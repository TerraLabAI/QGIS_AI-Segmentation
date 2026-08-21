"""
UV Package Installer Manager for QGIS AI-Segmentation Plugin.

Downloads and manages Astral's uv binary for faster package installation.
Falls back to pip seamlessly if uv is unavailable.

Source: https://github.com/astral-sh/uv
"""
from __future__ import annotations

import glob
import hashlib
import os
import platform
import shutil
import stat
import subprocess  # nosec B404
import sys
import tarfile
import tempfile
import time
import zipfile
from typing import Callable

from qgis.core import Qgis, QgsBlockingNetworkRequest
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from .archive_utils import safe_extract_tar as _safe_extract_tar
from .archive_utils import safe_extract_zip as _safe_extract_zip
from .cache_paths import plugin_cache_tmp_dir
from .logging_utils import log as _log
from .model_config import IS_ROSETTA
from .subprocess_utils import get_clean_env_for_venv, get_subprocess_kwargs  # nosec B404 - our own helper

# Plain home path on purpose: the binary stays put when AI_SEGMENTATION_CACHE_DIR
# moves the rest of the cache, so an existing install keeps finding it.
UV_HOME_DIR = os.path.expanduser("~/.qgis_ai_segmentation")
UV_DIR = os.path.join(UV_HOME_DIR, "uv")
UV_VERSION = "0.10.10"

# SHA256 of each release asset, copied from the official uv-<triple><ext>.sha256
# files published next to every uv release asset. These digests are public
# integrity checks, not secrets. MUST be updated in the same commit as any
# UV_VERSION bump: the download fails closed on a mismatch or a missing entry,
# so a stale or partial table would brick installs. Keys are the exact asset
# filenames _get_uv_platform_info() can produce.
UV_SHA256 = {
    "uv-aarch64-apple-darwin.tar.gz": "8a09f0ef51ee7f7170731b4cb8bde5bf9ba6da5304f49a7df6cdab42a1f37b5d",  # noqa: E501  # pragma: allowlist secret
    "uv-x86_64-apple-darwin.tar.gz": "dd18420591d625f9b4ca2b57a7a6fe3cce43910f02e02d90e47a4101428de14a",  # noqa: E501  # pragma: allowlist secret
    "uv-x86_64-pc-windows-msvc.zip": "d31a30f1dfb96e630a08d5a9b3f3f551254b7ed6e9b7e495f46a4232661c7252",  # noqa: E501  # pragma: allowlist secret
    "uv-aarch64-unknown-linux-gnu.tar.gz": "2b80457b950deda12e8d5dc3b9b7494ac143eae47f1fb11b1c6e5a8495a6421e",  # noqa: E501  # pragma: allowlist secret
    "uv-x86_64-unknown-linux-gnu.tar.gz": "3e1027f26ce8c7e4c32e2277a7fed2cb410f2f1f9320d3df97653d40e21f415b",  # noqa: E501  # pragma: allowlist secret
}

# Inactivity timeout for the binary download (Qt 5.15+); generous for slow corporate links
DOWNLOAD_TIMEOUT_MS = 300000


def resolved_uv_version() -> str:
    """UV_VERSION, or the version the server asked for.

    A tool release can be pulled or turn out to be broken on one platform. The
    version and its digest table move together: see ``resolved_uv_digests``.
    """
    try:
        from .install_config import uv_version

        return uv_version(UV_VERSION)
    except Exception:  # noqa: BLE001 -- a bad config must never block an install
        return UV_VERSION


def resolved_uv_digests() -> dict:
    """The digest table for the version in force.

    While the served version equals the shipped one the table is additive and
    a shipped digest always wins. When the server moves the version the shipped
    digests describe different bytes, so they are dropped rather than checked
    against the wrong file. The check stays fail-closed either way: an asset
    with no digest is refused.
    """
    try:
        from .install_config import uv_digests

        return uv_digests(UV_SHA256, UV_VERSION)
    except Exception:  # noqa: BLE001 -- a bad config must never block an install
        return dict(UV_SHA256)


def resolved_download_timeout_ms() -> int:
    """Inactivity timeout on the tool download."""
    try:
        from .install_config import uv_download_timeout_ms

        return uv_download_timeout_ms(DOWNLOAD_TIMEOUT_MS)
    except Exception:  # noqa: BLE001 -- a bad config must never block an install
        return DOWNLOAD_TIMEOUT_MS


def is_musl_linux() -> bool:
    """True on a musl userland (Alpine and friends).

    The published tool builds are linked against glibc and cannot run here,
    so downloading one wastes tens of megabytes to fail verification.
    """
    if not sys.platform.startswith("linux"):
        return False
    return bool(glob.glob("/lib/ld-musl-*.so*") or glob.glob("/usr/lib/ld-musl-*.so*"))


# Machines the published builds cover. Anything else (i686, riscv64, ppc64le,
# s390x) has no asset to download.
_SUPPORTED_MACHINES = ("x86_64", "amd64", "arm64", "aarch64")


def unsupported_download_platform_reason() -> str:
    """Why no published build can serve this machine, or "" when one can.

    Answering early keeps a musl or exotic-architecture box off a 30 MB
    download that can only fail its verification, and sends it straight to a
    system interpreter with the reason written down.
    """
    if sys.platform != "linux":
        return ""
    if is_musl_linux():
        return "this Linux uses musl and the published builds need glibc"
    machine = platform.machine()
    if machine.lower() not in _SUPPORTED_MACHINES:
        return f"no published build for this machine ({machine})"
    return ""


class DownloadStallGuard:
    """Abort a blocking network request that stalls, or that the user cancelled.

    QgsBlockingNetworkRequest.get() spins its own event loop, so nothing else
    on the thread runs for the length of a transfer: Cancel does nothing until
    the transfer ends, and below Qt 5.15 there is no transfer timeout at all,
    so a stalled connection hangs for good. This watches the progress signal
    from inside that loop and aborts the request.

    Use as a context manager around the ``get()`` call, then read
    ``aborted_reason``: "cancelled", "stalled", or "" when it did not fire.
    """

    def __init__(self, request, timeout_ms: int, cancel_check: Callable[[], bool] | None = None):
        # Imported here rather than at module scope: this class is the only
        # thing in the file that needs a Qt timer.
        from qgis.PyQt.QtCore import QTimer

        self._request = request
        self._timeout_ms = max(1000, int(timeout_ms))
        self._cancel_check = cancel_check
        self._last_progress = time.monotonic()
        self._timer = QTimer()
        self._timer.setInterval(min(1000, self._timeout_ms))
        self._timer.timeout.connect(self._tick)
        self._watch_stall = False
        self.aborted_reason = ""

    def __enter__(self) -> DownloadStallGuard:
        self._last_progress = time.monotonic()
        try:
            self._request.downloadProgress.connect(self._on_progress)
            # Only arm the inactivity ceiling when progress is actually
            # observable. Without the signal it would be a total-duration
            # limit and would cut off a slow but healthy download.
            self._watch_stall = True
        except (AttributeError, TypeError):
            pass  # nosec B110 - cancel still works, the stall check does not
        try:
            self._timer.start()
        except (RuntimeError, TypeError):
            pass  # nosec B110 - no event dispatcher here, nothing to watch with
        return self

    def __exit__(self, *_exc) -> None:
        self._timer.stop()
        try:
            self._request.downloadProgress.disconnect(self._on_progress)
        except (AttributeError, TypeError, RuntimeError):
            pass  # nosec B110 - never connected, or already gone

    def _on_progress(self, *_args) -> None:
        self._last_progress = time.monotonic()

    def _tick(self) -> None:
        if self._cancel_check and self._cancel_check():
            self.aborted_reason = "cancelled"
            self._abort()
            return
        if not self._watch_stall:
            return
        if (time.monotonic() - self._last_progress) * 1000 >= self._timeout_ms:
            self.aborted_reason = "stalled"
            self._abort()

    def _abort(self) -> None:
        self._timer.stop()
        try:
            self._request.abort()
        except (AttributeError, RuntimeError) as e:
            _log(f"Could not abort the download: {e}", Qgis.MessageLevel.Info)


def get_uv_path() -> str:
    """Path to the uv binary."""
    if sys.platform == "win32":
        return os.path.join(UV_DIR, "uv.exe")
    return os.path.join(UV_DIR, "uv")


def uv_exists() -> bool:
    """Check if the uv binary exists on disk."""
    return os.path.isfile(get_uv_path())


def _get_uv_platform_info() -> tuple[str, str]:
    """Returns (platform_triple, extension) for the uv download URL.

    Returns ("", "") when no published asset fits this machine, so the caller
    skips the download instead of fetching one that cannot run.
    """
    system = sys.platform
    machine = platform.machine().lower()

    if system == "darwin":
        if machine in ("arm64", "aarch64") or IS_ROSETTA:
            return ("aarch64-apple-darwin", ".tar.gz")
        return ("x86_64-apple-darwin", ".tar.gz")
    if system == "win32":
        # Windows on ARM runs the x86_64 build under emulation on purpose: it
        # has to match the architecture of the QGIS process it manages an
        # environment for, and QGIS ships x86_64 on Windows.
        if machine in ("arm64", "aarch64"):
            _log(
                "Windows on ARM: using the x86_64 build, which matches this "
                "QGIS process.", Qgis.MessageLevel.Info)
        return ("x86_64-pc-windows-msvc", ".zip")
    if unsupported_download_platform_reason():
        return ("", "")
    if machine in ("arm64", "aarch64"):
        return ("aarch64-unknown-linux-gnu", ".tar.gz")
    return ("x86_64-unknown-linux-gnu", ".tar.gz")


def _get_uv_download_url() -> str:
    """Build the GitHub release URL for uv, or "" when there is no asset."""
    triple, ext = _get_uv_platform_info()
    if not triple:
        return ""
    return (
        "https://github.com/astral-sh/uv/releases/download/"
        f"{resolved_uv_version()}/uv-{triple}{ext}"
    )


def _find_file_in_dir(directory: str, filename: str) -> str | None:
    """Recursively find a file by name under directory."""
    for root, _dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


def _verify_uv_payload(content_bytes: bytes, asset_name: str) -> tuple[bool, str]:
    """Fail-closed SHA256 integrity check of a downloaded uv asset.

    Returns (ok, message). Both a missing pin and a digest mismatch fail, so a
    tampered CDN response or an unpinned variant is never written to disk or
    executed. Mirrors the fail-closed policy in checkpoint_manager.
    """
    expected = resolved_uv_digests().get(asset_name, "")
    if not expected:
        return False, f"No pinned digest for {asset_name}; refusing to install"
    if hashlib.sha256(content_bytes).hexdigest() != expected:
        return False, "uv download failed integrity verification"
    return True, ""


def _apply_resolved_proxy() -> Callable[[], None] | None:
    """Point this thread's network manager at the proxy the installer resolved.

    The blocking request already honours a proxy configured inside QGIS. It
    knows nothing about one that exists only in the environment or in the
    machine's automatic configuration script, and that is how many company
    networks publish theirs. This download is the first thing a first run
    fetches, so on such a machine it is also the first thing to fail.

    Only a worker thread is touched: that manager belongs to this download
    alone, while the GUI thread's is the one the whole application uses.
    Returns a callable that puts the previous setting back, or None when there
    was nothing to change.
    """
    try:
        from urllib.parse import urlparse

        from qgis.core import QgsNetworkAccessManager
        from qgis.PyQt.QtNetwork import QNetworkProxy

        from .qt_compat import resolve_qt_enum
        from .venv_network import _get_effective_proxy_url, _on_gui_thread

        if _on_gui_thread():
            return None
        proxy_url = _get_effective_proxy_url()
        if not proxy_url:
            return None
        parsed = urlparse(proxy_url)
        if not parsed.hostname:
            return None
        nam = QgsNetworkAccessManager.instance()
        previous = nam.proxy()
        nam.setProxy(QNetworkProxy(
            resolve_qt_enum(QNetworkProxy, "ProxyType", "HttpProxy"),
            parsed.hostname,
            parsed.port or 80,
            parsed.username or "",
            parsed.password or "",
        ))
        # Never log the URL itself: it can carry credentials.
        _log("Routing the installer download through the configured proxy")
        return lambda: nam.setProxy(previous)
    except Exception as e:  # noqa: BLE001 - a direct attempt is the fallback
        _log(f"Could not apply the resolved proxy: {type(e).__name__}",
             Qgis.MessageLevel.Warning)
        return None


def download_uv(
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None
) -> tuple[bool, str]:
    """Download uv from GitHub releases using QgsBlockingNetworkRequest.

    Returns (success, message).
    """
    if uv_exists():
        _log(f"uv already exists at {get_uv_path()}")
        return True, "uv already installed"

    reason = unsupported_download_platform_reason()
    if reason:
        _log(f"Skipping the uv download: {reason}", Qgis.MessageLevel.Info)
        return False, f"No uv build for this system: {reason}"

    url = _get_uv_download_url()
    if not url:
        return False, "No uv build for this system"
    _log(f"Downloading uv {resolved_uv_version()} from: {url}")

    if progress_callback:
        progress_callback(0, "Downloading uv package installer...")

    if cancel_check and cancel_check():
        return False, "Download cancelled"

    # Retry up to 3 times with exponential backoff for unstable networks
    max_retries = 3
    err = None
    error_msg = ""
    restore_proxy = _apply_resolved_proxy()
    try:
        for attempt in range(max_retries):
            if cancel_check and cancel_check():
                return False, "Download cancelled"

            request = QgsBlockingNetworkRequest()
            net_req = QNetworkRequest(QUrl(url))
            timeout_ms = resolved_download_timeout_ms()
            if hasattr(net_req, "setTransferTimeout"):
                net_req.setTransferTimeout(timeout_ms)
            # The guard covers Cancel during the transfer, and the whole timeout
            # below Qt 5.15, where setTransferTimeout does not exist.
            with DownloadStallGuard(request, timeout_ms, cancel_check) as guard:
                err = request.get(net_req)
            if guard.aborted_reason == "cancelled":
                return False, "Download cancelled"

            if err == QgsBlockingNetworkRequest.ErrorCode.NoError:
                break

            error_msg = request.errorMessage()
            if guard.aborted_reason == "stalled":
                error_msg = "the download stalled, no data was received"
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)  # 5, 10s
                _log(
                    f"uv download failed (attempt {attempt + 1}/{max_retries}): {error_msg}. "
                    f"Retrying in {wait}s...",
                    Qgis.MessageLevel.Warning
                )
                if progress_callback:
                    progress_callback(
                        0, f"Network error, retrying in {wait}s...")
                time.sleep(wait)

    finally:
        if restore_proxy:
            restore_proxy()

    if err != QgsBlockingNetworkRequest.ErrorCode.NoError:
        _log(f"uv download failed: {error_msg}", Qgis.MessageLevel.Warning)
        return False, f"uv download failed: {error_msg}"

    if cancel_check and cancel_check():
        return False, "Download cancelled"

    reply = request.reply()
    content = reply.content()
    content_bytes = content.data()

    # Verify integrity BEFORE writing/extracting/executing the binary.
    asset_name = url.rsplit("/", 1)[-1]
    ok, verify_msg = _verify_uv_payload(content_bytes, asset_name)
    if not ok:
        _log(verify_msg, Qgis.MessageLevel.Warning)
        return False, verify_msg

    if progress_callback:
        size_mb = len(content_bytes) / (1024 * 1024)
        progress_callback(50, f"Downloaded uv ({size_mb:.1f} MB), extracting...")

    _, ext = _get_uv_platform_info()
    suffix = ".zip" if ext == ".zip" else ".tar.gz"
    # Contain the download temp on the cache volume (see plugin_cache_tmp_dir).
    tmp_root = plugin_cache_tmp_dir()
    fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=tmp_root)
    os.close(fd)

    try:
        with open(temp_path, "wb") as f:
            f.write(content_bytes)

        # Remove existing UV_DIR if present
        if os.path.exists(UV_DIR):
            shutil.rmtree(UV_DIR)
        os.makedirs(UV_DIR, exist_ok=True)

        # Extract to a temp dir first, then move uv binary to UV_DIR. Same
        # cache-volume containment as the download temp above.
        extract_dir = tempfile.mkdtemp(prefix="uv_extract_", dir=tmp_root)
        try:
            if suffix == ".tar.gz":
                with tarfile.open(temp_path, "r:gz") as tar:
                    _safe_extract_tar(tar, extract_dir)
            else:
                with zipfile.ZipFile(temp_path, "r") as z:
                    _safe_extract_zip(z, extract_dir)

            # Find the uv binary in extracted files
            binary_name = "uv.exe" if sys.platform == "win32" else "uv"
            found = _find_file_in_dir(extract_dir, binary_name)
            if not found:
                return False, "uv binary not found in archive"

            dest = get_uv_path()
            tmp_dest = dest + ".tmp"
            shutil.copy2(found, tmp_dest)

            # Set executable on Unix
            if sys.platform != "win32":
                os.chmod(tmp_dest, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

            # The same ladder the checkpoint and the interpreter move on:
            # an on-access scanner holds a file it has just seen written, and
            # a single move that loses to it wipes the whole uv directory and
            # drops the install onto the slow path.
            from .checkpoint_manager import _replace_with_retry

            if not _replace_with_retry(tmp_dest, dest):
                return False, "uv binary could not be moved into place"

        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

        if progress_callback:
            progress_callback(80, "Verifying uv...")

        if verify_uv():
            _installed = resolved_uv_version()
            _log(f"uv {_installed} installed successfully", Qgis.MessageLevel.Success)
            if progress_callback:
                progress_callback(100, "uv ready")
            return True, f"uv {_installed} installed"
        # Cleanup on verification failure
        shutil.rmtree(UV_DIR, ignore_errors=True)
        return False, "uv verification failed after download"

    except Exception as e:
        _log(f"uv installation failed: {e}", Qgis.MessageLevel.Warning)
        shutil.rmtree(UV_DIR, ignore_errors=True)
        return False, f"uv installation failed: {str(e)[:200]}"
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def verify_uv(retries: int = 3) -> bool:
    """Run `uv --version` to verify the binary works.

    Retried with a short wait between attempts: a security scanner routinely
    holds a freshly written executable on its first execute, and taking that
    for a broken binary deleted uv and dropped the whole install onto the
    slower path.

    Run with the same scrubbed environment as every other install subprocess.
    QGIS's own loader path points at the libraries it ships, and on a Flatpak,
    Snap or conda build uv resolves them instead of the system's and dies. The
    failure branch below then deletes a binary that works.
    """
    uv_path = get_uv_path()
    if not os.path.isfile(uv_path):
        return False

    attempts = max(1, retries)
    last_error = ""
    clean_env = get_clean_env_for_venv()
    for attempt in range(attempts):
        try:
            result = subprocess.run(  # nosec B603
                [uv_path, "--version"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
                env=clean_env,
                **get_subprocess_kwargs(),
            )
            if result.returncode == 0:
                version_out = result.stdout.strip()
                _log(f"uv verified: {version_out}")
                # Check version matches the one in force
                expected_version = resolved_uv_version()
                if expected_version not in version_out:
                    _log(
                        f"uv version mismatch: expected {expected_version}, got '{version_out}'. "
                        "Re-downloading.",
                        Qgis.MessageLevel.Warning
                    )
                    shutil.rmtree(UV_DIR, ignore_errors=True)
                    return False
                return True
            last_error = result.stderr or result.stdout or f"exit code {result.returncode}"
        except Exception as e:  # noqa: BLE001 - retried, then reported below
            last_error = str(e)
        if attempt < attempts - 1:
            _log(
                f"uv check {attempt + 1}/{attempts} failed ({last_error[:120]}), retrying...",
                Qgis.MessageLevel.Info
            )
            time.sleep(0.5 * (attempt + 1))

    _log(f"uv verification failed: {last_error}", Qgis.MessageLevel.Warning)
    # Cleanup on failure
    shutil.rmtree(UV_DIR, ignore_errors=True)
    return False


def remove_uv() -> tuple[bool, str]:
    """Remove the uv installation."""
    if not os.path.exists(UV_DIR):
        return True, "uv not installed"
    try:
        shutil.rmtree(UV_DIR)
        _log("Removed uv installation", Qgis.MessageLevel.Success)
        return True, "uv removed"
    except Exception as e:
        _log(f"Failed to remove uv: {e}", Qgis.MessageLevel.Warning)
        return False, f"Failed to remove uv: {str(e)[:200]}"
