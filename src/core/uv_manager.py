







from __future__ import annotations

import glob
import hashlib
import os
import platform
import shutil
import stat
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
from .cache_paths import PLUGIN_CACHE_DIR, plugin_cache_tmp_dir, remove_tree_quietly
from .gil_safe_qobject import prime as gil_safe
from .logging_utils import log as _log
from .model_config import IS_ROSETTA
from .streamed_download import sleep_unless_cancelled
from .subprocess_utils import (  # nosec B404
    get_clean_env_for_venv,
    get_subprocess_kwargs,
    run_unthrottled,
)







_LEGACY_UV_HOME_DIR = os.path.normpath(os.path.expanduser("~/.qgis_ai_segmentation"))


def _resolve_uv_dir() -> str:
    legacy = os.path.join(_LEGACY_UV_HOME_DIR, "uv")
    binary = "uv.exe" if sys.platform == "win32" else "uv"
    if os.path.isfile(os.path.join(legacy, binary)):
        return legacy
    return os.path.join(PLUGIN_CACHE_DIR, "uv")


UV_DIR = _resolve_uv_dir()
UV_VERSION = "0.10.10"







UV_SHA256 = {
    "uv-aarch64-apple-darwin.tar.gz": "8a09f0ef51ee7f7170731b4cb8bde5bf9ba6da5304f49a7df6cdab42a1f37b5d",  # noqa: E501  # pragma: allowlist secret
    "uv-x86_64-apple-darwin.tar.gz": "dd18420591d625f9b4ca2b57a7a6fe3cce43910f02e02d90e47a4101428de14a",  # noqa: E501  # pragma: allowlist secret
    "uv-x86_64-pc-windows-msvc.zip": "d31a30f1dfb96e630a08d5a9b3f3f551254b7ed6e9b7e495f46a4232661c7252",  # noqa: E501  # pragma: allowlist secret
    "uv-aarch64-unknown-linux-gnu.tar.gz": "2b80457b950deda12e8d5dc3b9b7494ac143eae47f1fb11b1c6e5a8495a6421e",  # noqa: E501  # pragma: allowlist secret
    "uv-x86_64-unknown-linux-gnu.tar.gz": "3e1027f26ce8c7e4c32e2277a7fed2cb410f2f1f9320d3df97653d40e21f415b",  # noqa: E501  # pragma: allowlist secret
}


DOWNLOAD_TIMEOUT_MS = 300000


def resolved_uv_version() -> str:





    try:
        from .install_config import uv_version

        return uv_version(UV_VERSION)
    except Exception:  # noqa: BLE001
        return UV_VERSION


def resolved_uv_digests() -> dict:








    try:
        from .install_config import uv_digests

        return uv_digests(UV_SHA256, UV_VERSION)
    except Exception:  # noqa: BLE001
        return dict(UV_SHA256)


def resolved_download_timeout_ms() -> int:

    try:
        from .install_config import uv_download_timeout_ms

        return uv_download_timeout_ms(DOWNLOAD_TIMEOUT_MS)
    except Exception:  # noqa: BLE001
        return DOWNLOAD_TIMEOUT_MS


def is_musl_linux() -> bool:





    if not sys.platform.startswith("linux"):
        return False
    return bool(glob.glob("/lib/ld-musl-*.so*") or glob.glob("/usr/lib/ld-musl-*.so*"))




_SUPPORTED_MACHINES = ("x86_64", "amd64", "arm64", "aarch64")


def unsupported_download_platform_reason() -> str:






    if sys.platform != "linux":
        return ""
    if is_musl_linux():
        return "this Linux uses musl and the published builds need glibc"
    machine = platform.machine()
    if machine.lower() not in _SUPPORTED_MACHINES:
        return f"no published build for this machine ({machine})"
    return ""


class DownloadStallGuard:












    def __init__(self, request, timeout_ms: int, cancel_check: Callable[[], bool] | None = None):


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



            self._watch_stall = True
        except (AttributeError, TypeError):
            pass  # nosec B110
        try:
            self._timer.start()
        except (RuntimeError, TypeError):
            pass  # nosec B110
        return self

    def __exit__(self, *_exc) -> None:
        self._timer.stop()
        try:
            self._request.downloadProgress.disconnect(self._on_progress)
        except (AttributeError, TypeError, RuntimeError):
            pass  # nosec B110

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

    if sys.platform == "win32":
        return os.path.join(UV_DIR, "uv.exe")
    return os.path.join(UV_DIR, "uv")


def uv_exists() -> bool:

    return os.path.isfile(get_uv_path())


def _discard_uv_dir() -> bool:








    from .checkpoint_manager import _remove_with_retry

    binary = get_uv_path()
    if os.path.isfile(binary) and not _remove_with_retry(binary):
        return False
    remove_tree_quietly(UV_DIR)
    return not os.path.isfile(binary)


def _get_uv_platform_info() -> tuple[str, str]:





    system = sys.platform
    machine = platform.machine().lower()

    if system == "darwin":
        if machine in ("arm64", "aarch64") or IS_ROSETTA:
            return ("aarch64-apple-darwin", ".tar.gz")
        return ("x86_64-apple-darwin", ".tar.gz")
    if system == "win32":



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

    triple, ext = _get_uv_platform_info()
    if not triple:
        return ""
    return (
        "https://github.com/astral-sh/uv/releases/download/"
        f"{resolved_uv_version()}/uv-{triple}{ext}"
    )


def _find_file_in_dir(directory: str, filename: str) -> str | None:

    for root, _dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


def _verify_uv_payload(content_bytes: bytes, asset_name: str) -> tuple[bool, str]:






    expected = resolved_uv_digests().get(asset_name, "")
    if not expected:
        return False, f"No pinned digest for {asset_name}; refusing to install"
    if hashlib.sha256(content_bytes).hexdigest() != expected:
        return False, "uv download failed integrity verification"
    return True, ""


def _apply_resolved_proxy() -> Callable[[], None] | None:













    try:
        from urllib.parse import unquote, urlparse

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
            unquote(parsed.username or ""),
            unquote(parsed.password or ""),
        ))

        _log("Routing the installer download through the configured proxy")
        return lambda: nam.setProxy(previous)
    except Exception as e:  # noqa: BLE001
        _log(f"Could not apply the resolved proxy: {type(e).__name__}",
             Qgis.MessageLevel.Warning)
        return None


def download_uv(
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None
) -> tuple[bool, str]:




    if uv_exists():



        if verify_uv():
            _log(f"uv already exists at {get_uv_path()}")
            return True, "uv already installed"
        if uv_exists():
            return False, "the uv binary on disk fails its check and cannot be replaced"

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


    from .server_dials import dial_in_range
    max_retries = dial_in_range("tuning.install.uv_download_max_retries", 3, 1, 10)
    backoff_base_s = dial_in_range("tuning.install.uv_download_backoff_base_s", 5, 1, 60)
    err = None
    error_msg = ""
    restore_proxy = _apply_resolved_proxy()
    try:
        for attempt in range(max_retries):
            if cancel_check and cancel_check():
                return False, "Download cancelled"



            request = gil_safe(QgsBlockingNetworkRequest())
            net_req = QNetworkRequest(QUrl(url))
            timeout_ms = resolved_download_timeout_ms()
            if hasattr(net_req, "setTransferTimeout"):
                net_req.setTransferTimeout(timeout_ms)


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
                wait = backoff_base_s * (2 ** attempt)
                _log(
                    f"uv download failed (attempt {attempt + 1}/{max_retries}): {error_msg}. "
                    f"Retrying in {wait}s...",
                    Qgis.MessageLevel.Warning
                )
                if progress_callback:
                    progress_callback(
                        0, f"Network error, retrying in {wait}s...")


                if sleep_unless_cancelled(wait, cancel_check):
                    return False, "Download cancelled"

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

    tmp_root = plugin_cache_tmp_dir()
    fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=tmp_root)
    os.close(fd)

    try:
        with open(temp_path, "wb") as f:
            f.write(content_bytes)


        if os.path.exists(UV_DIR) and not _discard_uv_dir():
            return False, "the previous uv binary is held by another program"
        os.makedirs(UV_DIR, exist_ok=True)



        extract_dir = tempfile.mkdtemp(prefix="uv_extract_", dir=tmp_root)
        try:
            if suffix == ".tar.gz":
                with tarfile.open(temp_path, "r:gz") as tar:
                    _safe_extract_tar(tar, extract_dir)
            else:
                with zipfile.ZipFile(temp_path, "r") as z:
                    _safe_extract_zip(z, extract_dir)


            binary_name = "uv.exe" if sys.platform == "win32" else "uv"
            found = _find_file_in_dir(extract_dir, binary_name)
            if not found:
                return False, "uv binary not found in archive"

            dest = get_uv_path()
            tmp_dest = dest + ".tmp"
            shutil.copy2(found, tmp_dest)


            if sys.platform != "win32":
                os.chmod(
                    tmp_dest,
                    stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)





            from .checkpoint_manager import _replace_with_retry

            if not _replace_with_retry(tmp_dest, dest):
                return False, "uv binary could not be moved into place"

        finally:
            remove_tree_quietly(extract_dir)

        if progress_callback:
            progress_callback(80, "Verifying uv...")

        if verify_uv():
            _installed = resolved_uv_version()
            _log(f"uv {_installed} installed successfully", Qgis.MessageLevel.Success)
            if progress_callback:
                progress_callback(100, "uv ready")
            return True, f"uv {_installed} installed"



        _discard_uv_dir()
        return False, "uv verification failed after download"

    except Exception as e:
        _log(f"uv installation failed: {e}", Qgis.MessageLevel.Warning)
        _discard_uv_dir()
        return False, f"uv installation failed: {str(e)[:200]}"
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def verify_uv(retries: int | None = None) -> bool:















    uv_path = get_uv_path()
    if not os.path.isfile(uv_path):
        return False

    from .server_dials import dial_in_range
    if retries is None:
        retries = dial_in_range("tuning.install.uv_verify_retries", 3, 1, 10)
    verify_timeout_s = dial_in_range("tuning.install.uv_verify_timeout_s", 15, 5, 60)
    attempts = max(1, retries)
    last_error = ""
    clean_env = get_clean_env_for_venv()
    for attempt in range(attempts):
        try:
            result = run_unthrottled(
                [uv_path, "--version"],
                text=True, encoding="utf-8", errors="replace", timeout=verify_timeout_s,
                env=clean_env,
                **get_subprocess_kwargs(),
            )
            if result.returncode == 0:
                version_out = result.stdout.strip()
                _log(f"uv verified: {version_out}")

                expected_version = resolved_uv_version()
                version_parts = version_out.split()
                if len(version_parts) < 2 or version_parts[:2] != ["uv", expected_version]:
                    _log(
                        f"uv version mismatch: expected {expected_version}, got '{version_out}'. "
                        "Re-downloading.",
                        Qgis.MessageLevel.Warning
                    )
                    _discard_uv_dir()
                    return False
                return True
            last_error = result.stderr or result.stdout or f"exit code {result.returncode}"
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
        if attempt < attempts - 1:
            _log(
                f"uv check {attempt + 1}/{attempts} failed ({last_error[:120]}), retrying...",
                Qgis.MessageLevel.Info
            )
            time.sleep(0.5 * (attempt + 1))

    _log(f"uv verification failed: {last_error}", Qgis.MessageLevel.Warning)

    _discard_uv_dir()
    return False


def remove_uv() -> tuple[bool, str]:

    if not os.path.exists(UV_DIR):
        return True, "uv not installed"
    try:
        if not _discard_uv_dir():
            return False, "the uv binary is held by another program"
        _log("Removed uv installation", Qgis.MessageLevel.Success)
        return True, "uv removed"
    except Exception as e:
        _log(f"Failed to remove uv: {e}", Qgis.MessageLevel.Warning)
        return False, f"Failed to remove uv: {str(e)[:200]}"
