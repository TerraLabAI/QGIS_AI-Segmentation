"""
UV Package Installer Manager for QGIS AI-Segmentation Plugin.

Downloads and manages Astral's uv binary for faster package installation.
Falls back to pip seamlessly if uv is unavailable.

Source: https://github.com/astral-sh/uv
"""
from __future__ import annotations

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
from .logging_utils import log as _log
from .model_config import IS_ROSETTA
from .subprocess_utils import get_subprocess_kwargs  # nosec B404 - our helper, name merely starts with "subprocess"

CACHE_DIR = os.path.expanduser("~/.qgis_ai_segmentation")
UV_DIR = os.path.join(CACHE_DIR, "uv")
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


def _download_tmp_dir() -> str | None:
    """Containment temp dir on the cache volume, mirroring venv_manager's
    _apply_cache_containment. The uv download temp file + extraction dir must
    land on the same volume the install uses, or a full SYSTEM drive still
    ENOSPCs even when AI_SEGMENTATION_CACHE_DIR points at a roomy secondary
    drive. The env-aware cache dir is derived here the same way venv_manager
    does (the module-level CACHE_DIR above deliberately stays the plain home
    path for the uv binary location). Returns None on any failure so the caller
    falls back to the system default temp (``dir=None`` == the old behaviour)."""
    cache_dir = os.environ.get("AI_SEGMENTATION_CACHE_DIR") or os.path.expanduser(
        "~/.qgis_ai_segmentation")
    tmp_dir = os.path.join(cache_dir, "tmp")
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir
    except OSError:
        return None


def get_uv_path() -> str:
    """Path to the uv binary."""
    if sys.platform == "win32":
        return os.path.join(UV_DIR, "uv.exe")
    return os.path.join(UV_DIR, "uv")


def uv_exists() -> bool:
    """Check if the uv binary exists on disk."""
    return os.path.isfile(get_uv_path())


def _get_uv_platform_info() -> tuple[str, str]:
    """Returns (platform_triple, extension) for the uv download URL."""
    system = sys.platform
    machine = platform.machine().lower()

    if system == "darwin":
        if machine in ("arm64", "aarch64") or IS_ROSETTA:
            return ("aarch64-apple-darwin", ".tar.gz")
        return ("x86_64-apple-darwin", ".tar.gz")
    if system == "win32":
        return ("x86_64-pc-windows-msvc", ".zip")
    if machine in ("arm64", "aarch64"):
        return ("aarch64-unknown-linux-gnu", ".tar.gz")
    return ("x86_64-unknown-linux-gnu", ".tar.gz")


def _get_uv_download_url() -> str:
    """Build the GitHub release URL for uv."""
    triple, ext = _get_uv_platform_info()
    return (
        "https://github.com/astral-sh/uv/releases/download/"
        f"{UV_VERSION}/uv-{triple}{ext}"
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
    expected = UV_SHA256.get(asset_name, "")
    if not expected:
        return False, f"No pinned digest for {asset_name}; refusing to install"
    if hashlib.sha256(content_bytes).hexdigest() != expected:
        return False, "uv download failed integrity verification"
    return True, ""


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

    url = _get_uv_download_url()
    _log(f"Downloading uv {UV_VERSION} from: {url}")

    if progress_callback:
        progress_callback(0, "Downloading uv package installer...")

    if cancel_check and cancel_check():
        return False, "Download cancelled"

    # Retry up to 3 times with exponential backoff for unstable networks
    max_retries = 3
    err = None
    error_msg = ""
    for attempt in range(max_retries):
        if cancel_check and cancel_check():
            return False, "Download cancelled"

        request = QgsBlockingNetworkRequest()
        net_req = QNetworkRequest(QUrl(url))
        if hasattr(net_req, "setTransferTimeout"):
            net_req.setTransferTimeout(DOWNLOAD_TIMEOUT_MS)
        err = request.get(net_req)

        if err == QgsBlockingNetworkRequest.ErrorCode.NoError:
            break

        error_msg = request.errorMessage()
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
    # Contain the download temp on the CACHE_DIR volume (see _download_tmp_dir).
    tmp_root = _download_tmp_dir()
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
        # CACHE_DIR containment as the download temp above.
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

            os.replace(tmp_dest, dest)

        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

        if progress_callback:
            progress_callback(80, "Verifying uv...")

        if verify_uv():
            _log(f"uv {UV_VERSION} installed successfully", Qgis.MessageLevel.Success)
            if progress_callback:
                progress_callback(100, "uv ready")
            return True, f"uv {UV_VERSION} installed"
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


def verify_uv() -> bool:
    """Run `uv --version` to verify the binary works."""
    uv_path = get_uv_path()
    if not os.path.isfile(uv_path):
        return False

    try:
        result = subprocess.run(  # nosec B603
            [uv_path, "--version"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
            **get_subprocess_kwargs(),
        )
        if result.returncode == 0:
            version_out = result.stdout.strip()
            _log(f"uv verified: {version_out}")
            # Check version matches expected UV_VERSION
            if UV_VERSION not in version_out:
                _log(
                    f"uv version mismatch: expected {UV_VERSION}, got '{version_out}'. "
                    "Re-downloading.",
                    Qgis.MessageLevel.Warning
                )
                shutil.rmtree(UV_DIR, ignore_errors=True)
                return False
            return True
        _log(f"uv --version failed: {result.stderr or result.stdout}", Qgis.MessageLevel.Warning)
    except Exception as e:
        _log(f"uv verification failed: {e}", Qgis.MessageLevel.Warning)

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
