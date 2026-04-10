"""
UV Package Installer Manager for QGIS AI-Segmentation Plugin.

Downloads and manages Astral's uv binary for faster package installation.
Falls back to pip seamlessly if uv is unavailable.

Source: https://github.com/astral-sh/uv
"""
from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
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

CACHE_DIR = os.path.expanduser("~/.qgis_ai_segmentation")
UV_DIR = os.path.join(CACHE_DIR, "uv")
UV_VERSION = "0.10.10"


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
        err = request.get(QNetworkRequest(QUrl(url)))

        if err == QgsBlockingNetworkRequest.NoError:
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

    if err != QgsBlockingNetworkRequest.NoError:
        _log(f"uv download failed: {error_msg}", Qgis.MessageLevel.Warning)
        return False, f"uv download failed: {error_msg}"

    if cancel_check and cancel_check():
        return False, "Download cancelled"

    reply = request.reply()
    content = reply.content()
    content_bytes = content.data()

    if progress_callback:
        size_mb = len(content_bytes) / (1024 * 1024)
        progress_callback(50, f"Downloaded uv ({size_mb:.1f} MB), extracting...")

    _, ext = _get_uv_platform_info()
    suffix = ".zip" if ext == ".zip" else ".tar.gz"
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        with open(temp_path, "wb") as f:
            f.write(content_bytes)

        # Remove existing UV_DIR if present
        if os.path.exists(UV_DIR):
            shutil.rmtree(UV_DIR)
        os.makedirs(UV_DIR, exist_ok=True)

        # Extract to a temp dir first, then move uv binary to UV_DIR
        extract_dir = tempfile.mkdtemp(prefix="uv_extract_")
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
        kwargs = {}
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = startupinfo

        result = subprocess.run(
            [uv_path, "--version"],
            capture_output=True, text=True, encoding="utf-8", timeout=15,
            **kwargs,
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
