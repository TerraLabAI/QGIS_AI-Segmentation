"""
Python Standalone Manager for QGIS AI-Segmentation Plugin.

Downloads and manages a standalone Python interpreter that matches
the QGIS Python version, ensuring 100% compatibility.

Source: https://github.com/astral-sh/python-build-standalone
"""
from __future__ import annotations

import os
import platform
import shutil
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
from .platform_detect import is_musl_linux, is_windows_arm64
from .uv_manager import DOWNLOAD_TIMEOUT_MS

PLUGIN_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("AI_SEGMENTATION_CACHE_DIR") or os.path.expanduser("~/.qgis_ai_segmentation")
STANDALONE_DIR = os.path.join(CACHE_DIR, "python_standalone")


# Release tag from python-build-standalone
# Update this periodically to get newer Python builds
RELEASE_TAG = "20251014"

# Mapping of Python minor versions to their latest patch versions in the release
PYTHON_VERSIONS = {
    (3, 9): "3.9.24",
    (3, 10): "3.10.19",
    (3, 11): "3.11.14",
    (3, 12): "3.12.12",
    (3, 13): "3.13.9",
    (3, 14): "3.14.0",
}

# Pinned standalone target when QGIS's own Python is outside the supported
# range (older QGIS on 3.7/3.8, future QGIS on 3.15+, or anything unknown).
# 3.12 is chosen because it has wheels for BOTH model paths: SAM2 (torch
# >=2.5.1, cp312) and SAM1 / Intel-mac (torch <=2.2.2 and <2.6, both ship
# cp312). It is a well-tested build in the release set above.
_PINNED_TARGET_VERSION = (3, 12)


def is_nixos() -> bool:
    """Detect NixOS where standalone Python binaries cannot run."""
    if sys.platform != "linux":
        return False
    nix_env = os.environ.get("NIX_PROFILES")
    return os.path.exists("/etc/NIXOS") or bool(nix_env)


def is_unsupported_windows() -> tuple[bool, str]:
    """Detect Windows versions below the standalone Python's official support.

    astral-sh/python-build-standalone targets Windows 8+ — Windows 7 binaries
    boot but commonly miss runtime APIs (e.g. ssl module loading fails because
    schannel-related symbols are absent in older kernel32), producing the
    "Can't connect to HTTPS URL because the SSL module is not available"
    failure observed in user reports. Detect early so we surface a clear
    "OS not supported" message instead of letting the install loop on a
    download that will never produce a working interpreter.
    """
    if sys.platform != "win32":
        return False, ""
    release = platform.release() or ""
    # platform.release() returns e.g. "7", "XP", "2008Server", "2012Server".
    # 2008/2012 Server share the pre-Windows-8 kernel (Server 2008 R2 == Win7,
    # Server 2012 == Win8 but its non-R2 base predates the standalone floor),
    # so they hit the same missing-runtime-API failures.
    unsupported = ("7", "Vista", "XP", "2003Server", "post2003", "2008Server", "2008ServerR2")
    if release in unsupported:
        return True, (
            f"Windows {release} is not supported by AI Segmentation. "
            "The bundled Python interpreter requires Windows 8 or later. "
            "Please upgrade to Windows 10 or 11."
        )
    return False, ""


def is_unsupported_macos() -> tuple[bool, str]:
    """Detect macOS versions too old for the standalone Python.

    Lowering qgisMinimumVersion (now 3.20) lets the plugin UI load on more
    machines, including some running old macOS, but python-build-standalone
    macOS builds need a modern system. Block clearly below 10.13 (High Sierra,
    2017) instead of
    letting the venv bootstrap fail with an opaque dynamic-loader error. The
    threshold is intentionally conservative so no working setup is rejected.
    """
    if sys.platform != "darwin":
        return False, ""
    ver = platform.mac_ver()[0] or ""
    try:
        parts = tuple(int(p) for p in ver.split(".")[:2])
    except ValueError:
        return False, ""
    if parts and parts < (10, 13):
        return True, (
            f"macOS {ver} is not supported by AI Segmentation. "
            "The bundled Python interpreter requires macOS 10.13 or later. "
            "Please update macOS."
        )
    return False, ""


def _get_windows_antivirus_help(plugin_path: str) -> str:
    """
    Return help message for Windows antivirus issues.
    """
    return (
        "Installation failed - this may be caused by antivirus software blocking the extraction.\n"
        "Please try:\n"
        "  1. Temporarily disable your antivirus (Windows Defender, etc.)\n"
        "  2. Add an exclusion for the QGIS plugins folder\n"
        "  3. Try the installation again\n"
        f"Folder to exclude: {plugin_path}"
    )


def get_qgis_python_version() -> tuple[int, int]:
    """Return the (major, minor) Python version QGIS itself runs on.

    This is the HOST interpreter version. It is NOT necessarily the version
    we download for the venv: use get_target_python_version() for that.
    """
    return (sys.version_info.major, sys.version_info.minor)


def get_target_python_version() -> tuple[int, int]:
    """Single source of truth for the standalone Python we download.

    Everything that downloads, verifies, names, or compares the standalone
    interpreter MUST call this so the three can never diverge again. A past
    bug downloaded 3.13 when QGIS's Python was unknown, then verify rejected
    it for not matching QGIS's version, looping forever. Selection rules:

    - Rosetta (x86_64 QGIS on Apple Silicon): (3, 10), so we fetch ARM64
      Python 3.10+ for SAM2 instead of matching QGIS's x86_64 3.9.
    - QGIS Python is a known, supported version (in PYTHON_VERSIONS and
      >= 3.9): match it, so in-process venv imports share the host ABI.
    - Anything else (3.7 / 3.8 on old QGIS, 3.15+ on future QGIS, or an
      unknown build): pin to a well-tested version that has wheels for both
      model paths. See _PINNED_TARGET_VERSION.
    """
    if IS_ROSETTA:
        return (3, 10)
    qgis_version = get_qgis_python_version()
    if qgis_version in PYTHON_VERSIONS and qgis_version >= (3, 9):
        return qgis_version
    return _PINNED_TARGET_VERSION


def get_python_full_version() -> str:
    """Get the full Python version string for download (e.g., '3.12.12')."""
    version_tuple = get_target_python_version()
    if version_tuple in PYTHON_VERSIONS:
        return PYTHON_VERSIONS[version_tuple]
    # Defensive only: get_target_python_version() always returns a tuple that
    # is in PYTHON_VERSIONS, so this branch is unreachable unless the pinned
    # default is removed from the map. Fall back to the pinned version, never
    # to a version that verify_standalone_python() would then reject.
    _log(
        f"Python {version_tuple[0]}.{version_tuple[1]} not in PYTHON_VERSIONS, "
        f"falling back to {_PINNED_TARGET_VERSION[0]}.{_PINNED_TARGET_VERSION[1]}",
        Qgis.MessageLevel.Warning)
    return PYTHON_VERSIONS[_PINNED_TARGET_VERSION]


def _create_python_symlinks(python_dir: str) -> None:
    """Create python3 symlink if only versioned binary exists (e.g. python3.12)."""
    bin_dir = os.path.join(python_dir, "bin")
    python3_path = os.path.join(bin_dir, "python3")
    if os.path.exists(python3_path):
        return
    # Find versioned binary like python3.12
    major, minor = get_target_python_version()
    versioned = os.path.join(bin_dir, f"python{major}.{minor}")
    if os.path.exists(versioned):
        os.symlink(f"python{major}.{minor}", python3_path)
        _log(f"Created python3 symlink -> python{major}.{minor}")


def get_standalone_python_path() -> str:
    """Get the path to the standalone Python executable."""
    python_dir = os.path.join(STANDALONE_DIR, "python")

    if sys.platform == "win32":
        return os.path.join(python_dir, "python.exe")
    return os.path.join(python_dir, "bin", "python3")


def standalone_python_exists() -> bool:
    """Check if standalone Python is already installed."""
    python_path = get_standalone_python_path()
    return os.path.exists(python_path)


def standalone_python_is_current() -> bool:
    """Check if installed standalone Python matches QGIS Python major.minor.

    Returns False if standalone doesn't exist or version doesn't match.
    """
    python_path = get_standalone_python_path()
    if not os.path.exists(python_path):
        return False

    try:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONIOENCODING"] = "utf-8"

        kwargs = {}
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = startupinfo

        result = subprocess.run(  # nosec B603
            [python_path, "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=15, env=env, **kwargs,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) == 2:
                installed = (int(parts[0]), int(parts[1]))
                expected = get_target_python_version()
                if installed != expected:
                    _log(
                        f"Standalone Python {installed[0]}.{installed[1]} "
                        f"doesn't match QGIS {expected[0]}.{expected[1]}",
                        Qgis.MessageLevel.Warning)
                    return False
                return True
    except Exception as e:
        _log(f"Failed to check standalone Python version: {e}", Qgis.MessageLevel.Warning)

    return False


def _get_platform_info() -> tuple[str, str]:
    """Get platform and architecture info for download URL."""
    system = sys.platform
    machine = platform.machine().lower()

    if system == "darwin":
        if machine in ("arm64", "aarch64") or IS_ROSETTA:
            return ("aarch64-apple-darwin", ".tar.gz")
        return ("x86_64-apple-darwin", ".tar.gz")
    if system == "win32":
        # Windows on ARM runs QGIS (and us) as emulated x86_64. Native
        # aarch64 Python exists, but torch publishes win_arm64 wheels only
        # from 2.7+, which is outside our pinned ranges, so a native venv
        # could not install torch. Keep the emulated x86_64 build (it works
        # under emulation) and just log the decision.
        if is_windows_arm64():
            _log(
                "ARM64 Windows detected; using emulated x86_64 Python "
                "(native torch wheels not yet available for pinned versions)",
                Qgis.MessageLevel.Info)
        return ("x86_64-pc-windows-msvc", ".tar.gz")
    # Linux: musl systems crash on the -gnu loader, so pick the -musl build.
    libc_suffix = "musl" if is_musl_linux() else "gnu"
    if machine in ("arm64", "aarch64"):
        return (f"aarch64-unknown-linux-{libc_suffix}", ".tar.gz")
    return (f"x86_64-unknown-linux-{libc_suffix}", ".tar.gz")


def get_download_urls() -> list[str]:
    """Candidate download URLs for the standalone Python, preferred first.

    install_only_stripped is the same build minus native debug symbols
    (about half the download on Windows, a third of it on Linux); it is
    what uv itself ships. Plain install_only stays as a fallback in case
    a release or platform lacks the stripped variant.
    """
    python_version = get_python_full_version()
    platform_str, ext = _get_platform_info()
    base = (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{RELEASE_TAG}"
    )
    prefix = f"cpython-{python_version}+{RELEASE_TAG}-{platform_str}"
    return [
        f"{base}/{prefix}-install_only_stripped{ext}",
        f"{base}/{prefix}-install_only{ext}",
    ]


def get_download_url() -> str:
    """Preferred download URL (kept for callers that want a single URL)."""
    return get_download_urls()[0]


def download_python_standalone(
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None
) -> tuple[bool, str]:
    """
    Download and install Python standalone using QGIS network manager.

    Uses QgsBlockingNetworkRequest to respect QGIS proxy settings.

    Args:
        progress_callback: Function called with (percent, message) for progress updates
        cancel_check: Function that returns True if operation should be cancelled

    Returns:
        Tuple of (success: bool, message: str)
    """
    unsupported, why = is_unsupported_windows()
    if unsupported:
        _log(why, Qgis.MessageLevel.Critical)
        return False, why

    mac_unsupported, mac_why = is_unsupported_macos()
    if mac_unsupported:
        _log(mac_why, Qgis.MessageLevel.Critical)
        return False, mac_why

    if standalone_python_exists():
        _log("Python standalone already exists", Qgis.MessageLevel.Info)
        return True, "Python standalone already installed"

    urls = get_download_urls()
    python_version = get_python_full_version()

    _log(f"Downloading Python {python_version} from: {urls[0]}", Qgis.MessageLevel.Info)

    if progress_callback:
        progress_callback(0, f"Downloading Python {python_version}...")

    # Create temp file for download
    fd, temp_path = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)

    try:
        if cancel_check and cancel_check():
            return False, "Download cancelled"

        if progress_callback:
            progress_callback(5, "Connecting to download server...")

        # Try each URL variant (stripped first, plain as fallback), each
        # with up to 3 attempts and exponential backoff. QGIS network
        # manager is used so QGIS proxy settings are respected.
        max_retries = 3
        err = None
        error_msg = ""
        request = None
        for url_idx, url in enumerate(urls):
            qurl = QUrl(url)
            for attempt in range(max_retries):
                if cancel_check and cancel_check():
                    return False, "Download cancelled"

                request = QgsBlockingNetworkRequest()
                net_req = QNetworkRequest(qurl)
                if hasattr(net_req, "setTransferTimeout"):
                    net_req.setTransferTimeout(DOWNLOAD_TIMEOUT_MS)
                err = request.get(net_req)

                if err == QgsBlockingNetworkRequest.ErrorCode.NoError:
                    break

                error_msg = request.errorMessage()
                if "404" in error_msg or "Not Found" in error_msg:
                    # No point retrying a 404; move on to the next variant
                    break

                if attempt < max_retries - 1:
                    wait = 5 * (2 ** attempt)  # 5, 10s
                    _log(
                        f"Download failed (attempt {attempt + 1}/{max_retries}): {error_msg}. "
                        f"Retrying in {wait}s...",
                        Qgis.MessageLevel.Warning
                    )
                    if progress_callback:
                        progress_callback(
                            5, f"Network error, retrying in {wait}s...")
                    time.sleep(wait)

            if err == QgsBlockingNetworkRequest.ErrorCode.NoError:
                break

            is_404 = "404" in error_msg or "Not Found" in error_msg
            if is_404:
                if url_idx + 1 < len(urls):
                    _log(
                        f"Archive variant not published ({url}), "
                        "trying the fallback variant...",
                        Qgis.MessageLevel.Warning)
                    continue
                error_msg = (
                    f"Python {python_version} not available for this platform. "
                    f"URL: {url}"
                )
                _log(error_msg, Qgis.MessageLevel.Critical)
                return False, error_msg

        if err != QgsBlockingNetworkRequest.ErrorCode.NoError:
            error_msg = f"Download failed: {error_msg}"
            _log(error_msg, Qgis.MessageLevel.Critical)
            return False, error_msg

        if cancel_check and cancel_check():
            return False, "Download cancelled"

        reply = request.reply()
        content = reply.content()

        content_size = len(content)
        if content_size == 0:
            return False, "Download failed: received empty file (0 bytes)"
        min_expected = 10 * 1024 * 1024  # 10 MB
        if content_size < min_expected:
            _log(
                f"Download suspiciously small: {content_size} bytes (expected >10 MB)", Qgis.MessageLevel.Warning)
            return False, (
                f"Download failed: file too small ({content_size / (1024 * 1024):.1f} MB). "
                "A firewall or proxy may be blocking the download."
            )

        if progress_callback:
            total_mb = content_size / (1024 * 1024)
            progress_callback(50, f"Downloaded {total_mb:.1f} MB, saving...")

        # Write content to temp file
        with open(temp_path, "wb") as f:
            f.write(content.data())

        # Validate archive magic bytes (catch proxy/firewall HTML pages)
        with open(temp_path, "rb") as f:
            magic = f.read(4)
        is_gzip = magic[:2] == b"\x1f\x8b"
        is_zip = magic[:2] == b"PK"
        if not is_gzip and not is_zip:
            try:
                preview_text = bytes(content.data()[:200]).decode(
                    "utf-8", errors="replace")[:150]
            except Exception:
                preview_text = "(binary data)"
            return False, (
                "Download failed: file is not a valid archive. "
                "A firewall or proxy may have returned an error page. "
                f"Preview: {preview_text}"
            )

        _log(f"Download complete ({content_size} bytes), extracting...", Qgis.MessageLevel.Info)

        if progress_callback:
            progress_callback(55, "Extracting Python...")

        # Remove existing standalone dir if it exists
        if os.path.exists(STANDALONE_DIR):
            shutil.rmtree(STANDALONE_DIR)

        os.makedirs(STANDALONE_DIR, exist_ok=True)

        # Extract archive with path traversal protection
        if temp_path.endswith(".tar.gz") or temp_path.endswith(".tgz"):
            with tarfile.open(temp_path, "r:gz") as tar:
                _safe_extract_tar(tar, STANDALONE_DIR)
        else:
            with zipfile.ZipFile(temp_path, "r") as z:
                _safe_extract_zip(z, STANDALONE_DIR)

        # Create python3 symlink if missing (archive symlinks skipped for safety)
        if sys.platform != "win32":
            _create_python_symlinks(os.path.join(STANDALONE_DIR, "python"))

        if progress_callback:
            progress_callback(80, "Verifying Python installation...")

        # Verify installation
        success, verify_msg = verify_standalone_python()

        if success:
            if progress_callback:
                progress_callback(100, f"✓ Python {python_version} installed")
            _log("Python standalone installed successfully", Qgis.MessageLevel.Success)
            return True, f"Python {python_version} installed successfully"
        # Clean up broken installation so _get_system_python() won't find it
        remove_standalone_python()
        return False, f"Verification failed: {verify_msg}"

    except InterruptedError:
        return False, "Download cancelled"
    except Exception as e:
        error_msg = f"Installation failed: {str(e)}"
        _log(error_msg, Qgis.MessageLevel.Critical)

        # On Windows, check for antivirus blocking (permission/access errors)
        if sys.platform == "win32":
            error_lower = str(e).lower()
            if "denied" in error_lower or "access" in error_lower or "permission" in error_lower:
                antivirus_help = _get_windows_antivirus_help(STANDALONE_DIR)
                _log(antivirus_help, Qgis.MessageLevel.Warning)
                error_msg = f"{error_msg}\n\n{antivirus_help}"

        return False, error_msg
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def verify_standalone_python() -> tuple[bool, str]:
    """Verify that the standalone Python installation works."""
    python_path = get_standalone_python_path()

    if not os.path.exists(python_path):
        return False, f"Python executable not found at {python_path}"

    # On Unix, make sure it's executable
    if sys.platform != "win32":
        try:
            import stat
            # Set executable permission (owner rwx, group rx, others rx)
            os.chmod(python_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        except OSError:
            pass

    try:
        # Test basic execution
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONIOENCODING"] = "utf-8"

        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            result = subprocess.run(  # nosec B603
                [python_path, "-c", "import sys; print(sys.version)"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                env=env,
                startupinfo=startupinfo,
            )
        else:
            result = subprocess.run(  # nosec B603
                [python_path, "-c", "import sys; print(sys.version)"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                env=env,
            )

        if result.returncode == 0:
            version_output = result.stdout.strip().split()[0]
            expected_version = get_python_full_version()

            # Verify major.minor matches the SAME source of truth used to
            # pick and download the build (never QGIS's host version).
            major, minor = get_target_python_version()
            if not version_output.startswith(f"{major}.{minor}"):
                msg = f"Python version mismatch: got {version_output}, expected {expected_version}"
                _log(msg, Qgis.MessageLevel.Warning)
                return False, f"Version mismatch: downloaded {version_output}, expected {expected_version}"

            _log(f"Verified Python standalone: {version_output}", Qgis.MessageLevel.Success)
            return True, f"Python {version_output} verified"
        error = result.stderr or "Unknown error"
        _log(f"Python verification failed: {error}", Qgis.MessageLevel.Warning)
        return False, f"Verification failed: {error[:100]}"

    except subprocess.TimeoutExpired:
        return False, "Python verification timed out"
    except Exception as e:
        return False, f"Verification error: {str(e)[:100]}"


def remove_standalone_python() -> tuple[bool, str]:
    """Remove the standalone Python installation."""
    if not os.path.exists(STANDALONE_DIR):
        return True, "Standalone Python not installed"

    try:
        shutil.rmtree(STANDALONE_DIR)
        _log("Removed standalone Python installation", Qgis.MessageLevel.Success)
        return True, "Standalone Python removed"
    except Exception as e:
        error_msg = f"Failed to remove: {str(e)}"
        _log(error_msg, Qgis.MessageLevel.Warning)
        return False, error_msg
