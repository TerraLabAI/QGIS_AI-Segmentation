"""
Python Standalone Manager for QGIS AI-Segmentation Plugin.

Downloads and manages a standalone Python interpreter that matches
the QGIS Python version, ensuring 100% compatibility.

Source: https://github.com/astral-sh/python-build-standalone
"""

import os
import sys
import platform
import subprocess
import tarfile
import zipfile
import tempfile
import shutil
from typing import Tuple, Optional, Callable

from qgis.core import QgsMessageLog, Qgis, QgsBlockingNetworkRequest
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest


PLUGIN_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.expanduser("~/.qgis_ai_segmentation")
STANDALONE_DIR = os.path.join(CACHE_DIR, "python_standalone")


def _safe_extract_tar(tar: tarfile.TarFile, dest_dir: str) -> None:
    """Safely extract tar archive with path traversal protection."""
    dest_dir = os.path.realpath(dest_dir)
    # Python 3.12+ supports the filter parameter (suppresses DeprecationWarning)
    use_filter = sys.version_info >= (3, 12)
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            continue
        member_path = os.path.realpath(os.path.join(dest_dir, member.name))
        if not member_path.startswith(dest_dir + os.sep) and member_path != dest_dir:
            raise ValueError(f"Attempted path traversal in tar archive: {member.name}")
        if use_filter:
            tar.extract(member, dest_dir, filter='data')
        else:
            tar.extract(member, dest_dir)


def _safe_extract_zip(zip_file: zipfile.ZipFile, dest_dir: str) -> None:
    """Safely extract zip archive with path traversal protection."""
    dest_dir = os.path.realpath(dest_dir)
    for member in zip_file.namelist():
        member_path = os.path.realpath(os.path.join(dest_dir, member))
        if not member_path.startswith(dest_dir + os.sep) and member_path != dest_dir:
            raise ValueError(f"Attempted path traversal in zip archive: {member}")
        zip_file.extract(member, dest_dir)


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


def _log(message: str, level=Qgis.Info):
    QgsMessageLog.logMessage(message, "AI Segmentation", level=level)


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
        "Folder to exclude: {}".format(plugin_path)
    )


def get_qgis_python_version() -> Tuple[int, int]:
    """Get the Python version used by QGIS."""
    return (sys.version_info.major, sys.version_info.minor)


def get_python_full_version() -> str:
    """Get the full Python version string for download (e.g., '3.12.12')."""
    version_tuple = get_qgis_python_version()
    if version_tuple in PYTHON_VERSIONS:
        return PYTHON_VERSIONS[version_tuple]
    # Fallback: use 3.13 (newest well-tested version) instead of X.Y.0
    # which likely doesn't exist in the release assets
    _log(
        "Python {}.{} not in PYTHON_VERSIONS, falling back to 3.13".format(
            version_tuple[0], version_tuple[1]),
        Qgis.Warning)
    return PYTHON_VERSIONS[(3, 13)]


def get_standalone_dir() -> str:
    """Get the directory where Python standalone is installed."""
    return STANDALONE_DIR


def get_standalone_python_path() -> str:
    """Get the path to the standalone Python executable."""
    python_dir = os.path.join(STANDALONE_DIR, "python")

    if sys.platform == "win32":
        return os.path.join(python_dir, "python.exe")
    else:
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

        result = subprocess.run(
            [python_path, "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
            capture_output=True, text=True, timeout=15, env=env, **kwargs,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) == 2:
                installed = (int(parts[0]), int(parts[1]))
                expected = get_qgis_python_version()
                if installed != expected:
                    _log(
                        "Standalone Python {}.{} doesn't match QGIS {}.{}".format(
                            installed[0], installed[1], expected[0], expected[1]),
                        Qgis.Warning)
                    return False
                return True
    except Exception as e:
        _log("Failed to check standalone Python version: {}".format(e), Qgis.Warning)

    return False


def _get_platform_info() -> Tuple[str, str]:
    """Get platform and architecture info for download URL."""
    system = sys.platform
    machine = platform.machine().lower()

    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return ("aarch64-apple-darwin", ".tar.gz")
        else:
            return ("x86_64-apple-darwin", ".tar.gz")
    elif system == "win32":
        return ("x86_64-pc-windows-msvc", ".tar.gz")
    else:  # Linux
        if machine in ("arm64", "aarch64"):
            return ("aarch64-unknown-linux-gnu", ".tar.gz")
        else:
            return ("x86_64-unknown-linux-gnu", ".tar.gz")


def get_download_url() -> str:
    """Construct the download URL for the standalone Python."""
    python_version = get_python_full_version()
    platform_str, ext = _get_platform_info()

    filename = f"cpython-{python_version}+{RELEASE_TAG}-{platform_str}-install_only{ext}"
    url = f"https://github.com/astral-sh/python-build-standalone/releases/download/{RELEASE_TAG}/{filename}"

    return url


def download_python_standalone(
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Tuple[bool, str]:
    """
    Download and install Python standalone using QGIS network manager.

    Uses QgsBlockingNetworkRequest to respect QGIS proxy settings.

    Args:
        progress_callback: Function called with (percent, message) for progress updates
        cancel_check: Function that returns True if operation should be cancelled

    Returns:
        Tuple of (success: bool, message: str)
    """
    if standalone_python_exists():
        _log("Python standalone already exists", Qgis.Info)
        return True, "Python standalone already installed"

    url = get_download_url()
    python_version = get_python_full_version()

    _log(f"Downloading Python {python_version} from: {url}", Qgis.Info)

    if progress_callback:
        progress_callback(0, f"Downloading Python {python_version}...")

    # Create temp file for download
    fd, temp_path = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)

    try:
        if cancel_check and cancel_check():
            return False, "Download cancelled"

        # Use QGIS network manager for proxy-aware downloads
        request = QgsBlockingNetworkRequest()
        qurl = QUrl(url)

        if progress_callback:
            progress_callback(5, "Connecting to download server...")

        # Perform the download (blocking)
        err = request.get(QNetworkRequest(qurl))

        if err != QgsBlockingNetworkRequest.NoError:
            error_msg = request.errorMessage()
            if "404" in error_msg or "Not Found" in error_msg:
                error_msg = f"Python {python_version} not available for this platform. URL: {url}"
            else:
                error_msg = f"Download failed: {error_msg}"
            _log(error_msg, Qgis.Critical)
            return False, error_msg

        if cancel_check and cancel_check():
            return False, "Download cancelled"

        reply = request.reply()
        content = reply.content()

        if progress_callback:
            total_mb = len(content) / (1024 * 1024)
            progress_callback(50, f"Downloaded {total_mb:.1f} MB, saving...")

        # Write content to temp file
        with open(temp_path, 'wb') as f:
            f.write(content.data())

        _log(f"Download complete ({len(content)} bytes), extracting...", Qgis.Info)

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

        if progress_callback:
            progress_callback(80, "Verifying Python installation...")

        # Verify installation
        success, verify_msg = verify_standalone_python()

        if success:
            if progress_callback:
                progress_callback(100, f"✓ Python {python_version} installed")
            _log("Python standalone installed successfully", Qgis.Success)
            return True, f"Python {python_version} installed successfully"
        else:
            return False, f"Verification failed: {verify_msg}"

    except InterruptedError:
        return False, "Download cancelled"
    except Exception as e:
        error_msg = f"Installation failed: {str(e)}"
        _log(error_msg, Qgis.Critical)

        # On Windows, check for antivirus blocking (permission/access errors)
        if sys.platform == "win32":
            error_lower = str(e).lower()
            if "denied" in error_lower or "access" in error_lower or "permission" in error_lower:
                antivirus_help = _get_windows_antivirus_help(STANDALONE_DIR)
                _log(antivirus_help, Qgis.Warning)
                error_msg = "{}\n\n{}".format(error_msg, antivirus_help)

        return False, error_msg
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def verify_standalone_python() -> Tuple[bool, str]:
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

            result = subprocess.run(
                [python_path, "-c", "import sys; print(sys.version)"],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                startupinfo=startupinfo,
            )
        else:
            result = subprocess.run(
                [python_path, "-c", "import sys; print(sys.version)"],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )

        if result.returncode == 0:
            version_output = result.stdout.strip().split()[0]
            expected_version = get_python_full_version()

            # Verify major.minor matches
            if not version_output.startswith(f"{sys.version_info.major}.{sys.version_info.minor}"):
                _log(f"Python version mismatch: got {version_output}, expected {expected_version}", Qgis.Warning)
                return False, f"Version mismatch: downloaded {version_output}, expected {expected_version}"

            _log(f"Verified Python standalone: {version_output}", Qgis.Success)
            return True, f"Python {version_output} verified"
        else:
            error = result.stderr or "Unknown error"
            _log(f"Python verification failed: {error}", Qgis.Warning)
            return False, f"Verification failed: {error[:100]}"

    except subprocess.TimeoutExpired:
        return False, "Python verification timed out"
    except Exception as e:
        return False, f"Verification error: {str(e)[:100]}"


def remove_standalone_python() -> Tuple[bool, str]:
    """Remove the standalone Python installation."""
    if not os.path.exists(STANDALONE_DIR):
        return True, "Standalone Python not installed"

    try:
        shutil.rmtree(STANDALONE_DIR)
        _log("Removed standalone Python installation", Qgis.Success)
        return True, "Standalone Python removed"
    except Exception as e:
        error_msg = f"Failed to remove: {str(e)}"
        _log(error_msg, Qgis.Warning)
        return False, error_msg


def get_standalone_status() -> Tuple[bool, str]:
    """Get the status of the standalone Python installation."""
    if not standalone_python_exists():
        return False, "Python standalone not installed"

    success, msg = verify_standalone_python()
    if success:
        python_version = get_python_full_version()
        return True, f"Python {python_version} ready"
    else:
        return False, f"Python standalone incomplete: {msg}"
