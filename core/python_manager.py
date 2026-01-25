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
import urllib.request
import shutil
from typing import Tuple, Optional, Callable

from qgis.core import QgsMessageLog, Qgis


PLUGIN_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STANDALONE_DIR = os.path.join(PLUGIN_ROOT_DIR, "python_standalone")

# Release tag from python-build-standalone
# Update this periodically to get newer Python builds
RELEASE_TAG = "20241219"

# Mapping of Python minor versions to their latest patch versions in the release
PYTHON_VERSIONS = {
    (3, 9): "3.9.21",
    (3, 10): "3.10.16",
    (3, 11): "3.11.11",
    (3, 12): "3.12.8",
    (3, 13): "3.13.1",
}


def _log(message: str, level=Qgis.Info):
    QgsMessageLog.logMessage(message, "AI Segmentation", level=level)


def get_qgis_python_version() -> Tuple[int, int]:
    """Get the Python version used by QGIS."""
    return (sys.version_info.major, sys.version_info.minor)


def get_python_full_version() -> str:
    """Get the full Python version string for download (e.g., '3.12.8')."""
    version_tuple = get_qgis_python_version()
    if version_tuple in PYTHON_VERSIONS:
        return PYTHON_VERSIONS[version_tuple]
    # Fallback: construct version string (may not exist in release)
    return f"{version_tuple[0]}.{version_tuple[1]}.0"


def get_standalone_dir() -> str:
    """Get the directory where Python standalone is installed."""
    return STANDALONE_DIR


def get_standalone_python_path() -> str:
    """Get the path to the standalone Python executable."""
    python_version = get_python_full_version()
    python_dir = os.path.join(STANDALONE_DIR, "python")

    if sys.platform == "win32":
        return os.path.join(python_dir, "python.exe")
    else:
        return os.path.join(python_dir, "bin", "python3")


def standalone_python_exists() -> bool:
    """Check if standalone Python is already installed."""
    python_path = get_standalone_python_path()
    return os.path.exists(python_path)


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
    Download and install Python standalone.

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
        # Download with progress reporting
        def report_progress(block_count, block_size, total_size):
            if cancel_check and cancel_check():
                raise InterruptedError("Download cancelled")

            if total_size > 0:
                downloaded = block_count * block_size
                percent = min(int((downloaded / total_size) * 100), 100)  # 0-100% for download
                size_mb = total_size / (1024 * 1024)
                downloaded_mb = min(downloaded, total_size) / (1024 * 1024)
                if progress_callback:
                    progress_callback(percent, f"Downloading Python {python_version}... ({downloaded_mb:.1f}/{size_mb:.1f} MB)")

        urllib.request.urlretrieve(url, temp_path, reporthook=report_progress)

        if cancel_check and cancel_check():
            return False, "Download cancelled"

        _log(f"Download complete, extracting...", Qgis.Info)

        if progress_callback:
            progress_callback(55, "Extracting Python...")

        # Remove existing standalone dir if it exists
        if os.path.exists(STANDALONE_DIR):
            shutil.rmtree(STANDALONE_DIR)

        os.makedirs(STANDALONE_DIR, exist_ok=True)

        # Extract archive
        if temp_path.endswith(".tar.gz") or temp_path.endswith(".tgz"):
            with tarfile.open(temp_path, "r:gz") as tar:
                tar.extractall(STANDALONE_DIR)
        else:
            with zipfile.ZipFile(temp_path, "r") as z:
                z.extractall(STANDALONE_DIR)

        if progress_callback:
            progress_callback(80, "Verifying Python installation...")

        # Verify installation
        success, verify_msg = verify_standalone_python()

        if success:
            if progress_callback:
                progress_callback(100, f"✓ Python {python_version} installed")
            _log(f"Python standalone installed successfully", Qgis.Success)
            return True, f"Python {python_version} installed successfully"
        else:
            return False, f"Verification failed: {verify_msg}"

    except urllib.error.HTTPError as e:
        if e.code == 404:
            error_msg = f"Python {python_version} not available for this platform. URL: {url}"
        else:
            error_msg = f"Download failed (HTTP {e.code}): {str(e)}"
        _log(error_msg, Qgis.Critical)
        return False, error_msg
    except urllib.error.URLError as e:
        error_msg = f"Network error: {str(e)}. Check your internet connection."
        _log(error_msg, Qgis.Critical)
        return False, error_msg
    except InterruptedError:
        return False, "Download cancelled"
    except Exception as e:
        error_msg = f"Installation failed: {str(e)}"
        _log(error_msg, Qgis.Critical)
        return False, error_msg
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def verify_standalone_python() -> Tuple[bool, str]:
    """Verify that the standalone Python installation works."""
    python_path = get_standalone_python_path()

    if not os.path.exists(python_path):
        return False, f"Python executable not found at {python_path}"

    # On Unix, make sure it's executable
    if sys.platform != "win32":
        try:
            os.chmod(python_path, 0o755)
        except:
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
