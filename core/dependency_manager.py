"""
Dependency Manager for AI Segmentation

Handles checking and installation of required Python packages (onnxruntime, numpy).
Designed for stability - never crashes QGIS, provides clear feedback.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional
import importlib.util

from qgis.core import QgsMessageLog, Qgis, QgsSettings

# Required packages with minimum versions
REQUIRED_PACKAGES = [
    ("numpy", "1.20.0"),
    ("onnxruntime", "1.15.0"),
]

# Settings key for remembering user choice
SETTINGS_KEY_DEPS_DISMISSED = "AI_Segmentation/dependencies_dismissed"


def get_python_path() -> Optional[str]:
    """
    Get the correct Python executable path for QGIS.

    On macOS QGIS.app, sys.executable returns the QGIS app path, not Python!
    We need to find the actual Python inside the bundle.

    Supports multiple QGIS versions:
    - QGIS 3.44+: Python directly in MacOS/ folder (e.g., MacOS/python3.12)
    - Older QGIS: Python in MacOS/bin/ folder (e.g., MacOS/bin/python3)

    Returns:
        Path to Python executable, or None if not found
    """
    # On macOS with QGIS.app
    if sys.platform == "darwin":
        possible_paths = []

        # First, try to find Python version from sys.version_info
        py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"

        # QGIS 3.44+ structure: Python directly in MacOS folder
        # Try both QGIS.app and QGIS-LTR.app
        for app_name in ["QGIS.app", "QGIS-LTR.app"]:
            base = f"/Applications/{app_name}/Contents/MacOS"
            # Version-specific Python (e.g., python3.12)
            possible_paths.append(f"{base}/{py_version}")
            # Generic python3
            possible_paths.append(f"{base}/python3")
            # Just python
            possible_paths.append(f"{base}/python")
            # Old structure with bin/ subfolder
            possible_paths.append(f"{base}/bin/{py_version}")
            possible_paths.append(f"{base}/bin/python3")
            possible_paths.append(f"{base}/bin/python")

        # Homebrew QGIS
        possible_paths.extend([
            "/opt/homebrew/opt/qgis/bin/python3",
            "/usr/local/opt/qgis/bin/python3",
        ])

        # Also try to find it relative to sys.prefix
        if sys.prefix:
            prefix_python = Path(sys.prefix) / "bin" / "python3"
            possible_paths.insert(0, str(prefix_python))
            # Also check directly in prefix for newer structures
            prefix_python_versioned = Path(sys.prefix) / py_version
            possible_paths.insert(0, str(prefix_python_versioned))

        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                QgsMessageLog.logMessage(
                    f"Found Python at: {path}",
                    "AI Segmentation",
                    level=Qgis.Info
                )
                return path

        # Last resort: try to use sys.executable if it looks like Python
        if "python" in sys.executable.lower():
            return sys.executable

        return None

    # On Windows/Linux, sys.executable should work
    return sys.executable


def is_package_installed(package_name: str) -> bool:
    """
    Check if a package is installed and importable.

    Args:
        package_name: Name of the package to check

    Returns:
        True if package is installed and importable
    """
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except Exception:
        return False


def get_installed_version(package_name: str) -> Optional[str]:
    """
    Get the installed version of a package.

    Args:
        package_name: Name of the package

    Returns:
        Version string or None if not installed
    """
    try:
        if package_name == "onnxruntime":
            import onnxruntime
            return onnxruntime.__version__
        elif package_name == "numpy":
            import numpy
            return numpy.__version__
        else:
            from importlib.metadata import version
            return version(package_name)
    except Exception:
        return None


def check_dependencies() -> List[Tuple[str, str, bool, Optional[str]]]:
    """
    Check which required dependencies are installed.

    Returns:
        List of tuples: (package_name, required_version, is_installed, installed_version)
    """
    results = []
    for package_name, min_version in REQUIRED_PACKAGES:
        installed = is_package_installed(package_name)
        version = get_installed_version(package_name) if installed else None
        results.append((package_name, min_version, installed, version))
    return results


def get_missing_dependencies() -> List[Tuple[str, str]]:
    """
    Get list of missing dependencies.

    Returns:
        List of tuples: (package_name, required_version)
    """
    missing = []
    for package_name, min_version, installed, _ in check_dependencies():
        if not installed:
            missing.append((package_name, min_version))
    return missing


def all_dependencies_installed() -> bool:
    """Quick check if all dependencies are installed."""
    return len(get_missing_dependencies()) == 0


def get_manual_install_instructions() -> str:
    """
    Get manual installation instructions for the user.

    Returns:
        String with instructions to show the user
    """
    if sys.platform == "darwin":
        return """To install dependencies manually on macOS:

RECOMMENDED METHOD (QGIS Python Console):

1. In QGIS, go to: Plugins → Python Console
2. In the console, type:

import pip
pip.main(['install', '--user', 'numpy', 'onnxruntime'])

3. Restart QGIS

ALTERNATIVE (if you have an older QGIS version):

Open Terminal and run:
/Applications/QGIS.app/Contents/MacOS/bin/pip3 install numpy onnxruntime

Then restart QGIS."""

    elif sys.platform == "win32":
        return """To install dependencies manually:

1. Open OSGeo4W Shell (installed with QGIS)
2. Run: pip install numpy onnxruntime
3. Restart QGIS"""

    else:  # Linux
        return """To install dependencies manually, open terminal and run:

pip3 install --user numpy onnxruntime

Then restart QGIS."""


def install_package_via_pip_module(package_name: str, version: str = None) -> Tuple[bool, str]:
    """
    Install a package using pip module directly (works inside QGIS context).

    This method imports pip and calls it programmatically, which works
    even when the Python executable cannot be called from outside QGIS
    (common issue with QGIS 3.44+ on macOS using vcpkg).

    Args:
        package_name: Name of the package to install
        version: Optional version specifier

    Returns:
        Tuple of (success, message)
    """
    try:
        import pip
        from pip._internal.cli.main import main as pip_main
    except ImportError:
        try:
            # Older pip versions
            from pip import main as pip_main
        except ImportError:
            return False, "pip module not available"

    # Build the package specifier
    if version:
        package_spec = f"{package_name}>={version}"
    else:
        package_spec = package_name

    QgsMessageLog.logMessage(
        f"Installing {package_spec} via pip module...",
        "AI Segmentation",
        level=Qgis.Info
    )

    try:
        # Install with --user flag to avoid permission issues
        # Use --quiet to reduce output noise
        args = ["install", "--user", "--quiet", package_spec]

        # Capture return code
        return_code = pip_main(args)

        if return_code == 0:
            QgsMessageLog.logMessage(
                f"Successfully installed {package_spec}",
                "AI Segmentation",
                level=Qgis.Success
            )
            return True, f"Installed {package_spec}"
        else:
            return False, f"pip returned error code {return_code}"

    except Exception as e:
        QgsMessageLog.logMessage(
            f"Error installing {package_name}: {str(e)}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return False, f"Error: {str(e)[:200]}"


def install_package_via_subprocess(package_name: str, version: str = None) -> Tuple[bool, str]:
    """
    Install a package using subprocess (traditional method).

    This works on Windows and Linux where the Python executable
    can be called directly.

    Args:
        package_name: Name of the package to install
        version: Optional version specifier

    Returns:
        Tuple of (success, message)
    """
    python_path = get_python_path()

    if python_path is None:
        return False, "Could not find Python executable."

    try:
        # Build the package specifier
        if version:
            package_spec = f"{package_name}>={version}"
        else:
            package_spec = package_name

        QgsMessageLog.logMessage(
            f"Installing {package_spec} using {python_path}...",
            "AI Segmentation",
            level=Qgis.Info
        )

        # Build pip command
        cmd = [
            python_path,
            "-m",
            "pip",
            "install",
            "--user",
            package_spec
        ]

        # Run pip install with proper error handling
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )

        if result.returncode == 0:
            QgsMessageLog.logMessage(
                f"Successfully installed {package_spec}",
                "AI Segmentation",
                level=Qgis.Success
            )
            return True, f"Installed {package_spec}"
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            QgsMessageLog.logMessage(
                f"Failed to install {package_spec}: {error_msg}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            return False, f"Failed: {error_msg[:200]}"

    except subprocess.TimeoutExpired:
        return False, f"Installation timed out (5 min)"
    except FileNotFoundError:
        return False, f"Python not found at {python_path}"
    except Exception as e:
        QgsMessageLog.logMessage(
            f"Error installing {package_name}: {str(e)}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return False, f"Error: {str(e)[:200]}"


def install_package(package_name: str, version: str = None) -> Tuple[bool, str]:
    """
    Install a Python package using the best available method.

    On macOS with QGIS 3.44+ (vcpkg build), uses pip module directly.
    On other platforms, uses subprocess to call pip.

    Args:
        package_name: Name of the package to install
        version: Optional version specifier

    Returns:
        Tuple of (success, message)
    """
    # On macOS, prefer pip module method (works with vcpkg Python)
    if sys.platform == "darwin":
        QgsMessageLog.logMessage(
            "macOS detected: using pip module method for installation",
            "AI Segmentation",
            level=Qgis.Info
        )
        success, msg = install_package_via_pip_module(package_name, version)
        if success:
            return success, msg
        # Fallback to subprocess if pip module fails
        QgsMessageLog.logMessage(
            "pip module method failed, trying subprocess...",
            "AI Segmentation",
            level=Qgis.Info
        )
        return install_package_via_subprocess(package_name, version)

    # On Windows/Linux, use subprocess (usually works fine)
    return install_package_via_subprocess(package_name, version)


def install_all_dependencies(
    progress_callback=None
) -> Tuple[bool, List[str]]:
    """
    Install all missing dependencies.

    Args:
        progress_callback: Optional callback function(current, total, message)

    Returns:
        Tuple of (all_success, list of messages)
    """
    missing = get_missing_dependencies()

    if not missing:
        return True, ["All dependencies are already installed"]

    # Check if we can find Python first
    python_path = get_python_path()
    if python_path is None:
        return False, ["Cannot find Python executable. " + get_manual_install_instructions()]

    messages = []
    all_success = True
    total = len(missing)

    for i, (package_name, version) in enumerate(missing):
        if progress_callback:
            progress_callback(i, total, f"Installing {package_name}...")

        success, msg = install_package(package_name, version)
        messages.append(msg)

        if not success:
            all_success = False
            break  # Stop on first failure

    if progress_callback:
        if all_success:
            progress_callback(total, total, "Installation complete! Please restart QGIS.")
        else:
            progress_callback(total, total, "Installation failed. See details below.")

    return all_success, messages


def verify_installation() -> Tuple[bool, str]:
    """
    Verify that all dependencies are properly installed and importable.

    Returns:
        Tuple of (success, message)
    """
    try:
        import numpy as np
        import onnxruntime as ort

        # Quick sanity check
        _ = np.array([1, 2, 3])
        providers = ort.get_available_providers()

        return True, f"numpy {np.__version__}, onnxruntime {ort.__version__} (providers: {providers})"

    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except Exception as e:
        return False, f"Verification error: {str(e)}"


def was_install_dismissed() -> bool:
    """Check if user previously dismissed the install prompt."""
    settings = QgsSettings()
    return settings.value(SETTINGS_KEY_DEPS_DISMISSED, False, type=bool)


def set_install_dismissed(dismissed: bool):
    """Remember that user dismissed the install prompt."""
    settings = QgsSettings()
    settings.setValue(SETTINGS_KEY_DEPS_DISMISSED, dismissed)


def reset_install_dismissed():
    """Reset the dismissed state (for retry)."""
    set_install_dismissed(False)
