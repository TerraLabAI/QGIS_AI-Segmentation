"""
Dependency Manager for AI Segmentation

Handles automatic installation of required Python packages (onnxruntime, numpy)
that are not bundled with QGIS by default.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import importlib.util

from qgis.core import QgsMessageLog, Qgis

# Required packages with minimum versions
REQUIRED_PACKAGES = [
    ("numpy", "1.20.0"),
    ("onnxruntime", "1.15.0"),
]


def get_pip_path() -> str:
    """
    Get the path to pip executable for the current Python environment.

    Returns:
        Path to pip executable
    """
    # Use the same Python that QGIS is using
    return sys.executable


def is_package_installed(package_name: str) -> bool:
    """
    Check if a package is installed and importable.

    Args:
        package_name: Name of the package to check

    Returns:
        True if package is installed and importable
    """
    spec = importlib.util.find_spec(package_name)
    return spec is not None


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
            # Generic approach using importlib.metadata
            from importlib.metadata import version
            return version(package_name)
    except Exception:
        return None


def check_dependencies() -> List[Tuple[str, str, bool]]:
    """
    Check which required dependencies are installed.

    Returns:
        List of tuples: (package_name, required_version, is_installed)
    """
    results = []
    for package_name, min_version in REQUIRED_PACKAGES:
        installed = is_package_installed(package_name)
        results.append((package_name, min_version, installed))
    return results


def get_missing_dependencies() -> List[Tuple[str, str]]:
    """
    Get list of missing dependencies.

    Returns:
        List of tuples: (package_name, required_version)
    """
    missing = []
    for package_name, min_version, installed in check_dependencies():
        if not installed:
            missing.append((package_name, min_version))
    return missing


def install_package(package_name: str, version: str = None) -> Tuple[bool, str]:
    """
    Install a Python package using pip.

    Args:
        package_name: Name of the package to install
        version: Optional version specifier

    Returns:
        Tuple of (success, message)
    """
    try:
        # Build the package specifier
        if version:
            package_spec = f"{package_name}>={version}"
        else:
            package_spec = package_name

        QgsMessageLog.logMessage(
            f"Installing {package_spec}...",
            "AI Segmentation",
            level=Qgis.Info
        )

        # Run pip install
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--user",  # Install to user directory to avoid permission issues
                "--quiet",
                package_spec
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            QgsMessageLog.logMessage(
                f"Successfully installed {package_spec}",
                "AI Segmentation",
                level=Qgis.Success
            )
            return True, f"Successfully installed {package_spec}"
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            QgsMessageLog.logMessage(
                f"Failed to install {package_spec}: {error_msg}",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return False, f"Failed to install {package_spec}: {error_msg}"

    except subprocess.TimeoutExpired:
        return False, f"Installation of {package_name} timed out"
    except Exception as e:
        return False, f"Error installing {package_name}: {str(e)}"


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

    if progress_callback:
        progress_callback(total, total, "Installation complete")

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
        _ = ort.get_available_providers()

        return True, f"Dependencies OK: numpy {np.__version__}, onnxruntime {ort.__version__}"

    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except Exception as e:
        return False, f"Verification error: {str(e)}"
