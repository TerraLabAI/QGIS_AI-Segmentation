import subprocess
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import importlib.util

from qgis.core import QgsMessageLog, Qgis, QgsSettings


REQUIRED_PACKAGES = [
    ("torch", "torch", "2.0.0"),
    ("torchvision", "torchvision", "0.15.0"),
    ("rtree", "rtree", "1.0.0"),
    ("pandas", "pandas", "1.3.0"),
    ("segment_anything", "segment-anything", "1.0"),
    ("torchgeo", "torchgeo", "0.5.0"),
]

QGIS_PROVIDED_PACKAGES = [
    ("numpy", "numpy"),
    ("rasterio", "rasterio"),
]

SETTINGS_KEY_DEPS_DISMISSED = "AI_Segmentation/dependencies_dismissed"

PYTHON_VERSION = sys.version_info
PLUGIN_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES_INSTALL_DIR = os.path.join(PLUGIN_ROOT_DIR, f'python{PYTHON_VERSION.major}.{PYTHON_VERSION.minor}')

CACHE_DIR = os.path.expanduser("~/.qgis_ai_segmentation")
CHECKPOINTS_DIR = os.path.join(CACHE_DIR, "checkpoints")

SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"


def is_package_installed(import_name: str) -> bool:
    try:
        spec = importlib.util.find_spec(import_name)
        return spec is not None
    except Exception:
        return False


def get_installed_version(import_name: str) -> Optional[str]:
    try:
        if import_name == "torch":
            import torch
            return torch.__version__
        elif import_name == "torchvision":
            import torchvision
            return torchvision.__version__
        elif import_name == "numpy":
            import numpy
            return numpy.__version__
        elif import_name == "rasterio":
            import rasterio
            return rasterio.__version__
        elif import_name == "segment_anything":
            import segment_anything
            return "installed"
        elif import_name == "torchgeo":
            import torchgeo
            return torchgeo.__version__
        elif import_name == "rtree":
            import rtree
            return rtree.__version__
        elif import_name == "pandas":
            import pandas
            return pandas.__version__
        else:
            from importlib.metadata import version
            return version(import_name)
    except Exception:
        return None


def check_dependencies() -> List[Tuple[str, str, str, bool, Optional[str]]]:
    results = []
    for import_name, pip_name, min_version in REQUIRED_PACKAGES:
        installed = is_package_installed(import_name)
        version = get_installed_version(import_name) if installed else None
        results.append((import_name, pip_name, min_version, installed, version))
    return results


def get_missing_dependencies() -> List[Tuple[str, str, str]]:
    missing = []
    for import_name, pip_name, min_version, installed, _ in check_dependencies():
        if not installed:
            missing.append((import_name, pip_name, min_version))
    return missing


def all_dependencies_installed() -> bool:
    if len(get_missing_dependencies()) > 0:
        return False
    for import_name, _ in QGIS_PROVIDED_PACKAGES:
        if not is_package_installed(import_name):
            return False
    return True


def get_manual_install_instructions() -> str:
    target_dir = PACKAGES_INSTALL_DIR

    if sys.platform == "darwin":
        return f"""Open Terminal and run:
pip3 install --target="{target_dir}" torch torchvision rtree pandas segment-anything torchgeo"""

    elif sys.platform == "win32":
        return f"""Open Command Prompt and run:
pip install --target="{target_dir}" torch torchvision rtree pandas segment-anything torchgeo"""

    else:
        return f"""Open terminal and run:
pip3 install --target="{target_dir}" torch torchvision rtree pandas segment-anything torchgeo"""


def ensure_packages_dir_in_path():
    os.makedirs(PACKAGES_INSTALL_DIR, exist_ok=True)
    if PACKAGES_INSTALL_DIR not in sys.path:
        sys.path.insert(0, PACKAGES_INSTALL_DIR)
        QgsMessageLog.logMessage(
            f"Added packages directory to sys.path: {PACKAGES_INSTALL_DIR}",
            "AI Segmentation",
            level=Qgis.Info
        )


def install_package_via_pip_module(pip_name: str, version: str = None) -> Tuple[bool, str]:
    try:
        from pip._internal.cli.main import main as pip_main
    except ImportError:
        try:
            from pip import main as pip_main
        except ImportError:
            return False, "pip module not available"

    ensure_packages_dir_in_path()

    if version:
        package_spec = f"{pip_name}>={version}"
    else:
        package_spec = pip_name

    QgsMessageLog.logMessage(
        f"Installing {package_spec} via pip module to {PACKAGES_INSTALL_DIR}...",
        "AI Segmentation",
        level=Qgis.Info
    )

    try:
        old_argv = sys.argv
        sys.argv = ['pip']

        args = [
            "install",
            "--upgrade",
            f"--target={PACKAGES_INSTALL_DIR}",
            "--no-warn-script-location",
            "--disable-pip-version-check",
            "--no-python-version-warning",
            package_spec
        ]

        return_code = pip_main(args)

        sys.argv = old_argv

        if return_code == 0:
            QgsMessageLog.logMessage(
                f"✓ Successfully installed {package_spec}",
                "AI Segmentation",
                level=Qgis.Success
            )
            return True, f"Installed {package_spec}"
        else:
            QgsMessageLog.logMessage(
                f"✗ pip returned error code {return_code} for {package_spec}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            return False, f"pip returned error code {return_code}"

    except SystemExit as e:
        if e.code == 0:
            return True, f"Installed {package_spec}"
        return False, f"pip exited with code {e.code}"
    except Exception as e:
        QgsMessageLog.logMessage(
            f"Error installing {pip_name}: {str(e)}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return False, f"Error: {str(e)[:200]}"


def install_all_dependencies(
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[bool, List[str]]:
    ensure_packages_dir_in_path()

    missing = get_missing_dependencies()

    if not missing:
        return True, ["All dependencies are already installed"]

    QgsMessageLog.logMessage(
        f"Installing {len(missing)} packages to: {PACKAGES_INSTALL_DIR}",
        "AI Segmentation",
        level=Qgis.Info
    )

    if progress_callback:
        packages_str = ", ".join([pip_name for _, pip_name, _ in missing])
        progress_callback(0, len(missing), f"Installing: {packages_str}")

    messages = []
    all_success = True
    total = len(missing)

    for i, (import_name, pip_name, version) in enumerate(missing):
        if progress_callback:
            progress_callback(i, total, f"Installing {pip_name}... ({i+1}/{total})")

        success, msg = install_package_via_pip_module(pip_name, version)
        messages.append(f"{pip_name}: {msg}")

        if success:
            QgsMessageLog.logMessage(
                f"✓ {pip_name} installed successfully",
                "AI Segmentation",
                level=Qgis.Success
            )
            if progress_callback:
                progress_callback(i + 1, total, f"✓ {pip_name} installed")
        else:
            QgsMessageLog.logMessage(
                f"✗ {pip_name} installation failed: {msg}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            all_success = False
            if progress_callback:
                progress_callback(i + 1, total, f"✗ {pip_name} failed")
            break

    if progress_callback:
        if all_success:
            progress_callback(total, total, "✓ Installation complete! Please restart QGIS.")
        else:
            progress_callback(total, total, "✗ Installation failed. See details below.")

    return all_success, messages


def verify_installation() -> Tuple[bool, str]:
    try:
        import numpy as np
        import torch
        import rasterio

        _ = np.array([1, 2, 3])
        _ = torch.tensor([1, 2, 3])

        return True, f"numpy {np.__version__}, torch {torch.__version__}, rasterio {rasterio.__version__}"

    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except Exception as e:
        return False, f"Verification error: {str(e)}"


def was_install_dismissed() -> bool:
    settings = QgsSettings()
    return settings.value(SETTINGS_KEY_DEPS_DISMISSED, False, type=bool)


def set_install_dismissed(dismissed: bool):
    settings = QgsSettings()
    settings.setValue(SETTINGS_KEY_DEPS_DISMISSED, dismissed)


def reset_install_dismissed():
    set_install_dismissed(False)


def init_packages_path():
    os.makedirs(PACKAGES_INSTALL_DIR, exist_ok=True)
    if PACKAGES_INSTALL_DIR not in sys.path:
        sys.path.insert(0, PACKAGES_INSTALL_DIR)


def get_packages_install_dir() -> str:
    return PACKAGES_INSTALL_DIR


def get_cache_dir() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def get_checkpoints_dir() -> str:
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    return CHECKPOINTS_DIR


def get_dependency_status_summary() -> str:
    deps = check_dependencies()
    installed = [(pip_name, ver) for _, pip_name, _, is_installed, ver in deps if is_installed]
    missing = [(pip_name, req_ver) for _, pip_name, req_ver, is_installed, _ in deps if not is_installed]

    if not missing:
        versions = ", ".join([f"{name} {ver}" for name, ver in installed])
        return f"OK: {versions}"
    else:
        missing_str = ", ".join([name for name, _ in missing])
        return f"Missing: {missing_str}"
