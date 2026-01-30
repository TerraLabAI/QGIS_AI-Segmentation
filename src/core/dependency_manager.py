import subprocess
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import importlib
import importlib.util

from qgis.core import QgsMessageLog, Qgis, QgsSettings


ALL_REQUIRED_PACKAGES = [
    ("torch", "torch", "2.0.0"),
    ("segment_anything", "segment-anything", "1.0"),
    ("pandas", "pandas", "1.3.0"),
    ("numpy", "numpy", "1.20.0,<2.0.0"),
    ("rasterio", "rasterio", "1.3.0"),
]

SETTINGS_KEY_DEPS_DISMISSED = "AI_Segmentation/dependencies_dismissed"

PYTHON_VERSION = sys.version_info
PLUGIN_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBS_DIR = os.path.join(PLUGIN_ROOT_DIR, 'libs')
PACKAGES_INSTALL_DIR = LIBS_DIR
PYTHON_VERSION_FILE = os.path.join(LIBS_DIR, '.python_version')

CACHE_DIR = os.path.expanduser("~/.qgis_ai_segmentation")


def _log(message: str, level=Qgis.Info):
    QgsMessageLog.logMessage(message, "AI Segmentation", level=level)


def migrate_from_old_directories() -> bool:
    """Remove old python3.X directories, return True if migration occurred."""
    old_dirs = []
    for version in ['python3.9', 'python3.10', 'python3.11', 'python3.12', 'python3.13']:
        old_dir = os.path.join(PLUGIN_ROOT_DIR, version)
        if os.path.exists(old_dir):
            old_dirs.append(old_dir)

    if not old_dirs:
        return False

    for old_dir in old_dirs:
        try:
            import shutil
            shutil.rmtree(old_dir)
            _log(f"Removed old directory: {old_dir}", Qgis.Info)
        except Exception as e:
            _log(f"Failed to remove {old_dir}: {e}", Qgis.Warning)

    return True


def check_python_version_match() -> bool:
    """Check if libs/ was built for current Python version."""
    if not os.path.exists(PYTHON_VERSION_FILE):
        return True

    try:
        with open(PYTHON_VERSION_FILE, 'r') as f:
            installed_version = f.read().strip()

        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        if installed_version != current_version:
            _log(
                f"Python version mismatch: libs/ built for {installed_version}, "
                f"running {current_version}. Reinstall required.",
                Qgis.Warning
            )
            return False

        return True
    except Exception:
        return True


def write_python_version():
    """Write current Python version marker file."""
    try:
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        with open(PYTHON_VERSION_FILE, 'w') as f:
            f.write(current_version)
    except Exception as e:
        _log(f"Failed to write version file: {e}", Qgis.Warning)


def clean_installation() -> bool:
    """Remove libs/ directory completely. User must reinstall."""
    try:
        if os.path.exists(LIBS_DIR):
            import shutil
            shutil.rmtree(LIBS_DIR)
            _log(f"Cleaned: {LIBS_DIR}", Qgis.Info)
            return True
    except Exception as e:
        _log(f"Cleanup failed: {e}", Qgis.Critical)
        return False
    return False


def _get_python_executable() -> str:
    """Get the correct Python executable path for pip installation.

    On macOS and Windows, sys.executable returns the QGIS executable, not Python.
    We use sys.prefix / 'bin' / 'python3' instead (following Geo-SAM/deepness approach).
    """
    if sys.platform == "darwin":
        python_path = Path(sys.prefix) / 'bin' / 'python3'
        if python_path.exists():
            _log(f"Using Python from sys.prefix: {python_path}", Qgis.Info)
            return str(python_path)
        _log(f"Warning: Python not found at {python_path}, using sys.executable", Qgis.Warning)
        return sys.executable

    elif sys.platform == "win32":
        python_path = os.path.join(os.path.dirname(sys.executable), "python.exe")
        if os.path.exists(python_path):
            return python_path
        python3_path = os.path.join(os.path.dirname(sys.executable), "python3.exe")
        if os.path.exists(python3_path):
            return python3_path
        return 'python'

    else:
        return sys.executable


def is_package_installed(import_name: str) -> bool:
    try:
        spec = importlib.util.find_spec(import_name)
        return spec is not None
    except Exception:
        return False


def get_installed_version(import_name: str) -> Optional[str]:
    try:
        if import_name == "numpy":
            import numpy
            return numpy.__version__
        elif import_name == "torch":
            import torch
            return torch.__version__
        elif import_name == "rasterio":
            import rasterio
            return rasterio.__version__
        elif import_name == "pandas":
            import pandas
            return pandas.__version__
        elif import_name == "segment_anything":
            return "installed"
        else:
            from importlib.metadata import version
            return version(import_name)
    except Exception:
        return None


def check_dependencies() -> List[Tuple[str, str, str, bool, Optional[str], bool]]:
    """Check if packages are installed AND isolated."""
    results = []

    for import_name, pip_name, min_version in ALL_REQUIRED_PACKAGES:
        installed = is_package_installed(import_name)
        version = get_installed_version(import_name) if installed else None

        isolated = False
        if installed:
            try:
                from .import_guard import verify_package_source
                isolated, _ = verify_package_source(import_name)
            except Exception:
                pass

        results.append((import_name, pip_name, min_version, installed, version, isolated))

    return results


def get_missing_dependencies() -> List[Tuple[str, str, str]]:
    """Get list of packages not installed in libs/ directory."""
    missing = []

    if not os.path.exists(LIBS_DIR):
        return ALL_REQUIRED_PACKAGES

    for import_name, pip_name, min_version in ALL_REQUIRED_PACKAGES:
        package_dir = os.path.join(LIBS_DIR, import_name)
        package_dir_alt = os.path.join(LIBS_DIR, import_name.replace('_', '-'))

        if not (os.path.exists(package_dir) or os.path.exists(package_dir_alt)):
            missing.append((import_name, pip_name, min_version))

    return missing


def all_dependencies_installed() -> bool:
    """Check if all packages are installed in libs/ directory."""
    if not os.path.exists(LIBS_DIR):
        _log("libs/ directory does not exist", Qgis.Info)
        return False

    libs_contents = os.listdir(LIBS_DIR)
    if not libs_contents or libs_contents == ['.python_version']:
        _log("libs/ directory is empty", Qgis.Info)
        return False

    for import_name, _, _ in ALL_REQUIRED_PACKAGES:
        package_dir = os.path.join(LIBS_DIR, import_name)
        package_dir_alt = os.path.join(LIBS_DIR, import_name.replace('_', '-'))

        if not (os.path.exists(package_dir) or os.path.exists(package_dir_alt)):
            _log(f"Package {import_name} not found in libs/", Qgis.Info)
            return False

    return True


def get_install_size_warning() -> str:
    missing = get_missing_dependencies()
    has_torch = any(pip_name == "torch" for _, pip_name, _ in missing)

    if has_torch:
        return (
            "⚠️ PyTorch (~2GB) will be downloaded.\n"
            "This may take several minutes depending on your connection."
        )
    return ""


def ensure_packages_dir_in_path():
    os.makedirs(LIBS_DIR, exist_ok=True)
    if LIBS_DIR not in sys.path:
        sys.path.insert(0, LIBS_DIR)
        _log(f"Added packages directory to sys.path: {LIBS_DIR}")


def _run_pip_install(pip_name: str, version: str = None, target_dir: str = None) -> Tuple[bool, str]:
    python_exe = _get_python_executable()

    if version:
        package_spec = f"{pip_name}>={version}"
    else:
        package_spec = pip_name

    cmd = [
        python_exe, "-m", "pip", "install",
        "--upgrade",
        "--no-warn-script-location",
        "--disable-pip-version-check",
    ]

    if target_dir:
        cmd.extend([f"--target={target_dir}"])

    cmd.append(package_spec)

    _log(f"Running: {' '.join(cmd)}")

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
                startupinfo=startupinfo,
            )
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )

        if result.returncode == 0:
            _log(f"✓ Successfully installed {package_spec}", Qgis.Success)
            if result.stdout:
                _log(f"pip stdout: {result.stdout[:500]}")
            return True, f"Installed {package_spec}"
        else:
            _log(f"✗ pip install failed with returncode {result.returncode}", Qgis.Warning)
            if result.stdout:
                _log(f"pip stdout: {result.stdout[:1000]}", Qgis.Warning)
            if result.stderr:
                _log(f"pip stderr: {result.stderr[:1000]}", Qgis.Warning)
            error_msg = result.stderr or result.stdout or f"Return code {result.returncode}"
            return False, f"pip error: {error_msg[:300]}"

    except subprocess.TimeoutExpired:
        _log(f"✗ Installation timed out for {package_spec}", Qgis.Warning)
        return False, "Installation timed out (10 minutes)"
    except FileNotFoundError:
        _log(f"✗ Python executable not found: {python_exe}", Qgis.Warning)
        return False, f"Python not found: {python_exe}"
    except Exception as e:
        _log(f"✗ Exception during install of {package_spec}: {str(e)}", Qgis.Warning)
        return False, f"Error: {str(e)[:200]}"


def install_package(pip_name: str, version: str = None) -> Tuple[bool, str]:
    ensure_packages_dir_in_path()

    _log(f"Installing {pip_name} to plugin directory: {PACKAGES_INSTALL_DIR}")
    success, msg = _run_pip_install(pip_name, version, PACKAGES_INSTALL_DIR)

    if success:
        try:
            import_name = pip_name.replace("-", "_")
            if import_name in sys.modules:
                del sys.modules[import_name]
            importlib.invalidate_caches()

            if is_package_installed(import_name):
                from .import_guard import verify_package_source
                is_isolated, source = verify_package_source(import_name)

                if is_isolated:
                    _log(f"✓ {import_name} isolated: {source}", Qgis.Success)
                else:
                    _log(f"⚠ {import_name} NOT isolated: {source}", Qgis.Warning)
            else:
                _log("Package installed but import check failed - may need restart", Qgis.Warning)
        except Exception as e:
            _log(f"Post-install verification error (non-fatal): {str(e)}")

    return success, msg


def install_all_dependencies(
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Tuple[bool, List[str]]:
    ensure_packages_dir_in_path()

    missing = get_missing_dependencies()

    if not missing:
        _log("All dependencies are already installed", Qgis.Success)
        return True, ["All dependencies are already installed"]

    _log("=" * 50)
    _log(f"Installing {len(missing)} packages to: {PACKAGES_INSTALL_DIR}")
    _log(f"Python executable: {_get_python_executable()}")
    _log(f"Platform: {sys.platform}")
    _log("=" * 50)

    if progress_callback:
        packages_str = ", ".join([pip_name for _, pip_name, _ in missing])
        progress_callback(0, len(missing), f"Installing: {packages_str}")

    messages = []
    all_success = True
    total = len(missing)

    for i, (import_name, pip_name, version) in enumerate(missing):
        if cancel_check and cancel_check():
            _log("Installation cancelled by user", Qgis.Warning)
            return False, messages + ["Installation cancelled"]

        if progress_callback:
            if pip_name == "torch":
                progress_callback(i, total, f"Installing {pip_name} (~2GB)... ({i + 1}/{total})")
            else:
                progress_callback(i, total, f"Installing {pip_name}... ({i + 1}/{total})")

        _log(f"[{i + 1}/{total}] Installing {pip_name}>={version}...")

        success, msg = install_package(pip_name, version)
        messages.append(f"{pip_name}: {msg}")

        if success:
            if progress_callback:
                progress_callback(i + 1, total, f"✓ {pip_name} installed")
        else:
            all_success = False
            if progress_callback:
                progress_callback(i + 1, total, f"✗ {pip_name} failed: {msg[:50]}")
            break

    if progress_callback:
        if all_success:
            progress_callback(total, total, "✓ Installation complete!")
        else:
            progress_callback(total, total, "✗ Installation failed - see logs")

    if all_success:
        write_python_version()
        _log("=" * 50)
        _log("All dependencies installed successfully!", Qgis.Success)
        _log(f"Install location: {PACKAGES_INSTALL_DIR}")
        _log("Restart QGIS to ensure all packages are properly loaded.")
        _log("=" * 50)

    return all_success, messages


def verify_installation() -> Tuple[bool, str]:
    try:
        import numpy as np
        numpy_ver = np.__version__

        import torch
        torch_ver = torch.__version__

        import rasterio
        rasterio_ver = rasterio.__version__

        import segment_anything  # noqa: F401 - verify import works

        _ = np.array([1, 2, 3])

        _log(f"Verification OK: numpy {numpy_ver}, torch {torch_ver}, rasterio {rasterio_ver}", Qgis.Success)
        return True, f"numpy {numpy_ver}, torch {torch_ver}"

    except ImportError as e:
        _log(f"Verification failed - import error: {str(e)}", Qgis.Warning)
        return False, f"Import error: {str(e)}"
    except Exception as e:
        _log(f"Verification failed: {str(e)}", Qgis.Warning)
        return False, f"Error: {str(e)}"


def was_install_dismissed() -> bool:
    settings = QgsSettings()
    return settings.value(SETTINGS_KEY_DEPS_DISMISSED, False, type=bool)


def set_install_dismissed(dismissed: bool):
    settings = QgsSettings()
    settings.setValue(SETTINGS_KEY_DEPS_DISMISSED, dismissed)


def reset_install_dismissed():
    set_install_dismissed(False)


def init_packages_path():
    """Initialize package paths and handle migration."""
    if migrate_from_old_directories():
        _log("Migrated from old directory structure. Please reinstall dependencies.", Qgis.Warning)

    os.makedirs(LIBS_DIR, exist_ok=True)

    if LIBS_DIR not in sys.path:
        sys.path.insert(0, LIBS_DIR)


def get_packages_install_dir() -> str:
    return PACKAGES_INSTALL_DIR


def get_cache_dir() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def get_dependency_status_summary() -> str:
    deps = check_dependencies()
    installed = [(pip_name, ver) for _, pip_name, _, is_installed, ver, _ in deps if is_installed]
    missing = [(pip_name, req_ver) for _, pip_name, req_ver, is_installed, _, _ in deps if not is_installed]

    if not missing:
        versions = ", ".join([f"{name} {ver}" for name, ver in installed])
        return f"OK: {versions}"
    else:
        missing_str = ", ".join([name for name, _ in missing])
        return f"Missing: {missing_str}"


def get_system_info() -> str:
    info = []
    info.append(f"Platform: {sys.platform}")
    info.append(f"Python: {sys.version}")
    info.append(f"Python executable: {_get_python_executable()}")
    info.append(f"Package install dir: {LIBS_DIR}")
    info.append(f"sys.path includes install dir: {LIBS_DIR in sys.path}")
    return "\n".join(info)
