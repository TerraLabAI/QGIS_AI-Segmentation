import subprocess
import sys
import os
import shutil
from pathlib import Path
from typing import Tuple, Optional, Callable

from qgis.core import QgsMessageLog, Qgis


PLUGIN_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_VERSION = f"py{sys.version_info.major}.{sys.version_info.minor}"
VENV_DIR = os.path.join(PLUGIN_ROOT_DIR, f'venv_{PYTHON_VERSION}')
LIBS_DIR = os.path.join(PLUGIN_ROOT_DIR, 'libs')

REQUIRED_PACKAGES = [
    ("numpy", "<2.0,>=1.20.0"),
    ("torch", ">=2.0.0"),
    ("torchvision", ">=0.15.0"),
    ("segment-anything", ">=1.0"),
    ("pandas", ">=1.3.0"),
    ("rasterio", ">=1.3.0"),
]


def _log(message: str, level=Qgis.Info):
    QgsMessageLog.logMessage(message, "AI Segmentation", level=level)


def get_venv_dir() -> str:
    return VENV_DIR


def get_venv_site_packages(venv_dir: str = None) -> str:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        return os.path.join(venv_dir, "Lib", "site-packages")
    else:
        py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        return os.path.join(venv_dir, "lib", py_version, "site-packages")


def ensure_venv_packages_available():
    if not venv_exists():
        return False

    site_packages = get_venv_site_packages()
    if not os.path.exists(site_packages):
        _log(f"Venv site-packages not found: {site_packages}", Qgis.Warning)
        return False

    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)
        _log(f"Added venv site-packages to sys.path: {site_packages}", Qgis.Info)

    return True


def get_venv_python_path(venv_dir: str = None) -> str:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        return os.path.join(venv_dir, "bin", "python3")


def get_venv_pip_path(venv_dir: str = None) -> str:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        return os.path.join(venv_dir, "bin", "pip")


def _get_system_python() -> str:
    py_major = sys.version_info.major
    py_minor = sys.version_info.minor

    if sys.platform == "darwin":
        candidates = [
            f"/opt/homebrew/bin/python{py_major}.{py_minor}",
            f"/usr/local/bin/python{py_major}.{py_minor}",
            f"/opt/homebrew/bin/python{py_major}",
            f"/usr/local/bin/python{py_major}",
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                _log(f"Found Python at: {candidate}", Qgis.Info)
                return candidate

        _log(f"No compatible Python {py_major}.{py_minor} found via Homebrew", Qgis.Warning)
        return "python3"

    elif sys.platform == "win32":
        candidates = []

        osgeo4w_root = os.environ.get("OSGEO4W_ROOT", r"C:\OSGeo4W")
        candidates.append(os.path.join(osgeo4w_root, "apps", f"Python{py_major}{py_minor}", "python.exe"))
        candidates.append(os.path.join(osgeo4w_root, "bin", "python.exe"))

        qgis_dir = os.path.dirname(sys.executable)
        candidates.append(os.path.join(qgis_dir, "python.exe"))
        candidates.append(os.path.join(qgis_dir, "python3.exe"))

        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            candidates.append(os.path.join(local_app_data, "Programs", "Python", f"Python{py_major}{py_minor}", "python.exe"))

        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        candidates.append(os.path.join(program_files, f"Python{py_major}{py_minor}", "python.exe"))
        candidates.append(os.path.join(program_files, "Python", f"Python{py_major}{py_minor}", "python.exe"))

        for candidate in candidates:
            if os.path.exists(candidate):
                _log(f"Found Python at: {candidate}", Qgis.Info)
                return candidate

        _log(f"No Python {py_major}.{py_minor} found, trying system PATH", Qgis.Warning)
        return "python"

    else:
        return sys.executable


def venv_exists(venv_dir: str = None) -> bool:
    if venv_dir is None:
        venv_dir = VENV_DIR

    python_path = get_venv_python_path(venv_dir)
    return os.path.exists(python_path)


def create_venv(venv_dir: str = None, progress_callback: Optional[Callable[[int, str], None]] = None) -> Tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    _log(f"Creating virtual environment at: {venv_dir}", Qgis.Info)

    if progress_callback:
        progress_callback(10, "Creating virtual environment...")

    system_python = _get_system_python()
    _log(f"Using Python: {system_python}", Qgis.Info)

    cmd = [system_python, "-m", "venv", venv_dir]

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
                timeout=120,
                env=env,
                startupinfo=startupinfo,
            )
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )

        if result.returncode == 0:
            _log(f"Virtual environment created successfully", Qgis.Success)
            if progress_callback:
                progress_callback(20, "Virtual environment created")
            return True, "Virtual environment created"
        else:
            error_msg = result.stderr or result.stdout or f"Return code {result.returncode}"
            _log(f"Failed to create venv: {error_msg}", Qgis.Critical)
            return False, f"Failed to create venv: {error_msg[:200]}"

    except subprocess.TimeoutExpired:
        _log("Virtual environment creation timed out", Qgis.Critical)
        return False, "Virtual environment creation timed out"
    except FileNotFoundError:
        _log(f"Python executable not found: {system_python}", Qgis.Critical)
        return False, f"Python not found: {system_python}"
    except Exception as e:
        _log(f"Exception during venv creation: {str(e)}", Qgis.Critical)
        return False, f"Error: {str(e)[:200]}"


def install_dependencies(
    venv_dir: str = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not venv_exists(venv_dir):
        return False, "Virtual environment does not exist"

    pip_path = get_venv_pip_path(venv_dir)
    _log(f"Installing dependencies using: {pip_path}", Qgis.Info)

    total_packages = len(REQUIRED_PACKAGES)
    base_progress = 20
    progress_per_package = 80 // total_packages

    for i, (package_name, version_spec) in enumerate(REQUIRED_PACKAGES):
        if cancel_check and cancel_check():
            _log("Installation cancelled by user", Qgis.Warning)
            return False, "Installation cancelled"

        package_spec = f"{package_name}{version_spec}"
        current_progress = base_progress + (i * progress_per_package)

        if progress_callback:
            if package_name == "torch":
                progress_callback(current_progress, f"Installing {package_name} (~2GB)... ({i+1}/{total_packages})")
            else:
                progress_callback(current_progress, f"Installing {package_name}... ({i+1}/{total_packages})")

        _log(f"[{i+1}/{total_packages}] Installing {package_spec}...", Qgis.Info)

        cmd = [
            pip_path, "install",
            "--upgrade",
            "--no-warn-script-location",
            "--disable-pip-version-check",
            package_spec
        ]

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
                if progress_callback:
                    progress_callback(current_progress + progress_per_package, f"✓ {package_name} installed")
            else:
                error_msg = result.stderr or result.stdout or f"Return code {result.returncode}"
                _log(f"✗ Failed to install {package_spec}: {error_msg[:500]}", Qgis.Critical)
                return False, f"Failed to install {package_name}: {error_msg[:200]}"

        except subprocess.TimeoutExpired:
            _log(f"Installation of {package_spec} timed out", Qgis.Critical)
            return False, f"Installation of {package_name} timed out"
        except Exception as e:
            _log(f"Exception during installation of {package_spec}: {str(e)}", Qgis.Critical)
            return False, f"Error installing {package_name}: {str(e)[:200]}"

    if progress_callback:
        progress_callback(100, "✓ All dependencies installed")

    _log("=" * 50, Qgis.Success)
    _log("All dependencies installed successfully!", Qgis.Success)
    _log(f"Virtual environment: {venv_dir}", Qgis.Success)
    _log("=" * 50, Qgis.Success)

    return True, "All dependencies installed successfully"


def _get_clean_env_for_venv() -> dict:
    env = os.environ.copy()

    vars_to_remove = [
        'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV',
        'QGIS_PREFIX_PATH', 'QGIS_PLUGINPATH',
    ]
    for var in vars_to_remove:
        env.pop(var, None)

    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _get_subprocess_kwargs() -> dict:
    kwargs = {}
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs['startupinfo'] = startupinfo
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
    return kwargs


def verify_venv(venv_dir: str = None) -> Tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not venv_exists(venv_dir):
        return False, "Virtual environment not found"

    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    for package_name, _ in REQUIRED_PACKAGES:
        import_name = package_name.replace("-", "_")

        cmd = [python_path, "-c", f"import {import_name}"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                **subprocess_kwargs
            )

            if result.returncode != 0:
                _log(f"Package {package_name} not importable in venv: {result.stderr[:200] if result.stderr else ''}", Qgis.Warning)
                return False, f"Missing package: {package_name}"

        except Exception as e:
            _log(f"Failed to verify {package_name}: {str(e)}", Qgis.Warning)
            return False, f"Verification error: {package_name}"

    _log("✓ Virtual environment verified successfully", Qgis.Success)
    return True, "Virtual environment ready"


def cleanup_old_libs() -> bool:
    if not os.path.exists(LIBS_DIR):
        return False

    _log("Detected old 'libs/' installation. Cleaning up...", Qgis.Info)

    try:
        shutil.rmtree(LIBS_DIR)
        _log("Old libs/ directory removed successfully", Qgis.Success)
        return True
    except Exception as e:
        _log(f"Failed to remove libs/: {e}. Please delete manually.", Qgis.Warning)
        return False


def create_venv_and_install(
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Tuple[bool, str]:
    cleanup_old_libs()

    if venv_exists():
        _log("Virtual environment already exists", Qgis.Info)
    else:
        success, msg = create_venv(progress_callback=progress_callback)
        if not success:
            return False, msg

        if cancel_check and cancel_check():
            return False, "Installation cancelled"

    success, msg = install_dependencies(
        progress_callback=progress_callback,
        cancel_check=cancel_check
    )

    if not success:
        return False, msg

    is_valid, verify_msg = verify_venv()
    if not is_valid:
        return False, f"Verification failed: {verify_msg}"

    return True, "Virtual environment ready"


def get_venv_status() -> Tuple[bool, str]:
    if not venv_exists():
        if os.path.exists(LIBS_DIR):
            return False, "Old installation detected. Migration required."
        return False, "Virtual environment not configured"

    is_valid, msg = verify_venv()
    if is_valid:
        return True, "Virtual environment ready"
    else:
        return False, f"Virtual environment incomplete: {msg}"


def remove_venv(venv_dir: str = None) -> Tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not os.path.exists(venv_dir):
        return True, "Virtual environment does not exist"

    try:
        shutil.rmtree(venv_dir)
        _log(f"Removed virtual environment: {venv_dir}", Qgis.Success)
        return True, "Virtual environment removed"
    except Exception as e:
        _log(f"Failed to remove venv: {e}", Qgis.Warning)
        return False, f"Failed to remove venv: {str(e)[:200]}"
