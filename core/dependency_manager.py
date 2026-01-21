

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import importlib.util

from qgis.core import QgsMessageLog, Qgis, QgsSettings


if sys.platform == "win32":
    
    CREATE_NO_WINDOW = 0x08000000
    STARTUPINFO = subprocess.STARTUPINFO()
    STARTUPINFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    STARTUPINFO.wShowWindow = subprocess.SW_HIDE
else:
    CREATE_NO_WINDOW = 0
    STARTUPINFO = None


REQUIRED_PACKAGES = [
    ("numpy", "1.20.0"),
    ("onnxruntime", "1.15.0"),
]


SETTINGS_KEY_DEPS_DISMISSED = "AI_Segmentation/dependencies_dismissed"


PYTHON_VERSION = sys.version_info
PLUGIN_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES_INSTALL_DIR = os.path.join(PLUGIN_ROOT_DIR, f'python{PYTHON_VERSION.major}.{PYTHON_VERSION.minor}')


def get_python_path() -> Optional[str]:
    
    
    
    if sys.platform == "win32":
        QgsMessageLog.logMessage(
            "Windows detected: using 'python' command (sys.executable returns QGIS.exe)",
            "AI Segmentation",
            level=Qgis.Info
        )
        return "python"

    
    if sys.platform == "darwin":
        possible_paths = []

        
        py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"

        
        
        for app_name in ["QGIS.app", "QGIS-LTR.app"]:
            base = f"/Applications/{app_name}/Contents/MacOS"
            
            possible_paths.append(f"{base}/{py_version}")
            
            possible_paths.append(f"{base}/python3")
            
            possible_paths.append(f"{base}/python")
            
            possible_paths.append(f"{base}/bin/{py_version}")
            possible_paths.append(f"{base}/bin/python3")
            possible_paths.append(f"{base}/bin/python")

        
        possible_paths.extend([
            "/opt/homebrew/opt/qgis/bin/python3",
            "/usr/local/opt/qgis/bin/python3",
        ])

        
        if sys.prefix:
            prefix_python = Path(sys.prefix) / "bin" / "python3"
            possible_paths.insert(0, str(prefix_python))
            
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

        
        if "python" in sys.executable.lower():
            return sys.executable

        return None

    
    return sys.executable


def is_package_installed(package_name: str) -> bool:
    
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except Exception:
        return False


def get_installed_version(package_name: str) -> Optional[str]:
    
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
    
    results = []
    for package_name, min_version in REQUIRED_PACKAGES:
        installed = is_package_installed(package_name)
        version = get_installed_version(package_name) if installed else None
        results.append((package_name, min_version, installed, version))
    return results


def get_missing_dependencies() -> List[Tuple[str, str]]:
    
    missing = []
    for package_name, min_version, installed, _ in check_dependencies():
        if not installed:
            missing.append((package_name, min_version))
    return missing


def all_dependencies_installed() -> bool:
    
    return len(get_missing_dependencies()) == 0


def get_manual_install_instructions() -> str:
    
    target_dir = PACKAGES_INSTALL_DIR
    
    if sys.platform == "darwin":
        return f

    elif sys.platform == "win32":
        return f

    else:  
        return f


def install_package_via_pip_module(package_name: str, version: str = None) -> Tuple[bool, str]:
    
    try:
        import pip
        from pip._internal.cli.main import main as pip_main
    except ImportError:
        try:
            
            from pip import main as pip_main
        except ImportError:
            return False, "pip module not available"

    
    ensure_packages_dir_in_path()

    
    if version:
        package_spec = f"{package_name}>={version}"
    else:
        package_spec = package_name

    QgsMessageLog.logMessage(
        f"Installing {package_spec} via pip module to {PACKAGES_INSTALL_DIR}...",
        "AI Segmentation",
        level=Qgis.Info
    )
    QgsMessageLog.logMessage(
        "Using pip module method (no console window)",
        "AI Segmentation",
        level=Qgis.Info
    )

    try:
        
        
        
        args = [
            "install", 
            "-U", 
            f"--target={PACKAGES_INSTALL_DIR}", 
            "--progress-bar", "off",
            package_spec
        ]

        QgsMessageLog.logMessage(
            f"pip args: {args}",
            "AI Segmentation",
            level=Qgis.Info
        )

        
        
        return_code = pip_main(args)

        if return_code == 0:
            QgsMessageLog.logMessage(
                f"✓ Successfully installed {package_spec} via pip module",
                "AI Segmentation",
                level=Qgis.Success
            )
            return True, f"Installed {package_spec}"
        else:
            QgsMessageLog.logMessage(
                f"✗ pip module returned error code {return_code} for {package_spec}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            return False, f"pip returned error code {return_code}"

    except Exception as e:
        QgsMessageLog.logMessage(
            f"Error installing {package_name} via pip module: {str(e)}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return False, f"Error: {str(e)[:200]}"


def ensure_packages_dir_in_path():
    
    os.makedirs(PACKAGES_INSTALL_DIR, exist_ok=True)
    if PACKAGES_INSTALL_DIR not in sys.path:
        sys.path.insert(0, PACKAGES_INSTALL_DIR)
        QgsMessageLog.logMessage(
            f"Added packages directory to sys.path: {PACKAGES_INSTALL_DIR}",
            "AI Segmentation",
            level=Qgis.Info
        )


def install_package_via_subprocess(
    package_name: str, 
    version: str = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[bool, str]:
    
    python_path = get_python_path()

    if python_path is None:
        return False, "Could not find Python executable."

    
    ensure_packages_dir_in_path()

    try:
        
        if version:
            package_spec = f"{package_name}>={version}"
        else:
            package_spec = package_name

        QgsMessageLog.logMessage(
            f"Installing {package_spec} using {python_path} to {PACKAGES_INSTALL_DIR}...",
            "AI Segmentation",
            level=Qgis.Info
        )

        
        
        cmd = [
            python_path,
            "-m",
            "pip",
            "install",
            "-U",  # Upgrade if already exists
            f"--target={PACKAGES_INSTALL_DIR}",
            "--progress-bar", "off",  # Cleaner output for parsing
            package_spec
        ]

        QgsMessageLog.logMessage(
            f"Running command: {' '.join(cmd)}",
            "AI Segmentation",
            level=Qgis.Info
        )

        
        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,  # Merge stderr into stdout
            "text": True,
            "bufsize": 1,  # Line buffered
            "env": {**os.environ, "PYTHONIOENCODING": "utf-8"}
        }
        
        
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = CREATE_NO_WINDOW
            popen_kwargs["startupinfo"] = STARTUPINFO

        
        process = subprocess.Popen(cmd, **popen_kwargs)
        
        output_lines = []
        last_status = ""
        
        
        while True:
            line = process.stdout.readline()
            if line == "" and process.poll() is not None:
                break
            if line:
                line = line.strip()
                output_lines.append(line)
                
                
                QgsMessageLog.logMessage(
                    f"  pip: {line}",
                    "AI Segmentation",
                    level=Qgis.Info
                )
                
                
                if progress_callback:
                    if "Downloading" in line:
                        last_status = f"Downloading {package_name}..."
                        progress_callback(last_status)
                    elif "Installing" in line:
                        last_status = f"Installing {package_name}..."
                        progress_callback(last_status)
                    elif "Collecting" in line:
                        last_status = f"Collecting {package_name}..."
                        progress_callback(last_status)
                    elif "Successfully installed" in line:
                        last_status = f"Installed {package_name} ✓"
                        progress_callback(last_status)
        
        return_code = process.poll()
        
        if return_code == 0:
            QgsMessageLog.logMessage(
                f"Successfully installed {package_spec}",
                "AI Segmentation",
                level=Qgis.Success
            )
            return True, f"Installed {package_spec}"
        else:
            error_output = "\n".join(output_lines[-10:])  # Last 10 lines
            QgsMessageLog.logMessage(
                f"Failed to install {package_spec}: exit code {return_code}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            return False, f"Failed (exit {return_code}): {error_output[:200]}"

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


def install_package(
    package_name: str, 
    version: str = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[bool, str]:
    
    
    if sys.platform in ("darwin", "win32"):
        platform_name = "macOS" if sys.platform == "darwin" else "Windows"
        QgsMessageLog.logMessage(
            f"{platform_name} detected: using pip module method (no console window)",
            "AI Segmentation",
            level=Qgis.Info
        )
        if progress_callback:
            progress_callback(f"Installing {package_name} via pip module...")
            
        success, msg = install_package_via_pip_module(package_name, version)
        if success:
            return success, msg
            
        
        QgsMessageLog.logMessage(
            "pip module method failed, trying subprocess (hidden console)...",
            "AI Segmentation",
            level=Qgis.Info
        )
        if progress_callback:
            progress_callback(f"Retrying {package_name} via subprocess...")
            
        return install_package_via_subprocess(package_name, version, progress_callback)

    
    return install_package_via_subprocess(package_name, version, progress_callback)


def install_all_dependencies(
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[bool, List[str]]:
    
    
    ensure_packages_dir_in_path()

    missing = get_missing_dependencies()

    if not missing:
        return True, ["All dependencies are already installed"]

    
    python_path = get_python_path()
    QgsMessageLog.logMessage(
        f"Python path: {python_path}",
        "AI Segmentation",
        level=Qgis.Info
    )

    QgsMessageLog.logMessage(
        f"Installing {len(missing)} packages to: {PACKAGES_INSTALL_DIR}",
        "AI Segmentation",
        level=Qgis.Info
    )
    
    if progress_callback:
        packages_str = ", ".join([name for name, _ in missing])
        progress_callback(0, len(missing), f"Preparing to install: {packages_str}")

    messages = []
    all_success = True
    total = len(missing)

    for i, (package_name, version) in enumerate(missing):
        if progress_callback:
            progress_callback(i, total, f"Installing {package_name}... ({i+1}/{total})")

        
        def pip_progress(msg: str):
            if progress_callback:
                
                progress_callback(i, total, msg)
        
        success, msg = install_package(package_name, version, pip_progress)
        messages.append(f"{package_name}: {msg}")

        if success:
            QgsMessageLog.logMessage(
                f"✓ {package_name} installed successfully",
                "AI Segmentation",
                level=Qgis.Success
            )
            if progress_callback:
                progress_callback(i + 1, total, f"✓ {package_name} installed")
        else:
            QgsMessageLog.logMessage(
                f"✗ {package_name} installation failed: {msg}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            all_success = False
            if progress_callback:
                progress_callback(i + 1, total, f"✗ {package_name} failed")
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
        import onnxruntime as ort

        
        _ = np.array([1, 2, 3])
        providers = ort.get_available_providers()

        return True, f"numpy {np.__version__}, onnxruntime {ort.__version__} (providers: {providers})"

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


def get_dependency_status_summary() -> str:
    
    deps = check_dependencies()
    installed = [(name, ver) for name, _, is_installed, ver in deps if is_installed]
    missing = [(name, req_ver) for name, req_ver, is_installed, _ in deps if not is_installed]
    
    if not missing:
        versions = ", ".join([f"{name} {ver}" for name, ver in installed])
        return f"OK: {versions}"
    else:
        missing_str = ", ".join([name for name, _ in missing])
        return f"Missing: {missing_str}"
