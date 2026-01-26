import subprocess
import sys
import os
import shutil
from pathlib import Path
from typing import Tuple, Optional, Callable, List

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
        # Detect actual Python version in venv (may differ from QGIS Python)
        lib_dir = os.path.join(venv_dir, "lib")
        if os.path.exists(lib_dir):
            for entry in os.listdir(lib_dir):
                if entry.startswith("python") and os.path.isdir(os.path.join(lib_dir, entry)):
                    site_packages = os.path.join(lib_dir, entry, "site-packages")
                    if os.path.exists(site_packages):
                        return site_packages

        # Fallback to QGIS Python version (for new venv creation)
        py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        return os.path.join(venv_dir, "lib", py_version, "site-packages")


def ensure_venv_packages_available():
    if not venv_exists():
        _log("Venv does not exist, cannot load packages", Qgis.Warning)
        return False

    site_packages = get_venv_site_packages()
    if not os.path.exists(site_packages):
        # Log detailed info for debugging
        venv_dir = get_venv_dir()
        lib_dir = os.path.join(venv_dir, "lib")
        if os.path.exists(lib_dir):
            contents = os.listdir(lib_dir)
            _log(f"Venv lib/ contents: {contents}", Qgis.Warning)
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
    """
    Get the path to the Python executable for creating venvs.

    Uses the standalone Python downloaded by python_manager.
    No fallback to system Python - fails clearly if standalone not installed.
    """
    from .python_manager import standalone_python_exists, get_standalone_python_path

    if standalone_python_exists():
        python_path = get_standalone_python_path()
        _log(f"Using standalone Python: {python_path}", Qgis.Info)
        return python_path

    # No fallback - require standalone Python
    raise RuntimeError(
        "Python standalone not installed. "
        "Please click 'Install Dependencies' to download Python automatically."
    )


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


def _detect_cuda_before_install() -> Optional[str]:
    """
    Detect CUDA availability before PyTorch installation to optimize download.
    Returns CUDA version string (e.g., 'cu121') if CUDA is available, None otherwise.
    """
    # Method 1: Check for nvidia-smi (works on all platforms)
    try:
        if sys.platform == "win32":
            # On Windows, try common locations for nvidia-smi
            nvidia_smi_paths = [
                "nvidia-smi.exe",
                r"C:\Windows\System32\nvidia-smi.exe",
                r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            ]
            for nvidia_smi in nvidia_smi_paths:
                try:
                    result = subprocess.run(
                        [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        _log("CUDA detected via nvidia-smi", Qgis.Info)
                        # PyTorch 2.0+ supports CUDA 11.8 and 12.1
                        # Default to CUDA 12.1 for better compatibility
                        return "cu121"
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
        else:
            # Linux/macOS: nvidia-smi should be in PATH
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                _log("CUDA detected via nvidia-smi", Qgis.Info)
                return "cu121"
    except Exception:
        pass

    # Method 2: Check environment variables (CUDA_PATH, CUDA_HOME)
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path and os.path.exists(cuda_path):
        _log(f"CUDA detected via environment variable: {cuda_path}", Qgis.Info)
        return "cu121"

    # Method 3: On Windows, check registry for CUDA installation
    if sys.platform == "win32":
        try:
            import winreg
            # Check common CUDA registry keys
            for key_path in [
                r"SOFTWARE\NVIDIA Corporation\CUDA",
                r"SOFTWARE\WOW6432Node\NVIDIA Corporation\CUDA",
            ]:
                try:
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                    _log("CUDA detected via Windows registry", Qgis.Info)
                    winreg.CloseKey(key)
                    return "cu121"
                except (FileNotFoundError, OSError):
                    continue
        except ImportError:
            pass  # winreg not available (shouldn't happen on Windows)
        except Exception:
            pass

    return None


def _get_pytorch_index_url(cuda_version: Optional[str] = None) -> str:
    """Get PyTorch index URL based on CUDA availability."""
    if cuda_version:
        # Use PyTorch's CUDA-specific index
        return f"https://download.pytorch.org/whl/{cuda_version}"
    else:
        # Use CPU-only index (smaller download)
        return "https://download.pytorch.org/whl/cpu"


def _install_packages_batch(
    pip_path: str,
    packages: List[Tuple[str, str]],
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    pytorch_index: Optional[str] = None,
    base_progress: int = 0,
    progress_range: int = 0
) -> Tuple[bool, str]:
    """Install multiple packages in a single pip command for efficiency."""
    if not packages:
        return True, "No packages to install"

    package_specs = [f"{name}{version_spec}" for name, version_spec in packages]
    package_names = [name for name, _ in packages]
    
    if progress_callback:
        names_str = ", ".join(package_names)
        progress_callback(base_progress, f"Installing: {names_str}")

    cmd = [
        pip_path, "install",
        "--upgrade",
        "--no-warn-script-location",
        "--disable-pip-version-check",
        "--prefer-binary",
    ]

    # For PyTorch: use --index-url (not --extra-index-url) to FORCE the specific build
    # This ensures CUDA version is installed when CUDA is detected
    if pytorch_index:
        cmd.extend(["--index-url", pytorch_index])
        _log(f"FORCING PyTorch index: {pytorch_index}", Qgis.Info)

    cmd.extend(package_specs)

    _log(f"Batch installing: {', '.join(package_names)}", Qgis.Info)

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
                timeout=900,  # 15 minutes for batch install
                env=env,
                startupinfo=startupinfo,
            )
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,
                env=env,
            )

        if result.returncode == 0:
            _log(f"✓ Successfully installed batch: {', '.join(package_names)}", Qgis.Success)
            if progress_callback:
                progress_callback(base_progress + progress_range, f"✓ Installed: {', '.join(package_names)}")
            return True, f"Installed {len(packages)} packages"
        else:
            error_msg = result.stderr or result.stdout or f"Return code {result.returncode}"
            _log(f"✗ Batch install failed: {error_msg[:500]}", Qgis.Critical)
            return False, f"Batch install failed: {error_msg[:200]}"

    except subprocess.TimeoutExpired:
        _log(f"Batch installation timed out", Qgis.Critical)
        return False, "Batch installation timed out"
    except Exception as e:
        _log(f"Exception during batch installation: {str(e)}", Qgis.Critical)
        return False, f"Error: {str(e)[:200]}"


def install_dependencies(
    venv_dir: str = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Tuple[bool, str]:
    """
    Optimized dependency installation:
    1. Batch install small packages together
    2. Install PyTorch separately with optimized index
    3. Use pip cache and prefer binary wheels
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not venv_exists(venv_dir):
        return False, "Virtual environment does not exist"

    pip_path = get_venv_pip_path(venv_dir)
    _log(f"Installing dependencies using: {pip_path}", Qgis.Info)

    # Separate PyTorch from other packages for optimization
    torch_packages = []
    other_packages = []
    
    for package_name, version_spec in REQUIRED_PACKAGES:
        if package_name in ("torch", "torchvision"):
            torch_packages.append((package_name, version_spec))
        else:
            other_packages.append((package_name, version_spec))

    base_progress = 20
    # Allocate 60% for small packages, 40% for PyTorch
    other_progress_range = 60
    torch_progress_range = 20

    # Step 1: Detect CUDA before installation
    if progress_callback:
        progress_callback(base_progress, "Detecting GPU capabilities...")
    
    cuda_version = _detect_cuda_before_install()
    pytorch_index = _get_pytorch_index_url(cuda_version)
    
    if cuda_version:
        _log(f"CUDA detected, using {cuda_version} PyTorch build", Qgis.Info)
    else:
        _log("No CUDA detected, using CPU-only PyTorch (smaller download)", Qgis.Info)

    # Step 2: Batch install small packages (much faster)
    if other_packages:
        if cancel_check and cancel_check():
            return False, "Installation cancelled"
        
        success, msg = _install_packages_batch(
            pip_path,
            other_packages,
            progress_callback,
            cancel_check,
            pytorch_index=None,  # Don't use PyTorch index for non-PyTorch packages
            base_progress=base_progress,
            progress_range=other_progress_range
        )
        
        if not success:
            return False, msg

    # Step 3: Install PyTorch packages separately with optimized index
    if torch_packages:
        if cancel_check and cancel_check():
            return False, "Installation cancelled"

        if progress_callback:
            torch_names = ", ".join([name for name, _ in torch_packages])
            size_note = " (~2GB)" if cuda_version else " (~500MB CPU-only)"
            progress_callback(
                base_progress + other_progress_range,
                f"Installing {torch_names}{size_note}..."
            )

        success, msg = _install_packages_batch(
            pip_path,
            torch_packages,
            progress_callback,
            cancel_check,
            pytorch_index=pytorch_index,  # Use PyTorch index for faster downloads
            base_progress=base_progress + other_progress_range,
            progress_range=torch_progress_range
        )

        if not success:
            return False, msg

    if progress_callback:
        progress_callback(100, "✓ All dependencies installed")

    _log("=" * 50, Qgis.Success)
    _log("All dependencies installed successfully!", Qgis.Success)
    _log(f"Virtual environment: {venv_dir}", Qgis.Success)
    if cuda_version:
        _log(f"PyTorch installed with CUDA {cuda_version} support", Qgis.Success)
    else:
        _log("PyTorch installed (CPU-only, smaller download)", Qgis.Success)
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


def check_pytorch_cuda_mismatch(venv_dir: str = None) -> Tuple[bool, str]:
    """
    Check if CUDA is available on system but PyTorch is CPU-only.
    Returns (has_mismatch, message).
    """
    if venv_dir is None:
        venv_dir = VENV_DIR
    
    if not venv_exists(venv_dir):
        return False, ""
    
    # Check if system has CUDA
    cuda_available = _detect_cuda_before_install() is not None
    if not cuda_available:
        return False, ""  # No CUDA on system, no mismatch possible
    
    # Check if PyTorch can use CUDA
    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()
    
    try:
        result = subprocess.run(
            [python_path, "-c", "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')"],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            **subprocess_kwargs
        )
        
        if result.returncode == 0:
            pytorch_device = result.stdout.strip()
            if pytorch_device == "CPU":
                _log("MISMATCH: System has CUDA but PyTorch is CPU-only!", Qgis.Warning)
                return True, "PyTorch CPU-only installed but CUDA is available"
            else:
                _log("PyTorch CUDA support verified", Qgis.Success)
                return False, ""
    except Exception as e:
        _log(f"Failed to check PyTorch CUDA: {str(e)}", Qgis.Warning)
    
    return False, ""


def reinstall_pytorch_cuda(venv_dir: str = None, progress_callback: Optional[Callable[[int, str], None]] = None) -> Tuple[bool, str]:
    """Force reinstall PyTorch with CUDA support."""
    if venv_dir is None:
        venv_dir = VENV_DIR
    
    pip_path = get_venv_pip_path(venv_dir)
    cuda_version = _detect_cuda_before_install()
    
    if not cuda_version:
        return False, "No CUDA detected on system"
    
    pytorch_index = _get_pytorch_index_url(cuda_version)
    
    _log(f"Reinstalling PyTorch with CUDA ({cuda_version})...", Qgis.Info)
    
    if progress_callback:
        progress_callback(10, f"Reinstalling PyTorch with CUDA {cuda_version}...")
    
    # Uninstall existing PyTorch first
    cmd_uninstall = [pip_path, "uninstall", "-y", "torch", "torchvision"]
    try:
        subprocess.run(cmd_uninstall, capture_output=True, timeout=60)
    except Exception:
        pass  # Ignore uninstall errors
    
    # Reinstall with CUDA
    cmd = [
        pip_path, "install",
        "--index-url", pytorch_index,
        "torch>=2.0.0", "torchvision>=0.15.0"
    ]
    
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env, startupinfo=startupinfo)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
        
        if result.returncode == 0:
            _log(f"✓ PyTorch CUDA reinstalled successfully", Qgis.Success)
            if progress_callback:
                progress_callback(100, "✓ PyTorch CUDA installed")
            return True, f"PyTorch CUDA ({cuda_version}) installed"
        else:
            error = result.stderr or result.stdout or "Unknown error"
            _log(f"Failed to reinstall PyTorch CUDA: {error[:500]}", Qgis.Critical)
            return False, f"Installation failed: {error[:200]}"
    except Exception as e:
        _log(f"Exception reinstalling PyTorch: {str(e)}", Qgis.Critical)
        return False, str(e)


def verify_venv(venv_dir: str = None) -> Tuple[bool, str]:
    """
    Verify that all required packages are installed and importable in the venv.
    Also checks for CUDA mismatch and fixes it automatically.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not venv_exists(venv_dir):
        return False, "Virtual environment not found"

    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    for package_name, _ in REQUIRED_PACKAGES:
        import_name = package_name.replace("-", "_")

        # PyTorch can take a long time to import, especially with CUDA
        # Use longer timeout for torch and torchvision
        timeout = 120 if package_name in ("torch", "torchvision") else 30

        # Use a more robust import check that handles initialization delays
        if package_name == "torch":
            # For torch, check both import and basic functionality
            cmd = [python_path, "-c", "import torch; print(torch.__version__)"]
        else:
            cmd = [python_path, "-c", f"import {import_name}"]

        _log(f"Verifying {package_name} (timeout: {timeout}s)...", Qgis.Info)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                **subprocess_kwargs
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                _log(
                    f"Package {package_name} not importable in venv: {error_msg[:500]}",
                    Qgis.Warning
                )
                return False, f"Missing package: {package_name}"

            # Log success for torch to show version
            if package_name == "torch" and result.stdout:
                version = result.stdout.strip()
                _log(f"✓ {package_name} verified: {version}", Qgis.Success)

        except subprocess.TimeoutExpired:
            _log(
                f"Verification of {package_name} timed out after {timeout}s. "
                f"This may indicate a slow import (common with PyTorch/CUDA). "
                f"Package may still be installed correctly.",
                Qgis.Warning
            )
            # For torch, timeout might be acceptable if it's just slow initialization
            # Try a simpler check
            if package_name == "torch":
                _log("Attempting simpler torch verification...", Qgis.Info)
                try:
                    # Just check if torch module exists, don't initialize it
                    simple_cmd = [
                        python_path, "-c",
                        "import sys; import importlib.util; "
                        "spec = importlib.util.find_spec('torch'); "
                        "sys.exit(0 if spec is not None else 1)"
                    ]
                    simple_result = subprocess.run(
                        simple_cmd,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        env=env,
                        **subprocess_kwargs
                    )
                    if simple_result.returncode == 0:
                        _log("✓ torch module found (import may be slow but package is installed)", Qgis.Success)
                        continue  # Skip to next package
                except Exception:
                    pass
            return False, f"Verification timeout: {package_name}"

        except Exception as e:
            _log(f"Failed to verify {package_name}: {str(e)}", Qgis.Warning)
            return False, f"Verification error: {package_name}"

    _log("✓ All packages verified", Qgis.Success)
    
    # Check for CUDA mismatch and fix automatically
    has_mismatch, mismatch_msg = check_pytorch_cuda_mismatch(venv_dir)
    if has_mismatch:
        _log("Detected CUDA mismatch - reinstalling PyTorch with CUDA...", Qgis.Warning)
        success, msg = reinstall_pytorch_cuda(venv_dir)
        if success:
            _log("✓ PyTorch CUDA installed - GPU acceleration enabled!", Qgis.Success)
            return True, "Virtual environment ready (GPU accelerated)"
        else:
            _log(f"Failed to install PyTorch CUDA: {msg}. Using CPU.", Qgis.Warning)
            return True, "Virtual environment ready (CPU only)"
    
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
    """
    Complete installation: download Python standalone + create venv + install packages.

    Progress breakdown:
    - 0-10%: Download Python standalone (~50MB)
    - 10-15%: Create virtual environment
    - 15-95%: Install packages (~2.5GB)
    - 95-100%: Verify installation
    """
    from .python_manager import (
        standalone_python_exists,
        download_python_standalone,
        get_python_full_version
    )

    cleanup_old_libs()

    # Step 1: Download Python standalone if needed
    if not standalone_python_exists():
        python_version = get_python_full_version()
        _log(f"Downloading Python {python_version} standalone...", Qgis.Info)

        def python_progress(percent, msg):
            # Map 0-100 to 0-10
            if progress_callback:
                progress_callback(int(percent * 0.10), msg)

        success, msg = download_python_standalone(
            progress_callback=python_progress,
            cancel_check=cancel_check
        )

        if not success:
            return False, f"Failed to download Python: {msg}"

        if cancel_check and cancel_check():
            return False, "Installation cancelled"
    else:
        _log("Python standalone already installed", Qgis.Info)
        if progress_callback:
            progress_callback(10, "Python standalone ready")

    # Step 2: Create virtual environment if needed
    if venv_exists():
        _log("Virtual environment already exists", Qgis.Info)
        if progress_callback:
            progress_callback(15, "Virtual environment ready")
    else:
        def venv_progress(percent, msg):
            # Map 10-20 to 10-15
            if progress_callback:
                progress_callback(10 + int(percent * 0.05), msg)

        success, msg = create_venv(progress_callback=venv_progress)
        if not success:
            return False, msg

        if cancel_check and cancel_check():
            return False, "Installation cancelled"

    # Step 3: Install dependencies
    def deps_progress(percent, msg):
        # Map 20-100 to 15-95
        if progress_callback:
            mapped = 15 + int((percent - 20) * 0.80 / 0.80)
            progress_callback(min(mapped, 95), msg)

    success, msg = install_dependencies(
        progress_callback=deps_progress,
        cancel_check=cancel_check
    )

    if not success:
        return False, msg

    # Step 4: Verify installation
    if progress_callback:
        progress_callback(95, "Verifying installation...")

    is_valid, verify_msg = verify_venv()
    if not is_valid:
        return False, f"Verification failed: {verify_msg}"

    if progress_callback:
        progress_callback(100, "✓ All dependencies installed")

    return True, "Virtual environment ready"


def get_venv_status() -> Tuple[bool, str]:
    """Get the status of the complete installation (Python standalone + venv)."""
    from .python_manager import standalone_python_exists, get_python_full_version

    # Check for old libs/ installation
    if os.path.exists(LIBS_DIR):
        return False, "Old installation detected. Migration required."

    # Check Python standalone
    if not standalone_python_exists():
        return False, "Dependencies not installed"

    # Check venv
    if not venv_exists():
        return False, "Virtual environment not configured"

    # Verify packages
    is_valid, msg = verify_venv()
    if is_valid:
        python_version = get_python_full_version()
        return True, f"Ready (Python {python_version})"
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
