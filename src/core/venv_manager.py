import subprocess
import sys
import os
import shutil
import platform
from typing import Tuple, Optional, Callable, List

from qgis.core import QgsMessageLog, Qgis


PLUGIN_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = PLUGIN_ROOT_DIR  # src/ directory
PYTHON_VERSION = f"py{sys.version_info.major}.{sys.version_info.minor}"
VENV_DIR = os.path.join(PLUGIN_ROOT_DIR, f'venv_{PYTHON_VERSION}')
LIBS_DIR = os.path.join(PLUGIN_ROOT_DIR, 'libs')

REQUIRED_PACKAGES = [
    ("numpy", ">=1.26.0,<2.0.0"),
    ("torch", ">=2.0.0"),
    ("torchvision", ">=0.15.0"),
    ("segment-anything", ">=1.0"),
    ("pandas", ">=1.3.0"),
    ("rasterio", ">=1.3.0"),
]


def _log(message: str, level=Qgis.Info):
    QgsMessageLog.logMessage(message, "AI Segmentation", level=level)


def _log_system_info():
    """Log system information for debugging installation issues."""
    try:
        qgis_version = Qgis.QGIS_VERSION
    except Exception:
        qgis_version = "Unknown"

    info_lines = [
        "=" * 50,
        "Installation Environment:",
        f"  OS: {sys.platform} ({platform.system()} {platform.release()})",
        f"  Architecture: {platform.machine()}",
        f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"  QGIS: {qgis_version}",
        "=" * 50,
    ]
    for line in info_lines:
        _log(line, Qgis.Info)


def _check_rosetta_warning() -> Optional[str]:
    """
    On macOS ARM, detect if running under Rosetta (x86_64 emulation).
    Returns warning message if Rosetta detected, None otherwise.
    """
    if sys.platform != "darwin":
        return None

    machine = platform.machine()

    # On Apple Silicon running native: machine = "arm64"
    # If running under Rosetta: machine = "x86_64" but CPU is Apple Silicon
    if machine == "x86_64":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            if "Apple" in result.stdout:
                return (
                    "Warning: QGIS is running under Rosetta (x86_64 emulation) "
                    "on Apple Silicon. This may cause compatibility issues. "
                    "Consider using the native ARM64 version of QGIS for best performance."
                )
        except Exception:
            pass

    return None


def _needs_cu128(gpu_name: str) -> bool:
    """
    Detect if the GPU requires CUDA 12.8 wheels.
    RTX 50-series (Blackwell architecture, compute capability sm_120)
    needs cu128 because cu121/cu126 wheels don't include SM_120 kernels.
    """
    if not gpu_name:
        return False
    return "RTX 50" in gpu_name.upper()


def detect_nvidia_gpu() -> Tuple[bool, str]:
    """
    Detect if an NVIDIA GPU is present by querying nvidia-smi.
    Returns (True, "GPU Name") or (False, "").
    """
    try:
        subprocess_kwargs = {}
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            subprocess_kwargs["startupinfo"] = startupinfo

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            **subprocess_kwargs,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split("\n")[0].strip()
            _log(f"NVIDIA GPU detected: {gpu_name}", Qgis.Info)
            return True, gpu_name
    except FileNotFoundError:
        pass  # nvidia-smi not found = no NVIDIA GPU
    except subprocess.TimeoutExpired:
        _log("nvidia-smi timed out", Qgis.Warning)
    except Exception as e:
        _log(f"nvidia-smi check failed: {e}", Qgis.Warning)

    return False, ""


def cleanup_old_venv_directories() -> List[str]:
    """
    Remove old venv_pyX.Y directories that don't match current Python version.
    Returns list of removed directories.
    """
    current_venv_name = f"venv_{PYTHON_VERSION}"
    removed = []

    try:
        for entry in os.listdir(SRC_DIR):
            if entry.startswith("venv_py") and entry != current_venv_name:
                old_path = os.path.join(SRC_DIR, entry)
                if os.path.isdir(old_path):
                    try:
                        shutil.rmtree(old_path)
                        _log(f"Cleaned up old venv directory: {entry}", Qgis.Info)
                        removed.append(entry)
                    except Exception as e:
                        _log(f"Failed to remove old venv {entry}: {e}", Qgis.Warning)
    except Exception as e:
        _log(f"Error scanning for old venv directories: {e}", Qgis.Warning)

    return removed


def _check_gdal_available() -> Tuple[bool, str]:
    """
    Check if GDAL system library is available (Linux only).
    Returns (is_available, help_message).
    """
    if sys.platform != "linux":
        return True, ""

    try:
        result = subprocess.run(
            ["gdal-config", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return True, f"GDAL {result.stdout.strip()} found"
        return False, ""
    except FileNotFoundError:
        return False, (
            "GDAL library not found. Rasterio requires GDAL to be installed.\n"
            "Please install GDAL:\n"
            "  Ubuntu/Debian: sudo apt install libgdal-dev\n"
            "  Fedora: sudo dnf install gdal-devel\n"
            "  Arch: sudo pacman -S gdal"
        )
    except Exception:
        return True, ""  # Assume OK if check fails


_SSL_ERROR_PATTERNS = [
    "ssl",
    "certificate verify failed",
    "CERTIFICATE_VERIFY_FAILED",
    "SSLError",
    "SSLCertVerificationError",
    "tlsv1 alert",
    "unable to get local issuer certificate",
    "self signed certificate in certificate chain",
]


def _is_ssl_error(stderr: str) -> bool:
    """Detect SSL/certificate errors in pip output."""
    stderr_lower = stderr.lower()
    return any(pattern.lower() in stderr_lower for pattern in _SSL_ERROR_PATTERNS)


def _get_pip_ssl_flags() -> List[str]:
    """Get pip flags to bypass SSL verification for corporate proxies."""
    return [
        "--trusted-host", "pypi.org",
        "--trusted-host", "pypi.python.org",
        "--trusted-host", "files.pythonhosted.org",
    ]


def _get_ssl_error_help() -> str:
    """Get actionable help message for persistent SSL errors."""
    return (
        "SSL certificate verification failed. "
        "This is usually caused by a corporate proxy or firewall "
        "intercepting HTTPS connections.\n\n"
        "Please try:\n"
        "  1. Ask your IT team to whitelist: pypi.org, "
        "pypi.python.org, files.pythonhosted.org\n"
        "  2. Check your proxy settings in QGIS "
        "(Settings > Options > Network)\n"
        "  3. If using a VPN, try disconnecting temporarily"
    )


def _is_antivirus_error(stderr: str) -> bool:
    """Detect antivirus/permission blocking in pip output."""
    stderr_lower = stderr.lower()
    patterns = [
        "access is denied",
        "winerror 5",
        "winerror 225",
        "permission denied",
        "operation did not complete successfully because the file contains a virus",
        "blocked by your administrator",
        "blocked by group policy",
    ]
    return any(p in stderr_lower for p in patterns)


def _get_pip_antivirus_help(venv_dir: str) -> str:
    """Get actionable help message for antivirus blocking pip."""
    return (
        "Installation was blocked, likely by antivirus software "
        "or security policy.\n\n"
        "Please try:\n"
        "  1. Temporarily disable real-time antivirus scanning\n"
        "  2. Add an exclusion for the plugin folder:\n"
        "     {}\n"
        "  3. Run QGIS as administrator (right-click > Run as administrator)\n"
        "  4. Try the installation again"
    ).format(venv_dir)


# Windows NTSTATUS crash codes (both signed and unsigned representations)
_WINDOWS_CRASH_CODES = {
    3221225477,   # 0xC0000005 unsigned - ACCESS_VIOLATION
    -1073741819,  # 0xC0000005 signed   - ACCESS_VIOLATION
    3221225725,   # 0xC00000FD unsigned - STACK_OVERFLOW
    -1073741571,  # 0xC00000FD signed   - STACK_OVERFLOW
    3221225781,   # 0xC0000135 unsigned - DLL_NOT_FOUND
    -1073741515,  # 0xC0000135 signed   - DLL_NOT_FOUND
}


def _is_windows_process_crash(returncode: int) -> bool:
    """Detect Windows process crashes (ACCESS_VIOLATION, STACK_OVERFLOW, etc.)."""
    if sys.platform != "win32":
        return False
    return returncode in _WINDOWS_CRASH_CODES


def _get_crash_help(venv_dir: str) -> str:
    """Get actionable help for Windows process crash during pip install."""
    return (
        "The installer process crashed unexpectedly (access violation).\n\n"
        "This is usually caused by:\n"
        "  - Antivirus software (Windows Defender, etc.) blocking pip\n"
        "  - Corrupted virtual environment\n\n"
        "Please try:\n"
        "  1. Temporarily disable real-time antivirus scanning\n"
        "  2. Add an exclusion for the plugin folder:\n"
        "     {}\n"
        "  3. Click 'Reinstall Dependencies' to recreate the environment\n"
        "  4. If the issue persists, run QGIS as administrator"
    ).format(venv_dir)


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


def _get_qgis_python() -> Optional[str]:
    """
    Get the path to QGIS's bundled Python on Windows.

    QGIS ships with a signed Python interpreter. This is used as a fallback
    when the standalone Python download is blocked by anti-malware software.

    Returns the path to the Python executable, or None if not found/not Windows.
    """
    if sys.platform != "win32":
        return None

    # QGIS on Windows bundles Python under sys.prefix
    python_path = os.path.join(sys.prefix, "python.exe")
    if not os.path.exists(python_path):
        # Some QGIS installs place it under a python3 name
        python_path = os.path.join(sys.prefix, "python3.exe")

    if not os.path.exists(python_path):
        _log("QGIS bundled Python not found at sys.prefix", Qgis.Warning)
        return None

    # Verify it can execute
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

        result = subprocess.run(
            [python_path, "-c", "import sys; print(sys.version)"],
            capture_output=True, text=True, timeout=15,
            env=env, startupinfo=startupinfo,
        )
        if result.returncode == 0:
            _log(f"QGIS Python verified: {result.stdout.strip()}", Qgis.Info)
            return python_path
        else:
            _log(f"QGIS Python failed verification: {result.stderr}", Qgis.Warning)
            return None
    except Exception as e:
        _log(f"QGIS Python verification error: {e}", Qgis.Warning)
        return None


def _get_system_python() -> str:
    """
    Get the path to the Python executable for creating venvs.

    Uses the standalone Python downloaded by python_manager.
    On Windows, falls back to QGIS's bundled Python if standalone is unavailable
    (e.g. when anti-malware blocks the standalone download).
    """
    from .python_manager import standalone_python_exists, get_standalone_python_path

    if standalone_python_exists():
        python_path = get_standalone_python_path()
        _log(f"Using standalone Python: {python_path}", Qgis.Info)
        return python_path

    # On Windows, try QGIS's bundled Python as fallback
    if sys.platform == "win32":
        qgis_python = _get_qgis_python()
        if qgis_python:
            _log(
                "Standalone Python unavailable, using QGIS Python as fallback",
                Qgis.Warning
            )
            return qgis_python

    # No fallback available
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
            _log("Virtual environment created successfully", Qgis.Success)

            # Ensure pip is available (QGIS Python fallback may not include pip)
            pip_path = get_venv_pip_path(venv_dir)
            if not os.path.exists(pip_path):
                _log("pip not found in venv, bootstrapping with ensurepip...", Qgis.Info)
                python_in_venv = get_venv_python_path(venv_dir)
                ensurepip_cmd = [python_in_venv, "-m", "ensurepip", "--upgrade"]
                try:
                    ensurepip_result = subprocess.run(
                        ensurepip_cmd,
                        capture_output=True, text=True, timeout=120,
                        env=env,
                        **({"startupinfo": startupinfo} if sys.platform == "win32" else {}),
                    )
                    if ensurepip_result.returncode == 0:
                        _log("pip bootstrapped via ensurepip", Qgis.Success)
                    else:
                        err = ensurepip_result.stderr or ensurepip_result.stdout
                        _log(f"ensurepip failed: {err[:200]}", Qgis.Warning)
                        return False, f"Failed to bootstrap pip: {err[:200]}"
                except Exception as e:
                    _log(f"ensurepip exception: {e}", Qgis.Warning)
                    return False, f"Failed to bootstrap pip: {str(e)[:200]}"

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
    cancel_check: Optional[Callable[[], bool]] = None,
    cuda_enabled: bool = False
) -> Tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not venv_exists(venv_dir):
        return False, "Virtual environment does not exist"

    pip_path = get_venv_pip_path(venv_dir)
    _log(f"Installing dependencies using: {pip_path}", Qgis.Info)
    if cuda_enabled:
        _log("CUDA mode enabled - will install GPU-accelerated PyTorch", Qgis.Info)

    total_packages = len(REQUIRED_PACKAGES)
    base_progress = 20
    progress_per_package = 80 // total_packages

    python_path = get_venv_python_path(venv_dir)

    for i, (package_name, version_spec) in enumerate(REQUIRED_PACKAGES):
        if cancel_check and cancel_check():
            _log("Installation cancelled by user", Qgis.Warning)
            return False, "Installation cancelled"

        package_spec = f"{package_name}{version_spec}"
        current_progress = base_progress + (i * progress_per_package)

        is_cuda_package = cuda_enabled and package_name in ("torch", "torchvision")

        if progress_callback:
            if package_name == "torch" and cuda_enabled:
                progress_callback(
                    current_progress,
                    "Installing {} (CUDA ~2.5GB)... ({}/{})".format(
                        package_name, i + 1, total_packages))
            elif package_name == "torch":
                progress_callback(
                    current_progress,
                    "Installing {} (~600MB)... ({}/{})".format(
                        package_name, i + 1, total_packages))
            else:
                progress_callback(
                    current_progress,
                    "Installing {}... ({}/{})".format(
                        package_name, i + 1, total_packages))

        _log(f"[{i + 1}/{total_packages}] Installing {package_spec}...", Qgis.Info)

        pip_args = [
            "install",
            "--upgrade",
            "--no-warn-script-location",
            "--disable-pip-version-check",
            "--prefer-binary",  # Prefer pre-built wheels to avoid C extension build issues
        ]
        pip_args.extend(_get_pip_proxy_args())
        pip_args.append(package_spec)

        # For CUDA-enabled torch/torchvision, use PyTorch's CUDA index
        if is_cuda_package:
            _, gpu_name = detect_nvidia_gpu()
            if _needs_cu128(gpu_name):
                cuda_index = "cu128"
            else:
                cuda_index = "cu121"
            pip_args.extend([
                "--index-url", "https://download.pytorch.org/whl/{}".format(cuda_index)
            ])
            _log("Using CUDA {} index for {}".format(cuda_index, package_name), Qgis.Info)

        # Use clean env to avoid QGIS PYTHONPATH/PYTHONHOME interference
        env = _get_clean_env_for_venv()

        subprocess_kwargs = _get_subprocess_kwargs()

        # CUDA wheels are ~2.5GB, need more time than standard packages
        if is_cuda_package and package_name in ("torch", "torchvision"):
            pkg_timeout = 1800  # 30 min for CUDA wheels
        else:
            pkg_timeout = 600  # 10 min for standard packages

        install_failed = False
        install_error_msg = ""
        last_returncode = None

        try:
            # Use python -m pip (more reliable than pip.exe on Windows
            # where antivirus may block the standalone pip executable)
            base_cmd = [python_path, "-m", "pip"] + pip_args

            # First attempt: standard pip install
            result = subprocess.run(
                base_cmd,
                capture_output=True, text=True, timeout=pkg_timeout,
                env=env, **subprocess_kwargs,
            )

            # If Windows process crash, retry with pip.exe as fallback
            if _is_windows_process_crash(result.returncode):
                _log(
                    "Process crash detected (code {}), "
                    "retrying with pip.exe...".format(result.returncode),
                    Qgis.Warning
                )
                if progress_callback:
                    progress_callback(
                        current_progress,
                        "Retrying {}... ({}/{})".format(
                            package_name, i + 1, total_packages)
                    )

                fallback_cmd = [pip_path] + pip_args
                result = subprocess.run(
                    fallback_cmd,
                    capture_output=True, text=True, timeout=pkg_timeout,
                    env=env, **subprocess_kwargs,
                )

            # If failed, check for SSL errors and retry with --trusted-host
            if result.returncode != 0 and not _is_windows_process_crash(result.returncode):
                error_output = result.stderr or result.stdout or ""

                if _is_ssl_error(error_output):
                    _log(
                        "SSL error detected, retrying with --trusted-host flags...",
                        Qgis.Warning
                    )
                    if progress_callback:
                        progress_callback(
                            current_progress,
                            "SSL error, retrying {}... ({}/{})".format(
                                package_name, i + 1, total_packages)
                        )

                    ssl_cmd = base_cmd[:3] + _get_pip_ssl_flags() + base_cmd[3:]
                    result = subprocess.run(
                        ssl_cmd,
                        capture_output=True, text=True, timeout=pkg_timeout,
                        env=env, **subprocess_kwargs,
                    )

            if result.returncode == 0:
                _log(f"✓ Successfully installed {package_spec}", Qgis.Success)
                if progress_callback:
                    progress_callback(current_progress + progress_per_package, f"✓ {package_name} installed")
            else:
                error_msg = result.stderr or result.stdout or f"Return code {result.returncode}"
                _log(f"✗ Failed to install {package_spec}: {error_msg[:500]}", Qgis.Critical)
                install_failed = True
                install_error_msg = error_msg
                last_returncode = result.returncode

        except subprocess.TimeoutExpired:
            _log(f"Installation of {package_spec} timed out", Qgis.Critical)
            install_failed = True
            install_error_msg = f"Installation of {package_name} timed out"
        except Exception as e:
            _log(f"Exception during installation of {package_spec}: {str(e)}", Qgis.Critical)
            install_failed = True
            install_error_msg = f"Error installing {package_name}: {str(e)[:200]}"

        # CUDA → CPU silent fallback: if CUDA install failed, retry with CPU wheel
        if install_failed and is_cuda_package:
            _log(
                "CUDA install of {} failed, falling back to CPU version...".format(package_name),
                Qgis.Warning
            )
            if progress_callback:
                progress_callback(
                    current_progress,
                    "CUDA failed, installing {} (CPU)...".format(package_name)
                )

            cpu_pip_args = [
                "install", "--upgrade", "--no-warn-script-location",
                "--disable-pip-version-check", "--prefer-binary",
                package_spec
            ]
            cpu_cmd = [python_path, "-m", "pip"] + cpu_pip_args
            try:
                cpu_result = subprocess.run(
                    cpu_cmd,
                    capture_output=True, text=True, timeout=600,
                    env=env, **subprocess_kwargs,
                )
                if cpu_result.returncode == 0:
                    _log("✓ Successfully installed {} (CPU version)".format(package_spec), Qgis.Success)
                    if progress_callback:
                        progress_callback(
                            current_progress + progress_per_package,
                            "✓ {} installed (CPU)".format(package_name)
                        )
                    install_failed = False
                else:
                    cpu_err = cpu_result.stderr or cpu_result.stdout or ""
                    install_error_msg = "CUDA and CPU install both failed for {}: {}".format(
                        package_name, cpu_err[:200])
            except subprocess.TimeoutExpired:
                install_error_msg = "CUDA and CPU install both timed out for {}".format(package_name)
            except Exception as e:
                install_error_msg = "CUDA and CPU install both failed for {}: {}".format(
                    package_name, str(e)[:200])

        if install_failed:
            # Check for Windows process crash
            if last_returncode is not None and _is_windows_process_crash(last_returncode):
                crash_help = _get_crash_help(venv_dir)
                _log(crash_help, Qgis.Warning)
                install_error_msg = "{}\n\n{}".format(
                    "Process crashed (code {})".format(last_returncode),
                    crash_help)
                return False, f"Failed to install {package_name}: {install_error_msg}"

            # Check for SSL errors
            if _is_ssl_error(install_error_msg):
                ssl_help = _get_ssl_error_help()
                _log(ssl_help, Qgis.Warning)
                install_error_msg = "{}\n\n{}".format(install_error_msg[:200], ssl_help)
                return False, f"Failed to install {package_name}: {install_error_msg}"

            # Check for antivirus blocking
            if _is_antivirus_error(install_error_msg):
                av_help = _get_pip_antivirus_help(venv_dir)
                _log(av_help, Qgis.Warning)
                install_error_msg = "{}\n\n{}".format(install_error_msg[:200], av_help)
                return False, f"Failed to install {package_name}: {install_error_msg}"

            # Check for GDAL issues on Linux when rasterio fails
            if package_name == "rasterio":
                gdal_ok, gdal_help = _check_gdal_available()
                if not gdal_ok and gdal_help:
                    _log(gdal_help, Qgis.Warning)
                    install_error_msg = "{}\n\n{}".format(install_error_msg[:200], gdal_help)
                    return False, f"Failed to install {package_name}: {install_error_msg}"

            return False, f"Failed to install {package_name}: {install_error_msg[:200]}"

    if progress_callback:
        progress_callback(100, "✓ All dependencies installed")

    _log("=" * 50, Qgis.Success)
    _log("All dependencies installed successfully!", Qgis.Success)
    _log(f"Virtual environment: {venv_dir}", Qgis.Success)
    _log("=" * 50, Qgis.Success)

    return True, "All dependencies installed successfully"


def _get_qgis_proxy_settings() -> Optional[str]:
    """Read proxy configuration from QGIS settings.

    Returns a proxy URL string like 'http://user:pass@host:port'
    or None if proxy is not configured or disabled.
    """
    try:
        from qgis.core import QgsSettings
        from urllib.parse import quote as url_quote

        settings = QgsSettings()
        enabled = settings.value("proxy/proxyEnabled", False, type=bool)
        if not enabled:
            return None

        host = settings.value("proxy/proxyHost", "", type=str)
        if not host:
            return None

        port = settings.value("proxy/proxyPort", "", type=str)
        user = settings.value("proxy/proxyUser", "", type=str)
        password = settings.value("proxy/proxyPassword", "", type=str)

        proxy_url = "http://"
        if user:
            proxy_url += url_quote(user, safe="")
            if password:
                proxy_url += ":" + url_quote(password, safe="")
            proxy_url += "@"
        proxy_url += host
        if port:
            proxy_url += ":{}".format(port)

        return proxy_url
    except Exception as e:
        _log("Could not read QGIS proxy settings: {}".format(e), Qgis.Warning)
        return None


def _get_pip_proxy_args() -> List[str]:
    """Get pip --proxy argument if QGIS proxy is configured."""
    proxy_url = _get_qgis_proxy_settings()
    if proxy_url:
        _log("Using QGIS proxy for pip: {}".format(
            proxy_url.split("@")[-1] if "@" in proxy_url else proxy_url),
            Qgis.Info
        )
        return ["--proxy", proxy_url]
    return []


def _get_clean_env_for_venv() -> dict:
    env = os.environ.copy()

    vars_to_remove = [
        'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV',
        'QGIS_PREFIX_PATH', 'QGIS_PLUGINPATH',
    ]
    for var in vars_to_remove:
        env.pop(var, None)

    env["PYTHONIOENCODING"] = "utf-8"

    # Propagate QGIS proxy settings to environment for pip/network calls
    proxy_url = _get_qgis_proxy_settings()
    if proxy_url:
        env.setdefault("HTTP_PROXY", proxy_url)
        env.setdefault("HTTPS_PROXY", proxy_url)

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


def _get_verification_code(package_name: str) -> str:
    """
    Get verification code that actually TESTS the package works, not just imports.

    This catches issues like pandas C extensions not being built properly.
    """
    if package_name == "pandas":
        # Test that pandas C extensions work by creating a DataFrame
        return "import pandas as pd; df = pd.DataFrame({'a': [1, 2, 3]}); print(df.sum())"
    elif package_name == "numpy":
        # Test numpy array operations
        return "import numpy as np; a = np.array([1, 2, 3]); print(np.sum(a))"
    elif package_name == "torch":
        # Test torch tensor creation
        return "import torch; t = torch.tensor([1, 2, 3]); print(t.sum())"
    elif package_name == "rasterio":
        # Just import - rasterio needs a file to test fully
        return "import rasterio; print(rasterio.__version__)"
    elif package_name == "segment-anything":
        return "import segment_anything; print('ok')"
    elif package_name == "torchvision":
        return "import torchvision; print(torchvision.__version__)"
    else:
        import_name = package_name.replace("-", "_")
        return f"import {import_name}"


def verify_venv(
    venv_dir: str = None,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not venv_exists(venv_dir):
        return False, "Virtual environment not found"

    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    total_packages = len(REQUIRED_PACKAGES)
    for i, (package_name, _) in enumerate(REQUIRED_PACKAGES):
        if progress_callback:
            # Report progress for each package (0-100% within verification phase)
            percent = int((i / total_packages) * 100)
            progress_callback(percent, f"Verifying {package_name}... ({i + 1}/{total_packages})")

        # Get functional test code, not just import
        verify_code = _get_verification_code(package_name)
        cmd = [python_path, "-c", verify_code]

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
                error_detail = result.stderr[:300] if result.stderr else result.stdout[:300]
                _log(
                    f"Package {package_name} verification failed: {error_detail}",
                    Qgis.Warning
                )
                return False, f"Package {package_name} is broken: {error_detail[:100]}"

        except Exception as e:
            _log(f"Failed to verify {package_name}: {str(e)}", Qgis.Warning)
            return False, f"Verification error: {package_name}"

    if progress_callback:
        progress_callback(100, "Verification complete")

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
    cancel_check: Optional[Callable[[], bool]] = None,
    cuda_enabled: bool = False
) -> Tuple[bool, str]:
    """
    Complete installation: download Python standalone + create venv + install packages.

    Progress breakdown:
    - 0-10%: Download Python standalone (~50MB)
    - 10-15%: Create virtual environment
    - 15-95%: Install packages (~800MB, or ~2.5GB with CUDA)
    - 95-100%: Verify installation
    """
    from .python_manager import (
        standalone_python_exists,
        download_python_standalone,
        get_python_full_version
    )

    # Log system info for debugging
    _log_system_info()

    # Check for Rosetta emulation on macOS (warning only, don't block)
    rosetta_warning = _check_rosetta_warning()
    if rosetta_warning:
        _log(rosetta_warning, Qgis.Warning)

    # Clean up old venv directories from previous Python versions
    removed_venvs = cleanup_old_venv_directories()
    if removed_venvs:
        _log(f"Removed {len(removed_venvs)} old venv directories", Qgis.Info)

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
            # On Windows, try QGIS Python fallback before giving up
            if sys.platform == "win32":
                qgis_python = _get_qgis_python()
                if qgis_python:
                    _log(
                        "Standalone Python download failed, "
                        "falling back to QGIS Python: {}".format(msg),
                        Qgis.Warning
                    )
                    if progress_callback:
                        progress_callback(10, "Using QGIS Python (fallback)...")
                else:
                    return False, f"Failed to download Python: {msg}"
            else:
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
        cancel_check=cancel_check,
        cuda_enabled=cuda_enabled
    )

    if not success:
        return False, msg

    # Step 4: Verify installation (95-100%)
    def verify_progress(percent: int, msg: str):
        """Map verification progress (0-100%) to overall progress (95-99%)."""
        if progress_callback:
            # Map 0-100 to 95-99 (leave 100% for final success message)
            mapped = 95 + int(percent * 0.04)
            progress_callback(min(mapped, 99), msg)

    is_valid, verify_msg = verify_venv(progress_callback=verify_progress)
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


def _remove_broken_symlinks(directory: str) -> int:
    """
    Recursively find and remove broken symlinks in a directory.
    Returns the number of broken symlinks removed.
    """
    removed_count = 0
    try:
        for root, dirs, files in os.walk(directory, topdown=False):
            # Check all entries (files and dirs can be symlinks)
            for name in files + dirs:
                path = os.path.join(root, name)
                # Check if it's a symlink (broken or not)
                if os.path.islink(path):
                    # Check if the symlink target exists
                    if not os.path.exists(path):
                        # Broken symlink - remove it
                        try:
                            os.unlink(path)
                            removed_count += 1
                        except Exception:
                            pass
    except Exception:
        pass
    return removed_count


def _remove_all_symlinks(directory: str) -> int:
    """
    Recursively find and remove ALL symlinks in a directory.
    This is more aggressive but ensures Qt can delete the directory.
    Returns the number of symlinks removed.
    """
    removed_count = 0
    try:
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files + dirs:
                path = os.path.join(root, name)
                if os.path.islink(path):
                    try:
                        os.unlink(path)
                        removed_count += 1
                    except Exception:
                        pass
    except Exception:
        pass
    return removed_count


def prepare_for_uninstall() -> bool:
    """
    Prepare plugin directory for uninstallation.

    On macOS, two issues prevent Qt's QDir.removeRecursively() from working:
    1. Extended attributes like 'com.apple.provenance'
    2. Broken symlinks (Qt cannot delete them)

    This function:
    - Removes extended attributes from all files
    - Removes all symlinks (broken or not) that Qt can't handle

    Returns True if cleanup was performed, False otherwise.
    """
    if sys.platform != "darwin":
        return False

    # Directories that may have problematic files
    dirs_to_clean = [
        VENV_DIR,
        os.path.join(SRC_DIR, "python_standalone"),
        os.path.join(SRC_DIR, "checkpoints"),
    ]

    cleaned = False
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            # Step 1: Remove all symlinks (Qt can't handle them properly)
            symlinks_removed = _remove_all_symlinks(dir_path)
            if symlinks_removed > 0:
                _log(f"Removed {symlinks_removed} symlinks from: {dir_path}", Qgis.Info)
                cleaned = True

            # Step 2: Remove extended attributes
            try:
                result = subprocess.run(
                    ["xattr", "-r", "-c", dir_path],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    _log(f"Cleaned extended attributes from: {dir_path}", Qgis.Info)
                    cleaned = True
            except subprocess.TimeoutExpired:
                _log(f"xattr cleanup timed out for: {dir_path}", Qgis.Warning)
            except FileNotFoundError:
                _log("xattr command not found", Qgis.Warning)
            except Exception as e:
                _log(f"Failed to clean extended attributes: {e}", Qgis.Warning)

    return cleaned
