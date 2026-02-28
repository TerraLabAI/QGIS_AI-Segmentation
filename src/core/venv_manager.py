import subprocess
import sys
import os
import shutil
import platform
import tempfile
import time
import re
import hashlib
from typing import Tuple, Optional, Callable, List

from qgis.core import QgsMessageLog, Qgis

from .model_config import SAM_PACKAGE, TORCH_MIN, TORCHVISION_MIN, USE_SAM2
from .uv_manager import (
    uv_exists, get_uv_path, download_uv, verify_uv, remove_uv,
)


# Module-level uv state (set during create_venv_and_install)
_uv_available = False
_uv_path = None  # type: Optional[str]

PLUGIN_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = PLUGIN_ROOT_DIR  # src/ directory
PYTHON_VERSION = f"py{sys.version_info.major}.{sys.version_info.minor}"
CACHE_DIR = os.environ.get("AI_SEGMENTATION_CACHE_DIR") or os.path.expanduser("~/.qgis_ai_segmentation")
VENV_DIR = os.path.join(CACHE_DIR, f'venv_{PYTHON_VERSION}')
LIBS_DIR = os.path.join(PLUGIN_ROOT_DIR, 'libs')

REQUIRED_PACKAGES = [
    ("numpy", ">=1.26.0,<2.0.0"),
    ("torch", TORCH_MIN),
    ("torchvision", TORCHVISION_MIN),
    SAM_PACKAGE,
    ("pandas", ">=1.3.0"),
    ("rasterio", ">=1.3.0"),
]

DEPS_HASH_FILE = os.path.join(VENV_DIR, "deps_hash.txt")
CUDA_FLAG_FILE = os.path.join(VENV_DIR, "cuda_installed.txt")

# Bump this when install logic changes significantly (e.g., --no-cache-dir,
# new retry strategies) to force a dependency re-install on plugin update.
# This invalidates the deps hash so users with stale cuda_fallback flags
# get a clean retry with the improved install logic.
_INSTALL_LOGIC_VERSION = "3"

# Bumped independently of _INSTALL_LOGIC_VERSION so only users with a stale
# cuda_fallback flag get a targeted CUDA retry — without forcing a full
# dependency reinstall for everyone.  Increment this whenever the CUDA
# install logic is fixed in a way that makes a previous fallback worth
# retrying (e.g. adding --force-reinstall to overcome pip version skipping).
_CUDA_LOGIC_VERSION = "3"


def _write_cuda_flag(value: str):
    """Persist CUDA install state.

    Valid values:
      'cuda'          – CUDA torch installed and working
      'cpu'           – CPU torch installed, no GPU available
      'cuda_fallback' – GPU available but CUDA failed

    For 'cuda_fallback' the current _CUDA_LOGIC_VERSION is appended so that
    needs_cuda_upgrade() can allow a retry when the install logic improves.
    Example on disk: 'cuda_fallback:3'
    """
    if value == "cuda_fallback":
        content = "cuda_fallback:{}".format(_CUDA_LOGIC_VERSION)
    else:
        content = value
    try:
        os.makedirs(os.path.dirname(CUDA_FLAG_FILE), exist_ok=True)
        with open(CUDA_FLAG_FILE, "w", encoding="utf-8") as f:
            f.write(content)
    except (OSError, IOError) as e:
        _log("Failed to write CUDA flag: {}".format(e), Qgis.Warning)


def _read_cuda_flag() -> Optional[str]:
    """Returns 'cuda', 'cpu', 'cuda_fallback', or None.

    Handles both the legacy bare format ('cuda_fallback') and the versioned
    format ('cuda_fallback:3') written by newer plugin versions.
    """
    try:
        with open(CUDA_FLAG_FILE, "r", encoding="utf-8") as f:
            value = f.read().strip()
        base = value.split(":")[0]
        if base in ("cuda", "cpu", "cuda_fallback"):
            return base
    except (OSError, IOError):
        pass
    return None


def _read_cuda_fallback_version() -> Optional[str]:
    """Return the _CUDA_LOGIC_VERSION stored with a cuda_fallback flag, or None.

    Returns None when the flag is missing, not a fallback, or was written by
    an older plugin version that did not record the logic version.
    """
    try:
        with open(CUDA_FLAG_FILE, "r", encoding="utf-8") as f:
            value = f.read().strip()
        parts = value.split(":", 1)
        if len(parts) == 2 and parts[0] == "cuda_fallback":
            return parts[1]
    except (OSError, IOError):
        pass
    return None


def needs_cuda_upgrade() -> bool:
    """GPU install disabled (CPU-only mode). Kept for future reactivation."""
    return False


def _compute_deps_hash() -> str:
    """Compute MD5 hash of REQUIRED_PACKAGES + install logic version.

    Changing either REQUIRED_PACKAGES or _INSTALL_LOGIC_VERSION will
    invalidate the stored hash and trigger a dependency re-install.
    """
    data = repr(REQUIRED_PACKAGES).encode("utf-8")
    data += _INSTALL_LOGIC_VERSION.encode("utf-8")
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _read_deps_hash() -> Optional[str]:
    """Read stored deps hash from the venv directory."""
    try:
        with open(DEPS_HASH_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, IOError):
        return None


def _write_deps_hash():
    """Write the current deps hash to the venv directory."""
    try:
        os.makedirs(os.path.dirname(DEPS_HASH_FILE), exist_ok=True)
        with open(DEPS_HASH_FILE, "w", encoding="utf-8") as f:
            f.write(_compute_deps_hash())
    except (OSError, IOError) as e:
        _log("Failed to write deps hash: {}".format(e), Qgis.Warning)


def _log(message: str, level=Qgis.Info):
    QgsMessageLog.logMessage(message, "AI Segmentation", level=level)


def _log_system_info():
    """Log system information for debugging installation issues."""
    try:
        qgis_version = Qgis.QGIS_VERSION
    except Exception:
        qgis_version = "Unknown"

    custom_cache = os.environ.get("AI_SEGMENTATION_CACHE_DIR")
    info_lines = [
        "=" * 50,
        "Installation Environment:",
        f"  OS: {sys.platform} ({platform.system()} {platform.release()})",
        f"  Architecture: {platform.machine()}",
        f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"  QGIS: {qgis_version}",
        f"  Install dir: {CACHE_DIR}",
    ]
    if custom_cache:
        info_lines.append("  (via AI_SEGMENTATION_CACHE_DIR)")
    info_lines.append("=" * 50)
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


# Minimum NVIDIA driver versions for each CUDA toolkit version.
# cu128 (Blackwell) needs driver >= 570, cu121 needs >= 530.
_CUDA_DRIVER_REQUIREMENTS = {
    "cu128": 570,
    "cu121": 530,
}

# Blackwell (sm_120+) requires cu128; everything else works with cu121.
_MIN_COMPUTE_CAP_FOR_CU128 = 12.0

# Cache for detect_nvidia_gpu() result — avoids re-running nvidia-smi
# each time (subprocess is expensive and can block up to 5 seconds).
_gpu_detect_cache = None  # type: Optional[Tuple[bool, dict]]


def detect_nvidia_gpu() -> Tuple[bool, dict]:
    """
    Detect if an NVIDIA GPU is present by querying nvidia-smi.

    Results are cached for the lifetime of the QGIS session so that
    nvidia-smi is only invoked once.

    Returns (True, info_dict) or (False, {}).
    info_dict keys: name, compute_cap, driver_version, memory_mb
    (any key may be missing if nvidia-smi didn't report it).
    """
    global _gpu_detect_cache
    if _gpu_detect_cache is not None:
        return _gpu_detect_cache
    try:
        subprocess_kwargs = {}
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            subprocess_kwargs["startupinfo"] = startupinfo

        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,compute_cap,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            **subprocess_kwargs,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            best_gpu = {}
            best_compute_cap = -1.0

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]

                gpu_info = {}
                if len(parts) >= 1 and parts[0]:
                    gpu_info["name"] = parts[0]
                if len(parts) >= 2 and parts[1]:
                    try:
                        gpu_info["compute_cap"] = float(parts[1])
                    except ValueError:
                        pass
                if len(parts) >= 3 and parts[2]:
                    gpu_info["driver_version"] = parts[2]
                if len(parts) >= 4 and parts[3]:
                    try:
                        gpu_info["memory_mb"] = int(float(parts[3]))
                    except ValueError:
                        pass

                cc = gpu_info.get("compute_cap", 0.0)
                if cc > best_compute_cap:
                    best_compute_cap = cc
                    best_gpu = gpu_info

            if not best_gpu:
                _gpu_detect_cache = (False, {})
                return _gpu_detect_cache

            _log("NVIDIA GPU detected (best of {}): {}".format(
                len(lines), best_gpu), Qgis.Info)
            _gpu_detect_cache = (True, best_gpu)
            return _gpu_detect_cache
    except FileNotFoundError:
        pass  # nvidia-smi not found = no NVIDIA GPU
    except subprocess.TimeoutExpired:
        _log("nvidia-smi timed out", Qgis.Warning)
    except Exception as e:
        _log("nvidia-smi check failed: {}".format(e), Qgis.Warning)

    _gpu_detect_cache = (False, {})
    return _gpu_detect_cache


def _select_cuda_index(gpu_info: dict) -> Optional[str]:
    """
    Choose the correct PyTorch CUDA wheel index based on GPU info.

    Returns "cu128", "cu121", or None.
    None means the driver is too old -> caller should install CPU torch.
    """
    # Determine which CUDA toolkit the GPU needs
    compute_cap = gpu_info.get("compute_cap")
    gpu_name = gpu_info.get("name", "")

    if compute_cap is not None:
        needs_cu128 = compute_cap >= _MIN_COMPUTE_CAP_FOR_CU128
    else:
        # Fallback: name-based heuristic when compute_cap unavailable
        needs_cu128 = "RTX 50" in gpu_name.upper()

    cuda_index = "cu128" if needs_cu128 else "cu121"

    # Validate driver version is sufficient
    driver_str = gpu_info.get("driver_version", "")
    if driver_str:
        try:
            driver_major = int(driver_str.split(".")[0])
            required = _CUDA_DRIVER_REQUIREMENTS.get(cuda_index, 0)
            if driver_major < required:
                _log(
                    "NVIDIA driver {} too old for {} (needs >= {}), "
                    "will use CPU instead".format(
                        driver_str, cuda_index, required),
                    Qgis.Warning
                )
                return None
        except (ValueError, IndexError):
            _log("Could not parse driver version: {}".format(driver_str),
                 Qgis.Warning)

    return cuda_index


def cleanup_old_venv_directories() -> List[str]:
    """
    Remove old venv_pyX.Y directories that don't match current Python version.
    Scans both the external cache dir (new location) and the plugin dir (legacy).
    Returns list of removed directories.
    """
    current_venv_name = f"venv_{PYTHON_VERSION}"
    removed = []

    for scan_dir in [CACHE_DIR, SRC_DIR]:
        try:
            if not os.path.exists(scan_dir):
                continue
            for entry in os.listdir(scan_dir):
                entry_cmp = os.path.normcase(entry)
                current_cmp = os.path.normcase(current_venv_name)
                if entry_cmp.startswith(os.path.normcase("venv_py")) and entry_cmp != current_cmp:
                    old_path = os.path.join(scan_dir, entry)
                    if os.path.isdir(old_path):
                        try:
                            shutil.rmtree(old_path)
                            _log("Cleaned up old venv: {}".format(old_path),
                                 Qgis.Info)
                            removed.append(old_path)
                        except Exception as e:
                            _log("Failed to remove old venv {}: {}".format(
                                old_path, e), Qgis.Warning)
        except Exception as e:
            _log("Error scanning for old venvs in {}: {}".format(
                scan_dir, e), Qgis.Warning)

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


def _is_hash_mismatch(output: str) -> bool:
    """Detect pip hash mismatch errors (corrupted cache from interrupted download)."""
    output_lower = output.lower()
    has_mismatch = "do not match the hashes" in output_lower
    has_hash_err = "hash mismatch" in output_lower
    return has_mismatch or has_hash_err


def _get_pip_ssl_flags() -> List[str]:
    """Get pip flags to bypass SSL verification for corporate proxies."""
    return [
        "--trusted-host", "pypi.org",
        "--trusted-host", "pypi.python.org",
        "--trusted-host", "files.pythonhosted.org",
    ]


def _is_ssl_module_missing(error_text: str) -> bool:
    """Check if the error is about a missing SSL module (not a certificate issue)."""
    lower = error_text.lower()
    patterns = ["ssl module is not available", "no module named '_ssl'",
                "ssl module", "importerror: _ssl"]
    return any(p in lower for p in patterns)


def _get_ssl_error_help(error_text: str = "") -> str:
    """Get actionable help message for SSL errors.

    Differentiates between SSL module missing (broken Python) and
    SSL certificate errors (network/proxy).
    """
    if _is_ssl_module_missing(error_text):
        return (
            "Installation failed: Python's SSL module is not available.\n\n"
            "This usually means the Python installation is incomplete or corrupted.\n"
            "Please try:\n"
            "  1. Delete the folder: {}\n"
            "  2. Restart QGIS and try again\n"
            "  3. If the issue persists, reinstall QGIS".format(CACHE_DIR)
        )
    return (
        "Installation failed due to network restrictions.\n\n"
        "Please contact your IT department to allow access to:\n"
        "  - pypi.org\n"
        "  - files.pythonhosted.org\n"
        "  - download.pytorch.org\n\n"
        "You can also try checking your proxy settings in QGIS "
        "(Settings > Options > Network)."
    )


_NETWORK_ERROR_PATTERNS = [
    "connectionreseterror",
    "connection aborted",
    "connection was forcibly closed",
    "remotedisconnected",
    "connectionerror",
    "newconnectionerror",
    "maxretryerror",
    "protocolerror",
    "readtimeouterror",
    "connecttimeouterror",
    "urlib3.exceptions",
    "requests.exceptions.connectionerror",
    "network is unreachable",
    "temporary failure in name resolution",
    "name or service not known",
]


def _is_network_error(output: str) -> bool:
    """Detect transient network/connection errors in pip output."""
    output_lower = output.lower()
    # Exclude SSL errors — they have their own retry path
    if _is_ssl_error(output):
        return False
    return any(p in output_lower for p in _NETWORK_ERROR_PATTERNS)


def _is_proxy_auth_error(output: str) -> bool:
    """Detect proxy authentication errors (HTTP 407)."""
    output_lower = output.lower()
    patterns = [
        "407 proxy authentication",
        "proxy authentication required",
        "proxyerror",
    ]
    return any(p in output_lower for p in patterns)


def _is_unable_to_create_process(output: str) -> bool:
    """Detect 'unable to create process' errors on Windows (broken pip.exe shim)."""
    return "unable to create process" in output.lower()


def _is_dll_init_error(output: str) -> bool:
    """Detect DLL initialization failures (missing VC++ Redistributables)."""
    lower = output.lower()
    patterns = [
        "winerror 1114",
        "dll initialization routine failed",
        "dll load failed",
        "_load_dll_libraries",
    ]
    return any(p in lower for p in patterns)


def _get_vcpp_help() -> str:
    """Get actionable help for DLL init errors (missing VC++ Redistributables)."""
    return (
        "A required DLL failed to initialize. This is usually caused by "
        "missing Visual C++ Redistributables.\n\n"
        "Please install the latest VC++ Redistributable:\n"
        "  https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"
        "Then restart your computer and try again."
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
        "application control policy",
        "control de aplicaciones",
        "applocker",
        "blocked by your organization",
    ]
    return any(p in stderr_lower for p in patterns)


def _get_pip_antivirus_help(venv_dir: str) -> str:
    """Get actionable help message for antivirus blocking pip."""
    steps = (
        "Installation was blocked, likely by antivirus software "
        "or security policy.\n\n"
        "Please try:\n"
        "  1. Temporarily disable real-time antivirus scanning\n"
        "  2. Add an exclusion for the plugin folder:\n"
        "     {}\n".format(venv_dir)
    )
    if sys.platform == "win32":
        steps += (
            "  3. Run QGIS as administrator "
            "(right-click > Run as administrator)\n"
            "  4. Try the installation again"
        )
    else:
        steps += (
            "  3. Check folder permissions: "
            "chmod -R u+rwX \"{}\"\n"
            "  4. Try the installation again".format(venv_dir)
        )
    return steps


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


def _add_windows_dll_directories(site_packages: str) -> None:
    """Register torch/torchvision DLL directories on Windows.

    Without this, importing torch from a foreign venv inside QGIS fails
    with 'DLL load failed' (WinError 126/127).
    """
    dll_dirs = [
        os.path.join(site_packages, "torch", "lib"),
        os.path.join(site_packages, "torch", "bin"),
        os.path.join(site_packages, "torchvision"),
    ]
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for dll_dir in dll_dirs:
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except OSError as exc:
                _log("add_dll_directory({}) failed: {}".format(
                    dll_dir, exc), Qgis.Warning)
            if dll_dir not in path_parts:
                path_parts.insert(0, dll_dir)
    os.environ["PATH"] = os.pathsep.join(path_parts)


def _fix_proj_gdal_data(site_packages: str) -> None:
    """Point PROJ_DATA and GDAL_DATA to venv's bundled data files.

    QGIS sets PROJ_LIB to its own PROJ data, but the venv's pyproj/rasterio
    may bundle a different version of proj.db. Without this fix, rasterio
    CRS operations can fail or crash.
    """
    proj_candidates = [
        os.path.join(site_packages, "pyproj", "proj_dir", "share", "proj"),
        os.path.join(site_packages, "rasterio", "proj_data"),
    ]
    for candidate in proj_candidates:
        proj_db = os.path.join(candidate, "proj.db")
        if os.path.exists(proj_db):
            os.environ["PROJ_DATA"] = candidate
            os.environ["PROJ_LIB"] = candidate
            _log("Set PROJ_DATA to venv: {}".format(candidate), Qgis.Info)
            break

    gdal_candidates = [
        os.path.join(site_packages, "rasterio", "gdal_data"),
    ]
    for candidate in gdal_candidates:
        if os.path.isdir(candidate):
            os.environ["GDAL_DATA"] = candidate
            _log("Set GDAL_DATA to venv: {}".format(candidate), Qgis.Info)
            break


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
        sys.path.append(site_packages)
        _log(f"Added venv site-packages to sys.path: {site_packages}", Qgis.Info)

    # On Windows, register DLL directories for torch/torchvision so the OS
    # loader can find their native libraries when importing from a foreign venv.
    if sys.platform == "win32":
        _add_windows_dll_directories(site_packages)

    # Fix PROJ_DATA/GDAL_DATA to point to venv's pyproj/rasterio data files.
    _fix_proj_gdal_data(site_packages)

    # SAFE FIX for old QGIS with stale typing_extensions (missing TypeIs)
    # QGIS may load an old typing_extensions at startup that lacks TypeIs,
    # which torch requires. Remove it so Python reimports from venv.
    if "typing_extensions" in sys.modules:
        try:
            te = sys.modules["typing_extensions"]
            if not hasattr(te, "TypeIs"):
                old_ver = getattr(te, "__version__", "unknown")
                del sys.modules["typing_extensions"]
                import typing_extensions as new_te
                _log(
                    "Reloaded typing_extensions {} -> {} from venv".format(
                        old_ver, new_te.__version__),
                    Qgis.Info
                )
        except Exception:
            _log("Failed to reload typing_extensions, torch may fail", Qgis.Warning)

    # FIX for QGIS bundling old numpy (< 1.22.4) incompatible with pandas >= 2.0
    # Check numpy version directly instead of gating on QGIS version, because
    # multiple QGIS releases (3.26-3.28.x) can ship old numpy. (issues #130/#133/#138)
    needs_numpy_fix = False
    old_version = "unknown"
    try:
        if "numpy" in sys.modules:
            old_np = sys.modules["numpy"]
            old_version = getattr(old_np, "__version__", "0.0.0")
        else:
            # numpy not loaded yet; check if QGIS Python path has an old one
            # by probing before we import (which would lock in the wrong one)
            old_version = "not_loaded"

        if old_version == "not_loaded":
            # numpy not yet imported; ensure venv path is first so the
            # first import picks up the venv copy
            needs_numpy_fix = True
        else:
            parts = old_version.split(".")[:3]
            vn = [int(x) for x in parts] + [0] * (3 - len(parts))
            np_old = (vn[0] < 1) or (vn[0] == 1 and vn[1] < 22)
            np_old = np_old or (vn[0] == 1 and vn[1] == 22 and vn[2] < 4)
            needs_numpy_fix = np_old
    except Exception:
        needs_numpy_fix = False

    if not needs_numpy_fix:
        return True

    qgis_ver = Qgis.QGIS_VERSION.split("-")[0]
    _log(
        "QGIS {} with old numpy {} detected. "
        "Forcing venv numpy/pandas...".format(qgis_ver, old_version),
        Qgis.Info)

    removed_paths = []
    try:
        import importlib

        # 1. Remove ALL numpy and pandas modules from cache
        mods_to_clear = [
            k for k in list(sys.modules.keys())
            if k.startswith("numpy") or k.startswith("pandas")
        ]
        for mod in mods_to_clear:
            del sys.modules[mod]

        # 2. Temporarily remove QGIS Python paths that contain numpy
        #    so the reimport finds the venv copy first
        for p in sys.path[:]:
            if p == site_packages:
                continue
            np_init = os.path.join(p, "numpy", "__init__.py")
            if os.path.exists(np_init):
                removed_paths.append(p)
                sys.path.remove(p)

        # 3. Invalidate import caches so Python re-scans directories
        importlib.invalidate_caches()

        # 4. Reimport numpy from venv
        import numpy as new_numpy  # noqa: E402

        # 5. Restore removed paths (after venv so venv stays first)
        for p in removed_paths:
            if p not in sys.path:
                sys.path.append(p)
        removed_paths = []

        # 6. Verify the reload actually worked
        new_ver = new_numpy.__version__
        if new_ver == old_version and old_version != "not_loaded":
            _log(
                "WARNING: numpy reload did not change version "
                "(still {}). pandas may fail on this QGIS.".format(
                    old_version),
                Qgis.Warning)
        else:
            _log(
                "Reloaded numpy {} -> {} from venv".format(
                    old_version, new_ver),
                Qgis.Info)

    except Exception as e:
        _log(
            "Failed to reload numpy: {}. "
            "Plugin may not work on this QGIS version.".format(e),
            Qgis.Warning)

    finally:
        # Always restore paths even on exception
        for p in removed_paths:
            if p not in sys.path:
                sys.path.append(p)

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


def _cleanup_partial_venv(venv_dir: str):
    """Remove a partially-created venv directory to prevent broken state on retry."""
    if os.path.exists(venv_dir):
        try:
            shutil.rmtree(venv_dir, ignore_errors=True)
            _log(f"Cleaned up partial venv: {venv_dir}", Qgis.Info)
        except Exception:
            _log(f"Could not clean up partial venv: {venv_dir}", Qgis.Warning)


def create_venv(
    venv_dir: str = None,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    _log(f"Creating virtual environment at: {venv_dir}", Qgis.Info)

    if progress_callback:
        progress_callback(10, "Creating virtual environment...")

    system_python = _get_system_python()
    _log(f"Using Python: {system_python}", Qgis.Info)

    # Use clean env to prevent QGIS PYTHONPATH/PYTHONHOME from leaking
    # into the standalone Python subprocess (issue #131)
    env = _get_clean_env_for_venv()

    global _uv_available, _uv_path

    # Try uv venv creation first (faster, no ensurepip needed)
    if _uv_available and _uv_path:
        _log("Creating venv with uv...", Qgis.Info)
        uv_cmd = [_uv_path, "venv", "--python", system_python, venv_dir]
        try:
            subprocess_kwargs = {}
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                subprocess_kwargs["startupinfo"] = startupinfo

            result = subprocess.run(
                uv_cmd,
                capture_output=True, text=True, timeout=120,
                env=env, **subprocess_kwargs,
            )
            if result.returncode == 0:
                _log("Virtual environment created with uv", Qgis.Success)
                if progress_callback:
                    progress_callback(20, "Virtual environment created (uv)")
                return True, "Virtual environment created"
            else:
                error_msg = result.stderr or result.stdout or ""
                _log("uv venv creation failed: {}".format(
                    error_msg[:200]), Qgis.Warning)
                _cleanup_partial_venv(venv_dir)
                # Fall through to standard venv creation
                remove_uv()
                _uv_available = False
                _uv_path = None
                _log("Falling back to python -m venv", Qgis.Warning)
        except Exception as e:
            _log("uv venv exception: {}, falling back to python -m venv".format(
                e), Qgis.Warning)
            _cleanup_partial_venv(venv_dir)
            remove_uv()
            _uv_available = False
            _uv_path = None

    # Standard venv creation with python -m venv
    cmd = [system_python, "-m", "venv", venv_dir]
    try:
        subprocess_kwargs = {}
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            subprocess_kwargs["startupinfo"] = startupinfo

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            **subprocess_kwargs,
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
                        _cleanup_partial_venv(venv_dir)
                        return False, f"Failed to bootstrap pip: {err[:200]}"
                except Exception as e:
                    _log(f"ensurepip exception: {e}", Qgis.Warning)
                    _cleanup_partial_venv(venv_dir)
                    return False, f"Failed to bootstrap pip: {str(e)[:200]}"

            if progress_callback:
                progress_callback(20, "Virtual environment created")
            return True, "Virtual environment created"
        else:
            error_msg = result.stderr or result.stdout or f"Return code {result.returncode}"
            _log(f"Failed to create venv: {error_msg}", Qgis.Critical)
            _cleanup_partial_venv(venv_dir)
            return False, f"Failed to create venv: {error_msg[:200]}"

    except subprocess.TimeoutExpired:
        _log("Venv creation timed out, retrying with --without-pip...", Qgis.Warning)
        _cleanup_partial_venv(venv_dir)
        # Retry with --without-pip (faster, avoids pip setup that AV scans)
        try:
            nopip_cmd = [system_python, "-m", "venv", "--without-pip", venv_dir]
            result2 = subprocess.run(
                nopip_cmd,
                capture_output=True, text=True, timeout=300,
                env=env, **subprocess_kwargs,
            )
            if result2.returncode == 0:
                _log("Venv created (--without-pip), bootstrapping pip...", Qgis.Info)
                python_in_venv = get_venv_python_path(venv_dir)
                ensurepip_cmd = [python_in_venv, "-m", "ensurepip", "--upgrade"]
                ep_result = subprocess.run(
                    ensurepip_cmd,
                    capture_output=True, text=True, timeout=120,
                    env=env, **subprocess_kwargs,
                )
                if ep_result.returncode == 0:
                    _log("pip bootstrapped via ensurepip", Qgis.Success)
                    if progress_callback:
                        progress_callback(20, "Virtual environment created")
                    return True, "Virtual environment created"
                else:
                    err = ep_result.stderr or ep_result.stdout or ""
                    _log("ensurepip failed: {}".format(err[:200]), Qgis.Warning)
                    _cleanup_partial_venv(venv_dir)
                    return False, "Failed to bootstrap pip: {}".format(err[:200])
            else:
                err = result2.stderr or result2.stdout or ""
                _log("Retry --without-pip failed: {}".format(err[:200]), Qgis.Critical)
                _cleanup_partial_venv(venv_dir)
                return False, "Virtual environment creation timed out"
        except Exception as e2:
            _log("Retry --without-pip exception: {}".format(e2), Qgis.Critical)
            _cleanup_partial_venv(venv_dir)
            return False, "Virtual environment creation timed out"
    except FileNotFoundError:
        _log(f"Python executable not found: {system_python}", Qgis.Critical)
        return False, f"Python not found: {system_python}"
    except Exception as e:
        _log(f"Exception during venv creation: {str(e)}", Qgis.Critical)
        _cleanup_partial_venv(venv_dir)
        return False, f"Error: {str(e)[:200]}"


def _build_install_cmd(python_path: str, pip_args: list) -> list:
    """Build an install command using uv (if available) or pip.

    Translates pip flags to uv equivalents when _uv_available is True.
    """
    if _uv_available and _uv_path:
        cmd = [_uv_path, "pip"]
        skip_next = False
        for i, arg in enumerate(pip_args):
            if skip_next:
                skip_next = False
                continue
            if arg == "--disable-pip-version-check":
                continue
            elif arg == "--no-warn-script-location":
                continue
            elif arg == "--no-cache-dir":
                cmd.append("--no-cache")
                continue
            elif arg == "--force-reinstall":
                cmd.append("--reinstall")
                continue
            elif arg == "--trusted-host":
                cmd.append("--allow-insecure-host")
                continue
            elif arg == "--proxy":
                # uv uses HTTP_PROXY/HTTPS_PROXY env vars (already set)
                skip_next = True
                continue
            cmd.append(arg)
        cmd.extend(["--python", python_path])
        return cmd
    return [python_path, "-m", "pip"] + pip_args


def _build_uninstall_cmd(python_path: str, packages: list) -> list:
    """Build an uninstall command using uv (if available) or pip."""
    if _uv_available and _uv_path:
        return [_uv_path, "pip", "uninstall", "--python", python_path] + packages
    return [python_path, "-m", "pip", "uninstall", "-y"] + packages


def _repin_numpy(venv_dir: str):
    """
    Check numpy version in the venv and force-downgrade if >= 2.0.

    This is a safety net: the CUDA torch index may pull numpy 2.x as a
    transitive dependency, which breaks torchvision and other packages.
    """
    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    try:
        result = subprocess.run(
            [python_path, "-c",
             "import numpy; print(numpy.__version__)"],
            capture_output=True, text=True, timeout=30,
            env=env, **subprocess_kwargs,
        )
        if result.returncode != 0:
            return  # numpy not installed or broken, nothing to fix here

        version_str = result.stdout.strip()
        major = int(version_str.split(".")[0])
        if major >= 2:
            _log(
                "numpy {} detected (>=2.0), forcing downgrade to <2.0.0...".format(version_str),
                Qgis.Warning
            )
            downgrade_args = [
                "install", "--force-reinstall", "--no-deps",
                "--disable-pip-version-check",
            ] + _get_pip_ssl_flags() + [
                "numpy>=1.26.0,<2.0.0",
            ]
            downgrade_cmd = _build_install_cmd(python_path, downgrade_args)
            downgrade_result = subprocess.run(
                downgrade_cmd,
                capture_output=True, text=True, timeout=120,
                env=env, **subprocess_kwargs,
            )
            if downgrade_result.returncode == 0:
                _log("numpy downgraded successfully to <2.0.0", Qgis.Success)
            else:
                err = downgrade_result.stderr or downgrade_result.stdout or ""
                _log("numpy downgrade failed: {}".format(err[:200]), Qgis.Warning)
    except Exception as e:
        _log("numpy version check failed: {}".format(e), Qgis.Warning)


def _reinstall_cpu_torch(
    venv_dir: str,
    progress_callback: Optional[Callable[[int, str], None]] = None
):
    """
    Uninstall CUDA torch/torchvision and reinstall CPU versions from PyPI.

    Called as a fallback when CUDA verification fails after installation.
    Also re-pins numpy to <2.0.0 since CPU wheels may pull a different numpy.
    """
    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    _log("Reinstalling CPU-only torch/torchvision...", Qgis.Warning)
    if progress_callback:
        progress_callback(96, "CUDA failed, reinstalling CPU torch...")

    # Uninstall existing torch and torchvision
    try:
        uninstall_cmd = _build_uninstall_cmd(
            python_path, ["torch", "torchvision"])
        subprocess.run(
            uninstall_cmd,
            capture_output=True, text=True, timeout=120,
            env=env, **subprocess_kwargs,
        )
    except Exception as e:
        _log("torch uninstall error (continuing): {}".format(e), Qgis.Warning)

    # Install CPU versions from default PyPI (use same specs as REQUIRED_PACKAGES)
    cpu_packages = [
        "torch{}".format(TORCH_MIN),
        "torchvision{}".format(TORCHVISION_MIN),
    ]
    for pkg in cpu_packages:
        pkg_timeout = 1800 if pkg.startswith("torch>=") else 1200
        try:
            install_args = [
                "install", "--no-warn-script-location",
                "--disable-pip-version-check",
                "--prefer-binary",
            ] + _get_pip_ssl_flags() + [pkg]
            cmd = _build_install_cmd(python_path, install_args)
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=pkg_timeout,
                env=env, **subprocess_kwargs,
            )
            if result.returncode == 0:
                _log("✓ Installed {} (CPU)".format(pkg), Qgis.Success)
            else:
                err = result.stderr or result.stdout or ""
                _log("Failed to install {} (CPU): {}".format(pkg, err[:200]), Qgis.Warning)
        except Exception as e:
            _log("Exception installing {} (CPU): {}".format(pkg, e), Qgis.Warning)

    # Re-pin numpy after torch reinstall
    _repin_numpy(venv_dir)

    if progress_callback:
        progress_callback(98, "CPU torch installed, re-verifying...")


def _verify_cuda_in_venv(venv_dir: str) -> bool:
    """
    Run a small CUDA smoke test inside the venv to confirm torch.cuda works.

    Returns True if CUDA is functional, False otherwise.
    """
    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    cuda_test_code = (
        "import torch; "
        "print('torch=' + torch.__version__); "
        "print('cuda_built=' + str(torch.version.cuda)); "
        "assert torch.cuda.is_available(), 'CUDA not available'; "
        "print('device=' + torch.cuda.get_device_name(0)); "
        "t = torch.zeros(1, device='cuda'); "
        "torch.cuda.synchronize(); "
        "print('CUDA OK')"
    )

    try:
        result = subprocess.run(
            [python_path, "-c", cuda_test_code],
            capture_output=True, text=True, timeout=120,
            env=env, **subprocess_kwargs,
        )
        if result.returncode == 0 and "CUDA OK" in result.stdout:
            _log("CUDA verification passed: {}".format(
                result.stdout.strip()[:200]), Qgis.Success)
            return True
        else:
            out = result.stdout or ""
            err = result.stderr or ""
            _log("CUDA verification failed.\nstdout: {}\nstderr: {}".format(
                out[:200], err[:200]), Qgis.Warning)
            return False
    except Exception as e:
        _log("CUDA verification exception: {}".format(e), Qgis.Warning)
        return False


def _is_cpu_torch_installed(python_path: str, env: dict, subprocess_kwargs: dict) -> bool:
    """Return True if the installed torch has no CUDA support (CPU-only build).

    Detects the case where pip would silently skip a CUDA wheel because the
    installed CPU torch version is numerically higher than what is available
    on the CUDA wheel index (e.g. 2.10.0+cpu > 2.5.1+cu121).
    """
    try:
        result = subprocess.run(
            [python_path, "-c", "import torch; print(torch.version.cuda)"],
            capture_output=True, text=True, timeout=30,
            env=env, **subprocess_kwargs
        )
        if result.returncode == 0:
            return result.stdout.strip() == "None"
    except Exception:
        pass
    return False


def _get_verification_timeout(package_name: str) -> int:
    """
    Get verification timeout in seconds for a given package.

    torch needs extra time because the first import loads CUDA DLLs on Windows,
    which can take >30s. torchvision also loads heavy native libraries.
    """
    if package_name == "torch":
        return 120
    elif package_name in ("torchvision", "pandas"):
        # pandas loads many .pyd C extensions on first import;
        # antivirus (Windows Defender) scans each one, easily exceeding 30s
        return 120
    else:
        return 30


class _PipResult:
    """Lightweight result object compatible with subprocess.CompletedProcess."""

    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _parse_pip_download_line(line: str) -> Optional[str]:
    """
    Extract a human-readable status from a pip stdout/stderr line.

    Pip outputs lines like:
      "Downloading https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp312-...-linux_x86_64.whl (2449.3 MB)"
    Returns e.g. "Downloading torch (2.4 GB)" or None.
    """
    m = re.search(r"Downloading\s+(\S+)\s+\(([^)]+)\)", line)
    if not m:
        return None

    raw_name = m.group(1)
    size = m.group(2)

    # Strip URL prefix: keep only the filename from the path
    if "/" in raw_name:
        raw_name = raw_name.rsplit("/", 1)[-1]

    # Extract just the package name (before version): "torch-2.5.1%2Bcu121-cp312-..." -> "torch"
    # URL-encoded + is %2B, so split on first "-" that is followed by a digit
    name_match = re.match(r"([A-Za-z][A-Za-z0-9_]*)", raw_name)
    pkg_name = name_match.group(1) if name_match else raw_name

    # Convert size to human-friendly: "2449.3 MB" -> "2.4 GB"
    size_match = re.match(r"([\d.]+)\s*(kB|MB|GB)", size)
    if size_match:
        num = float(size_match.group(1))
        unit = size_match.group(2)
        if unit == "MB" and num >= 1000:
            size = "{:.1f} GB".format(num / 1000)

    return "Downloading {} ({})".format(pkg_name, size)


def _run_pip_install(
    cmd: List[str],
    timeout: int,
    env: dict,
    subprocess_kwargs: dict,
    package_name: str,
    package_index: int,
    total_packages: int,
    progress_start: int,
    progress_end: int,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    is_cuda: bool = False,
) -> _PipResult:
    """
    Run a pip install command with real-time progress updates.

    Uses Popen with stdout/stderr redirected to temp files (not PIPE,
    per CLAUDE.md) and polls every 2 seconds to provide live feedback.
    """
    poll_interval = 2  # seconds

    # Create temp files for stdout and stderr
    stdout_fd, stdout_path = tempfile.mkstemp(
        suffix="_stdout.txt", prefix="pip_"
    )
    stderr_fd, stderr_path = tempfile.mkstemp(
        suffix="_stderr.txt", prefix="pip_"
    )

    try:
        stdout_file = os.fdopen(stdout_fd, "w", encoding="utf-8")
        stderr_file = os.fdopen(stderr_fd, "w", encoding="utf-8")
    except Exception:
        # If fdopen fails, close the raw fds and re-raise
        try:
            os.close(stdout_fd)
        except Exception:
            pass
        try:
            os.close(stderr_fd)
        except Exception:
            pass
        raise

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            env=env,
            **subprocess_kwargs,
        )

        start_time = time.monotonic()
        last_download_status = ""

        while True:
            try:
                process.wait(timeout=poll_interval)
                # Process finished
                break
            except subprocess.TimeoutExpired:
                pass  # Still running, continue polling

            elapsed = int(time.monotonic() - start_time)

            # Check cancellation
            if cancel_check and cancel_check():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                return _PipResult(-1, "", "Installation cancelled")

            # Check overall timeout
            if elapsed >= timeout:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise subprocess.TimeoutExpired(cmd, timeout)

            # Read last lines of stdout to find download progress
            try:
                with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
                    # Read last 4KB to find recent lines
                    f.seek(0, 2)  # seek to end
                    file_size = f.tell()
                    read_from = max(0, file_size - 4096)
                    f.seek(read_from)
                    tail = f.read()
                    lines = tail.strip().split("\n")
                    # Search from bottom for a Downloading line
                    for line in reversed(lines):
                        parsed = _parse_pip_download_line(line)
                        if parsed:
                            last_download_status = parsed
                            break
            except Exception:
                pass

            # Format elapsed time nicely
            if elapsed >= 60:
                elapsed_str = "{}m {}s".format(elapsed // 60, elapsed % 60)
            else:
                elapsed_str = "{}s".format(elapsed)

            # Build progress message
            if last_download_status:
                msg = "{}... {}".format(last_download_status, elapsed_str)
            elif is_cuda and package_name == "torch":
                msg = "Installing GPU PyTorch (~2.5 GB)... {}".format(
                    elapsed_str)
            elif package_name == "torch":
                msg = "Downloading PyTorch (~600 MB)... {}".format(
                    elapsed_str)
            else:
                msg = "Installing {}... {}".format(
                    package_name, elapsed_str)

            # Interpolate progress within the package's range
            # Use logarithmic-ish curve: fast at start, slows down
            # Cap interpolated progress at 90% of the range
            progress_range = progress_end - progress_start
            if timeout > 0:
                fraction = min(elapsed / timeout, 0.9)
            else:
                fraction = 0
            interpolated = progress_start + int(progress_range * fraction)
            interpolated = min(interpolated, progress_end - 1)

            if progress_callback:
                progress_callback(interpolated, msg)

        # Process finished — close files before reading
        stdout_file.close()
        stderr_file.close()
        stdout_file = None
        stderr_file = None

        # Read full output
        try:
            with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
                full_stdout = f.read()
        except Exception:
            full_stdout = ""

        try:
            with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
                full_stderr = f.read()
        except Exception:
            full_stderr = ""

        return _PipResult(process.returncode, full_stdout, full_stderr)

    except subprocess.TimeoutExpired:
        raise
    except Exception:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()
        raise
    finally:
        # Close files if still open
        if stdout_file is not None:
            try:
                stdout_file.close()
            except Exception:
                pass
        if stderr_file is not None:
            try:
                stderr_file.close()
            except Exception:
                pass
        # Clean up temp files
        try:
            os.unlink(stdout_path)
        except Exception:
            pass
        try:
            os.unlink(stderr_path)
        except Exception:
            pass


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

    # Upgrade pip before installing packages. The standalone Python bundles
    # pip 24.3.1 which can crash with internal exceptions on large packages
    # like torch (see issue #145). Skip when using uv.
    python_path_pre = get_venv_python_path(venv_dir)
    if _uv_available:
        _log("Using uv for installation, skipping pip upgrade", Qgis.Info)
        if progress_callback:
            progress_callback(20, "Using uv package installer...")
    else:
        if progress_callback:
            progress_callback(20, "Upgrading pip...")
        try:
            _log("Upgrading pip to latest version...", Qgis.Info)
            upgrade_cmd = [
                python_path_pre, "-m", "pip", "install",
                "--upgrade", "pip",
                "--disable-pip-version-check",
                "--no-warn-script-location",
            ]
            upgrade_cmd.extend(_get_pip_ssl_flags())
            upgrade_result = subprocess.run(
                upgrade_cmd,
                capture_output=True, text=True, timeout=120,
                env=_get_clean_env_for_venv(),
                **_get_subprocess_kwargs(),
            )
            if upgrade_result.returncode == 0:
                _log("pip upgraded successfully", Qgis.Success)
            else:
                _log("pip upgrade failed (non-critical): {}".format(
                    (upgrade_result.stderr or upgrade_result.stdout or "")[:200]),
                    Qgis.Warning)
        except Exception as e:
            _log("pip upgrade failed (non-critical): {}".format(str(e)[:200]),
                 Qgis.Warning)

    _cuda_fell_back = False  # Track if CUDA install fell back to CPU
    _driver_too_old = False  # Track if GPU driver was too old (not an error)

    total_packages = len(REQUIRED_PACKAGES)
    base_progress = 20
    progress_range = 80  # from 20% to 100%

    # Weighted progress allocation proportional to download size.
    # Name-based map so weights stay correct if REQUIRED_PACKAGES order changes.
    _weight_map_cuda = {
        "numpy": 5, "torch": 45, "torchvision": 15,
        "pandas": 5, "rasterio": 5,
    }
    _weight_map_cpu = {
        "numpy": 5, "torch": 30, "torchvision": 15,
        "pandas": 10, "rasterio": 10,
    }
    _wmap = _weight_map_cuda if cuda_enabled else _weight_map_cpu
    _weights = [_wmap.get(name, 10) for name, _ in REQUIRED_PACKAGES]
    weight_total = sum(_weights)
    # Cumulative start offsets for each package
    _cumulative = [0]
    for w in _weights:
        _cumulative.append(_cumulative[-1] + w)

    def _pkg_progress_start(idx):
        return base_progress + int(progress_range * _cumulative[idx] / weight_total)

    def _pkg_progress_end(idx):
        return base_progress + int(progress_range * _cumulative[idx + 1] / weight_total)

    python_path = get_venv_python_path(venv_dir)

    # Create a pip constraints file to prevent numpy 2.x drift.
    # When pip resolves torch from the CUDA index, it may pull numpy>=2.0
    # as a dependency, ignoring our version spec. The constraints file
    # forces pip to honour the upper bound on every install command.
    constraints_fd, constraints_path = tempfile.mkstemp(
        suffix=".txt", prefix="pip_constraints_"
    )
    try:
        with os.fdopen(constraints_fd, "w", encoding="utf-8") as f:
            f.write("numpy<2.0.0\n")
        _log(f"Created pip constraints file: {constraints_path}", Qgis.Info)
    except Exception as e:
        _log(f"Failed to write constraints file: {e}", Qgis.Warning)
        # Clean up fd/file on failure so nothing is leaked
        try:
            os.close(constraints_fd)
        except Exception:
            pass
        try:
            os.unlink(constraints_path)
        except Exception:
            pass
        constraints_path = None

    try:  # try/finally to guarantee constraints file cleanup

        # Detect CPU-only torch already in the venv. pip's --upgrade skips
        # the CUDA wheel when the installed CPU torch version is numerically
        # higher than the latest wheel on the CUDA index (e.g. 2.10.0+cpu >
        # 2.5.1+cu121). Use --force-reinstall to guarantee the CUDA build is
        # actually fetched in that case.
        _force_cuda_reinstall = False
        if cuda_enabled:
            _precheck_env = _get_clean_env_for_venv()
            _precheck_kwargs = _get_subprocess_kwargs()
            if _is_cpu_torch_installed(python_path, _precheck_env, _precheck_kwargs):
                _force_cuda_reinstall = True
                _log(
                    "CPU torch detected in venv, CUDA packages will use "
                    "--force-reinstall", Qgis.Info)

        for i, (package_name, version_spec) in enumerate(REQUIRED_PACKAGES):
            if cancel_check and cancel_check():
                _log("Installation cancelled by user", Qgis.Warning)
                return False, "Installation cancelled"

            package_spec = f"{package_name}{version_spec}"
            pkg_start = _pkg_progress_start(i)
            pkg_end = _pkg_progress_end(i)

            is_cuda_package = cuda_enabled and package_name in ("torch", "torchvision")

            if progress_callback:
                if package_name == "torch" and cuda_enabled:
                    progress_callback(
                        pkg_start,
                        "Installing GPU dependencies... ({}/{})".format(
                            i + 1, total_packages))
                elif package_name == "torch":
                    progress_callback(
                        pkg_start,
                        "Installing {} (~600MB)... ({}/{})".format(
                            package_name, i + 1, total_packages))
                else:
                    progress_callback(
                        pkg_start,
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
            # sam2/segment-anything list torch as a build dependency.
            # Without --no-build-isolation pip creates a separate env and
            # re-downloads torch (~2.5 GB), which often fails. Since torch
            # is already installed in the venv at this point, skip isolation.
            if package_name in ("sam2", "segment-anything"):
                pip_args.append("--no-build-isolation")
            # Add SSL bypass flags upfront for corporate proxies (not just as retry)
            pip_args.extend(_get_pip_ssl_flags())
            if constraints_path:
                pip_args.extend(["--constraint", constraints_path])
            pip_args.extend(_get_pip_proxy_args())
            pip_args.append(package_spec)

            # For CUDA-enabled torch/torchvision, use PyTorch's CUDA index
            if is_cuda_package:
                _, gpu_info = detect_nvidia_gpu()
                cuda_index = _select_cuda_index(gpu_info)
                if cuda_index is None:
                    # Driver too old for any CUDA toolkit -> fall back to CPU
                    _log(
                        "Driver too old for CUDA, installing CPU {} instead".format(
                            package_name),
                        Qgis.Warning
                    )
                    is_cuda_package = False
                    _driver_too_old = True
                else:
                    pip_args.extend([
                        "--index-url",
                        "https://download.pytorch.org/whl/{}".format(cuda_index),
                        "--no-cache-dir",
                    ])
                    _log("Using CUDA {} index for {}".format(
                        cuda_index, package_name), Qgis.Info)

            # Use clean env to avoid QGIS PYTHONPATH/PYTHONHOME interference
            env = _get_clean_env_for_venv()

            subprocess_kwargs = _get_subprocess_kwargs()

            # If CPU torch/torchvision is installed and we need CUDA, uninstall
            # the CPU version first. This forces pip to fetch the CUDA wheel
            # from the index instead of silently skipping because the installed
            # CPU version has a higher version number.
            # We do NOT use --force-reinstall because --index-url points to the
            # CUDA-only index which lacks transitive deps (typing-extensions,
            # sympy, etc.), causing pip to fail resolving them.
            if _force_cuda_reinstall and is_cuda_package:
                _log("Uninstalling CPU {} before CUDA install".format(
                    package_name), Qgis.Info)
                try:
                    uninstall_cmd = _build_uninstall_cmd(
                        python_path, [package_name])
                    subprocess.run(
                        uninstall_cmd,
                        capture_output=True, text=True, timeout=120,
                        env=env, **subprocess_kwargs
                    )
                except Exception as exc:
                    _log("Failed to uninstall CPU {}: {}".format(
                        package_name, exc), Qgis.Warning)

            # Large packages need more time than standard packages
            if is_cuda_package and package_name in ("torch", "torchvision"):
                pkg_timeout = 2400  # 40 min for CUDA wheels (~2.5GB)
            elif package_name == "torch":
                pkg_timeout = 3600  # 60 min for CPU torch on slow connections
            elif package_name == "torchvision":
                pkg_timeout = 1200  # 20 min for CPU torchvision
            else:
                pkg_timeout = 600  # 10 min for standard packages

            install_failed = False
            install_error_msg = ""
            last_returncode = None

            try:
                # Build install command (uv or pip depending on availability)
                base_cmd = _build_install_cmd(python_path, pip_args)

                # First attempt: pip install with real-time progress
                result = _run_pip_install(
                    cmd=base_cmd,
                    timeout=pkg_timeout,
                    env=env,
                    subprocess_kwargs=subprocess_kwargs,
                    package_name=package_name,
                    package_index=i,
                    total_packages=total_packages,
                    progress_start=pkg_start,
                    progress_end=pkg_end,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                    is_cuda=is_cuda_package,
                )

                # If cancelled
                if result.returncode == -1 and "cancelled" in (result.stderr or "").lower():
                    _log("Installation cancelled by user", Qgis.Warning)
                    return False, "Installation cancelled"

                # If Windows process crash, retry with pip.exe as fallback
                # (skip when using uv - it doesn't have this issue)
                if not _uv_available and _is_windows_process_crash(result.returncode):
                    _log(
                        "Process crash detected (code {}), "
                        "retrying with pip.exe...".format(result.returncode),
                        Qgis.Warning
                    )
                    if progress_callback:
                        progress_callback(
                            pkg_start,
                            "Retrying {}... ({}/{})".format(
                                package_name, i + 1, total_packages)
                        )

                    fallback_cmd = [pip_path] + pip_args
                    result = _run_pip_install(
                        cmd=fallback_cmd,
                        timeout=pkg_timeout,
                        env=env,
                        subprocess_kwargs=subprocess_kwargs,
                        package_name=package_name,
                        package_index=i,
                        total_packages=total_packages,
                        progress_start=pkg_start,
                        progress_end=pkg_end,
                        progress_callback=progress_callback,
                        cancel_check=cancel_check,
                        is_cuda=is_cuda_package,
                    )

                # If "unable to create process" (broken pip shim), retry with pip.exe
                # (skip when using uv - it doesn't have this issue)
                if not _uv_available and result.returncode != 0:
                    error_output = result.stderr or result.stdout or ""
                    if _is_unable_to_create_process(error_output):
                        _log(
                            "Unable to create process detected, "
                            "retrying with pip.exe...",
                            Qgis.Warning
                        )
                        if progress_callback:
                            progress_callback(
                                pkg_start,
                                "Retrying {}... ({}/{})".format(
                                    package_name, i + 1, total_packages)
                            )
                        fallback_cmd = [pip_path] + pip_args
                        result = _run_pip_install(
                            cmd=fallback_cmd,
                            timeout=pkg_timeout,
                            env=env,
                            subprocess_kwargs=subprocess_kwargs,
                            package_name=package_name,
                            package_index=i,
                            total_packages=total_packages,
                            progress_start=pkg_start,
                            progress_end=pkg_end,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                            is_cuda=is_cuda_package,
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
                                pkg_start,
                                "SSL error, retrying {}... ({}/{})".format(
                                    package_name, i + 1, total_packages)
                            )

                        # Re-run with SSL flags (already present but retry in case)
                        ssl_flags = _get_pip_ssl_flags()
                        ssl_cmd = base_cmd + ssl_flags
                        result = _run_pip_install(
                            cmd=ssl_cmd,
                            timeout=pkg_timeout,
                            env=env,
                            subprocess_kwargs=subprocess_kwargs,
                            package_name=package_name,
                            package_index=i,
                            total_packages=total_packages,
                            progress_start=pkg_start,
                            progress_end=pkg_end,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                            is_cuda=is_cuda_package,
                        )

                # If failed, check for hash mismatch (corrupted cache) and retry
                if result.returncode != 0 and not _is_windows_process_crash(result.returncode):
                    error_output = result.stderr or result.stdout or ""

                    if _is_hash_mismatch(error_output):
                        _log(
                            "Hash mismatch detected (corrupted cache), "
                            "retrying with --no-cache-dir...",
                            Qgis.Warning
                        )
                        if progress_callback:
                            progress_callback(
                                pkg_start,
                                "Cache error, retrying {}... ({}/{})".format(
                                    package_name, i + 1, total_packages)
                            )

                        nocache_flag = "--no-cache" if _uv_available else "--no-cache-dir"
                        nocache_cmd = base_cmd + [nocache_flag]
                        result = _run_pip_install(
                            cmd=nocache_cmd,
                            timeout=pkg_timeout,
                            env=env,
                            subprocess_kwargs=subprocess_kwargs,
                            package_name=package_name,
                            package_index=i,
                            total_packages=total_packages,
                            progress_start=pkg_start,
                            progress_end=pkg_end,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                            is_cuda=is_cuda_package,
                        )

                # If failed, check for network errors and retry after delay
                if result.returncode != 0 and not _is_windows_process_crash(result.returncode):
                    error_output = result.stderr or result.stdout or ""

                    if _is_network_error(error_output):
                        for attempt in range(1, 5):  # up to 4 retries
                            wait = 5 * (2 ** (attempt - 1))  # 5, 10, 20, 40s
                            _log(
                                "Network error detected, retrying in {}s "
                                "(attempt {}/4)...".format(wait, attempt),
                                Qgis.Warning
                            )
                            if progress_callback:
                                progress_callback(
                                    pkg_start,
                                    "Network error, retry {}/4 in {}s...".format(
                                        attempt, wait)
                                )
                            time.sleep(wait)
                            if cancel_check and cancel_check():
                                return False, "Installation cancelled"
                            result = _run_pip_install(
                                cmd=base_cmd,
                                timeout=pkg_timeout,
                                env=env,
                                subprocess_kwargs=subprocess_kwargs,
                                package_name=package_name,
                                package_index=i,
                                total_packages=total_packages,
                                progress_start=pkg_start,
                                progress_end=pkg_end,
                                progress_callback=progress_callback,
                                cancel_check=cancel_check,
                                is_cuda=is_cuda_package,
                            )
                            if result.returncode == 0:
                                break

                # If "no matching distribution" for torch, retry with --no-cache-dir
                if result.returncode != 0 and package_name in ("torch", "torchvision"):
                    error_output = result.stderr or result.stdout or ""
                    if "no matching distribution" in error_output.lower():
                        _log(
                            "No matching distribution for {}, "
                            "retrying with --no-cache-dir...".format(package_name),
                            Qgis.Warning
                        )
                        nocache2 = "--no-cache" if _uv_available else "--no-cache-dir"
                        nocache_cmd = base_cmd + [nocache2]
                        result = _run_pip_install(
                            cmd=nocache_cmd,
                            timeout=pkg_timeout,
                            env=env,
                            subprocess_kwargs=subprocess_kwargs,
                            package_name=package_name,
                            package_index=i,
                            total_packages=total_packages,
                            progress_start=pkg_start,
                            progress_end=pkg_end,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                            is_cuda=is_cuda_package,
                        )

                if result.returncode == 0:
                    _log(f"✓ Successfully installed {package_spec}", Qgis.Success)
                    if progress_callback:
                        progress_callback(pkg_end, f"✓ {package_name} installed")
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
                        pkg_start,
                        "CUDA failed, installing {} (CPU)...".format(package_name)
                    )

                cpu_pip_args = [
                    "install", "--upgrade", "--no-warn-script-location",
                    "--disable-pip-version-check", "--prefer-binary",
                ]
                # Add SSL bypass flags for corporate proxies
                cpu_pip_args.extend(_get_pip_ssl_flags())
                cpu_pip_args.append(package_spec)
                if constraints_path:
                    cpu_pip_args.extend(["--constraint", constraints_path])
                cpu_cmd = _build_install_cmd(python_path, cpu_pip_args)
                cpu_timeout = 1800 if package_name == "torch" else 1200
                try:
                    cpu_result = _run_pip_install(
                        cmd=cpu_cmd,
                        timeout=cpu_timeout,
                        env=env,
                        subprocess_kwargs=subprocess_kwargs,
                        package_name=package_name,
                        package_index=i,
                        total_packages=total_packages,
                        progress_start=pkg_start,
                        progress_end=pkg_end,
                        progress_callback=progress_callback,
                        cancel_check=cancel_check,
                        is_cuda=False,
                    )
                    if cpu_result.returncode == 0:
                        _log("✓ Successfully installed {} (CPU version)".format(package_spec), Qgis.Success)
                        if progress_callback:
                            progress_callback(
                                pkg_end,
                                "✓ {} installed (CPU)".format(package_name)
                            )
                        install_failed = False
                        _cuda_fell_back = True
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
                # Log detailed pip output for debugging
                _log("pip error output: {}".format(install_error_msg[:500]), Qgis.Critical)

                # Check for Windows process crash
                if last_returncode is not None and _is_windows_process_crash(last_returncode):
                    _log(_get_crash_help(venv_dir), Qgis.Warning)
                    return False, "Failed to install {}: process crashed (code {})".format(
                        package_name, last_returncode)

                # Check for DLL load failure (missing system libraries)
                is_dll_err = sys.platform == "win32" and _is_dll_init_error(install_error_msg)
                if is_dll_err and package_name in ("torch", "torchvision"):
                    _log(_get_vcpp_help(), Qgis.Warning)
                    return False, "Failed to install {}: {}".format(
                        package_name, _get_vcpp_help())

                # Check for SSL errors
                if _is_ssl_error(install_error_msg):
                    _log(_get_ssl_error_help(install_error_msg), Qgis.Warning)
                    return False, "Failed to install {}: SSL error".format(
                        package_name)

                # Check for proxy authentication errors (407)
                if _is_proxy_auth_error(install_error_msg):
                    _log(
                        "Proxy authentication failed (HTTP 407). "
                        "Configure proxy credentials in: "
                        "QGIS > Settings > Options > Network > Proxy "
                        "(User and Password fields).",
                        Qgis.Warning
                    )
                    return False, "Failed to install {}: proxy authentication required (407)".format(
                        package_name)

                # Check for network/connection errors (after retries exhausted)
                if _is_network_error(install_error_msg):
                    _log(
                        "Network connection failed after multiple retries. "
                        "Check internet connection, VPN/proxy settings, "
                        "and firewall rules for pypi.org and files.pythonhosted.org.",
                        Qgis.Warning
                    )
                    return False, "Failed to install {}: network error".format(
                        package_name)

                # Check for antivirus blocking
                if _is_antivirus_error(install_error_msg):
                    _log(_get_pip_antivirus_help(venv_dir), Qgis.Warning)
                    return False, "Failed to install {}: blocked by antivirus or security policy".format(
                        package_name)

                # Check for "unable to create process" (broken pip shim)
                if _is_unable_to_create_process(install_error_msg):
                    return False, (
                        "Failed to install {}: unable to create process.\n\n"
                        "Please try:\n"
                        "  1. Delete the folder: {}\n"
                        "  2. Restart QGIS and reinstall dependencies".format(
                            package_name, CACHE_DIR)
                    )

                # Check for GDAL issues on Linux when rasterio fails
                if package_name == "rasterio":
                    gdal_ok, gdal_help = _check_gdal_available()
                    if not gdal_ok and gdal_help:
                        _log(gdal_help, Qgis.Warning)
                        return False, "Failed to install {}: GDAL library not found".format(
                            package_name)

                return False, "Failed to install {}: {}".format(
                    package_name, install_error_msg[:200])

        # Post-install numpy version safety net:
        # Even with constraints, the CUDA index may have pulled numpy>=2.0.
        # Check and force-downgrade if needed.
        _repin_numpy(venv_dir)

        if progress_callback:
            progress_callback(100, "✓ All dependencies installed")

        _log("=" * 50, Qgis.Success)
        _log("All dependencies installed successfully!", Qgis.Success)
        _log(f"Virtual environment: {venv_dir}", Qgis.Success)
        _log("=" * 50, Qgis.Success)

        if _driver_too_old:
            return True, "All dependencies installed successfully [DRIVER_TOO_OLD]"
        if _cuda_fell_back:
            return True, "All dependencies installed successfully [CUDA_FALLBACK]"
        return True, "All dependencies installed successfully"

    finally:
        # Always clean up the constraints temp file
        if constraints_path:
            try:
                os.unlink(constraints_path)
            except Exception:
                pass


def _get_qgis_proxy_settings() -> Optional[str]:
    """Read proxy configuration from QGIS settings.

    Returns a proxy URL string (with optional authentication)
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
        'PROJ_DATA', 'PROJ_LIB',
        'GDAL_DATA', 'GDAL_DRIVER_PATH',
    ]
    for var in vars_to_remove:
        env.pop(var, None)

    env["PYTHONIOENCODING"] = "utf-8"

    # Skip sam2 CUDA extension compilation (Python fallback works fine)
    env["SAM2_BUILD_CUDA"] = "0"

    # On Linux, QGIS desktop launchers often don't inherit LD_LIBRARY_PATH,
    # so CUDA libraries may not be discoverable. Probe standard locations.
    if sys.platform == "linux":
        cuda_lib_dirs = []
        cuda_path = env.get("CUDA_PATH", "")
        if cuda_path:
            cuda_lib_dirs.append(os.path.join(cuda_path, "lib64"))
        for candidate in ("/usr/local/cuda/lib64", "/opt/cuda/lib64"):
            if os.path.isdir(candidate) and candidate not in cuda_lib_dirs:
                cuda_lib_dirs.append(candidate)
        if cuda_lib_dirs:
            existing = env.get("LD_LIBRARY_PATH", "")
            parts = [p for p in existing.split(":") if p]
            for d in cuda_lib_dirs:
                if d not in parts:
                    parts.append(d)
            env["LD_LIBRARY_PATH"] = ":".join(parts)

    # Propagate QGIS proxy settings to environment for pip/network calls
    proxy_url = _get_qgis_proxy_settings()
    if proxy_url:
        env.setdefault("HTTP_PROXY", proxy_url)
        env.setdefault("HTTPS_PROXY", proxy_url)

    return env


def _get_subprocess_kwargs() -> dict:
    # Set cwd to CACHE_DIR so the subprocess cannot accidentally discover
    # the plugin package if launched from the plugin directory.
    os.makedirs(CACHE_DIR, exist_ok=True)
    kwargs = {"cwd": CACHE_DIR}
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
    elif package_name == "sam2":
        return "from sam2.build_sam import build_sam2; print('ok')"
    elif package_name == "segment-anything":
        return "from segment_anything import sam_model_registry; print('ok')"
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
        pkg_timeout = _get_verification_timeout(package_name)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=pkg_timeout,
                env=env,
                **subprocess_kwargs
            )

            if result.returncode != 0:
                error_detail = result.stderr[:300] if result.stderr else result.stdout[:300]
                _log(
                    "Package {} verification failed: {}".format(
                        package_name, error_detail),
                    Qgis.Warning
                )

                # DLL init error (WinError 1114) = missing VC++ Redistributables
                if _is_dll_init_error(error_detail):
                    _log(_get_vcpp_help(), Qgis.Warning)
                    return False, (
                        "Package {} failed: {}".format(
                            package_name, _get_vcpp_help())
                    )

                # Detect broken C extensions (antivirus may have quarantined .pyd files)
                error_lower = error_detail.lower()
                broken_markers = [
                    "no module named", "_libs",
                    "dll load failed", "importerror",
                    "applocker", "application control",
                    "blocked by your organization",
                ]
                is_broken = any(m in error_lower for m in broken_markers)

                if is_broken:
                    _log(
                        "Package {} has broken C extensions, "
                        "attempting force-reinstall...".format(package_name),
                        Qgis.Warning
                    )
                    # Find the version spec from REQUIRED_PACKAGES
                    pkg_spec = package_name
                    for name, spec in REQUIRED_PACKAGES:
                        if name == package_name:
                            pkg_spec = "{}{}".format(name, spec)
                            break
                    reinstall_cmd = [
                        python_path, "-m", "pip", "install",
                        "--force-reinstall", "--no-deps",
                        "--disable-pip-version-check",
                        "--prefer-binary",
                    ] + _get_pip_ssl_flags() + [
                        pkg_spec,
                    ]
                    try:
                        subprocess.run(
                            reinstall_cmd,
                            capture_output=True,
                            text=True,
                            timeout=300,
                            env=env,
                            **subprocess_kwargs
                        )
                    except Exception:
                        pass
                    # Retry verification after force-reinstall
                    try:
                        result2 = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=pkg_timeout,
                            env=env,
                            **subprocess_kwargs
                        )
                        if result2.returncode == 0:
                            _log(
                                "Package {} fixed after force-reinstall".format(
                                    package_name),
                                Qgis.Success
                            )
                            continue  # Move to next package
                    except Exception:
                        pass
                    # Still broken after reinstall - check for AppLocker
                    detail_lower = error_detail.lower()
                    applocker_markers = [
                        "applocker", "application control",
                        "blocked by your organization",
                    ]
                    if any(m in detail_lower for m in applocker_markers):
                        return False, (
                            "Package {} is blocked by AppLocker or "
                            "application control policy.\n\n"
                            "Ask your IT administrator to whitelist "
                            "this folder:\n  {}\n\n"
                            "Then restart QGIS and reinstall "
                            "dependencies.".format(
                                package_name, venv_dir)
                        )
                    return False, (
                        "Package {} is broken (antivirus may be "
                        "interfering): {}".format(
                            package_name, error_detail[:200])
                    )

                if sys.platform == "win32":
                    vcpp_url = (
                        "https://aka.ms/vs/17/release/"
                        "vc_redist.x64.exe"
                    )
                    if package_name == "torch":
                        hint = (
                            "\n\nPlease try:\n"
                            "  1. Install Visual C++ "
                            "Redistributable:\n"
                            "     {}\n"
                            "  2. Add an antivirus exclusion for "
                            "the plugin folder\n"
                            "  3. Restart QGIS".format(vcpp_url)
                        )
                    else:
                        hint = (
                            "\n\nPlease try:\n"
                            "  1. Install Visual C++ "
                            "Redistributable:\n"
                            "     {}\n"
                            "  2. Click 'Reinstall dependencies' "
                            "in the plugin panel\n"
                            "  3. Restart QGIS".format(vcpp_url)
                        )
                else:
                    hint = (
                        "\n\nPlease try:\n"
                        "  1. Click 'Reinstall dependencies' "
                        "in the plugin panel\n"
                        "  2. Restart QGIS"
                    )
                return False, "Package {} is broken: {}{}".format(
                    package_name, error_detail[:200], hint)

        except subprocess.TimeoutExpired:
            # Retry once - antivirus scanning .pyd files on first import
            # causes timeouts; the 2nd import is near-instant once cached
            _log(
                "Verification of {} timed out ({}s), retrying...".format(
                    package_name, pkg_timeout),
                Qgis.Info
            )
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=pkg_timeout,
                    env=env,
                    **subprocess_kwargs
                )
                if result.returncode != 0:
                    error_detail = (
                        result.stderr[:300] if result.stderr
                        else result.stdout[:300]
                    )
                    _log(
                        "Package {} verification failed on retry: {}".format(
                            package_name, error_detail),
                        Qgis.Warning
                    )
                    # Check for AppLocker on retry too
                    retry_lower = error_detail.lower()
                    applocker_kw = [
                        "applocker", "application control",
                        "blocked by your organization",
                    ]
                    if any(m in retry_lower for m in applocker_kw):
                        return False, (
                            "Package {} is blocked by AppLocker "
                            "or application control policy.\n\n"
                            "Ask your IT administrator to "
                            "whitelist this folder:\n"
                            "  {}\n\nThen restart QGIS and "
                            "reinstall dependencies.".format(
                                package_name, venv_dir)
                        )
                    return False, "Package {} is broken: {}".format(
                        package_name, error_detail[:200])
            except subprocess.TimeoutExpired:
                _log(
                    "Verification of {} timed out twice".format(package_name),
                    Qgis.Warning
                )
                return False, (
                    "Verification error: {} "
                    "(timed out - antivirus may be blocking)".format(
                        package_name)
                )
            except Exception as e:
                _log(
                    "Failed to verify {} on retry: {}".format(
                        package_name, str(e)),
                    Qgis.Warning
                )
                return False, "Verification error: {}".format(package_name)

        except Exception as e:
            _log(
                "Failed to verify {}: {}".format(package_name, str(e)),
                Qgis.Warning
            )
            return False, "Verification error: {}".format(package_name)

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
    Complete installation: download Python standalone + download uv + create venv + install packages.

    Progress breakdown:
    - 0-10%:   Download Python standalone (~50MB)
    - 10-13%:  Download uv package installer (non-fatal if fails)
    - 13-18%:  Create virtual environment
    - 18-95%:  Install packages (~800MB, or ~2.5GB with CUDA)
    - 95-100%: Verify installation
    """
    from .python_manager import (
        standalone_python_exists,
        standalone_python_is_current,
        download_python_standalone,
        get_python_full_version,
        remove_standalone_python,
    )

    # Early Python version check (Issue #148)
    if sys.version_info < (3, 9):
        py_ver = "{}.{}.{}".format(
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro)
        return False, (
            "Python {} is too old. AI Segmentation requires "
            "Python 3.9+.\nPlease upgrade to QGIS 3.22 or later.".format(py_ver)
        )

    # Log system info for debugging
    _log_system_info()

    # Early writability check on CACHE_DIR
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        test_file = os.path.join(CACHE_DIR, ".write_test")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_file)
    except (OSError, IOError) as e:
        hint = (
            "Cannot write to install directory: {}\n"
            "Error: {}\n\n"
            "Set the AI_SEGMENTATION_CACHE_DIR environment variable "
            "to a writable directory, then restart QGIS."
        ).format(CACHE_DIR, e)
        _log(hint, Qgis.Critical)
        return False, hint

    # Check for Rosetta emulation on macOS (warning only, don't block)
    rosetta_warning = _check_rosetta_warning()
    if rosetta_warning:
        _log(rosetta_warning, Qgis.Warning)

    # Clean up old venv directories from previous Python versions
    removed_venvs = cleanup_old_venv_directories()
    if removed_venvs:
        _log(f"Removed {len(removed_venvs)} old venv directories", Qgis.Info)

    cleanup_old_libs()

    # If standalone Python exists but version doesn't match QGIS, delete and re-download
    if standalone_python_exists() and not standalone_python_is_current():
        _log(
            "Standalone Python version mismatch, re-downloading...",
            Qgis.Warning)
        remove_standalone_python()
        # Also remove the venv since it was built with the wrong Python
        if venv_exists():
            try:
                shutil.rmtree(VENV_DIR)
                _log("Removed stale venv after Python version mismatch", Qgis.Info)
            except Exception as e:
                _log("Failed to remove stale venv: {}".format(e), Qgis.Warning)

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

    # Step 1b: Download uv package installer (non-fatal if fails)
    global _uv_available, _uv_path
    if uv_exists() and verify_uv():
        _uv_available = True
        _uv_path = get_uv_path()
        _log("uv already installed, using for package management", Qgis.Info)
        if progress_callback:
            progress_callback(13, "uv package installer ready")
    else:
        if progress_callback:
            progress_callback(10, "Downloading uv package installer...")
        try:
            def uv_progress(percent, uv_msg):
                if progress_callback:
                    progress_callback(10 + int(percent * 0.03), uv_msg)

            uv_ok, uv_msg = download_uv(
                progress_callback=uv_progress,
                cancel_check=cancel_check,
            )
            if uv_ok:
                _uv_available = True
                _uv_path = get_uv_path()
                _log("uv downloaded, using for package management", Qgis.Info)
            else:
                _uv_available = False
                _uv_path = None
                _log("uv download failed (non-fatal), using pip: {}".format(
                    uv_msg), Qgis.Warning)
        except Exception as e:
            _uv_available = False
            _uv_path = None
            _log("uv download failed (non-fatal): {}".format(e), Qgis.Warning)
        if progress_callback:
            progress_callback(13, "uv: {}".format(
                "ready" if _uv_available else "unavailable, using pip"))

    if cancel_check and cancel_check():
        return False, "Installation cancelled"

    # Step 2: Create virtual environment if needed
    if venv_exists():
        _log("Virtual environment already exists", Qgis.Info)
        if progress_callback:
            progress_callback(18, "Virtual environment ready")
    else:
        def venv_progress(percent, msg):
            # Map 0-100 to 13-18
            if progress_callback:
                progress_callback(13 + int(percent * 0.05), msg)

        success, msg = create_venv(progress_callback=venv_progress)
        if not success:
            return False, msg

        if cancel_check and cancel_check():
            return False, "Installation cancelled"

    # Step 3: Install dependencies
    def deps_progress(percent, msg):
        # Map 20-100 to 18-95
        if progress_callback:
            mapped = 18 + int((percent - 20) * (95 - 18) / 80)
            progress_callback(min(mapped, 95), msg)

    success, msg = install_dependencies(
        progress_callback=deps_progress,
        cancel_check=cancel_check,
        cuda_enabled=cuda_enabled
    )

    if not success:
        return False, msg

    # Track if CUDA fell back to CPU at any level (install or verification)
    _driver_too_old = "[DRIVER_TOO_OLD]" in msg
    _cuda_fell_back = "[CUDA_FALLBACK]" in msg

    # Step 4: Verify installation (95-100%)
    def verify_progress(percent: int, msg: str):
        """Map verification progress (0-100%) to overall progress (95-99%)."""
        if progress_callback:
            # Map 0-100 to 95-99 (leave 100% for final success message)
            mapped = 95 + int(percent * 0.04)
            progress_callback(min(mapped, 99), msg)

    is_valid, verify_msg = verify_venv(progress_callback=verify_progress)

    # CUDA → CPU fallback at verification level:
    # If verification fails and CUDA was enabled, the CUDA torch build may be
    # incompatible with the user's GPU/driver. Silently fall back to CPU torch.
    if not is_valid and cuda_enabled:
        _log(
            "Verification failed with CUDA torch, "
            "falling back to CPU: {}".format(verify_msg),
            Qgis.Warning
        )
        _reinstall_cpu_torch(VENV_DIR, progress_callback=progress_callback)
        is_valid, verify_msg = verify_venv(progress_callback=verify_progress)
        _cuda_fell_back = True

    # Step 4b: CUDA smoke test — verify torch.cuda actually works in the venv
    # Always run when CUDA was requested, even after fallback: a later package
    # install (e.g. torchvision CUDA) may have pulled in CUDA torch as a
    # dependency, resolving the earlier failure (issue #132).
    if is_valid and cuda_enabled:
        if progress_callback:
            progress_callback(99, "Verifying CUDA functionality...")
        cuda_works = _verify_cuda_in_venv(VENV_DIR)
        if cuda_works and _cuda_fell_back:
            # CUDA actually works despite earlier fallback — clear the flag
            _log(
                "CUDA smoke test passed after earlier fallback. "
                "GPU acceleration is available.",
                Qgis.Info)
            _cuda_fell_back = False
        elif not cuda_works and not _cuda_fell_back:
            _log(
                "CUDA smoke test failed, falling back to CPU torch",
                Qgis.Warning
            )
            _reinstall_cpu_torch(VENV_DIR, progress_callback=progress_callback)
            is_valid, verify_msg = verify_venv(progress_callback=verify_progress)
            _cuda_fell_back = True

    if not is_valid:
        return False, f"Verification failed: {verify_msg}"

    # Persist deps hash so future upgrades can detect spec changes
    _write_deps_hash()

    # Persist whether CUDA or CPU torch was installed
    if cuda_enabled and not _cuda_fell_back and not _driver_too_old:
        _write_cuda_flag("cuda")
    elif cuda_enabled and _cuda_fell_back:
        _write_cuda_flag("cuda_fallback")
    else:
        _write_cuda_flag("cpu")

    if progress_callback:
        progress_callback(100, "✓ All dependencies installed")

    if _driver_too_old:
        return True, "Virtual environment ready [DRIVER_TOO_OLD]"
    if _cuda_fell_back:
        return True, "Virtual environment ready [CUDA_FALLBACK]"
    return True, "Virtual environment ready"


def _quick_check_packages(venv_dir: str = None) -> Tuple[bool, str]:
    """
    Fast filesystem-based check that packages exist in the venv site-packages.

    Does NOT spawn subprocesses — safe to call from the main thread.
    Checks for known package directories/files in site-packages.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    site_packages = get_venv_site_packages(venv_dir)
    if not os.path.exists(site_packages):
        _log("Quick check: site-packages not found: {}".format(site_packages),
             Qgis.Warning)
        return False, "site-packages directory not found"

    # Map package names to their expected directory names in site-packages
    sam_marker = ("sam2", "sam2") if USE_SAM2 else ("segment-anything", "segment_anything")
    package_markers = {
        "numpy": "numpy",
        "torch": "torch",
        "torchvision": "torchvision",
        sam_marker[0]: sam_marker[1],
        "pandas": "pandas",
        "rasterio": "rasterio",
    }

    for package_name, dir_name in package_markers.items():
        pkg_dir = os.path.join(site_packages, dir_name)
        if not os.path.exists(pkg_dir):
            _log("Quick check: {} not found at {}".format(
                package_name, pkg_dir), Qgis.Warning)
            return False, "Package {} not found".format(package_name)

    _log("Quick check: all packages found in {}".format(site_packages),
         Qgis.Info)
    return True, "All packages found"


def get_venv_status() -> Tuple[bool, str]:
    """Get the status of the complete installation (Python standalone + venv)."""
    from .python_manager import standalone_python_exists, get_python_full_version

    # Check for old libs/ installation
    if os.path.exists(LIBS_DIR):
        _log("get_venv_status: old libs/ detected at {}".format(LIBS_DIR),
             Qgis.Warning)
        return False, "Old installation detected. Migration required."

    # Check Python standalone
    if not standalone_python_exists():
        _log("get_venv_status: standalone Python not found", Qgis.Info)
        return False, "Dependencies not installed"

    # Check venv
    if not venv_exists():
        _log("get_venv_status: venv not found at {}".format(VENV_DIR),
             Qgis.Info)
        return False, "Virtual environment not configured"

    # Quick filesystem check (no subprocess, safe for main thread)
    is_present, msg = _quick_check_packages()
    if is_present:
        # Check if dependency specs have changed since last install
        stored_hash = _read_deps_hash()
        current_hash = _compute_deps_hash()
        if stored_hash is not None and stored_hash != current_hash:
            _log(
                "get_venv_status: deps hash mismatch "
                "(stored={}, current={})".format(stored_hash, current_hash),
                Qgis.Warning
            )
            return False, "Dependencies need updating"
        if stored_hash is None:
            # First run after upgrade from a version without hash tracking.
            # Packages already passed quick check — write hash and proceed.
            _log(
                "get_venv_status: no deps hash file, "
                "writing current hash (packages already present)",
                Qgis.Info
            )
            _write_deps_hash()
        python_version = get_python_full_version()
        _log("get_venv_status: ready (quick check passed)", Qgis.Success)
        return True, "Ready (Python {})".format(python_version)
    else:
        _log("get_venv_status: quick check failed: {}".format(msg),
             Qgis.Warning)
        return False, "Virtual environment incomplete: {}".format(msg)


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

    # Only clean directories inside the plugin dir (legacy locations).
    # The external cache (~/.qgis_ai_segmentation/) should survive uninstall.
    dirs_to_clean = []
    old_venv = os.path.join(SRC_DIR, f'venv_{PYTHON_VERSION}')
    old_standalone = os.path.join(SRC_DIR, "python_standalone")
    old_checkpoints = os.path.join(SRC_DIR, "checkpoints")
    for d in [old_venv, old_standalone, old_checkpoints]:
        if os.path.exists(d):
            dirs_to_clean.append(d)

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
