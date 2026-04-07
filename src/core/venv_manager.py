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

from .model_config import SAM_PACKAGE, TORCH_MIN, TORCHVISION_MIN, USE_SAM2, IS_ROSETTA
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


def _numpy_version_spec() -> str:
    """Return the numpy version constraint based on the Python version.

    Python 3.13+ has no numpy <2.0 wheel, so we allow numpy 2.x there.
    Modern torch (>=2.1) supports numpy 2.x.
    """
    if sys.version_info >= (3, 13):
        return ">=2.0.0,<3.0.0"
    return ">=1.26.0,<2.0.0"


REQUIRED_PACKAGES = [
    ("setuptools", ">=70.0"),
    ("numpy", _numpy_version_spec()),
    ("torch", TORCH_MIN),
    ("torchvision", TORCHVISION_MIN),
    SAM_PACKAGE,
    ("pandas", ">=1.3.0"),
    ("rasterio", ">=1.3.0"),
]

DEPS_HASH_FILE = os.path.join(VENV_DIR, "deps_hash.txt")

# Bump this when install logic changes significantly (e.g., --no-cache-dir,
# new retry strategies) to force a dependency re-install on plugin update.
_INSTALL_LOGIC_VERSION = "3"


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
    """Write the current deps hash to the venv directory (atomic via tmp + replace)."""
    try:
        hash_dir = os.path.dirname(DEPS_HASH_FILE)
        os.makedirs(hash_dir, exist_ok=True)
        tmp_path = DEPS_HASH_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(_compute_deps_hash())
        os.replace(tmp_path, DEPS_HASH_FILE)
    except (OSError, IOError) as e:
        _log("Failed to write deps hash: {}".format(e), Qgis.MessageLevel.Warning)


def _log(message: str, level=Qgis.MessageLevel.Info):
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
        "  Architecture: {}{}".format(
            platform.machine(),
            " (Rosetta on Apple Silicon)" if IS_ROSETTA else ""
        ),
        f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"  QGIS: {qgis_version}",
        f"  Install dir: {CACHE_DIR}",
    ]
    if custom_cache:
        info_lines.append("  (via AI_SEGMENTATION_CACHE_DIR)")
    info_lines.append("=" * 50)
    for line in info_lines:
        _log(line, Qgis.MessageLevel.Info)


def _check_rosetta_warning() -> Optional[str]:
    """
    On macOS ARM, detect if running under Rosetta (x86_64 emulation).
    Returns info message if Rosetta detected, None otherwise.
    """
    if not IS_ROSETTA:
        return None

    return (
        "Rosetta detected: QGIS is running as x86_64 on Apple Silicon. "
        "Installing native ARM64 Python 3.10+ for SAM2 support."
    )


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
                                 Qgis.MessageLevel.Info)
                            removed.append(old_path)
                        except Exception as e:
                            _log("Failed to remove old venv {}: {}".format(
                                old_path, e), Qgis.MessageLevel.Warning)
        except Exception as e:
            _log("Error scanning for old venvs in {}: {}".format(
                scan_dir, e), Qgis.MessageLevel.Warning)

    return removed


def _check_gdal_available() -> Tuple[bool, str]:
    """
    Check if GDAL system library is available (Linux and macOS).
    Returns (is_available, help_message).
    """
    if sys.platform not in ("linux", "darwin"):
        return True, ""

    try:
        result = subprocess.run(
            ["gdal-config", "--version"],
            capture_output=True, text=True, encoding="utf-8", timeout=5
        )
        if result.returncode == 0:
            return True, f"GDAL {result.stdout.strip()} found"
        return False, ""
    except FileNotFoundError:
        if sys.platform == "darwin":
            return False, (
                "GDAL library not found. Rasterio requires GDAL to be installed.\n"
                "Please install GDAL:\n"
                "  brew install gdal"
            )
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
    "ssl error",
    "ssl:",
    "sslerror",
    "sslcertverificationerror",
    "certificate verify failed",
    "CERTIFICATE_VERIFY_FAILED",
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
    """Get pip flags to bypass SSL verification for corporate proxies.

    Note: --trusted-host may be deprecated in future pip versions (>= 21.0),
    but is still needed for older pip on QGIS 3.22-3.28 bundled Python.
    """
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
    "network timeout",
    "failed to download",
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
    msg = (
        "A required DLL failed to initialize.\n\n"
        "Try these steps in order:\n"
        "  1. Install the latest VC++ Redistributable (x64):\n"
        "     https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
        "  2. Restart your computer after installing\n"
        "  3. If the error persists after reboot, click 'Reinstall Dependencies'\n"
        "     to force a clean reinstall of PyTorch\n"
        "  4. Check that no other Python (Anaconda, Miniconda, standalone Python)\n"
        "     puts conflicting torch DLLs on your system PATH.\n"
        "     Open a terminal and run: where python\n"
        "     If you see multiple results, remove the extra ones from PATH"
    )
    return msg


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


def _is_rename_or_record_error(output: str) -> bool:
    """Detect dist-info rename/RECORD errors during torch upgrade on Windows."""
    lower = output.lower()
    if ("rename" in lower and "dist-info" in lower):
        return True
    if ("record" in lower and "dist-info" in lower):
        return True
    # uv may truncate "dist-info" from output
    if ("failed to install" in lower and "failed to rename" in lower):
        return True
    return False


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
                    dll_dir, exc), Qgis.MessageLevel.Warning)
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
            _log("Set PROJ_DATA to venv: {}".format(candidate), Qgis.MessageLevel.Info)
            break

    gdal_candidates = [
        os.path.join(site_packages, "rasterio", "gdal_data"),
    ]
    for candidate in gdal_candidates:
        if os.path.isdir(candidate):
            os.environ["GDAL_DATA"] = candidate
            _log("Set GDAL_DATA to venv: {}".format(candidate), Qgis.MessageLevel.Info)
            break


def ensure_venv_packages_available():
    if not venv_exists():
        _log("Venv does not exist, cannot load packages", Qgis.MessageLevel.Warning)
        return False

    site_packages = get_venv_site_packages()
    if not os.path.exists(site_packages):
        # Log detailed info for debugging
        venv_dir = get_venv_dir()
        lib_dir = os.path.join(venv_dir, "lib")
        if os.path.exists(lib_dir):
            contents = os.listdir(lib_dir)
            _log(f"Venv lib/ contents: {contents}", Qgis.MessageLevel.Warning)
        _log(f"Venv site-packages not found: {site_packages}", Qgis.MessageLevel.Warning)
        return False

    if site_packages not in sys.path:
        sys.path.append(site_packages)
        _log(f"Added venv site-packages to sys.path: {site_packages}", Qgis.MessageLevel.Info)

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
                    Qgis.MessageLevel.Info
                )
        except Exception:
            _log("Failed to reload typing_extensions, torch may fail", Qgis.MessageLevel.Warning)

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
        Qgis.MessageLevel.Info)

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
                Qgis.MessageLevel.Warning)
        else:
            _log(
                "Reloaded numpy {} -> {} from venv".format(
                    old_version, new_ver),
                Qgis.MessageLevel.Info)

    except Exception as e:
        _log(
            "Failed to reload numpy: {}. "
            "Plugin may not work on this QGIS version.".format(e),
            Qgis.MessageLevel.Warning)

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
        _log("QGIS bundled Python not found at sys.prefix", Qgis.MessageLevel.Warning)
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
            capture_output=True, text=True, encoding="utf-8", timeout=15,
            env=env, startupinfo=startupinfo,
        )
        if result.returncode == 0:
            _log(f"QGIS Python verified: {result.stdout.strip()}", Qgis.MessageLevel.Info)
            return python_path
        else:
            _log(f"QGIS Python failed verification: {result.stderr}", Qgis.MessageLevel.Warning)
            return None
    except Exception as e:
        _log(f"QGIS Python verification error: {e}", Qgis.MessageLevel.Warning)
        return None


def _get_system_python() -> str:
    """
    Get the path to the Python executable for creating venvs.

    Uses the standalone Python downloaded by python_manager.
    On Windows, falls back to QGIS's bundled Python if standalone is unavailable
    (e.g. when anti-malware blocks the standalone download).
    """
    from .python_manager import (
        standalone_python_exists, get_standalone_python_path,
        verify_standalone_python, remove_standalone_python
    )

    if standalone_python_exists():
        ok, msg = verify_standalone_python()
        if ok:
            python_path = get_standalone_python_path()
            _log(f"Using standalone Python: {python_path}", Qgis.MessageLevel.Info)
            return python_path
        else:
            _log(
                "Standalone Python broken ({}), removing...".format(msg),
                Qgis.MessageLevel.Warning
            )
            remove_standalone_python()

    # On NixOS, use system Python (standalone binaries can't run)
    from .python_manager import is_nixos
    if is_nixos():
        python3 = shutil.which("python3")
        if python3:
            _log("NixOS: using system Python: {}".format(python3), Qgis.MessageLevel.Info)
            return python3

    # On Windows, try QGIS's bundled Python as fallback
    if sys.platform == "win32":
        qgis_python = _get_qgis_python()
        if qgis_python:
            _log(
                "Standalone Python unavailable, using QGIS Python as fallback",
                Qgis.MessageLevel.Warning
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
            _log(f"Cleaned up partial venv: {venv_dir}", Qgis.MessageLevel.Info)
        except Exception:
            _log(f"Could not clean up partial venv: {venv_dir}", Qgis.MessageLevel.Warning)


def create_venv(
    venv_dir: str = None,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    _log(f"Creating virtual environment at: {venv_dir}", Qgis.MessageLevel.Info)

    if progress_callback:
        progress_callback(10, "Creating virtual environment...")

    system_python = _get_system_python()
    _log(f"Using Python: {system_python}", Qgis.MessageLevel.Info)

    # Use clean env to prevent QGIS PYTHONPATH/PYTHONHOME from leaking
    # into the standalone Python subprocess (issue #131)
    env = _get_clean_env_for_venv()

    global _uv_available, _uv_path

    # Try uv venv creation first (faster, no ensurepip needed)
    if _uv_available and _uv_path:
        _log("Creating venv with uv...", Qgis.MessageLevel.Info)
        # Resolve 8.3 short paths (e.g. PROGRA~1) - uv can't inspect them
        uv_python = _win_long_path(system_python)
        uv_cmd = [_uv_path, "venv", "--python", uv_python, venv_dir]
        try:
            subprocess_kwargs = {}
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                subprocess_kwargs["startupinfo"] = startupinfo

            result = subprocess.run(
                uv_cmd,
                capture_output=True, text=True, encoding="utf-8", timeout=120,
                env=env, **subprocess_kwargs,
            )
            if result.returncode == 0:
                _log("Virtual environment created with uv", Qgis.MessageLevel.Success)
                if progress_callback:
                    progress_callback(20, "Virtual environment created (uv)")
                return True, "Virtual environment created"
            else:
                error_msg = result.stderr or result.stdout or ""
                _log("uv venv creation failed: {}".format(
                    error_msg[:200]), Qgis.MessageLevel.Warning)
                _cleanup_partial_venv(venv_dir)
                # Fall through to standard venv creation
                remove_uv()
                _uv_available = False
                _uv_path = None
                _log("Falling back to python -m venv", Qgis.MessageLevel.Warning)
        except Exception as e:
            _log("uv venv exception: {}, falling back to python -m venv".format(
                e), Qgis.MessageLevel.Warning)
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
            encoding="utf-8",
            timeout=300,
            env=env,
            **subprocess_kwargs,
        )

        if result.returncode == 0:
            _log("Virtual environment created successfully", Qgis.MessageLevel.Success)

            # Ensure pip is available (QGIS Python fallback may not include pip)
            pip_path = get_venv_pip_path(venv_dir)
            if not os.path.exists(pip_path):
                _log("pip not found in venv, bootstrapping with ensurepip...", Qgis.MessageLevel.Info)
                python_in_venv = get_venv_python_path(venv_dir)
                ensurepip_cmd = [python_in_venv, "-m", "ensurepip", "--upgrade"]
                ensurepip_ok = False
                try:
                    ensurepip_result = subprocess.run(
                        ensurepip_cmd,
                        capture_output=True, text=True, encoding="utf-8", timeout=120,
                        env=env,
                        **({"startupinfo": startupinfo} if sys.platform == "win32" else {}),
                    )
                    if ensurepip_result.returncode == 0:
                        _log("pip bootstrapped via ensurepip", Qgis.MessageLevel.Success)
                        ensurepip_ok = True
                    else:
                        err = ensurepip_result.stderr or ensurepip_result.stdout or ""
                        _log("ensurepip failed: {}".format(err[:200]),
                             Qgis.MessageLevel.Warning)
                except Exception as e:
                    _log("ensurepip exception: {}".format(e),
                         Qgis.MessageLevel.Warning)

                if not ensurepip_ok:
                    # Anaconda and some managed Pythons strip ensurepip.
                    # If uv is available it can manage packages without pip.
                    if _uv_available and _uv_path:
                        _log(
                            "ensurepip unavailable but uv is present, "
                            "continuing without pip",
                            Qgis.MessageLevel.Warning)
                    else:
                        _cleanup_partial_venv(venv_dir)
                        return False, (
                            "Failed to bootstrap pip (ensurepip unavailable). "
                            "This often happens with Anaconda Python."
                        )

            if progress_callback:
                progress_callback(20, "Virtual environment created")
            return True, "Virtual environment created"
        else:
            error_msg = result.stderr or result.stdout or f"Return code {result.returncode}"
            _log(f"Failed to create venv: {error_msg}", Qgis.MessageLevel.Critical)
            _cleanup_partial_venv(venv_dir)
            return False, f"Failed to create venv: {error_msg[:200]}"

    except subprocess.TimeoutExpired:
        _log("Venv creation timed out, retrying with --without-pip...", Qgis.MessageLevel.Warning)
        _cleanup_partial_venv(venv_dir)
        # Retry with --without-pip (faster, avoids pip setup that AV scans)
        try:
            nopip_cmd = [system_python, "-m", "venv", "--without-pip", venv_dir]
            result2 = subprocess.run(
                nopip_cmd,
                capture_output=True, text=True, encoding="utf-8", timeout=300,
                env=env, **subprocess_kwargs,
            )
            if result2.returncode == 0:
                _log("Venv created (--without-pip), bootstrapping pip...", Qgis.MessageLevel.Info)
                python_in_venv = get_venv_python_path(venv_dir)
                ensurepip_cmd = [python_in_venv, "-m", "ensurepip", "--upgrade"]
                ep_result = subprocess.run(
                    ensurepip_cmd,
                    capture_output=True, text=True, encoding="utf-8", timeout=120,
                    env=env, **subprocess_kwargs,
                )
                if ep_result.returncode == 0:
                    _log("pip bootstrapped via ensurepip", Qgis.MessageLevel.Success)
                    if progress_callback:
                        progress_callback(20, "Virtual environment created")
                    return True, "Virtual environment created"
                else:
                    err = ep_result.stderr or ep_result.stdout or ""
                    _log("ensurepip failed: {}".format(err[:200]), Qgis.MessageLevel.Warning)
                    if _uv_available and _uv_path:
                        _log(
                            "ensurepip unavailable but uv present, "
                            "continuing without pip",
                            Qgis.MessageLevel.Warning)
                        if progress_callback:
                            progress_callback(20, "Virtual environment created")
                        return True, "Virtual environment created"
                    _cleanup_partial_venv(venv_dir)
                    return False, "Failed to bootstrap pip: {}".format(err[:200])
            else:
                err = result2.stderr or result2.stdout or ""
                _log("Retry --without-pip failed: {}".format(err[:200]), Qgis.MessageLevel.Critical)
                _cleanup_partial_venv(venv_dir)
                return False, "Virtual environment creation timed out"
        except Exception as e2:
            _log("Retry --without-pip exception: {}".format(e2), Qgis.MessageLevel.Critical)
            _cleanup_partial_venv(venv_dir)
            return False, "Virtual environment creation timed out"
    except FileNotFoundError:
        _log(f"Python executable not found: {system_python}", Qgis.MessageLevel.Critical)
        return False, f"Python not found: {system_python}"
    except Exception as e:
        _log(f"Exception during venv creation: {str(e)}", Qgis.MessageLevel.Critical)
        _cleanup_partial_venv(venv_dir)
        return False, f"Error: {str(e)[:200]}"


def _win_short_path(path: str) -> str:
    """Convert a Windows path to 8.3 short form if it contains spaces.

    Returns the original path on non-Windows or if conversion fails.
    """
    if sys.platform != "win32" or " " not in path:
        return path
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(512)
        ret = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
        if ret and ret < 512:
            return buf.value
    except Exception:
        pass
    return path


def _win_long_path(path: str) -> str:
    """Convert a Windows 8.3 short path to its long form.

    uv cannot inspect Python behind short path aliases like
    C:\\PROGRA~1\\QGIS34~1.8\\..., so we resolve them first.
    Returns the original path on non-Windows or if conversion fails.
    """
    if sys.platform != "win32" or "~" not in path:
        return path
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(512)
        ret = ctypes.windll.kernel32.GetLongPathNameW(path, buf, 512)
        if ret and ret < 512:
            return buf.value
    except Exception:
        pass
    return path


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
            elif arg == "--prefer-binary":
                continue
            elif arg in ("--retries", "--timeout"):
                skip_next = True  # skip flag and its value
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
            if arg == "--constraint" and i + 1 < len(pip_args):
                cmd.append(arg)
                cmd.append(_win_short_path(pip_args[i + 1]))
                skip_next = True
                continue
            cmd.append(arg)
        cmd.extend(["--python", _win_short_path(python_path)])
        return cmd
    return [python_path, "-m", "pip"] + pip_args


def _build_uninstall_cmd(python_path: str, packages: list) -> list:
    """Build an uninstall command using uv (if available) or pip."""
    if _uv_available and _uv_path:
        return [_uv_path, "pip", "uninstall", "--python", _win_short_path(python_path)] + packages
    return [python_path, "-m", "pip", "uninstall", "-y"] + packages


def _repin_numpy(venv_dir: str):
    """
    Check numpy version in the venv and force-downgrade if >= 2.0.

    This is a safety net: torch may pull numpy 2.x as a transitive
    dependency, which breaks torchvision and other packages.
    On Python 3.13+ numpy 2.x is expected, so skip the check.
    """
    if sys.version_info >= (3, 13):
        _log("Python >= 3.13: numpy 2.x is expected, skipping repin",
             Qgis.MessageLevel.Info)
        return

    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    try:
        result = subprocess.run(
            [python_path, "-c",
             "import numpy; print(numpy.__version__)"],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
            env=env, **subprocess_kwargs,
        )
        if result.returncode != 0:
            return  # numpy not installed or broken, nothing to fix here

        version_str = result.stdout.strip()
        major = int(version_str.split(".")[0])
        if major >= 2:
            _log(
                "numpy {} detected (>=2.0), forcing downgrade to <2.0.0...".format(version_str),
                Qgis.MessageLevel.Warning
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
                capture_output=True, text=True, encoding="utf-8", timeout=120,
                env=env, **subprocess_kwargs,
            )
            if downgrade_result.returncode == 0:
                _log("numpy downgraded successfully to <2.0.0", Qgis.MessageLevel.Success)
            else:
                err = downgrade_result.stderr or downgrade_result.stdout or ""
                _log("numpy downgrade failed: {}".format(err[:200]), Qgis.MessageLevel.Warning)
    except Exception as e:
        _log("numpy version check failed: {}".format(e), Qgis.MessageLevel.Warning)


def _get_verification_timeout(package_name: str) -> int:
    """
    Get verification timeout in seconds for a given package.

    torch needs extra time because the first import loads native DLLs on Windows,
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
            encoding="utf-8",
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
) -> Tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not venv_exists(venv_dir):
        return False, "Virtual environment does not exist"

    pip_path = get_venv_pip_path(venv_dir)
    _log(f"Installing dependencies using: {pip_path}", Qgis.MessageLevel.Info)

    # Upgrade pip before installing packages. The standalone Python bundles
    # pip 24.3.1 which can crash with internal exceptions on large packages
    # like torch (see issue #145). Skip when using uv.
    python_path_pre = get_venv_python_path(venv_dir)
    if _uv_available:
        _log("Using uv for installation, skipping pip upgrade", Qgis.MessageLevel.Info)
        if progress_callback:
            progress_callback(20, "Using uv package installer...")
    else:
        if progress_callback:
            progress_callback(20, "Upgrading pip...")
        try:
            _log("Upgrading pip to latest version...", Qgis.MessageLevel.Info)
            upgrade_cmd = [
                python_path_pre, "-m", "pip", "install",
                "--upgrade", "pip",
                "--disable-pip-version-check",
                "--no-warn-script-location",
            ]
            upgrade_cmd.extend(_get_pip_ssl_flags())
            upgrade_result = subprocess.run(
                upgrade_cmd,
                capture_output=True, text=True, encoding="utf-8", timeout=120,
                env=_get_clean_env_for_venv(),
                **_get_subprocess_kwargs(),
            )
            if upgrade_result.returncode == 0:
                _log("pip upgraded successfully", Qgis.MessageLevel.Success)
            else:
                _log("pip upgrade failed (non-critical): {}".format(
                    (upgrade_result.stderr or upgrade_result.stdout or "")[:200]),
                    Qgis.MessageLevel.Warning)
        except Exception as e:
            _log("pip upgrade failed (non-critical): {}".format(str(e)[:200]),
                 Qgis.MessageLevel.Warning)

    total_packages = len(REQUIRED_PACKAGES)
    base_progress = 20
    progress_range = 80  # from 20% to 100%

    # Weighted progress allocation proportional to download size.
    # Name-based map so weights stay correct if REQUIRED_PACKAGES order changes.
    _wmap = {
        "numpy": 5, "torch": 30, "torchvision": 15,
        "pandas": 10, "rasterio": 10,
    }
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
    # pip may pull numpy>=2.0 as a dependency, ignoring our version spec.
    # The constraints file forces pip to honour the upper bound on every
    # install command.
    os.makedirs(CACHE_DIR, exist_ok=True)
    constraints_fd, constraints_path = tempfile.mkstemp(
        suffix=".txt", prefix="pip_constraints_", dir=CACHE_DIR
    )
    try:
        with os.fdopen(constraints_fd, "w", encoding="utf-8") as f:
            if sys.version_info >= (3, 13):
                f.write("numpy<3.0.0\n")
            else:
                f.write("numpy<2.0.0\n")
        _log(f"Created pip constraints file: {constraints_path}", Qgis.MessageLevel.Info)
    except Exception as e:
        _log(f"Failed to write constraints file: {e}", Qgis.MessageLevel.Warning)
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

        for i, (package_name, version_spec) in enumerate(REQUIRED_PACKAGES):
            if cancel_check and cancel_check():
                _log("Installation cancelled by user", Qgis.MessageLevel.Warning)
                return False, "Installation cancelled"

            package_spec = f"{package_name}{version_spec}"
            pkg_start = _pkg_progress_start(i)
            pkg_end = _pkg_progress_end(i)

            if progress_callback:
                if package_name == "torch":
                    progress_callback(
                        pkg_start,
                        "Installing {} (~600MB)... ({}/{})".format(
                            package_name, i + 1, total_packages))
                else:
                    progress_callback(
                        pkg_start,
                        "Installing {}... ({}/{})".format(
                            package_name, i + 1, total_packages))

            _log(f"[{i + 1}/{total_packages}] Installing {package_spec}...", Qgis.MessageLevel.Info)

            pip_args = [
                "install",
                "--upgrade",
                "--no-warn-script-location",
                "--disable-pip-version-check",
                "--prefer-binary",  # Prefer pre-built wheels to avoid C extension build issues
                "--retries", "10",  # More retries for unstable networks (default 5)
                "--timeout", "30",  # Longer timeout per connection attempt (default 15)
            ]
            # sam2/segment-anything list torch as a build dependency.
            # Without --no-build-isolation pip creates a separate env and
            # re-downloads torch (~2.5 GB), which often fails. Since torch
            # is already installed in the venv at this point, skip isolation.
            if package_name in ("sam2", "segment-anything"):
                pip_args.append("--no-build-isolation")
            # Force binary-only for rasterio to prevent source builds that
            # fail when gdal-config is missing (macOS Rosetta, etc.) (#186)
            if package_name == "rasterio":
                pip_args.extend(["--only-binary", "rasterio"])
            # Add SSL bypass flags upfront for corporate proxies (not just as retry)
            pip_args.extend(_get_pip_ssl_flags())
            if constraints_path:
                pip_args.extend(["--constraint", constraints_path])
            pip_args.extend(_get_pip_proxy_args())
            pip_args.append(package_spec)

            # Use clean env to avoid QGIS PYTHONPATH/PYTHONHOME interference
            env = _get_clean_env_for_venv()

            subprocess_kwargs = _get_subprocess_kwargs()

            # Large packages need more time than standard packages
            if package_name == "torch":
                pkg_timeout = 5400  # 90 min for CPU torch on slow connections
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
                )

                # If cancelled
                if result.returncode == -1 and "cancelled" in (result.stderr or "").lower():
                    _log("Installation cancelled by user", Qgis.MessageLevel.Warning)
                    return False, "Installation cancelled"

                # If Windows process crash, retry with pip.exe as fallback
                # (skip when using uv - it doesn't have this issue)
                if not _uv_available and _is_windows_process_crash(result.returncode):
                    _log(
                        "Process crash detected (code {}), "
                        "retrying with pip.exe...".format(result.returncode),
                        Qgis.MessageLevel.Warning
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
                    )

                # If "unable to create process" (broken pip shim), retry with pip.exe
                # (skip when using uv - it doesn't have this issue)
                if not _uv_available and result.returncode != 0:
                    error_output = result.stderr or result.stdout or ""
                    if _is_unable_to_create_process(error_output):
                        _log(
                            "Unable to create process detected, "
                            "retrying with pip.exe...",
                            Qgis.MessageLevel.Warning
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
                        )

                # If failed, check for SSL errors and retry with --trusted-host
                if result.returncode != 0 and not _is_windows_process_crash(result.returncode):
                    error_output = result.stderr or result.stdout or ""

                    if _is_ssl_error(error_output):
                        _log(
                            "SSL error detected, retrying...",
                            Qgis.MessageLevel.Warning
                        )
                        if progress_callback:
                            progress_callback(
                                pkg_start,
                                "SSL error, retrying {}... ({}/{})".format(
                                    package_name, i + 1, total_packages)
                            )

                        # Retry (SSL flags already in base_cmd)
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
                        )

                # If failed, check for hash mismatch (corrupted cache) and retry
                if result.returncode != 0 and not _is_windows_process_crash(result.returncode):
                    error_output = result.stderr or result.stdout or ""

                    if _is_hash_mismatch(error_output):
                        _log(
                            "Hash mismatch detected (corrupted cache), "
                            "retrying with --no-cache-dir...",
                            Qgis.MessageLevel.Warning
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
                                Qgis.MessageLevel.Warning
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
                            )
                            if result.returncode == 0:
                                break

                # If "no matching distribution" for torch, retry with --no-cache-dir
                if result.returncode != 0 and package_name in ("torch", "torchvision"):
                    error_output = result.stderr or result.stdout or ""
                    err_lower = error_output.lower()
                    no_dist = "no matching distribution" in err_lower
                    if no_dist:
                        _log(
                            "No matching distribution for {}, "
                            "retrying with --no-cache-dir...".format(package_name),
                            Qgis.MessageLevel.Warning
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
                        )

                # If rename/RECORD error (stale dist-info on Windows), clean and force-reinstall
                if result.returncode != 0 and package_name in ("torch", "torchvision"):
                    error_output = result.stderr or result.stdout or ""
                    if _is_rename_or_record_error(error_output):
                        _log(
                            "Stale dist-info detected for {}, cleaning and "
                            "retrying with --force-reinstall...".format(package_name),
                            Qgis.MessageLevel.Warning
                        )
                        try:
                            site_pkgs = get_venv_site_packages()
                            if os.path.isdir(site_pkgs):
                                import glob as _glob
                                import shutil as _shutil
                                pattern = os.path.join(
                                    site_pkgs,
                                    "{}-*.dist-info".format(package_name))
                                for dist_dir in _glob.glob(pattern):
                                    _shutil.rmtree(dist_dir, ignore_errors=True)
                                    _log("Removed stale {}".format(dist_dir),
                                         Qgis.MessageLevel.Warning)
                        except Exception as exc:
                            _log("Failed to clean dist-info: {}".format(exc),
                                 Qgis.MessageLevel.Warning)

                        if progress_callback:
                            progress_callback(
                                pkg_start,
                                "Retrying {}... ({}/{})".format(
                                    package_name, i + 1, total_packages))
                        reinstall_cmd = base_cmd + ["--force-reinstall"]
                        result = _run_pip_install(
                            cmd=reinstall_cmd,
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
                        )

                if result.returncode == 0:
                    _log(f"✓ Successfully installed {package_spec}", Qgis.MessageLevel.Success)
                    if progress_callback:
                        progress_callback(pkg_end, f"✓ {package_name} installed")
                else:
                    error_msg = result.stderr or result.stdout or f"Return code {result.returncode}"
                    _log(f"✗ Failed to install {package_spec}: {error_msg[:500]}", Qgis.MessageLevel.Critical)
                    install_failed = True
                    install_error_msg = error_msg
                    last_returncode = result.returncode

            except subprocess.TimeoutExpired:
                _log(f"Installation of {package_spec} timed out", Qgis.MessageLevel.Critical)
                install_failed = True
                install_error_msg = f"Installation of {package_name} timed out"
            except Exception as e:
                _log(f"Exception during installation of {package_spec}: {str(e)}", Qgis.MessageLevel.Critical)
                install_failed = True
                install_error_msg = f"Error installing {package_name}: {str(e)[:200]}"

            if install_failed:
                # Log detailed pip output for debugging
                _log("pip error output: {}".format(install_error_msg[:500]), Qgis.MessageLevel.Critical)

                # Check for Windows process crash
                if last_returncode is not None and _is_windows_process_crash(last_returncode):
                    _log(_get_crash_help(venv_dir), Qgis.MessageLevel.Warning)
                    return False, "Failed to install {}: process crashed (code {})".format(
                        package_name, last_returncode)

                # Check for DLL load failure (missing system libraries)
                is_dll_err = sys.platform == "win32" and _is_dll_init_error(install_error_msg)
                if is_dll_err and package_name in ("torch", "torchvision"):
                    _log(_get_vcpp_help(), Qgis.MessageLevel.Warning)
                    return False, "Failed to install {}: {}".format(
                        package_name, _get_vcpp_help())

                # Check for rename/record errors (antivirus blocking on Windows) - before SSL
                # because uv output may contain SSL_CERT_DIR warnings alongside rename errors
                if sys.platform == "win32" and _is_rename_or_record_error(install_error_msg):
                    help_msg = (
                        "Failed to install {}: file rename blocked.\n\n"
                        "This is typically caused by antivirus or security software "
                        "scanning files during installation.\n\n"
                        "Please try:\n"
                        "  1. Temporarily disable real-time antivirus scanning\n"
                        "  2. Add an exclusion for: {}\n"
                        "  3. Restart QGIS and reinstall dependencies"
                    ).format(package_name, CACHE_DIR)
                    _log(help_msg, Qgis.MessageLevel.Warning)
                    return False, "Failed to install {}: blocked by antivirus (rename failed)".format(
                        package_name)

                # Check for SSL errors
                if _is_ssl_error(install_error_msg):
                    _log(_get_ssl_error_help(install_error_msg), Qgis.MessageLevel.Warning)
                    return False, "Failed to install {}: SSL error".format(
                        package_name)

                # Check for proxy authentication errors (407)
                if _is_proxy_auth_error(install_error_msg):
                    _log(
                        "Proxy authentication failed (HTTP 407). "
                        "Configure proxy credentials in: "
                        "QGIS > Settings > Options > Network > Proxy "
                        "(User and Password fields).",
                        Qgis.MessageLevel.Warning
                    )
                    return False, "Failed to install {}: proxy authentication required (407)".format(
                        package_name)

                # Check for network/connection errors (after retries exhausted)
                if _is_network_error(install_error_msg):
                    _log(
                        "Network connection failed after multiple retries. "
                        "Check internet connection, VPN/proxy settings, "
                        "and firewall rules for pypi.org and files.pythonhosted.org.",
                        Qgis.MessageLevel.Warning
                    )
                    return False, "Failed to install {}: network error".format(
                        package_name)

                # Check for platform-unsupported wheels
                if "no matching platform tag" in install_error_msg.lower():
                    return False, (
                        "Failed to install {}: no wheels with a matching "
                        "platform tag for this OS/architecture."
                    ).format(package_name)

                # Check for antivirus blocking
                if _is_antivirus_error(install_error_msg):
                    _log(_get_pip_antivirus_help(venv_dir), Qgis.MessageLevel.Warning)
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

                # Check for GDAL issues on Linux/macOS when rasterio fails
                if package_name == "rasterio":
                    gdal_ok, gdal_help = _check_gdal_available()
                    if not gdal_ok and gdal_help:
                        _log(gdal_help, Qgis.MessageLevel.Warning)
                        return False, "Failed to install {}: GDAL library not found".format(
                            package_name)

                return False, "Failed to install {}: {}".format(
                    package_name, install_error_msg[:200])

        # Post-install numpy version safety net:
        # Check and force-downgrade if needed.
        _repin_numpy(venv_dir)

        if progress_callback:
            progress_callback(100, "✓ All dependencies installed")

        _log("=" * 50, Qgis.MessageLevel.Success)
        _log("All dependencies installed successfully!", Qgis.MessageLevel.Success)
        _log(f"Virtual environment: {venv_dir}", Qgis.MessageLevel.Success)
        _log("=" * 50, Qgis.MessageLevel.Success)

        return True, "All dependencies installed successfully"

    finally:
        # Always clean up the constraints temp file
        if constraints_path:
            for _attempt in range(3):
                try:
                    os.unlink(constraints_path)
                    break
                except PermissionError:
                    time.sleep(0.5)
                except Exception:
                    break


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
        _log("Could not read QGIS proxy settings: {}".format(e), Qgis.MessageLevel.Warning)
        return None


def _get_pip_proxy_args() -> List[str]:
    """Get pip --proxy argument if QGIS proxy is configured."""
    proxy_url = _get_qgis_proxy_settings()
    if proxy_url:
        _log("Using QGIS proxy for pip: {}".format(
            proxy_url.split("@")[-1] if "@" in proxy_url else proxy_url),
            Qgis.MessageLevel.Info
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

    # Remove SSL_CERT_DIR if it points to a non-existent directory.
    # Invalid paths cause uv to emit "SSL_CERT_DIR" warnings that the
    # error classifier would otherwise misread as real SSL errors (#184).
    ssl_cert_dir = env.get("SSL_CERT_DIR", "")
    if ssl_cert_dir and not os.path.isdir(ssl_cert_dir):
        env.pop("SSL_CERT_DIR", None)

    env["PYTHONIOENCODING"] = "utf-8"

    # Skip sam2 CUDA extension compilation (Python fallback works fine)
    env["SAM2_BUILD_CUDA"] = "0"

    # Increase uv download timeout (default 30s too short for large wheels)
    env["UV_HTTP_TIMEOUT"] = "300"

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
                encoding="utf-8",
                timeout=pkg_timeout,
                env=env,
                **subprocess_kwargs
            )

            if result.returncode != 0:
                error_detail = result.stderr[:300] if result.stderr else result.stdout[:300]
                _log(
                    "Package {} verification failed: {}".format(
                        package_name, error_detail),
                    Qgis.MessageLevel.Warning
                )

                # DLL init error (WinError 1114) - try force-reinstall first,
                # as a conflicting DLL from another Python install may be the cause
                if _is_dll_init_error(error_detail):
                    _log(
                        "DLL init error for {}, attempting "
                        "force-reinstall...".format(package_name),
                        Qgis.MessageLevel.Warning
                    )
                    pkg_spec = package_name
                    for name, spec in REQUIRED_PACKAGES:
                        if name == package_name:
                            pkg_spec = "{}{}".format(name, spec)
                            break
                    reinstall_cmd = _build_install_cmd(
                        python_path,
                        ["install", "--force-reinstall", "--no-deps",
                         "--prefer-binary", pkg_spec])
                    try:
                        subprocess.run(
                            reinstall_cmd,
                            capture_output=True, text=True,
                            encoding="utf-8", timeout=600,
                            env=env, **subprocess_kwargs
                        )
                        result2 = subprocess.run(
                            cmd, capture_output=True, text=True,
                            encoding="utf-8", timeout=pkg_timeout,
                            env=env, **subprocess_kwargs
                        )
                        if result2.returncode == 0:
                            _log(
                                "Package {} fixed after "
                                "force-reinstall".format(package_name),
                                Qgis.MessageLevel.Success)
                            continue
                    except Exception:
                        pass

                    # Nuclear option: delete torch dirs and reinstall fresh
                    _log(
                        "Force-reinstall did not fix DLL error for "
                        "{}. Nuking and reinstalling...".format(
                            package_name),
                        Qgis.MessageLevel.Warning
                    )
                    try:
                        site_pkgs = get_venv_site_packages(venv_dir)
                        for pkg_dir_name in (
                            "torch", "torchvision",
                        ):
                            for d in os.listdir(site_pkgs):
                                if d == pkg_dir_name or d.startswith(
                                    pkg_dir_name + "-"
                                ):
                                    target = os.path.join(site_pkgs, d)
                                    if os.path.isdir(target):
                                        shutil.rmtree(
                                            target, ignore_errors=True
                                        )
                        # Reinstall both packages
                        torch_spec = "torch{}".format(TORCH_MIN)
                        tv_spec = "torchvision{}".format(
                            TORCHVISION_MIN)
                        nuke_cmd = _build_install_cmd(
                            python_path,
                            ["install", "--prefer-binary",
                             torch_spec, tv_spec])
                        subprocess.run(
                            nuke_cmd,
                            capture_output=True, text=True,
                            encoding="utf-8", timeout=600,
                            env=env, **subprocess_kwargs
                        )
                        result3 = subprocess.run(
                            cmd, capture_output=True, text=True,
                            encoding="utf-8", timeout=pkg_timeout,
                            env=env, **subprocess_kwargs
                        )
                        if result3.returncode == 0:
                            _log(
                                "Package {} fixed after nuke "
                                "reinstall".format(package_name),
                                Qgis.MessageLevel.Success)
                            continue
                    except Exception:
                        pass

                    # 4th attempt (Windows only): try known-good pinned
                    # torch version that doesn't have DLL issues.
                    from .model_config import (
                        TORCH_WINDOWS_FALLBACK,
                        TORCHVISION_WINDOWS_FALLBACK,
                    )
                    if TORCH_WINDOWS_FALLBACK:
                        _log(
                            "Nuke reinstall did not fix DLL error. "
                            "Trying pinned torch{} fallback...".format(
                                TORCH_WINDOWS_FALLBACK),
                            Qgis.MessageLevel.Warning
                        )
                        try:
                            site_pkgs = get_venv_site_packages(venv_dir)
                            for pkg_dir_name in ("torch", "torchvision"):
                                for d in os.listdir(site_pkgs):
                                    if d == pkg_dir_name or d.startswith(
                                        pkg_dir_name + "-"
                                    ):
                                        target = os.path.join(site_pkgs, d)
                                        if os.path.isdir(target):
                                            shutil.rmtree(
                                                target, ignore_errors=True
                                            )
                            fallback_cmd = _build_install_cmd(
                                python_path,
                                ["install", "--prefer-binary",
                                 "torch{}".format(TORCH_WINDOWS_FALLBACK),
                                 "torchvision{}".format(
                                     TORCHVISION_WINDOWS_FALLBACK)])
                            subprocess.run(
                                fallback_cmd,
                                capture_output=True, text=True,
                                encoding="utf-8", timeout=600,
                                env=env, **subprocess_kwargs
                            )
                            result4 = subprocess.run(
                                cmd, capture_output=True, text=True,
                                encoding="utf-8", timeout=pkg_timeout,
                                env=env, **subprocess_kwargs
                            )
                            if result4.returncode == 0:
                                _log(
                                    "Package {} fixed with pinned "
                                    "torch{} fallback".format(
                                        package_name,
                                        TORCH_WINDOWS_FALLBACK),
                                    Qgis.MessageLevel.Success)
                                continue
                        except Exception:
                            pass

                    _log(_get_vcpp_help(), Qgis.MessageLevel.Warning)
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
                        Qgis.MessageLevel.Warning
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
                            encoding="utf-8",
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
                            encoding="utf-8",
                            timeout=pkg_timeout,
                            env=env,
                            **subprocess_kwargs
                        )
                        if result2.returncode == 0:
                            _log(
                                "Package {} fixed after force-reinstall".format(
                                    package_name),
                                Qgis.MessageLevel.Success
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
                Qgis.MessageLevel.Info
            )
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
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
                        Qgis.MessageLevel.Warning
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
                    Qgis.MessageLevel.Warning
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
                    Qgis.MessageLevel.Warning
                )
                return False, "Verification error: {}".format(package_name)

        except Exception as e:
            _log(
                "Failed to verify {}: {}".format(package_name, str(e)),
                Qgis.MessageLevel.Warning
            )
            return False, "Verification error: {}".format(package_name)

    if progress_callback:
        progress_callback(100, "Verification complete")

    _log("✓ Virtual environment verified successfully", Qgis.MessageLevel.Success)
    return True, "Virtual environment ready"


def cleanup_old_libs() -> bool:
    if not os.path.exists(LIBS_DIR):
        return False

    _log("Detected old 'libs/' installation. Cleaning up...", Qgis.MessageLevel.Info)

    try:
        shutil.rmtree(LIBS_DIR)
        _log("Old libs/ directory removed successfully", Qgis.MessageLevel.Success)
        return True
    except Exception as e:
        _log(f"Failed to remove libs/: {e}. Please delete manually.", Qgis.MessageLevel.Warning)
        return False


def create_venv_and_install(
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[bool, str]:
    """
    Complete installation: download Python standalone + download uv + create venv + install packages.

    Progress breakdown:
    - 0-10%:   Download Python standalone (~50MB)
    - 10-13%:  Download uv package installer (non-fatal if fails)
    - 13-18%:  Create virtual environment
    - 18-95%:  Install packages (~800MB)
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
        _log(hint, Qgis.MessageLevel.Critical)
        return False, hint

    # Check for Rosetta emulation on macOS (warning only, don't block)
    rosetta_warning = _check_rosetta_warning()
    if rosetta_warning:
        _log(rosetta_warning, Qgis.MessageLevel.Warning)

    # Clean up old venv directories from previous Python versions
    removed_venvs = cleanup_old_venv_directories()
    if removed_venvs:
        _log(f"Removed {len(removed_venvs)} old venv directories", Qgis.MessageLevel.Info)

    cleanup_old_libs()

    # If standalone Python exists but version doesn't match QGIS, delete and re-download
    if standalone_python_exists() and not standalone_python_is_current():
        _log(
            "Standalone Python version mismatch, re-downloading...",
            Qgis.MessageLevel.Warning)
        remove_standalone_python()
        # Also remove the venv since it was built with the wrong Python
        if venv_exists():
            try:
                shutil.rmtree(VENV_DIR)
                _log("Removed stale venv after Python version mismatch", Qgis.MessageLevel.Info)
            except Exception as e:
                _log("Failed to remove stale venv: {}".format(e), Qgis.MessageLevel.Warning)

    # Step 1: Download Python standalone if needed
    from .python_manager import is_nixos
    if not standalone_python_exists():
        if is_nixos():
            _log("NixOS detected, using system Python", Qgis.MessageLevel.Info)
            if progress_callback:
                progress_callback(10, "Using system Python (NixOS)...")
        else:
            python_version = get_python_full_version()
            _log(f"Downloading Python {python_version} standalone...", Qgis.MessageLevel.Info)

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
                            Qgis.MessageLevel.Warning
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
        _log("Python standalone already installed", Qgis.MessageLevel.Info)
        if progress_callback:
            progress_callback(10, "Python standalone ready")

    # Step 1b: Download uv package installer (non-fatal if fails)
    global _uv_available, _uv_path
    if uv_exists() and verify_uv():
        _uv_available = True
        _uv_path = get_uv_path()
        _log("uv already installed, using for package management", Qgis.MessageLevel.Info)
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
                _log("uv downloaded, using for package management", Qgis.MessageLevel.Info)
            else:
                _uv_available = False
                _uv_path = None
                _log("uv download failed (non-fatal), using pip: {}".format(
                    uv_msg), Qgis.MessageLevel.Warning)
        except Exception as e:
            _uv_available = False
            _uv_path = None
            _log("uv download failed (non-fatal): {}".format(e), Qgis.MessageLevel.Warning)
        if progress_callback:
            progress_callback(13, "uv: {}".format(
                "ready" if _uv_available else "unavailable, using pip"))

    if cancel_check and cancel_check():
        return False, "Installation cancelled"

    # Step 2: Create virtual environment if needed
    if venv_exists():
        _log("Virtual environment already exists", Qgis.MessageLevel.Info)
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

    # Persist deps hash so future upgrades can detect spec changes
    _write_deps_hash()

    if progress_callback:
        progress_callback(100, "✓ All dependencies installed")

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
             Qgis.MessageLevel.Warning)
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
                package_name, pkg_dir), Qgis.MessageLevel.Warning)
            return False, "Package {} not found".format(package_name)

    _log("Quick check: all packages found in {}".format(site_packages),
         Qgis.MessageLevel.Info)
    return True, "All packages found"


def get_venv_status() -> Tuple[bool, str]:
    """Get the status of the complete installation (Python standalone + venv)."""
    from .python_manager import standalone_python_exists, get_python_full_version

    # Check for old libs/ installation
    if os.path.exists(LIBS_DIR):
        _log("get_venv_status: old libs/ detected at {}".format(LIBS_DIR),
             Qgis.MessageLevel.Warning)
        return False, "Old installation detected. Migration required."

    # Check Python standalone (skip on NixOS where system Python is used)
    from .python_manager import is_nixos
    if not standalone_python_exists() and not is_nixos():
        _log("get_venv_status: standalone Python not found", Qgis.MessageLevel.Info)
        return False, "Dependencies not installed"

    # Check venv
    if not venv_exists():
        _log("get_venv_status: venv not found at {}".format(VENV_DIR),
             Qgis.MessageLevel.Info)
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
                Qgis.MessageLevel.Warning
            )
            return False, "Dependencies need updating"
        if stored_hash is None:
            # First run after upgrade from a version without hash tracking.
            # Packages passed quick check (filesystem), but on Windows the
            # DLLs may be broken. Verify torch actually imports before
            # writing the hash (prevents "phantom ready" state).
            python_path = get_venv_python_path()
            if python_path and sys.platform == "win32":
                try:
                    env = _get_clean_env_for_venv()
                    kwargs = _get_subprocess_kwargs()
                    probe = subprocess.run(
                        [python_path, "-c", "import torch"],
                        capture_output=True, text=True,
                        encoding="utf-8", timeout=60,
                        env=env, **kwargs,
                    )
                    if probe.returncode != 0:
                        _log(
                            "get_venv_status: torch import failed "
                            "(possible DLL error), marking as incomplete",
                            Qgis.MessageLevel.Warning
                        )
                        return False, "Package verification failed (torch import error)"
                except Exception as exc:
                    _log(
                        "get_venv_status: torch probe failed: {}".format(exc),
                        Qgis.MessageLevel.Warning
                    )
                    return False, "Package verification failed"
            _log(
                "get_venv_status: no deps hash file, "
                "writing current hash (packages already present)",
                Qgis.MessageLevel.Info
            )
            _write_deps_hash()
        python_version = get_python_full_version()
        _log("get_venv_status: ready (quick check passed)", Qgis.MessageLevel.Success)
        return True, "Ready (Python {})".format(python_version)
    else:
        _log("get_venv_status: quick check failed: {}".format(msg),
             Qgis.MessageLevel.Warning)
        return False, "Virtual environment incomplete: {}".format(msg)


def remove_venv(venv_dir: str = None) -> Tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not os.path.exists(venv_dir):
        return True, "Virtual environment does not exist"

    try:
        shutil.rmtree(venv_dir)
        _log(f"Removed virtual environment: {venv_dir}", Qgis.MessageLevel.Success)
        return True, "Virtual environment removed"
    except Exception as e:
        _log(f"Failed to remove venv: {e}", Qgis.MessageLevel.Warning)
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
                _log(f"Removed {symlinks_removed} symlinks from: {dir_path}", Qgis.MessageLevel.Info)
                cleaned = True

            # Step 2: Remove extended attributes
            try:
                result = subprocess.run(
                    ["xattr", "-r", "-c", dir_path],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=60
                )
                if result.returncode == 0:
                    _log(f"Cleaned extended attributes from: {dir_path}", Qgis.MessageLevel.Info)
                    cleaned = True
            except subprocess.TimeoutExpired:
                _log(f"xattr cleanup timed out for: {dir_path}", Qgis.MessageLevel.Warning)
            except FileNotFoundError:
                _log("xattr command not found", Qgis.MessageLevel.Warning)
            except Exception as e:
                _log(f"Failed to clean extended attributes: {e}", Qgis.MessageLevel.Warning)

    return cleaned
