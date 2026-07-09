from __future__ import annotations

import glob
import hashlib
import os
import platform
import re
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import threading
import time
from typing import Callable

from qgis.core import Qgis

from .install_lock import InstallLock
from .logging_utils import log as _log
from .model_config import IS_ROSETTA, SAM_PACKAGE, TORCH_MIN, TORCHVISION_MIN
from .pip_diagnostics import (
    get_crash_help as _get_crash_help,
)
from .pip_diagnostics import (
    get_file_locked_help as _get_file_locked_help,
)
from .pip_diagnostics import (
    get_pip_antivirus_help as _get_pip_antivirus_help,
)
from .pip_diagnostics import (
    get_pip_ssl_bypass_flags as _get_pip_ssl_bypass_flags,
)
from .pip_diagnostics import (
    get_ssl_error_help,
)
from .pip_diagnostics import (
    get_vcpp_help as _get_vcpp_help,
)
from .pip_diagnostics import (
    is_antivirus_error as _is_antivirus_error,
)
from .pip_diagnostics import (
    is_dll_init_error as _is_dll_init_error,
)
from .pip_diagnostics import (
    is_file_locked_error as _is_file_locked_error,
)
from .pip_diagnostics import (
    is_hash_mismatch as _is_hash_mismatch,
)
from .pip_diagnostics import (
    is_network_error as _is_network_error,
)
from .pip_diagnostics import (
    is_proxy_auth_error as _is_proxy_auth_error,
)
from .pip_diagnostics import (
    is_rename_or_record_error as _is_rename_or_record_error,
)
from .pip_diagnostics import (
    is_ssl_error as _is_ssl_error,
)
from .pip_diagnostics import (
    is_unable_to_create_process as _is_unable_to_create_process,
)
from .pip_diagnostics import (
    is_windows_process_crash as _is_windows_process_crash,
)
from .subprocess_utils import get_clean_env_for_venv as _get_base_clean_env  # nosec B404
from .uv_manager import (
    download_uv,
    get_uv_path,
    remove_uv,
    uv_exists,
    verify_uv,
)

# Module-level uv state (set during create_venv_and_install)
_uv_available = False
_uv_path: str | None = None


def _insecure_install_opt_in() -> bool:
    """True only if the user has explicitly allowed disabling TLS verification
    during dependency install.

    The safe corporate-CA path (--system-certs, OS trust store) is always
    tried first. Fully disabling TLS verification is an integrity risk (an
    active MITM could force the initial verified attempt to fail and then
    serve a poisoned wheel, which runs code on install), so it never happens
    silently: it requires an explicit opt-in via the QSettings key
    ``TerraLab/allow_insecure_install`` or the ``QGIS_AI_ALLOW_INSECURE_INSTALL``
    environment variable. Default is off.
    """
    if os.environ.get("QGIS_AI_ALLOW_INSECURE_INSTALL", "").strip().lower() in ("1", "true", "yes"):
        return True
    try:
        from qgis.PyQt.QtCore import QSettings

        val = QSettings().value("TerraLab/allow_insecure_install", False, type=bool)
        return bool(val)
    except Exception:  # noqa: BLE001 - settings unavailable: fail closed (secure default)
        return False


PLUGIN_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = PLUGIN_ROOT_DIR  # src/ directory
PYTHON_VERSION = f"py{sys.version_info.major}.{sys.version_info.minor}"
CACHE_DIR = os.environ.get("AI_SEGMENTATION_CACHE_DIR") or os.path.expanduser("~/.qgis_ai_segmentation")
VENV_DIR = os.path.join(CACHE_DIR, f"venv_{PYTHON_VERSION}")
LIBS_DIR = os.path.join(PLUGIN_ROOT_DIR, "libs")


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
    ("rasterio", ">=1.3.0"),
]

# Packages older venvs contain but the plugin never imports. pandas was
# installed up to 1.1.0 as a leftover (44 MB on disk, zero imports in src/,
# and a recurring Windows verification failure when the VC++ Redistributable
# is missing, see #66). It is no longer installed; existing venvs keep their
# copy harmlessly. If one of these fails verification we warn and continue.
NON_ESSENTIAL_PACKAGES = {"pandas"}

# Official CPU-only torch wheel index, used on Linux where PyPI's default
# torch wheels bundle CUDA (multi-GB with nvidia-* dependencies).
TORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"

DEPS_HASH_FILE = os.path.join(VENV_DIR, "deps_hash.txt")

# Marker written for the duration of create_venv_and_install. It only
# survives a hard crash (QGIS killed mid-install); finding it on a later
# run means the venv may be half-built and must be reinstalled.
INSTALL_MARKER_FILE = os.path.join(CACHE_DIR, "install_in_progress")

# Cross-process install lock. CACHE_DIR is shared across QGIS instances and
# profiles, so two windows can install concurrently; without a mutex the
# second one would rmtree the venv the first is still building. This lock
# complements the crash marker above: the marker recovers a same-process
# crash, the lock keeps two LIVE processes from touching the venv at once.
INSTALL_LOCK_FILE = os.path.join(CACHE_DIR, "install.lock")

# Bump this when install logic changes significantly (e.g., --no-cache-dir,
# new retry strategies) to force a dependency re-install on plugin update.
# v4: uv/rustls SSL errors ("invalid peer certificate") now trigger the
#     system-certs / TLS-bypass retry cascade; re-run installs that failed
#     silently behind corporate MITM proxies.
_INSTALL_LOGIC_VERSION = "4"


def _compute_deps_hash() -> str:
    """Compute MD5 hash of REQUIRED_PACKAGES + install logic version.

    Changing either REQUIRED_PACKAGES or _INSTALL_LOGIC_VERSION will
    invalidate the stored hash and trigger a dependency re-install.
    """
    data = repr(REQUIRED_PACKAGES).encode("utf-8")
    data += _INSTALL_LOGIC_VERSION.encode("utf-8")
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _read_deps_hash() -> str | None:
    """Read stored deps hash from the venv directory."""
    try:
        with open(DEPS_HASH_FILE, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
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
    except OSError as e:
        _log(f"Failed to write deps hash: {e}", Qgis.MessageLevel.Warning)


def _write_install_marker():
    """Record that an install of the current venv is in progress."""
    try:
        with open(INSTALL_MARKER_FILE, "w", encoding="utf-8") as f:
            f.write(os.path.basename(VENV_DIR))
    except OSError as e:
        _log(f"Could not write install marker: {e}", Qgis.MessageLevel.Warning)


def _clear_install_marker():
    try:
        os.unlink(INSTALL_MARKER_FILE)
    except OSError:
        pass  # nosec B110


def _install_marker_present() -> bool:
    """True when a previous install of the current venv was interrupted by a crash."""
    try:
        with open(INSTALL_MARKER_FILE, encoding="utf-8") as f:
            return f.read().strip() == os.path.basename(VENV_DIR)
    except OSError:
        return False


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


def _check_rosetta_warning() -> str | None:
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


def cleanup_old_venv_directories() -> list[str]:
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
                            shutil.rmtree(_win_extended_path(old_path))
                            _log(f"Cleaned up old venv: {old_path}",
                                 Qgis.MessageLevel.Info)
                            removed.append(old_path)
                        except Exception as e:
                            _log(f"Failed to remove old venv {old_path}: {e}", Qgis.MessageLevel.Warning)
        except Exception as e:
            _log(f"Error scanning for old venvs in {scan_dir}: {e}", Qgis.MessageLevel.Warning)

    return removed


def _check_gdal_available() -> tuple[bool, str]:
    """
    Check if GDAL system library is available (Linux and macOS).
    Returns (is_available, help_message).
    """
    if sys.platform not in ("linux", "darwin"):
        return True, ""

    try:
        result = subprocess.run(  # nosec B603 B607
            ["gdal-config", "--version"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5
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


def _get_ssl_error_help(error_text: str = "") -> str:
    """Wrapper that passes CACHE_DIR to the shared SSL error help."""
    return get_ssl_error_help(error_text, cache_dir=CACHE_DIR)


def get_venv_dir() -> str:
    return VENV_DIR


def get_venv_site_packages(venv_dir: str = None) -> str:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        return os.path.join(venv_dir, "Lib", "site-packages")
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
                _log(f"add_dll_directory({dll_dir}) failed: {exc}", Qgis.MessageLevel.Warning)
            if dll_dir not in path_parts:
                path_parts.insert(0, dll_dir)
    os.environ["PATH"] = os.pathsep.join(path_parts)


_rasterio_data_scoped = False


def _scope_rasterio_data_paths() -> None:
    """Point rasterio's own PROJ/GDAL at its bundled data, library-scoped.

    QGIS exports PROJ_LIB/PROJ_DATA/GDAL_DATA for its own GIS stack, and
    rasterio honors those env vars at import time, so without this its
    bundled PROJ would read QGIS's proj.db (different schema version) and
    CRS operations fail. The override must stay inside rasterio's own
    GDAL/PROJ instance: writing os.environ instead poisons QGIS itself and
    every other plugin in the process (PROJ failures observed in AI Edit
    on Windows). Never mutate os.environ here.
    """
    global _rasterio_data_scoped
    if _rasterio_data_scoped:
        return
    try:
        import rasterio
        from rasterio._env import set_gdal_config, set_proj_data_search_path
    except Exception as e:
        _log(f"rasterio unavailable, GIS data paths not scoped: {e}",
             Qgis.MessageLevel.Warning)
        return

    pkg_dir = os.path.dirname(os.path.abspath(rasterio.__file__))
    try:
        proj_dir = os.path.join(pkg_dir, "proj_data")
        if os.path.exists(os.path.join(proj_dir, "proj.db")):
            set_proj_data_search_path(proj_dir)
            _log(f"Scoped rasterio PROJ data to {proj_dir}", Qgis.MessageLevel.Info)
        gdal_dir = os.path.join(pkg_dir, "gdal_data")
        if os.path.isdir(gdal_dir):
            set_gdal_config("GDAL_DATA", gdal_dir)
            _log(f"Scoped rasterio GDAL data to {gdal_dir}", Qgis.MessageLevel.Info)
        _rasterio_data_scoped = True
    except Exception as e:  # nosec B110
        _log(f"Failed to scope rasterio data paths: {e}", Qgis.MessageLevel.Warning)


def _repair_poisoned_environment() -> None:
    """Undo global PROJ/GDAL env overrides left by plugin versions <= 1.1.0.

    Older releases pointed PROJ_DATA/PROJ_LIB/GDAL_DATA at the venv for the
    whole QGIS process. After a hot plugin upgrade those values are still in
    os.environ and keep breaking QGIS and other plugins until restart.
    """
    token = os.path.normcase(CACHE_DIR)
    poisoned = [
        var for var in ("PROJ_DATA", "PROJ_LIB", "GDAL_DATA")
        if token in os.path.normcase(os.environ.get(var, ""))
    ]
    if not poisoned:
        return

    qgis_proj_dir = None
    try:
        from qgis.core import QgsProjUtils
        for path in QgsProjUtils.searchPaths():
            if token in os.path.normcase(path):
                continue
            if os.path.exists(os.path.join(path, "proj.db")):
                qgis_proj_dir = path
                break
    except Exception:  # nosec B110
        pass
    for var in ("PROJ_DATA", "PROJ_LIB"):
        if var in poisoned:
            if qgis_proj_dir:
                os.environ[var] = qgis_proj_dir
            else:
                os.environ.pop(var, None)

    if "GDAL_DATA" in poisoned:
        gdal_candidates = []
        try:
            from qgis.core import QgsApplication
            gdal_candidates.append(os.path.join(QgsApplication.prefixPath(), "share", "gdal"))
            gdal_candidates.append(os.path.join(QgsApplication.pkgDataPath(), "gdal"))
        except Exception:  # nosec B110
            pass
        osgeo4w_root = os.environ.get("OSGEO4W_ROOT")
        if osgeo4w_root:
            gdal_candidates.append(os.path.join(osgeo4w_root, "apps", "gdal", "share", "gdal"))
            gdal_candidates.append(os.path.join(osgeo4w_root, "share", "gdal"))
        qgis_gdal_dir = next(
            (c for c in gdal_candidates if os.path.exists(os.path.join(c, "gdalvrt.xsd"))),
            None,
        )
        if qgis_gdal_dir:
            os.environ["GDAL_DATA"] = qgis_gdal_dir
        else:
            os.environ.pop("GDAL_DATA", None)

    _log(
        "Repaired PROJ/GDAL environment poisoned by an older plugin version: " + ", ".join(poisoned),
        Qgis.MessageLevel.Info,
    )


# Serializes ensure_venv_packages_available across threads. The old-numpy
# branch below does sys.modules/sys.path surgery (delete numpy*/pandas*,
# re-import from the venv); it is called from worker threads (DeviceInfoWorker,
# VerifyWorker) AND from the main thread (module-top of polygon_exporter /
# device_manager on their first import). Two threads interleaving that swap can
# observe numpy half-deleted / half-imported: an ImportError at best, a native
# C-extension re-init fault (= all of QGIS down) at worst. RLock, not Lock: the
# re-imports inside can transitively re-enter this function via a module-top
# ensure_venv_packages_available() call.
_ensure_packages_lock = threading.RLock()


def ensure_venv_packages_available():
    with _ensure_packages_lock:
        return _ensure_venv_packages_available_locked()


def _ensure_venv_packages_available_locked():
    # Heal sessions where an older plugin version globally overrode
    # PROJ/GDAL env vars (hot upgrade without QGIS restart).
    _repair_poisoned_environment()

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
                    f"Reloaded typing_extensions {old_ver} -> {new_te.__version__} from venv",
                    Qgis.MessageLevel.Info
                )
        except Exception:
            _log("Failed to reload typing_extensions, torch may fail", Qgis.MessageLevel.Warning)

    # --- sys.modules / sys.path surgery for old QGIS numpy ---
    #
    # WHY:  QGIS 3.22-3.28 bundle numpy < 1.22.4 which is incompatible with
    #        pandas >= 2.0 and modern torch. Without this fix, importing pandas
    #        or torch fails on those QGIS versions. (issues #130/#133/#138)
    #
    # WHAT: If old numpy is detected, this block:
    #        1. Deletes all numpy/pandas entries from sys.modules
    #        2. Temporarily removes QGIS Python paths containing numpy from sys.path
    #        3. Invalidates import caches so Python rescans directories
    #        4. Re-imports numpy from the venv site-packages
    #        5. Restores removed paths (in finally block, guaranteed)
    #
    # RISK: Modifies sys.path and sys.modules in the running QGIS process.
    #        This can affect module resolution for other plugins in the same
    #        session. The finally block ensures sys.path is always restored.
    #
    # REMOVE WHEN: Minimum supported QGIS is raised above 3.28.
    #
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
        # Importing rasterio pulls in numpy, so this must stay AFTER the
        # numpy version check above.
        _scope_rasterio_data_paths()
        return True

    qgis_ver = Qgis.QGIS_VERSION.split("-")[0]
    _log(
        f"QGIS {qgis_ver} with old numpy {old_version} detected. "
        "Forcing venv numpy/pandas...",
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
                f"(still {old_version}). pandas may fail on this QGIS.",
                Qgis.MessageLevel.Warning)
        else:
            _log(
                f"Reloaded numpy {old_version} -> {new_ver} from venv",
                Qgis.MessageLevel.Info)

    except Exception as e:
        _log(
            f"Failed to reload numpy: {e}. "
            "Plugin may not work on this QGIS version.",
            Qgis.MessageLevel.Warning)

    finally:
        # Always restore paths even on exception
        for p in removed_paths:
            if p not in sys.path:
                sys.path.append(p)

    # After numpy surgery so rasterio's numpy import resolves to the venv copy.
    _scope_rasterio_data_paths()
    return True


def get_venv_python_path(venv_dir: str = None) -> str:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python3")


def get_venv_pip_path(venv_dir: str = None) -> str:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "pip.exe")
    return os.path.join(venv_dir, "bin", "pip")


def _get_qgis_python() -> str | None:
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

        result = subprocess.run(  # nosec B603
            [python_path, "-c", "import sys; print(sys.version)"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
            env=env, **_get_subprocess_kwargs(),
        )
        if result.returncode == 0:
            _log(f"QGIS Python verified: {result.stdout.strip()}", Qgis.MessageLevel.Info)
            return python_path
        _log(f"QGIS Python failed verification: {result.stderr}", Qgis.MessageLevel.Warning)
        return None
    except Exception as e:
        _log(f"QGIS Python verification error: {e}", Qgis.MessageLevel.Warning)
        return None


def _system_python_matches_target(python3_path: str) -> bool:
    """Check a system python3 is a runnable interpreter at the target version.

    Used as the Linux fallback when the standalone build cannot execute
    (NixOS, musl/old glibc, sandboxed). We only need major.minor to match
    the same target the standalone Python would have been (get_qgis_python_
    version); an exact patch match is not required.
    """
    from .python_manager import get_qgis_python_version
    try:
        env = _get_clean_env_for_venv()
        result = subprocess.run(  # nosec B603
            [python3_path, "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
            env=env, **_get_subprocess_kwargs(),
        )
        if result.returncode != 0:
            return False
        parts = result.stdout.strip().split()
        if len(parts) != 2:
            return False
        installed = (int(parts[0]), int(parts[1]))
        return installed == get_qgis_python_version()
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        _log(f"System Python version check failed: {e}", Qgis.MessageLevel.Warning)
        return False


def _get_system_python() -> str:
    """
    Get the path to the Python executable for creating venvs.

    Uses the standalone Python downloaded by python_manager.
    On Windows, falls back to QGIS's bundled Python if standalone is unavailable
    (e.g. when anti-malware blocks the standalone download).

    Exception: on unsupported Windows versions (7 / Vista / XP / Server 2003)
    the QGIS-bundled Python is itself the broken-`_ssl` interpreter the user
    is hitting, so falling back loops them back into the same SSL failure the
    plugin reports. Short-circuit with the unsupported-OS message instead.
    """
    from .python_manager import (
        get_standalone_python_path,
        is_unsupported_windows,
        remove_standalone_python,
        standalone_python_exists,
        verify_standalone_python,
    )

    unsupported, why = is_unsupported_windows()
    if unsupported:
        raise RuntimeError(why)

    if standalone_python_exists():
        ok, msg = verify_standalone_python()
        if ok:
            python_path = get_standalone_python_path()
            _log(f"Using standalone Python: {python_path}", Qgis.MessageLevel.Info)
            return python_path
        _log(
            f"Standalone Python broken ({msg}), removing...",
            Qgis.MessageLevel.Warning
        )
        remove_standalone_python()

    # On Linux, fall back to a matching system Python when the standalone
    # build cannot run: NixOS (no dynamic FHS paths), musl/Alpine and old
    # glibc distros (astral's *-unknown-linux-gnu binaries need glibc
    # >= ~2.17), and sandboxed installs (Flatpak/Snap). Only NixOS was
    # covered before; generalize so any Linux with a compatible python3
    # on PATH does not dead-end on the raise below.
    if sys.platform == "linux":
        python3 = shutil.which("python3")
        if python3 and _system_python_matches_target(python3):
            _log(f"Linux fallback: using system Python: {python3}", Qgis.MessageLevel.Info)
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
            shutil.rmtree(_win_extended_path(venv_dir), ignore_errors=True)
            _log(f"Cleaned up partial venv: {venv_dir}", Qgis.MessageLevel.Info)
        except Exception:
            _log(f"Could not clean up partial venv: {venv_dir}", Qgis.MessageLevel.Warning)


def _venv_is_functional(venv_dir: str = None) -> bool:
    """Check that an existing venv is actually usable, not just present.

    ``venv_exists`` only checks that the python executable file is there. A
    venv left half-created by an interrupted run can have the executable but
    no valid ``pyvenv.cfg`` - uv then refuses it ("No virtual environment or
    system Python installation found for path ...; run `uv venv`") and the
    whole install fails. Require both a valid venv marker AND a working
    interpreter so a broken venv is detected and recreated. (#64)
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    python_path = get_venv_python_path(venv_dir)
    if not os.path.exists(python_path):
        return False
    if not os.path.exists(os.path.join(venv_dir, "pyvenv.cfg")):
        _log("Existing venv has no pyvenv.cfg (incomplete), will recreate",
             Qgis.MessageLevel.Warning)
        return False
    try:
        result = subprocess.run(  # nosec B603
            [python_path, "-c", "pass"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            env=_get_clean_env_for_venv(),
            **_get_subprocess_kwargs(),
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError) as e:
        _log(f"Existing venv interpreter is not runnable, will recreate: {e}",
             Qgis.MessageLevel.Warning)
        return False


def _venv_base_python_ok(venv_dir: str = None) -> tuple[bool, str]:
    """Check that pyvenv.cfg still points at an existing base interpreter.

    A venv does not embed Python: its launcher re-executes the interpreter
    recorded in the ``home`` line of pyvenv.cfg. When that base install was
    deleted by a cleanup tool, or the recorded path is corrupt (a stray
    quote has been seen in the field), every worker spawn dies with
    "No Python at ..." while site-packages still look complete, so the
    panel kept saying ready on an install that could never run.
    Filesystem-only, no subprocess: safe from the main thread.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    cfg_path = os.path.join(venv_dir, "pyvenv.cfg")
    home = None
    try:
        with open(cfg_path, encoding="utf-8") as f:
            for line in f:
                key, sep, value = line.partition("=")
                if sep and key.strip().lower() == "home":
                    home = value.strip()
                    break
    except OSError as e:
        return False, f"pyvenv.cfg unreadable: {e}"
    if not home:
        return False, "pyvenv.cfg has no home entry"
    if '"' in home or "'" in home:
        return False, f"pyvenv.cfg home path is corrupt: {home}"
    if not os.path.isdir(home):
        return False, f"venv base Python missing: {home}"
    try:
        entries = os.listdir(home)
    except OSError:
        entries = []
    if not any(name.startswith("python") for name in entries):
        return False, f"no python executable in venv base: {home}"
    return True, "venv base Python present"


def venv_needs_repair() -> bool:
    """True when the venv exists but cannot actually run anymore.

    Used by the UI to route a runtime worker-spawn failure to the
    one-click repair path (reinstall recreates the venv in place).
    """
    if not venv_exists():
        return False
    base_ok, base_msg = _venv_base_python_ok()
    if not base_ok:
        _log(f"venv_needs_repair: {base_msg}", Qgis.MessageLevel.Warning)
        return True
    return not _venv_is_functional()


def create_venv(
    venv_dir: str = None,
    progress_callback: Callable[[int, str], None] | None = None
) -> tuple[bool, str]:
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
            subprocess_kwargs = _get_subprocess_kwargs()

            result = subprocess.run(  # nosec B603
                uv_cmd,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
                env=env, **subprocess_kwargs,
            )
            if result.returncode == 0:
                _log("Virtual environment created with uv", Qgis.MessageLevel.Success)
                if progress_callback:
                    progress_callback(20, "Virtual environment created (uv)")
                return True, "Virtual environment created"
            error_msg = result.stderr or result.stdout or ""
            _log(f"uv venv creation failed: {error_msg[:200]}", Qgis.MessageLevel.Warning)
            _cleanup_partial_venv(venv_dir)
            # Fall through to standard venv creation
            remove_uv()
            _uv_available = False
            _uv_path = None
            _log("Falling back to python -m venv", Qgis.MessageLevel.Warning)
        except Exception as e:
            _log(f"uv venv exception: {e}, falling back to python -m venv", Qgis.MessageLevel.Warning)
            _cleanup_partial_venv(venv_dir)
            remove_uv()
            _uv_available = False
            _uv_path = None

    # Standard venv creation with python -m venv
    cmd = [system_python, "-m", "venv", venv_dir]
    try:
        subprocess_kwargs = _get_subprocess_kwargs()

        result = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8", errors="replace",
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
                    ensurepip_result = subprocess.run(  # nosec B603
                        ensurepip_cmd,
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
                        env=env,
                        **subprocess_kwargs,
                    )
                    if ensurepip_result.returncode == 0:
                        _log("pip bootstrapped via ensurepip", Qgis.MessageLevel.Success)
                        ensurepip_ok = True
                    else:
                        err = ensurepip_result.stderr or ensurepip_result.stdout or ""
                        _log(f"ensurepip failed: {err[:200]}",
                             Qgis.MessageLevel.Warning)
                except Exception as e:
                    _log(f"ensurepip exception: {e}",
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
            result2 = subprocess.run(  # nosec B603
                nopip_cmd,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                env=env, **subprocess_kwargs,
            )
            if result2.returncode == 0:
                _log("Venv created (--without-pip), bootstrapping pip...", Qgis.MessageLevel.Info)
                python_in_venv = get_venv_python_path(venv_dir)
                ensurepip_cmd = [python_in_venv, "-m", "ensurepip", "--upgrade"]
                ep_result = subprocess.run(  # nosec B603
                    ensurepip_cmd,
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
                    env=env, **subprocess_kwargs,
                )
                if ep_result.returncode == 0:
                    _log("pip bootstrapped via ensurepip", Qgis.MessageLevel.Success)
                    if progress_callback:
                        progress_callback(20, "Virtual environment created")
                    return True, "Virtual environment created"
                err = ep_result.stderr or ep_result.stdout or ""
                _log(f"ensurepip failed: {err[:200]}", Qgis.MessageLevel.Warning)
                if _uv_available and _uv_path:
                    _log(
                        "ensurepip unavailable but uv present, "
                        "continuing without pip",
                        Qgis.MessageLevel.Warning)
                    if progress_callback:
                        progress_callback(20, "Virtual environment created")
                    return True, "Virtual environment created"
                _cleanup_partial_venv(venv_dir)
                return False, f"Failed to bootstrap pip: {err[:200]}"
            err = result2.stderr or result2.stdout or ""
            _log(f"Retry --without-pip failed: {err[:200]}", Qgis.MessageLevel.Critical)
            _cleanup_partial_venv(venv_dir)
            return False, "Virtual environment creation timed out"
        except Exception as e2:
            _log(f"Retry --without-pip exception: {e2}", Qgis.MessageLevel.Critical)
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
        pass  # nosec B110
    return path


def _win_extended_path(path: str) -> str:
    """Prefix an absolute Windows path with the \\\\?\\ extended-length marker.

    The torch venv tree (deep Lib/site-packages/torch/... module paths) can
    exceed the legacy 260-char MAX_PATH, which is still the Windows default
    (LongPathsEnabled=0). The \\\\?\\ prefix routes shutil/glob's underlying
    file APIs around that limit. No-op on non-Windows and on paths already
    prefixed; UNC paths use the distinct \\\\?\\UNC\\ form.
    """
    if sys.platform != "win32" or not path:
        return path
    if path.startswith("\\\\?\\"):
        return path
    abs_path = os.path.abspath(path)
    if abs_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")
    return "\\\\?\\" + abs_path


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
        pass  # nosec B110
    return path


def _build_install_cmd(python_path: str, pip_args: list) -> list:
    """Build an install command using uv (if available) or pip.

    Translates pip flags to uv equivalents when _uv_available is True.
    Note: --trusted-host is translated to --allow-insecure-host for uv,
    which disables TLS verification. This should only be present in
    pip_args during SSL error retry, never in the default install path.
    """
    if _uv_available and _uv_path:
        cmd = [_uv_path, "pip"]
        skip_next = False
        for i, arg in enumerate(pip_args):
            if skip_next:
                skip_next = False
                continue
            if arg == "--disable-pip-version-check" or arg == "--no-warn-script-location" or arg == "--prefer-binary":
                continue
            if arg in ("--retries", "--timeout"):
                skip_next = True  # skip flag and its value
                continue
            if arg == "--no-cache-dir":
                cmd.append("--no-cache")
                continue
            if arg == "--force-reinstall":
                cmd.append("--reinstall")
                continue
            if arg == "--trusted-host":
                cmd.append("--allow-insecure-host")
                continue
            if arg == "--proxy":
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
    # Pure-Python pip fallback (no uv): pip's own file writes under a deep
    # torch venv tree can exceed Windows' legacy 260-char MAX_PATH, so route
    # the interpreter path through the extended-length form as cheap insurance.
    return [_win_extended_path(python_path), "-m", "pip"] + pip_args


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
        result = subprocess.run(  # nosec B603
            [python_path, "-c",
             "import numpy; print(numpy.__version__)"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
            env=env, **subprocess_kwargs,
        )
        if result.returncode != 0:
            return  # numpy not installed or broken, nothing to fix here

        version_str = result.stdout.strip()
        major = int(version_str.split(".")[0])
        if major >= 2:
            _log(
                f"numpy {version_str} detected (>=2.0), forcing downgrade to <2.0.0...",
                Qgis.MessageLevel.Warning
            )
            downgrade_args = [
                "install", "--force-reinstall", "--no-deps",
                "--disable-pip-version-check",
            ] + [
                "numpy>=1.26.0,<2.0.0",
            ]
            downgrade_cmd = _build_install_cmd(python_path, downgrade_args)
            downgrade_result = subprocess.run(  # nosec B603
                downgrade_cmd,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
                env=env, **subprocess_kwargs,
            )
            if downgrade_result.returncode == 0:
                _log("numpy downgraded successfully to <2.0.0", Qgis.MessageLevel.Success)
            else:
                err = downgrade_result.stderr or downgrade_result.stdout or ""
                _log(f"numpy downgrade failed: {err[:200]}", Qgis.MessageLevel.Warning)
    except Exception as e:
        _log(f"numpy version check failed: {e}", Qgis.MessageLevel.Warning)


def _get_verification_timeout(package_name: str) -> int:
    """
    Get verification timeout in seconds for a given package.

    torch needs extra time because the first import loads native DLLs on Windows,
    which can take >30s. torchvision also loads heavy native libraries.
    """
    if package_name == "torch":
        return 120
    if package_name in ("torchvision", "pandas"):
        # pandas loads many .pyd C extensions on first import;
        # antivirus (Windows Defender) scans each one, easily exceeding 120s
        return 180
    return 30


class _PipResult:
    """Lightweight result object compatible with subprocess.CompletedProcess."""

    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _parse_pip_download_line(line: str) -> str | None:
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
            size = f"{num / 1000:.1f} GB"

    return f"Downloading {pkg_name} ({size})"


def _run_pip_install(
    cmd: list[str],
    timeout: int,
    env: dict,
    subprocess_kwargs: dict,
    package_name: str,
    package_index: int,
    total_packages: int,
    progress_start: int,
    progress_end: int,
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> _PipResult:
    """
    Run a pip install command with real-time progress updates.

    Uses Popen with stdout/stderr redirected to temp files (never PIPE:
    an undrained PIPE deadlocks once the buffer fills) and polls every
    2 seconds to provide live feedback.
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
        # If fdopen fails, close the raw fds, remove the temp files, re-raise
        try:
            os.close(stdout_fd)
        except Exception:
            pass  # nosec B110
        try:
            os.close(stderr_fd)
        except Exception:
            pass  # nosec B110
        for leaked_path in (stdout_path, stderr_path):
            try:
                os.unlink(leaked_path)
            except OSError:
                pass  # nosec B110
        raise

    process = None
    try:
        process = subprocess.Popen(  # nosec B603
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            encoding="utf-8", errors="replace",
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
                with open(stdout_path, encoding="utf-8", errors="replace") as f:
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
                pass  # nosec B110

            # Format elapsed time nicely
            if elapsed >= 60:
                elapsed_str = f"{elapsed // 60}m {elapsed % 60}s"
            else:
                elapsed_str = f"{elapsed}s"

            # Build progress message
            if last_download_status:
                msg = f"{last_download_status}... {elapsed_str}"
            elif package_name == "torch":
                msg = f"Downloading PyTorch (~600 MB)... {elapsed_str}"
            else:
                msg = f"Installing {package_name}... {elapsed_str}"

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

        # Honor a cancellation that lands just as the process finishes
        if cancel_check and cancel_check():
            return _PipResult(-1, "", "Installation cancelled")

        # Process finished - close files before reading
        stdout_file.close()
        stderr_file.close()
        stdout_file = None
        stderr_file = None

        # Read full output
        try:
            with open(stdout_path, encoding="utf-8", errors="replace") as f:
                full_stdout = f.read()
        except Exception:
            full_stdout = ""

        try:
            with open(stderr_path, encoding="utf-8", errors="replace") as f:
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
                pass  # nosec B110
        if stderr_file is not None:
            try:
                stderr_file.close()
            except Exception:
                pass  # nosec B110
        # Clean up temp files
        try:
            os.unlink(stdout_path)
        except Exception:
            pass  # nosec B110
        try:
            os.unlink(stderr_path)
        except Exception:
            pass  # nosec B110


def install_dependencies(
    venv_dir: str = None,
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[bool, str]:
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
            upgrade_result = subprocess.run(  # nosec B603
                upgrade_cmd,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
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
            _log(f"pip upgrade failed (non-critical): {str(e)[:200]}",
                 Qgis.MessageLevel.Warning)

    total_packages = len(REQUIRED_PACKAGES)
    base_progress = 20
    progress_range = 80  # from 20% to 100%

    # Weighted progress allocation proportional to download size.
    # Name-based map so weights stay correct if REQUIRED_PACKAGES order changes.
    _wmap = {
        "numpy": 5, "torch": 30, "torchvision": 15,
        "rasterio": 10,
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
            pass  # nosec B110
        try:
            os.unlink(constraints_path)
        except Exception:
            pass  # nosec B110
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
                        f"Installing {package_name} (~600MB)... ({i + 1}/{total_packages})")
                else:
                    progress_callback(
                        pkg_start,
                        f"Installing {package_name}... ({i + 1}/{total_packages})")

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
            # SSL bypass flags (--trusted-host) are NOT added here by default.
            # They are only applied on retry after an SSL error is detected,
            # to avoid disabling TLS verification for all users.
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

                # On Linux, PyPI's default torch wheels bundle CUDA and pull
                # multi-GB nvidia-* dependencies. The official CPU index serves
                # the same versions as ~180 MB CPU-only wheels. Windows and
                # macOS wheels on PyPI are already CPU-only, so the extra
                # index (and its proxy-block risk) stays Linux-only.
                use_cpu_index = (
                    package_name in ("torch", "torchvision") and sys.platform.startswith("linux")
                )
                if use_cpu_index:
                    cpu_args = pip_args[:-1] + [
                        "--index-url", TORCH_CPU_INDEX_URL, pip_args[-1]]
                    base_cmd = _build_install_cmd(python_path, cpu_args)

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

                # CPU index unreachable (some corporate proxies whitelist
                # pypi.org only): fall back to PyPI, then let the normal
                # error handling judge that second attempt.
                if use_cpu_index and result.returncode != 0:
                    _log(
                        "CPU wheel index failed for "
                        f"{package_name} (code {result.returncode}), "
                        "retrying from the default index...",
                        Qgis.MessageLevel.Warning
                    )
                    base_cmd = _build_install_cmd(python_path, pip_args)
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
                    if result.returncode == -1 and "cancelled" in (result.stderr or "").lower():
                        _log("Installation cancelled by user", Qgis.MessageLevel.Warning)
                        return False, "Installation cancelled"

                # If Windows process crash, retry with pip.exe as fallback
                # (skip when using uv - it doesn't have this issue)
                if not _uv_available and _is_windows_process_crash(result.returncode):
                    _log(
                        f"Process crash detected (code {result.returncode}), "
                        "retrying with pip.exe...",
                        Qgis.MessageLevel.Warning
                    )
                    if progress_callback:
                        progress_callback(
                            pkg_start,
                            f"Retrying {package_name}... ({i + 1}/{total_packages})"
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
                                f"Retrying {package_name}... ({i + 1}/{total_packages})"
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

                # If failed, check for SSL errors and retry.
                # Strategy: for uv, try --system-certs (uses OS trust store,
                # handles corporate proxy CAs safely). Fully disabling TLS
                # verification is NOT done unless the user has explicitly opted
                # in (see _insecure_install_opt_in): otherwise an active MITM
                # could force the verified attempt to fail and then serve a
                # poisoned wheel.
                if result.returncode != 0 and not _is_windows_process_crash(result.returncode):
                    error_output = result.stderr or result.stdout or ""

                    if _is_ssl_error(error_output):
                        if _uv_available:
                            # uv: try OS native certificate store first (safe)
                            _log(
                                "SSL error detected, retrying with system certificates...",
                                Qgis.MessageLevel.Warning
                            )
                            if progress_callback:
                                progress_callback(
                                    pkg_start,
                                    f"SSL error, retrying {package_name} (system certs)... ({i + 1}/{total_packages})"
                                )
                            ssl_cmd_safe = base_cmd + ["--system-certs"]
                            result = _run_pip_install(
                                cmd=ssl_cmd_safe,
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

                        # If still failing, only disable TLS verification when
                        # the user has explicitly opted in. Otherwise surface the
                        # SSL error so a MITM can never silently downgrade the
                        # install to an unverified channel.
                        if result.returncode != 0 and not _insecure_install_opt_in():
                            _log(
                                "SSL error persists after the system-certificate retry. "
                                "Not disabling TLS verification (secure default). If you are "
                                "behind a corporate proxy, add its CA to your OS trust store, "
                                "or set TerraLab/allow_insecure_install to opt in to an "
                                "unverified install at your own risk.",
                                Qgis.MessageLevel.Warning
                            )
                        elif result.returncode != 0:
                            _log(
                                "SSL error persists, retrying with TLS verification bypass "
                                "(user opted in via allow_insecure_install)...",
                                Qgis.MessageLevel.Warning
                            )
                            if progress_callback:
                                progress_callback(
                                    pkg_start,
                                    f"SSL bypass retry for {package_name}... ({i + 1}/{total_packages})"
                                )
                            # Build a new command with SSL bypass flags injected.
                            # For uv: --trusted-host → --allow-insecure-host (via _build_install_cmd)
                            # For pip: --trusted-host directly
                            ssl_bypass_pip_args = list(pip_args) + _get_pip_ssl_bypass_flags()
                            ssl_cmd_bypass = _build_install_cmd(python_path, ssl_bypass_pip_args)
                            result = _run_pip_install(
                                cmd=ssl_cmd_bypass,
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
                                f"Cache error, retrying {package_name}... ({i + 1}/{total_packages})"
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
                                f"Network error detected, retrying in {wait}s "
                                f"(attempt {attempt}/4)...",
                                Qgis.MessageLevel.Warning
                            )
                            if progress_callback:
                                progress_callback(
                                    pkg_start,
                                    f"Network error, retry {attempt}/4 in {wait}s..."
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
                            f"No matching distribution for {package_name}, "
                            "retrying with --no-cache-dir...",
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

                # Stale dist-info rename/RECORD error applies to any package (#20).
                if result.returncode != 0:
                    error_output = result.stderr or result.stdout or ""
                    if _is_rename_or_record_error(error_output):
                        _log(
                            f"Stale dist-info detected for {package_name}, cleaning and "
                            "retrying with --force-reinstall...",
                            Qgis.MessageLevel.Warning
                        )
                        try:
                            site_pkgs = get_venv_site_packages()
                            if os.path.isdir(site_pkgs):
                                import glob as _glob
                                import shutil as _shutil
                                pattern = os.path.join(
                                    site_pkgs,
                                    f"{package_name}-*.dist-info")
                                for dist_dir in _glob.glob(pattern):
                                    _shutil.rmtree(dist_dir, ignore_errors=True)
                                    _log(f"Removed stale {dist_dir}",
                                         Qgis.MessageLevel.Warning)
                        except Exception as exc:
                            _log(f"Failed to clean dist-info: {exc}",
                                 Qgis.MessageLevel.Warning)

                        if progress_callback:
                            progress_callback(
                                pkg_start,
                                f"Retrying {package_name}... ({i + 1}/{total_packages})")
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
                # Log the END of pip output: that is where the real error is.
                # This line lands in the log buffer that error telemetry and
                # the "copy logs" button read, so keep it informative.
                _log(f"pip error output (tail): {install_error_msg[-1000:]}", Qgis.MessageLevel.Critical)

                # Check for Windows process crash
                if last_returncode is not None and _is_windows_process_crash(last_returncode):
                    _log(_get_crash_help(venv_dir), Qgis.MessageLevel.Warning)
                    return False, f"Failed to install {package_name}: process crashed (code {last_returncode})"

                # Check for DLL load failure (missing system libraries)
                is_dll_err = sys.platform == "win32" and _is_dll_init_error(install_error_msg)
                if is_dll_err and package_name in ("torch", "torchvision"):
                    _log(_get_vcpp_help(), Qgis.MessageLevel.Warning)
                    return False, f"Failed to install {package_name}: {_get_vcpp_help()}"

                # Check for native-module file-lock errors FIRST: a .pyd/.dll/.so
                # that cannot be removed because it is loaded in the running
                # QGIS process. Fix is "restart QGIS", NOT antivirus exclusion
                # - so this must precede the antivirus/rename branches below.
                if _is_file_locked_error(install_error_msg):
                    _log(_get_file_locked_help(), Qgis.MessageLevel.Warning)
                    return False, (
                        f"Failed to install {package_name}: file in use by QGIS. "
                        "Please close and reopen QGIS, then retry."
                    )

                # Check for rename/record errors (antivirus blocking on Windows) - before SSL
                # because uv output may contain SSL_CERT_DIR warnings alongside rename errors
                if sys.platform == "win32" and _is_rename_or_record_error(install_error_msg):
                    help_msg = (
                        f"Failed to install {package_name}: file rename blocked.\n\n"
                        "This is typically caused by antivirus or security software "
                        "scanning files during installation.\n\n"
                        "Please try:\n"
                        "  1. Temporarily disable real-time antivirus scanning\n"
                        f"  2. Add an exclusion for: {CACHE_DIR}\n"
                        "  3. Restart QGIS and reinstall dependencies"
                    )
                    _log(help_msg, Qgis.MessageLevel.Warning)
                    return False, f"Failed to install {package_name}: blocked by antivirus (rename failed)"

                # Check for SSL errors
                if _is_ssl_error(install_error_msg):
                    _log(_get_ssl_error_help(install_error_msg), Qgis.MessageLevel.Warning)
                    return False, f"Failed to install {package_name}: SSL error"

                # Check for proxy authentication errors (407)
                if _is_proxy_auth_error(install_error_msg):
                    _log(
                        "Proxy authentication failed (HTTP 407). "
                        "Configure proxy credentials in: "
                        "QGIS > Settings > Options > Network > Proxy "
                        "(User and Password fields).",
                        Qgis.MessageLevel.Warning
                    )
                    return False, f"Failed to install {package_name}: proxy authentication required (407)"

                # Check for network/connection errors (after retries exhausted)
                if _is_network_error(install_error_msg):
                    _log(
                        "Network connection failed after multiple retries. "
                        "Check internet connection, VPN/proxy settings, "
                        "and firewall rules for pypi.org and files.pythonhosted.org.",
                        Qgis.MessageLevel.Warning
                    )
                    return False, f"Failed to install {package_name}: network error"

                # Check for platform-unsupported wheels
                if "no matching platform tag" in install_error_msg.lower():
                    return False, (
                        f"Failed to install {package_name}: no wheels with a matching "
                        "platform tag for this OS/architecture."
                    )

                # Check for antivirus blocking
                if _is_antivirus_error(install_error_msg):
                    _log(_get_pip_antivirus_help(venv_dir), Qgis.MessageLevel.Warning)
                    return False, f"Failed to install {package_name}: blocked by antivirus or security policy"

                # Check for "unable to create process" (broken pip shim)
                if _is_unable_to_create_process(install_error_msg):
                    return False, (
                        f"Failed to install {package_name}: unable to create process.\n\n"
                        "Please try:\n"
                        f"  1. Delete the folder: {CACHE_DIR}\n"
                        "  2. Restart QGIS and reinstall dependencies"
                    )

                # Check for GDAL issues on Linux/macOS when rasterio fails
                if package_name == "rasterio":
                    gdal_ok, gdal_help = _check_gdal_available()
                    if not gdal_ok and gdal_help:
                        _log(gdal_help, Qgis.MessageLevel.Warning)
                        return False, f"Failed to install {package_name}: GDAL library not found"

                return False, f"Failed to install {package_name}: {install_error_msg[:200]}"

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


def _get_qgis_proxy_settings() -> str | None:
    """Read proxy configuration from QGIS settings.

    Returns a proxy URL string (with optional authentication)
    or None if proxy is not configured or disabled.
    """
    try:
        from urllib.parse import quote as url_quote

        from qgis.core import QgsSettings

        settings = QgsSettings()
        enabled = settings.value("proxy/proxyEnabled", False, type=bool)
        if not enabled:
            return None

        # pip/uv only speak HTTP proxies; SOCKS needs extra packages we
        # don't ship, so fall back to a direct/environment connection.
        proxy_type = settings.value("proxy/proxyType", "", type=str)
        if proxy_type == "Socks5Proxy":
            _log(
                "QGIS is configured with a SOCKS5 proxy, which is not "
                "supported for dependency installs. Trying a direct "
                "connection instead.",
                Qgis.MessageLevel.Warning
            )
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
            proxy_url += f":{port}"

        return proxy_url
    except Exception as e:
        _log(f"Could not read QGIS proxy settings: {e}", Qgis.MessageLevel.Warning)
        return None


def _get_system_proxy_settings() -> str | None:
    """Fall back to the OS-level proxy (Windows registry, macOS network
    settings, HTTP(S)_PROXY env vars) when no proxy is set in QGIS.

    Corporate machines often have a system proxy that browsers and QGIS
    use implicitly, while our pip/uv subprocesses get a clean environment
    and fail with a network error unless we pass it explicitly.
    """
    try:
        import urllib.request

        proxies = urllib.request.getproxies()
        proxy_url = proxies.get("https") or proxies.get("http")
        if proxy_url and proxy_url.lower().startswith(("http://", "https://")):
            return proxy_url
    except Exception as e:
        _log(f"Could not read system proxy settings: {e}", Qgis.MessageLevel.Warning)
    return None


def _get_effective_proxy_url() -> str | None:
    """QGIS-configured proxy first, then the OS system proxy."""
    return _get_qgis_proxy_settings() or _get_system_proxy_settings()


def _get_pip_proxy_args() -> list[str]:
    """Get pip --proxy argument if a QGIS or system proxy is configured."""
    proxy_url = _get_effective_proxy_url()
    if proxy_url:
        _log("Using proxy for pip: {}".format(
            proxy_url.split("@")[-1] if "@" in proxy_url else proxy_url),
            Qgis.MessageLevel.Info
        )
        return ["--proxy", proxy_url]
    return []


def _apply_cache_containment(env: dict) -> None:
    """Point uv/pip caches and build temp at CACHE_DIR for install subprocesses.

    By default uv, pip and the OS temp all live on the home/system volume even
    when the user points AI_SEGMENTATION_CACHE_DIR at a big secondary drive.
    That user then passes the single CACHE_DIR disk preflight and still fills
    the system drive mid torch install. Co-locating the caches and temp on the
    CACHE_DIR volume makes that one preflight cover the whole install, lets uv
    hardlink from its cache into the venv (faster, no cross-filesystem copy)
    and stops polluting the OS drive.

    This only touches the env dict handed to install subprocesses, never the
    QGIS process environment. Best-effort: if the directories cannot be
    created, the env is left untouched so the install still runs on the OS
    defaults rather than failing on our own bookkeeping.
    """
    uv_cache = os.path.join(CACHE_DIR, "uv_cache")
    pip_cache = os.path.join(CACHE_DIR, "pip_cache")
    tmp_dir = os.path.join(CACHE_DIR, "tmp")
    try:
        for path in (uv_cache, pip_cache, tmp_dir):
            os.makedirs(path, exist_ok=True)
    except OSError as e:
        _log(
            f"Cache containment skipped, using default locations: {e}",
            Qgis.MessageLevel.Info)
        return
    env["UV_CACHE_DIR"] = uv_cache
    env["PIP_CACHE_DIR"] = pip_cache
    env["TMPDIR"] = tmp_dir   # POSIX
    env["TEMP"] = tmp_dir     # Windows
    env["TMP"] = tmp_dir      # Windows


def _get_clean_env_for_venv() -> dict:
    # Shared sanitization (QGIS/PROJ/GDAL vars, SSL_CERT_DIR) lives in
    # subprocess_utils; install-specific extras are layered on top here.
    env = _get_base_clean_env()

    # Contain package-manager caches and build temp on the CACHE_DIR volume so
    # the disk preflight actually covers the whole install (see the helper).
    _apply_cache_containment(env)

    # Skip sam2 CUDA extension compilation (Python fallback works fine)
    env["SAM2_BUILD_CUDA"] = "0"

    # CPU-only on NVIDIA hardware (#31). MPS is unaffected.
    env["CUDA_VISIBLE_DEVICES"] = ""

    # Increase uv download timeout (default 30s too short for large wheels)
    env["UV_HTTP_TIMEOUT"] = "300"

    # Corporate MITM proxies re-sign TLS with a company CA that uv's bundled
    # Mozilla roots reject (pip >= 24.2 already trusts the OS store via
    # truststore). Native TLS makes uv trust the same store as the rest of
    # the machine. setdefault so a user override wins.
    env.setdefault("UV_NATIVE_TLS", "1")

    # More retries with backoff for unstable home/corporate connections
    # (uv default is 3).
    env.setdefault("UV_HTTP_RETRIES", "5")

    # Propagate QGIS or system proxy settings to environment for pip/uv
    proxy_url = _get_effective_proxy_url()
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
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def _get_verification_code(package_name: str) -> str:
    """
    Get verification code that actually TESTS the package works, not just imports.

    This catches issues like pandas C extensions not being built properly.
    """
    if package_name == "pandas":
        # Test that pandas C extensions work by creating a DataFrame
        return "import pandas as pd; df = pd.DataFrame({'a': [1, 2, 3]}); print(df.sum())"
    if package_name == "numpy":
        # Test numpy array operations
        return "import numpy as np; a = np.array([1, 2, 3]); print(np.sum(a))"
    if package_name == "torch":
        # Test torch tensor creation
        return "import torch; t = torch.tensor([1, 2, 3]); print(t.sum())"
    if package_name == "rasterio":
        # Just import - rasterio needs a file to test fully
        return "import rasterio; print(rasterio.__version__)"
    if package_name == "sam2":
        return "from sam2.build_sam import build_sam2; print('ok')"
    if package_name == "segment-anything":
        return "from segment_anything import sam_model_registry; print('ok')"
    if package_name == "torchvision":
        return "import torchvision; print(torchvision.__version__)"
    import_name = package_name.replace("-", "_")
    return f"import {import_name}"


def _remove_torch_dirs(site_pkgs: str) -> None:
    """Remove torch/torchvision dirs from site-packages, including .dist-info
    and partial wheel leftovers, so a fresh reinstall starts clean."""
    site_pkgs = _win_extended_path(site_pkgs)
    for pattern in ("torch*", "torchvision*"):
        for target in glob.glob(os.path.join(site_pkgs, pattern)):
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)


def verify_venv(
    venv_dir: str = None,
    progress_callback: Callable[[int, str], None] | None = None
) -> tuple[bool, str]:
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
            result = subprocess.run(  # nosec B603
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8", errors="replace",
                timeout=pkg_timeout,
                env=env,
                **subprocess_kwargs
            )

            if result.returncode != 0:
                # A Python traceback puts the real cause on its LAST line, so
                # keep the full text for classification and surface the TAIL (not
                # the first 300 chars, which is only the traceback header and
                # hid errors like "blocked by an application control policy" that
                # live at the bottom, sending the user to a dead-end VC++
                # message) (#bug-kees).
                full_error = result.stderr or result.stdout or ""
                error_detail = full_error[-400:] if full_error else ""
                _log(
                    f"Package {package_name} verification failed: {error_detail}",
                    Qgis.MessageLevel.Warning
                )

                # Non-essential packages (e.g. pandas) are not used by the
                # plugin. A failure here is usually a missing Visual C++
                # Redistributable that a reinstall cannot fix, so don't block
                # the whole install - warn and move on. (#66)
                if package_name in NON_ESSENTIAL_PACKAGES:
                    _log(
                        f"Package {package_name} is not required by the plugin; "
                        "skipping it and continuing. If you need it, install the "
                        "Microsoft Visual C++ Redistributable (x64) and restart "
                        "QGIS.",
                        Qgis.MessageLevel.Warning
                    )
                    continue

                # Security-policy / antivirus block (AppLocker, WDAC, corporate
                # application-control, quarantined file): the library is present
                # but blocked from loading, so the fix is to whitelist the
                # folder, NOT to reinstall or install a VC++ runtime. Checked
                # BEFORE the DLL branch, which the same "DLL load failed" text
                # would otherwise trigger (#bug-kees).
                if _is_antivirus_error(full_error):
                    _log(_get_pip_antivirus_help(venv_dir), Qgis.MessageLevel.Warning)
                    return False, (
                        f"Package {package_name} is blocked by a security policy "
                        "(antivirus, AppLocker or application control).\n\n"
                        f"{_get_pip_antivirus_help(venv_dir)}"
                    )

                # DLL init error (WinError 1114) - try force-reinstall first,
                # as a conflicting DLL from another Python install may be the cause
                if _is_dll_init_error(full_error):
                    _log(
                        f"DLL init error for {package_name}, attempting "
                        "force-reinstall...",
                        Qgis.MessageLevel.Warning
                    )
                    pkg_spec = package_name
                    for name, spec in REQUIRED_PACKAGES:
                        if name == package_name:
                            pkg_spec = f"{name}{spec}"
                            break
                    reinstall_cmd = _build_install_cmd(
                        python_path,
                        ["install", "--force-reinstall", "--no-deps",
                         "--prefer-binary", pkg_spec])
                    try:
                        subprocess.run(  # nosec B603
                            reinstall_cmd,
                            capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=600,
                            env=env, **subprocess_kwargs
                        )
                        result2 = subprocess.run(  # nosec B603
                            cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=pkg_timeout,
                            env=env, **subprocess_kwargs
                        )
                        if result2.returncode == 0:
                            _log(
                                f"Package {package_name} fixed after "
                                "force-reinstall",
                                Qgis.MessageLevel.Success)
                            continue
                    except Exception:
                        pass  # nosec B110

                    # Nuclear option: delete torch dirs and reinstall fresh
                    _log(
                        "Force-reinstall did not fix DLL error for "
                        f"{package_name}. Nuking and reinstalling...",
                        Qgis.MessageLevel.Warning
                    )
                    try:
                        _remove_torch_dirs(get_venv_site_packages(venv_dir))
                        # Reinstall both packages
                        torch_spec = f"torch{TORCH_MIN}"
                        tv_spec = f"torchvision{TORCHVISION_MIN}"
                        nuke_cmd = _build_install_cmd(
                            python_path,
                            ["install", "--prefer-binary",
                             torch_spec, tv_spec])
                        subprocess.run(  # nosec B603
                            nuke_cmd,
                            capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=600,
                            env=env, **subprocess_kwargs
                        )
                        result3 = subprocess.run(  # nosec B603
                            cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=pkg_timeout,
                            env=env, **subprocess_kwargs
                        )
                        if result3.returncode == 0:
                            _log(
                                f"Package {package_name} fixed after nuke "
                                "reinstall",
                                Qgis.MessageLevel.Success)
                            continue
                    except Exception:
                        pass  # nosec B110

                    # 4th attempt (Windows only): try known-good pinned
                    # torch version that doesn't have DLL issues.
                    from .model_config import (
                        TORCH_WINDOWS_FALLBACK,
                        TORCHVISION_WINDOWS_FALLBACK,
                    )
                    if TORCH_WINDOWS_FALLBACK:
                        _log(
                            "Nuke reinstall did not fix DLL error. "
                            f"Trying pinned torch{TORCH_WINDOWS_FALLBACK} fallback...",
                            Qgis.MessageLevel.Warning
                        )
                        try:
                            _remove_torch_dirs(
                                get_venv_site_packages(venv_dir))
                            fallback_cmd = _build_install_cmd(
                                python_path,
                                ["install", "--prefer-binary",
                                 f"torch{TORCH_WINDOWS_FALLBACK}",
                                 f"torchvision{TORCHVISION_WINDOWS_FALLBACK}"])
                            subprocess.run(  # nosec B603
                                fallback_cmd,
                                capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=600,
                                env=env, **subprocess_kwargs
                            )
                            result4 = subprocess.run(  # nosec B603
                                cmd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=pkg_timeout,
                                env=env, **subprocess_kwargs
                            )
                            if result4.returncode == 0:
                                _log(
                                    f"Package {package_name} fixed with pinned "
                                    f"torch{TORCH_WINDOWS_FALLBACK} fallback",
                                    Qgis.MessageLevel.Success)
                                continue
                        except Exception:
                            pass  # nosec B110

                    _log(_get_vcpp_help(), Qgis.MessageLevel.Warning)
                    return False, (
                        f"Package {package_name} failed: {_get_vcpp_help()}"
                    )

                # Detect broken C extensions (antivirus may have quarantined .pyd files)
                error_lower = full_error.lower()
                broken_markers = [
                    "no module named", "_libs",
                    "dll load failed", "importerror",
                    "applocker", "application control",
                    "blocked by your organization",
                ]
                is_broken = any(m in error_lower for m in broken_markers)

                if is_broken:
                    _log(
                        f"Package {package_name} has broken C extensions, "
                        "attempting force-reinstall...",
                        Qgis.MessageLevel.Warning
                    )
                    # Find the version spec from REQUIRED_PACKAGES
                    pkg_spec = package_name
                    for name, spec in REQUIRED_PACKAGES:
                        if name == package_name:
                            pkg_spec = f"{name}{spec}"
                            break
                    reinstall_cmd = [
                        python_path, "-m", "pip", "install",
                        "--force-reinstall", "--no-deps",
                        "--disable-pip-version-check",
                        "--prefer-binary",
                        pkg_spec,
                    ]
                    try:
                        subprocess.run(  # nosec B603
                            reinstall_cmd,
                            capture_output=True,
                            text=True,
                            encoding="utf-8", errors="replace",
                            timeout=300,
                            env=env,
                            **subprocess_kwargs
                        )
                    except Exception:
                        pass  # nosec B110
                    # Retry verification after force-reinstall
                    try:
                        result2 = subprocess.run(  # nosec B603
                            cmd,
                            capture_output=True,
                            text=True,
                            encoding="utf-8", errors="replace",
                            timeout=pkg_timeout,
                            env=env,
                            **subprocess_kwargs
                        )
                        if result2.returncode == 0:
                            _log(
                                f"Package {package_name} fixed after force-reinstall",
                                Qgis.MessageLevel.Success
                            )
                            continue  # Move to next package
                    except Exception:
                        pass  # nosec B110
                    # Still broken after reinstall - check for AppLocker
                    detail_lower = full_error.lower()
                    applocker_markers = [
                        "applocker", "application control",
                        "blocked by your organization",
                    ]
                    if any(m in detail_lower for m in applocker_markers):
                        return False, (
                            f"Package {package_name} is blocked by AppLocker or "
                            "application control policy.\n\n"
                            "Ask your IT administrator to whitelist "
                            f"this folder:\n  {venv_dir}\n\n"
                            "Then restart QGIS and reinstall "
                            "dependencies."
                        )
                    return False, (
                        f"Package {package_name} is broken (antivirus may be "
                        f"interfering): {error_detail[:200]}"
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
                            f"     {vcpp_url}\n"
                            "  2. Add an antivirus exclusion for "
                            "the plugin folder\n"
                            "  3. Restart QGIS"
                        )
                    else:
                        hint = (
                            "\n\nPlease try:\n"
                            "  1. Install Visual C++ "
                            "Redistributable:\n"
                            f"     {vcpp_url}\n"
                            "  2. Click 'Reinstall dependencies' "
                            "in the plugin panel\n"
                            "  3. Restart QGIS"
                        )
                else:
                    hint = (
                        "\n\nPlease try:\n"
                        "  1. Click 'Reinstall dependencies' "
                        "in the plugin panel\n"
                        "  2. Restart QGIS"
                    )
                return False, f"Package {package_name} is broken: {error_detail[:200]}{hint}"

        except subprocess.TimeoutExpired:
            # Retry once - antivirus scanning .pyd files on first import
            # causes timeouts; the 2nd import is near-instant once cached
            _log(
                f"Verification of {package_name} timed out ({pkg_timeout}s), retrying...",
                Qgis.MessageLevel.Info
            )
            try:
                result = subprocess.run(  # nosec B603
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8", errors="replace",
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
                        f"Package {package_name} verification failed on retry: {error_detail}",
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
                            f"Package {package_name} is blocked by AppLocker "
                            "or application control policy.\n\n"
                            "Ask your IT administrator to "
                            "whitelist this folder:\n"
                            f"  {venv_dir}\n\nThen restart QGIS and "
                            "reinstall dependencies."
                        )
                    return False, f"Package {package_name} is broken: {error_detail[:200]}"
            except subprocess.TimeoutExpired:
                _log(
                    f"Verification of {package_name} timed out twice",
                    Qgis.MessageLevel.Warning
                )
                return False, (
                    f"Verification error: {package_name} "
                    "(timed out - antivirus may be blocking)"
                )
            except Exception as e:
                _log(
                    f"Failed to verify {package_name} on retry: {str(e)}",
                    Qgis.MessageLevel.Warning
                )
                return False, f"Verification error: {package_name}"

        except Exception as e:
            _log(
                f"Failed to verify {package_name}: {str(e)}",
                Qgis.MessageLevel.Warning
            )
            return False, f"Verification error: {package_name}"

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
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[bool, str]:
    """
    Complete installation: download Python standalone + download uv + create venv + install packages.

    Progress breakdown:
    - 0-10%:   Download Python standalone (~50MB)
    - 10-13%:  Download uv package installer (non-fatal if fails)
    - 13-18%:  Create virtual environment
    - 18-95%:  Install packages (~800MB)
    - 95-100%: Verify installation
    """
    # Cross-process install lock (try-once, never blocks the GUI). CACHE_DIR is
    # shared across QGIS instances/profiles, so a second window installing at
    # the same time would rmtree the venv this one is building. If another LIVE
    # process holds the lock, report it through the normal failure channel and
    # bail out untouched. Everything below (preflight, marker-triggered
    # cleanup, venv rebuild) then runs while we exclusively hold the lock.
    lock = InstallLock(INSTALL_LOCK_FILE)
    if not lock.acquire():
        busy = (
            "Another QGIS window is installing the AI components. "
            "Wait for it to finish, then try again."
        )
        _log(busy, Qgis.MessageLevel.Warning)
        return False, busy

    try:
        # Disk space pre-flight. Package-manager caches and build temp are now
        # contained under CACHE_DIR for the install subprocesses (see
        # _get_clean_env_for_venv), so this single measurement covers the whole
        # install on one volume: the extracted venv (torch dominates, ~3 GB) +
        # the retained download cache (compressed wheels, up to ~1.5 GB when
        # pip is used since it does not hardlink from cache) + transient
        # extraction temp (~0.5 GB). 5 GB covers that with a small margin.
        min_free_gb = 5.0
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            free_gb = shutil.disk_usage(CACHE_DIR).free / (1024 ** 3)
        except OSError:
            # Let the writability check inside the install flow report the real error
            free_gb = None
        if free_gb is not None and free_gb < min_free_gb:
            hint = (
                f"Not enough free disk space to install dependencies: "
                f"{free_gb:.1f} GB available at {CACHE_DIR}, "
                f"at least {min_free_gb:.0f} GB is required.\n\n"
                "Free up disk space, or set the AI_SEGMENTATION_CACHE_DIR "
                "environment variable to a directory on a larger drive, "
                "then restart QGIS."
            )
            _log(hint, Qgis.MessageLevel.Critical)
            return False, hint

        # A leftover marker means QGIS was killed mid-install: the venv may be
        # half-built yet pass superficial checks, so rebuild it from scratch.
        # Safe to rmtree here because we hold the lock, so no other live
        # process can be building this venv right now.
        if _install_marker_present():
            _log(
                "Previous installation was interrupted, "
                "recreating the virtual environment...",
                Qgis.MessageLevel.Warning)
            _cleanup_partial_venv(VENV_DIR)

        _write_install_marker()
        try:
            return _create_venv_and_install(progress_callback, cancel_check)
        finally:
            # Any normal return or exception here is reported to the user; the
            # marker only needs to survive a hard crash, where finally never runs.
            _clear_install_marker()
    finally:
        lock.release()


def _create_venv_and_install(
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[bool, str]:
    """Run the actual install flow (see create_venv_and_install)."""
    from .python_manager import (
        download_python_standalone,
        get_python_full_version,
        remove_standalone_python,
        standalone_python_exists,
        standalone_python_is_current,
    )

    # Early Python version check (Issue #148)

    # Log system info for debugging
    _log_system_info()

    # Early writability check on CACHE_DIR
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        test_file = os.path.join(CACHE_DIR, ".write_test")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_file)
    except OSError as e:
        hint = (
            f"Cannot write to install directory: {CACHE_DIR}\n"
            f"Error: {e}\n\n"
            "Set the AI_SEGMENTATION_CACHE_DIR environment variable "
            "to a writable directory, then restart QGIS."
        )
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
                _log(f"Failed to remove stale venv: {e}", Qgis.MessageLevel.Warning)

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
                            f"falling back to QGIS Python: {msg}",
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
                _log(f"uv download failed (non-fatal), using pip: {uv_msg}", Qgis.MessageLevel.Warning)
        except Exception as e:
            _uv_available = False
            _uv_path = None
            _log(f"uv download failed (non-fatal): {e}", Qgis.MessageLevel.Warning)
        if progress_callback:
            progress_callback(13, "uv: {}".format(
                "ready" if _uv_available else "unavailable, using pip"))

    if cancel_check and cancel_check():
        return False, "Installation cancelled"

    # Step 2: Create virtual environment if needed.
    # A venv left broken/incomplete by an interrupted run still has its python
    # executable, so venv_exists() returns True, but uv refuses it and the
    # install fails. Recreate it instead of installing into a dead venv. (#64)
    def venv_progress(percent, msg):
        # Map 0-100 to 13-18
        if progress_callback:
            progress_callback(13 + int(percent * 0.05), msg)

    if venv_exists() and _venv_is_functional():
        _log("Virtual environment already exists", Qgis.MessageLevel.Info)
        if progress_callback:
            progress_callback(18, "Virtual environment ready")
    else:
        if venv_exists():
            _log(
                "Existing virtual environment is broken or incomplete, "
                "recreating it...", Qgis.MessageLevel.Warning)
            _cleanup_partial_venv(VENV_DIR)

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


def _quick_check_packages(venv_dir: str = None) -> tuple[bool, str]:
    """
    Fast filesystem-based check that packages exist in the venv site-packages.

    Does NOT spawn subprocesses - safe to call from the main thread.
    Checks for known package directories/files in site-packages.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    site_packages = get_venv_site_packages(venv_dir)
    if not os.path.exists(site_packages):
        _log(f"Quick check: site-packages not found: {site_packages}",
             Qgis.MessageLevel.Warning)
        return False, "site-packages directory not found"

    # Expected import directories, derived from REQUIRED_PACKAGES so this
    # check can never drift from what install_dependencies installs
    # (a hardcoded copy of the list once kept demanding pandas after it was
    # dropped). setuptools is skipped: not an install health signal.
    package_markers = {
        name: name.replace("-", "_")
        for name, _spec in REQUIRED_PACKAGES
        if name != "setuptools"
    }

    for package_name, dir_name in package_markers.items():
        pkg_dir = os.path.join(site_packages, dir_name)
        if not os.path.exists(pkg_dir):
            _log(f"Quick check: {package_name} not found at {pkg_dir}", Qgis.MessageLevel.Warning)
            return False, f"Package {package_name} not found"

    _log(f"Quick check: all packages found in {site_packages}",
         Qgis.MessageLevel.Info)
    return True, "All packages found"


def get_venv_status() -> tuple[bool, str]:
    """Get the status of the complete installation (Python standalone + venv)."""
    from .python_manager import get_python_full_version, standalone_python_exists

    # Check for old libs/ installation
    if os.path.exists(LIBS_DIR):
        _log(f"get_venv_status: old libs/ detected at {LIBS_DIR}",
             Qgis.MessageLevel.Warning)
        return False, "Old installation detected. Migration required."

    # A crash mid-install leaves the marker behind; the venv may be
    # half-built even if package directories look present.
    if _install_marker_present():
        _log("get_venv_status: previous installation was interrupted",
             Qgis.MessageLevel.Warning)
        return False, "Previous installation was interrupted"

    # Check Python standalone (skip where a matching system Python is used
    # instead, e.g. NixOS or other Linux where the standalone build cannot
    # run: musl/old glibc, sandboxed Flatpak/Snap)
    if not standalone_python_exists():
        python3 = shutil.which("python3") if sys.platform == "linux" else None
        if not (python3 and _system_python_matches_target(python3)):
            _log("get_venv_status: standalone Python not found", Qgis.MessageLevel.Info)
            return False, "Dependencies not installed"

    # Check venv
    if not venv_exists():
        _log(f"get_venv_status: venv not found at {VENV_DIR}",
             Qgis.MessageLevel.Info)
        return False, "Virtual environment not configured"

    # The venv launcher needs the base Python recorded in pyvenv.cfg.
    # If that install was deleted or the path is corrupt, packages look
    # complete but nothing can run: route to Install, which recreates it.
    base_ok, base_msg = _venv_base_python_ok()
    if not base_ok:
        _log(f"get_venv_status: {base_msg}", Qgis.MessageLevel.Warning)
        return False, "Python runtime is damaged. Reinstall required."

    # Quick filesystem check (no subprocess, safe for main thread)
    is_present, msg = _quick_check_packages()
    if is_present:
        # Check if dependency specs have changed since last install
        stored_hash = _read_deps_hash()
        current_hash = _compute_deps_hash()
        if stored_hash is not None and stored_hash != current_hash:
            _log(
                "get_venv_status: deps hash mismatch "
                f"(stored={stored_hash}, current={current_hash})",
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
                    probe = subprocess.run(  # nosec B603
                        [python_path, "-c", "import torch"],
                        capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=60,
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
                        f"get_venv_status: torch probe failed: {exc}",
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
        return True, f"Ready (Python {python_version})"
    _log(f"get_venv_status: quick check failed: {msg}",
         Qgis.MessageLevel.Warning)
    return False, f"Virtual environment incomplete: {msg}"
