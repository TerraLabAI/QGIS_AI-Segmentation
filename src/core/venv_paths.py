








from __future__ import annotations

import glob
import os
import shutil
import sys
import tempfile
import threading
from typing import Callable

from qgis.core import Qgis

from .cache_paths import PLUGIN_CACHE_DIR, remove_tree_quietly
from .logging_utils import log as _log

PLUGIN_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC_DIR = PLUGIN_ROOT_DIR

PYTHON_VERSION = f"py{sys.version_info.major}.{sys.version_info.minor}"

VENV_DIR = os.path.join(PLUGIN_CACHE_DIR, f"venv_{PYTHON_VERSION}")

LIBS_DIR = os.path.join(PLUGIN_ROOT_DIR, "libs")

DEPS_HASH_FILE = os.path.join(VENV_DIR, "deps_hash.txt")




INSTALL_MARKER_FILE = os.path.join(PLUGIN_CACHE_DIR, "install_in_progress")






INSTALL_LOCK_FILE = os.path.join(PLUGIN_CACHE_DIR, "install.lock")




INSTALL_LOG_FILE = os.path.join(PLUGIN_CACHE_DIR, "install.log")

INSTALL_LOG_MAX_BYTES = 2 * 1024 * 1024

INSTALL_LOG_STREAM_TAIL = 200_000




PENDING_DELETE_DIR = os.path.join(PLUGIN_CACHE_DIR, ".pending_delete")


def get_venv_dir() -> str:
    return VENV_DIR


def get_venv_site_packages(venv_dir: str = None) -> str:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        return os.path.join(venv_dir, "Lib", "site-packages")

    lib_dir = os.path.join(venv_dir, "lib")
    if os.path.exists(lib_dir):
        for entry in os.listdir(lib_dir):
            if entry.startswith("python") and os.path.isdir(os.path.join(lib_dir, entry)):
                site_packages = os.path.join(lib_dir, entry, "site-packages")
                if os.path.exists(site_packages):
                    return site_packages


    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return os.path.join(venv_dir, "lib", py_version, "site-packages")


def site_packages_python_version(site_packages: str) -> str | None:





    parent = os.path.basename(os.path.dirname(site_packages))
    if parent.startswith("python") and "." in parent:
        return parent
    return None


def site_packages_loadable_in_process(site_packages: str) -> bool:








    built_for = site_packages_python_version(site_packages)
    if built_for is None:
        return True
    running = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return built_for == running


def _add_windows_dll_directories(site_packages: str) -> None:











    dll_dirs = [
        os.path.join(site_packages, "torch", "lib"),
        os.path.join(site_packages, "torch", "bin"),
        os.path.join(site_packages, "torchvision"),
    ]
    for dll_dir in dll_dirs:
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except OSError as exc:
                _log(f"add_dll_directory({dll_dir}) failed: {exc}", Qgis.MessageLevel.Warning)


_rasterio_scope = {"done": False}


def _scope_rasterio_data_paths() -> None:










    if _rasterio_scope["done"]:
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
        _rasterio_scope["done"] = True
    except Exception as e:  # nosec B110
        _log(f"Failed to scope rasterio data paths: {e}", Qgis.MessageLevel.Warning)


def _repair_poisoned_environment() -> None:






    token = os.path.normcase(PLUGIN_CACHE_DIR)
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











_ensure_packages_lock = threading.RLock()


def ensure_venv_packages_available():
    with _ensure_packages_lock:
        return _ensure_venv_packages_available_locked()


def _ensure_venv_packages_available_locked():


    _repair_poisoned_environment()

    if not venv_exists():
        _log("Venv does not exist, cannot load packages", Qgis.MessageLevel.Warning)
        return False

    site_packages = get_venv_site_packages()
    if not os.path.exists(site_packages):

        venv_dir = get_venv_dir()
        lib_dir = os.path.join(venv_dir, "lib")
        if os.path.exists(lib_dir):
            contents = os.listdir(lib_dir)
            _log(f"Venv lib/ contents: {contents}", Qgis.MessageLevel.Warning)
        _log(f"Venv site-packages not found: {site_packages}", Qgis.MessageLevel.Warning)
        return False




    if not site_packages_loadable_in_process(site_packages):
        _log(
            f"Packages were installed for {site_packages_python_version(site_packages)} "
            f"but QGIS runs python{sys.version_info.major}.{sys.version_info.minor}. "
            "Reinstall the dependencies from the plugin panel to rebuild them "
            "for this QGIS.",
            Qgis.MessageLevel.Warning)
        return False




    _sp_key = os.path.normcase(site_packages)
    if all(os.path.normcase(p) != _sp_key for p in sys.path):
        sys.path.append(site_packages)
        _log(f"Added venv site-packages to sys.path: {site_packages}", Qgis.MessageLevel.Info)



    if sys.platform == "win32":
        _add_windows_dll_directories(site_packages)




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




















    needs_numpy_fix = False



    numpy_fix_applies = Qgis.QGIS_VERSION_INT < 33000
    try:
        if "numpy" in sys.modules:
            old_np = sys.modules["numpy"]
            old_version = getattr(old_np, "__version__", "0.0.0")
        else:
            old_version = "not_loaded"

        if old_version == "not_loaded":




            if sys.path and os.path.normcase(sys.path[0]) != _sp_key:
                if site_packages in sys.path:
                    sys.path.remove(site_packages)
                sys.path.insert(0, site_packages)
        elif numpy_fix_applies:
            parts = old_version.split(".")[:3]
            vn = [int(x) for x in parts] + [0] * (3 - len(parts))
            np_old = (vn[0] < 1) or (vn[0] == 1 and vn[1] < 22)
            np_old = np_old or (vn[0] == 1 and vn[1] == 22 and vn[2] < 4)
            needs_numpy_fix = np_old
    except Exception:
        needs_numpy_fix = False

    if not needs_numpy_fix:


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


        mods_to_clear = [
            k for k in list(sys.modules.keys())
            if k.startswith("numpy") or k.startswith("pandas")
        ]
        for mod in mods_to_clear:
            del sys.modules[mod]



        for p in sys.path[:]:



            if os.path.normcase(p) == os.path.normcase(site_packages):
                continue
            np_init = os.path.join(p, "numpy", "__init__.py")
            if os.path.exists(np_init):
                removed_paths.append(p)
                sys.path.remove(p)


        importlib.invalidate_caches()


        import numpy as new_numpy  # noqa: E402


        for p in removed_paths:
            if p not in sys.path:
                sys.path.append(p)
        removed_paths = []


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

        for p in removed_paths:
            if p not in sys.path:
                sys.path.append(p)


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


def venv_exists(venv_dir: str = None) -> bool:
    if venv_dir is None:
        venv_dir = VENV_DIR

    python_path = get_venv_python_path(venv_dir)
    return os.path.exists(python_path)




_VENV_REMOVAL_PREFIX = "_removing_venv_"


def _remove_dead_venv(venv_dir: str) -> bool:







    parent = os.path.dirname(venv_dir)
    staged = os.path.join(
        parent, f"{_VENV_REMOVAL_PREFIX}{os.path.basename(venv_dir)}.{os.getpid()}")
    try:
        os.replace(_win_extended_path(venv_dir), _win_extended_path(staged))
    except OSError as e:
        _log(f"Left {venv_dir} in place, it is open elsewhere: {e}",
             Qgis.MessageLevel.Warning)
        return False
    remove_tree_quietly(_win_extended_path(staged))
    return True


def _sweep_venv_removals(scan_dir: str) -> None:

    pattern = os.path.join(_win_extended_path(scan_dir), f"{_VENV_REMOVAL_PREFIX}*")
    for leftover in glob.glob(pattern):
        remove_tree_quietly(leftover)


def cleanup_old_venv_directories() -> list[str]:









    current_cmp = os.path.normcase(f"venv_{PYTHON_VERSION}")
    venv_prefix = os.path.normcase("venv_py")
    removed = []

    for scan_dir in [PLUGIN_CACHE_DIR, SRC_DIR]:
        try:
            if not os.path.exists(scan_dir):
                continue
            _sweep_venv_removals(scan_dir)
            for entry in os.listdir(scan_dir):
                entry_cmp = os.path.normcase(entry)
                if not entry_cmp.startswith(venv_prefix) or entry_cmp == current_cmp:
                    continue
                old_path = os.path.join(scan_dir, entry)
                if not os.path.isdir(old_path):
                    continue
                try:
                    if venv_exists(old_path):
                        _log(f"Kept {old_path}: another QGIS may be using it.",
                             Qgis.MessageLevel.Info)
                        continue
                    if _remove_dead_venv(old_path):
                        _log(f"Cleaned up old venv: {old_path}",
                             Qgis.MessageLevel.Info)
                        removed.append(old_path)
                except Exception as e:
                    _log(f"Failed to remove old venv {old_path}: {e}", Qgis.MessageLevel.Warning)
        except Exception as e:
            _log(f"Error scanning for old venvs in {scan_dir}: {e}", Qgis.MessageLevel.Warning)

    return removed


def cleanup_old_libs() -> bool:
    if not os.path.exists(LIBS_DIR):
        return False

    _log("Detected old 'libs/' installation. Cleaning up...", Qgis.MessageLevel.Info)

    try:


        shutil.rmtree(_win_extended_path(LIBS_DIR))
        _log("Old libs/ directory removed successfully", Qgis.MessageLevel.Success)
        return True
    except Exception as e:
        _log(f"Failed to remove libs/: {e}. Please delete manually.", Qgis.MessageLevel.Warning)
        return False


def _venv_loaded_in_process(venv_dir: str) -> bool:





    if sys.platform != "win32":
        return False
    root = os.path.normcase(os.path.abspath(venv_dir)) + os.sep
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and os.path.normcase(os.path.abspath(path)).startswith(root):
            return True
    return False


def _cleanup_partial_venv(venv_dir: str) -> bool:










    if not os.path.exists(venv_dir):
        return True
    if _venv_loaded_in_process(venv_dir):
        if _remove_dead_venv(venv_dir):
            _log(f"Cleaned up partial venv: {venv_dir}", Qgis.MessageLevel.Info)
            return True
        _log(f"Left the venv in place, this QGIS has libraries from it loaded. "
             f"Restart QGIS to rebuild it: {venv_dir}", Qgis.MessageLevel.Warning)
        return False
    if remove_tree_quietly(_win_extended_path(venv_dir)):
        _log(f"Cleaned up partial venv: {venv_dir}", Qgis.MessageLevel.Info)
        return True
    _log(f"Could not fully remove the partial venv, a file in it is still "
         f"open in another program: {venv_dir}", Qgis.MessageLevel.Warning)
    return False


def _win_path_api(func_name: str, path: str) -> str | None:







    try:
        import ctypes
        api = getattr(ctypes.windll.kernel32, func_name)
        size = 512
        for _attempt in range(2):
            buf = ctypes.create_unicode_buffer(size)
            ret = api(path, buf, size)
            if not ret:
                return None
            if ret < size:
                return buf.value
            size = ret + 1
    except Exception:
        return None
    return None


def _win_short_path(path: str) -> str:







    if sys.platform != "win32" or " " not in path:
        return path
    result = _win_path_api("GetShortPathNameW", path)
    if result and " " not in result:
        return result
    _log("Short path unavailable for a path containing a space; "
         "8.3 names may be disabled on this volume.",
         Qgis.MessageLevel.Warning)
    return path


def _win_extended_path(path: str) -> str:








    if sys.platform != "win32" or not path:
        return path
    if path.startswith("\\\\?\\"):
        return path
    abs_path = os.path.abspath(path)
    if abs_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")
    return "\\\\?\\" + abs_path


def _win_long_path(path: str) -> str:






    if sys.platform != "win32" or "~" not in path:
        return path
    return _win_path_api("GetLongPathNameW", path) or path


def _sweep_pending_delete() -> None:





    if not os.path.isdir(PENDING_DELETE_DIR):
        return
    remove_tree_quietly(_win_extended_path(PENDING_DELETE_DIR))


def _defer_cache_entry(path: str) -> bool:






    try:
        os.makedirs(PENDING_DELETE_DIR, exist_ok=True)
        holding = tempfile.mkdtemp(prefix="stale_", dir=PENDING_DELETE_DIR)
    except OSError as e:
        _log(f"Could not stage {path} for deletion: {e}", Qgis.MessageLevel.Warning)
        return False
    long_path = _win_extended_path(path)
    if os.path.isdir(path) and not os.path.islink(path):
        moved = 0
        for root, _dirs, files in os.walk(long_path):
            for name in files:
                try:
                    os.replace(os.path.join(root, name),
                               os.path.join(holding, f"{moved}_{name}"))
                    moved += 1
                except OSError:
                    pass  # nosec B110
        remove_tree_quietly(long_path)
    else:
        try:
            os.replace(long_path, os.path.join(holding, os.path.basename(path)))
        except OSError:
            pass  # nosec B110
    return not os.path.exists(long_path)


def purge_cache_dir(keep_install_lock: bool = True,
                    cancel_check: Callable[[], bool] | None = None) -> bool:



























    from .local_model_cache import invalidate as _forget_local_model
    _forget_local_model()
    if not os.path.isdir(PLUGIN_CACHE_DIR):
        return True
    _sweep_pending_delete()
    keep = {os.path.normcase(os.path.abspath(PENDING_DELETE_DIR))}
    if keep_install_lock:
        keep.add(os.path.normcase(os.path.abspath(INSTALL_LOCK_FILE)))

    def _deletable() -> list[str] | None:

        try:
            names = os.listdir(PLUGIN_CACHE_DIR)
        except OSError:
            return None
        return [
            name for name in names
            if os.path.normcase(os.path.abspath(os.path.join(PLUGIN_CACHE_DIR, name))) not in keep
        ]

    for name in _deletable() or []:
        if cancel_check is not None and cancel_check():
            return False
        path = os.path.join(PLUGIN_CACHE_DIR, name)
        long_path = _win_extended_path(path)
        if os.path.isdir(path) and not os.path.islink(path):
            remove_tree_quietly(long_path)
            continue
        try:
            os.unlink(long_path)
        except OSError:
            pass  # nosec B110
    leftovers = _deletable()
    if leftovers:
        leftovers = [
            name for name in leftovers
            if not _defer_cache_entry(os.path.join(PLUGIN_CACHE_DIR, name))
        ]
        if not leftovers:
            _log(
                "Some AI data is loaded in this QGIS session and was moved "
                "aside; it is removed from disk the next time QGIS starts.",
                Qgis.MessageLevel.Info
            )
    return leftovers == []


def _clear_installer_caches(include_tmp: bool = False) -> None:











    names = ("uv_cache", "pip_cache", "tmp") if include_tmp else ("uv_cache", "pip_cache")
    freed_any = False
    for name in names:
        path = os.path.join(PLUGIN_CACHE_DIR, name)
        if not os.path.isdir(path):
            continue
        remove_tree_quietly(_win_extended_path(path))
        freed_any = True
    if freed_any:
        _log("Cleared the installer wheel caches", Qgis.MessageLevel.Info)
