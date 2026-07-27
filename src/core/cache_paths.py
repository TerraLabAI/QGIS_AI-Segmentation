










from __future__ import annotations

import os
import shutil
import stat
import sys
import time


def _cache_dir_override() -> str:







    raw = os.environ.get("AI_SEGMENTATION_CACHE_DIR") or ""
    if os.name == "nt":
        raw = os.path.expandvars(raw.strip().strip('"').strip())
    return raw







PLUGIN_CACHE_DIR = os.path.normpath(
    _cache_dir_override() or os.path.expanduser("~/.qgis_ai_segmentation")
)



_WIN_SHARING_ERRORS = (32, 33)


def _retry_tree_entry(func, target, exc) -> None:







    if func not in (os.unlink, os.remove, os.rmdir):
        return
    for attempt in range(3):
        try:
            os.chmod(target, os.stat(target).st_mode | stat.S_IWRITE)
        except OSError:
            pass  # nosec B110
        try:
            func(target)
            return
        except FileNotFoundError:
            return
        except OSError as err:
            exc = err
        if getattr(exc, "winerror", None) not in _WIN_SHARING_ERRORS:
            return
        time.sleep(0.2 * (attempt + 1))


def remove_tree_quietly(path: str) -> bool:





    if sys.platform != "win32":
        shutil.rmtree(path, ignore_errors=True)
        return not os.path.lexists(path)
    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_retry_tree_entry)
        else:
            shutil.rmtree(
                path, onerror=lambda func, target, info: _retry_tree_entry(func, target, info[1]))
    except OSError:
        pass  # nosec B110
    return not os.path.lexists(path)


def plugin_cache_tmp_dir() -> str | None:






    tmp_dir = os.path.join(PLUGIN_CACHE_DIR, "tmp")
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir
    except OSError:
        return None
