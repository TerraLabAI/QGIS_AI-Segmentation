







from __future__ import annotations

import os
import time



_TEMP_PREFIXES = ("run_", "pip_", "pip_constraints_")



_TMP_LOCK_PREFIX = "uv-setuptools-"
_TMP_COMPILE_CACHE_PREFIX = "torchinductor_"


_INSTALL_LOCK_NAME = "install.lock"



_MAX_AGE_S = 24 * 60 * 60


def _max_age_s() -> float:






    from .server_dials import dial_in_range

    return dial_in_range(
        "tuning.install.temp_sweep_max_age_s", _MAX_AGE_S,
        4 * 60 * 60, 7 * 24 * 60 * 60)


def sweep_stale_install_temp_files(cache_dir: str) -> int:





    if os.path.lexists(os.path.join(cache_dir, _INSTALL_LOCK_NAME)):
        return 0
    try:
        names = os.listdir(cache_dir)
    except OSError:
        return 0
    cutoff = time.time() - _max_age_s()
    removed = 0
    for name in names:
        if not name.endswith(".txt") or not name.startswith(_TEMP_PREFIXES):
            continue
        path = os.path.join(cache_dir, name)
        try:
            if not os.path.isfile(path) or os.path.getmtime(path) > cutoff:
                continue
            os.unlink(path)
            removed += 1
        except OSError:
            continue
    return removed + _sweep_containment_tmp(cache_dir, cutoff)


def _newest_mtime(path: str) -> float:

    newest = os.path.getmtime(path)
    scanned = 0
    for root, dirs, files in os.walk(path):
        scanned += len(dirs) + len(files)
        if scanned > 10_000:
            return time.time()
        for name in dirs + files:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                continue
    return newest


def _sweep_containment_tmp(cache_dir: str, cutoff: float) -> int:






    tmp_dir = os.path.join(cache_dir, "tmp")
    if os.path.lexists(os.path.join(cache_dir, _INSTALL_LOCK_NAME)):
        return 0
    try:
        names = os.listdir(tmp_dir)
    except OSError:
        return 0
    from .cache_paths import remove_tree_quietly

    removed = 0
    for name in names:
        path = os.path.join(tmp_dir, name)
        try:
            if name.startswith(_TMP_LOCK_PREFIX) and name.endswith(".lock"):
                if os.path.isfile(path) and os.path.getmtime(path) <= cutoff:
                    os.unlink(path)
                    removed += 1
            elif name.startswith(_TMP_COMPILE_CACHE_PREFIX):
                if (os.path.isdir(path) and not os.path.islink(path)
                        and _newest_mtime(path) <= cutoff
                        and remove_tree_quietly(path)):
                    removed += 1
        except OSError:
            continue
    return removed
