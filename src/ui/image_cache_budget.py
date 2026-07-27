















from __future__ import annotations

import os
from pathlib import Path

from qgis.core import Qgis

from ..core.logging_utils import log






IMAGE_CACHE_BUDGET_BYTES = 200 * 1024 * 1024

_BYTES_PER_MB = 1024 * 1024



_state = {"sweep_done": False}


def image_cache_budget_bytes() -> int:







    try:
        from ..core.server_dials import dial_in_range

        shipped_mb = IMAGE_CACHE_BUDGET_BYTES // _BYTES_PER_MB
        return int(dial_in_range("ui.image_cache_mb", shipped_mb, 16, 4096)) * _BYTES_PER_MB
    except Exception:  # noqa: BLE001  # nosec B110
        return IMAGE_CACHE_BUDGET_BYTES


def touch_for_lru(path: Path) -> None:







    try:
        os.utime(path, None)
    except OSError:
        pass  # nosec B110


def _is_inside(path: Path, root: Path) -> bool:







    try:
        if path.is_symlink():
            return False
        resolved = os.path.normcase(str(path.resolve()))
    except OSError:
        return False
    return resolved.startswith(os.path.normcase(str(root)) + os.sep)


def _collect_entries(root: Path) -> tuple[list[tuple[float, int, Path]], int]:

    try:
        walked = list(root.rglob("*"))
    except OSError:
        return [], 0
    entries: list[tuple[float, int, Path]] = []
    total = 0
    for path in walked:
        try:
            if not path.is_file():
                continue
            stat = path.stat()
        except OSError:
            continue
        entries.append((stat.st_mtime, stat.st_size, path))
        total += stat.st_size
    return entries, total


def sweep_image_cache(root: Path, expected_dir_name: str,
                      budget_bytes: int | None = None) -> int:









    if budget_bytes is None:
        budget_bytes = image_cache_budget_bytes()
    try:
        root = root.resolve()
    except OSError:
        return 0
    if not expected_dir_name or root.name != expected_dir_name or not root.is_dir():
        return 0
    entries, total = _collect_entries(root)
    if total <= budget_bytes:
        return 0

    entries.sort(key=lambda entry: entry[0])
    freed = 0
    emptied: set[Path] = set()
    for _mtime, size, path in entries:
        if (total - freed) <= budget_bytes:
            break
        if not _is_inside(path, root):
            continue
        try:
            path.unlink()
        except OSError:
            continue
        freed += size
        emptied.add(path.parent)
    for folder in emptied:
        if folder == root or not _is_inside(folder, root):
            continue
        try:
            folder.rmdir()
        except OSError:
            continue
    log(f"Image cache swept: freed {freed // 1024} KiB of {total // 1024} KiB",
        Qgis.MessageLevel.Info)
    return freed


def sweep_image_cache_once(root: Path, expected_dir_name: str) -> None:

    if _state["sweep_done"]:
        return
    _state["sweep_done"] = True
    try:
        sweep_image_cache(root, expected_dir_name)
    except Exception as err:  # noqa: BLE001
        log(f"Image cache sweep failed: {err}", Qgis.MessageLevel.Warning)
