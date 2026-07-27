"""Keeps the on-disk image cache under a size budget, by evicting the oldest.

``template_demo_loader`` writes library images (template demos and archived run
artifacts) to a per-user cache dir and never deleted anything, so the dir grew
forever on the user's disk. This module is the other half: one deferred sweep
per QGIS session walks that dir and unlinks least-recently-used files until the
total fits ``IMAGE_CACHE_BUDGET_BYTES``.

Recency is ``st_mtime``: writing an entry sets it, and ``touch_for_lru`` bumps
it on a cache hit so "least recently used" means used, not downloaded.

Everything here is fail-safe. A cache dir that cannot be walked, a file that
cannot be stat'ed, a read-only disk: each frees less than it hoped and returns,
never raises into the caller. Production-safe logging: local cache paths and
sizes only.
"""
from __future__ import annotations

import os
from pathlib import Path

from qgis.core import Qgis

from ..core.logging_utils import log

# Disk budget for the whole image cache. An archived run tile weighs about
# 1.5 MB and a full template demo set about 5 MB, so 200 MB holds well over a
# hundred past runs plus every demo: enough that a returning user never
# refetches, small enough to stay unnoticed next to the tile and WMS caches
# QGIS itself keeps in the same place. Past it, the least recently used go.
IMAGE_CACHE_BUDGET_BYTES = 200 * 1024 * 1024

# One sweep per QGIS session. The cache only grows by what this session
# downloads, so re-walking it on every dialog open would buy nothing.
_SWEEP_DONE = False


def touch_for_lru(path: Path) -> None:
    """Record a cache hit, so eviction can tell live entries from forgotten ones.

    Only for entries whose caller declared them immutable: on an entry that can
    expire, ``st_mtime`` doubles as the TTL stamp and refreshing it on every
    read would make it immortal. A read-only cache dir must never break image
    loading, hence the swallow.
    """
    try:
        os.utime(path, None)
    except OSError:
        pass  # nosec B110


def _is_inside(path: Path, root: Path) -> bool:
    """True only for a path that physically sits under ``root``.

    Eviction deletes files, so this is deliberately paranoid: a symlink is
    refused outright rather than followed, and the comparison runs on resolved,
    normcase'd strings so a case or drive-letter difference on Windows cannot
    read as "outside" (or, worse, let something outside read as inside).
    """
    try:
        if path.is_symlink():
            return False
        resolved = os.path.normcase(str(path.resolve()))
    except OSError:
        return False
    return resolved.startswith(os.path.normcase(str(root)) + os.sep)


def _collect_entries(root: Path) -> tuple[list[tuple[float, int, Path]], int]:
    """Return (mtime, size, path) for every regular file under root, and the total."""
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
                      budget_bytes: int = IMAGE_CACHE_BUDGET_BYTES) -> int:
    """Evict least-recently-used images under ``root`` until they fit the budget.

    ``expected_dir_name`` is the cache dir's own folder name, checked against
    the resolved root before a single unlink: this function must never be able
    to delete anything outside the plugin's own cache. Returns the bytes freed.
    """
    try:
        root = root.resolve()
    except OSError:
        return 0
    if not expected_dir_name or root.name != expected_dir_name or not root.is_dir():
        return 0
    entries, total = _collect_entries(root)
    if total <= budget_bytes:
        return 0

    entries.sort(key=lambda entry: entry[0])  # least recently used first
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
            folder.rmdir()  # only succeeds once the folder is empty
        except OSError:
            continue
    log(f"Image cache swept: freed {freed // 1024} KiB of {total // 1024} KiB",
        Qgis.MessageLevel.Info)
    return freed


def sweep_image_cache_once(root: Path, expected_dir_name: str) -> None:
    """Run the sweep at most once per QGIS session, and never raise."""
    global _SWEEP_DONE
    if _SWEEP_DONE:
        return
    _SWEEP_DONE = True
    try:
        sweep_image_cache(root, expected_dir_name)
    except Exception as err:  # noqa: BLE001
        log(f"Image cache sweep failed: {err}", Qgis.MessageLevel.Warning)
