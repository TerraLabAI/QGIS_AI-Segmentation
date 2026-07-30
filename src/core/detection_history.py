"""Local history of committed Automatic detection runs (library Recent tab).

One JSON file plus small PNG thumbnails under the plugin's local cache dir
(``~/.qgis_ai_segmentation/detection_history/``, same root as the env/weights
caches). Each entry remembers enough to bring the user back to a run: the
prompt, when it ran, the zone extent + CRS authid, the exported layer name,
the object count, and a thumbnail rendered at Finish.

Strictly LOCAL-ONLY state: nothing here is ever sent anywhere (no telemetry,
no network). It is the richer sibling of ``presets/segment_history`` (which
keeps only the prompt token in QSettings for one-click reuse).

One store per account: a signed-in session reads and writes its own
subdirectory, named after the account fingerprint, while the signed-out state
keeps the shared root. Nothing is copied between them, so signing in to another
account on the same machine never surfaces the previous account's runs.

Fail-safe by design: reads return [] on any problem, and the store is capped
at :func:`history_max_entries` with orphaned thumbnails garbage-collected on
write.
"""
from __future__ import annotations

import json
import os
import time
import uuid

_HISTORY_DIR = os.path.join(
    os.environ.get("AI_SEGMENTATION_CACHE_DIR") or os.path.expanduser("~/.qgis_ai_segmentation"),
    "detection_history",
)
_HISTORY_FILE = "history.json"
_ACCOUNT_DIR_PREFIX = "account_"

# Shipped cap on the runs kept locally, and the fallback the getter below
# returns whenever the server says nothing usable. This is the user's own
# segmentation history and we keep all of it (there is no in-app delete): the
# server is the true unbounded archive for signed-in users, and this local
# store is the offline feed for the library's Recent tab. Each entry costs a
# 256px PNG thumb plus a tiny JSON row, so the cap is what keeps the store
# bounded on disk. Oldest beyond it roll off.
MAX_ENTRIES = 500


def history_max_entries() -> int:
    """How many committed runs the local store keeps.

    Bounded on both sides: below the floor the Recent tab stops being a
    history, above the ceiling the thumbnails outgrow what a cache dir should
    hold. Cache-only and never raises, so it stays safe in this fail-safe
    module and on a machine with no configuration at all.
    """
    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("library.history_max_entries", MAX_ENTRIES, 20, 5000))
    except Exception:  # noqa: BLE001 -- the cap is best-effort  # nosec B110
        return MAX_ENTRIES


def account_history_dir() -> str:
    """The store directory for the account signed in right now (not created).

    The lookup goes through the presets package, which is QGIS-bound, so a
    context without QGIS falls back to the signed-out store instead of raising:
    this module has to stay importable and fail-safe on its own.
    """
    try:
        from .presets.run_history_cache import account_fingerprint

        fingerprint = account_fingerprint()
    except Exception:
        return _HISTORY_DIR
    if not fingerprint:
        return _HISTORY_DIR
    return os.path.join(_HISTORY_DIR, f"{_ACCOUNT_DIR_PREFIX}{fingerprint}")


def history_dir() -> str:
    """The store directory for the current account (created on demand)."""
    path = account_history_dir()
    os.makedirs(path, exist_ok=True)
    return path


def _history_path() -> str:
    return os.path.join(history_dir(), _HISTORY_FILE)


def get_entries() -> list[dict]:
    """Stored runs, newest first. [] on any read problem (fail-safe).

    Each entry: ``{id, prompt, ts, layer_name, objects, crs, extent?, thumb?}``
    where ``extent`` is ``[xmin, ymin, xmax, ymax]`` in the CRS named by the
    ``crs`` authid, and ``thumb`` a PNG filename inside :func:`history_dir`.
    """
    try:
        with open(_history_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict)]


def _write_entries(entries: list[dict]) -> None:
    """Atomic JSON write: temp file + os.replace, so a crash mid-write can
    never corrupt the existing store."""
    path = _history_path()
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False)
    os.replace(tmp, path)


def new_thumb_filename() -> str:
    """A fresh unique thumbnail filename to save inside :func:`history_dir`."""
    return f"thumb_{uuid.uuid4().hex[:16]}.png"


def thumb_abspath(entry: dict) -> str | None:
    """Absolute path of an entry's thumbnail, or None when absent/missing.

    The stored name must be a bare filename (defense against a hand-edited
    store pointing outside the history dir)."""
    name = str(entry.get("thumb") or "")
    if not name or os.path.basename(name) != name:
        return None
    path = os.path.join(account_history_dir(), name)
    return path if os.path.isfile(path) else None


def add_entry(
    prompt: str,
    layer_name: str,
    objects: int,
    extent: tuple[float, float, float, float] | None,
    crs_authid: str,
    thumb: str | None = None,
) -> None:
    """Prepend one committed run, cap the store, GC orphaned thumbnails.

    ``extent`` is (xmin, ymin, xmax, ymax) in the CRS named by ``crs_authid``;
    ``thumb`` a filename already saved inside :func:`history_dir` (or None).
    """
    entries = get_entries()
    entry: dict = {
        "id": uuid.uuid4().hex[:16],
        "prompt": (prompt or "").strip(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "layer_name": layer_name or "",
        "objects": int(objects),
        "crs": crs_authid or "",
    }
    if extent is not None and len(extent) == 4:
        entry["extent"] = [float(v) for v in extent]
    if thumb:
        entry["thumb"] = os.path.basename(thumb)
    entries.insert(0, entry)
    entries = entries[:history_max_entries()]
    _write_entries(entries)
    _gc_thumbs(entries)


def clear_detection_history() -> None:
    """Drop the current account's stored runs and thumbnails.

    Other accounts' stores sit in their own subdirectories and are left alone.
    """
    _write_entries([])
    _gc_thumbs([])


def _gc_thumbs(entries: list[dict]) -> None:
    """Best-effort delete of thumbnail files no kept entry references.

    Scoped to the current account's directory, and to thumbnail file names, so
    the per-account subdirectories under the signed-out root are never touched.
    """
    # normcase both sides: this decides what gets deleted, and Windows file
    # names are case-insensitive, so a plain `not in` would read a live
    # thumbnail spelled with different case as an orphan and remove it.
    keep = set()
    for entry in entries:
        thumb = entry.get("thumb")
        # isinstance, because this set decides what gets deleted and the
        # entries come from a JSON file on disk: a non-string there would
        # raise out of normcase, and the caller catches nothing.
        if isinstance(thumb, str) and thumb:
            keep.add(os.path.normcase(thumb))
    directory = account_history_dir()
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        if name.startswith("thumb_") and name.endswith(".png") and os.path.normcase(name) not in keep:
            try:
                os.remove(os.path.join(directory, name))
            except OSError:
                pass  # nosec B110 -- GC is best-effort; retried on next write
