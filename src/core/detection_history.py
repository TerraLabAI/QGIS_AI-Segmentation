"""Local history of committed Automatic detection runs (library Recent tab).

One JSON file plus small PNG thumbnails under the plugin's local cache dir
(``~/.qgis_ai_segmentation/detection_history/``, same root as the env/weights
caches). Each entry remembers enough to bring the user back to a run: the
prompt, when it ran, the zone extent + CRS authid, the exported layer name,
the object count, and a thumbnail rendered at Finish.

Strictly LOCAL-ONLY state: nothing here is ever sent anywhere (no telemetry,
no network). It is the richer sibling of ``presets/segment_history`` (which
keeps only the prompt token in QSettings for one-click reuse).

Fail-safe by design: reads return [] on any problem, and the store is capped
at :data:`MAX_ENTRIES` with orphaned thumbnails garbage-collected on write.
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

# Last N runs kept locally. This is the user's own segmentation history and we
# keep all of it (there is no in-app delete): the server is the true unbounded
# archive for signed-in users, and this local store is the offline / pre-deploy
# feed for the library's Recent tab. 500 keeps every practical run while staying
# safely bounded on disk (a 256px PNG thumb + a tiny JSON row per entry, so a
# full store is a few tens of MB at most). Oldest beyond the cap roll off.
MAX_ENTRIES = 500


def history_dir() -> str:
    """The store directory (created on demand)."""
    os.makedirs(_HISTORY_DIR, exist_ok=True)
    return _HISTORY_DIR


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
    tmp = "{}.tmp".format(path)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False)
    os.replace(tmp, path)


def new_thumb_filename() -> str:
    """A fresh unique thumbnail filename to save inside :func:`history_dir`."""
    return "thumb_{}.png".format(uuid.uuid4().hex[:16])


def thumb_abspath(entry: dict) -> str | None:
    """Absolute path of an entry's thumbnail, or None when absent/missing.

    The stored name must be a bare filename (defense against a hand-edited
    store pointing outside the history dir)."""
    name = str(entry.get("thumb") or "")
    if not name or os.path.basename(name) != name:
        return None
    path = os.path.join(_HISTORY_DIR, name)
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
    entries = entries[:MAX_ENTRIES]
    _write_entries(entries)
    _gc_thumbs(entries)


def clear() -> None:
    """Drop every stored run and its thumbnails."""
    _write_entries([])
    _gc_thumbs([])


def _gc_thumbs(entries: list[dict]) -> None:
    """Best-effort delete of thumbnail files no kept entry references."""
    keep = {e.get("thumb") for e in entries if e.get("thumb")}
    try:
        names = os.listdir(_HISTORY_DIR)
    except OSError:
        return
    for name in names:
        if name.startswith("thumb_") and name.endswith(".png") and name not in keep:
            try:
                os.remove(os.path.join(_HISTORY_DIR, name))
            except OSError:
                pass  # nosec B110 -- GC is best-effort; retried on next write
