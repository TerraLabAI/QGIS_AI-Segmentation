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
import tempfile
import time
import uuid
from contextlib import contextmanager

from .cache_paths import PLUGIN_CACHE_DIR

_HISTORY_DIR = os.path.join(PLUGIN_CACHE_DIR, "detection_history")
_HISTORY_FILE = "history.json"
_ACCOUNT_DIR_PREFIX = "account_"
_HISTORY_LOCK_FILE = ".history.lock"
# How long a lock left behind is honoured before it is taken over. A process
# that died mid-write must not stop every later run from being recorded.
_LOCK_STALE_S = 30.0

# Shipped cap on the runs kept locally, and the fallback the getter below
# returns whenever the server says nothing usable. This is the user's own
# segmentation history and we keep all of it (there is no in-app delete): the
# server is the true unbounded archive for signed-in users, and this local
# store is the offline feed for the library's Recent tab. Each entry costs a
# 256px PNG thumb plus a tiny JSON row, so the cap is what keeps the store
# bounded on disk. Oldest beyond it roll off.
MAX_ENTRIES = 500

# Ceiling on the stored zone outline, in WKT characters. A drawn zone is a
# handful of points and lands well under it; a traced coastline can be tens of
# thousands, and the store is a small JSON file read whole on every library
# open. Past the ceiling the entry keeps its bounding box alone, which is what
# every entry had before the outline was stored at all.
MAX_ZONE_WKT_CHARS = 64_000


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


def _log_history_problem(message: str) -> None:
    """Put one line in the QGIS log, or nowhere at all.

    The logger is imported here rather than at the top, so this module keeps
    importing outside QGIS the way the rest of it already does.
    """
    try:
        from qgis.core import Qgis

        from .logging_utils import log

        log(message, Qgis.MessageLevel.Warning)
    except Exception:  # noqa: BLE001 -- a log line is never worth an exception
        pass  # nosec B110


def _load_entries() -> tuple[list[dict], bool]:
    """Stored runs, newest first, plus whether the store was actually read.

    The flag is False only when a store may exist and could not be read: a
    lock, a permission refusal, an I/O error, a directory that would not open.
    Absent (nothing written yet) and unparseable (nothing left to keep) both
    answer True with an empty list, because replacing either loses nothing.

    Anything that OVERWRITES the file has to tell those apart. Read as an empty
    store, one bad read turns the next saved run into a one-entry file written
    over the whole history, and the thumbnail collection then deletes every
    image the vanished entries referenced.
    """
    try:
        with open(_history_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return [], True
    except ValueError:
        return [], True
    except OSError:
        return [], False
    if not isinstance(data, list):
        return [], True
    return [e for e in data if isinstance(e, dict)], True


def get_entries() -> list[dict]:
    """Stored runs, newest first. [] on any read problem (fail-safe).

    Each entry: ``{id, prompt, ts, layer_name, objects, crs, extent?,
    zone_wkt?, thumb?}`` where ``extent`` is ``[xmin, ymin, xmax, ymax]`` in
    the CRS named by the ``crs`` authid, ``zone_wkt`` the drawn zone polygon
    in that same CRS, and ``thumb`` a PNG filename inside :func:`history_dir`.
    """
    return _load_entries()[0]


def _write_entries(entries: list[dict]) -> None:
    """Atomic JSON write: temp file + os.replace, so a crash mid-write can
    never corrupt the existing store.

    The temp file carries a unique name. On a fixed one, two QGIS windows
    sharing a profile write the same path at the same time, and the move then
    publishes whichever buffer was half way through.
    """
    path = _history_path()
    directory = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".history-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, ensure_ascii=False)
            # Durable BEFORE the rename. Without this a power loss can make
            # the rename durable while the bytes are not, publishing a
            # zero-length store: the next read calls it unreadable and the
            # write after that takes every thumbnail with it.
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass  # nosec B110 -- the move already took it, or it never landed
        raise


def _lock_is_stale(path: str) -> bool:
    """Whether a lock file has sat there longer than any write could take."""
    try:
        return (time.time() - os.stat(path).st_mtime) > _LOCK_STALE_S
    except OSError:
        return True


@contextmanager
def _history_lock():
    """Hold the store's write lock for one load-modify-write, or yield False.

    add_entry reads, changes and writes, and two QGIS windows sharing a profile
    interleave those steps: an entry is lost, and the thumbnail sweep that
    follows then deletes that run's picture, which cannot be got back. A lock
    older than the stale window is taken over, so nothing is blocked for good.
    """
    path = os.path.join(account_history_dir(), _HISTORY_LOCK_FILE)
    held = False
    try:
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            held = True
        except FileExistsError:
            held = _lock_is_stale(path)
        except OSError:
            held = False
        yield held
    finally:
        if held:
            try:
                os.unlink(path)
            except OSError:
                pass  # nosec B110 -- another window took it over; nothing owed


def _drop_thumb(name: str | None) -> None:
    """Delete a thumbnail written for an entry that never got recorded.

    Only a successful write sweeps thumbnails, so this file would otherwise sit
    in the history directory until the next one.
    """
    if not name:
        return
    try:
        os.unlink(os.path.join(account_history_dir(), os.path.basename(name)))
    except OSError:
        pass  # nosec B110 -- a stray thumbnail is not worth an error


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
    zone_wkt: str | None = None,
) -> None:
    """Prepend one committed run, cap the store, GC orphaned thumbnails.

    ``extent`` is (xmin, ymin, xmax, ymax) in the CRS named by ``crs_authid``;
    ``thumb`` a filename already saved inside :func:`history_dir` (or None).

    ``zone_wkt`` is the polygon the user actually drew, in the same CRS as
    ``extent``. Optional, because a rectangle or headless zone has none. It is
    what "Run again here" points at: the bounding box of an L-shaped or
    diagonal zone covers ground the run never looked at, and re-running it
    bills for that ground. Stored only up to :data:`MAX_ZONE_WKT_CHARS`, so a
    hand-traced outline with thousands of vertices cannot bloat the store.
    """
    with _history_lock() as locked:
        _add_entry_locked(prompt, layer_name, objects, extent, crs_authid,
                          thumb, zone_wkt, locked)


def _add_entry_locked(
    prompt: str,
    layer_name: str,
    objects: int,
    extent: tuple[float, float, float, float] | None,
    crs_authid: str,
    thumb: str | None,
    zone_wkt: str | None,
    locked: bool,
) -> None:
    """The body of :func:`add_entry`, inside the store's write lock."""
    entries, readable = _load_entries()
    if not readable:
        # The store is there and would not open. Writing now would replace
        # every kept run with this one and take their thumbnails with it, so
        # this run goes unrecorded instead and the history survives. Its
        # thumbnail is already on disk and nothing else sweeps this path.
        _drop_thumb(thumb)
        _log_history_problem(
            "Could not read the run history, so this run was not added to it. "
            "The stored runs are left untouched.")
        return
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
    zone = (zone_wkt or "").strip()
    if zone and len(zone) <= MAX_ZONE_WKT_CHARS:
        entry["zone_wkt"] = zone
    if thumb:
        entry["thumb"] = os.path.basename(thumb)
    entries.insert(0, entry)
    entries = entries[:history_max_entries()]
    _write_entries(entries)
    if locked:
        # Only under the lock: without it another window may have written an
        # entry this load never saw, and the sweep would delete its thumbnail.
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
