




















from __future__ import annotations

import json
import math
import os
import tempfile
import time
import uuid
from contextlib import contextmanager

from .cache_paths import PLUGIN_CACHE_DIR
from .file_replace_retry import replace_file_with_retry

_HISTORY_DIR = os.path.join(PLUGIN_CACHE_DIR, "detection_history")
_HISTORY_FILE = "history.json"
_ACCOUNT_DIR_PREFIX = "account_"
_HISTORY_LOCK_FILE = ".history.lock"








MAX_ENTRIES = 500






MAX_ZONE_WKT_CHARS = 64_000
MAX_HISTORY_BYTES = 128 * 1024 * 1024


def history_max_entries() -> int:







    try:
        from .server_dials import dial_in_range

        return int(dial_in_range("library.history_max_entries", MAX_ENTRIES, 20, 5000))
    except Exception:  # noqa: BLE001  # nosec B110
        return MAX_ENTRIES


def _zone_wkt_max_chars() -> int:

    from .server_dials import dial_in_range

    return int(dial_in_range(
        "tuning.library.zone_wkt_max_chars", MAX_ZONE_WKT_CHARS, 1_000, 500_000))


def account_history_dir() -> str:






    try:
        from .presets.run_history_cache import account_fingerprint

        fingerprint = account_fingerprint()
    except Exception:
        return _HISTORY_DIR
    if not fingerprint:
        return _HISTORY_DIR
    return os.path.join(_HISTORY_DIR, f"{_ACCOUNT_DIR_PREFIX}{fingerprint}")


def history_dir() -> str:

    path = account_history_dir()
    os.makedirs(path, exist_ok=True)
    return path


def _history_path() -> str:
    return os.path.join(history_dir(), _HISTORY_FILE)


def _log_history_problem(message: str) -> None:





    try:
        from qgis.core import Qgis

        from .logging_utils import log

        log(message, Qgis.MessageLevel.Warning)
    except Exception:  # noqa: BLE001
        pass  # nosec B110


def _load_entries() -> tuple[list[dict], bool]:












    try:



        with open(_history_path(), encoding="utf-8-sig") as fh:
            if os.fstat(fh.fileno()).st_size > MAX_HISTORY_BYTES:
                return [], False
            data = json.load(fh)
    except FileNotFoundError:
        return [], True
    except (OSError, ValueError, RecursionError):
        return [], False
    if not isinstance(data, list):
        return [], False
    return [e for e in data if isinstance(e, dict)][:history_max_entries()], True


def get_entries() -> list[dict]:







    return _load_entries()[0]


def _write_entries(entries: list[dict]) -> None:







    path = _history_path()
    directory = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".history-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, ensure_ascii=False, allow_nan=False)




            fh.flush()
            os.fsync(fh.fileno())


        replace_file_with_retry(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass  # nosec B110
        raise


@contextmanager
def _history_lock():






    handle = None
    held = False
    try:
        try:
            path = os.path.join(history_dir(), _HISTORY_LOCK_FILE)
            handle = open(path, "a+b")
            if os.name == "nt":
                import msvcrt

                if os.fstat(handle.fileno()).st_size == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = True
        except (OSError, ImportError):
            pass  # nosec B110
        yield held
    finally:
        if handle is not None:
            if held and os.name == "nt":




                try:
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass  # nosec B110
            try:
                handle.close()
            except OSError:
                pass  # nosec B110


def _drop_thumb(name: str | None) -> None:





    if not _history_thumbnail_name(name):
        return
    try:
        os.unlink(os.path.join(account_history_dir(), os.path.basename(name)))
    except OSError:
        pass  # nosec B110


def new_thumb_filename() -> str:

    return f"thumb_{uuid.uuid4().hex[:16]}.png"


def _history_thumbnail_name(name) -> bool:

    return (isinstance(name, str) and name.startswith("thumb_")
            and name.endswith(".png") and len(name) <= 255
            and not any(char in name for char in ("/", "\\", ":", "\0")))


def thumb_abspath(entry: dict) -> str | None:




    name = entry.get("thumb")
    if not _history_thumbnail_name(name):
        return None
    path = os.path.join(account_history_dir(), name)
    root = os.path.normcase(os.path.realpath(account_history_dir()))
    if not os.path.normcase(os.path.realpath(path)).startswith(root + os.sep):
        return None
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












    with _history_lock() as locked:
        if not locked:
            _drop_thumb(thumb)
            _log_history_problem(
                "Could not lock the run history, so this run was not added to it.")
            return
        try:
            _add_entry_locked(prompt, layer_name, objects, extent, crs_authid,
                              thumb, zone_wkt)
        except (OSError, TypeError, ValueError, OverflowError):
            _drop_thumb(thumb)
            _log_history_problem("Could not save the run history. The stored runs are left untouched.")


def _add_entry_locked(
    prompt: str,
    layer_name: str,
    objects: int,
    extent: tuple[float, float, float, float] | None,
    crs_authid: str,
    thumb: str | None,
    zone_wkt: str | None,
) -> None:

    if prompt is not None and not isinstance(prompt, str):
        raise TypeError("history prompt must be text")
    if layer_name is not None and not isinstance(layer_name, str):
        raise TypeError("history layer name must be text")
    if crs_authid is not None and not isinstance(crs_authid, str):
        raise TypeError("history CRS must be text")
    if zone_wkt is not None and not isinstance(zone_wkt, str):
        raise TypeError("history zone must be text")
    entries, readable = _load_entries()
    if not readable:




        _drop_thumb(thumb)
        _log_history_problem(
            "Could not read the run history, so this run was not added to it. "
            "The stored runs are left untouched.")
        return
    entry: dict = {
        "id": uuid.uuid4().hex[:16],
        "prompt": (prompt or "").strip()[:4096],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "layer_name": (layer_name or "")[:4096],
        "objects": max(0, int(objects)),
        "crs": (crs_authid or "")[:128],
    }
    if extent is not None and len(extent) == 4:
        coords = [float(v) for v in extent]
        if all(math.isfinite(v) for v in coords):
            entry["extent"] = coords
    zone = (zone_wkt or "").strip()
    if zone and len(zone) <= _zone_wkt_max_chars():
        entry["zone_wkt"] = zone
    if _history_thumbnail_name(thumb):
        entry["thumb"] = thumb
    entries.insert(0, entry)
    entries = entries[:history_max_entries()]
    _write_entries(entries)
    _gc_thumbs(entries)


def clear_detection_history() -> None:




    with _history_lock() as locked:
        if not locked:
            _log_history_problem(
                "Could not lock the run history, so it was not cleared.")
            return
        try:
            _write_entries([])
        except OSError:
            _log_history_problem("Could not clear the run history. The stored runs are left untouched.")
            return
        _gc_thumbs([])


def _gc_thumbs(entries: list[dict]) -> None:








    keep = set()
    for entry in entries:
        thumb = entry.get("thumb")



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
                pass  # nosec B110
