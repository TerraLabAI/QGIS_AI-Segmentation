













from __future__ import annotations

import json
import os
import tempfile
import threading
import time

from .cache_paths import PLUGIN_CACHE_DIR

_FILE_NAME = "telemetry_queue.jsonl"
_MAX_BYTES = 256 * 1024
_MAX_AGE_S = 24 * 3600

_lock = threading.Lock()


def _path() -> str:
    return os.path.join(PLUGIN_CACHE_DIR, _FILE_NAME)


def _caps() -> tuple[int, float]:
    try:
        from .server_dials import dial_in_range

        return (int(dial_in_range("telemetry.disk_queue_max_bytes", _MAX_BYTES, 0, 4_194_304)),
                float(dial_in_range("telemetry.disk_queue_max_age_s", _MAX_AGE_S, 60, 7 * 24 * 3600)))
    except Exception:  # noqa: BLE001
        return _MAX_BYTES, _MAX_AGE_S


def _read_lines_locked(max_age_s: float) -> list[str]:
    try:
        with open(_path(), encoding="utf-8") as fh:
            raw = fh.read().splitlines()
    except OSError:
        return []
    cutoff = time.time() - max_age_s
    kept = []
    for line in raw:
        try:
            row = json.loads(line)
            if float(row["t"]) >= cutoff and isinstance(row["e"], dict):
                kept.append(line)
        except (ValueError, KeyError, TypeError):
            continue
    return kept


def _write_lines_locked(lines: list[str]) -> None:
    path = _path()
    if not lines:
        try:
            os.unlink(path)
        except OSError:
            pass
        return
    tmp = None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handle, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tq-", suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        from .file_replace_retry import replace_file_with_retry

        replace_file_with_retry(tmp, path)
    except Exception:  # noqa: BLE001  # nosec B110
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def park(events: list) -> None:

    try:
        max_bytes, max_age = _caps()
        if max_bytes <= 0 or not events:
            return
        now = time.time()
        new_lines = []
        for evt in events:
            try:
                new_lines.append(json.dumps({"t": now, "e": evt}, allow_nan=False))
            except (TypeError, ValueError, OverflowError, RecursionError):
                continue
        with _lock:
            lines = _read_lines_locked(max_age) + new_lines
            size = sum(len(x.encode("utf-8")) + 1 for x in lines)
            while lines and size > max_bytes:
                size -= len(lines.pop(0).encode("utf-8")) + 1
            _write_lines_locked(lines)
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def take(limit_bytes: int) -> list:



    try:
        _max_bytes, max_age = _caps()
        with _lock:
            lines = _read_lines_locked(max_age)
            taken, size = [], 0
            while lines:
                cost = len(lines[0].encode("utf-8")) + 1
                if taken and size + cost > limit_bytes:
                    break
                size += cost
                taken.append(lines.pop(0))
            _write_lines_locked(lines)
        return [json.loads(x)["e"] for x in taken]
    except Exception:  # noqa: BLE001  # nosec B110
        return []


def clear() -> None:

    try:
        with _lock:
            os.unlink(_path())
    except OSError:
        pass
    except Exception:  # noqa: BLE001  # nosec B110
        pass
