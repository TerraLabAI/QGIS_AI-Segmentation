

























from __future__ import annotations

import itertools
import threading
import time
from contextlib import contextmanager

_lock = threading.Lock()
_active: dict[int, str] = {}
_tokens = itertools.count(1)


_last_activity = float("-inf")


def begin(kind: str = "network") -> int:

    global _last_activity
    token = next(_tokens)
    with _lock:
        _active[token] = str(kind)
        _last_activity = time.monotonic()
    return token


def end(token: int | None) -> None:

    global _last_activity
    if token is None:
        return
    with _lock:
        if _active.pop(token, None) is not None:
            _last_activity = time.monotonic()


@contextmanager
def network_busy(kind: str = "network"):

    token = begin(kind)
    try:
        yield token
    finally:
        end(token)


def touch() -> None:

    global _last_activity
    with _lock:
        _last_activity = time.monotonic()


def is_busy() -> bool:

    with _lock:
        return bool(_active)


def idle_for(quiet_s: float) -> bool:

    with _lock:
        if _active:
            return False
        return (time.monotonic() - _last_activity) >= float(quiet_s)


def active_kinds() -> list[str]:

    with _lock:
        return sorted(set(_active.values()))


def low_priority_slot_free() -> bool:









    try:
        from qgis.core import QgsApplication
        from qgis.PyQt.QtCore import QThreadPool

        limit = int(QThreadPool.globalInstance().maxThreadCount())
        running = int(QgsApplication.taskManager().countActiveTasks())
        return running < max(1, limit - 1)
    except Exception:  # noqa: BLE001
        return True
