










from __future__ import annotations

import threading
import time




RECENT_WARMING_S = 0.0

_lock = threading.Lock()
_warming_since: float | None = None
_last_warming_at: float | None = None


_wait_listener = None


def mark_warming() -> None:

    global _warming_since, _last_warming_at
    now = time.monotonic()
    with _lock:
        if _warming_since is None:
            _warming_since = now
        _last_warming_at = now


def mark_ready() -> None:


    global _warming_since
    with _lock:
        _warming_since = None


def is_warming() -> bool:
    with _lock:
        return _warming_since is not None


def warming_since() -> float | None:
    with _lock:
        return _warming_since


def _recent_window_s() -> float:
    try:
        from .server_dials import dial_in_range

        return float(dial_in_range("tuning.click.recent_warming_s",
                                   RECENT_WARMING_S, 0.0, 900.0))
    except Exception:  # noqa: BLE001
        return RECENT_WARMING_S


def recently_warming(window_s: float | None = None) -> bool:

    if window_s is None:
        window_s = _recent_window_s()
    with _lock:
        if _warming_since is not None:
            return True
        last = _last_warming_at
    return last is not None and time.monotonic() - last < window_s


def set_wait_listener(listener) -> None:
    global _wait_listener
    _wait_listener = listener


def notify_wait(elapsed_s: int | None) -> None:

    listener = _wait_listener
    if listener is None:
        return
    try:
        listener(elapsed_s)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
