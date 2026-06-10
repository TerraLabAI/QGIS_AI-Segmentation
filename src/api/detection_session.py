







from __future__ import annotations

import threading
import time

SESSION_END_PATH = "/session/end"

_lock = threading.Lock()
_last_use: float | None = None
_end_sent_at: float | None = None


def note_use(url: str) -> None:

    if url.endswith(SESSION_END_PATH):
        return
    global _last_use
    with _lock:
        _last_use = time.monotonic()


def claim_session_end(max_idle_s: float) -> bool:




    global _end_sent_at
    with _lock:
        if _last_use is None:
            return False
        if _end_sent_at is not None and _end_sent_at >= _last_use:
            return False
        if not 0.0 <= time.monotonic() - _last_use <= max_idle_s:
            return False
        _end_sent_at = time.monotonic()
        return True
