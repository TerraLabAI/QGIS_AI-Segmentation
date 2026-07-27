














from __future__ import annotations


def run_is_stalled(
    worker_running: bool,
    last_progress_ts: float | None,
    now: float,
    timeout_s: float,
) -> bool:








    if not worker_running or last_progress_ts is None or timeout_s <= 0:
        return False
    return (now - last_progress_ts) >= timeout_s


def terminal_is_lost(
    worker_running: bool,
    last_progress_ts: float | None,
    now: float,
    grace_s: float,
) -> bool:












    if worker_running or last_progress_ts is None or grace_s <= 0:
        return False
    return (now - last_progress_ts) >= grace_s
