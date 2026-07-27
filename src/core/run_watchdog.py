"""Stall detection for an Automatic detection run.

The detection worker's poll loop gives every in-flight tile a per-tile
deadline, so no tile can sit "pending" forever at the protocol level. What that
cannot catch is the network CALL itself blocking forever (a half-open socket, a
reply that never signals finished): the loop is stuck INSIDE the call, so it
cannot self-check and no terminal signal ever fires. The run then shows forever
progress with no way out.

The guard against that lives on the MAIN thread (a timer in the plugin), not in
the worker: only a thread that is still running its event loop can notice that
another thread has gone quiet. The main thread watches the age of the last
progress; this module holds the one decision it makes, kept pure so it can be
tested without QGIS or a live worker.
"""
from __future__ import annotations


def run_is_stalled(
    worker_running: bool,
    last_progress_ts: float | None,
    now: float,
    timeout_s: float,
) -> bool:
    """Whether a run should be declared stalled.

    True only when a worker is still running AND it has a recorded progress
    time AND that time is at least ``timeout_s`` old. A run with no worker, no
    recorded progress, or recent progress is never stalled (so a slow-but-alive
    run is left to finish on its own). A non-positive timeout never fires, which
    disables the watchdog rather than declaring every run stalled.
    """
    if not worker_running or last_progress_ts is None or timeout_s <= 0:
        return False
    return (now - last_progress_ts) >= timeout_s
