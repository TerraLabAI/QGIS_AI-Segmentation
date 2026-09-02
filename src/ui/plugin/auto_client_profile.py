"""The GUI side of one run's client profile, and the slow-notice event.

Plain functions over the plugin instance, not a mixin: the run terminals
live in three different mixins (all finished, error, cancelled) and each of
them needs the same two calls, take the worker's profile before the worker
goes, then hand the merged profile to its terminal event. A mixin method
name shared across those files is exactly the collision the MRO guard
exists for.

State kept on the plugin, all reset at the Detect click by
``reset_run_profile``:

- ``_auto_client_profile``: the worker's ``client_profile()`` plus the
  render bridge summary, taken at the terminal while both still exist.
- ``_auto_longest_silence_s``: the longest gap between two progress updates
  the GUI saw, which is what the user experienced as "stuck".
- ``_auto_detect_done_mono``: when the last answer landed.
- ``_auto_finalize_s``: the finalize pass wall clock, written by the
  finalize steps when they log their phases.
- ``_auto_slow_notice_sent``: the slow-notice event goes out once per run.
- ``_review_pass_profile``: what the review's shape, gap fill and snap passes
  cost, filled in by the passes and read at export or abandon.
"""
from __future__ import annotations

import time

from ...core.telemetry_run_profile import machine_profile, slow_notice_phase


def reset_run_profile(plugin) -> None:
    """Fresh per-run state at the Detect click."""
    plugin._auto_client_profile = {}
    plugin._auto_longest_silence_s = 0.0
    plugin._auto_detect_done_mono = None
    plugin._auto_finalize_s = 0.0
    plugin._auto_slow_notice_sent = False
    plugin._review_pass_profile = {}


def review_pass_profile(plugin) -> dict:
    """The review pass accumulator, created on first use."""
    prof = getattr(plugin, "_review_pass_profile", None)
    if not isinstance(prof, dict):
        prof = {}
        plugin._review_pass_profile = prof
    return prof


def add_review_pass_seconds(plugin, key: str, seconds: float) -> None:
    """Add ``seconds`` to one review pass bucket (shape_pass_s, gap_fill_s,
    snap_s). Never raises."""
    try:
        prof = review_pass_profile(plugin)
        prof[key] = float(prof.get(key, 0.0)) + max(0.0, float(seconds))
    except Exception:  # noqa: BLE001 -- a profile figure is never worth a pass  # nosec B110
        pass


def note_progress_gap(plugin, gap_s: float) -> None:
    """Remember the longest silence between two progress updates."""
    try:
        if gap_s > float(getattr(plugin, "_auto_longest_silence_s", 0.0) or 0.0):
            plugin._auto_longest_silence_s = float(gap_s)
    except (TypeError, ValueError):
        pass


def snapshot_worker_profile(plugin, worker) -> None:
    """Take the worker's profile and the render bridge summary now, before
    either is dropped. Called at every terminal; the first call per run wins
    the detect wall clock, later ones only fill in what is missing."""
    if getattr(plugin, "_auto_detect_done_mono", None) is None:
        plugin._auto_detect_done_mono = time.monotonic()
    profile = dict(getattr(plugin, "_auto_client_profile", None) or {})
    if worker is not None and not profile:
        try:
            profile.update(worker.client_profile())
        except Exception:  # noqa: BLE001 -- a worker mid-crash may not answer  # nosec B110
            pass
    bridge = getattr(plugin, "_auto_tile_bridge", None)
    if bridge is not None and "gui_worst_gap_ms" not in profile:
        try:
            summary = bridge.gui_thread_summary()
            profile["gui_worst_gap_ms"] = int(
                round(float(summary.get("worst_queue_s", 0.0)) * 1000))
            profile["gui_queue_s"] = float(summary.get("queue_s", 0.0))
        except Exception:  # noqa: BLE001 -- the bridge may be half torn down  # nosec B110
            pass
    plugin._auto_client_profile = profile


def client_profile_props(plugin) -> dict:
    """The full client profile for a terminal event: the worker snapshot,
    the machine, and the GUI-side wall clocks. Never raises."""
    try:
        profile = dict(getattr(plugin, "_auto_client_profile", None) or {})
        profile.update(machine_profile())
        profile["longest_bar_silence_s"] = float(
            getattr(plugin, "_auto_longest_silence_s", 0.0) or 0.0)
        started = getattr(plugin, "_auto_run_started_mono", None)
        done = getattr(plugin, "_auto_detect_done_mono", None)
        now = time.monotonic()
        if started is not None:
            if done is not None:
                profile["detect_wall_s"] = max(0.0, done - started)
            profile["end_to_end_s"] = max(0.0, now - started)
        finalize_s = float(getattr(plugin, "_auto_finalize_s", 0.0) or 0.0)
        if finalize_s:
            profile["finalize_s"] = finalize_s
        return profile
    except Exception:  # noqa: BLE001 -- the terminal event still goes out
        return {}


def slow_notice_state(plugin, worker) -> dict:
    """What the client is doing while the card says the run is quiet.

    Read off the worker's plain counters; nothing here locks. ``phase`` is
    one of telemetry_run_profile.SLOW_PHASES.
    """
    answered = int(getattr(worker, "tiles_succeeded", 0) or 0)
    settled = answered
    for name in ("tiles_timed_out", "tiles_failed_server", "tiles_skipped_network"):
        settled += int(getattr(worker, name, 0) or 0)
    try:
        awaiting = int(worker.tiles_awaiting_conversion())
    except Exception:  # noqa: BLE001 -- no pool yet
        awaiting = 0
    inflight = int(getattr(worker, "inflight_now", 0) or 0)
    total = int((getattr(plugin, "_auto_run_ctx", None) or {}).get("total", 0) or 0)
    all_answered = total > 0 and settled >= total
    return {
        "tiles_answered": answered,
        "tiles_awaiting_conversion": awaiting,
        "inflight": inflight,
        "convert_pool": str(getattr(worker, "_convert_pool_kind", "") or ""),
        "phase": slow_notice_phase(all_answered, awaiting, inflight),
    }
