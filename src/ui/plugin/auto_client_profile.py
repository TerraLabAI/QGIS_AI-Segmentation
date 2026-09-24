

























from __future__ import annotations

import time

from ...core.telemetry_run_profile import machine_profile, slow_notice_phase


class GuiGapWatch:











    INTERVAL_MS = 25
    GAP_MS = 50.0

    def __init__(self) -> None:
        from qgis.PyQt.QtCore import QTimer

        self._blocked_ms = 0.0
        self._last = time.perf_counter()



        self._timer = QTimer()
        self._timer.setInterval(self.INTERVAL_MS)
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._last = time.perf_counter()
        self._timer.start()

    def _tick(self) -> None:
        now = time.perf_counter()
        gap_ms = (now - self._last) * 1000.0
        self._last = now
        if gap_ms > self.GAP_MS:
            self._blocked_ms += gap_ms - self.INTERVAL_MS

    def stop(self) -> float:


        try:
            if self._timer.isActive():
                self._tick()
            self._timer.stop()
            self._timer.timeout.disconnect(self._tick)
        except (RuntimeError, TypeError):
            pass
        return self._blocked_ms / 1000.0


def start_gui_gap_watch(plugin) -> None:

    stop_gui_gap_watch(plugin)
    plugin._auto_gui_blocked_s = None
    try:
        watch = GuiGapWatch()
        watch.start()
    except Exception:  # noqa: BLE001
        watch = None
    plugin._auto_gui_gap_watch = watch


def stop_gui_gap_watch(plugin) -> None:


    watch = getattr(plugin, "_auto_gui_gap_watch", None)
    plugin._auto_gui_gap_watch = None
    if watch is None:
        return
    try:
        plugin._auto_gui_blocked_s = float(watch.stop())
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def reset_run_profile(plugin) -> None:

    plugin._auto_client_profile = {}
    plugin._auto_longest_silence_s = 0.0
    plugin._auto_detect_done_mono = None
    plugin._auto_finalize_s = 0.0
    plugin._auto_slow_notice_sent = False
    plugin._review_pass_profile = {}
    start_gui_gap_watch(plugin)


def review_pass_profile(plugin) -> dict:

    prof = getattr(plugin, "_review_pass_profile", None)
    if not isinstance(prof, dict):
        prof = {}
        plugin._review_pass_profile = prof
    return prof


def add_review_pass_seconds(plugin, key: str, seconds: float) -> None:


    try:
        prof = review_pass_profile(plugin)
        prof[key] = float(prof.get(key, 0.0)) + max(0.0, float(seconds))
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def note_progress_gap(plugin, gap_s: float) -> None:

    try:
        if gap_s > float(getattr(plugin, "_auto_longest_silence_s", 0.0) or 0.0):
            plugin._auto_longest_silence_s = float(gap_s)
    except (TypeError, ValueError):
        pass


def snapshot_worker_profile(plugin, worker) -> None:



    if getattr(plugin, "_auto_detect_done_mono", None) is None:
        plugin._auto_detect_done_mono = time.monotonic()
    profile = dict(getattr(plugin, "_auto_client_profile", None) or {})
    if worker is not None and not profile:
        try:
            profile.update(worker.client_profile())
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    bridge = getattr(plugin, "_auto_tile_bridge", None)
    if bridge is not None and "gui_worst_gap_ms" not in profile:
        try:
            summary = bridge.gui_thread_summary()
            profile["gui_worst_gap_ms"] = int(
                round(float(summary.get("worst_queue_s", 0.0)) * 1000))
            profile["gui_queue_s"] = float(summary.get("queue_s", 0.0))
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    plugin._auto_client_profile = profile


def client_profile_props(plugin) -> dict:


    try:
        profile = dict(getattr(plugin, "_auto_client_profile", None) or {})


        for key, value in (getattr(plugin, "_auto_density_props", None) or {}).items():
            profile.setdefault(key, value)
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


        stop_gui_gap_watch(plugin)
        blocked = getattr(plugin, "_auto_gui_blocked_s", None)
        if blocked is not None:
            profile["gui_blocked_s"] = round(float(blocked), 2)
        return profile
    except Exception:  # noqa: BLE001
        return {}


def slow_notice_state(plugin, worker) -> dict:





    answered = int(getattr(worker, "tiles_succeeded", 0) or 0)
    settled = answered
    for name in ("tiles_timed_out", "tiles_failed_server", "tiles_skipped_network"):
        settled += int(getattr(worker, name, 0) or 0)
    try:
        awaiting = int(worker.tiles_awaiting_conversion())
    except Exception:  # noqa: BLE001
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
