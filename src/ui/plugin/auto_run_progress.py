








from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsMessageLog,
)

from ...core.interaction_dials import lost_terminal_grace_s
from .shared import park_orphaned_worker










_STALL_TIMEOUT_S = 300.0

_STALL_CHECK_INTERVAL_MS = 5000





_SLOW_NOTICE_S = 20.0



_LOST_TERMINAL_GRACE_S = 15.0







_WIND_DOWN_DETACH = (
    ("tile_completed", "_on_auto_tile_completed"),
    ("progress", "_on_auto_progress"),
    ("all_tiles_finished", "_on_auto_all_finished"),
    ("warning", "_on_auto_warning"),
    ("nothing_found_yet", "_on_auto_nothing_found_yet"),
    ("error", "_on_auto_error"),
    ("credits_exhausted", "_on_auto_credits_exhausted"),
    ("queue_state", "_on_auto_queue_state"),
    ("run_phase", "_on_auto_run_phase"),
    ("rescan_state", "_on_auto_rescan_state"),
)







_STALL_WIND_DOWN_DETACH = tuple(
    entry for entry in _WIND_DOWN_DETACH if entry[0] != "tile_completed")


class AutoRunProgressMixin:






    def _start_auto_stall_watchdog(self) -> None:





        import time as _t
        self._auto_last_progress_ts = _t.monotonic()
        timer = getattr(self, "_auto_stall_timer", None)



        if timer is not None and not self._stall_timer_owns_dock(timer):
            timer = None
            self._auto_stall_timer = None
        if timer is None:
            from qgis.PyQt.QtCore import QTimer

            from ...core.server_dials import dial_in_range
            timer = QTimer(self.dock_widget)

            timer.setInterval(dial_in_range(
                "tuning.auto.stall_check_interval_ms", _STALL_CHECK_INTERVAL_MS,
                500, 30000))
            timer.timeout.connect(self._on_auto_stall_check)
            self._auto_stall_timer = timer
        try:
            timer.start()
        except (RuntimeError, AttributeError):
            self._auto_stall_timer = None

    def _stall_timer_owns_dock(self, timer) -> bool:




        try:
            return timer.parent() is self.dock_widget
        except (RuntimeError, AttributeError):
            return False

    def _stop_auto_stall_watchdog(self) -> None:



        timer = getattr(self, "_auto_stall_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except (RuntimeError, AttributeError):
                pass

    def _note_auto_progress(self) -> None:


        import time as _t
        now = _t.monotonic()
        last = getattr(self, "_auto_last_progress_ts", None)
        if last is not None:


            from .auto_client_profile import note_progress_gap
            note_progress_gap(self, now - last)
        self._auto_last_progress_ts = now



        if getattr(self, "_auto_link_slow_shown", False) and self.dock_widget is not None:
            self._auto_link_slow_shown = False
            try:
                self.dock_widget.set_auto_link_slow(False)
            except (RuntimeError, AttributeError):
                pass

    def _on_auto_run_phase(self, name: str) -> None:



        if self.dock_widget is None:
            return
        try:
            self.dock_widget.set_auto_wait_phase(name)
        except (RuntimeError, AttributeError):
            pass

    def _note_auto_link_slow(self, silent_for_s: float) -> None:





        if self.dock_widget is None:
            return
        from ...core.detection_policy import slow_notice_s
        slow = silent_for_s >= slow_notice_s(_SLOW_NOTICE_S)
        self._auto_link_slow_shown = slow




        local = False
        if slow:
            local = self._report_auto_slow_notice(silent_for_s)
        try:
            self.dock_widget.set_auto_link_slow(slow, local=local)
        except (RuntimeError, AttributeError):
            pass

    def _report_auto_slow_notice(self, silent_for_s: float) -> bool:


        worker = getattr(self, "_auto_worker", None)
        if worker is None:
            return False
        try:
            from .auto_client_profile import slow_notice_state
            state = slow_notice_state(self, worker)
        except Exception:  # noqa: BLE001
            return False
        local = state.get("phase") in ("converting", "assembling")
        if not getattr(self, "_auto_slow_notice_sent", False):
            self._auto_slow_notice_sent = True
            try:
                from ...core.telemetry_run_profile import track_auto_run_slow_notice
                track_auto_run_slow_notice(
                    run_id=self._auto_run_id or "",
                    silent_for_s=silent_for_s,
                    tiles_answered=state["tiles_answered"],
                    tiles_awaiting_conversion=state["tiles_awaiting_conversion"],
                    inflight=state["inflight"],
                    convert_pool=state["convert_pool"],
                    phase=state["phase"],
                )
            except Exception:  # noqa: BLE001
                pass  # nosec B110
        return local

    def _on_auto_stall_check(self) -> None:






        worker = self._auto_worker
        if worker is None:
            self._stop_auto_stall_watchdog()
            return
        try:
            running = worker.isRunning()
        except RuntimeError:
            return
        import time as _t

        from ...core.detection_policy import stall_timeout_s
        from ...core.run_watchdog import run_is_stalled, terminal_is_lost
        last = getattr(self, "_auto_last_progress_ts", None)
        if running and last is not None:
            self._note_auto_link_slow(_t.monotonic() - last)
        grace_s = lost_terminal_grace_s(_LOST_TERMINAL_GRACE_S)
        if terminal_is_lost(running, last, _t.monotonic(), grace_s):



            self._stop_auto_stall_watchdog()
            self._handle_auto_stall(worker, int(grace_s))
            return
        timeout = stall_timeout_s(_STALL_TIMEOUT_S)
        if not run_is_stalled(running, last, _t.monotonic(), timeout):
            return

        self._stop_auto_stall_watchdog()
        self._handle_auto_stall(worker, int(timeout))

    def _handle_auto_stall(self, worker, timeout_s: int) -> None:






        if worker is None or self._auto_worker is not worker:
            return
        QgsMessageLog.logMessage(
            "Auto detection: stall watchdog forcing wind-down "
            f"(no progress for {timeout_s}s)",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )




        try:
            worker._stop_reason = "stalled"
            worker.request_stop()
        except (RuntimeError, AttributeError):
            pass
        self._cancel_active_tile_render()





        for sig_name, slot_name in _STALL_WIND_DOWN_DETACH:
            try:
                getattr(worker, sig_name).disconnect(getattr(self, slot_name))
            except (TypeError, RuntimeError, AttributeError):
                pass
        self._finish_auto_stall_when_worker_stops(worker)

    def _finish_auto_stall_when_worker_stops(self, worker) -> None:









        state = {"done": False}

        def _salvage() -> None:

            if state["done"]:
                return
            state["done"] = True


            try:
                worker.tile_completed.disconnect(self._on_auto_tile_completed)
            except (TypeError, RuntimeError, AttributeError):
                pass
            if self._auto_worker is not worker:


                return




            try:
                stopped = worker.isFinished() and not worker.isRunning()
            except (RuntimeError, AttributeError):
                stopped = False
            if stopped:
                park_orphaned_worker(worker)
            self._on_auto_cancelled(reason="stalled")

        try:
            worker.finished.connect(_salvage)
        except (RuntimeError, AttributeError):


            _salvage()
            return
        try:
            still_running = worker.isRunning()
        except (RuntimeError, AttributeError):
            still_running = False
        if not still_running:



            _salvage()
            return
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(self._auto_cancel_grace_ms(worker), _salvage)
