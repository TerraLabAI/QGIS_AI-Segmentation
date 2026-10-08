






from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsMessageLog,
)

from ...core.i18n import tr
from ...core.interaction_dials import cancel_watchdog_ms
from ...core.telemetry_errors import slot_guard
from .auto_run_progress import (
    _WIND_DOWN_DETACH,
)
from .shared import park_orphaned_worker, release_worker_ref







_CANCEL_WATCHDOG_MS = 5000



_NOTHING_FOUND_COPY_ID = "copy.auto.nothing_found_yet"


def nothing_found_notice_text() -> str:













    from ...core.server_dials import dial_copy
    return dial_copy(_NOTHING_FOUND_COPY_ID, tr(
        "Nothing found in the first {n} tiles. Check the spelling of "
        "your prompt, try a simpler word, or check the zone and the "
        "imagery. The run continues and each tile still counts."))


class AutoRunCancelMixin:


    @staticmethod
    def _auto_cancel_grace_ms(worker) -> int:







        grace = cancel_watchdog_ms(_CANCEL_WATCHDOG_MS)
        try:
            drain_s = float(getattr(worker, "_stop_drain_budget_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            drain_s = 0.0
        return max(grace, int(drain_s * 1000) + 2000)

    def _take_auto_cancel_gesture(self) -> bool:







        dock = self.dock_widget
        if dock is None:
            return False
        try:
            return bool(dock.take_auto_cancel_gesture())
        except (RuntimeError, AttributeError):
            return False

    def _auto_cancel_is_confirmed(self, worker) -> bool:


















        dock = self.dock_widget
        armed = self._take_auto_cancel_gesture()
        if not armed or dock is None or self._auto_headless_run:
            return True
        run_id = getattr(self, "_auto_run_id", None)
        try:
            if not dock._confirm_auto_cancel():
                return False
        except (RuntimeError, AttributeError):
            return True




        if (getattr(self, "_auto_finalize_state", None) is not None
                or self._auto_review is not None):
            return False
        if self._auto_worker is worker:
            return True




        restart_pending = isinstance(
            getattr(self, "_auto_density_forced", None), dict)
        if (run_id and getattr(self, "_auto_run_id", None) == run_id
                and (self._auto_worker is not None or restart_pending)):
            self._stop_auto_run_restarted_during_question()
        return False

    def _stop_auto_run_restarted_during_question(self) -> None:






        if self._auto_worker is None:
            self._density_clear_forced()
            if getattr(self, "_auto_imagery_probe", None) is not None:
                self._abandon_imagery_probe()
        self._on_auto_cancel_clicked()

    @slot_guard(stage="segment")
    def _on_auto_cancel_clicked(self) -> None:







        if getattr(self, "_auto_finalize_state", None) is not None:






            self._take_auto_cancel_gesture()
            return
        worker = self._auto_worker
        if worker is None:
            self._take_auto_cancel_gesture()


            self._density_clear_forced()
            if self._auto_review is not None:





                return



            if self.dock_widget is not None:
                try:
                    self.dock_widget.set_auto_run_active(False)
                    self.dock_widget.set_auto_status("idle")
                except (RuntimeError, AttributeError):
                    pass
            self._reset_auto_live_pipeline()
            return
        if not self._auto_cancel_is_confirmed(worker):
            return


        self._pop_nothing_found_notice()




        if self.dock_widget is not None:
            try:
                self.dock_widget.set_auto_cancelling()
            except (RuntimeError, AttributeError):
                pass
        try:
            worker.request_stop()
        except (RuntimeError, AttributeError):
            pass

        self._cancel_active_tile_render()





        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(
            self._auto_cancel_grace_ms(worker),
            lambda w=worker: self._auto_cancel_watchdog(w))

    def _auto_cancel_watchdog(self, worker) -> None:


        if worker is None or self._auto_worker is not worker:
            return
        QgsMessageLog.logMessage(
            "Auto detection: cancel watchdog forcing wind-down "
            f"(worker did not confirm within {cancel_watchdog_ms(_CANCEL_WATCHDOG_MS)}ms)",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )



        for sig_name, slot_name in _WIND_DOWN_DETACH:
            try:
                getattr(worker, sig_name).disconnect(getattr(self, slot_name))
            except (TypeError, RuntimeError, AttributeError):
                pass






        try:
            still_running = worker.isRunning()
        except RuntimeError:
            still_running = False
        if still_running:
            park_orphaned_worker(worker)


        self._on_auto_cancelled()

    def _pop_nothing_found_notice(self) -> None:








        try:
            head, _sep, tail = nothing_found_notice_text().partition("{n}")
            head = head.strip()
            tail = tail.strip()
            if not head and not tail:
                return
            bar = self.iface.messageBar()
            try:
                items = list(bar.items())
            except (AttributeError, TypeError):
                items = [bar.currentItem()]
            for item in items:
                if item is None:
                    continue
                shown = str(item.text() or "")
                if ((not head or shown.startswith(head))
                        and (not tail or shown.endswith(tail))):
                    bar.popWidget(item)
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _cancel_active_tile_render(self) -> None:







        try:
            from ...core.cloud_detection import cancel_active_tile_render
            cancel_active_tile_render()
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _send_auto_cancelled_terminal(self, tiles_done: int) -> None:





        if getattr(self, "_auto_tel_stop_reason", None) not in (None, "completed"):
            return
        try:
            from ...core import telemetry_run_events
            from .auto_client_profile import client_profile_props
            telemetry_run_events.track_auto_detect_cancelled(
                run_id=self._auto_run_id or "",
                tiles_done=tiles_done,
                tiles_total=int((self._auto_run_ctx or {}).get("total", tiles_done) or tiles_done),
                salvaged_to_review=False,
                duration_ms=self._auto_duration_ms(),
                client_profile=client_profile_props(self),
            )
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        self._auto_tel_stop_reason = "cancelled"

    def _stop_auto_detection(self, send_terminal: bool = True) -> None:























        self._stop_auto_stall_watchdog()
        self._pop_nothing_found_notice()

        self._density_clear_forced()





        self._cancel_active_tile_render()
        worker = self._auto_worker
        if worker is None:






            self._end_superseded_finalize(send_terminal=send_terminal)
            self._reset_auto_live_pipeline()
            return











        for sig_name, slot_name in _WIND_DOWN_DETACH:
            try:
                getattr(worker, sig_name).disconnect(getattr(self, slot_name))
            except (TypeError, RuntimeError, AttributeError):
                pass

        try:


            if getattr(worker, "_stop_reason", None) is None:
                worker._stop_reason = "error"
            worker.request_stop()
        except (RuntimeError, AttributeError):
            pass



        self._auto_merger = None







        slot = getattr(self, "_auto_cancelled_slot", None)
        if slot is not None:
            try:
                worker.cancelled.disconnect(slot)
            except (TypeError, RuntimeError, AttributeError):
                pass
        self._auto_cancelled_slot = None

        if send_terminal:
            self._send_auto_cancelled_terminal(int(getattr(worker, "tiles_succeeded", 0) or 0))
        release_worker_ref(worker)
        self._auto_worker = None
        self._drop_auto_tile_bridge()
        if getattr(self, "_auto_review", None) is None:
            try:
                from ...core.detection_policy_core import release_run_policy
                release_run_policy()
            except Exception:  # noqa: BLE001  # nosec B110
                pass


        self._reset_auto_live_pipeline()
        self._remove_auto_selection_layer()
        self._set_zone_badge_enabled(True)
        if self.dock_widget:
            try:



                self.dock_widget.set_auto_finalizing(False)
                self.dock_widget.set_auto_run_active(False)





                self.dock_widget.set_auto_status("info", tr("Stopping..."))
            except (RuntimeError, AttributeError):
                pass
