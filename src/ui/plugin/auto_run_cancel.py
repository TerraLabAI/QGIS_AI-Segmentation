






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
from .shared import park_orphaned_worker







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
        try:
            if not dock._confirm_auto_cancel():
                return False
        except (RuntimeError, AttributeError):
            return True




        return (self._auto_worker is worker
                and getattr(self, "_auto_finalize_state", None) is None
                and self._auto_review is None)

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

    def _stop_auto_detection(self) -> None:


















        self._stop_auto_stall_watchdog()
        self._pop_nothing_found_notice()

        self._density_clear_forced()





        self._cancel_active_tile_render()
        worker = self._auto_worker
        if worker is None:




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
