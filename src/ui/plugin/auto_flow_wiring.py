







from __future__ import annotations

from ...core.i18n import tr
from ...core.telemetry_errors import slot_guard
from .shared import _debounce_timer


class AutoFlowWiringMixin:







    _WARMUP_MIN_INTERVAL_S = 90.0

    def _maybe_warmup_auto(self) -> None:

















        import time

        from ...core.activation_manager import get_auth_header, is_plugin_activated


        if not is_plugin_activated():
            return


        if self._auto_worker is not None:
            return
        auth = get_auth_header()
        if not auth:
            return

        from ...core.server_dials import dial_in_range
        warmup_min_interval_s = dial_in_range(
            "tuning.auto.warmup_min_interval_s", self._WARMUP_MIN_INTERVAL_S, 10.0, 600.0)
        now = time.monotonic()
        if now - self._last_warmup_monotonic < warmup_min_interval_s:
            return

        if self._warmup_task is not None and self._warmup_task.is_active():
            return

        self._last_warmup_monotonic = now
        try:
            from qgis.core import QgsApplication

            from ...api.terralab_client import TerraLabClient
            from ...workers.generic_request_task import GenericRequestTask
            client = TerraLabClient()
            self._warm_click_connection(client)
            self._warmup_task = GenericRequestTask(
                tr("Warming up AI Segmentation"),



                lambda: {"ok": bool(client.warmup(auth=auth))},
                hidden=True,
            )
            self._warmup_task.succeeded.connect(
                lambda res: self._on_warmup_finished(
                    bool(res.get("ok")) if isinstance(res, dict) else False))
            self._warmup_task.failed.connect(lambda *_a: self._on_warmup_finished(False))
            QgsApplication.taskManager().addTask(self._warmup_task)
        except Exception:

            self._warmup_task = None




    _SESSION_END_MAX_IDLE_S = 120.0

    def _cloud_work_in_flight(self) -> bool:
        if self._auto_worker is not None:
            return True
        if getattr(self, "_auto_headless_run", False) or getattr(self, "_headless", False):
            return True
        read = getattr(self, "_crop_read", None)
        return isinstance(read, dict) and read.get("worker") is not None

    def _signal_gpu_session_end(self, reason: str) -> None:








        try:
            if self._cloud_work_in_flight():
                return
            from ...api.detection_session import claim_session_end
            from ...core.activation_manager import get_auth_header, is_plugin_activated
            from ...core.server_dials import dial_in_range

            if not is_plugin_activated():
                return
            auth = get_auth_header()
            if not auth:
                return
            max_idle_s = dial_in_range(
                "tuning.session_end.max_idle_s", self._SESSION_END_MAX_IDLE_S, 0.0, 600.0)
            if not claim_session_end(max_idle_s):
                return

            self._last_warmup_monotonic = 0.0

            from qgis.core import QgsApplication

            from ...api.terralab_client import TerraLabClient
            from ...workers.generic_request_task import GenericRequestTask
            client = TerraLabClient()
            if not client.detection_direct:
                return
            task = GenericRequestTask(
                tr("Closing AI Segmentation session"),
                lambda: {"stopping": client.end_detection_session(auth)},
                hidden=True,
            )
            self._session_end_task = task
            task.succeeded.connect(lambda *_a: setattr(self, "_session_end_task", None))
            task.failed.connect(lambda *_a: setattr(self, "_session_end_task", None))
            QgsApplication.taskManager().addTask(task)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _warm_click_connection(self, client) -> None:












        try:
            if not getattr(client, "detection_direct", False):
                return
            from ...api.click_transport import warm_click_connection

            warm_click_connection(f"{client.detection_base_url}/health")
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _on_warmup_finished(self, ok: object = None) -> None:



        self._warmup_task = None
        if ok is False:
            self._last_warmup_monotonic = 0.0

    @slot_guard(stage="segment", user_message=tr(
        "Something went wrong starting the detection. Please try again."))
    def _on_auto_detect_requested(self) -> None:



        dock = self.dock_widget
        if dock is not None and not dock.confirm_prompt_for_detect():
            return





        if dock is not None:
            try:
                if not dock.require_privacy_notice(self._on_auto_detect_requested):
                    return
                dock.seal_tos_consent()
            except (RuntimeError, AttributeError):
                pass


        if dock is not None and not self._free_clip_question_passes(
                self._on_auto_detect_requested):
            return



        self._start_auto_detection()

    def _refresh_rerun_guard(self) -> None:







        dock = self.dock_widget
        if dock is None:
            return
        try:
            busy = (getattr(dock, "_auto_run_active", False) or getattr(dock, "_auto_review_active", False))
            row = getattr(dock, "auto_zero_assist_row", None)
            zero_up = bool(row is not None and row.isVisible())
            if busy or zero_up:
                dock.hide_auto_rerun_guard()
                return
            last = getattr(self, "_auto_last_run_sig", None)
            if not last:
                dock.hide_auto_rerun_guard()
                return
            cur = (dock.auto_prompt_input.text().strip(),
                   self._get_auto_detail_level(),
                   self._auto_exemplar_store.count())
        except (RuntimeError, AttributeError):
            return
        if cur != last:
            dock.hide_auto_rerun_guard()
            return

        if not dock.show_auto_rerun_guard():
            return
        if getattr(self, "_rerun_guard_emitted_sig", None) != cur:
            self._rerun_guard_emitted_sig = cur
            try:
                from ...core import telemetry_run_events
                telemetry_run_events.track_auto_prompt_hint_shown(
                    kind="identical_rerun", prompt=cur[0])
            except Exception:
                pass  # nosec B110

    def _on_auto_library_clicked(self) -> None:






        from ...core.server_dials import feature_enabled

        if not feature_enabled("library"):
            return
        try:
            from ..dialogs.segment_library_dialog import SegmentLibraryDialog
        except Exception as err:  # noqa: BLE001
            from qgis.core import Qgis

            from ...core.logging_utils import log
            log(f"Segment library unavailable: {err}", Qgis.MessageLevel.Warning)
            return
        from ...core.presets import segment_history



        run_active = getattr(self.dock_widget, "_auto_run_active", False)
        review_active = getattr(self.dock_widget, "_auto_review_active", False)
        view_only = bool(run_active or review_active)
        dlg = SegmentLibraryDialog(
            self.dock_widget, recent=segment_history.get_recent(), plugin=self,
            view_only=view_only)
        chosen = dlg.exec()
        token = dlg.get_selected_prompt() if chosen and not view_only else ""



        dlg.deleteLater()
        if token:
            self.dock_widget.set_prompt_text(token)

    def _on_zone_draw_requested(self) -> None:

        self._activate_zone_drawing()

    def _on_auto_detail_changed(self, _value: int) -> None:









        self._auto_detail_user_locked = True
        self._auto_detail_lock_prompt = self._resolved_auto_object_class().lower()




        self._schedule_credit_estimate()







        self._schedule_detail_telemetry("user")

    def _schedule_credit_estimate(self) -> None:



        _debounce_timer(self, "_credit_est_timer", self.dock_widget, 130,
                        self._update_credit_estimate)

    def _schedule_detail_telemetry(self, source: str) -> None:

        self._detail_tel_source = source
        _debounce_timer(self, "_detail_tel_timer", self.dock_widget, 1000,
                        self._emit_detail_telemetry)

    def _emit_detail_telemetry(self) -> None:
        try:
            from ...core import telemetry_run_events
            tiles = getattr(self, "_auto_est_tiles", -1)


            slider = self.dock_widget.auto_detail_slider
            telemetry_run_events.track_detail_changed(
                detail=self._get_auto_detail_level(),
                tiles=tiles if tiles is not None else -1,
                source=getattr(self, "_detail_tel_source", "user"),
                band_lo=int(slider.minimum()),
                band_hi=int(slider.maximum()),
                object_bound=bool(getattr(
                    self.dock_widget, "_auto_detail_object_bound", False)),
            )
        except Exception:
            pass  # nosec B110

    def _on_auto_layer_combo_changed(self, _layer) -> None:


        self._update_credit_estimate()

    def _on_auto_step_changed(self, index: int) -> None:







        if self._auto_worker is not None:
            return
        if index == 1:


            self._maybe_warmup_auto()


            if self._auto_zone is not None:
                self._on_zone_cleared()
            self._activate_zone_drawing()
        else:


            self._restore_maptool_after_zone()
