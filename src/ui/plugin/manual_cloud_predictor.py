
















from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr


class ManualCloudPredictorMixin:


    def _manual_cloud_route_ready(self) -> bool:










        try:
            from ...core.manual_cloud_route import (
                manual_cloud_route_enabled,
                manual_cloud_route_offered,
            )

            if not (manual_cloud_route_enabled()
                    and manual_cloud_route_offered()):
                return False
            from ...core.activation_manager import get_auth_token

            return bool(get_auth_token())
        except Exception:  # noqa: BLE001
            return False

    def _manual_cloud_predictor_active(self) -> bool:







        try:
            from ...core.cloud_first_predictor import CloudFirstPredictor

            return isinstance(getattr(self, "predictor", None), CloudFirstPredictor)
        except Exception:  # noqa: BLE001
            return False

    def _ensure_manual_cloud_predictor(self) -> bool:








        if not self._manual_cloud_route_ready():
            return False
        if self._manual_cloud_predictor_active():
            return True
        try:
            from ...core.activation_manager import get_auth_header
            from ...core.cloud_first_predictor import CloudFirstPredictor
            from ...core.cloud_sam_predictor import CloudSamPredictor

            if not self._cloud_correct_predictor_active():




                self._local_predictor_held = getattr(self, "predictor", None)






            ledger = getattr(self, "_manual_credit_ledger", None)
            self.predictor = CloudFirstPredictor(
                CloudSamPredictor(
                    auth=get_auth_header(),
                    session_id=getattr(ledger, "session_id", None)),
                local_source=lambda: getattr(self, "_local_predictor_held", None),
                on_fallback=self._note_manual_cloud_fallback,
                on_remote_answer=self._note_manual_cloud_answer,
            )
        except Exception as err:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Remote click route unavailable, staying on this computer: {err}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return False
        QgsMessageLog.logMessage(
            "Semi-Auto: clicks answered off the machine",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        return True

    def _end_cloud_click_session(self, say_signed_out: bool = True) -> None:




















        for step in (getattr(self, "_invalidate_manual_encode", None),



                     getattr(self, "_stop_hover_preview", None),
                     getattr(self, "_drop_cloud_correct_predictor", None),



                     getattr(self, "_end_manual_credit_session", None)):
            if step is None:
                continue
            try:
                step()
            except Exception:  # noqa: BLE001  # nosec B110
                pass



        self._hover_route_memo = None
        self._end_manual_session_with_no_predictor(say_signed_out)

    def _end_manual_session_with_no_predictor(self, say_signed_out: bool = True) -> None:





        try:



            dock = getattr(self, "dock_widget", None)
            if dock is None or not getattr(dock, "_segmentation_active", False):
                return
            if getattr(self, "predictor", None) is not None:
                return


            self._stop_manual_session(keep_saves=True)
            if not say_signed_out:
                return
            from ...core.server_dials import dial_in_range

            duration = dial_in_range("tuning.manual.session_ended_notice_s", 8, 5, 15)
            self.iface.messageBar().pushMessage(
                tr("Session ended"),
                tr("Cloud AI needs your account, and it is signed out. Sign "
                   "back in, or install the offline AI to work without one."),
                level=Qgis.MessageLevel.Warning, duration=duration)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _track_manual_click_answered(self, predict_ms: int, clock=None) -> None:







        try:
            from ...core import telemetry_session_events

            phases = clock.phase_properties() if clock is not None else {}
            if clock is not None:
                line = clock.summary_line()
                score = getattr(self, "current_score", None)
                if score is not None:
                    line += f", score {float(score):.3f}"
                QgsMessageLog.logMessage(line, "AI Segmentation",
                                         level=Qgis.MessageLevel.Info)
            on_cloud = (self._manual_cloud_predictor_active()
                        or self._cloud_correct_predictor_active())
            telemetry_session_events.track_manual_click_answered(
                engine="cloud" if on_cloud else "local",
                duration_ms=int(predict_ms),
                used_fallback=bool(getattr(self, "_manual_click_fell_back", False)),
                is_correct=bool(getattr(self, "_refine_handoff_active", False)),
                phases=phases,
            )
            if on_cloud and not getattr(self, "_cloud_notice_marked", False):





                from ...core.cloud_notice_seen import mark_cloud_notice_seen

                mark_cloud_notice_seen()
                self._cloud_notice_marked = True
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _note_manual_cloud_fallback(self) -> None:












        self._manual_click_fell_back = True
        try:
            if self._headless or not self.dock_widget:
                return
            from qgis.PyQt.QtCore import Q_ARG, QMetaObject, Qt

            from ...core.qt_compat import resolve_qt_enum

            text = tr("Answered on your computer this time. TerraLab could not "
                      "be reached.")
            queued = resolve_qt_enum(Qt, "ConnectionType", "QueuedConnection")
            QMetaObject.invokeMethod(
                self.dock_widget.instructions_label, "setText", queued,
                Q_ARG(str, text))
        except (RuntimeError, AttributeError, TypeError):
            pass  # nosec B110
