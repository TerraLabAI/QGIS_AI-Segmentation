























from __future__ import annotations

import time

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtCore import QTimer

from ...core.i18n import tr
from ...core.interaction_dials import route_memo_ms






_ROUTE_MEMO_MS = 3000.0

_FALLBACK_NOTICE_S = 6


class CorrectAiRouteMixin:


    def _correct_ai_route_is_remote(self) -> bool:








        if getattr(self, "_auto_review", None) is None:
            return False
        now = time.monotonic() * 1000.0
        memo = getattr(self, "_correct_route_memo", None)
        if memo is not None and now - memo[0] < route_memo_ms(_ROUTE_MEMO_MS):
            return memo[1]
        try:
            from ...core.server_dials import correct_ai_cloud_enabled

            if not correct_ai_cloud_enabled():
                answer = False
            else:
                from ...core.activation_manager import get_auth_token

                answer = bool(get_auth_token())
        except Exception:  # noqa: BLE001
            answer = False
        self._correct_route_memo = (now, answer)
        return answer

    def _cloud_correct_predictor_active(self) -> bool:










        predictor = getattr(self, "predictor", None)
        if predictor is None:
            return False
        try:
            from ...core.cloud_first_predictor import CloudFirstPredictor
            from ...core.cloud_sam_predictor import CloudSamPredictor

            return isinstance(predictor, (CloudSamPredictor, CloudFirstPredictor))
        except Exception:  # noqa: BLE001
            return False

    def _ensure_cloud_correct_predictor(self) -> bool:









        if not self._correct_ai_route_is_remote():
            return False
        if self._cloud_correct_predictor_active():
            return True
        try:
            from ...core.activation_manager import get_auth_header
            from ...core.cloud_sam_predictor import CloudSamPredictor

            held = getattr(self, "predictor", None)
            self._local_predictor_held = held
















            ledger = getattr(self, "_manual_credit_ledger", None)
            self.predictor = CloudSamPredictor(
                auth=get_auth_header(),
                on_remote_answer=self._note_manual_cloud_answer,
                session_id=getattr(ledger, "session_id", None),
            )
        except Exception as err:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Remote fix route unavailable, staying on-device: {err}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return False
        QgsMessageLog.logMessage(
            "Correct step: AI fix answered off the machine",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        return True

    def _degrade_correct_ai_to_manual(self, reason: str, failure_class: str = "") -> bool:




















        if not self._cloud_correct_predictor_active():
            return False
        if getattr(self, "_auto_review", None) is None:
            return False



        self._correct_route_memo = None
        QgsMessageLog.logMessage(
            f"Correct step: AI fix refused a click, handing it to Manual: {reason}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        self._correct_ai_failure_class = (
            failure_class or self._correct_ai_failure_kind(reason))
        QTimer.singleShot(0, self._switch_correct_to_manual_after_failure)
        return True

    def _correct_ai_failure_kind(self, reason: str) -> str:






        try:
            from ...core.error_policy import classify_run_error

            return classify_run_error(reason or "")
        except Exception:  # noqa: BLE001
            return "UNKNOWN"

    def _correct_ai_failure_line(self) -> str:










        from ...core.server_dials import dial_copy

        kind = getattr(self, "_correct_ai_failure_class", "") or "UNKNOWN"
        if kind == "CREDITS_EXHAUSTED":
            return dial_copy(
                "correct.ai_credits_exhausted",
                tr("Your cloud allowance for this month is used, so the AI "
                   "fix cannot answer. Switched to editing by hand, which "
                   "is free."))
        if kind == "AUTH":
            return dial_copy(
                "correct.ai_auth",
                tr("Sign in again to fix with the AI. Switched to editing "
                   "by hand, which needs no account."))
        return dial_copy(
            "correct.ai_unreachable",
            tr("AI fixing is not reachable right now. Switched to editing "
               "by hand, which works offline."))

    def _switch_correct_to_manual_after_failure(self) -> None:





        try:
            self._drop_cloud_correct_predictor()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        if getattr(self, "_auto_review", None) is None:
            return
        try:
            self.dock_widget.set_correct_method("manual")
        except (RuntimeError, AttributeError):
            pass
        try:
            self._on_correct_method_changed("manual")
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        try:
            from ...core.server_dials import dial_in_range

            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                self._correct_ai_failure_line(),
                level=Qgis.MessageLevel.Info,
                duration=dial_in_range(
                    "tuning.correct.fallback_notice_s", _FALLBACK_NOTICE_S, 4, 10),
            )
        except (RuntimeError, AttributeError):
            pass
        self._correct_ai_failure_class = ""

    def _drop_cloud_correct_predictor(self) -> None:






        if not self._cloud_correct_predictor_active():
            return
        try:
            self.predictor.cleanup()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        self.predictor = getattr(self, "_local_predictor_held", None)
        self._local_predictor_held = None


        self._correct_route_memo = None
