"""Which predictor answers the Automatic review's AI fix: on-device, or remote.

This is the ONE place that decides. Everything else asks
``_correct_ai_route_is_remote()`` or checks whether the remote predictor is in
place; nothing else reads the switch.

The decision is deliberately narrow. It says yes only inside the Automatic
review, only with the server switch explicitly on, and only for an account the
route will accept. Every other answer is no, and no means the plugin behaves
exactly as it did before this route existed: the on-device install, the model
download, the subprocess.

Semi-Auto mode does not get here on its own: the review check below keeps it
out, and its own choice lives in ``manual_cloud_predictor.py``, where the two
engine cards offer Cloud AI first and nothing is sent until the user has
accepted the notice that says what goes. The on-device predictor stays behind
it either way.

There is one door between the two, and it asks. Refine-in-Manual hands a
review session over to Semi-Auto and keeps whatever predictor is in the slot,
so ``_settle_refine_cloud_consent`` in ``manual_handoff.py`` puts the same
notice in front of that handoff and hands the remote predictor back on a no.
Anything else that turns a review predictor into a Semi-Auto session owes the
user the same question.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); methods here are
plain mixin members and state lives on the plugin instance.
"""
from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtCore import QTimer

from ...core.i18n import tr


class CorrectAiRouteMixin:
    """Picks the predictor behind the review's AI fix and Add lanes."""

    def _correct_ai_route_is_remote(self) -> bool:
        """True when the review's AI lanes should answer over the network.

        Fails closed on anything unexpected: a missing switch, an unreadable
        configuration or no account all land on the on-device path.
        """
        if getattr(self, "_auto_review", None) is None:
            return False
        try:
            from ...core.server_dials import correct_ai_cloud_enabled

            if not correct_ai_cloud_enabled():
                return False
            from ...core.activation_manager import get_auth_token

            return bool(get_auth_token())
        except Exception:  # noqa: BLE001 -- the on-device path is the fallback
            return False

    def _cloud_correct_predictor_active(self) -> bool:
        """Whether the predictor in hand answers over the network.

        Read by the local-install machinery, which must not start a model load
        under a session the network is already serving.

        Both remote routes count: the review's AI fix, and the Semi-Auto
        option that puts its own clicks on the network. The two never hold the
        slot at once, and the machinery that reads this needs the same answer
        from either.
        """
        predictor = getattr(self, "predictor", None)
        if predictor is None:
            return False
        try:
            from ...core.cloud_first_predictor import CloudFirstPredictor
            from ...core.cloud_sam_predictor import CloudSamPredictor

            return isinstance(predictor, (CloudSamPredictor, CloudFirstPredictor))
        except Exception:  # noqa: BLE001 -- an unimportable module is not active
            return False

    def _ensure_cloud_correct_predictor(self) -> bool:
        """Put the remote predictor in place, and say whether it is there.

        True means the click path is served over the network, so the caller
        skips every on-device install check. False leaves the plugin untouched.

        An on-device predictor already loaded is set aside rather than dropped,
        so a Manual session started later gets its own model back without
        reloading it.
        """
        if not self._correct_ai_route_is_remote():
            return False
        if self._cloud_correct_predictor_active():
            return True
        try:
            from ...core.activation_manager import get_auth_header
            from ...core.cloud_sam_predictor import CloudSamPredictor

            held = getattr(self, "predictor", None)
            self._local_predictor_held = held
            # Auth read here, on the GUI thread, and not on first use. The crop
            # now goes out from set_image, which the caller runs on a worker
            # thread, and the key store is not something to reach for from
            # there.
            # The ANSWER, not the callable, although the predictor accepts both.
            # A callable is asked again per request, and the first request is
            # sent from set_image, which the caller runs on a worker thread.
            # Reaching into the key store from there is the deadlock this
            # plugin already carries a scar from. The cost is that a key
            # rotated mid-session is only picked up by the next session.
            self.predictor = CloudSamPredictor(auth=get_auth_header())
        except Exception as err:  # noqa: BLE001 -- fall back to the on-device path
            QgsMessageLog.logMessage(
                f"Remote fix route unavailable, staying on-device: {err}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return False
        QgsMessageLog.logMessage(
            "Correct step: AI fix answered off the machine",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        return True

    def _degrade_correct_ai_to_manual(self, reason: str) -> bool:
        """The off-machine fix refused a click: hand the step to Manual.

        Returns True when the fallback took the failure, so the caller drops
        its own report. A network that will not answer is not something the
        user can act on, and a failure dialog on the first click of a step that
        opened itself on AI reads as the plugin breaking. One quiet line and
        the other method, which needs no network and no install, is the honest
        answer.

        Only ever fires inside a review served off the machine. Manual mode
        holds a different predictor, so it never reaches here.

        The switch itself waits one event-loop turn: the click that failed is
        still unwinding on this stack, and folding its session from under it
        would rewrite the very state the caller is about to roll back.
        """
        if not self._cloud_correct_predictor_active():
            return False
        if getattr(self, "_auto_review", None) is None:
            return False
        QgsMessageLog.logMessage(
            f"Correct step: AI fix refused a click, handing it to Manual: {reason}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        self._correct_ai_failure_class = self._correct_ai_failure_kind(reason)
        QTimer.singleShot(0, self._switch_correct_to_manual_after_failure)
        return True

    def _correct_ai_failure_kind(self, reason: str) -> str:
        """Why the click was refused, as one of ``error_policy``'s classes.

        The classifier reads the message the route answered with, and the
        server can teach it a new marker without a plugin release, which is
        why the branching below asks it rather than matching text here.
        """
        try:
            from ...core.error_policy import classify_run_error

            return classify_run_error(reason or "")
        except Exception:  # noqa: BLE001 -- an unreadable class is the generic one
            return "UNKNOWN"

    def _correct_ai_failure_line(self) -> str:
        """The one sentence the user reads, for the class of the refusal.

        Credits and sign-in are things they can act on, and each has its own
        fix; telling them "not reachable right now" sends them to look at their
        network instead. Everything else keeps the generic line, because a
        service that will not answer is not theirs to fix.

        All three are served copy, so a line that reads badly at the worst
        moment of the flow is one deploy away from fixed. The shipped wording
        stands whenever nothing is served."""
        from ...core.server_dials import dial_copy

        kind = getattr(self, "_correct_ai_failure_class", "") or "UNKNOWN"
        if kind == "CREDITS_EXHAUSTED":
            return dial_copy(
                "correct.ai_credits_exhausted",
                tr("You are out of credits, so the AI fix cannot answer. "
                   "Switched to editing by hand, which is free."))
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
        """Second half of the fallback, one turn after the failed click.

        Gives the predictor slot back, moves the switch, and tells the user in
        one line. Never raises: this runs off a timer with nothing above it.
        """
        try:
            self._drop_cloud_correct_predictor()
        except Exception:  # noqa: BLE001 -- teardown must never raise  # nosec B110
            pass
        if getattr(self, "_auto_review", None) is None:
            return
        try:
            self.dock_widget.set_correct_method("manual")
        except (RuntimeError, AttributeError):
            pass
        try:
            self._on_correct_method_changed("manual")
        except Exception:  # noqa: BLE001 -- the switch already reads Manual  # nosec B110
            pass
        try:
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                self._correct_ai_failure_line(),
                level=Qgis.MessageLevel.Info,
                duration=6,
            )
        except (RuntimeError, AttributeError):
            pass
        self._correct_ai_failure_class = ""

    def _drop_cloud_correct_predictor(self) -> None:
        """Give the predictor slot back to whatever held it before. Never raises.

        Called wherever a review-owned session ends, and at the top of a
        Semi-Auto start so that session decides its own route from a clean
        slot.
        """
        if not self._cloud_correct_predictor_active():
            return
        try:
            self.predictor.cleanup()
        except Exception:  # noqa: BLE001 -- teardown must never raise  # nosec B110
            pass
        self.predictor = getattr(self, "_local_predictor_held", None)
        self._local_predictor_held = None
