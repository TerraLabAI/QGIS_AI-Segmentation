"""Putting the Semi-Auto session's clicks on TerraLab's servers.

The ONE place that decides for this mode. It says yes only when the engine
card is on Cloud AI, the choice is still offered, the user has accepted the
data notice, and the account will be accepted. Every other answer is no, and no
means the session runs exactly as it always has: on this computer, with no
network. Cloud AI is the offered card, so the notice is what stands between a
fresh install and the first crop leaving.

Even on yes the mode stays usable offline. The predictor installed here keeps
the on-device one behind it and hands any click the network cannot answer
straight to it (see ``core/cloud_first_predictor.py``), so the option changes
where a click is answered, never whether it is.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); methods here are
plain mixin members and state lives on the plugin instance.
"""
from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr


class ManualCloudPredictorMixin:
    """Installs and removes the Semi-Auto session's remote click route."""

    def _manual_cloud_route_ready(self) -> bool:
        """True when this mode's clicks should be answered over the network.

        Fails closed on anything unexpected: an unreadable setting, an absent
        switch, an unanswered data notice or no account all land on the
        on-device path.

        The consent check is the one that carries the promise. The engine
        cards open on Cloud AI, so the stored route says yes for a user who has
        never touched it, and the dock asks the notice at Start. This
        is what makes that safe everywhere else: a headless caller, a script or
        an MCP session opens no dialog, and imagery must not leave the machine
        on a default nobody was shown.
        """
        try:
            from ...core.manual_cloud_route import (
                manual_cloud_route_enabled,
                manual_cloud_route_offered,
            )
            from ...core.manual_engine_choice import cloud_consent_given

            if not (manual_cloud_route_enabled() and manual_cloud_route_offered()
                    and cloud_consent_given()):
                return False
            from ...core.activation_manager import get_auth_token

            return bool(get_auth_token())
        except Exception:  # noqa: BLE001 -- the on-device path is the fallback
            return False

    def _manual_cloud_predictor_active(self) -> bool:
        """Whether the predictor in hand is THIS mode's remote route.

        Narrower than the shared cloud check on purpose. The review's route
        holds a bare remote predictor with nothing behind it, and letting that
        one stand for this mode's would leave the session with no on-device
        answer for a click the network refuses.
        """
        try:
            from ...core.cloud_first_predictor import CloudFirstPredictor

            return isinstance(getattr(self, "predictor", None), CloudFirstPredictor)
        except Exception:  # noqa: BLE001 -- an unimportable module is not active
            return False

    def _ensure_manual_cloud_predictor(self) -> bool:
        """Put the remote click route in place, and say whether it is there.

        True means the session's clicks travel, so the caller may skip the
        on-device install checks. False leaves the plugin untouched.

        An on-device predictor already loaded is set aside rather than dropped,
        so it stays available as the fallback and needs no second model load.
        """
        if not self._manual_cloud_route_ready():
            return False
        if self._manual_cloud_predictor_active():
            return True
        try:
            from ...core.activation_manager import get_auth_header
            from ...core.cloud_first_predictor import CloudFirstPredictor
            from ...core.cloud_sam_predictor import CloudSamPredictor

            if not self._cloud_correct_predictor_active():
                # Only an on-device predictor is worth parking. A remote one in
                # the slot means the on-device predictor is ALREADY parked, and
                # writing over that would make the fallback a second network
                # call instead of the offline answer it is there to be.
                self._local_predictor_held = getattr(self, "predictor", None)
            # Auth read here, on the GUI thread. The crop goes out from
            # set_image, which the caller runs on a worker thread, and the key
            # store is not something to reach for from there.
            self.predictor = CloudFirstPredictor(
                CloudSamPredictor(auth=get_auth_header()),
                local_source=lambda: getattr(self, "_local_predictor_held", None),
                on_fallback=self._note_manual_cloud_fallback,
                on_remote_answer=self._note_manual_cloud_answer,
            )
        except Exception as err:  # noqa: BLE001 -- fall back to the on-device path
            QgsMessageLog.logMessage(
                f"Remote click route unavailable, staying on this computer: {err}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return False
        QgsMessageLog.logMessage(
            "Semi-Auto: clicks answered off the machine",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        return True

    def _end_cloud_click_session(self) -> None:
        """Stop sending this account's clicks, now. Never raises.

        Called when the account goes away: the user signed out, or the stored
        key was refused. Nothing may keep travelling under it, so the crop work
        in flight is invalidated, the remote predictor gives the slot back to
        whatever held it, and its own generation moves, which ends the wait of
        any click still out and makes its answer unusable when it lands.

        The session is left standing WHEN there is something left to answer it.
        A user who signs out mid-session keeps the polygons they have not saved,
        and the on-device predictor this hands the slot back to answers the next
        click offline.

        On a machine that picked Cloud AI to skip the download there is no such
        predictor, and the slot goes back to None. Leaving the session open on
        that is the worst state this file can produce: the map tool stays armed,
        the panel still says a session is running, and every click from then on
        is swallowed with no outline, no message and no way back except Stop.
        So the session ends here instead, and says why.
        """
        for step in (getattr(self, "_invalidate_manual_encode", None),
                     getattr(self, "_drop_cloud_correct_predictor", None),
                     # The ledger travelled with the account that just went
                     # away. Left open it keeps refusing Saves for credits no
                     # charge would ever have been sent for.
                     getattr(self, "_end_manual_credit_session", None)):
            if step is None:
                continue
            try:
                step()
            except Exception:  # noqa: BLE001 -- teardown must never raise  # nosec B110
                pass
        self._end_manual_session_with_no_predictor()

    def _end_manual_session_with_no_predictor(self) -> None:
        """Close a live session the account took the only predictor from."""
        try:
            if not getattr(self, "_segmentation_active", False):
                return
            if getattr(self, "predictor", None) is not None:
                return
            # Keeps what is already saved. Only the untraced shape is lost, and
            # it could not have been finished on this machine anyway.
            self._stop_manual_session(keep_saves=True)
            self.iface.messageBar().pushMessage(
                tr("Session ended"),
                tr("Cloud AI needs your account, and it is signed out. Sign "
                   "back in, or install the offline AI to work without one."),
                level=Qgis.MessageLevel.Warning, duration=8)
        except Exception:  # noqa: BLE001 -- teardown must never raise  # nosec B110
            pass

    def _note_manual_cloud_fallback(self) -> None:
        """One quiet line when a click came back from this computer instead.

        The click already produced its outline, so this is a note and never a
        dialog. Best effort: it must not cost the click.

        The line is set through Qt's own hand-over rather than written here.
        Nothing in the predictor's shape says which thread answers a click, and
        a widget written from any thread but the one that drew it is a crash
        waiting for the wrong day.
        """
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
            pass  # nosec B110 -- the widget can be gone during teardown
