"""One local-AI install, one lock on the review that asked for it.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out so
agents and humans can work on one concern per file. Methods here are plain
mixin members: state lives on the plugin instance (self).

Two review actions need the on-device model and can be the first thing that
ever asks for it: the AI fix on a polygon (manual_handoff) and the AI Add lane
(manual_add). Each keeps its own pending flag, because each has its own thing
to open when the model arrives. What they share is the install itself, and the
install is the part that used to go wrong: it runs for minutes, and the review
stayed walkable while it did, so the user could leave the step that carries
the progress banner, toggle the method that drops the pending intent, or
Finish the review out from under it. Every one of those left the install
running with nothing on screen and nobody left to hand the model to.

So the two lanes come through here. Starting an install locks the review to
it, and every way it can end (loaded, failed, cancelled) releases the lock
through one door.
"""
from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtWidgets import QMessageBox

from ...core.i18n import tr


class LocalAiInstallLockMixin:
    """Start, hold and release a local-AI install asked for from the review."""

    def _local_ai_install_pending(self) -> bool:
        """True while an install started from the review still owes somebody
        an answer: the fix session or the Add lane it was started for."""
        pending = getattr(self, "_refine_install_pending", False)
        pending = pending or getattr(self, "_ai_add_install_pending", False)
        return bool(pending)

    def _begin_local_ai_install(self, lane: str) -> None:
        """Start the one-time setup for ``lane`` ("reshape" or "add") and hold
        the review still until it ends.

        The select tool stays armed on purpose, and `_set_correct_selection`
        refuses while this is pending: disarming it would clear the selection,
        and the selection IS what the install is running for. So the picked
        polygon keeps its yellow outline and a click meanwhile does nothing.

        `_warm_predictor_on_ready` rides along so the model subprocess starts
        the moment the predictor object exists, rather than on the first click
        after the install. It is the same prediction the Correct step makes
        when it arms on AI, and here the user has already said it out loud.
        """
        if lane == "add":
            self._ai_add_install_pending = True
        else:
            self._refine_install_pending = True
        # The review has now spent a setup on this. Only after that does a
        # later load failure stop the lanes offering it again: a first failure
        # with no attempt behind it is often a truncated model file, and the
        # setup is what re-downloads it.
        self._local_ai_install_attempted = True
        self._warm_predictor_on_ready = True
        if self.dock_widget is not None:
            try:
                self.dock_widget.set_auto_review_installing(True)
            except (RuntimeError, AttributeError):
                pass
        QgsMessageLog.logMessage(
            f"Local AI setup started from the review ({lane})",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        try:
            # Both lanes fall back to the on-device model, which is the only
            # reason either of them ever starts an install, so this one carries
            # it whatever mode the dock is showing.
            self._on_install_requested(include_local_model=True)
        finally:
            # _on_install_requested has early returns that show their own
            # dialog and never start a worker: a platform that cannot run
            # local inference at all, or another install already in flight.
            # The lock is taken above, so those paths left the review frozen
            # with Export, Next and the step dials dead, and only the banner's
            # Cancel to escape. A lock nobody will ever lift is worse than no
            # lock, so hand the review straight back when nothing is running.
            if not self._local_ai_install_running():
                QgsMessageLog.logMessage(
                    "Local AI setup did not start; releasing the review",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                self._release_local_ai_install()

    def _warn_local_ai_unavailable_once(self) -> None:
        """Say the on-device AI is out, at most once per session.

        Reached when the user keeps clicking polygons after the model failed.
        A message bar rather than a modal: the review is still fully usable on
        the Manual method, so this reports, it does not interrupt.
        """
        if getattr(self, "_local_ai_unavailable_warned", False):
            return
        self._local_ai_unavailable_warned = True
        try:
            self.iface.messageBar().pushWarning(
                "AI Segmentation",
                tr("The on-device AI is unavailable, so the AI fix is off. "
                   "Switch the fix method to Manual to keep correcting."),
            )
        except (RuntimeError, AttributeError):
            pass

    def _local_ai_install_running(self) -> bool:
        """True while a worker that can end the install lock is alive.

        The lock is only safe to hold when one of these will report back. The
        startup check is deliberately absent: it is not an install and it
        cannot release anything.
        """
        for name in ("deps_install_worker", "download_worker",
                     "_verify_worker", "_predictor_worker"):
            worker = getattr(self, name, None)
            try:
                if worker is not None and worker.isRunning():
                    return True
            except RuntimeError:
                continue  # C++ side already gone: not running
        return False

    def _release_local_ai_install(self) -> None:
        """The install is over, whichever way: drop both pending intents and
        give the review back. Every end goes through here, so the lock cannot
        outlive the thing it was holding for.

        Both lanes, not one. The failure paths in env_setup used to clear the
        fix lane only, so an install started from Add left its flag standing
        and its banner up for the rest of the session. Idempotent, and the
        unlock is unconditional: a lock with no pending flag behind it is
        exactly the state that would otherwise never lift.
        """
        try:
            self._clear_refine_install_pending()
        except (RuntimeError, AttributeError):
            pass
        try:
            self._clear_ai_add_install_pending()
        except (RuntimeError, AttributeError):
            pass
        if self.dock_widget is None:
            return
        try:
            self.dock_widget.set_auto_review_installing(False)
        except (RuntimeError, AttributeError):
            pass
        # Put the resting select tool back if anything took it while the lock
        # was on, so a review handed back answers a click like it did before.
        # A no-op when it is still armed, which is the normal case.
        if self._auto_review is not None and getattr(self, "_auto_review_step", 0) == 1:
            try:
                self._arm_correct_select()
            except (RuntimeError, AttributeError):
                pass

    def _on_review_install_cancel_requested(self) -> None:
        """The banner's Cancel: the only way out of a locked review.

        Asked once, because a cancelled install leaves the AI fix unavailable
        and the next attempt starts the download again from where the package
        cache left it, not from nothing."""
        if not self._local_ai_install_pending():
            self._release_local_ai_install()
            return
        if not getattr(self, "_headless", False):
            box = QMessageBox(self.iface.mainWindow())
            box.setWindowTitle(tr("Stop the setup?"))
            box.setText(tr(
                "The on-device AI will not be installed, so fixing a polygon "
                "with it stays unavailable. What is already downloaded is "
                "kept, so starting again resumes from there."))
            stop_btn = box.addButton(
                tr("Stop the setup"), QMessageBox.ButtonRole.DestructiveRole)
            keep_btn = box.addButton(
                tr("Keep installing"), QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(keep_btn)
            box.exec()
            if box.clickedButton() is not stop_btn:
                return
        QgsMessageLog.logMessage(
            "Local AI setup cancelled from the review banner",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        try:
            self._on_cancel_install()
        except (RuntimeError, AttributeError):
            pass
        self._release_local_ai_install()
