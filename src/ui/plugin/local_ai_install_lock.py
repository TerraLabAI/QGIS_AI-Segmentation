



















from __future__ import annotations

import time

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr




_UNAVAILABLE_NOTICE_QUIET_S = 12.0


class LocalAiInstallLockMixin:


    def _local_ai_install_pending(self) -> bool:


        pending = getattr(self, "_refine_install_pending", False)
        pending = pending or getattr(self, "_ai_add_install_pending", False)
        return bool(pending)

    def _begin_local_ai_install(self, lane: str) -> None:













        if lane == "add":
            self._ai_add_install_pending = True
        else:
            self._refine_install_pending = True




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



            self._on_install_requested(include_local_model=True)
        finally:







            if not self._local_ai_install_running():
                QgsMessageLog.logMessage(
                    "Local AI setup did not start; releasing the review",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                self._release_local_ai_install()

    def _warn_local_ai_unavailable_once(self) -> None:











        from ...core.server_dials import dial_in_range
        now = time.monotonic()
        said_at = getattr(self, "_local_ai_unavailable_warned_at", None)
        quiet_s = dial_in_range(
            "tuning.install.unavailable_notice_quiet_s", _UNAVAILABLE_NOTICE_QUIET_S, 4, 60)
        if said_at is not None and (now - said_at) < quiet_s:
            return
        self._local_ai_unavailable_warned_at = now
        try:
            self.iface.messageBar().pushWarning(
                "AI Segmentation",
                tr("The on-device AI is unavailable, so the AI fix is off. "
                   "Switch the fix method to Manual to keep correcting."),
            )
        except (RuntimeError, AttributeError):
            pass

    def _local_ai_install_running(self) -> bool:






        for name in ("deps_install_worker", "download_worker",
                     "_verify_worker", "_predictor_worker"):
            worker = getattr(self, name, None)
            try:
                if worker is not None and worker.isRunning():
                    return True
            except RuntimeError:
                continue
        return False

    def _release_local_ai_install(self) -> None:










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



        if self._auto_review is not None and getattr(self, "_auto_review_step", 0) == 1:
            try:
                self._arm_correct_select()
            except (RuntimeError, AttributeError):
                pass

    def _on_review_install_cancel_requested(self) -> None:





        if not self._local_ai_install_pending():
            self._release_local_ai_install()
            return
        if not getattr(self, "_headless", False):
            from ..dialogs.confirm_dialog import (
                DISCARD,
                PRIMARY,
                ChoiceButton,
                ask_choice,
            )

            if ask_choice(
                self.iface.mainWindow(), tr("Stop the setup?"),
                tr(
                    "The on-device AI will not be installed, so fixing a polygon "
                    "with it stays unavailable. What is already downloaded is "
                    "kept, so starting again resumes from there."),
                [ChoiceButton("stop", tr("Stop the setup"), DISCARD),
                 ChoiceButton("keep", tr("Keep installing"), PRIMARY)],
                default="keep", escape="keep",
            ) != "stop":
                return
        QgsMessageLog.logMessage(
            "Local AI setup cancelled from the review banner",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        try:
            self._on_cancel_install()
        except (RuntimeError, AttributeError):
            pass
        self._release_local_ai_install()
