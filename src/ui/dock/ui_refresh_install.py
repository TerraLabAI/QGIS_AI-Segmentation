







from __future__ import annotations

from ...core.i18n import tr
from ...core.server_dials import dial_in_range
from ...core.surface_dials import install_eta_ceiling_s, install_eta_honest_s
from .setup_status_text import setup_status_sentence
from .styles import _SETUP_STATUS_ERROR_QSS, _SETUP_STATUS_QSS





_INSTALL_ETA_HONEST_S = 20 * 60
_INSTALL_ETA_CEILING_S = 4 * 60 * 60



_INSTALL_ETA_MIN_PERCENT = 10
_INSTALL_ETA_MIN_ELAPSED_S = 5


_INSTALL_ETA_RECENT_WEIGHT = 0.7



_PROGRESS_TICK_MS = 500


class DockInstallStatusMixin:








    _READY_MARK = "\u2713"

    def _set_install_button_kind(self, kind: str, text: str) -> None:






        self._install_kind = (kind, text)
        try:
            self.install_button.setText(text)
        except (RuntimeError, AttributeError):

            pass

    def _install_button_kind(self) -> str:







        recorded = getattr(self, "_install_kind", None)
        try:
            live = self.install_button.text()
        except (RuntimeError, AttributeError):
            live = ""
        if isinstance(recorded, tuple) and len(recorded) == 2 and recorded[1] == live:
            return str(recorded[0])
        if live == tr("Update"):
            return "update"
        if live == tr("Retry"):
            return "retry"
        return "install"

    def set_dependency_status(self, ok: bool, message: str, mark: bool = True):





        self._dependencies_ok = ok

        if ok:
            text = message or ""
            if text and mark and not text.startswith(self._READY_MARK):
                text = f"{self._READY_MARK} {text}"
            self.setup_status_label.setText(text)
            self.setup_status_label.setToolTip("")



            self.setup_status_label.setVisible(bool(text))
            self.setup_status_label.setStyleSheet(_SETUP_STATUS_QSS)
            self.install_button.setVisible(False)
            self.cancel_button.setVisible(False)
            self.setup_progress.setVisible(False)
            self.setup_progress_label.setVisible(False)
        else:



            is_update = "updating" in message.lower() or "upgrading" in message.lower()
            is_dll_error = "dll" in message.lower() and "failed" in message.lower()
            display = setup_status_sentence(message) or message
            if is_dll_error:
                display = tr(
                    "Missing Visual C++ Redistributable. "
                    "Install it, restart your computer, then click Retry.")
                self.setup_status_label.setText(display)
                self.setup_status_label.setStyleSheet(_SETUP_STATUS_ERROR_QSS)
                self.setup_status_label.setVisible(True)
                self._set_install_button_kind("retry", tr("Retry"))
            else:





                self.setup_status_label.setText(display)
                self.setup_status_label.setStyleSheet(_SETUP_STATUS_QSS)
                self.setup_status_label.setVisible(bool(display))
                if is_update:
                    self._set_install_button_kind("update", tr("Update"))
                else:
                    self._set_install_button_kind("install", tr("Install"))



            self.setup_status_label.setToolTip(
                message if display != message else "")
            self.install_button.setVisible(True)
            self.install_button.setEnabled(True)





            self._setup_section_wanted = True






            if not self._manual_install_running():
                self._manual_install_wants_model = False
                self._finish_manual_install_window(False)

        self._update_full_ui()

    def set_install_progress(self, percent: int, message: str,
                             state: str | None = None):







        import time




        if getattr(self, "_auto_review_installing", False):
            try:
                self.auto_review_install_progress.setValue(
                    max(0, min(100, int(percent))))
                if message:
                    self.auto_review_install_label.setText(message)
            except (RuntimeError, AttributeError):
                pass

        self._target_progress = percent

        time_info = ""
        now = time.time()
        eta_min_percent = dial_in_range(
            "tuning.install.eta_min_percent", _INSTALL_ETA_MIN_PERCENT, 1, 50)
        eta_min_elapsed_s = dial_in_range(
            "tuning.install.eta_min_elapsed_s", _INSTALL_ETA_MIN_ELAPSED_S, 1, 60)
        if percent > eta_min_percent and percent < 100 and self._install_start_time:
            elapsed = now - self._install_start_time
            if elapsed > eta_min_elapsed_s:
                overall_speed = percent / elapsed
                remaining_pct = 100 - percent

                has_prev = self._last_percent_time is not None
                pct_increased = percent > self._last_percent
                time_increased = now > self._last_percent_time if has_prev else False
                if has_prev and pct_increased and time_increased:
                    dt = now - self._last_percent_time
                    dp = percent - self._last_percent
                    recent_speed = dp / dt
                    recent_weight = dial_in_range(
                        "tuning.install.eta_recent_weight",
                        _INSTALL_ETA_RECENT_WEIGHT, 0.1, 0.9)
                    blended_speed = (recent_weight * recent_speed
                                     + (1 - recent_weight) * overall_speed)
                else:
                    blended_speed = overall_speed

                if blended_speed > 0:
                    remaining = remaining_pct / blended_speed





                    honest_s = install_eta_honest_s(_INSTALL_ETA_HONEST_S)
                    remaining = min(remaining, install_eta_ceiling_s(_INSTALL_ETA_CEILING_S))
                    if remaining > honest_s:
                        time_info = " " + tr("(more than {n} min left)").format(
                            n=int(honest_s / 60))
                    elif remaining > 60:
                        time_info = " " + tr("(~{n} min left)").format(
                            n=int(remaining / 60))
                    elif remaining > 10:
                        time_info = " " + tr("(~{n} sec left)").format(
                            n=int(remaining))

        if percent > self._last_percent:
            self._last_percent_time = now
            self._last_percent = percent

        self.setup_progress_label.setText(f"{message}{time_info}")






        self._mirror_manual_install_progress(percent, f"{message}{time_info}")

        is_update = self._install_button_kind() == "update"



        lowered = (message or "").lower()
        install_cancelled = (state == "cancelled"
                             or (state is None and "cancel" in lowered))
        install_failed = (state == "failed"
                          or (state is None and "failed" in lowered))

        if percent == 0:


            self._setup_section_wanted = True


            self._manual_install_failed = False
            self._install_start_time = time.time()
            self._current_progress = 0
            self._last_percent = 0
            self._last_percent_time = None
            self._creep_counter = 0
            self.setup_progress.setValue(0)
            self.setup_progress.setVisible(True)
            self.setup_progress_label.setVisible(True)




            self.cancel_button.setText(tr("Cancel installation"))
            self.cancel_button.setVisible(True)
            self.install_button.setVisible(False)
            self.setup_status_label.setVisible(False)
            self.welcome_title.setText(tr("Installing AI Segmentation..."))
            self._progress_timer.start(dial_in_range(
                "tuning.install.progress_tick_ms", _PROGRESS_TICK_MS, 100, 2000))
        elif percent >= 100 or install_cancelled or install_failed:
            self._progress_timer.stop()
            self._install_start_time = None


            self._manual_install_wants_model = False
            self.setup_progress.setValue(percent)
            self.setup_progress.setVisible(False)
            self.setup_progress_label.setVisible(False)
            self.cancel_button.setVisible(False)
            self.install_button.setVisible(True)
            self.install_button.setEnabled(True)
            if is_update:
                self._set_install_button_kind("update", tr("Update"))
            else:
                self._set_install_button_kind("install", tr("Install"))
            if install_cancelled:


                self.setup_status_label.setStyleSheet(_SETUP_STATUS_QSS)
                self.setup_status_label.setVisible(True)
                self.setup_status_label.setText(tr("Installation cancelled"))
                self.welcome_title.setText(tr("Click Install to set up AI Segmentation"))
            elif install_failed:


                self.setup_status_label.setStyleSheet(_SETUP_STATUS_ERROR_QSS)
                self.setup_status_label.setVisible(True)
                self.setup_status_label.setText(tr("Installation failed"))
                self.welcome_title.setText(tr("Click Install to set up AI Segmentation"))



                self._manual_install_failed = True
            else:
                self.welcome_title.setText(tr("Click Install to set up AI Segmentation"))


            self._finish_manual_install_window(not install_failed)
        else:






            self.setup_progress.setVisible(True)
            self.setup_progress_label.setVisible(True)






            if not self._progress_timer.isActive():
                self._progress_timer.start(dial_in_range(
                    "tuning.install.progress_tick_ms",
                    _PROGRESS_TICK_MS, 100, 2000))
            if self._current_progress < percent:
                self._current_progress = percent
                self.setup_progress.setValue(percent)
            msg_lower = message.lower() if message else ""
            if "loading" in msg_lower and "model" in msg_lower:
                self.welcome_title.setText(tr("Loading AI model..."))
            elif "downloading" in msg_lower and "model" in msg_lower:
                self.welcome_title.setText(tr("Downloading AI model..."))
            elif "verifying" in msg_lower:
                self.welcome_title.setText(tr("Verifying installation..."))

    def _on_progress_tick(self):

        if self._current_progress < self._target_progress:
            step = max(1, (self._target_progress - self._current_progress) // 3)
            self._current_progress = min(
                self._current_progress + step, self._target_progress)
            self._creep_counter = 0
        elif self._current_progress < 99 and self._target_progress > 0:
            self._creep_counter += 1
            if self._creep_counter >= 4:
                self._creep_counter = 0
                if self._current_progress < self._target_progress + 5:
                    self._current_progress += 1

        self.setup_progress.setValue(self._current_progress)

    def set_checkpoint_status(self, ok: bool, message: str):
        self._checkpoint_ok = ok
        if not ok:





            try:
                self._progress_timer.stop()
            except (RuntimeError, AttributeError):
                pass  # nosec B110
            self._manual_install_wants_model = False






            try:
                self.welcome_title.setText(
                    tr("Click Install to set up AI Segmentation"))
            except (RuntimeError, AttributeError):
                pass  # nosec B110



            if message:
                self.setup_status_label.setText(message)
                self.setup_status_label.setStyleSheet(_SETUP_STATUS_ERROR_QSS)
                self.setup_status_label.setVisible(True)



                self._setup_section_wanted = True
        if ok:
            self.setup_status_label.setText(message)
            self.setup_status_label.setStyleSheet(_SETUP_STATUS_QSS)




            self._setup_section_wanted = False
            self._manual_install_failed = False
            self._manual_install_wants_model = False


            self._finish_manual_install_window(True)
        self._update_full_ui()
