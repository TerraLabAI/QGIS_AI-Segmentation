





















from __future__ import annotations

from .manual_local_install_dialog import ManualLocalInstallDialog






_INSTALL_START_WATCHDOG_MS = 20000




def _install_start_watchdog_ms() -> int:







    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range(
            "install.start_watchdog_ms", _INSTALL_START_WATCHDOG_MS,
            5_000, 120_000))
    except Exception:  # noqa: BLE001
        return _INSTALL_START_WATCHDOG_MS


class DockManualLocalInstallMixin:




    def _manual_install_window_owns_it(self) -> bool:





        dialog = getattr(self, "_manual_install_dialog", None)
        if dialog is None:
            return False
        try:
            return bool(dialog.is_installing())
        except RuntimeError:
            return False



    def _open_manual_local_install(self) -> bool:








        if self._manual_install_running():
            dialog = self._ensure_manual_install_dialog()
            if dialog is not None:
                try:
                    dialog.begin_progress()
                except RuntimeError:
                    self._manual_install_dialog = None




            self._set_manual_engine_cloud(False)
            return True

        dialog = self._ensure_manual_install_dialog()
        if dialog is None:


            self._setup_section_wanted = True
            self._manual_install_wants_model = True
            self._update_full_ui()
            self.install_requested.emit()
            return True

        try:
            answer = dialog.ask()
        except RuntimeError:
            self._manual_install_dialog = None
            return False
        if answer == "closed":




            self._close_manual_install_window()
            self._update_full_ui()
            return False
        if answer == "cloud":



            self._close_manual_install_window()
            self._set_manual_engine_cloud(True)
            return False




        self._set_manual_engine_cloud(False)
        try:
            dialog.begin_progress()
        except RuntimeError:
            self._manual_install_dialog = None
        self._setup_section_wanted = True


        self._manual_install_wants_model = True
        self.install_requested.emit()
        self._update_full_ui()
        self._watch_manual_install_started()
        return True

    def _watch_manual_install_started(self) -> None:
















        from ...core.qt_compat import safe_single_shot

        def _check() -> None:
            try:
                if not self._manual_install_window_owns_it():
                    return
                if self._manual_install_running():
                    return
                self._finish_manual_install_window(False)
            except (RuntimeError, AttributeError):
                pass  # nosec B110




        safe_single_shot(_install_start_watchdog_ms(), self, _check)

    def _ensure_manual_install_dialog(self) -> ManualLocalInstallDialog | None:

        dialog = getattr(self, "_manual_install_dialog", None)
        if dialog is not None:
            try:
                dialog.isVisible()
                return dialog
            except RuntimeError:
                dialog = None
        try:
            dialog = ManualLocalInstallDialog(self)
            dialog.cancel_requested.connect(self._on_manual_install_stop)
            dialog.finished.connect(self._on_manual_install_window_closed)
        except (RuntimeError, TypeError):
            return None
        self._manual_install_dialog = dialog
        return dialog



    def _mirror_manual_install_progress(self, percent: int, message: str) -> None:






        if not self._manual_install_window_owns_it():
            return
        try:
            self._manual_install_dialog.set_progress(percent, message)
        except (RuntimeError, AttributeError):
            self._manual_install_dialog = None

    def _finish_manual_install_window(self, ok: bool) -> None:






        if not self._manual_install_window_owns_it():
            return
        self._close_manual_install_window()
        if ok:
            return




        self._manual_install_failed = True
        self._setup_section_wanted = True
        self._update_full_ui()

    def _close_manual_install_window(self) -> None:

        dialog = getattr(self, "_manual_install_dialog", None)
        self._manual_install_dialog = None
        if dialog is None:
            return
        try:
            dialog.blockSignals(True)


            dialog.mark_install_ended()
            dialog.close()
            dialog.deleteLater()
        except RuntimeError:
            pass  # nosec B110



    def _end_manual_install_as_stopped(self) -> None:












        self._close_manual_install_window()
        self._manual_install_wants_model = False
        try:
            self.cancel_install_requested.emit()
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        if self._manual_engine_offered():
            self._set_manual_engine_cloud(True)

    def _on_manual_install_stop(self) -> None:

        self._end_manual_install_as_stopped()

    def _on_manual_install_window_closed(self, _result: int = 0) -> None:









        dialog = getattr(self, "_manual_install_dialog", None)
        if dialog is None:
            return
        try:
            if not dialog.is_installing():
                return
        except RuntimeError:
            self._manual_install_dialog = None
            return
        try:
            self._end_manual_install_as_stopped()
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        try:
            self._update_full_ui()
        except (RuntimeError, AttributeError):
            pass  # nosec B110
