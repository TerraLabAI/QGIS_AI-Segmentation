





from __future__ import annotations

from ...core.activation_manager import (
    has_tos_accepted,
    has_tos_locked,
    lock_tos,
)
from ...core.i18n import tr
from ...core.server_dials import dial_in_range
from .styles import (
    _BTN_GHOST,
    _BTN_GREEN,
    _BTN_GREEN_AUTH,
    _SETUP_STATUS_QSS,
    _msg_label_qss,
)
from .widgets import (
    Mode,
)



_VISIBILITY_DEBOUNCE_MS = 100


class DockActivationMixin:


    def set_activated_state(self, activated: bool):

        self._plugin_activated = activated
        if activated:

            self._stop_pairing_wait()
            self._pending_pairing_code = ""
        else:
            self.show_pairing_idle()
            self.activation_message_label.setVisible(False)




            self._auto_credits = None
            self._auto_credits_total = None
            self._auto_free_left = None
            self._auto_is_subscriber = False


            self._quota_envelopes = None



            try:
                self.set_auto_km2_block(None)
            except (RuntimeError, AttributeError):
                self._auto_km2_exceeded = False
        self._sync_pro_pill()
        self._update_full_ui()

    def set_activation_message(self, text: str, is_error: bool = False,
                               kind: str | None = None):

        self._show_activation_message(text, is_error, kind)

    def _show_activation_message(self, text: str, is_error: bool = False,
                                 kind: str | None = None):








        self.activation_message_label.setText(text)
        if not kind:
            kind = "error" if is_error else "success"
        self.activation_message_label.setStyleSheet(_msg_label_qss(kind))
        self.activation_message_label.setVisible(True)

    def _update_full_ui(self):







        self._update_full_ui_body()

    def _update_full_ui_body(self):

        activated = self._plugin_activated
        self._refresh_mode_switch_visibility()
        if not activated:
            self._show_signed_out_page()
            self._update_ui_state()
            self._refresh_manual_engine_ui()
            self._refresh_manual_credit_gate()
            return
        if self._mode == Mode.INTERACTIVE:
            self._update_full_ui_interactive()
        else:
            self._update_full_ui_automatic()
        self._update_ui_state()
        self._refresh_manual_engine_ui()
        self._refresh_manual_credit_gate()

    def _mode_flow_started(self) -> bool:





        if self._mode == Mode.INTERACTIVE:
            return bool(getattr(self, "_segmentation_active", False))
        return bool(getattr(self, "_auto_started", False))

    def _refresh_mode_switch_visibility(self) -> None:








        try:
            visible = self._plugin_activated and not self._mode_flow_started()
            self.mode_switch.setVisible(visible)
        except (RuntimeError, AttributeError):
            pass

    def _show_signed_out_page(self):


        self.welcome_widget.setVisible(False)



        self._refresh_setup_group_visibility()
        self.seg_widget.setVisible(False)
        self.batch_info_widget.setVisible(False)
        self.auto_page.setVisible(False)




        try:
            self.auto_review_view_row.setVisible(False)
        except AttributeError:
            pass

        self.activation_group.setVisible(True)
        if hasattr(self, "_connect_hint_label"):
            self._connect_hint_label.setText(
                tr("Free account - sign up takes 15 seconds in your browser."))
        if hasattr(self, "_connect_btn"):
            self._connect_btn.setStyleSheet(_BTN_GREEN_AUTH)


        self._settings_btn.setVisible(False)
        self._refresh_auto_credits_display()

    def _on_signed_out_contact(self) -> None:

        try:
            from ..settings.contact_dialog import show_contact_dialog

            show_contact_dialog(self)
        except (ImportError, RuntimeError, AttributeError):
            pass  # nosec B110

    def _update_full_ui_interactive(self):




        setup_complete = (self._dependencies_ok and self._checkpoint_ok) \
            or self._manual_cloud_route_picked()




        blocked = (self._manual_cloud_route_picked()
                   and self._manual_credits_exhausted()
                   and not self._segmentation_active)








        show_segmentation = setup_complete or self._manual_engine_offered()
        self.seg_widget.setVisible(show_segmentation)

        _start = getattr(self, "start_button", None)
        if _start is not None:
            try:
                _start.setVisible(not blocked)
            except (RuntimeError, AttributeError):

                pass



        _tutorial = getattr(self, "manual_tutorial_link", None)
        if _tutorial is not None:
            try:
                _tutorial.setVisible(not blocked)
            except RuntimeError:
                pass  # nosec B110










        show_setup = self._refresh_setup_group_visibility()






        engine_offered = self._manual_engine_offered()
        self.welcome_widget.setVisible(show_setup and not engine_offered)
        self._style_setup_for_engine_choice(engine_offered, show_setup)

        self.activation_group.setVisible(False)
        self._settings_btn.setVisible(True)
        if not show_segmentation:
            self.batch_info_widget.setVisible(False)




        self.auto_page.setVisible(False)
        self.auto_review_view_row.setVisible(False)
        self._refresh_auto_credits_display()

    def _setup_needed(self) -> bool:








        setup_complete = (self._dependencies_ok and self._checkpoint_ok) \
            or self._manual_cloud_route_picked()
        needed = not setup_complete or self._manual_install_running()
        if self._manual_engine_offered():
            needed = (
                self._manual_install_running()
                and not self._manual_install_window_owns_it()
            )
            needed = needed or bool(getattr(self, "_manual_install_failed", False))
        return needed

    def _refresh_setup_group_visibility(self) -> bool:



















        try:
            if not self._plugin_activated or self._mode == Mode.AUTOMATIC:
                self.setup_group.setVisible(False)
                return False
            show_setup = (
                getattr(self, "_setup_section_wanted", False)
                and self._setup_needed()
            )
            self.setup_group.setVisible(show_setup)
            return show_setup
        except (RuntimeError, AttributeError):

            return False

    def _style_setup_for_engine_choice(self, engine_offered: bool,
                                       show_setup: bool) -> None:


        try:
            look = "ghost" if engine_offered else "primary"
            if getattr(self, "_install_button_look", "primary") != look:
                self._install_button_look = look
                self.install_button.setStyleSheet(
                    _BTN_GHOST if engine_offered else _BTN_GREEN)
            cloud = self._manual_cloud_route_picked()
            installing = self._manual_install_running()
            start_offers_install = bool(
                engine_offered and not cloud
                and not self._manual_engine_local_ready())
            if engine_offered and start_offers_install and not installing:
                self.install_button.setVisible(False)
            link = getattr(self, "setup_cloud_link", None)
            if link is not None:
                link.setVisible(bool(show_setup and engine_offered and not cloud))
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _update_full_ui_automatic(self):


        self.welcome_widget.setVisible(False)



        self._refresh_setup_group_visibility()
        self.seg_widget.setVisible(False)
        self.batch_info_widget.setVisible(False)
        self.activation_group.setVisible(False)

        self._settings_btn.setVisible(True)

        self.auto_page.setVisible(True)
        self._update_auto_page_state()

    def _on_install_clicked(self):




        from ...core.logging_utils import log as _log
        _log("Install button clicked")








        self.install_button.setVisible(False)


        self._manual_install_wants_model = True
        self.install_requested.emit()

    def show_install_waiting_notice(self, message: str) -> None:











        try:
            self.setup_status_label.setText(message)
            self.setup_status_label.setToolTip("")
            self.setup_status_label.setStyleSheet(_SETUP_STATUS_QSS)
            self.setup_status_label.setVisible(True)
            self._setup_section_wanted = True
            self._refresh_setup_group_visibility()
        except (RuntimeError, AttributeError):

            pass

    def _on_cancel_clicked(self):


        from ..dialogs.confirm_dialog import question

        if question(
                self,
                tr("Cancel installation"),
                tr("Are you sure you want to cancel the installation?"),
                default_yes=False, destructive=True):



            self._end_manual_install_as_stopped()

    def _on_layer_changed(self, layer):

        self._update_ui_state()

    def _on_layers_added(self, _layers):








        self._update_ui_state()

    def _on_layers_removed(self, _layer_ids):

        self._update_ui_state()

    def _on_layer_visibility_changed(self, node):

        self._visibility_debounce_timer.start(dial_in_range(
            "tuning.ui.layer_visibility_debounce_ms",
            _VISIBILITY_DEBOUNCE_MS, 20, 2000))

    def seal_tos_consent(self):








        if not has_tos_locked():
            lock_tos()

    def require_privacy_notice(self, on_accept) -> bool:













        if has_tos_locked() or has_tos_accepted():
            return True
        self._privacy_notice_on_accept = on_accept
        self._show_privacy_notice()
        return False

    def _show_privacy_notice(self):



        from ..dialogs.privacy_notice_dialog import PrivacyNoticeDialog

        existing = getattr(self, "_privacy_notice_dialog", None)
        if existing is not None:
            existing.raise_()
            return


        dialog = PrivacyNoticeDialog(self.window())
        dialog.accepted.connect(self._on_privacy_notice_accepted)
        dialog.rejected.connect(self._on_privacy_notice_declined)
        dialog.finished.connect(self._on_privacy_notice_closed)
        self._privacy_notice_dialog = dialog
        dialog.open()

    def _on_privacy_notice_accepted(self):


        self.seal_tos_consent()
        queued = getattr(self, "_privacy_notice_on_accept", None)
        self._privacy_notice_on_accept = None
        self._update_ui_state()
        if queued is not None:
            queued()

    def _on_privacy_notice_declined(self):



        self._privacy_notice_on_accept = None
        message = tr("Nothing was sent. Press the button again to read the "
                     "notice.")
        if self._mode == Mode.AUTOMATIC:
            try:
                self.set_auto_status("info", message)
                return
            except (RuntimeError, AttributeError):

                pass
        self.show_manual_notice(message)

    def _on_privacy_notice_closed(self, _result: int):
        dialog = getattr(self, "_privacy_notice_dialog", None)
        self._privacy_notice_dialog = None
        if dialog is not None:
            dialog.deleteLater()

    def _on_start_clicked(self):
        layer = self.layer_combo.currentLayer()
        if not layer:
            return


        if not self._manual_engine_gate_start():
            return



        if not self.require_privacy_notice(self._on_start_clicked):
            return
        self.seal_tos_consent()
        self.start_segmentation_requested.emit(layer)

    def _on_start_shortcut(self):


        if self._mode == Mode.INTERACTIVE:
            if self.start_button.isEnabled() and self.start_button.isVisible():
                self._on_start_clicked()
            return
        try:
            if self.auto_start_btn.isVisible() and self.auto_start_btn.isEnabled():
                self._on_auto_start_clicked()
        except (RuntimeError, AttributeError):

            pass

    def _on_undo_clicked(self):
        self.undo_requested.emit()

    def _on_save_polygon_clicked(self):
        self.save_polygon_requested.emit()

    def _on_export_clicked(self):
        self.export_layer_requested.emit()

    def _on_clear_selection_clicked(self):
        self.clear_selection_requested.emit()

    def _on_stop_clicked(self):
        self.stop_segmentation_requested.emit()
