"""Activation clicks and state, full-UI refresh, install/cancel, layer signals.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations


from qgis.core import QgsRasterLayer
from qgis.PyQt.QtCore import Qt


from ...core.activation_manager import (
    has_tos_locked,
    lock_tos,
    set_tos_accepted,
)
from ...core.i18n import tr
from .styles import (
    _BTN_GREEN_AUTH,
    _msg_label_qss,
)
from .widgets import (
    Mode,
)


class DockActivationMixin:
    """Activation clicks and state, full-UI refresh, install/cancel, layer signals."""

    def set_activated_state(self, activated: bool):
        """Flip the signed-in state and refresh every dependent section."""
        self._plugin_activated = activated
        if activated:
            # A stale pairing spinner must never survive a successful activation.
            self._stop_pairing_wait()
            self._pending_pairing_code = ""
        else:
            self.show_pairing_idle()
            self.activation_message_label.setVisible(False)
            # B2: never carry a signed-out session's balance into the footer.
            # Clear the cached credit state and hide the gauge/pill/label so no
            # stale "N / M" lingers, and a later sign-in re-fetches before it
            # shows anything (avoids flashing the previous account's numbers).
            self._auto_credits = None
            self._auto_credits_total = None
            self._auto_free_left = None
            self._auto_is_subscriber = False
            for _w in (getattr(self, "_credit_ring", None),
                       getattr(self, "_footer_credits_label", None),
                       getattr(self, "_subscribe_pill", None)):
                if _w is not None:
                    try:
                        _w.setVisible(False)
                    except (RuntimeError, AttributeError):
                        pass
            if getattr(self, "_footer_credits_label", None) is not None:
                self._footer_credits_label.setText("")
        self._update_full_ui()

    def set_activation_message(self, text: str, is_error: bool = False):
        """Public alias used by the plugin's pairing handlers."""
        self._show_activation_message(text, is_error)

    def _show_activation_message(self, text: str, is_error: bool = False):
        """Display a message in the activation section, framed per the
        message taxonomy (error = accent-red text on a red-tinted card,
        success = a lime-tinted card with plain text)."""
        self.activation_message_label.setText(text)
        kind = "error" if is_error else "success"
        self.activation_message_label.setStyleSheet(_msg_label_qss(kind))
        self.activation_message_label.setVisible(True)

    def _update_full_ui(self):
        """Refresh the whole dock for the current signed-in state + mode.

        Signed-out is one neutral first step BEFORE any mode concept: the mode
        switch is hidden and both former per-mode sign-in variants funnel to the
        single login page. Only once activated does the mode switch appear and
        the per-mode paths run.
        """
        activated = self._plugin_activated
        self._refresh_mode_switch_visibility()
        if not activated:
            self._show_signed_out_page()
            self._update_ui_state()
            self._update_try_automatic_hint_visibility()
            self._update_auto_tutorial_banner_visibility()
            return
        if self._mode == Mode.INTERACTIVE:
            self._update_full_ui_interactive()
        else:
            self._update_full_ui_automatic()
        self._update_ui_state()
        self._update_try_automatic_hint_visibility()
        self._update_auto_tutorial_banner_visibility()

    def _mode_flow_started(self) -> bool:
        """True once the current mode has left its Start screen for a live flow:
        Manual after 'Start Manual AI Segmentation' (a session is active), or
        Automatic after 'Start Automatic AI Segmentation' (past step 0). This is
        the single test for hiding the mode switch mid-flow."""
        if self._mode == Mode.INTERACTIVE:
            return bool(getattr(self, "_segmentation_active", False))
        return bool(getattr(self, "_auto_started", False))

    def _refresh_mode_switch_visibility(self) -> None:
        """Show the Manual/Automatic switch ONLY on a mode's Start screen.

        The switch is how you pick a mode, so it belongs to the choosing moment.
        Once a flow is launched (Manual session active, or Automatic past step 0)
        it is hidden: the mode is committed, and hiding it removes the temptation
        to switch mid-flow while giving the running flow the vertical space back.
        Exiting/finishing a flow returns to the Start screen and brings it back.
        Never shown signed-out (there is no mode concept before sign-in)."""
        try:
            visible = self._plugin_activated and not self._mode_flow_started()
            self.mode_switch.setVisible(visible)
        except (RuntimeError, AttributeError):
            pass

    def _show_signed_out_page(self):
        """One neutral login page: hide every mode surface, show the sign-in
        card with mode-agnostic copy (no price, no local/cloud wording)."""
        self.welcome_widget.setVisible(False)
        self.setup_group.setVisible(False)
        self.seg_widget.setVisible(False)
        self.batch_info_widget.setVisible(False)
        self.auto_page.setVisible(False)

        self.activation_group.setVisible(True)
        if hasattr(self, "_connect_hint_label"):
            self._connect_hint_label.setText(
                tr("Free account - sign up takes 15 seconds in your browser."))
        if hasattr(self, "_connect_btn"):
            self._connect_btn.setStyleSheet(_BTN_GREEN_AUTH)

        # Footer: gear hidden (no account yet), cross-promo hidden (a CTA for
        # another product has no place before sign-in), credit gauge/Subscribe
        # stay hidden (an activated surface).
        self._settings_btn.setVisible(False)
        self._ai_edit_btn.setVisible(False)
        # The tutorial button stays reachable even before sign-in: a lost
        # visitor benefits most from the guide.
        self._tutorial_btn.setVisible(True)
        self._refresh_auto_credits_display()

    def _update_full_ui_interactive(self):
        """Interactive mode (activated): install gate + segmentation section."""
        setup_complete = self._dependencies_ok and self._checkpoint_ok

        # Segmentation section: only show if fully set up.
        show_segmentation = setup_complete
        self.seg_widget.setVisible(show_segmentation)

        # Setup section (install/download group + welcome title): shown while a
        # setup is genuinely needed or running, tracked in _setup_section_wanted
        # rather than the sticky setup_group.isVisible(). The old proxy was
        # wiped by _update_full_ui_automatic (which force-hides setup_group), so
        # switching to Automatic mid-install and back left this page empty while
        # the background install kept running. Gating on the flag also still
        # avoids flashing "Click Install..." during the silent startup re-check.
        show_setup = (not setup_complete) and getattr(
            self, "_setup_section_wanted", False)
        self.setup_group.setVisible(show_setup)
        self.welcome_widget.setVisible(show_setup)

        self.activation_group.setVisible(False)
        self._settings_btn.setVisible(True)
        self._ai_edit_btn.setVisible(True)
        if not show_segmentation:
            self.batch_info_widget.setVisible(False)

        # Always hide automatic page in Interactive mode
        self.auto_page.setVisible(False)
        # Footer credit gauge is an Automatic-mode surface only.
        self._refresh_auto_credits_display()

    def _update_full_ui_automatic(self):
        """Automatic mode (activated): hide install sections, show auto page."""
        # Never show install/setup sections in Automatic mode
        self.welcome_widget.setVisible(False)
        self.setup_group.setVisible(False)
        self.seg_widget.setVisible(False)
        self.batch_info_widget.setVisible(False)
        self.activation_group.setVisible(False)

        self._settings_btn.setVisible(True)
        # The credit gauge + Subscribe pill take the bottom-left slot in
        # Automatic mode; the AI Edit cross-promo yields it (still shown in
        # Interactive mode).
        self._ai_edit_btn.setVisible(False)

        self.auto_page.setVisible(True)
        self._update_auto_page_state()

    def _on_install_clicked(self):
        # Log on click receipt so users stuck on the install screen with no
        # apparent reaction (observed on macOS 26) can prove the click event
        # was actually delivered to Qt - separating UI-event regressions from
        # signal-propagation or background-worker bugs.
        from ...core.logging_utils import log as _log
        _log("Install button clicked")
        self.install_button.setEnabled(False)
        self.install_requested.emit()

    def _toggle_cancel_button(self):
        visible = not self.cancel_button.isVisible()
        self.cancel_button.setVisible(visible)
        arrow = Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow
        self.cancel_toggle.setArrowType(arrow)

    def _on_cancel_clicked(self):
        from qgis.PyQt.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            tr("Cancel installation"),
            tr("Are you sure you want to cancel the installation?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.cancel_install_requested.emit()

    def _on_layer_changed(self, layer):
        # Just update UI state - layer change handling is done by the plugin
        self._update_ui_state()

    def _on_layers_added(self, layers):
        """Handle new layers added to project - auto-select if none selected."""
        # Update UI state first (includes layer filter)
        self._update_ui_state()

        if self.layer_combo.currentLayer() is not None:
            return

        for layer in layers:
            if isinstance(layer, QgsRasterLayer):
                # Auto-select: prefer local georeferenced, then online, then any raster
                if self._is_online_layer(layer):
                    self.layer_combo.setLayer(layer)
                    break
                if self._is_layer_georeferenced(layer):
                    self.layer_combo.setLayer(layer)
                    break

    def _on_layers_removed(self, _layer_ids):
        """Handle layers removed from project."""
        self._update_ui_state()

    def _on_layer_visibility_changed(self, node):
        """Handle layer visibility toggle in the layer tree (debounced)."""
        self._visibility_debounce_timer.start(100)

    def _on_tos_toggled(self, checked: bool):
        """Persist the Terms + Privacy acceptance and refresh gating. Consent is
        one GLOBAL state shared by the Manual and Automatic checkboxes, so mirror
        the sibling box (without recursing) to keep the two modes in sync."""
        set_tos_accepted(checked)
        for _attr in ("tos_checkbox", "auto_tos_checkbox"):
            cb = getattr(self, _attr, None)
            if cb is not None and cb.isChecked() != checked:
                cb.blockSignals(True)
                cb.setChecked(checked)
                cb.blockSignals(False)
        self._update_ui_state()

    def seal_tos_consent(self):
        """Seal Terms + Privacy consent forever and drop both checkboxes.

        Called by the first committed action in either mode: Manual Start, or
        the first Automatic Detect (where the Automatic checkbox now lives,
        right above the button - consent friction at the latest possible
        moment). Subsequent sessions (or plugin updates) never re-display it.
        """
        if not has_tos_locked():
            lock_tos()
        for _attr in ("tos_container", "auto_tos_container"):
            container = getattr(self, _attr, None)
            if container is not None:
                container.setVisible(False)

    def _on_start_clicked(self):
        layer = self.layer_combo.currentLayer()
        if not layer:
            return
        self.seal_tos_consent()
        self.start_segmentation_requested.emit(layer)

    def _on_start_shortcut(self):
        """G starts the visible mode's flow: Manual Start in Manual mode,
        the Automatic Start button on its Start step in Automatic mode (K3)."""
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

    def _on_stop_clicked(self):
        self.stop_segmentation_requested.emit()
