







from __future__ import annotations

from ...core.cloud_notice_seen import cloud_notice_seen
from ...core.i18n import tr
from .cloud_notice_line import cloud_notice_line_html
from .styles import combo_theme_qss, locked_combo_qss
from .widgets import Mode


class DockSessionStateMixin:





    def set_segmentation_active(self, active: bool, layer=None):
        self._segmentation_active = active
        if active:


            self.hide_pro_after_success()
            self.clear_manual_export_success()





        if active:
            layer = layer or self.layer_combo.currentLayer()
            self._segmentation_layer_id = layer.id() if layer else None
        else:
            self._segmentation_layer_id = None





        self.layer_combo.set_view_tracking(not active)
        self.layer_combo.set_frozen(active)

        self._update_button_visibility()
        self._update_ui_state()


        self._refresh_mode_switch_visibility()




        self._refresh_manual_engine_ui()
        self._refresh_manual_credit_gate()
        if active:
            self._update_instructions()

    def _set_layer_combo_style(self, qss: str) -> None:


        if getattr(self, "_layer_combo_qss", None) == qss:
            return
        self._layer_combo_qss = qss
        try:
            self.layer_combo.setStyleSheet(qss)
        except (RuntimeError, AttributeError):
            pass

    def _update_button_visibility(self):
        if self._segmentation_active:








            self.layer_combo.setEnabled(False)
            self._set_layer_combo_style(locked_combo_qss("interactive"))

            self.start_container.setVisible(False)


            self.instructions_label.setVisible(not self._refine_handoff)
            self._update_instructions()


            self._update_refine_panel_visibility()




            self.save_mask_button.setEnabled(self._has_mask)
            if self._refine_handoff:
                self.save_mask_button.setVisible(False)
                self.secondary_buttons_widget.setVisible(False)
                self.undo_button.setVisible(False)
                self.clear_selection_button.setVisible(False)
                self.stop_button.setVisible(False)
                self._update_export_button_style()
            else:
                self._sync_manual_session_actions()
            self._sync_manual_session_tips()
        else:


            self.layer_combo.setEnabled(True)
            self._set_layer_combo_style(combo_theme_qss())

            self.start_container.setVisible(True)
            self.instructions_label.setVisible(False)
            self.preview_zoom_hint.setVisible(False)
            self.refine_group.setVisible(False)
            self.save_mask_button.setVisible(False)
            self.export_button.setVisible(False)
            self.undo_button.setVisible(False)
            self.clear_selection_button.setVisible(False)
            self.stop_button.setVisible(False)
            self.secondary_buttons_widget.setVisible(False)
            self.batch_info_widget.setVisible(False)

    def _manual_save_live(self) -> bool:



        if self._has_mask:
            return True
        try:
            return bool(self.save_mask_button.isEnabled())
        except (RuntimeError, AttributeError):
            return False

    def _sync_manual_session_actions(self) -> None:















        try:
            if not self._segmentation_active or self._refine_handoff:
                return
            saved = int(self._saved_polygon_count or 0)
            has_points = self._positive_count > 0 or self._negative_count > 0
            self.save_mask_button.setVisible(self._manual_save_live())
            self._update_export_button_style()
            can_undo = has_points or saved > 0
            self.undo_button.setEnabled(can_undo)
            self.undo_button.setVisible(can_undo)
            self.clear_selection_button.setEnabled(has_points)
            self.clear_selection_button.setVisible(has_points)
            self.stop_button.setEnabled(True)
            self.stop_button.setVisible(True)
            self.secondary_buttons_widget.setVisible(True)
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _sync_manual_session_tips(self) -> None:







        try:
            if not self._segmentation_active:
                return
            zoom = bool(self._preview_zoom_tip_wanted())
            if self.preview_zoom_hint.isHidden() == zoom:
                self.preview_zoom_hint.setVisible(zoom)


            zoom_shown = not self.preview_zoom_hint.isHidden()
            self.batch_info_widget.setVisible(
                not self._refine_handoff
                and int(self._saved_polygon_count or 0) == 0
                and not zoom_shown)
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def set_point_count(self, positive: int, negative: int):
        self._positive_count = positive
        self._negative_count = negative
        total = positive + negative
        has_points = total > 0
        old_has_mask = self._has_mask
        self._has_mask = has_points


        can_undo_saved = self._saved_polygon_count > 0
        self.undo_button.setEnabled((has_points or can_undo_saved) and self._segmentation_active)
        self.clear_selection_button.setEnabled(
            has_points and self._segmentation_active)
        self.save_mask_button.setEnabled(has_points)

        if self._segmentation_active:
            self._update_instructions()
            if self._refine_handoff:
                self._update_export_button_style()
            else:
                self._sync_manual_session_actions()
            if old_has_mask != self._has_mask:
                self._update_refine_panel_visibility()


                if self._refine_handoff:
                    self._update_button_visibility()

    def reset_session(self):
        self._has_mask = False
        self._segmentation_active = False
        self._segmentation_layer_id = None
        self._saved_polygon_count = 0
        self._positive_count = 0
        self._negative_count = 0
        self._manual_encoding = False
        self._manual_encoding_phase = "imagery"
        self.reset_refine_sliders()



        self.publish_refine_settings()
        self._update_button_visibility()
        self._update_ui_state()

    def set_saved_polygon_count(self, count: int):
        self._saved_polygon_count = count
        self._update_refine_panel_visibility()
        self._update_export_button_style()

        if self._segmentation_active:
            has_points = self._positive_count > 0 or self._negative_count > 0
            can_undo_saved = count > 0
            self.undo_button.setEnabled(has_points or can_undo_saved)
            if not self._refine_handoff:
                self._sync_manual_session_actions()
            self._sync_manual_session_tips()


        self._update_instructions()

    def _update_ui_state(self):
        if self._mode == Mode.INTERACTIVE:
            self._update_ui_state_interactive()
        else:
            self._update_ui_state_automatic()

        self._sync_update_card_for_work()

    def _update_ui_state_interactive(self):

        layer = self.layer_combo.currentLayer()
        has_layer = layer is not None

        has_rasters_available = self._sync_imagery_hero(self.layer_combo, self.no_rasters_widget)
        empty = not has_rasters_available and not self._segmentation_active





        self.no_rasters_widget.setVisible(empty)





        paywalled = self._manual_credit_gate_owns_page()
        self.layer_combo.setVisible(has_rasters_available and not paywalled)




        self.layer_label.setVisible(has_rasters_available and not paywalled)



        if not self._segmentation_active:
            self.start_container.setVisible(has_rasters_available
                                            and not paywalled)


        self._sync_manual_engine_card_visibility()


        setup_ok = (self._dependencies_ok and self._checkpoint_ok) \
            or self._manual_cloud_route_picked()
        activated = self._plugin_activated





        funded = not (self._manual_cloud_route_picked()
                      and self._manual_credits_exhausted())
        can_start = setup_ok and has_layer and activated and funded
        self.start_button.setEnabled(can_start and not self._segmentation_active)


        self._refresh_manual_engine_card_enabled()

    def _update_ui_state_automatic(self):

        if not self._plugin_activated:
            return









        if not self._auto_started:
            has_auto_rasters = self._sync_imagery_hero(
                self.auto_layer_combo, self.auto_no_rasters_widget)



            self.auto_no_rasters_widget.setVisible(not has_auto_rasters)
            self.auto_layer_combo.setVisible(has_auto_rasters)
            self.auto_layer_label.setVisible(has_auto_rasters)
            self.auto_steps.setVisible(has_auto_rasters)

            _start_ok = (has_auto_rasters
                         and self.auto_layer_combo.currentLayer() is not None)
            self.auto_start_btn.setEnabled(_start_ok)

            self.auto_start_btn.setToolTip("" if _start_ok else tr(
                "Load imagery in QGIS, then pick it above to start."))
        self._update_auto_detect_enabled()

    def _update_auto_page_state(self):

        if not self._plugin_activated:
            self.auto_upsell_card.setVisible(False)
            self.auto_controls_section.setVisible(False)
            return









        owns_page = bool(getattr(self, "_auto_run_active", False)
                         or getattr(self, "_auto_review_active", False))
        exhausted = self._is_free_exhausted() and not owns_page
        self.auto_upsell_card.setVisible(exhausted)



        blocked = self._refresh_auto_run_block(suppressed=exhausted)
        self.auto_controls_section.setVisible(not exhausted and not blocked)
        if exhausted:
            try:
                from ...core import telemetry_session_events
                telemetry_session_events.track_pro_upsell_viewed(
                    trigger="free_exhausted", cta_source="upsell_card")
            except Exception:
                pass  # nosec B110




        if not exhausted and not owns_page:
            if not self._auto_started:
                target = 0
            elif not self._auto_zone_is_set:
                target = 1
            else:
                target = 2



            try:
                settled = self.auto_steps.currentIndex() == target
            except (RuntimeError, AttributeError):
                settled = False
            if not settled:
                self._go_to_auto_step(target)



        try:
            self.auto_privacy_line.setText(cloud_notice_line_html())
            self.auto_privacy_line.setVisible(not cloud_notice_seen())
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        self._refresh_auto_credits_display()
        self._update_auto_detect_enabled()

    def _is_free_exhausted(self) -> bool:



        if self._auto_is_subscriber:
            return False
        env = getattr(self, "_quota_envelopes", None)
        if env is not None and env.has_km2_gauge():



            left = (env.km2_remaining if env.km2_remaining is not None
                    else max(0.0, env.km2_cap - env.km2_used))
            return left <= 0
        return self._auto_free_left is not None and self._auto_free_left <= 0

    def _update_auto_detect_enabled(self):





        self._apply_auto_run_block_takeover()


        self._sync_update_card_for_work()
        if not self._plugin_activated or self._auto_review_active:
            self.auto_detect_btn.setEnabled(False)



            self.auto_detect_btn.setToolTip(
                tr("Sign in to run Automatic.") if not self._plugin_activated
                else tr("Export or exit the review first."))
            return
        from ...core.detect_gate import can_detect
        has_layer = self.auto_layer_combo.currentLayer() is not None




        has_text = bool(self.auto_prompt_input.text().strip())
        positives = self._auto_positive_exemplars if self._EXEMPLARS_ENABLED else 0




        has_object = has_text or positives > 0
        can_run = can_detect(has_text, positives)
        self._apply_auto_detail_gate(has_object)
        not_too_large = not self._auto_zone_too_large



        credits_ok = not self._auto_balance_spent()



        km2_ok = not getattr(self, "_auto_km2_exceeded", False)






        auto_available = self._auto_service_available()
        hard_ok = has_layer and not_too_large and credits_ok
        hard_ok = hard_ok and km2_ok and auto_available
        run_allowed = hard_ok and can_run





        lookup_pending = getattr(self, "_prompt_lookup_key_pending", None) is not None
        self.auto_detect_btn.setEnabled(
            run_allowed and not self._auto_run_active and not lookup_pending)




        if not auto_available:
            tip = tr("Automatic is temporarily unavailable. Try again later.")
        elif not has_layer:
            tip = tr("Pick a raster layer at the top of the panel first.")
        elif self._auto_zone_too_large:
            tip = tr("This zone at this precision is too big for one run. "
                     "Draw a smaller zone, or lower the precision.")
        elif not credits_ok:



            tip = tr("You used your Automatic allowance for this month. "
                     "Semi-Auto on your computer keeps working, free, with no "
                     "counter.")
        elif not km2_ok:
            tip = tr("This zone is larger than the surface you have left this "
                     "month. Draw a smaller zone.")
        elif hard_ok and not can_run:
            tip = tr("Type what to find, or draw an example of it.")
        elif lookup_pending:
            tip = self._prompt_lookup_note()
        else:
            tip = ""
        self.auto_detect_btn.setToolTip(tip)




        self._note_detect_blocked(self._auto_run_block_reason())
