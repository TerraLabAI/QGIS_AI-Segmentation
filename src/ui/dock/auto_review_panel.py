






from __future__ import annotations

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAbstractButton, QApplication

from ...core.boundary_snap import snap_default_enabled
from ...core.i18n import tr
from ...core.review_defaults import (
    AUTO_REVIEW_CLEAN_DEFAULT as _AUTO_REVIEW_CLEAN_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_EXPAND_DEFAULT as _AUTO_REVIEW_EXPAND_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_FILL_HOLES_DEFAULT as _AUTO_REVIEW_FILL_HOLES_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT as _AUTO_REVIEW_FILL_MAX_M2_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_ORTHO_DEFAULT as _AUTO_REVIEW_ORTHO_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_POINTS_PCT_DEFAULT as _AUTO_REVIEW_POINTS_PCT_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_SIMPLIFY_DEFAULT as _AUTO_REVIEW_SIMPLIFY_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_SMOOTH_DEFAULT as _AUTO_REVIEW_SMOOTH_DEFAULT,
)
from ...core.server_dials import dial_in_range
from ...core.shape_policy_dials import auto_review_points_pct_default
from .auto_flow_look import _BTN_AUTO_QUIET
from .auto_review_build import _BTN_LINK_CONFIRM, _export_btn_label
from .styles import (
    _snap_review_conf,
)




_CONF_PREVIEW_DEBOUNCE_MS = 40
_CONF_REFILTER_DRAG_DEBOUNCE_MS = 250


_CONF_REFILTER_TYPE_DEBOUNCE_MS = 200


class DockAutoReviewPanelMixin:







    _PRE_REVIEW_ROWS = ("auto_detail_row",)

    def _remember_pre_review_visibility(self) -> None:


        store: dict[str, bool] = {}
        for name in self._PRE_REVIEW_ROWS:
            widget = getattr(self, name, None)
            try:
                store[name] = bool(widget is not None and widget.isVisible())
            except (RuntimeError, AttributeError):
                store[name] = False
        self._pre_review_visibility = store

    def _pre_review_visible(self, name: str) -> bool:

        store = getattr(self, "_pre_review_visibility", None)
        if isinstance(store, dict) and name in store:
            return bool(store[name])
        widget = getattr(self, name, None)
        try:
            return bool(widget is not None and widget.isVisible())
        except (RuntimeError, AttributeError):
            return False

    def set_auto_review_active(self, active: bool, count: int = 0,
                               reset_controls: bool = True,
                               preset: dict | None = None) -> None:














        if active and not bool(getattr(self, "_auto_review_active", False)):
            self._remember_pre_review_visibility()
        self._auto_review_active = active



        self._auto_finalizing = False



        if not active or reset_controls:
            self.set_auto_review_installing(False)




            self.set_boundary_snap_offered(False)
        if active:


            self._offer_right_angles_availability()
        if active and reset_controls:

            self._review_shape_tracked = set()



            self.reset_review_steps()





        if not self._refine_handoff:
            self.mode_switch.setEnabled(not active)
            self.mode_switch.setToolTip(


                tr("Export or exit the review to switch modes.") if active else "")
        self.auto_review_panel.setVisible(active)
        if active:



            self._clear_review_spin_focus()



            from ...core.qt_compat import safe_single_shot
            safe_single_shot(0, self, self._clear_review_spin_focus)


        self.auto_review_view_row.setVisible(
            active and not getattr(self, "_auto_zero_entry", False))





        try:
            if active:
                self.auto_upsell_card.setVisible(False)
                self.auto_controls_section.setVisible(True)
            else:
                self._update_auto_page_state()
        except (RuntimeError, AttributeError):
            pass




        self.auto_layer_combo.setVisible(not active)




        self.auto_prompt_card.setVisible(not active)
        try:
            self.auto_run_summary_card.setVisible(False)
        except (RuntimeError, AttributeError):
            pass
        self.auto_detect_row.setVisible(not active)
        self.auto_exemplar_panel.setVisible(not active)




        self.auto_detail_row.setVisible(
            not active and self._pre_review_visible("auto_detail_row"))


        self.auto_settings_box.setVisible(False)


        if active:
            self.auto_credit_cost_label.setVisible(False)
        else:
            self._refresh_auto_cost_label()



        if active:


            self.set_auto_status("idle")
            if reset_controls:




                p = preset or {}
                for w in (self.auto_min_size_spin, self.auto_max_size_spin):
                    w.blockSignals(True)
                self.auto_min_size_spin.setValue(float(p.get("min_size_m2", 0.0)))
                self.auto_max_size_spin.setValue(float(p.get("max_size_m2", 0.0)))
                for w in (self.auto_min_size_spin, self.auto_max_size_spin):
                    w.blockSignals(False)






                for w in (self.auto_points_spin, self.auto_simplify_spin,
                          self.auto_round_corners_check,
                          self.auto_expand_spin, self.auto_fill_holes_check,
                          self.auto_fill_max_spin,
                          self.auto_clean_spin, self.auto_ortho_check):
                    w.blockSignals(True)
                self.auto_points_spin.setValue(
                    int(p.get("points_pct", auto_review_points_pct_default(_AUTO_REVIEW_POINTS_PCT_DEFAULT))))
                self.auto_simplify_spin.setValue(
                    float(p.get("simplify_px", _AUTO_REVIEW_SIMPLIFY_DEFAULT)))
                self.auto_round_corners_check.setChecked(
                    bool(p.get("smooth", _AUTO_REVIEW_SMOOTH_DEFAULT)))
                self.auto_expand_spin.setValue(
                    int(p.get("expand_px", _AUTO_REVIEW_EXPAND_DEFAULT)))
                self.auto_fill_holes_check.setChecked(
                    bool(p.get("fill_holes", _AUTO_REVIEW_FILL_HOLES_DEFAULT)))
                self.auto_fill_max_spin.setValue(max(0.0, float(
                    p.get("fill_holes_max_m2", _AUTO_REVIEW_FILL_MAX_M2_DEFAULT))))
                self.auto_clean_spin.setValue(
                    float(p.get("clean_px", _AUTO_REVIEW_CLEAN_DEFAULT)))
                self.auto_ortho_check.setChecked(
                    bool(p.get("ortho", _AUTO_REVIEW_ORTHO_DEFAULT)))
                for w in (self.auto_points_spin, self.auto_simplify_spin,
                          self.auto_round_corners_check,
                          self.auto_expand_spin, self.auto_fill_holes_check,
                          self.auto_fill_max_spin,
                          self.auto_clean_spin, self.auto_ortho_check):
                    w.blockSignals(False)


                self._sync_auto_fill_max_row()
                self._sync_auto_right_angle_controls()
                was_blocked = self.auto_boundary_snap_check.blockSignals(True)
                self.auto_boundary_snap_check.setChecked(
                    bool(p.get("snap_boundaries", snap_default_enabled())))
                self.auto_boundary_snap_check.blockSignals(was_blocked)


                self.auto_show_tiles_check.blockSignals(True)
                self.auto_show_tiles_check.setChecked(False)
                self.auto_show_tiles_check.blockSignals(False)



                pct = _snap_review_conf(int(round(self.auto_confidence_spin.value() * 100)))



                self._review_conf_seeded_pct = pct
                self.auto_review_confidence_slider.blockSignals(True)
                self.auto_review_confidence_slider.setValue(pct)
                self.auto_review_confidence_slider.blockSignals(False)
                self.auto_review_confidence_spin.blockSignals(True)
                self.auto_review_confidence_spin.setValue(pct)
                self.auto_review_confidence_spin.blockSignals(False)



        if active and not reset_controls:



            self.set_auto_review_step(1)
        self._update_auto_detect_enabled()

    def _clear_review_spin_focus(self) -> None:


        try:
            spin = self.auto_review_confidence_spin
            if spin.hasFocus():
                spin.clearFocus()
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def set_closed_canopy_advice(self, on: bool) -> None:


        from .guidance import HINT_REVIEW_CLOSED_CANOPY, is_hint_dismissed
        try:
            self.auto_closed_canopy_hint.setVisible(
                bool(on) and not is_hint_dismissed(HINT_REVIEW_CLOSED_CANOPY))
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def set_free_zone_fit_offer(self, requested_km2: float | None,
                                processed_km2: float | None) -> None:





        card = getattr(self, "auto_review_free_fit_card", None)
        if card is None:
            return
        try:
            from .pro_nudges import PRO_CARD_FREE_ZONE_CLIPPED, pro_card_dismissed
            if (requested_km2 is None or processed_km2 is None
                    or pro_card_dismissed(PRO_CARD_FREE_ZONE_CLIPPED)):
                card.setVisible(False)
                return
            from ...core.server_dials import dial_copy
            from .ui_refresh_credits import format_km2_surface
            title = dial_copy("review.free_fit_title", tr(
                "Free allowance used: {done} of {zone} km² processed"))
            title = (title.replace("{done}", format_km2_surface(processed_km2))
                     .replace("{zone}", format_km2_surface(requested_km2)))
            left_out = max(0.0, float(requested_km2) - float(processed_km2))
            note = tr("Not processed: {x} km², dashed on the map").replace(
                "{x}", format_km2_surface(left_out))
            body = dial_copy("zone.free_cap_body", tr(
                "Pro has no size limit and runs the zone as you drew it."))
            from .upsell_card import keep_working_cta
            card.set_text(title, body, keep_working_cta(), note=note)
            card.set_pro_offer("plugin_free_zone_clipped")
            card.enable_dismiss()
            card.setVisible(True)
        except (RuntimeError, AttributeError):
            return
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_viewed(
                trigger="free_zone_clipped")
        except Exception:
            pass  # nosec B110

    def _on_free_zone_fit_offer_dismissed(self) -> None:
        from .pro_nudges import PRO_CARD_FREE_ZONE_CLIPPED
        self._dismiss_pro_card(
            PRO_CARD_FREE_ZONE_CLIPPED,
            getattr(self, "auto_review_free_fit_card", None))

    def _on_free_zone_fit_offer_clicked(self) -> None:

        from ...core.pro_page_link import open_pro_page
        open_pro_page("plugin_free_zone_clipped", "free_zone_clipped",
                      parent=self)

    def set_auto_review_score_useful(self, useful: bool) -> None:









        from qgis.PyQt.QtCore import Qt

        from .guidance import HINT_REVIEW_CONFIDENCE, is_hint_dismissed
        from .styles import _msg_label_qss, msg_rich
        try:
            for widget in (self.auto_review_confidence_header,
                           self.auto_conf_histogram,
                           self.auto_review_confidence_slider,
                           self.auto_review_confidence_ends):
                widget.setVisible(useful)


            self.auto_confidence_hint.setVisible(
                useful and not is_hint_dismissed(HINT_REVIEW_CONFIDENCE))
            if not useful:
                self.auto_review_flat_score_note.setStyleSheet(
                    _msg_label_qss("info"))
                self.auto_review_flat_score_note.setTextFormat(Qt.TextFormat.RichText)
                self.auto_review_flat_score_note.setText(msg_rich("info", tr(
                    "This model rates every object the same, so filtering by "
                    "confidence would show all of them or none. Use Size "
                    "below, or fix objects in the next step.")))
            self.auto_review_flat_score_note.setVisible(not useful)




            combo = getattr(self, "auto_display_combo", None)
            if combo is not None:
                idx = combo.findData("confidence")
                if idx >= 0:
                    combo.view().setRowHidden(idx, not useful)
                    item = combo.model().item(idx)
                    if item is not None:
                        item.setEnabled(useful)
                    if not useful and combo.currentData() == "confidence":
                        combo.setCurrentIndex(max(0, combo.findData("random")))
        except (RuntimeError, AttributeError):

            pass

    def _format_auto_review_count(self, visible: int, total: int, pct: int,
                                  bound: str = "confidence") -> str:


        from .review_count_line import format_review_count_line

        return format_review_count_line(visible, total, pct, bound)

    def set_auto_export_saving(self, saving: bool) -> None:









        btn = getattr(self, "auto_export_btn", None)
        if btn is None:
            return
        try:
            if saving:
                self._auto_export_label_before_save = btn.text()
                btn.setText(tr("Saving..."))
                btn.setEnabled(False)
            else:
                held = getattr(self, "_auto_export_label_before_save", "")
                if held:
                    btn.setText(held)
                self._auto_export_label_before_save = ""



                btn.setEnabled(
                    int(getattr(self, "_auto_review_visible_count", 1) or 0) > 0
                    and not self.review_install_locked())


            btn.repaint()
        except (RuntimeError, AttributeError):
            pass

    def _on_auto_review_reveal_clicked(self) -> None:







        bound = getattr(self, "_auto_review_hiding_bound", "confidence")
        try:
            if bound == "min":
                self.auto_min_size_spin.setValue(self.auto_min_size_spin.minimum())
            elif bound == "max":


                self.auto_max_size_spin.setValue(0.0)
            else:
                spin = self.auto_review_confidence_spin
                spin.setValue(spin.minimum())
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def update_auto_review_count(self, visible: int, total: int, pct: int,
                                 bound: str = "confidence") -> None:








        self._auto_review_visible_count = int(visible)
        try:
            self._auto_review_count_label.setText(
                self._format_auto_review_count(visible, total, pct, bound))
            self.auto_export_btn.setText(_export_btn_label(visible))


            self.auto_export_btn.setEnabled(
                visible > 0 and not self.review_install_locked())


            self._auto_review_hiding_bound = bound
            reveal = getattr(self, "auto_review_reveal_btn", None)
            if reveal is not None:
                reveal.setVisible(visible == 0 and total > 0)
            if visible == 0:
                if bound == "min":
                    tip = tr("Lower the Min size filter to show objects first.")
                elif bound == "max":
                    tip = tr("Raise the Max size filter to show objects first.")
                else:
                    tip = tr("Lower Confidence to show objects first.")
            else:


                tip = tr("Save the {visible} polygons shown as a layer.").format(
                    visible=visible)
            self.auto_export_btn.setToolTip(tip)





            if total > 0 and getattr(self, "_auto_zero_entry", False):
                self.set_zero_detection_entry(False)
                self.set_auto_review_step(
                    getattr(self, "_auto_review_step", 1))
        except (RuntimeError, AttributeError):
            pass



    def _on_conf_slider_moved(self, value: int) -> None:








        snapped = _snap_review_conf(
            value, floor=getattr(self, "_review_conf_floor_pct", None))
        if snapped != value:
            self.auto_review_confidence_slider.blockSignals(True)
            self.auto_review_confidence_slider.setValue(snapped)
            self.auto_review_confidence_slider.blockSignals(False)
        if self.auto_review_confidence_spin.value() != snapped:
            self.auto_review_confidence_spin.blockSignals(True)
            self.auto_review_confidence_spin.setValue(snapped)
            self.auto_review_confidence_spin.blockSignals(False)

        if getattr(self, "auto_conf_histogram", None) is not None:
            self.auto_conf_histogram.set_cutoff(snapped / 100.0)




        self._auto_conf_preview_timer.start(dial_in_range(
            "tuning.review.conf_preview_debounce_ms",
            _CONF_PREVIEW_DEBOUNCE_MS, 10, 500))
        self._auto_conf_debounce_timer.start(dial_in_range(
            "tuning.review.conf_refilter_drag_debounce_ms",
            _CONF_REFILTER_DRAG_DEBOUNCE_MS, 50, 2000))

    def _emit_auto_confidence_preview(self) -> None:
        self.auto_review_confidence_preview.emit(self.auto_review_confidence_spin.value())

    def _on_conf_spin_changed(self, value: int) -> None:

        if self.auto_review_confidence_slider.value() != value:
            self.auto_review_confidence_slider.blockSignals(True)
            self.auto_review_confidence_slider.setValue(value)
            self.auto_review_confidence_slider.blockSignals(False)


        if getattr(self, "auto_conf_histogram", None) is not None:
            self.auto_conf_histogram.set_cutoff(value / 100.0)
        self._schedule_conf_refilter()

    def seed_review_confidence(self, pct: int) -> None:











        try:
            value = int(pct)
            self._review_conf_seeded_pct = value
            self.auto_review_confidence_slider.blockSignals(True)
            self.auto_review_confidence_slider.setValue(value)
            self.auto_review_confidence_slider.blockSignals(False)
            self.auto_review_confidence_spin.blockSignals(True)
            self.auto_review_confidence_spin.setValue(value)
            self.auto_review_confidence_spin.blockSignals(False)
        except (RuntimeError, AttributeError):

            pass

    def set_review_conf_floor(self, floor_pct: int) -> None:











        try:
            floor = max(0, int(floor_pct))
        except (TypeError, ValueError):
            return
        self._review_conf_floor_pct = floor
        try:
            self.auto_review_confidence_slider.setMinimum(floor)
            self.auto_review_confidence_spin.setMinimum(floor)
        except (RuntimeError, AttributeError):
            return



        seeded = getattr(self, "_review_conf_seeded_pct", None)
        if seeded is not None and seeded >= floor:
            self.seed_review_confidence(seeded)

    def _schedule_conf_refilter(self) -> None:

        self._auto_conf_debounce_timer.start(dial_in_range(
            "tuning.review.conf_refilter_type_debounce_ms",
            _CONF_REFILTER_TYPE_DEBOUNCE_MS, 50, 2000))

    def _emit_auto_confidence_changed(self) -> None:



        self.auto_review_confidence_changed.emit(self.auto_review_confidence_spin.value())

    def _on_shape_control_changed(self, control: str, value) -> None:



        self.auto_refine_changed.emit()
        try:
            tracked = getattr(self, "_review_shape_tracked", None)
            if tracked is None:
                tracked = set()
                self._review_shape_tracked = tracked
            if control not in tracked:
                tracked.add(control)
                from ...core import telemetry, telemetry_run_events


                telemetry_run_events.track_review_shape_adjusted(
                    control=control, value=value,
                    run_id=telemetry.get_last_run_id() or "")
        except Exception:
            pass  # nosec B110

    def _offer_right_angles_availability(self) -> None:


        from .right_angles_support import offer_right_angles_availability

        offer_right_angles_availability(self)

    def _sync_auto_right_angle_controls(self) -> None:


        from .right_angles_support import sync_right_angle_conflicts

        sync_right_angle_conflicts(self)


        try:
            ortho_on = bool(self.auto_ortho_check.isChecked())
            self.auto_round_corners_row.setVisible(not ortho_on)
        except (RuntimeError, AttributeError):
            pass

    def set_boundary_snap_offered(self, offered: bool) -> None:







        self._auto_boundary_snap_offered = bool(offered)
        try:
            if not offered and self.auto_boundary_snap_check.isChecked():
                self.auto_boundary_snap_check.blockSignals(True)
                self.auto_boundary_snap_check.setChecked(False)
                self.auto_boundary_snap_check.blockSignals(False)
            self.auto_boundary_snap_row.setVisible(bool(offered))
        except (RuntimeError, AttributeError):

            pass

    def set_review_busy(self, busy: bool) -> None:


        from .review_card_rows import set_review_busy

        set_review_busy(self, busy)

    def set_boundary_snap_notice(self, reason: str) -> None:


        from .review_card_rows import set_boundary_snap_notice

        set_boundary_snap_notice(self, reason)

    def get_auto_boundary_snap(self) -> bool:






        try:
            offered = bool(getattr(self, "_auto_boundary_snap_offered", False))
            return offered and bool(self.auto_boundary_snap_check.isChecked())
        except (RuntimeError, AttributeError):
            return False





    def reset_review_steps(self) -> None:



        try:
            self._qgis_bridge_active_ui = False




            from .correct_method_default import correct_default_method
            self.set_correct_method(correct_default_method())
            self.set_correct_selection(0)
            self.set_correct_session_active(False)
            self.set_correct_armed(None)
            self.set_correct_status("neutral", "")
            self.set_correction_summary(0)
            self.set_retry_confirm_pending(False)
            self.set_zero_detection_entry(False)
            self.set_auto_review_step(0)
        except (RuntimeError, AttributeError):
            pass

    def set_auto_review_step(self, step: int) -> None:







        step = max(0, min(2, int(step)))
        self._auto_review_step = step
        try:
            self.auto_review_step_stack.setCurrentIndex(step)



            _focus = QApplication.focusWidget()
            if isinstance(_focus, QAbstractButton) and self.isAncestorOf(_focus):
                _focus.clearFocus()
            for i in range(3):
                if i == step:
                    state = "active"
                elif i < step and not (i == 0 and self._auto_zero_entry):
                    state = "done"
                else:


                    state = "todo"
                self._set_review_dial(i, state)




            self._set_review_dials_locked(self.review_install_locked(), step)




            self.auto_review_view_row.setVisible(
                bool(getattr(self, "_auto_review_active", False))
                and not getattr(self, "_auto_zero_entry", False))
            btn = self.auto_step_next_btn













            if step == 0:
                btn.setText(tr("Next: {step}").format(step=tr("Correct")))
            elif step == 1:
                btn.setText(tr("Next: {step}").format(step=tr("Shapes")))



            self._apply_step_next_visibility()


            self.auto_export_btn.setVisible(
                step == 2 and not self._auto_zero_entry)
            self._apply_review_links(step)
            self._refresh_correct_panels()
            self._keep_review_primary_in_view()
        except (RuntimeError, AttributeError):
            pass

    def _apply_step_next_visibility(self) -> None:













        try:
            step = int(getattr(self, "_auto_review_step", 0))
            visible = step in (0, 1)
            visible = visible and not bool(getattr(self, "_auto_zero_entry", False))
            visible = visible and not bool(
                getattr(self, "_auto_correct_session_active", False))
            self.auto_step_next_btn.setVisible(visible)
        except (RuntimeError, AttributeError):
            pass

    def _keep_review_primary_in_view(self) -> None:






        from ...core.qt_compat import safe_single_shot

        btn = (self.auto_export_btn
               if getattr(self, "_auto_review_step", 0) == 2
               else self.auto_step_next_btn)
        if btn.isHidden():
            return


        safe_single_shot(0, self, lambda: self._scroll_review_primary(btn))

    def _scroll_review_primary(self, btn) -> None:
        try:
            if btn.isHidden():
                return
            self._dock_scroll_area.ensureWidgetVisible(btn, 0, 8)
        except (RuntimeError, AttributeError):
            pass

    def _on_auto_step_next_clicked(self) -> None:


        self.auto_review_step_requested.emit(
            min(2, getattr(self, "_auto_review_step", 0) + 1))

    def _apply_review_links(self, step: int) -> None:


        try:
            self.auto_retry_btn.setVisible(True)
            self._auto_review_links_sep.setVisible(True)
            self.auto_review_exit_btn.setVisible(True)
        except (RuntimeError, AttributeError):
            pass

    def set_retry_confirm_pending(self, pending: bool) -> None:



        try:
            btn = self.auto_retry_btn
            from ..icons import icon_for
            if pending:
                btn.setText(
                    tr("Discard reviewed results and run again? Confirm"))
                btn.setStyleSheet(_BTN_LINK_CONFIRM)


                btn.setIcon(QIcon())
            else:
                btn.setText(tr("Re-run the whole zone"))
                btn.setStyleSheet(_BTN_AUTO_QUIET)
                btn.setIcon(icon_for(btn, "refresh", 14))
        except (RuntimeError, AttributeError):
            pass

    def set_auto_display_mode(self, mode: str) -> None:




        combo = getattr(self, "auto_display_combo", None)
        if combo is None:
            return
        idx = combo.findData(mode)
        if idx < 0:
            return
        combo.blockSignals(True)
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)

        from .review_view_block import sync_display_legend
        sync_display_legend(self)

    def set_display_legend(self, text: str) -> None:

        legend = getattr(self, "auto_display_legend", None)
        if legend is not None:
            legend.setText(text)
