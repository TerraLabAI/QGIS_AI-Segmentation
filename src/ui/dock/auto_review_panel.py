"""The post-run review panel: its own on/off, the confidence and shape
filters, the count readout, the three-step ladder and the display mode.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

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
from ...core.shape_policy_dials import auto_review_points_pct_default
from .auto_review_build import _BTN_LINK_CONFIRM, _export_btn_label
from .styles import (
    _BTN_LINK_STRONG,
    _snap_review_conf,
)


class DockAutoReviewPanelMixin:
    """The post-run review panel: its own on/off, the confidence and shape
    filters, the count readout, the three-step ladder and the display mode."""

    # set_auto_review_installing lives in dock/install_lock.py: the banner now
    # holds the review still while it runs, which is a concern of its own.

    # Rows the review hides on the way in and owes back on the way out.
    _PRE_REVIEW_ROWS = ("auto_detail_row", "auto_credit_cost_label")

    def _remember_pre_review_visibility(self) -> None:
        """Record what the pre-run rows were showing before the review took
        them."""
        store: dict[str, bool] = {}
        for name in self._PRE_REVIEW_ROWS:
            widget = getattr(self, name, None)
            try:
                store[name] = bool(widget is not None and widget.isVisible())
            except (RuntimeError, AttributeError):
                store[name] = False
        self._pre_review_visibility = store

    def _pre_review_visible(self, name: str) -> bool:
        """What one pre-run row was showing before the review opened."""
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
        """Post-run review state: refine + Finish replace the Detect/Exit row.

        reset_controls=False keeps the current filter values: used when returning
        from a Manual refine handoff, where a reset would wipe the locked
        confidence and any size filter the user set before handing off.

        preset: the run's smart review defaults (core.review_presets dict,
        keyed like the controls). When given with reset_controls, the shape
        refine + Min size seed from it instead of the static review_defaults
        constants, so the first result is already tuned to the prompt and the
        run resolution. Every NEW review reseeds (no cross-run memory).
        """
        # Snapshot the rows the review is about to take, BEFORE anything here
        # hides them, and only on the way in.
        if active and not bool(getattr(self, "_auto_review_active", False)):
            self._remember_pre_review_visibility()
        self._auto_review_active = active
        # The review is the end of the hand-over the run card was holding the
        # screen for (see set_auto_finalizing): release it here, BEFORE the
        # idle status below, which is what finally clears that card.
        self._auto_finalizing = False
        # A fresh review or a review teardown ends any prior background-install
        # banner (D1); a refine-handoff return (reset_controls=False) is left as
        # is (its env is already ready, no install running).
        if not active or reset_controls:
            self.set_auto_review_installing(False)
            # Shared borders starts hidden and OFF on every new review and on
            # teardown; the plugin re-offers it right after when this run's
            # object is land cover. A refine-handoff return (reset_controls
            # False) keeps whatever the user had.
            self.set_boundary_snap_offered(False)
        if active:
            # Answer the Right angles availability question now, once, rather
            # than waiting for the user to tick a box that may have no engine.
            self._offer_right_angles_availability()
        if active and reset_controls:
            # Fresh review: reset the per-control shape-adjust telemetry dedup set.
            self._review_shape_tracked = set()
            # The linear ladder restarts on Keep with every correction
            # surface cleared; the plugin re-applies zero-entry / queue /
            # Manual lock right after when they apply.
            self.reset_review_steps()
        # The top mode toggle is disabled during review: switching modes here is
        # the destructive path that discards (and red-autosaves) the review. The
        # only intended door to Manual is the "Refine in Manual mode" button,
        # which preserves the detections. (The handoff itself manages the toggle
        # separately, so don't fight it while a handoff is live.)
        if not self._refine_handoff:
            self.mode_switch.setEnabled(not active)
            self.mode_switch.setToolTip(
                tr("Finish or exit the review to switch modes.") if active else "")
        self.auto_review_panel.setVisible(active)
        # The View-as block sits at the dock bottom, outside the panel, so it
        # no longer inherits the panel's visibility: drive it here.
        self.auto_review_view_row.setVisible(active)
        # Credits can hit zero DURING the run, and the paywall swap hides the
        # section this panel is built inside. Claim the page as the review
        # opens rather than waiting for the next full refresh, or the user
        # meets a price list where their paid results should be. On the way
        # out, the full refresh puts the paywall back if it is owed.
        try:
            if active:
                self.auto_upsell_card.setVisible(False)
                self.auto_controls_section.setVisible(True)
            else:
                self._update_auto_page_state()
        except (RuntimeError, AttributeError):
            pass
        # The locked, greyed layer header is dead weight during review (the
        # raster cannot change until the review ends anyway): hide it so the
        # result + filters own the panel. Restored on review end and, for hard
        # teardowns that skip this call, by reset_auto_to_start.
        self.auto_layer_combo.setVisible(not active)
        # Hide the prompt card (Describe what to find + Library) during review:
        # the search is done, so the result + filters should own the panel. The
        # in-run receipt goes too, and does not come back: the review names
        # what was found, which is a better answer than what was asked.
        self.auto_prompt_card.setVisible(not active)
        try:
            self.auto_run_summary_card.setVisible(False)
        except (RuntimeError, AttributeError):
            pass
        self.auto_detect_row.setVisible(not active)
        self.auto_exemplar_panel.setVisible(not active)
        # Two rows come and go with the pre-run state, so the review gives back
        # what it took: reading isVisible() on the way out always answers
        # False, which made the restore a permanent no-op and lost the row for
        # the rest of the session.
        self.auto_detail_row.setVisible(
            not active and self._pre_review_visible("auto_detail_row"))
        # The pre-run confidence box is always hidden now (confidence lives in
        # the review panel below); keep it hidden here too.
        self.auto_settings_box.setVisible(False)
        # The run is paid for: the cost estimate is pre-run info that only
        # confuses the review. (The locked layer header keeps naming the raster.)
        self.auto_credit_cost_label.setVisible(
            not active and self._pre_review_visible("auto_credit_cost_label"))
        # Going back mid-review would silently desync the review from the
        # inputs; the exits (Finish, zone x / Escape, mode switch) all discard
        # or commit the review explicitly.
        if active:
            # Clear any leftover run status (tile progress / info banner) as
            # the review opens so it never lingers next to the result count.
            self.set_auto_status("idle")
            if reset_controls:
                # Fresh review: size filters seed from the run's smart preset
                # (prompt-aware Min size floor; no preset = neutral 0). block-
                # Signals avoids debounced preview refreshes; the plugin renders
                # the preview right after.
                p = preset or {}
                for w in (self.auto_min_size_spin, self.auto_max_size_spin):
                    w.blockSignals(True)
                self.auto_min_size_spin.setValue(float(p.get("min_size_m2", 0.0)))
                self.auto_max_size_spin.setValue(0)
                for w in (self.auto_min_size_spin, self.auto_max_size_spin):
                    w.blockSignals(False)
                # Shape refine seeds from the same preset (Right angles + Fill
                # holes for buildings, Round corners for vegetation, ...), with
                # the static faithful defaults as fallback. blockSignals so this
                # seed never fires the debounced re-derive. Points opens at the
                # class default (100%, the preset's own spacing does the
                # thinning); Simplify seeds from the preset's small de-noise.
                for w in (self.auto_points_spin, self.auto_simplify_spin,
                          self.auto_round_corners_check,
                          self.auto_expand_spin, self.auto_fill_holes_check,
                          self.auto_fill_max_spin,
                          self.auto_clean_spin, self.auto_ortho_check):
                    w.blockSignals(True)
                self.auto_points_spin.setValue(
                    auto_review_points_pct_default(_AUTO_REVIEW_POINTS_PCT_DEFAULT))
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
                # blockSignals swallowed the checkbox signal that drives the
                # threshold row's visibility; restate it from the seeded value.
                self._sync_auto_fill_max_row()
                self._sync_auto_right_angle_controls()
                # Debug tile overlay is off by default each new review (signal-free:
                # the plugin clears the grid when the run started).
                self.auto_show_tiles_check.blockSignals(True)
                self.auto_show_tiles_check.setChecked(False)
                self.auto_show_tiles_check.blockSignals(False)
                # Seed the review confidence slider from the pre-run dial so the
                # starting filter matches what the run used (no signal: the plugin
                # renders the preview right after).
                pct = _snap_review_conf(int(round(self.auto_confidence_spin.value() * 100)))
                # Remembered like any seed: the previous run's floor is still on
                # the controls here, so this value can be clamped on the way in
                # and needs re-applying once this run's floor lands.
                self._review_conf_seeded_pct = pct
                self.auto_review_confidence_slider.blockSignals(True)
                self.auto_review_confidence_slider.setValue(pct)
                self.auto_review_confidence_slider.blockSignals(False)
                self.auto_review_confidence_spin.blockSignals(True)
                self.auto_review_confidence_spin.setValue(pct)
                self.auto_review_confidence_spin.blockSignals(False)
            # The header + Export label are set by the plugin's first
            # update_auto_review_count call (right after this, via the review
            # preview push), so both counts and the pct are always consistent.
        if active and not reset_controls:
            # Returning from a Refine-in-Manual handoff lands on the Correct
            # step (step 1): that is where the hand-edit escalation lives, and
            # the dials reach Shapes/Export freely from there.
            self.set_auto_review_step(1)
        self._update_auto_detect_enabled()

    def set_closed_canopy_advice(self, on: bool) -> None:
        """Show the closed-forest card for this review, or hide it. A user
        who closed the card once never sees it again (dismissal is kept)."""
        from .guidance import HINT_REVIEW_CLOSED_CANOPY, is_hint_dismissed
        try:
            self.auto_closed_canopy_hint.setVisible(
                bool(on) and not is_hint_dismissed(HINT_REVIEW_CLOSED_CANOPY))
        except (RuntimeError, AttributeError):
            pass  # nosec B110 - a review page mid-teardown shows nothing

    def set_auto_review_score_useful(self, useful: bool) -> None:
        """Show or drop the whole Confidence group on the Keep step.

        A run whose objects all came back rated the same carries no ranking to
        filter on: every cutoff under the shared score keeps all of them, and
        the next step up keeps none. The control is a cliff wearing the clothes
        of a dial, and the strip above it sells one bar as a distribution. So
        the group leaves the step and one line takes its place. Size stays: it
        is the filter that still means something there.
        """
        from .guidance import HINT_REVIEW_CONFIDENCE, is_hint_dismissed
        from .styles import _msg_label_qss, _msg_text
        try:
            for widget in (self.auto_review_confidence_header,
                           self.auto_conf_histogram,
                           self.auto_review_confidence_slider,
                           self.auto_review_confidence_ends):
                widget.setVisible(useful)
            # The tip is dismissible, so it may only come back for a user who
            # never closed it: a plain setVisible(True) would resurrect it.
            self.auto_confidence_hint.setVisible(
                useful and not is_hint_dismissed(HINT_REVIEW_CONFIDENCE))
            if not useful:
                self.auto_review_flat_score_note.setStyleSheet(
                    _msg_label_qss("info"))
                self.auto_review_flat_score_note.setText(_msg_text("info", tr(
                    "This model rates every object the same, so filtering by "
                    "confidence would show all of them or none. Use Size "
                    "below, or fix objects in the next step.")))
            self.auto_review_flat_score_note.setVisible(not useful)
            # The View-detections-as combo offers a Confidence heatmap too.
            # Painting one colour on a run with no ranking to show is the
            # same dead-control problem as the slider above, so the item
            # leaves the list rather than sitting there disabled.
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
        """ONE compact review readout line, always honest. Body in
        review_count_line."""
        from .review_count_line import format_review_count_line

        return format_review_count_line(visible, total, pct, bound)

    def set_auto_export_saving(self, saving: bool) -> None:
        """Put the Export button in its saving state, and repaint it now.

        Everything the click does runs on the GUI thread: the shapes are
        checked, measured, written to a GeoPackage and drawn, and the map is
        held still for the redraw that follows. On a big run that is seconds
        with nothing moving on screen, which reads as a plugin that has
        stopped rather than one that is working. The button says what it is
        doing before the work starts, and refuses a second click while it runs.
        """
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
                btn.setEnabled(True)
            # The handler does not return to the event loop until the write is
            # done, so without this the new text is never painted.
            btn.repaint()
        except (RuntimeError, AttributeError):
            pass

    def update_auto_review_count(self, visible: int, total: int, pct: int,
                                 bound: str = "confidence") -> None:
        """Update the two-line review header + the Export button label after a
        live confidence re-filter. ``visible`` = objects shown now, ``total`` =
        objects the run found, ``pct`` = current confidence cutoff. ``bound``
        (only meaningful when visible == 0) names the filter that hides the
        objects, ``"confidence"``, ``"min"`` or ``"max"``, so the guidance sends
        the user to the dial that will reveal them."""
        # Remembered so the install lock can hand Export back to this rule
        # rather than to plain enabled when it releases.
        self._auto_review_visible_count = int(visible)
        try:
            self._auto_review_count_label.setText(
                self._format_auto_review_count(visible, total, pct, bound))
            self.auto_export_btn.setText(_export_btn_label(visible))
            # A local-AI install holds the review still, Export included: a
            # count refresh must not hand back a button the lock just took.
            self.auto_export_btn.setEnabled(
                visible > 0 and not self.review_install_locked())
            if visible == 0:
                if bound == "min":
                    tip = tr("Lower the Min size filter to show objects first.")
                elif bound == "max":
                    tip = tr("Raise the Max size filter to show objects first.")
                else:
                    tip = tr("Lower Confidence to show objects first.")
            else:
                # One short line: what the click does. The filtered-out
                # objects were the Keep step's call and are not restated here.
                tip = tr("Save the {visible} polygons shown as a layer.").format(
                    visible=visible)
            self.auto_export_btn.setToolTip(tip)
            # A run that found nothing hides both green primaries: there is
            # nothing to advance to or export. The moment the review holds an
            # object (the user drew one on the Correct step) that stops being
            # true, and without this the work just added could reach neither the
            # Shapes step nor Export.
            if total > 0 and getattr(self, "_auto_zero_entry", False):
                self.set_zero_detection_entry(False)
                self.set_auto_review_step(
                    getattr(self, "_auto_review_step", 1))
        except (RuntimeError, AttributeError):
            pass

    # -- Review confidence: slider <-> spinbox sync + debounced re-filter -----

    def _on_conf_slider_moved(self, value: int) -> None:
        """Slider dragged: snap to the nearest 5% stop, mirror into the spinbox
        only (cheap, no re-filter). QSlider singleStep only constrains keyboard
        input, so the mouse drag is snapped here to stop the handle (and the
        debounced re-filter on release) from landing on every tiny percentage.
        The spinbox keeps 1% precision; this snapped mirror just rounds it when
        the user drives from the slider. The snap floor follows the run (see
        set_review_conf_floor), so a run that keeps fainter detections than the
        design minimum can still be dragged back down to its own floor."""
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
        # A user-initiated move moves the histogram's dimmed/kept boundary.
        if getattr(self, "auto_conf_histogram", None) is not None:
            self.auto_conf_histogram.set_cutoff(snapped / 100.0)
        # Live feedback as the handle moves: a fast, cheap preview re-filter on a
        # short debounce, plus the accurate rebuild on a longer one so it also
        # runs after the user settles via the keyboard (no sliderReleased then).
        # A mouse release additionally triggers the accurate path immediately.
        self._auto_conf_preview_timer.start(40)
        self._auto_conf_debounce_timer.start(250)

    def _emit_auto_confidence_preview(self) -> None:
        self.auto_review_confidence_preview.emit(self.auto_review_confidence_spin.value())

    def _on_conf_spin_changed(self, value: int) -> None:
        """Spinbox edited: mirror into the slider, then schedule the re-filter."""
        if self.auto_review_confidence_slider.value() != value:
            self.auto_review_confidence_slider.blockSignals(True)
            self.auto_review_confidence_slider.setValue(value)
            self.auto_review_confidence_slider.blockSignals(False)
        # The slider mirror above is signal-blocked, so move the histogram's
        # kept/dimmed boundary here too (the slider path does it on its own move).
        if getattr(self, "auto_conf_histogram", None) is not None:
            self.auto_conf_histogram.set_cutoff(value / 100.0)
        self._schedule_conf_refilter()

    def seed_review_confidence(self, pct: int) -> None:
        """Signal-free mirror of the starting cutoff into the review slider and
        spinbox. The review page seeds its widgets from the pre-run dial when it
        opens, but the async finalize computes the real starting cutoff (class
        default / adaptive) AFTER that, so it pushes the final value here; the
        handle, the spin, the histogram boundary and the count line then all
        read the same number.

        The value is remembered: the finalize seeds BEFORE it clamps the
        controls to the run's noise floor, so a starting cutoff under the floor
        in force at that moment would be clamped away on the way in.
        set_review_conf_floor re-applies it once the controls can hold it."""
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
        """Clamp the review confidence controls to the run's noise floor:
        sub-floor detections are excluded from the review, so a cutoff under it
        would filter nothing.

        The floor moves BOTH ways. Above the design minimums it raises them as
        before. Below them it LOWERS them, because a run that keeps detections
        under the design minimum can open at a cutoff under it too: leaving the
        old floors in place showed a handle at 5% while the filter ran lower,
        and the value the user read was not the value applied. It only ever
        widens what the controls can reach downward, so the review still opens
        on the cutoff the run chose (never on zero shapes)."""
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
        # The seed lands before this clamp, so a starting cutoff the old floors
        # rejected is sitting on the controls as a clamped value. Now that they
        # reach lower, put the real one back.
        seeded = getattr(self, "_review_conf_seeded_pct", None)
        if seeded is not None and seeded >= floor:
            self.seed_review_confidence(seeded)

    def _schedule_conf_refilter(self) -> None:
        """Coalesce rapid confidence changes so the heavy re-merge runs once."""
        self._auto_conf_debounce_timer.start(200)

    def _emit_auto_confidence_changed(self) -> None:
        # Emit the spinbox value: it is the exact-cutoff source of truth (free 1%
        # precision). A slider drag has already snapped itself to a 5% step and
        # mirrored that into the spinbox, so this stays correct for both paths.
        self.auto_review_confidence_changed.emit(self.auto_review_confidence_spin.value())

    def _on_shape_control_changed(self, control: str, value) -> None:
        """A review shape-refine control changed: re-derive the visible set (via
        the debounced auto_refine_changed) and track the adjustment once per
        control per review."""
        self.auto_refine_changed.emit()
        try:
            tracked = getattr(self, "_review_shape_tracked", None)
            if tracked is None:
                tracked = set()
                self._review_shape_tracked = tracked
            if control not in tracked:
                tracked.add(control)
                from ...core import telemetry, telemetry_run_events
                # The dock does not hold the run id; the telemetry breadcrumb
                # does, and it is the run this review belongs to.
                telemetry_run_events.track_review_shape_adjusted(
                    control=control, value=value,
                    run_id=telemetry.get_last_run_id() or "")
        except Exception:
            pass  # nosec B110

    def _offer_right_angles_availability(self) -> None:
        """Ask once, when a review opens, whether Right angles can run at all.
        Body in right_angles_support."""
        from .right_angles_support import offer_right_angles_availability

        offer_right_angles_availability(self)

    def _sync_auto_right_angle_controls(self) -> None:
        """Make incompatible Shape controls unavailable with Right angles.
        Body in right_angles_support."""
        from .right_angles_support import sync_right_angle_conflicts

        sync_right_angle_conflicts(self)

    def set_boundary_snap_offered(self, offered: bool) -> None:
        """Show or hide the Keep step's shared-borders control.

        HIDDEN, never greyed: on a building or a car run the option does not
        exist at all (there the gap between two neighbours is real data), and
        it is out of reach on a result too big for one pass. Hiding it also
        turns it off, so a control the user cannot see never rewrites
        geometry. Driven by the plugin, which owns the land-cover gate."""
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
        """Show or hide the review's shape-pass line. Body in
        review_card_rows."""
        from .review_card_rows import set_review_busy

        set_review_busy(self, busy)

    def set_boundary_snap_notice(self, reason: str) -> None:
        """Say why shared borders did nothing on this set. Body in
        review_card_rows."""
        from .review_card_rows import set_boundary_snap_notice

        set_boundary_snap_notice(self, reason)

    def get_auto_boundary_snap(self) -> bool:
        """Whether the review should give neighbouring shapes shared borders.

        False whenever the control is not offered. Reads the stored offer flag,
        not the widget's visibility: the Keep page is hidden while the user
        stands on another step, and the setting has to survive that (the Export
        commits the same set it shows)."""
        try:
            offered = bool(getattr(self, "_auto_boundary_snap_offered", False))
            return offered and bool(self.auto_boundary_snap_check.isChecked())
        except (RuntimeError, AttributeError):
            return False

    # -- Linear review: step navigation + edit surfaces ---------------------
    # Store-only setters: the plugin owns every state machine (edit arming,
    # journal) and drives these.

    def reset_review_steps(self) -> None:
        """Restart the linear review ladder for a fresh review: Keep
        first, every edit surface cleared. The plugin re-applies the
        zero-detection entry right after when it applies."""
        try:
            self._qgis_bridge_active_ui = False
            # Re-read, never hardcode: the dock is built before the first
            # configuration fetch lands, and a sign-in or a finished install
            # since the last review changes which method is ready, so opening a
            # review is the moment the choice can actually take effect.
            from .correct_method_default import correct_default_method
            self.set_correct_method(correct_default_method())
            self.set_correct_selection(0)
            self.set_correct_session_active(False)
            self.set_correct_armed(None)
            self.set_correct_status("neutral", "")
            self.set_correction_summary(0)
            self.set_review_recap("")
            self.set_retry_confirm_pending(False)
            self.set_zero_detection_entry(False)
            self.set_auto_review_step(0)
        except (RuntimeError, AttributeError):
            pass

    def set_auto_review_step(self, step: int) -> None:
        """Land the review on one ladder step (0 Keep / 1 Correct /
        2 Shapes): stack page, dial states, the primary and the links.

        The ladder now matches the stack build order (Keep, Correct,
        Shapes), so the step index IS the page index. Correct comes before
        Shapes: you settle which objects exist before polishing how they
        look."""
        step = max(0, min(2, int(step)))
        self._auto_review_step = step
        try:
            self.auto_review_step_stack.setCurrentIndex(step)
            for i in range(3):
                if i < step or (i == 0 and self._auto_zero_entry):
                    state = "done"
                elif i == step:
                    state = "active"
                else:
                    state = "todo"
                self._set_review_dial(i, state)
            # Dials lock for one reason only: a local-AI install that owns the
            # review until it ends. The old Manual lock that dimmed them
            # guarded a paid re-detect batch that no longer exists (removed
            # 5706f2c); hand edits are protected by geometry, not by a lock.
            self._set_review_dials_locked(self.review_install_locked(), step)
            # The block is on every step, but only while a review is live: it
            # is a main_layout sibling now, so a step setter fired on a torn
            # down review would otherwise leave it alone above the footer.
            self.auto_review_view_row.setVisible(
                bool(getattr(self, "_auto_review_active", False)))
            btn = self.auto_step_next_btn
            # On a zero-detection run there is nothing to advance to, so the
            # green primary stays hidden on every step (not only on first
            # entry): re-clicking the Keep dial must not resurrect it.
            # "Next:" then what the NEXT step lets the user do, in that step's
            # own words. The bare verb phrase was dropped once because the dials
            # above already number the steps, and it came straight back: read on
            # its own, "Fix what looks wrong" is a task, not a way forward, and
            # users took the re-run link instead of moving one step on.
            # Text only: the green primary keeps the stylesheet it was built
            # with, so a step change never has to re-apply it.
            if step == 0:
                btn.setText(tr("Next: fix what looks wrong"))
            elif step == 1:
                btn.setText(tr("Next: clean up the outlines"))
            # Shapes is the last step: its primary is Export, so the step
            # primary goes away there (and on a zero-detection run there is
            # nothing to advance to on any step).
            self._apply_step_next_visibility()
            # Export is the end of the flow: one green primary per step, and
            # the save only once the outlines are done. Enter follows it.
            self.auto_export_btn.setVisible(
                step == 2 and not self._auto_zero_entry)
            # The edit recap belongs to the Correct step alone.
            recap = getattr(self, "auto_review_recap_label", None)
            if recap is not None:
                recap.setVisible(step == 1 and bool(recap.text()))
            self._apply_review_links(step)
            self._refresh_correct_panels()
            self._keep_review_primary_in_view()
        except (RuntimeError, AttributeError):
            pass

    def _apply_step_next_visibility(self) -> None:
        """Whether the green step primary is on screen. One place, because two
        different things decide it.

        Steps 0 and 1 carry it (Shapes ends on Export instead), and a
        zero-detection run has nothing to advance to on any step.

        On top of that it is HIDDEN while a fix session runs. Save is the only
        way out of an edit, so it has to be the loudest button on screen; a
        green Next beside it invited the user to leave the polygon mid-edit,
        and it read as the primary because it is bigger and further down. The
        Correct page announces the session through set_correct_session_active,
        which calls back here.
        """
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
        """Scroll the step's green primary into view when it lands below the
        fold. A tall step on a short dock used to leave the way forward off
        screen, and a user who wants to skip Correct should never hunt for it.
        Deferred one turn so the step's own layout has settled, and a no-op
        when the button is already on screen (ensureWidgetVisible scrolls the
        minimum, so the step content above stays put whenever it fits)."""
        from ...core.qt_compat import safe_single_shot

        btn = (self.auto_export_btn
               if getattr(self, "_auto_review_step", 0) == 2
               else self.auto_step_next_btn)
        if btn.isHidden():
            return
        # Parented to the dock: a review torn down inside that turn must not
        # scroll a button whose C++ half is already gone.
        safe_single_shot(0, self, lambda: self._scroll_review_primary(btn))

    def _scroll_review_primary(self, btn) -> None:
        try:
            if btn.isHidden():
                return
            self._dock_scroll_area.ensureWidgetVisible(btn, 0, 8)
        except (RuntimeError, AttributeError):
            pass

    def _on_auto_step_next_clicked(self) -> None:
        """The step primary: advance one step. Pure dispatch to the dock
        signals; the plugin owns the state."""
        self.auto_review_step_requested.emit(
            min(2, getattr(self, "_auto_review_step", 0) + 1))

    def _apply_review_links(self, step: int) -> None:
        """Quiet-links row: "Re-run the whole zone" + Exit on every step. The
        dials navigate freely both ways, so there is no back-link."""
        try:
            self.auto_retry_btn.setVisible(True)
            self._auto_review_links_sep.setVisible(True)
            self.auto_review_exit_btn.setVisible(True)
        except (RuntimeError, AttributeError):
            pass

    def set_retry_confirm_pending(self, pending: bool) -> None:
        """Two-stage retry guard (D11): the first click swaps the muted
        escape link to the error-warm confirm label; any other interaction
        resets it. The state machine is plugin-side."""
        try:
            btn = self.auto_retry_btn
            if pending:
                btn.setText(
                    tr("Discard reviewed results and run again? Confirm"))
                btn.setStyleSheet(_BTN_LINK_CONFIRM)
            else:
                btn.setText("↻  " + tr("Re-run the whole zone"))
                btn.setStyleSheet(_BTN_LINK_STRONG)
        except (RuntimeError, AttributeError):
            pass

    def set_review_recap(self, text: str) -> None:
        """Show what the Correct step has changed so far, under the ladder.

        The controller has always sent this line and nothing rendered it, so
        the user had no running trace of their own edits. Empty text clears it.
        """
        label = getattr(self, "auto_review_recap_label", None)
        if label is None:
            return
        try:
            label.setText(text or "")
            label.setVisible(
                bool(text) and int(getattr(self, "_auto_review_step", 0)) == 1)
        except (RuntimeError, AttributeError):
            pass

    def set_auto_display_mode(self, mode: str) -> None:
        """Programmatically select a review display colour mode ('normal' /
        'outline' / 'confidence' / 'random') without emitting
        auto_display_mode_changed: the plugin stores the mode and applies the
        renderer itself, so the combo must follow silently (never desync)."""
        combo = getattr(self, "auto_display_combo", None)
        if combo is None:
            return
        idx = combo.findData(mode)
        if idx < 0:
            return
        combo.blockSignals(True)
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def set_display_legend(self, text: str) -> None:
        """Update the visible colour legend for the selected display mode."""
        legend = getattr(self, "auto_display_legend", None)
        if legend is not None:
            legend.setText(text)
