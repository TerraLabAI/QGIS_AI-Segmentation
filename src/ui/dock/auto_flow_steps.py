"""The Automatic step ladder: raster lock, Start and Exit, the drawn zone,
the prompt focus hold and the keyboard shortcuts that drive them.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
)

from ...core.i18n import tr
from .widgets import (
    Mode,
)


class DockAutoFlowStepsMixin:
    """The Automatic step ladder: raster lock, Start and Exit, the drawn zone,
    the prompt focus hold and the keyboard shortcuts that drive them."""

    def _on_auto_layer_changed(self, layer) -> None:
        # A raster appearing or vanishing must show/hide the combo and the
        # no-rasters warning live, not only on the next mode switch.
        if self._mode == Mode.AUTOMATIC:
            self._update_ui_state_automatic()
        self._update_auto_detect_enabled()
        self._refresh_auto_layer_lock()

    def _on_auto_start_clicked(self) -> None:
        """Step 0 "Start": lock the chosen raster and open the draw-zone step.

        Mirrors the Interactive Start: from here on the layer combo is greyed
        and read-only (the header still names the locked raster) until the
        user clicks Exit.
        """
        layer = self.auto_layer_combo.currentLayer()
        if layer is None:
            return
        # Consent is NOT asked here: the checkbox sits on the last step, right
        # above Detect, and the first Detect seals it (see seal_tos_consent).
        # Clear any leftover "Saved N polygon(s)" banner from a previous run so
        # it never lingers into this fresh run's steps (it is shown on the Start
        # step right after Finish, and belongs only to that just-finished run).
        self.set_auto_status("idle")
        # Leaving step 0 for a fresh run retires the previous run's value recap
        # and the post-export success line.
        self.clear_last_run_recap()
        self.clear_auto_export_success()
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_auto_start_clicked(
                layer_kind=self._auto_layer_kind(layer),
                has_credits_known=self._auto_credits is not None,
            )
        except Exception:
            pass  # nosec B110
        self._auto_started = True
        self._go_to_auto_step(1)

    @staticmethod
    def _auto_layer_kind(layer) -> str:
        """Coarse raster-source class for telemetry: local / xyz / wms / other."""
        try:
            provider = (layer.dataProvider().name() or "").lower()
            if provider == "gdal":
                return "local"
            source = (layer.source() or "").lower()
            if "type=xyz" in source:
                return "xyz"
            if provider in ("wms", "wmts"):
                return "wms"
            return provider or "other"
        except Exception:
            return "other"

    def reset_auto_to_start(self) -> None:
        """Exit the Automatic flow back to the Start step, layer editable again.

        Called by the dock Exit button (via the plugin, which first clears the
        zone + any review) and on any hard teardown that should unlock the
        layer. Idempotent.
        """
        self._auto_started = False
        # A teardown mid-hand-over (mode switch, Exit) ends it: released here so
        # the idle status below is not swallowed and the run card really goes.
        self._auto_finalizing = False
        # Any hard teardown / mode switch retires the post-export success line
        # (the export path re-shows it AFTER calling this).
        self.clear_auto_export_success()
        # Leaving the flow retires the free-trial zone-cap message.
        self.set_auto_zone_rejected(None)
        self.set_auto_exhausted_subscribe_visible(False)
        # No zone left to price, so the credit-block callout goes with it.
        self.set_auto_credit_block(None)
        # Clear any leftover run status ("Saved N polygon(s) to ...") so it never
        # lingers on the Start / prompt page of the next run.
        self.set_auto_status("idle")
        # Re-enable the mode toggle, disabled during review: at Start
        # there is no review to protect (skip while a handoff owns the toggle).
        if not self._refine_handoff:
            self.mode_switch.setEnabled(True)
            self.mode_switch.setToolTip("")
        self.auto_prompt_input.blockSignals(True)
        self.auto_prompt_input.clear()
        self.auto_prompt_input.blockSignals(False)
        # The blocked clear() fired no textChanged, so the commit dedupe would
        # still hold the LAST run's prompt: typing the same object again next
        # run would silently skip its commit (no re-seed, no run plan). Reset
        # it with the box.
        self._last_committed_prompt = None
        self._boost_nudge_tracked = None
        self._auto_prompt_valid = False
        # Leaving the flow retires any in-flight commit-time prompt lookup, so
        # a late answer can never re-fire a detection on a flow that is gone.
        self._abandon_prompt_lookup()
        self._set_prompt_info()
        # The object is gone with the prompt: drop its slider verdict too.
        self._auto_detail_feedback = None
        # The prompt was cleared with signals blocked (no textChanged), so
        # refresh the Detect + Detail gates explicitly: the next run must
        # start with the Detail slider gated again until an object exists.
        self._update_auto_detect_enabled()
        # The review hides the layer header (see set_auto_review_active); a
        # hard teardown that skips the review-off call must restore it, and
        # retire the bottom View-as block the same way (it lives outside the
        # Automatic page, so nothing else hides it).
        self.auto_layer_combo.setVisible(True)
        self.auto_review_view_row.setVisible(False)
        self._go_to_auto_step(0)

    def _go_to_auto_step(self, index: int) -> None:
        """Switch the Automatic flow to the given step.

        step 0 Start (layer editable) | step 1 Draw zone | step 2 Prompt +
        settings. From step 1 on the layer header is locked; the canvas x badge
        re-draws the zone and the Exit button returns to step 0.
        """
        # A running detection pins the flow to the launch step.
        if self._auto_run_active:
            index = 2
        # Launching a run puts the prompt box and the reference panel in their
        # read-only form, and only set_auto_run_active(False) on a NON-hold path
        # takes them back out. A run that ends in the review never takes that
        # path (the review hides the whole card instead), so the read-only form
        # stayed latched for the rest of the session: "Re-run the whole zone"
        # then handed back a prompt the user could not type in and a reference
        # panel with no add button. Every route back to the setup screen goes
        # through this method, so release it here whenever no run owns the
        # screen. The hold (finalize hand-over) keeps its read-only form.
        if not self._auto_run_active and not getattr(self, "_auto_finalizing", False):
            self._set_auto_prompt_readonly(False)
            self._set_exemplar_readonly(False)
        # The stack is hidden in the empty state (hero only); any explicit step
        # change means the flow is live, so it must be visible again.
        self.auto_steps.setVisible(True)
        self.auto_steps.setCurrentIndex(index)
        self._refresh_auto_layer_lock()
        # Leaving step 0 launches the Automatic flow (returning to it exits): the
        # mode switch is only shown on the Start screen, so re-evaluate it here.
        self._refresh_mode_switch_visibility()
        self._update_auto_detect_enabled()
        # The exemplar panel belongs to the idle prompt step (2): an example is
        # drawn inside the zone, which exists by the time step 2 opens. Hidden on
        # the Start/Draw steps and while a run or review is active. Also gated
        # behind _EXEMPLARS_ENABLED (feature hidden for now).
        self.auto_exemplar_panel.setVisible(
            self._EXEMPLARS_ENABLED and index == 2 and not self._auto_run_active and not self._auto_review_active
        )
        # The bottom-pinned first-steps guide banner shows on the Start step only.
        self._update_auto_tutorial_banner_visibility()
        # The plugin reacts to step changes (e.g. arms the zone drawing tool
        # whenever the zone step opens without a zone set).
        self.auto_step_changed.emit(index)

    def _refresh_auto_layer_lock(self) -> None:
        """Greyed/locked layer header from step 1 on; editable on the Start step.

        Reuses the Interactive locked-combo look: disabled combo with the
        dropdown arrow hidden, and the label hidden so only the raster name
        shows. Step 0 restores the editable combo + its label.
        """
        # Empty-project guard: on the Start step with NO raster loaded the page
        # is the no-imagery hero, so the header must stay hidden. Without this,
        # this method (which runs LAST on a combo change) would re-show the
        # "Select a Raster Layer" label over the hero - the orphaned-label bug
        # seen after deleting the last layer. The fresh path never calls this,
        # which is why only the delete path broke.
        if self.auto_steps.currentIndex() == 0 and self.auto_layer_combo.count_layers() == 0:
            self.auto_layer_label.setVisible(False)
            self.auto_layer_combo.setVisible(False)
            return
        on_start = self.auto_steps.currentIndex() == 0
        self.auto_layer_label.setVisible(on_start)
        self.auto_layer_combo.setEnabled(on_start)
        # Freeze the combo whenever the header is locked (steps 1/2/run/review):
        # once a source is locked, hiding layers in the tree must not drop it from
        # the list or re-pick another. Unfreezes + resyncs on returning to Start.
        self.auto_layer_combo.set_frozen(not on_start)
        if on_start:
            self.auto_layer_combo.setStyleSheet(
                "QComboBox { color: palette(text); }")
        else:
            self.auto_layer_combo.setStyleSheet(
                "QComboBox { color: palette(text); }"
                "QComboBox::drop-down { width: 0px; border: none; }")

    def on_zone_deleted_from_canvas(self) -> None:
        """Called by the plugin when the user clicks the zone's x badge."""
        self.set_auto_zone_state("idle")
        self._go_to_auto_step(1)

    def set_auto_zone_state(self, state: str) -> None:
        """Reflect the canvas zone state. State: 'idle', 'drawing', 'zone_set'."""
        self._auto_zone_is_set = state == "zone_set"
        # A valid zone landing, or leaving the flow, retires the free-trial
        # zone-cap message; while state stays 'drawing' (a rejected zone puts
        # the user back there) the message remains as guidance.
        if state in ("idle", "zone_set"):
            self.set_auto_zone_rejected(None)
        if state in ("idle", "drawing"):
            self._auto_zone_too_large = False
            # No zone = no per-zone estimate; drop the stale cost label AND the
            # cached estimate/gate so neither lingers from the previous zone.
            self._auto_est_credits = None
            self._auto_insufficient_credits = False
            self.auto_credit_cost_label.setVisible(False)
            self.set_auto_credit_block(None)
        elif state == "zone_set":
            # A drawn zone completes step 2.
            self._go_to_auto_step(2)
            # Land AND HOLD the caret in the prompt box so the user types the
            # object without a click. A single setFocus is not enough: the zone
            # finishes on a canvas event, and for the first ~second the map
            # canvas can reclaim keyboard focus on a deferred repaint (the slow
            # basemap redraw), which yanked the caret back out after ~0.5 s. So
            # this holds focus across that window (see _begin_auto_prompt_focus).
            self._begin_auto_prompt_focus()
        self._update_auto_detect_enabled()

    def set_zone_draw_progress(self, count: int) -> None:
        """Live guidance under the 'Draw your zone' title while the user clicks
        points, so it is always clear what to do next and how to finish."""
        if count <= 0:
            txt = tr("Click on the map to outline the area to scan.")
        elif count < 3:
            txt = tr("Keep clicking around the area, at least 3 points.")
        else:
            txt = tr("Click the first point to close the zone.")
        try:
            self._auto_zone_hint.setText(txt)
        except (RuntimeError, AttributeError):
            pass

    def _begin_auto_prompt_focus(self) -> None:
        """Start holding the keyboard caret in the prompt box for ~1 s.

        The caret is re-asserted on a short repeating timer because a single
        setFocus loses to the canvas reclaiming focus a few hundred ms later
        (deferred basemap repaint after the zone is drawn). Each tick reclaims
        focus ONLY when it has drifted OUTSIDE the dock (i.e. to the map
        canvas) or nowhere, so a deliberate click on another dock control is
        respected and typing is never interrupted. Stops after the window."""
        from qgis.PyQt.QtCore import QTimer
        timer = getattr(self, "_auto_prompt_focus_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setInterval(120)  # ~8 ticks fill the ~1 s steal window
            timer.timeout.connect(self._tick_auto_prompt_focus)
            self._auto_prompt_focus_timer = timer
        self._auto_prompt_focus_ticks = 0
        self._tick_auto_prompt_focus()  # claim immediately, then hold
        timer.start()

    def _tick_auto_prompt_focus(self) -> None:
        """One focus-hold tick (see _begin_auto_prompt_focus)."""
        timer = getattr(self, "_auto_prompt_focus_timer", None)
        try:
            self._auto_prompt_focus_ticks = getattr(
                self, "_auto_prompt_focus_ticks", 0) + 1
            prompt = self.auto_prompt_input
            if (self.auto_steps.currentIndex() != 2 or not prompt.isVisible() or not prompt.isEnabled()):
                if timer is not None:
                    timer.stop()
                return
            focused = QApplication.focusWidget()
            # Reclaim only when focus sits outside the dock (the map canvas) or
            # nowhere; a real click on a sibling dock control (focus inside the
            # dock) is left alone, and if the prompt already holds it this is a
            # no-op.
            if focused is not prompt and (
                    focused is None or not self.isAncestorOf(focused)):
                prompt.setFocus(Qt.FocusReason.OtherFocusReason)
            if self._auto_prompt_focus_ticks >= 8 and timer is not None:
                timer.stop()
        except (RuntimeError, AttributeError):
            if timer is not None:
                timer.stop()

    def _is_auto_for_us(self) -> bool:
        """True when the Automatic flow should own Enter: in Automatic mode,
        the flow is started, and no run is in flight (Enter has no job during
        a run; Escape has its own gate so it can soft-cancel one)."""
        return self._mode == Mode.AUTOMATIC and self._auto_started and not self._auto_run_active

    def _on_auto_escape_shortcut(self) -> None:
        """Escape: delegate to the plugin's single dispatcher (_route_escape).

        Unlike Enter this also fires DURING a run (Escape = soft Cancel
        there), so it gates only on the Automatic flow being started."""
        if self._mode == Mode.AUTOMATIC and self._auto_started:
            self.auto_escape_pressed.emit()

    def _on_auto_enter_shortcut(self) -> None:
        """Enter: delegate to the plugin's single dispatcher (_route_enter),
        unless a text editor or spinbox has focus (its own returnPressed /
        value-commit handles Enter, avoiding a double trigger)."""
        if not self._is_auto_for_us():
            return
        fw = QApplication.focusWidget()
        if isinstance(fw, (QLineEdit, QComboBox, QAbstractSpinBox)):
            return
        self.auto_enter_pressed.emit()

    def _is_auto_correct_shortcut_active(self) -> bool:
        """Whether a Correct-step editing shortcut may act on the review."""
        active = self._mode == Mode.AUTOMATIC and bool(getattr(self, "_auto_review_active", False))
        active = active and getattr(self, "_auto_review_step", 0) == 1
        active = active and not bool(getattr(self, "_refine_handoff", False))
        return active and not bool(getattr(self, "_qgis_bridge_active_ui", False))

    def _on_auto_correct_remove_shortcut(self) -> None:
        """Remove the selected Correct-step detection with Delete."""
        if not self._is_auto_correct_shortcut_active():
            return
        try:
            if self.auto_correct_select_card.isVisible():
                self.auto_remove_requested.emit()
        except (RuntimeError, AttributeError):
            pass

    def _on_auto_correct_undo_shortcut(self) -> None:
        """Undo the last Correct-step change with the platform Undo shortcut.

        This gate is down everywhere else in the flow, which is why the plugin
        listens to the same shortcut: a press during a zone draw is routed to
        the draw tool there (see _on_auto_undo_pressed). The two states never
        overlap, so at most one of the two acts on a press.
        """
        if self._is_auto_correct_shortcut_active():
            self.auto_correction_undo_requested.emit()

    def set_auto_shortcuts_enabled(self, enabled: bool) -> None:
        """Toggle the Escape/Enter shortcuts. Disabled while an example box is
        being drawn so the draw tool owns Escape (cancel) with no race."""
        for sc in (self.auto_escape_shortcut, self.auto_enter_shortcut,
                   self.auto_enter_shortcut_kp):
            try:
                sc.setEnabled(enabled)
            except (RuntimeError, AttributeError):
                pass
