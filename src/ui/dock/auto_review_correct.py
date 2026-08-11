"""The review's Correct step: the picked object, the AI or Manual fix
method, the add lane, the per-edit status line and the change journal.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QSpinBox

from ...core.i18n import tr
from ...core.review_defaults import (
    AUTO_REVIEW_POINTS_PCT_DEFAULT as _AUTO_REVIEW_POINTS_PCT_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_SIMPLIFY_DEFAULT as _AUTO_REVIEW_SIMPLIFY_DEFAULT,
)
from .styles import (
    _msg_label_qss,
    _msg_text,
)
from .widgets import (
    Mode,
)


class DockAutoReviewCorrectMixin:
    """The review's Correct step: the picked object, the AI or Manual fix
    method, the add lane, the per-edit status line and the change journal."""

    def set_correct_selection(self, count: int) -> None:
        """Correct step: the selected-object panel and the resting hero are the
        two halves of one state. Exactly one shows: the hero asks for a map
        click, the panel offers what to do with the shape that was clicked.
        ``count`` is 0 or 1 (selection is single-object)."""
        try:
            selected = int(count) > 0
            self._auto_correct_has_selection = selected
            self.auto_correct_select_card.setVisible(selected)
            if not selected:
                # Deselecting ends the panel and any fix-session display. The
                # armed line belongs to the SESSION, not the selection:
                # enter_ai_reshape_state seeds it, so a rested panel (after
                # Save) never shows gesture help for a session that is not
                # running.
                self.set_correct_armed_line("")
                self.set_correct_session_active(False)
            self._refresh_correct_panels()
        except (RuntimeError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Method switch, armed line, fix session, class label (round 3)
    # ------------------------------------------------------------------

    def get_correct_method(self) -> str:
        """The Correct step's fix method: "ai" (on-device points) or
        "manual" (QGIS vertices)."""
        return getattr(self, "_correct_method", "ai")

    def set_correct_method(self, method: str) -> None:
        """Set the fix method programmatically (no auto_correct_method_changed
        signal). Drives the switch and the method-dependent labels."""
        method = "manual" if str(method) == "manual" else "ai"
        self._correct_method = method
        try:
            self.auto_correct_method_switch.set_method(method)
        except (RuntimeError, AttributeError):
            pass
        self._refresh_correct_method_ui()

    def _on_correct_method_toggled(self, method: str) -> None:
        """The user clicked the AI | Manual switch. Keep the panel coherent
        even before the plugin reacts, then announce the toggle."""
        method = "manual" if str(method) == "manual" else "ai"
        self._correct_method = method
        self._refresh_correct_method_ui()
        # The armed line is session-only, so re-seed it on a toggle only while
        # a session is actually running (the plugin folds and reopens it).
        if bool(getattr(self, "_auto_correct_session_active", False)):
            self.set_correct_armed_line(self._default_correct_armed_line())
        self.auto_correct_method_changed.emit(method)

    def _refresh_correct_method_ui(self) -> None:
        """Update everything that follows the method: the Add tile's label, the
        line under the Add branch's title, and the edit branch's glyph and
        method line. The armed lane owns its own button label ("Stop adding"),
        so this never writes over it."""
        method = getattr(self, "_correct_method", "ai")
        if not getattr(self, "_auto_add_lane_armed", False):
            try:
                # The tile wears its METHOD's glyph, the same pair the edit
                # branch uses, so one mark means one method across the page.
                # The card head keeps the "+" that names the branch.
                if method == "manual":
                    self.auto_add_lane_btn.setText(
                        "◇  " + tr("Draw its corners"))
                else:
                    self.auto_add_lane_btn.setText(
                        "◎  " + tr("Point at it on the map"))
            except (RuntimeError, AttributeError):
                pass
        try:
            self.auto_add_lane_method_line.setText(
                tr("You place the corners, the same as on any QGIS layer.")
                if method == "manual" else
                tr("The AI outlines it, free, on your computer."))
        except (RuntimeError, AttributeError):
            pass
        self._apply_shape_only_mode(method)
        self._apply_correct_hero_mode()

    def _default_correct_armed_line(self) -> str:
        """The panel's default armed line for the current method. The plugin
        overrides it with live gesture help once a session is armed."""
        if getattr(self, "_correct_method", "ai") == "manual":
            return tr(
                "Drag a corner to move it. Click an edge to add one, "
                "right-click removes.")
        return tr(
            "Left-click adds a keep point, right-click a trim point. The "
            "outline follows.")

    def set_correct_armed_line(self, text: str, kind: str = "armed") -> None:
        """The panel's armed line: what the active method is waiting for. Empty
        text hides it. The plugin owns the wording; the dock only renders it."""
        try:
            lbl = self.auto_correct_armed_line
        except AttributeError:
            return
        try:
            if not text:
                lbl.setText("")
                lbl.setVisible(False)
                return
            lbl.setStyleSheet(_msg_label_qss(kind))
            lbl.setText(_msg_text(kind, text))
            lbl.setVisible(True)
        except (RuntimeError, AttributeError):
            pass

    def set_correct_session_active(self, active: bool) -> None:
        """Add Save + Undo while a fix session runs, and take them away when it
        ends. The per-polygon dials ride along INSIDE the session.

        A map click goes straight into the edit, so the dials have to live
        there or nowhere: an intermediate menu was tried and cost a click on
        the one gesture this step exists for. The fold state carries the
        difference between the methods (_apply_shape_only_mode): Manual opens
        it, because thinning a dense outline is what makes its corners
        draggable and it is the FIRST thing the user does; AI leaves it shut,
        where the model already drew the shape and the dials are the exception.

        Merge is the one control that still steps aside for the session: it
        asks the user to click the OTHER polygons, and a session freezes them.
        """
        self._auto_correct_session_active = bool(active)
        try:
            self.auto_correct_session_row.setVisible(bool(active))
        except (RuntimeError, AttributeError):
            pass
        try:
            self.auto_correct_rest_box.setVisible(True)
        except (RuntimeError, AttributeError):
            pass
        self._apply_merge_tile()
        # Save is the only way out of an edit, so it must be the loudest button
        # on screen. The green step primary steps aside until the session ends.
        self._apply_step_next_visibility()
        if not active:
            self.set_correct_armed_line("")
        self._refresh_correct_summary_row()

    def set_correct_class_label(self, text: str) -> None:
        """The panel title: the run's class names the polygon. Falls back to a
        neutral label when the class is unknown."""
        try:
            self.auto_correct_selected_label.setText(
                str(text or "") or tr("This polygon"))
        except (RuntimeError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Add lane (round 3): one button, method-driven
    # ------------------------------------------------------------------

    def set_add_lane_visible(self, visible: bool) -> None:
        """Show or hide the "Add a missed object" card."""
        try:
            self.auto_add_lane_card.setVisible(bool(visible))
        except (RuntimeError, AttributeError):
            pass

    def set_add_lane_armed(self, armed: bool, method: str) -> None:
        """Arm the add lane, giving Add the same shape the fix panel has: the
        hero says what to do, one green primary commits, and the way out is a
        quiet button under it.

        At rest the lane is a single tile inviting the user in. Armed, that
        tile drops to a ghost "Stop adding" and Keep becomes the filled
        primary, so the loud button is always the one that keeps work."""
        method = "manual" if str(method) == "manual" else "ai"
        armed = bool(armed)
        if armed and not getattr(self, "_auto_add_lane_armed", False):
            self._auto_add_lane_kept = 0   # a fresh session counts from zero
        self._auto_add_lane_armed = armed
        self._auto_add_lane_method = method
        try:
            from .styles import _BTN_GHOST, _BTN_TILE
            if armed:
                self.auto_add_lane_btn.setStyleSheet(_BTN_GHOST)
                self.auto_add_lane_btn.setText(tr("Stop adding"))
                self.auto_add_lane_btn.setToolTip(tr(
                    "Go back to picking polygons. Everything you kept stays, "
                    "and so does the outline on screen."))
            else:
                self.auto_add_lane_btn.setStyleSheet(_BTN_TILE)
                self.set_add_lane_keep_available(False)
                self.auto_add_lane_btn.setToolTip(tr(
                    "Add an object the AI missed. In AI, point at it and the "
                    "on-device model outlines it, free; in Manual, draw its "
                    "corners."))
                self._refresh_correct_method_ui()
            self._refresh_add_lane_line()
        except (RuntimeError, AttributeError):
            pass
        # The resting hero still read "Click a polygon to edit it" while Add
        # was live, which is the opposite of what the lane was asking for.
        self._refresh_correct_selection_hint()

    def set_add_lane_keep_available(self, available: bool) -> None:
        """Show Keep exactly while an outline is waiting to be kept. Hidden,
        never grayed: the lane offers one action at a time. The armed line
        follows, so it always names the next gesture."""
        self._auto_add_lane_has_outline = bool(available)
        try:
            self.auto_add_lane_keep_btn.setVisible(bool(available))
        except (RuntimeError, AttributeError):
            pass
        self._refresh_add_lane_line()
        self._refresh_correct_selection_hint()

    def set_add_lane_progress(self, count: int) -> None:
        """How many objects this Add session has kept so far. The hero says
        what to do next, so the lane line is free to say what was already
        done; without it a Keep is silent and the user cannot tell it landed."""
        try:
            self._auto_add_lane_kept = max(0, int(count))
        except (TypeError, ValueError):
            self._auto_add_lane_kept = 0
        self._refresh_add_lane_line()

    def _refresh_add_lane_line(self) -> None:
        """The Add branch's one line while its tool is armed: the gesture it is
        waiting for, and what this session has already kept.

        The edit branch is hidden while Add is armed, so this line is the only
        instruction on screen and it has to carry the gesture. It used to carry
        the count alone, which left a freshly armed lane saying nothing at all.
        Hidden at rest: the card's title and method line already say what the
        branch is for."""
        try:
            lbl = self.auto_add_lane_line
        except AttributeError:
            return
        if not getattr(self, "_auto_add_lane_armed", False):
            try:
                lbl.setVisible(False)
            except (RuntimeError, AttributeError):
                pass
            return
        manual = getattr(self, "_auto_add_lane_method", "ai") == "manual"
        kept = int(getattr(self, "_auto_add_lane_kept", 0) or 0)
        if getattr(self, "_auto_add_lane_has_outline", False):
            kind = "armed"
            txt = (tr("Keep this one, or keep placing corners.") if manual
                   else tr("Keep this one, or click again to correct the "
                           "outline."))
        elif manual:
            kind = "armed"
            txt = tr("Click each corner on the map, then Finish the line.")
        else:
            kind = "armed"
            txt = tr("Click an object on the map and the AI outlines it.")
        if kept:
            kind = "success"
            txt += " " + (tr("1 polygon added so far.") if kept == 1
                          else tr("{count} polygons added so far.").format(
                              count=kept))
        try:
            lbl.setStyleSheet(_msg_label_qss(kind))
            lbl.setText(_msg_text(kind, txt))
            lbl.setVisible(True)
        except (RuntimeError, AttributeError):
            pass

    def _on_add_lane_clicked(self) -> None:
        """The add lane button: route to the AI outline or the hand draw by the
        current method."""
        if getattr(self, "_correct_method", "ai") == "manual":
            self.auto_add_polygon_requested.emit()
        else:
            self.auto_ai_add_requested.emit()

    def set_merge_available(self, available: bool) -> None:
        """Record whether the selected polygon has an overlapping or touching
        neighbour to join, and re-render the tile.

        Two conditions, not one: a neighbour to merge with, AND no live fix
        session. Merge asks the user to click the OTHER polygons on the map,
        and a session freezes every polygon but its own, so offering it mid
        session offers a click the canvas will refuse."""
        self._auto_merge_available = bool(available)
        self._apply_merge_tile()

    def _apply_merge_tile(self) -> None:
        try:
            self.auto_shape_merge_btn.setVisible(
                bool(getattr(self, "_auto_merge_available", False))
                and not bool(getattr(self, "_auto_correct_session_active", False)))
        except (RuntimeError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Selected detection: facts line and the per-shape settings panel
    # ------------------------------------------------------------------

    def set_correct_selection_info(self, text: str) -> None:
        """The measured facts of the selected shape (area, point count), shown
        on the title row. Empty hides nothing: the row keeps its height."""
        try:
            self.auto_correct_selected_info.setText(str(text or ""))
        except (RuntimeError, AttributeError):
            pass

    def _shape_only_widgets(self) -> dict:
        """The per-shape controls, keyed by the review param each one writes.
        One map, so seeding, reading and blocking signals cannot drift apart as
        controls are added."""
        return {
            "points_pct": self.auto_shape_only_points,
            "simplify_px": self.auto_shape_only_simplify,
            "open_px": self.auto_shape_only_clean,
            "expand_px": self.auto_shape_only_expand,
            "smooth": self.auto_shape_only_smooth,
            "fill_holes": self.auto_shape_only_fill,
            "ortho": self.auto_shape_only_ortho,
        }

    def set_shape_only_values(self, values: dict, overridden: bool) -> None:
        """Seed the per-shape controls for the object just selected.

        ``values`` is keyed by review param name; a missing key leaves that
        control alone. Store-only: setting a widget must not look like a user
        edit, so the signals are blocked while the values land."""
        try:
            widgets = self._shape_only_widgets()
        except (RuntimeError, AttributeError):
            return
        for w in widgets.values():
            w.blockSignals(True)
        try:
            for key, widget in widgets.items():
                if key not in values:
                    continue
                raw = values[key]
                if hasattr(widget, "setChecked"):
                    widget.setChecked(bool(raw))
                elif isinstance(widget, QSpinBox):
                    widget.setValue(int(raw))
                else:
                    widget.setValue(float(raw))
            self.auto_shape_only_reset.setVisible(bool(overridden))
            self._auto_shape_only_overridden = bool(overridden)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass
        finally:
            for w in widgets.values():
                w.blockSignals(False)
        self._sync_shape_only_right_angles()

    def _sync_shape_only_right_angles(self) -> None:
        """Refuse this polygon's Right angles when the geometry library behind
        it is missing, so a per-object exception cannot silently return the
        outline unchanged either (right_angles_support). Only a TICKED box is
        tested, so an untouched control never imports shapely."""
        ortho = getattr(self, "auto_shape_only_ortho", None)
        if ortho is None or not ortho.isChecked():
            return
        from .right_angles_support import gate_right_angles

        gate_right_angles(
            ortho, getattr(self, "auto_shape_only_ortho_label", None))

    def get_shape_only_values(self) -> dict:
        """Current per-shape control values, keyed by review param name."""
        try:
            widgets = self._shape_only_widgets()
        except (RuntimeError, AttributeError):
            return {"points_pct": _AUTO_REVIEW_POINTS_PCT_DEFAULT,
                    "simplify_px": _AUTO_REVIEW_SIMPLIFY_DEFAULT}
        out: dict = {}
        for key, widget in widgets.items():
            try:
                if hasattr(widget, "isChecked"):
                    out[key] = bool(widget.isChecked())
                elif isinstance(widget, QSpinBox):
                    out[key] = int(widget.value())
                else:
                    out[key] = float(widget.value())
            except (RuntimeError, AttributeError):
                continue
        return out

    def _emit_shape_only_changed(self, control: str, _value) -> None:
        """A per-shape control moved: publish every value and track the
        adjustment once per control per review (no new event name)."""
        values = self.get_shape_only_values()
        self.auto_shape_only_changed.emit(values)
        try:
            tracked = getattr(self, "_review_shape_tracked", None)
            if tracked is None:
                tracked = set()
                self._review_shape_tracked = tracked
            if control not in tracked:
                tracked.add(control)
                from ...core import telemetry, telemetry_run_events
                telemetry_run_events.track_review_shape_adjusted(
                    control=control,
                    value=int(values.get("points_pct", 0) or 0),
                    run_id=telemetry.get_last_run_id() or "")
        except Exception:
            pass  # nosec B110

    def _apply_shape_only_mode(self, method: str) -> None:
        """The per-polygon dials exist in BOTH methods, and open differently.

        They used to be an AI-only offer, on the reading that a control which
        rewrites a hand-traced outline undoes the user's work. What that missed
        is WHEN the user reaches for them in Manual: before dragging anything,
        to thin a 96-point outline into one with corners you can actually grab.
        So Manual rests OPEN with Points at the top, and AI rests shut, where
        the dials are the exception to a shape the model already drew.
        """
        manual = str(method) == "manual"
        if manual is not getattr(self, "_auto_shape_only_manual", None):
            self._auto_shape_only_manual = manual
            self._auto_shape_only_expanded = manual
        try:
            self.auto_shape_only_toggle.setVisible(True)
            self.auto_shape_only_scope_line.setText(
                tr("This polygon only. Fewer points means fewer corners to drag.")
                if manual else
                tr("This polygon only. Every other one follows the Shapes step."))
        except (RuntimeError, AttributeError):
            pass
        self._apply_shape_only_toggle()

    def _on_shape_only_toggle_clicked(self) -> None:
        """Fold the per-polygon settings open or shut."""
        self._auto_shape_only_expanded = not bool(
            getattr(self, "_auto_shape_only_expanded", False))
        self._apply_shape_only_toggle()

    def _apply_shape_only_toggle(self) -> None:
        """Render the fold: the head carries the chevron and the body follows.

        The head names what the dials DO in the current method, because that is
        what tells the two pages apart. "Settings for this polygon" named the
        scope and not the job, and read the same on both."""
        expanded = bool(getattr(self, "_auto_shape_only_expanded", False))
        manual = bool(getattr(self, "_auto_shape_only_manual", False))
        try:
            self.auto_shape_only_toggle.setText(
                ("▾  " if expanded else "▸  ") + (
                    tr("Simplify this outline first") if manual
                    else tr("Clean up this outline")))
            self.auto_shape_only_toggle.setToolTip(
                tr("Thin this outline before you drag its corners, without "
                   "moving the dials that drive the whole layer.") if manual
                else tr("Give this one polygon its own shape settings, without "
                        "moving the dials that drive the whole layer."))
            # `expanded` alone. Gating on the head's isVisible() read the
            # PARENT's state (Qt reports a child of a hidden card as invisible),
            # so the body stayed shut behind a chevron that said open, and it
            # never reopened when the review card came up.
            self.auto_shape_only_box.setVisible(expanded)
        except (RuntimeError, AttributeError):
            pass

    def _should_show_correct_selection_hint(self) -> bool:
        """Whether the edit branch card is on screen.

        It is offered while Correct waits for a click, and withdrawn whenever
        editing is not the branch the user is on: a run that found nothing has
        no polygon to edit, and an armed Add owns the map tool, so leaving the
        card up would invite a click that the add tool would swallow. Both used
        to rewrite the card's copy instead of hiding it, which put "click a
        polygon" on a page with no polygons."""
        resting = self._mode == Mode.AUTOMATIC and bool(getattr(self, "_auto_review_active", False))
        resting = resting and getattr(self, "_auto_review_step", 0) == 1
        resting = resting and not bool(getattr(self, "_auto_correct_has_selection", False))
        resting = resting and not bool(getattr(self, "_auto_correct_merge_armed", False))
        resting = resting and not bool(getattr(self, "_auto_zero_entry", False))
        return resting and not bool(getattr(self, "_auto_add_lane_armed", False))

    def _refresh_correct_selection_hint(self) -> None:
        """Show or hide the resting hero and set its copy.

        It is not dismissible: it is the step's only instruction, not a tip
        about it, and hiding it would leave the resting screen blank."""
        try:
            show = self._should_show_correct_selection_hint()
            self.auto_correct_pick_hero.setVisible(show)
            self._apply_correct_hero_mode()
            # The "or" belongs to the pair: the Add card is up whenever the
            # edit card is, so the edit card's own gate decides for both.
            self.auto_correct_or_row.setVisible(show)
        except (RuntimeError, AttributeError):
            pass

    def _apply_correct_hero_mode(self) -> None:
        """Drive the edit branch card: its glyph, and the one line that says
        what this method is waiting for.

        The gesture plate above the line carries the real difference; the glyph
        and the line follow it. The two methods do different work on the same
        polygon, and the page used to describe only one of them, in words that
        fitted neither."""
        method = getattr(self, "_correct_method", "ai")
        try:
            self.auto_correct_gesture_art.set_method(method)
        except (RuntimeError, AttributeError):
            pass
        try:
            if method == "manual":
                self.auto_correct_pick_glyph.setText("◇")
                line = tr("Click a polygon, then drag any corner.")
            else:
                self.auto_correct_pick_glyph.setText("◎")
                line = tr("Click a polygon, then click the spot the AI missed.")
            self.auto_correct_pick_title.setText(
                tr("Edit an existing polygon"))
            self.auto_correct_pick_hint.setText(_msg_text("armed", line))
        except (RuntimeError, AttributeError):
            pass

    def _correct_info_line_gate(self) -> bool:
        """The dismissible right-click delete tip shows only in the resting
        non-zero state on Correct, which is exactly where that gesture works.
        Also the reshow gate after a guidance reset, so it never flashes into
        the wrong state."""
        resting = self._mode == Mode.AUTOMATIC and bool(getattr(self, "_auto_review_active", False))
        resting = resting and getattr(self, "_auto_review_step", 0) == 1
        resting = resting and not bool(getattr(self, "_auto_correct_has_selection", False))
        return resting and not bool(getattr(self, "_auto_zero_entry", False))

    def _refresh_correct_info_line(self) -> None:
        """Show the right-click delete tip only when the gate allows and the
        user has not dismissed it."""
        try:
            hint = self.auto_correct_delete_tip
        except AttributeError:
            return
        try:
            from .guidance import HINT_REVIEW_RIGHT_CLICK_DELETE, is_hint_dismissed
            if (self._correct_info_line_gate()
                    and not is_hint_dismissed(HINT_REVIEW_RIGHT_CLICK_DELETE)):
                hint.show()
            else:
                hint.hide()
        except (RuntimeError, AttributeError):
            pass

    def _refresh_correct_panels(self) -> None:
        """Reconcile the resting hero, the info line, the add lane and the
        Delete row with the current selection and zero-detection state."""
        selected = bool(getattr(self, "_auto_correct_has_selection", False))
        try:
            # The add lane shows when nothing is picked (the zero-detection
            # entry is nothing-picked too, so it stays visible there). Delete
            # lives inside the panel now, so it needs no gating here.
            self.set_add_lane_visible(not selected)
        except (RuntimeError, AttributeError):
            pass
        self._refresh_correct_selection_hint()
        self._refresh_correct_info_line()

    def enter_ai_reshape_state(self) -> None:
        """Morph the panel into an in-place AI fix session: show Done/Undo and
        lock the dials (no separate screen). The panel and selection stay; only
        the fix method is live. ``_refine_handoff`` stays the internal flag the
        teardown paths and closeEvent already consult.

        The step primary stays put. A session opens on a single map click, so
        hiding it took the way forward away from anyone who clicked a polygon to
        see what happens; leaving Correct folds the live session first
        (_on_auto_review_step_requested), so the button is safe to press
        mid-edit. Save stays the session's own answer.
        """
        self._refine_handoff = True
        try:
            self.set_correct_session_active(True)
            # Seed the session's gesture help; the plugin refines the wording
            # (loading note, then the live keep/trim line) once the crop is up.
            self.set_correct_armed_line(self._default_correct_armed_line())
            self._set_review_dials_locked(True, 1)
        except (RuntimeError, AttributeError):
            pass

    def leave_ai_reshape_state(self) -> None:
        """Leave the AI fix session and restore the resting Correct panel: hide
        Done/Undo, re-drive the step primary and unlock the dials."""
        self._refine_handoff = False
        try:
            self.set_correct_session_active(False)
            self.set_auto_review_installing(False)
            self.set_auto_review_step(1)
        except (RuntimeError, AttributeError):
            pass

    def set_correct_armed(self, which: str | None) -> None:
        """Armed visual for the Correct step's Merge pick: 'merge' or None (at
        rest). The arming state machine is plugin-side; this only flips the
        armed dynamic property and re-polishes. Split is gone (it moved to the
        QGIS bridge), so there is a single toggle now."""
        try:
            btn = self.auto_shape_merge_btn
        except AttributeError:
            return
        try:
            from .styles import _BTN_TILE, _BTN_TILE_ACTIVE
            armed = (which == "merge")
            self._auto_correct_merge_armed = armed
            # The tile carries its armed look in its own stylesheet (a dynamic
            # property needs a matching rule, and the tile QSS has none).
            btn.setProperty("armed", armed)
            btn.setStyleSheet(_BTN_TILE_ACTIVE if armed else _BTN_TILE)
            # The label stays stable; the active look plus the panel's armed
            # line carry the "now click the pieces" state.
            btn.setText("⧉  " + tr("Merge with neighbours"))
            self._refresh_correct_selection_hint()
        except (RuntimeError, AttributeError):
            pass

    def set_correct_status(self, kind: str, text: str,
                           action_text: str = "") -> None:
        """Per-edit status line on the Correct step: an in-progress state
        (armed / neutral / warning), with an optional secondary ACTION link
        on its own second line (the merge confirm; routes
        auto_correct_status_action_requested). Empty text hides the line.
        There is no success/outcome message here: the persistent summary row
        ("N corrections this round · Undo last · Clear all") already reports
        what happened, so this line only carries what to do next."""
        try:
            lbl = self.auto_correct_status
        except AttributeError:
            return
        try:
            if not text:
                lbl.setText("")
                lbl.setVisible(False)
                return
            lbl.setStyleSheet(_msg_label_qss(kind))
            if action_text:
                import html
                body = _msg_text(kind, html.escape(text))
                act = f'<a href="action">{html.escape(action_text)}</a>'
                rows = [f'<tr><td colspan="2">{body}</td></tr>',
                        f'<tr><td colspan="2">{act}</td></tr>']
                lbl.setTextFormat(Qt.TextFormat.RichText)
                lbl.setText('<table width="100%">' + "".join(rows) + "</table>")
            else:
                lbl.setTextFormat(Qt.TextFormat.PlainText)
                lbl.setText(_msg_text(kind, text))
            lbl.setVisible(True)
        except (RuntimeError, AttributeError):
            pass

    def _on_correct_status_link(self, href: str) -> None:
        """Dispatch the status line's action link (the merge confirm)."""
        if href == "action":
            self.auto_correct_status_action_requested.emit()

    def set_correction_summary(self, count: int) -> None:
        """Persistent journal summary line "N corrections this round · Undo
        last · Clear all" (D9). Hidden while the journal is empty, and parked
        while a fix session runs (it is rest-state information; the session
        shows only the editing controls)."""
        self._correction_summary_count = max(0, int(count))
        try:
            if count == 1:
                text = tr("1 correction this round")
            elif count > 1:
                text = tr("{n} corrections this round").format(n=count)
            else:
                text = ""
            if text:
                self.auto_correct_summary_label.setText(text)
            # Name what it takes. "Clear all" beside "Undo last" reads as its
            # neighbour, and it undoes the whole round in one quiet click.
            if count > 1:
                clear_text = tr("Clear all {n}").format(n=count)
            else:
                clear_text = tr("Clear all")
            self.auto_correct_clear_btn.setText(clear_text)
        except (RuntimeError, AttributeError):
            pass
        self._refresh_correct_summary_row()

    def _refresh_correct_summary_row(self) -> None:
        """The journal summary shows only at rest, with something to report."""
        try:
            self.auto_correct_summary_row.setVisible(
                int(getattr(self, "_correction_summary_count", 0)) > 0 and not bool(getattr(
                    self, "_auto_correct_session_active", False)))
        except (RuntimeError, AttributeError):
            pass

    def set_zero_detection_entry(self, active: bool) -> None:
        """Zero-detection entry: an empty run opens the review on the Correct
        step (step 1) with the dedicated welcome line instead of the question,
        so the user who got nothing lands where they add what is missing; the
        green primary stays hidden while there is nothing to advance to."""
        self._auto_zero_entry = bool(active)
        try:
            if active:
                self.set_auto_review_step(1)
                self.auto_correct_zero_line.setVisible(True)
                self.auto_step_next_btn.setVisible(False)
            else:
                self.auto_correct_zero_line.setVisible(False)
            self._refresh_correct_panels()
        except (RuntimeError, AttributeError):
            pass
