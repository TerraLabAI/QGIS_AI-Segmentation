"""AI-assisted Add: point at a missed object, the local model outlines it, keep.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); a plain mixin like
its siblings, so state lives on the plugin instance (self) and the widgets on
the dock. The Add lane is the second half of the round-3 Correct step: with a
polygon selected the panel fixes it, and with nothing selected the Add lane
starts a fresh outline for an object the run missed.

The lane rides the SAME Manual point-refine session the reshape handoff uses, so
it inherits the overlap weld, the crop encode transport lock and the fold-back.
The only difference from a reshape is that no existing detection is opened: a
click on empty ground starts a normal prediction (the live outline), gated by
``_refine_add_mode_active`` in manual_predict's resting-click branch.

Keeping is the ordinary Manual save, reached from the lane's Keep button or S,
and the lane stays armed after it so several missed objects go in one session.
Done adding keeps the outline still on screen and folds everything into the
review as its own object (a fresh det_id, exempt from the confidence and size
gates). Escape is the only gesture that throws an outline away.
"""
from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtWidgets import QMessageBox

from ...core.i18n import tr


class ManualAddMixin:
    """The AI-assisted Add lane on the Correct step (method == 'ai')."""

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def _on_ai_add_requested(self) -> None:
        """Arm the AI-assisted Add lane: point at a missed object, the local AI
        outlines it, Keep (or S) commits it and leaves the lane armed for the
        next one. Reuses the Manual point-refine session (so weld/encode/fold
        all apply) but opens NO existing detection; the first click on empty
        ground starts a normal prediction.

        When the local AI is not installed, a one-time setup runs first and
        holds the review still while it does (local_ai_install_lock); the Add
        lane arms itself once the model is ready (see _on_predictor_loaded)."""
        review = self._auto_review
        if not review or not self.dock_widget:
            return
        # The armed lane button is also the way OUT (its label reads Done
        # adding). The outline on screen is the user's work, so Done KEEPS it:
        # _on_reshape_done folds it through the ordinary save, exactly like
        # pressing Keep first. Escape is the path that throws an outline away.
        # Only a click still waiting on an encode is dropped: it never became
        # an outline, and the session it would replay into is ending.
        if getattr(self, "_refine_add_mode_active", False):
            try:
                self._discard_pending_manual_click()
            except (RuntimeError, AttributeError):
                pass
            self._exit_ai_add_mode()
            try:
                self._on_reshape_done()
            except (RuntimeError, AttributeError):
                pass
            return
        # Add lives at rest (nothing selected); a live session/bridge owns the
        # canvas already, so ignore a stray Add there.
        if getattr(self, "_refine_handoff_active", False) or getattr(
                self, "_qgis_bridge_active", False):
            return
        layer = self._resolve_auto_source_layer()
        if layer is None:
            return
        # Env gate: without the local AI the predictor never arrives. Offer the
        # one-time setup, which takes the review until it ends and then arms the
        # lane itself.
        if not self._manual_env_ready():
            if self._local_ai_install_pending():
                return  # a setup is already running for this review
            # A setup already ran this session and the model still would not
            # load: say it once instead of reopening the same modal on every
            # Add click. With no attempt behind it the offer still stands.
            if (getattr(self, "_local_ai_load_failed", False)
                    and getattr(self, "_local_ai_install_attempted", False)):
                self._warn_local_ai_unavailable_once()
                return
            box = QMessageBox(self.iface.mainWindow())
            box.setWindowTitle(tr("Adding needs a one-time setup"))
            box.setText(tr(
                "Adding an object uses the free on-device AI, which is not "
                "installed yet. Install it now? It runs once and takes a few "
                "minutes. The review waits for it, then arms Add for you."))
            install_btn = box.addButton(
                tr("Install now"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(tr("Cancel"), QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(install_btn)
            box.exec()
            if box.clickedButton() is not install_btn:
                return  # review untouched
            self._begin_local_ai_install("add")
            return
        self._enter_ai_add_mode(layer)

    def _enter_ai_add_mode(self, layer) -> None:
        """Start the point-refine session for Add (no target opened) and arm the
        lane. Mirrors the reshape handoff setup minus _open_reshape_target."""
        # Drop the resting select tool before the Manual session takes the canvas.
        self._disarm_shape_tool()
        self._handoff_source_layer = layer
        self._pending_refine_import = False
        self._refine_handoff_active = True
        # Export reports a hand edit was used (adds are hand work like reshapes).
        self._auto_refined_in_manual = True
        # Hide (do NOT discard) the blue review layer + the zone/example overlays
        # while the SAM session runs on the same detections as editable polygons.
        self._remove_auto_selection_layer()
        self._set_exemplar_bands_visible(False)
        self._set_auto_zone_overlays_visible(False)
        try:
            self.dock_widget.enter_ai_reshape_state()
        except (RuntimeError, AttributeError):
            pass
        # Start the Manual SAM session directly (imports the detections so the
        # overlap weld can complete a partial object); if the predictor is still
        # loading it defers and _retry_predictor_load_for_handoff finishes it.
        self._ensure_interactive_setup()
        self._enter_manual_refine_session()
        # Add mode: a click on empty ground starts a normal prediction. The flag
        # scopes the resting-click gate flip in manual_predict.
        self._refine_add_mode_active = True
        self._ai_add_kept_count = 0
        try:
            self.dock_widget.set_add_lane_armed(True, "ai")
        except (RuntimeError, AttributeError):
            pass
        self._refresh_ai_add_keep_button()
        QgsMessageLog.logMessage(
            "AI Add: armed (point at a missed object)",
            "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _retry_predictor_load_for_ai_add(self) -> None:
        """Restart a stalled model load for a deferred AI Add, mirroring
        _retry_predictor_load_for_handoff. The two share the same deferred
        import machinery, so this simply reuses that retry (the handoff flag is
        set for Add too)."""
        self._retry_predictor_load_for_handoff()

    # ------------------------------------------------------------------
    # Keeping one outline (the lane's Keep button and the S shortcut)
    # ------------------------------------------------------------------

    def _ai_add_outline_in_progress(self) -> bool:
        """True while something is on screen that a Keep would commit: a live
        mask, points already placed, a frozen crop part, or an object opened
        for editing. The single definition of "in progress" for the lane, so
        Keep, Escape and the Keep button can never disagree about it."""
        if self.current_mask is not None:
            return True
        if getattr(self, "_active_crop_points_positive", None):
            return True
        if getattr(self, "_frozen_sessions", None):
            return True
        if getattr(self, "_unfrozen_display_polygon", None) is not None:
            return True
        return bool(getattr(self, "_is_refining_saved_object", False))

    def _keep_ai_add_outline(self) -> bool:
        """Commit the outline on screen as its own object and STAY armed, so
        the next missed object is one click away. Returns False when there was
        nothing to keep, or when a crop encode owns the predictor pipe (the
        save no-ops there, and reporting success would lie).

        The save is the ordinary Manual one, so the outline lands with the
        session's refine settings (inherited from the review) and folds at Done
        like every other hand edit."""
        if not getattr(self, "_refine_add_mode_active", False):
            return False
        if getattr(self, "_encoding_in_progress", False):
            return False
        if not self._ai_add_outline_in_progress():
            return False
        before = len(self.saved_polygons)
        try:
            self._on_save_polygon()
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            QgsMessageLog.logMessage(
                f"AI Add: keep failed ({exc})",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return False
        kept = len(self.saved_polygons) > before
        self._refresh_ai_add_keep_button()
        if kept:
            self._ai_add_kept_count = getattr(self, "_ai_add_kept_count", 0) + 1
            fn = getattr(self.dock_widget, "set_add_lane_progress", None)
            if callable(fn):
                try:
                    fn(self._ai_add_kept_count)
                except (RuntimeError, AttributeError, TypeError):
                    pass
        return kept

    def _route_save_add_mode(self) -> bool:
        """The Save gesture (S, or the lane's Keep button) while Add is armed.

        Keeps the outline and leaves the lane armed. Returns True when the
        gesture was consumed, so the caller does not fall through to the
        reshape Save (which would close a session the user is still adding
        in). Nothing drawn means nothing to save, and Add stays armed: the
        ways out are Done adding and Escape."""
        if not getattr(self, "_refine_add_mode_active", False):
            return False
        self._keep_ai_add_outline()
        return True

    def _refresh_ai_add_keep_button(self) -> None:
        """Show Keep exactly while the lane has an outline to commit. Called
        from every path that changes what is on screen (prediction, undo, save,
        cancel), so the button never outlives its outline."""
        dock = getattr(self, "dock_widget", None)
        if dock is None:
            return
        fn = getattr(dock, "set_add_lane_keep_available", None)
        if not callable(fn):
            return
        try:
            fn(bool(getattr(self, "_refine_add_mode_active", False)) and self._ai_add_outline_in_progress())
        except (RuntimeError, AttributeError, TypeError):
            pass

    # ------------------------------------------------------------------
    # Exit + install-pending lifecycle
    # ------------------------------------------------------------------

    def _exit_ai_add_mode(self) -> None:
        """Disarm the Add lane WITHOUT tearing the session down: clear the flag
        and drop the lane's armed look. Idempotent; the session teardown (Done /
        Esc-to-rest / step change) is a separate step, so callers that want to
        leave the session entirely fold first. Safe to call from any teardown."""
        self._refine_add_mode_active = False
        if self.dock_widget is None:
            return
        try:
            method = self.dock_widget.get_correct_method()
        except (RuntimeError, AttributeError):
            method = "ai"
        try:
            # Drops the Keep button with the lane (set_add_lane_armed(False)).
            self.dock_widget.set_add_lane_armed(False, method)
        except (RuntimeError, AttributeError):
            pass

    def _clear_ai_add_install_pending(self) -> None:
        """Drop the pending install-then-Add intent and hide its review banner.

        Idempotent, and never called on its own: `_release_local_ai_install`
        is the one door out of an install, and it clears both lanes."""
        if not getattr(self, "_ai_add_install_pending", False):
            return
        self._ai_add_install_pending = False
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_review_installing(False)
            except (RuntimeError, AttributeError):
                pass

    # ------------------------------------------------------------------
    # Escape while Add is armed
    # ------------------------------------------------------------------

    def _route_escape_add_mode(self) -> bool:
        """Escape while the Add lane is armed.

        A live in-progress outline (points placed, active mask, or an open
        edit) is cancelled first and the lane stays armed for the next point.
        With nothing in progress, Add is a pure add session (entered from the
        lane, not from selecting a polygon), so leave it entirely: fold the
        objects kept this session and return to the resting Correct step.
        Returns True (Escape consumed)."""
        if self._ai_add_outline_in_progress():
            try:
                self._clear_active_mask_without_saving()
            except (RuntimeError, AttributeError):
                pass
            self._is_refining_saved_object = False
            self._active_refine_origin_entry = None
            try:
                self._discard_pending_manual_click()
            except (RuntimeError, AttributeError):
                pass
            self._refresh_ai_add_keep_button()
            return True
        # Nothing in progress: exit Add and the session. _exit_ai_add_mode drops
        # the lane; _on_reshape_done folds and returns to rest.
        self._exit_ai_add_mode()
        try:
            self._on_reshape_done()
        except (RuntimeError, AttributeError):
            pass
        return True
