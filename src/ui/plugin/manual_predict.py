"""Manual clicks, SAM prediction, mask visualization, undo and session reset.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsGeometry,
    QgsMessageLog,
    QgsPointXY,
)
from qgis.gui import QgsRubberBand
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QMessageBox,
)

from ...core.i18n import tr
from ...core.prompt_manager import FrozenCropSession
from ...core.qt_compat import DashLine, PolygonGeometry, SolidLine
from ...core.review_defaults import (
    REFINE_CLEAN_DEFAULT,
    REFINE_EXPAND_DEFAULT,
    REFINE_FILL_HOLES_DEFAULT,
    REFINE_FILL_HOLES_MAX_M2_DEFAULT,
    REFINE_MAX_SIZE_M2_DEFAULT,
    REFINE_MIN_AREA_DEFAULT,
    REFINE_MIN_SIZE_M2_DEFAULT,
    REFINE_ORTHO_DEFAULT,
    REFINE_POINTS_PCT_DEFAULT,
    REFINE_SIMPLIFY_DEFAULT,
    REFINE_SMOOTH_DEFAULT,
    REFINE_SMOOTH_ITERATIONS,
)
from ..canvas_palette import (
    PENDING_FILL,
    PENDING_STROKE,
)
from ..error_report_dialog import show_error_report


class ManualPredictMixin:
    """Manual clicks, SAM prediction, mask visualization, undo and session reset."""

    def _refine_click_is_stale(self) -> bool:
        """True when a point click arrived from a fix session that is already
        over, and the click must be dropped instead of seeding a new one.

        The point tool and the session flag are separate state. If an exit put
        the flag down without taking the tool back, the tool keeps answering
        clicks under a dock that shows no session, so a click on empty ground
        starts refining nothing, anywhere. Only ever true inside a review:
        Manual mode has no session flag to disagree with.
        """
        if getattr(self, "_auto_review", None) is None:
            return False
        busy = getattr(self, "_refine_handoff_active", False)
        busy = busy or getattr(self, "_refine_add_mode_active", False)
        return not (busy or getattr(self, "_is_refining_saved_object", False))

    def _drop_stale_refine_click(self) -> None:
        """Undo the marker the tool painted for a stale click, then sweep."""
        if self.map_tool:
            self.map_tool.remove_last_marker()
        self._sweep_stale_refine_canvas()

    def _on_positive_click(self, point):
        """Handle left-click: add positive point (select this element)."""
        if self._refine_click_is_stale():
            self._drop_stale_refine_click()
            return
        if self.predictor is None:
            if self.map_tool:
                self.map_tool.remove_last_marker()
            return

        # Transform click from canvas CRS to raster CRS for all downstream use
        raster_pt = self._transform_to_raster_crs(point)

        if not self._is_point_in_raster_extent(raster_pt):
            if self.map_tool:
                self.map_tool.remove_last_marker()
            layer_name = ""
            sel = self.dock_widget.layer_combo.currentLayer()
            if sel:
                layer_name = sel.name()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Click is outside the '{layer}' raster. To segment another raster, stop the current segmentation first.").format(layer=layer_name),  # noqa: E501
                level=Qgis.MessageLevel.Warning,
                duration=8
            )
            return

        # Refine-in-Manual, resting state: EDIT-ONLY review of the run's
        # detections. A single click is always a SELECTION gesture (Ctrl+click
        # toggles multi-selection), empty ground only deselects, and NOTHING
        # here starts a new object or a 3-8s encode: opening an object for SAM
        # editing is the deliberate double-click / E gesture. Adding brand-new
        # objects is base Manual mode's job (an accidental empty-ground click
        # used to spawn overlapping new selections mid-review). Handled BEFORE
        # the transport lock: selection is pure canvas work (hit test + bands,
        # never the predictor pipe), so it stays instant even while a
        # background encode (speculative selection prewarm, a just-closed
        # edit's crop) is still in flight.
        is_resting_click = self._refine_handoff_active
        is_resting_click = is_resting_click and not self._is_refining_saved_object
        is_resting_click = is_resting_click and self.current_mask is None
        is_resting_click = is_resting_click and not self._active_crop_points_positive
        # AI-assisted Add flips the gate: a click on empty ground starts a normal
        # prediction (the live outline) instead of a select. Scoped to the flag
        # ONLY, so the select-not-create rule holds for every other session.
        is_resting_click = is_resting_click and not getattr(
            self, "_refine_add_mode_active", False)
        if is_resting_click:
            if self.map_tool:
                self.map_tool.remove_last_marker()
            idx = self._hit_test_saved_polygon(raster_pt)
            if idx is not None:
                self._select_saved_polygon(
                    idx, additive=self._click_was_additive())
            else:
                self._deselect_saved_polygons()
            return

        # Transport lock: while an off-thread encode owns the predictor pipe,
        # NOTHING on the GUI thread may touch it. Remember this click (last one
        # wins) and replay it through the normal path when the encode finishes;
        # never start a second encode (PERF-01).
        if self._encoding_in_progress:
            self._remember_pending_manual_click("positive", point)
            self._wear_busy_cursor_for_crop()
            return

        # Refine-in-Manual, while editing: a left-click INSIDE another saved
        # detection switches to it (auto-save the current object, then select
        # the target). Clicks on empty ground or inside the active shape stay
        # editing clicks, so growing an object is untouched.
        is_editing_click = False
        if self._refine_handoff_active:
            is_editing_click = self._is_refining_saved_object
            is_editing_click = is_editing_click or self.current_mask is not None
            is_editing_click = is_editing_click or self._active_crop_points_positive
        if is_editing_click:
            idx = self._hit_test_saved_polygon(raster_pt)
            if idx is not None:
                if self.map_tool:
                    self.map_tool.remove_last_marker()
                target = self.saved_polygons[idx]
                was_editing = self._is_refining_saved_object
                try:
                    self._on_save_polygon()
                except Exception as e:  # noqa: BLE001
                    QgsMessageLog.logMessage(
                        f"Refine switch: save fold error: {e}",
                        "AI Segmentation", level=Qgis.MessageLevel.Warning)
                # The save may have re-shuffled saved_polygons (append/absorb):
                # re-resolve the target by identity. Absorbed into the save =>
                # nothing left to select.
                for i, pg in enumerate(self.saved_polygons):
                    if pg is target:
                        self._select_saved_polygon(i)
                        break
                # Say it: the click landed on a different object, so the one that
                # was open got saved and closed. Without a word for it, the edit
                # session ending reads as the click having been swallowed.
                if was_editing and not self._headless:
                    self.iface.messageBar().pushMessage(
                        "AI Segmentation",
                        tr("That is another object. The one you were editing is saved, and this one is now selected."),  # noqa: E501
                        level=Qgis.MessageLevel.Info,
                        duration=4
                    )
                return

        # Check crop status BEFORE adding to active points, so the zoom
        # detection sees the true "no active points" state after a save.
        crop_status = self._check_crop_status(raster_pt)

        if crop_status != "ok":
            # The crop is not ready: defer. Remember the click (its marker is
            # dropped now, re-added on replay) and start the async encode. The
            # crop-transition tail + this click's prediction run from the encode
            # completion (_on_manual_encode_done). Nothing is registered here.
            self._remember_pending_manual_click("positive", point)
            if not self._begin_async_reencode(crop_status, raster_pt):
                # Crop extraction failed synchronously (error already surfaced);
                # the click cannot be honored, so drop it (marker already gone).
                self._discard_pending_manual_click()
            return

        # Refine edit session: no special click path. The open object seeds
        # the prediction as mask_input (see _run_prediction), so this click
        # falls through to the normal Manual predict and refines the whole
        # shape with the object as prior, exactly like base Manual.

        # --- Fast path: the crop is already encoded, so predict synchronously
        # (predict is a fast decoder round-trip). This is also the path the
        # replayed click lands on once the encode has committed the new crop.
        # Save current mask state for undo before modifying anything
        # Cap at 30 entries (~30MB) to prevent unbounded memory growth.
        if len(self._mask_state_history) >= 30:
            self._mask_state_history.pop(0)
        self._mask_state_history.append(self._snapshot_mask_state())

        self.prompts.add_positive_point(raster_pt.x(), raster_pt.y())
        self._active_crop_points_positive.append((raster_pt.x(), raster_pt.y()))

        QgsMessageLog.logMessage(
            f"POSITIVE POINT at ({raster_pt.x():.6f}, {raster_pt.y():.6f})",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        # Remember the newest click (raster CRS, same frame as the point arrays
        # in _run_prediction) so Progressive Merge can bound this click to its
        # local change, and so the prediction knows a keep click may only add.
        self._last_click_point = (raster_pt.x(), raster_pt.y())
        self._last_click_polarity = "positive"

        # The predict blocks this thread, so the busy cursor and the dashed
        # outline go up before it and come down whatever it returns. Only when
        # this call started them: a predict running under a read the session
        # already announced must not take that read's cursor down with it.
        waiting = self._begin_correct_wait()
        try:
            predicted = self._run_prediction()
        finally:
            if waiting:
                self._end_correct_wait()
        if not predicted:
            self._rollback_failed_click("positive")
            return

        # Auto-revert when THIS CLICK added nothing: it found nothing, or what it
        # found stood clear of the object being edited. Not "the shape is empty":
        # a keep click may only grow the shape, so once a shape exists the total
        # is never empty and neither message would ever come back.
        undo_note = None
        if self._last_prediction_found_nothing():
            undo_note = tr("No element detected at this point. Try clicking on a different area.")
        elif self._last_click_took_from_another_object():
            undo_note = tr("That ground belongs to another object, so nothing was added. Edit that object instead, or join the two with Merge with neighbours.")  # noqa: E501
        elif self._last_click_stood_clear_of_shape():
            undo_note = tr("That area does not touch the object you are editing, so nothing was added. Reshaping works on one object at a time.")  # noqa: E501
        if undo_note and self._mask_state_history:
            self.prompts.undo()
            if self._active_crop_points_positive:
                self._active_crop_points_positive.pop()
            self._restore_mask_state(self._mask_state_history.pop())
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self._update_ui_after_prediction()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                undo_note,
                level=Qgis.MessageLevel.Info,
                duration=5
            )
            return

        # Live complete-don't-stack: if this selection now overlaps an existing
        # detection, weld them into one shape on the canvas right away (refine
        # handoff only; no-op otherwise).
        self._weld_active_into_overlaps()

    def _on_negative_click(self, point):
        """Handle right-click: add negative point (exclude this area)."""
        if self._refine_click_is_stale():
            self._drop_stale_refine_click()
            return
        if self.predictor is None:
            if self.map_tool:
                self.map_tool.remove_last_marker()
            return

        # Refine-in-Manual, resting state: a right-click SELECTS like a left
        # click (never an encode, never an edit). Carving happens INSIDE an
        # edit session (open with double-click / E, then right-click removes
        # area), so the resting state stays purely non-destructive. Handled
        # BEFORE the transport lock (pure canvas work): selection stays
        # instant while a background encode is in flight.
        is_resting_click = self._refine_handoff_active
        is_resting_click = is_resting_click and not self._is_refining_saved_object
        is_resting_click = is_resting_click and self.current_mask is None
        is_resting_click = is_resting_click and not self._active_crop_points_positive
        # AI-assisted Add flips the gate (see _on_positive_click): scoped to the
        # flag ONLY, so a right-click during Add carves the new outline instead
        # of selecting.
        is_resting_click = is_resting_click and not getattr(
            self, "_refine_add_mode_active", False)
        if is_resting_click:
            raster_pt0 = self._transform_to_raster_crs(point)
            if self._is_point_in_raster_extent(raster_pt0):
                if self.map_tool:
                    self.map_tool.remove_last_marker()
                idx = self._hit_test_saved_polygon(raster_pt0)
                if idx is not None:
                    self._select_saved_polygon(
                        idx, additive=self._click_was_additive())
                else:
                    self._deselect_saved_polygons()
                return

        # Transport lock: defer to the encode completion while a worker owns the
        # predictor pipe (PERF-01), so a right-click during an encode is
        # remembered, never routed into a second encode.
        if self._encoding_in_progress:
            self._remember_pending_manual_click("negative", point)
            self._wear_busy_cursor_for_crop()
            return

        # Refine edit session: right-click removes area from the open object
        # through the normal Manual predict. It needs no prior positive point:
        # the object itself is the positive context (seeded as mask_input by
        # _run_prediction).
        refine_edit = self._refine_edit_session_active()

        # Block negative points until at least one positive point exists
        if not refine_edit and len(self.prompts.positive_points) == 0:
            if self.map_tool:
                self.map_tool.remove_last_marker()
            QgsMessageLog.logMessage(
                "Negative point ignored - need at least one positive point first",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )
            return

        # Transform click from canvas CRS to raster CRS for all downstream use
        raster_pt = self._transform_to_raster_crs(point)

        if not self._is_point_in_raster_extent(raster_pt):
            if self.map_tool:
                self.map_tool.remove_last_marker()
            layer_name = ""
            sel = self.dock_widget.layer_combo.currentLayer()
            if sel:
                layer_name = sel.name()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Click is outside the '{layer}' raster. To segment another raster, stop the current segmentation first.").format(layer=layer_name),  # noqa: E501
                level=Qgis.MessageLevel.Warning,
                duration=8
            )
            return

        crop_status = self._check_crop_status(raster_pt)

        # Negative points outside the current crop don't make sense - they're
        # meant to refine the current selection, not start a new one. In an
        # edit session the rule does not apply (a grown object can extend past
        # the crop encoded at open time): fall through to the re-encode below.
        if crop_status == "outside_bounds" and not refine_edit:
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Right-click must be inside the current selection area."),
                level=Qgis.MessageLevel.Info,
                duration=4
            )
            return

        if crop_status != "ok":
            # Zoom changed / no crop: defer. Remember the click (marker dropped
            # now, re-added on replay) and start the async encode; the transfer
            # tail + this click's prediction run on completion. Nothing is
            # registered here.
            self._remember_pending_manual_click("negative", point)
            if not self._begin_async_reencode(crop_status, raster_pt):
                # Crop extraction failed synchronously (error already surfaced).
                self._discard_pending_manual_click()
            return

        # --- Fast path: crop already encoded, predict synchronously (also the
        # path the replayed click lands on after the encode commits the crop).
        if len(self._mask_state_history) >= 30:
            self._mask_state_history.pop(0)
        self._mask_state_history.append(self._snapshot_mask_state())

        self.prompts.add_negative_point(raster_pt.x(), raster_pt.y())
        self._active_crop_points_negative.append((raster_pt.x(), raster_pt.y()))

        QgsMessageLog.logMessage(
            f"NEGATIVE POINT at ({raster_pt.x():.6f}, {raster_pt.y():.6f})",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        # Remember the newest click (raster CRS, same frame as the point arrays
        # in _run_prediction) so Progressive Merge can bound this click to its
        # local change. Read only when the server flag is on.
        self._last_click_point = (raster_pt.x(), raster_pt.y())
        self._last_click_polarity = "negative"

        # Same wait treatment as the keep click above: the predict blocks, so
        # say so on the polygon and on the cursor for as long as it runs.
        waiting = self._begin_correct_wait()
        try:
            predicted = self._run_prediction()
        finally:
            if waiting:
                self._end_correct_wait()
        if not predicted:
            self._rollback_failed_click("negative")
            return

        # Auto-revert when THIS CLICK found nothing. Not "the shape is empty": a
        # keep click may only grow the shape, so once a shape exists the total is
        # never empty and this message would never come back.
        if self._last_prediction_found_nothing() and self._mask_state_history:
            self.prompts.undo()
            if self._active_crop_points_negative:
                self._active_crop_points_negative.pop()
            self._restore_mask_state(self._mask_state_history.pop())
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self._update_ui_after_prediction()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("No element detected at this point. Try clicking on a different area."),
                level=Qgis.MessageLevel.Info,
                duration=4
            )
            return

    def _run_prediction(self) -> bool:
        """Run SAM prediction using active crop points only.

        When frozen sessions exist, only the active crop's points are sent
        to SAM (frozen polygons are composited during visualization).

        Returns True when a prediction was stored, False on any failure so
        the caller can roll the click back.
        """
        import numpy as np
        from rasterio.transform import from_bounds as transform_from_bounds

        # Use only active crop points for prediction (not frozen points)
        active_pos = self._active_crop_points_positive
        active_neg = self._active_crop_points_negative
        all_active = active_pos + active_neg
        if not all_active:
            return False

        if self._current_crop_info is None:
            QgsMessageLog.logMessage(
                "No crop encoded yet - cannot predict",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )
            return False

        # A dead worker (cleaned up after a transport error) leaves the crop
        # info in place but no encoded image: every click would fail silently
        # forever. Re-encode the same crop transparently. This is a rare
        # recovery path, reached only when the crop was expected "ok", so no
        # async worker owns the pipe: a SYNCHRONOUS blocking re-encode here is
        # transport-safe (the main thread owns the pipe) and keeps predict fully
        # synchronous, at the cost of a brief freeze in this recovery case only.
        if not self.predictor.is_image_set:
            QgsMessageLog.logMessage(
                "Worker has no encoded image - re-encoding current crop",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )
            b = self._current_crop_info["bounds"]
            center = QgsPointXY((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
            override = (self._current_crop_actual_mupp if self._is_online_layer
                        else self._current_crop_scale_factor)
            if not self._encode_crop_blocking(center, mupp_override=override):
                return False

        crop_bounds = self._current_crop_info["bounds"]
        img_shape = self._current_crop_info["img_shape"]
        img_height, img_width = img_shape

        minx, miny, maxx, maxy = crop_bounds
        img_clip_transform = transform_from_bounds(
            minx, miny, maxx, maxy, img_width, img_height)

        # Build point arrays from active crop points only
        from rasterio import transform as rio_transform
        point_coords_list = []
        point_labels_list = []
        for x, y in active_pos:
            row, col = rio_transform.rowcol(img_clip_transform, x, y)
            point_coords_list.append([col, row])
            point_labels_list.append(1)
        for x, y in active_neg:
            row, col = rio_transform.rowcol(img_clip_transform, x, y)
            point_coords_list.append([col, row])
            point_labels_list.append(0)

        point_coords = np.array(point_coords_list)
        point_labels = np.array(point_labels_list)

        # Use previous low_res_mask for iterative refinement (includes
        # transferred mask context after zoom re-encode)
        mask_input = None
        if self.current_low_res_mask is not None:
            mask_input = self.current_low_res_mask
        elif self._is_refining_saved_object or self._frozen_sessions:
            # There is a shape in progress, so this click continues it: seed SAM
            # with that shape rasterized onto the current crop (the same context
            # transfer a zoom re-encode uses), and the click REFINES the shape
            # instead of segmenting an unrelated element under the cursor.
            #
            # The frozen parts count. A click that lands outside the encoded crop
            # freezes the shape so far and reads new imagery, and without the
            # shape arriving on the new crop as context, clicking the garden
            # beside an open house segmented the garden on its own.
            mask_input = self._refine_polygon_mask_input()

        # Snapshot the previous mask for Progressive Merge BEFORE predict. Only a
        # continuation click has prior context (mask_input set); the first point
        # of a fresh object has none, so the merge is skipped there.
        prev_mask_for_merge = self.current_mask if mask_input is not None else None
        if prev_mask_for_merge is None and mask_input is not None:
            # A continuation click with no SAM mask yet (a freshly opened
            # detection, or the first click after moving to another crop). Seed
            # the previous shape from the geometry rasterized to the crop grid,
            # so even the FIRST negative click is bounded to a local window
            # instead of collapsing the whole footprint.
            try:
                shape = self._shape_in_progress_geometry()
                if shape is not None:
                    prev_mask_for_merge = self._rasterize_geom_to_crop(
                        shape, crop_bounds, img_shape)
            except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
                prev_mask_for_merge = None
        elif prev_mask_for_merge is not None and self._frozen_sessions:
            # The live mask holds only this crop. Parts frozen when an earlier
            # click moved crops are shape too, and a click that reaches one of
            # them reaches the shape, so they join the prior for this crop.
            try:
                shape = self._shape_in_progress_geometry()
                frozen_here = (None if shape is None else
                               self._rasterize_geom_to_crop(
                                   shape, crop_bounds, img_shape))
                if frozen_here is not None:
                    prev_mask_for_merge = np.logical_or(
                        prev_mask_for_merge[:img_height, :img_width].astype(bool),
                        frozen_here[:img_height, :img_width])
            except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
                pass

        # Use multimask only on the very first point of a new polygon/crop
        # (more accurate initial selection). For subsequent points or
        # re-encoded crops with transferred mask, use single mask.
        one_positive = len(active_pos) == 1
        no_negatives = len(active_neg) == 0
        is_first_point = one_positive and no_negatives and mask_input is None
        use_multimask = is_first_point

        try:
            masks, scores, low_res_masks = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                multimask_output=use_multimask,
            )
        except RuntimeError as e:
            error_str = str(e)
            QgsMessageLog.logMessage(
                f"Prediction failed: {error_str}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            self._track_manual_run_failed()
            # Classified error alongside the boolean failure: a DLL crash, a
            # dead SAM subprocess and a transport error need different fixes.
            # Resolved before the telemetry try so the report dialog below always
            # has a code, even if the telemetry import fails.
            is_dll_error = "DLL" in error_str or "Visual C++" in error_str
            if is_dll_error:
                code = "predict_dll_error"
            elif "subprocess" in error_str.lower() or "rpc" in error_str.lower():
                code = "predict_worker_died"
            else:
                code = "predict_runtime_error"
            try:
                from ...core import telemetry_errors
                # A non-headless DLL error reports through show_error_report
                # below, which fires its own telemetry with the same code;
                # skip here so the same occurrence is not counted twice.
                if not (is_dll_error and not self._headless):
                    telemetry_errors.track_plugin_error(
                        stage="segment", error_code=code, message=error_str)
            except Exception:
                pass  # nosec B110
            if self._headless:
                self._headless_error = error_str
                return False
            if is_dll_error:
                show_error_report(
                    self.iface.mainWindow(),
                    tr("Segmentation failed"),
                    error_str,
                    error_code="predict_dll_error",
                )
            else:
                # Any other RuntimeError (SAM subprocess died, JSON-RPC transport
                # error) used to fail silently. Open the report dialog so the
                # failure is actionable (copy logs + email), not a dead-end toast;
                # track=False since it was already counted above.
                show_error_report(
                    self.iface.mainWindow(),
                    tr("Segmentation failed"),
                    error_str,
                    error_code=code,
                    track=False,
                )
            return False
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Unexpected prediction error: {str(e)}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            self._track_manual_run_failed()
            try:
                from ...core import telemetry_errors
                telemetry_errors.track_plugin_error(
                    stage="segment",
                    error_code=type(e).__name__ or "predict_unexpected_error",
                    message=str(e))
            except Exception:
                pass  # nosec B110
            if not self._headless:
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("Segmentation failed. Please try again."),
                    level=Qgis.MessageLevel.Warning,
                    duration=5,
                )
            return False

        if use_multimask:
            total_pixels = masks[0].shape[0] * masks[0].shape[1]
            mask_areas = [int(m.sum()) for m in masks]

            # Avoid selecting the whole crop when clicking on small elements
            # in repetitive patterns (e.g. trees in an orchard). SAM's highest
            # score often goes to the "all similar elements" interpretation.
            small_enough = [
                i for i in range(len(scores))
                if 0 < mask_areas[i] < 0.8 * total_pixels
            ]
            if small_enough:
                best_idx = max(small_enough, key=lambda i: scores[i])
            else:
                best_idx = min(range(len(scores)), key=lambda i: mask_areas[i])

            QgsMessageLog.logMessage(
                f"Multimask: areas={mask_areas}, scores={[round(float(s), 3) for s in scores]}, picked={best_idx}",
                "AI Segmentation", level=Qgis.MessageLevel.Info
            )
            self.current_mask = masks[best_idx]
            self.current_score = float(scores[best_idx])
            self.current_low_res_mask = low_res_masks[best_idx:best_idx + 1]
        else:
            self.current_mask = masks[0]
            self.current_score = float(scores[0])
            self.current_low_res_mask = low_res_masks

        # SAM masks cover the full padded square; keep only the real image
        # area so reflect padding at raster edges cannot leak mirrored
        # ghost polygons outside the raster. Materialised, not left as a view:
        # a slice would pin the whole decoded multimask buffer (three full
        # masks) for as long as the session holds the one mask it uses.
        self.current_mask = np.ascontiguousarray(
            self.current_mask[:img_height, :img_width])

        # The click's OWN answer, before anything merges the shape into it. Both
        # the "nothing here" revert and the one-piece rule below judge the click
        # on this, never on a mask that already carries the shape.
        raw_answer = self.current_mask

        # The newest click as mask pixel row/col, in the SAME img_clip_transform
        # used for the point arrays above. Both locality rules below need it.
        click_rc = None
        try:
            if getattr(self, "_last_click_point", None) is not None:
                cx, cy = self._last_click_point
                crow, ccol = rio_transform.rowcol(img_clip_transform, cx, cy)
                click_rc = (int(crow), int(ccol))
        except Exception:  # noqa: BLE001 -- click path is best-effort  # nosec B110
            click_rc = None

        # Progressive Merge (FocalClick): bound this click to the region it
        # changed so it cannot reshape a part the user already accepted. Off when
        # the dial is unset or unreachable, which fails open to a full-mask
        # update. Best-effort: on any error self.current_mask stays exactly the
        # raw new SAM mask, so the click path can never crash here.
        try:
            from ...core.detection_policy import progressive_merge_enabled
            may_merge = prev_mask_for_merge is not None and progressive_merge_enabled()
            if may_merge and click_rc is not None:
                from ...core.progressive_merge import progressive_merge_masks
                # Crop the previous mask the same way as the new one so shapes
                # match; a mismatch is handled inside progressive_merge_masks.
                prev_c = prev_mask_for_merge[:img_height, :img_width]
                self.current_mask = progressive_merge_masks(
                    prev_c, self.current_mask, click_rc[0], click_rc[1])
        except Exception:  # noqa: BLE001 -- click path is best-effort  # nosec B110
            pass

        # A positive click says "this belongs to the shape too", so it may only
        # ADD to it. The model is given the shape so far as context, but that is
        # a hint it can overrule: clicking the garden beside an open house
        # returned the garden on its own, and the house was gone. The click's own
        # mask is kept as the leading edge and simply cannot lose ground the
        # shape already held. Trimming is what the right click is for, and it
        # skips this.
        self._last_prediction_empty = not bool(self.current_mask.any())
        self._last_click_stood_clear = False
        self._last_click_took_from_another = False
        if getattr(self, "_last_click_polarity", "positive") == "positive":
            if self._is_refining_saved_object:
                # Editing ONE object: the answer joins that object, or it is not
                # part of it. Pixel size comes from this crop, so the weld gap is
                # a ground distance whatever the resolution.
                px_size = (maxx - minx) / float(img_width) if img_width else 0.0
                (self.current_mask, self._last_click_stood_clear,
                 self._last_click_took_from_another) = \
                    self._grow_open_object_with_click(
                        self.current_mask, raw_answer, prev_mask_for_merge,
                        crop_bounds, img_shape, px_size)
            else:
                self.current_mask = self._grown_by_shape_so_far(
                    self.current_mask, prev_mask_for_merge, img_height, img_width)
        elif prev_mask_for_merge is not None and click_rc is not None:
            # A trim click says "not this bit", so it may only TAKE ground, and
            # only the piece under the cursor. Its answer is a fresh reading of
            # the whole object rather than an edit of it, and on a long shape
            # the model returns one short section, so applied whole it deleted
            # the road the user was trimming a car off. Unconditional, like the
            # keep rule above: this is the shape of the edit, not a tuning.
            try:
                from ...core.progressive_merge import subtract_click_region
                self.current_mask = subtract_click_region(
                    prev_mask_for_merge[:img_height, :img_width],
                    self.current_mask, click_rc[0], click_rc[1])
            except Exception:  # noqa: BLE001 -- click path is best-effort  # nosec B110
                pass

        # The prediction supersedes the display-only polygon INSIDE this crop.
        # Whatever lay outside it cannot be represented in a mask, so it is kept
        # as a frozen part rather than dropped.
        self._freeze_display_polygon_outside_crop(crop_bounds)
        self._unfrozen_display_polygon = None

        # Get CRS from layer
        crs_value = None
        try:
            if self._current_layer and self._current_layer.crs().isValid():
                crs_value = self._current_layer.crs().authid()
        except RuntimeError:
            pass

        self.current_transform_info = {
            "bbox": (minx, maxx, miny, maxy),
            "img_shape": (img_height, img_width),
            "crs": crs_value,
        }

        self._update_ui_after_prediction()
        return True

    def _last_prediction_found_nothing(self) -> bool:
        """Did the last click's own answer come back empty? Judged on the raw
        answer, before it was merged into the shape being edited."""
        if self.current_mask is None:
            return False
        return bool(getattr(self, "_last_prediction_empty", False))

    @staticmethod
    def _grown_by_shape_so_far(new_mask, prior_mask, img_height, img_width):
        """``new_mask`` plus everything the shape already covered.

        ``prior_mask`` is the shape before this click, on this crop's grid (the
        previous mask, or the open object's geometry rasterized onto it). None,
        or a mask of another shape, leaves the new mask untouched: a click can
        then still come back smaller, which is the pre-existing behaviour and
        better than pairing two grids that do not line up.
        """
        if prior_mask is None:
            return new_mask
        try:
            import numpy as np
            prior = prior_mask[:img_height, :img_width]
            if prior.shape != new_mask.shape:
                return new_mask
            grown = np.logical_or(new_mask.astype(bool), prior.astype(bool))
            return grown.astype(new_mask.dtype, copy=False)
        except Exception:  # noqa: BLE001 -- never break a click over this
            return new_mask

    def _grow_open_object_with_click(self, new_mask, raw_answer, prior_mask,
                                     crop_bounds, img_shape, pixel_size_m):
        """``(mask, stood_clear, took_from_another)`` for a keep click on the one
        object that is open for editing.

        Two rules, in this order. The click may not take ground that belongs to
        another detection, and what is left must reach the object being edited.
        Order matters: clipping second could leave the far side of a neighbour
        hanging off the shape as an island again.

        Ground the shape ALREADY held is never clipped, so a detection that
        arrived overlapping a neighbour is not carved up by a click that went
        nowhere near it.
        """
        import numpy as np

        img_height, img_width = img_shape
        took = False
        answer = raw_answer
        try:
            others = self._other_objects_mask_for_crop(crop_bounds, img_shape)
            if others is not None:
                others = others[:img_height, :img_width].astype(bool)
                if prior_mask is not None:
                    prior_b = prior_mask[:img_height, :img_width].astype(bool)
                    if prior_b.shape == others.shape:
                        others = np.logical_and(others, np.logical_not(prior_b))
                if others.shape == new_mask.shape:
                    keep = np.logical_not(others)
                    took = bool(np.logical_and(
                        raw_answer.astype(bool), others).any())
                    new_mask = np.logical_and(
                        new_mask.astype(bool), keep).astype(new_mask.dtype,
                                                            copy=False)
                    answer = np.logical_and(raw_answer.astype(bool), keep)
        except Exception as e:  # noqa: BLE001 -- a click must not fail over this
            QgsMessageLog.logMessage(
                f"Could not keep the click off the other objects: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)

        grown, stood_clear = self._grown_in_one_piece(
            new_mask, prior_mask, img_height, img_width, pixel_size_m, answer)

        # Which of the two rules to report: the neighbour wins, because it is the
        # one the user can act on (fix that object, or merge the two).
        if took and prior_mask is not None:
            try:
                prior_b = prior_mask[:img_height, :img_width].astype(bool)
                added = np.logical_and(np.asarray(grown).astype(bool),
                                       np.logical_not(prior_b))
                if not added.any():
                    return grown, False, True
            except Exception:  # noqa: BLE001 -- reporting aid only  # nosec B110
                pass
        return grown, stood_clear, False

    def _grown_in_one_piece(self, new_mask, prior_mask, img_height, img_width,
                            pixel_size_m, raw_answer=None):
        """``(mask, click_stood_clear)`` for a keep click on an open object.

        Refining works on ONE object, so the click's answer is kept where it
        reaches that object and dropped where it stands clear of it: clicking
        the garden beside an open house used to leave the garden sitting there
        as a second island. ``raw_answer`` is the click's own answer before the
        progressive-merge step folded the shape into ``new_mask``; without it
        every click would read as having landed on the shape. Falls back to the
        plain union on any failure, so the rule can cost a click nothing worse
        than the old behaviour.
        """
        if prior_mask is None:
            return new_mask, False
        try:
            prior = prior_mask[:img_height, :img_width]
            if prior.shape != new_mask.shape:
                return new_mask, False
            from ...core.shape_growth import grow_shape_with_click
            return grow_shape_with_click(
                prior, new_mask, pixel_size_m, click_answer=raw_answer)
        except Exception as e:  # noqa: BLE001 -- never break a click over this
            QgsMessageLog.logMessage(
                f"Could not keep the shape in one piece: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return self._grown_by_shape_so_far(
                new_mask, prior_mask, img_height, img_width), False

    def _last_click_stood_clear_of_shape(self) -> bool:
        """Did the last keep click answer with something that never reached the
        object being edited? Its area was dropped and the shape is unchanged."""
        return bool(getattr(self, "_last_click_stood_clear", False))

    def _last_click_took_from_another_object(self) -> bool:
        """Did the last keep click ask for ground that belongs to another
        detection, and have nothing left once that was refused?"""
        return bool(getattr(self, "_last_click_took_from_another", False))

    def _freeze_display_polygon_outside_crop(self, crop_bounds) -> None:
        """Park the part of the open object that lies outside the encoded crop.

        A mask can only describe what is inside its own crop, so the rest has to
        survive as geometry or a single click would silently shorten an object
        wider than one crop. No-op when there is no display polygon or it fits.
        """
        base = getattr(self, "_unfrozen_display_polygon", None)
        if base is None or base.isEmpty():
            return
        try:
            from qgis.core import QgsRectangle
            minx, miny, maxx, maxy = crop_bounds
            crop = QgsGeometry.fromRect(QgsRectangle(minx, miny, maxx, maxy))
            if crop.contains(base):
                return
            outside = base.difference(crop)
            if outside is None or outside.isEmpty():
                return
            self._frozen_sessions.append(
                FrozenCropSession(polygon=QgsGeometry(outside)))
        except Exception as e:  # noqa: BLE001 -- best effort, never break a click
            QgsMessageLog.logMessage(
                f"Could not keep the part outside the crop: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)

    def _rollback_failed_click(self, polarity: str):
        """Undo all state added by a click whose prediction failed.

        Without this, a failed prediction leaves a marker and a prompt point
        that never contributed to the mask, silently desyncing every later
        prediction and undo.
        """
        self.prompts.undo()
        if polarity == "positive" and self._active_crop_points_positive:
            self._active_crop_points_positive.pop()
        elif polarity == "negative" and self._active_crop_points_negative:
            self._active_crop_points_negative.pop()
        if self._mask_state_history:
            self._restore_mask_state(self._mask_state_history.pop())
        if self.map_tool:
            self.map_tool.remove_last_marker()
        if not self._headless:
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Something went wrong with this click, so it was not applied. Please try again."),
                level=Qgis.MessageLevel.Warning,
                duration=5
            )

    def _update_ui_after_prediction(self):
        if not self.dock_widget:
            return
        pos_count, neg_count = self.prompts.point_count
        self.dock_widget.set_point_count(pos_count, neg_count)
        # The AI Add lane offers Keep only while there is an outline to keep,
        # and this is the beat every click and every undo passes through.
        refresh = getattr(self, "_refresh_ai_add_keep_button", None)
        if refresh is not None:
            refresh()

        if self.current_mask is not None:
            mask_pixels = int(self.current_mask.sum())
            # A mask restored by undo can carry no score (e.g. seeded from a
            # saved polygon before any prediction): log 0, never crash.
            score = self.current_score if self.current_score is not None else 0.0
            QgsMessageLog.logMessage(
                f"Segmentation result: score={score:.3f}, mask_pixels={mask_pixels}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )
            self._update_mask_visualization()
        else:
            # No active mask: _update_mask_visualization keeps any frozen
            # or unfrozen polygons on screen instead of wiping them.
            self._update_mask_visualization()

        self._safe_restore_canvas_focus()

    def _apply_mask_band_style(self) -> None:
        """Colour the active-mask band. One color language: blue = editable (not
        yet saved), green = validated. So the object OPEN for editing stays the
        same pending-blue as every other unsaved seed; it only reads as "the one
        I'm editing" through a thicker outline, never a third hue (the old amber
        active-state broke the blue -> green story).

        One exception, and it is the review's, not Manual's: a fix session
        opened from the Automatic review under the Outline display mode drops
        the fill. Outline mode is a promise that the imagery stays visible, and
        the polygon being edited is the one the user is comparing against the
        ground. The bolder blue stroke still says which object is open."""
        if self.mask_rubber_band is None:
            return
        fill = PENDING_FILL
        if (getattr(self, "_refine_handoff_active", False)
                and getattr(self, "_auto_display_mode", "") == "outline"):
            fill = QColor(PENDING_FILL)
            fill.setAlpha(0)
        self.mask_rubber_band.setColor(fill)
        self.mask_rubber_band.setStrokeColor(PENDING_STROKE)
        # A bolder outline while an object is open for editing in a
        # refine/handoff, so it stands apart from the flat pending seeds
        # without introducing a non-blue colour.
        editing = self._refine_handoff_active or self._is_refining_saved_object
        self.mask_rubber_band.setWidth(3 if editing else 2)
        # Waiting on the model: the outline goes dashed, so the polygon says on
        # its own that it is busy instead of leaving the cursor to carry it
        # alone. Every path that ends the wait comes back through here, so the
        # dash cannot outlive the work (see correct_focus.py).
        try:
            self.mask_rubber_band.setLineStyle(
                DashLine if self._correct_wait_showing() else SolidLine)
        except (RuntimeError, AttributeError):
            pass

    def _crop_pixel_size_units(self, transform_info) -> float:
        """Ground size of one mask pixel in the crop, in CRS units. 0 when it
        cannot be measured.

        The crop covers a square region of GROUND, so on a raster whose two
        axes measure differently a pixel is not the same size in CRS units
        across as down. Every consumer of this number feeds it to a
        direction-agnostic buffer or simplify, so it reports the FINER of the
        two: the coarser one would erode or erase real geometry along the fine
        axis, while the finer one only leaves a few extra vertices along the
        coarse one. The two axes agree on a projected raster, so the answer
        there is unchanged.
        """
        if not transform_info:
            return 0.0
        bbox = transform_info.get("bbox", [0, 1, 0, 1])
        img_shape = transform_info.get("img_shape", (1024, 1024))
        # bbox is (minx, maxx, miny, maxy), not the usual corner pair order.
        width_pixels = max(int(img_shape[1]), 1)
        bbox_width = float(bbox[1]) - float(bbox[0])
        if bbox_width == 0:
            return 0.0
        pixel_size = bbox_width / width_pixels
        height_pixels = max(int(img_shape[0]), 1)
        bbox_height = float(bbox[3]) - float(bbox[2])
        if bbox_height != 0:
            pixel_size = min(pixel_size, bbox_height / height_pixels)
        return pixel_size

    def _manual_metres_per_unit(self, ref_x: float, ref_y: float) -> float:
        """Ground metres per unit of the current layer CRS near (ref_x, ref_y).

        Mirrors the Automatic review's _auto_crs_metres_per_unit: a Web Mercator
        unit is well under a metre, a geographic CRS counts in degrees, so a
        ground-metre dial has to cross that gap before it touches a geometry.
        Answers 1.0 on any failure (reads a metre setting as CRS units)."""
        layer = getattr(self, "_current_layer", None)
        if layer is None:
            return 1.0
        try:
            crs = layer.crs()
            if not crs.isValid():
                return 1.0
            geographic = bool(crs.isGeographic())
        except Exception:  # noqa: BLE001 -- an unusable CRS means no conversion
            return 1.0
        try:
            from ...core.layer_conventions import make_area_measurer
            step = 0.001 if geographic else 1.0
            metres = float(make_area_measurer(crs).measureLine(
                QgsPointXY(ref_x, ref_y), QgsPointXY(ref_x + step, ref_y)))
            return metres / step if metres > 0 else 1.0
        except Exception:  # noqa: BLE001 -- never block a refine on a measure
            return 1.0

    def _manual_unit_aspect(self, ref_x: float, ref_y: float) -> float:
        """How much longer one y unit of the current layer CRS is than one x
        unit near (ref_x, ref_y). 1.0 in a projected CRS, above 1 in a
        geographic one, where squaring a footprint on raw coordinates leaves
        every corner tilted. See core.layer_conventions.ground_unit_aspect."""
        layer = getattr(self, "_current_layer", None)
        if layer is None:
            return 1.0
        try:
            from ...core.layer_conventions import ground_unit_aspect
            return ground_unit_aspect(layer.crs(), ref_x, ref_y)
        except Exception:  # noqa: BLE001 -- never block a refine on a measure
            return 1.0

    def _manual_apply_right_angles(self, combined, transform_info, tolerance):
        """Square ``combined`` with the SAME engine and server dials the
        Automatic review uses. Manual has no prompt (no class), so the tick
        means "treat as man-made" and the GENERIC server dials apply: the
        regularizer's ground snap tolerance (regularize_tolerance_m against the
        object's own ground size) and de-staircase distance (destair_tolerance_m),
        both converted from ground metres to the layer's CRS units. Best-effort:
        returns ``combined`` unchanged on any failure."""
        from ...core.polygon_exporter import apply_right_angles
        pixel_units = self._crop_pixel_size_units(transform_info)
        # Resolve the safety envelope up front so the fallback path below still
        # has a bound value if the policy import inside the try fails. Neutral
        # (no-op) by default, so this changes nothing until the server tunes it.
        try:
            from ...core.detection_policy import regularize_envelope
            _envelope = regularize_envelope()
        except Exception:  # noqa: BLE001 -- neutral envelope on any failure  # nosec B110
            _envelope = None
        try:
            from ...core.detection_policy import (
                destair_tolerance_m,
                regularize_settings,
                regularize_tolerance_m,
            )
            bbox = combined.boundingBox()
            factor = self._manual_metres_per_unit(
                bbox.center().x(), bbox.center().y())
            if factor <= 0:
                factor = 1.0
            span_units = min(bbox.width(), bbox.height())
            reg_tol_m = regularize_tolerance_m(
                pixel_units * factor, span_units * factor)
            reg_tol = reg_tol_m / factor
            destair = destair_tolerance_m(pixel_units * factor) / factor
            s = regularize_settings()
            return apply_right_angles(
                combined,
                destair_tol=max(0.0, destair - tolerance),
                tolerance_m=reg_tol,
                allow_diagonal=bool(s["allow_diagonal"]),
                allow_circles=bool(s["allow_circles"]),
                min_keep_iou=float(s["min_keep_iou"]),
                diagonal_reduction=float(s["diagonal_reduction"]),
                circle_threshold=float(s["circle_threshold"]),
                # Multi-direction path: OFF unless the server turns it on, the
                # same dial the Automatic review forwards. With it on, a
                # building whose wing sits at an angle to the main block keeps
                # each wing on its own grid instead of coming back staircased.
                multi_direction=bool(s["multi_direction"]),
                multi_max_groups=int(s["multi_max_groups"]),
                multi_min_separation_deg=float(s["multi_min_separation_deg"]),
                unit_aspect=self._manual_unit_aspect(
                    bbox.center().x(), bbox.center().y()),
                envelope=_envelope)
        except Exception:  # noqa: BLE001 -- fall back to the pixel-anchored path
            destair3 = self._compute_simplification_tolerance(transform_info, 1.5)
            centre = combined.boundingBox().center()
            return apply_right_angles(
                combined,
                destair_tol=max(0.0, destair3 - tolerance),
                tolerance_m=destair3,
                unit_aspect=self._manual_unit_aspect(centre.x(), centre.y()),
                envelope=_envelope)

    def _manual_despike_distance(self, combined, transform_info) -> float:
        """The spike-cut opening distance for ``combined``, in the layer's CRS
        units. Resolved the way the Automatic review resolves it: a ground dial
        (core.detection_policy.despike_tolerance_m) crossed into CRS units by
        the ground scale under the object. 0.0 is the OFF state and is what an
        untuned server gives, so this stays offline-safe like the rest of
        Manual."""
        try:
            from ...core.detection_policy import despike_tolerance_m
            pixel_units = self._crop_pixel_size_units(transform_info)
            centre = combined.boundingBox().center()
            factor = self._manual_metres_per_unit(centre.x(), centre.y())
            if factor <= 0:
                factor = 1.0
            return despike_tolerance_m(pixel_units * factor) / factor
        except Exception:  # noqa: BLE001 -- the step stays off on any failure
            return 0.0

    def _shape_active_geometry(self, combined, transform_info):
        """Trim spikes, simplify, right angles, round corners on the active mask
        polygon, in geometry space. The SINGLE geometry-refine tail shared by
        the live preview and the save/export path, so the polygon a user gets is
        always exactly the one previewed. Mirrors the Automatic review's order
        and dials. Size filtering is left to each caller. Returns None if the
        geometry is empty."""
        if combined is None or combined.isEmpty():
            return None
        if self._refine_ortho:
            # Cut thin spikes and necks BEFORE squaring, the same slot and the
            # same op as the Automatic review, so the regularizer never snaps a
            # spike into a rotated diamond. A Manual selection can legitimately
            # be multipart, so its parts are kept.
            from ...core.polygon_exporter import despike_thin_necks
            combined = despike_thin_necks(
                combined,
                self._manual_despike_distance(combined, transform_info),
                preserve_parts=bool(combined.isMultipart()))
            if combined is None or combined.isEmpty():
                return None
        # Right angles owns the outline: an extra generic cleanup can erase a
        # narrow part and corner rounding reverses what was just squared. The
        # panel disables both controls, but a disabled widget is not a value:
        # a programmatic restore (a saved polygon, a handoff seed) sets the
        # plugin's fields without ever emitting, so the decision has to be made
        # HERE, where the geometry is, and not only in the dock's emit.
        ortho_on = bool(getattr(self, "_refine_ortho", False))
        # Trim spikes (morphological opening): strip thin attached fringe. The
        # same op as the Automatic review; px scaled to CRS units by the crop.
        open_px = (0.0 if ortho_on
                   else float(getattr(self, "_refine_clean", 0.0) or 0.0))
        if open_px > 0:
            open_dist = open_px * self._crop_pixel_size_units(transform_info)
            if open_dist > 0:
                try:
                    r = combined.buffer(-open_dist, 8).buffer(open_dist, 8)
                    if r is not None and not r.isEmpty():
                        combined = r
                except Exception:  # noqa: BLE001 -- refine is best-effort  # nosec B110
                    pass
        tolerance = self._compute_simplification_tolerance(
            transform_info, self._refine_simplify)
        if tolerance > 0:
            r = combined.simplify(tolerance)
            if r is not None and not r.isEmpty():
                combined = r
        # The point budget: cut the traced outline down to the number of points
        # a person would have drawn. Manual has no prompt and so no shape class:
        # it gets the one generic spacing, read from the cached policy, so this
        # stays offline like the rest of Manual.
        #
        # It runs under Right angles too, and BEFORE it, exactly as the
        # Automatic review does. Squaring wants a de-staircased outline, and the
        # budget is the pass that produces one; skipping it here fed the
        # regularizer every traced pixel corner, which is what brings a wall
        # back as a staircase. It is also what makes the Points control mean
        # something with Right angles on.
        combined = self._apply_manual_vertex_budget(combined)
        if combined is None or combined.isEmpty():
            return None
        if ortho_on:
            combined = self._manual_apply_right_angles(
                combined, transform_info, tolerance)
        if not ortho_on and self._refine_smooth > 0:
            # Round corners plus its vertex diet, the SAME shared pass the
            # Automatic review runs: one Chaikin iteration, the simplify
            # tolerance as the minimum distance so no invisible vertex is
            # minted, 120 degrees so a near-straight wall corner is left alone,
            # then a simplify back to the same tolerance so the rounding does
            # not re-densify what the point budget just thinned.
            from ...core.polygon_exporter import rounded_corner_outline
            combined = rounded_corner_outline(combined, tolerance)
        return combined if combined is not None and not combined.isEmpty() else None

    def _apply_manual_vertex_budget(self, combined):
        """Thin one Manual outline to its point budget: the generic class
        density plus the user's Points dial. Returns the input unchanged on any
        failure, so a mask always yields a polygon."""
        if combined is None or combined.isEmpty():
            return combined
        try:
            from ...core.detection_policy import vertex_budget_settings
            from ...core.live_refine import points_dial_fraction
            from ...core.vertex_budget import (
                simplify_to_budget,
                smooth_budget_multiplier,
            )

            # The user's Points dial, through the SAME helper the Automatic
            # review uses, so both modes read one control the same way
            # (100% maps to 0.0, which is what turns the dial off).
            keep_fraction = points_dial_fraction(
                {"points_pct": self._refine_points_pct})
            s = vertex_budget_settings()
            spacing_m = float(s["spacing_m"])
            if spacing_m <= 0 and keep_fraction <= 0.0:
                return combined
            min_pts = int(s["min_vertices"])
            dev_m = float(s["max_deviation_m"])
            # How many rounding passes will actually follow. None under Right
            # angles (the squaring owns the outline), and never more than the
            # renderer runs: the budget pre-thins by a multiplier raised to this
            # power, so a stale larger count would shed most of the vertices to
            # pay for passes that never happen.
            smooth_iters = (
                0 if getattr(self, "_refine_ortho", False)
                else min(int(self._refine_smooth or 0), REFINE_SMOOTH_ITERATIONS))
            if smooth_iters > 0:
                # Round corners follows the budget and multiplies the points
                # every pass, so the budget thins ahead of it in step.
                spacing_m *= smooth_budget_multiplier(
                    float(s["smooth_spacing_factor"]), smooth_iters,
                    cap=float(s["smooth_multiplier_cap"]))
                min_pts = int(s["smooth_min_vertices"])
                dev_m = float(s["smooth_max_deviation_m"])
            centre = combined.boundingBox().center()
            factor = self._manual_metres_per_unit(centre.x(), centre.y())
            if factor <= 0:
                factor = 1.0
            r = simplify_to_budget(
                combined,
                spacing=spacing_m / factor,
                min_vertices=min_pts,
                max_deviation=dev_m / factor,
                max_deviation_fraction=float(s["max_deviation_fraction"]),
                dial_max_cap_fraction=float(s["dial_max_cap_fraction"]),
                keep_fraction=keep_fraction,
            )
            if r is not None and not r.isEmpty():
                return r
        except Exception:  # noqa: BLE001 -- refine is best-effort  # nosec B110
            pass
        return combined

    def _refined_active_mask_geometry(self):
        """The active SAM mask as ONE refined geometry: mask refinement (fill
        holes, expand, min region), polygonize, then the shared geometry tail
        (trim spikes, simplify, right angles, corner rounding) and the user
        Min/Max size window. The shared tail of the preview, save, export and
        freeze paths, so the polygon a user gets is always exactly the one
        previewed. None when no active mask or nothing survives refinement."""
        if self.current_mask is None or self.current_transform_info is None:
            return None
        from ...core.polygon_exporter import (
            apply_mask_refinement,
            mask_to_polygons,
        )
        mask = self.current_mask
        if self._refine_fill_holes or self._refine_expand != 0 or self._refine_min_area > 0:
            mask = apply_mask_refinement(
                self.current_mask,
                expand_value=self._refine_expand,
                fill_holes=self._refine_fill_holes,
                min_area=self._refine_min_area,
                max_hole_px=self._fill_holes_pixel_cap(),
            )
        # Mask-level simplify tolerance: a multiple of the mask pixel size, OFF
        # (0.0) unless the server tunes it, so today's polygonize is unchanged.
        from ...core.detection_policy import manual_simplify_multiple_of_px
        _mult = manual_simplify_multiple_of_px()
        _tol = (_mult * self._crop_pixel_size_units(self.current_transform_info)
                if _mult > 0 else 0.0)
        geometries = mask_to_polygons(mask, self.current_transform_info, _tol)
        if not geometries:
            return None
        combined = QgsGeometry.unaryUnion(geometries)
        combined = self._shape_active_geometry(combined, self.current_transform_info)
        if combined is None or combined.isEmpty():
            return None
        combined = self._filter_geometry_parts_by_size(combined)
        if combined is None or combined.isEmpty():
            return None
        return combined

    def _fill_holes_pixel_cap(self):
        """The Fill-holes size threshold in MASK PIXELS, or None to fill every
        hole (the control at 0, and every path where the ground size of a pixel
        cannot be measured).

        The user's number is true ground m2, like Min/Max size, so it crosses to
        pixels through the same area convention (layer_conventions.
        make_area_measurer): measure the crop's ground area, divide by its pixel
        count, and one mask pixel has a ground area whatever the layer CRS is.
        """
        max_m2 = float(getattr(self, "_refine_fill_holes_max_m2", 0.0) or 0.0)
        if max_m2 <= 0:
            return None
        info = self.current_transform_info
        if not info:
            return None
        try:
            from qgis.core import QgsRectangle

            from ...core.hole_size import hole_pixels
            from ...core.layer_conventions import make_area_measurer
            minx, maxx, miny, maxy = (float(v) for v in info["bbox"])
            rows, cols = int(info["img_shape"][0]), int(info["img_shape"][1])
            if rows <= 0 or cols <= 0:
                return None
            rect = QgsGeometry.fromRect(QgsRectangle(minx, miny, maxx, maxy))
            ground_m2 = 0.0
            if self._current_layer is not None and self._current_layer.crs().isValid():
                ground_m2 = float(
                    make_area_measurer(self._current_layer.crs()).measureArea(rect))
            if ground_m2 <= 0:
                ground_m2 = float(rect.area())
            if ground_m2 <= 0:
                return None
            return hole_pixels(max_m2, ground_m2 / (rows * cols))
        except (RuntimeError, AttributeError, KeyError, TypeError, ValueError):
            return None

    def _filter_geometry_parts_by_size(self, geom):
        """Drop polygon parts outside the user's Min/Max size window (true
        ground m2 so degree CRSs measure correctly; 0 = off). Returns the input
        unchanged when no filter applies, an empty geometry when nothing
        survives (the preview then shows exactly what a save would keep)."""
        # A hand-added object skips the size window, the same exemption the
        # review gives it (_object_is_manual): the window carries the run's
        # Min/Max size, so a missed object smaller than the floor would preview
        # as nothing and save as nothing, with no way for the user to tell why.
        # Keyed on "brand-new object inside a handoff" (no origin entry means
        # no detection was reopened), not on the Add flag, so every fold path
        # keeps the same shape even after the lane disarms. Base Manual is
        # untouched: there the window is a tool the user reaches for.
        if (getattr(self, "_refine_handoff_active", False) and not getattr(self, "_active_refine_origin_entry", None)):
            return geom
        min_a = float(getattr(self, "_refine_min_size_m2", 0.0) or 0.0)
        max_a = float(getattr(self, "_refine_max_size_m2", 0.0) or 0.0)
        if (min_a <= 0 and max_a <= 0) or geom is None or geom.isEmpty():
            return geom
        measurer = None
        try:
            from ...core.layer_conventions import make_area_measurer
            if self._current_layer is not None and self._current_layer.crs().isValid():
                measurer = make_area_measurer(self._current_layer.crs())
        except (RuntimeError, AttributeError):
            measurer = None
        parts = (geom.asGeometryCollection() if geom.isMultipart()
                 else [geom])
        kept = []
        dropped = False
        for part in parts:
            if part is None or part.isEmpty():
                dropped = True
                continue
            try:
                area = (float(measurer.measureArea(part)) if measurer is not None
                        else float(part.area()))
            except (RuntimeError, AttributeError):
                area = float(part.area())
            if (min_a > 0 and area < min_a) or (max_a > 0 and area > max_a):
                dropped = True
                continue
            kept.append(part)
        if not dropped:
            return geom
        if not kept:
            return QgsGeometry()
        if len(kept) == 1:
            return QgsGeometry(kept[0])
        return QgsGeometry.unaryUnion(kept)

    def _update_mask_visualization(self):
        if self.mask_rubber_band is None:
            # Recreate rubber band if it was lost (e.g. after RuntimeError)
            try:
                self.mask_rubber_band = QgsRubberBand(
                    self.iface.mapCanvas(),
                    PolygonGeometry
                )
                self.mask_rubber_band.setColor(PENDING_FILL)
                self.mask_rubber_band.setStrokeColor(PENDING_STROKE)
                self.mask_rubber_band.setWidth(2)
            except Exception:
                return
        # All editable selections are pending-blue; the refine/handoff active
        # object only differs by a bolder outline (see _apply_mask_band_style).
        self._apply_mask_band_style()

        if self.current_mask is None or self.current_transform_info is None:
            # No active mask - but may have frozen/unfrozen polygons to display
            if self._frozen_sessions or self._unfrozen_display_polygon is not None:
                self._display_frozen_composite_with_extra(
                    self._unfrozen_display_polygon)
            else:
                self._clear_mask_visualization()
            return

        try:
            from ...core.detection_policy import manual_simplify_multiple_of_px
            from ...core.polygon_exporter import (
                apply_mask_refinement,
                count_significant_regions,
                mask_to_polygons,
            )

            # Mask-level simplify tolerance: same OFF-by-default server dial as
            # the save/export path, so the preview equals what a save keeps.
            _mult = manual_simplify_multiple_of_px()
            _tol = (_mult * self._crop_pixel_size_units(self.current_transform_info)
                    if _mult > 0 else 0.0)

            # Everything the mask stage reads. Six of the ten refine controls
            # (Points, Simplify, Trim spikes, Round corners, Right angles,
            # Min/Max size) change NONE of it, yet a move on any of them used to
            # re-clean and re-polygonize the whole mask: tens of ms of scipy and
            # rasterio per settled slider tick, on the GUI thread.
            mask_key = (self._refine_expand, self._refine_fill_holes,
                        self._refine_min_area, self._fill_holes_pixel_cap(),
                        _tol)
            memo = getattr(self, "_mask_preview_memo", None)
            memo_hit = memo is not None
            memo_hit = memo_hit and memo[0] is self.current_mask
            memo_hit = memo_hit and memo[1] is self.current_transform_info
            memo_hit = memo_hit and memo[2] == mask_key
            if memo_hit:
                mask_to_display, geometries = memo[3], memo[4]
            else:
                # Apply refinement to preview in both modes (refine affects current mask only)
                mask_to_display = self.current_mask
                # Apply mask-level refinements (fill holes, expand/contract, min region)
                if self._refine_fill_holes or self._refine_expand != 0 or self._refine_min_area > 0:
                    mask_to_display = apply_mask_refinement(
                        self.current_mask,
                        expand_value=self._refine_expand,
                        fill_holes=self._refine_fill_holes,
                        min_area=self._refine_min_area,
                        max_hole_px=self._fill_holes_pixel_cap(),
                    )
                geometries = mask_to_polygons(
                    mask_to_display, self.current_transform_info, _tol)
                # Holding the mask keeps the `is` check above honest: a live
                # reference cannot have its id reused by a later array.
                self._mask_preview_memo = (
                    self.current_mask, mask_key, mask_to_display, geometries)

            # Detect disjoint regions and show message bar warning. The region
            # count dilates and labels the whole mask, so it is asked for only
            # when the one-shot warning can still fire; every repaint used to
            # pay it (refine sliders repaint on every drag step).
            may_warn = not self._disjoint_warning_shown and len(self._active_crop_points_positive) >= 2
            if may_warn and count_significant_regions(mask_to_display) > 1:
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("Disconnected parts detected. For best accuracy, segment one element at a time."),
                    level=Qgis.MessageLevel.Warning,
                    duration=6
                )
                self._disjoint_warning_shown = True

            # Build composite: frozen polygons + active mask polygons
            all_geoms = [s.polygon for s in self._frozen_sessions]

            if geometries:
                active_combined = QgsGeometry.unaryUnion(geometries)
                if active_combined and not active_combined.isEmpty():
                    # Trim spikes, simplify, right angles, round corners: the
                    # SAME geometry tail the save/export path runs, so the
                    # preview equals what a save keeps.
                    active_combined = self._shape_active_geometry(
                        active_combined, self.current_transform_info)
                    # User Min/Max size window (ground m2): the preview drops
                    # exactly the parts a save would drop.
                    if active_combined and not active_combined.isEmpty():
                        active_combined = self._filter_geometry_parts_by_size(
                            active_combined)
                    if active_combined and not active_combined.isEmpty():
                        all_geoms.append(active_combined)

            if all_geoms:
                combined = QgsGeometry.unaryUnion(all_geoms)
                if combined and not combined.isEmpty():
                    # Geometry is in raster CRS; transform to canvas CRS
                    self._transform_geometry_to_canvas_crs(combined)
                    self.mask_rubber_band.setToGeometry(combined, None)
                else:
                    self._clear_mask_visualization()
            else:
                self._clear_mask_visualization()

        except (ValueError, TypeError, RuntimeError) as e:
            QgsMessageLog.logMessage(
                f"Mask visualization error ({type(e).__name__}): {str(e)}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
            self._clear_mask_visualization()
        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Unexpected mask visualization error ({type(e).__name__}): {str(e)}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            self._clear_mask_visualization()

    def _clear_mask_visualization(self):
        if self.mask_rubber_band:
            try:
                self.mask_rubber_band.reset(PolygonGeometry)
            except RuntimeError:
                self.mask_rubber_band = None

    def _on_undo(self):
        """Undo last point added, or restore last saved mask in batch mode."""
        # Transport lock: while an off-thread encode owns the predictor pipe,
        # the session state is mid-transition (a deferred click may be waiting
        # to replay against the incoming crop). Rewinding points, mask history
        # or the delete stack here would corrupt that replay's context, so the
        # gesture is ignored exactly like save (the synchronous-encode era
        # never allowed it either: the GUI was blocked for the whole encode).
        if self._encoding_in_progress and not self._abandon_speculative_manual_crop():
            # The pipe is held by work somebody asked for. One exception: a
            # point dropped WHILE the crop is encoding is not committed yet, it
            # is stashed as a pending click (its marker already removed) to
            # replay when the encode finishes. Undo must cancel that pending
            # click, or Ctrl+Z right after the first point of a Refine-with-AI
            # session does nothing until the encode ends. Committed points and
            # mask history still stay locked mid-encode.
            #
            # A cursor-less warm-up nobody asked for is given up instead, so a
            # speculative read can never swallow Ctrl+Z.
            if getattr(self, "_pending_manual_click", None) is not None:
                self._discard_pending_manual_click()
            return
        self._manual_undos_session = getattr(self, "_manual_undos_session", 0) + 1
        # Refine edit session, geometry sub-state (open object, no editing
        # click yet): step back one Shape-settings reshape. This branch ABSORBS
        # undo entirely (even with nothing to undo yet): falling through would
        # restore unrelated deleted objects or pop the base-Manual re-edit
        # dialog mid-edit. Once editing clicks exist (current_mask set), undo
        # unwinds them through the normal point-history path below.
        if self._refine_edit_session_active() and self.current_mask is None:
            history = getattr(self, "_refine_geom_history", None)
            if history:
                self._unfrozen_display_polygon = history.pop()
                if self.map_tool:
                    self.map_tool.remove_last_marker()
                self._update_mask_visualization()
            return
        # Check if we have points in current mask
        current_point_count = self.prompts.point_count[0] + self.prompts.point_count[1]
        # With no point history to unwind, Ctrl+Z restores the most recent
        # Delete-key removal (stacked: repeated presses walk back deletions).
        should_restore_deleted = current_point_count == 0
        should_restore_deleted = should_restore_deleted and getattr(self, "_deleted_objects_stack", None)
        should_restore_deleted = should_restore_deleted and self._restore_deleted_object()
        if should_restore_deleted:
            return

        if current_point_count > 0:
            # Normal undo: remove last point from current mask
            result = self.prompts.undo()
            if result is None:
                return

            if self.map_tool:
                self.map_tool.remove_last_marker()

            # Restore the exact mask state from before this point was added,
            # including the SAM logits so the next click continues the same
            # refinement chain the user sees on screen. _current_crop_info is
            # kept so the next click reuses the encoding (no 3-8s re-encode).
            state = (self._mask_state_history.pop()
                     if self._mask_state_history else None)
            if state:
                self._restore_mask_state(state)
            else:
                self.current_low_res_mask = None

            # Also remove from per-crop point tracking
            if result[0] == "positive" and self._active_crop_points_positive:
                self._active_crop_points_positive.pop()
            elif result[0] == "negative" and self._active_crop_points_negative:
                self._active_crop_points_negative.pop()

            if self.prompts.point_count[0] + self.prompts.point_count[1] > 0:
                # Update visualization with restored mask (no re-prediction)
                self._update_ui_after_prediction()
            elif self._unfrozen_display_polygon is not None:
                # The snapshot restored a display-only polygon (undoing the
                # FIRST editing click of an open handoff detection, or the
                # click after an unfreeze): keep showing it, the object is
                # still open exactly as before that click.
                self._update_mask_visualization()
                self.dock_widget.set_point_count(0, 0)
            else:
                # Active crop is empty - check if we can unfreeze a previous crop
                if self._frozen_sessions:
                    self._unfreeze_last_session()
                else:
                    self.current_mask = None
                    self.current_score = 0.0
                    self._clear_mask_visualization()
                    self.dock_widget.set_point_count(0, 0)
        elif self._frozen_sessions:
            # No active points but have frozen sessions - unfreeze last one
            self._unfreeze_last_session()
        elif len(self.saved_polygons) > 0 and not self._refine_handoff_active:
            # Base Manual only: with no points, offer to re-open the LAST saved
            # mask. NEVER in a refine handoff - there saved_polygons holds the
            # whole imported review, so this dialog would grab an arbitrary
            # detection (the last one), which the next Delete then destroyed
            # ("deleting removes another polygon" on Mac, where the delete key
            # is Backspace and used to land here via undo).
            reply = QMessageBox.warning(
                self.iface.mainWindow(),
                tr("Edit saved polygon"),
                "{}\n{}".format(
                    tr("Warning: you are about to edit an already saved polygon."),
                    tr("Do you want to continue?")),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._restore_last_saved_mask()
            self._safe_restore_canvas_focus()

    def _unfreeze_last_session(self):
        """Unfreeze the last frozen crop session back to active display.

        The frozen polygon is displayed as the active mask. No re-encode is
        performed - SAM state is invalidated and will re-encode on next click.
        """
        if not self._frozen_sessions:
            return

        session = self._frozen_sessions.pop()

        # Clear active crop state
        self.current_mask = None
        self.current_score = 0.0
        self.current_low_res_mask = None
        self._current_crop_info = None  # Force re-encode on next click
        self._mask_state_history = []

        # Restore only the unfrozen session's points to prompts and markers.
        # Frozen sessions' points are NOT added to prompts - they are already
        # baked into frozen polygons and should not inflate the point count.
        self.prompts.clear()
        if self.map_tool:
            self.map_tool.clear_markers()

        self._active_crop_points_positive = list(session.points_positive)
        self._active_crop_points_negative = list(session.points_negative)
        for pt in session.points_positive:
            self.prompts.add_positive_point(pt[0], pt[1])
            if self.map_tool:
                canvas_pt = self._transform_to_canvas_crs(QgsPointXY(pt[0], pt[1]))
                if canvas_pt is not None:
                    self.map_tool.add_marker(canvas_pt, is_positive=True)
        for pt in session.points_negative:
            self.prompts.add_negative_point(pt[0], pt[1])
            if self.map_tool:
                canvas_pt = self._transform_to_canvas_crs(QgsPointXY(pt[0], pt[1]))
                if canvas_pt is not None:
                    self.map_tool.add_marker(canvas_pt, is_positive=False)

        # Display: frozen polygons + unfrozen session polygon as rubberband
        # The unfrozen polygon becomes a "display-only" active state
        # (no numpy mask - will re-encode on next click). Keep it around so
        # undo/save/export still see it until a prediction replaces it.
        self._unfrozen_display_polygon = session.polygon
        self._display_frozen_composite_with_extra(session.polygon)

        pos_count, neg_count = self.prompts.point_count
        self.dock_widget.set_point_count(pos_count, neg_count)

        QgsMessageLog.logMessage(
            f"Unfroze crop session, {len(self._frozen_sessions)} frozen remaining",
            "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _display_frozen_composite_with_extra(self, extra_polygon=None):
        """Display all frozen polygons (+ optional extra) as the rubberband."""
        if self.mask_rubber_band is None:
            return

        all_geoms = [s.polygon for s in self._frozen_sessions]
        if extra_polygon is not None:
            all_geoms.append(extra_polygon)

        if not all_geoms:
            self._clear_mask_visualization()
            return

        combined = QgsGeometry.unaryUnion(all_geoms)
        if combined and not combined.isEmpty():
            self._transform_geometry_to_canvas_crs(combined)
            self.mask_rubber_band.setToGeometry(combined, None)
        else:
            self._clear_mask_visualization()

    def _restore_last_saved_mask(self):
        """Restore the last saved mask for editing in batch mode."""
        if not self.dock_widget:
            return
        self._ensure_polygon_rubberband_sync()

        if not self.saved_polygons or not self.saved_rubber_bands:
            return

        # Pop the last saved polygon data
        last_polygon = self.saved_polygons.pop()

        # Remove the corresponding rubber band (green). In a handoff it is a None
        # placeholder; drop only the restored object's feature from its seed
        # layer (it becomes the active mask). No-op in base Manual.
        if self.saved_rubber_bands:
            last_rb = self.saved_rubber_bands.pop()
            self._safe_remove_rubber_band(last_rb)
        if not self._handoff_remove_entry_feature(last_polygon):
            self._rebuild_handoff_layers()

        # Clear current state first
        self.prompts.clear()
        self._mask_state_history = []
        self._frozen_sessions = []
        self._unfrozen_display_polygon = None
        self._active_crop_points_positive = []
        self._active_crop_points_negative = []
        if self.map_tool:
            self.map_tool.clear_markers()

        # Restore points
        points_positive = last_polygon.get("points_positive", [])
        points_negative = last_polygon.get("points_negative", [])

        # Rebuild prompts (stored in raster CRS) and markers (displayed in canvas CRS)
        for pt in points_positive:
            self.prompts.add_positive_point(pt[0], pt[1])
            if self.map_tool:
                canvas_pt = self._transform_to_canvas_crs(QgsPointXY(pt[0], pt[1]))
                if canvas_pt is not None:
                    self.map_tool.add_marker(canvas_pt, is_positive=True)

        for pt in points_negative:
            self.prompts.add_negative_point(pt[0], pt[1])
            if self.map_tool:
                canvas_pt = self._transform_to_canvas_crs(QgsPointXY(pt[0], pt[1]))
                if canvas_pt is not None:
                    self.map_tool.add_marker(canvas_pt, is_positive=False)

        # Restore mask data
        self.current_mask = last_polygon.get("raw_mask")
        self.current_score = last_polygon.get("score", 0.0)
        self.current_transform_info = last_polygon.get("transform_info")
        if self.current_mask is None or self.current_transform_info is None:
            # Saved without an active SAM mask (an unfrozen session shape, a
            # handoff entry): only its geometry survives. Restore THAT as the
            # display-only active shape, or re-opening the polygon would wipe it
            # off the canvas and out of the save/export set.
            geom = last_polygon.get("geom_obj")
            if geom is None:
                geom = QgsGeometry.fromWkt(last_polygon.get("geometry_wkt") or "")
            if geom is not None and not geom.isEmpty():
                self._unfrozen_display_polygon = QgsGeometry(geom)

        # Restore refine settings (fallbacks shared with __init__/_reset_session
        # and the dock via core/review_defaults.py)
        self._refine_simplify = float(
            last_polygon.get("refine_simplify", REFINE_SIMPLIFY_DEFAULT) or 0.0)
        self._refine_points_pct = int(
            last_polygon.get("refine_points_pct") or REFINE_POINTS_PCT_DEFAULT)
        self._refine_smooth = last_polygon.get("refine_smooth", REFINE_SMOOTH_DEFAULT)
        self._refine_clean = float(
            last_polygon.get("refine_clean") or REFINE_CLEAN_DEFAULT)
        self._refine_expand = last_polygon.get("refine_expand", REFINE_EXPAND_DEFAULT)
        self._refine_fill_holes = last_polygon.get("refine_fill_holes", REFINE_FILL_HOLES_DEFAULT)
        self._refine_fill_holes_max_m2 = float(
            last_polygon.get("refine_fill_holes_max_m2") or REFINE_FILL_HOLES_MAX_M2_DEFAULT)
        self._refine_ortho = last_polygon.get("refine_ortho", REFINE_ORTHO_DEFAULT)
        self._refine_min_area = last_polygon.get(
            "refine_min_area", REFINE_MIN_AREA_DEFAULT)
        self._refine_min_size_m2 = float(last_polygon.get("refine_min_size_m2") or REFINE_MIN_SIZE_M2_DEFAULT)
        self._refine_max_size_m2 = float(last_polygon.get("refine_max_size_m2") or REFINE_MAX_SIZE_M2_DEFAULT)

        # Update UI sliders without emitting signals
        self.dock_widget.set_refine_values(
            self._refine_simplify,
            self._refine_smooth,
            self._refine_expand,
            self._refine_fill_holes,
            self._refine_min_area,
            right_angles=self._refine_ortho,
            fill_holes_max_m2=self._refine_fill_holes_max_m2,
            clean=self._refine_clean,
            points_pct=self._refine_points_pct,
        )
        self.dock_widget.set_size_filter_values(
            self._refine_min_size_m2, self._refine_max_size_m2)

        # Update visualization
        self._update_mask_visualization()

        # Update UI counters
        pos_count, neg_count = self.prompts.point_count
        self.dock_widget.set_point_count(pos_count, neg_count)
        self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))

        QgsMessageLog.logMessage(
            f"Restored mask with {pos_count} positive, {neg_count} negative points. "
            f"Refine: simplify={self._refine_simplify}, "
            f"points_pct={self._refine_points_pct}, smooth={self._refine_smooth}, "
            f"expand={self._refine_expand}, fill_holes={self._refine_fill_holes}, "
            f"min_area={self._refine_min_area}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

    def _reset_session(self):
        # Invalidate any in-flight async crop encode FIRST: this session's state
        # is about to be wiped, so a still-running SetImageWorker's completion
        # must be dropped (generation bump) and its remembered click discarded so
        # nothing replays into a torn-down session. The worker itself keeps
        # running set_image on its own predictor ref and drains harmlessly; its
        # completion restores the busy cursor. (PERF-01)
        self._invalidate_manual_encode()
        # Emit the manual session summary for the session that is ending (>=1
        # save; skip mid-handoff so a Refine-in-Manual import does not spawn a
        # spurious summary), then reset the per-session counters.
        try:
            saves = getattr(self, "_manual_saves_session", 0)
            if saves >= 1 and not self._refine_handoff_active:
                import time as _time

                from ...core import telemetry_session_events
                t0 = getattr(self, "_manual_session_t0", None)
                telemetry_session_events.track_manual_session_summary(
                    saves=saves,
                    undos=getattr(self, "_manual_undos_session", 0),
                    duration_ms=int((_time.time() - t0) * 1000) if t0 else None,
                )
        except Exception:
            pass  # nosec B110
        self._manual_saves_session = 0
        self._manual_undos_session = 0
        self._manual_session_t0 = None
        # Active-object edit state does not survive a session reset.
        self._is_refining_saved_object = False
        self._active_refine_origin_entry = None
        self._refine_geom_history = []
        self._deleted_objects_stack = []
        # Selection-first review state dies with the session (bands removed).
        self._handoff_selected_entries = []
        self._handoff_hover_entry = None
        self._handoff_hit_index = None
        self._handoff_tok2entry = {}
        self._handoff_det_id_seq = None
        # A leaked import set would let the NEXT session's fold delete objects
        # it never showed, so it dies with the session that recorded it.
        self._handoff_imported_det_ids = set()
        for attr in ("_handoff_selection_band", "_handoff_hover_band"):
            band = getattr(self, attr, None)
            if band is not None:
                self._safe_remove_rubber_band(band)
                setattr(self, attr, None)
        if self.dock_widget:
            try:
                self.dock_widget.set_handoff_selected(0)
                self.dock_widget.set_handoff_editing(False)
            except (RuntimeError, AttributeError):
                pass
        # Clear the telemetry start timestamp so the next successful run does
        # not attribute duration to an abandoned previous run.
        self._segmentation_start_ts = None
        self.prompts.clear()
        self._mask_state_history = []
        self._frozen_sessions = []
        self._unfrozen_display_polygon = None
        self._active_crop_points_positive = []
        self._active_crop_points_negative = []
        self._disjoint_warning_shown = False
        self.saved_polygons = []

        for rb in self.saved_rubber_bands:
            self._safe_remove_rubber_band(rb)
        self.saved_rubber_bands = []
        # The handoff seed layers die with the session. No-op
        # outside a handoff (refs are None).
        self._remove_handoff_layers()

        if self.map_tool:
            self.map_tool.clear_markers()

        self._clear_mask_visualization()

        self.current_mask = None
        self.current_score = 0.0
        self.current_transform_info = None
        self.current_low_res_mask = None
        # The last click coordinate feeds Progressive Merge; a fresh session
        # starts with none so the first click of a new object is never bounded.
        self._last_click_point = None
        self._last_click_polarity = "positive"
        self._last_prediction_empty = False
        self._last_click_stood_clear = False
        self._last_click_took_from_another = False

        # Reset on-demand encoding state. The layer reference goes too:
        # keeping it while the raster path is cleared leaves a half-dead
        # session that later flows trust and then fail on (crop_error_no_path).
        self._current_layer = None
        self._current_layer_name = ""
        self._current_crop_info = None
        self._current_raster_path = None
        self._current_crop_canvas_mupp = None
        self._current_crop_actual_mupp = None
        self._current_crop_scale_factor = None

        # Reset online layer state
        self._is_online_layer = False

        # Reset refinement settings to defaults (#12, #23)
        self._refine_simplify = float(REFINE_SIMPLIFY_DEFAULT)
        self._refine_points_pct = REFINE_POINTS_PCT_DEFAULT
        self._refine_smooth = REFINE_SMOOTH_DEFAULT
        self._refine_clean = REFINE_CLEAN_DEFAULT
        self._refine_expand = REFINE_EXPAND_DEFAULT
        self._refine_fill_holes = REFINE_FILL_HOLES_DEFAULT
        self._refine_fill_holes_max_m2 = REFINE_FILL_HOLES_MAX_M2_DEFAULT
        self._refine_ortho = REFINE_ORTHO_DEFAULT
        # overridden by _compute_auto_min_area() x 2
        self._refine_min_area = REFINE_MIN_AREA_DEFAULT
        self._refine_min_size_m2 = REFINE_MIN_SIZE_M2_DEFAULT
        self._refine_max_size_m2 = REFINE_MAX_SIZE_M2_DEFAULT

        if self.dock_widget:
            self.dock_widget.set_point_count(0, 0)
            self.dock_widget.set_saved_polygon_count(0)
