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
from qgis.PyQt.QtWidgets import (
    QMessageBox,
)

from ...core.i18n import tr
from ...core.qt_compat import PolygonGeometry
from ...core.review_defaults import (
    REFINE_EXPAND_DEFAULT,
    REFINE_FILL_HOLES_DEFAULT,
    REFINE_MAX_SIZE_M2_DEFAULT,
    REFINE_MIN_SIZE_M2_DEFAULT,
    REFINE_ORTHO_DEFAULT,
    REFINE_SIMPLIFY_DEFAULT,
    REFINE_SMOOTH_DEFAULT,
)
from ..canvas_palette import (
    PENDING_FILL,
    PENDING_STROKE,
)
from ..error_report_dialog import show_error_report


class ManualPredictMixin:
    """Manual clicks, SAM prediction, mask visualization, undo and session reset."""

    def _on_positive_click(self, point):
        """Handle left-click: add positive point (select this element)."""
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

        if not self._run_prediction():
            self._rollback_failed_click("positive")
            return

        # Auto-revert if prediction produced an empty mask (no element detected)
        if self.current_mask is not None and self.current_mask.sum() == 0 and self._mask_state_history:
            self.prompts.undo()
            if self._active_crop_points_positive:
                self._active_crop_points_positive.pop()
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

        # Live complete-don't-stack: if this selection now overlaps an existing
        # detection, weld them into one shape on the canvas right away (refine
        # handoff only; no-op otherwise).
        self._weld_active_into_overlaps()

    def _on_negative_click(self, point):
        """Handle right-click: add negative point (exclude this area)."""
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

        if not self._run_prediction():
            self._rollback_failed_click("negative")
            return

        # Auto-revert if prediction produced an empty mask (no element detected)
        if self.current_mask is not None and self.current_mask.sum() == 0 and self._mask_state_history:
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
        elif (self._is_refining_saved_object and self._unfrozen_display_polygon is not None):
            # First editing click on an open handoff detection: seed SAM with
            # the object's polygon rasterized onto the current crop (the same
            # context transfer a zoom re-encode uses), so the click REFINES
            # the whole shape exactly like a base-Manual click instead of
            # segmenting an unrelated element under the cursor.
            mask_input = self._refine_polygon_mask_input()

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
            is_dll_error = "DLL" in error_str or "Visual C++" in error_str
            try:
                from ...core import telemetry
                if is_dll_error:
                    code = "predict_dll_error"
                elif "subprocess" in error_str.lower() or "rpc" in error_str.lower():
                    code = "predict_worker_died"
                else:
                    code = "predict_runtime_error"
                # A non-headless DLL error reports through show_error_report
                # below, which fires its own telemetry with the same code;
                # skip here so the same occurrence is not counted twice.
                if not (is_dll_error and not self._headless):
                    telemetry.track_plugin_error(
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
                # error) used to fail silently: the click did nothing with no
                # explanation. Surface it so the user knows to retry.
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("Segmentation failed. Please try again."),
                    level=Qgis.MessageLevel.Warning,
                    duration=5,
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
                from ...core import telemetry
                telemetry.track_plugin_error(
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
        # ghost polygons outside the raster.
        self.current_mask = self.current_mask[:img_height, :img_width]

        # A real prediction replaces any display-only unfrozen polygon
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
        active-state broke the blue -> green story)."""
        if self.mask_rubber_band is None:
            return
        self.mask_rubber_band.setColor(PENDING_FILL)
        self.mask_rubber_band.setStrokeColor(PENDING_STROKE)
        # A bolder outline while an object is open for editing in a
        # refine/handoff, so it stands apart from the flat pending seeds
        # without introducing a non-blue colour.
        editing = self._refine_handoff_active or self._is_refining_saved_object
        self.mask_rubber_band.setWidth(3 if editing else 2)

    def _refined_active_mask_geometry(self):
        """The active SAM mask as ONE refined geometry: mask refinement (fill
        holes, expand, min region), polygonize, simplify, right angles, corner
        rounding, then the user Min/Max size window. The shared tail of the
        preview, save, export and freeze paths, so the polygon a user gets is
        always exactly the one previewed. None when no active mask or nothing
        survives refinement."""
        if self.current_mask is None or self.current_transform_info is None:
            return None
        from ...core.polygon_exporter import (
            apply_mask_refinement,
            apply_right_angles,
            mask_to_polygons,
        )
        mask = self.current_mask
        if self._refine_fill_holes or self._refine_expand != 0 or self._refine_min_area > 0:
            mask = apply_mask_refinement(
                self.current_mask,
                expand_value=self._refine_expand,
                fill_holes=self._refine_fill_holes,
                min_area=self._refine_min_area,
            )
        geometries = mask_to_polygons(mask, self.current_transform_info)
        if not geometries:
            return None
        combined = QgsGeometry.unaryUnion(geometries)
        if combined is None or combined.isEmpty():
            return None
        tolerance = self._compute_simplification_tolerance(
            self.current_transform_info, self._refine_simplify)
        if tolerance > 0:
            combined = combined.simplify(tolerance)
        if self._refine_ortho:
            combined = apply_right_angles(
                combined,
                destair_tol=max(
                    0.0,
                    self._compute_simplification_tolerance(
                        self.current_transform_info, 3) - tolerance))
        if self._refine_smooth > 0:
            combined = combined.smooth(self._refine_smooth, 0.25)
        if combined is None or combined.isEmpty():
            return None
        combined = self._filter_geometry_parts_by_size(combined)
        if combined is None or combined.isEmpty():
            return None
        return combined

    def _filter_geometry_parts_by_size(self, geom):
        """Drop polygon parts outside the user's Min/Max size window (true
        ground m2 so degree CRSs measure correctly; 0 = off). Returns the input
        unchanged when no filter applies, an empty geometry when nothing
        survives (the preview then shows exactly what a save would keep)."""
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
            from ...core.polygon_exporter import (
                apply_mask_refinement,
                apply_right_angles,
                count_significant_regions,
                mask_to_polygons,
            )

            # Apply refinement to preview in both modes (refine affects current mask only)
            mask_to_display = self.current_mask
            # Apply mask-level refinements (fill holes, expand/contract, min region)
            if self._refine_fill_holes or self._refine_expand != 0 or self._refine_min_area > 0:
                mask_to_display = apply_mask_refinement(
                    self.current_mask,
                    expand_value=self._refine_expand,
                    fill_holes=self._refine_fill_holes,
                    min_area=self._refine_min_area,
                )

            # Detect disjoint regions and show message bar warning
            region_count = count_significant_regions(mask_to_display)
            is_disjoint = region_count > 1
            has_multiple_positive = len(self._active_crop_points_positive) >= 2
            if is_disjoint and not self._disjoint_warning_shown and has_multiple_positive:
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("Disconnected parts detected. For best accuracy, segment one element at a time."),
                    level=Qgis.MessageLevel.Warning,
                    duration=6
                )
                self._disjoint_warning_shown = True

            geometries = mask_to_polygons(mask_to_display, self.current_transform_info)

            # Build composite: frozen polygons + active mask polygons
            all_geoms = [s.polygon for s in self._frozen_sessions]

            if geometries:
                active_combined = QgsGeometry.unaryUnion(geometries)
                if active_combined and not active_combined.isEmpty():
                    # Apply simplification to active mask preview
                    tolerance = self._compute_simplification_tolerance(
                        self.current_transform_info, self._refine_simplify
                    )
                    if tolerance > 0:
                        active_combined = active_combined.simplify(tolerance)
                    # Right angles: orthogonalize man-made shapes (needs a
                    # de-staircased outline; top up when simplify is weak).
                    if self._refine_ortho:
                        active_combined = apply_right_angles(
                            active_combined,
                            destair_tol=max(
                                0.0,
                                self._compute_simplification_tolerance(
                                    self.current_transform_info, 3) - tolerance))
                    # Apply corner rounding (QGIS native C++ Chaikin)
                    if self._refine_smooth > 0:
                        active_combined = active_combined.smooth(
                            self._refine_smooth, 0.25)
                    # User Min/Max size window (ground m2): the preview drops
                    # exactly the parts a save would drop.
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
        if self._encoding_in_progress:
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
                self.map_tool.add_marker(canvas_pt, is_positive=True)
        for pt in session.points_negative:
            self.prompts.add_negative_point(pt[0], pt[1])
            if self.map_tool:
                canvas_pt = self._transform_to_canvas_crs(QgsPointXY(pt[0], pt[1]))
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
                self.map_tool.add_marker(canvas_pt, is_positive=True)

        for pt in points_negative:
            self.prompts.add_negative_point(pt[0], pt[1])
            if self.map_tool:
                canvas_pt = self._transform_to_canvas_crs(QgsPointXY(pt[0], pt[1]))
                self.map_tool.add_marker(canvas_pt, is_positive=False)

        # Restore mask data
        self.current_mask = last_polygon.get("raw_mask")
        self.current_score = last_polygon.get("score", 0.0)
        self.current_transform_info = last_polygon.get("transform_info")

        # Restore refine settings (fallbacks shared with __init__/_reset_session
        # and the dock via core/review_defaults.py)
        self._refine_simplify = last_polygon.get("refine_simplify", REFINE_SIMPLIFY_DEFAULT)
        self._refine_smooth = last_polygon.get("refine_smooth", REFINE_SMOOTH_DEFAULT)
        self._refine_expand = last_polygon.get("refine_expand", REFINE_EXPAND_DEFAULT)
        self._refine_fill_holes = last_polygon.get("refine_fill_holes", REFINE_FILL_HOLES_DEFAULT)
        self._refine_ortho = last_polygon.get("refine_ortho", REFINE_ORTHO_DEFAULT)
        self._refine_min_area = last_polygon.get("refine_min_area", 200)
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
            f"Refine: simplify={self._refine_simplify}, smooth={self._refine_smooth}, "
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
                from ...core import telemetry
                t0 = getattr(self, "_manual_session_t0", None)
                telemetry.track_manual_session_summary(
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
        self._refine_simplify = REFINE_SIMPLIFY_DEFAULT
        self._refine_smooth = REFINE_SMOOTH_DEFAULT
        self._refine_expand = REFINE_EXPAND_DEFAULT
        self._refine_fill_holes = REFINE_FILL_HOLES_DEFAULT
        self._refine_ortho = REFINE_ORTHO_DEFAULT
        self._refine_min_area = 200  # overridden by _compute_auto_min_area() × 2
        self._refine_min_size_m2 = REFINE_MIN_SIZE_M2_DEFAULT
        self._refine_max_size_m2 = REFINE_MAX_SIZE_M2_DEFAULT

        if self.dock_widget:
            self.dock_widget.set_point_count(0, 0)
            self.dock_widget.set_saved_polygon_count(0)
