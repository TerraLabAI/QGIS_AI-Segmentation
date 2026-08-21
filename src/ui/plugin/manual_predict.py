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
    QApplication,
    QMessageBox,
)

from ...core.i18n import tr
from ...core.prompt_manager import FrozenCropSession
from ...core.qt_compat import DashLine, PolygonGeometry, SolidLine, WaitCursor
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
from ...core.telemetry_errors import slot_guard
from ..canvas_palette import (
    PENDING_FILL,
    PENDING_STROKE,
)
from ..error_report_dialog import show_error_report

# The ground size of a crop pixel could not be measured, so the Fill holes
# cutoff is unknown. Its own value because None already means "no cutoff, fill
# every hole": read as that, a failed measure swallows every courtyard of an
# object whose owner asked for a bounded fill.
FILL_HOLES_CAP_UNKNOWN = object()

# Why a click unwound with nothing to report. Neither is a failure, and neither
# gives the user anything to act on, so neither opens a message.
QUIET_CLICK_SUPERSEDED = "superseded"   # its crop moved on while it was out
QUIET_CLICK_REREAD = "reread"           # its imagery is being read again for it
QUIET_CLICK_REFUSED = "refused"         # the far side said no, and said why


def _click_was_superseded(err: Exception) -> bool:
    """True when the crop moved on while this click's answer was travelling.

    Not a failure to report: the picture the click was asked about is gone, so
    there is nothing to draw and nothing the user can do. The class is imported
    at the moment of the failure because the module holding it carries the
    heavy numeric imports and plugin start must not pay for them.
    """
    try:
        from ...core.cloud_sam_predictor import RefineSupersededError

        return isinstance(err, RefineSupersededError)
    except Exception:  # noqa: BLE001 -- an unimportable class is not a match
        return False


# How many clicks back the per-point undo can reach. Each entry holds a copy
# of the mask, so the depth is a memory ceiling as much as a UI one. The point
# lists themselves are not capped: dropping the oldest point would change what
# the next prediction is given, and the shape on screen with it.
MASK_UNDO_DEPTH = 30


def _click_refusal_answer(err: Exception) -> str:
    """The answer a refused click deserves, or "" when it was not a refusal.

    "CREDITS" and "SIGN_IN" are the user's own to settle, so they never earn a
    bug report. Everything else keeps the report dialog. Imported at the moment
    of the failure for the same reason as the class above.
    """
    try:
        from ...core.cloud_sam_predictor import REFUSAL_OTHER, RefineRefusedError

        if not isinstance(err, RefineRefusedError):
            return ""
        answer = err.refusal_class()
        return "" if answer == REFUSAL_OTHER else str(answer)
    except Exception:  # noqa: BLE001 -- an unimportable class is not a refusal
        return ""


class ManualPredictMixin:
    """Manual clicks, SAM prediction, mask visualization, undo and session reset."""

    def _report_click_without_model(self) -> None:
        """Say why a click did nothing, once per session.

        A click with no model in the slot took the marker back off the map and
        returned, so the map answered a click by erasing it and the user had
        no idea whether the click, the layer or the plugin was at fault.
        """
        if getattr(self, "_click_without_model_reported", False):
            return
        if getattr(self, "_headless", False):
            return
        self._click_without_model_reported = True
        try:
            if getattr(self, "_local_ai_load_failed", False):
                line = tr("The AI did not load, so this click was not "
                          "answered. Use the Install button in the panel to "
                          "set it up again.")
            else:
                line = tr("The AI is still loading, so this click was not "
                          "answered. Try again in a few seconds.")
            self.iface.messageBar().pushMessage(
                "AI Segmentation", line,
                level=Qgis.MessageLevel.Warning, duration=6)
        except (RuntimeError, AttributeError):
            pass

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

    @slot_guard(stage="segment")
    def _on_positive_click(self, point):
        """Handle left-click: add positive point (select this element).

        Guarded because this runs on the map tool's own stack frame, straight
        out of a reimplemented Qt virtual, where an escaping exception is the
        abort path in PyQt. What the click knows how to report, it reports
        below; the guard is for everything else.
        """
        # The click pipeline takes over from here, so the shape that was
        # following the cursor comes off the map before anything else runs.
        # Its answer is taken first, because clearing the ghost drops it, and
        # a click landing on the ghost asks the service what the ghost already
        # asked. Read and cleared in _run_prediction; a click that never gets
        # there leaves it for the next one to overwrite on this line.
        self._hover_click_answer = self._take_hover_preview_answer()
        self._stop_hover_preview("click")
        if self._refine_click_is_stale():
            self._drop_stale_refine_click()
            return
        if self.predictor is None:
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self._report_click_without_model()
            return

        # Transform click from canvas CRS to raster CRS for all downstream use
        raster_pt = self._transform_to_raster_crs(point)

        if not self._is_point_in_raster_extent(raster_pt):
            if self.map_tool:
                self.map_tool.remove_last_marker()
            # The session's own raster, never the combo: the combo can have
            # moved on to another layer, and naming that one told the user the
            # click missed a raster they were not working in.
            layer_name = ""
            try:
                if self._current_layer is not None:
                    layer_name = self._current_layer.name()
            except RuntimeError:
                layer_name = ""
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
        #
        # Unless the pipe is held by a crop NOBODY asked for. A warm-up reads a
        # neighbourhood the user may never click, and making a real click wait
        # it out is the one thing a warm-up must never cost. The abandon
        # re-checks everything and refuses when the crop was asked for by name
        # or already has a click waiting on it, so this can only ever drop
        # speculative work.
        if self._encoding_in_progress and not self._abandon_speculative_manual_crop():
            self._remember_pending_manual_click("positive", point)
            self._wear_busy_cursor_for_crop()
            return

        # An edit in progress owns every click. A click that lands on another
        # detection used to save and jump to it, which took the session away
        # mid-gesture and made the ground under a neighbour unreachable: the
        # user could never grow the open object over it. One object is edited at
        # a time, and the way to another one is to finish this one first.

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
        if len(self._mask_state_history) >= MASK_UNDO_DEPTH:
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
        waiting = self._begin_click_wait()
        try:
            predicted = self._run_prediction()
        except Exception:
            # The point and its marker are on screen already. An error on its
            # way to the slot guard would leave them there, and every later
            # predict would carry a point that produced nothing.
            self._rollback_failed_click("positive", point)
            raise
        finally:
            if waiting:
                self._end_click_wait()
        if not predicted:
            self._rollback_failed_click("positive", point)
            return

        # Auto-revert when THIS CLICK added nothing: it found nothing, or what it
        # found stood clear of the object being edited. Not "the shape is empty":
        # a keep click may only grow the shape, so once a shape exists the total
        # is never empty and neither message would ever come back.
        undo_note = None
        if self._last_prediction_found_nothing():
            undo_note = tr("No object found here. Try clicking somewhere else.")
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

    @slot_guard(stage="segment")
    def _on_negative_click(self, point):
        """Handle right-click: add negative point (exclude this area).

        Guarded like the keep click above, and for the same reason: it runs on
        the map tool's C++ stack frame, where an escaping exception aborts.
        """
        self._stop_hover_preview("click")
        if self._refine_click_is_stale():
            self._drop_stale_refine_click()
            return
        if self.predictor is None:
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self._report_click_without_model()
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
        # remembered, never routed into a second encode. A crop nobody asked
        # for is dropped instead of waited out, same rule as the left click.
        if self._encoding_in_progress and not self._abandon_speculative_manual_crop():
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
            dock = self.dock_widget
            sel = dock.layer_combo.currentLayer() if dock is not None else None
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
        if len(self._mask_state_history) >= MASK_UNDO_DEPTH:
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
        waiting = self._begin_click_wait()
        try:
            predicted = self._run_prediction()
        except Exception:
            # Same reason as the keep click: the point is committed before the
            # predict, so an error must take it back down on its way out.
            self._rollback_failed_click("negative", point)
            raise
        finally:
            if waiting:
                self._end_click_wait()
        if not predicted:
            self._rollback_failed_click("negative", point)
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
                tr("No object found here. Try clicking somewhere else."),
                level=Qgis.MessageLevel.Info,
                duration=4
            )
            return

    def _run_prediction(self) -> bool:
        """Run SAM prediction using active crop points only.

        When frozen sessions exist, only the active crop's points are sent
        to SAM (frozen polygons are composited during visualization).

        Returns True when a prediction was stored, False otherwise, so the
        caller can roll the click back. False is not always a failure: a click
        whose crop moved on, and one whose imagery has to be read again, both
        say so with `_end_click_quietly` and the rollback reads it there.
        """
        import numpy as np

        # Two lines of affine arithmetic, from a library that lives in the
        # on-device environment. The Automatic review's remote fix route runs on
        # machines that have none, so a plain-arithmetic stand-in takes over when
        # the import is missing. Where the library is there, it is still used, so
        # Manual's answers do not move by a pixel.
        try:
            from rasterio import transform as rio_transform
            from rasterio.transform import from_bounds as transform_from_bounds
        except ImportError:
            rio_transform = None
            transform_from_bounds = None

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
        # forever. Read the same crop again, transparently. A rare recovery
        # path, reached only when the crop was expected "ok", so no async worker
        # owns the pipe.
        #
        # On a file raster the read is local and the blocking form keeps predict
        # fully synchronous, at the cost of a brief freeze here. An online layer
        # reads from the tile network, with retries, and that is the whole
        # interactive path's reason for existing: doing it here froze QGIS for
        # as long as the tiles took. So the click hands itself to the same
        # deferred replay a click on a cold crop uses (see _rollback_failed_click).
        if not self.predictor.is_image_set:
            QgsMessageLog.logMessage(
                "Worker has no encoded image - re-encoding current crop",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )
            b = self._current_crop_info["bounds"]
            center = QgsPointXY((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
            override = (self._current_crop_actual_mupp if self._is_online_layer
                        else self._current_crop_scale_factor)
            if self._is_online_layer and not self._headless:
                if self._extract_and_encode_crop(center, mupp_override=override):
                    self._end_click_quietly(QUIET_CLICK_REREAD)
                return False
            if not self._encode_crop_blocking(center, mupp_override=override):
                return False

        crop_bounds = self._current_crop_info["bounds"]
        img_shape = self._current_crop_info["img_shape"]
        img_height, img_width = img_shape

        minx, miny, maxx, maxy = crop_bounds
        # Both branches below are built on this window, and the library one
        # raises on a window with no size instead of answering None. No window,
        # no click.
        if maxx <= minx or maxy <= miny or img_width <= 0 or img_height <= 0:
            QgsMessageLog.logMessage(
                "Crop window has no size - cannot place the click in it",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return False
        if rio_transform is not None:
            img_clip_transform = transform_from_bounds(
                minx, miny, maxx, maxy, img_width, img_height)

            def crop_pixel_of(px, py):
                return rio_transform.rowcol(img_clip_transform, px, py)
        else:
            from ...core.crop_window import crop_pixel_of_point

            def crop_pixel_of(px, py):
                return crop_pixel_of_point(crop_bounds, img_shape, px, py)

        # Build point arrays from active crop points only. A window with no size
        # cannot place a point in it, and the answer for that is no answer: the
        # top-left corner is a valid pixel address, so the model would be
        # prompted there and the user would get a mask nowhere near their click.
        point_coords_list = []
        point_labels_list = []
        for points, label in ((active_pos, 1), (active_neg, 0)):
            for x, y in points:
                pixel = crop_pixel_of(x, y)
                if pixel is None:
                    QgsMessageLog.logMessage(
                        "Crop window has no size - cannot place the click in it",
                        "AI Segmentation", level=Qgis.MessageLevel.Warning)
                    return False
                row, col = pixel
                point_coords_list.append([col, row])
                point_labels_list.append(label)

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

        # The ghost the user clicked on asked this crop this question already,
        # so its answer stands in for the round trip instead of paying it a
        # second time. Taken whatever happens, so a held answer can never reach
        # a later click.
        held_hover = getattr(self, "_hover_click_answer", None)
        self._hover_click_answer = None
        reused = (self._reused_hover_answer(held_hover, crop_bounds, img_shape,
                                            point_coords_list)
                  if held_hover is not None and use_multimask else None)

        # Timed from here: this is the wait the user actually sits through, on
        # whichever route answers them.
        import time as _click_clock
        self._manual_click_fell_back = False
        click_started_at = _click_clock.monotonic()

        try:
            if reused is not None:
                masks, scores, low_res_masks = reused
                # The network answered this, as a preview. The ledger hangs the
                # object's charge on a network answer, so it is noted here
                # exactly as the predictor notes one of its own.
                self._note_manual_cloud_answer()
            else:
                masks, scores, low_res_masks = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    mask_input=mask_input,
                    multimask_output=use_multimask,
                )
        except RuntimeError as e:
            if _click_was_superseded(e):
                # The session moved to another crop while this click was out.
                # Nothing failed and nothing is the user's to fix, so the click
                # ends here without a report and without changing the route.
                QgsMessageLog.logMessage(
                    "Click dropped: the crop changed while its answer was on "
                    "the way", "AI Segmentation", level=Qgis.MessageLevel.Info)
                self._end_click_quietly(QUIET_CLICK_SUPERSEDED)
                return False
            error_str = str(e)
            refusal = _click_refusal_answer(e)
            if refusal and not self._headless:
                # An empty balance or a session that signed out. The far side
                # already says what to do about it, and neither is a fault to
                # report, so the sentence goes to the message bar instead of a
                # dialog offering to mail us about the user's own account.
                QgsMessageLog.logMessage(
                    f"Click refused ({refusal})", "AI Segmentation",
                    level=Qgis.MessageLevel.Warning)
                try:
                    from ...core import telemetry_errors
                    telemetry_errors.track_plugin_error(
                        stage="segment", error_code="predict_refused",
                        message=error_str)
                except Exception:
                    pass  # nosec B110
                try:
                    if refusal == "SIGN_IN":
                        line = tr("Session expired. Sign in again to continue.")
                    else:
                        line = tr("You saved your cloud objects for this "
                                  "month. Switch to your own computer to keep "
                                  "working free, or upgrade from the panel.")
                    # The wire's own sentence stays in the log: it is English,
                    # it can name internals, and the user cannot act on it.
                    self.iface.messageBar().pushWarning(
                        "AI Segmentation", line)
                except (RuntimeError, AttributeError):
                    pass
                self._end_click_quietly(QUIET_CLICK_REFUSED)
                return False
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
            # The review's AI fix answered off the machine and the answer never
            # came. Nothing here is the user's to fix, so the step falls back to
            # editing by hand instead of opening a report dialog.
            if self._degrade_correct_ai_to_manual(error_str):
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
            if not self._headless and not self._degrade_correct_ai_to_manual(str(e)):
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("Segmentation failed. Please try again."),
                    level=Qgis.MessageLevel.Warning,
                    duration=5,
                )
            return False

        self._track_manual_click_answered(click_started_at)

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

        # The newest click as mask pixel row/col, through the SAME mapping used
        # for the point arrays above. Both locality rules below need it.
        click_rc = None
        try:
            if getattr(self, "_last_click_point", None) is not None:
                cx, cy = self._last_click_point
                crow, ccol = crop_pixel_of(cx, cy)
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
        if getattr(self, "_last_click_polarity", "positive") == "positive":
            if self._is_refining_saved_object:
                # Editing ONE object: the answer joins that object, or it is not
                # part of it. Pixel size comes from this crop, so the weld gap is
                # a ground distance whatever the resolution. What the OTHER
                # detections cover is not consulted: the object open for editing
                # is the only one in play, and it may grow wherever the user
                # points, overlap included.
                px_size = (maxx - minx) / float(img_width) if img_width else 0.0
                (self.current_mask, self._last_click_stood_clear) = \
                    self._grown_in_one_piece(
                        self.current_mask, prev_mask_for_merge,
                        img_height, img_width, px_size, raw_answer)
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

    # How far, in crop pixels, a click may land from the pixel a ghost was
    # asked about and still be answered by it. The preview loop's own drift,
    # so the two agree on what one spot is. Only the fallback test below reads
    # it: a click inside the drawn outline is the ghost's own question at any
    # distance, because the outline is what the user was aiming at.
    _HOVER_REUSE_NEAR_PX = 32

    def _reused_hover_answer(self, held, crop_bounds, img_shape, points):
        """The ghost's own answer for this click, as ``(masks, scores, seed)``.

        None whenever the click is not the question the ghost asked: another
        crop, another spot, more than one point, or an answer carrying no seed
        for the click after it.

        Two ways to be the same question, and the first is the one that matches
        what the user did. They aimed at an outline, so a click inside that
        outline is a click on the ghost, however far it sits from the pixel the
        hover happened to ask about. Simplify, Round corners and Expand all
        move that outline off the raw mask, so a shaped ghost regularly covers
        ground the mask does not, and testing the mask alone refused reuse for
        clicks that landed squarely on the drawn shape. The mask-and-distance
        test stays as the fallback for a click just off the drawn edge.

        Every doubt refuses: a refused reuse costs one round trip, a wrong one
        hands the user another object.
        """
        import numpy as np

        try:
            bounds, shape, asked, mask, score, logits, drawn = held
        except (TypeError, ValueError):
            return None
        if logits is None or len(points) != 1:
            return None
        if tuple(bounds) != tuple(crop_bounds):
            return None
        if (int(shape[0]), int(shape[1])) != (int(img_shape[0]), int(img_shape[1])):
            return None
        col, row = int(points[0][0]), int(points[0][1])
        if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]):
            return None
        answer = np.asarray([mask]), np.asarray([float(score)]), logits
        # The drawn outline arrives in the raster CRS, the frame crop_bounds is
        # in, so the click pixel goes back to the ground rather than the shape
        # coming to the grid. Its centre is the point the pixel address stands
        # for.
        if drawn is not None:
            try:
                minx, miny, maxx, maxy = (float(v) for v in crop_bounds)
                height, width = int(shape[0]), int(shape[1])
                if height > 0 and width > 0 and maxx > minx and maxy > miny:
                    x = minx + (col + 0.5) * (maxx - minx) / width
                    y = maxy - (row + 0.5) * (maxy - miny) / height
                    if drawn.contains(QgsPointXY(x, y)):
                        return answer
            except Exception:  # noqa: BLE001 -- an unreadable ghost answers nothing  # nosec B110
                pass
        if (abs(row - int(asked[0])) > self._HOVER_REUSE_NEAR_PX
                or abs(col - int(asked[1])) > self._HOVER_REUSE_NEAR_PX):
            return None
        if not mask[row, col]:
            # Off the drawn shape and off the mask under it, so the click is
            # asking about something else and the service has to say what.
            return None
        return answer

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

    def _end_click_quietly(self, reason: str) -> None:
        """Mark the click now unwinding as one with nothing to report.

        Read once by the rollback below, which is the only place that decides
        whether a click that stored no prediction owes the user a message."""
        self._quiet_click_end = reason

    def _take_quiet_click_end(self):
        """Why the click now unwinding ended quietly, or None, and clear it."""
        reason = getattr(self, "_quiet_click_end", None)
        self._quiet_click_end = None
        return reason

    def _rollback_failed_click(self, polarity: str, canvas_point=None):
        """Undo all state added by a click that stored no prediction.

        Without this, a failed prediction leaves a marker and a prompt point
        that never contributed to the mask, silently desyncing every later
        prediction and undo.

        Not every such click FAILED. One whose crop moved on while its answer
        travelled, and one whose imagery has to be read again, both end here
        with nothing wrong and nothing the user can act on, so they unwind
        without the message. The second keeps going: it is handed to the same
        deferred replay a click on a cold crop uses, and comes back by itself
        once the imagery lands, which is why it needs ``canvas_point``.
        """
        self.prompts.undo()
        if polarity == "positive" and self._active_crop_points_positive:
            self._active_crop_points_positive.pop()
        elif polarity == "negative" and self._active_crop_points_negative:
            self._active_crop_points_negative.pop()
        if self._mask_state_history:
            self._restore_mask_state(self._mask_state_history.pop())
        quiet = self._take_quiet_click_end()
        if quiet == QUIET_CLICK_REREAD and canvas_point is not None:
            # Hand the click to the replay, which takes its marker down now and
            # puts it back when it re-drives the click.
            self._remember_pending_manual_click(polarity, canvas_point)
            return
        if self.map_tool:
            self.map_tool.remove_last_marker()
        if quiet is None and not self._headless:
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
            self._warn_if_unsure(score)
            self._update_mask_visualization()
        else:
            # No active mask: _update_mask_visualization keeps any frozen
            # or unfrozen polygons on screen instead of wiping them.
            self._update_mask_visualization()

        self._safe_restore_canvas_focus()

    def _warn_if_unsure(self, score: float) -> None:
        """Say so when the model's own score says the outline is probably wrong.

        The outline is still drawn and still keepable: this is a hint, not a
        gate. It has to be, because the signal is real but weak.

        The score does NOT mean the same thing on the two engines, so this
        reads which one answered before it says anything. Only the off-machine
        answer carries a score the floor was set against. Half right is worse
        than silent, so the hint stays quiet on the other one.

        Off unless the server names a floor (``review.click_unsure_below``), and
        once per object, because a user correcting a hard shape would otherwise
        read the same sentence on every click of it.

        The sentence is served too, so the floor and the words it triggers can
        be retuned in the same deploy.
        """
        if self._unsure_warning_shown:
            return
        if not getattr(getattr(self, "predictor", None),
                       "last_answer_was_remote", False):
            return
        try:
            from ...core.detection_policy import click_unsure_below
            floor = click_unsure_below()
        except Exception:  # noqa: BLE001 -- a hint must never cost the click  # nosec B110
            return
        if floor <= 0 or score <= 0 or score >= floor:
            return
        self._unsure_warning_shown = True
        from ...core.server_dials import dial_copy

        self.iface.messageBar().pushMessage(
            "AI Segmentation",
            dial_copy(
                "manual.click_unsure",
                tr("The model is unsure about this outline. Click again to correct "
                   "it, or draw it by hand.")),
            level=Qgis.MessageLevel.Info,
            duration=5,
        )

    def _click_answer_travels(self) -> bool:
        """True when the predictor in hand answers this click over the network.

        Asked BEFORE the click goes out, so the wait can be shown while it is
        out. The predictor's own record of where the last answer came from is
        no use for that: it is written after."""
        active = getattr(self, "_cloud_correct_predictor_active", None)
        if active is None:
            return False
        try:
            return bool(active())
        except Exception:  # noqa: BLE001 -- a click must not fail over this
            return False

    def _remote_click_wait_showing(self) -> bool:
        """True while a click of base Semi-Auto waits on an answer travelling
        back."""
        return bool(getattr(self, "_remote_click_wait_active", False))

    def _begin_click_wait(self) -> bool:
        """Say the model is working, for as long as this click blocks: the
        polygon goes dashed and the cursor turns busy. True only when THIS call
        started the wait, which is what the caller ends.

        The review's fix session has its own wait, and that one is used where it
        applies. A click answered over the network in base Semi-Auto has no
        session and waits just as long, the first one of a sitting longest of
        all, and an arrow cursor with nothing moving for that long reads as a
        crash. So it gets the same treatment here."""
        if self._begin_correct_wait():
            return True
        if self._correct_wait_showing() or self._remote_click_wait_showing():
            return False
        if not self._click_answer_travels():
            return False
        self._remote_click_wait_active = True
        self._remote_click_wait_cursor = False
        if not self._headless:
            try:
                QApplication.setOverrideCursor(WaitCursor)
                self._remote_click_wait_cursor = True
            except (RuntimeError, AttributeError):
                self._remote_click_wait_cursor = False
        self._arm_remote_click_note()
        self._apply_mask_band_style()
        return True

    # How long a travelling answer may be silent before the panel names the
    # wait. Short enough that a machine which has to start up says so long
    # before the user decides the plugin is broken, long enough that an answer
    # on a warm machine, which is a tenth of a second, never flashes a line.
    _REMOTE_CLICK_NOTE_MS = 1_200

    def _arm_remote_click_note(self) -> None:
        """Name the wait in the panel, but only once it is long enough to be
        one. Never raises: a click must not fail over a line of text."""
        if self._headless:
            return
        try:
            from .shared import _debounce_timer

            _debounce_timer(self, "_remote_click_note_timer", self.dock_widget,
                            self._REMOTE_CLICK_NOTE_MS, self._show_remote_click_note)
        except Exception:  # noqa: BLE001 -- a note must never break a click  # nosec B110
            pass

    def _show_remote_click_note(self) -> None:
        """The answer is still travelling: say so. Re-checks the wait, because
        the timer outlives the click that armed it."""
        if not self._remote_click_wait_showing():
            return
        self._set_manual_encoding_note(True, phase="remote")

    def _clear_remote_click_note(self) -> None:
        """Take the line down, and disarm a timer that has not fired yet."""
        timer = getattr(self, "_remote_click_note_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except (RuntimeError, AttributeError):
                pass
        self._set_manual_encoding_note(False)

    def _end_click_wait(self) -> None:
        """Take this click's waiting outline and busy cursor back down. ONE pop
        for the one push, whichever of the two waits went up."""
        if not self._remote_click_wait_showing():
            self._end_correct_wait()
            return
        self._remote_click_wait_active = False
        if getattr(self, "_remote_click_wait_cursor", False):
            self._remote_click_wait_cursor = False
            try:
                QApplication.restoreOverrideCursor()
            except (RuntimeError, AttributeError):
                pass
        self._clear_remote_click_note()
        self._apply_mask_band_style()

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
        waiting = (self._correct_wait_showing()
                   or self._remote_click_wait_showing())
        try:
            self.mask_rubber_band.setLineStyle(
                DashLine if waiting else SolidLine)
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

    def _manual_simplify_tolerance(self, geom, transform_info) -> float:
        """The Simplify spinbox as a distance, capped by the object's own size.

        The spinbox speaks crop pixels, which is the right unit for a staircase
        and the wrong one for a small object: the same two pixels are nothing on
        a warehouse and a tenth of a car, and a zoomed-out crop makes one pixel
        several metres. So the tolerance is also bounded by a share of the
        object's narrow dimension, the same guard the point budget puts on its
        own deviation cap. Returns 0 when the control is off.
        """
        tolerance = self._compute_simplification_tolerance(
            transform_info, self._refine_simplify)
        if tolerance <= 0:
            return 0.0
        from ...core.review_defaults import REFINE_SIMPLIFY_MAX_NARROW_FRACTION

        if REFINE_SIMPLIFY_MAX_NARROW_FRACTION <= 0:
            return tolerance
        try:
            _pt, _area, _angle, width, height = geom.orientedMinimumBoundingBox()
            narrow = min(float(width), float(height))
        except Exception:  # noqa: BLE001 -- unmeasurable, keep the flat value  # nosec B110
            return tolerance
        if narrow <= 0:
            return tolerance
        return min(tolerance, REFINE_SIMPLIFY_MAX_NARROW_FRACTION * narrow)

    def _manual_vertex_deviation_cap(self, base_cap_m: float, transform_info,
                                     served_flat_m: float = 0.0,
                                     metres_per_unit: float = 1.0) -> float:
        """The furthest the point budget may move a Semi-Auto outline, in ground
        metres.

        The served cap is shared with Automatic, where it is the value a run's
        classes were calibrated against. Semi-Auto has no class and one object on
        screen, and at the shared value the budget is allowed to cut a real
        building step off as if it were noise, so it keeps a tighter one of its
        own. The floor matters as much as the value: one staircase step costs
        about 0.7 of a ground pixel, so a cap under that can drop no vertex at
        all and the raw traced outline ships instead.

        ``served_flat_m`` is the run's plain cap when ``base_cap_m`` is one of
        its variants (Round corners runs on a deliberately looser one). The
        tighter Semi-Auto value scales by the same ratio, so this replaces the
        cap without flattening the difference between the two.

        ``metres_per_unit`` is the ground scale under the object, resolved by
        the caller (which cannot run the budget at all without it).
        """
        from ...core.review_defaults import (
            REFINE_VERTEX_DEVIATION_PIXEL_FLOOR,
            REFINE_VERTEX_MAX_DEVIATION_M,
        )

        cap = REFINE_VERTEX_MAX_DEVIATION_M
        if cap <= 0:
            return base_cap_m
        if served_flat_m > 0 and base_cap_m > 0:
            cap *= base_cap_m / served_flat_m
        # Never looser than what the run would have used anyway.
        if base_cap_m > 0:
            cap = min(cap, base_cap_m)
        # The pixel floor goes on LAST. Clamped after it, it was thrown away on
        # exactly the crops it exists for: past a certain ground pixel size one
        # staircase step moves the boundary further than the flat cap, so no
        # vertex may be dropped at all and the raw traced outline ships.
        px_units = self._crop_pixel_size_units(transform_info)
        if px_units > 0:
            cap = max(cap, REFINE_VERTEX_DEVIATION_PIXEL_FLOOR
                      * px_units * metres_per_unit)
        return cap

    def _manual_metres_per_unit(self, ref_x: float, ref_y: float):
        """Ground metres per X unit of the current layer CRS near (ref_x, ref_y),
        or None when it cannot be measured.

        Mirrors the Automatic review's _auto_crs_metres_per_unit: a Web Mercator
        unit is well under a metre, a geographic CRS counts in degrees, so a
        ground-metre dial has to cross that gap before it touches a geometry.

        None, and never 1.0. Substituting 1.0 does not fail safe, it changes the
        unit: on a degree-based layer a spacing of a few ground metres is then
        read as a few degrees, the budget falls to its floor and a building
        ships as an octagon, with nothing on screen to say so. A caller with no
        use for a wrong number skips its step instead.

        The X axis only. In a geographic CRS the Y axis measures differently,
        and pairing this with ``_manual_unit_aspect`` is what covers that."""
        layer = getattr(self, "_current_layer", None)
        if layer is None:
            return None
        try:
            crs = layer.crs()
            if not crs.isValid():
                return None
            geographic = bool(crs.isGeographic())
        except Exception:  # noqa: BLE001 -- an unusable CRS means no conversion
            return None
        try:
            from ...core.layer_conventions import make_area_measurer
            step = 0.001 if geographic else 1.0
            metres = float(make_area_measurer(crs).measureLine(
                QgsPointXY(ref_x, ref_y), QgsPointXY(ref_x + step, ref_y)))
            return metres / step if metres > 0 else None
        except Exception:  # noqa: BLE001 -- never block a refine on a measure
            return None

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
        bbox = combined.boundingBox()
        centre = bbox.center()
        aspect = self._manual_unit_aspect(centre.x(), centre.y())
        # The served dials are in ground metres, so with no ground scale they
        # cannot be crossed into this CRS at all, and the object takes the
        # pixel-anchored path below, which needs none.
        factor = self._manual_metres_per_unit(centre.x(), centre.y())
        if factor is not None and factor > 0:
            try:
                from ...core.detection_policy import (
                    destair_tolerance_m,
                    regularize_settings,
                    regularize_tolerance_m,
                )
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
                    # Multi-direction path: OFF unless the server turns it on,
                    # the same dial the Automatic review forwards. With it on, a
                    # building whose wing sits at an angle to the main block
                    # keeps each wing on its own grid instead of staircased.
                    multi_direction=bool(s["multi_direction"]),
                    multi_max_groups=int(s["multi_max_groups"]),
                    multi_min_separation_deg=float(
                        s["multi_min_separation_deg"]),
                    unit_aspect=aspect,
                    envelope=_envelope)
            except Exception:  # noqa: BLE001 -- take the pixel-anchored path
                pass  # nosec B110
        destair3 = self._compute_simplification_tolerance(transform_info, 1.5)
        return apply_right_angles(
            combined,
            destair_tol=max(0.0, destair3 - tolerance),
            tolerance_m=destair3,
            unit_aspect=aspect,
            envelope=_envelope)

    def _manual_despike_distance(self, combined, transform_info) -> float:
        """The spike-cut opening distance for ``combined``, in the layer's CRS
        units. Resolved the way the Automatic review resolves it: a ground dial
        (core.detection_policy.despike_tolerance_m) crossed into CRS units by
        the ground scale under the object. 0.0 is the OFF state and is what an
        untuned server gives, so this stays offline-safe like the rest of
        Manual, and it is also the answer when there is no ground scale to
        cross the dial with: a metre distance applied to degrees would open the
        shape away entirely."""
        try:
            from ...core.detection_policy import despike_tolerance_m
            pixel_units = self._crop_pixel_size_units(transform_info)
            centre = combined.boundingBox().center()
            factor = self._manual_metres_per_unit(centre.x(), centre.y())
            if factor is None or factor <= 0:
                return 0.0
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
        tolerance = self._manual_simplify_tolerance(combined, transform_info)
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
        combined = self._apply_manual_vertex_budget(combined, transform_info)
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

    def _apply_manual_vertex_budget(self, combined, transform_info=None):
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
            if factor is None or factor <= 0:
                # The spacing and the cap are ground metres, and this is what
                # turns them into distances on this layer. Without it they would
                # be read as raw CRS units: on a degree-based layer that hands
                # every object the floor of the budget. The outline keeps all
                # its points instead, which is honest and reversible.
                return combined
            r = simplify_to_budget(
                combined,
                spacing=spacing_m / factor,
                min_vertices=min_pts,
                max_deviation=self._manual_vertex_deviation_cap(
                    dev_m, transform_info, float(s["max_deviation_m"]),
                    metres_per_unit=factor) / factor,
                max_deviation_fraction=float(s["max_deviation_fraction"]),
                dial_max_cap_fraction=float(s["dial_max_cap_fraction"]),
                keep_fraction=keep_fraction,
                # The two axes of a geographic CRS do not cover the same ground,
                # and every length in the budget is measured on raw coordinates.
                unit_aspect=self._manual_unit_aspect(centre.x(), centre.y()),
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
        previewed. None when no active mask or nothing survives refinement.

        The work itself lives in ManualShapeCacheMixin, which remembers the
        answer: the preview has usually computed this shape already, and Save
        and Export used to pay for it a second time.
        """
        return self._manual_active_outline()

    def _fill_holes_pixel_cap(self, info=None):
        """The Fill-holes size threshold in MASK PIXELS.

        Three answers, and the last two are NOT the same. A number is the
        cutoff. None is "no cutoff, fill every hole", which is the control at 0.
        FILL_HOLES_CAP_UNKNOWN is "the ground size of a pixel cannot be
        measured here", so the cutoff the user asked for cannot be worked out:
        answering None there fills every courtyard of an object whose owner
        asked for a bounded fill, and nothing on screen says why.

        The user's number is true ground m2, like Min/Max size, so it crosses to
        pixels through the same area convention (layer_conventions.
        make_area_measurer): measure the crop's ground area, divide by its pixel
        count, and one mask pixel has a ground area whatever the layer CRS is.

        ``info`` is the crop window the cap is asked about; the click session's
        own window when left out. The hover ghost passes its answer's window,
        which exists before any click has set the session's.
        """
        max_m2 = float(getattr(self, "_refine_fill_holes_max_m2", 0.0) or 0.0)
        if max_m2 <= 0:
            return None
        if info is None:
            info = self.current_transform_info
        if not info:
            return FILL_HOLES_CAP_UNKNOWN
        try:
            from qgis.core import QgsRectangle

            from ...core.hole_size import hole_pixels
            from ...core.layer_conventions import make_area_measurer
            minx, maxx, miny, maxy = (float(v) for v in info["bbox"])
            rows, cols = int(info["img_shape"][0]), int(info["img_shape"][1])
            if rows <= 0 or cols <= 0:
                return FILL_HOLES_CAP_UNKNOWN
            rect = QgsGeometry.fromRect(QgsRectangle(minx, miny, maxx, maxy))
            ground_m2 = 0.0
            if self._current_layer is not None and self._current_layer.crs().isValid():
                ground_m2 = float(
                    make_area_measurer(self._current_layer.crs()).measureArea(rect))
            if ground_m2 <= 0:
                ground_m2 = float(rect.area())
            if ground_m2 <= 0:
                return FILL_HOLES_CAP_UNKNOWN
            cap = hole_pixels(max_m2, ground_m2 / (rows * cols))
            return FILL_HOLES_CAP_UNKNOWN if cap is None else cap
        except (RuntimeError, AttributeError, KeyError, TypeError, ValueError):
            return FILL_HOLES_CAP_UNKNOWN

    def _fill_holes_arguments(self, info=None):
        """``(fill_holes, max_hole_px)`` for apply_mask_refinement.

        The step is OFF when the user asked for a bounded fill and the bound
        cannot be measured. apply_mask_refinement reads a max of None as "fill
        every hole", so passing an unmeasurable bound through as None does the
        opposite of what was asked, on the objects that have courtyards to
        lose. ``info`` names the crop window, as in _fill_holes_pixel_cap."""
        if not self._refine_fill_holes:
            return False, None
        cap = self._fill_holes_pixel_cap(info)
        if cap is FILL_HOLES_CAP_UNKNOWN:
            return False, None
        return True, cap

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
            from ...core.polygon_exporter import count_significant_regions

            # Mask-level simplify tolerance: same OFF-by-default server dial as
            # the save/export path, so the preview equals what a save keeps.
            _mult = manual_simplify_multiple_of_px()
            _tol = (_mult * self._crop_pixel_size_units(self.current_transform_info)
                    if _mult > 0 else 0.0)

            # The mask stage is memoized in ManualShapeCacheMixin: six of the
            # ten refine controls (Points, Simplify, Trim spikes, Round corners,
            # Right angles, Min/Max size) change none of its inputs, and a move
            # on any of them used to re-clean and re-polygonize the whole mask.
            fill_holes, max_hole_px = self._fill_holes_arguments()
            mask_to_display, geometries = self._manual_mask_polygons(
                fill_holes, max_hole_px, _tol)

            # Detect disjoint regions and show message bar warning. The region
            # count dilates and labels the whole mask, so it is asked for only
            # when the one-shot warning can still fire; every repaint used to
            # pay it (refine sliders repaint on every drag step). A mask that
            # polygonized to at most one piece cannot hold more than one
            # significant region, so that already-computed count gates the
            # expensive dilate-and-label pass too.
            may_warn = not self._disjoint_warning_shown and len(self._active_crop_points_positive) >= 2
            if (may_warn and len(geometries) > 1
                    and count_significant_regions(mask_to_display) > 1):
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("Disconnected parts detected. For best accuracy, segment one element at a time."),
                    level=Qgis.MessageLevel.Warning,
                    duration=6
                )
                self._disjoint_warning_shown = True

            # Build composite: frozen polygons + active mask polygons
            all_geoms = [s.polygon for s in self._frozen_sessions]

            # Trim spikes, simplify, right angles, round corners, then the user
            # Min/Max size window: the SAME shape Save and Export keep, and the
            # same memo they read it from, so the preview cannot drift from the
            # file and neither of them recomputes it.
            if geometries:
                active_combined = self._manual_active_outline()
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
        """Undo last point added, or restore last saved mask in batch mode.

        A wrapper so the Add lane's two buttons follow every path out of the
        undo below, including the early ones: what is on screen after an undo
        is what decides whether Keep and Undo are still offered.
        """
        try:
            self._undo_one_gesture()
        finally:
            refresh = getattr(self, "_refresh_ai_add_keep_button", None)
            if refresh is not None:
                refresh()

    def _undo_one_gesture(self):
        """Take back one gesture: the last point, else the last frozen part,
        else the last deleted object."""
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
                # No points is not nothing to save: the shape on screen is
                # what a Save commits, and the point count alone greyed the
                # button out from under it.
                self._keep_save_alive_for_display_polygon()
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
                # No by default: the press that lands here is usually one undo
                # too many, and Yes re-opens an object that was finished.
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._restore_last_saved_mask()
            self._safe_restore_canvas_focus()

    def _keep_save_alive_for_display_polygon(self) -> None:
        """Re-arm Save after a point count of zero left a shape on the map.

        The dock drives Save off the point count, which is right for a click
        session and wrong for an object opened for editing: it carries a shape
        and no points of its own until the first click lands.
        """
        if self._unfrozen_display_polygon is None:
            return
        button = getattr(self.dock_widget, "save_mask_button", None)
        if button is None:
            return
        try:
            button.setEnabled(True)
        except RuntimeError:
            pass

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
        # Same rule as a reopened polygon: the shape is back on the map, and
        # the point count it came back with must not decide Save.
        self._keep_save_alive_for_display_polygon()

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
        # It carries its identity back into the edit. Without this the re-save
        # minted a fresh det_id, so the object lost the colour it was saved
        # with AND read as brand new to the credit ledger: re-opening a
        # cloud-traced object and saving it again spent a second credit on the
        # one object. The handoff re-open has always done this (see
        # manual_handoff._open_saved_polygon_for_edit).
        self._active_refine_origin_entry = dict(last_polygon)

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
        # A polygon reopened for editing carries a shape and no points of its
        # own, and the dock drives Save off the point count. Without this the
        # object comes back on the map with Save greyed out under it.
        self._keep_save_alive_for_display_polygon()
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
        self._unsure_warning_shown = False
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

        # Reset online layer state. The private copy the warm-ups read imagery
        # through goes with it: it belongs to the layer this session had.
        self._is_online_layer = False
        try:
            from ...core.online_layer_twin import release_online_layer_twin
            release_online_layer_twin()
        except Exception:  # noqa: BLE001 -- a reset must never raise  # nosec B110
            pass
        # The raster held open for this session's windowed reads goes with it,
        # so the file is free to be moved or deleted the moment the user is
        # done with it.
        try:
            from ...core.raster_dataset_cache import release_raster_datasets
            release_raster_datasets()
        except Exception:  # noqa: BLE001 -- a reset must never raise  # nosec B110
            pass
        self._manual_shape_cache_reset()

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

        # One-shot lines and a parked tool belong to the session that raised
        # them: a new session must be able to say the same thing again, and
        # must never inherit an older session's park.
        self._tool_rearm_notice_shown = False
        self._manual_session_parked = False
        self._click_without_model_reported = False

        if self.dock_widget:
            self.dock_widget.set_point_count(0, 0)
            self.dock_widget.set_saved_polygon_count(0)
            try:
                # The notice describes a click in the session that is ending.
                self.dock_widget.clear_manual_notice()
            except (RuntimeError, AttributeError):
                pass
