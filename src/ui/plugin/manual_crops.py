"""Manual CRS transforms, crop extraction and encoding, re-encode decisions.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

import math

from qgis.core import (
    Qgis,
    QgsGeometry,
    QgsMessageLog,
    QgsPointXY,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QApplication,
)

from ...core.i18n import tr
from ...core.prompt_manager import FrozenCropSession
from ..error_report_dialog import show_error_report


class ManualCropsMixin:
    """Manual CRS transforms, crop extraction and encoding, re-encode decisions."""

    def _transform_to_raster_crs(self, point):
        """Transform a QgsPointXY from canvas CRS to raster CRS.

        Returns the original point unchanged when both CRS are identical.
        """
        if self._canvas_to_raster_xform is not None:
            return self._canvas_to_raster_xform.transform(point)
        return point

    def _transform_geometry_to_canvas_crs(self, geometry):
        """Transform a QgsGeometry from raster CRS to canvas CRS (in-place).

        Does nothing when both CRS are identical.
        """
        if self._raster_to_canvas_xform is not None:
            geometry.transform(self._raster_to_canvas_xform)

    def _transform_to_canvas_crs(self, point):
        """Transform a QgsPointXY from raster CRS to canvas CRS.

        Returns the original point unchanged when both CRS are identical.
        """
        if self._raster_to_canvas_xform is not None:
            return self._raster_to_canvas_xform.transform(point)
        return point

    def _is_point_in_raster_extent(self, point):
        """Check if a point (in raster CRS) falls within the layer extent."""
        if not self._is_layer_valid():
            return False
        try:
            ext = self._current_layer.extent()
            # Transform extent to raster CRS if needed
            if self._canvas_to_raster_xform is not None:
                # Layer extent is in layer CRS, point is already in raster CRS
                pass
            in_x = ext.xMinimum() <= point.x() <= ext.xMaximum()
            in_y = ext.yMinimum() <= point.y() <= ext.yMaximum()
            return in_x and in_y
        except RuntimeError:
            return False

    def _check_crop_status(self, point):
        """Check if a point (in raster CRS) is usable in the current crop.

        Returns a reason code:
        - "ok": point is inside the crop
        - "no_crop": no crop has been encoded yet
        - "outside_bounds": point is geographically outside the crop
        - "zoom_changed": user zoomed in significantly, crop should be re-encoded
        """
        if self._current_crop_info is None:
            return "no_crop"
        bounds = self._current_crop_info["bounds"]
        in_x = bounds[0] <= point.x() <= bounds[2]
        in_y = bounds[1] <= point.y() <= bounds[3]
        if not (in_x and in_y):
            return "outside_bounds"

        # Detect significant zoom-in requiring higher resolution.
        # Skip zoom re-encode when there are active points - re-encoding
        # destroys the current mask via lossy 64x64 logit transfer.
        # The existing crop is still valid (point is in bounds), so SAM
        # can predict just fine on the current encoding.
        has_active_points = (self._active_crop_points_positive
                             or self._active_crop_points_negative)  # noqa: W503
        if not has_active_points:
            # No active points - always use tight thresholds so any
            # meaningful zoom change triggers re-encode at the correct
            # resolution.  Loose thresholds (old 0.7/1.5) caused SAM to
            # reuse a closer-zoom encoding when the user zoomed out,
            # segmenting a small element instead of the full object.
            zoom_in_thresh = 0.85
            zoom_out_thresh = 1.15

            if self._is_online_layer:
                canvas = self.iface.mapCanvas()
                current_canvas_mupp = canvas.mapUnitsPerPixel()
                if self._current_crop_canvas_mupp and current_canvas_mupp > 0:
                    ratio = current_canvas_mupp / self._current_crop_canvas_mupp
                    if ratio < zoom_in_thresh or ratio > zoom_out_thresh:
                        return "zoom_changed"
            else:
                if self._current_crop_canvas_mupp is not None:
                    canvas = self.iface.mapCanvas()
                    current_mupp = canvas.mapUnitsPerPixel()
                    if current_mupp > 0:
                        ratio = current_mupp / self._current_crop_canvas_mupp
                        if ratio < zoom_in_thresh or ratio > zoom_out_thresh:
                            return "zoom_changed"

        return "ok"

    def _snapshot_mask_state(self) -> dict:
        """Snapshot the full prediction state for one undo step.

        The low-res logits are part of the state: without them, the click
        after an undo re-predicts from scratch and can produce a mask that
        no longer matches what the user sees on screen.
        """
        return {
            "mask": self.current_mask.copy() if self.current_mask is not None else None,
            "score": self.current_score,
            "transform_info": self.current_transform_info,
            "low_res_mask": (self.current_low_res_mask.copy()
                             if self.current_low_res_mask is not None else None),
            # Display-only polygon (an open handoff edit before its first
            # click, or an unfrozen session): the first prediction consumes
            # it, so undoing that click must bring it back.
            "display_polygon": (QgsGeometry(self._unfrozen_display_polygon)
                                if self._unfrozen_display_polygon is not None
                                else None),
        }

    def _restore_mask_state(self, state: dict) -> None:
        """Restore a state captured by _snapshot_mask_state."""
        self.current_mask = state["mask"]
        self.current_score = state["score"]
        self.current_transform_info = state["transform_info"]
        self.current_low_res_mask = state.get("low_res_mask")
        self._unfrozen_display_polygon = state.get("display_polygon")

    def _invalidate_history_logits(self) -> None:
        """Drop low-res logits from undo history after the crop changed.

        Logits live in crop-image space: pairing them with a different
        encoding corrupts refinement. Geographic masks stay valid.
        """
        for state in self._mask_state_history:
            state["low_res_mask"] = None

    @staticmethod
    def _resize_nearest(arr, target_h, target_w):
        """Resize a 2D numpy array using nearest-neighbor interpolation."""
        import numpy as np
        src_h, src_w = arr.shape
        row_idx = (np.arange(target_h) * src_h / target_h).astype(int)
        col_idx = (np.arange(target_w) * src_w / target_w).astype(int)
        np.clip(row_idx, 0, src_h - 1, out=row_idx)
        np.clip(col_idx, 0, src_w - 1, out=col_idx)
        return arr[row_idx[:, None], col_idx[None, :]]

    def _binary_mask_to_logits(self, mask, target: int = 256):
        """Convert a binary mask (H x W, 0/1 or bool) to SAM low-res logits of
        shape (1, 1, target, target): foreground=+6, background=-6. Shared by the
        zoom mask-transfer and the Refine-in-Manual polygon seeding so both seed
        SAM the same way."""
        import numpy as np
        m = np.asarray(mask, dtype=np.float32)
        logits = (m * 2.0 - 1.0) * 6.0
        logits_t = self._resize_nearest(logits, target, target)
        return logits_t[None, None, :, :]

    def _build_mask_input_from_previous(
        self, old_mask, old_bounds, old_shape, new_bounds, new_shape
    ):
        """Transfer a binary mask from old crop space to new crop as SAM logits.

        Computes geographic overlap between old and new crops, maps the
        overlapping region, converts to logits, and resizes to (1, 256, 256).
        Returns None if there is no overlap.
        """
        import numpy as np

        old_minx, old_miny, old_maxx, old_maxy = old_bounds
        new_minx, new_miny, new_maxx, new_maxy = new_bounds

        # Geographic overlap
        ovlp_minx = max(old_minx, new_minx)
        ovlp_miny = max(old_miny, new_miny)
        ovlp_maxx = min(old_maxx, new_maxx)
        ovlp_maxy = min(old_maxy, new_maxy)
        if ovlp_minx >= ovlp_maxx or ovlp_miny >= ovlp_maxy:
            return None

        old_h, old_w = old_shape
        new_h, new_w = new_shape

        def geo_to_pixel(gx, gy, bminx, bminy, bmaxx, bmaxy, pw, ph):
            col = (gx - bminx) / (bmaxx - bminx) * pw
            row = (bmaxy - gy) / (bmaxy - bminy) * ph
            return int(round(col)), int(round(row))

        # Overlap region in old pixel coords
        o_c0, o_r0 = geo_to_pixel(
            ovlp_minx, ovlp_maxy, old_minx, old_miny, old_maxx, old_maxy,
            old_w, old_h)
        o_c1, o_r1 = geo_to_pixel(
            ovlp_maxx, ovlp_miny, old_minx, old_miny, old_maxx, old_maxy,
            old_w, old_h)
        o_r0 = max(0, min(o_r0, old_h))
        o_r1 = max(0, min(o_r1, old_h))
        o_c0 = max(0, min(o_c0, old_w))
        o_c1 = max(0, min(o_c1, old_w))
        if o_r0 >= o_r1 or o_c0 >= o_c1:
            return None

        patch = old_mask[o_r0:o_r1, o_c0:o_c1]

        # Overlap region in new pixel coords
        n_c0, n_r0 = geo_to_pixel(
            ovlp_minx, ovlp_maxy, new_minx, new_miny, new_maxx, new_maxy,
            new_w, new_h)
        n_c1, n_r1 = geo_to_pixel(
            ovlp_maxx, ovlp_miny, new_minx, new_miny, new_maxx, new_maxy,
            new_w, new_h)
        n_r0 = max(0, min(n_r0, new_h))
        n_r1 = max(0, min(n_r1, new_h))
        n_c0 = max(0, min(n_c0, new_w))
        n_c1 = max(0, min(n_c1, new_w))
        target_h = n_r1 - n_r0
        target_w = n_c1 - n_c0
        if target_h < 1 or target_w < 1:
            return None

        resized_patch = self._resize_nearest(patch, target_h, target_w)

        # Place patch into full-size new crop mask
        new_mask = np.zeros((new_h, new_w), dtype=np.float32)
        new_mask[n_r0:n_r1, n_c0:n_c1] = resized_patch

        # Convert to SAM's low-res logits (1, 1, 256, 256): foreground=+6, bg=-6.
        return self._binary_mask_to_logits(new_mask)

    def _compute_crop_center_and_mupp(self, all_points_geo, crop_size=1024):
        """Compute optimal crop center and mupp to fit all points.

        When points span more than crop_size pixels at the current resolution,
        this zooms out (increases mupp) so all points fit in one image.

        Args:
            all_points_geo: list of (x, y) tuples in raster CRS
            crop_size: target image size in pixels (1024)

        Returns:
            (center_x, center_y, mupp_or_scale) where the third value is:
            - For online layers: the mupp needed (at least canvas mupp)
            - For file-based layers: the scale_factor (>= 1.0)
        """
        xs = [p[0] for p in all_points_geo]
        ys = [p[1] for p in all_points_geo]
        bbox_width = max(xs) - min(xs)
        bbox_height = max(ys) - min(ys)
        center_x = (min(xs) + max(xs)) / 2.0
        center_y = (min(ys) + max(ys)) / 2.0

        # Add 20% margin on each side (1.4x total)
        needed_geo_size = max(bbox_width, bbox_height) * 1.4

        if self._is_online_layer:
            canvas_mupp = self.iface.mapCanvas().mapUnitsPerPixel()
            # mupp must cover at least the needed area, but not less than canvas
            needed_mupp = needed_geo_size / crop_size
            mupp = max(canvas_mupp, needed_mupp)
            return center_x, center_y, mupp
        # For file-based layers, compute scale_factor from native pixel size
        native_pixel_size = self._get_native_pixel_size()
        if native_pixel_size > 0:
            needed_mupp = needed_geo_size / crop_size
            scale_factor = max(1.0, needed_mupp / native_pixel_size)
        else:
            scale_factor = 1.0
        return center_x, center_y, scale_factor

    def _get_native_pixel_size(self):
        """Get the native pixel size of the current file-based raster layer."""
        try:
            ext = self._current_layer.extent()
            w = self._current_layer.width()
            h = self._current_layer.height()
            if w > 0 and h > 0:
                px = (ext.xMaximum() - ext.xMinimum()) / w
                py = (ext.yMaximum() - ext.yMinimum()) / h
                return max(px, py)
        except (RuntimeError, AttributeError):
            pass
        return 0.0

    def _compute_auto_min_area(self):
        """Compute min_area for artifact removal based on current crop scale.

        SAM artifacts are small disconnected blobs (1-25 pixels) that appear
        regardless of input content.  They get slightly larger when the input
        image is heavily downsampled (high scale_factor = zoomed out).

        Uses sqrt scaling for a gentle progression that stays well below the
        size of any intentionally selected object (~50+ pixels).

        Returns pixel count in the 1024x1024 SAM mask.
        """
        scale = self._current_crop_scale_factor
        if scale is None or scale <= 0:
            # Online layers or unknown: use the MUPP ratio as proxy
            if (self._current_crop_actual_mupp
                    and self._current_crop_canvas_mupp  # noqa: W503
                    and self._current_crop_canvas_mupp > 0):  # noqa: W503
                scale = max(1.0, self._current_crop_actual_mupp
                            / self._current_crop_canvas_mupp * 2.0)  # noqa: W503
            else:
                scale = 1.0
        # Power curve centered on 200 (bumped ×2 per #12 for cleaner defaults).
        return max(100, int(200 * max(0.6, scale) ** 0.3))

    def _compute_initial_scale_factor(self):
        """Compute initial scale_factor from canvas zoom for file-based rasters.

        For high-res imagery where the user is zoomed out, the crop should cover
        a proportionally larger geographic area instead of just 1024 native pixels.

        Uses canvas extent in raster CRS to avoid unit mismatches (e.g. canvas
        in meters vs raster in degrees).
        """
        if self._is_online_layer:
            return None
        native_pixel_size = self._get_native_pixel_size()
        if native_pixel_size <= 0:
            return None

        canvas = self.iface.mapCanvas()
        canvas_extent = canvas.extent()
        # Transform canvas extent to raster CRS if needed
        if self._canvas_to_raster_xform is not None:
            try:
                canvas_extent = self._canvas_to_raster_xform.transformBoundingBox(
                    canvas_extent)
            except Exception:
                return None

        # Compute canvas mupp in raster CRS units
        canvas_width_px = canvas.width()
        if canvas_width_px <= 0:
            return None
        canvas_geo_width = canvas_extent.xMaximum() - canvas_extent.xMinimum()
        canvas_mupp_raster_crs = canvas_geo_width / canvas_width_px

        ratio = canvas_mupp_raster_crs / native_pixel_size
        return max(0.25, min(ratio, 8.0))

    # ------------------------------------------------------------------
    # Crop encoding (PERF-01): async on the interactive path, sync headless.
    #
    # set_image() writes a ~4MB crop over stdin to the SAM subprocess and blocks
    # on stdout until encoding finishes (~3-8s on CPU, longer on old laptops).
    # On the interactive path that round-trip now runs on a background worker
    # (SetImageWorker) so QGIS never freezes; crop EXTRACTION and every QGIS/
    # canvas access stay on the GUI thread. Only ONE encode worker touches the
    # predictor pipe at a time (the `_encoding_in_progress` transport lock), so
    # the JSON-RPC stream is never interleaved. Headless/MCP stays fully
    # synchronous (the API contract is a blocking call).
    #
    # State (lazy-initialised, see _ensure_manual_encode_state; the plugin
    # __init__ already owns `_encoding_in_progress`, the main-thread anchor):
    # - _manual_encode_worker: the live SetImageWorker (None when idle).
    # - _manual_encode_gen: generation counter; a teardown bumps it so a stale
    #   worker completion is dropped (mirrors auto_results `_auto_finalize_gen`).
    # - _pending_encode: {crop_info, tail, predictor, gen} committed on success.
    # - _pending_manual_click: {polarity, canvas_point} to replay on completion
    #   (last click wins; replaced by any click during the encode).
    # ------------------------------------------------------------------

    def _ensure_manual_encode_state(self) -> None:
        """Lazily create the async-encode fields (the plugin __init__ is owned
        by another change in flight and must not be touched)."""
        if not hasattr(self, "_manual_encode_gen"):
            self._manual_encode_gen = 0
            self._manual_encode_worker = None
            self._pending_encode = None
            self._pending_manual_click = None
        if not hasattr(self, "_online_fetch"):
            # The in-flight online crop fetch (async interactive path), a dict
            # {fetcher, gen, on_encoded, cursor}, or None when idle.
            self._online_fetch = None

    def _extract_and_encode_crop(self, center_point, mupp_override=None, *, on_encoded=None):
        """Extract a crop centered on the point and encode it with SAM.

        The contract depends on the run mode:

        - HEADLESS (MCP): fully SYNCHRONOUS. Extracts and encodes on the calling
          thread and returns True on success / False on error. When given,
          ``on_encoded`` runs synchronously on success before returning, so a
          headless caller shares the same continuation as the interactive path.
          This preserves the blocking MCP contract.

        - INTERACTIVE: ASYNCHRONOUS. Returns immediately; the SAM encode
          round-trip runs on a background worker. A True return means the crop
          is being PREPARED (either the encode STARTED, or for an online layer
          the crop FETCH started and the encode follows on completion), NOT
          "crop ready": the caller MUST NOT run a prediction synchronously after
          this. On a successful encode ``on_encoded()`` runs on the GUI thread
          from the completion callback (_on_manual_encode_done). A False return
          means crop extraction failed synchronously (nothing was started), or
          an encode/fetch is already in flight (never start a second one).

          File-based layers extract on the GUI thread first (a fast local
          windowed read), then encode on the worker. Online layers extract on
          the tile network, which can block ~18s of retries, so their fetch is
          driven off the event loop (_begin_online_crop_fetch) and only feeds
          the encode once the tiles stabilize.

        Args:
            center_point: QgsPointXY center in raster CRS
            mupp_override: For online layers, override mupp (zoom-out).
                For file-based layers, this is the scale_factor [0.25, 8.0].
            on_encoded: continuation to run after a successful encode (the
                crop-transition tail). Interactive: runs from the completion
                callback. Headless: runs inline on success.
        """
        if self._encoding_in_progress:
            # A worker (or an online fetch) already owns the predictor pipe;
            # never start a second one. Interactive click callers route here
            # only via the pending-click mechanism, which handles the busy case.
            return False

        if self._headless:
            ok = self._encode_crop_blocking(center_point, mupp_override)
            if ok and on_encoded is not None:
                on_encoded()
            return ok

        if self._is_online_layer:
            # Online tile extraction blocks on the network (fetch + progressive
            # retry, up to ~18s). Drive it asynchronously so the GUI never
            # freezes; True means the crop FETCH started (the encode follows on
            # completion), not that the crop is ready.
            return self._begin_online_crop_fetch(
                center_point, mupp_override, on_encoded)

        # File-based: a fast local windowed read, kept synchronous. The encode
        # itself still runs on the worker (_start_manual_encode) with its own
        # busy cursor, restored on the completion callback.
        image_np, crop_info = self._extract_crop_only(center_point, mupp_override)
        if image_np is None:
            return False  # extraction failed and already surfaced the error
        self._start_manual_encode(image_np, crop_info, on_encoded)
        return True  # encode STARTED (not done); on_encoded runs on completion

    def _online_crop_mupp(self, mupp_override):
        """Canvas -> raster map-units-per-pixel for an online crop, stashing the
        per-crop mupp state. Shared by the sync (_extract_crop_only) and async
        (_begin_online_crop_fetch) online paths so both compute it identically.
        """
        canvas = self.iface.mapCanvas()
        canvas_mupp = canvas.mapUnitsPerPixel()
        # When canvas CRS != raster CRS, the MUPP is in canvas units (e.g.
        # degrees) but crop is in raster units (e.g. meters). Convert by
        # measuring a small canvas-pixel offset in raster CRS.
        if self._canvas_to_raster_xform is not None:
            canvas_center = canvas.center()
            cx, cy = canvas_center.x(), canvas_center.y()
            p1 = self._canvas_to_raster_xform.transform(QgsPointXY(cx, cy))
            p2 = self._canvas_to_raster_xform.transform(
                QgsPointXY(cx + canvas_mupp, cy))
            raster_mupp = math.sqrt(
                (p2.x() - p1.x()) ** 2 + (p2.y() - p1.y()) ** 2)
        else:
            raster_mupp = canvas_mupp
        self._current_crop_canvas_mupp = canvas_mupp
        actual_mupp = mupp_override or raster_mupp
        self._current_crop_actual_mupp = actual_mupp
        return actual_mupp

    def _extract_crop_only(self, center_point, mupp_override, quiet=False):
        """Extract one crop on the GUI thread (no predictor access).

        Sets the per-crop mupp/scale state and returns (image_np, crop_info),
        or (None, None) after surfacing the crop error (headless_error in
        headless mode, an error report otherwise; ``quiet=True`` logs only,
        for the speculative session-start prewarm which must never pop a
        dialog). Split out of the old _do_extract_and_encode so the same
        extraction feeds both the sync (headless/recovery) and the async
        (interactive) encode paths.
        """
        from ...core.feature_encoder import extract_crop_from_online_layer, extract_crop_from_raster

        raster_pt_x = center_point.x()
        raster_pt_y = center_point.y()

        if self._is_online_layer:
            actual_mupp = self._online_crop_mupp(mupp_override)
            image_np, crop_info, error, error_code_from_crop = extract_crop_from_online_layer(
                self._current_layer, raster_pt_x, raster_pt_y,
                actual_mupp, crop_size=1024
            )
        elif not self._current_raster_path:
            if quiet:
                return None, None
            if self._headless:
                self._headless_error = tr("No raster file path available. Please restart segmentation.")
                return None, None
            show_error_report(
                self.iface.mainWindow(),
                tr("Crop Error"),
                tr("No raster file path available. Please restart segmentation."),
                error_code="crop_error_no_path",
            )
            return None, None
        else:
            layer_crs_wkt = None
            layer_extent = None
            try:
                if self._current_layer.crs().isValid():
                    layer_crs_wkt = self._current_layer.crs().toWkt()
                ext = self._current_layer.extent()
                if ext and not ext.isEmpty():
                    layer_extent = (ext.xMinimum(), ext.yMinimum(),
                                    ext.xMaximum(), ext.yMaximum())
            except RuntimeError:
                pass

            scale_factor = mupp_override or 1.0
            self._current_crop_scale_factor = scale_factor
            self._current_crop_canvas_mupp = self.iface.mapCanvas().mapUnitsPerPixel()
            image_np, crop_info, error, error_code_from_crop = extract_crop_from_raster(
                self._current_raster_path, raster_pt_x, raster_pt_y,
                crop_size=1024,
                layer_crs_wkt=layer_crs_wkt,
                layer_extent=layer_extent,
                scale_factor=scale_factor,
            )

        if error:
            QgsMessageLog.logMessage(
                f"Crop extraction failed: {error}",
                "AI Segmentation", level=Qgis.MessageLevel.Critical
            )
            if quiet:
                return None, None
            if error_code_from_crop == "crop_error_rasterio_unavailable":
                # The panel said ready but the in-process rasterio import
                # failed: the package is present-but-broken in the venv
                # (antivirus quarantine, interrupted install). An error report
                # dead-ends the user, so purge the broken artifacts (pip would
                # otherwise consider rasterio satisfied and skip it) and route
                # to the same one-click repair as a broken runtime. (#64)
                try:
                    from ...core.telemetry import track_plugin_error
                    track_plugin_error(
                        stage="segment",
                        error_code="crop_error_rasterio_unavailable",
                        message=error,
                        module="manual_crops",
                    )
                except Exception:
                    pass  # nosec B110
                from ...core.venv_manager import purge_package_from_venv
                purge_package_from_venv("rasterio")
                self._recover_broken_venv(error)
                return None, None
            if self._headless:
                self._headless_error = error
                return None, None
            show_error_report(
                self.iface.mainWindow(),
                tr("Crop Error"),
                error,
                error_code=error_code_from_crop or "crop_error_unknown",
            )
            return None, None

        return image_np, crop_info

    def _encode_crop_blocking(self, center_point, mupp_override) -> bool:
        """Synchronous crop extract + SAM encode on the CALLING thread.

        Used by the headless/MCP path (which must block) and by the rare
        _run_prediction recovery re-encode (the worker died, so there is no
        encode in flight and a main-thread set_image is transport-safe). Blocks
        ~3-8s. Returns True on success / False on error. This is the pre-PERF-01
        behaviour, kept intact for these two callers.
        """
        if self._encoding_in_progress:
            return False
        self._encoding_in_progress = True
        if not self._headless:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            QApplication.processEvents()
        try:
            image_np, crop_info = self._extract_crop_only(center_point, mupp_override)
            if image_np is None:
                return False
            try:
                self.predictor.set_image(image_np)
            except Exception as e:
                return self._handle_encode_error(str(e))
            self._apply_encode_result_ok(crop_info)
            return True
        finally:
            # Always drop the re-entrancy latch: any raise between the guard
            # above and here (e.g. a QgsCsException from the online transform
            # math) used to leave it True forever, dead-ending Manual mode.
            self._encoding_in_progress = False
            if not self._headless:
                QApplication.restoreOverrideCursor()

    def _handle_encode_error(self, err_str: str) -> bool:
        """Classify + surface a set_image failure on the MAIN thread. Shared by
        the sync (_encode_crop_blocking) and async (_on_manual_encode_done)
        paths so both trigger the exact same recovery. Always returns False so a
        sync caller aborts the current encode."""
        QgsMessageLog.logMessage(
            f"Image encoding failed: {err_str}",
            "AI Segmentation", level=Qgis.MessageLevel.Critical
        )
        # A corrupt model checkpoint makes every encode fail forever with a
        # raw PyTorch traceback. Detect that, delete the bad file and
        # re-download it (the download path re-verifies the hash). (#65)
        from ...core.checkpoint_manager import (
            delete_checkpoint,
            is_corrupt_checkpoint_error,
        )
        if is_corrupt_checkpoint_error(err_str):
            return self._recover_corrupt_checkpoint(delete_checkpoint())
        # A venv whose base Python was deleted or corrupted fails every
        # worker spawn with "No Python at ..." while the panel still
        # says ready. Route it to the one-click repair. (#64)
        from ...core.venv_manager import venv_needs_repair
        if venv_needs_repair():
            return self._recover_broken_venv(err_str)
        if self._headless:
            self._headless_error = err_str
            return False
        show_error_report(
            self.iface.mainWindow(),
            tr("Encoding Error"),
            err_str,
            error_code="encoding_error",
        )
        return False

    def _apply_encode_result_ok(self, crop_info) -> None:
        """Commit a successful encode on the MAIN thread: adopt the new crop,
        recompute the auto min-area and restore canvas focus. Shared by the sync
        and async success paths."""
        self._current_crop_info = crop_info
        # Auto-compute min_area based on current crop scale. The value is not
        # surfaced in the UI anymore - it is applied transparently.
        self._refine_min_area = self._compute_auto_min_area()
        self._safe_restore_canvas_focus()
        QgsMessageLog.logMessage(
            "Encoded crop: bounds={}, shape={}, auto_min_area={}".format(
                crop_info["bounds"], crop_info["img_shape"],
                self._refine_min_area),
            "AI Segmentation", level=Qgis.MessageLevel.Info
        )

    # ---- Async encode worker lifecycle --------------------------------------

    def _start_manual_encode(self, image_np, crop_info, on_encoded,
                             show_busy: bool = True) -> None:
        """Start the off-thread SAM encode for an already-extracted crop.

        Takes the transport lock (`_encoding_in_progress`), shows the busy
        cursor, and launches a SetImageWorker. The worker is parked up-front so
        its QThread C++ object can never be GC-dropped mid-run if the plugin
        instance goes away (unload cannot reach it: the controller is not a
        QObject and unload is frozen); park releases it on `finished`.

        ``show_busy=False`` runs the encode with NO cursor change: the
        speculative selection prewarm must be invisible (a busy cursor on
        every selection click would read as the selection freezing). The
        completion restores the cursor only when one was set (the flag rides
        in _pending_encode so set/restore always pair up).
        """
        self._ensure_manual_encode_state()
        from ..background_workers import SetImageWorker
        from .shared import park_orphaned_worker

        self._manual_encode_gen += 1
        gen = self._manual_encode_gen
        self._pending_encode = {
            "crop_info": crop_info,
            "tail": on_encoded,
            # Identity guard: a swapped/cleared predictor (env reset, reload)
            # means the completion belongs to a torn-down session.
            "predictor": self.predictor,
            "gen": gen,
            "cursor": bool(show_busy),
        }
        # Mirror of the cursor flag that survives _invalidate_manual_encode
        # (which nulls _pending_encode): the completion must never restore a
        # cursor this encode did not set, even after an invalidation.
        self._encode_cursor_set = bool(show_busy)
        self._encoding_in_progress = True
        # The busy cursor is the required affordance for the 3-8s wait (no
        # Manual status line is exposed on the dock without touching it). It is
        # restored in the completion callback for every outcome.
        if show_busy:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)

        worker = SetImageWorker(self.predictor, image_np, gen)
        self._manual_encode_worker = worker
        worker.done.connect(self._on_manual_encode_done)
        park_orphaned_worker(worker)
        worker.start()

    def _on_manual_encode_done(self, gen: int, ok: bool, err: str) -> None:
        """Main-thread completion of an off-thread encode (queued via the
        SetImageWorker.done signal, exactly like PredictorLoadWorker so it lands
        on the GUI thread). Commits the crop + runs the crop-transition tail +
        replays the user's click on success; drops everything on a stale/torn
        completion; reproduces the sync error recovery on failure."""
        self._ensure_manual_encode_state()
        pending = self._pending_encode

        # Teardown detection that does NOT rely on a generation bump: unload
        # (dock gone) and env-reset/reload (predictor cleared or swapped) cannot
        # bump the counter, so detect them by state and only clean up.
        torn_down = (
            self.dock_widget is None
            or self.predictor is None  # noqa: W503
            or (pending is not None  # noqa: W503
                and self.predictor is not pending.get("predictor"))  # noqa: W503
        )

        # Release the pipe lock, cursor and worker ref for every outcome. Only
        # restore the cursor when this encode SET one (the speculative prewarm
        # runs cursor-less; an unpaired restore would pop someone else's).
        self._manual_encode_worker = None
        self._pending_encode = None
        self._encoding_in_progress = False
        cursor_was_set = (pending.get("cursor", True) if pending is not None
                          else getattr(self, "_encode_cursor_set", True))
        self._encode_cursor_set = True
        if cursor_was_set:
            try:
                QApplication.restoreOverrideCursor()
            except Exception:  # nosec B110 -- cursor restore is best-effort
                pass

        if torn_down:
            return

        if gen != self._manual_encode_gen:
            # Invalidated by an in-session teardown (reset/mode switch/layer
            # removal). Drop this crop; honor a fresh remembered click (a new
            # session may have started) by re-driving it, which self-heals to a
            # new encode if needed.
            if self._pending_manual_click is not None:
                self._replay_pending_manual_click()
            return

        if not ok:
            # Same recovery + surfacing as the old synchronous except branch,
            # on the main thread. The deferred click cannot be applied: drop it.
            self._handle_encode_error(err)
            self._discard_pending_manual_click()
            return

        # Success: adopt the new crop, run the crop-transition tail (freeze /
        # transfer / refine seed), then replay the user's click (last wins).
        if pending is not None and pending.get("crop_info") is not None:
            self._apply_encode_result_ok(pending["crop_info"])
        tail = pending.get("tail") if pending is not None else None
        if tail is not None:
            try:
                tail()
            except Exception as e:  # noqa: BLE001 - never wedge the flow on the tail
                QgsMessageLog.logMessage(
                    f"Manual encode continuation failed: {e}",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
        if self._pending_manual_click is not None:
            self._replay_pending_manual_click()

    # ---- Remembered-click (defer + replay) ----------------------------------

    def _remember_pending_manual_click(self, polarity: str, canvas_point) -> None:
        """Record the click that triggered (or arrived during) an encode so it
        is replayed when the encode finishes. Last click wins: this replaces any
        previously remembered click. The map tool added a marker for the
        physical click; drop it here (the replay re-adds it) so a deferred click
        never leaves an orphan marker on the canvas."""
        self._ensure_manual_encode_state()
        if self.map_tool:
            self.map_tool.remove_last_marker()
        self._pending_manual_click = {"polarity": polarity, "canvas_point": canvas_point}

    def _discard_pending_manual_click(self) -> None:
        """Forget a remembered click without replaying it (teardown / encode
        error). Its marker was already removed when it was deferred."""
        self._ensure_manual_encode_state()
        self._pending_manual_click = None

    def _replay_pending_manual_click(self) -> None:
        """Re-drive the remembered click through the normal click handler now
        that the pipe is free. Re-adds the marker (removed on defer) so a
        successful replay keeps it and a failing one removes it via the
        handler's own last-marker rollback. The handler re-checks crop status,
        so a click that no longer fits the fresh crop self-heals to a new
        encode."""
        self._ensure_manual_encode_state()
        pending = self._pending_manual_click
        if not pending:
            return
        self._pending_manual_click = None
        point = pending["canvas_point"]
        is_positive = pending["polarity"] == "positive"
        if self.map_tool:
            self.map_tool.add_marker(point, is_positive=is_positive)
        if is_positive:
            self._on_positive_click(point)
        else:
            self._on_negative_click(point)

    # ---- Async online crop fetch (interactive) ------------------------------
    # An online-layer crop reads tiles off the network and retries with
    # progressive back-off (up to ~18s). Doing that on the GUI thread froze
    # QGIS, so the interactive path drives the OnlineCropFetcher's discrete
    # attempt steps with QTimer.singleShot: the retry waits happen off the
    # event loop, the busy cursor + transport lock are held across the whole
    # fetch, and a click that arrives meanwhile defers (via
    # _encoding_in_progress) and replays once the crop is ready. On success the
    # crop hands off to the same async SAM encode as the file-based path.

    def _begin_online_crop_fetch(self, center_point, mupp_override,
                                 on_encoded) -> bool:
        """Start the asynchronous online-tile crop fetch. Returns True when the
        fetch STARTED (the encode follows on completion), False when it could
        not start (provider unavailable / setup error, already surfaced)."""
        from ...core.feature_encoder import OnlineCropFetcher

        actual_mupp = self._online_crop_mupp(mupp_override)
        fetcher = OnlineCropFetcher(
            self._current_layer, center_point.x(), center_point.y(),
            actual_mupp, crop_size=1024)
        if fetcher.error is not None:
            self._surface_online_crop_error(fetcher.error, fetcher.error_code)
            return False
        try:
            fetcher.begin()
        except Exception as e:  # noqa: BLE001 - never leave the provider mutated
            fetcher.restore()
            self._surface_online_crop_error(str(e), "crop_error_online_exception")
            return False

        self._ensure_manual_encode_state()
        self._manual_encode_gen += 1
        gen = self._manual_encode_gen
        self._online_fetch = {
            "fetcher": fetcher,
            "gen": gen,
            "on_encoded": on_encoded,
            "cursor": True,
        }
        # The fetch owns the transport lock + busy cursor for its whole life. A
        # click meanwhile defers via _encoding_in_progress and replays after the
        # crop is ready. _encode_cursor_set mirrors the cursor so handoff
        # gestures treat the fetch like a foreground encode.
        self._encoding_in_progress = True
        self._encode_cursor_set = True
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        QApplication.processEvents()
        self._step_online_crop_fetch()
        return True

    def _step_online_crop_fetch(self) -> None:
        """Run one online-fetch attempt on the GUI thread, then finish (on
        success/exhaustion) or schedule the next attempt after its back-off via
        QTimer.singleShot. A stale step (session torn down / superseded) is
        dropped: its lock and cursor were already released by the teardown."""
        from qgis.PyQt.QtCore import QTimer

        self._ensure_manual_encode_state()
        if self.dock_widget is None:
            # Unloaded while a fetch was scheduled: revert the provider state
            # and pop the cursor (unload does not run the encode invalidation).
            self._release_online_fetch()
            return
        fetch = self._online_fetch
        if fetch is None or fetch.get("gen") != self._manual_encode_gen:
            return  # superseded; teardown already released the lock/cursor
        try:
            action, delay = fetch["fetcher"].step()
        except Exception as e:  # noqa: BLE001
            self._fail_online_fetch(str(e), "crop_error_online_exception")
            return
        if action in ("stabilized", "exhausted"):
            self._complete_online_crop_fetch()
            return
        # ("refetch", 0.5) or ("retry", delay): wait off the event loop, then
        # take the next attempt.
        QTimer.singleShot(max(0, int(delay * 1000)), self._step_online_crop_fetch)

    def _complete_online_crop_fetch(self) -> None:
        """Fetch stabilized: read the bands, revert the provider state, then
        either surface the fetch error or hand the crop off to the async SAM
        encode (which reuses the busy cursor and the transport lock)."""
        fetch = self._online_fetch
        if fetch is None:
            return
        fetcher = fetch["fetcher"]
        on_encoded = fetch.get("on_encoded")
        try:
            image_np, crop_info, error, error_code = fetcher.finish()
        except Exception as e:  # noqa: BLE001
            self._fail_online_fetch(str(e), "crop_error_online_exception")
            return
        finally:
            # The provider fetch is done; revert the user's live resampling
            # state before anything else reads the layer.
            try:
                fetcher.restore()
            except Exception:  # nosec B110
                pass
        if error:
            # Provider state already reverted above.
            self._fail_online_fetch(error, error_code, restore_provider=False)
            return
        # Success: drop the fetch bookkeeping and pop the fetch's busy cursor,
        # then start the async encode (it re-asserts the lock + pushes its own
        # cursor, restored on the encode completion). The lock stays held across
        # the handoff so a click cannot race in between.
        self._online_fetch = None
        if fetch.get("cursor"):
            try:
                QApplication.restoreOverrideCursor()
            except Exception:  # nosec B110
                pass
        self._start_manual_encode(image_np, crop_info, on_encoded)

    def _fail_online_fetch(self, error, error_code,
                           restore_provider: bool = True) -> None:
        """Online crop fetch failed: release the lock + cursor (+ provider state
        unless already reverted), surface the same error the sync path would,
        and drop the deferred click (it cannot be honored)."""
        self._release_online_fetch(restore_provider=restore_provider)
        self._surface_online_crop_error(error, error_code)
        self._discard_pending_manual_click()

    def _release_online_fetch(self, restore_provider: bool = True) -> None:
        """Tear down the in-flight online crop fetch: revert the provider's live
        resampling state, drop the transport lock, pop the busy cursor.
        Idempotent; safe when no fetch is active. A live encode WORKER never
        owns _online_fetch (it starts only after a fetch clears it), so clearing
        the lock here can never clobber a worker's pipe ownership."""
        self._ensure_manual_encode_state()
        fetch = self._online_fetch
        self._online_fetch = None
        if fetch is None:
            return
        if restore_provider:
            try:
                fetch["fetcher"].restore()
            except Exception:  # nosec B110
                pass
        self._encoding_in_progress = False
        if fetch.get("cursor"):
            try:
                QApplication.restoreOverrideCursor()
            except Exception:  # nosec B110
                pass

    def _surface_online_crop_error(self, error, error_code) -> None:
        """Log + report an online crop-fetch failure on the GUI thread (mirrors
        the interactive branch of _extract_crop_only; online never returns the
        rasterio-unavailable code, so no venv recovery is wired here)."""
        QgsMessageLog.logMessage(
            f"Crop extraction failed: {error}",
            "AI Segmentation", level=Qgis.MessageLevel.Critical
        )
        show_error_report(
            self.iface.mainWindow(),
            tr("Crop Error"),
            error,
            error_code=error_code or "crop_error_unknown",
        )

    def _prewarm_manual_encode(self) -> None:
        """Pre-encode the visible view at session start (first-click latency).

        warm_up() only pre-starts the SAM subprocess; the very first click of
        a session still paid the model-load tail + the first in-process
        rasterio import + the first ~3-8s encode, which read as "my first
        click searches for 5 seconds". Encoding a crop centered on the canvas
        as soon as the session starts absorbs all three while the user aims:
        a first click near the view center predicts instantly, any other
        click self-heals through the normal re-encode with the model already
        warm. Speculative and SILENT: extraction failures only log (quiet),
        an encode failure surfaces once through the shared error handler, and
        the whole thing steps aside for the Refine handoff, which seeds its
        own encode and must never find the pipe lock taken."""
        self._ensure_manual_encode_state()
        if self._headless or self._encoding_in_progress or self.predictor is None or self._refine_handoff_active:
            return
        try:
            canvas = self.iface.mapCanvas()
            # Fired via singleShot: the session may already be over (fast
            # Stop, layer removed). The session's map tool being active is
            # the cheapest reliable "still segmenting" signal.
            if canvas.mapTool() is not self.map_tool:
                return
            center = self._transform_to_raster_crs(QgsPointXY(canvas.center()))
        except Exception:  # noqa: BLE001 - prewarm is best-effort, never blocks the session
            return
        if not self._is_online_layer and not self._is_point_in_raster_extent(center):
            return  # view is off the raster; the first real click drives the encode
        image_np, crop_info = self._extract_crop_only(
            center, self._compute_initial_scale_factor(), quiet=True)
        if image_np is None:
            return
        QgsMessageLog.logMessage(
            "Prewarming first crop at view center",
            "AI Segmentation", level=Qgis.MessageLevel.Info
        )
        self._start_manual_encode(image_np, crop_info, None)

    def _invalidate_manual_encode(self) -> None:
        """Invalidate any pending encode completion so it never touches
        torn-down session state (session end / layer removal / mode switch).
        Bumps the generation (mirrors the _auto_finalize_gen pattern) and drops
        the pending crop + remembered click.

        `_encoding_in_progress` is deliberately NOT cleared here when a WORKER
        owns the predictor pipe: the lock must stay until its own completion
        fires (transport safety), which also restores the busy cursor. An
        in-flight online crop FETCH is different: no worker will fire to release
        it, so _release_online_fetch clears its lock + cursor + provider state
        (fetch and worker are mutually exclusive). A new encode therefore cannot
        race the draining one."""
        self._ensure_manual_encode_state()
        self._manual_encode_gen += 1
        self._pending_encode = None
        self._discard_pending_manual_click()
        self._release_online_fetch()
        # The handoff crop-spec memo describes an encode of the session being
        # torn down; a new session must never skip its own encode over it.
        self._handoff_crop_spec = None

    def _current_crop_covers_bbox(self, bb) -> bool:
        """True when the CURRENT encoded crop fully frames the given raster-CRS
        bounding box. Used by the handoff open to skip a re-encode when the
        speculative selection prewarm already encoded this object's crop."""
        info = self._current_crop_info
        if info is None:
            return False
        try:
            minx, miny, maxx, maxy = info["bounds"]
            return (bb.xMinimum() >= minx and bb.xMaximum() <= maxx
                    and bb.yMinimum() >= miny and bb.yMaximum() <= maxy)  # noqa: W503
        except (KeyError, TypeError, AttributeError):
            return False

    def _freeze_active_crop(self, crop_info_override=None):
        """Freeze the current active crop's mask as a geographic polygon.

        The polygon is stored in raster CRS (same as save/export) and added
        to _frozen_sessions for composite display.

        Args:
            crop_info_override: If provided, use this instead of
                self._current_crop_info (needed when the caller has already
                overwritten _current_crop_info with a new crop).
        """
        if self.current_mask is None or self.current_transform_info is None:
            # A handoff edit before its first editing click has no mask, only
            # the imported display geometry. Freeze THAT, so a click outside
            # the crop cannot drop the open object: the new crop's prediction
            # composites and unions with it exactly like any frozen session.
            base = self._unfrozen_display_polygon
            if base is not None and not base.isEmpty():
                self._frozen_sessions.append(
                    FrozenCropSession(polygon=QgsGeometry(base)))
                self._unfrozen_display_polygon = None
            return
        try:
            # Freeze exactly what the preview shows: the shared refine tail
            # (refinement, simplification, smoothing, size window). Freezing
            # the raw mask made the polygon visibly "jump" the moment the
            # user clicked elsewhere.
            combined = self._refined_active_mask_geometry()
            if combined is not None and not combined.isEmpty():
                session = FrozenCropSession(
                    polygon=combined,
                    points_positive=list(self._active_crop_points_positive),
                    points_negative=list(self._active_crop_points_negative),
                    crop_info=crop_info_override if crop_info_override is not None else self._current_crop_info,
                )
                self._frozen_sessions.append(session)
                QgsMessageLog.logMessage(
                    f"Froze crop session #{len(self._frozen_sessions)} "
                    f"with {len(session.points_positive) + len(session.points_negative)} points",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to freeze active crop: {str(e)}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)

        # Reset active crop tracking (mask/low_res cleared by caller)
        self._active_crop_points_positive = []
        self._active_crop_points_negative = []
        self._mask_state_history = []

    def _handle_reencode(self, crop_status, raster_pt):
        """Re-encode the crop for a click that fell outside the current one.

        HEADLESS/MCP: fully synchronous, returns True on success / False on
        error (the blocking contract the MCP detect() path relies on).

        INTERACTIVE: kicks off the async encode and returns True when it was
        STARTED (the crop-transition + prediction run later from the completion
        callback), False when crop extraction failed synchronously. Interactive
        click handlers do NOT run a prediction after this returns; the remembered
        click is replayed once the encode completes.
        """
        if self._headless:
            return self._handle_reencode_sync(crop_status, raster_pt)
        return self._begin_async_reencode(crop_status, raster_pt)

    def _handle_reencode_sync(self, crop_status, raster_pt):
        """Synchronous re-encode (headless/MCP). Encodes, then does the
        crop-transition work inline. Returns True on success."""
        if crop_status == "no_crop":
            self.current_low_res_mask = None
            # After an unfreeze the session's points are restored but no crop
            # is encoded. Fit ALL points in the new crop: centering on the
            # new click alone can leave the others outside the image, which
            # sends garbage coordinates to SAM.
            all_pts = [
                (p[0], p[1]) for p in
                self.prompts.positive_points + self.prompts.negative_points
            ]
            if (raster_pt.x(), raster_pt.y()) not in all_pts:
                all_pts.append((raster_pt.x(), raster_pt.y()))
            if len(all_pts) > 1:
                center_x, center_y, mupp_or_scale = (
                    self._compute_crop_center_and_mupp(all_pts))
                if not self._extract_and_encode_crop(
                        QgsPointXY(center_x, center_y),
                        mupp_override=mupp_or_scale):
                    return False
            else:
                initial_scale = self._compute_initial_scale_factor()
                if not self._extract_and_encode_crop(
                        raster_pt, mupp_override=initial_scale):
                    return False
            self._invalidate_history_logits()
            return True

        if crop_status == "outside_bounds":
            old_crop_info = self._current_crop_info
            old_history = list(self._mask_state_history)
            initial_scale = self._compute_initial_scale_factor()
            if not self._extract_and_encode_crop(
                    raster_pt, mupp_override=initial_scale):
                return False

            self._freeze_active_crop(crop_info_override=old_crop_info)
            # Clear stale prompts from old crop (#11, #35). The click handler
            # that called us will re-add the new point with correct polarity.
            self.prompts.clear()
            self._active_crop_points_positive = []
            self._active_crop_points_negative = []
            # Preserve history so the empty-mask rollback path still works.
            self._mask_state_history = old_history
            self._invalidate_history_logits()
            self.current_mask = None
            self.current_low_res_mask = None
            return True

        # zoom_changed: re-encode same crop at new resolution, keep all points
        old_crop_info = self._current_crop_info
        old_mask = self.current_mask

        all_pts = [
            (p[0], p[1]) for p in
            self.prompts.positive_points + self.prompts.negative_points
        ]
        if len(all_pts) > 1:
            center_x, center_y, mupp_or_scale = (
                self._compute_crop_center_and_mupp(all_pts))
            new_center = QgsPointXY(center_x, center_y)
        else:
            new_center = raster_pt
            mupp_or_scale = self._compute_initial_scale_factor()
        self.current_low_res_mask = None
        if not self._extract_and_encode_crop(
                new_center, mupp_override=mupp_or_scale):
            return False
        self._invalidate_history_logits()

        # Transfer previous mask as context to the new crop
        if old_mask is not None and old_crop_info is not None:
            transferred = self._build_mask_input_from_previous(
                old_mask.astype(float),
                old_crop_info["bounds"],
                old_crop_info["img_shape"],
                self._current_crop_info["bounds"],
                self._current_crop_info["img_shape"],
            )
            if transferred is not None:
                self.current_low_res_mask = transferred

        return True

    def _begin_async_reencode(self, crop_status, raster_pt):
        """Interactive re-encode: extract the crop on the GUI thread, encode on
        a worker, and defer the crop-transition work to the completion (as the
        ``on_encoded`` tail). The tail mirrors _handle_reencode_sync's per-status
        post-encode block; the remembered click is replayed after the tail. The
        triggering click is NOT registered here (the replay registers it once),
        so it stays undone in prompts/history until the crop is ready. Returns
        True when the encode was started, False when extraction failed."""
        if crop_status == "no_crop":
            self.current_low_res_mask = None
            # Fit ALL restored points plus this click in the new crop.
            all_pts = [
                (p[0], p[1]) for p in
                self.prompts.positive_points + self.prompts.negative_points
            ]
            if (raster_pt.x(), raster_pt.y()) not in all_pts:
                all_pts.append((raster_pt.x(), raster_pt.y()))
            if len(all_pts) > 1:
                center_x, center_y, mupp_or_scale = (
                    self._compute_crop_center_and_mupp(all_pts))
                center = QgsPointXY(center_x, center_y)
            else:
                center = raster_pt
                mupp_or_scale = self._compute_initial_scale_factor()

            def _tail():
                self._invalidate_history_logits()

            return self._extract_and_encode_crop(
                center, mupp_override=mupp_or_scale, on_encoded=_tail)

        if crop_status == "outside_bounds":
            old_crop_info = self._current_crop_info
            old_history = list(self._mask_state_history)
            initial_scale = self._compute_initial_scale_factor()

            def _tail():
                self._freeze_active_crop(crop_info_override=old_crop_info)
                # Clear stale prompts from the old crop (#11, #35). The replayed
                # click re-adds the new point with correct polarity.
                self.prompts.clear()
                self._active_crop_points_positive = []
                self._active_crop_points_negative = []
                # Preserve history so the empty-mask rollback path still works.
                self._mask_state_history = old_history
                self._invalidate_history_logits()
                self.current_mask = None
                self.current_low_res_mask = None

            return self._extract_and_encode_crop(
                raster_pt, mupp_override=initial_scale, on_encoded=_tail)

        # zoom_changed: re-encode same crop at new resolution, keep all points.
        old_crop_info = self._current_crop_info
        old_mask = self.current_mask
        all_pts = [
            (p[0], p[1]) for p in
            self.prompts.positive_points + self.prompts.negative_points
        ]
        # Include this click in the centering (the sync path had it registered
        # in prompts already; here it is not, so add it explicitly).
        if (raster_pt.x(), raster_pt.y()) not in all_pts:
            all_pts.append((raster_pt.x(), raster_pt.y()))
        if len(all_pts) > 1:
            center_x, center_y, mupp_or_scale = (
                self._compute_crop_center_and_mupp(all_pts))
            new_center = QgsPointXY(center_x, center_y)
        else:
            new_center = raster_pt
            mupp_or_scale = self._compute_initial_scale_factor()
        self.current_low_res_mask = None

        def _tail():
            self._invalidate_history_logits()
            # Transfer the previous mask as context to the new crop.
            if old_mask is not None and old_crop_info is not None:
                transferred = self._build_mask_input_from_previous(
                    old_mask.astype(float),
                    old_crop_info["bounds"],
                    old_crop_info["img_shape"],
                    self._current_crop_info["bounds"],
                    self._current_crop_info["img_shape"],
                )
                if transferred is not None:
                    self.current_low_res_mask = transferred

        return self._extract_and_encode_crop(
            new_center, mupp_override=mupp_or_scale, on_encoded=_tail)
