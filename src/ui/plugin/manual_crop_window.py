"""Where the square crop that answers a click is cut from.

One question, asked from several places: given the layer, the canvas and
whatever shape is already in hand, which ground window should the next crop
cover, and at what resolution? Everything that decides that lives here;
`manual_crops.py` next door reads the pixels and encodes them.

Two rules run through the whole file, both in `core/crop_window.py`:

- never blow a crop up. A window carrying fewer source pixels than the crop it
  fills is stretched to fit, and stretching invents no detail. Zoomed in past
  the source's own pixels the window takes more GROUND instead, and a source
  too coarse for the crop sends what it holds.
- do not let the window cut the object. Once a shape is known and still whole,
  the window is chosen to hold all of it, because the answer has to come back
  covering all of it.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); methods here are
plain mixin members and state lives on the plugin instance.
"""
from __future__ import annotations

import math

from qgis.core import QgsPointXY


class ManualCropWindowMixin:
    """Picks the ground window and resolution each Manual crop is read at."""

    def _crop_resolution_would_change(self) -> bool:
        """Would re-reading at the current zoom give the crop a DIFFERENT
        resolution than the one in hand?

        The zoom test above watches the canvas, and the canvas can move without
        moving the crop: past the source's own pixels the window is held at
        native resolution, so zooming further asks for exactly the crop already
        encoded. Re-reading it costs seconds and drops the mask, for the same
        pixels. Unknown either side answers yes, which is the behaviour this
        guard was added under.

        A session holding a shape still whole is the same story one level up:
        its window is chosen to hold that shape, so the zoom cannot move it and
        a re-read would fetch the crop already encoded.
        """
        if self._untouched_shape_window_is_held():
            return False
        try:
            if self._is_online_layer:
                held = self._current_crop_actual_mupp
                if not held or held <= 0:
                    return True
                _canvas_mupp, fresh = self._online_crop_mupp_now(None)
                return round(float(fresh), 9) != round(float(held), 9)
            held = self._current_crop_scale_factor
            fresh = self._compute_initial_scale_factor()
            if held is None or fresh is None:
                return True
            return round(float(fresh), 9) != round(float(held), 9)
        except (RuntimeError, AttributeError, TypeError, ValueError, ZeroDivisionError):
            return True

    def _untouched_shape_window_is_held(self) -> bool:
        """Is the crop in hand already the window a still-whole shape asks for?

        False whenever there is no such shape, no crop, or the two windows
        differ, so the caller's own test decides. Never raises: a click path
        must not fail on a comparison.
        """
        info = self._current_crop_info
        if info is None:
            return False
        try:
            from ...core.crop_window import crop_window_key

            minx, miny, maxx, maxy = info["bounds"]
            centre = QgsPointXY((minx + maxx) / 2.0, (miny + maxy) / 2.0)
            whole = self._untouched_shape_crop_window(centre)
            if whole is None:
                return False
            point, scale = whole
            asks = crop_window_key(point.x(), point.y(), scale)
            return asks == getattr(self, "_encoded_crop_window", None)
        except Exception:  # noqa: BLE001 -- a comparison must not cost a click
            return False

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

    def _compute_initial_scale_factor(self):
        """Compute initial scale_factor from canvas zoom for file-based rasters.

        For high-res imagery where the user is zoomed out, the crop should cover
        a proportionally larger geographic area instead of just 1024 native pixels.

        Zoomed IN past the raster's own pixels the answer is NOT a smaller
        window: `crop_window.scale_at_least_native` holds the window at native
        resolution and lets it cover more ground instead.

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

        from ...core.crop_window import scale_at_least_native

        ratio = canvas_mupp_raster_crs / native_pixel_size
        return min(scale_at_least_native(ratio), 8.0)

    # ------------------------------------------------------------------
    # Crop encoding (PERF-01): async on the interactive path, sync headless.
    #
    # set_image() writes a ~4MB crop over stdin to the prediction subprocess and blocks
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
    # - _encode_lock_gen: generation that owns the transport lock (None = free);
    #   a completion from any other generation must leave every field alone.
    # - _pending_encode: {crop_info, tail, predictor, gen} committed on success.
    # - _pending_manual_click: {polarity, canvas_point} to replay on completion
    #   (last click wins; replaced by any click during the encode).
    # ------------------------------------------------------------------

    def _crop_window_key_for(self, center_point, mupp_override):
        """Window identity of the crop a read is about to fetch, comparable to
        what the fix session computes for an object. None when the centre is
        unusable, which keeps an unknown window from matching another one."""
        from ...core.crop_window import crop_window_key
        try:
            return crop_window_key(center_point.x(), center_point.y(),
                                   mupp_override or 1.0)
        except (AttributeError, TypeError):
            return None

    def _grid_center_for_manual_click(self, raster_pt, scale):
        """Where to cut the crop that answers ONE Manual click:
        ``(center, scale)``.

        The click is moved onto the shared grid of `core/crop_window.py`, the
        same one the warm-ups use, so a second click in the same neighbourhood
        asks for a window the model already holds instead of one a few metres
        off it. One grid cell is a quarter of the crop, so the click still lands
        well inside the window it is answered from, edge clamping included.

        The crop already in hand is preferred over a fresh cell, but ONLY while
        no crop is current: a click that fell outside the current crop is
        outside the window that crop was cut from, and handing that window back
        would read the same imagery and reject the click a second time.

        Online layers keep the click as given here and are snapped in
        `_begin_online_crop_fetch`, the one place their grid unit (ground per
        pixel, not a zoom-out factor on native pixels) is known.
        """
        if self._is_online_layer:
            return raster_pt, scale
        from ...core.crop_window import snap_center_to_grid, window_frames_bounds
        native = self._get_native_pixel_size()
        grid_scale = scale or 1.0
        held = getattr(self, "_encoded_crop_window", None)
        reuse_held = self._current_crop_info is None and held is not None
        reuse_held = reuse_held and round(float(held[2]), 6) == round(float(grid_scale), 6)
        if reuse_held and window_frames_bounds(
                held,
                (raster_pt.x(), raster_pt.y(), raster_pt.x(), raster_pt.y()),
                native):
            return QgsPointXY(held[0], held[1]), held[2]
        cx, cy = snap_center_to_grid(
            raster_pt.x(), raster_pt.y(), grid_scale, native)
        return QgsPointXY(cx, cy), scale

    def _untouched_shape_crop_window(self, raster_pt, extra_points=()):
        """The window an object still whole should be re-read in, or None.

        An edit session opens on a KNOWN shape and frames the crop to hold all
        of it, because the answer to that first click has to come back covering
        the whole thing. A re-read (the user zoomed, or clicked outside the
        crop) then picked its window from the click alone and threw that
        framing away, so a shape wider than one crop came back stopping at the
        crop edge.

        Only while the shape is untouched. Once a click has kept part of it,
        that part is held whole and unioned back whatever the window does, so
        the window's job is the NEW part and the click is the better centre.

        The click joins the bounds, and so does every point in ``extra_points``
        (``(x, y)`` in raster CRS), because a prompt point outside the crop
        sends garbage coordinates to the model.
        """
        if getattr(self, "_frozen_sessions", None) or self.current_mask is not None:
            return None
        try:
            shape = self._shape_in_progress_geometry()
        except (RuntimeError, AttributeError):
            return None
        if shape is None or shape.isEmpty():
            return None
        try:
            box = shape.boundingBox()
        except (RuntimeError, AttributeError):
            return None
        xs = [box.xMinimum(), box.xMaximum(), raster_pt.x()]
        ys = [box.yMinimum(), box.yMaximum(), raster_pt.y()]
        for point in extra_points or ():
            xs.append(point[0])
            ys.append(point[1])
        return self._crop_window_for_bounds(
            (min(xs), min(ys), max(xs), max(ys)))

    def _crop_window_for_bounds(self, bounds):
        """``(center, scale_or_mupp)`` framing one ground rectangle, in the unit
        the crop reader expects for this layer. The one place the file and
        online forms of that call are spelled out.

        The bounds are in the RASTER's CRS, so the floor under the window has
        to be too. The canvas measures its step in canvas units, which are the
        same thing only while the two CRSs agree; the read itself already
        converts, and this asks it for the same number rather than converting
        twice and differently.
        """
        from ...core.crop_window import crop_window_for_object

        held = (getattr(self, "_encoded_crop_window", None)
                if self._current_crop_info is None else None)
        if self._is_online_layer:
            try:
                _canvas_mupp, raster_mupp = self._online_crop_mupp_now(None)
            except Exception:  # noqa: BLE001 -- an unusable floor is checked below
                raster_mupp = 0.0
            cx, cy, scale = crop_window_for_object(
                bounds, 1.0, held_window=held,
                min_scale=raster_mupp,
                max_scale=float("inf"))
        else:
            cx, cy, scale = crop_window_for_object(
                bounds, self._get_native_pixel_size(), held_window=held)
        return QgsPointXY(cx, cy), scale

    def _manual_crop_window_for_points(self, points_geo):
        """The crop window a whole set of Manual prompt points is answered from:
        ``(center, scale_or_mupp)``.

        Same grid as the warm-ups and as the review's fix session, so a
        neighbourhood keeps ONE crop instead of one per gesture. Every point is
        checked to sit clear of the window's edges before the shared cell is
        accepted, which a click-centred crop gave for free and a shared one has
        to earn: a point outside the image sends garbage coordinates to the
        model. The third value is in the unit the crop reader expects, a
        zoom-out factor on native pixels for a file raster and ground units per
        pixel for an online layer.
        """
        xs = [p[0] for p in points_geo]
        ys = [p[1] for p in points_geo]
        return self._crop_window_for_bounds(
            (min(xs), min(ys), max(xs), max(ys)))

    def _online_native_ground_per_pixel(self) -> float:
        """Finest ground per pixel the current tiled source serves, in RASTER CRS
        units, or 0.0 when it cannot be told. Fails open on purpose: an unknown
        source is left exactly as it was."""
        try:
            native = self._online_native_mupp(self._current_layer)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return 0.0
        return float(native) if native and native > 0 else 0.0

    def _online_crop_mupp_now(self, mupp_override) -> tuple:
        """``(canvas mupp, crop ground per pixel)`` an online crop would use
        RIGHT NOW, stashing nothing. Split out of `_online_crop_mupp` so the
        zoom test can ask what a fresh read would give without writing the
        per-crop state the crop in hand still owns.

        The crop step is held at the source's own finest by
        `crop_window.ground_per_pixel_at_least_native`, for the reason that
        function gives: a canvas zoomed past what the tiles carry would
        otherwise buy interpolated pixels with ground the crop could have
        covered.
        """
        from ...core.crop_window import ground_per_pixel_at_least_native

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
        return canvas_mupp, ground_per_pixel_at_least_native(
            mupp_override or raster_mupp, self._online_native_ground_per_pixel())

    def _online_crop_mupp(self, mupp_override):
        """Canvas -> raster map-units-per-pixel for an online crop, stashing the
        per-crop mupp state. Shared by the sync (_extract_crop_only) and async
        (_begin_online_crop_fetch) online paths so both compute it identically.
        """
        canvas_mupp, actual_mupp = self._online_crop_mupp_now(mupp_override)
        self._current_crop_canvas_mupp = canvas_mupp
        self._current_crop_actual_mupp = actual_mupp
        return actual_mupp
