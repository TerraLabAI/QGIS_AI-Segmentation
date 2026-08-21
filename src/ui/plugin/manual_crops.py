"""Manual CRS transforms, crop extraction and encoding, re-encode decisions.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

import os
import threading

from qgis.core import (
    Qgis,
    QgsGeometry,
    QgsMessageLog,
    QgsPointXY,
)
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QApplication,
)

from ...core.i18n import tr
from ...core.prompt_manager import FrozenCropSession
from ..error_report_dialog import show_error_report

# Transport-lock watchdog: beat interval, and the ceiling past which a lock
# is considered stranded even if its owner still looks alive. set_image's own
# transport timeout is 180s, so a lock held past 240s means the pipe is dead.
ENCODE_WATCHDOG_INTERVAL_MS = 5000
ENCODE_LOCK_CEILING_S = 240.0


class CropReadWorker(QThread):
    """Runs one windowed raster read off the GUI thread.

    The SAM encode already ran on a worker, but the read that feeds it did
    not, so a click on a big or slow raster still froze QGIS before the encode
    started. This carries that read across.

    It takes ONLY plain data (a file path, numbers) and touches no QGIS or Qt
    object, so it is safe on a secondary thread and safe to let finish after
    its owner is gone. `done` carries the generation the read started with, so
    a completion the main thread no longer owns is dropped there. The result
    is the 4-tuple extract_crop_from_raster returns, unchanged: the main
    thread surfaces the error exactly as the synchronous path does.
    """

    done = pyqtSignal(int, object)  # (generation, (image, info, err, code))

    def __init__(self, args: dict, generation: int, parent=None):
        super().__init__(parent)
        self._args = dict(args)
        self._generation = generation

    def run(self):
        from ...core.feature_encoder import extract_crop_from_raster
        try:
            result = extract_crop_from_raster(**self._args)
        except Exception as e:  # noqa: BLE001 - a raise here must not kill the thread
            result = (None, None, str(e), "crop_error_unknown")
        self.done.emit(self._generation, result)


class DirectTileFetchWorker(QThread):
    """Downloads one online crop's tiles off the GUI thread.

    A crop whose tiles can be asked for by number comes down in one go, and
    that one go used to sit on the thread that draws the window, so the click
    that started it froze QGIS until the last tile landed. Only the download
    moves here. The provider read ladder that serves every other online layer
    stays where it was, because it drives a QGIS raster provider and that
    belongs to the thread that owns it.

    It takes ONLY the plain request the fetcher built (strings and numbers) and
    touches no QGIS or Qt object, so it is safe on a secondary thread and safe
    to let finish after its owner is gone. `done` carries the generation the
    fetch started with, so a completion the main thread no longer owns is
    dropped there. `cancel_check` is polled before the download starts and again
    between tiles, so a fetch superseded while its tiles are coming down stops
    paying for them.
    """

    done = pyqtSignal(int, object)  # (generation, (image, error_code, elapsed_ms))

    def __init__(self, request, generation: int, cancel_check=None, parent=None):
        super().__init__(parent)
        self._request = request
        self._generation = generation
        self._cancel_check = cancel_check

    def run(self):
        try:
            if self._cancel_check is not None and self._cancel_check():
                result = (None, "crop_error_online_cancelled", 0)
            else:
                result = self._fetch_tiles()
        except Exception as e:  # noqa: BLE001 - a raise here must not kill the thread
            result = (None, str(e), 0)
        self.done.emit(self._generation, result)

    def _fetch_tiles(self):
        """The download itself, with the cancel handed down to the tile loop."""
        from ...core.feature_encoder import run_direct_tile_fetch

        return run_direct_tile_fetch(self._request,
                                     cancel_check=self._cancel_check)


class ManualCropsMixin:
    """Manual CRS transforms, crop extraction and encoding, re-encode decisions."""

    def _transform_to_raster_crs(self, point):
        """Transform a QgsPointXY from canvas CRS to raster CRS.

        Returns the original point unchanged when both CRS are identical, and
        None when the point has no image in the raster CRS. QGIS raises for a
        point outside a projection's valid domain, which a user hits by simply
        clicking or hovering far from the raster; callers treat None as
        "outside the raster".
        """
        if self._canvas_to_raster_xform is not None:
            try:
                return self._canvas_to_raster_xform.transform(point)
            except Exception:  # noqa: BLE001 - outside the projection domain
                return None
        return point

    def _transform_geometry_to_canvas_crs(self, geometry):
        """Transform a QgsGeometry from raster CRS to canvas CRS (in-place).

        Does nothing when both CRS are identical.
        """
        if self._raster_to_canvas_xform is not None:
            geometry.transform(self._raster_to_canvas_xform)

    def _transform_to_canvas_crs(self, point):
        """Transform a QgsPointXY from raster CRS to canvas CRS.

        Returns the original point unchanged when both CRS are identical, and
        None when the point has no image in the canvas CRS (see
        _transform_to_raster_crs); callers then skip the marker.
        """
        if self._raster_to_canvas_xform is not None:
            try:
                return self._raster_to_canvas_xform.transform(point)
            except Exception:  # noqa: BLE001 - outside the projection domain
                return None
        return point

    def _is_point_in_raster_extent(self, point):
        """Check if a point (in raster CRS) falls within the layer extent."""
        if point is None:
            return False  # the canvas -> raster transform had no image
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
        has_active_points = self._active_crop_points_positive or self._active_crop_points_negative
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
                        if self._crop_resolution_would_change():
                            return "zoom_changed"
            else:
                if self._current_crop_canvas_mupp is not None:
                    canvas = self.iface.mapCanvas()
                    current_mupp = canvas.mapUnitsPerPixel()
                    if current_mupp > 0:
                        ratio = current_mupp / self._current_crop_canvas_mupp
                        if ratio < zoom_in_thresh or ratio > zoom_out_thresh:
                            if self._crop_resolution_would_change():
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

    def _seed_side(self, fallback: int = 256) -> int:
        """The side a mask seed has to be for whoever answers the next click.

        Not one number any more. The on-device checkpoint works in 256, the
        tracker the service now answers with works in 288, and a seed at the
        wrong side is refused outright rather than resized. The predictor is
        the only thing that knows which is listening, so it is asked; before
        any answer has said, 256 is the honest guess, and one refusal teaches
        it the real one.
        """
        side = getattr(getattr(self, "predictor", None), "low_res_side", None)
        try:
            side = int(side)
        except (TypeError, ValueError):
            return fallback
        return side if side > 0 else fallback

    def _binary_mask_to_logits(self, mask, target: int | None = None):
        """Convert a binary mask (H x W, 0/1 or bool) to SAM low-res logits of
        shape (1, target, target): foreground=+6, background=-6. Shared by the
        zoom mask-transfer and the Refine-in-Manual polygon seeding so both seed
        SAM the same way.

        `target` defaults to whatever the active predictor works in, so a seed
        built here is never the wrong size for the model that reads it.

        Must stay 3D like the low_res_masks a predict returns: the SAM
        predictors add the batch dimension themselves, so a 4D mask_input
        reaches conv2d as 5D and crashes the prompt encoder."""
        import numpy as np
        side = self._seed_side() if target is None else int(target)
        m = np.asarray(mask, dtype=np.float32)
        logits = (m * 2.0 - 1.0) * 6.0
        logits_t = self._resize_nearest(logits, side, side)
        return logits_t[None, :, :]

    def _build_mask_input_from_previous(
        self, old_mask, old_bounds, old_shape, new_bounds, new_shape
    ):
        """Transfer a binary mask from old crop space to new crop as SAM logits.

        Computes geographic overlap between old and new crops, maps the
        overlapping region, converts to logits, and resizes to the side the
        active predictor reads.
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

        # Convert to SAM's low-res logits at the side the model reads:
        # foreground=+6, bg=-6.
        return self._binary_mask_to_logits(new_mask)

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
            if self._current_crop_actual_mupp and self._current_crop_canvas_mupp and self._current_crop_canvas_mupp > 0:
                scale = max(1.0, self._current_crop_actual_mupp / self._current_crop_canvas_mupp * 2.0)
            else:
                scale = 1.0
        # Power curve centered on 200 (bumped ×2 per #12 for cleaner defaults).
        return max(100, int(200 * max(0.6, scale) ** 0.3))

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
        if not hasattr(self, "_crop_read"):
            # The in-flight off-thread file crop read, a dict
            # {worker, gen, on_encoded, cursor, quiet, show_busy}, or None when
            # idle. Mutually exclusive with _online_fetch and with a live
            # encode worker: all three own the same transport lock.
            self._crop_read = None
        if not hasattr(self, "_queued_crop_request"):
            # The crop a caller asked for while the pipe was busy, started as
            # soon as the lock frees. One slot, newest wins: the user's latest
            # gesture is the only one still worth reading imagery for.
            self._queued_crop_request = None
        if not hasattr(self, "_inflight_crop_window"):
            # Window key of the crop being read/encoded right now, and of the
            # one the predictor actually holds. Two facts, deliberately not one
            # intent: a read that starts is not a crop that arrived, and only
            # the second is safe to skip an encode over.
            self._inflight_crop_window = None
            self._encoded_crop_window = None
        if not hasattr(self, "_encode_lock_gen"):
            # Generation of the encode that currently OWNS the transport lock
            # (None when the lock is free). A completion whose generation no
            # longer owns the lock must not release it, must not null the
            # pending crop and must not pop a cursor: all three now belong to a
            # newer encode. Distinct from _manual_encode_gen, which a plain
            # invalidation bumps while deliberately leaving the lock to the
            # draining worker's own completion.
            self._encode_lock_gen = None

    def _extract_and_encode_crop(self, center_point, mupp_override=None, *,
                                 on_encoded=None, show_busy=True, quiet=False):
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

          File-based layers read the window on a worker thread
          (_begin_file_crop_read, which falls back to an inline read for the
          few formats whose GDAL path cannot be scoped to one thread), then
          encode on another. Online layers read from the tile network, which
          can block ~18s of retries, so their
          fetch is driven off the event loop (_begin_online_crop_fetch) and
          only feeds the encode once the tiles stabilize. Either way the crop
          work holds the same transport lock and busy cursor as the encode, so
          a click meanwhile defers and replays exactly as before.

        Args:
            center_point: QgsPointXY center in raster CRS
            mupp_override: For online layers, override mupp (zoom-out).
                For file-based layers, this is the scale_factor [0.25, 8.0].
            on_encoded: continuation to run after a successful encode (the
                crop-transition tail). Interactive: runs from the completion
                callback. Headless: runs inline on success.
            show_busy: whether the wait wears the application busy cursor.
                False for a crop the user did not ask for by name (a warm-up),
                and for the fix-session open, which has its own panel line and
                must leave hover and the next pick alive.
            quiet: log a read failure instead of reporting it. For speculative
                crops only: nobody asked, so nobody should get a dialog.
        """
        self._ensure_manual_encode_state()
        if self._encoding_in_progress:
            # Another crop owns the predictor pipe. Interactive callers must not
            # be dropped here: queue this one (newest wins) and let the owner's
            # completion start it, or a pick made during a warm-up would open on
            # whatever crop happened to be loaded.
            if self._headless:
                return False
            self._queued_crop_request = {
                "center": center_point,
                "mupp": mupp_override,
                "on_encoded": on_encoded,
                "show_busy": bool(show_busy),
                "quiet": bool(quiet),
            }
            return True

        if self._headless:
            ok = self._encode_crop_blocking(center_point, mupp_override)
            if ok and on_encoded is not None:
                on_encoded()
            return ok

        if self._is_online_layer:
            # Online tile extraction blocks on the network (fetch + progressive
            # retry, up to ~18s). Drive it asynchronously so the GUI never
            # freezes; True means the crop FETCH started (the encode follows on
            # completion), not that the crop is ready. A quiet read is a
            # speculative one, and it goes through a private copy of the layer
            # so the user's basemap is left alone.
            return self._begin_online_crop_fetch(
                center_point, mupp_override, on_encoded,
                show_busy=show_busy, quiet=quiet)

        # File-based: the windowed read goes off the GUI thread too. It is fast
        # on a small local GeoTIFF, but on a big, heavily compressed or
        # network-mounted raster (and at a zoomed-out scale, where it decodes
        # up to 8x the crop per side) it costs as much as the encode, and it
        # used to run inline, so a click still froze QGIS before the encode
        # started. True means the READ started; the encode follows on its
        # completion, exactly like the online-tile path.
        return self._begin_file_crop_read(
            center_point, mupp_override, on_encoded,
            show_busy=show_busy, quiet=quiet)

    def _set_manual_encoding_note(self, active: bool, phase: str = "imagery") -> None:
        """Say in the panel what the click is waiting on.

        For the 3 to 8 seconds an encode takes, the busy cursor was the only
        sign anything was happening, and it names nothing. Base Manual only: the
        review's fix session has its own armed line, and a warm-up nobody asked
        for shows neither cursor nor line. The clear is never gated, so a
        handoff that opens mid-encode cannot strand the note on the panel.
        """
        dock = self.dock_widget
        if dock is None:
            return
        if active and getattr(self, "_refine_handoff_active", False):
            return
        try:
            dock.set_manual_encoding(bool(active), phase)
        except TypeError:
            # A dock from before the wait had two phases.
            dock.set_manual_encoding(bool(active))
        except (RuntimeError, AttributeError):
            pass

    def _wear_busy_cursor_for_crop(self) -> None:
        """Put the busy cursor on a crop read that is ALREADY in flight.

        A hover warm-up starts cursor-less on purpose: nobody asked for it, so
        nothing should look busy. The moment the user picks that same polygon
        the wait stops being speculative and becomes a click that takes
        seconds, and a click that takes seconds with an arrow cursor reads as a
        freeze. Each stage dict carries the cursor flag its own release checks,
        so flipping it here keeps the one push paired with the one pop.

        Both paths that can land on a silent crop call this: the review's AI
        fix session when it opens on an in-flight window, and a Manual click
        deferred behind one. Manual gets the panel note with it, so the two
        methods answer a click the same way."""
        self._ensure_manual_encode_state()
        if not getattr(self, "_encoding_in_progress", False):
            return
        stage = self._crop_read or self._pending_encode or self._online_fetch
        if stage is None or stage.get("cursor"):
            return
        stage["cursor"] = True
        if "show_busy" in stage:
            stage["show_busy"] = True  # carried across the read -> encode handoff
        self._encode_cursor_set = True
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        self._set_manual_encoding_note(True)

    def _drain_queued_crop_request(self) -> bool:
        """Start the crop a caller asked for while the pipe was busy.

        Called from every completion that hands the transport lock back. True
        when a crop was started, which also means the caller must NOT replay a
        remembered click yet: that click belongs to the crop now being read, not
        to the one that just finished."""
        self._ensure_manual_encode_state()
        request = self._queued_crop_request
        self._queued_crop_request = None
        if request is None or self._encoding_in_progress:
            return False
        if self.dock_widget is None or self.predictor is None:
            return False
        return bool(self._extract_and_encode_crop(
            request["center"], request["mupp"],
            on_encoded=request["on_encoded"],
            show_busy=request["show_busy"], quiet=request["quiet"]))

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
        else:
            args = self._file_crop_read_args(center_point, mupp_override, quiet)
            if args is None:
                return None, None
            image_np, crop_info, error, error_code_from_crop = extract_crop_from_raster(**args)

        if error:
            self._report_crop_error(error, error_code_from_crop, quiet)
            return None, None

        return image_np, crop_info

    def _file_crop_read_args(self, center_point, mupp_override, quiet=False):
        """Everything the file-based windowed read needs, as PLAIN data.

        Reads the layer CRS, the layer extent and the canvas scale, which are
        main-thread-only, and stashes the per-crop scale state. The result is a
        kwargs dict for extract_crop_from_raster holding nothing but a path,
        numbers and strings, so the read itself can run on any thread. Returns
        None after surfacing the missing-path error (the one failure that can
        be seen before reading a single pixel).
        """
        if not self._current_raster_path:
            if quiet:
                return None
            if self._crop_error_went_to_panel("crop_error_no_path"):
                return None
            message = tr("This layer has no file to read. Pick another layer at "
                         "the top of the panel, then start again.")
            if self._headless:
                self._headless_error = message
                return None
            show_error_report(
                self.iface.mainWindow(),
                tr("Crop Error"),
                message,
                error_code="crop_error_no_path",
            )
            return None

        from ...core.layer_conventions import ground_unit_aspect

        layer_crs_wkt = None
        layer_extent = None
        # How much more ground one y unit of the raster CRS covers than one x
        # unit, at the click. 1.0 on every CRS whose axes agree, so the read is
        # untouched there. Measured HERE because it reads the project, which is
        # main-thread only, and the read below may run on a worker.
        ground_aspect = 1.0
        try:
            if self._current_layer.crs().isValid():
                layer_crs_wkt = self._current_layer.crs().toWkt()
                ground_aspect = ground_unit_aspect(
                    self._current_layer.crs(), center_point.x(), center_point.y())
            ext = self._current_layer.extent()
            if ext and not ext.isEmpty():
                layer_extent = (ext.xMinimum(), ext.yMinimum(),
                                ext.xMaximum(), ext.yMaximum())
        except RuntimeError:
            pass

        scale_factor = mupp_override or 1.0
        # The zoom baseline the next click is measured against, held back until
        # the crop it describes is actually in hand (see
        # _apply_encode_result_ok). Written here, a read that failed left the
        # baseline describing a crop that never arrived, and the click after it
        # matched the baseline, skipped the re-encode and was answered from the
        # older, coarser imagery still loaded.
        self._pending_crop_zoom_baseline = (
            scale_factor, self.iface.mapCanvas().mapUnitsPerPixel())
        return {
            "raster_path": self._current_raster_path,
            "center_x": center_point.x(),
            "center_y": center_point.y(),
            "crop_size": 1024,
            "layer_crs_wkt": layer_crs_wkt,
            "layer_extent": layer_extent,
            "scale_factor": scale_factor,
            "ground_aspect": ground_aspect,
        }

    def _crop_error_went_to_panel(self, error_code, center_point=None) -> bool:
        """Try to answer a failed click with one line in the panel.

        True when the notice took it and no dialog is owed. Wrong basemap,
        wrong zoom, a click off the edge of the raster: the user fixes those in
        one move, and asking them to mail logs for it was the whole complaint.
        Which codes qualify, and what each one says, is in
        core/click_error_advice.py.
        """
        from ...core.click_error_advice import click_error_notice

        selected = ""
        try:
            selected = self._current_layer.name()
        except (RuntimeError, AttributeError):  # nosec B110 -- layer gone
            pass
        notice = click_error_notice(
            error_code or "", selected,
            self._visible_raster_under_click(center_point))
        if not notice:
            return False
        if self._headless:
            self._headless_error = notice
        else:
            try:
                self.dock_widget.show_manual_notice(notice)
            except (RuntimeError, AttributeError):
                # No panel to write on. Hand it back to the dialog, which
                # fires its own event, so nothing is counted twice.
                return False
        # The dialog used to fire this event on its way up. These failures still
        # need counting (how often a run dies on the wrong basemap is the whole
        # reason to know), so the code travels even though the dialog does not.
        # The sentence itself never does: it carries layer names.
        try:
            from ...core.telemetry_errors import track_plugin_error
            track_plugin_error(
                stage="segment", error_code=error_code or "crop_error_unknown",
                message="")
        except Exception:  # noqa: BLE001 -- telemetry never blocks a click
            pass  # nosec B110
        return True

    def _report_crop_error(self, error, error_code_from_crop, quiet=False) -> None:
        """Log and surface a file/online crop extraction failure on the MAIN
        thread. Shared by the synchronous path (_extract_crop_only) and the
        off-thread read completion, so both run the same repair and the same
        one-dialog-per-session dedup."""
        QgsMessageLog.logMessage(
            f"Crop extraction failed: {error}",
            "AI Segmentation", level=Qgis.MessageLevel.Critical
        )
        if quiet:
            return
        # Before the dedup, not after: a notice is cheap to repeat, and the
        # user who clicks the same wrong layer twice should read the same
        # sentence twice rather than nothing at all.
        if self._crop_error_went_to_panel(error_code_from_crop):
            return
        if error_code_from_crop == "crop_error_rasterio_unavailable":
            # The panel said ready but the in-process rasterio import
            # failed: the package is present-but-broken in the venv
            # (antivirus quarantine, interrupted install). An error report
            # dead-ends the user, so purge the broken artifacts (pip would
            # otherwise consider rasterio satisfied and skip it) and route
            # to the same one-click repair as a broken runtime. (#64)
            # Repair ONCE per session. Reinstalling only cures a damaged
            # package; an import that fails for any other reason (a native
            # extension the host process cannot load) survives every purge,
            # and repeating the cycle traps the user in a modal loop that
            # can never succeed. After the first attempt, say what actually
            # broke and let them report it.
            if not self._rasterio_repair_attempted:
                self._rasterio_repair_attempted = True
                # Counted here, so ONE broken environment sends one event per
                # session. Counting it on every failed read let a single stuck
                # machine emit hundreds in minutes, which reads as a broken
                # release rather than one broken install. A user who stays
                # stuck still reports once per launch.
                try:
                    from ...core.telemetry_errors import track_plugin_error
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
                return
            QgsMessageLog.logMessage(
                "rasterio still unavailable after a repair; not repairing "
                "again this session",
                "AI Segmentation", level=Qgis.MessageLevel.Critical,
            )
            if self._headless:
                self._headless_error = error
                return
            # One dialog per session, like the generic branch below. The reader
            # is broken for every file, so the key carries no path. The repair
            # has already failed by the time a click reaches here, so a second
            # dialog asks the user to fix what this dialog cannot, and they
            # used to arrive faster than they could be closed.
            report_key = ("", "crop_error_rasterio_unavailable")
            if report_key in self._crop_errors_reported:
                return
            self._crop_errors_reported.add(report_key)
            show_error_report(
                self.iface.mainWindow(),
                tr("Crop Error"),
                tr("The imagery reader could not be loaded, and repairing "
                   "the installation did not fix it. Please report this "
                   "so we can look into it.\n\n{details}").format(
                    details=error),
                error_code="crop_error_rasterio_unavailable",
                track=False,
            )
            return
        if self._headless:
            self._headless_error = error
            return
        # Same file, same failure: one dialog per session. A corrupt or
        # unsupported raster stays that way, and users retry-click several
        # times in a row. Repeats go to the log panel only; a different
        # file or failure still surfaces.
        report_key = (
            os.path.normcase(self._current_raster_path or ""),
            error_code_from_crop or "crop_error_unknown",
        )
        if report_key in self._crop_errors_reported:
            QgsMessageLog.logMessage(
                "Same crop error already reported this session; "
                "not showing the dialog again",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return
        self._crop_errors_reported.add(report_key)
        show_error_report(
            self.iface.mainWindow(),
            tr("Crop Error"),
            error,
            error_code=error_code_from_crop or "crop_error_unknown",
        )

    def _encode_crop_blocking(self, center_point, mupp_override) -> bool:
        """Synchronous crop extract + SAM encode on the CALLING thread.

        Used by the headless/MCP path (which must block) and by the rare
        _run_prediction recovery re-encode (the worker died, so there is no
        encode in flight and a main-thread set_image is transport-safe). Blocks
        ~3-8s. Returns True on success / False on error. This is the pre-PERF-01
        behaviour, kept intact for these two callers.
        """
        # Neither caller checks, and the predictor is None for the length of an
        # install. Without this the attribute error reaches the user as a raw
        # Python message in an error-report dialog.
        if self.predictor is None:
            return False
        if self._encoding_in_progress:
            return False
        self._encoding_in_progress = True
        if not self._headless:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            self._set_manual_encoding_note(True)
            QApplication.processEvents()
        try:
            self._inflight_crop_window = self._crop_window_key_for(
                center_point, mupp_override)
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
                self._set_manual_encoding_note(False)

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
        # No subprocess probe: this runs on the MAIN thread, on any Manual click
        # whose encode failed, and the probe is a 30 second interpreter spawn.
        # It froze QGIS for half a minute exactly when the environment is
        # broken, which is when that timeout is most likely to be paid in full.
        # The filesystem half is what catches the case named above, a venv whose
        # base Python is gone.
        if venv_needs_repair(allow_subprocess_probe=False):
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
        # The file read's zoom baseline lands with the crop it describes, for
        # the same reason the window below does. Online crops set their own
        # baseline in _online_crop_mupp and leave this empty.
        baseline = getattr(self, "_pending_crop_zoom_baseline", None)
        self._pending_crop_zoom_baseline = None
        if baseline is not None and not self._is_online_layer:
            self._current_crop_scale_factor, self._current_crop_canvas_mupp = baseline
        # The predictor now HOLDS this window. Recording it only here is the
        # point: a read that started, or an encode that failed, must never let a
        # later open skip its own encode over a crop that never arrived.
        self._encoded_crop_window = getattr(self, "_inflight_crop_window", None)
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
        self._encode_lock_gen = gen
        # The busy cursor and the panel note are the affordances for the 3-8s
        # wait. Both are taken down in the completion callback for every
        # outcome.
        if show_busy:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            self._set_manual_encoding_note(True)

        try:
            worker = SetImageWorker(self.predictor, image_np, gen)
            self._manual_encode_worker = worker
            worker.done.connect(self._on_manual_encode_done)
            park_orphaned_worker(worker)
            worker.start()
        except Exception as e:  # noqa: BLE001 - a failed start must release the lock
            # No worker ran, so no completion will ever release the lock or
            # pop the cursor: undo both here, or every later click is dropped
            # and the app-global busy cursor outlives the session.
            self._manual_encode_worker = None
            self._pending_encode = None
            self._encoding_in_progress = False
            self._encode_lock_gen = None
            if show_busy:
                try:
                    QApplication.restoreOverrideCursor()
                except Exception:  # nosec B110 -- cursor restore is best-effort
                    pass
                self._set_manual_encoding_note(False)
            self._encode_cursor_set = True
            self._discard_pending_manual_click()
            self._handle_encode_error(str(e))
            return
        self._arm_encode_watchdog()

    def _on_manual_encode_done(self, gen: int, ok: bool, err: str) -> None:
        """Main-thread completion of an off-thread encode (queued via the
        SetImageWorker.done signal, exactly like PredictorLoadWorker so it lands
        on the GUI thread). Commits the crop + runs the crop-transition tail +
        replays the user's click on success; drops everything on a stale/torn
        completion; reproduces the sync error recovery on failure."""
        self._ensure_manual_encode_state()

        # Does this completion still own the transport lock? A watchdog force
        # release drops the lock while the worker is still alive, so the user's
        # next click can legitimately start a NEW encode. When that late
        # completion finally lands it must touch nothing: releasing the lock
        # would let the next click write to the pipe under the live worker,
        # nulling _pending_encode would lose the new crop, and popping the
        # cursor would unbalance the new encode's own restore.
        if gen != self._encode_lock_gen:
            return

        pending = self._pending_encode

        # Teardown detection that does NOT rely on a generation bump: unload
        # (dock gone) and env-reset/reload (predictor cleared or swapped) cannot
        # bump the counter, so detect them by state and only clean up.
        torn_down = self.dock_widget is None or self.predictor is None
        torn_down = torn_down or (pending is not None and self.predictor is not pending.get("predictor"))

        # Release the pipe lock, cursor and worker ref for every outcome. Only
        # restore the cursor when this encode SET one (the speculative prewarm
        # runs cursor-less; an unpaired restore would pop someone else's).
        self._manual_encode_worker = None
        self._pending_encode = None
        self._encoding_in_progress = False
        self._encode_lock_gen = None
        cursor_was_set = (pending.get("cursor", True) if pending is not None
                          else getattr(self, "_encode_cursor_set", True))
        self._encode_cursor_set = True
        if cursor_was_set:
            try:
                QApplication.restoreOverrideCursor()
            except Exception:  # nosec B110 -- cursor restore is best-effort
                pass
            self._set_manual_encoding_note(False)

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
            self._inflight_crop_window = None
            self._handle_encode_error(err)
            self._discard_pending_manual_click()
            self._drain_queued_crop_request()
            return

        # Success: adopt the new crop, run the crop-transition tail (freeze /
        # transfer / refine seed), then replay the user's click (last wins).
        if pending is not None and pending.get("crop_info") is not None:
            self._apply_encode_result_ok(pending["crop_info"])
        # The imagery around the selected polygon is now read: swap the panel's
        # loading note for the keep/trim gesture help (handoff only, no-op else).
        setter = getattr(self, "_set_ai_session_armed_line", None)
        if setter is not None:
            setter(loading=False)
        tail = pending.get("tail") if pending is not None else None
        if tail is not None:
            try:
                tail()
            except Exception as e:  # noqa: BLE001 - never wedge the flow on the tail
                QgsMessageLog.logMessage(
                    f"Manual encode continuation failed: {e}",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
        # A crop queued behind this one goes first: the remembered click was
        # aimed at THAT crop, so replaying it against this one would predict on
        # the wrong imagery. Its own completion replays the click.
        if self._drain_queued_crop_request():
            return
        if self._pending_manual_click is not None:
            self._replay_pending_manual_click()

    # ---- Async file crop read (interactive) ---------------------------------
    # Third owner of the transport lock, alongside the encode worker and the
    # online fetch. All three hold `_encoding_in_progress` + the busy cursor +
    # a generation + the watchdog, and exactly one of them is live at a time: a
    # read clears itself BEFORE handing the crop to the encode, and a read and
    # an online fetch belong to different branches of the same decision.

    def _begin_file_crop_read(self, center_point, mupp_override, on_encoded,
                              *, quiet: bool = False,
                              show_busy: bool = True) -> bool:
        """Start the off-thread windowed read of a file-based crop.

        Returns True when the crop is under way (the read started and the
        encode follows on its completion, or, on the inline fallback below, the
        read is already done and the encode started), False when nothing was
        started (no raster path, a failed read, or a thread that refused to
        start; all three already surfaced). The lock, the cursor and the
        watchdog are taken here and released on EVERY exit: the failed start
        below, the completion, _release_crop_read (teardown) and the watchdog's
        force release.
        """
        self._ensure_manual_encode_state()
        args = self._file_crop_read_args(center_point, mupp_override, quiet)
        if args is None:
            return False
        self._inflight_crop_window = self._crop_window_key_for(
            center_point, mupp_override)

        from ...core.feature_encoder import crop_read_is_thread_safe

        if not crop_read_is_thread_safe(args["raster_path"]):
            # This build reads that format through a GDAL whose config options
            # are process-global, and the read blanks PROJ's data paths while
            # it runs (see feature_encoder). Off the GUI thread that would
            # blank them under a live canvas render. So read inline: the
            # window blocks for the read, exactly as it did before, and the
            # map keeps its projection.
            from ...core.feature_encoder import extract_crop_from_raster

            # The window freezes for the whole read, and at scale 8 that read
            # decodes an 8192x8192 window of native pixels. The busy cursor and
            # the panel note say why, so they go up FIRST and get one paint pass
            # to reach the screen; the encode handoff below used to be the first
            # to raise them, which is after the freeze they explain. User input
            # stays held back over that pass, so a click cannot start a second
            # read underneath this one.
            self._show_inline_read_busy(show_busy)
            try:
                result = extract_crop_from_raster(**args)
            finally:
                self._hide_inline_read_busy(show_busy)
            return self._deliver_crop_read(result, on_encoded, quiet=quiet,
                                           show_busy=show_busy)

        self._manual_encode_gen += 1
        gen = self._manual_encode_gen
        try:
            from .shared import park_orphaned_worker
            worker = CropReadWorker(args, gen)
        except Exception as e:  # noqa: BLE001 - nothing was taken yet
            self._report_crop_error(str(e), "crop_error_unknown", quiet)
            return False

        self._crop_read = {
            "worker": worker,
            "gen": gen,
            "on_encoded": on_encoded,
            "cursor": bool(show_busy),
            "quiet": bool(quiet),
            "show_busy": bool(show_busy),
        }
        self._encoding_in_progress = True
        self._encode_lock_gen = gen
        self._encode_cursor_set = bool(show_busy)
        if show_busy:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            self._set_manual_encoding_note(True)
        try:
            worker.done.connect(self._on_file_crop_read_done)
            park_orphaned_worker(worker)
            worker.start()
        except Exception as e:  # noqa: BLE001 - a failed start must release
            # No thread ran, so no completion will ever release the lock or pop
            # the cursor: undo both here, or every later click is dropped and
            # the app-global busy cursor outlives the session.
            self._release_crop_read()
            self._discard_pending_manual_click()
            self._report_crop_error(str(e), "crop_error_unknown", quiet)
            return False
        self._arm_encode_watchdog()
        return True

    def _show_inline_read_busy(self, show_busy: bool) -> None:
        """Put the busy cursor and the panel note on screen before a read that
        blocks the GUI thread. Paired with _hide_inline_read_busy.

        The paint pass holds back user input, so the wait is announced without
        letting a click in behind it.
        """
        if not show_busy:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        self._set_manual_encoding_note(True)
        try:
            from qgis.PyQt.QtCore import QEventLoop

            from ...core.qt_compat import resolve_qt_enum

            QApplication.processEvents(resolve_qt_enum(
                QEventLoop, "ProcessEventsFlag", "ExcludeUserInputEvents"))
        except Exception:  # noqa: BLE001 -- a missed repaint must not cost the read
            pass  # nosec B110

    def _hide_inline_read_busy(self, show_busy: bool) -> None:
        """Take back what _show_inline_read_busy put up. The encode handoff
        raises its own cursor and note straight after, in the same turn, so
        nothing repaints in between."""
        if not show_busy:
            return
        try:
            QApplication.restoreOverrideCursor()
        except Exception:  # nosec B110 -- cursor restore is best-effort
            pass
        self._set_manual_encoding_note(False)

    def _on_file_crop_read_done(self, gen: int, result) -> None:
        """Main-thread completion of an off-thread crop read (queued via the
        worker's `done` signal). Releases the read's hold on the lock and the
        cursor for every outcome, then either surfaces the read error or hands
        the crop to the async SAM encode, which takes both back."""
        self._ensure_manual_encode_state()
        # A watchdog force release, a teardown or a newer read already took the
        # lock away from this generation: touch nothing at all.
        if gen != self._encode_lock_gen:
            return
        read = self._crop_read
        quiet = bool(read.get("quiet")) if read is not None else False
        show_busy = bool(read.get("show_busy", True)) if read is not None else True
        on_encoded = read.get("on_encoded") if read is not None else None
        self._release_crop_read()

        # Teardown that cannot bump the generation (unload, env reset): the
        # lock and cursor are already back, so there is nothing left to do.
        if self.dock_widget is None or self.predictor is None:
            return
        if gen != self._manual_encode_gen:
            # Invalidated in-session (stop, mode switch, layer removal). Drop
            # this crop; a click remembered by a NEW session self-heals.
            if self._pending_manual_click is not None:
                self._replay_pending_manual_click()
            return

        if self._deliver_crop_read(result, on_encoded, quiet=quiet,
                                   show_busy=show_busy):
            return
        # The read failed, and everything waiting behind it needs THAT read to
        # succeed. Replaying a deferred click re-runs the read that just failed,
        # and its failure replays the click again: a loop at machine speed, one
        # error dialog per turn, that a user can only leave by killing QGIS. The
        # encode-failure path already stops here for the same reason. Drop both,
        # and let the user retry with a click, which is a retry at human pace
        # with room for the cause to be fixed in between.
        self._queued_crop_request = None
        self._discard_pending_manual_click()

    def _deliver_crop_read(self, result, on_encoded, *, quiet: bool,
                           show_busy: bool) -> bool:
        """Hand one finished windowed read to the SAM encode, or surface its
        error. True when the encode was started. Shared by the off-thread
        completion and the inline fallback, so both report and hand off the
        same way."""
        image_np, crop_info, error, error_code = result
        if error or image_np is None:
            self._inflight_crop_window = None
            # The handoff panel says "reading the imagery" from the moment the
            # read starts. A read that fails here is the only way that line
            # never reaches an encode completion, so take it down now or the
            # panel waits forever on work that already gave up.
            setter = getattr(self, "_set_ai_session_armed_line", None)
            if setter is not None:
                setter(loading=False)
            self._report_crop_error(error or "crop read failed",
                                    error_code or "crop_error_unknown", quiet)
            return False
        self._start_manual_encode(image_np, crop_info, on_encoded,
                                  show_busy=show_busy)
        return True

    def _release_crop_read(self) -> None:
        """Drop the in-flight crop read's hold: transport lock, busy cursor,
        bookkeeping. Idempotent and safe when no read is active. The thread
        itself is left to finish (it is parked, and it touches nothing but
        plain data); its completion sees the generation is no longer the lock
        owner and no-ops."""
        self._ensure_manual_encode_state()
        read = self._crop_read
        self._crop_read = None
        if read is None:
            return
        self._encoding_in_progress = False
        self._encode_lock_gen = None
        if read.get("cursor"):
            self._encode_cursor_set = False
            try:
                QApplication.restoreOverrideCursor()
            except Exception:  # nosec B110 -- cursor restore is best-effort
                pass
            self._set_manual_encoding_note(False)

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

    def _begin_online_crop_fetch(self, center_point, mupp_override, on_encoded,
                                 *, show_busy: bool = True,
                                 quiet: bool = False) -> bool:
        """Start the asynchronous online-tile crop fetch. Returns True when the
        fetch STARTED (the encode follows on completion), False when it could
        not start (provider unavailable / setup error, already surfaced).

        A SPECULATIVE fetch (``quiet``) reads through a private twin of the
        layer and refuses to start without one. The fetch switches the provider
        it reads to bilinear and makes it drop its tiles, and nobody asked for a
        basemap that reloads under the cursor. Being unasked, it also wears no
        busy cursor and reports no failure: the click behind it pays its own way
        exactly as it always has.
        """
        from ...core.feature_encoder import OnlineCropFetcher
        from ...core.online_layer_twin import online_layer_twin

        read_layer = self._current_layer
        if quiet:
            read_layer = online_layer_twin(read_layer)
            if read_layer is None:
                return False
        actual_mupp = self._online_crop_mupp(mupp_override)
        if mupp_override is None:
            # No resolution was named, so no window was chosen either: put this
            # one on the shared grid. An online crop is the most expensive read
            # there is, and two clicks that share a cell mean the second
            # downloads no tiles at all. The grid unit here is ground per pixel,
            # so the mupp IS the scale and the pixel size is 1. This only moves
            # WHERE a crop the user already asked for is centred; it never
            # starts a read on its own, because an online read switches the live
            # layer to bilinear and makes it drop its tiles.
            from ...core.crop_window import snap_center_to_grid
            cx, cy = snap_center_to_grid(
                center_point.x(), center_point.y(), actual_mupp, 1.0)
            center_point = QgsPointXY(cx, cy)
        # The window is keyed on the mupp actually used, which for an online crop
        # is the ground size per pixel the fetch asked the provider for.
        self._inflight_crop_window = self._crop_window_key_for(
            center_point, actual_mupp)
        fetcher = OnlineCropFetcher(
            read_layer, center_point.x(), center_point.y(),
            actual_mupp, crop_size=1024)
        if fetcher.error is not None:
            self._surface_online_crop_error(
                fetcher.error, fetcher.error_code, center_point, quiet=quiet)
            return False
        try:
            fetcher.begin()
        except Exception as e:  # noqa: BLE001 - never leave the provider mutated
            fetcher.restore()
            self._surface_online_crop_error(
                str(e), "crop_error_online_exception", center_point, quiet=quiet)
            return False

        self._ensure_manual_encode_state()
        self._manual_encode_gen += 1
        gen = self._manual_encode_gen
        self._online_fetch = {
            "fetcher": fetcher,
            "gen": gen,
            "on_encoded": on_encoded,
            "cursor": bool(show_busy),
            # The tile download, when it runs on a worker thread, and the flag
            # that tells it nobody wants the tiles any more. Set by the one
            # release site, so every way this fetch can end stops the download.
            "worker": None,
            "cancel": threading.Event(),
            # Rides the fetch so a failure several steps later still knows
            # nobody asked for this crop.
            "quiet": bool(quiet),
            # The click this fetch serves, kept for the failure report: it is
            # what lets the report name the layer the user is actually seeing.
            "center": center_point,
        }
        # The fetch owns the transport lock for its whole life, and the busy
        # cursor too when it has one. A click meanwhile defers via
        # _encoding_in_progress and replays after the crop is ready.
        # _encode_cursor_set mirrors the cursor so handoff gestures treat the
        # fetch like a foreground encode, and so an unasked one stays
        # abandonable (_abandon_speculative_manual_crop reads it).
        self._encoding_in_progress = True
        self._encode_cursor_set = bool(show_busy)
        self._encode_lock_gen = gen
        if show_busy:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            self._set_manual_encoding_note(True)
        self._arm_encode_watchdog()
        if show_busy:
            # Let the cursor and the note paint before the first read blocks.
            # A silent fetch has nothing to paint, so it skips the re-entrant
            # pass through the event loop entirely.
            QApplication.processEvents()
        self._step_online_crop_fetch()
        return True

    def _step_online_crop_fetch(self) -> None:
        """Run one online-fetch attempt on the GUI thread, then finish (on
        success/exhaustion) or schedule the next attempt after its back-off via
        QTimer.singleShot. A stale step (session torn down / superseded) is
        dropped: its lock and cursor were already released by the teardown.

        The one attempt that does NOT run here is a crop whose tiles can be
        asked for by number: that is a single uninterrupted download, so it
        goes to a worker and comes back through its completion."""
        self._ensure_manual_encode_state()
        if self.dock_widget is None:
            # Unloaded while a fetch was scheduled: revert the provider state
            # and pop the cursor (unload does not run the encode invalidation).
            self._release_online_fetch()
            return
        fetch = self._online_fetch
        if fetch is None or fetch.get("gen") != self._manual_encode_gen:
            return  # superseded; teardown already released the lock/cursor
        if self._start_direct_tile_fetch(fetch):
            return
        try:
            action, delay = fetch["fetcher"].step()
        except Exception as e:  # noqa: BLE001
            self._fail_online_fetch(str(e), "crop_error_online_exception")
            return
        self._route_online_fetch_action(action, delay)

    def _route_online_fetch_action(self, action, delay) -> None:
        """Act on what one fetch attempt answered: finish it, or wait out its
        back-off off the event loop and take the next attempt. Shared by the
        inline step and the tile-download completion, so both read the same
        answers the same way."""
        from qgis.PyQt.QtCore import QTimer

        if action in ("stabilized", "exhausted"):
            self._complete_online_crop_fetch()
            return
        # ("refetch", 0.5) or ("retry", delay): wait off the event loop, then
        # take the next attempt.
        QTimer.singleShot(max(0, int(delay * 1000)), self._step_online_crop_fetch)

    def _start_direct_tile_fetch(self, fetch) -> bool:
        """Send this fetch's tile download to a worker thread. True when the
        worker started, and the step then routes through its completion instead
        of downloading here.

        False leaves the caller to take the attempt inline exactly as before:
        there is no request to download (the crop reads the layer through the
        provider ladder), or the thread refused to start. A click is worth more
        than the thread it would have run on, so a refusal costs the freeze it
        always cost and nothing else.
        """
        if fetch.get("worker") is not None:
            # A download is already on its way, and its completion drives what
            # comes next. Reading the provider now would race it.
            return True
        fetcher = fetch.get("fetcher")
        request = fetcher.direct_tile_request() if fetcher is not None else None
        if request is None:
            return False
        cancel = fetch.get("cancel")
        try:
            from .shared import park_orphaned_worker
            worker = DirectTileFetchWorker(
                request, fetch["gen"],
                cancel_check=None if cancel is None else cancel.is_set)
            worker.done.connect(self._on_direct_tile_fetch_done)
            park_orphaned_worker(worker)
            worker.start()
        except Exception as e:  # noqa: BLE001 - a failed start reads inline
            QgsMessageLog.logMessage(
                f"Tile download stays on the main thread: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return False
        # The fetcher must not download these tiles a second time, whatever the
        # worker comes back with. The completion names the request again, so it
        # is kept here rather than left on the fetcher.
        fetcher.take_direct_tile_request()
        fetch["worker"] = worker
        fetch["request"] = request
        return True

    def _on_direct_tile_fetch_done(self, gen: int, result) -> None:
        """Main-thread completion of an off-thread tile download (queued via the
        worker's `done` signal). Hands the outcome to the fetcher, which either
        keeps the picture or falls back to the layer read, then routes the
        answer exactly as an inline step would.

        The fetch keeps the transport lock and the busy cursor across this, so
        nothing is taken or released here: this is one attempt of a fetch that
        is still running, not the end of it."""
        self._ensure_manual_encode_state()
        if self.dock_widget is None:
            # Unloaded while the tiles were coming down.
            self._release_online_fetch()
            return
        fetch = self._online_fetch
        if fetch is None or fetch.get("gen") != gen:
            return  # superseded; teardown already released the lock/cursor
        if gen != self._manual_encode_gen:
            return
        fetch["worker"] = None
        request = fetch.pop("request", None)
        try:
            action, delay = fetch["fetcher"].accept_direct_tiles(request, *result)
        except Exception as e:  # noqa: BLE001
            self._fail_online_fetch(str(e), "crop_error_online_exception")
            return
        self._route_online_fetch_action(action, delay)

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
        # finish() can open a nested event loop (its renderer pass), so a layer
        # removal, a Stop or an unload delivered inside it has already torn this
        # session down and released the lock. Everything below would then take
        # the lock again on a dead session, so re-check before handing off.
        if self._online_fetch is not fetch:
            return
        if self.dock_widget is None or self.predictor is None:
            self._release_online_fetch(restore_provider=False)
            return
        if fetch.get("gen") != self._manual_encode_gen:
            self._release_online_fetch(restore_provider=False)
            if self._pending_manual_click is not None:
                self._replay_pending_manual_click()
            return
        if error:
            # Provider state already reverted above.
            self._fail_online_fetch(error, error_code, restore_provider=False)
            return
        # Success: drop the fetch bookkeeping and pop the fetch's busy cursor,
        # then start the async encode (it re-asserts the lock + pushes its own
        # cursor, restored on the encode completion). The lock stays held across
        # the handoff so a click cannot race in between. The encode inherits the
        # fetch's cursor state, read HERE rather than at the start: a click that
        # landed on a silent fetch promoted it to a foreground one, and the
        # encode behind it belongs to that click.
        self._online_fetch = None
        show_busy = bool(fetch.get("cursor"))
        if show_busy:
            try:
                QApplication.restoreOverrideCursor()
            except Exception:  # nosec B110
                pass
        self._start_manual_encode(image_np, crop_info, on_encoded,
                                  show_busy=show_busy)

    def _fail_online_fetch(self, error, error_code,
                           restore_provider: bool = True) -> None:
        """Online crop fetch failed: release the lock + cursor (+ provider state
        unless already reverted), surface the same error the sync path would,
        and drop the deferred click (it cannot be honored)."""
        fetch = getattr(self, "_online_fetch", None) or {}
        center_point = fetch.get("center")
        quiet = bool(fetch.get("quiet"))
        self._release_online_fetch(restore_provider=restore_provider)
        self._surface_online_crop_error(error, error_code, center_point,
                                        quiet=quiet)
        self._discard_pending_manual_click()

    def _release_online_fetch(self, restore_provider: bool = True) -> None:
        """Tear down the in-flight online crop fetch: revert the provider's live
        resampling state, drop the transport lock, pop the busy cursor.
        Idempotent; safe when no fetch is active. A live encode WORKER never
        owns _online_fetch (it starts only after a fetch clears it), so clearing
        the lock here can never clobber a worker's pipe ownership.

        A tile download still on its thread is told to stop and cut loose here,
        which is what makes this the single release site for the worker too:
        unload runs it (through _invalidate_manual_encode) while the dock is
        still alive, so the download's completion is disconnected before
        anything it could touch is deleted. The thread itself is never stopped
        by force. It is parked, it holds nothing but plain data, and
        park_orphaned_worker joins it before its object goes."""
        self._ensure_manual_encode_state()
        fetch = self._online_fetch
        self._online_fetch = None
        if fetch is None:
            return
        cancel = fetch.get("cancel")
        if cancel is not None:
            cancel.set()
        worker = fetch.get("worker")
        if worker is not None:
            fetch["worker"] = None
            try:
                worker.done.disconnect()
            except (TypeError, RuntimeError):
                pass  # never connected, or the C++ half is already gone
        if restore_provider:
            try:
                fetch["fetcher"].restore()
            except Exception:  # nosec B110
                pass
        self._encoding_in_progress = False
        self._encode_lock_gen = None
        if fetch.get("cursor"):
            try:
                QApplication.restoreOverrideCursor()
            except Exception:  # nosec B110
                pass
            self._set_manual_encoding_note(False)

    def _surface_online_crop_error(self, error, error_code,
                                   center_point=None,
                                   quiet: bool = False) -> None:
        """Log + report an online crop-fetch failure on the GUI thread (mirrors
        the interactive branch of _extract_crop_only; online never returns the
        rasterio-unavailable code, so no venv recovery is wired here).

        ``center_point`` is the click in the CURRENT layer's CRS, when the
        caller has one. It buys the one diagnosis a user cannot make from the
        error text alone: the layer picked in the panel answers nothing, while
        the imagery they are LOOKING at comes from another layer underneath.
        Without the hint that reads as the plugin failing on a picture that is
        right there on screen.

        ``quiet`` logs and stops there, for a crop nobody asked for: the click
        that follows reads its own imagery and reports its own failure. It logs
        as information, not as a fault. A preparation the user never asked for
        must not colour the log red, or the first thing they see on a basemap
        that answers slowly is a critical line about something that was going
        to work anyway.
        """
        if quiet:
            QgsMessageLog.logMessage(
                f"Prepared crop not read, the click will read its own: {error}",
                "AI Segmentation", level=Qgis.MessageLevel.Info
            )
            return
        QgsMessageLog.logMessage(
            f"Crop extraction failed: {error}",
            "AI Segmentation", level=Qgis.MessageLevel.Critical
        )
        if self._crop_error_went_to_panel(error_code, center_point):
            return
        show_error_report(
            self.iface.mainWindow(),
            tr("Crop Error"),
            error,
            error_code=error_code or "crop_error_unknown",
        )

    def _visible_raster_under_click(self, center_point) -> str:
        """Name of the topmost VISIBLE raster, other than the session's layer,
        that covers the clicked spot. Empty when there is none, when there is
        no point to test, or on any doubt: this feeds a hint, and a wrong name
        would send the user to a layer that answers nothing either.

        Walked in layer-tree render order, top first, because "the image you
        see" is by definition the highest visible one that draws there.
        """
        if center_point is None:
            return ""
        try:
            from qgis.core import (
                QgsCoordinateTransform,
                QgsProject,
                QgsRasterLayer,
            )

            project = QgsProject.instance()
            root = project.layerTreeRoot()
            current = self._current_layer
            for layer in root.layerOrder():
                if layer is current or not isinstance(layer, QgsRasterLayer):
                    continue
                node = root.findLayer(layer.id())
                if node is None or not node.isVisible():
                    continue
                point = center_point
                try:
                    if current is not None and layer.crs() != current.crs():
                        point = QgsCoordinateTransform(
                            current.crs(), layer.crs(),
                            project.transformContext()).transform(center_point)
                except Exception:  # noqa: BLE001 -- an untransformable point is no evidence
                    point = None
                if point is None:
                    continue
                extent = layer.extent()
                if extent is None or extent.isEmpty():
                    continue
                if extent.contains(point):
                    return layer.name()
        except Exception:  # noqa: BLE001 -- a hint must never break the report  # nosec B110
            return ""
        return ""

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
        if not self._is_point_in_raster_extent(center):
            return  # view is off the raster; the first real click drives the encode
        if self._is_online_layer:
            self._prewarm_online_manual_encode(center)
            return
        QgsMessageLog.logMessage(
            "Prewarming first crop at view center",
            "AI Segmentation", level=Qgis.MessageLevel.Info
        )
        # Off the GUI thread like every other interactive crop: a prewarm that
        # read inline froze the session for as long as the read took, which is
        # the exact freeze the prewarm exists to remove. Quiet, so a failed
        # speculative read only logs, and cursor-less: nobody asked for this
        # crop, so nobody should watch an hourglass for it.
        from ...core.crop_window import snap_center_to_grid
        scale = self._compute_initial_scale_factor()
        # On the shared grid, so a later warm-up or click in this neighbourhood
        # asks the model for a crop it already holds.
        cx, cy = snap_center_to_grid(
            center.x(), center.y(), scale or 1.0, self._get_native_pixel_size())
        self._begin_file_crop_read(
            QgsPointXY(cx, cy), scale, None, quiet=True, show_busy=False)

    def _prewarm_online_manual_encode(self, center) -> None:
        """Same prewarm for a layer QGIS renders, read through a private copy.

        A file raster starts a session with its first crop already in hand,
        while a basemap user paid the whole imagery read at the click. The copy
        is what makes the difference payable early: the tiles it downloads sit
        in the shared cache, so the user's own layer answers the same ground
        almost at once, and the layer on screen is never touched.

        Best effort in every direction. With no copy to be had, or with the
        served switch off, nothing is read and the first click drives its own
        fetch, exactly as before. No resolution is named, so the fetch puts the
        window on the shared grid itself, which is the window a click with
        nothing in hand asks for.
        """
        from ...core.online_layer_twin import online_prewarm_enabled

        if not online_prewarm_enabled():
            return
        QgsMessageLog.logMessage(
            "Prewarming first crop at view center through a private copy "
            "of the rendered layer",
            "AI Segmentation", level=Qgis.MessageLevel.Info
        )
        self._extract_and_encode_crop(
            center, mupp_override=None, show_busy=False, quiet=True)

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
        (fetch and worker are mutually exclusive). A file crop READ is released
        the same way: its thread does finish, but it touches nothing, and its
        completion no-ops once the lock generation moved. A new encode therefore
        cannot race the draining one."""
        self._ensure_manual_encode_state()
        self._manual_encode_gen += 1
        self._pending_encode = None
        self._discard_pending_manual_click()
        self._release_online_fetch()
        self._release_crop_read()
        # A draining WORKER keeps the transport lock until its own completion
        # fires (pipe safety), but the session that showed the busy cursor is
        # gone: pop the cursor NOW so leaving the mode never keeps the
        # hourglass for the encode's remaining seconds. The completion reads
        # _encode_cursor_set (pending was nulled above) and skips its own
        # restore, so the pair stays balanced.
        if (self._manual_encode_worker is not None and getattr(self, "_encode_cursor_set", False)):
            self._encode_cursor_set = False
            try:
                QApplication.restoreOverrideCursor()
            except Exception:  # nosec B110 -- cursor restore is best-effort
                pass
        # These describe crops of the session being torn down: a new session
        # must never skip its own encode over one of them, and a queued crop
        # belongs to a gesture that no longer exists.
        self._queued_crop_request = None
        self._inflight_crop_window = None
        self._encoded_crop_window = None
        # The panel note belongs to the session being torn down. Cleared
        # unconditionally: the note has no lock to protect, only a line to
        # remove, and leaving it would outlive its own encode.
        self._set_manual_encoding_note(False)

    def _drop_inflight_crop_for_gesture(self) -> None:
        """Give up the crop being read so a gesture the user just made runs now.

        A busy pipe never blocks the gesture itself: committing a shape reads
        session state and touches no predictor. What the read carries is a click
        that has not landed yet plus the crop transition it was going to replay
        into, and both belong to the shape the gesture is closing, so both go.
        The generation bump makes the completion drop its tail and its deferred
        click, while each owner's own release site still hands the lock back.

        The crop the predictor holds is unknown afterwards, so the two facts
        that claim to know it go with it and the next click re-reads. One
        re-read is cheap next to a gesture nobody can see refused.
        """
        self._invalidate_manual_encode()
        self._current_crop_info = None
        self._encoded_crop_window = None
        # The crop this flag described is gone, and a flag left standing would
        # let a later gesture abandon a crop somebody did ask for.
        self._speculative_manual_crop = False
        QgsMessageLog.logMessage(
            "Dropped the crop being read: a shape was committed instead",
            "AI Segmentation", level=Qgis.MessageLevel.Info)

    # ---- Transport-lock watchdog --------------------------------------------
    # `_encoding_in_progress` has exactly one release site per owner (worker
    # completion, online-fetch release). If that site never runs (a worker
    # thread that died without delivering `done`, a wedged subprocess pipe),
    # the lock and the busy cursor were stranded FOREVER: every click dropped,
    # hourglass stuck, and no code path ever recovered. The watchdog is the
    # last-resort self-heal for exactly that.

    def _arm_encode_watchdog(self) -> None:
        """(Re)arm the lock watchdog when the transport lock is taken. Restamps
        the hold start; idempotent while a beat chain is already scheduled."""
        import time
        self._encode_lock_since = time.monotonic()
        self._encode_watchdog_strikes = 0
        if getattr(self, "_encode_watchdog_armed", False):
            return
        self._encode_watchdog_armed = True
        self._schedule_encode_watchdog()

    def _schedule_encode_watchdog(self) -> None:
        """Queue the next watchdog beat (seam for tests; the QTimer holds the
        bound method alive until it fires)."""
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(ENCODE_WATCHDOG_INTERVAL_MS, self._encode_watchdog_tick)

    def _encode_watchdog_tick(self) -> None:
        """One watchdog beat (main thread). Goes quiet as soon as the lock is
        free; force-releases it when its owner is gone or past the ceiling.
        Two consecutive bad beats are required: a `done` emitted between beats
        is already queued on the event loop and always lands before the next
        singleShot turn, so a single bad beat can still resolve itself."""
        self._encode_watchdog_armed = False
        self._ensure_manual_encode_state()
        if not self._encoding_in_progress:
            return
        import time
        held_s = time.monotonic() - getattr(self, "_encode_lock_since", 0.0)
        worker = self._manual_encode_worker
        owner_alive = worker is not None and bool(worker.isRunning())
        owner_alive = owner_alive or self._online_fetch is not None
        # A file crop read owns the lock too. Without this it looked like an
        # ownerless lock and the watchdog force-released a read that was doing
        # exactly what it should, right in the middle of a slow raster.
        owner_alive = owner_alive or self._crop_read is not None
        if owner_alive and held_s < ENCODE_LOCK_CEILING_S:
            self._encode_watchdog_strikes = 0
        else:
            self._encode_watchdog_strikes = getattr(
                self, "_encode_watchdog_strikes", 0) + 1
            if self._encode_watchdog_strikes >= 2:
                self._force_release_encode_lock(
                    "held past ceiling" if owner_alive else "owner gone")
                return
        self._encode_watchdog_armed = True
        self._schedule_encode_watchdog()

    def _force_release_encode_lock(self, reason: str) -> None:
        """Last-resort self-heal: drop every piece of in-flight encode state
        and pop the cursor so the session is clickable again. The generation
        bump makes any late completion stale, and the completion's cursor
        restore is disabled via _encode_cursor_set, so a worker that turns out
        to still deliver can only no-op. If the owner was truly wedged, the
        next set_image runs the predictor's own transport recovery."""
        QgsMessageLog.logMessage(
            f"Encode lock force-released by watchdog ({reason})",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        self._ensure_manual_encode_state()
        self._manual_encode_gen += 1
        self._pending_encode = None
        self._manual_encode_worker = None
        self._discard_pending_manual_click()
        # An online fetch or a file read pops its own cursor in its release
        # below; only a worker-owned (or ownerless) lock still needs one here.
        crop_owned = self._online_fetch is not None or self._crop_read is not None
        self._release_online_fetch()
        self._release_crop_read()
        self._encoding_in_progress = False
        self._encode_lock_gen = None
        if not crop_owned and getattr(self, "_encode_cursor_set", False):
            try:
                QApplication.restoreOverrideCursor()
            except Exception:  # nosec B110 -- cursor restore is best-effort
                pass
        self._encode_cursor_set = False
        self._set_manual_encoding_note(False)

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

    def _reencode_plan(self, crop_status, raster_pt, include_click_in_zoom):
        """Shared crop-transition decision for the sync (headless/MCP) and
        async (interactive) re-encode paths. Computes the new crop center and
        resolution for the given status, applies the pre-encode state resets,
        and returns (center, mupp_or_scale, tail) where tail is the
        post-encode transition closure both paths run (inline for sync, as the
        ``on_encoded`` callback for async).

        include_click_in_zoom: on the zoom_changed path the interactive caller
        has NOT registered the click in prompts yet (it is replayed after the
        tail), so it must be added to the centering point set explicitly; the
        sync caller already registered it."""
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
            whole = self._untouched_shape_crop_window(raster_pt, all_pts)
            if whole is not None:
                center, mupp_or_scale = whole
            elif len(all_pts) > 1:
                center, mupp_or_scale = self._manual_crop_window_for_points(
                    all_pts)
            else:
                center, mupp_or_scale = self._grid_center_for_manual_click(
                    raster_pt, self._compute_initial_scale_factor())

            def _tail():
                self._invalidate_history_logits()

            return center, mupp_or_scale, _tail

        if crop_status == "outside_bounds":
            old_crop_info = self._current_crop_info
            old_history = list(self._mask_state_history)

            def _tail():
                self._freeze_active_crop(crop_info_override=old_crop_info)
                # Clear stale prompts from the old crop (#11, #35). The click
                # that follows (handler re-add or interactive replay) re-adds
                # the new point with correct polarity.
                self.prompts.clear()
                self._active_crop_points_positive = []
                self._active_crop_points_negative = []
                # Preserve history so the empty-mask rollback path still works.
                self._mask_state_history = old_history
                self._invalidate_history_logits()
                self.current_mask = None
                self.current_low_res_mask = None

            # Asked BEFORE the tail freezes anything: while the shape is whole,
            # the window has to hold all of it, not just the click.
            whole = self._untouched_shape_crop_window(raster_pt)
            if whole is not None:
                return whole[0], whole[1], _tail
            center, scale = self._grid_center_for_manual_click(
                raster_pt, self._compute_initial_scale_factor())
            return center, scale, _tail

        # zoom_changed: re-encode same crop at new resolution, keep all points.
        old_crop_info = self._current_crop_info
        old_mask = self.current_mask
        all_pts = [
            (p[0], p[1]) for p in
            self.prompts.positive_points + self.prompts.negative_points
        ]
        if include_click_in_zoom and (raster_pt.x(), raster_pt.y()) not in all_pts:
            all_pts.append((raster_pt.x(), raster_pt.y()))
        whole = self._untouched_shape_crop_window(raster_pt, all_pts)
        if whole is not None:
            new_center, mupp_or_scale = whole
        elif len(all_pts) > 1:
            new_center, mupp_or_scale = self._manual_crop_window_for_points(
                all_pts)
        else:
            new_center, mupp_or_scale = self._grid_center_for_manual_click(
                raster_pt, self._compute_initial_scale_factor())
        self.current_low_res_mask = None
        # The old full-res mask belongs to the OLD crop frame; the next click is
        # in the NEW frame. Null it so Progressive Merge skips the first
        # post-zoom click rather than merge two masks that cover different
        # ground (both are padded to the model's square, so a shape check would
        # not catch the frame mismatch). The transferred low-res mask below
        # still carries the shape context into the re-encode.
        self.current_mask = None

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

        return new_center, mupp_or_scale, _tail

    def _handle_reencode_sync(self, crop_status, raster_pt):
        """Synchronous re-encode (headless/MCP). Encodes, then runs the
        transition tail inline. Returns True on success. The triggering click
        is already registered in prompts here, so the zoom centering must not
        re-add it (include_click_in_zoom=False)."""
        center, mupp_or_scale, tail = self._reencode_plan(
            crop_status, raster_pt, include_click_in_zoom=False)
        if not self._extract_and_encode_crop(center, mupp_override=mupp_or_scale):
            return False
        tail()
        return True

    def _begin_async_reencode(self, crop_status, raster_pt):
        """Interactive re-encode: read the crop on a worker, encode it on a
        worker, and defer the crop-transition tail to the completion (as the
        ``on_encoded`` callback); the remembered click is replayed after the
        tail. The triggering click is NOT registered here (the replay registers
        it once), so it stays undone in prompts/history until the crop is ready
        and must join the zoom centering explicitly (include_click_in_zoom).
        Returns True when the encode was started, False when extraction failed
        synchronously."""
        center, mupp_or_scale, tail = self._reencode_plan(
            crop_status, raster_pt, include_click_in_zoom=True)
        return self._extract_and_encode_crop(
            center, mupp_override=mupp_or_scale, on_encoded=tail)
