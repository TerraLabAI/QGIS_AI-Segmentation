"""Live Automatic run: the on-canvas selection layer, the stitcher thread that
turns arriving tile geometry into drawable objects, and the provider write that
puts them on the map.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).

The division of labour matters here. NOTHING per-object runs on the GUI thread
during a run: merging fragments, measuring objects, gating them and applying the
run's shape preset all happen on the LiveStitchThread, which hands this file a
queue of ready geometries. What is left is a provider write per repaint tick,
proportional to the objects that CHANGED, not to the size of the set.
"""
from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsFeature,
    QgsField,
    QgsMessageLog,
    QgsProject,
    QgsVectorLayer,
)

from ...core.i18n import tr
from ...workers.live_stitch_thread import (
    DELTA_REMOVE,
    STITCH_JOIN_TIMEOUT_MS,
    LiveStitchThread,
)
from .shared import (
    _AUTO_LIVE_FRAME_COST_RATIO,
    _AUTO_LIVE_REPAINT_MAX_MS,
    _AUTO_LIVE_REPAINT_MS,
    _FIELD_TYPE_DOUBLE,
    _FIELD_TYPE_INT,
    _FIELD_TYPE_STRING,
    _add_features_with_ids,
    _apply_fast_render,
    _notify_provider_write,
    park_orphaned_worker,
)

#: One cooperative slice of the finalize's wait on the stitcher, in ms. Short
#: enough that the event loop keeps painting, long enough that the tail of a run
#: does not turn into a timer ladder.
_STITCH_WAIT_SLICE_MS = 25

#: How long the end of a run waits for the stitcher to fold its whole backlog
#: before dropping what is still QUEUED. It never shortens the wait for the tile
#: being folded right now: the merger belongs to that thread until it leaves.
_STITCH_DRAIN_BUDGET_S = 30.0


def _diff_live_fid_map(old_map: dict, current: list):
    """Diff the previous live selection set against the current one.

    Pure bookkeeping (no QGIS types), so a preview updates the provider
    incrementally instead of truncating + re-adding every feature each repaint.
    Used by the REVIEW push, which rebuilds a whole visible set per reslice; the
    live run gets its delta from the stitcher instead.

    ``old_map`` maps a merger keeper fid -> (provider_fid, stamp, is_full,
    score) for every object currently ON the layer. ``current`` is the ordered
    list of (merger_fid, stamp, is_full, score) for every renderable-visible
    object this repaint. ``stamp`` identifies the keeper geometry (bbox + vertex
    count) and ``is_full`` is False for a cheap budget-fallback render, so a
    later full refine of the SAME keeper still counts as a geometry change.

    Returns (adds, geom_changes, attr_changes, deletes, kept_map):
      adds         = merger fids new since last repaint (need a feature created)
      geom_changes = {provider_fid: merger_fid} whose stamp or is_full changed
      attr_changes = {provider_fid: merger_fid} whose score changed
      deletes      = provider fids gone from the layer (retired or gated out)
      kept_map     = the surviving old entries with stamp/is_full/score refreshed
                     (the caller folds the added fids into it).
    """
    adds = []
    geom_changes = {}
    attr_changes = {}
    kept_map = {}
    seen = set()
    for fid, stamp, is_full, score in current:
        seen.add(fid)
        rec = old_map.get(fid)
        if rec is None:
            adds.append(fid)
            continue
        prov_fid, old_stamp, old_is_full, old_score = rec
        if old_stamp != stamp or old_is_full != is_full:
            geom_changes[prov_fid] = fid
        if old_score != score:
            attr_changes[prov_fid] = fid
        kept_map[fid] = (prov_fid, stamp, is_full, score)
    deletes = [rec[0] for fid, rec in old_map.items() if fid not in seen]
    return adds, geom_changes, attr_changes, deletes, kept_map


def _latest_delta_per_object(deltas: list) -> tuple[list, dict]:
    """Collapse a delta batch to one entry per object, first-seen order.

    Several tiles can land between two repaint ticks, so the same object may
    appear more than once in one batch. Only its last state is drawn, and the
    collapse is what stops an object that was added AND then updated inside the
    batch from being written to the provider twice.
    """
    order: list = []
    latest: dict = {}
    for kind, fid, geom, score in deltas:
        if fid not in latest:
            order.append(fid)
        latest[fid] = (kind, geom, score)
    return order, latest


class AutoResultsMixin:
    """Live selection layer plus the stitcher that feeds it whole objects."""

    def _create_auto_selection_layer(self, source_layer) -> QgsVectorLayer | None:
        """Create an in-memory vector layer to display detection results live."""
        try:
            # Detections stream in numerically expressed in the resolved run
            # CRS, not necessarily the raster's own CRS: a memory provider
            # stores raw coordinates under the declared CRS with no
            # reprojection on insert, so this must match. Fall back to the
            # raster CRS when no run CRS has been resolved yet.
            crs_authid = self._auto_crs_authid or source_layer.crs().authid()
            layer = QgsVectorLayer(
                f"MultiPolygon?crs={crs_authid}",
                tr("Auto detection (live)"),
                "memory",
            )
            if not layer.isValid():
                return None
            pr = layer.dataProvider()
            pr.addAttributes([
                QgsField("label", _FIELD_TYPE_STRING),
                QgsField("score", _FIELD_TYPE_DOUBLE),
                # Stable per-detection identity (the merger's keeper fid, fixed
                # the moment an object first appears and shared by the live run
                # and the review). The Random display mode hues on it so an object
                # KEEPS its colour as tiles stream in AND across reslices
                # (truncate + re-add renumbers $id, so hueing on $id would
                # reshuffle every repaint).
                QgsField("det_id", _FIELD_TYPE_INT),
            ])
            layer.updateFields()

            # Live results render in the SAME display mode as the post-run
            # review (Random by default: one distinct colour per instance, so
            # building footprints read apart as they stream in tile by tile).
            # _auto_display_mode is seeded to Random at run start; if unset (an
            # early creation before any run) it falls back to the blue review
            # outline. Red stays reserved for the saved export.
            self._apply_review_display_mode(layer)
            # Keep pan/zoom smooth as detections pile up live and through review:
            # render-time simplification + a provider spatial index.
            _apply_fast_render(layer)

            # Private working layer: renders on canvas (the tree node below is
            # what the canvas bridge draws) but stays out of the Layers panel.
            # Must be flagged BEFORE the add so the panel never flashes it.
            from ...core.output_store import drop_from_snapping, mark_temp_layer
            mark_temp_layer(layer)
            QgsProject.instance().addMapLayer(layer, False)
            # Post-add: keep the scratch layer out of the snapping config (a
            # dangling entry there crashes the next project save; see helper).
            drop_from_snapping(layer)
            root = QgsProject.instance().layerTreeRoot()
            root.insertLayer(0, layer)
            # Fresh empty layer: the incremental review push starts from a
            # clean provider mapping (stale fids would corrupt the diff).
            self._review_fid_map = {}
            return layer
        except (RuntimeError, AttributeError):
            return None

    # ---- Ground the run is re-reading finer ----------------------------------

    def _on_auto_rescan_state(self, tile_idx: int, bbox, active: bool) -> None:
        """Slot: a tile came back holding more objects than one read can carry,
        so the run is re-reading its ground finer.

        That tile's own objects are withheld until the finer read lands (worker
        _settle_converted), which leaves the busiest part of a zone bare for as
        long as the re-read takes. The marks are tracked whatever the display
        flag says, so turning the flag on mid-run paints what is pending;
        _auto_rescan_band_allowed decides whether they reach the canvas.
        tile_idx below zero clears every mark (the run's terminal)."""
        rects = getattr(self, "_auto_rescan_rects", None)
        if rects is None:
            rects = self._auto_rescan_rects = {}
        if tile_idx < 0:
            rects.clear()
        elif not active:
            rects.pop(tile_idx, None)
        elif bbox:
            try:
                rects[tile_idx] = (float(bbox[0]), float(bbox[1]),
                                   float(bbox[2]), float(bbox[3]))
            except (TypeError, ValueError, IndexError):
                return
        self._repaint_auto_rescan_band()

    def _auto_rescan_band_allowed(self) -> bool:
        """Whether the re-read marks may be drawn.

        Off for users: overlapping tile rectangles over their own results read
        as a broken render, not as progress, and the dock already says the run
        is working. Same developer flag as the review's tile grid toggle."""
        from qgis.PyQt.QtCore import QSettings
        try:
            return bool(QSettings().value(
                "TerraLab/auto_debug_tiles", False, type=bool))
        except (RuntimeError, TypeError, ValueError):
            return False

    def _repaint_auto_rescan_band(self) -> None:
        """Paint every marked tile as ONE rubber band geometry.

        One band, one setToGeometry: each addGeometry recomputes the band's
        bounding rect and schedules a canvas update, and a dense zone marks
        tens of tiles at once."""
        from qgis.core import QgsGeometry, QgsRectangle
        from qgis.gui import QgsRubberBand

        from ...core.qt_compat import PolygonGeometry
        from ..canvas_palette import RESCAN_FILL, RESCAN_STROKE

        rects = getattr(self, "_auto_rescan_rects", None) or {}
        band = getattr(self, "_auto_rescan_band", None)
        layer = self._auto_selection_layer
        if not rects or layer is None or not self._auto_rescan_band_allowed():
            if band is not None:
                self._safe_remove_rubber_band(band)
                self._auto_rescan_band = None
            return
        try:
            if band is None:
                band = QgsRubberBand(self.iface.mapCanvas(), PolygonGeometry)
                band.setFillColor(RESCAN_FILL)
                band.setStrokeColor(RESCAN_STROKE)
                band.setWidth(1)
                self._auto_rescan_band = band
            parts = [QgsGeometry.fromRect(QgsRectangle(*r))
                     for r in rects.values()]
            # CRS overload, not the layer: setToGeometry has no QgsMapLayer
            # overload on any QGIS build, and the bboxes arrive in the run CRS,
            # which is the selection layer's.
            band.setToGeometry(QgsGeometry.collectGeometry(parts), layer.crs())
        except (RuntimeError, AttributeError, TypeError):
            self._clear_auto_rescan_band()

    def _clear_auto_rescan_band(self) -> None:
        """Drop every mark and the band itself. Called on every run teardown."""
        self._auto_rescan_rects = {}
        band = getattr(self, "_auto_rescan_band", None)
        self._auto_rescan_band = None
        if band is not None:
            self._safe_remove_rubber_band(band)

    def _remove_auto_selection_layer(self) -> None:
        """Remove the live selection layer from the project (if present)."""
        # An armed hand edit never outlives the selection layer it points at:
        # every teardown path (export, discard, zone redraw, fresh run,
        # unload) funnels through here.
        # Close the QGIS edit bridge HERE rather than trusting each of the
        # seventeen callers to do it first. Removing a layer that still has an
        # open edit buffer, a live vertex tool bound to it and a deferred write
        # in flight is a use-after-free that takes QGIS down, and the Reshape
        # lane reached this point with the bridge open. The call is idempotent,
        # so the callers that already do it lose nothing.
        try:
            self._abort_qgis_edit_bridge_if_active()
        except (RuntimeError, AttributeError):
            pass
        try:
            self._disarm_shape_tool()
        except (RuntimeError, AttributeError):
            pass
        # The review's pushes repaint through the same pacer the live run uses,
        # so the canvas connection outlives the run. Drop it with the layer it
        # paints, or a finished review leaves a slot firing on every frame for
        # the rest of the session. Any later repaint re-connects it.
        # Guarded like the two above: nothing here may skip the removal below.
        # A leaked selection layer is Private, so it keeps painting detections
        # the user cannot reach in the Layers panel to delete.
        try:
            self._disconnect_live_repaint_pacer()
            self._clear_auto_rescan_band()
        except Exception:  # nosec B110 -- the layer goes even if a step fails
            pass
        layer = self._auto_selection_layer
        self._auto_selection_layer = None
        self._review_fid_map = {}
        if layer is None:
            return
        try:
            lid = layer.id()
            QgsProject.instance().removeMapLayer(lid)
        except (RuntimeError, AttributeError):
            pass

    # ---- The live stitcher ---------------------------------------------------

    def _start_auto_stitcher(self, geo_transform: dict | None = None) -> None:
        """Hand the run's merger to the stitcher thread and start it.

        From here until _finish_auto_stitcher returns, the merger belongs to
        that thread: nothing on the GUI may read it. Called once per run, right
        before the detection worker starts, so the first tile already has
        somewhere to go.

        ``geo_transform`` supplies the point the run's metres-per-CRS-unit
        factor is measured at. One factor for the whole zone is enough: it moves
        with latitude, not across a drawn zone.
        """
        self._abort_auto_stitcher()
        if self._auto_merger is None:
            return
        params = self._fresh_review_params()
        factor = self._auto_run_metres_per_unit(geo_transform)
        hard_cov = 0.0
        if self._auto_is_exemplar_only:
            from ...core.detection_policy import hard_tile_coverage
            from ...workers.auto_detection_worker import _HARD_TILE_COVERAGE
            hard_cov = hard_tile_coverage(_HARD_TILE_COVERAGE)
        # A previous run's shapes must never be seeded into this one's review.
        # Stale until the finalize proves otherwise, so a run that never reaches
        # the check reuses nothing.
        self._auto_stitch_shapes = None
        self._auto_stitch_shape_px = 0.0
        self._auto_stitch_shape_mpu = 0.0
        self._auto_stitch_shapes_stale = True
        stitcher = LiveStitchThread(
            self._auto_merger,
            params=params,
            pixel_size=self._auto_refine_pixel_size(),
            metres_per_unit=factor,
            crs_authid=self._auto_crs_authid or "EPSG:4326",
            unit_aspect=self._auto_run_unit_aspect(geo_transform),
            retain_fragments=bool(getattr(self, "_auto_retain_raw", False)),
            retain_coverage=bool(self._auto_is_exemplar_only),
            tile_ground_area=float(getattr(self, "_auto_tile_ground_area", 0.0)),
            hard_coverage=hard_cov,
        )
        from qgis.PyQt.QtCore import Qt
        stitcher.batch_ready.connect(
            self._on_auto_stitch_batch, Qt.ConnectionType.QueuedConnection)
        self._auto_stitcher = stitcher
        stitcher.start()

    def _auto_run_centre_xy(self, geo_transform: dict | None):
        """Run CRS coordinates of the middle of the run grid, else of the drawn
        zone, else None. The point the run's ground measures are taken at: both
        of them move with latitude, and neither moves across one zone."""
        bbox = (geo_transform or {}).get("bbox")
        if bbox and len(bbox) == 4:
            return ((float(bbox[0]) + float(bbox[2])) / 2.0,
                    (float(bbox[1]) + float(bbox[3])) / 2.0)
        if self._auto_clip_polygon is not None:
            try:
                point = self._auto_clip_polygon.boundingBox().center()
                return (point.x(), point.y())
            except (RuntimeError, AttributeError):
                return None
        return None

    def _auto_run_metres_per_unit(self, geo_transform: dict | None) -> float:
        """Ground metres per unit of the run CRS, at the centre of the run grid.

        Falls back to the drawn zone, then to 1.0 (which reads every metre
        setting as a CRS unit, the behaviour before the conversion existed).
        """
        centre = self._auto_run_centre_xy(geo_transform)
        if centre is None:
            return 1.0
        try:
            return self._auto_crs_metres_per_unit(centre[0], centre[1])
        except (RuntimeError, AttributeError, TypeError):
            return 1.0

    def _auto_run_unit_aspect(self, geo_transform: dict | None) -> float:
        """How much longer one y unit of the run CRS is than one x unit, at the
        centre of the run grid. 1.0 for a projected run and on any failure,
        which squares on raw coordinates: right there, wrong in a geographic
        CRS (see core.layer_conventions.ground_unit_aspect)."""
        centre = self._auto_run_centre_xy(geo_transform)
        if centre is None:
            return 1.0
        try:
            return self._auto_crs_unit_aspect(centre[0], centre[1])
        except (RuntimeError, AttributeError, TypeError):
            return 1.0

    def _finish_auto_stitcher(self, timeout_ms: int = STITCH_JOIN_TIMEOUT_MS) -> bool:
        """Let the stitcher finish everything queued, then take the merger back.

        Returns False while it is still folding, so a caller with an event loop
        to protect can come back in a moment. Nothing may read the merger until
        this has returned True. Idempotent.
        """
        stitcher = self._auto_stitcher
        if stitcher is None:
            return True
        try:
            stitcher.finish()
            if not stitcher.join_run(timeout_ms):
                return False
        except RuntimeError:
            # The thread object died under us (teardown raced us): the merger is
            # nobody's now, so the caller may proceed.
            self._auto_stitcher = None
            return True
        self._auto_stitcher = None
        self._harvest_auto_stitch_counters(stitcher)
        return True

    def _abort_auto_stitch_queue(self) -> None:
        """Skip what is still queued, but leave the baton where it is.

        For a caller that must go on to READ the merger. The thread stays its
        owner until it has left the tile it is folding, so the caller has to
        keep polling _finish_auto_stitcher; only that returning True means the
        merger is safe to touch. Abort alone never makes it safe.
        """
        stitcher = self._auto_stitcher
        if stitcher is None:
            return
        try:
            stitcher.abort()
        except RuntimeError:
            pass

    def _abort_auto_stitcher(self) -> None:
        """Drop the stitcher without waiting for its queue (teardown, cancel).

        For a caller that will NOT read the merger afterwards. Whatever was
        already folded stays in it, which is what the user was billed for; the
        queue is discarded. A thread that will not stop is PARKED rather than
        dropped: garbage-collecting a running QThread aborts QGIS.
        """
        stitcher = self._auto_stitcher
        self._auto_stitcher = None
        if stitcher is None:
            return
        try:
            stitcher.abort()
            if not stitcher.join_run(STITCH_JOIN_TIMEOUT_MS):
                park_orphaned_worker(stitcher)
        except RuntimeError:
            pass

    def _harvest_auto_stitch_counters(self, stitcher) -> None:
        """Take back the run-wide numbers the stitcher accumulated.

        It is the only place that saw every fragment once the fold moved off the
        GUI, so the count-vs-map decision and the run summary read them here.
        """
        if stitcher.raw_count > self._auto_raw_count:
            self._auto_raw_count = stitcher.raw_count
        if self._auto_is_exemplar_only:
            self._auto_raw_n_total = stitcher.raw_n_total
            self._auto_raw_cov_sum = stitcher.raw_coverage_sum
            self._auto_raw_cov_sq_sum = stitcher.raw_coverage_sq_sum
        if getattr(self, "_auto_retain_raw", False):
            self._auto_raw_fragments = stitcher.raw_fragments
        # The shapes it already built, for the finalize to reuse instead of
        # rebuilding them on the GUI thread. Taken with the two scalars they
        # were built at, because the finalize resolves both again and only a
        # match makes them the same shape (see _seed_review_refine_cache).
        self._auto_stitch_shapes = stitcher.shaped_geoms
        self._auto_stitch_shape_px = stitcher.shape_pixel_size
        self._auto_stitch_shape_mpu = stitcher.shape_metres_per_unit

    # ---- Auto detection signal handlers -------------------------------------

    def _on_auto_tile_completed(self, tile_idx: int, tagged_detections: list) -> None:
        """Slot: one tile finished. tagged_detections = [(geom_wkb: bytes, score)].

        This is the WHOLE of the GUI thread's per-tile work: hand the fragments
        to the stitcher and return. Merging, measuring, gating and shaping all
        happen on that thread. Keep new work out of this slot.
        """
        stitcher = self._auto_stitcher
        if stitcher is None or not tagged_detections:
            return
        stitcher.submit(tagged_detections, self._auto_refine_pixel_size())

    def _on_auto_stitch_batch(self) -> None:
        """Slot: the stitcher finished some objects. Ask for a repaint on the
        coalesced tick, so a burst of tiles becomes one provider write."""
        self._request_auto_live_repaint()

    def _drain_auto_tiles_now(self) -> None:
        """Block until the stitcher has folded everything, because the run is
        ending and the caller is about to read the merger.

        The headless path has no UI to protect and its caller blocks anyway; the
        interactive path waits cooperatively instead (the finalize "drain"
        phase). Both get the same patience and say the same thing when they run
        out of it: a headless run that quietly dropped its backlog would hand
        the caller fewer objects than the run billed for.
        """
        import time as _t

        deadline = _t.monotonic() + _STITCH_DRAIN_BUDGET_S
        hard_deadline = None
        while not self._finish_auto_stitcher(timeout_ms=_STITCH_WAIT_SLICE_MS):
            now = _t.monotonic()
            if hard_deadline is None:
                if now < deadline:
                    continue
                QgsMessageLog.logMessage(
                    "Auto detection: live stitcher did not finish in "
                    f"{int(_STITCH_DRAIN_BUDGET_S)}s; finalizing what it folded",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                self._abort_auto_stitch_queue()
                hard_deadline = now + STITCH_JOIN_TIMEOUT_MS / 1000.0
            elif now >= hard_deadline:
                # It is not even leaving the tile it is on. Park it and read the
                # merger anyway: a run held open for ever is worse than a stale
                # read of results the user already paid for.
                QgsMessageLog.logMessage(
                    "Auto detection: live stitcher would not stop; keeping "
                    "results as they stand", "AI Segmentation",
                    level=Qgis.MessageLevel.Warning)
                self._abort_auto_stitcher()
                return

    def _apply_auto_live_deltas(self) -> None:
        """Draw whatever the stitcher finished since the last tick.

        Every object arrives already merged, gated and shaped, so this is a
        provider write and a repaint request. The cost tracks the objects that
        CHANGED, which is why the cadence no longer has to stretch as the set
        grows.
        """
        layer = self._auto_selection_layer
        stitcher = self._auto_stitcher
        if layer is None or stitcher is None:
            return
        deltas = stitcher.take_deltas()
        if not deltas:
            return
        try:
            if not layer.isValid():
                return
            pr = layer.dataProvider()
            fields = layer.fields()
            score_idx = fields.indexOf("score")
            fid_map = self._auto_live_fid_map
            order, latest = _latest_delta_per_object(deltas)

            deletes: list = []
            adds: list = []            # (merger_fid, QgsFeature)
            geom_changes: dict = {}    # provider_fid -> geometry
            attr_changes: dict = {}    # provider_fid -> score
            for fid in order:
                kind, geom, score = latest[fid]
                if kind == DELTA_REMOVE:
                    rec = fid_map.pop(fid, None)
                    if rec is not None:
                        deletes.append(rec[0])
                    continue
                rec = fid_map.get(fid)
                if rec is None:
                    feat = QgsFeature(fields)
                    feat.setGeometry(geom)
                    # det_id is the merger's stable keeper fid: the renderer hues
                    # on it, so an object keeps its colour as it grows.
                    feat.setAttributes(["", score, fid])
                    adds.append((fid, feat))
                    continue
                prov_fid, old_score = rec
                geom_changes[prov_fid] = geom
                if old_score != score:
                    attr_changes[prov_fid] = score
                fid_map[fid] = (prov_fid, score)

            changed = False
            if deletes:
                pr.deleteFeatures(deletes)
                changed = True
            if adds:
                # The assigned provider fids come back on the RETURNED feature
                # copies only (addFeatures never mutates its inputs); without
                # them every later delta re-adds the same objects and the
                # stacked translucent fills read as opaque.
                ok, added = _add_features_with_ids(
                    pr, [feat for _, feat in adds])
                if ok and len(added) == len(adds):
                    for (fid, _feat), out in zip(adds, added):
                        pfid = out.id()
                        if pfid is not None and pfid >= 0:
                            fid_map[fid] = (pfid, latest[fid][2])
                changed = True
            if geom_changes:
                pr.changeGeometryValues(geom_changes)
                changed = True
            if attr_changes and score_idx >= 0:
                pr.changeAttributeValues(
                    {pfid: {score_idx: sc} for pfid, sc in attr_changes.items()})
                changed = True
            if not changed:
                return
            # Live "found so far" count in the run label, on the same tick as
            # the repaint: a slow zone never reads as dead. The tracked fid map
            # IS the current on-layer visible set.
            if self.dock_widget is not None:
                try:
                    # The run's own prompt, as sent (the guard rewrites the box
                    # before Detect, so this is the word the service was
                    # given). Empty on an example-only run: there is no word to
                    # quote, and the line stays "N found so far".
                    self.dock_widget.set_auto_run_found_count(
                        (self._auto_run_ctx or {}).get("prompt") or "",
                        len(fid_map))
                except (RuntimeError, AttributeError):
                    pass
            # No updateExtents() here: it rescans every feature (O(N)). Rendering
            # requests features by the canvas viewport via the spatial index, NOT
            # the layer's cached extent, so the live preview draws correctly
            # without it. The extent is refreshed once at review
            # (_push_review_geoms) for zoom-to-layer.
            _notify_provider_write(layer)
            self._repaint_live_layer(layer)
        except (RuntimeError, AttributeError):
            # Live-preview drawing is best-effort (the layer can be deleted from
            # under us mid-run). Narrow catch: do not mask geometry/merger bugs.
            pass

    def _repaint_live_layer(self, layer) -> None:
        """Show the objects that just landed WITHOUT killing the frame in flight.

        Serves the live run AND the review's pushes (``_push_review_geoms``):
        both write a delta to the provider faster than a dense set can be drawn,
        and both have the same way of making that visible.

        ``triggerRepaint()`` makes the canvas cancel whatever it is drawing and
        start over. A zone of many thousands of objects takes longer to draw
        than the live cadence, so every tick threw away a frame that was never
        finished: the polygons blinked in and out, and a zoom could not settle
        until the run ended. Measured on 17655 objects, whole zone visible: 1.4 s
        to draw, against a tick every 1.2 s.

        So a tick that lands mid-draw only marks the layer dirty (the deferred
        flag: the cached image is dropped, the canvas is NOT told to refresh)
        and remembers there is something new to show. The real repaint goes out
        when the canvas says it finished, which paces the preview at exactly the
        rate the map can actually draw."""
        canvas = None
        try:
            canvas = self.iface.mapCanvas()
        except (RuntimeError, AttributeError):
            canvas = None
        if canvas is None:
            layer.triggerRepaint()
            return
        self._connect_live_repaint_pacer(canvas)
        try:
            drawing = bool(canvas.isDrawing())
        except (RuntimeError, AttributeError):
            drawing = False
        if drawing or not self._live_repaint_allowed_now():
            # Either a frame is in flight (asking now would kill it) or the last
            # one was expensive enough that the run needs the machine back for a
            # moment. Mark the layer dirty and let the cool-down push it.
            self._auto_live_repaint_pending = True
            try:
                layer.triggerRepaint(True)
            except (RuntimeError, AttributeError, TypeError):
                pass
            self._arm_live_repaint_cooldown()
            return
        self._auto_live_repaint_pending = False
        self._note_live_repaint_sent()
        layer.triggerRepaint()

    # ---- Frame-cost pacing ---------------------------------------------------

    def _live_repaint_allowed_now(self) -> bool:
        """Whether the preview has sat out its share since the last repaint."""
        import time
        not_before = getattr(self, "_auto_live_repaint_not_before", 0.0)
        return time.monotonic() >= not_before

    def _note_live_repaint_sent(self) -> None:
        """Open the cool-down for the frame this repaint is about to cost.

        Sized on the last frame we MEASURED, so the preview is free on a small
        zone (a 40 ms frame gives a 120 ms cool-down, invisible) and steps back
        on a big one (a 1.5 s frame gives 4.5 s), which is exactly when the run
        needs its cores for folding tiles."""
        import time
        frame_s = getattr(self, "_auto_live_frame_s", 0.0)
        cool = min(_AUTO_LIVE_REPAINT_MAX_MS / 1000.0,
                   frame_s * _AUTO_LIVE_FRAME_COST_RATIO)
        self._auto_live_repaint_not_before = time.monotonic() + cool
        self._auto_live_frame_started = time.monotonic()

    def _arm_live_repaint_cooldown(self) -> None:
        """Wake up once when the cool-down ends, so a deferred repaint is not
        left waiting for the next tile (the last tiles of a run are minutes
        apart, and the preview would look frozen)."""
        import time

        from qgis.PyQt.QtCore import QTimer
        if getattr(self, "_auto_live_cooldown_timer", None) is None:
            timer = QTimer(self.iface.mainWindow())
            timer.setSingleShot(True)
            timer.timeout.connect(self._on_live_repaint_cooldown)
            self._auto_live_cooldown_timer = timer
        if self._auto_live_cooldown_timer.isActive():
            return
        wait_ms = int(max(0.0, getattr(self, "_auto_live_repaint_not_before", 0.0) - time.monotonic()) * 1000) + 20
        self._auto_live_cooldown_timer.start(wait_ms)

    def _on_live_repaint_cooldown(self) -> None:
        """The cool-down is over: show what landed while the preview sat out."""
        if not getattr(self, "_auto_live_repaint_pending", False):
            return
        layer = getattr(self, "_auto_selection_layer", None)
        if layer is None:
            return
        try:
            if not layer.isValid():
                return
            canvas = self.iface.mapCanvas()
            if canvas is not None and canvas.isDrawing():
                return  # the finished-frame signal will take it from here
            self._auto_live_repaint_pending = False
            self._note_live_repaint_sent()
            layer.triggerRepaint()
        except (RuntimeError, AttributeError):
            pass

    def _connect_live_repaint_pacer(self, canvas) -> None:
        """Listen once for "the canvas finished a frame". Held as the canvas we
        connected to, so the teardown disconnects the same object even if the
        canvas has been swapped under us."""
        if self._auto_live_pacer_canvas is canvas:
            return
        self._disconnect_live_repaint_pacer()
        try:
            canvas.mapCanvasRefreshed.connect(self._on_live_canvas_refreshed)
            # renderStarting gives the frame's true start, so the cool-down is
            # sized on what drawing costs and not on how long we waited.
            try:
                canvas.renderStarting.connect(self._on_live_canvas_render_started)
            except (RuntimeError, AttributeError, TypeError):
                pass
            self._auto_live_pacer_canvas = canvas
        except (RuntimeError, AttributeError, TypeError):
            self._auto_live_pacer_canvas = None

    def _disconnect_live_repaint_pacer(self) -> None:
        """Drop the canvas connection. Every run teardown calls this: a live
        slot left connected to a finished run repaints a layer that is gone."""
        canvas = self._auto_live_pacer_canvas
        self._auto_live_pacer_canvas = None
        self._auto_live_repaint_pending = False
        if canvas is None:
            return
        try:
            canvas.mapCanvasRefreshed.disconnect(self._on_live_canvas_refreshed)
        except (RuntimeError, AttributeError, TypeError):
            pass
        try:
            canvas.renderStarting.disconnect(self._on_live_canvas_render_started)
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _on_live_canvas_render_started(self) -> None:
        """A frame started drawing: the clock the cool-down is sized on."""
        import time
        self._auto_live_frame_started = time.monotonic()

    def _on_live_canvas_refreshed(self) -> None:
        """A frame finished. Record what it cost, then push the deferred repaint
        if data landed while it was drawing AND the preview has sat out its
        share. Repainting unconditionally here made every frame ask for the next
        one, which is how the preview came to own the machine on a long run."""
        import time
        started = getattr(self, "_auto_live_frame_started", 0.0)
        if started:
            measured = max(0.0, time.monotonic() - started)
            prev = getattr(self, "_auto_live_frame_s", 0.0)
            # Follow a slowdown at once, come back down slowly: a cheap frame
            # straight after an expensive one is usually a partial redraw, and
            # believing it would put the preview back in charge of the machine.
            self._auto_live_frame_s = (
                measured if measured > prev else prev * 0.7 + measured * 0.3)
            self._auto_live_frame_started = 0.0
        if not self._auto_live_repaint_pending:
            return
        if not self._live_repaint_allowed_now():
            self._arm_live_repaint_cooldown()
            return
        self._auto_live_repaint_pending = False
        layer = getattr(self, "_auto_selection_layer", None)
        if layer is None:
            return
        try:
            if layer.isValid():
                self._note_live_repaint_sent()
                layer.triggerRepaint()
        except (RuntimeError, AttributeError):
            pass

    def _request_auto_live_repaint(self) -> None:
        """Coalesce live-preview writes onto one timer tick.

        The tick applies a DELTA, so its cost follows the arriving objects and
        not the size of the set. What still grows is the canvas render itself,
        so the cadence eases off a little on very large sets rather than
        re-rendering thousands of polygons several times a second."""
        from qgis.PyQt.QtCore import QTimer
        if self._auto_repaint_timer is None:
            self._auto_repaint_timer = QTimer(self.iface.mainWindow())
            self._auto_repaint_timer.setSingleShot(True)
            self._auto_repaint_timer.timeout.connect(self._apply_auto_live_deltas)
        if not self._auto_repaint_timer.isActive():
            n = len(self._auto_live_fid_map)
            interval = _AUTO_LIVE_REPAINT_MS
            if n >= 1500:
                interval = min(1200, interval + (n // 1500) * 200)
            self._auto_repaint_timer.start(interval)

    def _pause_preview_jobs(self) -> None:
        """Suspend the canvas's preview jobs for the duration of a run (they
        render the just-outside-the-view margin after each repaint, which on
        an online basemap also re-fetches tiles; wasted work during live tile
        streaming). The user's setting is remembered and restored; idempotent
        and version-defensive (older builds without the getter no-op)."""
        if getattr(self, "_preview_jobs_paused", None) is not None:
            return  # already paused (nested run paths)
        try:
            canvas = self.iface.mapCanvas()
            prev = bool(canvas.previewJobsEnabled())
            if prev:
                canvas.setPreviewJobsEnabled(False)
            self._preview_jobs_paused = prev
        except (RuntimeError, AttributeError):
            self._preview_jobs_paused = None

    def _resume_preview_jobs(self) -> None:
        """Restore the canvas preview-jobs setting saved by the pause (no-op
        when nothing was paused or the user had them off anyway)."""
        prev = getattr(self, "_preview_jobs_paused", None)
        self._preview_jobs_paused = None
        if prev:
            try:
                self.iface.mapCanvas().setPreviewJobsEnabled(True)
            except (RuntimeError, AttributeError):
                pass

    def _stop_auto_live_pump(self) -> None:
        """Stop the live machinery: the stitcher thread, the coalesced repaint
        timer, and the provider mapping. Does NOT touch the finalize/preview
        generations, so the finalize state machine can call it from its own
        step without invalidating itself."""
        self._resume_preview_jobs()
        self._disconnect_live_repaint_pacer()
        self._abort_auto_stitcher()
        # Invalidate the incremental provider mapping: the layer is recreated
        # fresh per run, and the review rebuilds it (truncate + re-add), so no
        # stale provider fid may survive into the next run or into review.
        self._auto_live_fid_map = {}
        for attr in ("_auto_repaint_timer", "_auto_live_cooldown_timer"):
            timer = getattr(self, attr, None)
            if timer is not None:
                try:
                    timer.stop()
                except (RuntimeError, AttributeError):
                    pass
        # The next run opens on an empty layer, so it must not inherit this
        # run's frame cost (it would start throttled for no reason).
        self._auto_live_frame_s = 0.0
        self._auto_live_frame_started = 0.0
        self._auto_live_repaint_not_before = 0.0

    def _reset_auto_live_pipeline(self) -> None:
        """Stop the live stitcher and the pending repaint.

        Called from every path that abandons or finishes a run so a late delta,
        a pending repaint, or a stale finalize step can never touch a torn-down
        layer/merger."""
        self._stop_auto_live_pump()
        # Invalidate any in-flight cooperative finalize refine / reslice.
        self._auto_finalize_gen += 1
        self._auto_finalize_state = None
        # Invalidate any in-flight background preview-cache build.
        self._auto_preview_build_gen += 1
        self._auto_preview_build_state = None

    def _drop_auto_tile_bridge(self) -> None:
        """Release the per-tile render bridge once the worker thread has exited.

        Called at every worker-null terminal. The worker emits its terminal only
        after its run loop ends, so no further render_tile call can reach the
        bridge; cancel() it first (defensive) then drop the reference so it is
        garbage-collected. Safe to call when there is no bridge."""
        bridge = self._auto_tile_bridge
        self._auto_tile_bridge = None
        if bridge is not None:
            try:
                bridge.cancel()
            except (RuntimeError, AttributeError):
                pass
            # Main thread here, unlike cancel(), so this is where the cached
            # render clone can be dropped: it is a QgsRasterLayer and it holds
            # a handle on the user's raster until it goes.
            try:
                bridge.release_render_clone()
            except (RuntimeError, AttributeError):
                pass
