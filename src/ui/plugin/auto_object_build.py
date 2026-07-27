"""The canonical whole-object set: run pixel size, geodesic area, the noise
and false-positive gates, and the stable colour ids that survive a rebuild.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from qgis.core import QgsCoordinateReferenceSystem


class AutoObjectBuildMixin:
    """Build _auto_objects from a merged set and keep its identities stable."""

    # ---- Canonical whole-object helpers (merge-then-filter) -----------------

    def _capture_auto_mask_gsd(self, worker) -> None:
        """Harvest the worker's observed returned-mask ground resolution at a
        terminal (finished/cancelled/exhausted/error), BEFORE the worker ref is
        nulled, so the review's px<->ground refine keeps the run's true pixel.
        Keeps the previous run's value when this run saw no mask (0.0). Also
        captures the pre-submit tile drops (blank/nodata skips and render/provider
        holes) so _finalize_auto_results can surface them once per run."""
        obs = getattr(worker, "observed_mask_gsd", 0.0)
        if obs > 0:
            self._auto_mask_gsd = obs
        self._auto_skipped_blank_tiles = int(
            getattr(worker, "tiles_skipped_blank", 0) or 0)
        self._auto_render_failed_tiles = int(
            getattr(worker, "tiles_render_failed", 0) or 0)
        # Keep the run summary's "raw detection(s)" meaning what the model
        # RETURNED: the worker's MAP-mode per-tile pre-merge shrinks the stream
        # the GUI folds, so the GUI-side fold counter alone would under-report.
        raw_total = int(getattr(worker, "raw_detections_total", 0) or 0)
        if raw_total > self._auto_raw_count:
            self._auto_raw_count = raw_total
        # Residual truncation only: tiles still at the model's per-inference
        # object ceiling AFTER the saturated-tile re-split ladder ran (or was
        # unavailable). Rescued parents are excluded, so the review dense hint
        # no longer warns about truncation the run already repaired.
        self._auto_dense_tiles = int(
            getattr(worker, "tiles_capped_final", 0) or 0)
        self._auto_subdiv_tiles = int(
            getattr(worker, "tiles_subdivided", 0) or 0)

    def _auto_refine_pixel_size(self) -> float:
        """Ground units per DETECTION pixel, for px<->ground conversion in the
        review shape-refine (Simplify/Clean edges/Expand are px in the UI).

        The reference is the RUN's pixel, not the source raster's: the refine
        tolerances must scale with the staircase step of the polygons actually
        produced, which is the returned-mask grid of this run. That makes the
        px defaults dynamic: a close-up single-tile run gets tiny ground
        tolerances (detail preserved), a coarse wide run gets proportionally
        larger ones (staircase removed). Preference order:

        1. observed_mask_gsd from the live worker (mid-run live preview);
        2. _auto_mask_gsd captured at the worker's terminal (review/reslice);
        3. _auto_gsd, the render mupp (no mask seen yet, right magnitude);
        4. the source raster's native pixel (legacy fallback), then 1.0.
        """
        worker = self._auto_worker
        if worker is not None:
            obs = getattr(worker, "observed_mask_gsd", 0.0)
            if obs > 0:
                return obs
        obs = getattr(self, "_auto_mask_gsd", 0.0)
        if obs > 0:
            return obs
        if self._auto_gsd > 0:
            return self._auto_gsd
        return self._auto_source_pixel_size()

    def _auto_source_pixel_size(self) -> float:
        """Ground units per pixel of the source raster. Legacy FALLBACK for the
        px<->ground conversion when the run recorded no resolution (see
        _auto_refine_pixel_size, which callers should use instead). 1.0 fallback."""
        source_layer = self._get_active_raster_layer()
        try:
            if source_layer is not None:
                ext = source_layer.extent()
                w = source_layer.width()
                if w > 0 and ext.width() > 0:
                    return ext.width() / w
        except (RuntimeError, AttributeError):
            pass
        return 1.0

    def _make_auto_area_measurer(self):
        """One geodesic area measurer for the run CRS, reused for every object so
        a per-object ellipsoid reload never makes the size filter laggy."""
        try:
            from ...core.layer_conventions import make_area_measurer
            crs = QgsCoordinateReferenceSystem(self._auto_crs_authid or "EPSG:4326")
            return make_area_measurer(crs)
        except Exception:  # noqa: BLE001 -- never block finalize on a measurer
            return None

    def _object_area_m2(self, geom, measurer) -> float:
        """Geodesic ground area (m2) of a whole object, matching the number the
        export writes to area_m2 so the size filter agrees with the saved layer."""
        try:
            if measurer is not None:
                return float(measurer.measureArea(geom))
            return float(geom.area())
        except (RuntimeError, AttributeError):
            try:
                return float(geom.area())
            except (RuntimeError, AttributeError):
                return 0.0

    def _review_noise_floor(self) -> float:
        """Confidence fraction below which a detection is dropped from the review
        entirely (never counted, never rendered). Server-delivered value, fails
        open to one generic client default. The run still FETCHES everything (the
        worker recall floor is untouched); this is a review-side cut so the
        totals never include sub-noise masks and the review has fewer shapes to
        convert and render."""
        from ...core.detection_policy import review_noise_floor
        return review_noise_floor()

    def _auto_fp_rules(self) -> list:
        """Geometry-attribute false-positive rules for THIS run's object class,
        or [] (filter OFF). Keyed by the prompt's shape class (the same taxonomy
        the review presets and the server class tables use), so the server can
        drop obvious false positives per object kind (e.g. shadow slivers read as
        water) without a plugin release. Empty for an exemplar-only run (no text
        class) and whenever the server ships no table, so the filter is a no-op by
        default. The tuned thresholds live server-side; the client ships none."""
        prompt = str((self._auto_run_ctx or {}).get("prompt") or "").strip()
        if not prompt:
            return []
        try:
            from ...core.detection_policy import fp_rules
            from ...core.review_presets import shape_class_for
            return fp_rules(shape_class_for(prompt))
        except Exception:  # noqa: BLE001 -- never block finalize on the FP filter
            return []

    def _object_is_fp(self, geom, area: float, rules: list, measurer) -> bool:
        """Whether one whole object is a geometry-attribute false positive under
        ``rules``. No rules (the default) short-circuits to False, so the object
        is always kept and the code path is a no-op. The object's already
        measured ``area`` is reused so no area is computed twice."""
        if not rules:
            return False
        try:
            from ...core.geometry_attrs import matches_drop_rule, polygon_attributes
            return matches_drop_rule(
                polygon_attributes(geom, area_m2=area, measurer=measurer), rules)
        except Exception:  # noqa: BLE001 -- the FP filter must never break finalize
            return False

    def _build_auto_objects(self, merged_ided) -> list:
        """Synchronous (geom, score, area) build from the merger's ided result.
        Also records the parallel stable fid per object (_auto_object_fids) so the
        Random hue matches the live run. Used by the headless path; the
        interactive path builds it cooperatively. Detections below the review
        noise floor, and objects the server's per-class geometry-attribute filter
        marks as false positives, are dropped here so they never reach the review
        at all."""
        measurer = self._make_auto_area_measurer()
        floor = self._review_noise_floor()
        fp_rules = self._auto_fp_rules()
        out = []
        fids = []
        for fid, geom, score in merged_ided:
            if geom is None or geom.isEmpty():
                continue
            if float(score) < floor:
                continue
            area = self._object_area_m2(geom, measurer)
            if self._object_is_fp(geom, area, fp_rules, measurer):
                continue
            out.append((geom, float(score), area))
            fids.append(fid)
        self._auto_object_fids = fids
        return out

    def _preserve_object_fids(self, old_objects, old_fids) -> None:
        """Carry each unchanged object's stable colour id (det_id) across an
        object-set rebuild (batch fold, grouping toggle), so the Random hues
        never reshuffle on screen: losing every colour at once makes the
        result impossible to compare with what was there before. Matching is
        the pure core.review_corrections.remap_object_ids; on any failure the
        fresh merger ids stay (colours reshuffle, nothing breaks)."""
        try:
            from ...core.review_corrections import remap_object_ids
            old_rows = []
            for idx, (g, _s, _a) in enumerate(old_objects):
                if g is None or g.isEmpty():
                    continue
                c = g.centroid().asPoint()
                fid = old_fids[idx] if idx < len(old_fids) else idx
                old_rows.append(
                    (bytes(g.asWkb()), (c.x(), c.y()), int(fid)))
            new_rows = []
            for (g, _s, _a) in self._auto_objects:
                bb = g.boundingBox()
                c = g.centroid().asPoint()
                new_rows.append(
                    (bytes(g.asWkb()), (c.x(), c.y()),
                     (bb.xMinimum(), bb.yMinimum(),
                      bb.xMaximum(), bb.yMaximum())))
            self._auto_object_fids = remap_object_ids(old_rows, new_rows)
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _object_fid_for(self, idx: int) -> int:
        """Stable colour id (the merger's keeper fid) for a canonical object
        index, so the Random hue is identical live and in review. Falls back to
        the index itself when the fid list is missing or short (never raises into
        a repaint)."""
        fids = getattr(self, "_auto_object_fids", None)
        if fids is not None and 0 <= idx < len(fids):
            return fids[idx]
        return idx

    def _preserve_review_display_ids(self, old_geoms, old_ids, new_geoms) -> list | None:
        """Map review display ids onto a rebuilt visible geometry set.

        The Distinct renderer keys its hue on ``det_id``.  Review settings can
        reshape every geometry, and a Manual-returned shape can replace an
        automatic one altogether; neither is a reason to give the object a new
        visual identity.  Every object that can be matched keeps its id; only a
        genuinely new one gets a fresh id, minted above all the old ones.

        Every object gets an id, always.  Handing back ``None`` used to look
        like a safe "no opinion", but it writes an EMPTY det_id on every row,
        which sends the renderer to the QGIS feature id, and that is renumbered
        on every write: the colours then reshuffled on each move of the
        confidence slider.  So a row that cannot be read is skipped, not
        allowed to void the whole mapping.
        """
        if not isinstance(new_geoms, list):
            return None
        try:
            from ...core.review_corrections import remap_object_ids
            old_rows = []
            if isinstance(old_geoms, list) and isinstance(old_ids, list):
                for geom, det_id in zip(old_geoms, old_ids):
                    if geom is None or geom.isEmpty() or det_id is None:
                        continue  # unreadable prior row: it just matches nothing
                    center = geom.centroid().asPoint()
                    old_rows.append(
                        (bytes(geom.asWkb()), (center.x(), center.y()), int(det_id)))
            new_rows = []
            for geom in new_geoms:
                if geom is None or geom.isEmpty():
                    # Keeps the returned list aligned with new_geoms; the push
                    # drops the empty geometry anyway.
                    new_rows.append((b"", (0.0, 0.0), (0.0, 0.0, 0.0, 0.0)))
                    continue
                box = geom.boundingBox()
                center = geom.centroid().asPoint()
                new_rows.append(
                    (bytes(geom.asWkb()), (center.x(), center.y()),
                     (box.xMinimum(), box.yMinimum(),
                      box.xMaximum(), box.yMaximum())))
            return remap_object_ids(old_rows, new_rows)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return None
