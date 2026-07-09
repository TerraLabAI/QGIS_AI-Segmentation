"""Refine-in-Manual handoff and click-to-refine of imported detections.

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
    QgsProject,
)
from qgis.gui import QgsRubberBand
from qgis.PyQt.QtWidgets import (
    QMessageBox,
)

from ...core.i18n import tr
from ...core.prompt_manager import FrozenCropSession
from ...core.qt_compat import PolygonGeometry
from ..canvas_palette import PENDING_FILL, PENDING_STROKE


class ManualHandoffMixin:
    """Refine-in-Manual handoff and click-to-refine of imported detections."""

    # ------------------------------------------------------------------
    # Refine in Manual: hand the Automatic review to the Manual flow, refine
    # specific objects with point-and-click, then return to Finish.
    # ------------------------------------------------------------------

    def _resolve_auto_source_layer(self):
        """The raster layer the Automatic run segmented, for the Manual refine
        session. Resolved from the run context (by id), then by name."""
        ctx = self._auto_run_ctx or {}
        lid = ctx.get("layer_id")
        if lid:
            lyr = QgsProject.instance().mapLayer(lid)
            if lyr is not None:
                return lyr
        name = (self._auto_review or {}).get("source_layer_name")
        if name:
            for lyr in QgsProject.instance().mapLayersByName(name):
                return lyr
        return None

    def _update_handoff_progress(self) -> None:
        """Push the kept (hand-refined) count to the handoff banner counter."""
        if not (self._refine_handoff_active and self.dock_widget):
            return
        kept = sum(1 for p in self.saved_polygons if p.get("manual_touched"))
        try:
            self.dock_widget.update_handoff_progress(kept)
        except (RuntimeError, AttributeError):
            pass

    def _manual_env_ready(self) -> bool:
        """Best-effort 'the local AI is fully installed, or install/load is in
        flight'. Requires BOTH the venv AND the model checkpoint: deps-ready with
        a missing checkpoint leaves the predictor unable to load, which used to
        hang the handoff forever on 'Preparing Manual mode'. `_env_ready` is NOT
        treated as authoritative (it is a one-way cache that never re-validates
        false), so on the click path we re-run the cheap status checks and clear
        a stale positive if the env broke since. Fail-open on any check error so
        Refine is never wrongly blocked."""
        if self.predictor is not None:
            return True
        # An install/download/load already in flight counts as ready-in-progress:
        # the deferred handoff completes when it finishes.
        for w in (self.deps_install_worker, self._verify_worker,
                  self.download_worker, self._predictor_worker,
                  self._startup_check_worker):
            if w is not None and w.isRunning():
                return True
        # The model checkpoint must exist too, else the predictor can never load.
        try:
            from ...core.checkpoint_manager import checkpoint_exists
            if not checkpoint_exists():
                self._env_ready = False
                return False
        except Exception:  # nosec B110 -- fail-open on a checkpoint-probe error
            pass
        # Re-validate the venv on this click path (cheap, not per-frame): clear a
        # stale cached positive so a venv that broke since re-routes to install.
        try:
            from ...core.venv_manager import get_venv_status
            ready, _msg = get_venv_status()
            self._env_ready = bool(ready)
            return bool(ready)
        except Exception:
            return True  # never block a legitimate refine on a check error

    def _on_refine_in_manual_clicked(self) -> None:
        """Hand the reviewed detections to Manual mode for point-and-click fixes.

        Confidence stays EDITABLE: hand-edited objects are protected by geometry
        (_auto_protected_geoms) across a later confidence change, so no lock is
        needed. When the local AI is not installed, offer a one-time install that
        runs in the BACKGROUND while the user stays on this review (fully usable):
        an inline banner shows progress and the handoff opens automatically once
        the AI is ready (D1). No page detour, the review is never lost.
        """
        review = self._auto_review
        if not review or not self.dock_widget:
            return
        layer = self._resolve_auto_source_layer()
        if layer is None:
            return
        # Env gate: without the local AI the predictor never arrives, so the
        # handoff would hang on "Preparing Manual mode" forever. Offer to install
        # it in the background; the review stays on screen and the refine opens
        # itself when the predictor is ready (see _on_predictor_loaded).
        if not self._manual_env_ready():
            if self._refine_install_pending:
                return  # a background install is already running for this refine
            box = QMessageBox(self.iface.mainWindow())
            box.setWindowTitle(tr("Manual mode needs a one-time setup"))
            box.setText(tr(
                "Refining uses the free local AI, which is not installed yet. "
                "Install it now (a few minutes, in the background)? "
                "You can keep reviewing, and refining will open automatically "
                "when it is ready."))
            install_btn = box.addButton(tr("Install now"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(tr("Cancel"), QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(install_btn)
            box.exec()
            if box.clickedButton() is not install_btn:
                return  # review untouched
            # Start the install in the background WITHOUT leaving the review: the
            # inline banner shows progress, the Refine button is disabled until
            # the AI is ready, and _on_predictor_loaded re-invokes this method.
            self._refine_install_pending = True
            try:
                self.dock_widget.set_auto_review_installing(True)
            except (RuntimeError, AttributeError):
                pass
            self._on_install_requested()
            return
        # Refining runs the LOCAL SAM model, which loads lazily/async: the
        # predictor is None until the setup worker finishes, even when the model
        # is fully installed (e.g. an Automatic-first open never ran Interactive
        # setup). We therefore must NOT gate the handoff on `predictor is None`:
        # that wrongly routed legitimate refine clicks to a plain mode switch,
        # which discarded+autosaved the review to a RED committed layer. Instead
        # we enter the locked handoff now; begin_refine_handoff -> the mode
        # guard -> _ensure_interactive_setup kicks off the model load, and
        # _enter_manual_refine_session defers the import until it is ready.
        self._handoff_source_layer = layer
        self._pending_refine_import = False
        self._refine_handoff_active = True
        self._auto_refined_in_manual = True  # export will report the handoff was used
        try:
            from ...core import telemetry
            telemetry.track_refine_in_manual_entered(
                run_id=self._auto_run_id or "",
                instances=len(review.get("geoms", [])),
            )
        except Exception:
            pass  # nosec B110
        # Hide (do NOT discard) the blue review layer: the manual session shows
        # the same detections as editable saved polygons.
        self._remove_auto_selection_layer()
        try:
            # Confidence stays editable: hand-edited objects are protected across
            # a confidence change (see _auto_protected_geoms), so no lock needed.
            self.dock_widget.set_protected_note(False)
            self.dock_widget.begin_refine_handoff(len(review.get("geoms", [])))
        except (RuntimeError, AttributeError):
            pass

    def _clear_refine_install_pending(self) -> None:
        """Drop the pending background-install-then-refine intent and hide its
        inline review banner. Idempotent; safe to call from any review teardown."""
        if not getattr(self, "_refine_install_pending", False):
            return
        self._refine_install_pending = False
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_review_installing(False)
            except (RuntimeError, AttributeError):
                pass

    def _abort_refine_install(self) -> None:
        """A background install started from the review failed: clear the pending
        refine intent and re-enable the Refine button (the review is unharmed; the
        install's own error report already told the user what went wrong)."""
        self._clear_refine_install_pending()

    def _enter_manual_refine_session(self) -> None:
        """Start a Manual session on the run's raster and load the reviewed
        detections as editable saved polygons. Called from the mode-switch guard."""
        review = self._auto_review
        layer = getattr(self, "_handoff_source_layer", None)
        if not review or layer is None:
            return
        # The predictor loads asynchronously. If it is not up yet, DEFER: stash
        # the intent and let _on_predictor_loaded complete the start+import once
        # the model is ready. The mode-switch guard already called
        # _ensure_interactive_setup(), which triggers install/load if needed.
        if self.predictor is None:
            self._pending_refine_import = True
            if self.dock_widget:
                try:
                    self.dock_widget.set_refine_handoff_preparing(True)
                except (RuntimeError, AttributeError):
                    pass
            return
        self._pending_refine_import = False
        # Sync the (locked, grayed) manual combo to the handoff raster so the
        # header names the SAME raster the run segmented, not the stale auto-fill.
        try:
            combo = self.dock_widget.layer_combo
            combo.blockSignals(True)
            combo.setLayer(layer)
            combo.blockSignals(False)
        except (RuntimeError, AttributeError):
            pass
        # Full manual setup (this calls _reset_session, clearing saved_polygons).
        self._on_start_segmentation(layer)
        # Inherit the review's CURRENT refine settings for the Manual panel:
        # the handoff refines the SAME objects the Automatic review just
        # tuned (buildings keep holes filled + right angles, vegetation keeps
        # round corners), so the panel must not snap back to the generic
        # Manual defaults - including any switch the user flipped in the
        # review, not just the run's preset. Simplify/expand keep theirs
        # (Manual px is the 1024 SAM mask grid, a different scale). Set
        # BEFORE the import below so every imported entry records these
        # values as its refine baseline.
        try:
            params = self._widget_review_params()
            self._refine_smooth = 5 if params.get("smooth") else 0
            self._refine_fill_holes = bool(params.get("fill_holes"))
            self._refine_ortho = bool(params.get("ortho"))
            self.dock_widget.set_refine_values(
                self._refine_simplify, self._refine_smooth,
                self._refine_expand, self._refine_fill_holes,
                right_angles=self._refine_ortho)
        except (RuntimeError, AttributeError):
            pass
        self._import_review_geoms_as_saved(review)

    def _import_review_geoms_as_saved(self, review) -> None:
        """Load review geometries (raster CRS) into saved_polygons as 'pending'
        seeds rendered by ONE in-memory layer (blue), NOT a rubber band each: a
        1000-object handoff used to create 1000 canvas items and freeze QGIS.
        saved_rubber_bands gets a None per entry (kept index-
        locked with saved_polygons); the ACTIVE object is the only band. Refine-
        in-Manual still reads like Manual Mode (blue = pending / green = validated
        this session): click a detection to edit it, Save to turn it green. The
        pending/validated split is carried by the entry's `validated` flag, which
        _rebuild_handoff_layers uses to draw pending on the blue layer and
        validated on the green layer. Base Manual saves are unaffected (real green
        bands, no memory layer)."""
        geoms = review.get("geoms") or []
        crs = review.get("crs")
        authid = crs.authid() if crs is not None and crs.isValid() else None
        # The seed layers live in the SOURCE RASTER CRS (the run CRS the geoms are
        # already in), so geoms are pushed directly with no per-object canvas
        # transform (the layer reprojects for display).
        layer_crs = authid
        src = getattr(self, "_handoff_source_layer", None)
        if src is not None:
            try:
                layer_crs = src.crs().authid() or authid
            except (RuntimeError, AttributeError):
                layer_crs = authid
        if layer_crs:
            self._ensure_handoff_layers(layer_crs)
        for g in geoms:
            if g is None or g.isEmpty():
                continue
            # Carry protection across repeat refine visits: a geom matching a
            # previously hand-edited one is re-imported already marked touched, so
            # it stays protected from confidence re-filtering.
            self.saved_polygons.append({
                "geometry_wkt": g.asWkt(),
                # Cache the parsed geometry so absorb/click/collect don't re-parse
                # this polygon's WKT on every Save over a big handoff.
                "geom_obj": g,
                "transform_info": {"crs": authid} if authid else None,
                "raw_mask": None,
                "points_positive": [],
                "points_negative": [],
                "refine_simplify": self._refine_simplify,
                "refine_smooth": self._refine_smooth,
                "refine_expand": self._refine_expand,
                "refine_fill_holes": self._refine_fill_holes,
                "refine_ortho": self._refine_ortho,
                "refine_min_area": self._refine_min_area,
                "manual_touched": self._geom_overlaps_any(g, self._auto_protected_geoms),
                # Not yet hand-validated: drawn on the blue pending layer.
                "validated": False,
            })
            # None placeholder keeps the two lists index-locked; the geometry is
            # drawn by _handoff_pending_layer, not a per-object band.
            self.saved_rubber_bands.append(None)
        self._rebuild_handoff_layers()
        if self.dock_widget:
            try:
                self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
            except (RuntimeError, AttributeError):
                pass

    def _on_back_to_review_clicked(self) -> None:
        """Return from the Manual refine to the Automatic review (where Finish
        commits). Harvests the manual edits back into the held review first."""
        if not self._refine_handoff_active:
            return
        self._collect_manual_refine_into_review()
        try:
            from ...core import telemetry
            telemetry.track_refine_in_manual_back(
                run_id=self._auto_run_id or "",
                validated_count=len((self._auto_review or {}).get("geoms", [])),
            )
        except Exception:
            pass  # nosec B110
        try:
            from ..ai_segmentation_dockwidget import Mode
            self.dock_widget.end_refine_handoff(Mode.AUTOMATIC)
        except (RuntimeError, AttributeError):
            pass

    def _collect_manual_refine_into_review(self) -> None:
        """Fold every manual edit (saved + any in-progress mask) back into
        _auto_review["geoms"], then tear the manual session down."""
        # If the import never completed (the predictor was still loading when
        # the user backed out), there is no manual session to harvest: leave the
        # held review untouched so Back to review restores it intact.
        if self._pending_refine_import:
            self._pending_refine_import = False
            self._teardown_manual_session()
            return
        review = self._auto_review
        if review is not None:
            # Fold any in-progress (unsaved) mask into saved_polygons via the
            # normal save path so all edits are captured uniformly.
            try:
                self._on_save_polygon()
            except Exception as e:  # noqa: BLE001
                # Never block the handoff harvest on one bad save, but do not
                # swallow it silently: log so a lost edit is diagnosable (plan
                # 11 §1.7). Runs once per Back-to-review, so no spam guard needed.
                QgsMessageLog.logMessage(
                    f"Refine handoff: save fold error: {e}",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
            geoms = []
            protected = []
            for pg in self.saved_polygons:
                g = self._entry_geom(pg)
                if g is not None and not g.isEmpty():
                    geoms.append(g)
                    if pg.get("manual_touched"):
                        protected.append(g)
            # Dissolve any remaining overlaps so the committed output is uniform
            # (never stacked layers), while distinct touching objects stay split.
            review["geoms"] = self._dissolve_overlapping(geoms)
            # The geoms were rewritten, so the parallel score/id lists no longer
            # align: drop them (neutral heatmap, $id random colours) rather than
            # colour the wrong objects.
            review["scores"] = None
            review["ids"] = None
            # Remember the hand-edited objects so a later confidence change keeps
            # them and only re-filters the untouched auto detections.
            self._auto_protected_geoms = protected
            # And remember the DELETED ones: a later reslice (confidence or any
            # shape param) recomputes the visible set from the canonical
            # _auto_objects, so without this memory an object removed in Manual
            # comes straight back on the first slider move.
            self._auto_manual_removed = self._removed_canonical_objects(
                review["geoms"])
        self._teardown_manual_session()

    def _removed_canonical_objects(self, kept_geoms: list) -> set:
        """Indices into _auto_objects of detections deleted during the Manual
        refine: canonical objects no longer meaningfully covered (>= 30% of
        their area) by any harvested geometry. Spatial-index candidates keep
        this linear-ish; it runs once per Back-to-review."""
        objects = getattr(self, "_auto_objects", None) or []
        if not objects:
            return set()
        from qgis.core import QgsFeature, QgsGeometry, QgsSpatialIndex

        index = QgsSpatialIndex()
        kept = []
        for g in kept_geoms or []:
            if g is None or g.isEmpty():
                continue
            feat = QgsFeature(len(kept))
            feat.setGeometry(QgsGeometry.fromRect(g.boundingBox()))
            index.addFeature(feat)
            kept.append(g)
        removed = set()
        for det_idx, (base, _score, _area) in enumerate(objects):
            if base is None or base.isEmpty():
                continue
            area = base.area()
            if area <= 0:
                continue
            still_present = False
            for j in index.intersects(base.boundingBox()):
                try:
                    inter = base.intersection(kept[j])
                except Exception:  # nosec B112
                    continue
                if inter is not None and not inter.isEmpty() and inter.area() / area >= 0.3:
                    still_present = True
                    break
            if not still_present:
                removed.add(det_idx)
        return removed

    def _restore_auto_review_after_handoff(self) -> None:
        """Rebuild the Automatic review UI + blue layer after a Manual refine.
        Confidence stays locked so the hand edits survive."""
        review = self._auto_review
        layer = getattr(self, "_handoff_source_layer", None)
        self._refine_handoff_active = False
        self._handoff_source_layer = None
        if layer is not None:
            self._remove_auto_selection_layer()
            self._auto_selection_layer = self._create_auto_selection_layer(layer)
        if self.dock_widget and review is not None:
            try:
                self.dock_widget.set_auto_review_active(
                    True, count=len(review.get("geoms") or []),
                    reset_controls=False)
                # Confidence stays editable; hand edits are protected across a
                # confidence change (see _auto_protected_geoms). Show the truth
                # note when there is at least one protected object.
                self.dock_widget.set_protected_note(bool(self._auto_protected_geoms))
            except (RuntimeError, AttributeError):
                pass
        self._refresh_auto_review_preview()

    def _teardown_manual_session(self) -> None:
        """Stop the Manual session without the confirm dialog (its edits are
        being harvested, not discarded)."""
        if self._shortcut_filter is not None:
            try:
                self.iface.mainWindow().removeEventFilter(self._shortcut_filter)
                canvas = self.iface.mapCanvas()
                canvas.viewport().removeEventFilter(self._shortcut_filter)
                canvas.removeEventFilter(self._shortcut_filter)
            except RuntimeError:
                pass
        self._stopping_segmentation = True
        try:
            self.iface.mapCanvas().unsetMapTool(self.map_tool)
            self._restore_previous_map_tool()
        finally:
            self._stopping_segmentation = False
        self._reset_session()
        if self.dock_widget:
            try:
                self.dock_widget.reset_session()
            except (RuntimeError, AttributeError):
                pass

    def _discard_refine_handoff(self) -> None:
        """Abort a Manual refine handoff (mode switch / unload) without committing
        manual edits: clear the flag + UI lock and tear the manual session down.
        The held _auto_review is left for its own teardown path to handle."""
        if not self._refine_handoff_active:
            return
        self._refine_handoff_active = False
        self._auto_protected_geoms = []
        self._auto_manual_removed = set()
        self._handoff_source_layer = None
        self._teardown_manual_session()
        if self.dock_widget:
            try:
                self.dock_widget._refine_handoff = False
                self.dock_widget.set_protected_note(False)
                self.dock_widget.refine_handoff_banner.setVisible(False)
                self.dock_widget.mode_switch.setEnabled(True)
            except (RuntimeError, AttributeError):
                pass

    # --- click-to-refine an imported detection -------------------------------

    @staticmethod
    def _entry_geom(pg):
        """Cached QgsGeometry for a saved_polygons entry: parse its WKT once and
        memoize it on the entry as `geom_obj`. Entries are never mutated in place
        (geometry_wkt is only ever set at append time), so the cache cannot go
        stale. Avoids re-parsing every polygon's WKT on each Save / canvas click
        / absorb over a big handoff set. Callers must NOT mutate
        the returned geometry in place - it is shared; copy with QgsGeometry(g)
        before transform()."""
        g = pg.get("geom_obj")
        if g is not None:
            return g
        g = QgsGeometry.fromWkt(pg.get("geometry_wkt") or "")
        pg["geom_obj"] = g
        return g

    def _hit_test_saved_polygon(self, raster_pt):
        """Index of the topmost saved polygon containing raster_pt, else None.
        Iterates last-first so the most recently added object wins overlaps."""
        pt = QgsGeometry.fromPointXY(raster_pt)
        for i in range(len(self.saved_polygons) - 1, -1, -1):
            g = self._entry_geom(self.saved_polygons[i])
            if g is not None and not g.isEmpty() and g.contains(pt):
                return i
        return None

    # Fraction of the SMALLER object's area that must overlap for a new save to
    # be treated as "completing" an existing detection (vs a distinct object).
    # A real completion overlaps a lot; two distinct neighbours that merely touch
    # share ~zero area, so they stay separate (the instance count is preserved).
    _COMPLETE_OVERLAP_FRAC = 0.1

    def _absorb_overlapping_saved(self, geom):
        """Refine handoff only: union `geom` with any already-saved detections it
        genuinely OVERLAPS (shared area, not a mere shared edge) and drop those,
        so clicking to complete a partial detection grows ONE uniform polygon
        instead of stacking a new layer on top. Distinct neighbours that only
        touch are left alone. Returns the (possibly grown) geometry."""
        if not self._refine_handoff_active or geom is None or geom.isEmpty():
            return geom
        merged = geom
        merged_bb = merged.boundingBox()
        new_polys: list = []
        new_bands: list = []
        for i in range(len(self.saved_polygons)):
            pg = self.saved_polygons[i]
            rb = self.saved_rubber_bands[i] if i < len(self.saved_rubber_bands) else None
            g = self._entry_geom(pg)
            absorb = False
            # Cheap bbox pre-filter before the costly intersection().
            if (g is not None and not g.isEmpty()
                    and merged_bb.intersects(g.boundingBox())  # noqa: W503
                    and merged.intersects(g)):  # noqa: W503
                inter = merged.intersection(g)
                if inter is not None and not inter.isEmpty():
                    smaller = min(merged.area(), g.area())
                    if smaller > 0 and inter.area() / smaller >= self._COMPLETE_OVERLAP_FRAC:
                        union = merged.combine(g)
                        if union is not None and not union.isEmpty():
                            merged = union
                            merged_bb = merged.boundingBox()
                            absorb = True
            if absorb:
                if rb is not None:
                    self._safe_remove_rubber_band(rb)
            else:
                new_polys.append(pg)
                # Append the band UNCONDITIONALLY (even when None): the two lists
                # must stay index-locked or _ensure_polygon_rubberband_sync will
                # truncate saved_polygons as "repair" and drop real detections.
                new_bands.append(rb)
        self.saved_polygons = new_polys
        self.saved_rubber_bands = new_bands
        self._rebuild_handoff_layers()
        return merged

    def _geom_overlaps_any(self, geom, others) -> bool:
        """True if `geom` overlaps any geometry in `others` by area (>= the
        complete-overlap fraction of the smaller), i.e. they are the same object,
        not merely touching neighbours. Cheap bbox pre-filter first."""
        if geom is None or geom.isEmpty() or not others:
            return False
        bb = geom.boundingBox()
        for o in others:
            if o is None or o.isEmpty():
                continue
            if not bb.intersects(o.boundingBox()) or not geom.intersects(o):
                continue
            inter = geom.intersection(o)
            if inter is None or inter.isEmpty():
                continue
            smaller = min(geom.area(), o.area())
            if smaller > 0 and inter.area() / smaller >= self._COMPLETE_OVERLAP_FRAC:
                return True
        return False

    def _dissolve_overlapping(self, geoms: list) -> list:
        """Union geometries that OVERLAP by area into one (no stacked layers in
        the committed output), leaving merely-touching or disjoint objects
        separate so the instance count is preserved. Spatially indexed (cheap
        bbox prune first), so it stays light even on a big review set."""
        from qgis.core import QgsFeature, QgsSpatialIndex
        items = [g for g in geoms if g is not None and not g.isEmpty()]
        if len(items) <= 1:
            return items
        index = QgsSpatialIndex()
        keep: dict = {}
        nid = 0
        for g in items:
            merged = g
            matches = []
            for fid in index.intersects(merged.boundingBox()):
                h = keep.get(fid)
                if h is None or not merged.intersects(h):
                    continue
                inter = merged.intersection(h)
                if inter is None or inter.isEmpty():
                    continue
                smaller = min(merged.area(), h.area())
                if smaller > 0 and inter.area() / smaller >= self._COMPLETE_OVERLAP_FRAC:
                    matches.append(fid)
            for fid in matches:
                union = merged.combine(keep[fid])
                if union is not None and not union.isEmpty():
                    merged = union
                    keep[fid] = None
            feat = QgsFeature(nid)
            feat.setGeometry(merged)
            index.insertFeature(feat)
            keep[nid] = merged
            nid += 1
        return [g for g in keep.values() if g is not None]

    def _weld_active_into_overlaps(self) -> None:
        """Live 'complete-don't-stack' during a refine handoff: if the active SAM
        selection now overlaps existing saved detection(s) by area, fold each into
        a FROZEN session (and drop its saved entry) so the canvas shows ONE welded
        shape immediately and a Save commits it as one polygon. Frozen sessions are
        already composited with the active mask in both the preview and the save,
        so no SAM re-call is needed - just a polygonize of the current crop mask
        (cheap) plus a bbox-pruned overlap scan. Touching-only neighbours (~0
        shared area) are left alone, preserving the instance count."""
        if not self._refine_handoff_active:
            return
        if self.current_mask is None or self.current_transform_info is None:
            return
        from ...core.polygon_exporter import mask_to_polygons
        geoms = mask_to_polygons(self.current_mask, self.current_transform_info)
        if not geoms:
            return
        active = QgsGeometry.unaryUnion(geoms)
        if active is None or active.isEmpty():
            return
        active_bb = active.boundingBox()
        new_polys: list = []
        new_bands: list = []
        folded = False
        for i in range(len(self.saved_polygons)):
            pg = self.saved_polygons[i]
            rb = self.saved_rubber_bands[i] if i < len(self.saved_rubber_bands) else None
            g = self._entry_geom(pg)
            absorb = False
            if (g is not None and not g.isEmpty()
                    and active_bb.intersects(g.boundingBox())  # noqa: W503
                    and active.intersects(g)):  # noqa: W503
                inter = active.intersection(g)
                if inter is not None and not inter.isEmpty():
                    smaller = min(active.area(), g.area())
                    if smaller > 0 and inter.area() / smaller >= self._COMPLETE_OVERLAP_FRAC:
                        absorb = True
            if absorb:
                self._frozen_sessions.append(FrozenCropSession(polygon=g))
                folded = True
                if rb is not None:
                    self._safe_remove_rubber_band(rb)
            else:
                new_polys.append(pg)
                # Append UNCONDITIONALLY (even None) to keep the two lists
                # index-locked; see _absorb_overlapping_saved.
                new_bands.append(rb)
        if folded:
            self.saved_polygons = new_polys
            self.saved_rubber_bands = new_bands
            self._rebuild_handoff_layers()
            if self.dock_widget:
                try:
                    self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
                except (RuntimeError, AttributeError):
                    pass
            self._update_mask_visualization()

    def _clear_active_mask_without_saving(self) -> None:
        """Drop the active mask + its clicks/markers WITHOUT saving it (used by
        the Delete-key object removal). Does not touch the saved set."""
        self.current_mask = None
        self.current_score = 0.0
        self.current_transform_info = None
        self.current_low_res_mask = None
        try:
            self.prompts.clear()
        except (RuntimeError, AttributeError):
            pass
        self._mask_state_history = []
        if self.map_tool:
            try:
                self.map_tool.clear_markers()
            except (RuntimeError, AttributeError):
                pass
        self._clear_mask_visualization()
        if self.dock_widget:
            try:
                self.dock_widget.set_point_count(0, 0)
            except (RuntimeError, AttributeError):
                pass

    def _on_delete_active_object(self) -> None:
        """Delete the object currently OPEN for editing. One-shot undo
        (Ctrl+Z) restores it. Only active during a refine handoff or when a
        saved object is open for editing."""
        if not (self._refine_handoff_active or self._is_refining_saved_object):
            return
        # Snapshot for undo: prefer the exact original entry re-opened for edit;
        # otherwise synthesize one from the current active mask geometry.
        origin = self._active_refine_origin_entry
        if origin is not None:
            backup = dict(origin)
        else:
            wkt = None
            if self.current_mask is not None and self.current_transform_info is not None:
                from ...core.polygon_exporter import mask_to_polygons
                gs = mask_to_polygons(self.current_mask, self.current_transform_info)
                if gs:
                    u = QgsGeometry.unaryUnion(gs)
                    if u is not None and not u.isEmpty():
                        wkt = u.asWkt()
            if not wkt:
                return
            authid = (self.current_transform_info or {}).get("crs")
            backup = {
                "geometry_wkt": wkt,
                "transform_info": {"crs": authid} if authid else None,
                "manual_touched": self._refine_handoff_active,
            }
        self._deleted_object_backup = backup
        self._clear_active_mask_without_saving()
        self._is_refining_saved_object = False
        self._active_refine_origin_entry = None
        if self.dock_widget:
            try:
                self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
            except (RuntimeError, AttributeError):
                pass
        self._update_handoff_progress()
        QgsMessageLog.logMessage(
            "Object deleted. Ctrl+Z restores it.",
            "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _restore_deleted_object(self) -> bool:
        """Re-append a Delete-key backup as a PENDING blue saved polygon. Returns
        True if a restore happened."""
        backup = self._deleted_object_backup
        if not backup:
            return False
        self._deleted_object_backup = None
        wkt = backup.get("geometry_wkt")
        g = QgsGeometry.fromWkt(wkt) if wkt else None
        if g is None or g.isEmpty():
            return False
        entry = dict(backup)
        entry["validated"] = False  # a restored object is pending (blue) again
        self.saved_polygons.append(entry)
        if self._refine_handoff_active:
            # Drawn by the blue pending layer; None keeps saved_rubber_bands
            # index-locked with saved_polygons.
            self.saved_rubber_bands.append(None)
            self._rebuild_handoff_layers()
        else:
            # Base Manual re-edit: pending (not-yet-validated) blue band.
            rb = QgsRubberBand(
                self.iface.mapCanvas(), PolygonGeometry)
            rb.setColor(PENDING_FILL)
            rb.setStrokeColor(PENDING_STROKE)
            rb.setWidth(2)
            display_geom = QgsGeometry(g)
            self._transform_geometry_to_canvas_crs(display_geom)
            rb.setToGeometry(display_geom, None)
            self.saved_rubber_bands.append(rb)
        if self.dock_widget:
            try:
                self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
            except (RuntimeError, AttributeError):
                pass
        self._update_handoff_progress()
        return True

    def _activate_saved_polygon_for_refine(self, idx, raster_pt, label: int = 1) -> None:
        """Re-open an imported detection as the active SAM mask so +/- clicks
        reshape it: encode a crop fitting the whole object, seed SAM with the
        object's rasterized mask, register the click, and predict. Falls back to
        keeping the original shape selected if the local model diverges.

        label=1: the click is the first POSITIVE point (grow/keep at the click).
        label=0: the click is a NEGATIVE point (carve at the click) - a positive
        anchor is added at the polygon's interior so SAM keeps the rest of the
        shape while the negative removes the clicked area. This lets a right-click
        directly carve an over-segmented detection with no prior left-click."""
        entry = self.saved_polygons[idx]
        geom = QgsGeometry.fromWkt(entry.get("geometry_wkt") or "")
        if geom is None or geom.isEmpty():
            return
        # Remove it from the saved set + canvas: it becomes the active selection.
        popped = self.saved_polygons.pop(idx)
        if idx < len(self.saved_rubber_bands):
            self._safe_remove_rubber_band(self.saved_rubber_bands.pop(idx))
        # It just left saved_polygons, so refresh the seed layers to drop it (it
        # now shows as the active mask band, not on the pending/kept layer).
        self._rebuild_handoff_layers()
        # This object is now OPEN for editing: rendered in pending-blue with a
        # bolder outline (no separate hue) and the Delete key enabled. Keep the
        # original entry so a Delete-undo can restore it.
        self._is_refining_saved_object = True
        self._active_refine_origin_entry = dict(popped)
        if self.dock_widget:
            try:
                self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
            except (RuntimeError, AttributeError):
                pass

        # Keep the object VISIBLE during the async encode (PERF-01): it just left
        # the saved layer, so without this it would blank out for the 3-8s encode.
        # Show it as the display-only unfrozen polygon; the tail's prediction
        # replaces it (or the failure fallback keeps it).
        self._unfrozen_display_polygon = geom
        self._update_mask_visualization()

        # Encode a crop that fits the WHOLE object (bbox corners + click), so a
        # large detection is not clipped by a click-centered 1024px crop. The
        # encode is async on the interactive path (PERF-01): the seed + predict
        # tail runs from the completion callback, not synchronously here.
        bb = geom.boundingBox()
        pts = [(bb.xMinimum(), bb.yMinimum()), (bb.xMaximum(), bb.yMaximum()),
               (raster_pt.x(), raster_pt.y())]
        cx, cy, scale = self._compute_crop_center_and_mupp(pts)

        def _seed_and_predict():
            # Seed SAM with the object's mask rasterized into the crop pixel grid
            # so the refine starts from the cloud detection, not a blank slate.
            seed = self._rasterize_polygon_to_crop(geom)
            if seed is not None and seed.any():
                self.current_low_res_mask = self._binary_mask_to_logits(seed)

            # Register the click for this refine. A positive click is the first
            # positive point. A negative click carves: anchor a positive at the
            # polygon interior (so SAM keeps the rest of the shape) and add the
            # click as a negative point.
            self.prompts.clear()
            self._active_crop_points_positive = []
            self._active_crop_points_negative = []
            if label == 0:
                anchor = geom.pointOnSurface()
                if anchor is not None and not anchor.isEmpty():
                    ap = anchor.asPoint()
                    self._active_crop_points_positive.append((ap.x(), ap.y()))
                    self.prompts.add_positive_point(ap.x(), ap.y())
                self._active_crop_points_negative.append((raster_pt.x(), raster_pt.y()))
                self.prompts.add_negative_point(raster_pt.x(), raster_pt.y())
            else:
                self._active_crop_points_positive.append((raster_pt.x(), raster_pt.y()))
                self.prompts.add_positive_point(raster_pt.x(), raster_pt.y())
            self._mask_state_history = []

            if not self._run_prediction():
                self.current_low_res_mask = None
                self._unfrozen_display_polygon = geom
                self._update_mask_visualization()
                return

            # Guard the cloud-vs-local-SAM mismatch: if local SAM returned an
            # (almost) empty mask, keep the original object as the active shape
            # so it never vanishes; the user's +/- clicks then drive it directly.
            if self.current_mask is None or int(self.current_mask.sum()) == 0:
                self.current_mask = None
                self.current_low_res_mask = None
                self._unfrozen_display_polygon = geom
                self._update_mask_visualization()

        if not self._extract_and_encode_crop(
                QgsPointXY(cx, cy), mupp_override=scale, on_encoded=_seed_and_predict):
            # Crop extraction failed synchronously (interactive) or the encode
            # failed (headless): keep the object visible so it is not lost.
            self._unfrozen_display_polygon = geom
            self._update_mask_visualization()
            return

    def _rasterize_polygon_to_crop(self, geom):
        """Rasterize a raster-CRS polygon into the current crop's pixel grid (the
        inverse of mask_to_polygons). Returns a uint8 H x W mask, or None."""
        info = self._current_crop_info
        if info is None:
            return None
        try:
            import json

            from rasterio.features import rasterize
            from rasterio.transform import from_bounds as transform_from_bounds
            minx, miny, maxx, maxy = info["bounds"]
            h, w = info["img_shape"]
            transform = transform_from_bounds(minx, miny, maxx, maxy, w, h)
            geojson = json.loads(geom.asJson())
            return rasterize([(geojson, 1)], out_shape=(h, w), transform=transform,
                             fill=0, all_touched=True, dtype="uint8")
        except Exception:  # noqa: BLE001
            return None
