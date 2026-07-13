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
from .shared import _debounce_timer


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
        """Push the kept count to the handoff header progress. Kept = validated
        (green on the canvas), so the dock bar and the map always agree; a
        shape-only tweak protects an object but does not turn it green."""
        if not (self._refine_handoff_active and self.dock_widget):
            return
        kept = sum(1 for p in self.saved_polygons if p.get("validated"))
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
            import time as _time

            from ...core import telemetry
            self._refine_handoff_t0 = _time.monotonic()
            telemetry.track_refine_in_manual_entered(
                run_id=self._auto_run_id or "",
                instances=len(review.get("geoms", [])),
            )
        except Exception:
            pass  # nosec B110
        # Hide (do NOT discard) the blue review layer: the manual session shows
        # the same detections as editable saved polygons.
        self._remove_auto_selection_layer()
        # Hide the Automatic canvas overlays for the hand-edit: the green
        # example outline reads as one more editable polygon and the zone
        # band/grid only distract. Restored on Back to review / discard.
        self._set_exemplar_bands_visible(False)
        self._set_auto_zone_overlays_visible(False)
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
            # The model load is one-shot: once _ensure_interactive_setup has run,
            # a load that FAILED (or was never kicked off) is never retried, so
            # the deferred import would wait on "Preparing Manual mode" forever.
            # On the NEXT event tick (after the first-time setup, itself a
            # singleShot, has had its turn to start its own worker) restart the
            # load only if nothing is in flight, so this never races or double-
            # starts it (see _retry_predictor_load_for_handoff).
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._retry_predictor_load_for_handoff)
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
        # Inherit the review's CURRENT refine settings BEFORE the import so every
        # imported entry records them as its refine baseline (see
        # _seed_refine_from_review).
        self._seed_refine_from_review()
        self._import_review_geoms_as_saved(review)

    def _seed_refine_from_review(self) -> None:
        """Seed the Manual refine panel from the Automatic review's CURRENT
        widget values, so a Refine-in-Manual handoff refines the SAME objects
        with the SAME settings the review just tuned (buildings keep holes
        filled + right angles, vegetation keeps round corners) instead of
        snapping back to the generic Manual defaults, including any switch the
        user flipped in the review, not just the run's preset. Simplify/expand
        keep their Manual values (Manual px is the 1024 SAM mask grid, a
        different scale); Min/Max size carry over 1:1 (both sides are true
        ground m2). Must run AFTER _on_start_segmentation (which resets the
        session) and BEFORE the import. Shared by the direct handoff and the
        deferred (predictor-still-loading) completion in _on_predictor_loaded."""
        try:
            params = self._widget_review_params()
            self._refine_smooth = 5 if params.get("smooth") else 0
            self._refine_fill_holes = bool(params.get("fill_holes"))
            self._refine_ortho = bool(params.get("ortho"))
            self._refine_min_size_m2 = max(0.0, float(params.get("min_a") or 0.0))
            self._refine_max_size_m2 = max(0.0, float(params.get("max_a") or 0.0))
            self.dock_widget.set_refine_values(
                self._refine_simplify, self._refine_smooth,
                self._refine_expand, self._refine_fill_holes,
                right_angles=self._refine_ortho)
            self.dock_widget.set_size_filter_values(
                self._refine_min_size_m2, self._refine_max_size_m2)
        except (RuntimeError, AttributeError):
            pass

    def _retry_predictor_load_for_handoff(self) -> None:
        """Restart a stalled model load for a deferred Refine handoff.

        Runs one tick after the handoff defers on a None predictor. The load is
        one-shot (a failed attempt is never retried by _ensure_interactive_setup),
        so without this a prior load failure strands the handoff on "Preparing
        Manual mode" forever. No-op unless the handoff is still pending, the
        predictor is still down, and NO install/download/verify/load/startup-check
        worker is running: each of those already ends in a load that completes
        the import from _on_predictor_loaded, so skipping while one runs also
        guards against double-starting the first-time setup's own load."""
        if not (self._pending_refine_import and self._refine_handoff_active):
            return
        if self.predictor is not None:
            return
        for w in (self.deps_install_worker, self._verify_worker,
                  self.download_worker, self._predictor_worker,
                  self._startup_check_worker):
            try:
                if w is not None and w.isRunning():
                    return
            except RuntimeError:
                continue
        self._load_predictor()

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
        # Parallel identity lists (may be None after older flows): score feeds
        # the review heatmap on return, det_id keeps the Random per-instance
        # colour stable across the whole handoff round trip.
        scores = review.get("scores") or []
        ids = review.get("ids") or []
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
        # Synthetic det_id sequence for objects with no canonical id (hand-drawn
        # saves, legacy reviews without ids): keeps every entry hue-stable and
        # the return arrays free of NULLs. Starts above the largest real id.
        max_id = max((int(i) for i in ids if i is not None), default=-1)
        self._handoff_det_id_seq = max_id + 1
        for n, g in enumerate(geoms):
            if g is None or g.isEmpty():
                continue
            det_id = ids[n] if n < len(ids) and ids[n] is not None else None
            if det_id is None:
                det_id = self._next_handoff_det_id()
            score = scores[n] if n < len(scores) and scores[n] is not None else None
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
                "refine_min_size_m2": self._refine_min_size_m2,
                "refine_max_size_m2": self._refine_max_size_m2,
                "manual_touched": self._geom_overlaps_any(g, self._auto_protected_geoms),
                # Not yet hand-validated: drawn on the pending layer.
                "validated": False,
                # Per-instance identity, carried through the whole handoff so
                # the Random colour and the review heatmap survive the round
                # trip (they used to be dropped here, which flattened Manual
                # refine to one uniform blue).
                "det_id": int(det_id),
                "score": float(score) if score is not None else None,
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
            import time as _time

            from ...core import telemetry
            t0 = getattr(self, "_refine_handoff_t0", None)
            telemetry.track_refine_in_manual_back(
                run_id=self._auto_run_id or "",
                validated_count=len((self._auto_review or {}).get("geoms", [])),
                duration_ms=int((_time.monotonic() - t0) * 1000) if t0 else None,
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
            # The save no-ops while a crop encode is in flight: an object still
            # OPEN for editing was popped from saved_polygons at open time, so
            # without this fold it would vanish from the harvest entirely.
            if self._is_refining_saved_object:
                self._close_active_edit_to_pending()
            entries = []
            protected = []
            for pg in self.saved_polygons:
                g = self._entry_geom(pg)
                if g is not None and not g.isEmpty():
                    entries.append((g, pg.get("det_id"), pg.get("score")))
                    if pg.get("manual_touched"):
                        protected.append(g)
            # Dissolve any remaining overlaps so the committed output is uniform
            # (never stacked layers), while distinct touching objects stay split.
            # Identity-aware: a dissolved group keeps its first member's det_id
            # (so the Random colour survives the round trip) and its max score,
            # instead of dropping both lists and reshuffling every colour.
            geoms, ids, scores = self._dissolve_overlapping_entries(entries)
            review["geoms"] = geoms
            review["scores"] = scores
            review["ids"] = ids
            # Harvested geoms are hand-edited/dissolved, NOT the refine cache's
            # output: drop the provenance stamp so the next review push does a
            # full rebuild instead of wrongly diffing against pre-handoff state.
            review["stamp"] = None
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
        # Bring the Automatic canvas overlays back (hidden for the hand-edit).
        self._set_exemplar_bands_visible(True)
        self._set_auto_zone_overlays_visible(True)
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
        self._auto_protected_geoms = []
        self._auto_manual_removed = set()
        self._handoff_source_layer = None
        self._set_exemplar_bands_visible(True)
        self._set_auto_zone_overlays_visible(True)
        # Tear the manual session down BEFORE clearing the handoff flag: the
        # teardown's _reset_session suppresses the manual_session_summary event
        # only while the flag is still set, so clearing it first fires a spurious
        # summary for this discarded (never-committed) handoff.
        self._teardown_manual_session()
        self._refine_handoff_active = False
        if self.dock_widget:
            try:
                self.dock_widget._refine_handoff = False
                self.dock_widget.set_protected_note(False)
                self.dock_widget.refine_handoff_banner.setVisible(False)
                self.dock_widget.back_to_review_btn.setVisible(False)
                self.dock_widget.handoff_state_card.setVisible(False)
                # Zero + hide the footer tally so it cannot linger under a torn
                # down handoff (the recap gates on the now-cleared flag).
                self.dock_widget._reset_handoff_counters()
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

    def _saved_index_of(self, entry):
        """Index of an entry in saved_polygons by IDENTITY (entries are stable
        dict objects; indices shift on structural changes), else None."""
        for i, pg in enumerate(self.saved_polygons):
            if pg is entry:
                return i
        return None

    def _hit_test_saved_entry(self, raster_pt):
        """The topmost saved ENTRY containing raster_pt, else None. Last-drawn
        wins overlaps. Uses the token-keyed handoff spatial index when present
        so hover stays cheap over thousands of seeds; falls back to the plain
        scan outside the handoff."""
        pt = QgsGeometry.fromPointXY(raster_pt)
        # The click must land on what the USER SEES on top. In the handoff the
        # kept layer is created after (so renders above) the pending layer, so
        # a kept entry beats any pending one it overlaps; ties break to the
        # highest provider fid (the memory provider iterates fids ascending, so
        # the highest fid within a layer is drawn last = on top). Base Manual
        # draws one band per entry in append order, so there the list index IS
        # the z-order.
        prefer_kept = bool(self._refine_handoff_active)

        def _pick(cands):
            best = None
            for order, pg in cands:
                g = self._entry_geom(pg)
                if g is None or g.isEmpty() or not g.contains(pt):
                    continue
                kept = 1 if (prefer_kept and pg.get("validated")) else 0
                key = (kept, order)
                if best is None or key > best[0]:
                    best = (key, pg)
            return None if best is None else best[1]

        index = getattr(self, "_handoff_hit_index", None)
        if index is not None:
            from qgis.core import QgsRectangle
            x, y = raster_pt.x(), raster_pt.y()
            tok2entry = getattr(self, "_handoff_tok2entry", None) or {}
            cands = []
            for tok in index.intersects(QgsRectangle(x, y, x, y)):
                pg = tok2entry.get(tok)
                if pg is not None:
                    cands.append((pg.get("_hfid", -1), pg))
            return _pick(cands)
        return _pick(enumerate(self.saved_polygons))

    def _hit_test_saved_polygon(self, raster_pt):
        """Index wrapper over _hit_test_saved_entry for the callers that need
        the list position (select / open-for-edit)."""
        entry = self._hit_test_saved_entry(raster_pt)
        return None if entry is None else self._saved_index_of(entry)

    # --- selection-first review of the imported detections --------------------
    # Resting-state model (mirrors the annotation-review standard: selection is
    # never destructive and never triggers the 3-8s SAM encode): hover
    # highlights, click selects, Ctrl+click multi-selects, Suppr deletes the
    # selection instantly, S keeps it, a second click (or E / double-click)
    # opens ONE object for actual SAM editing.

    def _selected_saved_indices(self) -> list:
        """Current selection as indices into saved_polygons (identity-matched:
        entries are stable dict objects, indices shift on structural changes)."""
        sel = getattr(self, "_handoff_selected_entries", None) or []
        if not sel:
            return []
        return [i for i, pg in enumerate(self.saved_polygons)
                if any(pg is e for e in sel)]

    def _select_saved_polygon(self, idx: int, additive: bool = False) -> None:
        """Select the idx-th saved polygon (Ctrl+click toggles membership)."""
        if not (0 <= idx < len(self.saved_polygons)):
            return
        entry = self.saved_polygons[idx]
        sel = list(getattr(self, "_handoff_selected_entries", None) or [])
        if additive:
            for e in sel:
                if e is entry:
                    sel = [x for x in sel if x is not entry]
                    break
            else:
                sel.append(entry)
        else:
            sel = [entry]
        self._handoff_selected_entries = sel
        self._refresh_handoff_selection_band()
        self._notify_handoff_selection()
        self._schedule_handoff_crop_prewarm()

    def _deselect_saved_polygons(self) -> None:
        """Clear the selection (Esc / click on empty ground)."""
        timer = getattr(self, "_handoff_prewarm_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:
                self._handoff_prewarm_timer = None  # C++ side gone (unload)
        if getattr(self, "_handoff_selected_entries", None):
            self._handoff_selected_entries = []
            self._refresh_handoff_selection_band()
            self._notify_handoff_selection()

    # --- speculative selection prewarm ---------------------------------------
    # Opening a detection for SAM editing needs its crop encoded (~3-8s on
    # CPU). The select-then-act flow gives that time away for free: the moment
    # ONE detection is selected, its crop encode can start in the background,
    # so a following Edit (E / double-click / the state-card button) opens with
    # the crop already warm and the first editing click predicts instantly.

    def _schedule_handoff_crop_prewarm(self) -> None:
        """(Re)arm the selection-prewarm debounce. 400 ms: long enough that a
        double-click's opening press never races the open's own encode, short
        enough that a deliberate select-then-Edit gets the crop warm. Armed
        only for a SINGLE selected entry on a LOCAL raster (an online crop
        extraction blocks the GUI on tile fetches, unacceptable per selection
        click)."""
        if not self._refine_handoff_active or self.dock_widget is None:
            return
        timer = getattr(self, "_handoff_prewarm_timer", None)
        sel = getattr(self, "_handoff_selected_entries", None) or []
        if len(sel) != 1 or self._is_online_layer or self._headless:
            if timer is not None:
                timer.stop()
            return
        _debounce_timer(self, "_handoff_prewarm_timer", self.dock_widget, 400,
                        self._maybe_prewarm_selected_crop)

    def _handoff_crop_spec_for(self, geom, anchor_pt) -> tuple:
        """Deterministic crop identity for one detection: center + scale from
        the object's bbox corners (+ an interior anchor, which never widens the
        bounds), exactly what the open computes, so the prewarm and the open
        agree on whether an encode is already covered."""
        bb = geom.boundingBox()
        pts = [(bb.xMinimum(), bb.yMinimum()), (bb.xMaximum(), bb.yMaximum()),
               (anchor_pt.x(), anchor_pt.y())]
        cx, cy, scale = self._compute_crop_center_and_mupp(pts)
        return cx, cy, scale

    def _maybe_prewarm_selected_crop(self) -> None:
        """Speculatively encode the single selected detection's crop (silent:
        no busy cursor, quiet extraction). Every guard re-checks at fire time;
        a stale or wasted prewarm is harmless because the first editing click
        self-heals through _check_crop_status either way."""
        if not self._refine_handoff_active or self._encoding_in_progress:
            return
        skip = self.predictor is None or self._headless or self._is_online_layer
        skip = skip or self._is_refining_saved_object
        skip = skip or self.current_mask is not None
        if skip:
            return
        sel = getattr(self, "_handoff_selected_entries", None) or []
        if len(sel) != 1:
            return
        g = self._entry_geom(sel[0])
        if g is None or g.isEmpty():
            return
        anchor = g.pointOnSurface()
        if anchor is None or anchor.isEmpty():
            return
        pt = anchor.asPoint()
        cx, cy, scale = self._handoff_crop_spec_for(
            g, QgsPointXY(pt.x(), pt.y()))
        spec = (round(cx, 6), round(cy, 6), round(float(scale or 0.0), 6))
        if spec == getattr(self, "_handoff_crop_spec", None) and self._current_crop_info is not None:
            return  # this object's crop is already encoded
        image_np, crop_info = self._extract_crop_only(
            QgsPointXY(cx, cy), scale, quiet=True)
        if image_np is None:
            return
        self._handoff_crop_spec = spec
        QgsMessageLog.logMessage(
            "Refine handoff: prewarming selected detection's crop",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        self._start_manual_encode(image_np, crop_info, None, show_busy=False)

    def _notify_handoff_selection(self) -> None:
        """Push the selection count to the dock state card."""
        sel = getattr(self, "_handoff_selected_entries", None) or []
        if self.dock_widget:
            try:
                self.dock_widget.set_handoff_selected(len(sel))
            except (RuntimeError, AttributeError):
                pass

    def _refresh_handoff_selection_band(self) -> None:
        """Redraw the white selection outline over the selected entries. Prunes
        entries that left saved_polygons (deleted/opened) from the selection."""
        alive = []
        for e in getattr(self, "_handoff_selected_entries", None) or []:
            if any(e is pg for pg in self.saved_polygons):
                alive.append(e)
        self._handoff_selected_entries = alive
        band = getattr(self, "_handoff_selection_band", None)
        if not alive:
            if band is not None:
                band.reset(PolygonGeometry)
            return
        if band is None:
            from qgis.PyQt.QtGui import QColor
            band = QgsRubberBand(self.iface.mapCanvas(), PolygonGeometry)
            # QGIS-native selection yellow, NOT white: the white hover outline
            # and a white selection were near-twins, so Delete acted on a
            # selection made earlier while the user thought it acted on the
            # polygon under the cursor. Yellow = selected is the reflex every
            # QGIS user already has.
            band.setColor(QColor(255, 255, 0, 255))
            band.setFillColor(QColor(255, 255, 0, 60))
            band.setWidth(3)
            self._handoff_selection_band = band
        band.reset(PolygonGeometry)
        displays = []
        for e in alive:
            g = self._entry_geom(e)
            if g is None or g.isEmpty():
                continue
            display = QgsGeometry(g)
            self._transform_geometry_to_canvas_crs(display)
            # Flatten multiparts: collectGeometry must only see single parts
            # to build one clean MultiPolygon on every QGIS 3.22-4.x build.
            if display.isMultipart():
                displays.extend(display.asGeometryCollection())
            else:
                displays.append(display)
        if displays:
            # One collected geometry: every addGeometry call recomputes the
            # band's bounding rect and schedules a canvas update, which adds
            # up on a large Ctrl+multi-selection.
            band.setToGeometry(QgsGeometry.collectGeometry(displays), None)
        band.show()

    def _set_handoff_hover(self, idx) -> None:
        """Show/clear the hover highlight (thin white outline) for idx/None."""
        entry = self.saved_polygons[idx] if idx is not None else None
        self._set_handoff_hover_entry(entry)

    def _set_handoff_hover_entry(self, entry) -> None:
        """Entry-identity variant of _set_handoff_hover: the hover path works
        on entries directly, skipping the O(N) entry-to-index resolution on
        every mouse move."""
        if entry is getattr(self, "_handoff_hover_entry", None):
            return
        self._handoff_hover_entry = entry
        band = getattr(self, "_handoff_hover_band", None)
        if entry is None:
            if band is not None:
                band.reset(PolygonGeometry)
            return
        if band is None:
            from qgis.PyQt.QtGui import QColor
            band = QgsRubberBand(self.iface.mapCanvas(), PolygonGeometry)
            band.setColor(QColor(255, 255, 255, 170))
            band.setFillColor(QColor(255, 255, 255, 18))
            band.setWidth(2)
            self._handoff_hover_band = band
        g = self._entry_geom(entry)
        if g is None or g.isEmpty():
            return
        display = QgsGeometry(g)
        self._transform_geometry_to_canvas_crs(display)
        band.reset(PolygonGeometry)
        band.addGeometry(display, None)
        band.show()

    def _encode_blocks_ui(self) -> bool:
        """True while a FOREGROUND (busy-cursor) encode owns the pipe, which is
        when resting-state gestures defer to it. A silent speculative prewarm
        must never freeze hover or swallow the double-click open: those are
        pure canvas work, and the open attaches to the in-flight prewarm."""
        return bool(self._encoding_in_progress) and bool(
            getattr(self, "_encode_cursor_set", True))

    def _on_handoff_cursor_moved(self, point) -> None:
        """Map-tool hover: highlight the detection under the cursor (handoff
        only; pure canvas work, never a model call)."""
        if not self._refine_handoff_active or self._encode_blocks_ui():
            return
        if not self.saved_polygons:
            return
        try:
            raster_pt = self._transform_to_raster_crs(point)
        except (RuntimeError, AttributeError):
            return
        self._set_handoff_hover_entry(self._hit_test_saved_entry(raster_pt))

    def _click_was_additive(self) -> bool:
        """True when the last map-tool click carried Ctrl/Cmd or Shift (additive
        selection). Qt maps Cmd to ControlModifier on macOS."""
        tool = self.map_tool
        if tool is None:
            return False
        from qgis.PyQt.QtCore import Qt
        mods = getattr(tool, "last_click_modifiers", Qt.KeyboardModifier.NoModifier)
        return bool(mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier))

    def _on_canvas_double_click(self, point) -> None:
        """Double-click on a detection = open it for editing (the first press of
        the pair already selected it). Resting state only: while editing, the
        first press was already routed (point or switch), so this no-ops."""
        if not self._refine_handoff_active or self._encode_blocks_ui():
            return
        if self.current_mask is not None or self._active_crop_points_positive or self._is_refining_saved_object:
            return
        try:
            raster_pt = self._transform_to_raster_crs(point)
        except (RuntimeError, AttributeError):
            return
        if not self._is_point_in_raster_extent(raster_pt):
            return
        idx = self._hit_test_saved_polygon(raster_pt)
        if idx is not None:
            self._open_saved_polygon_for_edit(idx, raster_pt)

    def _delete_selected_saved_polygons(self) -> bool:
        """Instantly delete the selected detections (NO SAM round trip: this is
        a list removal + repaint). One undo unit on the stack. Returns True if
        anything was deleted."""
        idxs = self._selected_saved_indices()
        if not idxs:
            return False
        unit = []
        inc_ok = True
        for i in sorted(idxs, reverse=True):
            pg = self.saved_polygons.pop(i)
            if i < len(self.saved_rubber_bands):
                self._safe_remove_rubber_band(self.saved_rubber_bands.pop(i))
            # Remove BEFORE the undo copy so the snapshot never carries the
            # (now dead) provider bookkeeping keys.
            inc_ok = self._handoff_remove_entry_feature(pg) and inc_ok
            unit.append(dict(pg))
        self._push_deleted_unit(unit)
        self._handoff_selected_entries = []
        self._refresh_handoff_selection_band()
        self._set_handoff_hover(None)
        self._notify_handoff_selection()
        if not inc_ok:
            self._rebuild_handoff_layers()
        if self.dock_widget:
            try:
                self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
                if self._refine_handoff_active:
                    self.dock_widget.note_handoff_shape_removed(len(unit))
            except (RuntimeError, AttributeError):
                pass
        self._update_handoff_progress()
        QgsMessageLog.logMessage(
            f"{len(unit)} object(s) deleted. Ctrl+Z restores them.",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        return True

    def _on_handoff_edit_clicked(self) -> None:
        """State-card Edit shape: opens the single selected detection (the
        button only shows when exactly one is selected)."""
        self._edit_selected_saved_polygon()

    def _on_handoff_delete_clicked(self) -> None:
        """State-card Remove: deletes the open edit or the selection."""
        self._on_delete_active_object()

    def _edit_selected_saved_polygon(self) -> bool:
        """Open the (single) selected detection for SAM editing, seeded at its
        interior point. Returns True if an edit session started."""
        idxs = self._selected_saved_indices()
        if len(idxs) != 1:
            return False
        idx = idxs[0]
        g = self._entry_geom(self.saved_polygons[idx])
        if g is None or g.isEmpty():
            return False
        anchor = g.pointOnSurface()
        if anchor is None or anchor.isEmpty():
            return False
        pt = anchor.asPoint()
        self._open_saved_polygon_for_edit(idx, QgsPointXY(pt.x(), pt.y()))
        return True

    def _open_saved_polygon_for_edit(self, idx: int, raster_pt, label: int = 1) -> None:
        """Enter SAM editing on one saved detection: clear the selection/hover
        first (the entry is about to leave saved_polygons), then activate."""
        self._handoff_selected_entries = []
        self._refresh_handoff_selection_band()
        self._set_handoff_hover(None)
        self._notify_handoff_selection()
        self._activate_saved_polygon_for_refine(idx, raster_pt, label=label)

    def _push_deleted_unit(self, unit: list) -> None:
        """Append one undo unit (list of entry dicts) to the delete stack."""
        stack = getattr(self, "_deleted_objects_stack", None)
        if stack is None:
            stack = []
            self._deleted_objects_stack = stack
        stack.append(unit)
        del stack[:-25]  # bounded: 25 undo units is plenty for a review pass

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
        # This is the single choke point for a saved shape in the handoff (the
        # Save-shape button and the S key both land here), so it drives the
        # footer's "edited" tally.
        if self.dock_widget:
            try:
                self.dock_widget.note_handoff_shape_edited()
            except (RuntimeError, AttributeError):
                pass
        merged = geom
        merged_bb = merged.boundingBox()
        new_polys: list = []
        new_bands: list = []
        absorbed_any = False
        inc_ok = True
        for i in range(len(self.saved_polygons)):
            pg = self.saved_polygons[i]
            rb = self.saved_rubber_bands[i] if i < len(self.saved_rubber_bands) else None
            g = self._entry_geom(pg)
            absorb = False
            # Cheap bbox pre-filter before the costly intersection().
            if g is not None and not g.isEmpty() and merged_bb.intersects(g.boundingBox()) and merged.intersects(g):
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
                absorbed_any = True
                inc_ok = self._handoff_remove_entry_feature(pg) and inc_ok
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
        # The common case absorbs NOTHING: zero provider work (the full
        # both-layers rebuild used to run on every single Save here).
        if absorbed_any:
            if not inc_ok:
                self._rebuild_handoff_layers()
            else:
                try:
                    self._refresh_handoff_selection_band()
                    self._set_handoff_hover(None)
                except (RuntimeError, AttributeError):
                    pass
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

    def _merge_kept_with_protected(self, geoms: list, protected: list) -> list:
        """Re-merge a re-filtered auto detection set against hand-edited
        ``protected`` geometries, scoped to the protected neighbourhood so the
        tail stays O(protected), not O(all detections).

        Drops any detection that overlaps a protected geom by area (same
        object, so the manual edit wins), then dissolves ONLY the protected
        geoms plus the kept detections whose bounding box touches a protected
        bounding box. The dissolve threshold equals the filter threshold, so a
        detection that survived the filter can never reach the dissolve
        threshold against a protected geom; the only pairs the dissolve can
        actually merge live in that neighbourhood, so the untouched remainder
        (kept detections whose bbox touches no protected geom) is appended
        as-is."""
        prot = [g for g in protected if g is not None and not g.isEmpty()]
        if not prot:
            return [g for g in geoms if g is not None and not g.isEmpty()]
        from qgis.core import QgsFeature, QgsSpatialIndex
        # One index over the small protected set, queried per detection (the
        # linear scan per geom is gone). The candidate list it returns is the
        # same bbox pre-filter _geom_overlaps_any applies, so scanning it is
        # identical to scanning all of protected.
        index = QgsSpatialIndex()
        pmap: dict = {}
        for i, g in enumerate(prot):
            feat = QgsFeature(i)
            feat.setGeometry(g)
            index.insertFeature(feat)
            pmap[i] = g
        near: list = []
        remainder: list = []
        for g in geoms:
            if g is None or g.isEmpty():
                continue
            cands = [pmap[fid] for fid in index.intersects(g.boundingBox())]
            if not cands:
                # No protected bbox to touch: this detection can never merge.
                remainder.append(g)
                continue
            if self._geom_overlaps_any(g, cands):
                continue  # same object as a protected geom: the manual edit wins
            near.append(g)
        return self._dissolve_overlapping(near + prot) + remainder

    def _next_handoff_det_id(self) -> int:
        """Next synthetic per-instance id for entries with no canonical det_id
        (hand-drawn saves, legacy reviews). Monotonic within the session."""
        seq = getattr(self, "_handoff_det_id_seq", None)
        if seq is None:
            seq = 100000  # clear of any plausible canonical id range
        self._handoff_det_id_seq = seq + 1
        return seq

    def _dissolve_overlapping_entries(self, entries: list):
        """Identity-aware `_dissolve_overlapping`: entries are (geom, det_id,
        score) triples. Overlapping-by-area geometries union into one whose
        det_id is the FIRST member's (colour stability) and whose score is the
        max (a stitched object is as confident as its best part). Returns the
        aligned (geoms, ids, scores) lists; ids are always ints (synthetic ones
        were assigned at entry creation), scores may carry a 1.0 fallback."""
        from qgis.core import QgsFeature, QgsSpatialIndex
        items = [(g, i, s) for g, i, s in entries
                 if g is not None and not g.isEmpty()]
        if len(items) <= 1:
            geoms = [g for g, _i, _s in items]
            ids = [int(i) if i is not None else self._next_handoff_det_id()
                   for _g, i, _s in items]
            scores = [float(s) if s is not None else 1.0 for _g, _i, s in items]
            return geoms, ids, scores
        index = QgsSpatialIndex()
        keep: dict = {}
        nid = 0
        for g, det_id, score in items:
            merged = g
            m_id = det_id
            m_score = score
            matches = []
            for fid in index.intersects(merged.boundingBox()):
                rec = keep.get(fid)
                if rec is None:
                    continue
                h = rec[0]
                if not merged.intersects(h):
                    continue
                inter = merged.intersection(h)
                if inter is None or inter.isEmpty():
                    continue
                smaller = min(merged.area(), h.area())
                if smaller > 0 and inter.area() / smaller >= self._COMPLETE_OVERLAP_FRAC:
                    matches.append(fid)
            for fid in matches:
                h, h_id, h_score = keep[fid]
                union = merged.combine(h)
                if union is not None and not union.isEmpty():
                    merged = union
                    # The EARLIER keeper's identity wins: its colour is what the
                    # user has been looking at since the run streamed in.
                    if h_id is not None:
                        m_id = h_id if m_id is None else min(m_id, h_id)
                    if h_score is not None:
                        m_score = h_score if m_score is None else max(m_score, h_score)
                    keep[fid] = None
            feat = QgsFeature(nid)
            feat.setGeometry(merged)
            index.insertFeature(feat)
            keep[nid] = (merged, m_id, m_score)
            nid += 1
        geoms, ids, scores = [], [], []
        for rec in keep.values():
            if rec is None:
                continue
            g, i, s = rec
            geoms.append(g)
            ids.append(int(i) if i is not None else self._next_handoff_det_id())
            scores.append(float(s) if s is not None else 1.0)
        return geoms, ids, scores

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
        if self._is_refining_saved_object:
            # An OPEN edit absorbs its neighbours at Save time only
            # (_absorb_overlapping_saved): a live weld here would delete them
            # outside the Ctrl+Z history (undo would shrink the object but
            # never bring the neighbours back).
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
        inc_ok = True
        for i in range(len(self.saved_polygons)):
            pg = self.saved_polygons[i]
            rb = self.saved_rubber_bands[i] if i < len(self.saved_rubber_bands) else None
            g = self._entry_geom(pg)
            absorb = False
            if g is not None and not g.isEmpty() and active_bb.intersects(g.boundingBox()) and active.intersects(g):
                inter = active.intersection(g)
                if inter is not None and not inter.isEmpty():
                    smaller = min(active.area(), g.area())
                    if smaller > 0 and inter.area() / smaller >= self._COMPLETE_OVERLAP_FRAC:
                        absorb = True
            if absorb:
                self._frozen_sessions.append(FrozenCropSession(polygon=g))
                folded = True
                inc_ok = self._handoff_remove_entry_feature(pg) and inc_ok
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
            if not inc_ok:
                self._rebuild_handoff_layers()
            else:
                try:
                    self._refresh_handoff_selection_band()
                    self._set_handoff_hover(None)
                except (RuntimeError, AttributeError):
                    pass
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
        # Geometry-based edit session state dies with the active object too.
        self._unfrozen_display_polygon = None
        self._refine_geom_history = []
        self._refine_edit_pristine = None
        self._refine_edit_last_applied = None
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
                self.dock_widget.set_handoff_editing(False)
            except (RuntimeError, AttributeError):
                pass

    def _on_delete_active_object(self) -> None:
        """Delete the object currently OPEN for editing, or (when nothing is
        open) the current SELECTION. Ctrl+Z restores from the delete stack.
        Only active during a refine handoff or when a saved object is open."""
        if not (self._refine_handoff_active or self._is_refining_saved_object):
            return
        # Selection-first: with no active edit, Suppr rejects the selected
        # detections instantly (no SAM round trip, no open-first detour).
        should_delete_selected = self.current_mask is None and self._active_refine_origin_entry is None
        should_delete_selected = should_delete_selected and not self._active_crop_points_positive
        if should_delete_selected:
            self._delete_selected_saved_polygons()
            return
        # Snapshot for undo: prefer the exact entry re-opened for edit, updated
        # to the CURRENTLY EDITED shape (what the user saw at delete time, not
        # the pre-edit original); otherwise synthesize one from the active mask.
        origin = self._active_refine_origin_entry
        if origin is not None:
            backup = dict(origin)
            base = self._harvest_open_edit_geometry()
            if base is not None and not base.isEmpty():
                backup["geometry_wkt"] = base.asWkt()
                # Keep the cached geometry consistent with the WKT (a stale
                # geom_obj would win in _entry_geom after a restore).
                backup["geom_obj"] = QgsGeometry(base)
                # And drop any pre-edit pristine anchor for the same reason.
                backup.pop("shape_base_wkt", None)
            # Editing clicks and reshapes are hand edits: a restore must keep
            # the object protected from confidence re-filtering (mirrors
            # close-to-pending).
            if getattr(self, "_refine_geom_history", None) or any(self.prompts.point_count):
                backup["manual_touched"] = True
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
                "det_id": self._next_handoff_det_id(),
                "score": None,
            }
        self._push_deleted_unit([backup])
        # Drop any click remembered during this edit's encode (its context is
        # gone; replaying it later would select something out of nowhere).
        self._discard_pending_manual_click()
        self._clear_active_mask_without_saving()
        self._is_refining_saved_object = False
        self._active_refine_origin_entry = None
        if self.dock_widget:
            try:
                self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
                if self._refine_handoff_active:
                    self.dock_widget.note_handoff_shape_removed(1)
            except (RuntimeError, AttributeError):
                pass
        self._update_handoff_progress()
        QgsMessageLog.logMessage(
            "Object deleted. Ctrl+Z restores it.",
            "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _restore_deleted_object(self) -> bool:
        """Pop the last delete-stack UNIT (one Delete press = one unit, possibly
        several selected objects) and re-append its entries as PENDING saved
        polygons, identity intact. Returns True if a restore happened."""
        stack = getattr(self, "_deleted_objects_stack", None) or []
        if not stack:
            return False
        unit = stack.pop()
        restored = 0
        inc_ok = True
        for backup in unit:
            wkt = backup.get("geometry_wkt")
            g = QgsGeometry.fromWkt(wkt) if wkt else None
            if g is None or g.isEmpty():
                continue
            entry = dict(backup)
            entry["validated"] = False  # a restored object is pending again
            self.saved_polygons.append(entry)
            if self._refine_handoff_active:
                # Drawn by the pending layer; None keeps saved_rubber_bands
                # index-locked with saved_polygons.
                self.saved_rubber_bands.append(None)
                inc_ok = self._handoff_add_entry_feature(entry) and inc_ok
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
            restored += 1
        if not restored:
            return False
        if self._refine_handoff_active and not inc_ok:
            self._rebuild_handoff_layers()
        if self.dock_widget:
            try:
                self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
            except (RuntimeError, AttributeError):
                pass
        self._update_handoff_progress()
        return True

    def _activate_saved_polygon_for_refine(self, idx, raster_pt, label: int = 1) -> None:
        """Open an imported detection for editing WITHOUT re-predicting it.

        Opening keeps the shape exactly as the Automatic run (or a previous
        edit) produced it. Editing then behaves exactly like base Manual: the
        first click predicts WITH the object as prior (its polygon is
        rasterized into SAM's mask_input by _run_prediction), so a click just
        outside the shape grows it along the underlying object instead of
        dropping an unrelated island, and every later click continues the same
        refinement chain (accumulated points + logits). The crop encode starts
        here, async, so the first editing click is fast. `label` is kept for
        signature stability; the opening gesture no longer doubles as an
        editing click."""
        entry = self.saved_polygons[idx]
        geom = QgsGeometry.fromWkt(entry.get("geometry_wkt") or "")
        if geom is None or geom.isEmpty():
            return
        # Remove it from the saved set + canvas: it becomes the active selection.
        popped = self.saved_polygons.pop(idx)
        if idx < len(self.saved_rubber_bands):
            self._safe_remove_rubber_band(self.saved_rubber_bands.pop(idx))
        # It just left saved_polygons: drop only ITS feature from the seed
        # layers (the full both-layers rebuild per open WAS the double-click
        # lag on big handoffs).
        if not self._handoff_remove_entry_feature(popped):
            self._rebuild_handoff_layers()
        # This object is now OPEN for editing: rendered in pending-blue with a
        # bolder outline (no separate hue) and the Delete key enabled. Keep the
        # original entry so a Delete-undo can restore it.
        self._is_refining_saved_object = True
        self._active_refine_origin_entry = dict(popped)
        # Per-polygon Shape settings: the panel shows THIS object's stored
        # values, and the pristine geometry anchors non-destructive re-shaping
        # (a settings change always recomputes from it, never compounds).
        self._seed_refine_panel_from_entry(popped)
        self._refine_edit_pristine = QgsGeometry(geom)
        self._refine_edit_last_applied = self._entry_refine_tuple(popped)
        if self.dock_widget:
            try:
                self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
                self.dock_widget.set_handoff_editing(True)
            except (RuntimeError, AttributeError):
                pass

        # The session starts geometry-only (no mask, no prompt points): the
        # display polygon IS the shape until the first editing click seeds a
        # Manual mask session from it. Frozen sessions from a previous edit
        # are cleared defensively so a leak could not be unioned into this
        # object.
        self.current_mask = None
        self.current_score = 0.0
        self.current_low_res_mask = None
        self._frozen_sessions = []
        self._mask_state_history = []
        self._refine_geom_history = []
        self.prompts.clear()
        self._active_crop_points_positive = []
        self._active_crop_points_negative = []
        if self.map_tool:
            try:
                self.map_tool.clear_markers()
            except (RuntimeError, AttributeError):
                pass
        self._unfrozen_display_polygon = geom
        self._update_mask_visualization()

        # Encode a crop that fits the WHOLE object (bbox corners + click), so a
        # large detection is not clipped by a click-centered 1024px crop. The
        # encode is async on the interactive path (PERF-01); clicks that land
        # while it runs are remembered and replayed on completion. When the
        # speculative selection prewarm already encoded (or is encoding) this
        # exact crop, skip the duplicate extract + encode entirely: the first
        # editing click lands on a warm crop. A wrong skip only costs the
        # normal self-heal re-encode at first click, never correctness.
        cx, cy, scale = self._handoff_crop_spec_for(geom, raster_pt)
        spec = (round(cx, 6), round(cy, 6), round(float(scale or 0.0), 6))
        if spec == getattr(self, "_handoff_crop_spec", None) and (
                self._encoding_in_progress or self._current_crop_covers_bbox(geom.boundingBox())):
            return
        # Record the spec only AFTER the encode actually started: a False
        # return (pipe busy with another crop, extraction error) must not
        # stamp this spec, or a later re-open could skip its encode over a
        # neighbouring crop that merely covers the bbox at the wrong scale.
        if self._extract_and_encode_crop(QgsPointXY(cx, cy), mupp_override=scale):
            self._handoff_crop_spec = spec

    def _refine_edit_session_active(self) -> bool:
        """True while a detection is open for editing: the geometry state
        before the first editing click (display polygon only), or the live
        Manual mask session after it (current_mask / frozen crop parts).
        Self-heals the half-open state (flag set but no shape at all, e.g.
        after an interrupted teardown) so a stray click can never fall through
        to the base-Manual new-object path inside a handoff."""
        if not self._is_refining_saved_object:
            return False
        if self.current_mask is not None or self._frozen_sessions:
            return True
        base = self._unfrozen_display_polygon
        if base is None or base.isEmpty():
            self._is_refining_saved_object = False
            self._active_refine_origin_entry = None
            self._refine_geom_history = []
            if self.dock_widget:
                try:
                    self.dock_widget.set_handoff_editing(False)
                except (RuntimeError, AttributeError):
                    pass
            return False
        return True

    def _close_active_edit_to_pending(self) -> None:
        """Close the open edit session WITHOUT validating it: the object (with
        any deltas applied) returns to the pending set, identity intact. Used
        by Esc and by the harvest fallback when a Save is not possible (encode
        in flight). No-op when no edit is open."""
        if not self._is_refining_saved_object:
            return
        base = self._harvest_open_edit_geometry()
        origin = self._active_refine_origin_entry or {}
        appended = None
        if base is not None and not base.isEmpty():
            entry = dict(origin)
            entry["geometry_wkt"] = base.asWkt()
            entry["geom_obj"] = QgsGeometry(base)
            # The edited shape supersedes any pre-edit pristine anchor; a
            # stale one would make a later Shape-settings change erase the
            # deltas by re-shaping from the old geometry.
            entry.pop("shape_base_wkt", None)
            entry["validated"] = False
            # Editing clicks and Shape-settings changes count as hand edits
            # (protected from confidence re-filtering); an untouched close
            # keeps the original flag.
            if getattr(self, "_refine_geom_history", None) or any(self.prompts.point_count):
                entry["manual_touched"] = True
            self.saved_polygons.append(entry)
            appended = entry
            if self._refine_handoff_active:
                # Drawn by the pending layer; None keeps the lists index-locked.
                self.saved_rubber_bands.append(None)
            else:
                rb = QgsRubberBand(self.iface.mapCanvas(), PolygonGeometry)
                rb.setColor(PENDING_FILL)
                rb.setStrokeColor(PENDING_STROKE)
                rb.setWidth(2)
                display_geom = QgsGeometry(base)
                self._transform_geometry_to_canvas_crs(display_geom)
                rb.setToGeometry(display_geom, None)
                self.saved_rubber_bands.append(rb)
        self._is_refining_saved_object = False
        self._active_refine_origin_entry = None
        # A click remembered during this edit's encode belongs to a context
        # that no longer exists: without this, it would replay seconds later
        # in the resting state and select something out of nowhere.
        self._discard_pending_manual_click()
        # Clears the display band, markers, dock counts and the editing flag;
        # also nulls _unfrozen_display_polygon and the delta history.
        self._clear_active_mask_without_saving()
        # The open already dropped this object's feature at activate time, so
        # only the (re)appended entry needs drawing: one incremental add, not
        # a full rebuild of both seed layers.
        if appended is not None and not self._handoff_add_entry_feature(appended):
            self._rebuild_handoff_layers()
        if self.dock_widget:
            try:
                self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
            except (RuntimeError, AttributeError):
                pass
        self._update_handoff_progress()

    def _refine_polygon_mask_input(self):
        """SAM mask_input (low-res logits) built from the OPEN object's current
        display geometry, rasterized onto the encoded crop grid. This is the
        base-Manual context seed: the first editing click predicts WITH the
        object as prior, so it refines the whole shape (a click beside the
        polygon grows it along the underlying object) instead of segmenting an
        unrelated element. None when there is no crop or no geometry."""
        info = self._current_crop_info
        base = self._unfrozen_display_polygon
        if info is None or base is None or base.isEmpty():
            return None
        mask = self._rasterize_geom_to_crop(
            base, info["bounds"], info["img_shape"])
        if mask is None or not mask.any():
            return None
        return self._binary_mask_to_logits(mask)

    def _harvest_open_edit_geometry(self):
        """The open edit's CURRENT shape as one geometry, exactly what the
        canvas shows: the pre-click display polygon, or (after editing clicks)
        the refined active mask, composed with any frozen crop parts. None
        when the session holds no shape at all."""
        parts = [s.polygon for s in self._frozen_sessions
                 if s.polygon is not None and not s.polygon.isEmpty()]
        base = self._unfrozen_display_polygon
        if base is not None and not base.isEmpty():
            parts.append(base)
        active = self._refined_active_mask_geometry()
        if active is not None and not active.isEmpty():
            parts.append(active)
        if not parts:
            return None
        if len(parts) == 1:
            return QgsGeometry(parts[0])
        combined = QgsGeometry.unaryUnion(parts)
        if combined is None or combined.isEmpty():
            return None
        return combined

    def _rasterize_geom_to_crop(self, geom, bounds, img_shape):
        """Rasterize a raster-CRS geometry onto the crop pixel grid (bool
        mask), for pixel-space overlap scoring. None on any failure."""
        try:
            import json as _json

            from rasterio import features
            from rasterio.transform import from_bounds as transform_from_bounds
            minx, miny, maxx, maxy = bounds
            h, w = img_shape
            tfm = transform_from_bounds(minx, miny, maxx, maxy, w, h)
            shape = _json.loads(geom.asJson())
            m = features.rasterize(
                [(shape, 1)], out_shape=(h, w), transform=tfm, fill=0)
            return m.astype(bool)
        except Exception:  # noqa: BLE001 -- scoring aid only, never fatal
            return None
