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
from ...core.review_defaults import REFINE_SMOOTH_ITERATIONS
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
        # The local model packages must be there. get_venv_status below no
        # longer covers them: it answers "can the plugin run", and a venv with
        # no model still runs Automatic. Without this check the install could
        # finish without the model, the venv would read ready, and the failure
        # would surface as a traceback on the first click instead of the
        # one-time-setup offer.
        try:
            from ...core.venv_manager import local_model_ready
            model_ok, _why = local_model_ready()
            if not model_ok:
                self._env_ready = False
                return False
        except Exception:  # nosec B110 -- fail-open on a probe error
            pass
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
            # UI thread, on a click: never a subprocess probe. An environment
            # with no stored deps hash would otherwise pay a cold torch import
            # (up to a minute on Windows) right here, so the first Refine after
            # an upgrade froze the window. Packages on disk count as ready; the
            # background startup check runs the real verification.
            ready, _msg = get_venv_status(allow_subprocess_probe=False)
            self._env_ready = bool(ready)
            return bool(ready)
        except Exception:
            return True  # never block a legitimate refine on a check error

    def _on_reshape_ai_requested(self) -> None:
        """Reshape the SELECTED detection with the on-device AI, in place.

        No mode switch, no separate screen: the review stays open on the Correct
        step and shows the AI-reshaping sub-state. Under the hood this reuses the
        Manual SAM session (start the model on the run's raster, load the
        detections as editable polygons, open the selected one for point-and-
        click), then folds the result back into the review on Done.

        Confidence stays EDITABLE: on return every reshaped object is folded
        into its canonical row and its det_id skips the gates, so a later
        confidence change re-filters only the untouched detections and never
        drops the hand work. When the local AI is not installed, a one-time
        setup runs first and holds the review still while it does
        (local_ai_install_lock); the fix opens itself once the AI is ready
        (see _on_predictor_loaded)."""
        review = self._auto_review
        if not review or not self.dock_widget:
            return
        # A live session or the native bridge already owns the canvas, so ignore
        # a stray Reshape there. The same guard as the Add lane in manual_add,
        # and it is load-bearing here: this method reaches
        # _remove_auto_selection_layer, which would take the layer out of the
        # project while the bridge still holds it in an open edit session, with
        # a vertex tool bound to it and an identity write possibly queued.
        if getattr(self, "_refine_handoff_active", False) or getattr(
                self, "_qgis_bridge_active", False):
            return
        idx = getattr(self, "_correct_selected_idx", None)
        objects = getattr(self, "_auto_objects", None) or []
        if idx is None or idx < 0 or idx >= len(objects):
            return
        layer = self._resolve_auto_source_layer()
        if layer is None:
            return
        # Remember which object to open once the session is up: its interior
        # anchor point, in the run CRS the stored geometries live in.
        geom = objects[idx][0]
        anchor = geom.pointOnSurface() if geom is not None else None
        if anchor is not None and not anchor.isEmpty():
            pt = anchor.asPoint()
            self._reshape_open_anchor = (pt.x(), pt.y())
        else:
            self._reshape_open_anchor = None
        # Env gate: without the local AI the predictor never arrives. Offer the
        # one-time setup, which takes the review until it ends and then opens
        # this polygon itself.
        if not self._manual_env_ready():
            if self._local_ai_install_pending():
                return  # a setup is already running for this review
            # A setup already ran this session and the model still would not
            # load. Re-offering it on every polygon click is a modal the user
            # cannot get rid of, so say it once and leave them on Manual. With
            # no attempt behind it the offer still stands: it is what fixes a
            # truncated model file.
            if (getattr(self, "_local_ai_load_failed", False)
                    and getattr(self, "_local_ai_install_attempted", False)):
                self._warn_local_ai_unavailable_once()
                return
            box = QMessageBox(self.iface.mainWindow())
            box.setWindowTitle(tr("Fixing needs a one-time setup"))
            box.setText(tr(
                "Fixing a polygon uses the free on-device AI, which is not "
                "installed yet. Install it now? It runs once and takes a few "
                "minutes. The review waits for it, then opens this polygon "
                "for you."))
            install_btn = box.addButton(tr("Install now"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(tr("Cancel"), QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(install_btn)
            box.exec()
            if box.clickedButton() is not install_btn:
                return  # review untouched
            # No reshape state here: nothing is being reshaped yet, and a panel
            # showing Save and Undo over an install is a session that does not
            # exist. The install lock is what holds the review still.
            self._begin_local_ai_install("reshape")
            return
        # Drop the resting select tool before the Manual session takes the canvas.
        self._disarm_shape_tool()
        self._handoff_source_layer = layer
        self._pending_refine_import = False
        self._refine_handoff_active = True
        self._auto_refined_in_manual = True  # export will report a reshape was used
        try:
            import time as _time

            from ...core import telemetry_run_events
            self._refine_handoff_t0 = _time.monotonic()
            telemetry_run_events.track_refine_in_manual_entered(
                run_id=self._auto_run_id or "",
                instances=len(review.get("geoms", [])),
            )
        except Exception:
            pass  # nosec B110
        # Hide (do NOT discard) the blue review layer + the zone/example overlays
        # while the SAM edit runs on the same detections as editable polygons.
        self._remove_auto_selection_layer()
        self._set_exemplar_bands_visible(False)
        self._set_auto_zone_overlays_visible(False)
        # Re-anchor the selection ONLY NOW: _remove_auto_selection_layer just
        # disarmed the shape tool, and every disarm clears the panel card. The
        # session works ON this polygon, so the card must come back after the
        # last clearing call or the dock rests on the pick hero with the
        # session's Save out of reach.
        self._correct_selected_idx = idx
        try:
            self.dock_widget.set_correct_selection(1)
        except (RuntimeError, AttributeError):
            pass
        _push = getattr(self, "_push_shape_only_state", None)
        if _push is not None:
            _push()
        try:
            self.dock_widget.enter_ai_reshape_state()
        except (RuntimeError, AttributeError):
            pass
        # Start the Manual SAM session directly (no mode switch): this mirrors
        # the old _on_mode_changed handoff branch. _enter_manual_refine_session
        # imports the detections (or defers on a still-loading predictor); the
        # target object is opened afterwards, here or from _on_predictor_loaded.
        self._ensure_interactive_setup()
        self._enter_manual_refine_session()
        self._open_reshape_target()

    def _open_reshape_target(self) -> None:
        """Open the pre-selected detection for SAM editing, once the Manual
        session has imported the detections. No-op while the import is still
        deferred (predictor loading): _on_predictor_loaded calls this again."""
        if self._pending_refine_import:
            return
        anchor = getattr(self, "_reshape_open_anchor", None)
        self._reshape_open_anchor = None
        if anchor is None or not self.saved_polygons:
            return
        from qgis.core import QgsPointXY
        pt = QgsPointXY(anchor[0], anchor[1])
        # The anchor was read off a canonical object, so it is in the run CRS,
        # while the imported seeds live in the raster CRS. Same conversion the
        # geometries got at import, and the same no-op when the two CRS match.
        pair = getattr(self, "_handoff_crs_pair", None)
        if pair:
            xform = self._handoff_crs_xform(pair[0], pair[1])
            if xform is not None:
                try:
                    pt = xform.transform(pt)
                except Exception:  # noqa: BLE001 -- keep the raw anchor  # nosec B110
                    pass
        idx = self._hit_test_saved_polygon(pt)
        if idx is not None:
            self._open_saved_polygon_for_edit(idx, pt)

    def _clear_refine_install_pending(self) -> None:
        """Drop the pending install-then-fix intent and hide its review banner.

        Idempotent, and never called on its own: `_release_local_ai_install`
        is the one door out of an install, and it clears both lanes."""
        if not getattr(self, "_refine_install_pending", False):
            return
        self._refine_install_pending = False
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_review_installing(False)
            except (RuntimeError, AttributeError):
                pass

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
        user flipped in the review, not just the run's preset. Simplify, Points
        and Min/Max size carry over 1:1: both sides now read the same tolerance
        from a Simplify number, the Points dial is one shared control, and both
        sizes are true ground m2. Expand keeps its Manual value, because it is
        still the odd one out (Manual px is the 1024 SAM mask grid, a different
        scale). Must run AFTER _on_start_segmentation (which resets the
        session) and BEFORE the import. Shared by the direct handoff and the
        deferred (predictor-still-loading) completion in _on_predictor_loaded."""
        try:
            params = self._widget_review_params()
            self._refine_simplify = max(
                0.0, float(params.get("simplify_px") or 0.0))
            self._refine_points_pct = max(
                1, min(100, int(params.get("points_pct") or 100)))
            self._refine_smooth = (
                REFINE_SMOOTH_ITERATIONS if params.get("smooth") else 0)
            self._refine_clean = max(0.0, float(params.get("open_px") or 0.0))
            self._refine_fill_holes = bool(params.get("fill_holes"))
            self._refine_fill_holes_max_m2 = max(
                0.0, float(params.get("fill_max_m2") or 0.0))
            self._refine_ortho = bool(params.get("ortho"))
            self._refine_min_size_m2 = max(0.0, float(params.get("min_a") or 0.0))
            self._refine_max_size_m2 = max(0.0, float(params.get("max_a") or 0.0))
            self.dock_widget.set_refine_values(
                self._refine_simplify, self._refine_smooth,
                self._refine_expand, self._refine_fill_holes,
                right_angles=self._refine_ortho,
                fill_holes_max_m2=self._refine_fill_holes_max_m2,
                clean=self._refine_clean,
                points_pct=self._refine_points_pct)
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

    # --- the one CRS boundary between the review and the Manual session ------
    # The review carries its geometries in the RUN CRS. Manual works end to end
    # in the RASTER CRS: the canvas transform, the crop bounds every rasterize
    # reads, and the raster-CRS points a click produces. The two CRS are the
    # same on every raster whose own CRS the run kept, so the conversion below
    # is a no-op there. Where they differ, converting here (and back on the
    # harvest) is what keeps one CRS on each side instead of two inside Manual.

    def _handoff_crs_xform(self, src_authid, dst_authid):
        """ONE transform between two CRS authids, built at a handoff boundary
        and reused for every geometry crossing it. None when either authid is
        missing, the two are equal, or the transform cannot be built: each of
        those means leave the geometries alone."""
        if not src_authid or not dst_authid or src_authid == dst_authid:
            return None
        try:
            from qgis.core import (
                QgsCoordinateReferenceSystem,
                QgsCoordinateTransform,
            )
            src = QgsCoordinateReferenceSystem(src_authid)
            dst = QgsCoordinateReferenceSystem(dst_authid)
            if not src.isValid() or not dst.isValid():
                return None
            xform = QgsCoordinateTransform(src, dst, QgsProject.instance())
            return xform if xform.isValid() else None
        except (RuntimeError, AttributeError, TypeError):
            return None

    @staticmethod
    def _handoff_reproject(geom, xform):
        """A COPY of `geom` in the transform's destination CRS, None when the
        transform refuses it. Never mutates the input: an entry geometry is
        shared with the seed layers and the hit index."""
        try:
            out = QgsGeometry(geom)
            out.transform(xform)
            return None if out.isEmpty() else out
        except Exception:  # noqa: BLE001 -- caller keeps the original shape
            return None

    @staticmethod
    def _handoff_layer_authid(layer):
        """A raster layer's CRS authid, or None when it has no valid one."""
        try:
            if layer is None:
                return None
            crs = layer.crs()
            return crs.authid() if crs is not None and crs.isValid() else None
        except (RuntimeError, AttributeError):
            return None

    def _handoff_entries_to_run_crs(self, entries: list) -> list:
        """Harvested (geom, det_id, score, touched) rows moved from the raster
        CRS Manual worked in back to the run CRS the review and the canonical
        objects are stored in, so the Automatic review stays in one CRS.

        Returns `entries` unchanged when the import found the two CRS equal,
        which is every raster whose own CRS the run kept. A geometry the
        transform refuses is passed through as it is: a shape in the wrong CRS
        can still be moved by hand, a dropped one is gone."""
        pair = getattr(self, "_handoff_crs_pair", None)
        self._handoff_crs_pair = None
        if not pair or not entries:
            return entries
        run_authid, raster_authid = pair
        xform = self._handoff_crs_xform(raster_authid, run_authid)
        if xform is None:
            QgsMessageLog.logMessage(
                "Refine handoff: no transform back to the run CRS; "
                f"{len(entries)} shape(s) kept in {raster_authid}.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return entries
        out, failed = [], 0
        for g, det_id, score, touched in entries:
            moved = self._handoff_reproject(g, xform)
            if moved is None:
                failed += 1
                moved = g
            out.append((moved, det_id, score, touched))
        if failed:
            QgsMessageLog.logMessage(
                f"Refine handoff: {failed} shape(s) kept in {raster_authid}, "
                "the transform back refused them.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return out

    def _import_review_geoms_as_saved(self, review) -> None:
        """Load the review geometries into saved_polygons as 'pending' seeds,
        converting them from the run CRS into the raster CRS Manual works in
        (see the boundary note above; a no-op when the two match). They are
        rendered by ONE in-memory layer (blue), NOT a rubber band each: a
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
        run_authid = crs.authid() if crs is not None and crs.isValid() else None
        raster_authid = self._handoff_layer_authid(
            getattr(self, "_handoff_source_layer", None))
        # Convert the seeds into the raster CRS once, here, so everything inside
        # Manual keeps reading one CRS. None when the two already match (the
        # usual case) or when one of them is unusable, and then nothing moves.
        self._handoff_crs_pair = None
        xform = self._handoff_crs_xform(run_authid, raster_authid)
        if xform is not None:
            self._handoff_crs_pair = (run_authid, raster_authid)
        # The seed layers are declared in the CRS the entries end up in: the
        # raster CRS whenever the raster has one, else the run CRS the review
        # carried. Geoms are pushed directly with no per-object canvas transform
        # (the layer reprojects for display).
        authid = raster_authid or run_authid
        if authid:
            self._ensure_handoff_layers(authid)
        else:
            # Neither side named a usable CRS, so there is nothing to declare a
            # memory layer in and the handoff runs with no seeds on the canvas.
            # Say so: this used to fail silently.
            QgsMessageLog.logMessage(
                "Refine handoff: neither the run nor the raster has a valid "
                "CRS; no seed layers were created.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        # Synthetic det_id sequence for objects with no canonical id (hand-drawn
        # saves, legacy reviews without ids): keeps every entry hue-stable and
        # the return arrays free of NULLs. Seeded above every CANONICAL id, not
        # just the ones this review carries: the review holds the VISIBLE set,
        # so a detection the confidence or size filter is hiding keeps an id
        # that seeding from the review alone would hand to the next object the
        # user adds. The fold then matches that object to the hidden row and
        # writes over it, and the added shape gets no row of its own.
        max_id = max((int(i) for i in ids if i is not None), default=-1)
        for fid in (getattr(self, "_auto_object_fids", None) or ()):
            if isinstance(fid, int) and fid > max_id:
                max_id = fid
        self._handoff_det_id_seq = max_id + 1
        # Carry the "hand edited" mark across repeat refine visits: an object
        # whose det_id was folded/exempted on an earlier pass re-imports already
        # marked touched, so it stays exempt from the confidence/size gates.
        exempt_ids = set(getattr(self, "_auto_manual_object_ids", None) or ())
        # Which canonical objects this session can speak for. The import carries
        # the VISIBLE set only, so a detection the confidence or size filter is
        # hiding was never on the user's canvas and they cannot have deleted it.
        # _removed_canonical_objects reads this to scope the deletion diff.
        imported_ids: set[int] = set()
        failed = 0
        for n, g in enumerate(geoms):
            if g is None or g.isEmpty():
                continue
            if xform is not None:
                moved = self._handoff_reproject(g, xform)
                if moved is None:
                    failed += 1  # kept as it is: never lose the user's shape
                else:
                    g = moved
            det_id = ids[n] if n < len(ids) and ids[n] is not None else None
            if det_id is None:
                det_id = self._next_handoff_det_id()
            imported_ids.add(int(det_id))
            score = scores[n] if n < len(scores) and scores[n] is not None else None
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
                "refine_points_pct": self._refine_points_pct,
                "refine_smooth": self._refine_smooth,
                "refine_clean": self._refine_clean,
                "refine_expand": self._refine_expand,
                "refine_fill_holes": self._refine_fill_holes,
                "refine_fill_holes_max_m2": self._refine_fill_holes_max_m2,
                "refine_ortho": self._refine_ortho,
                "refine_min_area": self._refine_min_area,
                "refine_min_size_m2": self._refine_min_size_m2,
                "refine_max_size_m2": self._refine_max_size_m2,
                "manual_touched": det_id in exempt_ids,
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
        if failed:
            QgsMessageLog.logMessage(
                f"Refine handoff: {failed} shape(s) kept in {run_authid}, "
                f"the transform to {raster_authid} refused them.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        self._handoff_imported_det_ids = imported_ids
        self._rebuild_handoff_layers()
        if self.dock_widget:
            try:
                self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
            except (RuntimeError, AttributeError):
                pass

    def _on_reshape_done(self) -> None:
        """Finish the live fix session: fold the edits back into the held review,
        rebuild the blue review layer in place, and return to the Correct step.
        No mode switch, so the review is never left.

        The panel's Done routes here for both methods. A native QGIS bridge edit
        commits through its own path, so route it there; the AI point-refine
        folds below."""
        if getattr(self, "_qgis_bridge_active", False):
            self.finish_qgis_edit_bridge()
            return
        if not self._refine_handoff_active:
            # The flag is already down, but the CANVAS may not be. Any path that
            # drops it without taking the point tool back leaves that tool armed
            # and the last edit's prompt markers painted, and then every click on
            # empty ground opens a refine point out of nowhere. Nothing to fold
            # here, so sweep and leave.
            self._sweep_stale_refine_canvas()
            return
        # The Add lane rides this session; leaving it drops the lane arm.
        if getattr(self, "_refine_add_mode_active", False):
            try:
                self._exit_ai_add_mode()
            except (RuntimeError, AttributeError):
                pass
        self._collect_manual_refine_into_review()
        try:
            import time as _time

            from ...core import telemetry_run_events
            t0 = getattr(self, "_refine_handoff_t0", None)
            telemetry_run_events.track_refine_in_manual_back(
                run_id=self._auto_run_id or "",
                validated_count=len((self._auto_review or {}).get("geoms", [])),
                duration_ms=int((_time.monotonic() - t0) * 1000) if t0 else None,
            )
        except Exception:
            pass  # nosec B110
        # Rebuild the review UI + blue layer (this clears _refine_handoff_active),
        # then leave the reshape sub-state and re-arm the resting select tool.
        self._restore_auto_review_after_handoff()
        try:
            self.dock_widget.leave_ai_reshape_state()
        except (RuntimeError, AttributeError):
            pass
        self._arm_correct_select()

    def _sweep_stale_refine_canvas(self) -> None:
        """Leave nothing of a fix session on the canvas. Idempotent, and a no-op
        outside a review so a Manual session's own points are never touched.

        The session flag and the canvas are two different pieces of state, and
        the exits do not all move them together. When they disagree the tool
        stays live under a dock that says the session is over, which is the one
        failure the user cannot undo by clicking: every click makes it worse.
        Sweeping here covers every exit at once, rather than trusting each one.
        """
        if getattr(self, "_auto_review", None) is None:
            return  # Manual mode owns its own points; not ours to clear
        # The canvas focus is part of that leftover state: without this, an exit
        # that never reached _restore_auto_review_after_handoff would leave the
        # map grey and the busy cursor on.
        self._end_correct_focus()
        tool = getattr(self, "map_tool", None)
        if tool is not None:
            try:
                tool.clear_markers()
            except (RuntimeError, AttributeError):
                pass
            try:
                canvas = self.iface.mapCanvas()
                if canvas.mapTool() is tool:
                    canvas.unsetMapTool(tool)
            except (RuntimeError, AttributeError):
                pass
        try:
            self.dock_widget.leave_ai_reshape_state()
        except (RuntimeError, AttributeError):
            pass
        # Back to the resting pick tool, so the next click selects a polygon
        # instead of landing on whatever tool the canvas kept.
        self._arm_correct_select()

    def _collect_manual_refine_into_review(self) -> None:
        """Fold every manual edit (saved + any in-progress mask) back into
        _auto_review["geoms"], then tear the manual session down."""
        # If the import never completed (the predictor was still loading when
        # the user backed out), there is no manual session to harvest: leave the
        # held review untouched so Back to review restores it intact.
        if self._pending_refine_import:
            self._pending_refine_import = False
            self._handoff_crs_pair = None
            self._teardown_manual_session()
            return
        review = self._auto_review
        if review is None:
            # Nothing to fold back into, so no CRS boundary to cross either.
            self._handoff_crs_pair = None
        else:
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
            for pg in self.saved_polygons:
                g = self._entry_geom(pg)
                if g is not None and not g.isEmpty():
                    entries.append((g, pg.get("det_id"), pg.get("score"),
                                    bool(pg.get("manual_touched"))))
            # Back to the run CRS the review and the canonical objects speak,
            # before anything downstream compares areas or overlaps against
            # them. No-op unless the import had to convert on the way in.
            entries = self._handoff_entries_to_run_crs(entries)
            # Dissolve any remaining overlaps so the committed output is uniform
            # (never stacked layers), while distinct touching objects stay split.
            # Identity-aware: a dissolved group keeps its first member's det_id
            # (so the Random colour survives the round trip) and its max score,
            # instead of dropping both lists and reshuffling every colour.
            geoms, ids, scores = self._dissolve_overlapping_entries(
                [(g, i, s) for g, i, s, _t in entries])
            review["geoms"] = geoms
            review["scores"] = scores
            review["ids"] = ids
            # Harvested geoms are hand-edited/dissolved, NOT the refine cache's
            # output: drop the provenance stamp so the next review push does a
            # full rebuild instead of wrongly diffing against pre-handoff state.
            review["stamp"] = None
            # Every hand edit becomes the polygon's canonical base, so the Shapes
            # pipeline drives it like any other object on the next reslice (no
            # freeze). The reslice itself waits for _restore_auto_review_after_
            # handoff to recreate the selection layer to push onto.
            self._fold_manual_refine_into_objects(entries, geoms, ids, scores)
            # No lock: this only disarms any stray armed gesture on return.
            self._disarm_after_handoff()
        self._teardown_manual_session()

    def _fold_manual_refine_into_objects(self, entries, geoms, ids, scores) -> None:
        """Fold the harvested hand edits into the canonical object rows.

        A reshaped detection overwrites its own canonical row (matched by
        det_id) with the repaired geometry, its carried score and a re-measured
        area, so a later Shapes reslice drives it from that base. Brand-new
        objects append. Deletions during the session are remembered in
        _auto_manual_removed. Every touched or added det_id skips the
        confidence/size gates. The whole fold is journalled so one Undo restores
        the pre-fold bases, pops the appended rows, drops the exemptions it
        added and puts back the removal set, strictly LIFO with merge/remove.
        """
        objects = getattr(self, "_auto_objects", None)
        if objects is None:
            return
        manual_removed_before = set(getattr(self, "_auto_manual_removed", None) or ())
        pre_len = len(objects)
        # Brand-new manual objects append canonical rows; their fresh det_ids
        # come back so the journal can drop their exemption on undo.
        added_ids = self._register_manual_only_review_objects(geoms, ids, scores)
        appended = len(objects) - pre_len
        # Existing detections the user reshaped: overwrite their base by det_id.
        fids = list(getattr(self, "_auto_object_fids", None) or [])
        by_id = {fid: idx for idx, fid in enumerate(fids)}
        measurer = self._make_auto_area_measurer()
        from ...core.layer_conventions import repair_polygon, to_multipolygon
        manual_ids = self._auto_manual_object_ids
        restored: list[tuple[int, object]] = []
        exempted: list[int] = []
        for g, det_id, score, touched in entries:
            if not touched or not isinstance(det_id, int):
                continue
            index = by_id.get(det_id)
            if index is None or index >= pre_len:
                continue  # a brand-new object, handled by the append path above
            repaired = to_multipolygon(repair_polygon(g) or g)
            if repaired is None or repaired.isEmpty():
                continue
            restored.append((index, objects[index]))
            carried = objects[index][1] if score is None else float(score)
            objects[index] = (
                repaired, float(carried), self._object_area_m2(repaired, measurer))
            if det_id not in manual_ids:
                manual_ids.add(det_id)
                exempted.append(det_id)
        for det_id in added_ids:
            if det_id not in exempted:
                exempted.append(det_id)
        # Deletions: recomputed against the folded bases, so a reshaped object
        # (its base now IS the edited geometry) is never read as removed.
        new_removed = self._removed_canonical_objects(geoms)
        self._auto_manual_removed = new_removed
        from ...core.shape_edits import KIND_REFINE, ShapeEdit
        edit = ShapeEdit(
            kind=KIND_REFINE,
            restored=tuple(restored),
            appended=appended,
            unremoved=(),
            exempted=tuple(exempted),
        )
        # Snapshot the removal set only when it actually moved, so a Done that
        # changed nothing (opened, looked, backed out) never journals a no-op.
        self._record_fold_edit(
            edit, fids=tuple(exempted),
            manual_removed_before=(manual_removed_before
                                   if manual_removed_before != new_removed
                                   else None))
        # Match the bridge fold's invalidation so the reslice recomputes cleanly.
        self._shape_hit_geoms = {}
        self._reset_review_refine_cache()
        try:
            pixel_size = (self._auto_review or {}).get("pixel_size", 1.0)
            self._start_build_preview_cache(pixel_size)
        except (RuntimeError, AttributeError):
            pass

    def _register_manual_only_review_objects(self, geoms, ids, scores) -> list:
        """Make every Manual-created review object addressable by Correct.

        Appends a canonical row for each review geometry whose det_id has no
        canonical object yet (a hand-drawn add), and marks that det_id exempt
        from the gates so a slider set for the detections never hides the user's
        own drawing. Returns the det_ids it appended, for the fold journal."""
        added: list[int] = []
        if not isinstance(ids, list):
            return added
        objects = getattr(self, "_auto_objects", None)
        if objects is None:
            return added
        known = {self._object_fid_for(index) for index in range(len(objects))}
        fids = list(getattr(self, "_auto_object_fids", None) or [])
        measurer = self._make_auto_area_measurer()
        manual_ids = self._auto_manual_object_ids
        imported = getattr(self, "_handoff_imported_det_ids", None) or set()
        for index, geom in enumerate(geoms or []):
            det_id = ids[index] if index < len(ids) else None
            if not isinstance(det_id, int) or geom is None or geom.isEmpty():
                continue
            if det_id in known:
                # An id this session imported is a detection the user reshaped:
                # the caller overwrites its canonical row, so not appending it
                # is right. Any other known id is a clash, and the shape behind
                # it would take a row that belongs to another object without a
                # word. Say so rather than let it pass as an ordinary skip.
                if det_id not in imported:
                    QgsMessageLog.logMessage(
                        f"Refine handoff: det_id {det_id} is already taken by "
                        f"another object; the added shape got no row of its own.",
                        "AI Segmentation", level=Qgis.MessageLevel.Warning)
                continue
            score = scores[index] if isinstance(scores, list) and index < len(scores) else 1.0
            try:
                score = float(score) if score is not None else 1.0
            except (TypeError, ValueError):
                score = 1.0
            objects.append((geom, score, self._object_area_m2(geom, measurer)))
            fids.append(det_id)
            known.add(det_id)
            if det_id not in manual_ids:
                manual_ids.add(det_id)
            added.append(det_id)
        self._auto_object_fids = fids
        return added

    def _removed_canonical_objects(self, kept_geoms: list) -> set:
        """Indices into _auto_objects of detections deleted during the Manual
        refine: canonical objects no longer meaningfully covered (>= 30% of
        their area) by any harvested geometry. Spatial-index candidates keep
        this linear-ish; it runs once per Back-to-review.

        Scoped to the det_ids the session actually IMPORTED. The import carries
        the visible set, so walking the whole canonical list read every
        detection the confidence or size filter was hiding as deleted: opening
        a fix session and closing it without touching anything wiped them, and
        a later slider move could no longer bring them back. With no recorded
        import set, nothing is treated as deleted: under-reporting a deletion
        costs one extra polygon, over-reporting one destroys work.
        """
        objects = getattr(self, "_auto_objects", None) or []
        session_ids = getattr(self, "_handoff_imported_det_ids", None)
        if not objects or not session_ids:
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
            if self._object_fid_for(det_idx) not in session_ids:
                continue  # filtered out of this session: not the user's doing
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
        """Rebuild the Automatic review UI + blue layer after a Manual refine,
        then reslice so the folded bases drive the display like any other
        Shapes change. Confidence stays editable: the touched objects are
        exempt from the gates, so they survive every later slider move."""
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
            except (RuntimeError, AttributeError):
                pass
        # Push the harvested interim set, then reslice from the folded canonical
        # objects (the selection layer exists again now, so the reslice has a
        # target to push onto).
        self._refresh_auto_review_preview()
        self._start_auto_reslice()
        # Every polygon gets its colour and its clicks back. Last, so the
        # repaint lands on the layer that was just rebuilt.
        self._end_correct_focus()

    # _teardown_manual_session lives in ManualWorkflowMixin (the session
    # owner); the harvest paths above call it through the assembled class.

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
        """The topmost saved ENTRY under raster_pt that the fix session allows,
        else None. Last-drawn wins overlaps, and a blocked entry on top is
        stepped over rather than eating the click (see the loop below).
        Uses the token-keyed handoff spatial index when present
        so hover stays cheap over thousands of seeds; falls back to the plain
        scan outside the handoff.

        This is also the gate for "one polygon at a time": while a fix session
        owns one, every other entry reads as a miss, so neither the hover
        outline nor the click-to-switch can leave the polygon under edit. The
        session's own polygon still answers, which is how the AI path opens it
        (`_open_reshape_target` comes through here too)."""
        pt = QgsGeometry.fromPointXY(raster_pt)
        # The click must land on what the USER SEES on top. In the handoff the
        # kept layer is created after (so renders above) the pending layer, so
        # a kept entry beats any pending one it overlaps; ties break to the
        # highest provider fid (the memory provider iterates fids ascending, so
        # the highest fid within a layer is drawn last = on top). Base Manual
        # draws one band per entry in append order, so there the list index IS
        # the z-order.
        prefer_kept = bool(self._refine_handoff_active)

        def _ranked(cands):
            """Every entry under the point, topmost first. `intersects` rather
            than `contains` so a click on a shared edge hits both neighbours
            instead of missing both."""
            hits = []
            for order, pg in cands:
                g = self._entry_geom(pg)
                if g is None or g.isEmpty() or not g.intersects(pt):
                    continue
                kept = 1 if (prefer_kept and pg.get("validated")) else 0
                hits.append(((kept, order), pg))
            hits.sort(key=lambda h: h[0], reverse=True)
            return [pg for _key, pg in hits]

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
            ranked = _ranked(cands)
        else:
            ranked = _ranked(enumerate(self.saved_polygons))
        # A blocked entry sitting on top must not swallow the click. Walk down
        # to the first entry the fix session allows: during a session that is
        # its own polygon, so an overlapping neighbour can no longer make it
        # unclickable. Outside a session nothing blocks, so the topmost wins
        # exactly as before.
        for pg in ranked:
            if not self._correct_focus_blocks_det_id(pg.get("det_id")):
                return pg
        return None

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
        """The crop window one detection gets refined in: ``(cx, cy, scale)``.

        On a local raster the window comes from the shared grid in
        `core/crop_window.py`, so neighbouring detections land on ONE crop and
        the second of them opens with no encode at all. Online layers keep the
        canvas-mupp path, which is a different unit and has no warm-up. The
        interior anchor is accepted for signature stability; it sits inside the
        bounds, so it never moved the window."""
        from ...core.crop_window import crop_window_for_object
        bb = geom.boundingBox()
        bounds = (bb.xMinimum(), bb.yMinimum(), bb.xMaximum(), bb.yMaximum())
        held = getattr(self, "_encoded_crop_window", None)
        if self._is_online_layer:
            # An online crop is described by a ground size per pixel rather than
            # a zoom-out factor on native pixels, so the same grid is asked for
            # in that unit: one ground unit per pixel, floored at the canvas
            # resolution. Sharing matters more here than anywhere else, since a
            # crop the model already holds is a set of map tiles NOT fetched.
            return crop_window_for_object(
                bounds, 1.0, held_window=held,
                min_scale=self.iface.mapCanvas().mapUnitsPerPixel(),
                max_scale=float("inf"))
        return crop_window_for_object(
            bounds, self._get_native_pixel_size(), held_window=held)

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
        from ...core.crop_window import crop_window_key
        cx, cy, scale = self._handoff_crop_spec_for(
            g, QgsPointXY(pt.x(), pt.y()))
        spec = crop_window_key(cx, cy, scale)
        if spec in (getattr(self, "_encoded_crop_window", None),
                    getattr(self, "_inflight_crop_window", None)):
            return  # this object's crop is encoded, or on its way
        QgsMessageLog.logMessage(
            "Refine handoff: prewarming selected detection's crop",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        # Off the GUI thread and silent: a selection click must not freeze for a
        # crop nobody asked for yet, and must not report a read nobody wanted.
        self._extract_and_encode_crop(
            QgsPointXY(cx, cy), mupp_override=scale, show_busy=False, quiet=True)

    def _notify_handoff_selection(self) -> None:
        """Push the selection count to the dock state card, then keep the Correct
        panel synced to the polygon the session is now on (facts, merge)."""
        sel = getattr(self, "_handoff_selected_entries", None) or []
        if self.dock_widget:
            try:
                self.dock_widget.set_handoff_selected(len(sel))
            except (RuntimeError, AttributeError):
                pass
        if len(sel) == 1:
            self._sync_correct_panel_to_handoff_entry(sel[0])

    def _sync_correct_panel_to_handoff_entry(self, entry) -> None:
        """Keep the round-3 panel pointed at the polygon the session is editing.

        In-session clicks move between polygons through the handoff (not through
        _set_correct_selection), so the panel's facts line and Merge tile would
        otherwise freeze on the first polygon. Resolve the entry's stable det_id
        back to its canonical row and re-drive the panel (title stays; the class
        is per-run). Best-effort and handoff-only; never opens a new session."""
        if not getattr(self, "_refine_handoff_active", False) or self.dock_widget is None:
            return
        det_id = entry.get("det_id") if isinstance(entry, dict) else None
        idx = None
        if det_id is not None:
            resolve = getattr(self, "_object_index_for_det_id", None)
            if resolve is not None:
                idx = resolve(det_id)
        if idx is None:
            return
        self._correct_selected_idx = idx
        try:
            self.dock_widget.set_correct_selection(1)
            self.dock_widget.enter_ai_reshape_state()
            self.dock_widget.set_merge_available(
                self._selected_has_mergeable_neighbor(idx))
        except (RuntimeError, AttributeError):
            pass
        _push = getattr(self, "_push_shape_only_state", None)
        if _push is not None:
            _push()

    def _set_ai_session_armed_line(self, loading: bool) -> None:
        """The panel armed line during an AI fix session: an honest loading note
        while the imagery around the polygon is being read, then the keep/trim
        gesture help once the crop is ready. Handoff-only; add mode has its own
        lane line, so it is left untouched there.

        The waiting outline and the busy cursor ride this same signal, because
        every path that starts a model wait and every path that ends one
        (completion, read failure, teardown) already passes here. They move
        BEFORE the two guards below: a session torn down mid-read still has to
        get its cursor back, and a line nobody shows is not a reason to keep
        QGIS looking busy."""
        if loading:
            self._begin_correct_wait()
        else:
            self._end_correct_wait()
        if not getattr(self, "_refine_handoff_active", False) or self.dock_widget is None:
            return
        if getattr(self, "_refine_add_mode_active", False):
            return
        try:
            if loading:
                self.dock_widget.set_correct_armed_line(
                    tr("Reading the imagery around this polygon..."), "info")
            else:
                self.dock_widget.set_correct_armed_line(
                    tr("Left-click adds a keep point, right-click a trim point. "
                       "The outline follows."), "armed")
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
        """Map-tool hover during a session on the map.

        In the review's fix session it highlights the detection under the cursor
        (pure canvas work, never a model call). In plain Manual there is no
        detection to highlight, so a resting cursor is used for the one thing it
        does say: where the next click is going, and therefore which imagery to
        have ready."""
        if not self._refine_handoff_active:
            self._schedule_manual_hover_warm(point)
            return
        if self._encode_blocks_ui():
            return
        if not self.saved_polygons:
            return
        try:
            raster_pt = self._transform_to_raster_crs(point)
        except (RuntimeError, AttributeError):
            return
        if raster_pt is None:
            return  # cursor is outside the raster CRS domain
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

    # Fraction of the SMALLER object's area that must overlap for two entries of
    # the SAME run to be treated as one object when the entry list is merged.
    # Tile fragments of one object overlap on the seam; distinct neighbours that
    # merely touch share ~zero area, so they stay separate (the instance count
    # is preserved).
    _COMPLETE_OVERLAP_FRAC = 0.1

    # Same ratio, but for absorbing an EXISTING saved detection into a shape the
    # user is drawing now. It has to mean "one of the two is essentially inside
    # the other", because absorbing deletes the neighbour outside the Ctrl+Z
    # history: _snapshot_mask_state carries neither _frozen_sessions nor
    # saved_polygons, so a wrongly eaten polygon cannot be undone. A big new
    # shape clipping a corner off a small old one is a neighbour, not a
    # completion, and must leave it alone and reachable.
    _ABSORB_COVER_FRAC = 0.8

    def _absorb_overlapping_saved(self, geom):
        """Refine handoff only: union `geom` with any already-saved detections it
        genuinely OVERLAPS (shared area, not a mere shared edge) and drop those,
        so a NEW shape drawn over an existing one grows into one polygon instead
        of stacking a layer on top. Distinct neighbours that only touch are left
        alone. Returns the (possibly grown) geometry."""
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
        # Saving an object that was OPENED for editing never eats its
        # neighbours. One object is edited at a time, and a click can no longer
        # grow it over another detection (_grow_open_object_with_click), so an
        # overlap here can only be one the run itself produced. Swallowing a
        # neighbour over 10% of it, silently and outside the undo history, is not
        # an edit of the object the user opened. Joining two detections is what
        # Merge with neighbours is for.
        if self._is_refining_saved_object:
            return geom
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
                    if smaller > 0 and inter.area() / smaller >= self._ABSORB_COVER_FRAC:
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

    def _next_handoff_det_id(self) -> int:
        """Next synthetic per-instance id for entries with no canonical det_id
        (hand-drawn saves, legacy reviews). Monotonic within the session, and
        never an id a canonical object already holds: the fold matches a
        harvested shape to a canonical row BY det_id, so a reused id reads as
        an object the review already knows, and the user's new shape overwrites
        that row instead of getting one."""
        seq = getattr(self, "_handoff_det_id_seq", None)
        if seq is None:
            seq = 100000  # clear of any plausible canonical id range
        taken = {fid for fid in (getattr(self, "_auto_object_fids", None) or ())
                 if isinstance(fid, int)}
        while seq in taken:
            seq += 1
        self._handoff_det_id_seq = seq + 1
        return seq

    def _dissolve_overlapping_entries(self, entries: list):
        """Union overlapping-by-area entries, identity-aware: entries are (geom,
        det_id, score) triples. Overlapping-by-area geometries union into one
        whose det_id is the FIRST member's (colour stability) and whose score is
        the max (a stitched object is as confident as its best part). Returns the
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
                    if smaller > 0 and inter.area() / smaller >= self._ABSORB_COVER_FRAC:
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

        # Encode a crop that fits the WHOLE object, so a large detection is not
        # clipped by a click-centered 1024px crop. Three cases, and only the
        # third makes the user wait:
        #   - the predictor already HOLDS this window (a neighbour opened it, or
        #     the hover warm-up did): ready for keep/trim clicks right now;
        #   - a read for this exact window is in flight: attach to it rather
        #     than starting a second one, and put the busy cursor on it, since
        #     a wait the hover warm-up started cursor-less is now a wait the
        #     user asked for;
        #   - otherwise start it, with the busy cursor: the panel line names
        #     the wait, and the cursor is what says it is still running. The
        #     line alone left the pointer looking idle over a dead canvas for
        #     the seconds the read takes, which reads as a crash.
        # The comparison is against windows that were actually encoded or are
        # actually being read, never against an intent, so a failed read can
        # never pass as a warm crop.
        from ...core.crop_window import crop_window_key
        cx, cy, scale = self._handoff_crop_spec_for(geom, raster_pt)
        spec = crop_window_key(cx, cy, scale)
        if (spec == getattr(self, "_encoded_crop_window", None) and self._current_crop_info is not None):
            self._set_ai_session_armed_line(loading=False)
            return
        if (spec == getattr(self, "_inflight_crop_window", None) and self._encoding_in_progress):
            self._wear_busy_cursor_for_crop()
            self._set_ai_session_armed_line(loading=True)
            return
        if self._extract_and_encode_crop(
                QgsPointXY(cx, cy), mupp_override=scale, show_busy=True):
            # Honest wait: the imagery is being read; the gesture help returns
            # when the encode completes (_on_manual_encode_done).
            self._set_ai_session_armed_line(loading=True)

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

    def _shape_in_progress_geometry(self):
        """Everything the object being edited holds right now, EXCEPT the live
        mask: the display polygon plus every part frozen when a click moved the
        session to another crop.

        The frozen parts matter. A click outside the encoded crop freezes the
        shape so far and reads new imagery, and the shape has to arrive on that
        new crop as context or the click segments whatever sits under it as an
        unrelated object: click a house, then click the garden beside it, and
        the garden came back on its own instead of the house growing into it.
        """
        parts = [s.polygon for s in self._frozen_sessions
                 if s.polygon is not None and not s.polygon.isEmpty()]
        base = self._unfrozen_display_polygon
        if base is not None and not base.isEmpty():
            parts.append(base)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        combined = QgsGeometry.unaryUnion(parts)
        if combined is None or combined.isEmpty():
            return None
        return combined

    def _refine_polygon_mask_input(self):
        """SAM mask_input (low-res logits) built from the shape being edited,
        rasterized onto the encoded crop grid. This is the base-Manual context
        seed: a click predicts WITH the shape so far as prior, so it refines
        that shape (a click beside it grows it along the underlying object)
        instead of segmenting an unrelated element.

        The rasterization clips to the crop, so a shape that shares no pixels
        with the current crop yields None and the click starts clean."""
        info = self._current_crop_info
        base = self._shape_in_progress_geometry()
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

    def _other_objects_mask_for_crop(self, bounds, img_shape):
        """Every OTHER detection's ground on this crop's pixel grid, or None.

        Editing runs on one object at a time, so the shape being edited must not
        grow over its neighbours: an overlap reads as two objects claiming the
        same ground, and the save used to answer that by swallowing the
        neighbour whole. The object being edited is not in here, since opening
        it took it out of ``saved_polygons``.
        """
        if not self._refine_handoff_active or not self.saved_polygons:
            return None
        try:
            from qgis.core import QgsRectangle
            crop = QgsRectangle(bounds[0], bounds[1], bounds[2], bounds[3])
            parts = []
            for entry in self.saved_polygons:
                g = self._entry_geom(entry)
                if g is None or g.isEmpty():
                    continue
                if not crop.intersects(g.boundingBox()):
                    continue
                parts.append(g)
            if not parts:
                return None
            union = parts[0] if len(parts) == 1 else QgsGeometry.unaryUnion(parts)
            if union is None or union.isEmpty():
                return None
            return self._rasterize_geom_to_crop(union, bounds, img_shape)
        except Exception:  # noqa: BLE001 -- a click must not fail over this
            return None

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
