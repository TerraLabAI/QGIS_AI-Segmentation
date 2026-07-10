"""Post-run review: confidence re-filter, reslice, export, exit/retry.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations


from qgis.core import (
    Qgis,
    QgsFeature,
    QgsGeometry,
    QgsMessageLog,
)
from qgis.PyQt.QtWidgets import (
    QMessageBox,
)

from ...core.i18n import tr
from ...core.telemetry import slot_guard
from .shared import (
    _add_features_fast,
)

# Single-slot memo for the merge-policy token/category sets, keyed on the policy
# merge-dict id so a repeated _default_merge_separate call does not re-normalize
# the lists. One live policy dict at a time, so a new id replaces the entry (no
# growth, no stale id-reuse across distinct dicts).
_MERGE_SETS_CACHE: dict[str, object] = {"id": None, "sets": None}


def _merge_token_sets(merge: dict) -> tuple[frozenset, frozenset, frozenset]:
    """(continuous_tokens, discrete_tokens, continuous_categories) as sets.

    Tokens are normalized lower + underscore->space (matching the prompt
    normalization); categories are lowercased only (preset categories carry
    underscores, e.g. ``land_water``). Empty sets when the policy omits a list.
    """
    key = id(merge)
    if _MERGE_SETS_CACHE["id"] == key and _MERGE_SETS_CACHE["sets"] is not None:
        return _MERGE_SETS_CACHE["sets"]  # type: ignore[return-value]

    def _tokens(vals: object) -> frozenset:
        return frozenset(
            str(v).strip().lower().replace("_", " ")
            for v in (vals or []) if isinstance(v, str)) if isinstance(vals, list) else frozenset()

    def _cats(vals: object) -> frozenset:
        return frozenset(
            str(v).strip().lower()
            for v in (vals or []) if isinstance(v, str)) if isinstance(vals, list) else frozenset()

    result = (
        _tokens(merge.get("continuous_tokens")),
        _tokens(merge.get("discrete_tokens")),
        _cats(merge.get("continuous_categories")),
    )
    _MERGE_SETS_CACHE["id"] = key
    _MERGE_SETS_CACHE["sets"] = result
    return result


class AutoReviewMixin:
    """Post-run review: confidence re-filter, reslice, export, exit/retry."""

    # ---- Post-run review helpers -------------------------------------------

    def _auto_seam_min_dim(self) -> float:
        """Bbox max-dim (ground units) below which a detection always fits whole
        in one tile, so it can only be a cross-tile duplicate, never a seam-split
        half. This is exactly the inter-tile overlap span; it drives the
        IncrementalMerger size-aware gate so distinct small neighbours (solar
        panels, cars, trees) with a gap between them are not fused.

        BOTH policies use the overlap span when the GSD is known. SEPARATE used
        to return +inf (nothing seam-eligible, dedup only), which worked only
        because the parent hypothesis masks glued seam halves transitively; the
        per-tile hypothesis NMS removed those parents, so without the size gate
        big buildings render truncated along the tile grid. With the span, an
        object larger than the overlap strip (the only kind a seam can cut)
        matches its other half at merge_ios, and the merger's selection branch
        unions the pair only when it genuinely extends (seam stitch) while
        still selecting among redundant same-footprint readings, so distinct
        objects are never fused and counting stays safe. Unknown GSD: +inf for
        SEPARATE (strict dedup, counting-safe), 0.0 for CONTINUOUS (gate off,
        original merge behaviour)."""
        from ...core.tile_manager import TILE_SIZE, OVERLAP_FRACTION

        if self._auto_gsd <= 0:
            return float("inf") if self._auto_merge_separate else 0.0
        return OVERLAP_FRACTION * TILE_SIZE * self._auto_gsd

    def _default_merge_separate(self, prompt: str) -> bool:
        """Smart default for the merge policy from the object token.

        True = keep SEPARATE (count). False = MERGE split pieces (map continuous
        features). Continuous when: the preset is flagged ``weak`` (continuous
        land cover), the token is a known linear/continuous feature, or its
        preset category is in a continuous family. A short list of discrete
        countable objects is forced SEPARATE even when its category is
        continuous. Defaults to SEPARATE (counting-safe) for everything else and
        on any lookup error.

        The server policy refines the default: the token and category lists
        arrive in the review.merge policy. Without it those lists are empty, so
        only the preset ``weak`` flag routes to MERGE and everything else applies
        the counting-safe SEPARATE default."""
        token = (prompt or "").strip().lower()
        if not token:
            return True
        # Normalise underscores/spaces so "farm_field" and "farm field" match.
        norm = token.replace("_", " ")
        from ...core.detection_policy import merge_policy
        continuous_tokens, discrete_tokens, continuous_categories = _merge_token_sets(
            merge_policy())
        # Discrete countable objects win over a continuous category.
        if norm in discrete_tokens or token in discrete_tokens:
            return True
        if norm in continuous_tokens or token in continuous_tokens:
            return False
        try:
            from ...core.presets import segmentation_presets as _sp
            for preset in _sp.all_presets():
                if str(preset.get("prompt", "")).strip().lower().replace("_", " ") != norm:
                    continue
                if preset.get("weak"):
                    return False
                cat = str(preset.get("category", "")).lower()
                if cat in continuous_categories:
                    return False
                break
        except Exception:  # noqa: BLE001 -- never block a run on a preset lookup  # nosec B110
            pass
        return True

    def _start_auto_reslice(self) -> None:
        """Cooperatively re-derive the review's VISIBLE geometry set from the
        canonical WHOLE objects at the current confidence + min/max size +
        shape-refine settings: filter each whole object by score and area, then
        apply the shape refine to those that pass. A pure recompute on the
        already-merged objects (NO re-merge, so an object is NEVER re-cut; no
        re-detection, no credits). Time-sliced via the filter phase of
        _step_auto_finalize_refine so it never freezes; generation-guarded so a
        later reslice or a new run supersedes an in-flight one (last one wins)."""
        if not self._auto_review:
            return
        self._auto_finalize_gen += 1
        self._review_push_err_logged = False  # new generation: allow one log again
        self._auto_finalize_state = {
            "mode": "reslice",
            "phase": "filter",
            # Enumerated so each visible geom carries its canonical det_id and
            # the Random colours stay stable across reslices.
            "filter_pending": list(enumerate(self._auto_objects)),
            "total_filter": len(self._auto_objects),
            "visible": [],
            "visible_scores": [],
            "visible_ids": [],
            "params": self._widget_review_params(),
            "pixel_size": (self._auto_review or {}).get("pixel_size", 1.0),
            "gen": self._auto_finalize_gen,
        }
        self._step_auto_finalize_refine()

    def _on_auto_show_tiles_toggled(self, show: bool) -> None:
        """Review debug toggle: overlay the tile grid on the finished result, or
        hide it again. The grid is cleared while a run is in flight (so the user
        watches segmentations appear cleanly); this brings it back on demand to
        inspect tile seams or which tile a detection came from."""
        if not show:
            self._clear_zone_tile_grid()
            return
        layer = self._get_active_raster_layer()
        if layer is None:
            return
        try:
            grid = self._compute_auto_grid(layer)
        except (RuntimeError, AttributeError):
            grid = None
        if grid is not None:
            self._show_zone_tile_grid(layer, grid)

    def _on_auto_review_confidence_preview(self, percent: int) -> None:
        """Live preview WHILE the confidence slider is dragged: re-show the
        detections at the new cutoff. The geometries are pre-simplified ONCE in
        _auto_preview_geoms (sorted by score desc), so a drag tick is just a
        prefix slice - no per-tick simplify, no merge, no orthogonalize. The
        accurate rebuild runs on release via _on_auto_review_confidence_changed."""
        if not self._auto_review:
            return
        conf = max(0.0, min(1.0, percent / 100.0))
        preview = []
        pscores = []
        pids = []
        # Objects deleted during a Manual refine stay deleted whatever the
        # cutoff (the caches predate the handoff, so filter at consumption).
        removed = getattr(self, "_auto_manual_removed", None) or set()
        if self._auto_preview_geoms:
            # Fast path: the cache is built (sorted by score desc), so the cutoff
            # is a prefix slice - no per-tick simplify.
            for geom, score, det_idx in self._auto_preview_geoms:
                if score < conf:
                    break  # everything after is below the cutoff
                if det_idx in removed:
                    continue
                preview.append(geom)
                pscores.append(score)
                pids.append(self._object_fid_for(det_idx))
        else:
            # Fallback while the background cache build is still running: filter
            # the canonical WHOLE objects directly (correct, just heavier). Whole
            # objects, never fragments, so a drag never shows half a building. The
            # accurate size filter + shape refine still runs on release.
            for det_idx, (g, s, _a) in enumerate(self._auto_objects):
                if det_idx in removed:
                    continue
                if s >= conf and g is not None:
                    preview.append(g)
                    pscores.append(s)
                    pids.append(self._object_fid_for(det_idx))
        self._push_review_geoms(preview, repair=False, scores=pscores, ids=pids)

    def _on_auto_review_confidence_changed(self, percent: int) -> None:
        """Review confidence slider released: re-filter the stored detections at
        the new cutoff and refresh the review preview, cooperatively (time-sliced)
        so a dense result never freezes. No server call, no credits."""
        if not self._auto_review:
            return
        self._auto_confidence = max(0.0, min(1.0, percent / 100.0))
        self._start_auto_reslice()
        # Telemetry: count the move and emit one review_confidence_final after
        # the slider settles (2s of no further change).
        self._review_conf_moves = getattr(self, "_review_conf_moves", 0) + 1
        from qgis.PyQt.QtCore import QTimer
        timer = getattr(self, "_review_conf_timer", None)
        if timer is None:
            # The plugin controller is a plain class, not a QObject; parent the
            # timer to the dock so Qt owns (and tears down) its lifetime.
            timer = QTimer(self.dock_widget)
            timer.setSingleShot(True)
            timer.timeout.connect(self._emit_review_confidence_final)
            self._review_conf_timer = timer
        timer.start(2000)

    def _emit_review_confidence_final(self) -> None:
        if not self._auto_review:
            return
        try:
            from ...core import telemetry
            telemetry.track_review_confidence_final(
                run_id=self._auto_run_id or "",
                final_pct=int(round((self._auto_confidence or 0.0) * 100)),
                visible_count=len(self._auto_review.get("geoms", [])),
                moves=getattr(self, "_review_conf_moves", 0),
            )
        except Exception:
            pass  # nosec B110

    def _refresh_auto_review_preview(self) -> None:
        """Re-populate the live selection layer with the current VISIBLE object
        set. _auto_review["geoms"] IS that visible set (already whole-object
        filtered by confidence + size and shape-refined by the cooperative
        reslice), so this is a plain push: the display, the Manual handoff and the
        Export all read the same single source of truth. Called when the review is
        entered and after every reslice completes.
        """
        if self._auto_review is None:
            return
        self._push_review_geoms(
            self._auto_review["geoms"], scores=self._auto_review.get("scores"),
            ids=self._auto_review.get("ids"))

    def _push_review_geoms(self, geoms: list, repair: bool = True,
                           scores: list | None = None,
                           ids: list | None = None) -> None:
        """Write geoms onto the live review selection layer (truncate + add +
        repaint) and update the review count. Shared by the accurate refresh and
        the fast confidence-drag preview. ``repair=False`` skips the per-geom
        makeValid for the fast path (raw geoms are usually valid; the accurate
        pass on release repairs anyway). ``scores`` is an optional parallel list
        (same order/length as ``geoms``) written to the per-object 'score' field
        so the review heatmap colors each detection; a neutral 1.0 fallback keeps
        any post-handoff or mismatched case green/trusted rather than crashing.
        ``ids`` is the parallel canonical-detection index written to 'det_id':
        the Random display mode hues on it so an object keeps its colour across
        reslices (NULL fallback lets the renderer hue on $id instead)."""
        layer = self._auto_selection_layer
        if layer is None:
            return
        try:
            if not layer.isValid():
                return
            from ...core.layer_conventions import repair_polygon, to_multipolygon

            pr = layer.dataProvider()
            pr.truncate()
            features_to_add = []
            for i, geom in enumerate(geoms):
                if geom is None or geom.isEmpty():
                    continue
                if repair:
                    geom = to_multipolygon(repair_polygon(geom) or geom)
                else:
                    geom = to_multipolygon(geom) or geom
                if geom is None or geom.isEmpty():
                    continue
                feat = QgsFeature(layer.fields())
                feat.setGeometry(geom)
                feat.setAttributes(
                    ["", float(scores[i]) if (scores is not None and i < len(scores))
                     else 1.0,
                     int(ids[i]) if (ids is not None and i < len(ids)) else None])
                features_to_add.append(feat)
            if features_to_add:
                _add_features_fast(pr, features_to_add)
            # updateExtents rescans every feature (O(N) on a memory provider):
            # only the accurate release pass needs it (zoom-to-layer); the 40ms
            # drag preview renders by viewport via the spatial index, so the
            # stale cached extent is invisible mid-drag.
            if repair:
                layer.updateExtents()
            # triggerRepaint alone schedules the canvas update; the extra
            # mapCanvas().refresh() forced a full re-render of EVERY layer on
            # each (debounced) slider tick, which is what made review sliders lag.
            layer.triggerRepaint()
            # Keep the review count honest with the size filter (it can hide
            # detections): show how many are actually on the layer now.
            self._update_review_header(len(features_to_add))
        except Exception as e:  # noqa: BLE001
            # Review must never crash the UI, so the guard stays broad; but a
            # swallowed geometry error was fully silent. Log once per rebuild
            # generation (reset in _start_auto_reslice) so a real bug surfaces
            # without spamming the log on every confidence-drag tick.
            if not self._review_push_err_logged:
                self._review_push_err_logged = True
                QgsMessageLog.logMessage(
                    f"Auto review: geometry rebuild error: {e}",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)

    def _on_auto_refine_changed_debounced(self) -> None:
        """Slot connected to auto_refine_changed; restarts the 150 ms debounce timer.

        The timer's timeout is wired to _start_auto_reslice once in
        _ensure_dock_widget. This slot just restarts it so rapid spinbox / slider
        changes coalesce into a single cooperative recompute of the visible set.
        """
        if not self.dock_widget:
            return
        try:
            self.dock_widget._auto_review_debounce_timer.start(150)
        except (RuntimeError, AttributeError):
            pass

    def _update_review_header(self, visible: int) -> None:
        """Push the honest two-line review header + Export label: ``visible`` are
        shown now, total = all found whole objects, pct = current cutoff."""
        if not self.dock_widget:
            return
        try:
            total = len(self._auto_objects)
            pct = int(round((self._auto_confidence or 0.0) * 100))
            # When nothing is visible, tell the user which filter is actually
            # hiding the objects so they reach for the right lever: the Min size
            # filter can hide everything even when Confidence would show them.
            size_bound = (visible == 0 and total > 0
                          and self._review_zero_is_size_bound())
            self.dock_widget.update_auto_review_count(
                visible, total, pct, size_bound=size_bound)
        except (RuntimeError, AttributeError):
            pass

    def _review_zero_is_size_bound(self) -> bool:
        """With nothing visible, decide whether the Min size filter (not
        Confidence) is what hides the objects: True when at least one found
        object scores at/above the current Confidence cutoff (so Confidence is
        NOT the binding filter and the size gate must be), so the header can
        point at the lever that will actually reveal them."""
        conf = self._auto_confidence or 0.0
        removed = getattr(self, "_auto_manual_removed", None) or set()
        for det_idx, (base, score, _area) in enumerate(self._auto_objects):
            if det_idx in removed or base is None or base.isEmpty():
                continue
            if score >= conf:
                return True
        return False

    def _current_visible_review_count(self) -> int:
        """Objects currently shown in the review (the last pushed visible set)."""
        review = self._auto_review or {}
        return len(review.get("geoms", []))

    def _full_found_review_geoms(self) -> tuple[list, list | None]:
        """The review's found objects with the Confidence gate dropped but the
        current size + shape refine kept. The safety-net exit paths export this
        so a billed detection hidden ONLY by the Confidence cutoff is never
        lost. Hand-edited (protected) objects are merged back in, mirroring the
        confidence reslice, so a Manual-refine detour survives the save too."""
        review = self._auto_review or {}
        pixel_size = review.get("pixel_size", 1.0) or 1.0
        params = dict(self._widget_review_params())
        params["conf"] = 0.0  # drop only the confidence gate; keep size + shape
        geoms, scores = self._compute_visible_objects(
            params, pixel_size, with_scores=True)
        if self._auto_protected_geoms:
            protected = self._auto_protected_geoms
            kept = [g for g in geoms if not self._geom_overlaps_any(g, protected)]
            geoms = self._dissolve_overlapping(kept + list(protected))
            scores = None  # dissolve rewrote geoms; parallel scores no longer align
        return geoms, scores

    def _export_auto_review(self, include_hidden: bool = False
                            ) -> tuple[str | None, int] | None:
        """Apply the refine settings to the pending review geometries, export
        them to a GeoPackage layer, clear the review state, and return
        (layer_name, polygon_count). Returns None when there is nothing to
        export. Shared by the interactive Export button and the headless MCP
        path so both commit the review identically.

        ``include_hidden`` is set ONLY on the safety-net exit paths (teardown
        autosave, the review Exit dialog's Save): a paid detection hidden by the
        Confidence cutoff must not be silently lost, so when the visible set is
        smaller than the full found set the FULL set is exported instead
        (confidence gate dropped, the user's size + shape refine kept). The
        normal Finish button leaves it False and exports exactly the visible set
        the user sees.
        """
        review = self._auto_review
        if not review:
            return None
        # Normally export EXACTLY the current VISIBLE set (review["geoms"] is
        # already the confidence + size filtered, shape-refined objects), so
        # what the user sees on the map is what gets saved. The safety-net exit
        # paths (include_hidden) instead export the full found set when the
        # cutoff is hiding billed detections. Copy each geom below so the
        # export's makeValid never mutates the stored review geometry.
        geoms = review["geoms"]
        scores = review.get("scores")
        if include_hidden:
            visible_n = sum(1 for g in geoms if g is not None and not g.isEmpty())
            full_geoms, full_scores = self._full_found_review_geoms()
            if len(full_geoms) > visible_n:
                geoms, scores = full_geoms, full_scores
        if scores is not None and len(scores) != len(geoms):
            scores = None  # e.g. after a protected-geoms dissolve
        refined, refined_scores = [], []
        for index, g in enumerate(geoms):
            if g is None or g.isEmpty():
                continue
            refined.append(QgsGeometry(g))
            refined_scores.append(scores[index] if scores else None)
        name = self._export_auto_detections(
            refined, review["crs"], review["source_layer_name"], review["prompt"],
            scores=refined_scores)
        # Capture the run's REAL outcome (chosen confidence, refine settings,
        # the kept geometry - even after a Refine-in-Manual detour) on a hidden
        # background task. Best-effort: queued only after the local export
        # succeeded, and can never block or fail it.
        try:
            from .run_export_upload import queue_run_export_upload
            queue_run_export_upload(self, review, refined, refined_scores)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        # Record the committed object so the Segment library's Recent tab can
        # re-run it. Runs once per commit (this is the shared interactive +
        # headless-MCP path); best-effort, never blocks the export.
        try:
            from ...core.presets import segment_history
            segment_history.add_recent(
                review.get("prompt", ""),
                detections=len(refined),
                detail=(self._auto_run_ctx or {}).get("detail"),
            )
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        try:
            from ...core import telemetry
            found = len(self._auto_objects)
            telemetry.track_auto_export_done(
                run_id=self._auto_run_id or "",
                exported_count=len(refined),
                visible_pct_of_found=int(round(len(refined) / found * 100)) if found else 0,
                final_confidence=int(round((self._auto_confidence or 0.0) * 100)),
                display_mode=self._auto_display_mode,
                refined_in_manual=getattr(self, "_auto_refined_in_manual", False),
            )
            if refined:
                telemetry.track_first_generation_milestone(mode="auto")
        except Exception:
            pass  # nosec B110
        self._auto_review = None
        self._auto_objects = []
        self._remove_auto_selection_layer()
        self._auto_protected_geoms = []
        self._auto_manual_removed = set()
        self._auto_refined_in_manual = False
        # A background install started from this review is now orphaned (the
        # review is committed): drop the pending refine so a late predictor load
        # does not auto-open a handoff on a gone or a different review.
        self._clear_refine_install_pending()
        if self.dock_widget:
            try:
                self.dock_widget.set_protected_note(False)
                self.dock_widget.set_auto_review_active(False)
            except (RuntimeError, AttributeError):
                pass
        return name, len(refined)

    @slot_guard(stage="export", user_message=tr(
        "Something went wrong saving your detections. Please try again."))
    def _on_auto_export_clicked(self) -> None:
        """Finish: commit the reviewed detections to a layer, then return to the
        Start step (pick a layer, begin a new segmentation), layer unlocked."""
        # Snapshot the prompt BEFORE _export_auto_review nulls the review, so the
        # end-of-run value recap (shown on the Start page) can name the object.
        recap_prompt = ((self._auto_review or {}).get("prompt") or "").strip()
        exported = self._export_auto_review()
        if exported is None:
            return
        name, count = exported
        status = tr("Saved {n} polygon(s) to {name}").format(n=count, name=name or "")
        # Free users: teach the meter by appending the balance (Moment E). The
        # run already consumed its credits DURING detection, so the cached
        # snapshot is the correct post-run value.
        try:
            credits_left, is_free = self._auto_credit_snapshot()
            if is_free and credits_left is not None:
                status += " " + tr("{remaining} free detections left.").format(
                    remaining=credits_left)
        except (RuntimeError, AttributeError):
            pass
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_status("info", status)
            except (RuntimeError, AttributeError):
                pass
        # Capture the remaining recap facts while the run context + clip polygon
        # still exist (reset clears the zone). credits_used = billed tiles of the
        # run; credits_left = post-run balance snapshot (may be None if the usage
        # refresh has not returned, in which case the recap drops the balance).
        try:
            recap_area = self._auto_zone_area_km2()
            recap_used = (self._auto_run_ctx or {}).get("total")
            recap_left, _is_free = self._auto_credit_snapshot()
        except Exception:  # nosec B110 -- recap is best-effort
            recap_area, recap_used, recap_left = 0.0, None, None
        self._reset_auto_for_new_run()
        # Value recap on the Start page (session only). Entirely best-effort:
        # the export already succeeded, so the recap must never raise here.
        try:
            if self.dock_widget and recap_used is not None:
                self.dock_widget.set_last_run_recap(
                    count=count,
                    object_word=recap_prompt or tr("Example match"),
                    area_km2=recap_area,
                    credits_used=recap_used,
                    credits_left=recap_left,
                )
        except Exception:  # nosec B110 -- never break Finish on the recap
            pass

    def _reset_auto_for_new_run(self) -> None:
        """After Finish: return to the Start step (pick a layer, begin a new
        segmentation) with the layer unlocked, rather than jumping straight back
        to drawing a zone. The committed detections stay on the map."""
        # Disarm the zone drawing tool if it is still active.
        self._restore_maptool_after_zone()
        # A committed run consumed its plan; the next run re-fetches per prompt.
        self._auto_run_plan = None
        self._cancel_task("_auto_run_plan_task")
        self._auto_zone = None
        self._auto_zone_polygon = None
        self._clear_auto_canvas()
        if self.dock_widget:
            try:
                # Back to the Start step (layer editable), like Exit, so the
                # user re-picks a layer and starts fresh.
                self.dock_widget.reset_auto_to_start()
            except (RuntimeError, AttributeError):
                pass

    def _discard_review_without_autosave(self) -> None:
        """Drop the pending review OUTPUT without the autosave that
        _discard_auto_review performs. Shared by Adjust & run again and Exit so
        the discard cleanup lives in exactly one place. Clears the review, the
        canonical objects, the selection layer, the protected/handoff markers,
        and supersedes any in-flight cooperative finalize/reslice."""
        self._auto_review = None
        self._auto_objects = []
        self._remove_auto_selection_layer()
        self._auto_protected_geoms = []
        self._auto_manual_removed = set()
        self._auto_refined_in_manual = False
        self._auto_finalize_gen += 1
        self._auto_finalize_state = None
        self._auto_preview_geoms = []
        # Orphan any background install started from this review (see above).
        self._clear_refine_install_pending()
        # Turn the review UI OFF here, in the one shared discard spot: the
        # Exit path used to skip it, leaving _auto_review_active stuck True
        # so the NEXT run's prompt step re-opened on the stale review panel.
        if self.dock_widget:
            try:
                self.dock_widget.set_protected_note(False)
                self.dock_widget.set_auto_review_active(False)
            except (RuntimeError, AttributeError):
                pass

    def _on_auto_review_exit_clicked(self) -> None:
        """Exit from the review: offer to Save (export) the detections, Discard
        them, or Cancel, so a billed result is NEVER silently dropped nor
        silently autosaved. On Save/Discard, leave to the Start step (unlocked)."""
        visible = self._current_visible_review_count()
        total = len(self._auto_objects)
        if total > 0 and not self._auto_headless_run:
            # Save exports the FULL found set (Confidence gate dropped, size +
            # shape kept) whenever it is larger than the visible set, so a billed
            # detection hidden by Confidence is never lost. The label must state
            # the count that will ACTUALLY be saved, not the smaller visible one.
            full_geoms, _full_scores = self._full_found_review_geoms()
            save_count = max(len(full_geoms), visible)
            hidden = save_count - visible
            if hidden > 0:
                label = tr(
                    "Save {save} detections ({hidden} currently hidden by "
                    "Confidence) to a layer before leaving?").format(
                        save=save_count, hidden=hidden)
            else:
                label = tr(
                    "Save {save} detections to a layer before leaving?").format(
                        save=save_count)
            box = QMessageBox(self.iface.mainWindow())
            box.setWindowTitle(tr("Keep your detections?"))
            box.setText(label)
            save_btn = box.addButton(tr("Save && exit"), QMessageBox.ButtonRole.AcceptRole)
            drop_btn = box.addButton(
                tr("Discard && exit"), QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(tr("Cancel"), QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(save_btn)
            box.exec()
            clicked = box.clickedButton()
            if clicked is save_btn:
                # Safety net: the dialog offered to save detections hidden by
                # Confidence, so export the FULL found set, not the (possibly
                # empty) visible one, or the promise silently drops paid work.
                self._export_auto_review(include_hidden=True)
                self._reset_auto_for_new_run()
                return
            if clicked is not drop_btn:
                return                              # Cancel: review intact
        self._discard_review_without_autosave()     # Discard: no autosave
        self._reset_auto_for_new_run()              # back to step 0, unlocked

    def _on_auto_retry_clicked(self) -> bool:
        """Retry: drop the just-finished detection results and return to the
        prompt step with the SAME zone, references and settings intact, so the
        user can adjust (sizes, examples, detail) and re-detect the same zone
        without redrawing it or re-selecting examples.

        Unlike Finish (which exports) and Exit (which drops everything), Retry is
        non-destructive: it does NOT auto-save the discarded results (the user is
        re-running, not committing, so an autosave would spawn a junk layer). The
        zone, its polygon, the exemplar store, _auto_run_ctx and every review
        widget value are left untouched; only the run OUTPUT is cleared.

        Discarding a billed result is confirmed first: a bare "Retry" hid that
        the detections are dropped and the next Detect bills again.

        Returns True when the retry proceeded (confirmed), False when the user
        cancelled the discard, so callers (the exemplar nudge) can chain a
        follow-up action only on a real retry."""
        total = len(self._auto_objects)
        discarded = len((self._auto_review or {}).get("geoms", [])) or total
        confirmed = True
        if total > 0 and not self._auto_headless_run:
            box = QMessageBox(self.iface.mainWindow())
            box.setWindowTitle(tr("Discard these detections?"))
            box.setText(tr(
                "Your {total} detections will be discarded. You keep your zone, "
                "object and settings. Running Detect again will use new credits."
            ).format(total=total))
            discard_btn = box.addButton(
                tr("Discard && adjust"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(tr("Cancel"), QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(discard_btn)
            box.exec()
            confirmed = box.clickedButton() is discard_btn
        try:
            from ...core import telemetry
            telemetry.track_auto_retry_clicked(
                run_id=self._auto_run_id or "",
                discarded_count=discarded,
                confirmed=confirmed,
            )
        except Exception:
            pass  # nosec B110
        if not confirmed:
            return False
        # Clear the review OUTPUT without the autosave _discard_auto_review does.
        self._discard_review_without_autosave()
        if not self.dock_widget:
            return True
        try:
            self.dock_widget.set_protected_note(False)
            self.dock_widget.set_auto_review_active(False)
            # Bring back the zone fill the review had dropped, so the kept zone
            # reads clearly on the map again.
            self._set_zone_band_fill_visible(True)
            # Return to the prompt step (step 2) with the zone kept + locked; the
            # references are still in the store, just re-show their chips.
            self.dock_widget.set_auto_zone_state("zone_set")
            # Redraw the tile-grid preview and re-show the detail slider + cost for
            # the kept zone (both are cleared during a run/review), so the user can
            # see and control the tiles again before re-detecting.
            self._update_credit_estimate()
            self._refresh_exemplar_chips()
            self.dock_widget.set_auto_status("idle")
        except (RuntimeError, AttributeError):
            pass
        return True

    def _on_auto_exit_clicked(self) -> None:
        """Exit the Automatic flow from the prompt step.

        Drops the zone + any pending review and returns to the Start step with
        the layer unlocked. The zone x badge re-draws the zone (same layer);
        Exit is the full way out, mirroring Interactive's Stop returning to the
        start. Only reachable when idle (the Detect/Exit row is hidden during a
        run or review).
        """
        try:
            from ...core import telemetry
            from_step = None
            try:
                from_step = int(self.dock_widget.auto_steps.currentIndex())
            except (RuntimeError, AttributeError):
                pass
            autosaved = len((self._auto_review or {}).get("geoms", []))
            telemetry.track_auto_exit_clicked(
                from_step=from_step if from_step is not None else -1,
                autosaved_count=autosaved,
            )
        except Exception:
            pass  # nosec B110
        # Disarm the zone drawing tool if it is still active.
        self._restore_maptool_after_zone()
        self._discard_auto_review()
        self._auto_zone = None
        self._auto_zone_polygon = None
        self._clear_auto_canvas()
        if self.dock_widget:
            try:
                self.dock_widget.reset_auto_to_start()
            except (RuntimeError, AttributeError):
                pass
