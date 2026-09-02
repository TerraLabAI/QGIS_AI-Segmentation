"""Post-run review: confidence re-filter, reslice, export, exit/retry.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProject,
)
from qgis.PyQt.QtWidgets import (
    QMessageBox,
)

from ...core.i18n import tr
from ...core.interaction_dials import reslice_screen_first_min_objects
from ...core.telemetry_errors import slot_guard
from ..error_report_dialog import show_error_report
from .shared import _debounce_timer

# Single-slot memo for the merge-policy token/category sets, keyed on the policy
# merge-dict id so a repeated _default_merge_separate call does not re-normalize
# the lists. The dict itself is held in the slot: without a reference the policy
# dict can be collected and a different dict can land on the same id, which would
# serve one policy's sets for another. One live entry, so no growth.
_MERGE_SETS_CACHE: dict[str, object] = {"id": None, "sets": None, "merge": None}

# Below this many objects a reslice finishes in a slice or two, so partitioning
# the work by what is on screen would cost more than it saves.
_RESLICE_SCREEN_FIRST_MIN_OBJECTS = 400


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
    _MERGE_SETS_CACHE["merge"] = merge
    return result


def _union_review_sets(geoms: list, scores: list | None, ids: object,
                       full_geoms: list, full_scores: list,
                       full_ids: list) -> tuple[list, list | None, int] | None:
    """The visible set plus every found object it does not already hold, keyed
    on det_id. Returns (geoms, scores, added_count), or None when either side
    lacks a usable id list.

    The rescue export used to compare the two SIZES. Two sets of the same size
    can hold different objects (a hand edit revealed one and the size gate hid
    another), and the size test then swapped the visible set out for a full set
    that was missing what the user had on screen. Identity keeps both.
    """
    vis_ids = list(ids) if isinstance(ids, (list, tuple)) else None
    if vis_ids is None or len(vis_ids) != len(geoms):
        return None
    if len(full_ids) != len(full_geoms):
        return None
    out_geoms = list(geoms)
    out_scores = list(scores) if scores is not None and len(scores) == len(geoms) else None
    seen = {det_id for det_id in vis_ids if det_id is not None}
    added = 0
    for i, det_id in enumerate(full_ids):
        if det_id is not None and det_id in seen:
            continue
        g = full_geoms[i]
        if g is None or g.isEmpty():
            continue
        out_geoms.append(g)
        if out_scores is not None:
            out_scores.append(full_scores[i] if i < len(full_scores) else None)
        if det_id is not None:
            seen.add(det_id)
        added += 1
    return out_geoms, out_scores, added


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
        from ...core.tile_manager import OVERLAP_FRACTION, TILE_SIZE

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
            from ...core.review_presets import live_catalog_categories
            for category in live_catalog_categories():
                for preset in category.get("presets") or []:
                    if str(preset.get("prompt", "")).strip().lower().replace("_", " ") != norm:
                        continue
                    if preset.get("weak"):
                        return False
                    cat = str(preset.get("category") or category.get("key") or "").lower()
                    return cat not in continuous_categories
        except Exception:  # noqa: BLE001 -- never block a run on a preset lookup  # nosec B110
            pass
        return True

    def _on_auto_review_refine_debounced(self) -> None:
        """The review's shape/size debounce settled: mark the review as
        actively refined (feeds the abandonment telemetry's refined-vs-left
        split), then run the normal cooperative reslice."""
        if self._auto_review is not None:
            self._review_tel_refined = True
        self._start_auto_reslice()

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
        self._review_push_editable_logged = False
        self._auto_finalize_state = {
            "mode": "reslice",
            "phase": "filter",
            # Enumerated so each visible geom carries its canonical det_id and
            # the Random colours stay stable across reslices.
            "filter_pending": self._reslice_pending_screen_last(),
            "total_filter": len(self._auto_objects),
            "visible": [],
            "visible_scores": [],
            "visible_ids": [],
            "visible_order": [],
            "params": self._widget_review_params(),
            "pixel_size": (self._auto_review or {}).get("pixel_size", 1.0),
            "gen": self._auto_finalize_gen,
        }
        # Say the pass is running, and offer the way out of it. Only on a set
        # big enough for the pass to be visible: on a small one the line would
        # appear and vanish inside a frame.
        self._set_review_busy(
            len(self._auto_objects)
            >= reslice_screen_first_min_objects(_RESLICE_SCREEN_FIRST_MIN_OBJECTS))
        self._step_auto_finalize_refine()

    def _set_review_busy(self, busy: bool) -> None:
        """Show or hide the review's shape-pass line, and wire its Stop once."""
        dock = self.dock_widget
        if dock is None:
            return
        try:
            if busy and not getattr(self, "_review_stop_wired", False):
                btn = getattr(dock, "auto_review_stop_btn", None)
                if btn is not None:
                    btn.clicked.connect(self._on_review_stop_clicked)
                    self._review_stop_wired = True
            dock.set_review_busy(bool(busy))
        except (RuntimeError, AttributeError):
            pass

    def _on_review_stop_clicked(self) -> None:
        """Stop the shape pass where it is, keeping what it has already drawn.

        The generation bump supersedes the cooperative pump, so a slice already
        queued finds itself stale and does nothing. The refine thread goes with
        it: every job still queued describes work nobody is waiting for now.
        The visible set stays whatever the last push wrote, which is exactly
        what the user can see on the map.
        """
        if not self._auto_review:
            return
        self._auto_finalize_gen += 1
        self._auto_finalize_state = None
        stop = getattr(self, "_stop_review_refine_thread", None)
        if stop is not None:
            stop()
        self._set_review_busy(False)

    def _reslice_pending_screen_last(self) -> list:
        """``_auto_objects`` enumerated, with the ones under the user's eyes at
        the END of the list.

        The pump takes its next object with ``.pop()``, so last in the list is
        first refined, and the progressive push writes what has been refined so
        far every 250 ms. Order therefore decides WHEN the shapes on screen
        change, not what they become: the whole set is refined either way, and
        the completion sorts the result back into canonical order.

        Falls back to plain enumeration whenever the canvas extent cannot be
        read or brought into the run CRS, and skips the partition on a small
        set, where the whole pass lands inside one or two slices anyway.
        """
        pending = list(enumerate(self._auto_objects))
        if len(pending) < reslice_screen_first_min_objects(_RESLICE_SCREEN_FIRST_MIN_OBJECTS):
            return pending
        boxes = self._reslice_object_boxes()
        try:
            canvas = self.iface.mapCanvas()
            extent = canvas.extent()
            canvas_crs = canvas.mapSettings().destinationCrs()
            run_crs = QgsCoordinateReferenceSystem(
                self._auto_crs_authid or "EPSG:4326")
            if not run_crs.isValid() or extent.isEmpty():
                return pending
            if canvas_crs.isValid() and canvas_crs != run_crs:
                extent = QgsCoordinateTransform(
                    canvas_crs, run_crs,
                    QgsProject.instance()).transformBoundingBox(extent)
            offscreen, onscreen = [], []
            for row in pending:
                box = boxes[row[0]]
                if box is not None and extent.intersects(box):
                    onscreen.append(row)
                else:
                    offscreen.append(row)
        except Exception:  # noqa: BLE001 -- ordering is an optimisation only
            return pending
        if not onscreen or not offscreen:
            return pending
        return offscreen + onscreen

    def _reslice_object_boxes(self) -> list:
        """Bounding boxes of ``_auto_objects``, one per index, memoised.

        ``boundingBox()`` walks the whole geometry, so a dense result paid a
        full scan of every object on each reslice just to decide the refine
        order. The memo is keyed on the identity and the length of the object
        list; a geometry swapped in place at a stable index therefore serves a
        stale box, which can only misplace an object in the ORDER the pump
        refines in, never change what it becomes."""
        objects = self._auto_objects or []
        key = (id(objects), len(objects))
        memo = getattr(self, "_reslice_box_memo", None)
        if memo is not None and memo[0] == key:
            return memo[1]
        boxes = [row[0].boundingBox() if row[0] is not None else None
                 for row in objects]
        self._reslice_box_memo = (key, boxes)
        return boxes

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
            # force: the review owns the canvas, so the grid is refused to
            # every other caller here. This one is the user asking for it.
            self._show_zone_tile_grid(layer, grid, force=True)

    def _on_auto_review_confidence_preview(self, percent: int) -> None:
        """Live preview WHILE the confidence slider is dragged: re-show the
        detections at the new cutoff. The geometries are pre-simplified ONCE in
        _auto_preview_geoms (gate-exempt rows first, then score desc), so a drag
        tick is just a prefix slice - no per-tick simplify, no merge. The
        accurate rebuild runs on release via _on_auto_review_confidence_changed.

        The Min/Max size gate and the hand-drawn exemption are applied here too,
        on the SAME params the release pass uses: without them the drag showed
        (and counted) shapes that vanished on release, and a split piece that
        inherited a low parent score blinked out mid-drag. The cache carries the
        exemption per row, so applying it costs no extra scan."""
        if not self._auto_review:
            return
        conf = max(0.0, min(1.0, percent / 100.0))
        # Shared borders is deliberately NOT applied here. It is a whole-set
        # pass over every neighbour, so running it on each drag tick would
        # stall the handle; the accurate pass on release snaps the set, and
        # its stamp differs from this preview's, so the layer is rewritten
        # then. Mid-drag the shapes show their own, unsnapped outlines.
        # Adopt the cutoff NOW, not on the accurate pass: the preview's push
        # refreshes the count header, whose pct reads _auto_confidence. With a
        # dense result the accurate reslice takes seconds, so a stale value
        # here left the header saying "below 90%" while the dial sat at 85.
        self._auto_confidence = conf
        preview = []
        pscores = []
        pids = []
        # Objects deleted during a Manual refine or a review correction stay
        # deleted whatever the cutoff (the caches predate those gestures, so
        # filter at consumption). Revealed objects only rejoin on the accurate
        # release pass (the drag cache is an ordered prefix slice).
        removed = self._review_removed_fids()
        # The size half of the visible-set gate, read from the same widgets the
        # release pass reads. Hand-drawn and split objects skip both gates.
        params = self._widget_review_params()
        if self._auto_preview_geoms:
            # Fast path: the cache is built (exempt rows first, then score desc),
            # so the cutoff is a prefix slice - no per-tick simplify, and the
            # scan stops at the first row below it. The stamp names the cache
            # build: within one build an object's preview geometry never changes,
            # so a drag tick pushes only the prefix DELTA (adds/deletes).
            stamp = ("prev", self._auto_preview_build_gen)
            for geom, score, area, det_idx, exempt in self._auto_preview_geoms:
                if not exempt:
                    if score < conf:
                        break  # ordered: everything after is below the cutoff
                    if not self._passes_size_filters(area, params):
                        continue
                if det_idx in removed:
                    continue
                preview.append(geom)
                pscores.append(score)
                pids.append(self._object_fid_for(det_idx))
        else:
            # Fallback while the background cache build is still running: filter
            # the canonical WHOLE objects directly (correct, just heavier). Whole
            # objects, never fragments, so a drag never shows half a building. The
            # shape refine still runs on release.
            stamp = ("base", 0)
            for det_idx, (g, s, area) in enumerate(self._auto_objects):
                if det_idx in removed or g is None:
                    continue
                if not self._object_is_manual(det_idx):
                    if s < conf or not self._passes_size_filters(area, params):
                        continue
                preview.append(g)
                pscores.append(s)
                pids.append(self._object_fid_for(det_idx))
        self._push_review_geoms(preview, repair=False, scores=pscores, ids=pids,
                                stamp=stamp)

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
        self._review_tel_conf_changed = True
        self._review_conf_moves = getattr(self, "_review_conf_moves", 0) + 1
        _debounce_timer(self, "_review_conf_timer", self.dock_widget, 2000,
                        self._emit_review_confidence_final)

    def _emit_review_confidence_final(self) -> None:
        if not self._auto_review:
            return
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_review_confidence_final(
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
        # repair=False: _auto_review["geoms"] are the reslice output, already
        # repaired + MultiPolygon-coerced once at cache-fill
        # (_review_refined_geom), so a second per-feature makeValid here is pure
        # waste. On a 26k-object run that redundant repair was the multi-minute
        # freeze at review open. Extents still refresh (zoom-to-layer needs them).
        self._push_review_geoms(
            self._auto_review["geoms"], repair=False, update_extents=True,
            scores=self._auto_review.get("scores"),
            ids=self._auto_review.get("ids"),
            stamp=self._auto_review.get("stamp"))

    def _push_review_geoms(self, geoms: list, repair: bool = True,
                           scores: list | None = None,
                           ids: list | None = None,
                           stamp: tuple | None = None,
                           partial: bool = False,
                           update_extents: bool | None = None) -> None:
        """Write geoms onto the live review selection layer and update the
        review count. Body, and the two push strategies behind it, in
        review_layer_push."""
        from .review_layer_push import push_review_geoms

        push_review_geoms(self, geoms, repair=repair, scores=scores, ids=ids,
                          stamp=stamp, partial=partial,
                          update_extents=update_extents)

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
        # This runs on the COMPLETE push only (a progressive mid-pass batch
        # returns before it), so it is the point the shape pass is finished
        # with and the busy line has nothing left to announce.
        self._set_review_busy(False)
        try:
            # Rows are overwritten and appended, never deleted, so the raw
            # length keeps counting what the user removed. The header is a
            # count of what is still there, hand-drawn objects included: they
            # are what lifts a review that found nothing out of its empty state.
            removed = self._review_removed_fids()
            # Counted off the removal set, not by walking every row: this runs
            # on every push, and the removals are a handful next to the objects.
            n_objects = len(self._auto_objects)
            total = n_objects - sum(
                1 for det_idx in removed if 0 <= det_idx < n_objects)
            pct = int(round((self._auto_confidence or 0.0) * 100))
            # When nothing is visible, tell the user which filter is actually
            # hiding the objects so they reach for the right lever: the Min size
            # filter can hide everything even when Confidence would show them.
            bound = ("confidence" if visible or total <= 0
                     else self._review_zero_binding_gate())
            self.dock_widget.update_auto_review_count(
                visible, total, pct, bound=bound)
        except (RuntimeError, AttributeError):
            pass

    def _review_zero_binding_gate(self) -> str:
        """With nothing visible, name the filter that hides the objects:
        ``"confidence"``, ``"min"`` or ``"max"``.

        An object scoring at or above the cutoff proves Confidence is not what
        hides it, so one of the two size gates is, and the header has to point
        at the one the user can actually pull. Saying "lower the Min size" when
        Max size is what binds sends them to a dial that changes nothing."""
        conf = self._auto_confidence or 0.0
        params = self._widget_review_params()
        min_a = float(params.get("min_a") or 0.0)
        max_a = float(params.get("max_a") or 0.0)
        removed = self._review_removed_fids()
        below_min = 0
        above_max = 0
        for det_idx, (base, score, area) in enumerate(self._auto_objects):
            if det_idx in removed or base is None or base.isEmpty():
                continue
            if score < conf:
                continue
            if min_a > 0 and area < min_a:
                below_min += 1
            elif max_a > 0 and area > max_a:
                above_max += 1
        if not below_min and not above_max:
            return "confidence"
        return "min" if below_min >= above_max else "max"

    def _drop_preview_geom_cache(self) -> None:
        """Free the confidence-slider preview cache and disarm any build still
        pumping into it.

        The cache holds one simplified copy of EVERY object the run found, the
        ones the filters hide included, so on a dense run it is the largest
        thing the review keeps alive. The export path left it resident: the
        user pressed Finish and the whole set stayed in memory while they
        panned the layer they had just saved.

        Bumping the build generation matters as much as emptying the list. The
        build is cooperative, so a slice already queued would otherwise publish
        a fresh full copy one event-loop turn after the clear.
        """
        self._auto_preview_geoms = []
        self._auto_preview_build_state = None
        self._auto_preview_build_gen = (
            getattr(self, "_auto_preview_build_gen", 0) + 1)

    def _current_visible_review_count(self) -> int:
        """Objects currently shown in the review (the last pushed visible set)."""
        review = self._auto_review or {}
        return len(review.get("geoms", []))

    def _full_found_review_count(self) -> int:
        """How many objects the safety-net export would save (Confidence gate
        dropped, Min/Max size kept), WITHOUT running the shape refine: the Exit
        dialog label must be instant, and refining the whole hidden cohort
        synchronously froze QGIS for seconds on a dense review. The rare
        refine-emptied geometry can make this off by a hair; the actual Save
        export recomputes exactly."""
        params = dict(self._widget_review_params())
        params["conf"] = 0.0
        removed = self._review_removed_fids()
        n = 0
        for det_idx, (base, score, area) in enumerate(self._auto_objects):
            if det_idx in removed or base is None or base.isEmpty():
                continue
            # A hand-drawn object skips the gates everywhere else, so counting
            # it under them here promised the user fewer objects than the save
            # actually wrote.
            if (self._object_is_manual(det_idx)
                    or self._passes_review_filters(score, area, params)):
                n += 1
        return n

    def _full_found_review_geoms(self) -> tuple[list, list, list]:
        """The review's found objects with the Confidence gate dropped but the
        current size + shape refine kept. The safety-net exit paths export this
        so a billed detection hidden ONLY by the Confidence cutoff is never
        lost. Hand edits are folded into the canonical rows and their det_ids
        skip the gates, so _compute_visible_objects already carries them; no
        separate protected set to merge back.

        Comes back with its parallel det_id list so the export can union the two
        sets by identity."""
        review = self._auto_review or {}
        pixel_size = review.get("pixel_size", 1.0) or 1.0
        params = dict(self._widget_review_params())
        params["conf"] = 0.0  # drop only the confidence gate; keep size + shape
        # Budgeted: the hidden cohort was never refined, so on a dense review
        # this owes thousands of GEOS passes, and its callers are the exit
        # paths, teardown and QGIS quit among them. See _RESCUE_REFINE_BUDGET_S.
        from .auto_review_geometry import rescue_refine_budget
        return self._compute_visible_objects(
            params, pixel_size, with_scores=True,
            refine_budget_s=rescue_refine_budget(), with_ids=True)

    def _export_auto_review(self, include_hidden: bool = False,
                            autosave: bool = False
                            ) -> tuple[str | None, int] | None:
        """Apply the refine settings to the pending review geometries, export
        them to a GeoPackage layer, clear the review state, and return
        (layer_name, polygon_count). Returns None when there is nothing to
        export. Shared by the interactive Export button and the headless MCP
        path so both commit the review identically.

        The review state is cleared only when the write produced a layer name.
        A write every target refused comes back as (None, count) with the
        review, the objects and the selection layer left alone, so the user can
        free the file and press Finish again.

        ``include_hidden`` is set ONLY on the safety-net exit paths (teardown
        autosave, the review Exit dialog's Save): a paid detection hidden by the
        Confidence cutoff must not be silently lost, so when the visible set is
        smaller than the full found set the FULL set is exported instead
        (confidence gate dropped, the user's size + shape refine kept). The
        normal Finish button leaves it False and exports exactly the visible set
        the user sees.

        ``autosave`` marks the passive leave-safety export (mode switch, new
        run, unload) in telemetry, so the funnel can tell an explicit Finish
        from a rescue save.
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
        # The confidence that actually filtered the exported set. The safety net
        # drops the gate ONLY when it really falls back to the full found set;
        # when the visible set is already the whole of it, the slider value still
        # applies. Recording the gate that ran is the difference between reading
        # this run later and guessing at it: meta records the slider position
        # either way, so an export saved past a 0.55 cutoff and an export saved
        # with the cutoff dropped look identical from the outside.
        conf_applied = float(getattr(self, "_auto_confidence", 0.0) or 0.0)
        if include_hidden:
            visible_n = sum(1 for g in geoms if g is not None and not g.isEmpty())
            full_geoms, full_scores, full_ids = self._full_found_review_geoms()
            merged = _union_review_sets(
                geoms, scores, review.get("ids"), full_geoms, full_scores, full_ids)
            if merged is not None:
                geoms, scores, extra = merged
                if extra:
                    conf_applied = 0.0
            elif len(full_geoms) > visible_n:
                # No usable identity on one side, so fall back to the size test.
                geoms, scores = full_geoms, full_scores
                conf_applied = 0.0
        if scores is not None and len(scores) != len(geoms):
            scores = None  # e.g. after a protected-geoms dissolve
        refined, refined_scores = [], []
        for index, g in enumerate(geoms):
            if g is None or g.isEmpty():
                continue
            refined.append(QgsGeometry(g))
            refined_scores.append(scores[index] if scores else None)
        # Freed BEFORE the write, not with the rest of the teardown below: it
        # holds a simplified copy of every object the run found, and the write
        # is where this path peaks (the staged features, the writer's own
        # buffers and the reprojection all land on top of it).
        self._drop_preview_geom_cache()
        name = self._export_auto_detections(
            refined, review["crs"], review["source_layer_name"], review["prompt"],
            scores=refined_scores, confidence_applied=conf_applied)
        if name:
            # The billed set reached a real layer, so this run's crash-net
            # copy is a duplicate of it: the pointer goes, and so does the
            # table it points at, which nothing pruned before.
            # One event-loop turn later: dropping the table is a SQLite delete
            # of a duplicate of everything just written, and it was sitting
            # between the user's click and their layer.
            run_id = self._auto_run_id or None
            exported_layer_id = self._auto_export_layer_id or ""

            def _drop_autosave_copy():
                self._pending_autosave_drop = None
                try:
                    from ...core.run_autosave import clear_pending
                    clear_pending(run_id, drop_table=True)
                except Exception:  # nosec B110
                    pass
                # The drop is the LAST write into the file the saved layer
                # reads, and its commit holds the file exactly when the
                # layer's first render may be reading it: that render comes
                # back with no rows, paints an empty map, and nothing asks
                # for another frame. Ask for one after the write, the same
                # rule persist_layer_to_file_later applies to its writes.
                try:
                    layer = QgsProject.instance().mapLayer(exported_layer_id)
                    if layer is not None:
                        layer.triggerRepaint()
                except (RuntimeError, AttributeError):  # nosec B110
                    pass

            # The timer belongs to the dock and dies with it, so an unload
            # between the click and the next turn would leave the duplicate
            # table on disk with nothing left to drop it. unload() runs
            # whatever is still parked here.
            self._pending_autosave_drop = _drop_autosave_copy

            try:
                from ...core.qt_compat import safe_single_shot

                safe_single_shot(0, self.dock_widget, _drop_autosave_copy)
            except Exception:  # noqa: BLE001  # nosec B110
                _drop_autosave_copy()
        else:
            # Every write target refused. Keep the review, the objects and the
            # selection layer exactly as they are, and report the failure with
            # the count: the teardown below would take a paid run off the screen
            # and leave the user no Finish to press, which is what the caller's
            # message asks them to do. The autosave pointer stays too, so the
            # run is still recoverable.
            # The cache freed above goes back too: the review stays open, and
            # without it every confidence move the user makes while they free
            # the file takes the slow path.
            try:
                self._start_build_preview_cache(
                    (self._auto_review or {}).get("pixel_size", 1.0) or 1.0)
            except (RuntimeError, AttributeError):
                pass
            return None, len(refined)
        # Capture the run's REAL outcome (chosen confidence, refine settings,
        # the kept geometry - even after a Refine-in-Manual detour) on a hidden
        # background task. Best-effort: queued only after the local export
        # succeeded, and can never block or fail it.
        try:
            from .run_export_upload import queue_run_export_upload
            queue_run_export_upload(
                self, review, refined, refined_scores,
                export_path=("autosave" if autosave
                             else "exit_save" if include_hidden else "finish"),
                confidence_applied=conf_applied)
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
            from ...core import telemetry_run_events, telemetry_session_events
            from .auto_client_profile import review_pass_profile
            found = len(self._auto_objects)
            pass_profile = dict(review_pass_profile(self))
            pass_profile["objects"] = len(refined)
            pass_profile.setdefault("on_pool", False)
            telemetry_run_events.track_auto_export_done(
                run_id=self._auto_run_id or "",
                exported_count=len(refined),
                visible_pct_of_found=int(round(len(refined) / found * 100)) if found else 0,
                final_confidence=int(round((self._auto_confidence or 0.0) * 100)),
                display_mode=self._auto_display_mode,
                refined_in_manual=getattr(self, "_auto_refined_in_manual", False),
                autosave=autosave,
                pass_profile=pass_profile,
            )
            if refined:
                telemetry_session_events.track_first_generation_milestone(mode="auto")
        except Exception:
            pass  # nosec B110
        self._auto_review = None
        self._auto_objects = []
        self._auto_object_fids = []
        self._drop_preview_geom_cache()
        self._reset_review_refine_cache()
        self._remove_auto_selection_layer()
        self._auto_manual_removed = set()
        self._auto_refined_in_manual = False
        self._clear_auto_raw_fragments()
        # An install started from this review is now orphaned (the review is
        # committed): drop the pending fix/add so a late predictor load does
        # not auto-open one on a gone or different review.
        self._release_local_ai_install()
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_review_active(False)
            except (RuntimeError, AttributeError):
                pass
        # The written count, not the offered one: a shape the writer could not
        # take is not a saved object, and "Saved 12" over 11 rows is the one
        # error a user cannot check.
        written = int(getattr(self, "_auto_export_feature_count", 0) or 0)
        return name, (written or len(refined))

    @slot_guard(stage="export", user_message=tr(
        "Something went wrong saving your detections. Please try again."))
    def _on_auto_export_clicked(self) -> None:
        """Finish: commit the reviewed detections to a layer, then return to the
        Start step (pick a layer, begin a new segmentation), layer unlocked."""
        # Snapshot the prompt BEFORE _export_auto_review nulls the review, so the
        # end-of-run value recap (shown on the Start page) can name the object.
        recap_prompt = ((self._auto_review or {}).get("prompt") or "").strip()
        # Export saves what is on screen. The Confidence cutoff is the Keep
        # step's decision, already made, so the finish line asks nothing.
        include_hidden = False
        # Say it is working before the work starts: the whole commit is
        # synchronous, and the map is deliberately held still for the redraw
        # that follows, so nothing else on screen can answer the click.
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import QApplication
        try:
            self.dock_widget.set_auto_export_saving(True)
        except (RuntimeError, AttributeError):
            pass
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            exported = self._export_auto_review(include_hidden=include_hidden)
        finally:
            QApplication.restoreOverrideCursor()
            try:
                self.dock_widget.set_auto_export_saving(False)
            except (RuntimeError, AttributeError):
                pass
        if exported is None:
            return
        name, count = exported
        # A total write failure comes back as (None, count), NOT as None: every
        # target in the fallback chain refused, and a tuple is never None, so
        # the old check waved it through and announced "saved N objects" with a
        # blank layer name for a run the user paid for and did not get. Report
        # the failure instead, and leave the run context alone so nothing
        # downstream reads it as a completed export.
        if not name:
            reason = getattr(self, "_auto_export_failure", "") or "file_refused"
            if reason == "nothing_visible":
                detail = tr("Nothing is visible to save. Lower Confidence, or "
                            "widen the size range, then try Finish again.")
            elif reason == "no_shapes":
                detail = tr("None of the objects came out as a shape the file "
                            "could take. Turn the cleanup settings down and "
                            "try Finish again.")
            else:
                detail = tr("The file may be open in QGIS or in another "
                            "program. Close it and try Finish again.")
            show_error_report(
                self.iface.mainWindow(),
                tr("Export Failed"),
                "{}\n\n{}".format(
                    tr("Could not save your detections to a file."), detail),
                error_code="export_failed",
            )
            return
        # Capture the layer id while the run context still exists (the reset
        # clears it). It is what makes the layer name a link on the success
        # line.
        try:
            recap_layer_id = getattr(self, "_auto_export_layer_id", "")
        except Exception:  # nosec B110 -- the success line is best-effort
            recap_layer_id = ""
        self._reset_auto_for_new_run()
        # ONE message right after Finish, saying how many objects were saved and
        # where. Entirely best-effort: the export already succeeded, so nothing
        # here may raise. Set AFTER the reset, which clears the Start page, so
        # the line survives the return to Start.
        #
        # A second, quieter card used to carry the same run plus its credit cost
        # for the rest of the session. Removed: it repeated the
        # legend and the footer ring on the page about the NEXT run.
        try:
            if self.dock_widget:
                self.dock_widget.set_auto_export_success(
                    count, name or "",
                    object_word=recap_prompt or None,
                    layer_id=recap_layer_id,
                )
        except Exception:  # nosec B110 -- never break Finish on the success line
            pass

    def _reset_auto_for_new_run(self) -> None:
        """After Finish: return to the Start step (pick a layer, begin a new
        segmentation) with the layer unlocked, rather than jumping straight back
        to drawing a zone. The committed detections stay on the map."""
        # Disarm the zone drawing tool if it is still active.
        self._restore_maptool_after_zone()
        # A committed run consumed its plan (and any attribute filters); the
        # next run re-fetches per prompt.
        self._auto_run_plan = None
        self._auto_attribute_filters = []
        self._cancel_task("_auto_run_plan_task")
        self._cancel_task("_auto_token_task")
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

    def _track_review_abandoned(self, exit_path: str) -> None:
        """One review_abandoned per review: fired by every leave-without-Finish
        path (the autosaving ones AND the discarding ones), with how the user
        left and whether they had engaged with the review controls first. The
        per-review flags are reset where review_opened fires. Best-effort."""
        if self._auto_review is None or not self._auto_objects:
            return
        if getattr(self, "_review_abandon_tracked", False):
            return
        self._review_abandon_tracked = True
        try:
            from ...core import telemetry_run_events
            from .auto_client_profile import review_pass_profile
            instances = len((self._auto_review or {}).get("geoms", []))
            pass_profile = dict(review_pass_profile(self))
            pass_profile["objects"] = instances
            pass_profile.setdefault("on_pool", False)
            telemetry_run_events.track_review_abandoned(
                run_id=self._auto_run_id or "",
                instances_at_exit=instances,
                refined=bool(getattr(self, "_review_tel_refined", False)),
                confidence_changed=bool(
                    getattr(self, "_review_tel_conf_changed", False)),
                exit_path=exit_path,
                pass_profile=pass_profile,
            )
        except Exception:
            pass  # nosec B110

    def _discard_review_without_autosave(self, exit_path: str = "other") -> None:
        """Drop the pending review OUTPUT without the autosave that
        _discard_auto_review performs. Shared by Adjust & run again and Exit so
        the discard cleanup lives in exactly one place. Clears the review, the
        canonical objects, the selection layer, the protected/handoff markers,
        and supersedes any in-flight cooperative finalize/reslice."""
        self._track_review_abandoned(exit_path)
        # A fix session open on this review dies with it, and dies WITHOUT
        # folding: its edits belong to the output being dropped. Runs while
        # _auto_review is still set, because the canvas sweep keys on it.
        self._abandon_fix_session_for_discard()
        # An explicit discard also drops this run's crash-net autosave pointer:
        # the user chose to throw the results away, so the next start must not
        # offer them back (the disk table itself is left in place).
        try:
            from ...core.run_autosave import clear_pending
            clear_pending(self._auto_run_id or None)
        except Exception:  # nosec B110
            pass
        self._auto_review = None
        self._auto_objects = []
        self._auto_object_fids = []
        self._reset_review_refine_cache()
        self._remove_auto_selection_layer()
        self._auto_manual_removed = set()
        self._auto_refined_in_manual = False
        self._auto_finalize_gen += 1
        self._auto_finalize_state = None
        self._drop_preview_geom_cache()
        self._clear_auto_raw_fragments()
        # Orphan any install started from this review (see above).
        self._release_local_ai_install()
        # Turn the review UI OFF here, in the one shared discard spot: the
        # Exit path used to skip it, leaving _auto_review_active stuck True
        # so the NEXT run's prompt step re-opened on the stale review panel.
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_review_active(False)
            except (RuntimeError, AttributeError):
                pass

    def _on_auto_review_exit_clicked(self) -> None:
        """Exit from the review: offer to Save (export) the detections, Discard
        them, or Cancel, so a billed result is NEVER silently dropped nor
        silently autosaved. On Save/Discard, leave to the Start step (unlocked).

        A fix session left open on the Correct step is folded first, the same
        way the step switch folds it. The Save below writes the review's own
        geometry, so without the fold the vertices the user had just moved were
        counted, offered and then dropped."""
        try:
            if getattr(self, "_refine_add_mode_active", False):
                self._exit_ai_add_mode()
            if (getattr(self, "_refine_handoff_active", False)
                    or getattr(self, "_qgis_bridge_active", False)):
                self._fold_active_correct_session()
            self._disarm_shape_tool()
        except (RuntimeError, AttributeError):  # nosec B110
            pass
        visible = self._current_visible_review_count()
        # Save exports the FULL found set (Confidence gate dropped, size + shape
        # kept) whenever it is larger than the visible set, so a billed
        # detection hidden by Confidence is never lost. The label states the
        # count that will be saved via the CHEAP filter-only count (the full
        # shape refine of the hidden cohort used to freeze this click). Counting
        # the rows instead would offer to save 0 objects on a review the user
        # emptied by hand, since a removed object keeps its row.
        save_count = max(self._full_found_review_count(), visible)
        if save_count > 0 and not self._auto_headless_run:
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
                saved = self._export_auto_review(include_hidden=True)
                if not saved or not saved[0]:
                    # Every write target refused. The export already told the
                    # user and left the review standing; resetting here would
                    # take the paid run off the screen right after promising to
                    # save it, with no Finish left to press.
                    return
                self._reset_auto_for_new_run()
                return
            if clicked is not drop_btn:
                return                              # Cancel: review intact
        self._discard_review_without_autosave(
            exit_path="exit_button")                # Discard: no autosave
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
                "object and settings. Running Detect again spends new cloud "
                "detections."
            ).format(total=total))
            discard_btn = box.addButton(
                tr("Discard && adjust"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(tr("Cancel"), QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(discard_btn)
            box.exec()
            confirmed = box.clickedButton() is discard_btn
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_auto_retry_clicked(
                run_id=self._auto_run_id or "",
                discarded_count=discarded,
                confirmed=confirmed,
            )
        except Exception:
            pass  # nosec B110
        if not confirmed:
            return False
        # Clear the review OUTPUT without the autosave _discard_auto_review does.
        self._discard_review_without_autosave(exit_path="new_run")
        if not self.dock_widget:
            return True
        try:
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
            self._restore_tile_grid_after_run()
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
            from ...core import telemetry_run_events
            from_step = None
            try:
                from_step = int(self.dock_widget.auto_steps.currentIndex())
            except (RuntimeError, AttributeError):
                pass
            autosaved = len((self._auto_review or {}).get("geoms", []))
            telemetry_run_events.track_auto_exit_clicked(
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
