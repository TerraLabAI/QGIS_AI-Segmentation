"""Live results: selection/handoff layers, renderers, tile pump, finalize.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

import math

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsVectorLayer,
)

from ...core.i18n import tr
from ...core.qt_compat import symbol_fill_color_property
from ..canvas_palette import KEPT_STROKE, OUTLINE_MODE_STROKE_STR
from ...core.review_defaults import (
    AUTO_REVIEW_CLEAN_DEFAULT as _AUTO_REVIEW_CLEAN_DEFAULT,
    AUTO_REVIEW_EXPAND_DEFAULT as _AUTO_REVIEW_EXPAND_DEFAULT,
    AUTO_REVIEW_FILL_HOLES_DEFAULT as _AUTO_REVIEW_FILL_HOLES_DEFAULT,
    AUTO_REVIEW_ORTHO_DEFAULT as _AUTO_REVIEW_ORTHO_DEFAULT,
    AUTO_REVIEW_SIMPLIFY_DEFAULT as _AUTO_REVIEW_SIMPLIFY_DEFAULT,
    AUTO_REVIEW_SMOOTH_DEFAULT as _AUTO_REVIEW_SMOOTH_DEFAULT,
)
from .shared import (
    _AUTO_LIVE_REFINE_BUDGET_S,
    _AUTO_LIVE_REPAINT_MS,
    _AUTO_PUMP_BUDGET_S,
    _FIELD_TYPE_DOUBLE,
    _FIELD_TYPE_INT,
    _FIELD_TYPE_STRING,
    _add_features_with_ids,
    _apply_fast_render,
)


# Upper bound on retained exemplar-only raw fragments (for the count-vs-map
# re-merge + the review override). Past it the list is freed and the override is
# marked unavailable; the map-likeness counters keep counting either way, so the
# automatic decision still runs. A run this dense is continuous cover in practice.
_RAW_FRAGMENT_RETAIN_CAP = 40000


def _diff_live_fid_map(old_map: dict, current: list):
    """Diff the previous live selection set against the current one.

    Pure bookkeeping (no QGIS types), so the live preview updates the provider
    incrementally instead of truncating + re-adding every feature each repaint.

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


class AutoResultsMixin:
    """Live results: selection/handoff layers, renderers, tile pump, finalize."""

    def _create_auto_selection_layer(self, source_layer) -> "QgsVectorLayer | None":
        """Create an in-memory vector layer to display detection results live."""
        try:
            crs_authid = source_layer.crs().authid()
            layer = QgsVectorLayer(
                "MultiPolygon?crs={}".format(crs_authid),
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

    # --- Refine-in-Manual handoff: seeds as memory layers, not N bands (§1.1) ---

    def _create_handoff_layer(self, crs_authid: str, kind: str):
        """Create ONE in-memory MultiPolygon layer (raster CRS) for the handoff
        seeds: kind='pending' (blue hairline, not yet validated) or kind='kept'
        (green fill, validated this session). Same fast-render pattern as the
        review layer (sub-pixel simplify + spatial index) so 1000s of seeds pan
        smoothly. Geometries are pushed in raster CRS directly (no per-object
        canvas transform, unlike the old rubber bands). Returns the layer or
        None."""
        try:
            layer = QgsVectorLayer(
                "MultiPolygon?crs={}".format(crs_authid),
                tr("Refine seeds"), "memory")
            if not layer.isValid():
                return None
            # Same identity fields as the review selection layer: det_id drives
            # the per-instance Random hue (one colour per object, stable across
            # the whole handoff), score rides along for the return trip.
            pr = layer.dataProvider()
            pr.addAttributes([
                QgsField("score", _FIELD_TYPE_DOUBLE),
                QgsField("det_id", _FIELD_TYPE_INT),
            ])
            layer.updateFields()
            self._apply_handoff_random_renderer(layer, kept=(kind == "kept"))
            _apply_fast_render(layer)
            # Private working layer, same rationale as the live selection
            # layer: flag before add, keep the tree node for canvas render.
            from ...core.output_store import mark_temp_layer
            mark_temp_layer(layer)
            QgsProject.instance().addMapLayer(layer, False)
            QgsProject.instance().layerTreeRoot().insertLayer(0, layer)
            return layer
        except (RuntimeError, AttributeError, ImportError):
            return None

    def _ensure_handoff_layers(self, crs_authid: str) -> None:
        """Create the pending + kept seed layers for the handoff if absent."""
        if self._handoff_pending_layer is None:
            self._handoff_pending_layer = self._create_handoff_layer(
                crs_authid, "pending")
        if self._handoff_kept_layer is None:
            self._handoff_kept_layer = self._create_handoff_layer(
                crs_authid, "kept")

    def _apply_handoff_random_renderer(self, layer, kept: bool) -> None:
        """Per-instance Random hue for a handoff seed layer (same expression as
        the review's Random mode, so an object KEEPS its colour through the
        handoff). Pending = dark hairline outline; kept = bold green outline
        (the validated ring) over the same instance hue. Best-effort."""
        try:
            from qgis.core import (QgsFillSymbol, QgsSingleSymbolRenderer,
                                   QgsProperty)
            stroke = KEPT_STROKE if kept else None
            symbol = QgsFillSymbol.createSimple({
                "color": "120,120,120,120",
                "outline_color": ("{},{},{},255".format(
                    stroke.red(), stroke.green(), stroke.blue())
                    if stroke is not None else "20,20,20,200"),
                "outline_width": "0.6" if kept else "0.2",
            })
            sl = symbol.symbolLayer(0)
            expr = "color_hsla((coalesce(\"det_id\", $id) * 67) % 360, 78, 55, 205)"
            prop_key = symbol_fill_color_property()
            sl.setDataDefinedProperty(prop_key, QgsProperty.fromExpression(expr))
            symbol.setOpacity(0.75)
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            layer.triggerRepaint()
        except (RuntimeError, AttributeError, ImportError):
            pass

    def _push_geoms_to_layer(self, layer, rows: list) -> None:
        """Replace a handoff seed layer's features with `rows` of
        (entry, geom, score, det_id) in raster CRS: truncate + bulk add + one
        repaint. Records each entry's assigned provider fid (_hfid/_hkept) so
        later single-object edits can update the layer incrementally instead of
        rebuilding it. Best-effort; never raises."""
        if layer is None:
            return
        try:
            from ...core.layer_conventions import to_multipolygon
            pr = layer.dataProvider()
            pr.truncate()
            feats = []
            kept_flag = layer is self._handoff_kept_layer
            for pg, g, score, det_id in rows:
                # Fresh rebuild: any prior bookkeeping is stale by definition.
                pg.pop("_hfid", None)
                pg.pop("_hkept", None)
                if g is None or g.isEmpty():
                    continue
                mg = to_multipolygon(g) or g
                if mg is None or mg.isEmpty():
                    continue
                feat = QgsFeature(layer.fields())
                feat.setGeometry(mg)
                feat.setAttributes([
                    float(score) if score is not None else 1.0,
                    int(det_id) if det_id is not None else None,
                ])
                feats.append((pg, feat))
            if feats:
                # The assigned provider fids (needed for the incremental
                # single-object updates) come back on the RETURNED copies;
                # addFeatures never mutates its inputs.
                ok, added = _add_features_with_ids(pr, [f for _pg, f in feats])
                if ok and len(added) == len(feats):
                    for (pg, _f), out in zip(feats, added):
                        pfid = out.id()
                        if pfid is not None and pfid >= 0:
                            pg["_hfid"] = pfid
                            pg["_hkept"] = kept_flag
            # No updateExtents(): rendering fetches by viewport via the provider
            # spatial index, and the handoff never zooms to these layers, so the
            # O(N) extent rescan per rebuild bought nothing.
            layer.triggerRepaint()
        except (RuntimeError, AttributeError):
            pass

    def _rebuild_handoff_layers(self) -> None:
        """Refresh both seed layers from saved_polygons: not-yet-validated
        entries go pending, validated ones go kept (green ring). Also rebuilds
        the hover/click spatial index and prunes the selection outline, so every
        structural change keeps canvas, hit-testing and selection in lockstep.
        The ACTIVE object is already popped out of saved_polygons, so it is
        naturally excluded (it shows as the active mask band). No-op outside the
        handoff.

        This is the BULK path (import, teardown) and the fallback when an
        incremental single-object update reports failure; routine per-object
        changes (open, close, save, delete, undo, absorb) go through
        _handoff_add_entry_feature / _handoff_remove_entry_feature instead,
        which was the fix for the double-click-to-edit lag on big handoffs."""
        if not self._refine_handoff_active:
            return
        pending, kept = [], []
        for pg in self.saved_polygons:
            row = (pg, self._entry_geom(pg), pg.get("score"), pg.get("det_id"))
            (kept if pg.get("validated") else pending).append(row)
        self._push_geoms_to_layer(self._handoff_pending_layer, pending)
        self._push_geoms_to_layer(self._handoff_kept_layer, kept)
        self._rebuild_handoff_hit_index()
        try:
            self._refresh_handoff_selection_band()
            self._set_handoff_hover(None)
        except (RuntimeError, AttributeError):
            pass

    def _rebuild_handoff_hit_index(self) -> None:
        """Bbox spatial index over saved_polygons so the hover highlight and
        click hit-test stay cheap over thousands of seeds. Keyed by a STABLE
        per-entry token (_htok, not the shifting list index) so single-object
        changes maintain it incrementally between full rebuilds; the token's
        bbox is kept on the entry because the QGIS < 3.36 deleteFeature API
        needs the exact inserted fid + bounds back."""
        try:
            from qgis.core import QgsFeature, QgsSpatialIndex
            index = QgsSpatialIndex()
            tok2entry = {}
            for pg in self.saved_polygons:
                g = self._entry_geom(pg)
                if g is None or g.isEmpty():
                    continue
                tok = pg.get("_htok")
                if tok is None:
                    tok = self._next_handoff_hit_token()
                    pg["_htok"] = tok
                bbox = g.boundingBox()
                pg["_hbbox"] = bbox
                feat = QgsFeature(tok)
                feat.setGeometry(QgsGeometry.fromRect(bbox))
                index.addFeature(feat)
                tok2entry[tok] = pg
            self._handoff_hit_index = index
            self._handoff_tok2entry = tok2entry
        except (RuntimeError, AttributeError):
            self._handoff_hit_index = None
            self._handoff_tok2entry = {}

    def _next_handoff_hit_token(self) -> int:
        """Monotonic stable token for the hit index (never reused in-session)."""
        tok = getattr(self, "_handoff_hit_tok_seq", 0) + 1
        self._handoff_hit_tok_seq = tok
        return tok

    def _handoff_hit_insert(self, pg) -> None:
        """Add one entry to the hover/click spatial index (no-op without one)."""
        index = getattr(self, "_handoff_hit_index", None)
        if index is None:
            return
        try:
            from qgis.core import QgsFeature
            g = self._entry_geom(pg)
            if g is None or g.isEmpty():
                return
            tok = pg.get("_htok")
            if tok is None:
                tok = self._next_handoff_hit_token()
                pg["_htok"] = tok
            bbox = g.boundingBox()
            pg["_hbbox"] = bbox
            feat = QgsFeature(tok)
            feat.setGeometry(QgsGeometry.fromRect(bbox))
            index.addFeature(feat)
            self._handoff_tok2entry[tok] = pg
        except (RuntimeError, AttributeError):
            pass

    def _handoff_hit_remove(self, pg) -> None:
        """Drop one entry from the hover/click spatial index (no-op without
        one, or when the entry was never indexed). The token/bbox keys are
        POPPED off the entry so any dict(pg) snapshot taken afterwards (undo
        backup, close-to-pending copy) mints a fresh token on re-insert; a
        reused token could otherwise end up indexed for two entries at once."""
        index = getattr(self, "_handoff_hit_index", None)
        tok = pg.pop("_htok", None)
        bbox = pg.pop("_hbbox", None)
        if index is None or tok is None or bbox is None:
            return
        try:
            from qgis.core import QgsFeature
            feat = QgsFeature(tok)
            feat.setGeometry(QgsGeometry.fromRect(bbox))
            index.deleteFeature(feat)
            self._handoff_tok2entry.pop(tok, None)
        except (RuntimeError, AttributeError):
            pass

    def _handoff_add_entry_feature(self, pg) -> bool:
        """Incremental twin of _rebuild_handoff_layers for ONE appended entry:
        add its feature to the right seed layer (+ hit index) and record its
        provider fid. Returns False when the incremental path is unavailable,
        so the caller falls back to the full rebuild. True no-op outside the
        handoff (base Manual draws rubber bands instead)."""
        if not self._refine_handoff_active:
            return True
        # A restored/copied entry can carry provider bookkeeping from a
        # previous life: strip it so a later remove never targets a dead fid.
        pg.pop("_hfid", None)
        pg.pop("_hkept", None)
        kept = bool(pg.get("validated"))
        layer = self._handoff_kept_layer if kept else self._handoff_pending_layer
        if layer is None:
            return False
        try:
            if not layer.isValid():
                return False
            from ...core.layer_conventions import to_multipolygon
            g = self._entry_geom(pg)
            if g is None or g.isEmpty():
                return True  # nothing to draw for this entry
            mg = to_multipolygon(g) or g
            if mg is None or mg.isEmpty():
                return True
            feat = QgsFeature(layer.fields())
            feat.setGeometry(mg)
            score = pg.get("score")
            det_id = pg.get("det_id")
            feat.setAttributes([
                float(score) if score is not None else 1.0,
                int(det_id) if det_id is not None else None,
            ])
            ok, added = _add_features_with_ids(layer.dataProvider(), [feat])
            pfid = added[0].id() if ok and added else None
            if pfid is None or pfid < 0:
                return False
            pg["_hfid"] = pfid
            pg["_hkept"] = kept
            self._handoff_hit_insert(pg)
            layer.triggerRepaint()
            return True
        except (RuntimeError, AttributeError):
            return False

    def _handoff_remove_entry_feature(self, pg) -> bool:
        """Incremental twin of _rebuild_handoff_layers for ONE removed entry:
        delete its feature from its seed layer (+ hit index). Returns False
        when the entry has no recorded fid (caller falls back to the full
        rebuild). True no-op outside the handoff."""
        if not self._refine_handoff_active:
            return True
        self._handoff_hit_remove(pg)
        pfid = pg.pop("_hfid", None)
        kept = pg.pop("_hkept", None)
        g = self._entry_geom(pg)
        if g is None or g.isEmpty():
            return True  # was never drawn on a seed layer
        if pfid is None or kept is None:
            return False
        layer = self._handoff_kept_layer if kept else self._handoff_pending_layer
        if layer is None:
            return False
        try:
            if not layer.isValid():
                return False
            layer.dataProvider().deleteFeatures([pfid])
            layer.triggerRepaint()
            return True
        except (RuntimeError, AttributeError):
            return False

    def _remove_handoff_layers(self) -> None:
        """Remove both handoff seed layers from the project (teardown)."""
        for attr in ("_handoff_pending_layer", "_handoff_kept_layer"):
            layer = getattr(self, attr, None)
            if layer is not None:
                try:
                    QgsProject.instance().removeMapLayer(layer.id())
                except (RuntimeError, AttributeError):
                    pass
                setattr(self, attr, None)

    def _apply_review_heatmap_renderer(self, layer) -> None:
        """Color the review selection layer as a confidence heatmap: high score
        (confident) = yellow, low score (uncertain) = purple, so dragging the
        confidence slider visibly shows what gets cut. Viridis is used instead of
        a red->green ramp because red-green is the most common colorblind
        confusion; Viridis is perceptually uniform and colorblind-safe (the legend
        states the yellow/purple meaning). One translucent fill symbol with a
        data-defined ramp over the per-object 'score' field (stable 0..1 domain).
        Best-effort: a render failure must never break the review."""
        try:
            from qgis.core import (QgsFillSymbol, QgsSingleSymbolRenderer,
                                   QgsProperty, QgsStyle)
            symbol = QgsFillSymbol.createSimple({
                "color": "0,150,80,110",
                "outline_color": "40,40,40,180",
                "outline_width": "0.2",
            })
            sl = symbol.symbolLayer(0)
            # Viridis ships with QGIS, but guard so an unusual install without it
            # still gets a colorblind-safe (non red-green) ramp.
            ramp_name = ("Viridis"
                         if QgsStyle.defaultStyle().colorRamp("Viridis")
                         else "Spectral")
            expr = "ramp_color('{}', coalesce(\"score\", 0))".format(ramp_name)
            # FillColor property key spelling moved across QGIS 3.x -> 4; the
            # shim resolves whichever this build exposes.
            prop_key = symbol_fill_color_property()
            sl.setDataDefinedProperty(prop_key, QgsProperty.fromExpression(expr))
            symbol.setOpacity(0.55)
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            layer.triggerRepaint()
        except (RuntimeError, AttributeError, ImportError):
            pass

    def _apply_review_random_renderer(self, layer) -> None:
        """Colour each detection a distinct pseudo-random colour (stable per
        feature id) so touching or merged objects are easy to tell apart: a visual
        debug aid. One translucent fill with a data-defined hue from the feature
        id. ONE identical style from the first live tile through the review and
        up to export: any style switch at review-open reads as the results
        "changing" once the run ends. Best-effort: a render failure must never
        break the review."""
        try:
            from qgis.core import (QgsFillSymbol, QgsSingleSymbolRenderer,
                                   QgsProperty)
            symbol = QgsFillSymbol.createSimple({
                "color": "120,120,120,120",
                "outline_color": "20,20,20,200",
                "outline_width": "0.2",
            })
            sl = symbol.symbolLayer(0)
            # Stable pseudo-random hue per DETECTION (det_id = the merger's keeper
            # id, constant from the moment an object first appears through the
            # whole live run and review), so new tiles or changed confidence/shape
            # settings never reshuffle colours; * 67 spreads adjacent ids. $id
            # fallback covers rows without det_id (hand-edited
            # dissolve path). Saturation 78 + lightness 55 + alpha 205, combined
            # with the symbol opacity below, lands ~0.60 effective: richer and
            # clearly readable over dark imagery (rooftops) instead of washed out,
            # while still translucent enough to see the imagery underneath.
            expr = "color_hsla((coalesce(\"det_id\", $id) * 67) % 360, 78, 55, 205)"
            prop_key = symbol_fill_color_property()
            sl.setDataDefinedProperty(prop_key, QgsProperty.fromExpression(expr))
            symbol.setOpacity(0.75)
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            layer.triggerRepaint()
        except (RuntimeError, AttributeError, ImportError):
            pass

    def _apply_review_outline_renderer(self, layer) -> None:
        """Red outline only, NO fill, so the imagery INSIDE each detection stays
        visible: the filled modes (Normal blue, Confidence heatmap, Random) tint
        the interior and hide what is underneath. Red matches the saved-polygon
        convention. Best-effort: a render failure must never break the review."""
        try:
            from qgis.core import QgsFillSymbol, QgsSingleSymbolRenderer
            symbol = QgsFillSymbol.createSimple({
                "style": "no",                       # no fill brush = see through
                "outline_color": OUTLINE_MODE_STROKE_STR,  # red outline
                "outline_width": "0.5",
                "outline_style": "solid",
            })
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            layer.triggerRepaint()
        except (RuntimeError, AttributeError, ImportError):
            pass

    def _apply_review_display_mode(self, layer) -> None:
        """Apply the current review DISPLAY colour mode to the selection layer:
        'normal' (blue review fill), 'outline' (red outline only, see-through),
        'confidence' (green->red heatmap on the per-object score) or 'random' (one
        colour per object). The SAME renderer serves the live run and the review
        (one visual identity end to end). Visual only: never touches geometry,
        filters or export."""
        if layer is None:
            return
        mode = getattr(self, "_auto_display_mode", "normal")
        if mode == "random":
            self._apply_review_random_renderer(layer)
        elif mode == "outline":
            self._apply_review_outline_renderer(layer)
        elif mode == "confidence":
            self._apply_review_heatmap_renderer(layer)
        else:
            try:
                from ...core.layer_conventions import make_review_renderer
                layer.setRenderer(make_review_renderer())
                layer.triggerRepaint()
            except (RuntimeError, AttributeError, ImportError):
                pass

    def _display_legend_text(self, mode: str) -> str:
        """Legend line for a display mode: swatch dots in the real renderer
        colours + what they mean (single source: the dock-side helper, so the
        build-time seed and every mode switch render the same line)."""
        from ..dock.auto_review_build import display_legend_html
        return display_legend_html(mode)

    def _seed_review_display_mode(self) -> None:
        """Default the display colour to Random (one distinct colour per object)
        so touching instances read apart at a glance. Called at RUN START (so the
        live results stream in Random, not blue) and again when review opens.
        Syncs the dock combo + legend signal-free so control and renderer never
        desync; the user's later switches then work normally. The committed
        export keeps the red-outline convention (make_committed_renderer)."""
        self._auto_display_mode = "random"
        if self.dock_widget is not None:
            try:
                self.dock_widget.set_auto_display_mode("random")
                self.dock_widget.set_display_legend(
                    self._display_legend_text("random"))
            except (RuntimeError, AttributeError):
                pass

    def _on_auto_display_mode_changed(self, mode: str) -> None:
        """Review display colour mode changed (Normal / Outline / Confidence /
        Random): store it and re-render the selection layer. Visual only, no
        re-detection."""
        self._auto_display_mode = (
            mode if mode in ("normal", "outline", "confidence", "random")
            else "normal")
        if self._auto_selection_layer is not None:
            self._apply_review_display_mode(self._auto_selection_layer)
        # Muted legend line under the combo: what the colours MEAN for this mode.
        if self.dock_widget is not None:
            try:
                self.dock_widget.set_display_legend(
                    self._display_legend_text(self._auto_display_mode))
            except (RuntimeError, AttributeError):
                pass
        try:
            from ...core import telemetry
            telemetry.track_review_display_mode(mode=self._auto_display_mode)
        except Exception:
            pass  # nosec B110

    def _remove_auto_selection_layer(self) -> None:
        """Remove the live selection layer from the project (if present)."""
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

    def _refresh_auto_live_from_merger(self) -> None:
        """Redraw the live layer from the running merged set (stitched objects).

        The merger holds one geometry per stitched object, so the preview shows
        whole objects as tiles complete instead of cut tile fragments. We repaint
        from already-built geometries (no re-polygonization), and the set is the
        number of distinct objects (far smaller than the fragment count), so this
        stays cheap per tile. The merged set is confidence-agnostic, so filter by
        the current cutoff here (on whole objects, never fragments) so the live
        preview matches the review. layer.triggerRepaint() schedules the canvas
        update; no heavy mapCanvas().refresh() per tile.

        The run's FULL smart preset (fill holes, right angles, min size, clean
        edges...) is applied live, so what streams in already looks like the
        reviewed result: no hole-riddled masks or tiny noise specks that later
        vanish. Kept cheap by two facts: a merger keeper is IMMUTABLE (a merge
        retires old fids and inserts a new one), so the refined copy is cached
        per fid and each object is refined exactly once; and each repaint
        refines new arrivals for at most _AUTO_LIVE_REFINE_BUDGET_S, falling
        back to the plain simplify for the overflow until the next repaint.
        """
        layer = self._auto_selection_layer
        if layer is None or self._auto_merger is None:
            return
        try:
            if not layer.isValid():
                return
            import time as _t
            from ...core.layer_conventions import to_multipolygon
            conf = self._auto_confidence
            # Fallback tolerance for over-budget objects: the review's default
            # Simplify-outline in px converted to ground units by the run's
            # detection pixel size (the live worker reports the returned-mask
            # grid); computed once per repaint (not per object).
            px = self._auto_refine_pixel_size()
            live_tol = _AUTO_REVIEW_SIMPLIFY_DEFAULT * px if px > 0 else 0.0
            # The cached refines are px-scaled: when the observed mask grid
            # changes the scale (0 -> real after the first tiles), redo them.
            if self._auto_live_refine_px != px:
                self._auto_live_refine_cache.clear()
                self._auto_live_refine_px = px
                self._auto_live_params = None
            if self._auto_live_measurer is None:
                self._auto_live_measurer = self._make_auto_area_measurer()
            # Prompt, grid and confidence are all fixed for the duration of a
            # run, so resolve the review preset once per run (it regex-scans
            # the server policy's keyword tables) instead of on every repaint.
            params = self._auto_live_params
            if params is None:
                params = self._fresh_review_params()
                self._auto_live_params = params
            # Fill holes is a native, O(rings) drop of interior rings, so it is
            # decoupled from the expensive shape refine: when the preset fills
            # holes, apply it to EVERY over-budget object below (not just the ones
            # the per-repaint refine budget reached), so the live preview never
            # streams in hole-riddled polygons the review would fill.
            fill_holes_live = bool(params.get("fill_holes"))
            cache = self._auto_live_refine_cache
            fresh_cache = {}
            refine_deadline = _t.monotonic() + _AUTO_LIVE_REFINE_BUDGET_S
            pr = layer.dataProvider()
            fields = layer.fields()
            score_idx = fields.indexOf("score")
            # Build the CURRENT renderable set, then diff it against the previous
            # one and push only the delta to the provider (add new objects, change
            # grown/upgraded geometries and bumped scores, delete retired ones). A
            # per-repaint truncate + re-add of every feature was the dominant GUI
            # cost on dense runs (thousands of objects, O(N) every repaint); the
            # diff keeps a quiet repaint (nothing merged) nearly free.
            current: list = []          # ordered (fid, stamp, is_full, score)
            geom_by_fid: dict = {}      # fid -> geometry to render
            # Track by the merger's STABLE keeper id (det_id), not the loop
            # position, so an object KEEPS its hue as more tiles stream in (the
            # incremental writes carry det_id = fid unchanged, and the renderer
            # hues on det_id).
            for fid, geom, score in self._auto_merger.result_scored_ided():
                if geom is None or geom.isEmpty():
                    continue
                # Cheap stamp of the KEEPER geometry. The merger now keeps an
                # object's fid stable across the merges that GROW it, so a cache
                # hit on fid alone would serve the geometry from before the merge
                # (a stale render). Invalidate on any geometry change (bbox to 3
                # decimals + vertex count, both O(1)-ish on the keeper in hand)
                # and treat a mismatch exactly like a cache miss (recompute the
                # visibility gate + refine from the current geometry).
                stamp = (geom.boundingBox().toString(3), geom.constGet().nCoordinates())
                entry = cache.get(fid)
                if entry is not None and entry[2] != stamp:
                    entry = None
                if entry is None and _t.monotonic() < refine_deadline:
                    # New (or just-grown) object: apply the run preset ONCE (size
                    # gate on the same geodesic area the review uses, then the
                    # shape refine on a copy; the keeper is never mutated).
                    area = self._object_area_m2(geom, self._auto_live_measurer)
                    visible = self._passes_review_filters(score, area, params)
                    refined = None
                    if visible:
                        refined = self._refine_geom_for_review(geom, params, px)
                        if refined is None or refined.isEmpty():
                            refined = geom
                        # Coerce ONCE at cache-fill time: cache hits then skip
                        # the per-repaint to_multipolygon deep copy below.
                        refined = to_multipolygon(refined)
                        if refined is None or refined.isEmpty():
                            visible, refined = False, None
                    entry = (visible, refined, stamp)
                # is_full flags a full-refine render (from the cache) vs a cheap
                # budget-fallback render: the fallback is is_full=False, so the
                # next repaint's full refine of the SAME keeper (same stamp) still
                # registers as a geometry change below and upgrades the on-layer
                # polygon in place (the budget-fallback convergence).
                if entry is not None:
                    fresh_cache[fid] = entry
                    if not entry[0]:
                        continue  # preset min/max size or confidence gate
                    gg = entry[1]  # already MultiPolygon (coerced at fill)
                    is_full = True
                elif score < conf:
                    continue
                else:
                    # Refine budget spent this repaint: do the CHEAP ops now (drop
                    # interior rings if the preset fills holes, plus a plain
                    # simplify), all on a COPY. The costly ops (smooth / ortho /
                    # expand / clean-edges) land on the next repaint's full refine.
                    if fill_holes_live:
                        try:
                            r = geom.removeInteriorRings(-1.0)
                            if r is not None and not r.isEmpty():
                                geom = r
                        except (AttributeError, TypeError, ValueError):
                            pass
                    if live_tol > 0:
                        sg = geom.simplify(live_tol)
                        if sg is not None and not sg.isEmpty():
                            geom = sg
                    # Last-line guard: a MultiPolygon provider rejects a
                    # GeometryCollection, so coerce to polygon-only MultiPolygon and
                    # skip anything with no areal content (never raise mid-repaint).
                    # No defensive copy: to_multipolygon never mutates its input (the
                    # Polygon branch copies internally; the MultiPolygon branch
                    # returns it as-is) and setGeometry copies into the feature, so
                    # the keeper stays intact.
                    gg = to_multipolygon(geom)
                    is_full = False
                if gg is None or gg.isEmpty():
                    continue
                current.append((fid, stamp, is_full, float(score)))
                geom_by_fid[fid] = gg
            # Swap in the visited-fid cache: retired keepers (absorbed by a
            # merge) drop out here, so the cache never outgrows the live set.
            self._auto_live_refine_cache = fresh_cache
            # Diff the current renderable set against the previous one and push
            # only the delta to the provider.
            old_map = self._auto_live_fid_map
            adds, geom_changes, attr_changes, deletes, new_map = _diff_live_fid_map(
                old_map, current)
            by_fid = {f: (s, full, sc) for f, s, full, sc in current}
            changed = False
            if deletes:
                pr.deleteFeatures(deletes)
                changed = True
            if adds:
                add_feats = []
                for f in adds:
                    feat = QgsFeature(fields)
                    feat.setGeometry(geom_by_fid[f])
                    feat.setAttributes(["", by_fid[f][2], f])
                    add_feats.append((f, feat))
                # The assigned provider fids come back on the RETURNED feature
                # copies only (addFeatures never mutates its inputs); without
                # them every later repaint re-adds the same objects and the
                # stacked translucent fills read as opaque.
                ok, added = _add_features_with_ids(
                    pr, [feat for _, feat in add_feats])
                if ok and len(added) == len(add_feats):
                    for (f, _feat), out in zip(add_feats, added):
                        pfid = out.id()
                        if pfid is not None and pfid >= 0:
                            s, full, sc = by_fid[f]
                            new_map[f] = (pfid, s, full, sc)
                changed = True
            if geom_changes:
                pr.changeGeometryValues(
                    {prov_fid: geom_by_fid[f] for prov_fid, f in geom_changes.items()})
                changed = True
            if attr_changes and score_idx >= 0:
                pr.changeAttributeValues(
                    {prov_fid: {score_idx: by_fid[f][2]}
                     for prov_fid, f in attr_changes.items()})
                changed = True
            self._auto_live_fid_map = new_map
            # Live "found so far" count in the run label (same cadence as the
            # repaint, so no extra throttle): a slow zone never reads as dead. The
            # tracked fid map IS the current on-layer visible set.
            if self.dock_widget is not None:
                try:
                    self.dock_widget.set_auto_run_found_count(
                        (self._auto_run_ctx or {}).get("prompt") or tr("Example match"),
                        len(new_map))
                except (RuntimeError, AttributeError):
                    pass
            # No updateExtents() here: it rescans every feature (O(N)). Rendering
            # requests features by the canvas viewport via the spatial index, NOT
            # the layer's cached extent, so the live preview draws correctly
            # without it. The extent is refreshed once at review
            # (_push_review_geoms) for zoom-to-layer. Repaint only when the
            # provider actually changed (already throttled by the caller).
            if changed:
                layer.triggerRepaint()
        except (RuntimeError, AttributeError):
            # Live-preview repaint is best-effort (the layer can be deleted from
            # under us mid-run). Narrow catch: do not mask geometry/merger bugs.
            pass

    # ---- Auto detection signal handlers -------------------------------------

    def _on_auto_tile_completed(self, tile_idx: int, tagged_detections: list) -> None:
        """Slot: one tile finished. tagged_detections = [(geom_wkb: bytes, score)].

        Cheap on purpose: just queue the ready geometries and kick the pump. The
        heavy mask->geometry conversion now runs on the worker thread (it used to
        run here, which froze QGIS). The pump then only rehydrates WKB + folds
        each geom into the merger, a short time-slice at a time, yielding to the
        event loop between slices so the UI stays responsive during the run.
        """
        if self._auto_merger is None:
            return
        if tagged_detections:
            self._auto_tile_queue.append(tagged_detections)
        self._schedule_auto_pump()

    def _schedule_auto_pump(self) -> None:
        """Schedule one cooperative pump turn on the next event-loop iteration."""
        if self._auto_pump_scheduled or self._auto_merger is None:
            return
        self._auto_pump_scheduled = True
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(0, self._pump_auto_tiles)

    def _pump_auto_tiles(self) -> None:
        """Process queued tile detections for one short time slice, then yield.

        Reschedules itself while work remains, so a long run is converted in
        many small main-thread turns instead of a few long blocking ones. A
        (coalesced) live-preview repaint is requested whenever a slice made
        progress.
        """
        self._auto_pump_scheduled = False
        if self._auto_merger is None:
            self._auto_tile_queue.clear()
            return
        import time as _t
        progressed = self._process_auto_queue(_t.monotonic() + _AUTO_PUMP_BUDGET_S)
        if progressed:
            self._request_auto_live_repaint()
        if self._auto_tile_queue:
            self._schedule_auto_pump()

    def _drain_auto_tiles_now(self) -> None:
        """Process every queued tile synchronously because the run is ending.

        Called before _finalize_auto_results reads the merger so the result and
        the scored-geom store are complete (the pump is asynchronous, so the
        queue can still hold tiles when all_tiles_finished/cancelled fires).
        Bounded by whatever is still queued.
        """
        if self._auto_merger is None:
            self._auto_tile_queue.clear()
            return
        self._process_auto_queue(None)

    def _process_auto_queue(self, deadline) -> bool:
        """Fold queued tile detections into the running merger.

        ``deadline`` is a time.monotonic() instant to stop at (cooperative
        slice), or None to drain everything (run finishing). Returns True if any
        detection was processed. Pops as it goes, and yields mid-tile so one
        dense tile cannot blow the slice budget.

        The heavy mask -> refine -> polygonize -> clip -> repair pipeline now
        runs on the worker thread (it used to run here and froze the GUI at the
        end of a dense run). The worker emits READY geometry as WKB bytes; this
        loop only rehydrates each WKB into a QgsGeometry, stores it scored for
        the run summary, and folds EVERY fragment (confidence-agnostic) into the
        merger carrying its score. Merging confidence-agnostic is what fixes the
        "confidence cuts buildings" bug: a seam half that scored low still unions
        with its strong other half, and the merged object keeps the MAX score, so
        the review confidence slider drops weak WHOLE objects, never half a
        building. The live preview then filters the merged set by the current
        cutoff. That is cheap, so even the synchronous end-of-run drain no longer
        freezes.
        """
        from qgis.core import QgsGeometry
        import time as _t

        progressed = False
        queue = self._auto_tile_queue

        # Exemplar-only runs additionally retain each raw fragment + accumulate
        # the area-weighted mean tile coverage (failure blobs above the hard cap
        # excluded), so finalize can decide count-vs-map from the run's own masks
        # (and the review can re-group with no re-detection).
        retain = getattr(self, "_auto_is_exemplar_only", False)
        tile_area = getattr(self, "_auto_tile_ground_area", 0.0) if retain else 0.0
        hard_cov = 0.0
        if retain:
            from ...core.detection_policy import hard_tile_coverage
            from ...workers.auto_detection_worker import _HARD_TILE_COVERAGE as _HTC
            hard_cov = hard_tile_coverage(_HTC)

        while queue and self._auto_merger is not None:
            detections = queue[0]
            while detections:
                wkb, score = detections.pop()
                geom = QgsGeometry()
                geom.fromWkb(wkb)
                if geom.isEmpty():
                    continue
                # Keep EVERY detection (scored) for the run summary, then fold
                # EVERY fragment (confidence-agnostic) into the merger with its
                # score so seam-split halves always stitch and the merged object
                # keeps the MAX score. The live preview / review filter by whole-
                # object score afterward.
                self._auto_raw_count += 1
                if retain:
                    self._auto_raw_n_total += 1
                    if tile_area > 0:
                        cov = geom.area() / tile_area
                        # Failure blobs (near-whole-tile texture fills) are
                        # excluded so a handful cannot fake continuous cover.
                        if 0.0 < cov <= hard_cov:
                            self._auto_raw_cov_sum += cov
                            self._auto_raw_cov_sq_sum += cov * cov
                    frags = self._auto_raw_fragments
                    if frags is not None:
                        if len(frags) >= _RAW_FRAGMENT_RETAIN_CAP:
                            # Overflow: free the list, mark the override
                            # unavailable; the counters above still decide.
                            self._auto_raw_fragments = None
                        else:
                            frags.append((wkb, float(score)))
                self._auto_merger.add(geom, float(score))
                progressed = True
                if deadline is not None and _t.monotonic() >= deadline:
                    break
            if not detections:
                queue.popleft()
            if deadline is not None and _t.monotonic() >= deadline:
                break
        return progressed

    def _request_auto_live_repaint(self) -> None:
        """Coalesce live-preview repaints: rebuilding the whole selection layer
        on every completed tile was a second per-tile GUI cost. Fire at most
        once per _AUTO_LIVE_REPAINT_MS via a single-shot timer. Each repaint
        truncates + re-adds the whole live set (O(N)), so the cadence stretches
        as the set grows: on dense runs the extra latency is invisible (objects
        already stream in continuously) but the saved GUI churn is not."""
        from qgis.PyQt.QtCore import QTimer
        if self._auto_repaint_timer is None:
            self._auto_repaint_timer = QTimer(self.iface.mainWindow())
            self._auto_repaint_timer.setSingleShot(True)
            self._auto_repaint_timer.timeout.connect(self._refresh_auto_live_from_merger)
        if not self._auto_repaint_timer.isActive():
            n = len(self._auto_live_refine_cache)
            interval = _AUTO_LIVE_REPAINT_MS
            if n >= 800:
                interval = min(2500, interval + (n // 800) * 400)
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
        """Stop the live tile pump machinery: queue, pump flag, per-run refine
        cache, coalesced repaint timer. Does NOT touch the finalize/preview
        generations, so the finalize state machine can call it from its own
        drain step without invalidating itself."""
        self._resume_preview_jobs()
        self._auto_tile_queue.clear()
        self._auto_pump_scheduled = False
        # Fresh run / teardown: drop the live preset-refine cache (per-fid
        # refined copies) and its px/measurer, all tied to one run's grid/CRS.
        self._auto_live_refine_cache = {}
        self._auto_live_refine_px = -1.0
        self._auto_live_measurer = None
        self._auto_live_params = None
        # Invalidate the incremental provider mapping: the layer is recreated
        # fresh per run, and the review rebuilds it (truncate + re-add), so no
        # stale provider fid may survive into the next run or into review.
        self._auto_live_fid_map = {}
        timer = self._auto_repaint_timer
        if timer is not None:
            try:
                timer.stop()
            except (RuntimeError, AttributeError):
                pass

    def _reset_auto_live_pipeline(self) -> None:
        """Clear the live tile queue and stop the coalesced repaint timer.

        Called from every path that abandons or finishes a run so a late pump
        turn, a pending repaint, or a stale finalize step can never touch a
        torn-down layer/merger."""
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

    def _on_auto_all_finished(self, results: list) -> None:
        """Slot: all tiles finished. Export deduplicated results to a GeoPackage."""
        self._set_zone_badge_enabled(True)
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_run_active(False)
                self.dock_widget.set_auto_status("idle")
            except (RuntimeError, AttributeError):
                pass
        # Refresh the credit display now that the run consumed credits.
        self._refresh_auto_credits()

        # Billable tile count (zero-mask tiles included; failed tiles are
        # refunded server side). `results` holds one entry per DETECTION,
        # so len(results) is wrong whenever a tile yields 0 or 2+ masks.
        tiles_succeeded = getattr(self._auto_worker, "tiles_succeeded", 0)
        self._capture_auto_mask_gsd(self._auto_worker)
        self._auto_worker = None
        self._drop_auto_tile_bridge()
        self._auto_tel_stop_reason = "completed"
        self._finalize_auto_results(tiles_succeeded)

    def _finalize_auto_results(self, tiles_succeeded: int) -> None:
        """Turn the current merged set into a result.

        Headless/MCP: export straight to a layer (stable API contract).
        Interactive: enter the post-run review (refine, then Finish).

        Shared by a completed run and a stopped run (user cancel) so the
        billed partial results are never orphaned: a cancel drops the user
        into the same review of whatever was found so far.
        """
        # Run summary for observability: split the wall-clock into render (upfront
        # basemap fetch) vs detection (submit/inference/poll). This is the single
        # log to read when a run felt slow - it says WHERE the time went without
        # needing the user. Production-safe (durations + counts only).
        try:
            import time as _time
            detect_ms = (
                int((_time.monotonic() - self._auto_detect_t0) * 1000)
                if self._auto_detect_t0 else 0
            )
            # mask/render px ratio: 1.0 = the server answered at the sent tile
            # resolution (full detail); ~2.0 = masks came back on a half grid.
            # Production-safe (a bare ratio), and the quickest field check that
            # a run really got full-resolution masks.
            ratio = (self._auto_mask_gsd / self._auto_gsd
                     if self._auto_mask_gsd > 0 and self._auto_gsd > 0 else 0.0)
            QgsMessageLog.logMessage(
                "Auto detection: run summary - render {} ms, detect {} ms, "
                "{} tile(s) billed, {} raw detection(s), mask/render px ratio "
                "{:.2f}, {} saturated tile(s) re-split".format(
                    self._auto_render_ms, detect_ms, tiles_succeeded,
                    self._auto_raw_count, ratio,
                    getattr(self, "_auto_subdiv_tiles", 0)),
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )
        except (RuntimeError, AttributeError):
            pass

        # Pre-submit, uncharged tile drops: make them legible once per run.
        # Blank/nodata skips saved the user credits; render/provider holes mean
        # a slow-server run may have coverage gaps. Both are already unbilled.
        blank_n = int(getattr(self, "_auto_skipped_blank_tiles", 0) or 0)
        holes_n = int(getattr(self, "_auto_render_failed_tiles", 0) or 0)
        if (blank_n or holes_n) and not self._auto_headless_run:
            try:
                # When the drops dominate the run (a basemap that would not
                # load), a soft info line reads as success: escalate to one
                # loud warning that says the area was NOT analyzed.
                dropped = blank_n + holes_n
                major_hole = dropped >= 10 or dropped > tiles_succeeded
                if major_hole:
                    self.iface.messageBar().pushWarning(
                        "AI Segmentation",
                        tr("{n} tiles had no imagery and were not analyzed "
                           "(not charged). Check the imagery layer loads over "
                           "this area, then run Detect again.").format(n=dropped),
                    )
                else:
                    if blank_n:
                        self.iface.messageBar().pushInfo(
                            "AI Segmentation",
                            tr("Skipped {n} empty tiles (not charged).").format(n=blank_n),
                        )
                    if holes_n:
                        self.iface.messageBar().pushWarning(
                            "AI Segmentation",
                            tr("{n} tiles could not be loaded from the layer server; "
                               "results may be incomplete.").format(n=holes_n),
                        )
            except (RuntimeError, AttributeError):
                pass
        if blank_n or holes_n:
            QgsMessageLog.logMessage(
                "Auto detection: {} blank tile(s) skipped, {} render hole(s)".format(
                    blank_n, holes_n),
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )

        # Telemetry: report degraded tiles once per run (skipped / timed out /
        # blank-skipped / render holes).
        if (self._auto_skipped_tiles or self._auto_timeout_tiles or blank_n or holes_n):
            try:
                from ...core import telemetry
                telemetry.track_auto_tiles_degraded(
                    run_id=self._auto_run_id or "",
                    skipped_tiles=self._auto_skipped_tiles,
                    timeout_tiles=self._auto_timeout_tiles,
                    blank_tiles=blank_n,
                    render_failed_tiles=holes_n,
                )
            except Exception:
                pass  # nosec B110

        # The live pump is asynchronous, so the queue can still hold tiles when
        # the run signals completion. Headless blocks its caller and has no UI
        # to protect: drain synchronously and finalize inline, as before.
        if self._auto_headless_run:
            self._drain_auto_tiles_now()
            self._reset_auto_live_pipeline()
            # Same exemplar-only count-vs-map decision as the interactive path
            # (a MAP decision keeps the live merger, a SEPARATE one re-merges the
            # retained fragments), so the MCP contract reflects the same grouping.
            merged_ided = self._resolve_exemplar_finalize_ided()
            self._auto_merger = None
            if not merged_ided:
                self._record_auto_zero_result(tiles_succeeded)
                return
            # Sweep + build synchronously and export the default-filtered
            # visible set. No forced shape refine here: the Automatic path
            # stays faithful by default (the user opts into simplify/round/
            # expand/fill in the review).
            from ...core.polygon_exporter import drop_covered_objects
            merged_ided = drop_covered_objects(merged_ided)
            if not merged_ided:
                self._record_auto_zero_result(tiles_succeeded)
                return
            self._auto_objects = self._build_auto_objects(merged_ided)
            self._reset_review_refine_cache()
            visible, vis_scores = self._compute_visible_objects(
                self._fresh_review_params(), self._auto_refine_pixel_size(),
                with_scores=True)
            self._complete_auto_finalize(
                visible, tiles_succeeded, scores=vis_scores)
            return

        # Interactive: drain the leftover queue COOPERATIVELY as finalize phase
        # "drain" (time-sliced, yielding to the event loop between slices). The
        # old synchronous drain blocked the GUI for the WHOLE backlog: on a
        # dense continuous run (roads) the last tiles' merges are the most
        # expensive, the async pump falls behind, and the backlog landed here
        # in one blocking bite - the end-of-run beachball. The gen bump
        # supersedes any in-flight finalize/reslice from a prior run.
        self._auto_finalize_gen += 1
        self._auto_finalize_state = {
            "mode": "finalize",
            "phase": "drain",
            "tiles_succeeded": tiles_succeeded,
            "gen": self._auto_finalize_gen,
        }
        self._step_auto_finalize_refine()

    def _finalize_drain_done(self, state: dict) -> None:
        """Drain phase complete: read the merger, stop the live pump machinery
        and seed the sweep phase (or record a zero result). Continues the
        cooperative finalize started in _finalize_auto_results."""
        # Stop the pump WITHOUT bumping the finalize gen (that would invalidate
        # the very state machine we are running); the preview-build gen is
        # bumped so a prior run's in-flight cache build can never leak in.
        self._stop_auto_live_pump()
        self._auto_preview_build_gen += 1
        self._auto_preview_build_state = None
        # Objects were stitched incrementally by the merger (confidence-agnostic),
        # so the result is already WHOLE objects (no tile seams), not raw
        # fragments. Read them SCORED (each object's representative score = the max
        # of its fragments) so the review filters act on whole objects, never on
        # cut fragments (the fix for "confidence cuts buildings in half").
        # (stable_id, geom, score) triples: the id carries each object's live
        # colour into the review so the hue is continuous across the run->review
        # boundary, not just stable during the run. For an exemplar-only run the
        # count-vs-map policy is decided HERE from the run's own masks, which may
        # replace the merged set with a fresh SEPARATE re-merge of the fragments.
        merged_ided = self._resolve_exemplar_finalize_ided()
        self._auto_merger = None
        # The end-of-run redundancy sweep (drop leftover partial readings mostly
        # painted over by larger objects: patches/strips double-painting a big
        # roof, which pairwise dedup cannot catch) is GEOS-heavy on dense runs,
        # so it runs cooperatively (phase "sweep") and never freezes the GUI at
        # "run finished".
        if not merged_ided:
            self._auto_finalize_state = None
            self._record_auto_zero_result(state["tiles_succeeded"])
            return

        from ...core.polygon_exporter import CoverSweep
        state.update({
            "phase": "sweep",
            "sweep": CoverSweep(list(merged_ided)),
            "sweep_before": len(merged_ided),
            "measurer": self._make_auto_area_measurer(),
            "params": self._fresh_review_params(),
            "pixel_size": self._auto_refine_pixel_size(),
        })
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(0, self._step_auto_finalize_refine)

    # ---- Exemplar-only count-vs-map auto decision + override ----------------

    def _resolve_exemplar_finalize_ided(self) -> list:
        """The (fid, geom, score) merged set finalize should build objects from.

        For a prompted run (or any run that is not exemplar-only) this is simply
        the live merger's result. For an exemplar-only run it applies the
        automatic count-vs-map decision from the run's own masks: the live merger
        already streamed MAP, so a MAP decision keeps it, while a SEPARATE
        decision re-merges the retained raw fragments client-side (gates + a
        fresh SEPARATE merger). The chosen mode is stored BEFORE the review
        opens, since downstream seam logic reads it."""
        live = (
            self._auto_merger.result_scored_ided()
            if self._auto_merger is not None else []
        )
        if not getattr(self, "_auto_is_exemplar_only", False):
            return live
        want_separate = self._decide_exemplar_merge_separate()
        self._auto_merge_mode_source = "signal"
        frags = getattr(self, "_auto_raw_fragments", None)
        if want_separate and frags:
            self._auto_merge_separate = True
            return self._remerge_raw_fragments(True)
        # MAP (or SEPARATE wanted but fragments overflowed / empty): the live
        # merger is already MAP, so keep it and record MAP as the mode.
        self._auto_merge_separate = False
        return live

    def _decide_exemplar_merge_separate(self) -> bool:
        """True = keep SEPARATE (count), False = MAP (continuous cover), decided
        from the run's own masks.

        The signal is the area-weighted mean tile coverage of the fragments
        (sum(cov^2)/sum(cov), cov = fragment ground area / tile ground area,
        failure blobs above the hard cap excluded): the tile fraction a typical
        unit of detected ground belongs to. Continuous cover (many medium
        fragments tiling the zone) scores high; small countable objects (each a
        tiny fraction of a tile) score near zero, and excluding the failure blobs
        keeps a handful from faking cover. Compared against the server-tunable
        map_likeness_min_share; when no fragment was measured the counting-safe
        policy default stands."""
        from ...core import detection_policy
        cov_sum = float(getattr(self, "_auto_raw_cov_sum", 0.0) or 0.0)
        if cov_sum <= 0.0:
            return detection_policy.exemplar_only_merge_separate()
        cov_sq_sum = float(getattr(self, "_auto_raw_cov_sq_sum", 0.0) or 0.0)
        map_likeness = cov_sq_sum / cov_sum
        threshold = detection_policy.map_likeness_min_share()
        is_map = map_likeness >= threshold
        QgsMessageLog.logMessage(
            "Auto detection: exemplar-only map-likeness {:.3f} vs threshold "
            "{:.3f} -> {}".format(
                map_likeness, threshold,
                "continuous cover" if is_map else "distinct objects"),
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )
        return not is_map

    def _remerge_raw_fragments(self, merge_separate: bool) -> list:
        """Re-merge the retained exemplar raw fragments the given way and return
        the (fid, geom, score) merged set. SEPARATE first applies the worker's
        coverage gates client-side (hard cap drop + a compactness check above the
        soft cap), then folds survivors into a fresh IncrementalMerger built with
        the exact kwargs the run merger uses. Bounded synchronous work (the
        SEPARATE branch only runs on runs the signal judged NOT map-like, so the
        fragment count is modest); logged for observability. The cover sweep runs
        downstream (finalize) or in the caller (override)."""
        import time as _t
        from qgis.core import QgsGeometry
        from ...core.polygon_exporter import IncrementalMerger
        from ...workers.auto_detection_worker import (
            AutoDetectionWorker, _HARD_TILE_COVERAGE, _MAX_TILE_COVERAGE,
        )

        frags = getattr(self, "_auto_raw_fragments", None) or []
        # _auto_seam_min_dim reads _auto_merge_separate, so set it first.
        self._auto_merge_separate = merge_separate
        ms = self._auto_merge_scalars or {}
        merger = IncrementalMerger(
            seam_min_dim=self._auto_seam_min_dim(),
            select_duplicates=merge_separate,
            gsd=self._auto_gsd,
            merge_ios=ms.get("merge_ios", 0.15),
            dedup_ios=ms.get("dedup_ios", 0.5),
            dup_ios_floor=ms.get("dup_ios_floor", 0.3),
            dup_centroid_frac=ms.get("dup_centroid_frac", 0.35),
            seam_span_ios=ms.get("seam_span_ios", 0.03),
        )
        tile_area = float(getattr(self, "_auto_tile_ground_area", 0.0) or 0.0)
        # Same server-overridable coverage gates the run worker resolved.
        from ...core.detection_policy import hard_tile_coverage, max_tile_coverage
        hard_cov = hard_tile_coverage(_HARD_TILE_COVERAGE)
        max_cov = max_tile_coverage(_MAX_TILE_COVERAGE)
        t0 = _t.monotonic()
        gated = 0
        for wkb, score in frags:
            geom = QgsGeometry()
            geom.fromWkb(wkb)
            if geom.isEmpty():
                continue
            if merge_separate and tile_area > 0:
                cov = geom.area() / tile_area
                if cov > hard_cov:
                    gated += 1
                    continue
                if cov > max_cov and not AutoDetectionWorker._is_compact_shape(geom):
                    gated += 1
                    continue
            merger.add(geom, float(score))
        out = merger.result_scored_ided()
        QgsMessageLog.logMessage(
            "Auto detection: re-merged {} raw fragment(s) as {} ({} gated) in "
            "{} ms".format(
                len(frags), "distinct objects" if merge_separate else "continuous cover",
                gated, int((_t.monotonic() - t0) * 1000)),
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )
        return out

    def _rebuild_auto_objects_for_mode(self, merge_separate: bool) -> bool:
        """Re-merge the retained exemplar fragments the given way, run the cover
        sweep, and rebuild _auto_objects (+ its stable fids). Used by the review
        override; returns False when there is nothing to re-merge."""
        frags = getattr(self, "_auto_raw_fragments", None)
        if not frags:
            return False
        from ...core.polygon_exporter import drop_covered_objects
        merged_ided = self._remerge_raw_fragments(merge_separate)
        merged_ided = drop_covered_objects(merged_ided)
        self._auto_objects = self._build_auto_objects(merged_ided)
        self._reset_review_refine_cache()
        # A re-grouping rebuilds the object set, so index-based per-object
        # removals from the prior grouping no longer map; drop them (geometry-
        # based protected edits still apply through the reslice overlap check).
        self._auto_manual_removed = set()
        return True

    def _clear_auto_raw_fragments(self) -> None:
        """Drop the exemplar raw-fragment retention + counters (a review end, a
        new run, a teardown). Cheap and idempotent."""
        self._auto_is_exemplar_only = False
        self._auto_raw_fragments = None
        self._auto_raw_n_total = 0
        self._auto_raw_cov_sum = 0.0
        self._auto_raw_cov_sq_sum = 0.0
        self._auto_tile_ground_area = 0.0

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

    def _build_auto_objects(self, merged_ided) -> list:
        """Synchronous (geom, score, area) build from the merger's ided result.
        Also records the parallel stable fid per object (_auto_object_fids) so the
        Random hue matches the live run. Used by the headless path; the
        interactive path builds it cooperatively. Detections below the review
        noise floor are dropped here so they never reach the review at all."""
        measurer = self._make_auto_area_measurer()
        floor = self._review_noise_floor()
        out = []
        fids = []
        for fid, geom, score in merged_ided:
            if geom is None or geom.isEmpty():
                continue
            if float(score) < floor:
                continue
            out.append((geom, float(score), self._object_area_m2(geom, measurer)))
            fids.append(fid)
        self._auto_object_fids = fids
        return out

    def _object_fid_for(self, idx: int) -> int:
        """Stable colour id (the merger's keeper fid) for a canonical object
        index, so the Random hue is identical live and in review. Falls back to
        the index itself when the fid list is missing or short (never raises into
        a repaint)."""
        fids = getattr(self, "_auto_object_fids", None)
        if fids is not None and 0 <= idx < len(fids):
            return fids[idx]
        return idx

    def _auto_review_preset(self) -> dict:
        """The run's smart review defaults: prompt-shaped regularizers (Right
        angles/Fill holes/Round corners per object kind) + the resolution-aware
        Min size floor. Recomputed per call from the run context, so every NEW
        run starts from ITS optimum (no cross-run memory by design); the user
        can still override any control in the review."""
        from ...core.review_presets import review_preset_for
        prompt = str((self._auto_run_ctx or {}).get("prompt") or "")
        # Meters per RETURNED-mask pixel: the run's meter GSD scaled by the
        # observed mask/render ratio (the mask grid is what the polygons
        # staircase on, so it is the real noise floor).
        gsd_m = getattr(self, "_auto_gsd_m", 0.0)
        mask_gsd = getattr(self, "_auto_mask_gsd", 0.0)
        if gsd_m > 0 and mask_gsd > 0 and self._auto_gsd > 0:
            gsd_m *= mask_gsd / self._auto_gsd
        # Prefer the server run plan's review block when it was fetched for this
        # run's prompt; else the blob/generic prompt-shaped preset.
        plan = self._active_run_plan(prompt)
        if plan is not None:
            preset = self._review_preset_from_plan(plan.get("review"), gsd_m)
            if preset is not None:
                return preset
        return review_preset_for(prompt, gsd_m)

    def _review_preset_from_plan(self, review: object, gsd_m: float) -> dict | None:
        """Build the review preset dict from a run plan's ``review`` block, or
        None when it is missing/malformed. The plan's ``min_size_m2`` is the
        OBJECT floor only; the client still maxes it with its own resolution
        noise floor (3 x mask ground pixel)^2, which stays client-side."""
        if not isinstance(review, dict):
            return None
        try:
            noise = (3.0 * gsd_m) ** 2 if gsd_m and gsd_m > 0 else 0.0
            object_floor = float(review.get("min_size_m2") or 0.0)
            return {
                "simplify_px": float(review.get("simplify_px", _AUTO_REVIEW_SIMPLIFY_DEFAULT)),
                "smooth": bool(review.get("smooth", _AUTO_REVIEW_SMOOTH_DEFAULT)),
                "expand_px": int(review.get("expand_px", _AUTO_REVIEW_EXPAND_DEFAULT)),
                "fill_holes": bool(review.get("fill_holes", _AUTO_REVIEW_FILL_HOLES_DEFAULT)),
                "clean_px": float(review.get("clean_px", _AUTO_REVIEW_CLEAN_DEFAULT)),
                "ortho": bool(review.get("ortho", _AUTO_REVIEW_ORTHO_DEFAULT)),
                "min_size_m2": round(max(object_floor, noise), 1),
                "shape_class": str(review.get("shape_class", "server")),
            }
        except (TypeError, ValueError):
            return None

    def _effective_confidence_default(self) -> float:
        """The review's starting confidence default: the run plan's value when
        it matches this run's prompt, else the exemplar-only default for an
        exemplar-only run, else the prompt's shape-class value from the server
        policy, else the generic default. Object classes (and an exemplar-only
        run's lack of a text prior) score differently on the same true object,
        so the start adapts instead of one flat cutoff. The live seed uses the
        SAME resolver (review_start_confidence_default) so the live preview and
        the review open at the same cutoff. The auto_lowered comparisons all
        read this so they compare against the SAME default the review seeds
        from."""
        prompt = str((self._auto_run_ctx or {}).get("prompt") or "")
        plan = self._active_run_plan(prompt)
        if plan is not None:
            c = plan.get("confidence_default")
            if isinstance(c, (int, float)) and not isinstance(c, bool):
                return float(c)
        from ...core.review_presets import review_start_confidence_default

        return review_start_confidence_default(
            prompt, bool(getattr(self, "_auto_is_exemplar_only", False)))

    def _fresh_review_params(self) -> dict:
        """Review filter/refine params for a fresh result: the pre-run
        confidence, no max size, and the run's smart preset for the shape
        refine + Min size (see _auto_review_preset). Used at finalize so a
        stale filter from a prior review never touches a fresh run before the
        review widgets are (re)seeded with the same preset."""
        preset = self._auto_review_preset()
        return {
            "conf": self._auto_confidence,
            "min_a": float(preset["min_size_m2"]),
            "max_a": 0.0,
            "simplify_px": float(preset["simplify_px"]),
            "smooth": bool(preset["smooth"]),
            "expand_px": int(preset["expand_px"]),
            "fill_holes": bool(preset["fill_holes"]),
            "open_px": float(preset["clean_px"]),
            "ortho": bool(preset["ortho"]),
        }

    def _widget_review_params(self) -> dict:
        """Current review filter/refine params read from the dock widgets (a
        reslice snapshot). Confidence comes from _auto_confidence (the confidence
        handler keeps it in sync)."""
        params = self._fresh_review_params()
        d = self.dock_widget
        if d is None:
            return params
        try:
            params["conf"] = self._auto_confidence
            params["min_a"] = d.get_auto_min_size()
            params["max_a"] = d.get_auto_max_size()
            simplify, smooth, expand, fill, clean, ortho = d.get_auto_refine_params()
            params["simplify_px"] = simplify
            params["smooth"] = smooth
            params["expand_px"] = expand
            params["fill_holes"] = fill
            params["open_px"] = clean
            params["ortho"] = ortho
        except (RuntimeError, AttributeError):
            pass
        return params

    def _passes_review_filters(self, score: float, area: float, params: dict) -> bool:
        """Whole-object confidence + min/max-size gate (the VISIBLE-set filter)."""
        if score < params.get("conf", 0.0):
            return False
        min_a = params.get("min_a", 0.0)
        max_a = params.get("max_a", 0.0)
        if min_a > 0 and area < min_a:
            return False
        if max_a > 0 and area > max_a:
            return False
        return True

    def _refine_geom_for_review(self, base, params: dict, pixel_size: float):
        """Apply the review shape-refine controls to a base object geometry.
        simplify/expand are px in the UI, converted to ground units by the run's
        detection pixel size (_auto_refine_pixel_size). Returns a NEW geometry
        (base is never mutated)."""
        from ...core.polygon_exporter import apply_geometry_refinement
        px = pixel_size if pixel_size and pixel_size > 0 else 1.0
        return apply_geometry_refinement(
            QgsGeometry(base),
            simplify_tol=float(params.get("simplify_px", 0)) * px,
            smooth=bool(params.get("smooth", False)),
            expand_dist=float(params.get("expand_px", 0)) * px,
            fill_holes=bool(params.get("fill_holes", False)),
            open_dist=float(params.get("open_px", 0)) * px,
            ortho=bool(params.get("ortho", False)),
            # Right angles needs a de-staircased outline first (a raw mask
            # outline is ALREADY all 90-degree stair steps, so orthogonalizing
            # it alone is a no-op): 2.5 detection px, the regularization
            # literature's simplify-then-orthogonalize tolerance.
            ortho_tol=2.5 * px,
        )

    @staticmethod
    def _review_shape_key(params: dict, pixel_size: float) -> tuple:
        """Hashable identity of the SHAPE portion of the review params (+ the
        px scale they are applied at). Confidence and Min/Max size are filters,
        not shape ops, so they stay OUT: two reslices that differ only by a
        filter share every refined geometry through _auto_reslice_cache."""
        return (
            round(float(params.get("simplify_px", 0) or 0.0), 4),
            bool(params.get("smooth", False)),
            round(float(params.get("expand_px", 0) or 0.0), 4),
            bool(params.get("fill_holes", False)),
            round(float(params.get("open_px", 0) or 0.0), 4),
            bool(params.get("ortho", False)),
            round(float(pixel_size or 0.0), 6),
        )

    def _reset_review_refine_cache(self) -> None:
        """Drop the reslice refine cache. Must run whenever _auto_objects is
        rebuilt or cleared (finalize build, exemplar re-group, export, discard,
        run start): the cache is keyed by object INDEX, so a stale entry after
        a rebuild would render the wrong object. The review provider map goes
        with it (det_ids from a rebuilt object set can collide with the old
        ones), which forces the next push to a full rebuild of the layer."""
        self._auto_reslice_cache = {"key": None, "geoms": {}}
        self._review_fid_map = {}

    def _review_refined_geom(self, det_idx: int, base, params: dict,
                             pixel_size: float):
        """Refined + normalized (repaired, MultiPolygon-coerced) geometry for
        one canonical object, memoized in _auto_reslice_cache. A cache hit is a
        dict lookup, so filter-only reslices (Confidence / Min / Max size) never
        re-run the GEOS refine on objects whose shape params did not change.
        Returns None when the refine emptied the geometry. Callers must not
        mutate the returned geometry (copy first)."""
        cache = self._auto_reslice_cache
        key = self._review_shape_key(params, pixel_size)
        if cache.get("key") != key:
            cache["key"] = key
            cache["geoms"] = {}
        geoms = cache["geoms"]
        if det_idx in geoms:
            return geoms[det_idx]
        g = self._refine_geom_for_review(base, params, pixel_size)
        if g is not None and not g.isEmpty():
            # Normalize ONCE at cache-fill time (repair + MultiPolygon coerce),
            # so every later push of this geometry skips both.
            from ...core.layer_conventions import repair_polygon, to_multipolygon
            g = to_multipolygon(repair_polygon(g) or g)
        result = g if (g is not None and not g.isEmpty()) else None
        geoms[det_idx] = result
        return result

    def _compute_visible_objects(
        self, params: dict, pixel_size: float, with_scores: bool = False,
    ) -> list | tuple[list, list]:
        """Synchronous filter+refine of _auto_objects into the visible geom set:
        whole-object confidence + size filter, then the shape refine. Used by the
        headless path; the interactive path does the same cooperatively.
        ``with_scores`` also returns the parallel per-object score list, so the
        headless export can fill the layer's `score` field like the review path
        does (it silently exported NULL scores before)."""
        removed = getattr(self, "_auto_manual_removed", None) or set()
        out = []
        out_scores = []
        for det_idx, (base, score, area) in enumerate(self._auto_objects):
            if det_idx in removed or base is None or base.isEmpty():
                continue
            if not self._passes_review_filters(score, area, params):
                continue
            g = self._review_refined_geom(det_idx, base, params, pixel_size)
            if g is not None:
                out.append(g)
                out_scores.append(float(score))
        if with_scores:
            return out, out_scores
        return out

    def _record_auto_zero_result(self, tiles_succeeded: int) -> None:
        """Shared 'no detections' bookkeeping for the finalize paths."""
        result = {
            "status": "completed",
            "instances": 0,
            "tiles_processed": tiles_succeeded,
            "layer_name": None,
        }
        prior = self._last_auto_result
        if isinstance(prior, dict) and prior.get("status") == "credits_exhausted":
            # Exhausted with nothing kept: preserve the exhaustion signal
            # (recorded before finalize) so the caller sees the real stop cause.
            result["status"] = "credits_exhausted"
            result["credits_remaining"] = prior.get("credits_remaining", 0)
        self._last_auto_result = result
        try:
            from ...core import telemetry
            ctx = self._auto_run_ctx or {}
            total = ctx.get("total", tiles_succeeded)
            telemetry.track_auto_zero_result(
                run_id=self._auto_run_id or "",
                tiles=tiles_succeeded,
                object_class=ctx.get("prompt") or "Example match",
                had_exemplar=self._auto_exemplar_store.count() > 0,
            )
            # A natural completion with no detections is still one completed run
            # (cancel/exhaust already emitted their own terminal event).
            if self._auto_tel_stop_reason in (None, "completed"):
                telemetry.track_auto_detect_completed(
                    run_id=self._auto_run_id or "",
                    duration_ms=self._auto_duration_ms(),
                    tiles_done=tiles_succeeded,
                    tiles_failed=max(0, total - tiles_succeeded),
                    instances_found=0,
                    instances_visible_at_default=0,
                    zero_at_default=True,
                    stop_reason="completed",
                    warming_ms=self._auto_warming_wait_ms(),
                    merge_mode_final="separate" if self._auto_merge_separate else "map",
                    merge_override_used=bool(getattr(self, "_auto_merge_override_used", False)),
                )
        except Exception:
            pass  # nosec B110
        self._on_auto_zero_detections(tiles_succeeded)
        self._remove_auto_selection_layer()
        self._clear_auto_raw_fragments()

    def _step_auto_finalize_refine(self) -> None:
        """One cooperative slice of the finalize/reslice pipeline, yielding to the
        event loop between slices so QGIS never freezes. Two phases, both
        time-sliced:

          - "build" (finalize only): measure each merged WHOLE object's geodesic
            area and store (geom, score, area) in _auto_objects. No shape refine
            here: the Automatic path stays faithful by default.
          - "filter" (both modes): filter the canonical whole objects by the
            snapshotted confidence + min/max size, then apply the user shape
            refine to those that pass, building the VISIBLE geometry set. This is
            a pure recompute on already-merged objects: it never re-merges and so
            never re-cuts an object (the fix for confidence cutting buildings).

        On completion, "finalize" enters the post-run review; "reslice" swaps the
        review's visible geoms. Generation-guarded so a new run / teardown / later
        reslice supersedes an in-flight one (last one wins)."""
        state = self._auto_finalize_state
        if state is None or state.get("gen") != self._auto_finalize_gen:
            return  # superseded by a new run, a later reslice, or torn down
        from qgis.PyQt.QtCore import QTimer
        import time as _t

        deadline = _t.monotonic() + _AUTO_PUMP_BUDGET_S

        # Phase -1 (finalize only): fold the tiles the async pump had not
        # processed yet into the merger, time-sliced. The old synchronous
        # end-of-run drain blocked the GUI for the whole backlog (the dense-run
        # beachball); this keeps the event loop and the live preview alive.
        if state.get("phase") == "drain":
            progressed = self._process_auto_queue(deadline)
            if self._auto_tile_queue:
                if progressed:
                    self._request_auto_live_repaint()
                QTimer.singleShot(0, self._step_auto_finalize_refine)
                return
            self._finalize_drain_done(state)
            return

        # Phase 0 (finalize only): the end-of-run redundancy sweep, time-sliced.
        # GEOS-heavy on dense runs, so chunk it and yield between chunks.
        if state.get("phase") == "sweep":
            sweep = state["sweep"]
            done = False
            while not done and _t.monotonic() < deadline:
                done = sweep.step(128)
            if not done:
                QTimer.singleShot(0, self._step_auto_finalize_refine)
                return
            merged_ided = sweep.result()
            before = state.get("sweep_before", len(merged_ided))
            if len(merged_ided) != before:
                QgsMessageLog.logMessage(
                    "Auto detection: redundancy sweep dropped %d covered "
                    "fragment(s) of %d objects" % (before - len(merged_ided), before),
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )
            if not merged_ided:
                self._auto_finalize_state = None
                self._record_auto_zero_result(state["tiles_succeeded"])
                return
            # Sweep done: seed the build phase and continue.
            state["phase"] = "build"
            state["build_pending"] = list(merged_ided)
            state["total_build"] = len(merged_ided)
            state["objects"] = []
            state["object_fids"] = []
            state.pop("sweep", None)
            QTimer.singleShot(0, self._step_auto_finalize_refine)
            return

        # Phase 1 (finalize only): build the canonical (geom, score, area) set.
        if state.get("phase") == "build":
            build_pending = state["build_pending"]
            objects = state["objects"]
            object_fids = state["object_fids"]
            measurer = state["measurer"]
            # Below-floor detections are pure noise: drop them here so they never
            # count in the review total nor render (also fewer shapes to build).
            floor = self._review_noise_floor()
            while build_pending:
                fid, geom, score = build_pending.pop()
                if geom is not None and not geom.isEmpty() and float(score) >= floor:
                    objects.append(
                        (geom, float(score), self._object_area_m2(geom, measurer)))
                    # Parallel to objects (same append guard = same order), so the
                    # review's Random hue keys on the same merger id the live run
                    # used: an object's colour survives the run->review handoff.
                    object_fids.append(fid)
                if _t.monotonic() >= deadline:
                    break
            if build_pending:
                QTimer.singleShot(0, self._step_auto_finalize_refine)
                return
            # Build done: publish the canonical set, seed the confidence-drag
            # preview cache, then move to the shared filter phase.
            self._auto_objects = objects
            self._auto_object_fids = object_fids
            self._reset_review_refine_cache()
            # Adaptive starting confidence: if 0.30 would hide EVERY found object,
            # drop to the highest 5% step that shows at least one, so the review
            # never opens reading "0 found". Set it BEFORE the filter phase below
            # so the visible set reflects the lowered cutoff; update the snapshot
            # params too. Headless keeps the seeded default (stable API contract).
            raw_start_conf = self._review_start_confidence()
            # Snap the cutoff to the review slider's 5% grid ONCE, at the source,
            # so the stored filter value, the histogram cutoff and the seeded
            # slider/spin all show the SAME number (the slider can only rest on
            # 5% steps, so an unsnapped 0.17 used to read as 15% while the filter
            # ran at 17%). The note flags below still read the UNSNAPPED value so
            # a pure snap never masquerades as a deliberate lowering.
            start_conf = self._snap_review_start_confidence(raw_start_conf)
            self._auto_confidence = start_conf
            state["params"]["conf"] = start_conf
            if not self._auto_headless_run and self.dock_widget is not None:
                default_conf = self._effective_confidence_default()
                auto_lowered = raw_start_conf < default_conf - 1e-9
                # "adaptive" = lowered while objects DO score above the default
                # (distribution fit); the other reason is "nothing scored above".
                adaptive_note = auto_lowered and any(
                    s >= default_conf for (_g, s, _a) in objects)
                # "tuned" = the start IS the default, but an object-specific
                # one (differs from the generic cutoff): explain it so a
                # non-30% opening value never looks arbitrary.
                from ...core.detection_policy import confidence_default
                tuned_note = (not auto_lowered) and abs(
                    default_conf - confidence_default()) > 1e-9
                try:
                    spin = self.dock_widget.auto_confidence_spin
                    spin.blockSignals(True)
                    spin.setValue(start_conf)
                    spin.blockSignals(False)
                    # The review page already seeded its slider/spin from the
                    # pre-run dial when it opened (before this async step), so
                    # push the REAL starting cutoff into them too: without
                    # this, a tuned/adaptive start filtered at one value while
                    # the visible handle rested at another.
                    self.dock_widget.seed_review_confidence(
                        int(round(start_conf * 100)))
                    self.dock_widget.set_review_conf_lowered_note(
                        auto_lowered or tuned_note, int(round(start_conf * 100)),
                        adaptive=adaptive_note, tuned=tuned_note)
                    # Clamp the confidence controls so neither the slider nor the
                    # spinbox can dial below the noise floor (sub-floor detections
                    # were already dropped, so a cutoff under it is meaningless).
                    floor_pct = int(math.ceil(self._review_noise_floor() * 100))
                    self.dock_widget.set_review_conf_floor(floor_pct)
                    hist = getattr(self.dock_widget, "auto_conf_histogram", None)
                    if hist is not None:
                        # The histogram must span EXACTLY the slider's range
                        # (its clamped minimum, not the raw floor), so the
                        # grey/blue boundary sits above the handle at every
                        # cutoff instead of drifting on a different scale.
                        from ..dock.styles import _REVIEW_CONF_MIN
                        hist.set_range(
                            max(floor_pct, _REVIEW_CONF_MIN) / 100.0, 0.95)
                        hist.set_scores([s for (_g, s, _a) in objects])
                        hist.set_cutoff(start_conf)
                except (RuntimeError, AttributeError):
                    pass
            self._start_build_preview_cache(state["pixel_size"])
            # Enumerate so each visible geom remembers WHICH canonical object it
            # came from; the visible id it records is the object's stable merger
            # fid (_object_fid_for), so the Random display colours key on the same
            # id the live run used and a reslice never reshuffles a colour.
            state["filter_pending"] = list(enumerate(objects))
            state["total_filter"] = len(objects)
            state["visible"] = []
            state["visible_scores"] = []
            state["visible_ids"] = []
            state["phase"] = "filter"
            QTimer.singleShot(0, self._step_auto_finalize_refine)
            return

        # Phase 2 (both modes): filter whole objects, then shape-refine the pass.
        filter_pending = state["filter_pending"]
        visible = state["visible"]
        visible_scores = state["visible_scores"]
        visible_ids = state["visible_ids"]
        params = state["params"]
        pixel_size = state["pixel_size"]
        removed = getattr(self, "_auto_manual_removed", None) or set()
        while filter_pending:
            det_idx, (base, score, area) = filter_pending.pop()
            base_ok = base is not None and not base.isEmpty()
            if det_idx not in removed and base_ok and self._passes_review_filters(score, area, params):
                # Cached per object + shape key: a filter-only reslice
                # (Confidence / Min / Max size) is pure dict lookups here.
                g = self._review_refined_geom(det_idx, base, params, pixel_size)
                if g is not None:
                    visible.append(g)
                    visible_scores.append(score)
                    visible_ids.append(self._object_fid_for(det_idx))
            if _t.monotonic() >= deadline:
                break
        if filter_pending:
            # Progressive apply (reslice only): every ~250 ms, write the geoms
            # refined SO FAR onto the layer (diff-only, objects not yet
            # processed keep their old shape). A shape-settings change then
            # visibly sweeps the map instead of freezing on the old state
            # until the whole cooperative pass ends.
            if state.get("mode") == "reslice" and visible:
                now = _t.monotonic()
                if now - state.get("last_partial", 0.0) >= 0.25:
                    state["last_partial"] = now
                    self._push_review_geoms(
                        visible, repair=False, scores=visible_scores,
                        ids=visible_ids,
                        stamp=("acc", (self._auto_reslice_cache or {}).get("key")),
                        partial=True)
            QTimer.singleShot(0, self._step_auto_finalize_refine)
            return
        vis_scores = state.get("visible_scores", [])
        vis_ids = state.get("visible_ids", [])
        self._auto_finalize_state = None
        if state.get("mode") == "reslice":
            self._apply_auto_reslice_result(visible, vis_scores, vis_ids)
        else:
            self._complete_auto_finalize(
                visible, state["tiles_succeeded"], vis_scores, vis_ids)

    def _apply_auto_reslice_result(self, geoms: list,
                                   scores: list | None = None,
                                   ids: list | None = None) -> None:
        """Completion of a cooperative review reslice: swap in the new geoms,
        update the count, refresh the preview. No re-detection, no credits.
        ``scores`` / ``ids`` are parallel per-object lists (same order as geoms)
        feeding the review heatmap and the stable Random colours.

        Hand-edited objects are PROTECTED: drop any re-filtered auto detection
        that overlaps one, then add the protected geoms back, so a confidence
        change never wipes manual work (it only re-filters untouched detections)."""
        if not self._auto_review:
            return
        if self._auto_protected_geoms:
            # Filter + re-merge scoped to the protected neighbourhood so this
            # cooperative-reslice tail does not re-freeze the GUI on a big
            # detection set (the whole point of the time-sliced reslice).
            geoms = self._merge_kept_with_protected(
                geoms, list(self._auto_protected_geoms))
            # Geoms were rewritten (dissolve), so the parallel scores/ids no
            # longer align: fall back to neutral coloring rather than mislabel.
            self._auto_review["scores"] = None
            self._auto_review["ids"] = None
            self._auto_review["stamp"] = None
        else:
            self._auto_review["scores"] = scores
            self._auto_review["ids"] = ids
            # Reslice output geoms are the cache-normalized refined objects:
            # stamp them with the shape key so the incremental push writes only
            # the delta (identical det_id + stamp = geometry unchanged).
            self._auto_review["stamp"] = (
                "acc", (self._auto_reslice_cache or {}).get("key"))
        self._auto_review["geoms"] = geoms
        self._update_review_header(len(geoms))
        self._refresh_auto_review_preview()

    def _start_build_preview_cache(self, pixel_size: float) -> None:
        """Kick a cooperative, time-sliced build of the confidence-slider preview
        cache: the canonical WHOLE objects (geom, score), lightly simplified and
        sorted by score desc, so a slider drag is a cheap prefix slice of whole
        objects (never fragments). Runs in the background after the objects are
        built so it never blocks entering review; the slider drag falls back to
        filtering _auto_objects directly until it is ready."""
        from qgis.PyQt.QtCore import QTimer

        self._auto_preview_build_gen += 1
        self._auto_preview_build_state = {
            "pending": [(i, g, s)
                        for i, (g, s, _a) in enumerate(self._auto_objects)],
            "out": [],
            "pixel_size": pixel_size,
            "gen": self._auto_preview_build_gen,
        }
        QTimer.singleShot(0, self._step_build_preview_cache)

    def _step_build_preview_cache(self) -> None:
        """One cooperative slice of the preview-cache build: simplify a batch of
        scored geoms, then reschedule. On completion sort by score desc once and
        publish to _auto_preview_geoms so subsequent slider drags use the cheap
        prefix-slice path. Generation-guarded against a new run / teardown."""
        state = self._auto_preview_build_state
        if state is None or state.get("gen") != self._auto_preview_build_gen:
            return  # superseded or torn down
        from qgis.PyQt.QtCore import QTimer
        import time as _t

        deadline = _t.monotonic() + _AUTO_PUMP_BUDGET_S
        pending = state["pending"]
        out = state["out"]
        pixel_size = state["pixel_size"]
        # Same tolerance as the committed reslice (AUTO_REVIEW_SIMPLIFY_DEFAULT
        # px): the drag preview used a 5x coarser 2.0 px simplify, so shapes
        # visibly "sharpened" on slider release and small objects could collapse
        # during the drag. Matching tolerances makes drag == release.
        tol = _AUTO_REVIEW_SIMPLIFY_DEFAULT * pixel_size if pixel_size > 0 else 0.0
        from ...core.layer_conventions import to_multipolygon
        while pending:
            det_idx, geom, score = pending.pop()
            if geom is not None and not geom.isEmpty():
                s = geom.simplify(tol) if tol > 0 else geom
                if s is None or s.isEmpty():
                    s = geom
                # Coerce to MultiPolygon ONCE here so a confidence-drag tick's
                # incremental adds never pay the per-geom deep copy again.
                mp = to_multipolygon(s)
                out.append((mp if mp is not None and not mp.isEmpty() else s,
                            score, det_idx))
            if _t.monotonic() >= deadline:
                break
        if pending:
            QTimer.singleShot(0, self._step_build_preview_cache)
            return
        out.sort(key=lambda gs: gs[1], reverse=True)
        self._auto_preview_geoms = out
        self._auto_preview_build_state = None

    def _snap_review_start_confidence(self, conf: float) -> float:
        """Snap a starting review cutoff to the review slider's 5% grid, clamped
        to [0, review-max]. The slider can only rest on 5% steps, so snapping the
        stored _auto_confidence (and thus the histogram cutoff and the seeded
        widgets) to the same grid keeps all three on one value. 0 is preserved,
        so a run whose best score is under 5% still opens showing every object."""
        from ..dock.styles import _REVIEW_CONF_STEP, _REVIEW_CONF_MAX
        try:
            c = float(conf)
        except (TypeError, ValueError):
            return conf
        pct = int(round(c * 100.0 / _REVIEW_CONF_STEP)) * _REVIEW_CONF_STEP
        pct = max(0, min(_REVIEW_CONF_MAX, pct))
        return pct / 100.0

    def _review_start_confidence(self) -> float:
        """Starting review cutoff. The default (0.30) unless either

        - it would hide EVERY found object: then the highest 5% step in
          [0, 0.30] that shows at least one, or
        - the run's own score distribution says the hidden cohort is the same
          physical population as the confidently detected one (dense scenes
          where most true objects score under the default): then the adaptive
          cutoff from core/review_defaults.adaptive_review_confidence, always
          strictly below the default (classes whose scores behave keep the
          old start exactly).

        Headless/MCP runs keep the seeded value (stable API contract)."""
        default = self._effective_confidence_default()
        if self._auto_headless_run or not self._auto_objects:
            return default
        scores = [s for (_g, s, _a) in self._auto_objects]
        best = max(scores)
        if best < default:
            # Highest 5% step <= best. A best below 5% starts at 0 so the review
            # never opens on "0 shown" for a run that DID find something (the old
            # floor of 5 broke exactly the guarantee this function exists for).
            import math
            step = max(0, int(math.floor(best * 100 / 5.0)) * 5)
            return step / 100.0
        from ...core.review_defaults import adaptive_review_confidence
        adaptive = adaptive_review_confidence(
            [(s, a) for (_g, s, a) in self._auto_objects],
            default=default,
            merge_separate=self._auto_merge_separate,
        )
        return adaptive if adaptive is not None else default

    def _complete_auto_finalize(self, visible: list, tiles_succeeded: int,
                                scores: list | None = None,
                                ids: list | None = None) -> None:
        """Finish a run: ``visible`` is the current VISIBLE object set (already
        whole-object filtered + shape-refined). ``scores`` / ``ids`` are the
        parallel per-object score and canonical-identity lists (same order as
        ``visible``) that feed the review heatmap and the stable Random colours.
        Headless exports it straight to a layer; interactive enters the post-run
        review."""
        crs = QgsCoordinateReferenceSystem(self._auto_crs_authid or "EPSG:4326")

        # Determine source layer name.
        source_layer = self._get_active_raster_layer()
        source_layer_name = ""
        try:
            if source_layer is not None:
                source_layer_name = source_layer.name()
        except (RuntimeError, AttributeError):
            pass

        # Get prompt text for output filename.
        prompt_text = ""
        try:
            if self.dock_widget:
                prompt_text = self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            pass

        if self._auto_headless_run:
            # MCP/headless: no human to review, so export the default-filtered
            # visible set (keeps the stable API contract: a saved layer name).
            if not visible:
                self._record_auto_zero_result(tiles_succeeded)
                return
            exported_layer_name = self._export_auto_detections(
                visible, crs, source_layer_name, prompt_text, scores=scores)
            self._remove_auto_selection_layer()
            result = {
                "status": "completed",
                "instances": len(visible),
                "tiles_processed": tiles_succeeded,
                "layer_name": exported_layer_name,
            }
            prior = self._last_auto_result
            if isinstance(prior, dict) and prior.get("status") == "credits_exhausted":
                # The exhausted terminal recorded its result before finalize:
                # keep BOTH facts (the saved layer and the quota) so the
                # headless caller neither orphans paid work nor misses the
                # exhaustion signal.
                result["status"] = "credits_exhausted"
                result["credits_remaining"] = prior.get("credits_remaining", 0)
            self._last_auto_result = result
            QgsMessageLog.logMessage(
                "Auto detection: exported {} polygon(s)".format(len(visible)),
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )
            return

        # Interactive: enter the review even if the fresh confidence cutoff hides
        # every object (as long as the run DID find objects), so the confidence
        # slider can bring them back. Only a truly empty run is a zero result.
        if not self._auto_objects:
            self._record_auto_zero_result(tiles_succeeded)
            return

        pixel_size = self._auto_refine_pixel_size()
        self._auto_review = {
            "geoms": visible,
            "scores": scores or [],
            "ids": ids or [],
            "crs": crs,
            "source_layer_name": source_layer_name,
            "prompt": prompt_text,
            "pixel_size": pixel_size,
            # Provenance stamp for the incremental review push: the visible set
            # is the cache-normalized refine output under this shape key.
            "stamp": ("acc", (self._auto_reslice_cache or {}).get("key")),
        }
        # Record result so MCP get_status stays consistent.
        self._last_auto_result = {
            "status": "completed",
            "instances": len(visible),
            "tiles_processed": tiles_succeeded,
            "layer_name": None,
        }
        # A tile still at the cloud model's per-inference mask ceiling AFTER the
        # re-split ladder means the scene was denser than the run could resolve.
        # Internal log only (this runs once per run); the review UI stays quiet.
        if getattr(self, "_auto_dense_tiles", 0):
            QgsMessageLog.logMessage(
                "Auto detection: {} tile(s) still at the max masks per "
                "inference after re-split; denser tiling (higher Detail) may "
                "catch more objects.".format(self._auto_dense_tiles),
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )
        # Seed the refine controls (confidence + size + shape) with this run's
        # smart preset so a value left over from a previous review cannot
        # wrongly filter this fresh result. The visible set was already computed
        # with the same preset (_fresh_review_params), so the widgets and the
        # geometry agree.
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_review_active(
                    True, count=len(visible), preset=self._auto_review_preset())
                # Drop the blue zone fill so the detections are not washed out by
                # the overlay during review (the outline stays for context).
                self._set_zone_band_fill_visible(False)
                # Exemplar-only count-vs-map override: one muted line naming the
                # auto grouping + a link to re-group the other way. Shown only
                # when the run was exemplar-only and its fragments were retained.
                if (getattr(self, "_auto_is_exemplar_only", False) and self._auto_raw_fragments is not None):
                    self.dock_widget.set_merge_override(
                        "separate" if self._auto_merge_separate else "map")
                else:
                    self.dock_widget.set_merge_override(None)
                # Exemplar nudge: bottom-heavy, no-example runs.
                self._maybe_show_exemplar_nudge(prompt_text, scores or [])
            except (RuntimeError, AttributeError):
                pass
        # Swap the live-run blue outline for the review's default Random colours
        # (one colour per object, seeded fresh for every NEW review; the combo
        # follows signal-free so control and renderer agree).
        self._seed_review_display_mode()
        if self._auto_selection_layer is not None:
            self._apply_review_display_mode(self._auto_selection_layer)
        self._refresh_auto_review_preview()
        self._review_conf_moves = 0  # fresh confidence-move counter for this review
        # Telemetry: the run's terminal event (completed only; cancel/exhaust
        # already emitted theirs) plus the review-opened funnel step.
        try:
            from ...core import telemetry
            ctx = self._auto_run_ctx or {}
            total = ctx.get("total", tiles_succeeded)
            instances_found = len(self._auto_objects)
            visible_n = len(visible)
            start_pct = int(round((self._auto_confidence or 0.0) * 100))
            if self._auto_tel_stop_reason in (None, "completed"):
                telemetry.track_auto_detect_completed(
                    run_id=self._auto_run_id or "",
                    duration_ms=self._auto_duration_ms(),
                    tiles_done=tiles_succeeded,
                    tiles_failed=max(0, total - tiles_succeeded),
                    instances_found=instances_found,
                    instances_visible_at_default=visible_n,
                    zero_at_default=visible_n == 0,
                    stop_reason="completed",
                    warming_ms=self._auto_warming_wait_ms(),
                    merge_mode_final="separate" if self._auto_merge_separate else "map",
                    merge_override_used=bool(getattr(self, "_auto_merge_override_used", False)),
                )
            telemetry.track_review_opened(
                run_id=self._auto_run_id or "",
                instances_found=instances_found,
                visible_at_start=visible_n,
                start_confidence=start_pct,
                auto_lowered=start_pct < int(round(self._effective_confidence_default() * 100)),
            )
        except Exception:
            pass  # nosec B110
        QgsMessageLog.logMessage(
            "Auto detection: {} object(s) ready for review".format(len(visible)),
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )

    def _on_auto_zero_detections(self, tiles_billed: int = 0) -> None:
        """A run that found nothing shows one quiet status line at the top of the
        prompt step ("No detection in this zone.") and leaves the user on the
        normal step-2 screen, where the Detect + Exit row already lets them
        adjust and run again or leave. No heavy guidance box. The terminal
        handlers already called set_auto_run_active(False), so the step-2
        controls are back; this just posts the message. Headless: just log."""
        # Back on the prompt step: restore the zone fill the run start dropped
        # (live/review visual parity).
        self._set_zone_band_fill_visible(True)
        # Distinguish a genuinely empty zone from a run where every tile failed
        # to reach the service (offline / server down / timeouts). Telling an
        # offline user their zone is empty is misleading, so surface the real
        # cause instead.
        network_failed = (
            tiles_billed == 0 and (getattr(self, "_auto_skipped_tiles", 0) or getattr(self, "_auto_timeout_tiles", 0))
        )
        if network_failed:
            msg = tr("Could not reach the service. Check your connection and try again.")
            log_msg = "Auto detection: run ended with no successful tiles (network/timeout)"
        else:
            # Actionable, not just declarative: a paid run that found nothing
            # is the worst moment of the flow, and the two levers that rescue
            # it (a more concrete object word, a drawn example) are both one
            # step away on this very screen.
            msg = tr(
                "No detection in this zone. Try a more specific object word, "
                "or draw an example of one (best for unusual objects).")
            log_msg = "Auto detection: run completed with zero detections"
        if self.dock_widget and not self._auto_headless_run:
            try:
                self.dock_widget.set_auto_status(
                    "error" if network_failed else "info", msg)
                if not network_failed:
                    # The two rescue levers named in the status, as one-click
                    # chips (draw an example / prefill a stronger word). A
                    # full example store would make the draw chip a silent
                    # no-op on click, so it is not offered then (the synonym
                    # chip alone is not worth the row: a full store means
                    # examples were already the strategy).
                    try:
                        _can_add_example = not self._auto_exemplar_store.is_full_for(1)
                    except (RuntimeError, AttributeError):
                        _can_add_example = False
                    if _can_add_example:
                        self.dock_widget.show_auto_zero_assist(
                            self.dock_widget.auto_prompt_input.text().strip())
                self._set_zone_badge_enabled(True)
            except (RuntimeError, AttributeError):
                pass
        QgsMessageLog.logMessage(
            log_msg, "AI Segmentation", level=Qgis.MessageLevel.Info,
        )

    def _on_auto_zero_assist_clicked(self, kind: str, to_prompt: str) -> None:
        """A zero-result rescue chip: record it, then perform the lever.

        'draw_example' arms the one-shot example draw (all the usual no-op
        guards in _on_add_exemplar_requested apply); 'synonym' prefills the
        suggested word - setText fires textChanged, which hides the chips and
        debounce-commits the new prompt, so the detail re-seed happens on the
        normal path."""
        from_prompt = ""
        try:
            from_prompt = self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            pass
        try:
            from ...core import telemetry
            telemetry.track_zero_assist_clicked(kind, from_prompt, to_prompt)
        except Exception:
            pass  # nosec B110
        try:
            self.dock_widget.hide_auto_zero_assist()
        except (RuntimeError, AttributeError):
            pass
        if kind == "synonym" and to_prompt:
            try:
                self.dock_widget.auto_prompt_input.setText(to_prompt)
                self.dock_widget.auto_prompt_input.setFocus()
            except (RuntimeError, AttributeError):
                pass
        elif kind == "draw_example":
            self._on_add_exemplar_requested(1)
