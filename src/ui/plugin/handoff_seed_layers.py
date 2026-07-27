"""Refine-in-Manual handoff: the two in-memory seed layers (pending, kept)
and the spatial index behind hover and click.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from qgis.core import (
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsProject,
    QgsVectorLayer,
)

from ...core.i18n import tr
from ...core.qt_compat import symbol_fill_color_property
from ..canvas_palette import KEPT_STROKE
from .shared import (
    _FIELD_TYPE_DOUBLE,
    _FIELD_TYPE_INT,
    _add_features_with_ids,
    _apply_fast_render,
    _clear_all_features,
    _notify_provider_write,
)


class HandoffSeedLayersMixin:
    """The handoff seed layers, their renderer and their hit index."""

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
                f"MultiPolygon?crs={crs_authid}",
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
            self._apply_handoff_display_renderer(layer, kept=(kind == "kept"))
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

    def _apply_handoff_display_renderer(self, layer, kept: bool) -> None:
        """Colour a handoff seed layer to MATCH the review's current display
        mode, so opening an object to edit with the AI keeps the Normal /
        Confidence / Distinct look the user picked instead of always forcing
        Distinct. Only the fill follows the mode; the kept layer keeps its bold
        green validated ring, the pending layer its dark hairline. Best-effort:
        a render failure must never break the handoff."""
        try:
            from qgis.core import (
                QgsFillSymbol,
                QgsProperty,
                QgsSingleSymbolRenderer,
                QgsStyle,
            )
            stroke = KEPT_STROKE if kept else None
            symbol = QgsFillSymbol.createSimple({
                "color": "120,120,120,120",
                "outline_color": (
                    f"{stroke.red()},{stroke.green()},{stroke.blue()},255"
                    if stroke is not None else "20,20,20,200"),
                "outline_width": "0.6" if kept else "0.2",
            })
            sl = symbol.symbolLayer(0)
            mode = getattr(self, "_auto_display_mode", "random")
            if mode == "confidence":
                ramp = ("Viridis"
                        if QgsStyle.defaultStyle().colorRamp("Viridis")
                        else "Spectral")
                expr = f"ramp_color('{ramp}', coalesce(\"score\", 0))"
            elif mode == "normal":
                from ...core.output_store import committed_color_for_prompt
                raw_prompt = (getattr(self, "_auto_run_ctx", None) or {}).get("prompt")
                raw_prompt = raw_prompt or (getattr(self, "_auto_review", None) or {}).get("prompt")
                prompt = str(raw_prompt or "").strip()
                c = committed_color_for_prompt(prompt)
                expr = f"color_rgba({c.red()}, {c.green()}, {c.blue()}, 205)"
            else:
                # Distinct (random) and outline keep the per-object stable hue,
                # so an object holds its colour across the whole handoff.
                expr = 'color_hsla((coalesce("det_id", $id) * 67) % 360, 78, 55, 205)'
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
            # Not truncate(): this layer carries a spatial index, and a
            # truncate leaves its entries behind (see _clear_all_features).
            _clear_all_features(pr)
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
            _notify_provider_write(layer)
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
            _notify_provider_write(layer)
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
            _notify_provider_write(layer)
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
