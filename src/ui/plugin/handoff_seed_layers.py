






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
from ..canvas_palette import KEPT_STROKE, OUTLINE_MODE_STROKE_STR
from .shared import (
    _FIELD_TYPE_DOUBLE,
    _FIELD_TYPE_INT,
    _add_features_with_ids,
    _apply_fast_render,
    _clear_all_features,
    _notify_provider_write,
)


class HandoffSeedLayersMixin:




    def _create_handoff_layer(self, crs_authid: str, kind: str):








        try:
            layer = QgsVectorLayer(
                f"MultiPolygon?crs={crs_authid}",
                tr("Refine seeds"), "memory")
            if not layer.isValid():
                return None



            pr = layer.dataProvider()
            pr.addAttributes([
                QgsField("score", _FIELD_TYPE_DOUBLE),
                QgsField("det_id", _FIELD_TYPE_INT),
            ])
            layer.updateFields()
            self._apply_handoff_display_renderer(layer, kept=(kind == "kept"))
            _apply_fast_render(layer)


            from ...core.output_store import mark_temp_layer
            mark_temp_layer(layer)
            QgsProject.instance().addMapLayer(layer, False)
            QgsProject.instance().layerTreeRoot().insertLayer(0, layer)
            return layer
        except (RuntimeError, AttributeError, ImportError):
            return None

    def _ensure_handoff_layers(self, crs_authid: str) -> None:

        if self._handoff_pending_layer is None:
            self._handoff_pending_layer = self._create_handoff_layer(
                crs_authid, "pending")
        if self._handoff_kept_layer is None:
            self._handoff_kept_layer = self._create_handoff_layer(
                crs_authid, "kept")

    def _apply_handoff_display_renderer(self, layer, kept: bool) -> None:








        try:
            from qgis.core import (
                QgsFillSymbol,
                QgsProperty,
                QgsSingleSymbolRenderer,
                QgsStyle,
            )
            stroke = KEPT_STROKE if kept else None
            mode = getattr(self, "_auto_display_mode", "random")
            if mode == "outline":


                self._apply_handoff_outline_renderer(layer, stroke)
                self._apply_correct_focus_dim(layer)
                return
            symbol = QgsFillSymbol.createSimple({
                "color": "120,120,120,120",
                "outline_color": (
                    f"{stroke.red()},{stroke.green()},{stroke.blue()},255"
                    if stroke is not None else "20,20,20,200"),
                "outline_width": "0.6" if kept else "0.2",
            })
            sl = symbol.symbolLayer(0)
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





                from .auto_review_display import random_mode_fill_expression
                expr = random_mode_fill_expression()
            prop_key = symbol_fill_color_property()
            sl.setDataDefinedProperty(prop_key, QgsProperty.fromExpression(expr))
            symbol.setOpacity(0.75)
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            layer.triggerRepaint()



            self._apply_correct_focus_dim(layer)
        except (RuntimeError, AttributeError, ImportError):
            pass

    def _apply_handoff_outline_renderer(self, layer, stroke) -> None:





        try:
            from qgis.core import QgsFillSymbol, QgsSingleSymbolRenderer
            symbol = QgsFillSymbol.createSimple({
                "style": "no",
                "outline_color": (
                    f"{stroke.red()},{stroke.green()},{stroke.blue()},255"
                    if stroke is not None else OUTLINE_MODE_STROKE_STR),
                "outline_width": "0.6" if stroke is not None else "0.5",
                "outline_style": "solid",
            })
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            layer.triggerRepaint()
        except (RuntimeError, AttributeError, ImportError):
            pass

    def _refresh_handoff_display_renderers(self) -> None:





        for attr, kept in (("_handoff_pending_layer", False),
                           ("_handoff_kept_layer", True)):
            layer = getattr(self, attr, None)
            if layer is None:
                continue
            try:
                if not layer.isValid():
                    continue
            except (RuntimeError, AttributeError):
                continue
            self._apply_handoff_display_renderer(layer, kept=kept)

    def _push_geoms_to_layer(self, layer, rows: list) -> None:





        if layer is None:
            return
        try:
            from ...core.layer_conventions import to_multipolygon
            pr = layer.dataProvider()


            _clear_all_features(pr)
            feats = []
            kept_flag = layer is self._handoff_kept_layer
            for pg, g, score, det_id in rows:

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



                ok, added = _add_features_with_ids(pr, [f for _pg, f in feats])
                if ok and len(added) == len(feats):
                    for (pg, _f), out in zip(feats, added):
                        pfid = out.id()
                        if pfid is not None and pfid >= 0:
                            pg["_hfid"] = pfid
                            pg["_hkept"] = kept_flag



            _notify_provider_write(layer)
            layer.triggerRepaint()
        except (RuntimeError, AttributeError):
            pass

    def _rebuild_handoff_layers(self) -> None:













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

        tok = getattr(self, "_handoff_hit_tok_seq", 0) + 1
        self._handoff_hit_tok_seq = tok
        return tok

    def _handoff_hit_insert(self, pg) -> None:

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





        if not self._refine_handoff_active:
            return True


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
                return True
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




        if not self._refine_handoff_active:
            return True
        self._handoff_hit_remove(pg)
        pfid = pg.pop("_hfid", None)
        kept = pg.pop("_hkept", None)
        g = self._entry_geom(pg)
        if g is None or g.isEmpty():
            return True
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

        for attr in ("_handoff_pending_layer", "_handoff_kept_layer"):
            layer = getattr(self, attr, None)
            if layer is not None:
                try:
                    QgsProject.instance().removeMapLayer(layer.id())
                except (RuntimeError, AttributeError):
                    pass
                setattr(self, attr, None)
