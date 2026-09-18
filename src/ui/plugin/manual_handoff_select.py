







from __future__ import annotations

from qgis.core import Qgis, QgsGeometry, QgsMessageLog, QgsPointXY
from qgis.gui import QgsRubberBand

from ...core.i18n import tr
from ...core.qt_compat import PolygonGeometry
from .shared import _debounce_timer


class ManualHandoffSelectMixin:










    @staticmethod
    def _entry_geom(pg):







        g = pg.get("geom_obj")
        if g is not None:
            return g
        g = QgsGeometry.fromWkt(pg.get("geometry_wkt") or "")
        pg["geom_obj"] = g
        return g

    def _saved_index_of(self, entry):


        for i, pg in enumerate(self.saved_polygons):
            if pg is entry:
                return i
        return None

    def _hit_test_saved_entry(self, raster_pt):












        pt = QgsGeometry.fromPointXY(raster_pt)







        prefer_kept = bool(self._refine_handoff_active)

        def _ranked(cands):



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





        for pg in ranked:
            if not self._correct_focus_blocks_det_id(pg.get("det_id")):
                return pg
        return None

    def _hit_test_saved_polygon(self, raster_pt):


        entry = self._hit_test_saved_entry(raster_pt)
        return None if entry is None else self._saved_index_of(entry)








    def _selected_saved_indices(self) -> list:


        sel = getattr(self, "_handoff_selected_entries", None) or []
        if not sel:
            return []
        return [i for i, pg in enumerate(self.saved_polygons)
                if any(pg is e for e in sel)]

    def _select_saved_polygon(self, idx: int, additive: bool = False) -> None:

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

        timer = getattr(self, "_handoff_prewarm_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:
                self._handoff_prewarm_timer = None
        if getattr(self, "_handoff_selected_entries", None):
            self._handoff_selected_entries = []
            self._refresh_handoff_selection_band()
            self._notify_handoff_selection()








    def _schedule_handoff_crop_prewarm(self) -> None:






        if not self._refine_handoff_active or self.dock_widget is None:
            return
        timer = getattr(self, "_handoff_prewarm_timer", None)
        sel = getattr(self, "_handoff_selected_entries", None) or []
        if len(sel) != 1 or self._is_online_layer or self._headless:
            if timer is not None:
                try:
                    timer.stop()
                except RuntimeError:
                    self._handoff_prewarm_timer = None
            return
        from ...core.server_dials import dial_in_range
        debounce_ms = int(dial_in_range(
            "tuning.manual.handoff_prewarm_debounce_ms", 400, 100, 2000))
        _debounce_timer(self, "_handoff_prewarm_timer", self.dock_widget,
                        debounce_ms, self._maybe_prewarm_selected_crop)

    def _handoff_crop_spec_for(self, geom, anchor_pt) -> tuple:








        from ...core.crop_window import crop_window_for_object
        bb = geom.boundingBox()
        bounds = (bb.xMinimum(), bb.yMinimum(), bb.xMaximum(), bb.yMaximum())
        held = getattr(self, "_encoded_crop_window", None)
        if self._is_online_layer:






            try:
                _canvas_mupp, raster_mupp = self._online_crop_mupp_now(None)
            except Exception:  # noqa: BLE001
                raster_mupp = 0.0
            return crop_window_for_object(
                bounds, 1.0, held_window=held,
                min_scale=raster_mupp,
                max_scale=float("inf"))
        return crop_window_for_object(
            bounds, self._get_native_pixel_size(), held_window=held)

    def _maybe_prewarm_selected_crop(self) -> None:




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
            return
        QgsMessageLog.logMessage(
            "Refine handoff: prewarming selected detection's crop",
            "AI Segmentation", level=Qgis.MessageLevel.Info)


        self._extract_and_encode_crop(
            QgsPointXY(cx, cy), mupp_override=scale, show_busy=False, quiet=True)

    def _notify_handoff_selection(self) -> None:

        sel = getattr(self, "_handoff_selected_entries", None) or []
        if len(sel) == 1:
            self._sync_correct_panel_to_handoff_entry(sel[0])

    def _sync_correct_panel_to_handoff_entry(self, entry) -> None:







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


            if display.isMultipart():
                displays.extend(display.asGeometryCollection())
            else:
                displays.append(display)
        if displays:



            band.setToGeometry(QgsGeometry.collectGeometry(displays), None)
        band.show()

    def _set_handoff_hover(self, idx) -> None:

        entry = self.saved_polygons[idx] if idx is not None else None
        self._set_handoff_hover_entry(entry)

    def _set_handoff_hover_entry(self, entry) -> None:



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




        return bool(self._encoding_in_progress) and bool(
            getattr(self, "_encode_cursor_set", True))

    def _on_handoff_cursor_moved(self, point) -> None:







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
            return
        self._set_handoff_hover_entry(self._hit_test_saved_entry(raster_pt))

    def _click_was_additive(self) -> bool:


        tool = self.map_tool
        if tool is None:
            return False
        from qgis.PyQt.QtCore import Qt
        mods = getattr(tool, "last_click_modifiers", Qt.KeyboardModifier.NoModifier)
        return bool(mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier))

    def _on_canvas_double_click(self, point) -> None:



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
