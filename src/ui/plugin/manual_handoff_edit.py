







from __future__ import annotations

from qgis.core import Qgis, QgsGeometry, QgsMessageLog, QgsPointXY
from qgis.gui import QgsRubberBand

from ...core.qt_compat import PolygonGeometry
from ..canvas_palette import PENDING_FILL, PENDING_STROKE


class ManualHandoffEditMixin:





    def _delete_selected_saved_polygons(self) -> bool:



        idxs = self._selected_saved_indices()
        if not idxs:
            return False
        unit = []
        inc_ok = True
        for i in sorted(idxs, reverse=True):
            pg = self.saved_polygons.pop(i)
            if i < len(self.saved_rubber_bands):
                self._safe_remove_rubber_band(self.saved_rubber_bands.pop(i))


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
            except (RuntimeError, AttributeError):
                pass
        QgsMessageLog.logMessage(
            f"{len(unit)} object(s) deleted. Ctrl+Z restores them.",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        return True

    def _edit_selected_saved_polygon(self) -> bool:


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


        self._handoff_selected_entries = []
        self._refresh_handoff_selection_band()
        self._set_handoff_hover(None)
        self._notify_handoff_selection()
        self._activate_saved_polygon_for_refine(idx, raster_pt, label=label)

    def _push_deleted_unit(self, unit: list) -> None:

        from ...core.server_dials import dial_in_range

        stack = getattr(self, "_deleted_objects_stack", None)
        if stack is None:
            stack = []
            self._deleted_objects_stack = stack
        stack.append(unit)

        cap = int(dial_in_range("tuning.manual.delete_undo_stack_cap", 25, 5, 200))
        del stack[:-cap]

    def _next_handoff_det_id(self) -> int:






        seq = getattr(self, "_handoff_det_id_seq", None)
        if seq is None:
            seq = 100000
        taken = {fid for fid in (getattr(self, "_auto_object_fids", None) or ())
                 if isinstance(fid, int)}
        while seq in taken:
            seq += 1
        self._handoff_det_id_seq = seq + 1
        return seq

    def _handoff_entry_identities(self, entries: list):









        items = [(g, i, s) for g, i, s in entries
                 if g is not None and not g.isEmpty()]
        geoms = [g for g, _i, _s in items]
        ids = [int(i) if i is not None else self._next_handoff_det_id()
               for _g, i, _s in items]
        scores = [float(s) if s is not None else 1.0 for _g, _i, s in items]
        return geoms, ids, scores

    def _clear_active_mask_without_saving(self) -> None:


        self.current_mask = None
        self.current_score = 0.0
        self.current_transform_info = None
        self.current_low_res_mask = None





        ledger = getattr(self, "_manual_credit_ledger", None)
        if ledger is not None:
            try:
                ledger.start_next_object()
            except (RuntimeError, AttributeError):
                pass

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
            except (RuntimeError, AttributeError):
                pass

    def _on_delete_active_object(self) -> None:



        if not (self._refine_handoff_active or self._is_refining_saved_object):
            return


        should_delete_selected = self.current_mask is None and self._active_refine_origin_entry is None
        should_delete_selected = should_delete_selected and not self._active_crop_points_positive
        if should_delete_selected:
            self._delete_selected_saved_polygons()
            return



        origin = self._active_refine_origin_entry
        if origin is not None:
            backup = dict(origin)
            base = self._harvest_open_edit_geometry()
            if base is not None and not base.isEmpty():
                backup["geometry_wkt"] = base.asWkt()


                backup["geom_obj"] = QgsGeometry(base)

                backup.pop("shape_base_wkt", None)



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


        self._discard_pending_manual_click()
        self._clear_active_mask_without_saving()
        self._is_refining_saved_object = False
        self._active_refine_origin_entry = None
        if self.dock_widget:
            try:
                self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
            except (RuntimeError, AttributeError):
                pass
        QgsMessageLog.logMessage(
            "Object deleted. Ctrl+Z restores it.",
            "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _restore_deleted_object(self) -> bool:



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
            entry["validated"] = False
            self.saved_polygons.append(entry)
            if self._refine_handoff_active:


                self.saved_rubber_bands.append(None)
                inc_ok = self._handoff_add_entry_feature(entry) and inc_ok
            else:

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
        return True

    def _activate_saved_polygon_for_refine(self, idx, raster_pt, label: int = 1) -> None:












        entry = self.saved_polygons[idx]
        geom = QgsGeometry.fromWkt(entry.get("geometry_wkt") or "")
        if geom is None or geom.isEmpty():
            return

        popped = self.saved_polygons.pop(idx)
        if idx < len(self.saved_rubber_bands):
            self._safe_remove_rubber_band(self.saved_rubber_bands.pop(idx))



        if not self._handoff_remove_entry_feature(popped):
            self._rebuild_handoff_layers()



        self._is_refining_saved_object = True
        self._active_refine_origin_entry = dict(popped)



        self._seed_refine_panel_from_entry(popped)
        self._refine_edit_pristine = QgsGeometry(geom)
        self._refine_edit_last_applied = self._entry_refine_tuple(popped)
        if self.dock_widget:
            try:
                self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
            except (RuntimeError, AttributeError):
                pass






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


            self._set_ai_session_armed_line(loading=True)

    def _refine_edit_session_active(self) -> bool:






        if not self._is_refining_saved_object:
            return False
        if self.current_mask is not None or self._frozen_sessions:
            return True
        base = self._unfrozen_display_polygon
        if base is None or base.isEmpty():
            self._is_refining_saved_object = False
            self._active_refine_origin_entry = None
            self._refine_geom_history = []
            return False
        return True

    def _close_active_edit_to_pending(self) -> None:







        if not self._is_refining_saved_object:
            return
        base = self._harvest_open_edit_geometry()
        origin = self._active_refine_origin_entry or {}
        appended = None
        if base is not None and not base.isEmpty():
            entry = dict(origin)
            entry["geometry_wkt"] = base.asWkt()
            entry["geom_obj"] = QgsGeometry(base)



            entry.pop("shape_base_wkt", None)
            entry["validated"] = False



            touched = bool(getattr(self, "_refine_geom_history", None)
                           or any(self.prompts.point_count))
            if touched:
                entry["manual_touched"] = True
            self.saved_polygons.append(entry)
            appended = entry
            if touched and not entry.get("run_counted"):





                entry["run_counted"] = True
                try:
                    import time as _time

                    from ...core.telemetry_session_events import track_segmentation_run
                    start_ts = getattr(self, "_segmentation_start_ts", None)
                    duration_ms = int((_time.time() - start_ts) * 1000) if start_ts else None
                    track_segmentation_run(success=True, duration_ms=duration_ms)
                    self._segmentation_start_ts = None
                except Exception:
                    pass  # nosec B110
            if self._refine_handoff_active:

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



        self._discard_pending_manual_click()


        self._clear_active_mask_without_saving()



        if appended is not None and not self._handoff_add_entry_feature(appended):
            self._rebuild_handoff_layers()
        if self.dock_widget:
            try:
                self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))
            except (RuntimeError, AttributeError):
                pass

    def _shape_in_progress_geometry(self):










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








        try:
            from ...core.geometry_raster import rasterize_geometry_to_grid

            return rasterize_geometry_to_grid(geom, bounds, img_shape)
        except Exception:  # noqa: BLE001
            return None
