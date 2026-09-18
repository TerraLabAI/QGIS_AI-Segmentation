






from __future__ import annotations

from qgis.core import Qgis, QgsGeometry, QgsMessageLog
from qgis.gui import QgsRubberBand

from ...core.i18n import tr
from ...core.qt_compat import PolygonGeometry
from ..canvas_palette import KEPT_FILL, KEPT_STROKE
from .shared import pixel_grid_crs


class ManualWorkflowSaveMixin:





    def _on_save_polygon(self):


        has_active = self.current_mask is not None and self.current_transform_info is not None
        if not has_active and not self._frozen_sessions and self._unfrozen_display_polygon is None:
            return






        origin = self._active_refine_origin_entry or {}
        origin_id = origin.get("det_id")
        billing_id = (int(origin_id) if origin_id is not None
                      else self._next_handoff_det_id())
        if self._manual_save_refused_for_credits(billing_id):
            return






        if self._encoding_in_progress:
            self._drop_inflight_crop_for_gesture()

        self._ensure_polygon_rubberband_sync()



        all_geoms = [s.polygon for s in self._frozen_sessions]
        if not has_active and self._unfrozen_display_polygon is not None:
            all_geoms.append(self._unfrozen_display_polygon)

        if has_active:







            active_combined = self._refined_active_mask_geometry()
            if active_combined is not None and not active_combined.isEmpty():
                all_geoms.append(active_combined)

        if all_geoms:
            combined = QgsGeometry.unaryUnion(all_geoms)
        else:
            combined = None

        if combined is None or combined.isEmpty():






            self._say_manual_save_found_nothing()
            return






        from .manual_save_alignment import align_manual_saved_shape
        combined = align_manual_saved_shape(self, combined)





        self.saved_polygons.append({
            "det_id": billing_id,
            "score": origin.get("score"),
            "manual_touched": self._refine_handoff_active,
            "geometry_wkt": combined.asWkt(),


            "geom_obj": combined,
            "transform_info": self.current_transform_info.copy() if self.current_transform_info else None,


            "raw_mask": (self.current_mask.copy()
                         if self.current_mask is not None and not self._frozen_sessions else None),
            "points_positive": list(self.prompts.positive_points),
            "points_negative": list(self.prompts.negative_points),
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

            "refine_max_size_m2": 0.0,




            "validated": not self._refine_handoff_active,
        })

        if self._refine_handoff_active:



            self.saved_rubber_bands.append(None)
            if not self._handoff_add_entry_feature(self.saved_polygons[-1]):
                self._rebuild_handoff_layers()
        else:
            saved_rb = QgsRubberBand(self.iface.mapCanvas(), PolygonGeometry)
            saved_rb.setColor(KEPT_STROKE)
            saved_rb.setFillColor(KEPT_FILL)
            saved_rb.setWidth(2)

            display_geom = QgsGeometry(combined)
            self._transform_geometry_to_canvas_crs(display_geom)
            saved_rb.setToGeometry(display_geom, None)
            self.saved_rubber_bands.append(saved_rb)

        QgsMessageLog.logMessage(
            f"Saved mask #{len(self.saved_polygons)}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))




        self._charge_manual_saved_object(self.saved_polygons[-1]["det_id"])


        try:
            import time as _time

            from ...core.telemetry_session_events import track_segmentation_run
            start_ts = getattr(self, "_segmentation_start_ts", None)
            duration_ms = int((_time.time() - start_ts) * 1000) if start_ts else None
            track_segmentation_run(success=True, duration_ms=duration_ms)
            self._segmentation_start_ts = None

            self._manual_saves_session = getattr(self, "_manual_saves_session", 0) + 1
            if getattr(self, "_manual_session_t0", None) is None:
                self._manual_session_t0 = _time.time()
        except Exception:
            pass  # nosec B110






        self._is_refining_saved_object = False
        self._active_refine_origin_entry = None
        self._refine_geom_history = []



        ledger = getattr(self, "_manual_credit_ledger", None)
        if ledger is not None:
            ledger.start_next_object()
        self.prompts.clear()
        self._mask_state_history = []
        self._frozen_sessions = []
        self._unfrozen_display_polygon = None
        self._active_crop_points_positive = []
        self._active_crop_points_negative = []
        if self.map_tool:
            self.map_tool.clear_markers()
        self._clear_mask_visualization()
        self.current_mask = None
        self.current_score = 0.0
        self.current_low_res_mask = None
        self.dock_widget.set_point_count(0, 0)

        refresh = getattr(self, "_refresh_ai_add_keep_button", None)
        if refresh is not None:
            refresh()



    def _say_manual_save_found_nothing(self) -> None:









        if getattr(self, "_refine_add_mode_active", False):
            return
        try:
            from ...core.server_dials import dial_in_range



            duration = dial_in_range("tuning.manual.add_keep_failed_notice_s", 6, 4, 10)
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("That shape was not added. Adjust it with a click and try "
                   "again."),
                level=Qgis.MessageLevel.Warning, duration=duration)
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _manual_saved_crs_definition(self) -> str:






        if self._is_non_georeferenced_mode:
            return pixel_grid_crs().toWkt()
        for entry in self.saved_polygons:
            info = entry.get("transform_info") or {}
            value = info.get("crs")
            if isinstance(value, str) and value.strip():
                return value.strip()


        live = (self.current_transform_info or {}).get("crs")
        if isinstance(live, str) and live.strip():
            return live.strip()
        try:
            if self._is_layer_valid() and self._current_layer.crs().isValid():
                crs = self._current_layer.crs()
                return crs.authid() or crs.toWkt()
        except RuntimeError:
            pass
        return ""

    def _live_manual_shape_geometry(self):




        try:
            parts = [s.polygon for s in self._frozen_sessions
                     if s.polygon is not None and not s.polygon.isEmpty()]
            active = None
            if self.current_mask is not None and self.current_transform_info is not None:
                active = self._refined_active_mask_geometry()
            if active is None and self._unfrozen_display_polygon is not None:
                active = self._unfrozen_display_polygon
            if active is not None and not active.isEmpty():
                parts.append(active)
            if not parts:
                return None
            combined = parts[0] if len(parts) == 1 else QgsGeometry.unaryUnion(parts)
            if combined is None or combined.isEmpty():
                return None
            return combined
        except Exception:  # noqa: BLE001
            return None

    def _autosave_manual_saved_polygons(self, include_live: bool = False) -> None:















        try:
            if not self.saved_polygons and not include_live:
                return


            if self._refine_handoff_active:
                return
            import time as _time

            from ...core import run_autosave

            saved = []
            for index, entry in enumerate(self.saved_polygons):

                try:
                    geom = entry.get("geom_obj")
                    if geom is None:
                        geom = QgsGeometry.fromWkt(entry.get("geometry_wkt") or "")
                    if geom is None or geom.isEmpty():
                        continue
                    det_id = entry.get("det_id")
                    score = entry.get("score")
                    saved.append((
                        int(det_id) if det_id is not None else index,
                        geom,


                        float(score) if score is not None else None,
                    ))
                except (TypeError, ValueError, AttributeError, RuntimeError):
                    continue
            if include_live:
                live = self._live_manual_shape_geometry()
                if live is not None:
                    saved.append((len(self.saved_polygons), live, None))
            if not saved:
                return
            try:
                source_layer = self._current_layer if self._is_layer_valid() else None
            except RuntimeError:
                source_layer = None
            info = run_autosave.write_autosave(
                saved, self._manual_saved_crs_definition(), "",
                f"manual-{int(_time.time() * 1000)}", source_layer=source_layer)
            if not info:
                return
            run_autosave.record_pending(info)
            QgsMessageLog.logMessage(
                "Semi-Auto: {n} saved polygon(s) written to {path}".format(
                    n=info.get("count", 0), path=info.get("path", "")),
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )
        except Exception:  # noqa: BLE001
            try:
                QgsMessageLog.logMessage(
                    "Semi-Auto: the session autosave failed",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
            except Exception:  # nosec B110
                pass
