











from __future__ import annotations

from qgis.core import QgsGeometry



_REFINE_HISTORY_DEPTH = 30


class HandoffShapeMixin:


    def _handoff_ground_mupp(self) -> float:








        layer = getattr(self, "_handoff_source_layer", None)
        if layer is None:
            return 1.0
        try:
            v = float(layer.rasterUnitsPerPixelX())
            if v > 0:
                return v
        except (RuntimeError, AttributeError):
            pass
        try:
            canvas = self.iface.mapCanvas()
            if canvas.mapSettings().destinationCrs() == layer.crs():
                v = float(canvas.mapUnitsPerPixel())
                if v > 0:
                    return v
        except (RuntimeError, AttributeError):
            pass
        return 1.0

    def _current_refine_tuple(self) -> tuple:


        return (
            float(self._refine_simplify or 0.0),
            self._refine_smooth > 0,
            int(self._refine_expand),
            bool(self._refine_fill_holes),
            bool(self._refine_ortho),
            float(getattr(self, "_refine_min_size_m2", 0.0) or 0.0),
            0.0,
            float(getattr(self, "_refine_fill_holes_max_m2", 0.0) or 0.0),
            float(getattr(self, "_refine_clean", 0.0) or 0.0),
            int(getattr(self, "_refine_points_pct", 100) or 100),
        )

    @staticmethod
    def _entry_refine_tuple(entry: dict) -> tuple:


        return (
            float(entry.get("refine_simplify") or 0.0),
            (entry.get("refine_smooth") or 0) > 0,
            int(entry.get("refine_expand") or 0),
            bool(entry.get("refine_fill_holes")),
            bool(entry.get("refine_ortho")),
            float(entry.get("refine_min_size_m2") or 0.0),
            0.0,
            float(entry.get("refine_fill_holes_max_m2") or 0.0),
            float(entry.get("refine_clean") or 0.0),
            int(entry.get("refine_points_pct") or 100),
        )

    def _seed_refine_panel_from_entry(self, entry: dict) -> None:



        from ...core.review_defaults import REFINE_POINTS_PCT_DEFAULT
        self._refine_simplify = float(entry.get("refine_simplify") or 0.0)
        self._refine_points_pct = int(
            entry.get("refine_points_pct") or REFINE_POINTS_PCT_DEFAULT)
        self._refine_smooth = int(entry.get("refine_smooth") or 0)
        self._refine_clean = float(entry.get("refine_clean") or 0.0)
        self._refine_expand = int(entry.get("refine_expand") or 0)
        self._refine_fill_holes = bool(entry.get("refine_fill_holes"))
        self._refine_fill_holes_max_m2 = float(
            entry.get("refine_fill_holes_max_m2") or 0.0)
        self._refine_ortho = bool(entry.get("refine_ortho"))
        self._refine_min_size_m2 = float(entry.get("refine_min_size_m2") or 0.0)
        self._refine_max_size_m2 = 0.0
        if self.dock_widget:
            try:
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

    def _ring_area_cutoff(self, geom, ground_m2: float) -> float:









        if not ground_m2 or ground_m2 <= 0 or geom is None or geom.isEmpty():
            return 0.0
        try:
            from ...core.hole_size import ring_area_arg, units2_per_m2
            from ...core.layer_conventions import make_area_measurer


            layer = (getattr(self, "_handoff_source_layer", None) or getattr(self, "_current_layer", None))
            if layer is None or not layer.crs().isValid():
                return float(ground_m2)
            ground = float(make_area_measurer(layer.crs()).measureArea(geom))
            cutoff = ring_area_arg(
                ground_m2, units2_per_m2(float(geom.area()), ground))
            return max(0.0, cutoff)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return float(ground_m2)

    def _apply_handoff_refine_settings(self) -> bool:





        if not (self._refine_handoff_active or self._is_refining_saved_object):
            return False
        if self.current_mask is not None:
            return False
        if self._refine_edit_session_active():
            self._reshape_open_edit()
        return True

    def _handoff_regularize_kwargs(self, geom, mupp: float) -> dict:






        try:
            from qgis.core import QgsPointXY

            from ...core.detection_policy import (
                destair_tolerance_m,
                regularize_settings,
                regularize_tolerance_m,
            )
            from ...core.layer_conventions import (
                ground_unit_aspect,
                make_area_measurer,
            )
            layer = (getattr(self, "_handoff_source_layer", None) or getattr(self, "_current_layer", None))
            factor = 1.0
            aspect = 1.0
            if layer is not None and layer.crs().isValid():
                crs = layer.crs()
                geographic = bool(crs.isGeographic())
                centre = geom.boundingBox().center()
                step = 0.001 if geographic else 1.0
                metres = float(make_area_measurer(crs).measureLine(
                    QgsPointXY(centre.x(), centre.y()),
                    QgsPointXY(centre.x() + step, centre.y())))
                if metres > 0:
                    factor = metres / step
                aspect = ground_unit_aspect(crs, centre.x(), centre.y())
            bbox = geom.boundingBox()
            span_units = min(bbox.width(), bbox.height())
            pixel_units = float(mupp)
            reg_tol_m = regularize_tolerance_m(
                pixel_units * factor, span_units * factor)
            destair_m = destair_tolerance_m(pixel_units * factor)
            s = regularize_settings()
            return {
                "regularize_tol": reg_tol_m / factor,
                "destair_tol": destair_m / factor,
                "allow_diagonal": bool(s["allow_diagonal"]),
                "allow_circles": bool(s["allow_circles"]),
                "min_keep_iou": float(s["min_keep_iou"]),
                "diagonal_reduction": float(s["diagonal_reduction"]),
                "circle_threshold": float(s["circle_threshold"]),


                "multi_direction": bool(s["multi_direction"]),
                "multi_max_groups": int(s["multi_max_groups"]),
                "multi_min_separation_deg": float(
                    s["multi_min_separation_deg"]),



                "unit_aspect": aspect,
            }
        except Exception:  # noqa: BLE001
            return {}

    def _handoff_points_kwargs(self, geom) -> dict:










        try:
            from ...core.detection_policy import vertex_budget_settings
            from ...core.live_refine import points_dial_fraction

            keep_fraction = points_dial_fraction(
                {"points_pct": getattr(self, "_refine_points_pct", 100)})
            if keep_fraction <= 0.0:
                return {}
            s = vertex_budget_settings()
            centre = geom.boundingBox().center()
            factor = self._manual_metres_per_unit(centre.x(), centre.y())





            if factor is None or factor <= 0:
                return {}
            return {
                "vertex_keep_fraction": keep_fraction,
                "vertex_spacing": 0.0,
                "vertex_min": int(s["min_vertices"]),
                "vertex_max_deviation": float(s["max_deviation_m"]) / factor,
                "vertex_max_deviation_fraction": float(
                    s["max_deviation_fraction"]),
                "vertex_dial_max_cap_fraction": float(
                    s["dial_max_cap_fraction"]),
            }
        except Exception:  # noqa: BLE001
            return {}

    def _reshape_open_edit(self) -> None:



        from ...core.detection_policy import regularize_envelope
        from ...core.polygon_exporter import shape_polygon_geometry
        base = self._unfrozen_display_polygon
        if base is None or base.isEmpty():
            return
        cur = self._current_refine_tuple()
        if cur == getattr(self, "_refine_edit_last_applied", None):
            return
        pristine = getattr(self, "_refine_edit_pristine", None)
        if pristine is None or pristine.isEmpty():
            pristine = QgsGeometry(base)
            self._refine_edit_pristine = pristine
        info = self._current_crop_info
        if info is not None:
            minx, miny, maxx, maxy = info["bounds"]
            w = max(info["img_shape"][1], 1)
            mupp = (maxx - minx) / w
        else:
            mupp = self._handoff_ground_mupp()
        ortho_kwargs = (
            self._handoff_regularize_kwargs(pristine, mupp) if cur[4] else {})
        try:
            shaped = shape_polygon_geometry(
                pristine, mupp,
                simplify_px=cur[0], smooth=cur[1], expand_px=cur[2],
                fill_holes=cur[3], ortho=cur[4],
                fill_holes_max_area=self._ring_area_cutoff(pristine, cur[7]),
                open_dist=cur[8] * mupp,
                envelope=regularize_envelope(),
                **self._handoff_points_kwargs(pristine),
                **ortho_kwargs)
        except Exception as exc:  # noqa: BLE001




            from qgis.core import Qgis, QgsMessageLog
            QgsMessageLog.logMessage(
                f"Refine handoff: reshape failed ({type(exc).__name__})",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return
        if shaped is None or shaped.isEmpty():
            return


        filtered = self._filter_geometry_parts_by_size(shaped)
        if filtered is not None and not filtered.isEmpty():
            shaped = filtered
        history = getattr(self, "_refine_geom_history", None)
        if history is None:
            history = []
            self._refine_geom_history = history
        history.append(base)
        from ...core.server_dials import dial_in_range

        depth = dial_in_range(
            "tuning.manual.refine_history_depth", _REFINE_HISTORY_DEPTH, 5, 100)
        del history[:-depth]
        self._refine_edit_last_applied = cur
        self._unfrozen_display_polygon = shaped
        self._update_mask_visualization()
