



















from __future__ import annotations

import math

from qgis.core import Qgis, QgsPointXY


class ManualCropWindowMixin:


    def _crop_resolution_would_change(self) -> bool:














        if self._untouched_shape_window_is_held():
            return False
        try:
            if self._is_online_layer:
                held = self._current_crop_actual_mupp
                if not held or held <= 0:
                    return True
                _canvas_mupp, fresh = self._online_crop_mupp_now(None)
                return round(float(fresh), 9) != round(float(held), 9)
            held = self._current_crop_scale_factor
            fresh = self._compute_initial_scale_factor()
            if held is None or fresh is None:
                return True
            return round(float(fresh), 9) != round(float(held), 9)
        except (RuntimeError, AttributeError, TypeError, ValueError, ZeroDivisionError):
            return True

    def _untouched_shape_window_is_held(self) -> bool:






        info = self._current_crop_info
        if info is None:
            return False
        try:
            from ...core.crop_window import crop_window_key

            minx, miny, maxx, maxy = info["bounds"]
            centre = QgsPointXY((minx + maxx) / 2.0, (miny + maxy) / 2.0)
            whole = self._untouched_shape_crop_window(centre)
            if whole is None:
                return False
            point, scale = whole
            asks = crop_window_key(point.x(), point.y(), scale)
            return asks == getattr(self, "_encoded_crop_window", None)
        except Exception:  # noqa: BLE001
            return False

    def _get_native_pixel_size(self):

        try:
            ext = self._current_layer.extent()
            w = self._current_layer.width()
            h = self._current_layer.height()
            if w > 0 and h > 0:
                px = (ext.xMaximum() - ext.xMinimum()) / w
                py = (ext.yMaximum() - ext.yMinimum()) / h
                return max(px, py)
        except (RuntimeError, AttributeError):
            pass
        return 0.0

    def _compute_initial_scale_factor(self):












        if self._is_online_layer:
            return None
        native_pixel_size = self._get_native_pixel_size()
        if native_pixel_size <= 0:
            return None

        canvas = self.iface.mapCanvas()
        canvas_extent = canvas.extent()

        if self._canvas_to_raster_xform is not None:
            try:
                canvas_extent = self._canvas_to_raster_xform.transformBoundingBox(
                    canvas_extent)
            except Exception:
                return None


        canvas_width_px = canvas.width()
        if canvas_width_px <= 0:
            return None
        canvas_geo_width = canvas_extent.xMaximum() - canvas_extent.xMinimum()
        canvas_mupp_raster_crs = canvas_geo_width / canvas_width_px

        from ...core.crop_window import scale_at_least_native
        from ...core.server_dials import dial_in_range

        ratio = canvas_mupp_raster_crs / native_pixel_size
        max_scale = dial_in_range("tuning.manual.max_crop_scale_factor", 8.0, 2.0, 20.0)
        return min(scale_at_least_native(ratio), max_scale)

























    def _crop_window_key_for(self, center_point, mupp_override):



        from ...core.crop_window import crop_window_key
        try:
            return crop_window_key(center_point.x(), center_point.y(),
                                   mupp_override or 1.0)
        except (AttributeError, TypeError):
            return None

    def _grid_center_for_manual_click(self, raster_pt, scale):


















        if self._is_online_layer:
            return raster_pt, scale
        from ...core.crop_window import snap_center_to_grid, window_frames_bounds
        native = self._get_native_pixel_size()
        grid_scale = scale or 1.0
        held = getattr(self, "_encoded_crop_window", None)
        reuse_held = self._current_crop_info is None and held is not None
        reuse_held = reuse_held and round(float(held[2]), 6) == round(float(grid_scale), 6)
        if reuse_held and window_frames_bounds(
                held,
                (raster_pt.x(), raster_pt.y(), raster_pt.x(), raster_pt.y()),
                native):
            return QgsPointXY(held[0], held[1]), held[2]
        cx, cy = snap_center_to_grid(
            raster_pt.x(), raster_pt.y(), grid_scale, native)
        return QgsPointXY(cx, cy), scale

    def _untouched_shape_crop_window(self, raster_pt, extra_points=()):

















        if getattr(self, "_frozen_sessions", None) or self.current_mask is not None:
            return None
        try:
            shape = self._shape_in_progress_geometry()
        except (RuntimeError, AttributeError):
            return None
        if shape is None or shape.isEmpty():
            return None
        try:
            box = shape.boundingBox()
        except (RuntimeError, AttributeError):
            return None
        xs = [box.xMinimum(), box.xMaximum(), raster_pt.x()]
        ys = [box.yMinimum(), box.yMaximum(), raster_pt.y()]
        for point in extra_points or ():
            xs.append(point[0])
            ys.append(point[1])
        return self._crop_window_for_bounds(
            (min(xs), min(ys), max(xs), max(ys)))

    def _crop_window_for_bounds(self, bounds):










        from ...core.crop_window import crop_window_for_object

        held = (getattr(self, "_encoded_crop_window", None)
                if self._current_crop_info is None else None)
        if self._is_online_layer:
            try:
                _canvas_mupp, raster_mupp = self._online_crop_mupp_now(None)
            except Exception:  # noqa: BLE001
                raster_mupp = 0.0
            cx, cy, scale = crop_window_for_object(
                bounds, 1.0, held_window=held,
                min_scale=raster_mupp,
                max_scale=float("inf"))
        else:
            cx, cy, scale = crop_window_for_object(
                bounds, self._get_native_pixel_size(), held_window=held)
        return QgsPointXY(cx, cy), scale

    def _manual_crop_window_for_points(self, points_geo):












        xs = [p[0] for p in points_geo]
        ys = [p[1] for p in points_geo]
        return self._crop_window_for_bounds(
            (min(xs), min(ys), max(xs), max(ys)))

    def _online_native_ground_per_pixel(self) -> float:



        try:
            native = self._online_native_mupp(self._current_layer)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return 0.0
        return float(native) if native and native > 0 else 0.0

    def _online_crop_step_ceiling(self) -> float:









        from qgis.core import QgsUnitTypes

        from ...core.crop_window import MAX_CROP_GROUND_WIDTH_M
        from ...core.server_dials import dial

        width_m = dial("manual.max_crop_ground_width_m", MAX_CROP_GROUND_WIDTH_M)


        metres = getattr(getattr(Qgis, "DistanceUnit", None), "Meters", None)
        if metres is None:
            metres = getattr(QgsUnitTypes, "DistanceMeters", None)
        if metres is None:
            return 0.0
        try:
            crs = self._current_layer.crs()
            per_metre = QgsUnitTypes.fromUnitToUnitFactor(metres, crs.mapUnits())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return 0.0
        if not per_metre or per_metre <= 0 or not math.isfinite(per_metre):
            return 0.0


        return (width_m * per_metre) / 1024.0

    def _online_crop_mupp_now(self, mupp_override) -> tuple:











        from ...core.crop_window import (
            ground_per_pixel_at_least_native,
            ground_per_pixel_within_ceiling,
        )

        canvas = self.iface.mapCanvas()
        canvas_mupp = canvas.mapUnitsPerPixel()



        if self._canvas_to_raster_xform is not None:
            canvas_center = canvas.center()
            cx, cy = canvas_center.x(), canvas_center.y()
            p1 = self._canvas_to_raster_xform.transform(QgsPointXY(cx, cy))
            p2 = self._canvas_to_raster_xform.transform(
                QgsPointXY(cx + canvas_mupp, cy))
            raster_mupp = math.sqrt(
                (p2.x() - p1.x()) ** 2 + (p2.y() - p1.y()) ** 2)
        else:
            raster_mupp = canvas_mupp
        step = ground_per_pixel_at_least_native(
            mupp_override or raster_mupp, self._online_native_ground_per_pixel())
        return canvas_mupp, ground_per_pixel_within_ceiling(
            step, self._online_crop_step_ceiling())

    def _online_grid_window(self, raster_pt):










        from ...core.crop_window import snap_center_to_grid
        try:
            _canvas_mupp, actual_mupp = self._online_crop_mupp_now(None)
        except Exception:  # noqa: BLE001
            return None
        if not actual_mupp or actual_mupp <= 0:
            return None
        cx, cy = snap_center_to_grid(
            raster_pt.x(), raster_pt.y(), actual_mupp, 1.0)
        return QgsPointXY(cx, cy), actual_mupp

    def _online_crop_mupp(self, mupp_override):










        canvas_mupp, actual_mupp = self._online_crop_mupp_now(mupp_override)
        self._pending_crop_zoom_baseline = (None, canvas_mupp, actual_mupp)
        return actual_mupp
