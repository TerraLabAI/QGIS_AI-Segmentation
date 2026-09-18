









from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsGeometry,
    QgsMessageLog,
)

from ...core.prompt_manager import FrozenCropSession





_ZOOM_IN_THRESH = 0.85
_ZOOM_OUT_THRESH = 1.15


class ManualCropsGeometryMixin:


    def _transform_to_raster_crs(self, point):








        if self._canvas_to_raster_xform is not None:
            try:
                return self._canvas_to_raster_xform.transform(point)
            except Exception:  # noqa: BLE001
                return None
        return point

    def _transform_geometry_to_canvas_crs(self, geometry) -> bool:






        if self._raster_to_canvas_xform is None:
            return True
        from ...core.qt_compat import geometry_op_succeeded
        return geometry_op_succeeded(
            geometry.transform(self._raster_to_canvas_xform))

    def _transform_to_canvas_crs(self, point):






        if self._raster_to_canvas_xform is not None:
            try:
                return self._raster_to_canvas_xform.transform(point)
            except Exception:  # noqa: BLE001
                return None
        return point

    def _is_point_in_raster_extent(self, point):

        if point is None:
            return False
        if not self._is_layer_valid():
            return False
        try:


            ext = self._current_layer.extent()
            in_x = ext.xMinimum() <= point.x() <= ext.xMaximum()
            in_y = ext.yMinimum() <= point.y() <= ext.yMaximum()
            return in_x and in_y
        except RuntimeError:
            return False

    def _check_crop_status(self, point):








        if self._current_crop_info is None:
            return "no_crop"
        bounds = self._current_crop_info["bounds"]
        in_x = bounds[0] <= point.x() <= bounds[2]
        in_y = bounds[1] <= point.y() <= bounds[3]
        if not (in_x and in_y):
            return "outside_bounds"






        has_active_points = self._active_crop_points_positive or self._active_crop_points_negative
        if not has_active_points:





            from ...core.server_dials import dial_in_range

            zoom_in_thresh = dial_in_range(
                "tuning.manual.crop_zoom_in_thresh", _ZOOM_IN_THRESH, 0.5, 0.99)
            zoom_out_thresh = dial_in_range(
                "tuning.manual.crop_zoom_out_thresh", _ZOOM_OUT_THRESH, 1.01, 2.0)

            if self._is_online_layer:
                canvas = self.iface.mapCanvas()
                current_canvas_mupp = canvas.mapUnitsPerPixel()
                if self._current_crop_canvas_mupp and current_canvas_mupp > 0:
                    ratio = current_canvas_mupp / self._current_crop_canvas_mupp
                    if ratio < zoom_in_thresh or ratio > zoom_out_thresh:
                        if self._crop_resolution_would_change():
                            return "zoom_changed"
            else:
                if self._current_crop_canvas_mupp is not None:
                    canvas = self.iface.mapCanvas()
                    current_mupp = canvas.mapUnitsPerPixel()
                    if current_mupp > 0:
                        ratio = current_mupp / self._current_crop_canvas_mupp
                        if ratio < zoom_in_thresh or ratio > zoom_out_thresh:
                            if self._crop_resolution_would_change():
                                return "zoom_changed"

        return "ok"

    def _snapshot_mask_state(self) -> dict:






        return {
            "mask": self.current_mask.copy() if self.current_mask is not None else None,
            "score": self.current_score,
            "transform_info": self.current_transform_info,
            "low_res_mask": (self.current_low_res_mask.copy()
                             if self.current_low_res_mask is not None else None),



            "display_polygon": (QgsGeometry(self._unfrozen_display_polygon)
                                if self._unfrozen_display_polygon is not None
                                else None),
        }

    def _restore_mask_state(self, state: dict) -> None:

        self.current_mask = state["mask"]
        self.current_score = state["score"]
        self.current_transform_info = state["transform_info"]
        self.current_low_res_mask = state.get("low_res_mask")
        self._unfrozen_display_polygon = state.get("display_polygon")

    def _invalidate_history_logits(self) -> None:





        for state in self._mask_state_history:
            state["low_res_mask"] = None

    @staticmethod
    def _resize_nearest(arr, target_h, target_w):

        import numpy as np
        src_h, src_w = arr.shape
        row_idx = (np.arange(target_h) * src_h / target_h).astype(int)
        col_idx = (np.arange(target_w) * src_w / target_w).astype(int)
        np.clip(row_idx, 0, src_h - 1, out=row_idx)
        np.clip(col_idx, 0, src_w - 1, out=col_idx)
        return arr[row_idx[:, None], col_idx[None, :]]

    def _seed_side(self, fallback: int = 256) -> int:









        side = getattr(getattr(self, "predictor", None), "low_res_side", None)
        try:
            side = int(side)
        except (TypeError, ValueError):
            return fallback
        return side if side > 0 else fallback

    def _binary_mask_to_logits(self, mask, target: int | None = None):











        import numpy as np
        side = self._seed_side() if target is None else int(target)



        m = np.asarray(self._resize_nearest(np.asarray(mask), side, side),
                       dtype=np.float32)
        logits = (m * 2.0 - 1.0) * 6.0
        return logits[None, :, :]

    def _build_mask_input_from_previous(
        self, old_mask, old_bounds, old_shape, new_bounds, new_shape
    ):







        import numpy as np

        old_minx, old_miny, old_maxx, old_maxy = old_bounds
        new_minx, new_miny, new_maxx, new_maxy = new_bounds


        ovlp_minx = max(old_minx, new_minx)
        ovlp_miny = max(old_miny, new_miny)
        ovlp_maxx = min(old_maxx, new_maxx)
        ovlp_maxy = min(old_maxy, new_maxy)
        if ovlp_minx >= ovlp_maxx or ovlp_miny >= ovlp_maxy:
            return None

        old_h, old_w = old_shape
        new_h, new_w = new_shape

        def geo_to_pixel(gx, gy, bminx, bminy, bmaxx, bmaxy, pw, ph):
            col = (gx - bminx) / (bmaxx - bminx) * pw
            row = (bmaxy - gy) / (bmaxy - bminy) * ph
            return int(round(col)), int(round(row))


        o_c0, o_r0 = geo_to_pixel(
            ovlp_minx, ovlp_maxy, old_minx, old_miny, old_maxx, old_maxy,
            old_w, old_h)
        o_c1, o_r1 = geo_to_pixel(
            ovlp_maxx, ovlp_miny, old_minx, old_miny, old_maxx, old_maxy,
            old_w, old_h)
        o_r0 = max(0, min(o_r0, old_h))
        o_r1 = max(0, min(o_r1, old_h))
        o_c0 = max(0, min(o_c0, old_w))
        o_c1 = max(0, min(o_c1, old_w))
        if o_r0 >= o_r1 or o_c0 >= o_c1:
            return None

        patch = old_mask[o_r0:o_r1, o_c0:o_c1]


        n_c0, n_r0 = geo_to_pixel(
            ovlp_minx, ovlp_maxy, new_minx, new_miny, new_maxx, new_maxy,
            new_w, new_h)
        n_c1, n_r1 = geo_to_pixel(
            ovlp_maxx, ovlp_miny, new_minx, new_miny, new_maxx, new_maxy,
            new_w, new_h)
        n_r0 = max(0, min(n_r0, new_h))
        n_r1 = max(0, min(n_r1, new_h))
        n_c0 = max(0, min(n_c0, new_w))
        n_c1 = max(0, min(n_c1, new_w))
        target_h = n_r1 - n_r0
        target_w = n_c1 - n_c0
        if target_h < 1 or target_w < 1:
            return None

        resized_patch = self._resize_nearest(patch, target_h, target_w)


        new_mask = np.zeros((new_h, new_w), dtype=np.float32)
        new_mask[n_r0:n_r1, n_c0:n_c1] = resized_patch



        return self._binary_mask_to_logits(new_mask)

    def _compute_auto_min_area(self):











        scale = self._current_crop_scale_factor
        if scale is None or scale <= 0:

            if self._current_crop_actual_mupp and self._current_crop_canvas_mupp and self._current_crop_canvas_mupp > 0:
                scale = max(1.0, self._current_crop_actual_mupp / self._current_crop_canvas_mupp * 2.0)
            else:
                scale = 1.0

        return max(100, int(200 * max(0.6, scale) ** 0.3))

    def _visible_raster_under_click(self, center_point) -> str:








        if center_point is None:
            return ""
        try:
            from qgis.core import (
                QgsCoordinateTransform,
                QgsProject,
                QgsRasterLayer,
            )

            project = QgsProject.instance()
            root = project.layerTreeRoot()
            current = self._current_layer
            for layer in root.layerOrder():
                if layer is current or not isinstance(layer, QgsRasterLayer):
                    continue
                node = root.findLayer(layer.id())
                if node is None or not node.isVisible():
                    continue
                point = center_point
                try:
                    if current is not None and layer.crs() != current.crs():
                        point = QgsCoordinateTransform(
                            current.crs(), layer.crs(),
                            project.transformContext()).transform(center_point)
                except Exception:  # noqa: BLE001
                    point = None
                if point is None:
                    continue
                extent = layer.extent()
                if extent is None or extent.isEmpty():
                    continue
                if extent.contains(point):
                    return layer.name()
        except Exception:  # noqa: BLE001  # nosec B110
            return ""
        return ""

    def _freeze_active_crop(self, crop_info_override=None):










        if self.current_mask is None or self.current_transform_info is None:




            base = self._unfrozen_display_polygon
            if base is not None and not base.isEmpty():
                self._frozen_sessions.append(
                    FrozenCropSession(polygon=QgsGeometry(base)))
                self._unfrozen_display_polygon = None
            return
        try:




            combined = self._refined_active_mask_geometry()
            if combined is not None and not combined.isEmpty():
                session = FrozenCropSession(
                    polygon=combined,
                    points_positive=list(self._active_crop_points_positive),
                    points_negative=list(self._active_crop_points_negative),
                    crop_info=crop_info_override if crop_info_override is not None else self._current_crop_info,
                )
                self._frozen_sessions.append(session)
                QgsMessageLog.logMessage(
                    f"Froze crop session #{len(self._frozen_sessions)} "
                    f"with {len(session.points_positive) + len(session.points_negative)} points",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to freeze active crop: {str(e)}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)


        self._active_crop_points_positive = []
        self._active_crop_points_negative = []
        self._mask_state_history = []
