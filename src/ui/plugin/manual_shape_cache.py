



















from __future__ import annotations

from qgis.core import QgsGeometry


class ManualShapeCacheMixin:


    def _manual_shape_cache_reset(self) -> None:


        self._mask_preview_memo = None
        self._manual_outline_memo = None

    def _manual_mask_polygons(self, fill_holes, max_hole_px, simplify_tol,
                              mask=None, info=None):










        from ...core.polygon_exporter import apply_mask_refinement, mask_to_polygons

        if mask is None:
            mask = self.current_mask
        if info is None:
            info = self.current_transform_info
        is_active = (mask is self.current_mask
                     and info is self.current_transform_info)
        key = (self._refine_expand, fill_holes, self._refine_min_area,
               max_hole_px, simplify_tol)
        if is_active:
            memo = getattr(self, "_mask_preview_memo", None)
            if (memo is not None and memo[0] is mask and memo[1] is info
                    and memo[2] == key):
                return memo[3], memo[4]

        cleaned = mask
        if fill_holes or self._refine_expand != 0 or self._refine_min_area > 0:
            cleaned = apply_mask_refinement(
                mask,
                expand_value=self._refine_expand,
                fill_holes=fill_holes,
                min_area=self._refine_min_area,
                max_hole_px=max_hole_px,
            )







            if (self._refine_min_area > 0 and not cleaned.any()
                    and mask is not None and mask.any()):
                cleaned = apply_mask_refinement(
                    mask,
                    expand_value=self._refine_expand,
                    fill_holes=fill_holes,
                    min_area=0,
                    max_hole_px=max_hole_px,
                )
        geometries = mask_to_polygons(cleaned, info, simplify_tol)
        if is_active:


            self._mask_preview_memo = (mask, info, key, cleaned, geometries)
        return cleaned, geometries

    def _manual_outline_for(self, mask, info):









        if mask is None or info is None:
            return None
        from ...core.detection_policy import manual_simplify_multiple_of_px

        multiple = manual_simplify_multiple_of_px()
        tolerance = (multiple * self._crop_pixel_size_units(info)
                     if multiple > 0 else 0.0)
        fill_holes, max_hole_px = self._fill_holes_arguments(info)
        _cleaned, geometries = self._manual_mask_polygons(
            fill_holes, max_hole_px, tolerance, mask=mask, info=info)
        if not geometries:
            return None
        combined = QgsGeometry.unaryUnion(geometries)
        if combined is None or combined.isEmpty():
            return None
        combined = self._shape_active_geometry(combined, info)
        if combined is not None and not combined.isEmpty():
            combined = self._filter_geometry_parts_by_size(combined)
        if combined is None or combined.isEmpty():
            return None
        return combined

    def _manual_outline_key(self, mask_stage):







        return mask_stage + (
            bool(getattr(self, "_refine_ortho", False)),
            float(getattr(self, "_refine_clean", 0.0) or 0.0),
            float(getattr(self, "_refine_simplify", 0.0) or 0.0),
            int(getattr(self, "_refine_points_pct", 100) or 100),
            int(getattr(self, "_refine_smooth", 0) or 0),
            float(getattr(self, "_refine_min_size_m2", 0.0) or 0.0),
            0.0,



            bool(getattr(self, "_refine_handoff_active", False)),
            bool(getattr(self, "_active_refine_origin_entry", None)),
        )

    def _manual_active_outline(self):






        mask = self.current_mask
        info = self.current_transform_info
        if mask is None or info is None:
            return None






        from ...core.config_cache import get_config
        from ...core.detection_policy import manual_simplify_multiple_of_px



        multiple = manual_simplify_multiple_of_px()
        tolerance = (multiple * self._crop_pixel_size_units(info)
                     if multiple > 0 else 0.0)
        fill_holes, max_hole_px = self._fill_holes_arguments()
        mask_stage = (self._refine_expand, fill_holes, self._refine_min_area,
                      max_hole_px, tolerance)

        config = get_config()
        key = self._manual_outline_key(mask_stage)
        memo = getattr(self, "_manual_outline_memo", None)
        if (memo is not None and memo[0] is mask and memo[1] is info
                and memo[2] is config and memo[3] == key):
            return QgsGeometry(memo[4]) if memo[4] is not None else None

        outline = self._manual_outline_for(mask, info)

        self._manual_outline_memo = (mask, info, config, key, outline)
        return QgsGeometry(outline) if outline is not None else None
