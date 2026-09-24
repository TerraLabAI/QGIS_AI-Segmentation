



















from __future__ import annotations

from qgis.core import QgsGeometry


def _mask_pixel_units(mask, info) -> float:


    try:
        minx, maxx, miny, maxy = (float(v) for v in info["bbox"])
        rows, cols = int(mask.shape[0]), int(mask.shape[1])
        if rows <= 0 or cols <= 0:
            return 0.0
        return max(0.0, min(abs(maxx - minx) / cols, abs(maxy - miny) / rows))
    except Exception:  # noqa: BLE001
        return 0.0


class ManualShapeCacheMixin:


    def _manual_shape_cache_reset(self) -> None:


        self._mask_preview_memo = None
        self._manual_outline_memo = None

    def _manual_outline_smooth_px(self) -> float:


        try:
            from ...core.detection_policy import manual_outline_smooth_px

            cloud = bool(self._manual_cloud_predictor_active())
            return max(0.0, float(manual_outline_smooth_px(cloud)))
        except Exception:  # noqa: BLE001
            return 0.0

    def _manual_mask_stage_key(self, fill_holes, max_hole_px, simplify_tol):






        return (self._refine_expand, fill_holes, self._refine_min_area,
                max_hole_px, simplify_tol,
                (self._manual_outline_smooth_px(),
                 self._manual_outline_smooth_size_fraction()))

    def _manual_outline_smooth_size_fraction(self) -> float:

        try:
            from ...core.detection_policy import manual_outline_smooth_size_fraction

            return max(0.0, float(manual_outline_smooth_size_fraction()))
        except Exception:  # noqa: BLE001
            return 0.0

    def _manual_mask_polygons(self, fill_holes, max_hole_px, simplify_tol,
                              mask=None, info=None):










        from ...core.polygon_exporter import apply_mask_refinement, mask_to_polygons

        if mask is None:
            mask = self.current_mask
        if info is None:
            info = self.current_transform_info
        is_active = (mask is self.current_mask
                     and info is self.current_transform_info)
        key = self._manual_mask_stage_key(fill_holes, max_hole_px, simplify_tol)
        smooth_px, smooth_size_fraction = key[-1]
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
        if smooth_px > 0:





            from ...core.semiauto_outline import (
                CORNER_CUT_PX,
                cut_corners,
                smooth_mask,
            )
            cleaned = smooth_mask(cleaned, smooth_px, smooth_size_fraction)
        geometries = mask_to_polygons(cleaned, info, simplify_tol)
        if smooth_px > 0 and geometries:
            cut = CORNER_CUT_PX * _mask_pixel_units(cleaned, info)
            if cut > 0:
                geometries = [cut_corners(g, cut) for g in geometries]
        if is_active:


            self._mask_preview_memo = (mask, info, key, cleaned, geometries)
        else:



            self._ghost_mask_stage = (mask, dict(info), key, cleaned, geometries)
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

    def _adopt_ghost_shape(self, ghost_mask, ghost_outline) -> bool:














        try:
            import numpy as np
            from qgis.core import QgsGeometry as _Geometry

            from ...core.config_cache import get_config
            from ...core.detection_policy import manual_simplify_multiple_of_px

            mask = self.current_mask
            info = self.current_transform_info
            stage = getattr(self, "_ghost_mask_stage", None)
            if (mask is None or info is None or stage is None or ghost_outline is None
                    or stage[0] is not ghost_mask or ghost_outline[0] is not ghost_mask):
                return False
            ghost_info = stage[1]
            if (tuple(ghost_info.get("bbox") or ()) != tuple(info.get("bbox") or ())
                    or tuple(ghost_info.get("img_shape") or ()) != tuple(info.get("img_shape") or ())
                    or ghost_info.get("crs") != info.get("crs")):
                return False
            if mask.shape != ghost_mask.shape or not np.array_equal(mask, ghost_mask):
                return False
            multiple = manual_simplify_multiple_of_px()
            tolerance = (multiple * self._crop_pixel_size_units(info)
                         if multiple > 0 else 0.0)
            fill_holes, max_hole_px = self._fill_holes_arguments()
            key = self._manual_mask_stage_key(fill_holes, max_hole_px, tolerance)
            config = get_config()
            outline_key = self._manual_outline_key(key)
            _mask, ghost_config, ghost_settings, outline = ghost_outline
            if stage[2] != key or ghost_config is not config or ghost_settings != outline_key:
                return False
            self._mask_preview_memo = (mask, info, key, stage[3], stage[4])
            self._manual_outline_memo = (
                mask, info, config, outline_key,
                _Geometry(outline) if outline is not None else None)
            return True
        except Exception:  # noqa: BLE001  # nosec B110
            return False

    def _warm_outline_chain(self) -> None:








        if getattr(self, "_headless", False):
            return
        try:
            import numpy as np


            from ...core import polygon_exporter, progressive_merge  # noqa: F401

            if self._manual_cloud_predictor_active():
                from ...core.cloud_detection import decode_rle_to_mask  # noqa: F401
            side = 32
            mask = np.zeros((side, side), dtype=bool)
            mask[8:24, 10:22] = True
            info = {"bbox": (0.0, float(side), 0.0, float(side)),
                    "img_shape": (side, side), "crs": None}
            self._manual_outline_for(mask, info)
            self._ghost_mask_stage = None
        except Exception:  # noqa: BLE001  # nosec B110
            pass

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
        mask_stage = self._manual_mask_stage_key(fill_holes, max_hole_px, tolerance)

        config = get_config()
        key = self._manual_outline_key(mask_stage)
        memo = getattr(self, "_manual_outline_memo", None)
        if (memo is not None and memo[0] is mask and memo[1] is info
                and memo[2] is config and memo[3] == key):
            return QgsGeometry(memo[4]) if memo[4] is not None else None

        outline = self._manual_outline_for(mask, info)

        self._manual_outline_memo = (mask, info, config, key, outline)
        return QgsGeometry(outline) if outline is not None else None
