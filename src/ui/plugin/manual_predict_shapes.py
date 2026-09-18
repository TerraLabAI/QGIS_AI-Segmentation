






from __future__ import annotations

from qgis.core import Qgis, QgsGeometry, QgsMessageLog, QgsPointXY

from ...core.prompt_manager import FrozenCropSession
from ...core.review_defaults import (
    REFINE_SMOOTH_ITERATIONS,
)
from .manual_measure_cache import narrow_dimension, session_area_measurer





FILL_HOLES_CAP_UNKNOWN = object()


class ManualShapeMixin:





    @staticmethod
    def _grown_by_shape_so_far(new_mask, prior_mask, img_height, img_width):








        if prior_mask is None:
            return new_mask
        try:
            import numpy as np
            prior = prior_mask[:img_height, :img_width]
            if prior.shape != new_mask.shape:
                return new_mask
            grown = np.logical_or(new_mask.astype(bool), prior.astype(bool))
            return grown.astype(new_mask.dtype, copy=False)
        except Exception:  # noqa: BLE001
            return new_mask

    def _grown_in_one_piece(self, new_mask, prior_mask, img_height, img_width,
                            pixel_size_m, raw_answer=None):











        if prior_mask is None:
            return new_mask, False
        try:
            prior = prior_mask[:img_height, :img_width]
            if prior.shape != new_mask.shape:
                return new_mask, False
            from ...core.shape_growth import grow_shape_with_click
            return grow_shape_with_click(
                prior, new_mask, pixel_size_m, click_answer=raw_answer)
        except Exception as e:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Could not keep the shape in one piece: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return self._grown_by_shape_so_far(
                new_mask, prior_mask, img_height, img_width), False

    def _freeze_display_polygon_outside_crop(self, crop_bounds) -> None:






        base = getattr(self, "_unfrozen_display_polygon", None)
        if base is None or base.isEmpty():
            return
        try:
            from qgis.core import QgsRectangle
            minx, miny, maxx, maxy = crop_bounds
            crop = QgsGeometry.fromRect(QgsRectangle(minx, miny, maxx, maxy))
            if crop.contains(base):
                return
            outside = base.difference(crop)
            if outside is None or outside.isEmpty():
                return
            self._frozen_sessions.append(
                FrozenCropSession(polygon=QgsGeometry(outside)))
        except Exception as e:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Could not keep the part outside the crop: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)

    def _crop_pixel_size_units(self, transform_info) -> float:












        if not transform_info:
            return 0.0
        bbox = transform_info.get("bbox", [0, 1, 0, 1])
        img_shape = transform_info.get("img_shape", (1024, 1024))

        width_pixels = max(int(img_shape[1]), 1)
        bbox_width = float(bbox[1]) - float(bbox[0])
        if bbox_width == 0:
            return 0.0
        pixel_size = bbox_width / width_pixels
        height_pixels = max(int(img_shape[0]), 1)
        bbox_height = float(bbox[3]) - float(bbox[2])
        if bbox_height != 0:
            pixel_size = min(pixel_size, bbox_height / height_pixels)
        return pixel_size

    def _manual_simplify_tolerance(self, geom, transform_info) -> float:









        tolerance = self._compute_simplification_tolerance(
            transform_info, self._refine_simplify)
        if tolerance <= 0:
            return 0.0
        from ...core.review_defaults import REFINE_SIMPLIFY_MAX_NARROW_FRACTION

        if REFINE_SIMPLIFY_MAX_NARROW_FRACTION <= 0:
            return tolerance
        narrow = narrow_dimension(self, geom)
        if narrow <= 0:
            return tolerance
        return min(tolerance, REFINE_SIMPLIFY_MAX_NARROW_FRACTION * narrow)

    def _manual_vertex_deviation_cap(self, base_cap_m: float, transform_info,
                                     served_flat_m: float = 0.0,
                                     metres_per_unit: float = 1.0) -> float:



















        from ...core.review_defaults import (
            REFINE_VERTEX_DEVIATION_PIXEL_FLOOR,
            REFINE_VERTEX_MAX_DEVIATION_M,
        )

        cap = REFINE_VERTEX_MAX_DEVIATION_M
        if cap <= 0:
            return base_cap_m
        if served_flat_m > 0 and base_cap_m > 0:
            cap *= base_cap_m / served_flat_m

        if base_cap_m > 0:
            cap = min(cap, base_cap_m)




        px_units = self._crop_pixel_size_units(transform_info)
        if px_units > 0:
            cap = max(cap, REFINE_VERTEX_DEVIATION_PIXEL_FLOOR
                      * px_units * metres_per_unit)
        return cap

    def _manual_metres_per_unit(self, ref_x: float, ref_y: float):















        layer = getattr(self, "_current_layer", None)
        if layer is None:
            return None
        try:
            crs = layer.crs()
            if not crs.isValid():
                return None
            geographic = bool(crs.isGeographic())
        except Exception:  # noqa: BLE001
            return None
        try:
            measurer = session_area_measurer(self, crs)
            if measurer is None:
                return None
            step = 0.001 if geographic else 1.0
            metres = float(measurer.measureLine(
                QgsPointXY(ref_x, ref_y), QgsPointXY(ref_x + step, ref_y)))
            return metres / step if metres > 0 else None
        except Exception:  # noqa: BLE001
            return None

    def _manual_unit_aspect(self, ref_x: float, ref_y: float) -> float:




        layer = getattr(self, "_current_layer", None)
        if layer is None:
            return 1.0
        try:
            from ...core.layer_conventions import ground_unit_aspect
            return ground_unit_aspect(layer.crs(), ref_x, ref_y)
        except Exception:  # noqa: BLE001
            return 1.0

    def _manual_apply_right_angles(self, combined, transform_info, tolerance):







        from ...core.polygon_exporter import apply_right_angles
        pixel_units = self._crop_pixel_size_units(transform_info)



        try:
            from ...core.detection_policy import regularize_envelope
            _envelope = regularize_envelope()
        except Exception:  # noqa: BLE001  # nosec B110
            _envelope = None
        bbox = combined.boundingBox()
        centre = bbox.center()
        aspect = self._manual_unit_aspect(centre.x(), centre.y())



        factor = self._manual_metres_per_unit(centre.x(), centre.y())
        if factor is not None and factor > 0:
            try:
                from ...core.detection_policy import (
                    destair_tolerance_m,
                    regularize_settings,
                    regularize_tolerance_m,
                )






                span_units = min(bbox.width(), bbox.height() * aspect)
                reg_tol_m = regularize_tolerance_m(
                    pixel_units * factor, span_units * factor)
                reg_tol = reg_tol_m / factor
                destair = destair_tolerance_m(pixel_units * factor) / factor
                s = regularize_settings()
                return apply_right_angles(
                    combined,
                    destair_tol=max(0.0, destair - tolerance),
                    tolerance_m=reg_tol,
                    allow_diagonal=bool(s["allow_diagonal"]),
                    allow_circles=bool(s["allow_circles"]),
                    min_keep_iou=float(s["min_keep_iou"]),
                    diagonal_reduction=float(s["diagonal_reduction"]),
                    circle_threshold=float(s["circle_threshold"]),




                    multi_direction=bool(s["multi_direction"]),
                    multi_max_groups=int(s["multi_max_groups"]),
                    multi_min_separation_deg=float(
                        s["multi_min_separation_deg"]),
                    unit_aspect=aspect,
                    envelope=_envelope)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
        destair3 = self._compute_simplification_tolerance(transform_info, 1.5)
        return apply_right_angles(
            combined,
            destair_tol=max(0.0, destair3 - tolerance),
            tolerance_m=destair3,
            unit_aspect=aspect,
            envelope=_envelope)

    def _manual_despike_distance(self, combined, transform_info) -> float:








        try:
            from ...core.detection_policy import despike_tolerance_m
            pixel_units = self._crop_pixel_size_units(transform_info)
            centre = combined.boundingBox().center()
            factor = self._manual_metres_per_unit(centre.x(), centre.y())
            if factor is None or factor <= 0:
                return 0.0
            return despike_tolerance_m(pixel_units * factor) / factor
        except Exception:  # noqa: BLE001
            return 0.0

    def _shape_active_geometry(self, combined, transform_info):






        if combined is None or combined.isEmpty():
            return None
        if self._refine_ortho:




            from ...core.polygon_exporter import despike_thin_necks
            combined = despike_thin_necks(
                combined,
                self._manual_despike_distance(combined, transform_info),
                preserve_parts=bool(combined.isMultipart()))
            if combined is None or combined.isEmpty():
                return None






        ortho_on = bool(getattr(self, "_refine_ortho", False))


        open_px = (0.0 if ortho_on
                   else float(getattr(self, "_refine_clean", 0.0) or 0.0))
        if open_px > 0:
            open_dist = open_px * self._crop_pixel_size_units(transform_info)
            if open_dist > 0:
                try:
                    r = combined.buffer(-open_dist, 8).buffer(open_dist, 8)
                    if r is not None and not r.isEmpty():
                        combined = r
                except Exception:  # noqa: BLE001  # nosec B110
                    pass
        tolerance = self._manual_simplify_tolerance(combined, transform_info)
        if tolerance > 0:



            from ...core.detection_policy import vertex_budget_settings
            from ...core.vertex_budget import destaircase_outline
            r = destaircase_outline(
                combined, tolerance,
                2 * int(vertex_budget_settings()["min_vertices"]),
                self._crop_pixel_size_units(transform_info))
            if r is not None and not r.isEmpty():
                combined = r











        combined = self._apply_manual_vertex_budget(combined, transform_info)
        if combined is None or combined.isEmpty():
            return None
        if ortho_on:
            combined = self._manual_apply_right_angles(
                combined, transform_info, tolerance)
        if not ortho_on and self._refine_smooth > 0:






            from ...core.polygon_exporter import rounded_corner_outline
            combined = rounded_corner_outline(combined, tolerance)
        return combined if combined is not None and not combined.isEmpty() else None

    def _apply_manual_vertex_budget(self, combined, transform_info=None):



        if combined is None or combined.isEmpty():
            return combined
        try:
            from ...core.detection_policy import vertex_budget_settings
            from ...core.live_refine import points_dial_fraction
            from ...core.vertex_budget import (
                simplify_to_budget,
                smooth_budget_multiplier,
            )




            keep_fraction = points_dial_fraction(
                {"points_pct": self._refine_points_pct})
            s = vertex_budget_settings()
            spacing_m = float(s["spacing_m"])
            if spacing_m <= 0 and keep_fraction <= 0.0:
                return combined
            min_pts = int(s["min_vertices"])
            dev_m = float(s["max_deviation_m"])





            smooth_iters = (
                0 if getattr(self, "_refine_ortho", False)
                else min(int(self._refine_smooth or 0), REFINE_SMOOTH_ITERATIONS))
            if smooth_iters > 0:


                spacing_m *= smooth_budget_multiplier(
                    float(s["smooth_spacing_factor"]), smooth_iters,
                    cap=float(s["smooth_multiplier_cap"]))
                min_pts = int(s["smooth_min_vertices"])
                dev_m = float(s["smooth_max_deviation_m"])
            centre = combined.boundingBox().center()
            factor = self._manual_metres_per_unit(centre.x(), centre.y())
            if factor is None or factor <= 0:





                return combined
            r = simplify_to_budget(
                combined,
                spacing=spacing_m / factor,
                min_vertices=min_pts,
                max_deviation=self._manual_vertex_deviation_cap(
                    dev_m, transform_info, float(s["max_deviation_m"]),
                    metres_per_unit=factor) / factor,
                max_deviation_fraction=float(s["max_deviation_fraction"]),
                dial_max_cap_fraction=float(s["dial_max_cap_fraction"]),
                keep_fraction=keep_fraction,


                unit_aspect=self._manual_unit_aspect(centre.x(), centre.y()),
            )
            if r is not None and not r.isEmpty():
                return r
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return combined

    def _refined_active_mask_geometry(self):




















        outline = self._manual_active_outline()
        if outline is not None and not outline.isEmpty():
            return outline
        return self._unrefined_active_mask_geometry()

    def _unrefined_active_mask_geometry(self):











        mask = self.current_mask
        info = self.current_transform_info
        if mask is None or info is None:
            return None
        try:
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
        except Exception:  # noqa: BLE001
            return None
        return combined if combined is not None and not combined.isEmpty() else None

    def _fill_holes_pixel_cap(self, info=None):


















        max_m2 = float(getattr(self, "_refine_fill_holes_max_m2", 0.0) or 0.0)
        if max_m2 <= 0:
            return None
        if info is None:
            info = self.current_transform_info
        if not info:
            return FILL_HOLES_CAP_UNKNOWN
        try:
            from qgis.core import QgsRectangle

            from ...core.hole_size import hole_pixels
            minx, maxx, miny, maxy = (float(v) for v in info["bbox"])
            rows, cols = int(info["img_shape"][0]), int(info["img_shape"][1])
            if rows <= 0 or cols <= 0:
                return FILL_HOLES_CAP_UNKNOWN
            rect = QgsGeometry.fromRect(QgsRectangle(minx, miny, maxx, maxy))
            ground_m2 = 0.0
            measurer = session_area_measurer(self)
            if measurer is not None:
                ground_m2 = float(measurer.measureArea(rect))
            if ground_m2 <= 0:
                ground_m2 = float(rect.area())
            if ground_m2 <= 0:
                return FILL_HOLES_CAP_UNKNOWN
            cap = hole_pixels(max_m2, ground_m2 / (rows * cols))
            return FILL_HOLES_CAP_UNKNOWN if cap is None else cap
        except (RuntimeError, AttributeError, KeyError, TypeError, ValueError):
            return FILL_HOLES_CAP_UNKNOWN

    def _fill_holes_arguments(self, info=None):







        if not self._refine_fill_holes:
            return False, None
        window = info if info is not None else self.current_transform_info
        max_m2 = float(getattr(self, "_refine_fill_holes_max_m2", 0.0) or 0.0)
        memo = getattr(self, "_fill_holes_args_memo", None)
        if memo is not None and memo[0] is window and memo[1] == max_m2:
            return memo[2]
        cap = self._fill_holes_pixel_cap(info)
        answer = (False, None) if cap is FILL_HOLES_CAP_UNKNOWN else (True, cap)
        self._fill_holes_args_memo = (window, max_m2, answer)
        return answer

    def _filter_geometry_parts_by_size(self, geom):












        if (getattr(self, "_refine_handoff_active", False) and not getattr(self, "_active_refine_origin_entry", None)):
            return geom
        min_a = float(getattr(self, "_refine_min_size_m2", 0.0) or 0.0)
        if min_a <= 0 or geom is None or geom.isEmpty():
            return geom
        measurer = session_area_measurer(self)
        parts = (geom.asGeometryCollection() if geom.isMultipart()
                 else [geom])
        kept = []
        dropped = False
        for part in parts:
            if part is None or part.isEmpty():
                dropped = True
                continue
            try:
                area = (float(measurer.measureArea(part)) if measurer is not None
                        else float(part.area()))
            except (RuntimeError, AttributeError):
                area = float(part.area())
            if area < min_a:
                dropped = True
                continue
            kept.append(part)
        if not dropped:
            return geom
        if not kept:
            return QgsGeometry()
        if len(kept) == 1:
            return QgsGeometry(kept[0])
        return QgsGeometry.unaryUnion(kept)
