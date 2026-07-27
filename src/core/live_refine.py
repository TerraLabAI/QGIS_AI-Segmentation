
















from __future__ import annotations

import math

from qgis.core import QgsGeometry




_UNMEASURABLE_VERTEX_BUDGET: tuple[float, int, float, float, float | None] = (
    0.0, 0, 0.0, 0.0, None)



_DESTAIR_PIXEL_MULTIPLE = 2.5


def points_dial_fraction(params: dict) -> float:






    try:
        pct = float(params.get("points_pct", 100) or 100)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if pct >= 100 else max(0.01, pct / 100.0)


class LiveRefiner:

















    def __init__(self, params: dict, pixel_size: float,
                 metres_per_unit: float, unit_aspect: float = 1.0) -> None:
        from .detection_policy import (
            despike_tolerance_m,
            destair_tolerance_m,
            regularize_envelope,
            regularize_settings,
            regularize_tolerance_m,
            vertex_budget_settings,
        )


        px = float(pixel_size) if pixel_size and pixel_size > 0 else 1.0
        factor = (float(metres_per_unit)
                  if metres_per_unit and metres_per_unit > 0 else 1.0)
        pixel_m = px * factor
        self._metres_per_unit = factor
        self._unit_aspect = (float(unit_aspect)
                             if unit_aspect and unit_aspect > 0 else 1.0)
        self._regularize_pixel_floor_m = pixel_m

        self._simplify_tol = float(params.get("simplify_px", 0)) * px
        self._expand_dist = float(params.get("expand_px", 0)) * px
        self._open_dist = float(params.get("open_px", 0)) * px



        close_m = float(params.get("close_notches_m", 0.0) or 0.0)
        self._close_dist = close_m / factor if close_m > 0 else 0.0
        self._smooth = bool(params.get("smooth", False))
        self._fill_holes = bool(params.get("fill_holes", False))
        self._keep_fraction = points_dial_fraction(params)
        self._ortho = bool(params.get("ortho", False))







        hole_m2 = float(params.get("fill_max_m2", 0.0) or 0.0)
        self._hole_cutoff_units2 = (
            hole_m2 / (factor * factor * self._unit_aspect) if hole_m2 > 0 else 0.0)











        self._destair_tol = _DESTAIR_PIXEL_MULTIPLE * px
        self._despike_tol = 0.0
        self._regularize = False
        self._regularize_tolerance_uncapped_m = 0.0
        self._regularize_max_object_fraction = 0.0
        self._allow_diagonal = True
        self._allow_circles = False
        self._regularize_min_iou = 0.0
        self._diagonal_reduction: float | None = None
        self._circle_threshold: float | None = None
        self._multi_direction = False
        self._multi_max_groups: int | None = None
        self._multi_min_separation_deg: float | None = None
        if self._ortho:
            try:
                self._destair_tol = destair_tolerance_m(pixel_m) / factor
            except Exception:  # noqa: BLE001
                self._destair_tol = _DESTAIR_PIXEL_MULTIPLE * px
            try:
                self._despike_tol = despike_tolerance_m(pixel_m) / factor
            except Exception:  # noqa: BLE001
                self._despike_tol = 0.0
            try:
                settings = regularize_settings()
                self._regularize = True






                self._regularize_tolerance_uncapped_m = regularize_tolerance_m(
                    pixel_m, 0.0)
                self._regularize_max_object_fraction = float(
                    settings["max_object_fraction"])
                self._allow_diagonal = bool(settings["allow_diagonal"])
                self._allow_circles = bool(settings["allow_circles"])
                self._regularize_min_iou = float(settings["min_keep_iou"])
                self._diagonal_reduction = float(settings["diagonal_reduction"])
                self._circle_threshold = float(settings["circle_threshold"])



                self._multi_direction = bool(settings["multi_direction"])
                self._multi_max_groups = int(settings["multi_max_groups"])
                self._multi_min_separation_deg = float(
                    settings["multi_min_separation_deg"])
            except Exception:  # noqa: BLE001
                self._regularize = False









        self._vertex_budget: tuple[float, int, float, float, float | None] = (
            _UNMEASURABLE_VERTEX_BUDGET)
        try:
            budget = vertex_budget_settings()
            spacing_m = float(params.get("vertex_spacing_m", 0.0) or 0.0)
            min_points = int(budget["min_vertices"])
            deviation_m = float(budget["max_deviation_m"])
            if params.get("smooth"):




                from .vertex_budget import smooth_budget_multiplier

                spacing_m *= smooth_budget_multiplier(
                    float(budget["smooth_spacing_factor"]), 1,
                    float(budget["smooth_multiplier_cap"]))
                min_points = int(budget["smooth_min_vertices"])
                deviation_m = float(budget["smooth_max_deviation_m"])
            self._vertex_budget = (
                spacing_m / factor if spacing_m > 0 else 0.0,
                min_points,
                deviation_m / factor,
                float(budget["max_deviation_fraction"]),
                float(budget["dial_max_cap_fraction"]),
            )
        except Exception:  # noqa: BLE001
            self._vertex_budget = _UNMEASURABLE_VERTEX_BUDGET




        self._envelope = regularize_envelope()





        from .polygon_exporter import apply_geometry_refinement

        self._refine_entry = apply_geometry_refinement




        self._smooth_settings: dict | None = None
        if self._smooth:
            try:
                from .detection_policy import smooth_pass_settings

                self._smooth_settings = smooth_pass_settings()
            except Exception:  # noqa: BLE001
                self._smooth_settings = None

    def regularize_tolerance_units(self, base: QgsGeometry) -> float:







        bbox = base.boundingBox()
        span_m = min(bbox.width(), bbox.height() * self._unit_aspect) * self._metres_per_unit
        tolerance_m = self._regularize_tolerance_uncapped_m
        if math.isfinite(span_m) and span_m > 0:
            tolerance_m = min(
                tolerance_m, self._regularize_max_object_fraction * span_m)
            if self._regularize_pixel_floor_m > 0:
                tolerance_m = max(tolerance_m, self._regularize_pixel_floor_m)
        return tolerance_m / self._metres_per_unit

    def refine(self, base: QgsGeometry | None) -> QgsGeometry | None:





        apply_geometry_refinement = self._refine_entry
        measurable = base is not None and not base.isEmpty()
        regularize = self._regularize
        regularize_tol = 0.0
        if regularize:
            try:
                regularize_tol = self.regularize_tolerance_units(base)
            except Exception:  # noqa: BLE001
                regularize = False
        (vertex_spacing, vertex_min, vertex_deviation, vertex_deviation_frac,
         vertex_dial_cap) = (
            self._vertex_budget if measurable else _UNMEASURABLE_VERTEX_BUDGET)
        return apply_geometry_refinement(
            QgsGeometry(base),
            smooth_settings=self._smooth_settings,
            simplify_tol=self._simplify_tol,
            vertex_keep_fraction=self._keep_fraction,
            smooth=self._smooth,
            expand_dist=self._expand_dist,
            fill_holes=self._fill_holes,
            fill_holes_max_area=(
                self._hole_cutoff_units2 if measurable else 0.0),
            open_dist=self._open_dist,
            close_dist=self._close_dist,
            despike_m=self._despike_tol,
            vertex_spacing=vertex_spacing,
            vertex_min=vertex_min,
            vertex_max_deviation=vertex_deviation,
            vertex_max_deviation_fraction=vertex_deviation_frac,
            vertex_dial_max_cap_fraction=vertex_dial_cap,
            ortho=self._ortho,



            ortho_tol=self._destair_tol,
            regularize=regularize,
            regularize_tol=regularize_tol,
            allow_diagonal=self._allow_diagonal,
            allow_circles=self._allow_circles,
            regularize_min_iou=self._regularize_min_iou,
            diagonal_reduction=self._diagonal_reduction,
            circle_threshold=self._circle_threshold,
            multi_direction=self._multi_direction,
            multi_max_groups=self._multi_max_groups,
            multi_min_separation_deg=self._multi_min_separation_deg,
            envelope=self._envelope,
            unit_aspect=self._unit_aspect,
        )





_NORMALIZE_TOOLS: tuple | None = None


def _refine_normalize_tools() -> tuple:





    global _NORMALIZE_TOOLS
    if _NORMALIZE_TOOLS is None:
        from .layer_conventions import repair_polygon, to_multipolygon

        _NORMALIZE_TOOLS = (repair_polygon, to_multipolygon)
    return _NORMALIZE_TOOLS


def plain_outline_geom(base: QgsGeometry | None) -> QgsGeometry | None:






    repair_polygon, to_multipolygon = _refine_normalize_tools()
    try:
        g = to_multipolygon(repair_polygon(base) or QgsGeometry(base))
    except Exception:  # noqa: BLE001
        g = QgsGeometry(base)
    return g if (g is not None and not g.isEmpty()) else None


def refine_review_geom(refiner: LiveRefiner,
                       base: QgsGeometry | None) -> tuple:











    repair_polygon, to_multipolygon = _refine_normalize_tools()
    try:
        g = refiner.refine(base)
        if g is not None and not g.isEmpty():
            g = to_multipolygon(repair_polygon(g) or g)
    except Exception as exc:  # noqa: BLE001
        return plain_outline_geom(base), exc
    if g is None or g.isEmpty():
        g = plain_outline_geom(base)
    return (g if (g is not None and not g.isEmpty()) else None), None
