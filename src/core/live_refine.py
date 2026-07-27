"""The Automatic review's shape refine, with the run's dials resolved once.

A review applies the same shape controls to every object of a run: one params
dict, one detection pixel size, one CRS. The server dials behind those controls
(the footprint regularizer's settings, the de-staircase and spike-cut
distances, the point budget, the safety envelope, the hole-area scale) are
constant for the whole run, yet the per-object refine used to re-read all of
them for every object. ``LiveRefiner`` reads them once in ``__init__`` and
leaves only the object-dependent work in ``refine()``.

Only one input genuinely varies per object: the regularizer's snap tolerance,
which is capped at a share of the object's own narrow dimension.

Nothing here touches Qt, QgsProject or the plugin instance, so a worker thread
can build a refiner and call it. The ground-metres-per-CRS-unit factor is
resolved by the caller, once for the run, and passed in.
"""
from __future__ import annotations

import math

from qgis.core import QgsGeometry

# What the point budget answers for an object it cannot measure: the spacing
# step off, no floor, no cap, and None for the dial ceiling, which reads as
# "use the engine's own value" and is not the same as a cap of zero.
_UNMEASURABLE_VERTEX_BUDGET: tuple[float, int, float, float, float | None] = (
    0.0, 0, 0.0, 0.0, None)

# De-staircase distance kept when Right angles is off, and when the server dial
# cannot be read: a small multiple of the detection pixel.
_DESTAIR_PIXEL_MULTIPLE = 2.5


def points_dial_fraction(params: dict) -> float:
    """The Points control as a fraction, or 0.0 when it asks for everything.

    100% must map to 0.0, not 1.0: 0.0 is what turns the user dial off, so an
    untouched control leaves the class density as the only budget and the
    geometry exactly as it was before this control existed.
    """
    try:
        pct = float(params.get("points_pct", 100) or 100)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if pct >= 100 else max(0.01, pct / 100.0)


class LiveRefiner:
    """The review shape controls of ONE run, ready to apply to any object.

    Build one per (params, pixel_size, metres_per_unit) and call ``refine()``
    per object. ``pixel_size`` is the run's detection pixel in CRS units, which
    is what the px-denominated UI dials scale by. ``metres_per_unit`` is ground
    metres per unit of the run CRS: every server dial is a ground distance, and
    a run often works in a CRS where one unit is well under a metre.

    Instances are read-only once built, so several threads may share one.
    """

    def __init__(self, params: dict, pixel_size: float,
                 metres_per_unit: float) -> None:
        from .detection_policy import (
            despike_tolerance_m,
            destair_tolerance_m,
            regularize_envelope,
            regularize_settings,
            regularize_tolerance_m,
            vertex_budget_settings,
        )
        # An unknown pixel size reads the px dials as CRS units, which is what
        # they meant before a detection pixel was known at all.
        px = float(pixel_size) if pixel_size and pixel_size > 0 else 1.0
        factor = (float(metres_per_unit)
                  if metres_per_unit and metres_per_unit > 0 else 1.0)
        pixel_m = px * factor
        self._metres_per_unit = factor
        self._regularize_pixel_floor_m = pixel_m

        self._simplify_tol = float(params.get("simplify_px", 0)) * px
        self._expand_dist = float(params.get("expand_px", 0)) * px
        self._open_dist = float(params.get("open_px", 0)) * px
        self._smooth = bool(params.get("smooth", False))
        self._fill_holes = bool(params.get("fill_holes", False))
        self._keep_fraction = points_dial_fraction(params)
        self._ortho = bool(params.get("ortho", False))

        # The Fill-holes threshold in CRS units squared (0 = every hole). The
        # dial is true ground m2 and one CRS unit is not one metre outside a
        # metric projection, so it is scaled by the run factor, squared because
        # it is an area.
        hole_m2 = float(params.get("fill_max_m2", 0.0) or 0.0)
        self._hole_cutoff_units2 = (
            hole_m2 / (factor * factor) if hole_m2 > 0 else 0.0)

        # Right angles runs the diagonals-aware footprint regularizer for EVERY
        # object, whatever the run's words were. What the user ticks is the
        # intent; the words only decide whether the box STARTS ticked
        # (core.review_presets). The regularizer's IoU guard reverts a shape
        # squaring would wreck, so an organic outline degrades to unchanged
        # rather than mangled.
        #
        # None of the fallbacks below changes today's behaviour: a failed
        # policy read leaves the pixel-anchored de-staircase distance in place,
        # the spike cut off, and the regularizer on the plain path.
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
            except Exception:  # noqa: BLE001 -- keep the client distance
                self._destair_tol = _DESTAIR_PIXEL_MULTIPLE * px
            try:
                self._despike_tol = despike_tolerance_m(pixel_m) / factor
            except Exception:  # noqa: BLE001 -- keep the step off on failure
                self._despike_tol = 0.0
            try:
                settings = regularize_settings()
                self._regularize = True
                # regularize_tolerance_m resolves the server dial, caps it at a
                # share of the object, then floors it at the detection pixel.
                # Only the cap depends on the object, so the resolution runs
                # here with the cap skipped (object size 0) and the two bounds
                # are re-applied per object. Applying the floor twice cannot
                # change the answer: it is a max against a constant.
                self._regularize_tolerance_uncapped_m = regularize_tolerance_m(
                    pixel_m, 0.0)
                self._regularize_max_object_fraction = float(
                    settings["max_object_fraction"])
                self._allow_diagonal = bool(settings["allow_diagonal"])
                self._allow_circles = bool(settings["allow_circles"])
                self._regularize_min_iou = float(settings["min_keep_iou"])
                self._diagonal_reduction = float(settings["diagonal_reduction"])
                self._circle_threshold = float(settings["circle_threshold"])
                # Multi-direction path: OFF unless the server turns it on. When
                # on, a many-axis building keeps each wing on its own grid
                # instead of staircasing off one global axis.
                self._multi_direction = bool(settings["multi_direction"])
                self._multi_max_groups = int(settings["multi_max_groups"])
                self._multi_min_separation_deg = float(
                    settings["multi_min_separation_deg"])
            except Exception:  # noqa: BLE001 -- fall open to the plain path
                self._regularize = False

        # The point budget, in CRS units. Squaring already sets its own sparse
        # vertex count, so the two never run together (apply_geometry_refinement
        # enforces it); resolving it anyway keeps the call site flat.
        #
        # The min-points floor and the deviation cap are ALWAYS resolved, even
        # when the class carries no spacing: the user's Points dial drives the
        # budget on its own and must keep the same boundary guard. Zeros come
        # back only when the settings cannot be read.
        self._vertex_budget: tuple[float, int, float, float, float | None] = (
            _UNMEASURABLE_VERTEX_BUDGET)
        try:
            budget = vertex_budget_settings()
            spacing_m = float(params.get("vertex_spacing_m", 0.0) or 0.0)
            min_points = int(budget["min_vertices"])
            deviation_m = float(budget["max_deviation_m"])
            if params.get("smooth"):
                # Round corners follows the budget and multiplies the points, so
                # the budget thins ahead of it or the Points promise is broken.
                # One review pass, through the shared helper so the ceiling on
                # the multiplier is the same server dial Manual obeys.
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
        except Exception:  # noqa: BLE001 -- leave the outline as traced
            self._vertex_budget = _UNMEASURABLE_VERTEX_BUDGET

        # Safety envelope + eligibility + rectangle substitution, all OFF unless
        # the server (or a local dev override) enables them; a neutral policy
        # reproduces today's geometry.
        self._envelope = regularize_envelope()

    def regularize_tolerance_units(self, base: QgsGeometry) -> float:
        """The regularizer's snap tolerance for ONE object, in RUN CRS UNITS.

        The only dial that varies per object: the run's ground tolerance capped
        at a share of the object's own narrow ground dimension, then floored
        back at the detection pixel so a small object is not squared with a
        tolerance no shape can use.
        """
        bbox = base.boundingBox()
        span_m = min(bbox.width(), bbox.height()) * self._metres_per_unit
        tolerance_m = self._regularize_tolerance_uncapped_m
        if math.isfinite(span_m) and span_m > 0:
            tolerance_m = min(
                tolerance_m, self._regularize_max_object_fraction * span_m)
            if self._regularize_pixel_floor_m > 0:
                tolerance_m = max(tolerance_m, self._regularize_pixel_floor_m)
        return tolerance_m / self._metres_per_unit

    def refine(self, base: QgsGeometry | None) -> QgsGeometry | None:
        """The run's shape controls applied to ONE object geometry.

        Returns a NEW geometry; ``base`` is never mutated. No plugin state and
        no Qt, so this is safe to call from a worker thread.
        """
        from .polygon_exporter import apply_geometry_refinement

        measurable = base is not None and not base.isEmpty()
        regularize = self._regularize
        regularize_tol = 0.0
        if regularize:
            try:
                regularize_tol = self.regularize_tolerance_units(base)
            except Exception:  # noqa: BLE001 -- fall open to the plain path
                regularize = False
        (vertex_spacing, vertex_min, vertex_deviation, vertex_deviation_frac,
         vertex_dial_cap) = (
            self._vertex_budget if measurable else _UNMEASURABLE_VERTEX_BUDGET)
        return apply_geometry_refinement(
            QgsGeometry(base),
            simplify_tol=self._simplify_tol,
            vertex_keep_fraction=self._keep_fraction,
            smooth=self._smooth,
            expand_dist=self._expand_dist,
            fill_holes=self._fill_holes,
            fill_holes_max_area=(
                self._hole_cutoff_units2 if measurable else 0.0),
            open_dist=self._open_dist,
            despike_m=self._despike_tol,
            vertex_spacing=vertex_spacing,
            vertex_min=vertex_min,
            vertex_max_deviation=vertex_deviation,
            vertex_max_deviation_fraction=vertex_deviation_frac,
            vertex_dial_max_cap_fraction=vertex_dial_cap,
            ortho=self._ortho,
            # Right angles needs a de-staircased outline first (a raw mask
            # outline is ALREADY all 90-degree stair steps, so orthogonalizing
            # it alone is a no-op).
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
        )
