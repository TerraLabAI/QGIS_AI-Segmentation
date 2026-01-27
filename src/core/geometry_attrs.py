













from __future__ import annotations

import math
from typing import Any





FP_ATTRS = frozenset({"area_m2", "elongation", "eccentricity"})
FP_OPS = frozenset({"gt", "lt", "gte", "lte"})
FP_ACTIONS = frozenset({"drop"})


def _compare(value: float, op: str, threshold: float) -> bool:

    if op == "gt":
        return value > threshold
    if op == "lt":
        return value < threshold
    if op == "gte":
        return value >= threshold
    if op == "lte":
        return value <= threshold
    return False


def matches_drop_rule(attrs: dict, rules: list | None) -> bool:








    for rule in rules or []:
        if not isinstance(rule, dict) or rule.get("action") not in FP_ACTIONS:
            continue
        attr = rule.get("attr")
        if attr not in FP_ATTRS or rule.get("op") not in FP_OPS:
            continue
        value = attrs.get(attr)
        if value is None:
            continue
        op: Any = rule.get("op")
        threshold: Any = rule.get("value")
        try:
            if isinstance(value, bool) or isinstance(threshold, bool):
                continue
            value, threshold = float(value), float(threshold)
            if not math.isfinite(value) or not math.isfinite(threshold):
                continue
            if _compare(value, op, threshold):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _oriented_sides(geom, measurer=None) -> tuple[float | None, float | None]:








    try:
        crs = measurer.sourceCrs() if hasattr(measurer, "sourceCrs") else None
        if crs is not None and crs.isValid():
            from qgis.core import QgsPointXY

            from .ground_frame import aspect_is_identity, stretch_y

            center = geom.boundingBox().center()
            step = 0.001 if crs.isGeographic() else 1.0
            along_x = measurer.measureLine(
                center, QgsPointXY(center.x() + step, center.y()))
            along_y = measurer.measureLine(
                center, QgsPointXY(center.x(), center.y() + step))
            if (not math.isfinite(along_x) or not math.isfinite(along_y)
                    or along_x <= 0.0 or along_y <= 0.0):
                return None, None
            aspect = along_y / along_x
            if not aspect_is_identity(aspect):
                geom = stretch_y(geom, aspect, origin_y=center.y())
                if geom is None:
                    return None, None
        res = geom.orientedMinimumBoundingBox()
    except (RuntimeError, AttributeError, TypeError):
        return None, None
    if isinstance(res, (tuple, list)) and len(res) >= 5:
        try:
            return float(res[-2]), float(res[-1])
        except (TypeError, ValueError):
            return None, None
    return None, None


def polygon_attributes(geom, area_m2: float | None = None, measurer=None,
                       *, include_shape: bool = True) -> dict:












    attrs: dict = {"area_m2": None, "elongation": None, "eccentricity": None}
    if geom is None:
        return attrs
    try:
        if geom.isEmpty():
            return attrs
    except (RuntimeError, AttributeError):
        return attrs

    if area_m2 is not None:
        try:
            measured_area = float(area_m2)
            if math.isfinite(measured_area) and measured_area >= 0.0:
                attrs["area_m2"] = measured_area
        except (TypeError, ValueError):
            attrs["area_m2"] = None
    if area_m2 is None:
        try:
            attrs["area_m2"] = (
                float(measurer.measureArea(geom)) if measurer is not None
                else float(geom.area()))
        except (RuntimeError, AttributeError):
            try:
                attrs["area_m2"] = float(geom.area())
            except (RuntimeError, AttributeError):
                attrs["area_m2"] = None
    if attrs["area_m2"] is not None and (
            not math.isfinite(attrs["area_m2"]) or attrs["area_m2"] < 0.0):
        attrs["area_m2"] = None
    if not include_shape:
        return attrs

    width, height = _oriented_sides(geom, measurer)
    if (width is not None and height is not None
            and math.isfinite(width) and math.isfinite(height)):
        long_side = max(width, height)
        short_side = min(width, height)
        if short_side > 0:
            attrs["elongation"] = long_side / short_side
            ratio = short_side / long_side
            attrs["eccentricity"] = math.sqrt(max(0.0, 1.0 - ratio * ratio))
    return attrs


def polygon_matches_drop_rule(geom, rules: list | None,
                              area_m2: float | None = None, measurer=None) -> bool:






    area_rules = []
    shape_rules = []
    for rule in rules or []:
        if (not isinstance(rule, dict) or rule.get("action") not in FP_ACTIONS
                or rule.get("op") not in FP_OPS):
            continue
        attr = rule.get("attr")
        if attr == "area_m2":
            area_rules.append(rule)
        elif attr in ("elongation", "eccentricity"):
            shape_rules.append(rule)
    if area_rules and matches_drop_rule(
            polygon_attributes(geom, area_m2, measurer, include_shape=False),
            area_rules):
        return True
    return bool(shape_rules) and matches_drop_rule(
        polygon_attributes(geom, area_m2, measurer), shape_rules)
