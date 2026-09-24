













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
        value = attrs.get(rule.get("attr"))
        if value is None:
            continue
        op: Any = rule.get("op")
        threshold: Any = rule.get("value")
        try:
            if _compare(float(value), op, float(threshold)):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _oriented_sides(geom, measurer=None) -> tuple[float | None, float | None]:







    try:
        crs = measurer.sourceCrs() if hasattr(measurer, "sourceCrs") else None
        if crs is not None and crs.isGeographic():
            from qgis.core import QgsPointXY

            from .ground_frame import aspect_is_identity, stretch_y

            center = geom.boundingBox().center()
            along_x = measurer.measureLine(
                center, QgsPointXY(center.x() + 0.001, center.y()))
            along_y = measurer.measureLine(
                center, QgsPointXY(center.x(), center.y() + 0.001))
            if (not math.isfinite(along_x) or not math.isfinite(along_y)
                    or along_x <= 0.0 or along_y <= 0.0):
                return None, None
            aspect = along_y / along_x
            if not aspect_is_identity(aspect):
                geom = stretch_y(geom, aspect)
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


def polygon_attributes(geom, area_m2: float | None = None, measurer=None) -> dict:










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
            attrs["area_m2"] = float(area_m2)
        except (TypeError, ValueError):
            attrs["area_m2"] = None
    if attrs["area_m2"] is None:
        try:
            attrs["area_m2"] = (
                float(measurer.measureArea(geom)) if measurer is not None
                else float(geom.area()))
        except (RuntimeError, AttributeError):
            try:
                attrs["area_m2"] = float(geom.area())
            except (RuntimeError, AttributeError):
                attrs["area_m2"] = None

    width, height = _oriented_sides(geom, measurer)
    if width is not None and height is not None:
        long_side = max(width, height)
        short_side = min(width, height)
        if short_side > 0:
            attrs["elongation"] = long_side / short_side
            ratio = short_side / long_side
            attrs["eccentricity"] = math.sqrt(max(0.0, 1.0 - ratio * ratio))
    return attrs
