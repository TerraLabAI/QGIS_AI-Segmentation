"""Per-polygon geometry attributes and a generic attribute-rule filter.

The MECHANISM behind the Automatic review's geometry-attribute false-positive
filter: measure an object's ground area, elongation and eccentricity, then drop
the objects whose attributes match a rule. The plugin ships NO thresholds; the
tuned per-shape-class rule tables live in the server detection policy and are
read (and validated) by ``detection_policy.fp_rules``. With no rules the filter
is an identity no-op, so nothing changes until the server delivers a table.

Pure geometry + math. QGIS is never imported here: the geometry objects are
passed in and their methods are called duck-typed, so the schema constants and
the rule evaluator import and run headless; only the attribute computation needs
a real QGIS geometry at call time.
"""
from __future__ import annotations

import math

# The attribute-filter schema and its single source of truth (the policy reader
# validates server rules against these). A rule names one attribute, one
# comparison op and one action; anything outside these sets is ignored, so a
# malformed server entry can only turn a rule OFF, never misfire.
FP_ATTRS = frozenset({"area_m2", "elongation", "eccentricity"})
FP_OPS = frozenset({"gt", "lt", "gte", "lte"})
FP_ACTIONS = frozenset({"drop"})


def _compare(value: float, op: str, threshold: float) -> bool:
    """Evaluate one known comparison op; an unknown op never matches."""
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
    """True when any DROP rule matches the attributes (pure Python, headless).

    A rule is ``{"attr", "op", "value", "action"}``; only ``drop`` actions are
    honoured. A rule naming an attribute the caller could not compute (a None or
    missing value) never matches, so a partial attribute set (e.g. an oriented
    box GEOS could not build) can only KEEP an object. Rules are assumed
    pre-validated by ``detection_policy.fp_rules``; an unknown op or a
    non-numeric value still fails safe here."""
    for rule in rules or []:
        if not isinstance(rule, dict) or rule.get("action") not in FP_ACTIONS:
            continue
        value = attrs.get(rule.get("attr"))
        if value is None:
            continue
        try:
            if _compare(float(value), rule.get("op"), float(rule.get("value"))):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _oriented_sides(geom) -> tuple[float | None, float | None]:
    """(width, height) of the oriented minimum bounding box, or (None, None).

    QGIS returns the box as a tuple ``(geometry, area, angle, width, height)``;
    the two side lengths are the last two values. A GEOS failure or any other
    return shape yields (None, None), so the caller treats shape attributes as
    unavailable rather than guessing."""
    try:
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
    """Geometry attributes of one polygon (a whole geometry; multipart allowed).

    Returns ``{"area_m2", "elongation", "eccentricity"}``; a value is None when
    it cannot be computed (an empty geometry, or an oriented box GEOS could not
    build), so a rule on it never fires. ``area_m2`` may be passed in when the
    caller already measured it (the review has the geodesic area on hand), else
    it is measured with ``measurer`` (geodesic) or falls back to the planar
    area. Elongation is the oriented minimum bounding box's long/short side ratio
    (>= 1, a square is 1); eccentricity is ``sqrt(1 - (short/long)^2)`` in
    [0, 1), a rectangle-derived proxy for how far the shape is from round."""
    attrs: dict = {"area_m2": None, "elongation": None, "eccentricity": None}
    if geom is None:
        return attrs
    try:
        if geom.isEmpty():
            return attrs
    except (RuntimeError, AttributeError):
        return attrs
    # Ground area: prefer the caller's measured value, else measure/estimate.
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
    # Shape: oriented minimum bounding box side lengths.
    width, height = _oriented_sides(geom)
    if width is not None and height is not None:
        long_side = max(width, height)
        short_side = min(width, height)
        if short_side > 0:
            attrs["elongation"] = long_side / short_side
            ratio = short_side / long_side
            attrs["eccentricity"] = math.sqrt(max(0.0, 1.0 - ratio * ratio))
    return attrs


def apply_fp_rules(parts, rules: list | None, measurer=None) -> tuple[list, int]:
    """Drop the geometries whose attributes match a DROP rule; keep the rest.

    ``parts`` is a list of geometries to evaluate INDEPENDENTLY; the caller
    picks the granularity (whole review objects, or the parts of one geometry it
    split beforehand). Returns ``(kept, dropped_count)``. Empty ``rules`` is an
    identity no-op (every geometry kept), so the filter stays OFF until the
    server delivers a table."""
    if not rules:
        return list(parts), 0
    kept = []
    dropped = 0
    for geom in parts:
        if matches_drop_rule(polygon_attributes(geom, measurer=measurer), rules):
            dropped += 1
        else:
            kept.append(geom)
    return kept, dropped
