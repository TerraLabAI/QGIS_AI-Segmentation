







from __future__ import annotations

import math
from dataclasses import dataclass

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsCoordinateTransformContext,
    QgsGeometry,
    QgsRectangle,
)

from .layer_conventions import (
    UTM_MAX_LATITUDE_N,
    UTM_MIN_LATITUDE_S,
    _crs_holds_ground_scale,
    _project_transform_context,
    ground_unit_aspect,
    run_crs_enabled,
    run_crs_max_latitude,
    run_crs_max_span_deg,
)
from .qt_compat import geometry_op_succeeded
from .zone_antimeridian import crosses_antimeridian, fold_into_lonlat_range


@dataclass(frozen=True)
class WrappedZoneFrame:


    crs: QgsCoordinateReferenceSystem
    geometry: QgsGeometry
    periodic_bounds: tuple[float, float, float, float]


def _source_components_are_local(geom, source_crs, wgs84, context) -> bool:







    transform = QgsCoordinateTransform(source_crs, wgs84, context)
    parts = geom.asGeometryCollection() if geom.isMultipart() else [geom]
    if not parts:
        return False
    for part in parts:
        box = transform.transformBoundingBox(
            part.boundingBox(), handle180Crossover=True)
        west, east = box.xMinimum(), box.xMaximum()
        if not math.isfinite(west) or not math.isfinite(east):
            return False
        span = east - west
        if span < 0.0:
            span += 360.0
        if span <= 0.0 or span > run_crs_max_span_deg():
            return False
    return True


def _periodic_bounds(geom: QgsGeometry) -> tuple[float, float, float, float] | None:






    box = geom.boundingBox()
    if box.width() <= 180.0:
        return None
    intervals = []
    parts = geom.asGeometryCollection() if geom.isMultipart() else [geom]
    for part in parts:
        part_box = part.boundingBox()
        if (part_box.width() >= 180.0 or part_box.xMinimum() < -180.0
                or part_box.xMaximum() > 180.0):
            return None
        intervals.append((part_box.xMinimum() + 180.0,
                          part_box.xMaximum() + 180.0))
    if not intervals:
        return None
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    gaps = []
    for i, (_lo, hi) in enumerate(merged):
        next_index = (i + 1) % len(merged)
        next_lo = merged[next_index][0] + (360.0 if next_index == 0 else 0.0)
        gaps.append((next_lo - hi, next_index))
    gap, index = max(gaps)
    width = 360.0 - gap
    if width <= 0.0 or width > run_crs_max_span_deg():
        return None
    if not (-90.0 < box.yMinimum() <= box.yMaximum() < 90.0):


        return None
    west = merged[index][0] - 180.0
    return west, box.yMinimum(), west + width, box.yMaximum()


def wrapped_zone_frame(
    geom: QgsGeometry,
    source_crs: QgsCoordinateReferenceSystem,
    transform_context: QgsCoordinateTransformContext | None = None,
) -> WrappedZoneFrame | None:







    if (not run_crs_enabled() or geom is None or geom.isEmpty()
            or source_crs is None or not source_crs.isValid()):
        return None
    try:
        context = _project_transform_context(transform_context)
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        canonical = QgsGeometry(geom)
        if source_crs != wgs84:
            transform = QgsCoordinateTransform(source_crs, wgs84, context)
            if not geometry_op_succeeded(canonical.transform(transform)):
                return None
        if (crosses_antimeridian(canonical)
                and not _source_components_are_local(geom, source_crs, wgs84, context)):
            return None
        canonical = fold_into_lonlat_range(canonical)
        if (canonical is None or canonical.isEmpty()
                or not canonical.isGeosValid()):
            return None
        bounds = _periodic_bounds(canonical)
        if bounds is None:
            return None
        west, south, east, north = bounds
        lon = math.remainder((west + east) / 2.0, 360.0)
        lat = (south + north) / 2.0
        polar = max(abs(south), abs(north)) > run_crs_max_latitude()
        if polar:


            code = 32661 if south >= 0.0 else 32761 if north <= 0.0 else 0
            if not code:
                return None
        else:
            zone = min(60, max(1, int((lon + 180.0) / 6.0) + 1))
            code = (32600 if lat >= 0.0 else 32700) + zone
        candidate = QgsCoordinateReferenceSystem(f"EPSG:{code}")
        if not candidate.isValid():
            return None
        if polar:
            declared = candidate.bounds()
            if south < declared.yMinimum() or north > declared.yMaximum():
                return None
        elif south < UTM_MIN_LATITUDE_S or north > UTM_MAX_LATITUDE_N:
            return None
        if not _crs_holds_ground_scale(
                candidate, wgs84, QgsRectangle(west, south, east, north), context):
            return None
        transform = QgsCoordinateTransform(wgs84, candidate, context)
        if not geometry_op_succeeded(canonical.transform(transform)):
            return None



        parts = canonical.asGeometryCollection() if canonical.isMultipart() else [canonical]
        joined = QgsGeometry.unaryUnion(parts)
        if joined is None or joined.isEmpty() or not joined.isGeosValid():
            return None
        centre = joined.boundingBox().center()
        if ground_unit_aspect(candidate, centre.x(), centre.y()) != 1.0:
            return None
        return WrappedZoneFrame(candidate, joined, bounds)
    except Exception:  # noqa: BLE001
        return None
