
















from __future__ import annotations

import math


_LON_MAX = 180.0
_LAT_MAX = 90.0


def crosses_antimeridian(geom) -> bool:





    try:
        box = geom.boundingBox()
        if box is None or box.isEmpty():
            return False
        if box.xMinimum() < -_LON_MAX or box.xMaximum() > _LON_MAX:
            return True


        return any(abs(end.x() - start.x()) > _LON_MAX
                   for polygon in _polygon_rings(geom) for ring in polygon
                   for start, end in zip(ring, ring[1:]))
    except Exception:  # noqa: BLE001
        return False


def fold_into_lonlat_range(geom):







    try:
        from qgis.core import QgsGeometry, QgsRectangle

        from .polygon_masks import polygonal_part_of

        if not crosses_antimeridian(geom):
            return geom
        pieces = []
        for polygon in _polygon_rings(geom):
            unwrapped = _unwrap_polygon(polygon)
            if unwrapped is None:
                return geom
            box = unwrapped.boundingBox()
            first = math.floor((box.xMinimum() + _LON_MAX) / 360.0)
            last = math.floor((box.xMaximum() + _LON_MAX) / 360.0)
            for lap in range(first, last + 1):
                west, east = -_LON_MAX + 360.0 * lap, _LON_MAX + 360.0 * lap
                part = polygonal_part_of(unwrapped.intersection(
                    QgsGeometry.fromRect(QgsRectangle(west, -_LAT_MAX, east, _LAT_MAX))))
                if part is None:
                    continue
                if lap:
                    part.translate(-360.0 * lap, 0.0)
                pieces.append(part)
        if not pieces:
            return geom
        joined = QgsGeometry.unaryUnion(pieces)
        if joined is not None and not joined.isEmpty() and joined.isGeosValid():
            return joined
        return geom
    except Exception:  # noqa: BLE001
        return geom


def _polygon_rings(geom):

    return geom.asMultiPolygon() if geom.isMultipart() else [geom.asPolygon()]


def _unwrap_polygon(rings):






    from qgis.core import QgsGeometry

    if not rings:
        return None
    exterior = _unwrap_ring(rings[0])
    if exterior is None:
        return None
    anchor = (min(point.x() for point in exterior)
              + max(point.x() for point in exterior)) / 2.0
    interiors = [_unwrap_ring(ring, anchor) for ring in rings[1:]]
    if any(ring is None for ring in interiors):
        return None
    polygon = QgsGeometry.fromPolygonXY([exterior, *interiors])
    return polygon if not polygon.isEmpty() and polygon.isGeosValid() else None


def _unwrap_ring(ring, anchor=None):

    from qgis.core import QgsPointXY

    if not ring:
        return None
    out = []
    previous = anchor
    for point in ring:
        x, y = point.x(), point.y()
        if not math.isfinite(x) or not math.isfinite(y) or abs(y) > _LAT_MAX:
            return None
        if previous is None:
            x -= 360.0 * math.floor((x + _LON_MAX) / 360.0)
        else:
            x += 360.0 * round((previous - x) / 360.0)
        out.append(QgsPointXY(x, y))
        previous = x
    if out[0] != out[-1]:
        return None
    return out
