
















from __future__ import annotations

import math


_LON_MAX = 180.0
_LAT_MAX = 90.0


def crosses_antimeridian(geom) -> bool:





    try:
        box = geom.boundingBox()
        if box is None or box.isEmpty():
            return False
        return box.xMinimum() < -_LON_MAX or box.xMaximum() > _LON_MAX
    except Exception:  # noqa: BLE001
        return False


def fold_into_lonlat_range(geom):







    try:
        from qgis.core import QgsGeometry, QgsRectangle

        if not crosses_antimeridian(geom):
            return geom
        shifted = QgsGeometry(geom)
        box = shifted.boundingBox()


        laps = math.floor((box.xMinimum() + _LON_MAX) / 360.0)
        if laps:
            shifted.translate(-360.0 * laps, 0.0)
        west = shifted.intersection(QgsGeometry.fromRect(
            QgsRectangle(-_LON_MAX, -_LAT_MAX, _LON_MAX, _LAT_MAX)))
        east = shifted.intersection(QgsGeometry.fromRect(
            QgsRectangle(_LON_MAX, -_LAT_MAX, _LON_MAX + 360.0, _LAT_MAX)))
        west_ok = west is not None and not west.isEmpty()
        east_ok = east is not None and not east.isEmpty()
        if east_ok:
            east.translate(-360.0, 0.0)
        if west_ok and east_ok:
            joined = west.combine(east)
            if joined is not None and not joined.isEmpty():
                return joined
            return geom
        if west_ok:
            return west
        if east_ok:
            return east
        return geom
    except Exception:  # noqa: BLE001
        return geom
