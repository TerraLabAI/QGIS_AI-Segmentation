"""Fold a WGS84 zone outline back inside the lon/lat range before it is sent.

A canvas does not stop at longitude 180. A user working over Fiji or Chukotka
draws one continuous rectangle, and reprojecting it to WGS84 gives a ring whose
longitudes run past 180 (or below -180). The geometry is right and its area is
right, but the coordinates are outside the range WGS84 defines, and every
reader downstream is entitled to refuse them: the account service drops the
whole outline, so the run loses both its stored surface and the shape a later
"Run again here" needs.

The fix is the usual one. Shift the ring back into range, then cut it at the
antimeridian and keep both halves as one multipolygon. The ground is the same,
the area is the same, and every coordinate is now a legal one.

Pure geometry: no project, no canvas, no network. Safe on a worker thread.
"""

from __future__ import annotations

import math

# The lon/lat rectangle a WGS84 coordinate has to sit in.
_LON_MAX = 180.0
_LAT_MAX = 90.0


def crosses_antimeridian(geom) -> bool:
    """True when any coordinate of ``geom`` sits outside the longitude range.

    False on anything unreadable, so a caller that cannot decide keeps its
    existing behaviour.
    """
    try:
        box = geom.boundingBox()
        if box is None or box.isEmpty():
            return False
        return box.xMinimum() < -_LON_MAX or box.xMaximum() > _LON_MAX
    except Exception:  # noqa: BLE001 -- an unreadable shape crosses nothing
        return False


def fold_into_lonlat_range(geom):
    """``geom`` (WGS84) with every longitude inside [-180, 180].

    Returns the input untouched when it is already in range or when anything
    fails, so this can never be the reason a run loses its outline. A shape
    that does cross comes back as a multipolygon of the pieces either side of
    the antimeridian, which is the same ground under a legal spelling.
    """
    try:
        from qgis.core import QgsGeometry, QgsRectangle

        if not crosses_antimeridian(geom):
            return geom
        shifted = QgsGeometry(geom)
        box = shifted.boundingBox()
        # Bring the western edge into [-180, 180) first: a zone drawn after
        # several turns around the globe can sit any number of laps away.
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
            return west
        if west_ok:
            return west
        if east_ok:
            return east
        return geom
    except Exception:  # noqa: BLE001 -- the unfolded shape is the safe answer
        return geom
