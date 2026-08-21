"""Does a drawn zone's geometry match the CRS the run says it is in.

A run takes its CRS id from the layer it renders. A layer with a local or
undefined CRS has no id to give, and the run records a geographic one instead,
so a polygon whose coordinates are ground metres can travel labelled as
degrees. Every geodesic measurement taken on it is then wrong rather than
missing, which is worse: a missing surface is skipped, a wrong one is stored.

Bounds are the only signal available before any measurement, and they refuse
both ways round. A geographic CRS cannot hold a coordinate outside the lon/lat
range. A projected CRS can hold any value, but a zone whose every coordinate
sits inside that same lon/lat range is a geographic layer wearing a projected
label: a real zone in metres would have to be within 180 m of the projection
origin, which nobody runs a detection over.
"""

from __future__ import annotations

# The lon/lat range, widened by one degree. QGIS lets a geographic coordinate
# sit a hair outside it after a transform, and the fault this catches is off by
# orders of magnitude, never by a rounding step.
_LON_LIMIT = 181.0
_LAT_LIMIT = 91.0


def zone_fits_declared_crs(geom, crs) -> bool:
    """True unless ``geom`` provably cannot be in ``crs``.

    ``geom`` is a QgsGeometry, ``crs`` a QgsCoordinateReferenceSystem. True on
    anything this cannot decide, including a missing or invalid argument or an
    empty geometry, so a caller keeps its existing behaviour and only loses the
    values that are demonstrably wrong.
    """
    try:
        if geom is None or crs is None or not crs.isValid():
            return True
        box = geom.boundingBox()
        if box is None or box.isEmpty():
            return True
        in_lonlat_range = (
            -_LON_LIMIT <= box.xMinimum() <= _LON_LIMIT
            and -_LON_LIMIT <= box.xMaximum() <= _LON_LIMIT
            and -_LAT_LIMIT <= box.yMinimum() <= _LAT_LIMIT
            and -_LAT_LIMIT <= box.yMaximum() <= _LAT_LIMIT
        )
        # A geographic CRS holds nothing outside the range; a projected one in
        # metres holds nothing inside it, unless the zone straddles the
        # projection origin at a scale of a few hundred metres.
        return in_lonlat_range if crs.isGeographic() else not in_lonlat_range
    except Exception:  # noqa: BLE001 - a check that raises must not block a run
        return True
