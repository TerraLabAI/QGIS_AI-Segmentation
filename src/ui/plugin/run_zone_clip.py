"""The zone a past run was confined to, and the clip that reapplies it.

A live Automatic run clips every detection to the polygon the user drew before
anything reaches the merger, so an object sitting in a boundary tile's
rectangular overflow never appears. A replay decodes the same archived tiles
and has to apply the same clip, at the same stage, or the restored map shows
objects the original run threw away, over ground the user never asked about.

The outline travels on the run row as ``zone_wkt``, always in WGS84 (see
``ZONE_WKT_CRS_AUTHID``). It is optional: a rectangle zone, a headless run, and
every run archived before the field existed carry none, and those restore
exactly as they did. Everything here fails open to "no clip" for the same
reason.

Plain values only, no plugin and no QgsProject: the decode this serves runs on
the library's fetch thread.
"""
from __future__ import annotations

# The one CRS a run row's ``zone_wkt`` is ever written in. The run's own
# predict call sends the drawn outline in WGS84, and so does the summary the
# review uploads when the user finishes, so a row that carries no
# ``zone_crs_authid`` at all (every row written before that field existed) is
# WGS84 too. Reading an absent field as the tiles' CRS instead put the outline
# tens of thousands of kilometres from the ground it describes, and the clip
# then dropped every object in the run.
ZONE_WKT_CRS_AUTHID = "EPSG:4326"


def zone_polygon_from_wkt(wkt) -> object | None:
    """A polygon geometry from stored WKT, or None when it cannot be read."""
    if not wkt or not isinstance(wkt, str):
        return None
    try:
        from qgis.core import QgsGeometry

        geom = QgsGeometry.fromWkt(wkt)
        return None if geom is None or geom.isEmpty() else geom
    except Exception:  # noqa: BLE001 -- an unreadable zone is simply absent
        return None


def zone_geometry_from_run(run: dict, crs_authid: str):
    """The polygon THIS run was confined to, in ``crs_authid``, or None.

    Reaches QgsProject for nothing, because the caller runs on the library's
    fetch thread: the transform is built on a context of its own.
    """
    geom = zone_polygon_from_wkt(run.get("zone_wkt"))
    if geom is None:
        return None
    try:
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsCoordinateTransformContext,
        )

        source_authid = (
            str(run.get("zone_crs_authid") or "").strip() or ZONE_WKT_CRS_AUTHID)
        target_authid = str(crs_authid or "").strip()
        if not target_authid:
            return None
        if source_authid == target_authid:
            return geom
        source = QgsCoordinateReferenceSystem(source_authid)
        target = QgsCoordinateReferenceSystem(target_authid)
        if not source.isValid() or not target.isValid():
            return None
        # transform() answers 0 on success and edits the geometry in place, so
        # anything else leaves half-moved coordinates behind. A zone on the
        # wrong ground clips every real object away, which is worse than no
        # clip at all: refuse it.
        if geom.transform(QgsCoordinateTransform(
                source, target, QgsCoordinateTransformContext())) != 0:
            return None
        return None if geom.isEmpty() else geom
    except Exception:  # noqa: BLE001 -- an unreadable zone clips nothing
        return None


def prepare_zone_engine(zone):
    """A prepared GEOS engine over ``zone``, or None when one cannot be built.

    Prepared once and reused: the caller asks it about every decoded polygon,
    and the whole point of preparing is that the index is built once. A
    prepared engine is bound to the geometry instance it was built on, so it
    never crosses a thread with the zone left behind.
    """
    if zone is None:
        return None
    try:
        from qgis.core import QgsGeometry

        engine = QgsGeometry.createGeometryEngine(zone.constGet())
        engine.prepareGeometry()
        return engine
    except Exception:  # noqa: BLE001 -- without an engine the clip still works
        return None


def clip_geometry_to_zone(geom, zone, engine=None):
    """One decoded polygon confined to ``zone``, or None when nothing of it is
    inside. ``geom`` unchanged when there is no zone.

    This is where the live run clips, one detection at a time and BEFORE the
    merge. Stage matters as much as the clip: a detection outside the outline
    that reaches the merger absorbs the one inside it, and the pair comes back
    as a single object crossing ground the run never looked at, which no later
    clip can undo.

    A geometry the prepared engine reports wholly inside is returned as it is,
    so only the boundary-crossing few pay for an intersection.
    """
    if zone is None or geom is None:
        return geom
    try:
        inside = False
        if engine is not None:
            try:
                inside = engine.contains(geom.constGet())
            except Exception:  # noqa: BLE001 -- fall back to the intersection
                inside = False
        if inside:
            return geom
        cut = geom.intersection(zone)
    except Exception:  # noqa: BLE001 -- keep what cannot be measured
        return geom
    if cut is None or cut.isEmpty() or cut.area() <= 0:
        return None
    return cut
