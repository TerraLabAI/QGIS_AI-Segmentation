
















from __future__ import annotations








ZONE_WKT_CRS_AUTHID = "EPSG:4326"


def zone_polygon_from_wkt(wkt) -> object | None:

    if not wkt or not isinstance(wkt, str):
        return None
    try:
        from qgis.core import QgsGeometry

        geom = QgsGeometry.fromWkt(wkt)
        return None if geom is None or geom.isEmpty() else geom
    except Exception:  # noqa: BLE001
        return None


def zone_geometry_from_run(run: dict, crs_authid: str):





    geom = zone_polygon_from_wkt(run.get("zone_wkt"))
    if geom is None:
        return None
    try:
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsCoordinateTransformContext,
        )

        from ...core.qt_compat import geometry_op_succeeded

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




        if not geometry_op_succeeded(geom.transform(QgsCoordinateTransform(
                source, target, QgsCoordinateTransformContext()))):
            return None
        return None if geom.isEmpty() else geom
    except Exception:  # noqa: BLE001
        return None


def prepare_zone_engine(zone):







    if zone is None:
        return None
    try:
        from qgis.core import QgsGeometry

        engine = QgsGeometry.createGeometryEngine(zone.constGet())
        engine.prepareGeometry()
        return engine
    except Exception:  # noqa: BLE001
        return None


def clip_geometry_to_zone(geom, zone, engine=None):












    if zone is None or geom is None:
        return geom
    try:
        inside = False
        if engine is not None:
            try:
                inside = engine.contains(geom.constGet())
            except Exception:  # noqa: BLE001
                inside = False
        if inside:
            return geom
        cut = geom.intersection(zone)
    except Exception:  # noqa: BLE001
        return geom
    if cut is None or cut.isEmpty() or cut.area() <= 0:
        return None
    return cut
