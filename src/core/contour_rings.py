














from __future__ import annotations

import contextlib
from typing import Any






MAX_TRACED_RINGS = 20000


def _max_traced_rings() -> int:


    from .server_dials import dial_in_range

    return int(dial_in_range(
        "tuning.export.max_traced_rings", MAX_TRACED_RINGS, 5_000, 200_000))


def rings_to_polygons(rings: list[list[Any]]) -> list[Any]:








    from qgis.core import QgsGeometry, QgsLineString, QgsPolygon

    usable = [r for r in rings if r and len(r) >= 4]
    if not usable:
        return []
    if len(usable) > _max_traced_rings():
        _log_ring_flood(len(usable))
        return _all_solid(usable)
    if len(usable) == 1:
        return _all_solid(usable)

    try:
        solids = [_solid(r) for r in usable]
        containers = _containers(usable, solids)
        outers: list[int] = []
        holes: dict[int, list[int]] = {}
        for i, inside in enumerate(containers):
            if solids[i] is None:
                continue
            if len(inside) % 2 == 0:
                outers.append(i)
                holes.setdefault(i, [])
        for i, inside in enumerate(containers):
            if solids[i] is None or len(inside) % 2 == 0:
                continue


            owner = max(inside, key=lambda j: len(containers[j]))
            if owner in holes:
                holes[owner].append(i)
        out = []
        for i in outers:
            polygon = QgsPolygon()
            polygon.setExteriorRing(QgsLineString(list(usable[i])))
            for h in holes.get(i, []):
                polygon.addInteriorRing(QgsLineString(list(usable[h])))
            out.append(_oriented(QgsGeometry(polygon)))
        return out or [s for s in solids if s is not None]
    except Exception:  # noqa: BLE001
        return _all_solid(usable)


def _containers(rings: list[list[Any]], solids: list[Any]) -> list[list[int]]:












    from qgis.core import QgsGeometry, QgsRectangle, QgsSpatialIndex

    index = QgsSpatialIndex()
    for i, solid in enumerate(solids):
        if solid is not None:
            index.addFeature(i, solid.boundingBox())

    out: list[list[int]] = [[] for _ in rings]
    for i, solid in enumerate(solids):
        if solid is None:
            continue
        probe = QgsGeometry.fromPointXY(rings[i][0])
        if probe is None or probe.isEmpty():
            continue
        point = probe.asPoint()
        x, y = point.x(), point.y()
        for j in index.intersects(QgsRectangle(x, y, x, y)):
            if j == i or j < 0 or j >= len(solids) or solids[j] is None:
                continue
            if solids[j].contains(probe):
                out[i].append(j)
    return out


def _oriented(geom: Any) -> Any:








    try:
        forced = geom.forceRHR()
        if forced is not None and not forced.isEmpty():
            return forced
    except (RuntimeError, AttributeError, TypeError):
        pass
    return geom


def _all_solid(rings: list[list[Any]]) -> list[Any]:

    return [g for g in (_solid(r) for r in rings) if g is not None]


def _solid(ring: list[Any]) -> Any:

    from qgis.core import QgsGeometry, QgsLineString, QgsPolygon

    try:
        polygon = QgsPolygon()
        polygon.setExteriorRing(QgsLineString(list(ring)))
        geom = QgsGeometry(polygon)
        return None if geom.isEmpty() else _oriented(geom)
    except Exception:  # noqa: BLE001
        return None


def _log_ring_flood(count: int) -> None:

    with contextlib.suppress(Exception):
        from qgis.core import Qgis, QgsMessageLog

        QgsMessageLog.logMessage(
            f"Fallback polygonize: {count} traced rings, past the "
            f"{_max_traced_rings()} ceiling; rings kept as outlines, holes not "
            f"attached",
            "AI Segmentation",
            level=Qgis.MessageLevel.Warning,
        )
