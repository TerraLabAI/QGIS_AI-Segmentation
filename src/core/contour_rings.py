"""Turn traced mask contours into polygons that keep their holes.

A contour tracer hands back one closed ring per boundary it meets, and it does
not say which is the outside of an object and which is a hole in one. Built as
separate polygons they all become solid, so a building's courtyard reaches the
map as a second building sitting inside the first, and its area is counted
twice.

A ring is a hole when it sits inside an odd number of the others, and its owner
is the innermost ring that contains it. That is the whole rule, and it handles
the nested case (an island inside a lake inside an island) for free.

Used by the no-rasterio polygonize fallback in ``polygon_exporter``, which is
the path a user without the optional geometry packages actually runs on.
"""
from __future__ import annotations

import contextlib
from typing import Any

# A ceiling against a mask that traced noise rather than objects. It is not a
# cost ceiling: containment runs through a spatial index, so the work grows
# with the ring count and not with its square, and a dense mask keeps its holes
# at any ordinary count. Past this the rings come back as plain outlines and
# the count is logged.
MAX_TRACED_RINGS = 20000


def rings_to_polygons(rings: list[list[Any]]) -> list[Any]:
    """Polygons built from closed rings of QgsPointXY, holes attached.

    Each ring must already be closed (first point repeated last). Returns one
    QgsGeometry per OUTER ring, each carrying the rings that fall inside it as
    interior rings. On any failure, and past ``MAX_TRACED_RINGS``, every ring
    comes back as its own solid polygon, which is what the caller did before
    holes were handled at all.
    """
    from qgis.core import QgsGeometry, QgsLineString, QgsPolygon

    usable = [r for r in rings if r and len(r) >= 4]
    if not usable:
        return []
    if len(usable) > MAX_TRACED_RINGS:
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
            # The owner is the innermost container, which is the one with the
            # most containers of its own.
            owner = max(inside, key=lambda j: len(containers[j]))
            if owner in holes:
                holes[owner].append(i)
        out = []
        for i in outers:
            polygon = QgsPolygon()
            polygon.setExteriorRing(QgsLineString(list(usable[i])))
            for h in holes.get(i, []):
                polygon.addInteriorRing(QgsLineString(list(usable[h])))
            out.append(QgsGeometry(polygon))
        return out or [s for s in solids if s is not None]
    except Exception:  # noqa: BLE001 -- a ring nobody can classify stays solid
        return _all_solid(usable)


def _containers(rings: list[list[Any]], solids: list[Any]) -> list[list[int]]:
    """For each ring, the indices of the rings that contain it.

    The probe is a VERTEX of the ring itself, never a point inside its filled
    disc. The middle of a building with a central courtyard sits inside that
    courtyard, and a probe there reports the building as sitting inside its own
    hole, so both come out odd, neither is an outer ring, and the courtyard
    reaches the map filled in. A point ON a ring lies inside another ring
    exactly when the whole ring does, which is the parity rule this wants.

    A spatial index over the ring boxes keeps the work near n log n: only the
    rings whose box covers the probe are asked the exact question.
    """
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


def _all_solid(rings: list[list[Any]]) -> list[Any]:
    """Every ring as its own exterior-only polygon, degenerate ones dropped."""
    return [g for g in (_solid(r) for r in rings) if g is not None]


def _solid(ring: list[Any]) -> Any:
    """One ring as an exterior-only polygon, or None when it will not build."""
    from qgis.core import QgsGeometry, QgsLineString, QgsPolygon

    try:
        polygon = QgsPolygon()
        polygon.setExteriorRing(QgsLineString(list(ring)))
        geom = QgsGeometry(polygon)
        return None if geom.isEmpty() else geom
    except Exception:  # noqa: BLE001 -- a degenerate ring is simply dropped
        return None


def _log_ring_flood(count: int) -> None:
    """Say that a mask traced so many rings that holes were given up on."""
    with contextlib.suppress(Exception):
        from qgis.core import Qgis, QgsMessageLog

        QgsMessageLog.logMessage(
            f"Fallback polygonize: {count} traced rings, past the "
            f"{MAX_TRACED_RINGS} ceiling; rings kept as outlines, holes not "
            f"attached",
            "AI Segmentation",
            level=Qgis.MessageLevel.Warning,
        )
