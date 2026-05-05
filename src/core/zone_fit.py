











from __future__ import annotations

from typing import Callable

from qgis.core import QgsGeometry, QgsRectangle



_ITERATIONS = 40

_GOOD_ENOUGH = 0.995


def _zone_anchor(geom: QgsGeometry):


    center = geom.centroid()
    if center is None or center.isEmpty() or not geom.contains(center):
        center = geom.pointOnSurface()
    if center is None or center.isEmpty():
        return None
    return center.asPoint()


def fit_zone_to_area(
    geom: QgsGeometry,
    target_km2: float,
    measure_km2: Callable[[QgsGeometry], float],
) -> QgsGeometry | None:






    try:
        if geom is None or geom.isEmpty() or not (target_km2 > 0):
            return None
        anchor = _zone_anchor(geom)
        if anchor is None:
            return None
        box = geom.boundingBox()
        hi = max(box.xMaximum() - anchor.x(), anchor.x() - box.xMinimum(),
                 box.yMaximum() - anchor.y(), anchor.y() - box.yMinimum())
        if not hi > 0:
            return None
        lo = 0.0
        best = None
        for _ in range(_ITERATIONS):
            mid = (lo + hi) / 2.0
            square = QgsGeometry.fromRect(QgsRectangle(
                anchor.x() - mid, anchor.y() - mid,
                anchor.x() + mid, anchor.y() + mid))
            part = geom.intersection(square)
            area = 0.0
            if part is not None and not part.isEmpty():
                area = float(measure_km2(part))
            if area <= target_km2:
                lo = mid
                if area > 0:
                    best = part
                    if area >= target_km2 * _GOOD_ENOUGH:
                        break
            else:
                hi = mid
        if best is None:
            return None


        from .layer_conventions import repair_polygon
        repaired = repair_polygon(QgsGeometry(best))
        if repaired is not None and not repaired.isEmpty():
            if float(measure_km2(repaired)) <= target_km2:
                return repaired
        return best
    except Exception:  # noqa: BLE001
        return None
