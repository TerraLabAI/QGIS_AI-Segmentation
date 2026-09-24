















from __future__ import annotations

import numpy as np
from qgis.core import QgsGeometry

from .polygon_exporter import suppress_redundant_hypotheses


def select_tile_hypotheses(
    items: list,
    ios_threshold: float = 0.5,
    dup_ios_floor: float = 0.3,
    dup_centroid_frac: float = 0.35,
) -> list:







    n = len(items)
    if n < 2:
        return list(items)
    order = sorted(items, key=lambda t: -t[1])
    try:
        xmin = np.empty(n)
        xmax = np.empty(n)
        ymin = np.empty(n)
        ymax = np.empty(n)
        area = np.empty(n)
        cx = np.empty(n)
        cy = np.empty(n)
        dim = np.empty(n)





        engines = []
        for i, (geom, _score) in enumerate(order):
            bb = geom.boundingBox()
            engine = QgsGeometry.createGeometryEngine(geom.constGet())
            c = engine.centroid()
            if c is None:
                c = geom.centroid().asPoint()
            engines.append(engine)
            xmin[i] = bb.xMinimum()
            xmax[i] = bb.xMaximum()
            ymin[i] = bb.yMinimum()
            ymax[i] = bb.yMaximum()
            area[i] = geom.area()
            cx[i] = c.x()
            cy[i] = c.y()
            dim[i] = max(bb.width(), bb.height())
    except Exception:  # noqa: BLE001
        return suppress_redundant_hypotheses(
            items, ios_threshold, dup_ios_floor, dup_centroid_frac)

    overlap_floor = min(ios_threshold, dup_ios_floor)


    with np.errstate(divide="ignore", invalid="ignore"):
        kept, _kept_index = _walk_candidates(
            order, xmin, xmax, ymin, ymax, area, cx, cy, dim, engines,
            ios_threshold, dup_ios_floor, dup_centroid_frac, overlap_floor, n)
    return kept


def _walk_candidates(order, xmin, xmax, ymin, ymax, area, cx, cy, dim, engines,
                     ios_threshold, dup_ios_floor, dup_centroid_frac,
                     overlap_floor, n):



    k_xmin = np.empty(n)
    k_xmax = np.empty(n)
    k_ymin = np.empty(n)
    k_ymax = np.empty(n)
    k_area = np.empty(n)
    kept: list = []
    kept_index: list[int] = []
    m = 0
    for i, (geom, score) in enumerate(order):
        conflict = False
        if m:




            iw = np.minimum(xmax[i], k_xmax[:m]) - np.maximum(xmin[i], k_xmin[:m])
            ih = np.minimum(ymax[i], k_ymax[:m]) - np.maximum(ymin[i], k_ymin[:m])
            small = np.minimum(k_area[:m], area[i])
            bbox_ios = (iw * ih) / small
            passing = (iw > 0.0) & (ih > 0.0) & (small > 0.0) & (bbox_ios >= overlap_floor)
            for j in np.flatnonzero(passing).tolist():
                kept_geom = kept[j][0]
                inter = engines[i].intersection(kept_geom.constGet())
                ia = inter.area() if inter is not None and not inter.isEmpty() else 0.0
                ios = ia / float(small[j])
                if ios >= ios_threshold:
                    conflict = True
                    break
                if ios >= dup_ios_floor:



                    kj = kept_index[j]
                    smax = float(dim[i] if area[i] <= area[kj] else dim[kj])
                    if smax > 0.0:
                        dx = float(cx[i]) - float(cx[kj])
                        dy = float(cy[i]) - float(cy[kj])
                        dist = (dx ** 2 + dy ** 2) ** 0.5
                        if dist < dup_centroid_frac * smax:
                            conflict = True
                            break
        if not conflict:
            kept.append((geom, score))
            kept_index.append(i)
            k_xmin[m] = xmin[i]
            k_xmax[m] = xmax[i]
            k_ymin[m] = ymin[i]
            k_ymax[m] = ymax[i]
            k_area[m] = area[i]
            m += 1
    return kept, kept_index
