




























from __future__ import annotations

import math

import numpy as np




SMOOTH_PX_CLOUD_DEFAULT = 1.5
SMOOTH_PX_LOCAL_DEFAULT = 1.5



SMOOTH_SIZE_FRACTION_DEFAULT = 0.01



CORNER_CUT_PX = 0.5




SMOOTH_MAX_AREA_CHANGE = 0.2



SIMPLIFY_MAX_NARROW_FRACTION = 0.25



SMOOTH_PX_MAX = 4.0


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def _blur_axis(a: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:


    r = (len(kernel) - 1) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    padded = np.pad(a, pad)
    n = a.shape[axis]
    out = np.zeros_like(a, dtype=np.float32)
    for i, w in enumerate(kernel):
        if axis == 0:
            out += w * padded[i:i + n, :]
        else:
            out += w * padded[:, i:i + n]
    return out


def smooth_mask(mask, sigma_px: float, size_fraction: float = 0.0):








    if mask is None or sigma_px is None:
        return mask
    try:
        sigma = float(sigma_px)
    except (TypeError, ValueError):
        return mask
    if not math.isfinite(sigma) or sigma <= 0:
        return mask
    m = np.asarray(mask)
    if m.ndim != 2:
        return mask
    fg = m.astype(bool, copy=False)
    rows = np.flatnonzero(fg.any(axis=1))
    if rows.size == 0:
        return mask
    try:
        fraction = float(size_fraction or 0.0)
    except (TypeError, ValueError):
        fraction = 0.0
    if math.isfinite(fraction) and fraction > 0:
        sigma = max(sigma, fraction * math.sqrt(float(np.count_nonzero(fg))))
    sigma = min(sigma, SMOOTH_PX_MAX)
    cols = np.flatnonzero(fg.any(axis=0))
    pad = int(math.ceil(3.0 * sigma)) + 1
    r0 = max(0, int(rows[0]) - pad)
    r1 = min(fg.shape[0], int(rows[-1]) + pad + 1)
    c0 = max(0, int(cols[0]) - pad)
    c1 = min(fg.shape[1], int(cols[-1]) + pad + 1)
    box = fg[r0:r1, c0:c1].astype(np.float32)
    kernel = _gaussian_kernel(sigma)
    soft = _blur_axis(_blur_axis(box, kernel, 0), kernel, 1)
    smoothed_box = soft >= 0.5
    before = int(box.sum())
    after = int(smoothed_box.sum())
    if after == 0 or abs(after - before) > SMOOTH_MAX_AREA_CHANGE * before:
        return mask
    out = np.zeros(fg.shape, dtype=bool)
    out[r0:r1, c0:c1] = smoothed_box
    return out.astype(m.dtype, copy=False)


def _cut_ring(pts: list, cut: float) -> list:

    n = len(pts)
    if n < 4:
        return pts
    out = []
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        d = min(cut, 0.5 * length)
        ux, uy = dx / length, dy / length
        p = (ax + ux * d, ay + uy * d)
        q = (bx - ux * d, by - uy * d)
        if not out or out[-1] != p:
            out.append(p)
        if q != p:
            out.append(q)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out if len(out) >= 3 else pts


def cut_corners(geom, cut: float):



    if geom is None or cut <= 0:
        return geom
    try:
        from qgis.core import QgsGeometry, QgsPointXY

        if geom.isEmpty():
            return geom
        multi = bool(geom.isMultipart())
        polys = geom.asMultiPolygon() if multi else [geom.asPolygon()]
        out_polys = []
        for poly in polys:
            rings = []
            for ring in poly:
                pts = [(p.x(), p.y()) for p in ring]
                if len(pts) > 1 and pts[0] == pts[-1]:
                    pts = pts[:-1]
                cut_pts = _cut_ring(pts, cut)
                ring_out = [QgsPointXY(x, y) for x, y in cut_pts]
                ring_out.append(QgsPointXY(cut_pts[0][0], cut_pts[0][1]))
                rings.append(ring_out)
            out_polys.append(rings)
        result = (QgsGeometry.fromMultiPolygonXY(out_polys) if multi
                  else QgsGeometry.fromPolygonXY(out_polys[0]))
        if result is None or result.isEmpty():
            return geom
        if not result.isGeosValid():
            fixed = result.makeValid()
            if fixed is None or fixed.isEmpty() or not fixed.isGeosValid():
                return geom
            result = fixed
        return result
    except Exception:  # noqa: BLE001
        return geom


def simplify_outline(geom, tolerance: float):


    if geom is None or tolerance <= 0:
        return geom
    try:
        if geom.isEmpty():
            return geom
        from .polygon_masks import simplify_and_revalidate

        result = simplify_and_revalidate(geom, tolerance)
        if result is None or result.isEmpty():
            return geom
        return result
    except Exception:  # noqa: BLE001
        return geom
