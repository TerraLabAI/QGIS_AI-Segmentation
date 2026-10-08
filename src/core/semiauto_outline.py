




























from __future__ import annotations

import math
import struct

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


def _cut_ring_corners(xy: np.ndarray, counts: np.ndarray, cut: float):












    n_rings = len(counts)
    end = np.cumsum(counts)
    start = end - counts
    last = end - 1
    closed = (counts > 1) & (xy[start, 0] == xy[last, 0]) & (xy[start, 1] == xy[last, 1])
    body = np.ones(len(xy), dtype=bool)
    body[last[closed]] = False
    x, y = xy[body, 0], xy[body, 1]
    m = counts - closed
    m_start = np.cumsum(m) - m
    ring_of = np.repeat(np.arange(n_rings), m)
    nxt = np.arange(1, len(x) + 1)
    nxt[m_start + m - 1] = m_start
    bx, by = x[nxt], y[nxt]
    dx, dy = bx - x, by - y
    length = np.hypot(dx, dy)


    slanted = np.flatnonzero((dx != 0) & (dy != 0))
    if slanted.size:
        length[slanted] = np.fromiter(
            map(math.hypot, dx[slanted].tolist(), dy[slanted].tolist()),
            dtype=np.float64, count=slanted.size)
    edge = (m >= 4)[ring_of] & ~(length <= 0)
    ax, ay, bx, by, dx, dy, length, er = (
        v[edge] for v in (x, y, bx, by, dx, dy, length, ring_of))
    step = np.minimum(cut, 0.5 * length)
    ux, uy = dx / length, dy / length
    px, py = ax + ux * step, ay + uy * step
    qx, qy = bx - ux * step, by - uy * step


    emit = np.empty(2 * len(er), dtype=bool)
    emit[:1] = True
    emit[2::2] = (er[1:] != er[:-1]) | (px[1:] != qx[:-1]) | (py[1:] != qy[:-1])
    emit[1::2] = (qx != px) | (qy != py)
    sx = np.empty(2 * len(er))
    sx[0::2], sx[1::2] = px, qx
    sy = np.empty(2 * len(er))
    sy[0::2], sy[1::2] = py, qy
    ox, oy, out_ring = sx[emit], sy[emit], np.repeat(er, 2)[emit]
    got = np.bincount(out_ring, minlength=n_rings)
    got_end = np.cumsum(got)
    got_start = got_end - got
    two = np.flatnonzero(got > 1)
    first, final_pt = got_start[two], got_end[two] - 1
    drop = np.zeros(n_rings, dtype=bool)
    drop[two] = (ox[first] == ox[final_pt]) & (oy[first] == oy[final_pt])
    got -= drop
    cut_ok = (m >= 4) & (got >= 3)

    final_n = np.where(cut_ok, got, m) + 1
    final_end = np.cumsum(final_n)
    final_start = final_end - final_n
    fx = np.empty(int(final_end[-1]))
    fy = np.empty(len(fx))
    at = np.arange(len(ox)) - got_start[out_ring]
    take = np.flatnonzero(cut_ok[out_ring] & (at < got[out_ring]))
    dest = final_start[out_ring[take]] + at[take]
    fx[dest], fy[dest] = ox[take], oy[take]
    kept = np.flatnonzero(~cut_ok[ring_of])
    if kept.size:
        ring = ring_of[kept]
        dest = final_start[ring] + kept - m_start[ring]
        fx[dest], fy[dest] = x[kept], y[kept]
    fx[final_end - 1], fy[final_end - 1] = fx[final_start], fy[final_start]
    return np.column_stack((fx, fy)), final_n


_WKB_POLYGON_2D = b"\x01\x03\x00\x00\x00"
_WKB_MULTIPOLYGON_2D = b"\x01\x06\x00\x00\x00"


def _plain_polygon_wkb(geom) -> bytes | None:




    from qgis.core import QgsGeometry

    if geom is None or geom.isEmpty():
        return None
    wkb = bytes(geom.asWkb())
    if wkb[:5] in (_WKB_POLYGON_2D, _WKB_MULTIPOLYGON_2D):
        return wkb
    plain = (QgsGeometry.fromMultiPolygonXY(geom.asMultiPolygon())
             if geom.isMultipart()
             else QgsGeometry.fromPolygonXY(geom.asPolygon()))
    if plain is None or plain.isEmpty():
        return None
    return bytes(plain.asWkb())


def _polygon_wkb_rings(wkb: bytes):


    view = memoryview(wkb)
    multi = wkb[1] == 6
    off, n_parts = (9, struct.unpack_from("<I", wkb, 5)[0]) if multi else (0, 1)
    headers, ring_counts, rings = [], [], []
    for _ in range(n_parts):
        if wkb[off:off + 5] != _WKB_POLYGON_2D:
            return None
        n_rings = struct.unpack_from("<I", wkb, off + 5)[0]
        headers.append(wkb[off:off + 9])
        ring_counts.append(n_rings)
        off += 9
        for _ in range(n_rings):
            n = struct.unpack_from("<I", wkb, off)[0]
            if n == 0:
                return None
            rings.append(view[off + 4:off + 4 + 16 * n])
            off += 4 + 16 * n
    return headers, ring_counts, rings


def cut_corners(geometries: list, cut: float) -> list:









    out = list(geometries)
    if cut <= 0 or not out:
        return out
    try:
        layouts, rings = [], []
        for geom in out:
            try:
                wkb = _plain_polygon_wkb(geom)
                layout = _polygon_wkb_rings(wkb) if wkb is not None else None
            except Exception:  # noqa: BLE001
                layout = None
            if layout is None:
                layouts.append(None)
                continue

            layouts.append((wkb[:9] if wkb[1] == 6 else b"", layout))
            rings += layout[2]
        if not rings:
            return out
        xy = np.frombuffer(b"".join(rings), dtype="<f8").reshape(-1, 2)
        counts = np.fromiter(map(len, rings), dtype=np.int64, count=len(rings)) // 16
        final, final_n = _cut_ring_corners(xy, counts, cut)
        coords = memoryview(final.astype("<f8", copy=False).tobytes())
        count_bytes = final_n.astype("<u4").tobytes()
        sizes = final_n.tolist()
    except Exception:  # noqa: BLE001
        return out
    ring = pos = 0
    for index, plan in enumerate(layouts):
        if plan is None:
            continue
        head, (headers, ring_counts, _rings) = plan
        pieces = [head]
        for header, n_rings in zip(headers, ring_counts):
            pieces.append(header)
            for _ in range(n_rings):
                size = sizes[ring]
                pieces += (count_bytes[4 * ring:4 * ring + 4], coords[16 * pos:16 * (pos + size)])
                ring += 1
                pos += size
        result = _outline_from_wkb(b"".join(pieces))
        if result is not None:
            out[index] = result
    return out


def _outline_from_wkb(wkb: bytes):

    try:
        from qgis.core import QgsGeometry

        result = QgsGeometry()
        result.fromWkb(wkb)
        if result.isEmpty():
            return None
        if not result.isGeosValid():
            fixed = result.makeValid()
            if fixed is None or fixed.isEmpty() or not fixed.isGeosValid():
                return None
            result = fixed
        return result
    except Exception:  # noqa: BLE001
        return None


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
