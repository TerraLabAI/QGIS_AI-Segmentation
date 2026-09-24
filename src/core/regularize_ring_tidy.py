





















from __future__ import annotations

import math
from typing import NamedTuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]



_PARALLEL_DEG = 20.0
_PARALLEL_SIN = math.sin(math.radians(_PARALLEL_DEG))


_DIAGONAL_WINDOW_DEG = 10.0


_ROUNDED_MIN_CUTS = 3



_CORNER_DOT_MAX = 0.2



_SQUARE_DOT_MAX = 0.02



_OFFSET_CHUNK = 128


class TidyDials(NamedTuple):



    parallel_sin: float = _PARALLEL_SIN
    diagonal_window_deg: float = _DIAGONAL_WINDOW_DEG
    rounded_min_cuts: int = _ROUNDED_MIN_CUTS
    corner_dot_max: float = _CORNER_DOT_MAX
    square_dot_max: float = _SQUARE_DOT_MAX


DEFAULT_TIDY_DIALS = TidyDials()


def resolve_tidy_dials() -> TidyDials:


    try:
        from .server_dials import dial_in_range

        parallel_deg = dial_in_range("tuning.review.tidy_parallel_deg", _PARALLEL_DEG, 5.0, 40.0)
        return TidyDials(
            parallel_sin=math.sin(math.radians(parallel_deg)),
            diagonal_window_deg=dial_in_range(
                "tuning.review.tidy_diagonal_window_deg", _DIAGONAL_WINDOW_DEG, 2.0, 20.0),
            rounded_min_cuts=dial_in_range(
                "tuning.review.tidy_rounded_min_cuts", _ROUNDED_MIN_CUTS, 2, 8),
            corner_dot_max=dial_in_range(
                "tuning.review.tidy_corner_dot_max", _CORNER_DOT_MAX, 0.05, 0.5),
            square_dot_max=dial_in_range(
                "tuning.review.tidy_square_dot_max", _SQUARE_DOT_MAX, 0.005, 0.1),
        )
    except Exception:  # noqa: BLE001  # nosec B110
        return DEFAULT_TIDY_DIALS


def _ring_axis_deg(pts: list) -> float:

    sx = sy = 0.0
    n = len(pts)
    for i in range(n):
        dx, dy = pts[(i + 1) % n] - pts[i]
        length = math.hypot(dx, dy)
        if length <= 0:
            continue

        ang = 4.0 * math.atan2(dy, dx)
        sx += length * math.cos(ang)
        sy += length * math.sin(ang)
    return math.degrees(math.atan2(sy, sx)) / 4.0


def _is_diagonal(vec, axis_deg: float, window_deg: float = _DIAGONAL_WINDOW_DEG) -> bool:
    ang = math.degrees(math.atan2(vec[1], vec[0]))
    off = (ang - axis_deg - 45.0) % 90.0
    return min(off, 90.0 - off) <= window_deg


def _is_corner_cut(pts: list, i: int, axis_deg: float,
                   tidy: TidyDials = DEFAULT_TIDY_DIALS) -> bool:

    n = len(pts)
    prev_vec = pts[i] - pts[(i - 1) % n]
    next_vec = pts[(i + 2) % n] - pts[(i + 1) % n]
    window = tidy.diagonal_window_deg
    if _is_diagonal(prev_vec, axis_deg, window) or _is_diagonal(next_vec, axis_deg, window):
        return False
    lp, ln = math.hypot(*prev_vec), math.hypot(*next_vec)
    if lp < 1e-12 or ln < 1e-12:
        return False
    return abs(float(np.dot(prev_vec, next_vec))) / (lp * ln) < tidy.corner_dot_max


def _corner_cut_count(pts: list, axis_deg: float,
                      tidy: TidyDials = DEFAULT_TIDY_DIALS) -> int:
    n = len(pts)
    return sum(
        1 for i in range(n)
        if _is_diagonal(pts[(i + 1) % n] - pts[i], axis_deg, tidy.diagonal_window_deg)
        and _is_corner_cut(pts, i, axis_deg, tidy))


def _drop_collinear(pts: list) -> list:







    out = list(pts)
    if len(out) <= 3:
        return out
    flags = _collinear_flags(out)
    while len(out) > 3:
        try:
            i = flags.index(True)
        except ValueError:
            break
        del out[i]
        del flags[i]
        n = len(out)
        if n <= 3:
            break
        for k in ((i - 1) % n, i % n):
            flags[k] = _collinear_at(out, k)
    return out


def _collinear_at(out: list, i: int) -> bool:

    n = len(out)
    a, b, c = out[i - 1], out[i], out[(i + 1) % n]
    v1, v2 = b - a, c - b
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return True
    return abs(v1[0] * v2[1] - v1[1] * v2[0]) / (n1 * n2) < 1e-6


def _collinear_flags(out: list) -> list:


    ring = np.asarray(out)
    v1 = ring - np.concatenate((ring[-1:], ring[:-1]))
    v2 = np.concatenate((ring[1:], ring[:1])) - ring
    n1 = np.array([math.hypot(x, y) for x, y in v1.tolist()])
    n2 = np.array([math.hypot(x, y) for x, y in v2.tolist()])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        flat = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) / (n1 * n2) < 1e-6
    return ((n1 < 1e-9) | (n2 < 1e-9) | flat).tolist()


def _square_to(p0, p1, direction, dot_max: float = _SQUARE_DOT_MAX) -> bool:

    vec = p1 - p0
    length = math.hypot(*vec)
    if length < 1e-12:
        return False
    return abs(float(np.dot(vec, direction))) / length < dot_max


def _ring_offset(ring_a, ring_b) -> float:


    def _directed(points, other):
        seg_a = other
        seg_b = np.roll(other, -1, axis=0)
        d = seg_b - seg_a
        len2 = np.maximum((d * d).sum(axis=1), 1e-24)
        worst = 0.0



        for start in range(0, len(points), _OFFSET_CHUNK):
            block = points[start:start + _OFFSET_CHUNK]
            rel = block[:, None, :] - seg_a[None, :, :]
            t = np.clip((rel * d).sum(axis=2) / len2, 0.0, 1.0)
            gap = block[:, None, :] - (seg_a + d * t[:, :, None])
            nearest = np.hypot(gap[..., 0], gap[..., 1]).min(axis=1)
            for value in nearest.tolist():
                worst = max(worst, value)
        return worst
    a, b = np.asarray(ring_a), np.asarray(ring_b)
    return max(_directed(a, b), _directed(b, a))


def _line_intersection(p0, d0, p1, d1):
    cross = d0[0] * d1[1] - d0[1] * d1[0]
    if abs(cross) < 1e-12:
        return None
    t = ((p1[0] - p0[0]) * d1[1] - (p1[1] - p0[1]) * d1[0]) / cross
    return p0 + d0 * t


def _remove_edge(pts: list, i: int, reach: float,
                 tidy: TidyDials = DEFAULT_TIDY_DIALS) -> list | None:


    n = len(pts)
    j = (i + 1) % n
    a, b = pts[(i - 1) % n], pts[i]
    c, d = pts[j], pts[(j + 1) % n]
    dp, dn = b - a, d - c
    lp, ln = math.hypot(*dp), math.hypot(*dn)
    if lp < 1e-12 or ln < 1e-12:
        return None
    up, un = dp / lp, dn / ln
    cross = up[0] * un[1] - up[1] * un[0]
    out = list(pts)
    if abs(cross) > tidy.parallel_sin:

        x = _line_intersection(b, up, c, un)
        if x is None or math.hypot(*(x - (b + c) / 2.0)) > reach:
            return None
        out[i] = x
        del out[j]
        return out
    if float(np.dot(up, un)) > 0:




        before, after = pts[(i - 2) % n], pts[(j + 2) % n]
        order = ((c, d, after, up, j), (b, a, before, un, i)) if lp >= ln else (
            (b, a, before, un, i), (c, d, after, up, j))
        for near, far, beyond, keep_dir, k in order:
            normal = np.array([-keep_dir[1], keep_dir[0]])
            anchor = b if k == j else c
            shift = normal * float(np.dot(near - anchor, normal))
            if (math.hypot(*shift) > reach
                    or not _square_to(far, beyond, keep_dir, tidy.square_dot_max)):
                continue
            if k == j:
                out[j] = c - shift
                out[(j + 1) % n] = d - shift
            else:
                out[i] = b - shift
                out[(i - 1) % n] = a - shift
            return out
        return None



    if min(lp, ln) > reach:
        return None
    if lp <= ln:
        x = c + un * float(np.dot(a - c, un))
        if math.hypot(*(x - a)) > reach:
            return None
    else:
        x = b + up * float(np.dot(d - b, up))
        if math.hypot(*(x - d)) > reach:
            return None
    out[i] = x
    del out[j]
    return out


def tidy_squared_ring(coords, min_edge: float, chamfer_max: float,
                      max_shift: float = 0.0, reference=None,
                      tidy: TidyDials = DEFAULT_TIDY_DIALS):












    if np is None or (min_edge <= 0 and chamfer_max <= 0):
        return coords
    try:
        arr = np.asarray(coords, dtype=float)
        if len(arr) < 5:
            return coords
        pts = [p.copy() for p in arr[:-1]] if np.allclose(arr[0], arr[-1]) else [
            p.copy() for p in arr]
        pts = _drop_collinear(pts)
        if len(pts) <= 4:
            return coords
        start = np.asarray(pts)
        if reference is not None:
            ref = np.asarray(reference, dtype=float)
            if len(ref) > 1 and np.allclose(ref[0], ref[-1]):
                ref = ref[:-1]
            if len(ref) >= 3:
                start = ref
        budget = max_shift if max_shift > 0 else max(min_edge, chamfer_max)
        budget = max(budget, _ring_offset(pts, start))
        axis = _ring_axis_deg(pts)



        rounded = (chamfer_max > 0
                   and _corner_cut_count(pts, axis, tidy) >= tidy.rounded_min_cuts)
        window = tidy.diagonal_window_deg
        blocked: set = set()
        for _ in range(4 * len(pts)):
            n = len(pts)
            if n <= 4:
                break
            best = None
            for i in range(n):
                vec = pts[(i + 1) % n] - pts[i]
                length = math.hypot(*vec)
                limit = min_edge
                if chamfer_max > 0 and _is_diagonal(vec, axis, window):
                    if rounded and _is_corner_cut(pts, i, axis, tidy):
                        limit = max(limit, chamfer_max, length * 1.01)
                    else:
                        limit = max(limit, chamfer_max)
                if length < limit and (tuple(pts[i]), tuple(pts[(i + 1) % n])) not in blocked:
                    if best is None or length < best[1]:
                        best = (i, length, limit)
            if best is None:
                break
            i, length, limit = best
            nxt = _remove_edge(pts, i, 2.0 * limit, tidy)


            allowed = budget
            if rounded and _is_corner_cut(pts, i, axis, tidy):
                allowed = max(budget, 0.5 * length + 1e-9 * max(1.0, length))
            if nxt is not None and len(nxt) >= 4 and _ring_offset(nxt, start) > allowed:
                nxt = None
            if nxt is None or len(nxt) < 4:
                blocked.add((tuple(pts[i]), tuple(pts[(i + 1) % n])))
                continue
            pts = _drop_collinear(nxt)
        if len(pts) < 4:
            return coords
        return np.vstack([np.asarray(pts), pts[0]])
    except Exception:  # noqa: BLE001  # nosec B110
        return coords
