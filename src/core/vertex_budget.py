


























from __future__ import annotations

import heapq
import math
from typing import Any

from .shape_policy_dials import floor_max_chord_steps



MIN_RING_VERTICES = 4


def _ring_metrics(pts: list, prev: list, nxt: list, i: int) -> tuple[float, float]:










    ax, ay = pts[prev[i]]
    bx, by = pts[i]
    cx, cy = pts[nxt[i]]
    cross = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
    base = math.hypot(cx - ax, cy - ay)
    return cross / 2.0, (cross / base if base > 0 else 0.0)


def thin_ring(pts: list, budget: int, max_deviation: float = 0.0) -> list:

    return [pts[i] for i in thin_ring_indices(pts, budget, max_deviation)]


def thin_ring_indices(pts: list, budget: int,
                      max_deviation: float = 0.0) -> list:





    n = len(pts)
    budget = max(int(budget), MIN_RING_VERTICES)
    if n <= budget:
        return list(range(n))
    prev = [(i - 1) % n for i in range(n)]
    nxt = [(i + 1) % n for i in range(n)]
    alive = [True] * n
    heap: list[tuple[float, int]] = []
    for i in range(n):
        heapq.heappush(heap, (_ring_metrics(pts, prev, nxt, i)[0], i))
    count = n
    while heap and count > budget:
        cost, i = heapq.heappop(heap)
        if not alive[i]:
            continue
        cur_cost, cur_dev = _ring_metrics(pts, prev, nxt, i)
        if cur_cost != cost:


            heapq.heappush(heap, (cur_cost, i))
            continue
        if max_deviation > 0.0 and cur_dev > max_deviation:


            continue
        alive[i] = False
        count -= 1
        p, q = prev[i], nxt[i]
        nxt[p] = q
        prev[q] = p
        for j in (p, q):
            if alive[j]:
                heapq.heappush(heap, (_ring_metrics(pts, prev, nxt, j)[0], j))
    return [i for i in range(n) if alive[i]]


def _edge_offset(pts: list, i: int, j: int) -> tuple | None:








    ax, ay = pts[i]
    bx, by = pts[j]
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length <= 0:
        return None
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux


    area2 = 0.0
    k = i
    px, py = ax, ay
    n = len(pts)
    while k != j:
        k = (k + 1) % n
        qx, qy = pts[k]
        area2 += (px - ax) * (qy - ay) - (qx - ax) * (py - ay)
        px, py = qx, qy
    shift = -(area2 / 2.0) / length
    return ((ax + shift * nx, ay + shift * ny), (ux, uy))


def _line_intersection(line_a: tuple, line_b: tuple, fallback: tuple,
                       max_shift: float) -> tuple:


    (ax, ay), (ux, uy) = line_a
    (bx, by), (vx, vy) = line_b
    denom = ux * vy - uy * vx
    if abs(denom) < 1e-9:
        return fallback
    t = ((bx - ax) * vy - (by - ay) * vx) / denom
    px, py = ax + t * ux, ay + t * uy
    if max_shift > 0.0 and math.hypot(px - fallback[0], py - fallback[1]) > max_shift:
        return fallback
    return (px, py)


def fit_edges(pts: list, kept: list, max_shift: float = 0.0) -> list:








    m = len(kept)
    if m < MIN_RING_VERTICES:
        return [pts[i] for i in kept]
    lines = []
    for k in range(m):
        lines.append(_edge_offset(pts, kept[k], kept[(k + 1) % m]))
    out = []
    for k in range(m):
        prev_line, next_line = lines[k - 1], lines[k]
        original = pts[kept[k]]
        if prev_line is None or next_line is None:
            out.append(original)
            continue
        out.append(_line_intersection(prev_line, next_line, original, max_shift))
    return out


def ring_budget(perimeter: float, spacing: float, min_vertices: int) -> int:

    if spacing <= 0:
        return 0
    return max(int(min_vertices), int(round(perimeter / spacing)))


def smooth_budget_multiplier(factor: float, iterations: int,
                             cap: float = 8.0) -> float:










    if iterations <= 0 or factor <= 0:
        return 1.0
    mult = float(factor) ** int(iterations)
    return min(mult, float(cap)) if cap > 0 else mult


def _perimeter(pts: list) -> float:
    total = 0.0
    for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
        total += math.hypot(bx - ax, by - ay)
    return total


def _thin_ring_xy(ring: list, spacing: float, min_vertices: int,
                  max_deviation: float, fit: bool = True,
                  keep_fraction: float = 0.0,
                  unit_aspect: float = 1.0) -> list:






    pts = [(p.x(), p.y() * unit_aspect) for p in ring]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) <= MIN_RING_VERTICES:
        return ring
    budget = (ring_budget(_perimeter(pts), spacing, min_vertices)
              if spacing > 0 else len(pts))
    if keep_fraction > 0.0:






        budget = min(budget, max(MIN_RING_VERTICES,
                                 int(round(len(pts) * keep_fraction))))
    if budget <= 0 or len(pts) <= budget:
        return ring
    idx = thin_ring_indices(pts, budget, max_deviation)
    kept = (fit_edges(pts, idx, max_deviation) if fit
            else [pts[i] for i in idx])
    if len(kept) < MIN_RING_VERTICES - 1:
        return ring
    from qgis.core import QgsPointXY

    out = [QgsPointXY(x, y / unit_aspect) for x, y in kept]
    out.append(QgsPointXY(kept[0][0], kept[0][1] / unit_aspect))
    return out


def deviation_cap_for(geom: Any, max_deviation: float,
                      object_fraction: float) -> float:








    if max_deviation <= 0 or object_fraction <= 0:
        return max_deviation
    try:
        _pt, _area, _angle, width, height = geom.orientedMinimumBoundingBox()
        narrow = min(float(width), float(height))
    except Exception:  # noqa: BLE001  # nosec B110
        return max_deviation
    if narrow <= 0:
        return max_deviation
    return min(max_deviation, object_fraction * narrow)







_DIAL_MAX_CAP_NARROW_FRACTION = 0.5


def _dial_relaxed_cap(geom: Any, cap: float, keep_fraction: float,
                      max_cap_fraction: float = _DIAL_MAX_CAP_NARROW_FRACTION
                      ) -> float:










    if keep_fraction <= 0.0 or cap <= 0.0:
        return cap
    relaxed = cap / keep_fraction
    try:
        _pt, _area, _angle, width, height = geom.orientedMinimumBoundingBox()
        narrow = min(float(width), float(height))
    except Exception:  # noqa: BLE001  # nosec B110
        return relaxed
    if narrow <= 0 or max_cap_fraction <= 0:
        return relaxed
    return min(relaxed, max_cap_fraction * narrow)


def _polygonal_parts_xy(geom: Any, want_type: Any) -> list:







    parts: list = []
    if geom is None:
        return parts
    try:
        members = geom.asGeometryCollection()
    except Exception:  # noqa: BLE001  # nosec B110
        return parts
    for member in members:
        try:
            if member is None or member.isEmpty() or member.type() != want_type:
                continue
            polys = (member.asMultiPolygon() if member.isMultipart()
                     else [member.asPolygon()])
        except Exception:  # noqa: BLE001  # nosec B112
            continue
        parts.extend(poly for poly in polys if poly)
    return parts


def _thinned_part_rings(rings: list, original: list, want_type: Any) -> list:








    from qgis.core import QgsGeometry

    try:
        part = QgsGeometry.fromPolygonXY(rings)
        if part is not None and not part.isEmpty():
            if part.isGeosValid():
                return [rings]
            repaired = _polygonal_parts_xy(part.makeValid(), want_type)
            if repaired:
                return repaired
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return [original]


def _polygon_from_parts(parts: list, multi: bool) -> Any:


    from qgis.core import QgsGeometry

    if multi or len(parts) > 1:
        return QgsGeometry.fromMultiPolygonXY(parts)
    return QgsGeometry.fromPolygonXY(parts[0])



_BUDGET_ASPECT_IDENTITY_EPSILON = 1e-9


def _measure_frame_polygon(geom: Any, polys: list, multi: bool,
                           unit_aspect: float) -> Any:








    if abs(unit_aspect - 1.0) < _BUDGET_ASPECT_IDENTITY_EPSILON:
        return geom
    try:
        from qgis.core import QgsPointXY

        stretched = _polygon_from_parts(
            [[[QgsPointXY(p.x(), p.y() * unit_aspect) for p in ring]
              for ring in poly] for poly in polys], multi)
        if stretched is not None and not stretched.isEmpty():
            return stretched
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return geom


def simplify_to_budget(
    geom: Any,
    *,
    spacing: float,
    min_vertices: int = 8,
    max_deviation: float = 0.0,
    max_deviation_fraction: float = 0.0,
    fit: bool = True,
    keep_fraction: float = 0.0,
    dial_max_cap_fraction: float = _DIAL_MAX_CAP_NARROW_FRACTION,
    unit_aspect: float = 1.0,
) -> Any:






























    if geom is None or (spacing <= 0 and keep_fraction <= 0.0):
        return geom
    try:
        if geom.isEmpty():
            return geom
    except Exception:  # noqa: BLE001  # nosec B110
        return geom
    try:
        multi = bool(geom.isMultipart())
        polys = geom.asMultiPolygon() if multi else [geom.asPolygon()]
    except Exception:  # noqa: BLE001  # nosec B110
        return geom
    if not polys or not any(polys):
        return geom
    if not (unit_aspect and math.isfinite(unit_aspect) and unit_aspect > 0.0):
        unit_aspect = 1.0


    measured = _measure_frame_polygon(geom, polys, multi, unit_aspect)
    cap = deviation_cap_for(measured, max_deviation, max_deviation_fraction)



    cap = _dial_relaxed_cap(measured, cap, keep_fraction, dial_max_cap_fraction)
    want_type = geom.type()
    out_polys: list = []
    changed = False
    for poly in polys:
        rings = []
        for ring in poly:
            thinned = _thin_ring_xy(ring, spacing, min_vertices, cap, fit,
                                    keep_fraction, unit_aspect)
            if len(thinned) != len(ring):
                changed = True
            rings.append(thinned)
        out_polys.extend(_thinned_part_rings(rings, poly, want_type))
    if not changed or not out_polys:
        return geom
    try:
        result = _polygon_from_parts(out_polys, multi)
    except Exception:  # noqa: BLE001  # nosec B110
        return geom
    if result is None or result.isEmpty():
        return geom
    try:
        if not result.isGeosValid():


            parts = _polygonal_parts_xy(result.makeValid(), want_type)
            if not parts:
                return geom
            result = _polygon_from_parts(parts, multi)
            if result is None or result.isEmpty():
                return geom
    except Exception:  # noqa: BLE001  # nosec B110
        return geom
    return result






DEFAULT_MIN_ROUND_SIDES = 16


def round_chord_share(min_round_sides: int) -> float:









    sides = max(4, int(min_round_sides))
    return math.tan(math.pi / (2.0 * sides)) / 2.0







_EVEN_STRADDLE = 0.5







_FLOOR_MAX_CHORD_STEPS = 4.0


def _dp_mark_keeps(seq: list, start: int, end: int, tolerance: float,
                   chord_share: float, grid_step: float, keep: list) -> None:










    floor_steps = floor_max_chord_steps(_FLOOR_MAX_CHORD_STEPS)
    stack = [(start, end)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        ax, ay = seq[i]
        bx, by = seq[j]
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy)
        worst, at = 0.0, -1
        high = low = 0.0
        walked = 0.0
        last_x, last_y = ax, ay
        for k in range(i + 1, j):
            px, py = seq[k]
            walked += math.hypot(px - last_x, py - last_y)
            last_x, last_y = px, py
            if norm > 0.0:
                side = ((px - ax) * dy - (py - ay) * dx) / norm
                d = abs(side)
                high = max(high, side)
                low = min(low, side)
            else:
                d = math.hypot(px - ax, py - ay)
            if d > worst:
                worst, at = d, k
        walked += math.hypot(bx - last_x, by - last_y)
        if norm > 0.0:
            limit = chord_share * norm
            small, large = min(high, -low), max(high, -low)






            if (grid_step > 0.0
                    and norm <= floor_steps * grid_step
                    and abs(walked - (abs(dx) + abs(dy))) <= 1e-9 * max(1.0, walked)
                    and large > 0.0
                    and small >= _EVEN_STRADDLE * large):
                limit = max(limit, grid_step)
            limit = min(tolerance, limit)
        else:
            limit = 0.0
        if worst > limit and at > 0:
            keep[at] = True
            stack.append((i, at))
            stack.append((at, j))


def outline_grid_step(ring: list) -> float:




    step = 0.0
    for a, b in zip(ring, ring[1:]):
        d = math.hypot(b.x() - a.x(), b.y() - a.y())
        if d > 0.0 and (step == 0.0 or d < step):
            step = d
    return step


def _destaircase_ring(ring: list, tolerance: float, chord_share: float,
                      grid_step: float) -> list:

    if grid_step <= 0.0:
        grid_step = outline_grid_step(ring)
    pts = [(p.x(), p.y()) for p in ring]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    n = len(pts)
    if n <= MIN_RING_VERTICES:
        return ring
    seq = pts + [pts[0]]


    far = max(range(1, n),
              key=lambda k: math.hypot(seq[k][0] - seq[0][0],
                                       seq[k][1] - seq[0][1]))
    keep = [False] * (n + 1)
    keep[0] = keep[far] = keep[n] = True
    _dp_mark_keeps(seq, 0, far, tolerance, chord_share, grid_step, keep)
    _dp_mark_keeps(seq, far, n, tolerance, chord_share, grid_step, keep)
    kept = [k for k in range(n + 1) if keep[k]]
    if len(kept) >= len(seq):
        return ring
    from qgis.core import QgsPointXY

    return [QgsPointXY(seq[k][0], seq[k][1]) for k in kept]


def destaircase_outline(geom: Any, tolerance: float,
                        min_round_sides: int = DEFAULT_MIN_ROUND_SIDES,
                        grid_step: float = 0.0) -> Any:













    if geom is None or tolerance <= 0:
        return geom
    chord_share = round_chord_share(min_round_sides)
    try:
        if geom.isEmpty():
            return geom
        multi = bool(geom.isMultipart())
        polys = geom.asMultiPolygon() if multi else [geom.asPolygon()]
    except Exception:  # noqa: BLE001  # nosec B110
        return geom
    if not polys or not any(polys):
        return geom
    want_type = geom.type()
    out_polys: list = []
    changed = False
    for poly in polys:
        rings = []
        for ring in poly:
            flat = _destaircase_ring(ring, tolerance, chord_share,
                                     grid_step)
            if len(flat) != len(ring):
                changed = True
            rings.append(flat)
        out_polys.extend(_thinned_part_rings(rings, poly, want_type))
    if not changed or not out_polys:
        return geom
    try:
        result = _polygon_from_parts(out_polys, multi)
        if result is None or result.isEmpty():
            return geom
        if not result.isGeosValid():
            parts = _polygonal_parts_xy(result.makeValid(), want_type)
            if not parts:
                return geom
            result = _polygon_from_parts(parts, multi)
    except Exception:  # noqa: BLE001  # nosec B110
        return geom
    return result if result is not None and not result.isEmpty() else geom
