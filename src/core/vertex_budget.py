"""Give an outline the number of points a person would have drawn.

A mask traced pixel by pixel carries one vertex every fraction of a metre. A
polygon digitized by hand, or shipped in a national reference database, carries
one vertex every few metres: the same shape, an order of magnitude fewer points.
Points that dense are not extra accuracy, they are the raster grid showing
through, and they make the layer heavy to draw and miserable to edit by hand.

This module cuts an outline down to a point BUDGET drawn from its own size,
using Visvalingam-Whyatt: repeatedly drop the vertex whose triangle with its two
neighbours is smallest, so the points that survive are the ones carrying the
shape. Three guards bound what it may do:

  - a budget, one point per ``spacing`` of outline, never below ``min_vertices``;
  - a deviation cap, so no single point is dropped if that would move the
    boundary further than the caller allows;
  - ``keep_fraction``, the share of its own points a ring may keep, which is
    what a user-facing dial moves.

Distances are in the geometry's own CRS units, along its X axis; callers
converting a ground setting cross the metres-per-unit factor first, like every
other ground dial, and pass ``unit_aspect`` when the two axes of that CRS cover
different ground.

Pure QGIS API and standard library: no numpy, no shapely, no venv. Manual mode
runs it offline like everything else it uses.
"""
from __future__ import annotations

import heapq
import math
from typing import Any

# A closed ring needs three distinct points; below four there is nothing left to
# thin and any budget under it is a caller mistake, not an instruction.
MIN_RING_VERTICES = 4


def _ring_metrics(pts: list, prev: list, nxt: list, i: int) -> tuple[float, float]:
    """(triangle area, boundary deviation) if vertex ``i`` were dropped.

    The deviation is the distance from the vertex to the chord that would
    replace it, which is exactly how far the outline moves.

    The reference vector-simplifying tools weight this area by the angle the
    vertex sits at, to drop spikes before bends. It changes nothing here,
    because the deviation cap and the edge fitting below already do that work.
    Left out rather than shipped as a dial that does nothing.
    """
    ax, ay = pts[prev[i]]
    bx, by = pts[i]
    cx, cy = pts[nxt[i]]
    cross = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
    base = math.hypot(cx - ax, cy - ay)
    return cross / 2.0, (cross / base if base > 0 else 0.0)


def thin_ring(pts: list, budget: int, max_deviation: float = 0.0) -> list:
    """The surviving points of ``thin_ring_indices``, in order."""
    return [pts[i] for i in thin_ring_indices(pts, budget, max_deviation)]


def thin_ring_indices(pts: list, budget: int,
                      max_deviation: float = 0.0) -> list:
    """Visvalingam-Whyatt one OPEN ring (no repeated closing point) down to
    ``budget`` points, never moving the outline by more than ``max_deviation``
    (0 = no cap). Returns the INDICES of the surviving points, so a caller can
    still see which original points each kept edge replaced.
    """
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
            # Stale entry: a neighbour moved since it was queued. Re-queue at
            # the price it costs now.
            heapq.heappush(heap, (cur_cost, i))
            continue
        if max_deviation > 0.0 and cur_dev > max_deviation:
            # Too expensive to drop. Left out of the heap; it comes back if one
            # of its neighbours goes and makes it cheap again.
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
    """The line that best replaces the original points from ``i`` to ``j``.

    Dropping points turns an arc into a chord, and a chord always cuts INSIDE
    the arc: every simplified outline shrinks. Instead of the chord, take the
    line parallel to it that leaves as much ground outside as it takes inside,
    which is the chord shifted by (area between arc and chord) / (chord length).
    Returns (point on line, unit direction), or None for a degenerate edge.
    """
    ax, ay = pts[i]
    bx, by = pts[j]
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length <= 0:
        return None
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    # Signed area of the closed loop (arc, then chord back), in the chord's own
    # frame: positive when the arc bulges along the left normal.
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
    """Where two fitted edges meet, or ``fallback`` when they cannot be trusted
    (near-parallel, or a corner that would fly further than ``max_shift``)."""
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
    """Move each kept edge onto the middle of the outline it replaced.

    Visvalingam-Whyatt picks WHICH points to keep; this decides WHERE they sit.
    Keeping them exactly where the traced mask put them makes every edge a
    chord, so the shape shrinks a little at every bend. Fitting each edge to
    its own arc and re-cutting the corners costs no extra point and hands back
    the area. ``max_shift`` bounds how far a corner may travel.
    """
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
    """How many points an outline of this length is allowed to keep."""
    if spacing <= 0:
        return 0
    return max(int(min_vertices), int(round(perimeter / spacing)))


def smooth_budget_multiplier(factor: float, iterations: int,
                             cap: float = 8.0) -> float:
    """How much wider the point spacing must be ahead of corner rounding.

    Corner rounding runs AFTER the budget and multiplies the outline points on
    every pass, so a budget applied first must thin by ``factor`` per pass to
    keep its promised density. ``cap`` bounds the total, so a high pass count
    cannot starve a shape of points; it is a server dial
    (``review.vertex_budget.smooth_multiplier_cap``) that callers pass in, and
    the default here is the client fallback. Returns 1.0 (spacing unchanged)
    when rounding is off (``iterations`` <= 0) or ``factor`` is unusable.
    """
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
    """One QgsPointXY ring (closed, first == last) thinned to its budget.

    ``unit_aspect`` stretches y into the x unit for the whole pass, so every
    length below is one distance again (see simplify_to_budget). The points go
    back into the caller's own frame on the way out.
    """
    pts = [(p.x(), p.y() * unit_aspect) for p in ring]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) <= MIN_RING_VERTICES:
        return ring
    budget = (ring_budget(_perimeter(pts), spacing, min_vertices)
              if spacing > 0 else len(pts))
    if keep_fraction > 0.0:
        # The user's own dial, applied on top of the class density: keep this
        # share of the points the ring came in with. Whichever of the two asks
        # for fewer points wins, so turning the dial down always thins. The
        # floor here is the hard ring minimum, NOT the class floor: when the
        # user explicitly dials Points down they may thin a clean shape past its
        # class density, which the higher class floor would otherwise block.
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
    """The deviation cap to use on ONE object, in the geometry's CRS units.

    A flat cap cannot be right for every object in a run: a metre is nothing on
    a 40 m warehouse and half the width of a hedge. So the caller's cap is also
    bounded by a share of the object's own NARROW dimension, the same ceiling
    the footprint regularizer puts on its snap tolerance. Falls back to the
    flat cap when the narrow dimension cannot be measured.
    """
    if max_deviation <= 0 or object_fraction <= 0:
        return max_deviation
    try:
        _pt, _area, _angle, width, height = geom.orientedMinimumBoundingBox()
        narrow = min(float(width), float(height))
    except Exception:  # noqa: BLE001 -- unmeasurable, keep the flat cap  # nosec B110
        return max_deviation
    if narrow <= 0:
        return max_deviation
    return min(max_deviation, object_fraction * narrow)


# How far a corner may travel when the user dials Points all the way down: the
# relaxed deviation cap saturates at this share of the object's narrow
# dimension, so aggressive simplification stays inside the object rather than
# throwing a spike outside it. Client fallback for the server policy's
# `review.vertex_budget.dial_max_cap_fraction`, which callers pass in.
_DIAL_MAX_CAP_NARROW_FRACTION = 0.5


def _dial_relaxed_cap(geom: Any, cap: float, keep_fraction: float,
                      max_cap_fraction: float = _DIAL_MAX_CAP_NARROW_FRACTION
                      ) -> float:
    """Loosen the boundary-movement cap as the user's Points dial drops.

    A lower dial is an explicit request for fewer points, so the fixed cap that
    protects real corners is relaxed in step (``cap / keep_fraction``): a low %
    can then shed the corners the fixed cap would keep. The relaxed cap is
    bounded by ``max_cap_fraction`` of the object's narrow dimension so a
    corner never flies outside the shape (0 = no such ceiling). Returns ``cap``
    unchanged when the dial is off (``keep_fraction`` 0) or the object cannot
    be measured.
    """
    if keep_fraction <= 0.0 or cap <= 0.0:
        return cap
    relaxed = cap / keep_fraction
    try:
        _pt, _area, _angle, width, height = geom.orientedMinimumBoundingBox()
        narrow = min(float(width), float(height))
    except Exception:  # noqa: BLE001 -- unmeasurable, keep the inverse-scaled cap  # nosec B110
        return relaxed
    if narrow <= 0 or max_cap_fraction <= 0:
        return relaxed
    return min(relaxed, max_cap_fraction * narrow)


def _polygonal_parts_xy(geom: Any, want_type: Any) -> list:
    """Every polygonal part of ``geom``, each as a list of QgsPointXY rings.

    ``makeValid`` answers a ring that self-touches with a collection: the
    repaired polygon next to the line the pinch collapsed to. Read the members
    and keep the polygons, rather than turning the whole repair down over the
    line that came with it.
    """
    parts: list = []
    if geom is None:
        return parts
    try:
        members = geom.asGeometryCollection()
    except Exception:  # noqa: BLE001 -- unreadable repair  # nosec B110
        return parts
    for member in members:
        try:
            if member is None or member.isEmpty() or member.type() != want_type:
                continue
            polys = (member.asMultiPolygon() if member.isMultipart()
                     else [member.asPolygon()])
        except Exception:  # noqa: BLE001 -- unreadable member  # nosec B112
            continue
        parts.extend(poly for poly in polys if poly)
    return parts


def _thinned_part_rings(rings: list, original: list, want_type: Any) -> list:
    """The thinned rings of ONE part, or that part's own rings when nothing
    valid can be built from them.

    Per part on purpose: a single ring self-touching after thinning must cost
    that part its budget and nothing else. Returning it whole-object, which is
    what a shared validity check does, hands the user the raw traced staircase
    for every other part of the same object.
    """
    from qgis.core import QgsGeometry

    try:
        part = QgsGeometry.fromPolygonXY(rings)
        if part is not None and not part.isEmpty():
            if part.isGeosValid():
                return [rings]
            repaired = _polygonal_parts_xy(part.makeValid(), want_type)
            if repaired:
                return repaired
    except Exception:  # noqa: BLE001 -- rebuild or repair failed  # nosec B110
        pass
    return [original]


def _polygon_from_parts(parts: list, multi: bool) -> Any:
    """One geometry out of ring lists, multipart when the input was multipart
    or a repair split one part into several."""
    from qgis.core import QgsGeometry

    if multi or len(parts) > 1:
        return QgsGeometry.fromMultiPolygonXY(parts)
    return QgsGeometry.fromPolygonXY(parts[0])


# Below this the stretch is a no-op and the caller's own geometry is measured.
_BUDGET_ASPECT_IDENTITY_EPSILON = 1e-9


def _measure_frame_polygon(geom: Any, polys: list, multi: bool,
                           unit_aspect: float) -> Any:
    """``geom`` with y stretched into the x unit, or ``geom`` itself when the
    two axes already agree or the stretched copy cannot be built.

    The deviation caps are a share of the object's narrow dimension, while the
    thinning pass measures every deviation in the stretched frame. Read in two
    different frames, a north-south object in a geographic CRS is capped
    against a length the pass never sees.
    """
    if abs(unit_aspect - 1.0) < _BUDGET_ASPECT_IDENTITY_EPSILON:
        return geom
    try:
        from qgis.core import QgsPointXY

        stretched = _polygon_from_parts(
            [[[QgsPointXY(p.x(), p.y() * unit_aspect) for p in ring]
              for ring in poly] for poly in polys], multi)
        if stretched is not None and not stretched.isEmpty():
            return stretched
    except Exception:  # noqa: BLE001 -- fall back to the caller's frame  # nosec B110
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
    """Thin every ring of a (multi)polygon to its own point budget.

    ``spacing`` and ``max_deviation`` are in the geometry's CRS units;
    ``spacing`` <= 0 drops the class density, leaving ``keep_fraction`` as the
    only budget (both off returns the input untouched).
    ``max_deviation_fraction`` tightens the cap on narrow objects (see
    deviation_cap_for), so one setting works across a run that mixes a
    warehouse and a hedge. ``keep_fraction`` (0 = off) is the user's
    share-of-points dial, applied on top of the class density and never able to
    ADD points back. ``dial_max_cap_fraction`` caps how far that dial may let a
    corner travel, as a share of the object's narrow dimension (0 = uncapped);
    it is a server dial (``review.vertex_budget.dial_max_cap_fraction``) and
    the default here is the client fallback.

    ``unit_aspect`` is ground metres per y unit over ground metres per x unit
    of the geometry's own CRS (see core.layer_conventions.ground_unit_aspect).
    Every measure below reads raw coordinates through ``math.hypot``, which
    takes the two axes for one unit. In a geographic CRS they are not: a
    north-south wall then measures short, so it is handed a smaller share of
    the budget and a looser deviation cap than a wall of the same ground length
    running east-west. Stretching y into the x unit for the pass makes all of
    them one distance again, and the points come back in the caller's frame.
    1.0 (a projected CRS) costs nothing.

    Best-effort by design, like the rest of the refine tail: a geometry that
    cannot be read yields the input unchanged rather than an exception, and a
    part whose thinned rings come back invalid and cannot be repaired keeps its
    own original rings while every other part keeps its budget. The answer is
    always polygonal and always of the input's own geometry type.
    """
    if geom is None or (spacing <= 0 and keep_fraction <= 0.0):
        return geom
    try:
        if geom.isEmpty():
            return geom
    except Exception:  # noqa: BLE001 -- unknown input  # nosec B110
        return geom
    try:
        multi = bool(geom.isMultipart())
        polys = geom.asMultiPolygon() if multi else [geom.asPolygon()]
    except Exception:  # noqa: BLE001 -- not a polygon we can read  # nosec B110
        return geom
    if not polys or not any(polys):
        return geom
    if not (unit_aspect and math.isfinite(unit_aspect) and unit_aspect > 0.0):
        unit_aspect = 1.0
    # Both caps below come off the object's narrow dimension, so they are read
    # in the same frame the thinning pass measures in, never in raw coordinates.
    measured = _measure_frame_polygon(geom, polys, multi, unit_aspect)
    cap = deviation_cap_for(measured, max_deviation, max_deviation_fraction)
    # The user's Points dial loosens this cap as it drops, so a low % actually
    # sheds the corners the fixed cap protects (bounded to the object). Off
    # (keep_fraction 0) leaves the cap exactly as it was.
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
    except Exception:  # noqa: BLE001 -- rebuild failed  # nosec B110
        return geom
    if result is None or result.isEmpty():
        return geom
    try:
        if not result.isGeosValid():
            # Each part is already valid on its own, so what is left is two of
            # them meeting, which only a whole-object repair can settle.
            parts = _polygonal_parts_xy(result.makeValid(), want_type)
            if not parts:
                return geom
            result = _polygon_from_parts(parts, multi)
            if result is None or result.isEmpty():
                return geom
    except Exception:  # noqa: BLE001 -- validity check is best-effort  # nosec B110
        return geom
    return result


# The fewest sides the de-staircase pass may leave a ROUND outline. Twice the
# budget's own floor, because the pre-pass is not the pass that decides a point
# count: it has to hand the budget something to choose from. Callers pass the
# served floor; this is the client fallback.
DEFAULT_MIN_ROUND_SIDES = 16


def round_chord_share(min_round_sides: int) -> float:
    """How far a run may leave its own chord, as a share of that chord's length.

    On a regular N-gon an edge's sagitta over its chord length is exactly
    tan(pi / 2N) / 2, and that ratio holds at every size. So bounding the ratio
    at N sides is the same statement as "never take a round outline below N
    sides", whatever the object measures in pixels, which a flat pixel bound
    cannot say: a crown of ten pixels' radius leaves its own chord by less than
    one pixel and a flat bound waves it through as a staircase.
    """
    sides = max(4, int(min_round_sides))
    return math.tan(math.pi / (2.0 * sides)) / 2.0


# How even the two sides have to be before a run counts as a staircase rather
# than an arc. A staircase's chord runs through the middle of its steps, so its
# two sides are equal and the ratio is 1. An arc bulges one way and only the
# grid noise reaches the other side, and that noise cannot pass half a step
# while the bulge that matters is over half a step. Half separates them.
_EVEN_STRADDLE = 0.5

# The longest chord, in grid steps, still allowed the one-step floor below.
# Erasing a sagitta of one step off a chord of c steps erases curvature of
# radius up to c * c / 8 steps, so at four the floor can only reach a radius of
# two steps: a round feature four pixels across, which is under what the mask
# itself can hold. Past four steps a departure of one step can be real shape,
# and only the chord share decides.
_FLOOR_MAX_CHORD_STEPS = 4.0


def _dp_mark_keeps(seq: list, start: int, end: int, tolerance: float,
                   chord_share: float, grid_step: float, keep: list) -> None:
    """Douglas-Peucker between two anchors, marking the points it keeps.

    Three bounds decide one edge. ``chord_share`` of its own chord, which says
    how round the outline may end up whatever it measures. One ``grid_step`` on
    a run that is a staircase and nothing else: at most four steps of chord,
    walking no further than |dx| + |dy|, and reaching either side of the chord.
    That floor is needed because the chord share of a run a few pixels long is
    under one step, which no staircase can meet, so on its own it would split a
    short staircase for ever. And the caller's flat ``tolerance``, over all.
    """
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
            # A staircase never turns back: it is monotone along both axes and
            # every segment of it is axis-aligned, so the ground it walks is
            # EXACTLY |dx| + |dy|, and only such a run is. The test is equality
            # and not an upper bound: a run of near-diagonal segments walks LESS
            # than |dx| + |dy| and an upper bound waved it through, flattening
            # real detail that happens to sit off the pixel grid.
            if (grid_step > 0.0
                    and norm <= _FLOOR_MAX_CHORD_STEPS * grid_step
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
    """The grid a ring was traced on: the shortest step between two of its
    points. A mask outline runs along pixel edges, so its shortest step IS one
    pixel, in whatever units the ring carries. 0 when there is nothing to read.
    """
    step = 0.0
    for a, b in zip(ring, ring[1:]):
        d = math.hypot(b.x() - a.x(), b.y() - a.y())
        if d > 0.0 and (step == 0.0 or d < step):
            step = d
    return step


def _destaircase_ring(ring: list, tolerance: float, chord_share: float,
                      grid_step: float) -> list:
    """One closed ring with its straight runs flattened and its arcs kept."""
    if grid_step <= 0.0:
        grid_step = outline_grid_step(ring)
    pts = [(p.x(), p.y()) for p in ring]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    n = len(pts)
    if n <= MIN_RING_VERTICES:
        return ring
    seq = pts + [pts[0]]
    # Two anchors, not one: Douglas-Peucker on a closed ring anchored at a
    # single point can only cut what that point already sees.
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
    """Flatten the pixel staircase on the STRAIGHT runs of an outline, and
    leave the curved ones alone.

    Douglas-Peucker at a flat tolerance cannot tell a staircase from an arc:
    both are a run of points near a chord, so a tolerance high enough to take a
    one-pixel step off a wall also takes a tree crown down to a hexagon. What
    separates them is how the departure from the chord scales. A staircase is
    bounded by its own step whatever the run's length, an arc's grows with the
    square of it, so each run here is judged against a share of its OWN chord
    (see round_chord_share) as well as against ``tolerance``, and a curved run
    is split until it carries ``min_round_sides``. Best-effort like the rest of
    the refine tail: the input comes back unchanged on any failure.
    """
    if geom is None or tolerance <= 0:
        return geom
    chord_share = round_chord_share(min_round_sides)
    try:
        if geom.isEmpty():
            return geom
        multi = bool(geom.isMultipart())
        polys = geom.asMultiPolygon() if multi else [geom.asPolygon()]
    except Exception:  # noqa: BLE001 -- not a polygon we can read  # nosec B110
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
    except Exception:  # noqa: BLE001 -- rebuild or repair failed  # nosec B110
        return geom
    return result if result is not None and not result.isEmpty() else geom
