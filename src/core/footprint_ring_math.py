







from __future__ import annotations

import math

import numpy as np



_MIN_DROP_ROUNDS = 20



_REACH_MARGIN = 1e-12




_CORNER_KEEP_MARGIN = 1e-9




_SIMPLE_ALL_PAIRS_MAX = 96


_PAIR_INDEX_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_PAIR_INDEX_CACHE_MAX = 128

_PAIR_GATHER_CACHE: dict[int, tuple] = {}


def angle_diff_mod90(a: float, b: float) -> float:

    d = abs(a - b) % 90.0
    return min(d, 90.0 - d)


def weighted_circular_mean_mod90(angles_deg: np.ndarray,
                                 weights: np.ndarray) -> float:

    a = np.radians(np.asarray(angles_deg, dtype=float) * 4.0)
    w = np.asarray(weights, dtype=float)
    s = float(np.sum(w * np.sin(a)))
    c = float(np.sum(w * np.cos(a)))
    if s == 0.0 and c == 0.0:
        return float(angles_deg[0]) % 90.0 if len(angles_deg) else 0.0
    return (math.degrees(math.atan2(s, c)) / 4.0) % 90.0


def ring_edge_azimuths_mod90(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    d = np.diff(coords, axis=0)
    lengths = np.hypot(d[:, 0], d[:, 1])
    az = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90.0
    return az, lengths


def ring_dominant_angle(coords: np.ndarray, bin_deg: float,
                        halo: int) -> tuple[float, float]:








    az, lengths = ring_edge_azimuths_mod90(coords)
    total = float(lengths.sum())
    if total <= 0.0:
        return 0.0, 0.0
    nbins = int(round(90.0 / bin_deg))
    bins = (az / bin_deg).astype(int) % nbins
    hist = np.bincount(bins, weights=lengths, minlength=nbins)
    win = int(np.argmax(hist))
    idx = [(win + k) % nbins for k in range(-halo, halo + 1)]


    in_window = np.zeros(nbins, dtype=bool)
    in_window[idx] = True
    mask = in_window[bins]
    if not mask.any():
        return float(win * bin_deg), 0.0
    angle = weighted_circular_mean_mod90(az[mask], lengths[mask])
    top_fraction = float(hist[idx].sum() / total)
    return angle, top_fraction


def _snapped_azimuth(theta_deg: float, base_deg: float, ortho_window_deg: float,
                     diag_window_deg: float) -> float | None:


    rel = (theta_deg - base_deg) % 90.0
    d_ortho = min(rel, 90.0 - rel)
    if d_ortho <= ortho_window_deg:
        n = round((theta_deg - base_deg) / 90.0)
        return (base_deg + 90.0 * n) % 180.0
    d_diag = abs(rel - 45.0)
    if d_diag <= diag_window_deg:
        n = round((theta_deg - base_deg - 45.0) / 90.0)
        return (base_deg + 45.0 + 90.0 * n) % 180.0
    return None


def ring_snap_segments(coords: np.ndarray, base_deg: float,
                       ortho_window_deg: float,
                       diag_window_deg: float) -> list:











    coords = np.asarray(coords)
    if len(coords) < 2:
        return []
    starts, ends = coords[:-1], coords[1:]
    mids = ((starts + ends) / 2.0).tolist()
    deltas = ends - starts
    lengths = np.hypot(deltas[:, 0], deltas[:, 1]).tolist()
    units: dict = {}
    lines = []
    for i, (dx, dy) in enumerate(deltas.tolist()):
        theta = math.degrees(math.atan2(dy, dx)) % 180.0
        snapped = _snapped_azimuth(
            theta, base_deg, ortho_window_deg, diag_window_deg)
        if snapped is None:
            snapped = theta
        u = units.get(snapped)
        if u is None:
            u = (math.cos(math.radians(snapped)), math.sin(math.radians(snapped)))
            units[snapped] = u
        lines.append([tuple(mids[i]), u, lengths[i]])
    return lines


def _merge_parallel_lines(lines: list, parallel_threshold: float) -> list:







    if not lines:
        return lines
    eps = math.sin(math.radians(1.0))
    merged = [list(lines[0])]
    for mid, u, ln in lines[1:]:
        pm, pu, pl = merged[-1]
        cross = abs(pu[0] * u[1] - pu[1] * u[0])
        if cross < eps:
            dvx, dvy = mid[0] - pm[0], mid[1] - pm[1]
            offset = abs(dvx * pu[1] - dvy * pu[0])
            if offset <= parallel_threshold:
                w = pl + ln
                merged[-1] = [((pm[0] * pl + mid[0] * ln) / w,
                               (pm[1] * pl + mid[1] * ln) / w), pu, w]
                continue
        merged.append([mid, u, ln])
    if len(merged) > 1:
        m0, u0, l0 = merged[0]
        m1, u1, l1 = merged[-1]
        cross = abs(u1[0] * u0[1] - u1[1] * u0[0])
        if cross < eps:
            dvx, dvy = m0[0] - m1[0], m0[1] - m1[1]
            offset = abs(dvx * u1[1] - dvy * u1[0])
            if offset <= parallel_threshold:
                w = l0 + l1
                merged[0] = [((m1[0] * l1 + m0[0] * l0) / w,
                              (m1[1] * l1 + m0[1] * l0) / w), u1, w]
                merged.pop()
    return merged


def ring_is_simple(coords: np.ndarray) -> bool:












    pts = np.asarray(coords, dtype=float)
    n = len(pts) - 1
    if n < 4:
        return True
    starts, ends = pts[:n], pts[1:n + 1]
    if n <= _SIMPLE_ALL_PAIRS_MAX:



        gather = _pair_gather(n)
        if gather is None:
            return True


        origin, first, second, count = gather
        signs = _cross_sign(pts[origin], pts[first], pts[second]).reshape(4, count)
        return not bool(np.any((signs[0] * signs[1] < 0) & (signs[2] * signs[3] < 0)))
    for i in range(n - 2):
        lo = i + 2
        hi = n - 1 if i == 0 else n
        if lo >= hi:
            continue
        p, q = starts[i], ends[i]
        a, b = starts[lo:hi], ends[lo:hi]
        if np.any(_pairs_cross(p, q, a, b)):
            return False
    return True


def _pairs_cross(p: np.ndarray, q: np.ndarray,
                 a: np.ndarray, b: np.ndarray) -> np.ndarray:


    d1 = _cross_sign(a, b, p)
    d2 = _cross_sign(a, b, q)
    d3 = _cross_sign(p, q, a)
    d4 = _cross_sign(p, q, b)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def _non_adjacent_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:



    hit = _PAIR_INDEX_CACHE.get(n)
    if hit is not None:
        return hit
    i_idx, j_idx = np.triu_indices(n, k=2)

    keep = ~((i_idx == 0) & (j_idx == n - 1))
    pair = (i_idx[keep], j_idx[keep])
    if len(_PAIR_INDEX_CACHE) < _PAIR_INDEX_CACHE_MAX:
        _PAIR_INDEX_CACHE[n] = pair
    return pair


def _pair_gather(n: int) -> tuple | None:






    hit = _PAIR_GATHER_CACHE.get(n)
    if hit is not None:
        return hit
    i_idx, j_idx = _non_adjacent_pairs(n)
    if i_idx.size == 0:
        return None
    origin = np.concatenate((j_idx, j_idx, i_idx, i_idx))
    first = np.concatenate((j_idx + 1, j_idx + 1, i_idx + 1, i_idx + 1))
    second = np.concatenate((i_idx, i_idx + 1, j_idx, j_idx + 1))
    gather = (origin, first, second, int(i_idx.size))
    if len(_PAIR_GATHER_CACHE) < _PAIR_INDEX_CACHE_MAX:
        _PAIR_GATHER_CACHE[n] = gather
    return gather


def _cross_sign(origin: np.ndarray, first: np.ndarray,
                second: np.ndarray) -> np.ndarray:



    return np.sign((first[..., 0] - origin[..., 0]) * (second[..., 1] - origin[..., 1])
                   - (first[..., 1] - origin[..., 1]) * (second[..., 0] - origin[..., 0]))


def ring_rebuild_corners(lines: list, parallel_threshold: float) -> np.ndarray | None:












    lines = _merge_parallel_lines(lines, parallel_threshold)
    n = len(lines)
    if n < 3:
        return None
    near_parallel = math.sin(math.radians(4.0))
    pts = []
    for i in range(n):
        (m1x, m1y), (u1x, u1y), l1 = lines[i]
        (m2x, m2y), (u2x, u2y), l2 = lines[(i + 1) % n]
        cross = u1x * u2y - u1y * u2x

        if not abs(cross) < near_parallel:
            dvx, dvy = m2x - m1x, m2y - m1y
            t1 = (dvx * u2y - dvy * u2x) / cross
            ix, iy = m1x + u1x * t1, m1y + u1y * t1


            reach = 2.0 * (l1 + l2) + 4.0
            spike = _beyond_reach(ix - m1x, iy - m1y, reach)
            if not spike:
                spike = _beyond_reach(ix - m2x, iy - m2y, reach)
            if not spike:
                pts.append((ix, iy))
                continue
        half1, half2 = l1 / 2.0, l2 / 2.0
        pts.append((m1x + u1x * half1, m1y + u1y * half1))
        pts.append((m2x - u2x * half2, m2y - u2y * half2))
    if len(pts) < 3:
        return None
    ring = np.asarray(pts + [pts[0]], dtype=float)
    return ring if ring_is_simple(ring) else None


def _beyond_reach(dx: float, dy: float, reach: float) -> bool:




    square = dx * dx + dy * dy
    limit = reach * reach
    if square < limit * (1.0 - _REACH_MARGIN):
        return False
    if square > limit * (1.0 + _REACH_MARGIN):
        return True
    return float(np.hypot(dx, dy)) > reach


def ring_drop_short_edges(coords: np.ndarray, min_edge_abs: float,
                          min_edge_rel: float,
                          min_corner_deg: float) -> np.ndarray:















    perim = float(np.sum(np.hypot(*(np.diff(coords, axis=0).T))))
    min_edge = max(min_edge_abs, min_edge_rel * perim)
    pts = list(coords[:-1])
    changed = True
    rounds = 0



    max_rounds = max(_MIN_DROP_ROUNDS, len(pts))

    corners_passed: set = set()
    keep_below = _corner_keep_below(min_corner_deg)
    while changed and len(pts) > 3 and rounds < max_rounds:
        changed = False
        rounds += 1
        n = len(pts)
        ring = np.asarray(pts)
        step = np.empty_like(ring)
        step[:-1] = ring[1:] - ring[:-1]
        step[-1] = ring[0] - ring[-1]
        lengths = np.hypot(step[:, 0], step[:, 1]).tolist()
        order = np.argsort(lengths)
        for i in order:
            if lengths[i] >= min_edge:
                break
            i = int(i)
            j = (i + 1) % n
            a_prev = pts[(i - 1) % n]
            b_next = pts[(j + 1) % n]
            mid = (pts[i] + pts[j]) / 2.0
            u1 = pts[i] - a_prev
            u2 = pts[j] - b_next
            n1, n2 = np.hypot(*u1), np.hypot(*u2)
            repl = mid
            if n1 > 1e-9 and n2 > 1e-9:
                u1, u2 = u1 / n1, u2 / n2
                cross = u1[0] * u2[1] - u1[1] * u2[0]
                if abs(cross) > math.sin(math.radians(20.0)):
                    dv = pts[j] - pts[i]
                    t1 = (dv[0] * u2[1] - dv[1] * u2[0]) / cross
                    inter = pts[i] + u1 * t1
                    if float(np.hypot(*(inter - mid))) <= 2.0 * min_edge:
                        repl = inter
            pts[i] = repl
            del pts[j]
            changed = True
            break
        if len(pts) <= 3:
            break
        n = len(pts)
        ring = np.asarray(pts)
        xy = ring.tolist()
        clearly_kept = _corners_clearly_kept(ring, keep_below)
        for i in range(n):
            key = (*xy[i - 1], *xy[i], *xy[(i + 1) % n])
            if key in corners_passed or clearly_kept[i]:
                continue
            a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
            v1, v2 = a - b, c - b
            n1, n2 = np.hypot(*v1), np.hypot(*v2)
            if n1 == 0 or n2 == 0:
                del pts[i]
                changed = True
                break
            cosang = float(np.dot(v1, v2) / (n1 * n2))
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
            if ang < min_corner_deg:
                del pts[i]
                changed = True
                break
            corners_passed.add(key)
    return np.asarray(pts + [pts[0]])


def _corner_keep_below(min_corner_deg: float) -> float | None:




    try:
        floor = float(min_corner_deg)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(floor) and 0.1 < floor < 180.0):
        return None
    return math.cos(math.radians(floor)) - _CORNER_KEEP_MARGIN


def _corners_clearly_kept(ring: np.ndarray, keep_below: float | None) -> list:








    n = len(ring)
    if keep_below is None or n == 0:
        return [False] * n
    prev = np.concatenate((ring[-1:], ring[:-1]))
    nxt = np.concatenate((ring[1:], ring[:1]))
    v1 = prev - ring
    v2 = nxt - ring
    n1 = np.hypot(v1[:, 0], v1[:, 1])
    n2 = np.hypot(v2[:, 0], v2[:, 1])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cos = (v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]) / (n1 * n2)
        kept = (cos < keep_below) & (n1 > 0) & (n2 > 0)
    return kept.tolist()


def circle_ring(center_x: float, center_y: float, area: float,
                segments: int = 32) -> np.ndarray:

    radius = math.sqrt(max(area, 0.0) / math.pi)
    ang = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    pts = np.column_stack([center_x + radius * np.cos(ang),
                           center_y + radius * np.sin(ang)])
    return np.vstack([pts, pts[:1]])
