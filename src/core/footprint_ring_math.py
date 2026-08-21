"""Ring-level math for the run-wide footprint alignment (numpy only, no QGIS).

Operates on closed exterior rings as (n, 2) float arrays (last point equal to
the first). The QGIS-side driver is core/footprint_alignment.py; this module
holds the coordinate math so it can run and be checked under any Python that
has numpy. Distances are in the caller's frame units (the driver hands it a
metre frame). Every function is pure: no state, no I/O, never touches QGIS.
"""
from __future__ import annotations

import math

import numpy as np


def angle_diff_mod90(a: float, b: float) -> float:
    """Smallest absolute difference between two mod-90 angles, in [0, 45]."""
    d = abs(a - b) % 90.0
    return min(d, 90.0 - d)


def weighted_circular_mean_mod90(angles_deg: np.ndarray,
                                 weights: np.ndarray) -> float:
    """Weighted circular mean of angles on the mod-90 circle, in [0, 90)."""
    a = np.radians(np.asarray(angles_deg, dtype=float) * 4.0)
    w = np.asarray(weights, dtype=float)
    s = float(np.sum(w * np.sin(a)))
    c = float(np.sum(w * np.cos(a)))
    if s == 0.0 and c == 0.0:
        return float(angles_deg[0]) % 90.0 if len(angles_deg) else 0.0
    return (math.degrees(math.atan2(s, c)) / 4.0) % 90.0


def ring_edge_azimuths_mod90(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Edge azimuths (degrees mod 90) and edge lengths of a closed ring."""
    d = np.diff(coords, axis=0)
    lengths = np.hypot(d[:, 0], d[:, 1])
    az = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90.0
    return az, lengths


def ring_dominant_angle(coords: np.ndarray, bin_deg: float,
                        halo: int) -> tuple[float, float]:
    """Length-weighted dominant edge azimuth of a closed ring, mod 90.

    Histogram of the edge azimuths in ``bin_deg`` bins; the answer is the
    weighted circular mean of the edges inside the winning bin plus ``halo``
    bins each side. Returns (angle_deg, top_fraction) where top_fraction is
    the share of total edge length inside that window: low means the ring has
    no clear grid and the caller should try other angle sources.
    """
    az, lengths = ring_edge_azimuths_mod90(coords)
    total = float(lengths.sum())
    if total <= 0.0:
        return 0.0, 0.0
    nbins = int(round(90.0 / bin_deg))
    bins = (az / bin_deg).astype(int) % nbins
    hist = np.bincount(bins, weights=lengths, minlength=nbins)
    win = int(np.argmax(hist))
    idx = [(win + k) % nbins for k in range(-halo, halo + 1)]
    mask = np.isin(bins, idx)
    if not mask.any():
        return float(win * bin_deg), 0.0
    angle = weighted_circular_mean_mod90(az[mask], lengths[mask])
    top_fraction = float(hist[idx].sum() / total)
    return angle, top_fraction


def _snapped_azimuth(theta_deg: float, base_deg: float, ortho_window_deg: float,
                     diag_window_deg: float) -> float | None:
    """Snap an absolute azimuth (deg, taken mod 180) onto base + n*90, or onto
    base + 45 + n*90 inside the tighter diagonal window. None = stays free."""
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
    """Rotate each ring edge about its midpoint onto its snapped azimuth.

    An edge outside both capture windows keeps its own azimuth. Returns one
    [midpoint, unit_direction, length] triple per edge, the input of
    ring_rebuild_corners.
    """
    lines = []
    for i in range(len(coords) - 1):
        a, b = coords[i], coords[i + 1]
        mid = (a + b) / 2.0
        theta = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
        snapped = _snapped_azimuth(
            theta, base_deg, ortho_window_deg, diag_window_deg)
        if snapped is None:
            snapped = theta
        u = np.array([math.cos(math.radians(snapped)),
                      math.sin(math.radians(snapped))])
        lines.append([mid, u, float(np.hypot(*(b - a)))])
    return lines


def _merge_parallel_lines(lines: list, parallel_threshold: float) -> list:
    """Merge consecutive near-parallel lines whose lateral offset is under the
    threshold (length-weighted midpoint), wrap-around pair included. Lines that
    are parallel but offset further stay separate; the corner rebuild joins
    them with a jog."""
    if not lines:
        return lines
    eps = math.sin(math.radians(1.0))
    merged = [list(lines[0])]
    for mid, u, ln in lines[1:]:
        pm, pu, pl = merged[-1]
        cross = abs(pu[0] * u[1] - pu[1] * u[0])
        if cross < eps:
            dv = mid - pm
            offset = abs(dv[0] * pu[1] - dv[1] * pu[0])
            if offset <= parallel_threshold:
                w = pl + ln
                merged[-1] = [(pm * pl + mid * ln) / w, pu, w]
                continue
        merged.append([mid, u, ln])
    if len(merged) > 1:
        m0, u0, l0 = merged[0]
        m1, u1, l1 = merged[-1]
        cross = abs(u1[0] * u0[1] - u1[1] * u0[0])
        if cross < eps:
            dv = m0 - m1
            offset = abs(dv[0] * u1[1] - dv[1] * u1[0])
            if offset <= parallel_threshold:
                w = l0 + l1
                merged[0] = [(m1 * l1 + m0 * l0) / w, u1, w]
                merged.pop()
    return merged


def ring_rebuild_corners(lines: list, parallel_threshold: float) -> np.ndarray | None:
    """Intersect consecutive snapped lines into a closed ring.

    Near-parallel neighbours, and pairs whose intersection lands far from both
    segments, are joined with a LOCAL JOG between the two segment ends instead
    of the intersection: a foot projection onto a distant line degenerates into
    a long zero-width spike, and the jog is the only join that cannot. Returns
    None when fewer than three corners survive.
    """
    lines = _merge_parallel_lines(lines, parallel_threshold)
    n = len(lines)
    if n < 3:
        return None
    near_parallel = math.sin(math.radians(4.0))
    pts = []
    for i in range(n):
        m1, u1, l1 = lines[i]
        m2, u2, l2 = lines[(i + 1) % n]
        cross = u1[0] * u2[1] - u1[1] * u2[0]
        dv = m2 - m1
        if abs(cross) < near_parallel:
            pts.append(m1 + u1 * (l1 / 2.0))
            pts.append(m2 - u2 * (l2 / 2.0))
            continue
        t1 = (dv[0] * u2[1] - dv[1] * u2[0]) / cross
        inter = m1 + u1 * t1
        # A corner should land near both segments; a very obtuse pair sends the
        # intersection far away and that degenerates into a spike.
        reach = 2.0 * (l1 + l2) + 4.0
        if float(np.hypot(*(inter - m1))) > reach or \
                float(np.hypot(*(inter - m2))) > reach:
            pts.append(m1 + u1 * (l1 / 2.0))
            pts.append(m2 - u2 * (l2 / 2.0))
            continue
        pts.append(inter)
    if len(pts) < 3:
        return None
    return np.asarray(pts + [pts[0]])


def ring_drop_short_edges(coords: np.ndarray, min_edge_abs: float,
                          min_edge_rel: float,
                          min_corner_deg: float) -> np.ndarray:
    """Remove edges under the length floor and corners under the angle floor.

    The floor is the larger of the absolute value and the relative share of
    the ring's perimeter. A removed short edge extends its two neighbour edges
    to their intersection when that lands nearby, else collapses to the edge
    midpoint. Iterates shortest-first until stable (bounded rounds).
    """
    perim = float(np.sum(np.hypot(*(np.diff(coords, axis=0).T))))
    min_edge = max(min_edge_abs, min_edge_rel * perim)
    pts = list(coords[:-1])
    changed = True
    rounds = 0
    while changed and len(pts) > 3 and rounds < 20:
        changed = False
        rounds += 1
        n = len(pts)
        lengths = [float(np.hypot(*(pts[(i + 1) % n] - pts[i])))
                   for i in range(n)]
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
        for i in range(n):
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
    return np.asarray(pts + [pts[0]])


def circle_ring(center_x: float, center_y: float, area: float,
                segments: int = 32) -> np.ndarray:
    """Closed ring of an equal-area circle around a center point."""
    radius = math.sqrt(max(area, 0.0) / math.pi)
    ang = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    pts = np.column_stack([center_x + radius * np.cos(ang),
                           center_y + radius * np.sin(ang)])
    return np.vstack([pts, pts[:1]])
