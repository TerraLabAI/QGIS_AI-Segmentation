"""Diagonals-aware footprint regularizer for man-made shapes.

The per-polygon geometry core here is adapted from Building-Regulariser by
DPIRD-DMA (https://github.com/DPIRD-DMA/Building-Regulariser), used under the
MIT License reproduced below. Only the single-polygon math is vendored: the
upstream geopandas / pandas / multiprocessing driver is dropped, so the sole
runtime dependency is shapely + numpy. No geopandas, no pandas.

Neither is installed by the plugin. numpy comes with QGIS and with the
isolated env used for mask post-processing; shapely comes with QGIS ALONE, and
only when the install carries it (OSGeo4W ships it with the qgis-full meta
package and the standalone installer, not with the bare qgis package). So this
module can be unusable on a healthy install, which is what
dependencies_available() exists to report.

What it does: find a polygon's dominant edge orientation, snap every edge to
parallel or perpendicular (and optional 45-degree diagonals) to that
direction, reconnect corners, optionally fit a circle, all bounded by a
distance tolerance in the geometry's own CRS units. That fixes the case the
native orthogonalize cannot: a genuinely diagonal wall (a 45-degree building
side) is snapped clean instead of left as mask stair steps.

A single dominant direction staircases the walls of an L, U or multi-wing
building and any wall at an arbitrary heading, because every edge is forced
onto the one axis of the whole footprint. An optional multi-direction path
(off by default, turned on by a caller flag) instead clusters the edges into
their own structural directions and snaps each edge to its group's grid, so
each wing keeps its axis. It is contained to the analyze / orient / connect /
enforce steps and reuses the same low-level geometry primitives.

An anti-distortion guard compares the regularized shape to the original by
IoU and reverts to the original when the tolerance could not be met without
wrecking the shape, so a hard case never ships a mangled polygon. On the
multi-direction path a revert returns the de-staircased (simplified) outline
rather than the raw pixel staircase.

Heavy imports (numpy, shapely) are bound lazily on first use, never at module
import, so importing this module at refine time never forces a native import
at plugin load. Best-effort throughout: any failure returns the input
unchanged and never raises.

MIT License

Copyright (c) 2025 DPIRD-DMA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import math
from typing import Any, NamedTuple

# Bound lazily by _ensure_deps() before any numeric helper runs. Kept out of the
# module top so plugin load never forces a native numpy/shapely import.
np: Any = None
Polygon: Any = None
MultiPolygon: Any = None
LinearRing: Any = None
_unary_union: Any = None
_affine_transform: Any = None

# Generic client defaults. One value each, the single fallback when a caller
# does not pass a value (the tuned values live in the server policy, read by
# core.detection_policy.regularize_settings).
_DEFAULT_MIN_KEEP_IOU = 0.70
_DEFAULT_DIAGONAL_REDUCTION = 15.0
_DEFAULT_CIRCLE_THRESHOLD = 0.90
# Below this symmetric-difference share the output is treated as identical to
# the input, so a run that changed nothing is reported as not regularized.
_CHANGED_MIN_FRACTION = 1.0e-3
# ONE generic client fallback for the per-ring revert. A regularized ring
# that overlaps its original this little is not the same shape, so the
# original is kept. The tuned value is server policy.
_RING_MIN_IOU = 0.1
# Under this much difference from 1, a unit aspect is treated as square ground
# (see _regularize_geometry_isotropic) and no frame change runs at all. A
# correction this small moves a corner by well under a degree, which no outline
# shows, and the dead band keeps a projected run's shapes untouched: the
# ellipsoidal measure answers a hair off 1 there, and a hair is not a reason to
# put every object through a frame change.
_ASPECT_IDENTITY_EPSILON = 0.01

# Multi-direction path defaults. Off by default, so a caller that passes nothing
# gets exactly the single dominant-direction behaviour. When on, edges are
# clustered into their own structural directions and each edge snaps to ITS
# group's grid, so an L or multi-wing building keeps each wing's axis instead of
# staircasing the walls off the one axis of the whole footprint. The tuned
# on/off and thresholds live in the server policy and arrive as kwargs.
_DEFAULT_MULTI_DIRECTION = False
_DEFAULT_MULTI_MAX_GROUPS = 3
_DEFAULT_MULTI_MIN_SEPARATION_DEG = 10.0
# Two oriented edges within this angular gap (degrees, folded to a line) are
# treated as parallel by the multi-direction corner reconnect.
_MULTI_PARALLEL_ANGLE_EPS = 1.5
# A second (or third) structural direction is only kept when the edges near it
# hold at least this share of the whole ring's perimeter. Without it the
# clustering promotes noise peaks on a near-single-direction footprint into
# spurious extra grids, which staircases a shape the single path handles clean.
# The top direction is always kept, so a genuinely one-grid building collapses
# to the single-direction path and its output is unchanged. A generic guard,
# not a primary tuning dial.
_MULTI_MIN_GROUP_WEIGHT_FRACTION = 0.20


class RegularizeResult(NamedTuple):
    """Outcome of a regularize call.

    geometry: the geometry to use (regularized, or the original on revert).
    regularized: True when the shape was actually changed and kept.
    reverted: True when regularization ran but the guard rejected it and the
        original was kept.
    """

    geometry: Any
    regularized: bool
    reverted: bool


class RegularizePolicy(NamedTuple):
    """Extra guards layered on the regularizer, all OFF/neutral by default so an
    absent policy reproduces the IoU-only behaviour exactly. The tuned values
    live in the server policy (core.detection_policy) and arrive here already
    validated. A field left at its neutral default disables that check."""
    # Safety envelope (candidate/reject the regularized shape)
    envelope_enabled: bool = False
    max_area_ratio: float = 0.0        # reject if reg.area/orig.area above this (0 = off)
    min_area_ratio: float = 0.0        # reject if reg.area/orig.area below this (0 = off)
    max_hausdorff_mult: float = 0.0    # reject if boundary displacement > this * tolerance (0 = off)
    max_vertex_growth: float = 0.0     # reject if reg vertices > orig vertices * this (0 = off)
    enforce_component_count: bool = False  # reject if exterior part count changed
    enforce_hole_count: bool = False       # reject if interior-ring count changed
    # Eligibility gate (skip regularizing shapes that are not plausible footprints)
    eligibility_enabled: bool = False
    min_rectangularity: float = 0.0    # require area/convex_hull_area >= this (0 = off)
    max_holes: int = -1                # only regularize shapes with <= this many holes (-1 = off)
    # Conditioned rectangle substitution (replace a clean rectangle with its OMBB)
    rectangle_enabled: bool = False
    rectangle_area_fill: float = 0.95  # trigger when area/OMBB.area >= this
    rectangle_min_aspect: float = 1.2  # AND OMBB long/short side ratio >= this


_NEUTRAL_POLICY = RegularizePolicy()


def _ensure_deps() -> bool:
    """Bind numpy + shapely names as module globals on first use. Returns False
    when either import fails (the caller then returns the input unchanged)."""
    global np, Polygon, MultiPolygon, LinearRing, _unary_union
    global _affine_transform
    if np is not None and Polygon is not None:
        return True
    try:
        import numpy as _numpy
        from shapely.affinity import affine_transform as _affine
        from shapely.geometry import LinearRing as _LinearRing
        from shapely.geometry import MultiPolygon as _MultiPolygon
        from shapely.geometry import Polygon as _Polygon
        from shapely.ops import unary_union as _uu
    except Exception:  # noqa: BLE001 -- deps optional  # nosec B110
        return False
    np = _numpy
    Polygon = _Polygon
    MultiPolygon = _MultiPolygon
    LinearRing = _LinearRing
    _unary_union = _uu
    _affine_transform = _affine
    return True


def dependencies_available() -> bool:
    """True when this module can actually regularize anything.

    Every public entry point here returns its input unchanged when
    ``_ensure_deps`` fails, and the callers keep that input: the user ticks
    "Right angles", waits for the slowest control in the panel, and gets an
    outline nothing squared, with nothing to tell it apart from the
    anti-distortion guard declining one shape. A UI can ask this first and
    refuse the control out loud instead.

    The test is the import itself, which is the only one that catches a shapely
    whose native GEOS library will not load. numpy ships with QGIS and with the
    plugin's own env; shapely ships with QGIS ALONE (it is not in
    venv_manager.REQUIRED_PACKAGES), so a QGIS build without it leaves this
    False and no plugin install fixes it.

    Not cached: a False can become True later in the session, when the plugin's
    env lands on sys.path. Call it when the control is reached, never at plugin
    load, or the lazy binding this module exists for is defeated.
    """
    return _ensure_deps()


# --------------------------------------------------------------------------
# Geometry math helpers (adapted from buildingregulariser.geometry_utils).
# --------------------------------------------------------------------------


def calculate_distance(point_1: Any, point_2: Any) -> float:
    """Euclidean distance between two points."""
    dx, dy = point_1[0] - point_2[0], point_1[1] - point_2[1]
    return math.hypot(dx, dy)


def calculate_azimuth_angle(start_point: Any, end_point: Any) -> float:
    """Azimuth of the line start->end in degrees, in [0, 360)."""
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    return math.degrees(math.atan2(dy, dx)) % 360


def create_line_equation(point1: Any, point2: Any) -> tuple[float, float, float]:
    """Line through two points as (A, B, C) with Ax + By + C = 0."""
    a = point1[1] - point2[1]
    b = point2[0] - point1[0]
    c = point1[0] * point2[1] - point2[0] * point1[1]
    return a, b, -c


def calculate_line_intersection(
    line1: tuple[float, float, float], line2: tuple[float, float, float]
) -> tuple[float, float] | None:
    """Intersection of two lines, or None when they are parallel."""
    d = line1[0] * line2[1] - line1[1] * line2[0]
    dx = line1[2] * line2[1] - line1[1] * line2[2]
    dy = line1[0] * line2[2] - line1[2] * line2[0]
    if d != 0:
        return dx / d, dy / d
    return None


def calculate_parallel_line_distance(
    line1: tuple[float, float, float], line2: tuple[float, float, float]
) -> float:
    """Distance between two near-parallel lines."""
    a1, _, c1 = line1
    a2, b2, c2 = line2
    eps = 1e-10
    new_c1 = c1 / (a1 + eps)
    new_b2 = b2 / (a2 + eps)
    new_c2 = c2 / (a2 + eps)
    return abs(new_c1 - new_c2) / math.sqrt(1.0 + new_b2 * new_b2)


def project_point_to_line(
    point_x: float,
    point_y: float,
    line_x1: float,
    line_y1: float,
    line_x2: float,
    line_y2: float,
) -> tuple[float, float]:
    """Project a point onto the line through two points."""
    eps = 1e-10
    dx = line_x2 - line_x1
    dy = line_y2 - line_y1
    denom = dx * dx + dy * dy + eps
    x = (
        point_x * dx * dx + point_y * dy * dx + (line_x1 * line_y2 - line_x2 * line_y1) * dy
    ) / denom
    y = (
        point_x * dx * dy + point_y * dy * dy + (line_x2 * line_y1 - line_x1 * line_y2) * dx
    ) / denom
    return (x, y)


def rotate_point(point: Any, center: Any, angle_degrees: float) -> tuple[float, float]:
    """Rotate a point clockwise around a center by angle_degrees."""
    x, y = point
    center_x, center_y = center
    angle_radians = math.radians(angle_degrees)
    tx = x - center_x
    ty = y - center_y
    rx = tx * math.cos(angle_radians) + ty * math.sin(angle_radians)
    ry = ty * math.cos(angle_radians) - tx * math.sin(angle_radians)
    return (rx + center_x, ry + center_y)


def rotate_edge(start_point: Any, end_point: Any, rotation_angle: float) -> list[Any]:
    """Rotate an edge around its midpoint by rotation_angle degrees."""
    midpoint = (start_point + end_point) / 2
    if rotation_angle > 0:
        rotated_start = rotate_point(start_point, midpoint, -rotation_angle)
        rotated_end = rotate_point(end_point, midpoint, -rotation_angle)
    elif rotation_angle < 0:
        rotated_start = rotate_point(start_point, midpoint, abs(rotation_angle))
        rotated_end = rotate_point(end_point, midpoint, abs(rotation_angle))
    else:
        rotated_start = (float(start_point[0]), float(start_point[1]))
        rotated_end = (float(end_point[0]), float(end_point[1]))
    return [np.array(rotated_start), np.array(rotated_end)]


# --------------------------------------------------------------------------
# Regularization core (adapted from buildingregulariser.regularization).
# --------------------------------------------------------------------------


def find_nearest_target_angle(
    current_azimuth: float, main_direction: float, allow_45_degree: bool
) -> float:
    """Closest allowed target azimuth (0-360) for one edge."""
    diff_angle = (current_azimuth - main_direction + 180) % 360 - 180
    if allow_45_degree:
        allowed_offsets = [0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0]
    else:
        allowed_offsets = [0.0, 90.0, 180.0, -90.0]
    best_offset = 0.0
    min_angle_dist = 181.0
    for offset in allowed_offsets:
        d = (diff_angle - offset + 180) % 360 - 180
        if abs(d) < min_angle_dist:
            min_angle_dist = abs(d)
            best_offset = offset
    return (main_direction + best_offset + 360) % 360


def enforce_angles_post_process(
    points: list[Any],
    main_direction: int,
    allow_45_degree: bool,
    angle_tolerance: float = 0.1,
    max_iterations: int = 2,
) -> list[Any]:
    """Nudge vertices iteratively so each segment hits its target angle.

    Adjusting one segment shifts its neighbour, so several passes run until no
    segment moves more than angle_tolerance. Input points are the open ring
    (last point != first), length >= 3.
    """
    if len(points) < 3:
        return points
    adjusted_points = [p.copy() for p in points]
    num_points = len(adjusted_points)
    for _ in range(max_iterations):
        changed = False
        for i in range(num_points):
            p1 = adjusted_points[i]
            p2_idx = (i + 1) % num_points
            p2 = adjusted_points[p2_idx]
            if calculate_distance(p1, p2) < 1e-7:
                continue
            current_azimuth = calculate_azimuth_angle(p1, p2)
            target_azimuth = find_nearest_target_angle(
                current_azimuth, main_direction, allow_45_degree
            )
            rotation_diff = (target_azimuth - current_azimuth + 180) % 360 - 180
            if abs(rotation_diff) > angle_tolerance:
                changed = True
                if rotation_diff > 0:
                    new_p2 = rotate_point(p2, p1, -rotation_diff)
                else:
                    new_p2 = rotate_point(p2, p1, abs(rotation_diff))
                adjusted_points[p2_idx] = np.array(new_p2)
        if not changed:
            break
    return adjusted_points


def _create_weighted_histogram(
    angles: Any, bin_size: float, weights: Any, num_bins_override=None, smooth=True
) -> Any:
    num_bins = int(90 / bin_size) if num_bins_override is None else num_bins_override
    indices = np.minimum(np.floor(angles / bin_size).astype(int), num_bins - 1)
    bins = np.bincount(indices, weights=weights, minlength=num_bins)
    if smooth:
        bins = _smooth_histogram(bins)
    return bins


def _smooth_histogram(hist: Any) -> Any:
    smoothed = hist.copy()
    for i in range(1, len(hist) - 1):
        smoothed[i] = (2 * hist[i] + hist[i - 1] + hist[i + 1]) / 4
    smoothed[0] = (2 * hist[0] + hist[1]) / 3
    smoothed[-1] = (2 * hist[-1] + hist[-2]) / 3
    return smoothed


def _find_best_symmetric_bin(hist: Any) -> int:
    """Dominant histogram bin, averaging the histogram with its mirror so the
    orientation and its perpendicular reinforce each other."""
    mirrored_mean = (hist + hist[::-1]) / 2
    sorted_indices = np.argsort(mirrored_mean)
    a, b = sorted_indices[-2:]
    return a if hist[a] > hist[b] else b


def analyze_edges(
    coordinates: Any, coarse_bin_size: int = 5, fine_bin_size: int = 1
) -> dict[str, Any]:
    """Edge azimuths, edge index pairs, and the dominant structural direction."""
    if len(coordinates) < 3:
        return {
            "azimuth_angles": np.array([]),
            "edge_indices": np.array([]),
            "main_direction": 0,
        }
    start_points = coordinates
    end_points = np.roll(coordinates, -1, axis=0)
    vectors = end_points - start_points
    edge_lengths = np.linalg.norm(vectors, axis=1)
    valid = edge_lengths > 1e-9
    if not np.any(valid):
        return {
            "azimuth_angles": np.array([]),
            "edge_indices": np.array([]),
            "main_direction": 0,
        }
    vectors = vectors[valid]
    lengths = edge_lengths[valid]
    azimuth_angles = (np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) + 360) % 360
    normalized_angles = azimuth_angles % 180
    orthogonal_angles = normalized_angles % 90
    indices = np.stack(
        [
            np.arange(len(coordinates)),
            (np.arange(len(coordinates)) + 1) % len(coordinates),
        ],
        axis=1,
    )
    edge_indices = indices[valid]
    coarse_bins = _create_weighted_histogram(orthogonal_angles, coarse_bin_size, lengths)
    fine_bins = _create_weighted_histogram(
        orthogonal_angles, fine_bin_size, lengths, num_bins_override=90
    )
    if np.sum(coarse_bins) == 0:
        refined_angle = 0
    else:
        main_bin = _find_best_symmetric_bin(coarse_bins)
        fine_start = main_bin * coarse_bin_size
        fine_end = fine_start + coarse_bin_size
        refined_bin = _find_best_symmetric_bin(fine_bins[fine_start:fine_end])
        refined_angle_center = fine_start + refined_bin + fine_bin_size / 2
        if refined_bin == 0:
            refined_angle = math.floor(refined_angle_center)
        elif refined_bin == (fine_end - fine_start - 1):
            refined_angle = math.ceil(refined_angle_center)
        else:
            left = fine_bins[fine_start + refined_bin - 1]
            right = fine_bins[fine_start + refined_bin + 1]
            refined_angle = (
                math.ceil(refined_angle_center)
                if right > left
                else math.floor(refined_angle_center)
            )
    return {
        "azimuth_angles": azimuth_angles,
        "edge_indices": edge_indices,
        "main_direction": refined_angle,
    }


def get_orientation_and_rotation(
    diff_angle: float,
    main_direction: float,
    azimuth: float,
    allow_45_degree: bool,
    diagonal_threshold_reduction: float,
    tolerance: float = 1e-9,
) -> tuple[int, float]:
    """Target orientation code + the rotation (degrees) to reach it for one edge.

    diagonal_threshold_reduction shrinks the angular zone that snaps to 45
    degrees, so larger values make diagonal edges less likely (range 0-22.5).
    """
    target_offset = 0.0
    orientation_code = 0
    if allow_45_degree:
        mod180 = diff_angle % 180
        dist_to_0 = min(abs(mod180), abs(mod180 - 180))
        dist_to_90 = abs(mod180 - 90)
        dist_to_45 = min(abs(mod180 - 45), abs(mod180 - 135))
        if dist_to_45 <= (22.5 - diagonal_threshold_reduction):
            angle_mod = diff_angle % 90
            if angle_mod < 45:
                target_offset = (diff_angle // 90) * 90 + 45
            else:
                target_offset = (diff_angle // 90 + 1) * 90 - 45
            normalized_angle = (main_direction + target_offset) % 180
            orientation_code = 2 if 0 <= normalized_angle < 90 else 3
        elif dist_to_0 <= dist_to_90:
            target_offset = round(diff_angle / 180.0) * 180.0
            orientation_code = 0
        else:
            target_offset = round(diff_angle / 90.0) * 90.0
            if abs(target_offset % 180) < tolerance:
                target_offset = 90.0 if diff_angle > 0 else -90.0
            orientation_code = 1
    else:
        if abs(diff_angle) < 45.0:
            target_offset = round(diff_angle / 180.0) * 180.0
            orientation_code = 0
        else:
            target_offset = round(diff_angle / 90.0) * 90.0
            if abs(target_offset % 180) < tolerance:
                target_offset = 90.0 if diff_angle > 0 else -90.0
            orientation_code = 1
    rotation_angle = (main_direction + target_offset - azimuth + 180) % 360 - 180
    return orientation_code, rotation_angle


def orient_edges(
    simplified_coordinates: Any,
    edge_data: dict,
    allow_45_degree: bool,
    diagonal_threshold_reduction: float,
) -> tuple[Any, list[int]]:
    """Rotate each edge onto its target orientation relative to main direction.

    Orientation codes: 0 parallel, 1 perpendicular, 2/3 the two diagonals
    (only when allow_45_degree).
    """
    oriented_edges = []
    edge_orientations: list[int] = []
    azimuth_angles = edge_data["azimuth_angles"]
    edge_indices = edge_data["edge_indices"]
    main_direction = edge_data["main_direction"]
    for azimuth, (start_idx, end_idx) in zip(azimuth_angles, edge_indices):
        diff_angle = (azimuth - main_direction + 180) % 360 - 180
        orientation_code, rotation_angle = get_orientation_and_rotation(
            diff_angle=diff_angle,
            main_direction=main_direction,
            azimuth=azimuth,
            allow_45_degree=allow_45_degree,
            diagonal_threshold_reduction=diagonal_threshold_reduction,
        )
        start_point = np.array(simplified_coordinates[start_idx], dtype=float)
        end_point = np.array(simplified_coordinates[end_idx], dtype=float)
        rotated_edge = rotate_edge(start_point, end_point, rotation_angle)
        oriented_edges.append(rotated_edge)
        edge_orientations.append(orientation_code)
    return np.array(oriented_edges, dtype=float), edge_orientations


def handle_perpendicular_edges(
    current_edge_start: Any,
    current_edge_end: Any,
    next_edge_start: Any,
    next_edge_end: Any,
) -> Any:
    """Corner between two non-parallel edges: their line intersection."""
    line1 = create_line_equation(current_edge_start, current_edge_end)
    line2 = create_line_equation(next_edge_start, next_edge_end)
    intersection_point = calculate_line_intersection(line1, line2)
    if intersection_point:
        return np.array(intersection_point)
    return current_edge_end


def handle_parallel_edges(
    current_edge_start: Any,
    current_edge_end: Any,
    next_edge_start: Any,
    next_edge_end: Any,
    parallel_threshold: float,
    next_index: int,
    oriented_edges: Any,
) -> list[Any]:
    """Join two same-orientation edges: shift the next edge onto the current
    line when they are within parallel_threshold, else add a short connecting
    step between them."""
    line1 = create_line_equation(current_edge_start, current_edge_end)
    line2 = create_line_equation(next_edge_start, next_edge_end)
    line_distance = calculate_parallel_line_distance(line1, line2)
    new_points = []
    if line_distance < parallel_threshold:
        projected_point = project_point_to_line(
            next_edge_start[0],
            next_edge_start[1],
            current_edge_start[0],
            current_edge_start[1],
            current_edge_end[0],
            current_edge_end[1],
        )
        new_points.append(np.array(projected_point))
        oriented_edges[next_index][0] = np.array(projected_point)
        oriented_edges[next_index][1] = np.array(
            project_point_to_line(
                next_edge_end[0],
                next_edge_end[1],
                current_edge_start[0],
                current_edge_start[1],
                current_edge_end[0],
                current_edge_end[1],
            )
        )
    else:
        midpoint = (current_edge_end + next_edge_start) / 2
        connecting_point1 = project_point_to_line(
            midpoint[0],
            midpoint[1],
            current_edge_start[0],
            current_edge_start[1],
            current_edge_end[0],
            current_edge_end[1],
        )
        connecting_point2 = project_point_to_line(
            midpoint[0],
            midpoint[1],
            next_edge_start[0],
            next_edge_start[1],
            next_edge_end[0],
            next_edge_end[1],
        )
        new_points.append(np.array(connecting_point1))
        new_points.append(np.array(connecting_point2))
    return new_points


def connect_regularized_edges(
    oriented_edges: Any, edge_orientations: list, parallel_threshold: float
) -> list[Any]:
    """Reconnect the oriented edges into a closed ring of corner points."""
    regularized_points: list[Any] = []
    for i in range(len(oriented_edges)):
        next_index = (i + 1) % len(oriented_edges)
        current_edge_start = oriented_edges[i][0]
        current_edge_end = oriented_edges[i][1]
        next_edge_start = oriented_edges[next_index][0]
        next_edge_end = oriented_edges[next_index][1]
        current_orientation = edge_orientations[i]
        next_orientation = edge_orientations[next_index]
        if current_orientation != next_orientation:
            regularized_points.append(
                handle_perpendicular_edges(
                    current_edge_start, current_edge_end, next_edge_start, next_edge_end
                )
            )
        else:
            regularized_points.extend(
                handle_parallel_edges(
                    current_edge_start,
                    current_edge_end,
                    next_edge_start,
                    next_edge_end,
                    parallel_threshold,
                    next_index,
                    oriented_edges,
                )
            )
    return regularized_points


def regularize_coordinate_array(
    coordinates: Any,
    parallel_threshold: float,
    allow_45_degree: bool,
    diagonal_threshold_reduction: float,
    angle_enforcement_tolerance: float = 0.1,
) -> tuple[Any, float]:
    """Regularize one closed ring: align edges to the main direction, reconnect
    corners, then enforce the target angles. Returns (closed coords, main dir).
    Falls back to the input coords on any degenerate step."""
    if len(coordinates) < 4:
        return coordinates, 0.0
    if np.allclose(coordinates[0], coordinates[-1]):
        processing_coords = coordinates[:-1]
    else:
        processing_coords = coordinates
    if len(processing_coords) < 3:
        return coordinates, 0.0
    edge_data = analyze_edges(processing_coords)
    oriented_edges, edge_orientations = orient_edges(
        processing_coords,
        edge_data,
        allow_45_degree=allow_45_degree,
        diagonal_threshold_reduction=diagonal_threshold_reduction,
    )
    initial_points = connect_regularized_edges(
        oriented_edges, edge_orientations, parallel_threshold
    )
    if not initial_points or len(initial_points) < 3:
        return coordinates, 0.0
    final_points = enforce_angles_post_process(
        points=initial_points,
        main_direction=edge_data["main_direction"],
        allow_45_degree=allow_45_degree,
        angle_tolerance=angle_enforcement_tolerance,
    )
    if not final_points or len(final_points) < 3:
        return coordinates, 0.0
    final_coords_array = np.array(list(final_points))
    closed_final_coords = np.vstack([final_coords_array, final_coords_array[0]])
    return closed_final_coords, edge_data["main_direction"]


# --------------------------------------------------------------------------
# Multi-direction path: cluster edges into their own structural directions and
# snap each edge to ITS group's grid (the IRMBR / per-edge-orientation line,
# arXiv:2006.13176 and arXiv:2007.12587). Contained to the analyze / orient /
# connect / enforce steps; the low-level primitives are shared with the
# single-direction path. Gated by a flag that defaults off.
# --------------------------------------------------------------------------


def _circular_dist_mod(a: float, b: float, period: float) -> float:
    """Shortest distance between two angles on a circle of the given period."""
    d = abs(a - b) % period
    return min(d, period - d)


def _refine_direction(
    folded_angles: Any, lengths: Any, center_bin: int, window: float
) -> float:
    """Length-weighted circular mean (mod 90) of the edges near a histogram
    peak, so a group direction is not stuck on a one-degree bin edge."""
    center = center_bin + 0.5
    sum_x = 0.0
    sum_y = 0.0
    total = 0.0
    for angle, weight in zip(folded_angles, lengths):
        if _circular_dist_mod(float(angle), center, 90.0) <= window:
            # Fold [0, 90) onto a full circle (x4) so the mean wraps correctly.
            radians = math.radians(float(angle) * 4.0)
            sum_x += float(weight) * math.cos(radians)
            sum_y += float(weight) * math.sin(radians)
            total += float(weight)
    if total <= 0.0:
        return center % 90.0
    return (math.degrees(math.atan2(sum_y, sum_x)) / 4.0) % 90.0


def _cluster_directions(
    azimuth_angles: Any, lengths: Any, max_groups: int, min_separation_deg: float,
    min_group_weight: float = _MULTI_MIN_GROUP_WEIGHT_FRACTION,
) -> list[float]:
    """The strongest few structural directions in a ring, in [0, 90).

    Builds a length-weighted, circularly smoothed 1-degree histogram of the
    edge angles folded to [0, 90), then greedily keeps the heaviest local peaks
    that sit at least min_separation_deg apart, up to max_groups. Each kept
    peak is refined to the weighted circular mean of its nearby edges.
    ``min_group_weight`` is the share of the ring's perimeter a second or third
    direction must carry to be kept at all.
    """
    folded = np.mod(azimuth_angles, 90.0)
    bins = np.zeros(90, dtype=float)
    indices = np.minimum(np.floor(folded).astype(int), 89)
    for idx, weight in zip(indices, lengths):
        bins[int(idx)] += float(weight)
    if bins.sum() <= 0.0:
        return []
    smoothed = bins.copy()
    for i in range(90):
        smoothed[i] = (
            2.0 * bins[i] + bins[(i - 1) % 90] + bins[(i + 1) % 90]
        ) / 4.0
    peaks = [
        i
        for i in range(90)
        if smoothed[i] > 0.0 and smoothed[i] >= smoothed[(i - 1) % 90] and smoothed[i] >= smoothed[(i + 1) % 90]
    ]
    peaks.sort(key=lambda i: smoothed[i], reverse=True)
    total_weight = float(np.sum(lengths))

    def _weight_near(peak: int) -> float:
        return float(
            np.sum(
                [
                    w
                    for a, w in zip(folded, lengths)
                    if _circular_dist_mod(float(a), peak + 0.5, 90.0) <= min_separation_deg
                ]
            )
        )

    min_weight = min_group_weight * total_weight
    chosen: list[int] = []
    for peak in peaks:
        if any(
            _circular_dist_mod(peak, other, 90.0) < min_separation_deg
            for other in chosen
        ):
            continue
        # The top direction always survives; a second or third only when it
        # carries a real share of the perimeter, never a noise peak.
        if chosen and _weight_near(peak) < min_weight:
            continue
        chosen.append(peak)
        if len(chosen) >= max_groups:
            break
    return [
        _refine_direction(folded, lengths, peak, min_separation_deg)
        for peak in chosen
    ]


def _analyze_edges_multi(
    coordinates: Any, max_groups: int, min_separation_deg: float,
    min_group_weight: float = _MULTI_MIN_GROUP_WEIGHT_FRACTION,
) -> dict[str, Any]:
    """Edge azimuths + index pairs, the kept group directions, and each edge's
    assigned group direction (its nearest structural axis, mod 90)."""
    start_points = coordinates
    end_points = np.roll(coordinates, -1, axis=0)
    vectors = end_points - start_points
    edge_lengths = np.linalg.norm(vectors, axis=1)
    valid = edge_lengths > 1e-9
    empty = {
        "azimuth_angles": np.array([]),
        "edge_indices": np.array([]),
        "edge_directions": np.array([]),
        "group_dirs": [],
    }
    if not np.any(valid):
        return empty
    vectors = vectors[valid]
    lengths = edge_lengths[valid]
    azimuth_angles = (np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) + 360) % 360
    indices = np.stack(
        [
            np.arange(len(coordinates)),
            (np.arange(len(coordinates)) + 1) % len(coordinates),
        ],
        axis=1,
    )
    edge_indices = indices[valid]
    group_dirs = _cluster_directions(
        azimuth_angles, lengths, max_groups, min_separation_deg, min_group_weight
    )
    if not group_dirs:
        return empty
    folded = np.mod(azimuth_angles, 90.0)
    edge_directions = np.array(
        [
            min(group_dirs, key=lambda d: _circular_dist_mod(float(a), d, 90.0))
            for a in folded
        ],
        dtype=float,
    )
    return {
        "azimuth_angles": azimuth_angles,
        "edge_indices": edge_indices,
        "edge_directions": edge_directions,
        "group_dirs": group_dirs,
    }


def _orient_edges_multi(
    simplified_coordinates: Any,
    edge_data: dict,
    allow_45_degree: bool,
    diagonal_threshold_reduction: float,
) -> tuple[Any, list[int], list[float]]:
    """Rotate each edge onto its own group's target orientation. Returns the
    oriented edges, their orientation codes, and their final target azimuths
    (used by the reconnect to tell actually-parallel edges from corners)."""
    oriented_edges = []
    edge_orientations: list[int] = []
    target_azimuths: list[float] = []
    azimuth_angles = edge_data["azimuth_angles"]
    edge_indices = edge_data["edge_indices"]
    edge_directions = edge_data["edge_directions"]
    for azimuth, (start_idx, end_idx), main_direction in zip(
        azimuth_angles, edge_indices, edge_directions
    ):
        diff_angle = (azimuth - main_direction + 180) % 360 - 180
        orientation_code, rotation_angle = get_orientation_and_rotation(
            diff_angle=diff_angle,
            main_direction=main_direction,
            azimuth=azimuth,
            allow_45_degree=allow_45_degree,
            diagonal_threshold_reduction=diagonal_threshold_reduction,
        )
        start_point = np.array(simplified_coordinates[start_idx], dtype=float)
        end_point = np.array(simplified_coordinates[end_idx], dtype=float)
        rotated_edge = rotate_edge(start_point, end_point, rotation_angle)
        oriented_edges.append(rotated_edge)
        edge_orientations.append(orientation_code)
        target_azimuths.append((azimuth + rotation_angle) % 360)
    return np.array(oriented_edges, dtype=float), edge_orientations, target_azimuths


def _connect_regularized_edges_multi(
    oriented_edges: Any, target_azimuths: list[float], parallel_threshold: float,
    parallel_eps_deg: float = _MULTI_PARALLEL_ANGLE_EPS,
) -> list[Any]:
    """Reconnect oriented edges into a ring, deciding parallel vs corner from
    the edges' actual target azimuths rather than group-relative codes, so two
    edges from different groups meet at a clean line intersection."""
    regularized_points: list[Any] = []
    count = len(oriented_edges)
    for i in range(count):
        next_index = (i + 1) % count
        current_edge_start = oriented_edges[i][0]
        current_edge_end = oriented_edges[i][1]
        next_edge_start = oriented_edges[next_index][0]
        next_edge_end = oriented_edges[next_index][1]
        if (
            _circular_dist_mod(target_azimuths[i], target_azimuths[next_index], 180.0) < parallel_eps_deg
        ):
            regularized_points.extend(
                handle_parallel_edges(
                    current_edge_start,
                    current_edge_end,
                    next_edge_start,
                    next_edge_end,
                    parallel_threshold,
                    next_index,
                    oriented_edges,
                )
            )
        else:
            regularized_points.append(
                handle_perpendicular_edges(
                    current_edge_start, current_edge_end, next_edge_start, next_edge_end
                )
            )
    return regularized_points


def _nearest_target_multi(
    current_azimuth: float, group_dirs: list[float], allow_45_degree: bool
) -> float:
    """Closest allowed target azimuth across every group's grid."""
    best_target = current_azimuth
    best_dist = 361.0
    for main_direction in group_dirs:
        target = find_nearest_target_angle(
            current_azimuth, main_direction, allow_45_degree
        )
        dist = abs((target - current_azimuth + 180) % 360 - 180)
        if dist < best_dist:
            best_dist = dist
            best_target = target
    return best_target


def _enforce_angles_multi(
    points: list[Any],
    group_dirs: list[float],
    allow_45_degree: bool,
    angle_tolerance: float = 0.1,
    max_iterations: int = 2,
) -> list[Any]:
    """Nudge each segment onto the nearest target across all group grids. Same
    shape as enforce_angles_post_process, but the target is chosen per segment
    from every kept direction rather than from one main direction."""
    if len(points) < 3:
        return points
    adjusted_points = [p.copy() for p in points]
    num_points = len(adjusted_points)
    for _ in range(max_iterations):
        changed = False
        for i in range(num_points):
            p1 = adjusted_points[i]
            p2_idx = (i + 1) % num_points
            p2 = adjusted_points[p2_idx]
            if calculate_distance(p1, p2) < 1e-7:
                continue
            current_azimuth = calculate_azimuth_angle(p1, p2)
            target_azimuth = _nearest_target_multi(
                current_azimuth, group_dirs, allow_45_degree
            )
            rotation_diff = (target_azimuth - current_azimuth + 180) % 360 - 180
            if abs(rotation_diff) > angle_tolerance:
                changed = True
                if rotation_diff > 0:
                    new_p2 = rotate_point(p2, p1, -rotation_diff)
                else:
                    new_p2 = rotate_point(p2, p1, abs(rotation_diff))
                adjusted_points[p2_idx] = np.array(new_p2)
        if not changed:
            break
    return adjusted_points


def _ring_min_iou() -> float:
    """IoU under which one regularized RING is thrown away and the original
    kept, resolved from the cached server policy when present.

    Not the same guard as ``min_keep_iou``, which is applied later and to the
    whole geometry. This one reverts per ring, silently: if it sits too high
    the regularizer declines every complex footprint and the user sees Right
    angles do nothing, with no message saying why. That is the failure this
    dial exists to be able to correct without a release.

    Cache-only, never raises, so a failure keeps the built-in value and the
    geometry is unchanged.
    """
    try:
        from .detection_policy import regularize_settings

        value = float(regularize_settings()["ring_min_iou"])
        return value if 0.0 < value < 1.0 else _RING_MIN_IOU
    except Exception:  # noqa: BLE001 - policy is optional, never break geometry
        return _RING_MIN_IOU


def _multi_clustering_dials() -> tuple[float, float]:
    """(parallel angle epsilon, minimum group weight) for the multi-direction
    clustering, resolved from the cached server policy when present, else the
    built-in defaults.

    The on/off switch and the other two multi dials already arrive as kwargs
    from the same policy block; these two are read here because the callers in
    between do not carry them. Cache-only (no network) and only reached on the
    multi path, which is off unless the server turns it on; any failure keeps
    the built-in values, so the geometry is unchanged.
    """
    try:
        from .detection_policy import regularize_settings
        settings = regularize_settings()
        return (
            float(settings["multi_parallel_eps_deg"]),
            float(settings["multi_min_group_weight"]),
        )
    except Exception:  # noqa: BLE001 - policy is optional, never break geometry
        return (_MULTI_PARALLEL_ANGLE_EPS, _MULTI_MIN_GROUP_WEIGHT_FRACTION)


def regularize_coordinate_array_multi(
    coordinates: Any,
    parallel_threshold: float,
    allow_45_degree: bool,
    diagonal_threshold_reduction: float,
    max_groups: int,
    min_separation_deg: float,
    angle_enforcement_tolerance: float = 0.1,
) -> tuple[Any, float]:
    """Multi-direction regularization of one closed ring.

    Clusters the ring's edges into up to max_groups structural directions and
    snaps each edge to its own group's grid. When only one direction survives
    the clustering (a simple rectangle), delegates to the single-direction path
    so the output is identical. Falls back to the input coords on any
    degenerate step.
    """
    if len(coordinates) < 4:
        return coordinates, 0.0
    if np.allclose(coordinates[0], coordinates[-1]):
        processing_coords = coordinates[:-1]
    else:
        processing_coords = coordinates
    if len(processing_coords) < 3:
        return coordinates, 0.0
    parallel_eps_deg, min_group_weight = _multi_clustering_dials()
    edge_data = _analyze_edges_multi(
        processing_coords, max_groups, min_separation_deg, min_group_weight
    )
    group_dirs = edge_data["group_dirs"]
    if len(group_dirs) <= 1:
        return regularize_coordinate_array(
            coordinates=coordinates,
            parallel_threshold=parallel_threshold,
            allow_45_degree=allow_45_degree,
            diagonal_threshold_reduction=diagonal_threshold_reduction,
            angle_enforcement_tolerance=angle_enforcement_tolerance,
        )
    oriented_edges, _orientations, target_azimuths = _orient_edges_multi(
        processing_coords,
        edge_data,
        allow_45_degree=allow_45_degree,
        diagonal_threshold_reduction=diagonal_threshold_reduction,
    )
    initial_points = _connect_regularized_edges_multi(
        oriented_edges, target_azimuths, parallel_threshold, parallel_eps_deg
    )
    if not initial_points or len(initial_points) < 3:
        return coordinates, 0.0
    final_points = _enforce_angles_multi(
        points=initial_points,
        group_dirs=group_dirs,
        allow_45_degree=allow_45_degree,
        angle_tolerance=angle_enforcement_tolerance,
    )
    if not final_points or len(final_points) < 3:
        return coordinates, 0.0
    final_coords_array = np.array(list(final_points))
    closed_final_coords = np.vstack([final_coords_array, final_coords_array[0]])
    return closed_final_coords, group_dirs[0]


def _regularize_one_ring(
    coordinates: Any,
    parallel_threshold: float,
    allow_45_degree: bool,
    diagonal_threshold_reduction: float,
    multi_direction: bool,
    multi_max_groups: int,
    multi_min_separation_deg: float,
) -> tuple[Any, float]:
    """Dispatch one ring to the multi-direction or single-direction path."""
    if multi_direction:
        return regularize_coordinate_array_multi(
            coordinates=coordinates,
            parallel_threshold=parallel_threshold,
            allow_45_degree=allow_45_degree,
            diagonal_threshold_reduction=diagonal_threshold_reduction,
            max_groups=multi_max_groups,
            min_separation_deg=multi_min_separation_deg,
        )
    return regularize_coordinate_array(
        coordinates=coordinates,
        parallel_threshold=parallel_threshold,
        allow_45_degree=allow_45_degree,
        diagonal_threshold_reduction=diagonal_threshold_reduction,
    )


def preprocess_polygon(polygon: Any, simplify: bool, simplify_tolerance: float) -> Any:
    """De-staircase a raw mask outline: simplify then densify so the edge
    analysis sees real edges, not single-pixel stair steps."""
    if simplify:
        simplified = polygon.simplify(
            tolerance=simplify_tolerance, preserve_topology=True
        )
        if polygon.is_empty:
            return polygon
        if isinstance(simplified, Polygon):
            polygon = simplified
        else:
            return polygon
    return polygon.segmentize(max_segment_length=simplify_tolerance * 5)


def flatten_to_polygons(geometries) -> list[Any]:
    """Flatten geometries into a list of non-empty Polygons. MultiPolygons are
    unwrapped; empty and unsupported geometries are dropped."""
    flat: list[Any] = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, MultiPolygon):
            flat.extend(p for p in geom.geoms if not p.is_empty)
        elif isinstance(geom, Polygon):
            flat.append(geom)
    return flat


def iou_and_symmetric_fraction(shape_a: Any, shape_b: Any) -> tuple[float, float]:
    """IoU of two polygons, and their symmetric-difference area as a fraction
    of ``shape_a``, from ONE overlay instead of three.

    Areas obey |A or B| = |A| + |B| - |A and B| and |A xor B| = |A| + |B| -
    2|A and B|, so the intersection answers both questions. The union and
    symmetric_difference calls a reader expects here each build a whole
    geometry that the caller only ever measures, and on a review-sized set
    that is the single biggest avoidable cost in the shape refine.

    Returns (0.0, 0.0) when either shape has no area to compare.
    """
    area_a = shape_a.area
    area_b = shape_b.area
    intersection_area = shape_a.intersection(shape_b).area
    union_area = area_a + area_b - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0
    symmetric_fraction = (
        (area_a + area_b - 2.0 * intersection_area) / area_a
        if area_a > 0 else 0.0)
    return iou, symmetric_fraction


def regularize_single_polygon(
    polygon: Any,
    parallel_threshold: float,
    allow_45_degree: bool,
    diagonal_threshold_reduction: float,
    allow_circles: bool,
    circle_threshold: float,
    simplify: bool,
    simplify_tolerance: float,
    multi_direction: bool = _DEFAULT_MULTI_DIRECTION,
    multi_max_groups: int = _DEFAULT_MULTI_MAX_GROUPS,
    multi_min_separation_deg: float = _DEFAULT_MULTI_MIN_SEPARATION_DEG,
) -> list[Any]:
    """Regularize one shapely Polygon (or MultiPolygon part-by-part).

    Returns a list of output Polygons (usually one, more when a
    self-intersecting shape splits under buffer(0)). Returns the input
    unchanged on any degenerate case. When multi_direction is on, each ring is
    snapped to its own clustered directions (see regularize_coordinate_array_multi).
    """
    if isinstance(polygon, MultiPolygon):
        results: list[Any] = []
        for p in polygon.geoms:
            results.extend(
                regularize_single_polygon(
                    polygon=p,
                    parallel_threshold=parallel_threshold,
                    allow_45_degree=allow_45_degree,
                    diagonal_threshold_reduction=diagonal_threshold_reduction,
                    allow_circles=allow_circles,
                    circle_threshold=circle_threshold,
                    simplify=simplify,
                    simplify_tolerance=simplify_tolerance,
                    multi_direction=multi_direction,
                    multi_max_groups=multi_max_groups,
                    multi_min_separation_deg=multi_min_separation_deg,
                )
            )
        return results or [polygon]
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        return [polygon]

    simple_polygon = preprocess_polygon(
        polygon, simplify=simplify, simplify_tolerance=simplify_tolerance
    ).buffer(0)

    if isinstance(simple_polygon, MultiPolygon):
        results = []
        for p in simple_polygon.geoms:
            results.extend(
                regularize_single_polygon(
                    polygon=p,
                    parallel_threshold=parallel_threshold,
                    allow_45_degree=allow_45_degree,
                    diagonal_threshold_reduction=diagonal_threshold_reduction,
                    allow_circles=allow_circles,
                    circle_threshold=circle_threshold,
                    simplify=simplify,
                    simplify_tolerance=simplify_tolerance,
                    multi_direction=multi_direction,
                    multi_max_groups=multi_max_groups,
                    multi_min_separation_deg=multi_min_separation_deg,
                )
            )
        return results or [polygon]

    if not isinstance(simple_polygon, Polygon) or simple_polygon.is_empty:
        return [polygon]

    exterior_coordinates = np.array(simple_polygon.exterior.coords)
    regularized_exterior, _main_direction = _regularize_one_ring(
        coordinates=exterior_coordinates,
        parallel_threshold=parallel_threshold,
        allow_45_degree=allow_45_degree,
        diagonal_threshold_reduction=diagonal_threshold_reduction,
        multi_direction=multi_direction,
        multi_max_groups=multi_max_groups,
        multi_min_separation_deg=multi_min_separation_deg,
    )

    if allow_circles and polygon.area > 0:
        radius = math.sqrt(polygon.area / math.pi)
        perfect_circle = polygon.centroid.buffer(radius, quad_segs=42)
        circle_iou, _ = iou_and_symmetric_fraction(perfect_circle, polygon)
        if circle_iou > circle_threshold:
            regularized_exterior = np.array(
                perfect_circle.exterior.coords, dtype=float
            )

    regularized_interiors: list[Any] = []
    for interior in simple_polygon.interiors:
        interior_coordinates = np.array(interior.coords)
        regularized_interior, _ = _regularize_one_ring(
            coordinates=interior_coordinates,
            parallel_threshold=parallel_threshold,
            allow_45_degree=allow_45_degree,
            diagonal_threshold_reduction=diagonal_threshold_reduction,
            multi_direction=multi_direction,
            multi_max_groups=multi_max_groups,
            multi_min_separation_deg=multi_min_separation_deg,
        )
        regularized_interiors.append(regularized_interior)

    try:
        exterior_ring = LinearRing(regularized_exterior)
        interior_rings = [LinearRing(r) for r in regularized_interiors]
        regularized_polygon = Polygon(exterior_ring, interior_rings).buffer(0)
        final_iou, _ = iou_and_symmetric_fraction(regularized_polygon, polygon)
        if final_iou < _ring_min_iou():
            return [polygon]
        pieces = flatten_to_polygons([regularized_polygon])
        return pieces or [polygon]
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        return [polygon]


def _regularize_part_local(
    part: Any,
    tolerance: float,
    allow_diagonal: bool,
    diagonal_reduction: float,
    allow_circles: bool,
    circle_threshold: float,
    multi_direction: bool,
    multi_max_groups: int,
    multi_min_separation_deg: float,
) -> list[Any]:
    """One part regularized in a frame centred on it and scaled so the snap
    tolerance is exactly 1, then mapped back.

    The snapping math reads raw coordinates. create_line_equation builds
    ``c = x1*y2 - x2*y1`` out of absolute positions, and
    calculate_line_intersection rejects a parallel pair with ``d != 0``, an
    exact test. Both therefore behave differently depending on how large the
    numbers in the layer happen to be, and a geographic CRS makes them tiny: on
    a WGS84 run the ring that came back had a symmetric difference of about 75
    million times the object's own area, and only the IoU guard below kept that
    off the map. What the user saw was Right angles doing nothing at all, on
    every object, with no message. Measured on the same run once the frame is
    local: square corners 0.10 to 0.72, which is what a projected run already
    gets.

    Centring and scaling costs nothing on a projected run (measured
    identical to four decimals on the shape metrics) and it makes every
    hardcoded epsilon in the path mean the same thing in every CRS. The
    de-staircase, the cleanup and the guards all still run in the caller's
    frame.
    """
    if tolerance <= 0:
        return regularize_single_polygon(
            polygon=part,
            parallel_threshold=tolerance,
            allow_45_degree=allow_diagonal,
            diagonal_threshold_reduction=diagonal_reduction,
            allow_circles=allow_circles,
            circle_threshold=circle_threshold,
            simplify=True,
            simplify_tolerance=tolerance,
            multi_direction=multi_direction,
            multi_max_groups=multi_max_groups,
            multi_min_separation_deg=multi_min_separation_deg,
        )
    centre = part.centroid
    origin_x, origin_y = float(centre.x), float(centre.y)
    scale = 1.0 / float(tolerance)
    into = (scale, 0.0, 0.0, scale, -scale * origin_x, -scale * origin_y)
    back = (tolerance, 0.0, 0.0, tolerance, origin_x, origin_y)
    results = regularize_single_polygon(
        polygon=_affine_transform(part, into),
        parallel_threshold=1.0,
        allow_45_degree=allow_diagonal,
        diagonal_threshold_reduction=diagonal_reduction,
        allow_circles=allow_circles,
        circle_threshold=circle_threshold,
        simplify=True,
        simplify_tolerance=1.0,
        multi_direction=multi_direction,
        multi_max_groups=multi_max_groups,
        multi_min_separation_deg=multi_min_separation_deg,
    )
    return [_affine_transform(piece, back) for piece in results or []]


def _cleanup_polygon(polygon: Any, simplify_tolerance: float) -> Any:
    """Remove thin slivers left by regularization with a square/mitre buffer
    open-close, then drop collinear vertices. Best-effort: returns the input on
    any failure or when the cleanup empties the shape."""
    if polygon is None or polygon.is_empty or simplify_tolerance <= 0:
        return polygon
    try:
        buffer_size = simplify_tolerance / 50.0
        cleaned = polygon.buffer(buffer_size, cap_style="square", join_style="mitre")
        cleaned = cleaned.buffer(
            buffer_size * -2, cap_style="square", join_style="mitre"
        )
        cleaned = cleaned.buffer(buffer_size, cap_style="square", join_style="mitre")
        if cleaned.is_empty:
            return polygon
        cleaned = cleaned.simplify(tolerance=buffer_size, preserve_topology=True)
        if cleaned.is_empty:
            return polygon
        return cleaned
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        return polygon


def _destaircase_geometry(geometry: Any, tolerance_m: float) -> Any:
    """A de-staircased (simplified) copy of a polygon, or None when the
    simplify empties it or fails. Used on revert so the caller gets a cleaned
    outline instead of the raw pixel staircase."""
    if geometry is None or tolerance_m <= 0:
        return None
    try:
        simplified = geometry.simplify(tolerance_m, preserve_topology=True)
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        return None
    if simplified is None or simplified.is_empty:
        return None
    return simplified


def _vertex_count(geom: Any) -> int:
    """Total ring vertices across the exterior + interiors of every part.
    Best-effort: 0 on failure."""
    try:
        total = 0
        for part in flatten_to_polygons([geom]):
            total += len(part.exterior.coords)
            for ring in part.interiors:
                total += len(ring.coords)
        return total
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        return 0


def _component_count(geom: Any) -> int:
    """Number of polygon parts. Best-effort: 0 on failure."""
    try:
        return len(flatten_to_polygons([geom]))
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        return 0


def _hole_count(geom: Any) -> int:
    """Total interior rings across all parts. Best-effort: 0 on failure."""
    try:
        return sum(len(part.interiors) for part in flatten_to_polygons([geom]))
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        return 0


def _rectangularity(geom: Any) -> float:
    """area / convex_hull.area: how close to convex/rectangular a shape is.
    Best-effort: 0.0 on failure or a zero-area hull."""
    try:
        hull_area = geom.convex_hull.area
        if hull_area <= 0:
            return 0.0
        return geom.area / hull_area
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        return 0.0


def _ombb_candidate(part: Any, policy: RegularizePolicy) -> Any | None:
    """The oriented minimum bounding box for ONE Polygon when it is a clean,
    hole-free, elongated rectangle (area fill and aspect above the policy
    triggers). None otherwise. A near-square is never substituted, since its
    OMBB orientation is unstable; the aspect gate handles that. Best-effort:
    None on any failure."""
    try:
        if len(part.interiors) != 0:
            return None
        mrr = part.minimum_rotated_rectangle
        if not isinstance(mrr, Polygon) or not mrr.is_valid or mrr.area <= 0:
            return None
        area_fill = part.area / mrr.area
        coords = list(mrr.exterior.coords)
        if len(coords) < 5:
            return None
        side_a = calculate_distance(coords[0], coords[1])
        side_b = calculate_distance(coords[1], coords[2])
        short = min(side_a, side_b)
        if short <= 0:
            return None
        aspect = max(side_a, side_b) / short
        if area_fill >= policy.rectangle_area_fill and aspect >= policy.rectangle_min_aspect:
            return mrr
        return None
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        return None


def _passes_envelope(
    regularized: Any, original: Any, tolerance_m: float, policy: RegularizePolicy
) -> bool:
    """True to KEEP the regularized candidate, False to REJECT it. Every enabled
    check must pass; a check whose dial is neutral is skipped. Fail-OPEN: any
    error returns True so an envelope crash never loses a good shape.

    The Hausdorff check is what catches a thin spike the IoU guard misses: a
    spike moves the boundary a long way while barely changing area, so IoU stays
    high but the boundary displacement is large."""
    try:
        if not regularized.is_valid:
            return False
        if policy.enforce_component_count and (
            _component_count(regularized) != _component_count(original)
        ):
            return False
        if policy.enforce_hole_count and (
            _hole_count(regularized) != _hole_count(original)
        ):
            return False
        orig_area = original.area
        if orig_area > 0:
            ratio = regularized.area / orig_area
            if policy.max_area_ratio > 0 and ratio > policy.max_area_ratio:
                return False
            if policy.min_area_ratio > 0 and ratio < policy.min_area_ratio:
                return False
        if policy.max_hausdorff_mult > 0 and tolerance_m > 0:
            d = regularized.hausdorff_distance(original)
            if d > policy.max_hausdorff_mult * tolerance_m:
                return False
        if policy.max_vertex_growth > 0:
            ov = _vertex_count(original)
            rv = _vertex_count(regularized)
            if ov > 0 and rv > ov * policy.max_vertex_growth:
                return False
        return True
    except Exception:  # noqa: BLE001 -- fail open  # nosec B110
        return True


def _is_eligible(original: Any, policy: RegularizePolicy) -> bool:
    """True when the shape may be regularized. Every enabled gate must pass; a
    gate whose dial is neutral is skipped. Fail-OPEN: any error returns True."""
    try:
        if policy.max_holes >= 0 and _hole_count(original) > policy.max_holes:
            return False
        return not (
            policy.min_rectangularity > 0 and _rectangularity(original) < policy.min_rectangularity
        )
    except Exception:  # noqa: BLE001 -- fail open  # nosec B110
        return True


def _regularize_geometry(
    geometry: Any,
    tolerance_m: float,
    allow_diagonal: bool,
    allow_circles: bool,
    min_keep_iou: float,
    diagonal_reduction: float = _DEFAULT_DIAGONAL_REDUCTION,
    circle_threshold: float = _DEFAULT_CIRCLE_THRESHOLD,
    multi_direction: bool = _DEFAULT_MULTI_DIRECTION,
    multi_max_groups: int = _DEFAULT_MULTI_MAX_GROUPS,
    multi_min_separation_deg: float = _DEFAULT_MULTI_MIN_SEPARATION_DEG,
    policy: RegularizePolicy | None = None,
) -> tuple[Any, bool, bool]:
    """Core shapely path. Returns (geometry, regularized, reverted).

    Regularizes every polygon part, cleans slivers, then applies the
    anti-distortion guard: when the result deviates from the input past the
    IoU floor, the original is kept (reverted=True). With multi_direction on,
    a revert returns the de-staircased outline rather than the raw staircase.

    ``policy`` layers extra guards on top: an eligibility gate (decline shapes
    that are not plausible footprints), a conditioned rectangle substitution,
    and a safety envelope that rejects a candidate whose boundary moved too far.
    It is OFF/neutral by default, so an absent policy reproduces the IoU-only
    behaviour bit-for-bit.
    """
    pol = policy or _NEUTRAL_POLICY
    if geometry is None:
        return geometry, False, False
    try:
        if geometry.is_empty:
            return geometry, False, False
    except Exception:  # noqa: BLE001 -- unknown input  # nosec B110
        return geometry, False, False
    if tolerance_m <= 0:
        return geometry, False, False

    original = geometry
    try:
        if not original.is_valid:
            fixed = original.buffer(0)
            if not fixed.is_empty:
                original = fixed
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        pass

    parts = flatten_to_polygons([original])
    if not parts:
        return geometry, False, False

    if pol.eligibility_enabled and not _is_eligible(original, pol):
        # Not a plausible man-made footprint: decline, keep the caller's input
        # (already de-staircased/simplified upstream), never a squared guess.
        return geometry, False, True

    out_pieces: list[Any] = []
    for part in parts:
        rect = _ombb_candidate(part, pol) if pol.rectangle_enabled else None
        if rect is not None:
            cleaned = _cleanup_polygon(rect, tolerance_m)
            if cleaned is not None and not cleaned.is_empty:
                out_pieces.append(cleaned)
            continue
        try:
            results = _regularize_part_local(
                part=part,
                tolerance=tolerance_m,
                allow_diagonal=allow_diagonal,
                diagonal_reduction=diagonal_reduction,
                allow_circles=allow_circles,
                circle_threshold=circle_threshold,
                multi_direction=multi_direction,
                multi_max_groups=multi_max_groups,
                multi_min_separation_deg=multi_min_separation_deg,
            )
        except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
            results = [part]
        for piece in results:
            cleaned = _cleanup_polygon(piece, tolerance_m)
            if cleaned is not None and not cleaned.is_empty:
                out_pieces.append(cleaned)

    if not out_pieces:
        return geometry, False, False

    try:
        regularized = out_pieces[0] if len(out_pieces) == 1 else _unary_union(out_pieces)
        if regularized is None or regularized.is_empty:
            return geometry, False, False
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        return geometry, False, False

    # Anti-distortion guard: keep the original when the snapped shape drifts
    # too far from it (IoU below the floor).
    try:
        iou, sym_frac = iou_and_symmetric_fraction(original, regularized)
        if iou < min_keep_iou:
            if multi_direction:
                destaired = _destaircase_geometry(original, tolerance_m)
                if destaired is not None:
                    return destaired, False, True
            return geometry, False, True
        changed = sym_frac > _CHANGED_MIN_FRACTION
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        return geometry, False, False

    # Safety envelope: the IoU guard passes a thin spike or a rotated diamond
    # (tiny area change keeps IoU high), so reject those by boundary
    # displacement, area ratio, vertex growth and topology change. On reject,
    # return the de-staircased outline, never the mangled candidate.
    if pol.envelope_enabled and not _passes_envelope(
        regularized, original, tolerance_m, pol
    ):
        destaired = _destaircase_geometry(original, tolerance_m)
        if destaired is not None:
            return destaired, False, True
        return geometry, False, True

    return regularized, changed, False


def _regularize_geometry_isotropic(
    geometry: Any,
    tolerance_m: float,
    allow_diagonal: bool,
    allow_circles: bool,
    min_keep_iou: float,
    diagonal_reduction: float = _DEFAULT_DIAGONAL_REDUCTION,
    circle_threshold: float = _DEFAULT_CIRCLE_THRESHOLD,
    multi_direction: bool = _DEFAULT_MULTI_DIRECTION,
    multi_max_groups: int = _DEFAULT_MULTI_MAX_GROUPS,
    multi_min_separation_deg: float = _DEFAULT_MULTI_MIN_SEPARATION_DEG,
    policy: RegularizePolicy | None = None,
    unit_aspect: float = 1.0,
) -> tuple[Any, bool, bool]:
    """_regularize_geometry, run in a frame where one x unit and one y unit
    cover the same distance on the ground.

    ``unit_aspect`` is ground metres per y unit divided by ground metres per x
    unit of the geometry's own CRS: 1.0 for a projected CRS, and well above 1
    in a geographic one away from the equator, where a degree of longitude is
    much shorter than a degree of latitude.

    Every step below reads raw coordinates: the snap builds its perpendicular
    from them, the sliver cleanup buffers by a distance, the envelope guard
    measures a boundary displacement. In a stretched frame a perpendicular is
    not a right angle on the ground, so a squared building reaches the map as a
    parallelogram, tilted by an amount that grows with latitude. Stretching y
    into the caller's x unit makes all of them mean one distance again, and the
    result is mapped straight back, so the caller still gets its own CRS.

    A unit_aspect of 1 costs nothing: the whole frame change is skipped.
    """
    if not (unit_aspect and math.isfinite(unit_aspect) and unit_aspect > 0.0):
        unit_aspect = 1.0
    plain = (
        geometry, tolerance_m, allow_diagonal, allow_circles, min_keep_iou,
        diagonal_reduction, circle_threshold,
        multi_direction, multi_max_groups, multi_min_separation_deg,
    )
    if abs(unit_aspect - 1.0) < _ASPECT_IDENTITY_EPSILON or not _ensure_deps():
        return _regularize_geometry(*plain, policy=policy)
    try:
        stretched = _affine_transform(
            geometry, (1.0, 0.0, 0.0, unit_aspect, 0.0, 0.0))
    except Exception:  # noqa: BLE001 -- fall back to the caller's frame  # nosec B110
        return _regularize_geometry(*plain, policy=policy)
    result, regularized, reverted = _regularize_geometry(
        stretched, tolerance_m, allow_diagonal, allow_circles, min_keep_iou,
        diagonal_reduction, circle_threshold,
        multi_direction, multi_max_groups, multi_min_separation_deg,
        policy=policy,
    )
    if result is stretched:
        # Untouched: hand back the caller's own object rather than a shape that
        # made a needless round trip through the stretch.
        return geometry, regularized, reverted
    try:
        return (
            _affine_transform(result, (1.0, 0.0, 0.0, 1.0 / unit_aspect, 0.0, 0.0)),
            regularized,
            reverted,
        )
    except Exception:  # noqa: BLE001 -- never ship a shape stuck in the frame  # nosec B110
        return geometry, False, reverted


def regularize_polygon(
    geometry: Any,
    *,
    tolerance_m: float,
    allow_diagonal: bool,
    allow_circles: bool = False,
    min_keep_iou: float = _DEFAULT_MIN_KEEP_IOU,
    diagonal_reduction: float = _DEFAULT_DIAGONAL_REDUCTION,
    circle_threshold: float = _DEFAULT_CIRCLE_THRESHOLD,
    multi_direction: bool = _DEFAULT_MULTI_DIRECTION,
    multi_max_groups: int = _DEFAULT_MULTI_MAX_GROUPS,
    multi_min_separation_deg: float = _DEFAULT_MULTI_MIN_SEPARATION_DEG,
    policy: RegularizePolicy | None = None,
    unit_aspect: float = 1.0,
) -> Any:
    """Regularize a shapely Polygon/MultiPolygon. Pure shapely, no QGIS, so it
    is unit-testable outside QGIS. Returns the input unchanged on any failure,
    empty input, or when the anti-distortion guard reverts (with multi_direction
    on, a revert returns the de-staircased outline instead). ``policy`` layers
    the eligibility/rectangle/envelope guards; OFF by default. ``unit_aspect``
    squares against ground distance rather than raw coordinates: see
    _regularize_geometry_isotropic. It defaults to 1.0, which is exact for a
    projected CRS and leaves an old caller unchanged."""
    if not _ensure_deps():
        return geometry
    result, _regularized, _reverted = _regularize_geometry_isotropic(
        geometry, tolerance_m, allow_diagonal, allow_circles, min_keep_iou,
        diagonal_reduction, circle_threshold,
        multi_direction, multi_max_groups, multi_min_separation_deg,
        policy=policy, unit_aspect=unit_aspect,
    )
    return result


def regularize_qgs_geometry_ex(
    geom,
    *,
    tolerance_m: float,
    allow_diagonal: bool,
    allow_circles: bool = False,
    min_keep_iou: float = _DEFAULT_MIN_KEEP_IOU,
    diagonal_reduction: float = _DEFAULT_DIAGONAL_REDUCTION,
    circle_threshold: float = _DEFAULT_CIRCLE_THRESHOLD,
    multi_direction: bool = _DEFAULT_MULTI_DIRECTION,
    multi_max_groups: int = _DEFAULT_MULTI_MAX_GROUPS,
    multi_min_separation_deg: float = _DEFAULT_MULTI_MIN_SEPARATION_DEG,
    policy: RegularizePolicy | None = None,
    unit_aspect: float = 1.0,
) -> RegularizeResult:
    """Regularize a QgsGeometry and report the guard outcome.

    Converts via WKB (QgsGeometry -> shapely -> QgsGeometry). Best-effort:
    returns the input geometry with regularized=False on any failure and never
    raises. Callers wanting only the geometry can use regularize_qgs_geometry.

    ``unit_aspect`` squares against ground distance rather than raw coordinates
    (see _regularize_geometry_isotropic). A caller working in a geographic CRS
    MUST pass it, or the square corners it asked for arrive tilted.
    """
    if geom is None:
        return RegularizeResult(geom, False, False)
    try:
        if geom.isEmpty():
            return RegularizeResult(geom, False, False)
    except Exception:  # noqa: BLE001 -- unknown input  # nosec B110
        return RegularizeResult(geom, False, False)
    if tolerance_m is None or tolerance_m <= 0 or not _ensure_deps():
        return RegularizeResult(geom, False, False)
    if min_keep_iou <= 0:
        min_keep_iou = _DEFAULT_MIN_KEEP_IOU
    try:
        from shapely import wkb as shapely_wkb
    except Exception:  # noqa: BLE001 -- deps optional  # nosec B110
        return RegularizeResult(geom, False, False)
    try:
        shp = shapely_wkb.loads(bytes(geom.asWkb()))
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        return RegularizeResult(geom, False, False)
    result_shp, regularized, reverted = _regularize_geometry_isotropic(
        shp, tolerance_m, allow_diagonal, allow_circles, min_keep_iou,
        diagonal_reduction, circle_threshold,
        multi_direction, multi_max_groups, multi_min_separation_deg,
        policy=policy, unit_aspect=unit_aspect,
    )
    if not regularized:
        # Nothing kept (guard reverted, or no change): return the input as-is so
        # a WKB round-trip never perturbs an untouched geometry.
        return RegularizeResult(geom, False, reverted)
    try:
        from qgis.core import QgsGeometry
        from qgis.PyQt.QtCore import QByteArray

        # QgsGeometry.fromWkb is an INSTANCE method (it fills self and returns a
        # status), unlike the static fromWkt. Calling it on the class raises
        # TypeError, which the guard below would swallow into "nothing changed",
        # so the whole regularizer silently returned its input. Bind on an
        # instance, and judge the outcome by the geometry, not the return value
        # (the sip binding reports None on some builds). WKT is the fallback so a
        # binding quirk still yields a shape rather than a silent no-op.
        out = QgsGeometry()
        out.fromWkb(QByteArray(result_shp.wkb))
        if out.isEmpty():
            out = QgsGeometry.fromWkt(result_shp.wkt)
        if out is None or out.isEmpty():
            return RegularizeResult(geom, False, reverted)
        return RegularizeResult(out, True, False)
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        return RegularizeResult(geom, False, reverted)


def regularize_qgs_geometry(
    geom,
    *,
    tolerance_m: float,
    allow_diagonal: bool,
    allow_circles: bool = False,
    min_keep_iou: float = _DEFAULT_MIN_KEEP_IOU,
    diagonal_reduction: float = _DEFAULT_DIAGONAL_REDUCTION,
    circle_threshold: float = _DEFAULT_CIRCLE_THRESHOLD,
    multi_direction: bool = _DEFAULT_MULTI_DIRECTION,
    multi_max_groups: int = _DEFAULT_MULTI_MAX_GROUPS,
    multi_min_separation_deg: float = _DEFAULT_MULTI_MIN_SEPARATION_DEG,
    policy: RegularizePolicy | None = None,
    unit_aspect: float = 1.0,
):
    """Regularize a QgsGeometry, returning the geometry only (simple happy
    path). See regularize_qgs_geometry_ex for the guard outcome and for
    ``unit_aspect``. Returns the input unchanged on any failure; never raises."""
    return regularize_qgs_geometry_ex(
        geom,
        tolerance_m=tolerance_m,
        allow_diagonal=allow_diagonal,
        allow_circles=allow_circles,
        min_keep_iou=min_keep_iou,
        diagonal_reduction=diagonal_reduction,
        circle_threshold=circle_threshold,
        multi_direction=multi_direction,
        multi_max_groups=multi_max_groups,
        multi_min_separation_deg=multi_min_separation_deg,
        policy=policy,
        unit_aspect=unit_aspect,
    ).geometry
