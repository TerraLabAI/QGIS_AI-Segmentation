










from __future__ import annotations

import math




try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from typing import Any, NamedTuple

__all__ = [
    "RegularizeDials",
    "_ASPECT_IDENTITY_EPSILON",
    "_CHANGED_MIN_FRACTION",
    "_DEFAULT_CIRCLE_THRESHOLD",
    "_DEFAULT_DIAGONAL_REDUCTION",
    "_DEFAULT_DIALS",
    "_DEFAULT_MIN_KEEP_IOU",
    "_DEFAULT_MULTI_DIRECTION",
    "_DEFAULT_MULTI_MAX_GROUPS",
    "_DEFAULT_MULTI_MIN_SEPARATION_DEG",
    "_DESTAIRCASE_NOOP_FRACTION",
    "_MULTI_MIN_GROUP_WEIGHT_FRACTION",
    "_MULTI_PARALLEL_ANGLE_EPS",
    "_RING_MIN_IOU",
    "_create_weighted_histogram",
    "_find_best_symmetric_bin",
    "_resolve_regularize_dials",
    "_smooth_histogram",
    "analyze_edges",
    "calculate_azimuth_angle",
    "calculate_distance",
    "calculate_line_intersection",
    "calculate_parallel_line_distance",
    "connect_regularized_edges",
    "create_line_equation",
    "enforce_angles_post_process",
    "find_nearest_target_angle",
    "get_orientation_and_rotation",
    "handle_parallel_edges",
    "handle_perpendicular_edges",
    "orient_edges",
    "project_point_to_line",
    "regularize_coordinate_array",
    "rotate_edge",
    "rotate_point",
]




_DEFAULT_MIN_KEEP_IOU = 0.70
_DEFAULT_DIAGONAL_REDUCTION = 15.0




_DEFAULT_CIRCLE_THRESHOLD = 0.94


_CHANGED_MIN_FRACTION = 1.0e-3



_RING_MIN_IOU = 0.1






_ASPECT_IDENTITY_EPSILON = 0.01







_DEFAULT_MULTI_DIRECTION = False
_DEFAULT_MULTI_MAX_GROUPS = 3
_DEFAULT_MULTI_MIN_SEPARATION_DEG = 10.0


_MULTI_PARALLEL_ANGLE_EPS = 1.5







_MULTI_MIN_GROUP_WEIGHT_FRACTION = 0.20




_DESTAIRCASE_NOOP_FRACTION = 1.0e-9







def calculate_distance(point_1: Any, point_2: Any) -> float:

    dx, dy = point_1[0] - point_2[0], point_1[1] - point_2[1]
    return math.hypot(dx, dy)


def calculate_azimuth_angle(start_point: Any, end_point: Any) -> float:

    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    return math.degrees(math.atan2(dy, dx)) % 360


def create_line_equation(point1: Any, point2: Any) -> tuple[float, float, float]:

    a = point1[1] - point2[1]
    b = point2[0] - point1[0]
    c = point1[0] * point2[1] - point2[0] * point1[1]
    return a, b, -c


def calculate_line_intersection(
    line1: tuple[float, float, float], line2: tuple[float, float, float]
) -> tuple[float, float] | None:

    d = line1[0] * line2[1] - line1[1] * line2[0]
    dx = line1[2] * line2[1] - line1[1] * line2[2]
    dy = line1[0] * line2[2] - line1[2] * line2[0]
    if d != 0:
        return dx / d, dy / d
    return None


def calculate_parallel_line_distance(
    line1: tuple[float, float, float], line2: tuple[float, float, float]
) -> float:

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

    x, y = point
    center_x, center_y = center
    angle_radians = math.radians(angle_degrees)
    tx = x - center_x
    ty = y - center_y
    rx = tx * math.cos(angle_radians) + ty * math.sin(angle_radians)
    ry = ty * math.cos(angle_radians) - tx * math.sin(angle_radians)
    return (rx + center_x, ry + center_y)


def rotate_edge(start_point: Any, end_point: Any, rotation_angle: float) -> list[Any]:

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







def find_nearest_target_angle(
    current_azimuth: float, main_direction: float, allow_45_degree: bool
) -> float:

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


    mirrored_mean = (hist + hist[::-1]) / 2
    sorted_indices = np.argsort(mirrored_mean)
    a, b = sorted_indices[-2:]
    return a if hist[a] > hist[b] else b


def analyze_edges(
    coordinates: Any, coarse_bin_size: int = 5, fine_bin_size: int = 1
) -> dict[str, Any]:

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


class RegularizeDials(NamedTuple):






















    ring_min_iou: float = _RING_MIN_IOU
    multi_parallel_eps_deg: float = _MULTI_PARALLEL_ANGLE_EPS
    multi_min_group_weight: float = _MULTI_MIN_GROUP_WEIGHT_FRACTION


_DEFAULT_DIALS = RegularizeDials()


def _resolve_regularize_dials() -> RegularizeDials:




    try:
        from .detection_policy import regularize_settings

        settings = regularize_settings()
        ring_iou = float(settings["ring_min_iou"])
        return RegularizeDials(
            ring_min_iou=ring_iou if 0.0 < ring_iou < 1.0 else _RING_MIN_IOU,
            multi_parallel_eps_deg=float(settings["multi_parallel_eps_deg"]),
            multi_min_group_weight=float(settings["multi_min_group_weight"]),
        )
    except Exception:  # noqa: BLE001
        return _DEFAULT_DIALS
