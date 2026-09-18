








from __future__ import annotations

import math




try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from typing import TYPE_CHECKING, Any

from .regularize_edge_pipeline import (
    _MULTI_MIN_GROUP_WEIGHT_FRACTION,
    _MULTI_PARALLEL_ANGLE_EPS,
    _resolve_regularize_dials,
    calculate_azimuth_angle,
    calculate_distance,
    find_nearest_target_angle,
    get_orientation_and_rotation,
    handle_parallel_edges,
    handle_perpendicular_edges,
    regularize_coordinate_array,
    rotate_edge,
    rotate_point,
)

if TYPE_CHECKING:
    from .regularize_edge_pipeline import (
        RegularizeDials,
    )










def _circular_dist_mod(a: float, b: float, period: float) -> float:

    d = abs(a - b) % period
    return min(d, period - d)


def _refine_direction(
    folded_angles: Any, lengths: Any, center_bin: int, window: float
) -> float:


    center = center_bin + 0.5
    sum_x = 0.0
    sum_y = 0.0
    total = 0.0
    for angle, weight in zip(folded_angles, lengths):
        if _circular_dist_mod(float(angle), center, 90.0) <= window:

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









    folded = np.mod(azimuth_angles, 90.0)
    weights = np.asarray(lengths, dtype=float)
    indices = np.minimum(np.floor(folded).astype(int), 89)
    bins = np.bincount(indices, weights=weights, minlength=90).astype(float)
    if bins.sum() <= 0.0:
        return []


    smoothed = (2.0 * bins + np.roll(bins, 1) + np.roll(bins, -1)) / 4.0
    peaks = [
        i
        for i in range(90)
        if smoothed[i] > 0.0 and smoothed[i] >= smoothed[(i - 1) % 90] and smoothed[i] >= smoothed[(i + 1) % 90]
    ]
    peaks.sort(key=lambda i: smoothed[i], reverse=True)
    total_weight = float(weights.sum())

    def _weight_near(peak: int) -> float:


        gaps = np.abs(folded - (peak + 0.5)) % 90.0
        gaps = np.minimum(gaps, 90.0 - gaps)


        return float(np.sum(weights[gaps <= min_separation_deg]))

    min_weight = min_group_weight * total_weight
    chosen: list[int] = []
    for peak in peaks:
        if any(
            _circular_dist_mod(peak, other, 90.0) < min_separation_deg
            for other in chosen
        ):
            continue


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


def regularize_coordinate_array_multi(
    coordinates: Any,
    parallel_threshold: float,
    allow_45_degree: bool,
    diagonal_threshold_reduction: float,
    max_groups: int,
    min_separation_deg: float,
    angle_enforcement_tolerance: float = 0.1,
    dials: RegularizeDials | None = None,
) -> tuple[Any, float]:








    if len(coordinates) < 4:
        return coordinates, 0.0
    if np.allclose(coordinates[0], coordinates[-1]):
        processing_coords = coordinates[:-1]
    else:
        processing_coords = coordinates
    if len(processing_coords) < 3:
        return coordinates, 0.0
    resolved = dials or _resolve_regularize_dials()
    parallel_eps_deg = resolved.multi_parallel_eps_deg
    edge_data = _analyze_edges_multi(
        processing_coords, max_groups, min_separation_deg,
        resolved.multi_min_group_weight,
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
