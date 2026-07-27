"""Copyright (c) 2025 DPIRD-DMA"""
































































from __future__ import annotations

from .regularize_edge_pipeline import (
    _ASPECT_IDENTITY_EPSILON,
    _CHANGED_MIN_FRACTION,
    _DEFAULT_CIRCLE_THRESHOLD,
    _DEFAULT_DIAGONAL_REDUCTION,
    _DEFAULT_DIALS,
    _DEFAULT_MIN_KEEP_IOU,
    _DEFAULT_MULTI_DIRECTION,
    _DEFAULT_MULTI_MAX_GROUPS,
    _DEFAULT_MULTI_MIN_SEPARATION_DEG,
    _DESTAIRCASE_NOOP_FRACTION,
    _MULTI_MIN_GROUP_WEIGHT_FRACTION,
    _MULTI_PARALLEL_ANGLE_EPS,
    _RING_MIN_IOU,
    RegularizeDials,
    _create_weighted_histogram,
    _find_best_symmetric_bin,
    _resolve_regularize_dials,
    _smooth_histogram,
    analyze_edges,
    calculate_azimuth_angle,
    calculate_distance,
    calculate_line_intersection,
    calculate_parallel_line_distance,
    connect_regularized_edges,
    create_line_equation,
    enforce_angles_post_process,
    find_nearest_target_angle,
    get_orientation_and_rotation,
    handle_parallel_edges,
    handle_perpendicular_edges,
    orient_edges,
    project_point_to_line,
    regularize_coordinate_array,
    rotate_edge,
    rotate_point,
)
from .regularize_multi_direction import (
    _analyze_edges_multi,
    _circular_dist_mod,
    _cluster_directions,
    _connect_regularized_edges_multi,
    _enforce_angles_multi,
    _nearest_target_multi,
    _orient_edges_multi,
    _refine_direction,
    regularize_coordinate_array_multi,
)
from .regularize_polygon_pipeline import (
    _NEUTRAL_POLICY,
    LinearRing,
    MultiPolygon,
    Polygon,
    RegularizePolicy,
    RegularizeResult,
    _affine_transform,
    _assemble_parts,
    _cleanup_polygon,
    _component_and_hole_counts,
    _component_count,
    _destaircase_geometry,
    _ensure_deps,
    _hole_count,
    _is_eligible,
    _ombb_candidate,
    _passes_envelope,
    _rectangularity,
    _regularize_geometry,
    _regularize_geometry_isotropic,
    _regularize_one_ring,
    _regularize_part_local,
    _rings_already_on_grid,
    _unary_union,
    _vertex_count,
    dependencies_available,
    flatten_to_polygons,
    iou_and_symmetric_fraction,
    np,
    preprocess_polygon,
    regularize_polygon,
    regularize_qgs_geometry,
    regularize_qgs_geometry_ex,
    regularize_single_polygon,
)

__all__ = [
    "np",
    "Polygon",
    "MultiPolygon",
    "LinearRing",
    "_unary_union",
    "_affine_transform",
    "_DEFAULT_MIN_KEEP_IOU",
    "_DEFAULT_DIAGONAL_REDUCTION",
    "_DEFAULT_CIRCLE_THRESHOLD",
    "_CHANGED_MIN_FRACTION",
    "_RING_MIN_IOU",
    "_ASPECT_IDENTITY_EPSILON",
    "_DEFAULT_MULTI_DIRECTION",
    "_DEFAULT_MULTI_MAX_GROUPS",
    "_DEFAULT_MULTI_MIN_SEPARATION_DEG",
    "_MULTI_PARALLEL_ANGLE_EPS",
    "_MULTI_MIN_GROUP_WEIGHT_FRACTION",
    "_DESTAIRCASE_NOOP_FRACTION",
    "RegularizeResult",
    "RegularizePolicy",
    "_NEUTRAL_POLICY",
    "_ensure_deps",
    "dependencies_available",
    "calculate_distance",
    "calculate_azimuth_angle",
    "create_line_equation",
    "calculate_line_intersection",
    "calculate_parallel_line_distance",
    "project_point_to_line",
    "rotate_point",
    "rotate_edge",
    "find_nearest_target_angle",
    "enforce_angles_post_process",
    "_create_weighted_histogram",
    "_smooth_histogram",
    "_find_best_symmetric_bin",
    "analyze_edges",
    "get_orientation_and_rotation",
    "orient_edges",
    "handle_perpendicular_edges",
    "handle_parallel_edges",
    "connect_regularized_edges",
    "regularize_coordinate_array",
    "_circular_dist_mod",
    "_refine_direction",
    "_cluster_directions",
    "_analyze_edges_multi",
    "_orient_edges_multi",
    "_connect_regularized_edges_multi",
    "_nearest_target_multi",
    "_enforce_angles_multi",
    "RegularizeDials",
    "_DEFAULT_DIALS",
    "_resolve_regularize_dials",
    "regularize_coordinate_array_multi",
    "_regularize_one_ring",
    "preprocess_polygon",
    "flatten_to_polygons",
    "iou_and_symmetric_fraction",
    "regularize_single_polygon",
    "_regularize_part_local",
    "_cleanup_polygon",
    "_assemble_parts",
    "_destaircase_geometry",
    "_vertex_count",
    "_component_count",
    "_hole_count",
    "_component_and_hole_counts",
    "_rectangularity",
    "_ombb_candidate",
    "_passes_envelope",
    "_is_eligible",
    "_rings_already_on_grid",
    "_regularize_geometry",
    "_regularize_geometry_isotropic",
    "regularize_polygon",
    "regularize_qgs_geometry_ex",
    "regularize_qgs_geometry",
]
