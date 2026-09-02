
















from __future__ import annotations

from .server_dials import dial_in_range



_SEED_CROP = "detection_policy.seed.crop."


def crop_grid_cell_fraction(fallback: float) -> float:

    return dial_in_range(_SEED_CROP + "grid_cell_fraction", fallback, 0.05, 1.0)


def crop_scale_step(fallback: float) -> float:

    return dial_in_range(_SEED_CROP + "scale_step", fallback, 1.01, 2.0)


def crop_edge_clearance(fallback: float) -> float:

    return dial_in_range(_SEED_CROP + "edge_clearance", fallback, 0.0, 0.5)




_REVIEW = "detection_policy.review."


def weld_gap_m(fallback: float) -> float:

    return dial_in_range(_REVIEW + "weld_gap_m", fallback, 0.0, 10.0)


def weld_radius_px_cap(fallback: int) -> int:

    return dial_in_range(_REVIEW + "weld_radius_px_cap", fallback, 0, 50)


def reach_work_budget(fallback: int) -> int:

    return dial_in_range(_REVIEW + "reach_work_budget", fallback, 1e5, 1e10)


def reach_min_steps(fallback: int) -> int:

    return dial_in_range(_REVIEW + "reach_min_steps", fallback, 1, 1000)





def close_max_area_growth(fallback: float) -> float:

    return dial_in_range(_REVIEW + "close_max_area_growth", fallback, 1.0, 3.0)


def smooth_area_keep(fallback: float) -> float:

    return dial_in_range(_REVIEW + "smooth.area_keep", fallback, 0.0, 1.0)


def smooth_diet_fraction(fallback: float) -> float:

    return dial_in_range(_REVIEW + "smooth.diet_fraction", fallback, 0.0, 1.0)


def refine_margin_px(fallback: int) -> int:

    return dial_in_range(_REVIEW + "refine_margin_px", fallback, 0, 20)




_VERTEX_BUDGET = _REVIEW + "vertex_budget."


def floor_max_chord_steps(fallback: float) -> float:

    return dial_in_range(_VERTEX_BUDGET + "floor_max_chord_steps", fallback, 1.0, 32.0)




_AUTO_REGULARIZE = "detection_policy.auto_regularize."


def consensus_neighbour_cap(fallback: int) -> int:

    return dial_in_range(_AUTO_REGULARIZE + "consensus_neighbour_cap", fallback, 1, 200)


def save_neighbour_cap(fallback: int) -> int:

    return dial_in_range(_AUTO_REGULARIZE + "save_neighbour_cap", fallback, 1, 200)


def min_angle_split_deg(fallback: float) -> float:

    return dial_in_range(_AUTO_REGULARIZE + "min_angle_split_deg", fallback, 0.1, 45.0)


def circle_segments(fallback: int) -> int:

    return dial_in_range(_AUTO_REGULARIZE + "circle_segments", fallback, 8, 128)


def align_thread_min_objects(fallback: int) -> int:

    return dial_in_range(_AUTO_REGULARIZE + "thread_min_objects", fallback, 1, 1e7)


def align_process_min_objects(fallback: int) -> int:

    return dial_in_range(_AUTO_REGULARIZE + "process_min_objects", fallback, 1, 1e7)


def align_process_workers(fallback: int) -> int:

    return dial_in_range(_AUTO_REGULARIZE + "process_workers", fallback, 0, 16)


def align_phase_budget_s(fallback: float) -> float:

    return dial_in_range(_AUTO_REGULARIZE + "phase_budget_s", fallback, 1.0, 600.0)


def align_offload_budget_s(fallback: float) -> float:

    return dial_in_range(_AUTO_REGULARIZE + "offload_budget_s", fallback, 1.0, 900.0)




_NETWORK = "detection_policy.network."


def archive_jpeg_quality(fallback: int) -> int:

    return dial_in_range("detection_policy.seed.archive_jpeg_quality", fallback, 50, 100)


def render_zone_timeout_ms(fallback: int) -> int:

    return dial_in_range(_NETWORK + "render_zone_timeout_ms", fallback, 2000, 120000)


def imagery_probe_px(fallback: int) -> int:

    return dial_in_range(_NETWORK + "imagery_probe_px", fallback, 64, 1024)


def imagery_probe_timeout_ms(fallback: int) -> int:

    return dial_in_range(_NETWORK + "imagery_probe_timeout_ms", fallback, 1000, 60000)




_AUTO_DEFAULTS = _REVIEW + "auto_defaults."


def auto_review_simplify_default(fallback: float) -> float:

    return dial_in_range(_AUTO_DEFAULTS + "simplify", fallback, 0.0, 1.0)


def auto_review_clean_default(fallback: float) -> float:

    return dial_in_range(_AUTO_DEFAULTS + "clean", fallback, 0.0, 1.0)


def auto_review_expand_default(fallback: int) -> int:

    return dial_in_range(_AUTO_DEFAULTS + "expand", fallback, -50, 50)


def auto_review_close_notches_default(fallback: float) -> float:

    return dial_in_range(_AUTO_DEFAULTS + "close_notches_m", fallback, 0.0, 50.0)


def auto_review_points_pct_default(fallback: int) -> int:

    return dial_in_range(_AUTO_DEFAULTS + "points_pct", fallback, 1, 100)
