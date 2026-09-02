"""Served dials for the geometry, crop and review-default constants.

Every getter here reads the cached product configuration and falls open to
the module constant its caller passes in, so a client with no configuration
(startup, offline, an older server) behaves exactly as it shipped. Nothing
here networks, raises, or touches disk, so a getter is safe on the GUI thread
and from a worker thread alike.

Each getter is bounded: a served value outside the range the code can live
with is refused and the shipped constant wins. The bound is the whole safety
story, so a getter never caps a served value with the constant.

Read a dial at call time (or once per run), never at import time, because the
cache fills after the modules that use it are imported. The caller keeps its
constant and hands it in as the fallback, so a test that patches the constant
still drives the code.
"""
from __future__ import annotations

from .server_dials import dial_in_range

# -- crop window (Semi-Auto shared crops) -----------------------------------

_SEED_CROP = "detection_policy.seed.crop."


def crop_grid_cell_fraction(fallback: float) -> float:
    """Ground size of one crop grid cell as a share of the crop's own ground."""
    return dial_in_range(_SEED_CROP + "grid_cell_fraction", fallback, 0.05, 1.0)


def crop_scale_step(fallback: float) -> float:
    """The multiplicative step a shared crop's zoom-out factor may grow in."""
    return dial_in_range(_SEED_CROP + "scale_step", fallback, 1.01, 2.0)


def crop_edge_clearance(fallback: float) -> float:
    """Share of the crop's ground an object must keep clear of the crop edge."""
    return dial_in_range(_SEED_CROP + "edge_clearance", fallback, 0.0, 0.5)


# -- shape growth (refine click welding and reach) --------------------------

_REVIEW = "detection_policy.review."


def weld_gap_m(fallback: float) -> float:
    """Widest gap, in ground metres, that still reads as one object."""
    return dial_in_range(_REVIEW + "weld_gap_m", fallback, 0.0, 10.0)


def weld_radius_px_cap(fallback: int) -> int:
    """Ceiling, in pixels, on the weld radius derived from the gap."""
    return dial_in_range(_REVIEW + "weld_radius_px_cap", fallback, 0, 50)


def reach_work_budget(fallback: int) -> int:
    """Pixel visits the reach walk may spend before giving up."""
    return dial_in_range(_REVIEW + "reach_work_budget", fallback, 1e5, 1e10)


def reach_min_steps(fallback: int) -> int:
    """Fewest dilation steps the reach walk always gets, whatever its size."""
    return dial_in_range(_REVIEW + "reach_min_steps", fallback, 1, 1000)


# -- refine tail (polygon_exporter) -----------------------------------------


def close_max_area_growth(fallback: float) -> float:
    """Ceiling on how much a notch closing may grow an object's area."""
    return dial_in_range(_REVIEW + "close_max_area_growth", fallback, 1.0, 3.0)


def smooth_area_keep(fallback: float) -> float:
    """Share of its area a shape must keep through a smoothing step."""
    return dial_in_range(_REVIEW + "smooth.area_keep", fallback, 0.0, 1.0)


def smooth_diet_fraction(fallback: float) -> float:
    """Share of the simplify tolerance the post-smooth diet runs at."""
    return dial_in_range(_REVIEW + "smooth.diet_fraction", fallback, 0.0, 1.0)


def refine_margin_px(fallback: int) -> int:
    """Rows and columns kept around a mask's box before the refinement steps."""
    return dial_in_range(_REVIEW + "refine_margin_px", fallback, 0, 20)


# -- vertex budget ----------------------------------------------------------

_VERTEX_BUDGET = _REVIEW + "vertex_budget."


def floor_max_chord_steps(fallback: float) -> float:
    """Longest chord, in grid steps, still allowed the one-step floor."""
    return dial_in_range(_VERTEX_BUDGET + "floor_max_chord_steps", fallback, 1.0, 32.0)


# -- footprint alignment ----------------------------------------------------

_AUTO_REGULARIZE = "detection_policy.auto_regularize."


def consensus_neighbour_cap(fallback: int) -> int:
    """Most neighbours one consensus angle averages, nearest first."""
    return dial_in_range(_AUTO_REGULARIZE + "consensus_neighbour_cap", fallback, 1, 200)


def save_neighbour_cap(fallback: int) -> int:
    """Most saved neighbours one save-time alignment reads."""
    return dial_in_range(_AUTO_REGULARIZE + "save_neighbour_cap", fallback, 1, 200)


def min_angle_split_deg(fallback: float) -> float:
    """Angle gap under which a second candidate adds nothing."""
    return dial_in_range(_AUTO_REGULARIZE + "min_angle_split_deg", fallback, 0.1, 45.0)


def circle_segments(fallback: int) -> int:
    """Sides of the polygon that stands in for a circular footprint."""
    return dial_in_range(_AUTO_REGULARIZE + "circle_segments", fallback, 8, 128)


def align_thread_min_objects(fallback: int) -> int:
    """Fewest objects that send the run-wide alignment to its own thread."""
    return dial_in_range(_AUTO_REGULARIZE + "thread_min_objects", fallback, 1, 1e7)


def align_phase_budget_s(fallback: float) -> float:
    """Seconds the alignment may take on the interface thread."""
    return dial_in_range(_AUTO_REGULARIZE + "phase_budget_s", fallback, 1.0, 600.0)


def align_offload_budget_s(fallback: float) -> float:
    """Seconds the alignment may take once it runs off the interface thread."""
    return dial_in_range(_AUTO_REGULARIZE + "offload_budget_s", fallback, 1.0, 900.0)


# -- cloud detection (archive copy, zone render, imagery probe) -------------

_NETWORK = "detection_policy.network."


def archive_jpeg_quality(fallback: int) -> int:
    """Encode quality of the un-stamped archive copy that rides beside a tile."""
    return dial_in_range("detection_policy.seed.archive_jpeg_quality", fallback, 50, 100)


def render_zone_timeout_ms(fallback: int) -> int:
    """Hard cap on one zone render wait."""
    return dial_in_range(_NETWORK + "render_zone_timeout_ms", fallback, 2000, 120000)


def imagery_probe_px(fallback: int) -> int:
    """Side, in pixels, of the finest imagery probe render."""
    return dial_in_range(_NETWORK + "imagery_probe_px", fallback, 64, 1024)


def imagery_probe_timeout_ms(fallback: int) -> int:
    """Hard cap on one imagery probe render."""
    return dial_in_range(_NETWORK + "imagery_probe_timeout_ms", fallback, 1000, 60000)


# -- review panel opening values (Automatic) --------------------------------

_AUTO_DEFAULTS = _REVIEW + "auto_defaults."


def auto_review_simplify_default(fallback: float) -> float:
    """Simplify tolerance, in pixels, an Automatic review opens with."""
    return dial_in_range(_AUTO_DEFAULTS + "simplify", fallback, 0.0, 1.0)


def auto_review_clean_default(fallback: float) -> float:
    """Clean-edges opening radius, in pixels, an Automatic review opens with."""
    return dial_in_range(_AUTO_DEFAULTS + "clean", fallback, 0.0, 1.0)


def auto_review_expand_default(fallback: int) -> int:
    """Expand or shrink, in pixels, an Automatic review opens with."""
    return dial_in_range(_AUTO_DEFAULTS + "expand", fallback, -50, 50)


def auto_review_close_notches_default(fallback: float) -> float:
    """Close-notches width, in metres, an Automatic review opens with."""
    return dial_in_range(_AUTO_DEFAULTS + "close_notches_m", fallback, 0.0, 50.0)


def auto_review_points_pct_default(fallback: int) -> int:
    """Share of traced points, in percent, an Automatic review opens with."""
    return dial_in_range(_AUTO_DEFAULTS + "points_pct", fallback, 1, 100)
