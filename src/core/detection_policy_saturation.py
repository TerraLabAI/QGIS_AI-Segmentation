






from __future__ import annotations

from .detection_policy_core import (
    _is_finite_policy_value,
    policy_scope,
    saturation_policy,
    seed_policy,
)
from .served_config import require_served_bool, require_served_int, require_served_number


def resplit_charge_every(policy: dict | None = None) -> int:





    val = seed_policy(policy).get("resplit_charge_every")
    if _is_finite_policy_value(val) and val >= 0:
        return int(val)
    return 1


def _sat_float(key: str, fallback: float, policy: dict | None) -> float:

    val = saturation_policy(policy).get(key)
    if _is_finite_policy_value(val):
        return float(val)
    return fallback


def mask_cap_trigger_frac(fallback: object = None, policy: dict | None = None) -> float:



    with policy_scope(policy):
        return require_served_number("detection_policy.seed.saturation.cap_trigger_frac", 0.05, 1.0)


def subdiv_max_depth(fallback: object = None, policy: dict | None = None) -> int:


    with policy_scope(policy):
        return require_served_int("detection_policy.seed.saturation.subdiv_max_depth", 0, 8)


def resplit_time_ratio(fallback: object = None, policy: dict | None = None) -> float:



    with policy_scope(policy):
        return require_served_number("detection_policy.seed.saturation.resplit_time_ratio", 0.0, 100.0)


def max_masks_per_tile(fallback: object = None, policy: dict | None = None) -> int:



    with policy_scope(policy):
        return require_served_int("detection_policy.seed.saturation.max_masks_per_tile", 1, 1000)


def subdivide_overlap_fraction(fallback: object = None, policy: dict | None = None) -> float:



    with policy_scope(policy):
        return require_served_number(
            "detection_policy.seed.saturation.subdivide_overlap_fraction", 0.0, 0.49)


def subdivide_min_parent_px(fallback: object = None, policy: dict | None = None) -> int:


    with policy_scope(policy):
        return require_served_int(
            "detection_policy.seed.saturation.subdivide_min_parent_px", 1, 100_000)


def subdivide_cap_params(
    fallback_max: int, fallback_min: int, fallback_scale: int,
    policy: dict | None = None,
) -> tuple[int, int, int]:



    sat = saturation_policy(policy)

    def _pos_int(key: str, fb: int) -> int:
        val = sat.get(key)
        if _is_finite_policy_value(val) and val > 0:
            return int(val)
        return fb

    return (
        _pos_int("subdivide_cap_max", fallback_max),
        _pos_int("subdivide_cap_min", fallback_min),
        _pos_int("subdivide_cap_scale", fallback_scale),
    )


def max_tile_coverage(fallback: object = None, policy: dict | None = None) -> float:


    with policy_scope(policy):
        return require_served_number("detection_policy.seed.saturation.max_tile_coverage", 0.0, 1.0)


def hard_tile_coverage(fallback: object = None, policy: dict | None = None) -> float:


    with policy_scope(policy):
        return require_served_number("detection_policy.seed.saturation.hard_tile_coverage", 0.0, 1.0)


def map_cover_score_floor(fallback: float, policy: dict | None = None) -> float:












    val = _sat_float("map_cover_score_floor", fallback, policy)
    return val if 0.0 < val <= 1.0 else fallback


def compact_min_fill(fallback: object = None, policy: dict | None = None) -> float:





    with policy_scope(policy):
        return require_served_number("detection_policy.seed.saturation.compact_min_fill", 0.01, 1.0)


def tile_span_fraction(fallback: object = None, policy: dict | None = None) -> float:





    with policy_scope(policy):
        return require_served_number("detection_policy.seed.saturation.tile_span_fraction", 0.01, 1.0)


def hard_cover_shape_escape(fallback: object = None, policy: dict | None = None) -> bool:














    with policy_scope(policy):
        return require_served_bool("detection_policy.seed.saturation.hard_cover_shape_escape")


def min_keep_px(fallback: object = None, policy: dict | None = None) -> float:




    with policy_scope(policy):
        return require_served_number("detection_policy.seed.saturation.min_keep_px", 0.0, 64.0)


def min_keep_floor_m2(fallback: object = None, policy: dict | None = None) -> float:



    with policy_scope(policy):
        return require_served_number(
            "detection_policy.seed.saturation.min_keep_floor_m2", 0.0, 1_000_000.0)
