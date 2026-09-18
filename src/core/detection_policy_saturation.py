






from __future__ import annotations

from .detection_policy_core import (
    _is_finite_policy_value,
    saturation_policy,
    seed_policy,
)


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


def mask_cap_trigger_frac(fallback: float, policy: dict | None = None) -> float:



    return _sat_float("cap_trigger_frac", fallback, policy)


def subdiv_max_depth(fallback: int, policy: dict | None = None) -> int:

    val = _sat_float("subdiv_max_depth", float(fallback), policy)
    return int(val) if val >= 0 else fallback


def resplit_time_ratio(fallback: float, policy: dict | None = None) -> float:



    val = _sat_float("resplit_time_ratio", fallback, policy)
    return val if val >= 0 else fallback


def max_masks_per_tile(fallback: int, policy: dict | None = None) -> int:



    val = _sat_float("max_masks_per_tile", float(fallback), policy)
    return int(val) if val > 0 else fallback


def subdivide_overlap_fraction(fallback: float, policy: dict | None = None) -> float:



    val = _sat_float("subdivide_overlap_fraction", fallback, policy)
    return val if 0 <= val < 0.5 else fallback


def subdivide_min_parent_px(fallback: int, policy: dict | None = None) -> int:


    val = _sat_float("subdivide_min_parent_px", float(fallback), policy)
    return int(val) if val > 0 else fallback


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


def max_tile_coverage(fallback: float, policy: dict | None = None) -> float:


    return _sat_float("max_tile_coverage", fallback, policy)


def hard_tile_coverage(fallback: float, policy: dict | None = None) -> float:


    return _sat_float("hard_tile_coverage", fallback, policy)


def map_cover_score_floor(fallback: float, policy: dict | None = None) -> float:












    val = _sat_float("map_cover_score_floor", fallback, policy)
    return val if 0.0 < val <= 1.0 else fallback


def compact_min_fill(fallback: float, policy: dict | None = None) -> float:





    val = _sat_float("compact_min_fill", fallback, policy)
    return val if 0.0 < val <= 1.0 else fallback


def tile_span_fraction(fallback: float, policy: dict | None = None) -> float:






    val = _sat_float("tile_span_fraction", fallback, policy)
    return val if 0.0 < val <= 1.0 else fallback


def hard_cover_shape_escape(fallback: bool, policy: dict | None = None) -> bool:















    val = saturation_policy(policy).get("hard_cover_shape_escape")
    return val if isinstance(val, bool) else bool(fallback)


def min_keep_px(fallback: float, policy: dict | None = None) -> float:




    val = _sat_float("min_keep_px", fallback, policy)
    return val if val >= 0 else fallback


def min_keep_floor_m2(fallback: float, policy: dict | None = None) -> float:





    val = _sat_float("min_keep_floor_m2", fallback, policy)
    return val if val >= 0 else fallback
