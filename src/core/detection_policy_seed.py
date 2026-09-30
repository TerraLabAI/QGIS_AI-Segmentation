







from __future__ import annotations

from .detection_policy_core import (
    _is_finite_policy_value,
    policy_scope,
    seed_policy,
)
from .prompt_taxonomy import first_entry_match, normalize_prompt
from .served_config import require_served_int, require_served_number
from .tile_manager import (
    AUTO_SEED_HEADROOM_LEVELS,
    AUTO_SEED_TILE_CAP,
    DEFAULT_AUTO_TILE_BUDGET,
    NATIVE_OVERSAMPLE_MAX,
    QUALITY_FLOOR_MUPP_M,
)


def max_concurrent(policy: dict | None = None) -> int:




    val = seed_policy(policy).get("max_concurrent")
    if _is_finite_policy_value(val) and 1 <= val <= 32:
        return int(val)
    return 6


def object_profile(prompt: str, policy: dict | None = None) -> tuple[float, float]:














    seed = seed_policy(policy)
    tiers = seed.get("object_tiers")
    generic = (10.0, zone_seed_mupp(policy)) if not isinstance(tiers, list) else None
    if generic is not None:
        return generic
    text = normalize_prompt(prompt)
    tier = first_entry_match(text, tiers)
    fallback_pair = (10.0, 0.0)
    if tier is not None:
        pair = _profile_pair(tier, fallback_pair)
    else:
        pair = _profile_pair(seed.get("default_object"), fallback_pair)
    return pair if pair[1] > 0 else (pair[0], zone_seed_mupp(policy))


def _profile_pair(entry: object, fallback: tuple[float, float]) -> tuple[float, float]:

    if isinstance(entry, dict):
        try:
            size_m = entry["size_m"]
            target_mupp = entry["target_mupp"]
        except KeyError:
            return fallback
        if _is_finite_policy_value(size_m) and _is_finite_policy_value(target_mupp):
            return float(size_m), float(target_mupp)
        return fallback
    return fallback


def object_tile_floor_m(prompt: str, policy: dict | None = None) -> float:













    seed = seed_policy(policy)
    entry: object = seed.get("default_object")
    tiers = seed.get("object_tiers")
    if isinstance(tiers, list):
        tier = first_entry_match(normalize_prompt(prompt), tiers)
        if tier is not None:
            entry = tier
    if isinstance(entry, dict):
        val = entry.get("min_tile_ground_m")
        if _is_finite_policy_value(val) and val > 0:
            return float(val)
    return 0.0


def object_tile_ceiling_m(prompt: str, policy: dict | None = None) -> float:














    seed = seed_policy(policy)
    entry: object = seed.get("default_object")
    tiers = seed.get("object_tiers")
    if isinstance(tiers, list):
        tier = first_entry_match(normalize_prompt(prompt), tiers)
        if tier is not None:
            entry = tier
    if isinstance(entry, dict):
        val = entry.get("max_tile_ground_m")
        if _is_finite_policy_value(val) and val > 0:
            return float(val)
    return 0.0


def _seed_float(key: str, fallback: float, policy: dict | None) -> float:

    val = seed_policy(policy).get(key)
    if _is_finite_policy_value(val):
        return float(val)
    return fallback


def zone_seed_mupp(policy: dict | None = None) -> float:


    with policy_scope(policy):
        return require_served_number("detection_policy.seed.zone_seed_mupp", 0.01, 10.0)


def soft_tile_budget(policy: dict | None = None) -> int:

    val = int(_seed_float("soft_tile_budget", DEFAULT_AUTO_TILE_BUDGET, policy))
    return val if val > 0 else DEFAULT_AUTO_TILE_BUDGET


def seed_tile_cap(policy: dict | None = None) -> int:

    val = int(_seed_float("seed_tile_cap", AUTO_SEED_TILE_CAP, policy))
    return val if val > 0 else AUTO_SEED_TILE_CAP


def seed_headroom_levels(policy: dict | None = None) -> int:



    val = int(_seed_float("seed_headroom_levels",
                          float(AUTO_SEED_HEADROOM_LEVELS), policy))
    return val if 0 <= val <= 10 else AUTO_SEED_HEADROOM_LEVELS


def free_run_fraction(policy: dict | None = None) -> float:







    val = _seed_float("free_run_fraction", 1.0, policy)
    return val if 0.0 < val <= 1.0 else 1.0


def free_monthly_allowance(policy: dict | None = None) -> int:








    val = int(_seed_float("free_monthly_allowance", 200.0, policy))
    return val if val > 0 else 200


def free_zone_max_km2(fallback: float, policy: dict | None = None) -> float:





    val = _seed_float("free_zone_max_km2", fallback, policy)
    return val if val > 0 else fallback


def max_tiles_per_run(fallback: int, policy: dict | None = None) -> int:




    val = int(_seed_float("max_tiles_per_run", float(fallback), policy))
    return val if val > 0 else fallback


def max_tiles_per_km2(fallback: float, policy: dict | None = None) -> float:



    val = _seed_float("max_tiles_per_km2", float(fallback), policy)
    return val if val > 0 else fallback


def max_tiles_floor(fallback: int, policy: dict | None = None) -> int:


    val = int(_seed_float("max_tiles_floor", float(fallback), policy))
    return val if val > 0 else fallback


def tile_jpeg_quality(fallback: int, policy: dict | None = None) -> int:





    val = int(_seed_float("tile_jpeg_quality", float(fallback), policy))
    return min(100, max(60, val)) if val > 0 else fallback


def object_min_px(policy: dict | None = None) -> int:


    with policy_scope(policy):
        return require_served_int("detection_policy.seed.object_min_px", 1, 1000)


def detail_coarse_travel_ratio(policy: dict | None = None) -> float:




    with policy_scope(policy):
        return require_served_number("detection_policy.seed.detail_coarse_travel_ratio", 1.0, 8.0)


def detail_fine_travel_ratio(policy: dict | None = None) -> float:




    with policy_scope(policy):
        return require_served_number("detection_policy.seed.detail_fine_travel_ratio", 1.0, 8.0)


def drawn_object_tile_frac(policy: dict | None = None) -> float:













    with policy_scope(policy):
        return require_served_number("detection_policy.seed.drawn_object_tile_frac", 0.01, 1.0)


def exemplar_size_ladder(policy: dict | None = None) -> list[tuple[float, float]]:








    raw = seed_policy(policy).get("exemplar_size_ladder")
    if not isinstance(raw, list):
        return []
    rungs: list[tuple[float, float]] = []
    for rung in raw:
        if not isinstance(rung, dict):
            continue
        size = rung.get("size_m")
        mupp = rung.get("target_mupp")
        if (_is_finite_policy_value(size) and _is_finite_policy_value(mupp)
                and size > 0 and mupp > 0):
            rungs.append((float(size), float(mupp)))
    return rungs


def exemplar_ladder_mupp(size_m: float, policy: dict | None = None) -> float:



    import math

    if not (isinstance(size_m, (int, float)) and math.isfinite(size_m) and size_m > 0):
        return 0.0
    best = 0.0
    best_dist = math.inf
    for rung_size, rung_mupp in exemplar_size_ladder(policy):
        dist = abs(math.log(rung_size) - math.log(size_m))
        if dist < best_dist or (dist == best_dist and rung_mupp < best):
            best, best_dist = rung_mupp, dist
    return best


def exemplar_band_enabled(policy: dict | None = None) -> bool:




    block = seed_policy(policy).get("tile_plan")
    return not (isinstance(block, dict) and block.get("exemplar_band") is False)


def sweet_spot_max_mupp(policy: dict | None = None) -> float:









    with policy_scope(policy):
        return require_served_number("detection_policy.seed.sweet_spot_max_mupp", 0.01, 10.0)


def quality_floor_mupp(policy: dict | None = None) -> float:

    return _seed_float("quality_floor_mupp", QUALITY_FLOOR_MUPP_M, policy)


def native_oversample_max(policy: dict | None = None) -> float:

    return _seed_float("native_oversample_max", NATIVE_OVERSAMPLE_MAX, policy)


def detail_over_ratio(policy: dict | None = None) -> float:





    return _seed_float("detail_over_ratio", 0.0, policy)


def detail_over_ratio_free(policy: dict | None = None) -> float:


    return _seed_float("detail_over_ratio_free", 0.0, policy)


def split_risk_tile_frac(policy: dict | None = None) -> float:





    with policy_scope(policy):
        return require_served_number("detection_policy.seed.split_risk_tile_frac", 0.01, 1.0)


def recall_floor(policy: dict | None = None) -> float:



    with policy_scope(policy):
        return require_served_number("detection_policy.seed.recall_floor", 0.0, 1.0)


def recall_floor_exemplar_only(policy: dict | None = None) -> float:


    with policy_scope(policy):
        return require_served_number(
            "detection_policy.seed.recall_floor_exemplar_only", 0.0, 1.0)


def confidence_default(policy: dict | None = None) -> float:


    from .review_defaults import AUTO_DEFAULT_CONFIDENCE

    val = _seed_float("confidence_default", AUTO_DEFAULT_CONFIDENCE, policy)
    return val if 0.0 <= val <= 1.0 else AUTO_DEFAULT_CONFIDENCE


def confidence_default_exemplar_only(policy: dict | None = None) -> float:






    from .review_defaults import AUTO_DEFAULT_CONFIDENCE

    val = _seed_float(
        "confidence_default_exemplar_only", AUTO_DEFAULT_CONFIDENCE, policy)
    return val if 0.0 <= val <= 1.0 else AUTO_DEFAULT_CONFIDENCE


def gsd_warn_max_mupp(fallback: float, policy: dict | None = None) -> float:



    val = _seed_float("gsd_warn_max_mupp", fallback, policy)
    return val if val > 0 else fallback
