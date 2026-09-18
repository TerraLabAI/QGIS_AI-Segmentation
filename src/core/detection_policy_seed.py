







from __future__ import annotations

from .detection_policy_core import (
    _is_finite_policy_value,
    seed_policy,
)
from .prompt_taxonomy import first_entry_match, iter_keywords, keyword_matches, normalize_prompt
from .tile_manager import (
    AUTO_OBJECT_MIN_PX,
    AUTO_SEED_HEADROOM_LEVELS,
    AUTO_SEED_TILE_CAP,
    DEFAULT_AUTO_TILE_BUDGET,
    DEFAULT_SEED_MUPP_M,
    DETAIL_COARSE_TRAVEL_RATIO,
    DETAIL_FINE_TRAVEL_RATIO,
    DRAWN_OBJECT_TILE_FRAC,
    MASK_SCALE_MIN_WIDTH_PX,
    NATIVE_OVERSAMPLE_MAX,
    QUALITY_FLOOR_MUPP_M,
    SPLIT_RISK_TILE_FRAC,
    SWEET_SPOT_MAX_MUPP_M,
)


def max_concurrent(policy: dict | None = None) -> int:




    val = seed_policy(policy).get("max_concurrent")
    if _is_finite_policy_value(val) and 1 <= val <= 32:
        return int(val)
    return 6


def object_profile(prompt: str, policy: dict | None = None) -> tuple[float, float]:













    generic = (10.0, DEFAULT_SEED_MUPP_M)
    seed = seed_policy(policy)
    tiers = seed.get("object_tiers")
    if not isinstance(tiers, list):
        return generic
    text = normalize_prompt(prompt)
    tier = first_entry_match(text, tiers)
    if tier is not None:
        return _profile_pair(tier, generic)
    return _profile_pair(seed.get("default_object"), generic)


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


def mask_scale_policy(policy: dict | None = None) -> dict:





    val = seed_policy(policy).get("mask_scale")
    return val if isinstance(val, dict) else {}


def mask_scale_min_width_px(policy: dict | None = None) -> float:




    val = mask_scale_policy(policy).get("min_width_px")
    if _is_finite_policy_value(val) and val > 0:
        return float(val)
    return MASK_SCALE_MIN_WIDTH_PX


def _matched_tiers(text: str, policy: dict | None) -> list[dict]:






    tiers = seed_policy(policy).get("object_tiers")
    if not isinstance(tiers, list):
        return []
    return [
        tier
        for tier in tiers
        if isinstance(tier, dict) and any(keyword_matches(text, kw) for kw in iter_keywords(tier))
    ]


def _entry_min_width_m(entry: dict) -> float | None:








    val = entry.get("min_width_m")
    if _is_finite_policy_value(val) and val > 0:
        return float(val)
    return None


def _entry_max_mupp(entry: dict) -> float | None:



    val = entry.get("max_mupp")
    if _is_finite_policy_value(val) and val > 0:
        return float(val)
    return None


def _prompt_names_an_unlisted_object(
    text: str, classes: list, policy: dict | None
) -> bool:







    class_keywords = [
        ckw for entry in classes if isinstance(entry, dict) for ckw in iter_keywords(entry)
    ]
    for tier in _matched_tiers(text, policy):
        for tier_kw in iter_keywords(tier):
            if not keyword_matches(text, tier_kw):
                continue



            if not any(keyword_matches(tier_kw, ckw) for ckw in class_keywords):
                return True
    return False


def mask_scale_for_run(
    prompt: str, run_mupp: float, policy: dict | None = None
) -> int:























    if not _is_finite_policy_value(run_mupp):
        return 1
    if run_mupp <= 0:
        return 1
    text = normalize_prompt(prompt)
    if not text:
        return 1
    classes = mask_scale_policy(policy).get("classes")
    if not isinstance(classes, list):
        return 1





    usable = [
        item
        for item in classes
        if isinstance(item, dict) and (_entry_min_width_m(item) is not None or _entry_max_mupp(item) is not None)
    ]
    entry = first_entry_match(text, usable)
    if entry is None:
        return 1
    if _prompt_names_an_unlisted_object(text, classes, policy):
        return 1

    width = _entry_min_width_m(entry)
    cap = _entry_max_mupp(entry)
    if width is not None and width / float(run_mupp) < mask_scale_min_width_px(policy):
        return 1
    if cap is not None and float(run_mupp) > cap:
        return 1
    return 2


def _seed_float(key: str, fallback: float, policy: dict | None) -> float:

    val = seed_policy(policy).get(key)
    if _is_finite_policy_value(val):
        return float(val)
    return fallback


def zone_seed_mupp(policy: dict | None = None) -> float:


    val = _seed_float("zone_seed_mupp", DEFAULT_SEED_MUPP_M, policy)
    return val if val > 0 else DEFAULT_SEED_MUPP_M


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


    val = int(_seed_float("object_min_px", AUTO_OBJECT_MIN_PX, policy))
    return val if val > 0 else AUTO_OBJECT_MIN_PX


def _travel_ratio(key: str, fallback: float, policy: dict | None) -> float:
    val = _seed_float(key, fallback, policy)
    return val if 1.0 <= val <= 8.0 else fallback


def detail_coarse_travel_ratio(policy: dict | None = None) -> float:




    return _travel_ratio("detail_coarse_travel_ratio",
                         DETAIL_COARSE_TRAVEL_RATIO, policy)


def detail_fine_travel_ratio(policy: dict | None = None) -> float:




    return _travel_ratio("detail_fine_travel_ratio",
                         DETAIL_FINE_TRAVEL_RATIO, policy)


def drawn_object_tile_frac(policy: dict | None = None) -> float:













    val = _seed_float("drawn_object_tile_frac", DRAWN_OBJECT_TILE_FRAC, policy)
    return val if 0 < val <= 1 else DRAWN_OBJECT_TILE_FRAC


def sweet_spot_max_mupp(policy: dict | None = None) -> float:









    return _seed_float("sweet_spot_max_mupp", SWEET_SPOT_MAX_MUPP_M, policy)


def quality_floor_mupp(policy: dict | None = None) -> float:

    return _seed_float("quality_floor_mupp", QUALITY_FLOOR_MUPP_M, policy)


def native_oversample_max(policy: dict | None = None) -> float:

    return _seed_float("native_oversample_max", NATIVE_OVERSAMPLE_MAX, policy)


def detail_over_ratio(policy: dict | None = None) -> float:





    return _seed_float("detail_over_ratio", 0.4, policy)


def detail_over_ratio_free(policy: dict | None = None) -> float:


    return _seed_float("detail_over_ratio_free", 0.5, policy)


def split_risk_tile_frac(policy: dict | None = None) -> float:





    val = _seed_float("split_risk_tile_frac", SPLIT_RISK_TILE_FRAC, policy)
    return val if 0.0 < val <= 1.0 else SPLIT_RISK_TILE_FRAC


def recall_floor(fallback: float, policy: dict | None = None) -> float:




    return _seed_float("recall_floor", fallback, policy)


def recall_floor_exemplar_only(fallback: float, policy: dict | None = None) -> float:


    return _seed_float("recall_floor_exemplar_only", fallback, policy)


def confidence_default(policy: dict | None = None) -> float:


    from .review_defaults import AUTO_DEFAULT_CONFIDENCE

    val = _seed_float("confidence_default", AUTO_DEFAULT_CONFIDENCE, policy)
    return val if 0.0 <= val <= 1.0 else AUTO_DEFAULT_CONFIDENCE


def confidence_default_exemplar_only(policy: dict | None = None) -> float:






    from .review_defaults import AUTO_DEFAULT_CONFIDENCE_EXEMPLAR_ONLY

    return _seed_float(
        "confidence_default_exemplar_only",
        AUTO_DEFAULT_CONFIDENCE_EXEMPLAR_ONLY, policy)


def gsd_warn_max_mupp(fallback: float, policy: dict | None = None) -> float:



    val = _seed_float("gsd_warn_max_mupp", fallback, policy)
    return val if val > 0 else fallback
