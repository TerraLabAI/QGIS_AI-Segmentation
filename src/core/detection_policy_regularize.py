






from __future__ import annotations

from typing import TYPE_CHECKING

from .detection_policy_core import (
    _is_finite_policy_value,
    auto_regularize_policy,
    review_policy,
)
from .prompt_taxonomy import keyword_matches, normalize_prompt

if TYPE_CHECKING:
    from .building_regularizer import RegularizePolicy



_REGULARIZE_FALLBACK_KEYWORDS: tuple[str, ...] = (
    "building",
    "rooftop",
    "roof",
    "house",
    "solar",
    "panel",
    "pv",
)




_REGULARIZE_FALLBACK_TOLERANCE_M = 1.0




_REGULARIZE_FALLBACK_OBJECT_FRACTION = 0.25







_REGULARIZE_FALLBACK_DIAGONAL_REDUCTION = 8.0

_REGULARIZE_DIAGONAL_REDUCTION_MAX = 22.5



_REGULARIZE_FALLBACK_CIRCLE_THRESHOLD = 0.94




_DESTAIR_FALLBACK_MULT = 2.5





_REGULARIZE_FALLBACK_MULTI_DIRECTION = False

_REGULARIZE_FALLBACK_MULTI_MAX_GROUPS = 3

_REGULARIZE_MULTI_MAX_GROUPS_MAX = 6

_REGULARIZE_FALLBACK_MULTI_MIN_SEPARATION_DEG = 10.0

_REGULARIZE_MULTI_MIN_SEPARATION_MAX = 45.0





_REGULARIZE_FALLBACK_MULTI_PARALLEL_EPS_DEG = 1.5

_REGULARIZE_FALLBACK_MULTI_MIN_GROUP_WEIGHT = 0.20


def _positive_number(value: object) -> float | None:

    if _is_finite_policy_value(value) and value > 0:
        return float(value)
    return None


def regularize_policy(policy: dict | None = None) -> dict:



    val = review_policy(policy).get("regularize")
    return val if isinstance(val, dict) else {}


def regularize_settings(policy: dict | None = None) -> dict:
































    reg = regularize_policy(policy)
    kws = reg.get("keywords")
    if isinstance(kws, list):
        keywords = tuple(k.lower() for k in kws if isinstance(k, str) and k.strip())
    else:
        keywords = ()
    if not keywords:
        keywords = _REGULARIZE_FALLBACK_KEYWORDS

    def _num(key: str, fallback: float) -> float:
        v = reg.get(key)
        if _is_finite_policy_value(v):
            return float(v)
        return fallback

    def _flag(key: str, fallback: bool) -> bool:
        v = reg.get(key)
        return bool(v) if isinstance(v, bool) else fallback

    tolerance_m = _num("tolerance_m", _REGULARIZE_FALLBACK_TOLERANCE_M)
    if tolerance_m <= 0:
        tolerance_m = _REGULARIZE_FALLBACK_TOLERANCE_M
    fraction = _num("max_object_fraction", _REGULARIZE_FALLBACK_OBJECT_FRACTION)
    if not 0 < fraction <= 1:
        fraction = _REGULARIZE_FALLBACK_OBJECT_FRACTION
    reduction = _num("diagonal_reduction", _REGULARIZE_FALLBACK_DIAGONAL_REDUCTION)
    if not 0 <= reduction < _REGULARIZE_DIAGONAL_REDUCTION_MAX:
        reduction = _REGULARIZE_FALLBACK_DIAGONAL_REDUCTION
    circle = _num("circle_threshold", _REGULARIZE_FALLBACK_CIRCLE_THRESHOLD)
    if not 0 < circle <= 1:
        circle = _REGULARIZE_FALLBACK_CIRCLE_THRESHOLD
    max_groups = int(_num(
        "multi_max_groups", float(_REGULARIZE_FALLBACK_MULTI_MAX_GROUPS)))
    if not 1 <= max_groups <= _REGULARIZE_MULTI_MAX_GROUPS_MAX:
        max_groups = _REGULARIZE_FALLBACK_MULTI_MAX_GROUPS
    separation = _num(
        "multi_min_separation_deg", _REGULARIZE_FALLBACK_MULTI_MIN_SEPARATION_DEG)
    if not 0 < separation <= _REGULARIZE_MULTI_MIN_SEPARATION_MAX:
        separation = _REGULARIZE_FALLBACK_MULTI_MIN_SEPARATION_DEG
    parallel_eps = _num(
        "multi_parallel_eps_deg", _REGULARIZE_FALLBACK_MULTI_PARALLEL_EPS_DEG)
    if not 0 < parallel_eps <= _REGULARIZE_MULTI_MIN_SEPARATION_MAX:
        parallel_eps = _REGULARIZE_FALLBACK_MULTI_PARALLEL_EPS_DEG
    group_weight = _num(
        "multi_min_group_weight", _REGULARIZE_FALLBACK_MULTI_MIN_GROUP_WEIGHT)
    if not 0 <= group_weight < 1:
        group_weight = _REGULARIZE_FALLBACK_MULTI_MIN_GROUP_WEIGHT

    return {
        "keywords": keywords,
        "tolerance_m": tolerance_m,
        "max_object_fraction": fraction,
        "tolerance_mult": _num("tolerance_mult", _DESTAIR_FALLBACK_MULT),
        "allow_diagonal": _flag("allow_diagonal", True),
        "diagonal_reduction": reduction,
        "allow_circles": _flag("allow_circles", False),
        "circle_threshold": circle,
        "min_keep_iou": _num("min_keep_iou", 0.7),




        "ring_min_iou": _num("ring_min_iou", 0.1),
        "multi_direction": _flag(
            "multi_direction", _REGULARIZE_FALLBACK_MULTI_DIRECTION),
        "multi_max_groups": max_groups,
        "multi_min_separation_deg": separation,
        "multi_parallel_eps_deg": parallel_eps,
        "multi_min_group_weight": group_weight,
    }


def regularize_tolerance_m(
    pixel_size_m: float,
    object_size_m: float = 0.0,
    policy: dict | None = None,
) -> float:



























    reg = regularize_policy(policy)
    settings = regularize_settings(policy)
    pixel_m = _positive_number(pixel_size_m) or 0.0

    tolerance = settings["tolerance_m"]
    if _positive_number(reg.get("tolerance_m")) is None:
        legacy_mult = _positive_number(reg.get("tolerance_mult"))
        if legacy_mult is not None and pixel_m > 0:
            tolerance = legacy_mult * pixel_m

    size_m = _positive_number(object_size_m)
    if size_m is not None:
        tolerance = min(tolerance, settings["max_object_fraction"] * size_m)
    if pixel_m > 0:
        tolerance = max(tolerance, pixel_m)
    return tolerance


def destair_tolerance_m(pixel_size_m: float, policy: dict | None = None) -> float:


















    reg = regularize_policy(policy)
    pixel_m = _positive_number(pixel_size_m) or 0.0

    ground = _positive_number(reg.get("destair_m"))
    if ground is not None:
        return ground
    mult = _positive_number(reg.get("destair_mult")) or _DESTAIR_FALLBACK_MULT
    return mult * pixel_m


def regularize_envelope(policy: dict | None = None) -> RegularizePolicy:






























    from .building_regularizer import RegularizePolicy




    try:
        env = regularize_policy(policy).get("envelope")
        if not isinstance(env, dict):
            return RegularizePolicy()

        def _flag(key: str) -> bool:
            v = env.get(key)
            return v if isinstance(v, bool) else False

        def _pos(key: str) -> float:
            return _positive_number(env.get(key)) or 0.0

        holes = env.get("max_holes")
        max_holes = int(holes) if isinstance(holes, int) and not isinstance(holes, bool) and holes >= 0 else -1

        rect = _positive_number(env.get("min_rectangularity")) or 0.0
        if not 0 < rect <= 1:
            rect = 0.0

        fill = _positive_number(env.get("rectangle_area_fill"))
        fill = fill if fill is not None and 0 < fill <= 1 else 0.95
        aspect = _positive_number(env.get("rectangle_min_aspect")) or 1.2

        return RegularizePolicy(
            envelope_enabled=_flag("envelope_enabled"),
            max_area_ratio=_pos("max_area_ratio"),
            min_area_ratio=_pos("min_area_ratio"),
            max_hausdorff_mult=_pos("max_hausdorff_mult"),
            max_vertex_growth=_pos("max_vertex_growth"),
            enforce_component_count=_flag("enforce_component_count"),
            enforce_hole_count=_flag("enforce_hole_count"),
            eligibility_enabled=_flag("eligibility_enabled"),
            min_rectangularity=rect,
            max_holes=max_holes,
            rectangle_enabled=_flag("rectangle_enabled"),
            rectangle_area_fill=fill,
            rectangle_min_aspect=aspect,
        )
    except Exception:  # noqa: BLE001  # nosec B110
        return RegularizePolicy()


def manual_simplify_multiple_of_px(policy: dict | None = None) -> float:



    v = regularize_policy(policy).get("manual_simplify_mult")
    if _is_finite_policy_value(v) and v >= 0:
        return float(v)
    return 0.0


def progressive_merge_enabled(policy: dict | None = None) -> bool:



    return regularize_policy(policy).get("progressive_merge_enabled") is True


def despike_tolerance_m(pixel_size_m: float, policy: dict | None = None) -> float:












    reg = regularize_policy(policy)
    ground = _positive_number(reg.get("despike_m"))
    return ground if ground is not None else 0.0


def regularize_enabled_for(prompt: str, policy: dict | None = None) -> bool:










    text = normalize_prompt(prompt)
    if not text:
        return False
    keywords = regularize_settings(policy)["keywords"]
    return any(keyword_matches(text, kw) for kw in keywords)







_AUTO_REGULARIZE_DEFAULTS: dict[str, float | int] = {
    "simplify_gsd_factor": 3.0,
    "simplify_min_m": 0.6,
    "simplify_max_m": 1.5,
    "ortho_window_deg": 15.0,
    "diag_window_deg": 7.0,
    "min_edge_abs_m": 1.0,
    "min_edge_rel": 0.02,
    "min_corner_deg": 30.0,
    "parallel_threshold_m": 1.0,
    "circularity_skip": 0.90,
    "hist_bin_deg": 1.0,
    "bin_halo": 3,
    "mrr_when_top_below": 0.50,
    "consensus_radius_m": 100.0,
    "consensus_min_neighbours": 3,
    "consensus_iou_margin": 0.01,
    "guard_iou_floor": 0.85,
    "guard_area_ceiling": 0.10,
}


def auto_regularize_settings(shape_class: str,
                             policy: dict | None = None) -> dict | None:














    reg = auto_regularize_policy(policy)
    if reg.get("enabled") is not True:
        return None
    classes = reg.get("classes")
    if not isinstance(classes, list) or shape_class not in classes:
        return None
    return _auto_regularize_dials(reg)




_AUTO_REGULARIZE_ZERO_OK = frozenset({
    "diag_window_deg", "bin_halo", "min_edge_rel", "min_edge_abs_m",
    "consensus_iou_margin",
})






_AUTO_REGULARIZE_RANGES: dict[str, tuple[float, float]] = {
    "hist_bin_deg": (0.1, 45.0),
    "bin_halo": (0.0, 45.0),
}


def _auto_regularize_dials(reg: dict) -> dict:










    out: dict[str, float | int | bool] = {
        "revert_to_simplified": reg.get("revert_to_simplified") is True,
    }
    for key, fallback in _AUTO_REGULARIZE_DEFAULTS.items():
        val = reg.get(key)
        if _is_finite_policy_value(val) and (
                val >= 0 if key in _AUTO_REGULARIZE_ZERO_OK else val > 0):
            bounds = _AUTO_REGULARIZE_RANGES.get(key)
            if bounds is not None and not bounds[0] <= val <= bounds[1]:
                out[key] = fallback
                continue
            out[key] = int(val) if isinstance(fallback, int) else float(val)
        else:
            out[key] = fallback
    return out


def manual_save_alignment_settings(policy: dict | None = None) -> dict | None:












    reg = auto_regularize_policy(policy)
    if reg.get("enabled") is not True:
        return None
    dials = _auto_regularize_dials(reg)
    dials["circularity_skip"] = 1.01
    return dials
