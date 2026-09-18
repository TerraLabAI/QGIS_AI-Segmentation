







from __future__ import annotations

from .detection_policy_core import (
    _is_finite_policy_value,
    review_policy,
)


def _review_float(key: str, fallback: float, policy: dict | None) -> float:

    val = review_policy(policy).get(key)
    if _is_finite_policy_value(val):
        return float(val)
    return fallback


def review_correct_default_method(policy: dict | None = None) -> str:






    val = review_policy(policy).get("correct_default_method")
    return val if val in ("ai", "manual") else "manual"


def review_correct_default_method_ready(policy: dict | None = None) -> str:












    val = review_policy(policy).get("correct_default_method_ready")
    if val in ("ai", "manual"):
        return val
    return review_correct_default_method(policy)


def review_noise_floor(policy: dict | None = None) -> float:





    val = review_policy(policy).get("noise_floor")
    if _is_finite_policy_value(val):
        f = float(val)
        if 0.0 <= f < 1.0:
            return f
    return 0.05


def click_unsure_below(policy: dict | None = None) -> float:












    val = review_policy(policy).get("click_unsure_below")
    if _is_finite_policy_value(val):
        f = float(val)
        if 0.0 <= f < 1.0:
            return f
    return 0.0


def pinhole_fill_m(fallback: float = 0.0, policy: dict | None = None) -> float:









    val = _review_float("pinhole_m", fallback, policy)
    return val if val > 0 else fallback


def tile_simplify_mult(fallback: float = 0.0, policy: dict | None = None) -> float:







    val = _review_float("tile_simplify_mult", fallback, policy)
    return val if val > 0 else fallback


def smooth_pass_settings(policy: dict | None = None) -> dict:











    src = review_policy(policy).get("smooth")
    src = src if isinstance(src, dict) else {}
    iterations = src.get("iterations")
    offset = src.get("offset")
    angle = src.get("max_angle_deg")
    return {
        "iterations": (
            int(iterations)
            if _is_finite_policy_value(iterations) and 1 <= iterations <= 5
            else 1
        ),
        "offset": (
            float(offset)
            if _is_finite_policy_value(offset) and 0.0 < offset <= 0.5
            else 0.25
        ),
        "max_angle_deg": (
            float(angle)
            if _is_finite_policy_value(angle) and 0.0 < angle <= 180.0
            else 120.0
        ),
    }


def min_size_noise_px(no_prompt: bool, fallback: float,
                      policy: dict | None = None) -> float:








    key = "min_size_noise_px_no_prompt" if no_prompt else "min_size_noise_px"
    val = _review_float(key, fallback, policy)
    return val if val > 0 else fallback


def fill_holes_floor_m2(fallback: float, policy: dict | None = None) -> float:







    val = review_policy(policy).get("fill_holes_floor_m2")
    if _is_finite_policy_value(val) and val >= 0:
        return float(val)
    return fallback


def vertex_budget_settings(policy: dict | None = None) -> dict:

































    pol = review_policy(policy).get("vertex_budget")
    pol = pol if isinstance(pol, dict) else {}

    def _zero_or_positive(value: object, fallback: float) -> float:


        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return float(value)
        return fallback

    min_v = pol.get("min_vertices")
    if not isinstance(min_v, int) or isinstance(min_v, bool) or min_v < 3:
        min_v = 8
    smooth_min = pol.get("smooth_min_vertices")
    if (not isinstance(smooth_min, int) or isinstance(smooth_min, bool) or smooth_min < 3):
        smooth_min = max(3, min_v // 2)
    return {
        "spacing_m": _zero_or_positive(pol.get("spacing_m"), 6.0),
        "min_vertices": int(min_v),
        "max_deviation_m": _zero_or_positive(pol.get("max_deviation_m"), 1.0),
        "max_deviation_fraction": _zero_or_positive(
            pol.get("max_deviation_fraction"), 0.10),





        "smooth_spacing_factor": _zero_or_positive(
            pol.get("smooth_spacing_factor"), 2.0),
        "smooth_max_deviation_m": _zero_or_positive(
            pol.get("smooth_max_deviation_m"), 2.0),
        "smooth_min_vertices": int(smooth_min),
        "dial_max_cap_fraction": _zero_or_positive(
            pol.get("dial_max_cap_fraction"), 0.5),
        "smooth_multiplier_cap": _zero_or_positive(
            pol.get("smooth_multiplier_cap"), 8.0),
    }


_CANOPY_HINT_FALLBACK = frozenset({"tree", "trees", "canopy", "forest"})


def prompt_suggests_canopy(prompt: str, policy: dict | None = None) -> bool:






    norm = (prompt or "").strip().lower().replace("_", " ")
    if not norm:
        return False
    tokens = review_policy(policy).get("canopy_hint_tokens")
    words: frozenset[str] = _CANOPY_HINT_FALLBACK
    if isinstance(tokens, list):
        server = frozenset(
            str(v).strip().lower().replace("_", " ")
            for v in tokens if isinstance(v, str) and str(v).strip())
        if server:
            words = server
    return norm in words or any(w in words for w in norm.split())


_CLOSED_CANOPY_FALLBACK = {"max_raw_per_tile": 8.0, "min_span_dropped": 3, "min_tiles": 12}


def closed_canopy_signature(raw_total: int, tiles: int, span_dropped: int,
                            policy: dict | None = None) -> bool:









    dial = review_policy(policy).get("closed_canopy_advice")
    cfg = dict(_CLOSED_CANOPY_FALLBACK)
    if isinstance(dial, dict):
        for key in cfg:
            v = dial.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
                cfg[key] = float(v)
    try:
        tiles = int(tiles)
        raw_total = int(raw_total)
        span_dropped = int(span_dropped)
    except (TypeError, ValueError):
        return False
    if tiles < cfg["min_tiles"] or tiles <= 0:
        return False
    if span_dropped < cfg["min_span_dropped"]:
        return False
    return (raw_total / tiles) <= cfg["max_raw_per_tile"]


def adaptive_confidence_policy(policy: dict | None = None) -> dict:


    val = review_policy(policy).get("adaptive_confidence")
    return val if isinstance(val, dict) else {}


def semantic_rescue_policy(policy: dict | None = None) -> dict:



    val = review_policy(policy).get("semantic_rescue")
    return val if isinstance(val, dict) else {}


def semantic_rescue_enabled(policy: dict | None = None) -> bool:



    return semantic_rescue_policy(policy).get("enabled") is True


def semantic_rescue_coverage_floor(policy: dict | None = None) -> float:




    val = semantic_rescue_policy(policy).get("coverage_floor")
    if _is_finite_policy_value(val) and 0.0 <= val <= 1.0:
        return float(val)
    return 0.45


def fp_filter_policy(policy: dict | None = None) -> dict:




    val = review_policy(policy).get("fp_filter")
    return val if isinstance(val, dict) else {}


def fp_rules(shape_class: str, policy: dict | None = None) -> list[dict]:







    raw = fp_filter_policy(policy).get(shape_class)
    if not isinstance(raw, list):
        return []
    from .geometry_attrs import FP_ACTIONS, FP_ATTRS, FP_OPS

    out: list[dict] = []
    for rule in raw:
        if not isinstance(rule, dict):
            continue
        attr = rule.get("attr")
        op = rule.get("op")
        action = rule.get("action")
        value = rule.get("value")
        if attr not in FP_ATTRS or op not in FP_OPS or action not in FP_ACTIONS:
            continue
        if not _is_finite_policy_value(value):
            continue
        out.append({"attr": attr, "op": op, "value": float(value), "action": action})
    return out
