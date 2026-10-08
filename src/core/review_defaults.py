









from __future__ import annotations

import math






AUTO_DEFAULT_CONFIDENCE = 0.30











AUTO_REVIEW_SIMPLIFY_DEFAULT = 0.0







AUTO_REVIEW_POINTS_PCT_DEFAULT = 100





AUTO_REVIEW_CLEAN_DEFAULT = 0.0


AUTO_REVIEW_SMOOTH_DEFAULT = False






AUTO_REVIEW_ORTHO_DEFAULT = False


AUTO_REVIEW_EXPAND_DEFAULT = 0








AUTO_REVIEW_CLOSE_NOTCHES_M_DEFAULT = 0.0





HOLE_NOISE_FALLBACK_M2 = 4.0



AUTO_REVIEW_FILL_HOLES_DEFAULT = True





AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT = HOLE_NOISE_FALLBACK_M2










def auto_review_simplify_default() -> float:

    try:
        from .server_dials import dial_in_range

        return dial_in_range(
            "tuning.review.simplify_default", AUTO_REVIEW_SIMPLIFY_DEFAULT, 0, 50)
    except Exception:  # noqa: BLE001  # nosec B110
        return AUTO_REVIEW_SIMPLIFY_DEFAULT


def auto_review_clean_default() -> float:

    try:
        from .server_dials import dial_in_range

        return dial_in_range(
            "tuning.review.clean_default", AUTO_REVIEW_CLEAN_DEFAULT, 0, 20)
    except Exception:  # noqa: BLE001  # nosec B110
        return AUTO_REVIEW_CLEAN_DEFAULT


def auto_review_smooth_default() -> bool:

    try:
        from .server_dials import dial_bool

        return dial_bool("tuning.review.smooth_default", AUTO_REVIEW_SMOOTH_DEFAULT)
    except Exception:  # noqa: BLE001  # nosec B110
        return AUTO_REVIEW_SMOOTH_DEFAULT


def auto_review_ortho_default() -> bool:

    try:
        from .server_dials import dial_bool

        return dial_bool("tuning.review.ortho_default", AUTO_REVIEW_ORTHO_DEFAULT)
    except Exception:  # noqa: BLE001  # nosec B110
        return AUTO_REVIEW_ORTHO_DEFAULT


def auto_review_expand_default() -> int:

    try:
        from .server_dials import dial_in_range

        return dial_in_range(
            "tuning.review.expand_default", AUTO_REVIEW_EXPAND_DEFAULT, -100, 100)
    except Exception:  # noqa: BLE001  # nosec B110
        return AUTO_REVIEW_EXPAND_DEFAULT


def auto_review_fill_holes_default() -> bool:

    try:
        from .server_dials import dial_bool

        return dial_bool(
            "tuning.review.fill_holes_default", AUTO_REVIEW_FILL_HOLES_DEFAULT)
    except Exception:  # noqa: BLE001  # nosec B110
        return AUTO_REVIEW_FILL_HOLES_DEFAULT


def _fill_holes_floor_m2() -> float:




    try:
        from .detection_policy import fill_holes_floor_m2

        return fill_holes_floor_m2(AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT)
    except Exception:  # noqa: BLE001  # nosec B110
        return AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT


def fill_holes_max_m2_with_floor(class_fills: object, class_ceiling: object) -> float:











    floor = _fill_holes_floor_m2()
    if not class_fills:
        return floor
    if isinstance(class_ceiling, bool) or not isinstance(class_ceiling, (int, float)):
        return floor
    ceiling = float(class_ceiling)
    if ceiling == 0.0:
        return 0.0
    if ceiling < 0.0:
        return floor
    return max(ceiling, floor)














MIN_SIZE_NOISE_MASK_PX = 1.0
MIN_SIZE_NOISE_MASK_PX_NO_PROMPT = 1.0


def min_size_noise_floor_m2(mask_gsd_m: float, *, no_prompt: bool = False) -> float:







    if not mask_gsd_m or mask_gsd_m <= 0:
        return 0.0
    px = MIN_SIZE_NOISE_MASK_PX_NO_PROMPT if no_prompt else MIN_SIZE_NOISE_MASK_PX
    try:
        from .detection_policy import min_size_noise_px

        px = min_size_noise_px(no_prompt, px)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return (px * float(mask_gsd_m)) ** 2






















REFINE_SIMPLIFY_DEFAULT = 2.0







REFINE_SIMPLIFY_MAX_NARROW_FRACTION = 0.08








REFINE_VERTEX_MAX_DEVIATION_M = 0.25





REFINE_VERTEX_DEVIATION_PIXEL_FLOOR = 1.5
REFINE_SMOOTH_DEFAULT = 0




REFINE_SMOOTH_ITERATIONS = 1



REFINE_POINTS_PCT_DEFAULT = AUTO_REVIEW_POINTS_PCT_DEFAULT



REFINE_CLEAN_DEFAULT = 0.0
REFINE_EXPAND_DEFAULT = 0
REFINE_FILL_HOLES_DEFAULT = True






REFINE_FILL_HOLES_MAX_M2_DEFAULT = HOLE_NOISE_FALLBACK_M2


def refine_fill_holes_max_m2_default() -> float:






    try:
        from .detection_policy_core import config_detection_policy
        from .detection_policy_review import pinhole_fill_m

        side_m = pinhole_fill_m(0.0, config_detection_policy())
    except Exception:  # noqa: BLE001
        side_m = 0.0
    if math.isfinite(side_m) and side_m > 0:
        return side_m * side_m
    return REFINE_FILL_HOLES_MAX_M2_DEFAULT







REFINE_ORTHO_DEFAULT = False
REFINE_MIN_SIZE_M2_DEFAULT = 0.0



REFINE_MIN_AREA_DEFAULT = 200
















def area_passes_size_gates(area: float, params: dict) -> bool:




    min_a = params.get("min_a", 0.0)
    max_a = params.get("max_a", 0.0)
    if min_a > 0 and area < min_a:
        return False
    return not (max_a > 0 and area > max_a)


def object_passes_review_gates(score: float, area: float, params: dict) -> bool:






    if score < params.get("conf", 0.0):
        return False
    return area_passes_size_gates(area, params)




















_ADAPTIVE_KEYS = (
    "min_objects", "hidden_trigger", "area_ratio_lo", "area_ratio_hi",
    "min_anchor", "floor",
)

_REVIEW_SLIDER_STEP = 5


_GRID_SNAP_EPSILON = 1e-9


def _adaptive_params() -> tuple[int, float, float, float, int, float] | None:



    try:
        from .detection_policy import adaptive_confidence_policy

        pol = adaptive_confidence_policy()
        values = []
        for key in _ADAPTIVE_KEYS:
            v = pol.get(key)
            if (not isinstance(v, (int, float)) or isinstance(v, bool)
                    or not math.isfinite(v)):
                return None
            values.append(float(v))
    except Exception:  # noqa: BLE001  # nosec B110
        return None
    min_objects, hidden_trigger, lo, hi, min_anchor, floor = values
    return int(min_objects), hidden_trigger, lo, hi, int(min_anchor), floor


def adaptive_review_confidence(
    scored: list[tuple[float, float]],
    default: float = AUTO_DEFAULT_CONFIDENCE,
    merge_separate: bool = True,
) -> float | None:











    params = _adaptive_params()
    if params is None:
        return None
    (min_objects, hidden_trigger, ratio_lo, ratio_hi,
     min_anchor, floor) = params
    n = len(scored)
    if n < min_objects:
        return None
    below = [(s, a) for s, a in scored if s < default]
    if len(below) / n <= hidden_trigger:
        return None
    if merge_separate:
        low_areas = sorted(a for s, a in below if a > 0)
        high_areas = sorted(a for s, a in scored if s >= default and a > 0)
        if len(high_areas) < min_anchor or not low_areas:
            return None
        med_low = low_areas[len(low_areas) // 2]
        med_high = high_areas[len(high_areas) // 2]
        if med_high <= 0:
            return None
        ratio = med_low / med_high
        if not (ratio_lo <= ratio <= ratio_hi):
            return None
    low_scores = sorted(s for s, _a in below)
    p25 = low_scores[int(0.25 * (len(low_scores) - 1))]
    cutoff = max(floor, p25)
    grid = _REVIEW_SLIDER_STEP
    step = int(cutoff * 100 / grid) * grid



    floor_step = int(math.ceil(floor * 100 / grid - _GRID_SNAP_EPSILON)) * grid
    step = max(step, floor_step)
    step = min(step, int(round(default * 100)) - grid)
    if step < floor_step:
        return None
    return step / 100.0
