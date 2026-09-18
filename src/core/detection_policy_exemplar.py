






from __future__ import annotations

from .detection_policy_core import (
    _is_finite_policy_value,
    exemplar_policy,
)


def exemplar_context_pad(policy: dict | None = None) -> float:





    val = exemplar_policy(policy).get("context_pad")
    if _is_finite_policy_value(val):
        return float(val)
    return 0.05


def exemplar_context_pad_px_cap(policy: dict | None = None) -> float:




    val = exemplar_policy(policy).get("context_pad_px")
    if _is_finite_policy_value(val) and val > 0:
        return float(val)
    return 12.0


def exemplar_min_paste_scale(policy: dict | None = None) -> float:





    val = exemplar_policy(policy).get("min_paste_scale")
    if _is_finite_policy_value(val) and 0 < val <= 1:
        return float(val)
    return 0.85


def _exemplar_positive_int(key: str, fallback: int, policy: dict | None) -> int:

    val = exemplar_policy(policy).get(key)
    if _is_finite_policy_value(val) and val >= 1:
        return int(val)
    return fallback


def exemplar_stamp_max_px(fallback: int, policy: dict | None = None) -> int:



    return _exemplar_positive_int("stamp_max_px", fallback, policy)


def exemplar_stamp_pad_px(fallback: int, policy: dict | None = None) -> int:

    return _exemplar_positive_int("stamp_pad_px", fallback, policy)


def exemplar_max_positive(fallback: int, policy: dict | None = None) -> int:


    return _exemplar_positive_int("max_positive", fallback, policy)


def exemplar_max_exclude(fallback: int, policy: dict | None = None) -> int:


    return _exemplar_positive_int("max_exclude", fallback, policy)


def exemplar_max_region(fallback: int, policy: dict | None = None) -> int:


    return _exemplar_positive_int("max_region", fallback, policy)


def exemplar_max_total(fallback: int, policy: dict | None = None) -> int:



    return _exemplar_positive_int("max_total", fallback, policy)


def _exemplar_nonneg_int(key: str, fallback: int, policy: dict | None) -> int:


    val = exemplar_policy(policy).get(key)
    if _is_finite_policy_value(val) and val >= 0:
        return int(val)
    return fallback


def exemplar_max_positive_free(fallback: int, policy: dict | None = None) -> int:


    return _exemplar_nonneg_int("max_positive_free", fallback, policy)


def exemplar_max_exclude_free(fallback: int, policy: dict | None = None) -> int:


    return _exemplar_nonneg_int("max_exclude_free", fallback, policy)


def exemplar_min_example_positives(fallback: int, policy: dict | None = None) -> int:




    return _exemplar_positive_int("min_example_positives", fallback, policy)


def exemplar_min_meta_positives(fallback: int, policy: dict | None = None) -> int:


    return _exemplar_positive_int("min_meta_positives", fallback, policy)


def exemplar_render_min_side_px(fallback: int, policy: dict | None = None) -> int:




    return _exemplar_positive_int("render_min_side_px", fallback, policy)


def exemplar_render_max_side_px(fallback: int, policy: dict | None = None) -> int:



    return _exemplar_positive_int("render_max_side_px", fallback, policy)


def exemplar_render_side_bounds(min_fallback: int, max_fallback: int,
                                policy: dict | None = None) -> tuple[int, int]:






    low = exemplar_render_min_side_px(min_fallback, policy)
    high = exemplar_render_max_side_px(max_fallback, policy)
    if low > high:
        return int(min_fallback), int(max_fallback)
    return low, high


def exemplar_render_abs_min_side_px(fallback: int, policy: dict | None = None) -> int:




    return _exemplar_positive_int("render_abs_min_side_px", fallback, policy)


def exemplar_render_fallback_gsd_m(fallback: float, policy: dict | None = None) -> float:



    val = exemplar_policy(policy).get("render_fallback_gsd_m")
    if _is_finite_policy_value(val) and val > 0:
        return float(val)
    return float(fallback)
