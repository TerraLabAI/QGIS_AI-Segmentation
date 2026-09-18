






from __future__ import annotations

from .detection_policy_core import (
    _is_finite_policy_value,
    gate_policy,
)
from .prompt_taxonomy import first_entry_match, normalize_prompt


def gate_enabled(policy: dict | None = None) -> bool:



    return gate_policy(policy).get("enabled") is True


def gate_group(policy: dict | None = None) -> int:




    val = gate_policy(policy).get("group")
    if _is_finite_policy_value(val) and 2 <= val <= 4:
        return int(val)
    return 2


def gate_max_group(policy: dict | None = None) -> int:






    val = gate_policy(policy).get("max_group")
    if _is_finite_policy_value(val) and 2 <= val <= 4:
        return int(val)
    return 2


def gate_min_pixels(policy: dict | None = None) -> int:


    val = gate_policy(policy).get("min_pixels")
    if _is_finite_policy_value(val) and val >= 1:
        return int(val)
    return 8


def gate_min_tiles(policy: dict | None = None) -> int:


    val = gate_policy(policy).get("min_tiles")
    if _is_finite_policy_value(val) and val >= 4:
        return int(val)
    return 12


def gate_prefilter_policy(policy: dict | None = None) -> dict:





    val = gate_policy(policy).get("prefilter")
    return val if isinstance(val, dict) else {}


def gate_prefilter_enabled(policy: dict | None = None) -> bool:




    return gate_prefilter_policy(policy).get("enabled") is not False


def gate_prefilter_nodata_frac(policy: dict | None = None) -> float:








    val = gate_prefilter_policy(policy).get("nodata_frac")
    if _is_finite_policy_value(val) and 0.0 < val <= 1.0:
        return float(val)
    return 1.0


def gate_prefilter_nodata_rgb_eps(fallback: int, policy: dict | None = None) -> int:







    val = gate_prefilter_policy(policy).get("nodata_rgb_eps")
    if _is_finite_policy_value(val) and -1 <= val <= 24:
        return int(val)
    return int(fallback)


def gate_prefilter_min_valid_px(fallback: float, policy: dict | None = None) -> float:




    val = gate_prefilter_policy(policy).get("min_valid_px")
    if _is_finite_policy_value(val) and 0.0 <= val <= 4096.0:
        return float(val)
    return float(fallback)


def gate_prefilter_band_eps(policy: dict | None = None) -> float:




    val = gate_prefilter_policy(policy).get("band_eps")
    if _is_finite_policy_value(val) and 0.0 <= val <= 16.0:
        return float(val)
    return 2.0


def gate_prefilter_config(policy: dict | None = None) -> dict | None:



    if not gate_prefilter_enabled(policy):
        return None
    from .cloud_detection import (  # noqa: PLC0415
        _PREFILTER_MIN_VALID_PX,
        _PREFILTER_NODATA_RGB_EPS,
    )
    return {
        "nodata_frac": gate_prefilter_nodata_frac(policy),
        "band_eps": gate_prefilter_band_eps(policy),
        "nodata_rgb_eps": gate_prefilter_nodata_rgb_eps(
            _PREFILTER_NODATA_RGB_EPS, policy),
        "min_valid_px": gate_prefilter_min_valid_px(
            _PREFILTER_MIN_VALID_PX, policy),
    }


def gate_blank_policy(policy: dict | None = None) -> dict:




    val = gate_policy(policy).get("blank")
    return val if isinstance(val, dict) else {}


def blank_dominant_frac(fallback: float, policy: dict | None = None) -> float:




    val = gate_blank_policy(policy).get("dominant_frac")
    if _is_finite_policy_value(val) and 0.0 < val <= 1.0:
        return float(val)
    return fallback


def blank_quant(fallback: int, policy: dict | None = None) -> int:



    val = gate_blank_policy(policy).get("quant")
    if _is_finite_policy_value(val) and val > 0:
        return int(val)
    return fallback


def blank_sample_px(fallback: int, policy: dict | None = None) -> int:


    val = gate_blank_policy(policy).get("sample_px")
    if _is_finite_policy_value(val) and val > 0:
        return int(val)
    return fallback


def gate_unavailable_policy(policy: dict | None = None) -> dict:






    val = gate_policy(policy).get("unavailable")
    return val if isinstance(val, dict) else {}


def unavailable_neutral_eps(fallback: int, policy: dict | None = None) -> int:


    val = gate_unavailable_policy(policy).get("neutral_eps")
    if _is_finite_policy_value(val) and val >= 0:
        return int(val)
    return fallback


def unavailable_neutral_frac(fallback: float, policy: dict | None = None) -> float:



    val = gate_unavailable_policy(policy).get("neutral_frac")
    if _is_finite_policy_value(val) and 0.0 < val <= 1.0:
        return float(val)
    return fallback


def unavailable_dominant_frac(fallback: float, policy: dict | None = None) -> float:




    val = gate_unavailable_policy(policy).get("dominant_frac")
    if _is_finite_policy_value(val) and 0.0 < val <= 1.0:
        return float(val)
    return fallback


def unavailable_agreement_min(fallback: float, policy: dict | None = None) -> float:







    val = gate_unavailable_policy(policy).get("agreement_min")
    if _is_finite_policy_value(val) and -1.0 <= val <= 1.0:
        return float(val)
    return fallback


def unavailable_agreement_sample_px(fallback: int, policy: dict | None = None) -> int:


    val = gate_unavailable_policy(policy).get("agreement_sample_px")
    if _is_finite_policy_value(val) and 8 <= val <= 256:
        return int(val)
    return fallback


def unavailable_backoff_steps(fallback: int, policy: dict | None = None) -> int:



    val = gate_unavailable_policy(policy).get("backoff_steps")
    if _is_finite_policy_value(val) and 0 <= val <= 8:
        return int(val)
    return fallback


def gate_class_for_prompt(
    prompt: str, fallback_class: str, policy: dict | None = None
) -> str:












    class_map = gate_policy(policy).get("class_map")
    if isinstance(class_map, list):
        text = normalize_prompt(prompt)
        usable = [
            entry
            for entry in class_map
            if isinstance(entry, dict) and isinstance(entry.get("gate_class"), str) and entry.get("gate_class")
        ]
        entry = first_entry_match(text, usable)
        if entry is not None:
            return str(entry["gate_class"])
    return fallback_class


def gate_class_rule(gate_class: str, policy: dict | None = None) -> dict | None:











    classes = gate_policy(policy).get("classes")
    if not isinstance(classes, dict):
        return None
    raw = classes.get(gate_class)
    if not isinstance(raw, dict):
        return None
    min_score = raw.get("min_score")
    if not _is_finite_policy_value(min_score):
        return None
    if not 0.0 < float(min_score) <= 1.0:
        return None
    rule: dict = {"min_score": float(min_score)}
    cap = raw.get("max_scan_mupp")
    if _is_finite_policy_value(cap) and cap > 0:
        rule["max_scan_mupp"] = float(cap)
    max_group = raw.get("max_group")
    if _is_finite_policy_value(max_group) \
            and 2 <= max_group <= 4:
        rule["max_group"] = int(max_group)
    return rule
