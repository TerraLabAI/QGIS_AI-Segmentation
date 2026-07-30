

















from __future__ import annotations

from .detection_policy import review_policy
from .prompt_taxonomy import keyword_matches, normalize_prompt
from .review_defaults import (
    AUTO_REVIEW_CLEAN_DEFAULT,
    AUTO_REVIEW_CLOSE_NOTCHES_M_DEFAULT,
    AUTO_REVIEW_EXPAND_DEFAULT,
    AUTO_REVIEW_FILL_HOLES_DEFAULT,
    AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT,
    AUTO_REVIEW_ORTHO_DEFAULT,
    AUTO_REVIEW_SIMPLIFY_DEFAULT,
    AUTO_REVIEW_SMOOTH_DEFAULT,
    fill_holes_max_m2_with_floor,
    min_size_noise_floor_m2,
)
from .shape_policy_dials import (
    auto_review_clean_default,
    auto_review_close_notches_default,
    auto_review_expand_default,
    auto_review_simplify_default,
)



_DEFAULT_SETTINGS: dict = {
    "ortho": AUTO_REVIEW_ORTHO_DEFAULT,
    "fill_holes": AUTO_REVIEW_FILL_HOLES_DEFAULT,
    "fill_holes_max_m2": AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT,
    "smooth": AUTO_REVIEW_SMOOTH_DEFAULT,
    "simplify_px": AUTO_REVIEW_SIMPLIFY_DEFAULT,
    "clean_px": AUTO_REVIEW_CLEAN_DEFAULT,
}


def _default_settings() -> dict:





    return {
        **_DEFAULT_SETTINGS,
        "simplify_px": auto_review_simplify_default(AUTO_REVIEW_SIMPLIFY_DEFAULT),
        "clean_px": auto_review_clean_default(AUTO_REVIEW_CLEAN_DEFAULT),
    }





_normalize = normalize_prompt
_matches = keyword_matches


def _class_settings_for(cls: str, policy: dict | None) -> dict:












    served = (review_policy(policy).get("class_settings") or {}).get(cls)
    if cls == "default":
        return (
            {**_default_settings(), **served}
            if isinstance(served, dict) and served
            else _default_settings()
        )
    return served if isinstance(served, dict) else _default_settings()


def min_size_m2_for(
    prompt: str, mask_gsd_m: float, policy: dict | None = None
) -> float:






    text = _normalize(prompt)
    object_floor = 0.0
    if text:
        floors = review_policy(policy).get("min_size_m2")
        if isinstance(floors, dict):
            for kw in sorted(floors, key=len, reverse=True):
                if _matches(text, kw):
                    try:
                        object_floor = float(floors[kw])
                    except (TypeError, ValueError):
                        object_floor = 0.0
                    break
    noise_floor = min_size_noise_floor_m2(mask_gsd_m, no_prompt=not text)
    return round(max(object_floor, noise_floor), 1)


def review_preset_for(
    prompt: str, mask_gsd_m: float, policy: dict | None = None
) -> dict:










    cls = "default"
    settings = _class_settings_for(cls, policy)
    return {
        "simplify_px": float(settings.get(
            "simplify_px", auto_review_simplify_default(AUTO_REVIEW_SIMPLIFY_DEFAULT))),
        "smooth": bool(settings.get("smooth", AUTO_REVIEW_SMOOTH_DEFAULT)),
        "expand_px": int(settings.get(
            "expand_px", auto_review_expand_default(AUTO_REVIEW_EXPAND_DEFAULT))),


        "fill_holes": AUTO_REVIEW_FILL_HOLES_DEFAULT,
        "fill_holes_max_m2": _fill_holes_max_m2(settings),
        "clean_px": float(settings.get(
            "clean_px", auto_review_clean_default(AUTO_REVIEW_CLEAN_DEFAULT))),
        "close_notches_m": _close_notches_m(settings),
        "ortho": _ortho_default_for(prompt, cls, settings, policy),
        "min_size_m2": min_size_m2_for(prompt, mask_gsd_m, policy),
        "vertex_spacing_m": _vertex_spacing_for(settings, policy),
        "shape_class": cls,
    }


def neutral_review_preset(prompt: str, mask_gsd_m: float) -> dict:



    return review_preset_for(prompt, mask_gsd_m, policy={})


def _vertex_spacing_for(settings: dict, policy: dict | None) -> float:








    from .detection_policy import vertex_budget_settings

    val = settings.get("vertex_spacing_m")
    if isinstance(val, (int, float)) and not isinstance(val, bool) and val >= 0:
        return float(val)
    return float(vertex_budget_settings(policy)["spacing_m"])


def _close_notches_m(settings: dict) -> float:










    val = settings.get("close_notches_m")
    if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
        return float(val)
    return auto_review_close_notches_default(AUTO_REVIEW_CLOSE_NOTCHES_M_DEFAULT)


def _fill_holes_max_m2(settings: dict) -> float:









    return fill_holes_max_m2_with_floor(
        settings.get("fill_holes"), settings.get("fill_holes_max_m2"))


def _class_pins_ortho(cls: str, settings: dict, policy: dict | None) -> bool:







    if cls != "default":
        return settings is not _DEFAULT_SETTINGS and "ortho" in settings
    served = review_policy(policy).get("class_settings")
    raw = served.get("default") if isinstance(served, dict) else None
    return isinstance(raw, dict) and "ortho" in raw


def _ortho_default_for(prompt: str, cls: str, settings: dict, policy) -> bool:









    if _class_pins_ortho(cls, settings, policy):
        return bool(settings["ortho"])
    from .detection_policy import regularize_enabled_for
    if regularize_enabled_for(prompt, policy):
        return True
    return AUTO_REVIEW_ORTHO_DEFAULT
