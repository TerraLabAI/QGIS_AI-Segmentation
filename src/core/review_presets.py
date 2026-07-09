

















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


def shape_class_for(prompt: str, policy: dict | None = None) -> str:









    text = _normalize(prompt)
    if not text:
        return "default"
    review = review_policy(policy)
    class_keywords = review.get("class_keywords")
    if not isinstance(class_keywords, dict):
        return "default"
    candidates = [
        (kw, cls)
        for cls, kws in class_keywords.items()
        for kw in (kws or [])
        if isinstance(kw, str)
    ]
    candidates.sort(key=lambda item: len(item[0]), reverse=True)
    for kw, cls in candidates:
        if _matches(text, kw):
            return cls
    category_to_class = review.get("category_to_class")
    if isinstance(category_to_class, dict):
        try:
            for cat in live_catalog_categories():
                for preset in cat.get("presets") or []:
                    if str(preset.get("prompt", "")).lower() == text:
                        return category_to_class.get(cat.get("key"), "default")
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    return "default"





_live_catalog_memo: dict[str, tuple[str, list[dict]]] = {}


def live_catalog_categories() -> list[dict]:



    from .presets.segmentation_presets import catalog_revision, fallback_categories

    stamp = catalog_revision()
    if not stamp:
        return fallback_categories()
    held = _live_catalog_memo.get("held")
    if held is not None and held[0] == stamp:
        return held[1]
    try:
        from .presets.segmentation_presets_client import cached_or_offline_catalog

        cats, _tops = cached_or_offline_catalog()
    except Exception:  # noqa: BLE001
        cats = fallback_categories()
    _live_catalog_memo["held"] = (stamp, cats)
    return cats


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








    cls = shape_class_for(prompt, policy)
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


def review_start_confidence_default(
    prompt: str, is_exemplar_only: bool, policy: dict | None = None
) -> float:










    if is_exemplar_only:
        from .detection_policy import confidence_default_exemplar_only

        return confidence_default_exemplar_only(policy)
    cls_conf = class_confidence_for(prompt, policy)
    if cls_conf is not None:
        return cls_conf
    from .detection_policy import confidence_default

    return confidence_default(policy)


def class_confidence_for(prompt: str, policy: dict | None = None) -> float | None:





    cls = shape_class_for(prompt, policy)
    if cls == "default":
        return None
    val = _class_settings_for(cls, policy).get("confidence")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        conf = float(val)
        if 0.0 <= conf <= 1.0:
            return conf
    return None
