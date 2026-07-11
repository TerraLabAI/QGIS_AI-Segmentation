"""Prompt-aware smart defaults for the Automatic post-run review controls.

The review's shape refine (Right angles, Fill holes, Round corners, Clean
edges) and the Min size filter each have one best starting value per kind of
object. The per-object tables (shape classes, keyword maps, size floors) are
provided by the plugin's server configuration; without it every prompt degrades
to one neutral shape class and Min size falls back to the resolution noise
floor only. The matching and fallback MECHANISMS live here; the tuned values do
not.

Min size is a ground-area floor (m2) combining a per-object minimum with the
run's resolution noise floor, (3 x mask ground pixel)^2, below which a
detection is mask noise by construction.

Pure data + functions, no Qt: callable from the plugin controller, the dock and
the headless MCP path alike. The per-run entry point is
``review_preset_for(prompt, mask_gsd_m)``.
"""
from __future__ import annotations

import re

from .detection_policy import review_policy
from .review_defaults import (
    AUTO_REVIEW_CLEAN_DEFAULT,
    AUTO_REVIEW_EXPAND_DEFAULT,
    AUTO_REVIEW_FILL_HOLES_DEFAULT,
    AUTO_REVIEW_ORTHO_DEFAULT,
    AUTO_REVIEW_SIMPLIFY_DEFAULT,
    AUTO_REVIEW_SMOOTH_DEFAULT,
)

# Neutral fallback shape class: the faithful review_defaults values. Used for
# an unknown prompt and whenever no server policy is available.
_DEFAULT_SETTINGS: dict = {
    "ortho": AUTO_REVIEW_ORTHO_DEFAULT,
    "fill_holes": AUTO_REVIEW_FILL_HOLES_DEFAULT,
    "smooth": AUTO_REVIEW_SMOOTH_DEFAULT,
    "simplify_px": AUTO_REVIEW_SIMPLIFY_DEFAULT,
    "clean_px": AUTO_REVIEW_CLEAN_DEFAULT,
}


def _normalize(prompt: str) -> str:
    text = (prompt or "").strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", text)


def _matches(text: str, keyword: str) -> bool:
    """Whole-word keyword match tolerating a trailing plural (s/es), so
    "buildings" hits "building" while "street" never hits "tree"."""
    pattern = r"\b" + re.escape(keyword) + r"(?:e?s)?\b"
    return re.search(pattern, text) is not None


def _class_settings_for(cls: str, policy: dict | None) -> dict:
    """The review regularizers for a shape class: the server policy's entry, or
    the neutral defaults for the fallback class / a missing or partial policy."""
    if cls == "default":
        return _DEFAULT_SETTINGS
    settings = (review_policy(policy).get("class_settings") or {}).get(cls)
    return settings if isinstance(settings, dict) else _DEFAULT_SETTINGS


def shape_class_for(prompt: str, policy: dict | None = None) -> str:
    """The shape class of an English prompt token, from the server policy's
    keyword and catalogue-category tables. "default" when there is no policy or
    no hit.

    Keywords are matched longest-first across ALL classes so a phrase beats a
    component word of another class and ordering between classes never decides a
    tie. A prompt with no keyword hit falls back to the library catalogue's
    category.
    """
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
            from .presets.segmentation_presets import fallback_categories

            for cat in fallback_categories():
                for preset in cat["presets"]:
                    if preset["prompt"].lower() == text:
                        return category_to_class.get(cat["key"], "default")
        except Exception:  # noqa: BLE001 -- catalogue lookup is best-effort  # nosec B110
            pass
    return "default"


def min_size_m2_for(
    prompt: str, mask_gsd_m: float, policy: dict | None = None
) -> float:
    """Min size (m2): the per-object floor from the server policy, never below
    the run's resolution noise floor (3 x mask ground pixel)^2 - a blob under 3
    mask pixels across is noise by construction, whatever the prompt. Longest
    keyword wins. 0.0 = off (unknown prompt at unknown resolution, or no
    policy)."""
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
    noise_floor = (3.0 * mask_gsd_m) ** 2 if mask_gsd_m and mask_gsd_m > 0 else 0.0
    return round(max(object_floor, noise_floor), 1)


def review_preset_for(
    prompt: str, mask_gsd_m: float, policy: dict | None = None
) -> dict:
    """Smart review defaults for one run: prompt-shaped regularizers plus the
    resolution-aware Min size. Keys mirror the dock's review controls:

        simplify_px (float), smooth (bool), expand_px (int),
        fill_holes (bool), clean_px (float), ortho (bool),
        min_size_m2 (float, 0 = off), shape_class (str, for logging).
    """
    cls = shape_class_for(prompt, policy)
    settings = _class_settings_for(cls, policy)
    return {
        "simplify_px": float(settings.get("simplify_px", AUTO_REVIEW_SIMPLIFY_DEFAULT)),
        "smooth": bool(settings.get("smooth", AUTO_REVIEW_SMOOTH_DEFAULT)),
        "expand_px": AUTO_REVIEW_EXPAND_DEFAULT,
        "fill_holes": bool(settings.get("fill_holes", AUTO_REVIEW_FILL_HOLES_DEFAULT)),
        "clean_px": float(settings.get("clean_px", AUTO_REVIEW_CLEAN_DEFAULT)),
        "ortho": bool(settings.get("ortho", AUTO_REVIEW_ORTHO_DEFAULT)),
        "min_size_m2": min_size_m2_for(prompt, mask_gsd_m, policy),
        "shape_class": cls,
    }


def review_start_confidence_default(
    prompt: str, is_exemplar_only: bool, policy: dict | None = None
) -> float:
    """The starting confidence cutoff for a run, used for BOTH the live preview
    and the post-run review so they open at the same value.

    An EXEMPLAR-only run (a drawn example, no text prompt) has no
    open-vocabulary text prior and surfaces more weak look-alikes, so it opens
    at the exemplar-only default (``prompt`` is ignored). A text run uses its
    shape-class value when the server policy carries one, else the generic text
    default. This is the single decision point so the live seed and the review
    seed can never drift apart.
    """
    if is_exemplar_only:
        from .detection_policy import confidence_default_exemplar_only

        return confidence_default_exemplar_only(policy)
    cls_conf = class_confidence_for(prompt, policy)
    if cls_conf is not None:
        return cls_conf
    from .detection_policy import confidence_default

    return confidence_default(policy)


def class_confidence_for(prompt: str, policy: dict | None = None) -> float | None:
    """The review's STARTING confidence for this prompt's shape class, or None
    when the policy carries none (the caller then uses the generic default).
    Object classes score differently on the same true object, so one flat
    cutoff over- or under-hides depending on the prompt; this reads the
    per-class value the server policy delivers."""
    cls = shape_class_for(prompt, policy)
    if cls == "default":
        return None
    val = _class_settings_for(cls, policy).get("confidence")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        conf = float(val)
        if 0.0 <= conf <= 1.0:
            return conf
    return None
