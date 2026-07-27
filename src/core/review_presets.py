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

from .detection_policy import review_policy
from .prompt_taxonomy import keyword_matches, normalize_prompt
from .review_defaults import (
    AUTO_REVIEW_CLEAN_DEFAULT,
    AUTO_REVIEW_EXPAND_DEFAULT,
    AUTO_REVIEW_FILL_HOLES_DEFAULT,
    AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT,
    AUTO_REVIEW_ORTHO_DEFAULT,
    AUTO_REVIEW_SIMPLIFY_DEFAULT,
    AUTO_REVIEW_SMOOTH_DEFAULT,
    fill_holes_max_m2_with_floor,
    min_size_noise_floor_m2,
)

# Neutral fallback shape class: the faithful review_defaults values. Used for
# an unknown prompt and whenever no server policy is available.
_DEFAULT_SETTINGS: dict = {
    "ortho": AUTO_REVIEW_ORTHO_DEFAULT,
    "fill_holes": AUTO_REVIEW_FILL_HOLES_DEFAULT,
    "fill_holes_max_m2": AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT,
    "smooth": AUTO_REVIEW_SMOOTH_DEFAULT,
    "simplify_px": AUTO_REVIEW_SIMPLIFY_DEFAULT,
    "clean_px": AUTO_REVIEW_CLEAN_DEFAULT,
}


# Normalization and keyword matching are shared with the other policy tables
# (seed tiers, mask-grid routing, scan gate) so the same prompt cannot classify
# differently across them; see core/prompt_taxonomy.
_normalize = normalize_prompt
_matches = keyword_matches


def _class_settings_for(cls: str, policy: dict | None) -> dict:
    """The review regularizers for a shape class: the server policy's entry, or
    the neutral defaults for the fallback class / a missing or partial policy.

    The "default" class is where every prompt with no keyword hit lands, and it
    used to be the one class the server could not reach, so retuning what an
    unrecognised prompt opens with meant a plugin release. A served "default"
    entry now applies, merged OVER the client defaults so a partial entry only
    moves the keys it names. Merging is done for that class alone: a real class
    entry is a complete server decision, and filling its gaps from the neutral
    class would tell _ortho_default_for that the server pinned keys it never
    mentioned.
    """
    served = (review_policy(policy).get("class_settings") or {}).get(cls)
    if cls == "default":
        return (
            {**_DEFAULT_SETTINGS, **served}
            if isinstance(served, dict) and served
            else _DEFAULT_SETTINGS
        )
    return served if isinstance(served, dict) else _DEFAULT_SETTINGS


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
    the run's resolution noise floor (a blob a few mask pixels across is noise
    by construction, whatever the prompt). Longest keyword wins. A run with no
    text has no per-object floor at all, so it starts at the wider no-prompt
    noise floor instead. 0.0 = off (no text at unknown resolution, or no
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
    noise_floor = min_size_noise_floor_m2(mask_gsd_m, no_prompt=not text)
    return round(max(object_floor, noise_floor), 1)


def review_preset_for(
    prompt: str, mask_gsd_m: float, policy: dict | None = None
) -> dict:
    """Smart review defaults for one run: prompt-shaped regularizers plus the
    resolution-aware Min size. Keys mirror the dock's review controls:

        simplify_px (float), smooth (bool), expand_px (int),
        fill_holes (bool), fill_holes_max_m2 (float, 0 = fill every hole),
        clean_px (float), ortho (bool),
        min_size_m2 (float, 0 = off), shape_class (str, for logging).
    """
    cls = shape_class_for(prompt, policy)
    settings = _class_settings_for(cls, policy)
    return {
        "simplify_px": float(settings.get("simplify_px", AUTO_REVIEW_SIMPLIFY_DEFAULT)),
        "smooth": bool(settings.get("smooth", AUTO_REVIEW_SMOOTH_DEFAULT)),
        "expand_px": int(settings.get("expand_px", AUTO_REVIEW_EXPAND_DEFAULT)),
        # Not settings.get(): a class does not get to turn hole filling off, it
        # only sets how far above the floor it goes (see _fill_holes_max_m2).
        "fill_holes": AUTO_REVIEW_FILL_HOLES_DEFAULT,
        "fill_holes_max_m2": _fill_holes_max_m2(settings),
        "clean_px": float(settings.get("clean_px", AUTO_REVIEW_CLEAN_DEFAULT)),
        "ortho": _ortho_default_for(prompt, cls, settings, policy),
        "min_size_m2": min_size_m2_for(prompt, mask_gsd_m, policy),
        "vertex_spacing_m": _vertex_spacing_for(settings, policy),
        "shape_class": cls,
    }


def _vertex_spacing_for(settings: dict, policy: dict | None) -> float:
    """One exported point per this much outline, in GROUND METRES.

    How closely a class wants its points depends on the class: a tree crown
    reads well with eight, a field boundary needs one every few metres to keep
    its bends. The per-class value is a server dial (``vertex_spacing_m`` in the
    review policy's class settings); a missing or unusable entry falls back to
    the ONE generic client value, never to a per-class copy.
    """
    from .detection_policy import vertex_budget_settings

    val = settings.get("vertex_spacing_m")
    if isinstance(val, (int, float)) and not isinstance(val, bool) and val >= 0:
        return float(val)
    return float(vertex_budget_settings(policy)["spacing_m"])


def _fill_holes_max_m2(settings: dict) -> float:
    """Fill holes SMALLER than this ground area (m2); 0 = fill every hole.

    The value is a per-class server dial (``fill_holes_max_m2`` in the review
    policy's class settings): a road wants its parked cars filled and its median
    kept, a building wants every mask pinhole gone. A missing or unusable entry
    falls back to the one generic client default, never to a per-class copy.
    A class that fills nothing, or names a ceiling under the client floor, still
    gets the floor: that is where "fill holes on every run" is enforced.
    """
    return fill_holes_max_m2_with_floor(
        settings.get("fill_holes"), settings.get("fill_holes_max_m2"))


def _class_pins_ortho(cls: str, settings: dict, policy: dict | None) -> bool:
    """Whether the SERVER named ``ortho`` for this shape class.

    The neutral class is asked against the RAW served entry, never against the
    merged settings: _class_settings_for lays a served "default" over the
    client defaults, so "ortho" is present in the merged dict either way and
    only the raw entry can say who put it there.
    """
    if cls != "default":
        return settings is not _DEFAULT_SETTINGS and "ortho" in settings
    served = review_policy(policy).get("class_settings")
    raw = served.get("default") if isinstance(served, dict) else None
    return isinstance(raw, dict) and "ortho" in raw


def _ortho_default_for(prompt: str, cls: str, settings: dict, policy) -> bool:
    """Default state of the "Right angles" toggle for a run.

    A class the footprint regularizer covers (buildings, solar) opens with
    Right angles ON, so those runs are squared without the user toggling it.
    This is the ONLY thing the keywords decide: whatever they say, the toggle
    itself always runs the regularizer, so a run with no prompt squares just as
    well once ticked. A server class entry that pins ``ortho`` still wins; the
    client's own neutral defaults do not count as a server decision, so the
    regularize default applies through them."""
    if _class_pins_ortho(cls, settings, policy):
        return bool(settings["ortho"])
    from .detection_policy import regularize_enabled_for
    if regularize_enabled_for(prompt, policy):
        return True
    return AUTO_REVIEW_ORTHO_DEFAULT


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
