"""Server-delivered detection policy for Automatic mode.

The per-object detail targets (which ground resolution to seed per object) and
the review shape defaults and size floors are provided by the plugin's server
configuration and cached in memory. Without that configuration the plugin uses
one neutral default for each value, so Automatic mode still runs (quality only
depends on it). Manual mode never touches this module.

This file holds the matching and fallback MECHANISMS only; the tuned tables
live in the server configuration. Pure Python with no Qt at import time, so it
is safe to import from the controller, the dock and the headless path alike.
"""
from __future__ import annotations

import re

from .tile_manager import (
    AUTO_OBJECT_MIN_PX,
    AUTO_SEED_TILE_CAP,
    DEFAULT_AUTO_TILE_BUDGET,
    DEFAULT_SEED_MUPP_M,
    NATIVE_OVERSAMPLE_MAX,
    QUALITY_FLOOR_MUPP_M,
    SWEET_SPOT_MAX_MUPP_M,
)


def get_detection_policy() -> dict:
    """The cached detection policy dict, or {} when none is available.

    Reads the server configuration cache only (never networks), so it is safe
    on the GUI thread and fails open to an empty dict.
    """
    try:
        from .activation_manager import get_server_config

        config = get_server_config()
    except Exception:  # noqa: BLE001 -- config is best-effort  # nosec B110
        return {}
    if not isinstance(config, dict):
        return {}
    policy = config.get("detection_policy")
    return policy if isinstance(policy, dict) else {}


def seed_policy(policy: dict | None = None) -> dict:
    """The seed sub-policy (per-object detail targets and grid tuning)."""
    policy = get_detection_policy() if policy is None else policy
    seed = policy.get("seed") if isinstance(policy, dict) else None
    return seed if isinstance(seed, dict) else {}


def review_policy(policy: dict | None = None) -> dict:
    """The review sub-policy (shape classes, keyword maps, size floors)."""
    policy = get_detection_policy() if policy is None else policy
    review = policy.get("review") if isinstance(policy, dict) else None
    return review if isinstance(review, dict) else {}


def review_noise_floor(policy: dict | None = None) -> float:
    """Confidence fraction below which a detection is excluded from the review
    entirely: it never counts in the total, never renders, and the review
    confidence controls cannot dial below it. Read from the server
    ``review.noise_floor``; the fallback is ONE generic client value (0.05),
    never a mirror of a tuned table. Clamped to [0, 1)."""
    val = review_policy(policy).get("noise_floor")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = float(val)
        if 0.0 <= f < 1.0:
            return f
    return 0.05


def merge_policy(policy: dict | None = None) -> dict:
    """The review.merge sub-policy (merge/dedup scalars + token/category lists).

    Empty dict when absent, so every consumer falls open to its own generic
    default (the counting-safe merge behaviour)."""
    review = review_policy(policy)
    merge = review.get("merge") if isinstance(review, dict) else None
    return merge if isinstance(merge, dict) else {}


def exemplar_only_merge_separate(policy: dict | None = None) -> bool:
    """Merge policy for an EXEMPLAR-only run (empty prompt token): the one case
    with no token signal at all for SEPARATE (count distinct objects) vs MAP
    (continuous cover union). Reads the server policy's ``exemplar_only`` key
    ("map" or "separate"); the fallback is the counting-safe default (True),
    one generic value, never a mirror of the tuned per-object table."""
    val = merge_policy(policy).get("exemplar_only")
    if isinstance(val, str) and val.strip().lower() == "map":
        return False
    return True


def map_likeness_min_share(policy: dict | None = None) -> float:
    """Minimum map-likeness for an EXEMPLAR-only run to be grouped as continuous
    cover (MAP) rather than counted as distinct objects.

    Map-likeness is the area-weighted mean tile coverage of the run's fragments
    (near zero for small countable objects, high for continuous cover). Read
    from the server ``review.merge`` policy (``map_likeness_min_share``); the
    fallback is ONE generic client value (0.15), never a mirror of the tuned
    server tables."""
    val = merge_policy(policy).get("map_likeness_min_share")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return 0.15


def max_concurrent(policy: dict | None = None) -> int:
    """Cap on concurrent in-flight tiles per run (fallback 6)."""
    val = seed_policy(policy).get("max_concurrent")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return int(val)
    return 6


# Generic client fallbacks for the review.merge scalars. One value per key, the
# single fallback source when the server policy omits or malforms a scalar.
_MERGE_SCALAR_DEFAULTS: dict[str, float] = {
    "merge_ios": 0.15,
    "dedup_ios": 0.5,
    "dup_ios_floor": 0.3,
    "dup_centroid_frac": 0.35,
    "seam_span_ios": 0.03,
    "ios_threshold": 0.5,
}


def merge_scalar(key: str, fallback: float | None = None, policy: dict | None = None) -> float:
    """One numeric review.merge scalar, or the client fallback.

    ``fallback`` defaults to the generic constant for ``key`` when omitted, so
    the fallback values live in one place."""
    if fallback is None:
        fallback = _MERGE_SCALAR_DEFAULTS.get(key, 0.0)
    val = merge_policy(policy).get(key)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return fallback


def merge_scalars(policy: dict | None = None) -> dict[str, float]:
    """All six review.merge scalars resolved (policy value or generic fallback)."""
    return {k: merge_scalar(k, d, policy) for k, d in _MERGE_SCALAR_DEFAULTS.items()}


def object_profile(prompt: str, policy: dict | None = None) -> tuple[float, float]:
    """(typical ground size m, target ground resolution m/px) for a prompt.

    The seed policy carries a tier list (finest first, targets already
    resolution-adjusted server-side); the first tier with a keyword that is a
    word-boundary prefix of the prompt wins. Without a policy the generic
    default applies; with a policy present but no tier match, the policy's own
    default object applies.
    """
    generic = (10.0, DEFAULT_SEED_MUPP_M)
    seed = seed_policy(policy)
    tiers = seed.get("object_tiers")
    if not isinstance(tiers, list):
        return generic
    text = (prompt or "").lower()
    for tier in tiers:
        if not isinstance(tier, dict):
            continue
        for kw in tier.get("keywords") or []:
            if isinstance(kw, str) and re.search(r"\b" + re.escape(kw), text):
                return _profile_pair(tier, generic)
    return _profile_pair(seed.get("default_object"), generic)


def _profile_pair(entry: object, fallback: tuple[float, float]) -> tuple[float, float]:
    """Read (size_m, target_mupp) from a policy entry, else the fallback."""
    if isinstance(entry, dict):
        try:
            return float(entry["size_m"]), float(entry["target_mupp"])
        except (KeyError, TypeError, ValueError):
            return fallback
    return fallback


def _seed_float(key: str, fallback: float, policy: dict | None) -> float:
    """A numeric seed-policy scalar, or the client fallback constant."""
    val = seed_policy(policy).get(key)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return fallback


def zone_seed_mupp(policy: dict | None = None) -> float:
    """Prompt-less default seed resolution (m/px)."""
    return _seed_float("zone_seed_mupp", DEFAULT_SEED_MUPP_M, policy)


def soft_tile_budget(policy: dict | None = None) -> int:
    """Soft tile (credit) preference the auto seed tries to stay within."""
    return int(_seed_float("soft_tile_budget", DEFAULT_AUTO_TILE_BUDGET, policy))


def seed_tile_cap(policy: dict | None = None) -> int:
    """Hard ceiling on tiles the auto-picked default may propose."""
    return int(_seed_float("seed_tile_cap", AUTO_SEED_TILE_CAP, policy))


def free_run_fraction(policy: dict | None = None) -> float:
    """Max share of the monthly free allowance one run may cost (0-1].

    Free-tier runs are capped so a single Detect can never drain the whole
    trial; subscribers are never capped. Fallback 0.25 (one quarter of the
    allowance, so the trial always covers several full runs)."""
    val = _seed_float("free_run_fraction", 0.25, policy)
    return val if 0.0 < val <= 1.0 else 0.25


def object_min_px(policy: dict | None = None) -> int:
    """Minimum pixels across for an object to count as resolvable."""
    return int(_seed_float("object_min_px", AUTO_OBJECT_MIN_PX, policy))


def sweet_spot_max_mupp(policy: dict | None = None) -> float:
    """Coarse edge (m/px) of the adequate-quality band."""
    return _seed_float("sweet_spot_max_mupp", SWEET_SPOT_MAX_MUPP_M, policy)


def quality_floor_mupp(policy: dict | None = None) -> float:
    """Resolution (m/px) below which the UI warns detail is too coarse."""
    return _seed_float("quality_floor_mupp", QUALITY_FLOOR_MUPP_M, policy)


def native_oversample_max(policy: dict | None = None) -> float:
    """How far past a source's native resolution a render may go (linear)."""
    return _seed_float("native_oversample_max", NATIVE_OVERSAMPLE_MAX, policy)


def detail_over_ratio(policy: dict | None = None) -> float:
    """Fraction of the object's target resolution below which the detail
    slider guidance flags the level as past diminishing returns (finer
    resolution than the object needs mostly adds cost and can fragment
    large objects across tiles). Subscriber ratio; free tier uses
    detail_over_ratio_free."""
    return _seed_float("detail_over_ratio", 0.4, policy)


def detail_over_ratio_free(policy: dict | None = None) -> float:
    """Free-tier variant of detail_over_ratio: warns earlier on the Fine
    end, because a free run spends scarce monthly free credits."""
    return _seed_float("detail_over_ratio_free", 0.5, policy)


def recall_floor(fallback: float, policy: dict | None = None) -> float:
    """Recall floor sent for a TEXT run so every plausible mask comes back.

    The fallback is the client's own generic constant (kept in the UI layer,
    passed in here so this core module stays free of any UI import)."""
    return _seed_float("recall_floor", fallback, policy)


def recall_floor_exemplar_only(fallback: float, policy: dict | None = None) -> float:
    """Recall floor sent for an EXEMPLAR-only run (no text prior). Fallback is
    the client's own generic constant, passed in by the caller."""
    return _seed_float("recall_floor_exemplar_only", fallback, policy)


def confidence_default(policy: dict | None = None) -> float:
    """The post-run review's starting confidence cutoff."""
    from .review_defaults import AUTO_DEFAULT_CONFIDENCE

    return _seed_float("confidence_default", AUTO_DEFAULT_CONFIDENCE, policy)


def confidence_default_exemplar_only(policy: dict | None = None) -> float:
    """The starting confidence cutoff for an EXEMPLAR-only run (a drawn example,
    no text prompt), used for BOTH the live preview and the post-run review so
    they open at the same value. Higher than the text default because, without
    an open-vocabulary text prior, the model surfaces more weak look-alikes.
    Read from the server ``seed.confidence_default_exemplar_only``; the fallback
    is ONE generic client value, never a mirror of a tuned table."""
    from .review_defaults import AUTO_DEFAULT_CONFIDENCE_EXEMPLAR_ONLY

    return _seed_float(
        "confidence_default_exemplar_only",
        AUTO_DEFAULT_CONFIDENCE_EXEMPLAR_ONLY, policy)


def resplit_charge_every(policy: dict | None = None) -> int:
    """How re-split quadrants are billed by the server: 1 credit per this many
    quadrants (1 = every quadrant billed, the pre-discount behavior; 4 = one
    credit per re-scanned saturated tile; 0 = quadrants fully free). Drives
    only the client's credit clamp on the re-split budget. Defaults to 1 so
    the clamp stays fully conservative against servers that bill everything."""
    val = seed_policy(policy).get("resplit_charge_every")
    if isinstance(val, (int, float)) and not isinstance(val, bool) and val >= 0:
        return int(val)
    return 1


def saturation_policy(policy: dict | None = None) -> dict:
    """The seed.saturation sub-policy (saturated-tile re-split tuning)."""
    sat = seed_policy(policy).get("saturation")
    return sat if isinstance(sat, dict) else {}


def _sat_float(key: str, fallback: float, policy: dict | None) -> float:
    """A numeric saturation scalar, or the caller's fallback constant."""
    val = saturation_policy(policy).get(key)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return fallback


def mask_cap_trigger_frac(fallback: float, policy: dict | None = None) -> float:
    """Fraction of the per-inference mask ceiling at/above which a tile counts
    as truncated (drives the re-split ladder and the dense hint). Fallback is
    the client's own constant, passed in by the caller."""
    return _sat_float("cap_trigger_frac", fallback, policy)


def subdiv_max_depth(fallback: int, policy: dict | None = None) -> int:
    """Saturated-tile re-split recursion ceiling (fallback: client constant)."""
    val = _sat_float("subdiv_max_depth", float(fallback), policy)
    return int(val) if val >= 0 else fallback


def max_tile_coverage(fallback: float, policy: dict | None = None) -> float:
    """Tile-coverage fraction above which a SEPARATE-mode mask must pass the
    compactness check (failure-blob gate). Fallback: client constant."""
    return _sat_float("max_tile_coverage", fallback, policy)


def hard_tile_coverage(fallback: float, policy: dict | None = None) -> float:
    """Tile-coverage fraction above which a SEPARATE-mode mask is dropped as a
    fill-everything failure regardless of shape. Fallback: client constant."""
    return _sat_float("hard_tile_coverage", fallback, policy)


def adaptive_confidence_policy(policy: dict | None = None) -> dict:
    """The review.adaptive_confidence sub-policy (data-driven starting-cutoff
    tuning). Empty dict when absent: the client's generic constants apply."""
    val = review_policy(policy).get("adaptive_confidence")
    return val if isinstance(val, dict) else {}


def exemplar_policy(policy: dict | None = None) -> dict:
    """The exemplar sub-policy (example-crop framing)."""
    policy = get_detection_policy() if policy is None else policy
    exemplar = policy.get("exemplar") if isinstance(policy, dict) else None
    return exemplar if isinstance(exemplar, dict) else {}


def exemplar_context_pad(policy: dict | None = None) -> float:
    """Fractional margin around a drawn example crop (0.10 = 10%)."""
    val = exemplar_policy(policy).get("context_pad")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return 0.10


def prompt_policy(policy: dict | None = None) -> dict:
    """The prompt sub-policy (guard-rail word sets and the offline lexicon).

    The tuned tables live server-side; without a policy this is an empty dict
    and the prompt guard degrades to its generic English fallbacks."""
    policy = get_detection_policy() if policy is None else policy
    prompt = policy.get("prompt") if isinstance(policy, dict) else None
    return prompt if isinstance(prompt, dict) else {}
