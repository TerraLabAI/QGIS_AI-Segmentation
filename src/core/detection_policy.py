"""Server-delivered detection policy for Automatic mode.

The per-object detail targets (which ground resolution to seed per object) and
the review shape defaults and size floors are provided by the plugin's server
configuration and cached in memory. Without that configuration the plugin uses
one neutral default for each value, so Automatic mode still runs (quality only
depends on it). Manual mode reads the shape dials from here too, but never the per-object
tiers: it has no prompt, so it cannot pick a class.

This file holds the matching and fallback MECHANISMS only; the tuned tables
live in the server configuration. Pure Python with no Qt at import time, so it
is safe to import from the controller, the dock and the headless path alike.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

    from .building_regularizer import RegularizePolicy

from .prompt_taxonomy import (
    first_entry_match,
    iter_keywords,
    keyword_matches,
    longest_keyword_match,
    normalize_prompt,
)
from .tile_manager import (
    AUTO_OBJECT_MIN_PX,
    AUTO_SEED_TILE_CAP,
    DEFAULT_AUTO_TILE_BUDGET,
    DEFAULT_SEED_MUPP_M,
    DETAIL_MAX_OBJECT_TILE_FRAC,
    MASK_SCALE_MIN_WIDTH_PX,
    NATIVE_OVERSAMPLE_MAX,
    QUALITY_FLOOR_MUPP_M,
    SPLIT_RISK_TILE_FRAC,
    SWEET_SPOT_MAX_MUPP_M,
)


def _is_finite_policy_value(value: object) -> TypeGuard[int | float]:
    """Whether a policy value is a finite numeric scalar, excluding bool."""
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
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


def policy_rev(policy: dict | None = None) -> int | None:
    """The server policy blob's version number as an int, or None.

    Per-run provenance: which policy revision produced a run. The blob carries
    a ``version`` key; fails open to None (no policy, or a malformed version),
    like every other getter here."""
    policy = get_detection_policy() if policy is None else policy
    val = policy.get("version") if isinstance(policy, dict) else None
    if _is_finite_policy_value(val):
        return int(val)
    return None


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


def review_correct_default_method(policy: dict | None = None) -> str:
    """Which fix method the Correct step opens on: ``"ai"`` or ``"manual"``.

    Read from the review sub-policy, beside its siblings, NOT from the top
    level: the blob carries no root ``review`` section, so a reader pointed
    there can never be moved by a deploy however the server is edited.
    """
    val = review_policy(policy).get("correct_default_method")
    return val if val in ("ai", "manual") else "manual"


def review_correct_default_method_ready(policy: dict | None = None) -> str:
    """Which fix method the Correct step opens on when the AI one is already
    usable: ``"ai"`` or ``"manual"``.

    A SECOND key beside ``correct_default_method``, never a replacement. The
    plain one answers for every plugin version and must keep its meaning, so
    this one carries the case the older versions cannot judge: the AI method in
    reach with nothing left to install. The caller decides that part locally
    and only asks here when the answer is yes.

    One generic fallback, and it is the plain default: an absent key leaves the
    step exactly where it opens today.
    """
    val = review_policy(policy).get("correct_default_method_ready")
    if val in ("ai", "manual"):
        return val
    return review_correct_default_method(policy)


def review_noise_floor(policy: dict | None = None) -> float:
    """Confidence fraction below which a detection is excluded from the review
    entirely: it never counts in the total, never renders, and the review
    confidence controls cannot dial below it. Read from the server
    ``review.noise_floor``; the fallback is ONE generic client value (0.05),
    never a mirror of a tuned table. Clamped to [0, 1)."""
    val = review_policy(policy).get("noise_floor")
    if _is_finite_policy_value(val):
        f = float(val)
        if 0.0 <= f < 1.0:
            return f
    return 0.05


def click_unsure_below(policy: dict | None = None) -> float:
    """Predicted-IoU floor under which a click's answer is called unsure.

    The model returns a score with every mask, and that score is a real but
    weak failure signal: it ranks bad answers above good ones more often than
    not, and no floor separates them cleanly. Any floor tight enough to be
    worth showing also marks some good outlines for nothing.

    Weak enough that the number belongs on the server, not in a release: this
    reads ``review.click_unsure_below`` and falls back to 0.0, which marks
    nothing. Turning it on is a blob edit, and turning it off again is another.
    Never a gate: the answer is always shown, it is only said to be uncertain.
    """
    val = review_policy(policy).get("click_unsure_below")
    if _is_finite_policy_value(val):
        f = float(val)
        if 0.0 <= f < 1.0:
            return f
    return 0.0


def _review_float(key: str, fallback: float, policy: dict | None) -> float:
    """A numeric review-policy scalar, or the caller's fallback constant."""
    val = review_policy(policy).get(key)
    if _is_finite_policy_value(val):
        return float(val)
    return fallback


def pinhole_fill_m(fallback: float = 0.0, policy: dict | None = None) -> float:
    """Ground size (metres across) of the largest interior hole treated as a
    pinhole to fill rather than a real courtyard to keep, read from
    ``review.pinhole_m``. This is the per-TILE half of the decision, applied
    before anything is polygonized; the review-side half is the per-class
    ``fill_holes`` / ``fill_holes_max_m2`` pair in the class settings.

    Must stay positive. The default fallback 0.0 means "no server value", which
    tells the vectorizer to apply its own constant, so the client fallback
    lives in one place (core.cloud_detection)."""
    val = _review_float("pinhole_m", fallback, policy)
    return val if val > 0 else fallback


def tile_simplify_mult(fallback: float = 0.0, policy: dict | None = None) -> float:
    """Multiple of the mask grid step used as the staircase simplify tolerance
    when a detection is vectorized, read from ``review.tile_simplify_mult``.
    Must stay positive; 0.0 means "no server value" and the vectorizer applies
    its own constant, exactly like pinhole_fill_m.

    Resolve it ONCE per run and carry it with the run: replaying an archived
    run must vectorize with the value that run used, not the current one."""
    val = _review_float("tile_simplify_mult", fallback, policy)
    return val if val > 0 else fallback


def smooth_pass_settings(policy: dict | None = None) -> dict:
    """The "Round corners" pass, read from ``review.smooth``.

    ``iterations`` is how many Chaikin passes the tick runs, ``offset`` how far
    along each edge a corner is cut, ``max_angle_deg`` the angle above which a
    vertex is left alone (a near-straight wall corner). Each pass roughly
    doubles the point count, so iterations is the dial that decides whether
    rounding pays for itself against the point budget that ran before it.

    Fallbacks are the ONE generic client set, so a client with no policy keeps
    today's outline. Values outside their valid range fall back individually.
    """
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
    """Width in RETURNED-MASK pixels under which a detection is mask noise,
    whatever the prompt. Read from ``review.min_size_noise_px`` (and
    ``review.min_size_noise_px_no_prompt`` for a run with no text, where no
    per-object floor holds the line). Squared with the mask ground pixel it is
    the area the Min size control opens at.

    The caller passes its own generic constant as the fallback, so the client
    keeps one value and the server keeps the tuned one. Must stay positive."""
    key = "min_size_noise_px_no_prompt" if no_prompt else "min_size_noise_px"
    val = _review_float(key, fallback, policy)
    return val if val > 0 else fallback


def fill_holes_floor_m2(fallback: float, policy: dict | None = None) -> float:
    """Ground area (m2) up to which interior holes are filled on EVERY run,
    whatever a class asked for, read from ``review.fill_holes_floor_m2``.

    It is a floor, not a ceiling: a class tuned to keep its holes still gets
    the mask pepper filled. 0 is a legal value and turns the floor off, so a
    class that fills nothing really fills nothing; a negative or non-numeric
    value falls back to the caller's generic constant."""
    val = review_policy(policy).get("fill_holes_floor_m2")
    if _is_finite_policy_value(val) and val >= 0:
        return float(val)
    return fallback


def vertex_budget_settings(policy: dict | None = None) -> dict:
    """How many points an exported outline is allowed to carry.

    A traced mask puts a vertex every fraction of a metre; a polygon drawn by
    hand, or shipped in a reference database, puts one every few metres. The
    dense version is not extra accuracy, it is the pixel grid, and it makes the
    layer slow to draw and painful to edit. So an outline gets a point BUDGET
    from its own length, in GROUND METRES, which keeps it the same shape at
    every detail level (a pixel-anchored setting would not).

        spacing_m (float): one point per this much outline; 0 = step off
        min_vertices (int): floor, so a small object stays a shape
        max_deviation_m (float): never drop a point if that moves the boundary
            further than this; 0 = no cap
        max_deviation_fraction (float): the same cap as a share of the object's
            own narrow dimension, so one run can hold a warehouse and a hedge;
            the tighter of the two wins. 0 = no per-object ceiling
        smooth_spacing_factor (float): spacing multiplier when Round corners
            is on; smoothing multiplies the outline points after the budget,
            so the budget thins ahead by the same factor. 1 = off
        smooth_max_deviation_m (float): the flat cap used pre-smooth; corner
            rounding moves the boundary on its own, so the pre-smooth cap can
            sit looser than the plain one
        smooth_min_vertices (int): the floor used pre-smooth
        dial_max_cap_fraction (float): ceiling on how far a corner may travel
            when the Points dial is turned all the way down, as a share of the
            object's narrow dimension; 0 = no relaxed-cap ceiling
        smooth_multiplier_cap (float): ceiling on the compounded pre-smooth
            spacing multiplier, so a high pass count cannot starve a shape of
            points; 0 = uncapped

    Read from the server ``review.vertex_budget``. The fallbacks are ONE
    generic client set, never a mirror of a tuned per-class table.
    """
    pol = review_policy(policy).get("vertex_budget")
    pol = pol if isinstance(pol, dict) else {}

    def _zero_or_positive(value: object, fallback: float) -> float:
        """A server 0 is an instruction (turn the step off), not a missing
        value, so it must survive where _positive_number would reject it."""
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
        # Round corners runs AFTER the budget and one Chaikin pass doubles
        # the outline points, so a smooth-on export overshoots the promised
        # density unless the budget compensates: spacing scales by the
        # factor, the flat cap and the floor swap to their pre-smooth
        # values (the fraction cap needs nothing, it is already relative).
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
    """Whether a prompt names tree cover, for guidance nudges only (the
    Coverage Continuous-cover tip, the shadow exclude-example tip). Never a
    behaviour switch: the merge grouping keeps its own policy.

    The word list arrives server-side at ``review.canopy_hint_tokens``
    (additive); the fallback is a short generic seed, not a tuned table."""
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


def restore_partitions_for(prompt: str, policy: dict | None = None,
                           exemplar_only: bool = False) -> bool:
    """Whether this run should give back the objects a coarse reading
    swallowed (IncrementalMerger.restore_absorbed_partitions).

    The merger's additive union skips a member that adds no new area. For a
    cross-tile jitter duplicate that is right. For the individual buildings of
    a complex the model ALSO read as one shape it is wrong: each sits whole
    inside that shape, adds nothing, and is dropped. Measured offline over ten
    archived runs, that is 8.6 points of recall, the largest single loss in
    the client pipeline.

    It is per CLASS, not global, and the numbers say why. Restoring the parts
    on building and house lifts recall 0.783 to 0.867 on the tuning runs and
    0.943 to 0.965 on the holdout, for 0.008 and 0.006 of precision. On tree,
    swimming pool and lake it only costs precision (lake 1.000 to 0.875): a
    canopy or a pool read in pieces is one object fragmented, not a group.
    Car and solar panel never trigger it at all.

    So the list arrives server-side at ``review.merge.restore_partition_
    prompts`` and the fallback is EMPTY: no class gets this until the server
    says so, and a client with no policy behaves exactly as before.

    ``exemplar_only`` is the run with a drawn example and NO text: it names no
    class, so the per-class list cannot decide it. It gets its own switch,
    ``review.merge.restore_partition_exemplar_only``, also off by default,
    because the measurement behind the class list covers text runs only (every
    archived run it used was a text prompt with no exemplars) and a drawn
    example of one building is exactly the case where the model is most likely
    to answer with the block.
    """
    if exemplar_only:
        return bool(merge_policy(policy).get("restore_partition_exemplar_only"))
    norm = (prompt or "").strip().lower().replace("_", " ")
    if not norm:
        return False
    names = merge_policy(policy).get("restore_partition_prompts")
    if not isinstance(names, list):
        return False
    wanted = frozenset(
        str(v).strip().lower().replace("_", " ")
        for v in names if isinstance(v, str) and str(v).strip())
    return norm in wanted


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
    return not (isinstance(val, str) and val.strip().lower() == "map")


def map_likeness_min_share(policy: dict | None = None) -> float:
    """Minimum map-likeness for an EXEMPLAR-only run to be grouped as continuous
    cover (MAP) rather than counted as distinct objects.

    Map-likeness is the area-weighted mean tile coverage of the run's fragments
    (near zero for small countable objects, high for continuous cover). Read
    from the server ``review.merge`` policy (``map_likeness_min_share``); the
    fallback is ONE generic client value (0.15), never a mirror of the tuned
    server tables."""
    val = merge_policy(policy).get("map_likeness_min_share")
    if _is_finite_policy_value(val):
        return float(val)
    return 0.15


def max_concurrent(policy: dict | None = None) -> int:
    """Cap on concurrent in-flight tiles per run (fallback 6)."""
    val = seed_policy(policy).get("max_concurrent")
    if _is_finite_policy_value(val):
        return int(val)
    return 6


# Generic client fallbacks for the review.merge scalars. One value per key, the
# single fallback source when the server policy omits or malforms a scalar.
#
# A key here says what it gates, never how it was tuned: the merger has to run
# with no config at all, so every key needs a shipped number, and this file is
# public. The tuning lives with the server table that owns these values.
_MERGE_SCALAR_DEFAULTS: dict[str, float] = {
    "merge_ios": 0.15,
    "dedup_ios": 0.5,
    "dup_ios_floor": 0.3,
    "dup_centroid_frac": 0.35,
    "seam_span_ios": 0.03,
    "ios_threshold": 0.5,
    # Share of the seam-strip width an overlap must span before the seam
    # rescue accepts it (mask edges land short of the tile border, so a real
    # seam strip measures slightly under the theoretical width).
    "seam_span_tol": 0.85,
    # Added-area floor, as a share of the largest member, under which a
    # stitched member counts as outline jitter (used when no pixel size is
    # known to erode by).
    "jitter_area_frac": 0.02,
    # Detection pixels of erosion the added area must survive for a stitched
    # member to count as a real seam complement rather than outline jitter.
    # The sibling above is the fallback when no pixel size is known; this is
    # the test a real run takes.
    "jitter_erode_px": 1.0,
    # Share of its own area an object must have painted over by LARGER objects
    # before the end-of-run sweep drops it as a leftover partial reading. It
    # outscoring every coverer still saves it.
    "cover_threshold": 0.40,
    # Area share of the largest member above which a member counts as a
    # co-extensive reading and may carry the group's score.
    "score_floor_frac": 0.5,
    # The five below shape the partition restore, so they are read only by a
    # run that turned restore_partitions on (see restore_partitions_for).
    # Share of a child's own area that must sit inside the coarse reading for
    # the child to count as one of its parts.
    "part_inside": 0.90,
    # Largest share of the coarse reading one child may take: above it the
    # child IS that reading, not a part of it.
    "part_max_frac": 0.70,
    # Overlap (IoS) at which two candidate children read as the same part, so
    # only the first is kept.
    "part_sibling_ios": 0.20,
    # Share of the coarse reading the kept children must account for together
    # before it is replaced by them.
    "part_cover_frac": 0.60,
    # How many children a coarse reading must break into to be replaced.
    "part_min_children": 2,
}

# review.merge scalars that carry a COUNT, not a ratio. They must arrive as
# ints, because the consumer compares them against a length.
_MERGE_INT_SCALARS: frozenset[str] = frozenset({"part_min_children"})


def merge_scalar(key: str, fallback: float | None = None, policy: dict | None = None) -> float:
    """One numeric review.merge scalar, or the client fallback.

    ``fallback`` defaults to the generic constant for ``key`` when omitted, so
    the fallback values live in one place. A key listed in _MERGE_INT_SCALARS
    comes back as an int, so a server float never reaches a count."""
    if fallback is None:
        fallback = _MERGE_SCALAR_DEFAULTS.get(key, 0.0)
    val = merge_policy(policy).get(key)
    # A negative is treated as absent. Every one of these is a ratio, a count
    # or a pixel width, so no served negative is meaningful, and one of them
    # reverses a shape operation rather than just mistuning it: a negative
    # jitter_erode_px turns the merger's erosion into a dilation, which swells
    # every outline fleet-wide with nothing to say why.
    ok = _is_finite_policy_value(val) and float(val) >= 0.0
    resolved = float(val) if ok else float(fallback)
    return int(resolved) if key in _MERGE_INT_SCALARS else resolved


def merge_scalars(policy: dict | None = None) -> dict[str, float]:
    """Every review.merge scalar resolved (policy value or generic fallback)."""
    return {k: merge_scalar(k, d, policy) for k, d in _MERGE_SCALAR_DEFAULTS.items()}


def merge_scalar_kwargs(target: object, scalars: dict | None = None,
                        policy: dict | None = None) -> dict[str, float]:
    """The review.merge scalars ``target`` accepts, ready to splat as kwargs.

    ``target`` is the consumer itself (a class or a callable): its own
    signature decides which scalars apply. A scalar added to the shared
    defaults then reaches every consumer that takes it without a second edit
    per call site, and a scalar meant for a different consumer is dropped
    rather than raising. ``scalars`` reuses an already resolved set so one run
    reads the policy once. An unreadable signature yields an empty dict, which
    leaves the consumer on its own defaults."""
    import inspect

    values = scalars if isinstance(scalars, dict) else merge_scalars(policy)
    try:
        accepted = inspect.signature(target).parameters  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in values.items() if k in accepted}


def object_profile(prompt: str, policy: dict | None = None) -> tuple[float, float]:
    """(typical ground size m, target ground resolution m/px) for a prompt.

    The seed policy carries a tier list (finest first, targets already
    resolution-adjusted server-side); the first tier with a matching keyword
    wins. Tier ORDER is the priority here on purpose: a multi-object prompt
    must be rendered fine enough for its smallest object, so "car parking lot"
    seeds at car scale, not parking scale. Matching itself is the shared strict
    whole-word rule (see prompt_taxonomy), so a keyword never lands inside a
    longer unrelated word ("dam" must not seed "damaged roof" as an 80 m dam).

    Without a policy the generic default applies; with a policy present but no
    tier match, the policy's own default object applies.
    """
    generic = (10.0, DEFAULT_SEED_MUPP_M)
    seed = seed_policy(policy)
    tiers = seed.get("object_tiers")
    if not isinstance(tiers, list):
        return generic
    text = normalize_prompt(prompt)
    tier = first_entry_match(text, tiers)
    if tier is not None:
        return _profile_pair(tier, generic)
    return _profile_pair(seed.get("default_object"), generic)


def _profile_pair(entry: object, fallback: tuple[float, float]) -> tuple[float, float]:
    """Read (size_m, target_mupp) from a policy entry, else the fallback."""
    if isinstance(entry, dict):
        try:
            size_m = entry["size_m"]
            target_mupp = entry["target_mupp"]
        except KeyError:
            return fallback
        if _is_finite_policy_value(size_m) and _is_finite_policy_value(target_mupp):
            return float(size_m), float(target_mupp)
        return fallback
    return fallback


def object_tile_floor_m(prompt: str, policy: dict | None = None) -> float:
    """Smallest tile ground side (metres) this object is known to detect at,
    or 0.0 when the server names none.

    Not the object's own size. Some objects need a fixed amount of ground in
    frame whatever their width, and under it the detection quietly degrades,
    so no per-object-size rule can express the bound. Read off the SAME seed
    tier ``object_profile`` resolves, so one prompt cannot take one tier's
    resolution and another tier's floor.

    0.0 is the shipped fallback and means the object-size rule alone bounds the
    fine end of the Precision slider, which is exactly the behaviour before
    this key existed.
    """
    seed = seed_policy(policy)
    entry: object = seed.get("default_object")
    tiers = seed.get("object_tiers")
    if isinstance(tiers, list):
        tier = first_entry_match(normalize_prompt(prompt), tiers)
        if tier is not None:
            entry = tier
    if isinstance(entry, dict):
        val = entry.get("min_tile_ground_m")
        if _is_finite_policy_value(val) and val > 0:
            return float(val)
    return 0.0


def mask_scale_policy(policy: dict | None = None) -> dict:
    """The seed.mask_scale sub-policy (coarse mask-grid routing table).

    Empty dict when absent; the routing is fail-CLOSED (no table = never ask
    the service for the coarser mask grid), so behaviour changes only once the
    server ships a class list."""
    val = seed_policy(policy).get("mask_scale")
    return val if isinstance(val, dict) else {}


def mask_scale_min_width_px(policy: dict | None = None) -> float:
    """Native-pixel floor an object's NARROW dimension must clear to survive the
    coarse mask grid. Below it a half-cell boundary shift erodes the object's
    thin parts, which is what kept car-scale and micro-pool runs on the full
    grid. Server-tunable; the fallback is the validated routing floor."""
    val = mask_scale_policy(policy).get("min_width_px")
    if _is_finite_policy_value(val) and val > 0:
        return float(val)
    return MASK_SCALE_MIN_WIDTH_PX


def _matched_tiers(text: str, policy: dict | None) -> list[dict]:
    """EVERY seed tier the prompt hits, not just the winning one.

    ``object_profile`` resolves one tier (the finest) because a run renders at
    one resolution. Mask-grid routing needs the full set instead: a prompt
    naming several objects ("building and road") may only take the coarse grid
    when EVERY object it names survives it."""
    tiers = seed_policy(policy).get("object_tiers")
    if not isinstance(tiers, list):
        return []
    return [
        tier
        for tier in tiers
        if isinstance(tier, dict) and any(keyword_matches(text, kw) for kw in iter_keywords(tier))
    ]


def _entry_min_width_m(entry: dict) -> float | None:
    """A routing class's measured NARROW ground width in metres, or None when
    the server has not annotated it. Presence is the opt-in: only an annotated
    class can take the derived coarse-grid path.

    This lives on the mask-grid CLASS, never on the seed tier: seed tiers group
    objects by the resolution they need, which mixes very different widths in
    one row (a road, a house and a pool share a tier), so a tier-level width
    would let a thin object inherit a fat one's clearance."""
    val = entry.get("min_width_m")
    if _is_finite_policy_value(val) and val > 0:
        return float(val)
    return None


def _entry_max_mupp(entry: dict) -> float | None:
    """A routing class's hand-set resolution ceiling, or None when absent or
    malformed. Kept alongside the derived width rule: it bounds a class at the
    coarsest resolution that class is known to work at."""
    val = entry.get("max_mupp")
    if _is_finite_policy_value(val) and val > 0:
        return float(val)
    return None


def _prompt_names_an_unlisted_object(
    text: str, classes: list, policy: dict | None
) -> bool:
    """Whether the prompt names an object the routing table does not cover.

    A prompt can name several objects ("building and road"). The mask-grid
    table only speaks for the classes it lists, so any OTHER object the seed
    tiers recognise in the same prompt is unproven for the coarse grid, and the
    whole run must stay on the full grid. Without this a thin unlisted object
    silently rode along on a listed fat one's eligibility."""
    class_keywords = [
        ckw for entry in classes if isinstance(entry, dict) for ckw in iter_keywords(entry)
    ]
    for tier in _matched_tiers(text, policy):
        for tier_kw in iter_keywords(tier):
            if not keyword_matches(text, tier_kw):
                continue
            # Match the routing keywords against the TIER's own word, not
            # against the prompt: that is what ties this named object to a
            # routing class, independently of the two tables' vocabularies.
            if not any(keyword_matches(tier_kw, ckw) for ckw in class_keywords):
                return True
    return False


def mask_scale_for_run(
    prompt: str, run_mupp: float, policy: dict | None = None
) -> int:
    """The mask-grid scale to request for a whole run: 2 (coarse) or 1 (full).

    The routing table is ``seed.mask_scale.classes``, an ordered GO list. A run
    takes the coarse grid only when ALL of these hold, so anything unproven
    stays on the full grid:

    1. The prompt matches a listed class (whole-word, see prompt_taxonomy).
    2. The prompt names no OTHER object the table does not cover, so a thin
       unlisted object ("building and road") cannot ride along on a listed fat
       one's eligibility.
    3. The class clears its resolution test:
       - DERIVED (preferred) when the class carries ``min_width_m``, its
         measured NARROW ground dimension: ``min_width_m / run_mupp >=
         min_width_px``. This is the validated routing rule computed rather
         than approximated, so re-tuning a class needs no plugin release.
         The floor keys on the NARROW dimension, never on a typical LENGTH: a
         car is ~4 m long but ~1.8 m wide, and it is the 1.8 m that decides
         whether the coarse grid erodes it.
       - LEGACY otherwise: the hand-set ``max_mupp`` ceiling, so a plugin newer
         than the config blob behaves exactly as the shipped fleet does.
       A class may carry both, and then both must pass (the width rule opens a
       class up, the ceiling still bounds it where measurement stops).
    """
    if not _is_finite_policy_value(run_mupp):
        return 1
    if run_mupp <= 0:
        return 1
    text = normalize_prompt(prompt)
    if not text:
        return 1
    classes = mask_scale_policy(policy).get("classes")
    if not isinstance(classes, list):
        return 1

    # Only an entry carrying a usable resolution test can decide: a row with
    # neither a width nor a ceiling (or with a malformed one) is skipped rather
    # than allowed to veto a later valid row for the same class, so one bad
    # value never disables routing for a class the table does cover.
    usable = [
        item
        for item in classes
        if isinstance(item, dict) and (_entry_min_width_m(item) is not None or _entry_max_mupp(item) is not None)
    ]
    entry = first_entry_match(text, usable)
    if entry is None:
        return 1
    if _prompt_names_an_unlisted_object(text, classes, policy):
        return 1

    width = _entry_min_width_m(entry)
    cap = _entry_max_mupp(entry)
    if width is not None and width / float(run_mupp) < mask_scale_min_width_px(policy):
        return 1
    if cap is not None and float(run_mupp) > cap:
        return 1
    return 2


def _seed_float(key: str, fallback: float, policy: dict | None) -> float:
    """A numeric seed-policy scalar, or the client fallback constant."""
    val = seed_policy(policy).get(key)
    if _is_finite_policy_value(val):
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
    trial; subscribers are never capped. Fallback 0.5, which still leaves two
    full runs in a month while letting the largest zone a free plan may draw
    land at a resolution the coarse-imagery warning does not fire on. A
    quarter did not: it paired a large permitted zone with a grid too coarse
    to detect anything on it."""
    val = _seed_float("free_run_fraction", 0.5, policy)
    return val if 0.0 < val <= 1.0 else 0.5


def free_monthly_allowance(policy: dict | None = None) -> int:
    """The free-tier monthly allowance (in tiles/credits) assumed when the
    server's usage payload omits its own total.

    Only the per-run share cap (``free_run_fraction``) is derived from it;
    nothing here grants or spends anything. Read from
    ``seed.free_monthly_allowance`` so the number can move without a plugin
    release. Fallback 200. The live number is served, and the UI reads the
    server's own ``free_detections_total``. Must stay positive."""
    val = int(_seed_float("free_monthly_allowance", 200.0, policy))
    return val if val > 0 else 200


def free_zone_max_km2(fallback: float, policy: dict | None = None) -> float:
    """Max geodesic area (km2) a free-tier Detect zone may cover before the
    draw is refused with an upsell. Served, so it can move without a plugin
    release. The fallback is the client's own
    generic constant (kept in the UI layer and passed in here so this core
    module stays free of any UI import). Must stay positive."""
    val = _seed_float("free_zone_max_km2", fallback, policy)
    return val if val > 0 else fallback


def max_tiles_per_run(fallback: int, policy: dict | None = None) -> int:
    """Hard ceiling on how many tiles (credits) one Detect may span. A served
    ceiling, so it can move without a plugin release. The fallback is the
    client's own constant, passed in so this core module stays UI-free. Must
    stay positive."""
    val = int(_seed_float("max_tiles_per_run", float(fallback), policy))
    return val if val > 0 else fallback


def tile_jpeg_quality(fallback: int, policy: dict | None = None) -> int:
    """JPEG quality for the tile upload encode. A fidelity/bandwidth dial the
    server can tune without a plugin release (detection scores near the review
    cutoff are sensitive to compression noise, so the client default stays
    near-lossless; the server may lower it for bandwidth). The fallback is the
    client's own constant. Clamped to a sane JPEG range."""
    val = int(_seed_float("tile_jpeg_quality", float(fallback), policy))
    return min(100, max(60, val)) if val > 0 else fallback


def object_min_px(policy: dict | None = None) -> int:
    """Minimum pixels across for an object to count as resolvable."""
    return int(_seed_float("object_min_px", AUTO_OBJECT_MIN_PX, policy))


def detail_max_object_tile_frac(policy: dict | None = None) -> float:
    """Share of a tile's ground side an object may take before extra detail
    stops paying for it. Bounds the fine end of the Precision slider.

    Its own key, not the sibling `max_object_tile_frac`: that one guards the
    SEED, so borrowing it would tie how far a user may push the slider to what
    a default run costs and returns. `split_risk_tile_frac` marks a later point
    again, where pieces stop being stitchable, which is what the amber warning
    claims. Three limits, three keys.
    """
    val = _seed_float("detail_max_object_tile_frac",
                      DETAIL_MAX_OBJECT_TILE_FRAC, policy)
    return val if 0 < val <= 1 else DETAIL_MAX_OBJECT_TILE_FRAC


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


def split_risk_tile_frac(policy: dict | None = None) -> float:
    """Object ground size as a fraction of a tile's ground side at or above
    which the slider guidance may say large ones can come back split.

    Below it the object is small next to the tile and the claim is false, so
    the guidance must not make it. Only a fraction in (0, 1] is accepted."""
    val = _seed_float("split_risk_tile_frac", SPLIT_RISK_TILE_FRAC, policy)
    return val if 0.0 < val <= 1.0 else SPLIT_RISK_TILE_FRAC


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
    if _is_finite_policy_value(val) and val >= 0:
        return int(val)
    return 1


def saturation_policy(policy: dict | None = None) -> dict:
    """The seed.saturation sub-policy (saturated-tile re-split tuning)."""
    sat = seed_policy(policy).get("saturation")
    return sat if isinstance(sat, dict) else {}


def _sat_float(key: str, fallback: float, policy: dict | None) -> float:
    """A numeric saturation scalar, or the caller's fallback constant."""
    val = saturation_policy(policy).get(key)
    if _is_finite_policy_value(val):
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


def resplit_time_ratio(fallback: float, policy: dict | None = None) -> float:
    """How much of the PAID grid's own duration the free re-split tail may
    spend before it stops queueing and drops what is left (fallback: client
    constant; 0 or less means no clock)."""
    val = _sat_float("resplit_time_ratio", fallback, policy)
    return val if val >= 0 else fallback


def max_masks_per_tile(fallback: int, policy: dict | None = None) -> int:
    """Per-inference instance ceiling requested from the service (drives the
    submission's max_masks and, with cap_trigger_frac, the saturation trigger).
    Fallback is the client's own constant; must stay positive."""
    val = _sat_float("max_masks_per_tile", float(fallback), policy)
    return int(val) if val > 0 else fallback


def gsd_warn_max_mupp(fallback: float, policy: dict | None = None) -> float:
    """Ground resolution (m/px) at/above which the detail slider shows the
    coarse-imagery quality warning. Fallback: client constant; must stay
    positive."""
    val = _seed_float("gsd_warn_max_mupp", fallback, policy)
    return val if val > 0 else fallback


def subdivide_overlap_fraction(fallback: float, policy: dict | None = None) -> float:
    """Sibling overlap of the 2x2 re-split quadrants, as a fraction of the
    parent side added to each half. Fallback: client constant; only sane
    fractions (0 <= f < 0.5) are accepted."""
    val = _sat_float("subdivide_overlap_fraction", fallback, policy)
    return val if 0 <= val < 0.5 else fallback


def subdivide_min_parent_px(fallback: int, policy: dict | None = None) -> int:
    """Smallest tile side (run-grid px) still worth re-splitting. Fallback:
    client constant; must stay positive."""
    val = _sat_float("subdivide_min_parent_px", float(fallback), policy)
    return int(val) if val > 0 else fallback


def subdivide_cap_params(
    fallback_max: int, fallback_min: int, fallback_scale: int,
    policy: dict | None = None,
) -> tuple[int, int, int]:
    """(hard cap, floor, per-base-tile scale) for the re-split tile budget
    ceiling (credit_gate.subdivide_cap). Fallbacks: the client's constants,
    passed in by the caller; each value must stay positive."""
    sat = saturation_policy(policy)

    def _pos_int(key: str, fb: int) -> int:
        val = sat.get(key)
        if _is_finite_policy_value(val) and val > 0:
            return int(val)
        return fb

    return (
        _pos_int("subdivide_cap_max", fallback_max),
        _pos_int("subdivide_cap_min", fallback_min),
        _pos_int("subdivide_cap_scale", fallback_scale),
    )


def network_policy(policy: dict | None = None) -> dict:
    """The top-level network sub-policy (worker retry/offline budgets). An
    operations dial: loosen retry budgets fleet-wide during a backend incident
    without a plugin release. Empty dict when absent."""
    src = get_detection_policy() if policy is None else policy
    val = src.get("network") if isinstance(src, dict) else None
    return val if isinstance(val, dict) else {}


def _net_float(
    key: str, fallback: float, policy: dict | None, high: float | None = None,
    low: float | None = None,
) -> float:
    """A served network dial, or the shipped constant.

    ``high`` is a ceiling, not a clamp: a value above it is refused whole and
    the constant stands. These dials decide how long a client waits and how
    many times it retries, so one bad deploy with no ceiling wedges the whole
    fleet at once with nothing client-side to stop it. install_config already
    bounds every equivalent through dial_in_range; this is the same rule for
    the network block. A reader with no ceiling has no plausible way to be too
    large (a count of retries, a worker count).

    ``low`` is the same refusal at the other end, and it exists because on a
    few of these small is the harmful direction: a budget that decides how
    long we wait for an answer we have ALREADY BILLED throws that tile away
    when it is near zero. A ceiling alone would not have caught that.
    """
    val = network_policy(policy).get(key)
    if _is_finite_policy_value(val) and val > 0:
        if high is not None and val > high:
            return fallback
        if low is not None and val < low:
            return fallback
        return float(val)
    return fallback


def max_rate_limit_retries(fallback: int, policy: dict | None = None) -> int:
    """Per-tile transient-network retry ceiling. Fallback: client constant."""
    return int(_net_float("max_rate_limit_retries", float(fallback), policy, high=50.0))


def queue_retry_budget_s(fallback: float, policy: dict | None = None) -> float:
    """Per-tile time budget (s) for queue-busy retries. Fallback: client
    constant."""
    return _net_float("queue_retry_budget_s", fallback, policy, high=1800.0)


def midrun_offline_streak(fallback: int, policy: dict | None = None) -> int:
    """Unbroken hard-connectivity failures after the first success before a
    run is declared offline. Fallback: client constant."""
    return int(_net_float("midrun_offline_streak", float(fallback), policy))


def backend_unavailable_retries(fallback: int, policy: dict | None = None) -> int:
    """Per-tile retry ceiling for a cold instance's transient backend-unavailable
    answer. Fallback: client constant."""
    return int(_net_float("backend_unavailable_retries", float(fallback), policy, high=50.0))


def backend_unavailable_delay_s(fallback: float, policy: dict | None = None) -> float:
    """Base backoff (s) between backend-unavailable retries. Fallback: client
    constant."""
    return _net_float("backend_unavailable_delay_s", fallback, policy, high=120.0)


def stall_timeout_s(fallback: float, policy: dict | None = None) -> float:
    """Seconds of zero run progress after which the main-thread watchdog
    declares the worker wedged and forces a terminal. An operations dial:
    tighten or loosen it fleet-wide without a plugin release. Fallback: client
    constant."""
    return _net_float("stall_timeout_s", fallback, policy, high=3600.0)


def busy_jitter(
    fallback: tuple[float, float], policy: dict | None = None
) -> tuple[float, float]:
    """(low, high) multipliers spread over a retry delay so tiles told to wait
    the same amount do not all come back in one synchronized wave. Read from
    ``network.busy_jitter_min`` / ``busy_jitter_max``; the fallback is the
    client's own pair. A server pair that is not ordered low <= high falls back
    whole, so the two never arrive crossed."""
    low = _net_float("busy_jitter_min", fallback[0], policy, high=10.0)
    high = _net_float("busy_jitter_max", fallback[1], policy, high=10.0)
    return (low, high) if low <= high else fallback


def prefetch_depth(fallback: int, policy: dict | None = None) -> int:
    """How many upcoming tiles the streaming path renders ahead of need.
    Fallback: client constant."""
    return int(_net_float("prefetch_depth", float(fallback), policy))


def convert_workers(fallback: int, policy: dict | None = None) -> int:
    """Threads the streaming path uses to turn a finished tile's masks into
    geometry, off the loop that drives the sockets. 0 = size from the machine.
    Fallback: client constant."""
    return int(_net_float("convert_workers", float(fallback), policy))


def prefetch_holdoff_s(fallback: float, policy: dict | None = None) -> float:
    """Pause (s) on the render prefetch after a blank or failed render, so a
    struggling imagery provider is not hit harder. Fallback: client constant."""
    return _net_float("prefetch_holdoff_s", fallback, policy, high=300.0)


def slow_notice_s(fallback: float, policy: dict | None = None) -> float:
    """Seconds of zero run progress after which the progress card tells the user
    the connection is slow rather than showing a bar that stopped moving. Below
    stall_timeout_s, which ENDS the run. Fallback: client constant."""
    return _net_float("slow_notice_s", fallback, policy, high=3600.0)


def render_slow_s(fallback: float, policy: dict | None = None) -> float:
    """Seconds a tile may wait on its imagery before the run narrows how many
    basemap fetches it keeps in flight. Fallback: client constant."""
    return _net_float("render_slow_s", fallback, policy, high=300.0)


def tile_render_timeout_ms(fallback: int, policy: dict | None = None) -> int:
    """Deadline (ms) on one tile's basemap render. Past it the tile has no
    imagery and is retried, then dropped uncharged. Fallback: client constant."""
    return int(_net_float(
        "tile_render_timeout_ms", float(fallback), policy, high=600_000.0))


def aimd_start(fallback: int, policy: dict | None = None) -> int:
    """In-flight tile width a run OPENS at before the adaptive controller grows
    it. Fallback: client constant."""
    return int(_net_float("aimd_start", float(fallback), policy))


def max_consecutive_tile_fatals(fallback: int, policy: dict | None = None) -> int:
    """Consecutive per-tile rejections, with no success in between, that abort
    the run. Fallback: client constant."""
    return int(_net_float("max_consecutive_tile_fatals", float(fallback), policy))


def render_retry_max(fallback: int, policy: dict | None = None) -> int:
    """Re-render attempts for a tile whose imagery came back blank or empty.
    Fallback: client constant."""
    return int(_net_float("render_retry_max", float(fallback), policy))


def render_retry_delay_s(fallback: float, policy: dict | None = None) -> float:
    """Base delay (s) before re-rendering a blank tile; it doubles per attempt.
    Fallback: client constant."""
    return _net_float("render_retry_delay_s", fallback, policy, high=120.0)


def gate_scan_render_tries(fallback: int, policy: dict | None = None) -> int:
    """Render attempts for a scan-phase tile before it falls open to the normal
    detect path. Fallback: client constant."""
    return int(_net_float("gate_scan_render_tries", float(fallback), policy))


def poll_interval_s(fallback: float, policy: dict | None = None) -> float:
    """Wait (s) between status polls when the answer names none. Fallback:
    client constant."""
    return _net_float("poll_interval_s", fallback, policy, high=60.0)


def poll_max_wait_s(fallback: float, policy: dict | None = None) -> float:
    """How long (s) one tile may stay pending before the poll gives up.
    Fallback: client constant."""
    return _net_float("poll_max_wait_s", fallback, policy, high=1800.0, low=5.0)


def min_poll_backoff_s(fallback: float, policy: dict | None = None) -> float:
    """Floor (s) under the coalesced poll back-off, so a tiny or zero server
    hint cannot turn the poll loop into a tight request storm. Fallback: client
    constant."""
    return _net_float("min_poll_backoff_s", fallback, policy, high=30.0)


def aimd_min(fallback: int, policy: dict | None = None) -> int:
    """Narrowest in-flight width the adaptive controller may fall back to.
    Fallback: client constant."""
    return int(_net_float("aimd_min", float(fallback), policy, high=32.0))


def convert_workers_ceiling(fallback: int, policy: dict | None = None) -> int:
    """Hard cap on the converter thread count, whatever the machine or the
    served worker count asks for. Every worker is a thread that touches the
    geometry library. Fallback: client constant."""
    return int(_net_float("convert_workers_ceiling", float(fallback), policy, high=64.0))


def convert_backlog_per_worker(fallback: int, policy: dict | None = None) -> int:
    """Finished-but-unconverted tiles allowed to queue per converter thread
    before the run loop spends its cycle draining instead of firing new tiles.
    Fallback: client constant."""
    return int(_net_float("convert_backlog_per_worker", float(fallback), policy, high=64.0))


def convert_drain_budget_s(fallback: float, policy: dict | None = None) -> float:
    """Ceiling (s) on the end-of-run wait for the last conversions. They carry
    billed geometry, so a normal end waits; this only stops a wedged converter
    from holding the terminal open for good. Fallback: client constant."""
    return _net_float("convert_drain_budget_s", fallback, policy, high=600.0, low=5.0)


def stop_drain_budget_s(fallback: float, policy: dict | None = None) -> float:
    """Ceiling (s) on the wait, after a user cancel, for tiles already in
    flight to land so their billed masks are kept. Fallback: client
    constant."""
    return _net_float("stop_drain_budget_s", fallback, policy, high=60.0, low=0.5)


def gate_render_cache_max(fallback: int, policy: dict | None = None) -> int:
    """Full-resolution scan renders held for the detect phase, so a kept tile
    renders once. Bounded to keep run memory flat at any tile count. Fallback:
    client constant."""
    return int(_net_float("gate_render_cache_max", float(fallback), policy, high=4096.0))


# Floor on the submit timeout. A window shorter than this would expire on
# almost every tile, so a bad server value must not be able to kill a run.
_SUBMIT_TIMEOUT_FLOOR_MS = 5_000


def submit_timeout_ms(fallback: int, policy: dict | None = None) -> int:
    """How long the client waits on ONE tile submission, in milliseconds.

    It is pinned against the request timeout on the receiving side: giving up
    on a tile that is still being computed would retry it and charge it twice.
    So when that side's timeout moves this has to move with it, and a plugin
    release must not be what carries the change. Read per request, since
    nothing here needs to stay constant across a run's tiles.

    Fallback: client constant. Floored at _SUBMIT_TIMEOUT_FLOOR_MS."""
    val = int(_net_float("submit_timeout_ms", float(fallback), policy, high=300_000.0))
    return val if val >= _SUBMIT_TIMEOUT_FLOOR_MS else fallback


def max_geojson_bytes(fallback: int, policy: dict | None = None) -> int:
    """Ceiling on the geometry text of the run-summary upload, in bytes.

    It mirrors a cap on the receiving side, so raising that cap must not need a
    plugin release. Fallback: client constant; must stay positive."""
    val = int(_net_float("max_geojson_bytes", float(fallback), policy))
    return val if val > 0 else fallback


def max_wkb_bytes(fallback: int, policy: dict | None = None) -> int:
    """Ceiling on the binary geometry the GUI thread collects for that same
    upload, in bytes. Sits beside max_geojson_bytes because it exists to drop,
    before any conversion, a set that is certain to blow it. Fallback: client
    constant; must stay positive."""
    val = int(_net_float("max_wkb_bytes", float(fallback), policy))
    return val if val > 0 else fallback


def max_tile_coverage(fallback: float, policy: dict | None = None) -> float:
    """Tile-coverage fraction above which a SEPARATE-mode mask must pass the
    compactness check (failure-blob gate). Fallback: client constant."""
    return _sat_float("max_tile_coverage", fallback, policy)


def hard_tile_coverage(fallback: float, policy: dict | None = None) -> float:
    """Tile-coverage fraction above which a SEPARATE-mode mask is dropped as a
    fill-everything failure regardless of shape. Fallback: client constant."""
    return _sat_float("hard_tile_coverage", fallback, policy)


def map_cover_score_floor(fallback: float, policy: dict | None = None) -> float:
    """Confidence a MAP-mode mask covering most of its tile must reach to be
    kept. 0 or less disables it, which is the behaviour that shipped.

    MAP takes coverage as the union of every hypothesis, so it has no guard
    against a tile answering "all of this is the thing" over ground that is not:
    the merger unions that mask with the correct outlines around it and the run
    comes back as one shape. A coverage cut cannot decide it, because the same
    mask is a real lake filling a tile. Confidence can: the bigger the claim,
    the surer it has to be. Nothing here is class-specific, so one value covers
    every continuous prompt.

    Fallback: caller-passed; only a value in (0, 1] arms it."""
    val = _sat_float("map_cover_score_floor", fallback, policy)
    return val if 0.0 < val <= 1.0 else fallback


def compact_min_fill(fallback: float, policy: dict | None = None) -> float:
    """Share of its oriented bounding box a large mask must fill to be kept as
    a real solid object instead of dropped as a whole-tile texture blob. Both
    gates that ARM that check are already server dials (max_tile_coverage,
    hard_tile_coverage), so the decision itself belongs on the same block.
    Fallback: client constant; only a fraction in (0, 1] is honoured."""
    val = _sat_float("compact_min_fill", fallback, policy)
    return val if 0.0 < val <= 1.0 else fallback


def tile_span_fraction(fallback: float, policy: dict | None = None) -> float:
    """Share of the tile a mask's bounding box must span, in both directions,
    for the tile to count as what drew its outline. Such a mask skips the
    compactness check entirely: it fills its oriented box perfectly, so that
    check would keep the very shape it exists to drop. Sits on the same block as
    the two gates that arm the check. Fallback: client constant; only a fraction
    in (0, 1] is honoured."""
    val = _sat_float("tile_span_fraction", fallback, policy)
    return val if 0.0 < val <= 1.0 else fallback


def min_keep_px(fallback: float, policy: dict | None = None) -> float:
    """Anti-sliver floor: a detection smaller than this many pixels on a side
    is dropped as sub-pixel noise rather than kept as an object. Expressed in
    detection pixels so it follows the run's resolution. Fallback: client
    constant; must stay non-negative (0 = never drop)."""
    val = _sat_float("min_keep_px", fallback, policy)
    return val if val >= 0 else fallback


def min_keep_floor_m2(fallback: float, policy: dict | None = None) -> float:
    """Ground floor under the anti-sliver drop, in square metres. The pixel
    floor above scales with resolution squared, so on very fine imagery it
    stops removing anything; this floor keeps a resolution-independent
    minimum. 0 = the pixel floor alone decides. Fallback: client constant;
    must stay non-negative."""
    val = _sat_float("min_keep_floor_m2", fallback, policy)
    return val if val >= 0 else fallback


# Generic client fallback keyword set for the diagonals-aware regularizer.
# One small generic set (buildings + solar), NOT a mirror of the tuned server
# table: the full keyword list and tuning live in the server policy.
_REGULARIZE_FALLBACK_KEYWORDS: tuple[str, ...] = (
    "building",
    "rooftop",
    "roof",
    "house",
    "solar",
    "panel",
    "pv",
)

# ONE generic client fallback for the regularizer's snap tolerance, in GROUND
# METRES (the unit every regularizer takes, ours and the commercial ones).
# The tuned per-class values live in the server policy.
_REGULARIZE_FALLBACK_TOLERANCE_M = 1.0
# Share of an object's own narrow ground dimension the tolerance may reach. A
# tolerance near an object's own size does not square it, it dissolves it, so
# small objects in a run keep a proportionally smaller snap distance.
_REGULARIZE_FALLBACK_OBJECT_FRACTION = 0.25
# ONE generic fallback for the 45-degree capture window, in degrees. An edge
# lands on a diagonal when it sits within (22.5 - this) of 45, so a larger
# value means fewer diagonals and more right angles, a smaller one keeps more
# near-diagonal walls as diagonals. The IoU guard backstops a corner a wide
# window would wrongly chamfer. Bounded below 22.5, past which the window
# closes on nothing. The tuned value lives server-side.
_REGULARIZE_FALLBACK_DIAGONAL_REDUCTION = 8.0
_REGULARIZE_DIAGONAL_REDUCTION_MAX = 22.5
# ONE generic fallback for the IoU above which a near-round shape is replaced
# by a true circle.
_REGULARIZE_FALLBACK_CIRCLE_THRESHOLD = 0.90
# ONE generic fallback for the de-staircase pass, as a multiple of the
# detection pixel. A raw mask outline is already all 90-degree pixel steps, so
# it is simplified before the snap; the tuned ground value lives server-side.
_DESTAIR_FALLBACK_MULT = 2.5
# Multi-direction path (per-edge structural orientation). Off by default, so an
# unset policy reproduces the single dominant-direction behaviour exactly; the
# server turns it on and tunes it. The client fallbacks are ONE generic value
# each, never a mirror of a tuned table.
_REGULARIZE_FALLBACK_MULTI_DIRECTION = False
_REGULARIZE_FALLBACK_MULTI_MAX_GROUPS = 3
_REGULARIZE_MULTI_MAX_GROUPS_MAX = 6
_REGULARIZE_FALLBACK_MULTI_MIN_SEPARATION_DEG = 10.0
_REGULARIZE_MULTI_MIN_SEPARATION_MAX = 45.0
# Angular gap (degrees) under which the multi-direction reconnect treats two
# edges as parallel, and the share of the ring's perimeter a second or third
# direction must hold to be kept. Generic guards on the same off-by-default
# path, so they move with the rest of the multi dials.
_REGULARIZE_FALLBACK_MULTI_PARALLEL_EPS_DEG = 1.5
_REGULARIZE_FALLBACK_MULTI_MIN_GROUP_WEIGHT = 0.20


def _positive_number(value: object) -> float | None:
    """The value as a positive float, or None when absent or malformed."""
    if _is_finite_policy_value(value) and value > 0:
        return float(value)
    return None


def regularize_policy(policy: dict | None = None) -> dict:
    """The review.regularize sub-policy (which classes get the diagonals-aware
    building regularizer, plus its tuning). Empty dict when absent, so the
    reader below falls open to its single generic client defaults."""
    val = review_policy(policy).get("regularize")
    return val if isinstance(val, dict) else {}


def regularize_settings(policy: dict | None = None) -> dict:
    """Settings for the footprint regularizer, resolved from the server policy
    or the single generic client fallback per value:

        keywords (tuple[str]): prompt words that opt a class into regularization
        tolerance_m (float): snap tolerance in GROUND METRES (the primary dial)
        max_object_fraction (float): ceiling on the tolerance, as a share of the
            object's own narrow ground dimension
        tolerance_mult (float): legacy tolerance = this x the mask pixel, kept
            for servers that only carry the pixel-anchored dial
        allow_diagonal (bool): snap 45-degree walls too, not just right angles
        diagonal_reduction (float): degrees taken off the 45-degree capture
            window; raise it to stop corners being cut into chamfers
        allow_circles (bool): fit near-circular blobs to circles
        circle_threshold (float): IoU above which a round blob becomes a circle
        min_keep_iou (float): revert to the original below this IoU (guard)
        multi_direction (bool): cluster edges into their own structural
            directions and snap each edge to its group's grid, so an L or
            multi-wing building keeps each wing's axis (off by default)
        multi_max_groups (int): how many structural directions the clustering
            may keep
        multi_min_separation_deg (float): smallest angular gap between two kept
            directions
        multi_parallel_eps_deg (float): angular gap under which two edges count
            as parallel when the corners are reconnected
        multi_min_group_weight (float): share of the ring's perimeter a second
            or third direction must hold to be kept at all

    The fallback is ONE generic value per key, never a mirror of the tuned
    server tables. Use regularize_tolerance_m to turn tolerance_m,
    max_object_fraction and the legacy multiplier into the tolerance for one
    object; reading them apart skips the floors.
    """
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
        # Per-RING revert, applied before min_keep_iou and to one ring rather
        # than the whole geometry. Too high and the regularizer declines every
        # complex footprint, silently, which reads as Right angles doing
        # nothing.
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
    """The regularizer's snap tolerance for ONE object, in ground metres.

    A regularizer tolerance is a ground distance: the same building must square
    the same way whatever detail the run was tiled at. So the dial is
    ``review.regularize.tolerance_m`` and the pixel size only sets a floor.

    Resolution, in order:

    1. the server's ``tolerance_m``;
    2. else, when the server carries only the older pixel-anchored
       ``tolerance_mult``, that multiplier times the detection pixel, so a
       server that predates the metre dial keeps its tuning;
    3. else the single generic client fallback.

    Then two bounds, both needed because one run mixes object sizes:

    - CEILING at ``max_object_fraction`` of ``object_size_m`` (the object's
      narrow ground dimension), so a tolerance never approaches the size of the
      thing it is squaring;
    - FLOOR at the detection pixel, applied last: under one pixel the snap has
      nothing to move, and the ceiling would otherwise send small objects to a
      tolerance no shape can use.

    ``pixel_size_m`` and ``object_size_m`` are GROUND METRES, so a caller
    working in a CRS whose unit is not the metre converts first. Pass
    ``object_size_m=0`` when the size is unknown: the ceiling is then skipped.
    """
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
    """The de-staircase simplify tolerance, in GROUND METRES.

    A detection mask outline is a pixel staircase: every step is already a
    right angle, so snapping it to right angles alone changes nothing. The
    outline is simplified first, and how hard decides whether a wall arrives at
    the snap as one edge or as twenty.

    Resolution, in order:

    1. the server's ``review.regularize.destair_m``, a ground distance, so one
       building de-staircases the same way whatever detail it was tiled at;
    2. else ``destair_mult`` times the detection pixel, the pixel-anchored form
       every shipped client uses today;
    3. else the generic client multiple of the pixel.

    Returns 0.0 when the pixel size is unusable and no ground dial is set: the
    caller then skips the pass rather than guessing a distance.
    """
    reg = regularize_policy(policy)
    pixel_m = _positive_number(pixel_size_m) or 0.0

    ground = _positive_number(reg.get("destair_m"))
    if ground is not None:
        return ground
    mult = _positive_number(reg.get("destair_mult")) or _DESTAIR_FALLBACK_MULT
    return mult * pixel_m


def regularize_envelope(policy: dict | None = None) -> RegularizePolicy:
    """The extra safety guards layered on the footprint regularizer, read from
    ``review.regularize.envelope`` (a dict) and returned as a
    building_regularizer.RegularizePolicy.

    OFF/neutral by default: when the key is absent (an old server, or a server
    that has not tuned these), every field returns its neutral default, so the
    regularizer reproduces today's IoU-only geometry bit-for-bit. Each field is
    read only when present and valid, else it stays neutral:

        envelope_enabled (bool): master switch for the candidate/reject envelope
        max_area_ratio / min_area_ratio (float > 0): reject when the regularized
            area / original area leaves this band (0 = off)
        max_hausdorff_mult (float > 0): reject when the boundary moved more than
            this times the snap tolerance (0 = off); catches thin spikes the IoU
            guard misses
        max_vertex_growth (float > 0): reject when vertices grew past this factor
        enforce_component_count / enforce_hole_count (bool): reject a candidate
            that changed the part count or hole count (weld / hole loss)
        eligibility_enabled (bool): master switch for the eligibility gate
        min_rectangularity (float in (0,1]): decline shapes below this
            area / convex-hull-area (curved blobs)
        max_holes (int >= 0): decline shapes with more holes than this
        rectangle_enabled (bool): master switch for OMBB substitution
        rectangle_area_fill (float in (0,1]) / rectangle_min_aspect (float > 0):
            substitute the oriented bounding box when a hole-free part fills at
            least this share of its OMBB and is at least this elongated

    The client fallback for every field is ONE neutral value, never a mirror of
    a tuned table.
    """
    from .building_regularizer import RegularizePolicy

    # Contractually non-raising: this is called per-object in the review's live
    # paint loop, so any parse failure must fall back to the neutral policy
    # (today's geometry), never break the repaint.
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
    except Exception:  # noqa: BLE001 -- policy parse is best-effort  # nosec B110
        return RegularizePolicy()


def manual_simplify_multiple_of_px(policy: dict | None = None) -> float:
    """A multiple of the mask's returned pixel size for the Manual polygonize
    simplify tolerance, read from ``review.regularize.manual_simplify_mult``.
    Fallback 0.0 (today's behaviour, no extra simplify). Clamped to >= 0."""
    v = regularize_policy(policy).get("manual_simplify_mult")
    if _is_finite_policy_value(v) and v >= 0:
        return float(v)
    return 0.0


def progressive_merge_enabled(policy: dict | None = None) -> bool:
    """The Manual FocalClick locality flag, read from
    ``review.regularize.progressive_merge_enabled``. OFF by default: a missing
    or non-true value keeps it off."""
    return regularize_policy(policy).get("progressive_merge_enabled") is True


def despike_tolerance_m(pixel_size_m: float, policy: dict | None = None) -> float:
    """The spike-cut opening distance, in GROUND METRES (0.0 = OFF).

    A raw detection mask can carry a thin spike or a neck joining a stray blob
    (a tile-seam merge, an uncertain point). Squaring turns such a spike into a
    rotated diamond, so it is cut first by a morphological opening of this
    radius (shrink then grow, mitre joins, keep the largest part). The dial is
    ``review.regularize.despike_m``, a ground distance so one building is
    cleaned the same way whatever detail it was tiled at. No pixel-anchored
    fallback: unset means 0.0 (OFF, today's behaviour). ``pixel_size_m`` is
    accepted for signature parity with the other tolerances and to let a future
    server tie the floor to the pixel; it is not used by the fallback.
    """
    reg = regularize_policy(policy)
    ground = _positive_number(reg.get("despike_m"))
    return ground if ground is not None else 0.0


def regularize_enabled_for(prompt: str, policy: dict | None = None) -> bool:
    """Whether a run's words suggest OPENING with Right angles already ticked:
    True when a regularize keyword hits the prompt as a whole word (the shared
    taxonomy rule, so "building" never fires inside an unrelated word). An
    empty prompt suggests nothing, so it opens unticked.

    This picks the DEFAULT state of the checkbox, never which engine runs
    behind it. The regularizer answers the tick, whatever the words were: a run
    with drawn examples and no prompt, or one saying "hangar", still squares
    its buildings once the user ticks the box.
    """
    text = normalize_prompt(prompt)
    if not text:
        return False
    keywords = regularize_settings(policy)["keywords"]
    return any(keyword_matches(text, kw) for kw in keywords)


def adaptive_confidence_policy(policy: dict | None = None) -> dict:
    """The review.adaptive_confidence sub-policy (data-driven starting-cutoff
    tuning). Empty dict when absent: the client's generic constants apply."""
    val = review_policy(policy).get("adaptive_confidence")
    return val if isinstance(val, dict) else {}


def semantic_rescue_policy(policy: dict | None = None) -> dict:
    """The review.semantic_rescue sub-policy (zero-instance coverage rescue for
    map-like prompts). Empty dict when absent; the feature is fail-closed, so an
    empty policy keeps it off."""
    val = review_policy(policy).get("semantic_rescue")
    return val if isinstance(val, dict) else {}


def semantic_rescue_enabled(policy: dict | None = None) -> bool:
    """Whether the semantic zero-instance rescue is turned on. Fail-CLOSED by
    design: a missing or non-true ``review.semantic_rescue.enabled`` keeps the
    rescue off, so the behaviour changes only once the server opts in."""
    return semantic_rescue_policy(policy).get("enabled") is True


def semantic_rescue_coverage_floor(policy: dict | None = None) -> float:
    """Minimum semantic coverage a zero-instance tile needs before its coverage
    mask is kept. Read from ``review.semantic_rescue.coverage_floor``; the
    fallback is ONE generic, conservative client value (0.45), never a mirror
    of a tuned table. Values outside [0, 1] fall open to the generic default."""
    val = semantic_rescue_policy(policy).get("coverage_floor")
    if _is_finite_policy_value(val) and 0.0 <= val <= 1.0:
        return float(val)
    return 0.45


def gate_policy(policy: dict | None = None) -> dict:
    """The top-level ``gate`` sub-policy (empty-tile scan gate). Empty dict
    when absent; the gate is fail-CLOSED, so an empty policy keeps it off and
    every run behaves exactly as today."""
    src = get_detection_policy() if policy is None else policy
    val = src.get("gate") if isinstance(src, dict) else None
    return val if isinstance(val, dict) else {}


def gate_enabled(policy: dict | None = None) -> bool:
    """Whether the empty-tile scan gate may run at all. Fail-CLOSED: a missing
    or non-true ``gate.enabled`` keeps the gate off, so behaviour changes only
    once the server opts in."""
    return gate_policy(policy).get("enabled") is True


def gate_group(policy: dict | None = None) -> int:
    """Scan block side: how many tiles per row/column share one packed scan
    image (2 = 2x2, one scan covers up to 4 tiles). Only 2..4 are sane (1
    cannot save anything, beyond 4 the per-tile scan resolution collapses);
    anything else falls back to the generic 2."""
    val = gate_policy(policy).get("group")
    if _is_finite_policy_value(val) and 2 <= val <= 4:
        return int(val)
    return 2


def gate_max_group(policy: dict | None = None) -> int:
    """Ceiling on the ADAPTIVE scan block side. On fine-resolution runs the
    packing deepens up to this side (see scan_gate.scan_group) while the
    per-class resolution cap keeps the packed scan within its validated
    range; a per-class ``max_group`` in the rule overrides this dial. Same
    sane bounds as gate_group; the generic fallback (2) keeps the packing
    fixed until the server raises it."""
    val = gate_policy(policy).get("max_group")
    if _is_finite_policy_value(val) and 2 <= val <= 4:
        return int(val)
    return 2


def gate_min_pixels(policy: dict | None = None) -> int:
    """Mask pixels (at the reference scan canvas) a detection needs inside a
    quadrant to count as evidence for that tile. One generic client value."""
    val = gate_policy(policy).get("min_pixels")
    if _is_finite_policy_value(val) and val >= 1:
        return int(val)
    return 8


def gate_min_tiles(policy: dict | None = None) -> int:
    """Smallest run (tiles) worth a scan phase; below it the fixed scan
    latency outweighs the possible savings. One generic client value."""
    val = gate_policy(policy).get("min_tiles")
    if _is_finite_policy_value(val) and val >= 4:
        return int(val)
    return 12


def gate_prefilter_policy(policy: dict | None = None) -> dict:
    """The ``gate.prefilter`` sub-policy (client-side degenerate-tile check).
    Nested under ``gate`` but INDEPENDENT of ``gate.enabled``: the prefilter
    settles only provably-empty tiles (all no-data or per-band uniform), so
    unlike the model scan it is safe on every run and fails OPEN. Empty dict
    when absent (the generic client defaults apply)."""
    val = gate_policy(policy).get("prefilter")
    return val if isinstance(val, dict) else {}


def gate_prefilter_enabled(policy: dict | None = None) -> bool:
    """Whether the degenerate-tile prefilter runs. Fail-OPEN by design: only
    an explicit ``gate.prefilter.enabled: false`` turns it off (the check can
    settle only tiles that provably contain no object, so running it without
    any policy is safe by construction)."""
    return gate_prefilter_policy(policy).get("enabled") is not False


def gate_prefilter_nodata_frac(policy: dict | None = None) -> float:
    """No-data fraction at/above which a rendered tile settles as empty
    without a model pass. The generic client default is 1.0 (only a FULLY
    no-data tile, the provably-safe point). Only fractions in (0, 1] are
    honoured.

    Only a fully no-data tile can be dropped without risk, because no no-data
    fraction below 1.0 separates a barren tile from a productive one. The
    server may loosen it."""
    val = gate_prefilter_policy(policy).get("nodata_frac")
    if _is_finite_policy_value(val) and 0.0 < val <= 1.0:
        return float(val)
    return 1.0


def gate_prefilter_nodata_rgb_eps(fallback: int, policy: dict | None = None) -> int:
    """Max channel value at which a pixel counts as no-data by COLOUR. The
    tile render paints an opaque black background, so ground the layer does not
    cover arrives black at full alpha and an alpha-only test never fires at
    all; this is what makes ``nodata_frac`` a live dial. Fallback: the client
    constant (exact black). Negative disables the colour half; values above 24
    are refused, because a wider tolerance starts eating deep shadow and dark
    water, which carry sensor noise a background fill does not."""
    val = gate_prefilter_policy(policy).get("nodata_rgb_eps")
    if _is_finite_policy_value(val) and -1 <= val <= 24:
        return int(val)
    return int(fallback)


def gate_prefilter_min_valid_px(fallback: float, policy: dict | None = None) -> float:
    """Valid (non-no-data) pixel count under which a rendered tile settles as
    empty. Fallback: the client constant, the pipeline's own noise floor
    squared, which is the provable bound. Capped well under a tile so a bad
    policy cannot turn a provable rule into a lossy skip."""
    val = gate_prefilter_policy(policy).get("min_valid_px")
    if _is_finite_policy_value(val) and 0.0 <= val <= 4096.0:
        return float(val)
    return float(fallback)


def gate_prefilter_band_eps(policy: dict | None = None) -> float:
    """Max per-band value spread for a rendered tile to count as uniform
    (degenerate). The generic client default is a small render-noise margin;
    only a narrow sane range is honoured, because a large spread would turn
    the provably-safe uniform rule into a lossy low-variance skip."""
    val = gate_prefilter_policy(policy).get("band_eps")
    if _is_finite_policy_value(val) and 0.0 <= val <= 16.0:
        return float(val)
    return 2.0


def gate_prefilter_config(policy: dict | None = None) -> dict | None:
    """Resolved degenerate-tile prefilter settings for one run, or None when
    the server kill switch turned it off. Read once at worker construction so
    the thresholds stay constant for the whole run."""
    if not gate_prefilter_enabled(policy):
        return None
    from .cloud_detection import (  # noqa: PLC0415 - avoids an import cycle
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
    """The ``gate.blank`` sub-policy (single-colour tile skip). A sibling of
    ``gate.prefilter`` and, like it, INDEPENDENT of ``gate.enabled``: the check
    settles tiles that render as essentially one colour, so it is safe on every
    run. Empty dict when absent (the generic client defaults apply)."""
    val = gate_policy(policy).get("blank")
    return val if isinstance(val, dict) else {}


def blank_dominant_frac(fallback: float, policy: dict | None = None) -> float:
    """Share of a sampled tile one quantized colour bucket must cover for the
    tile to be skipped before submit (and never billed). It trades free-trial
    protection against over-culling, so the server can tune it. Fallback:
    client constant; only a fraction in (0, 1] is honoured."""
    val = gate_blank_policy(policy).get("dominant_frac")
    if _is_finite_policy_value(val) and 0.0 < val <= 1.0:
        return float(val)
    return fallback


def blank_quant(fallback: int, policy: dict | None = None) -> int:
    """Colour quantization step of the dominant-bucket test: near-identical
    values collapse into one bucket while a real scene still spreads across
    several. Fallback: client constant; must stay positive."""
    val = gate_blank_policy(policy).get("quant")
    if _is_finite_policy_value(val) and val > 0:
        return int(val)
    return fallback


def blank_sample_px(fallback: int, policy: dict | None = None) -> int:
    """Side (px) the tile is downsampled to before the dominant-bucket test.
    Fallback: client constant; must stay positive."""
    val = gate_blank_policy(policy).get("sample_px")
    if _is_finite_policy_value(val) and val > 0:
        return int(val)
    return fallback


def gate_unavailable_policy(policy: dict | None = None) -> dict:
    """The ``gate.unavailable`` sub-policy: how a tile rendered from an ONLINE
    source is recognised as that source's "no image here" placeholder card
    instead of ground. A sibling of ``gate.blank``, independent of
    ``gate.enabled``. Empty dict when absent (the generic client defaults
    apply). Placeholder cards differ per provider, so this is exactly the kind
    of dial that has to be retunable without a plugin release."""
    val = gate_policy(policy).get("unavailable")
    return val if isinstance(val, dict) else {}


def unavailable_neutral_eps(fallback: int, policy: dict | None = None) -> int:
    """Max spread between a pixel's three channels for it to count as neutral
    grey. Fallback: client constant; must stay non-negative."""
    val = gate_unavailable_policy(policy).get("neutral_eps")
    if _is_finite_policy_value(val) and val >= 0:
        return int(val)
    return fallback


def unavailable_neutral_frac(fallback: float, policy: dict | None = None) -> float:
    """Share of a sampled tile that must be neutral grey before the tile can be
    read as a placeholder card. Fallback: client constant; only a fraction in
    (0, 1] is honoured."""
    val = gate_unavailable_policy(policy).get("neutral_frac")
    if _is_finite_policy_value(val) and 0.0 < val <= 1.0:
        return float(val)
    return fallback


def unavailable_dominant_frac(fallback: float, policy: dict | None = None) -> float:
    """Share one quantized colour bucket must cover for a neutral tile to be
    read as a placeholder card. Looser than the blank test's own fraction,
    because the card carries a line of text. Fallback: client constant; only a
    fraction in (0, 1] is honoured."""
    val = gate_unavailable_policy(policy).get("dominant_frac")
    if _is_finite_policy_value(val) and 0.0 < val <= 1.0:
        return float(val)
    return fallback


def gate_class_for_prompt(
    prompt: str, fallback_class: str, policy: dict | None = None
) -> str:
    """The gate's OWN class for a prompt, decoupled from the review shape
    classes.

    ``gate.class_map`` is an ordered list of ``{"gate_class": <name>,
    "keywords": [<keyword>, ...]}`` entries; the first entry with a keyword
    that is a word-boundary prefix of the prompt wins (the same matching style
    as the seed tiers). The review shape classes group prompts by their
    GEOMETRY, which can force two classes with very different scan behaviour
    to share one gate entry; this map lets the server key gate rules on its
    own taxonomy. ``fallback_class`` (the caller passes the review shape
    class) applies when the map is absent, malformed, or matches nothing, so
    a partial map extends the shape-class table instead of replacing it."""
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
    """The gate rule for one gate class, or None (gate OFF for it).

    ``gate.classes`` maps gate classes (from gate_class_for_prompt; the review
    shape classes remain valid keys via its fallback) to ``{"min_score":
    (0..1], "max_scan_mupp": ground m/px cap or absent, "max_group": optional
    per-class adaptive-packing ceiling}``: the score below which a scanned
    quadrant counts as empty, the coarsest packed scan resolution at which
    skipping that class stays proven-safe, and an optional override of the
    top-level gate_max_group dial. The client ships NO class table: without a
    server entry the run is not gated (fail-closed), so a malformed entry can
    only keep today's behaviour, never skip unsafely."""
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


def fp_filter_policy(policy: dict | None = None) -> dict:
    """The review.fp_filter sub-policy: a per-shape-class geometry-attribute
    false-positive rule table. Empty dict when absent; the filter is
    fail-CLOSED (empty = OFF), so nothing is dropped until the server ships a
    table."""
    val = review_policy(policy).get("fp_filter")
    return val if isinstance(val, dict) else {}


def fp_rules(shape_class: str, policy: dict | None = None) -> list[dict]:
    """Validated geometry-attribute FP rules for one shape class, or [] (OFF).

    Each rule is ``{"attr": <known attr>, "op": <known op>, "value": <number>,
    "action": "drop"}``. A rule with an unknown attr/op, a non-numeric value or
    an unsupported action is dropped defensively, so a malformed server entry can
    only turn rules OFF, never misfire. The client ships NO thresholds: without a
    server table this is always []."""
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


def exemplar_policy(policy: dict | None = None) -> dict:
    """The exemplar sub-policy (example-crop framing)."""
    policy = get_detection_policy() if policy is None else policy
    exemplar = policy.get("exemplar") if isinstance(policy, dict) else None
    return exemplar if isinstance(exemplar, dict) else {}


def exemplar_context_pad(policy: dict | None = None) -> float:
    """Fractional margin around a drawn example crop (0.05 = 5%). Thin on
    purpose: the example is traced as a polygon and cropped from its bounding
    box, so an irregular shape already brings its surroundings along, and a
    wider ring hands the model the neighbours instead of the object. Never 0:
    the boundary needs some outside pixels to sit against."""
    val = exemplar_policy(policy).get("context_pad")
    if _is_finite_policy_value(val):
        return float(val)
    return 0.05


def exemplar_context_pad_px_cap(policy: dict | None = None) -> float:
    """Absolute ceiling on the example-crop margin, in RUN-scale pixels. The
    fractional pad grows with the drawn object, so on a large object it drags
    whole neighbouring structures into the crop; this cap keeps the margin a
    thin context ring at any object size."""
    val = exemplar_policy(policy).get("context_pad_px")
    if _is_finite_policy_value(val) and val > 0:
        return float(val)
    return 12.0


def exemplar_min_paste_scale(policy: dict | None = None) -> float:
    """Minimum fraction of an example crop's run-scale size at which pasting
    it into tiles is still worthwhile. Below this, the pasted copy shows the
    object smaller than the tile's real objects (the model matches by
    apparent scale), so the paste is skipped (the example still points at its
    real in-situ object where a tile fully contains it)."""
    val = exemplar_policy(policy).get("min_paste_scale")
    if _is_finite_policy_value(val) and 0 < val <= 1:
        return float(val)
    return 0.85


def _exemplar_positive_int(key: str, fallback: int, policy: dict | None) -> int:
    """A positive integer exemplar dial, or the caller's fallback constant."""
    val = exemplar_policy(policy).get(key)
    if _is_finite_policy_value(val) and val >= 1:
        return int(val)
    return fallback


def exemplar_stamp_max_px(fallback: int, policy: dict | None = None) -> int:
    """Longest-side pixel cap for one pasted example crop. The strongest lever
    on how much of an example the model actually sees, so it sits with the rest
    of the framing dials. Fallback: client constant; must stay positive."""
    return _exemplar_positive_int("stamp_max_px", fallback, policy)


def exemplar_stamp_pad_px(fallback: int, policy: dict | None = None) -> int:
    """Gap (px) around each pasted example crop. Fallback: client constant."""
    return _exemplar_positive_int("stamp_pad_px", fallback, policy)


def exemplar_max_positive(fallback: int, policy: dict | None = None) -> int:
    """How many positive examples one run may carry. Fallback: client
    constant; must stay at least 1."""
    return _exemplar_positive_int("max_positive", fallback, policy)


def exemplar_max_exclude(fallback: int, policy: dict | None = None) -> int:
    """How many exclude examples one run may carry. Fallback: client
    constant; must stay at least 1."""
    return _exemplar_positive_int("max_exclude", fallback, policy)


def exemplar_max_region(fallback: int, policy: dict | None = None) -> int:
    """How many region markers one run may carry. Fallback: client constant;
    must stay at least 1."""
    return _exemplar_positive_int("max_region", fallback, policy)


def exemplar_min_example_positives(fallback: int, policy: dict | None = None) -> int:
    """How many positives the exclude example waits for before it is offered.

    Same block as the ceilings above so the gate and the store move together.
    Fallback: client constant; must stay at least 1."""
    return _exemplar_positive_int("min_example_positives", fallback, policy)


def exemplar_min_meta_positives(fallback: int, policy: dict | None = None) -> int:
    """How many positives the recommended prompt-plus-example combination asks
    for. Fallback: client constant; must stay at least 1."""
    return _exemplar_positive_int("min_meta_positives", fallback, policy)


def exemplar_render_min_side_px(fallback: int, policy: dict | None = None) -> int:
    """Floor on the longest side (px) a drawn example is RENDERED at, before it
    is pasted into a tile. The model matches by apparent scale, so this bounds
    how well a drawn example detects anything, the same way the paste dials do.
    Fallback: client constant; must stay positive."""
    return _exemplar_positive_int("render_min_side_px", fallback, policy)


def exemplar_render_max_side_px(fallback: int, policy: dict | None = None) -> int:
    """Ceiling on the longest side (px) a drawn example is RENDERED at. It has
    to stay small enough to fit the tile corner it is pasted into. Fallback:
    client constant; must stay positive."""
    return _exemplar_positive_int("render_max_side_px", fallback, policy)


def exemplar_render_side_bounds(min_fallback: int, max_fallback: int,
                                policy: dict | None = None) -> tuple[int, int]:
    """The (min, max) render sides in px, resolved together.

    A floor served above the ceiling leaves the render size nowhere to land, so
    BOTH go back to the client constants rather than clamping to a size nobody
    chose. Resolve the pair here and nowhere else, so every caller sees the same
    answer."""
    low = exemplar_render_min_side_px(min_fallback, policy)
    high = exemplar_render_max_side_px(max_fallback, policy)
    if low > high:
        return int(min_fallback), int(max_fallback)
    return low, high


def exemplar_render_abs_min_side_px(fallback: int, policy: dict | None = None) -> int:
    """Hard pixel floor for a rendered example: below it the crop carries no
    usable texture, whatever the run scale asked for. Also the size under which
    the dock warns that a drawn example is too small. Fallback: client constant;
    must stay positive."""
    return _exemplar_positive_int("render_abs_min_side_px", fallback, policy)


def exemplar_render_fallback_gsd_m(fallback: float, policy: dict | None = None) -> float:
    """Ground resolution (m/px) to render a drawn example at when neither the
    source's own resolution nor the run's is known. Fallback: client constant;
    must stay positive."""
    val = exemplar_policy(policy).get("render_fallback_gsd_m")
    if _is_finite_policy_value(val) and val > 0:
        return float(val)
    return float(fallback)


def prompt_policy(policy: dict | None = None) -> dict:
    """The prompt sub-policy (guard-rail word sets and the offline lexicon).

    The tuned tables live server-side; without a policy this is an empty dict
    and the prompt guard degrades to its generic English fallbacks."""
    policy = get_detection_policy() if policy is None else policy
    prompt = policy.get("prompt") if isinstance(policy, dict) else None
    return prompt if isinstance(prompt, dict) else {}


def prompt_hint_for(
    prompt: str, policy: dict | None = None
) -> tuple[str, str] | None:
    """Server advisory for one prompt, as ``(hint_id, sentence)``, or None.

    ``prompt.hints`` is a list of ``{"keywords": [...], "hint": "..."}``. The
    LONGEST matching keyword wins, so a phrase beats another entry's component
    word and table order never decides a tie.

    The id is derived from the entry's first keyword rather than shared by every
    served hint, so a user dismissing the road advisory does not also dismiss a
    future one about dams, and the server can suppress a single hint through
    ``guidance.suppressed``. An id this build has never seen is exactly what
    ``guidance.extra`` is for.

    Served copy arrives already in the user's language, so it carries no
    ``tr()``. There is no shipped fallback table: with no policy the prompt box
    simply shows nothing, which is what it did before this existed.
    """
    text = normalize_prompt(prompt)
    if not text:
        return None
    hints = prompt_policy(policy).get("hints")
    if not isinstance(hints, list):
        return None
    usable = [
        h for h in hints
        if isinstance(h, dict)
        and isinstance(h.get("hint"), str) and h["hint"].strip()
        and next(iter_keywords(h), None) is not None
    ]
    entry = longest_keyword_match(text, usable)
    if entry is None:
        return None
    first = next(iter_keywords(entry))
    slug = "".join(c if c.isalnum() else "_" for c in first.lower()).strip("_")
    return f"prompt_hint_{slug}", entry["hint"].strip()
