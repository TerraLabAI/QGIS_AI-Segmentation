"""Single source of truth for the Automatic-review defaults.

The dock (`ai_segmentation_dockwidget.py`) sets the review spin/slider start
values from these; the plugin controller (`ai_segmentation_plugin.py`) uses the
same numbers as fallbacks when it reads the review settings back. Both used to
carry private copies with "kept in sync" comments; the copies drifted risk is
gone now that they import from here (core layer, so both UI files can). Grep-
proof: each constant is defined exactly once, right here.
"""

from __future__ import annotations

# Cloud-model detection-confidence: the UI's default cutoff, AND the value the plugin
# uses as the recall-leaning starting point for the post-run review slider. 0.30
# is recall-leaning for aerial imagery (users prefer deleting a few false
# positives over re-running). The much lower
# recall FLOOR actually sent to the server (`_RECALL_FLOOR`) is plugin-private:
# every plausible mask comes back so the slider re-filters client-side, free.
AUTO_DEFAULT_CONFIDENCE = 0.30

# Exemplar-ONLY runs (a drawn example, no text prompt) open both the live
# preview and the post-run review at a HIGHER starting cutoff than a text run.
# Without an open-vocabulary text prior the model surfaces more weak
# look-alikes, so a higher start hides that tier by default (the slider still
# recovers them). Kept above AUTO_DEFAULT_CONFIDENCE so raising it never makes
# the review open below the text default. The real tuned value is
# server-delivered; this is only the ONE generic client fallback, never a
# mirror of a tuned table.
AUTO_DEFAULT_CONFIDENCE_EXEMPLAR_ONLY = 0.45

# The review's "px" unit is DYNAMIC: 1 px = one pixel of the run's returned
# masks (worker-observed, `_auto_refine_pixel_size`), NOT a fixed ground size
# and NOT the source raster's native pixel. So the same default is gentle on a
# close-up single-tile run (tiny ground tolerance, detail kept) and strong
# enough on a coarse wide run (tolerance grows with the staircase step).

# Simplify (px): a gentle 0.4 so detections keep their TRUE detail out of the
# box. No longer a visible control (see AUTO_REVIEW_POINTS_PCT_DEFAULT); it stays
# as the small de-noise pass the class presets set.
# Sub-pixel default, so its spinbox must be a QDoubleSpinBox.
AUTO_REVIEW_SIMPLIFY_DEFAULT = 0.4

# Points (%): the share of an outline's own points to keep, which is the visible
# vertex control. 100 = the density the class preset already produces, lower
# thins further. It drives the point budget (core.vertex_budget), NOT a
# distance tolerance: a tolerance decides how far the boundary may move and
# only reduces the count as a side effect, which is why turning one up eats a
# building instead of thinning it.
AUTO_REVIEW_POINTS_PCT_DEFAULT = 100

# Clean edges (morphological opening, px): 0.5 by default so detections come out
# clean; lower to 0 for faithful raw outlines, raise to strip thicker attached
# fringe the cloud model leaves around objects. Acts on the object's OWN outline (unlike Min
# size, which drops separate small features). 0 = off, adjustable live.
AUTO_REVIEW_CLEAN_DEFAULT = 0.5

# Round corners (QgsGeometry.smooth): off by default (faithful angular outlines).
AUTO_REVIEW_SMOOTH_DEFAULT = False

# Right angles (QgsGeometry.orthogonalize): off by default (faithful outlines).
# Opt-in regularizer for man-made shapes (buildings, pools, panels): snaps every
# edge to 90 degrees so footprints read as hand-digitized. Deliberately
# opinionated (it moves geometry toward what the shape "should" be), hence
# never on by default; the user checks it when the objects warrant it.
AUTO_REVIEW_ORTHO_DEFAULT = False

# Expand/shrink (px): 0 by default (no buffer).
AUTO_REVIEW_EXPAND_DEFAULT = 0

# Widest interior hole (metres across) that is mask noise rather than part of
# the object: compression pits, staircase specks, rooftop texture. THE single
# source for that judgement, because two code paths make it and they must agree:
# the cloud path fills up to here on every tile before anything is polygonized
# (cloud_detection.pinhole_fill_limit_px reads this), and Manual has no such
# pass, so its own control starts here instead (see the Manual defaults below).
# Squared it is a hole AREA, which is the unit both controls speak.
HOLE_NOISE_CEILING_M = 2.5
HOLE_NOISE_CEILING_M2 = HOLE_NOISE_CEILING_M * HOLE_NOISE_CEILING_M

# Fill interior holes: ON for every run, with no per-class opt-out. The per-tile
# pass above only reaches HOLE_NOISE_CEILING_M2, and what survives it is not
# structure: a field comes back peppered with bare-soil gaps, a forest with
# canopy pinholes, a building with rooftop clutter. Leaving that to the classes
# meant every class that says "keep my holes" (lake, forest, building) opened on
# a shredded object. So the tick is on and the SIZE decides instead, which is
# what a hole being noise ever depended on.
AUTO_REVIEW_FILL_HOLES_DEFAULT = True

# Fill holes SMALLER than this ground area (m2); bigger holes stay holes.
# The ONE generic client value, never a mirror of the server's per-class table
# (a road wants car-sized holes gone and its median kept: that is a server
# dial). It plays two roles, and they are the same number on purpose: the
# ceiling a run starts at when nothing better is served, and the FLOOR under any
# served ceiling, so a class tuned to keep its holes still gets the pepper
# filled. Sits well above HOLE_NOISE_CEILING_M2 (under it the per-tile pass has
# already filled everything and this would be a no-op) and well under a
# courtyard, a canopy clearing, a farm pond or an island, which stay holes at
# any prompt. About 7 m across.
AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT = 50.0


def _fill_holes_floor_m2() -> float:
    """The floor every run fills to, server-overridable.

    ``review.fill_holes_floor_m2`` retunes it fleet-wide with no plugin
    release; the module constant above is the client fallback."""
    try:
        from .detection_policy import fill_holes_floor_m2

        return fill_holes_floor_m2(AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT)
    except Exception:  # noqa: BLE001 -- policy is best-effort  # nosec B110
        return AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT


def fill_holes_max_m2_with_floor(class_fills: object, class_ceiling: object) -> float:
    """The ground area (m2) under which interior holes are filled, given what a
    class (or a server run plan) asked for. 0 = fill every hole, at any size.

    Every run fills at least up to the floor (AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT
    unless the server moved it); above it the server value decides, and an
    explicit 0 still means "no ceiling, fill everything". A class that fills
    nothing gets the floor, which is the whole point of there being one. A
    missing or malformed value reads as "no opinion", so it gets the floor too:
    it must never read as 0, which means fill EVERYTHING to the filler and is
    the opposite of a safe fallback.
    """
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


# ---------------------------------------------------------------------------
# Min size: the resolution noise floor
#
# A detection narrower than this many RETURNED-MASK pixels is mask noise by
# construction, whatever the prompt, so Min size never opens below its ground
# area. Squared with the mask ground pixel it is an area, the unit the control
# speaks. Two callers compute it (review_presets for the client-side preset, the
# plugin's run-plan path for a server preset) and they must agree.
# ---------------------------------------------------------------------------
MIN_SIZE_NOISE_MASK_PX = 3.0

# The same floor for a run with NO text prompt, where nothing else holds the
# line: no prompt means no server run plan, so no per-object floor, and the
# generic 3 px lets through every speck the model emits. Widened rather than
# replaced by a fixed area, because noise scales with the mask pixel and a
# ground number would be wrong at the next resolution. Raising it hides objects,
# it never deletes them: the review's Min size spinbox brings them straight back.
MIN_SIZE_NOISE_MASK_PX_NO_PROMPT = 5.0


def min_size_noise_floor_m2(mask_gsd_m: float, *, no_prompt: bool = False) -> float:
    """Ground area (m2) under which a detection is mask noise at this
    resolution. 0 when the resolution is unknown (no floor rather than a guess).

    The pixel width is server-overridable (``review.min_size_noise_px``), so
    the floor that decides what a run even shows can be retuned fleet-wide;
    the module constants are the client fallbacks.
    """
    if not mask_gsd_m or mask_gsd_m <= 0:
        return 0.0
    px = MIN_SIZE_NOISE_MASK_PX_NO_PROMPT if no_prompt else MIN_SIZE_NOISE_MASK_PX
    try:
        from .detection_policy import min_size_noise_px

        px = min_size_noise_px(no_prompt, px)
    except Exception:  # noqa: BLE001 -- policy is best-effort  # nosec B110
        pass
    return (px * float(mask_gsd_m)) ** 2


# ---------------------------------------------------------------------------
# Manual refine-panel defaults ("Refine selection" in base Manual, "Shape
# settings" in a Refine-in-Manual handoff). Used to be hardcoded independently
# at 5 call sites (plugin __init__, session reset, saved-polygon-restore
# fallback, and the dock's widget setup + slider reset); a drift between them
# silently changed what a fresh session or a restored polygon lacking the
# field started with. Grep-proof: defined once here.
# ---------------------------------------------------------------------------
# Simplify (px): the SAME unit and the same gentle start as the Automatic
# review, so a number typed in one mode means the same thing in the other. It
# used to be an integer on a half-pixel scale, which could not reach a sub-pixel
# tolerance at all and opened every Manual session at nearly four times this.
# The point budget below is what thins a Manual outline now; this only de-noises.
REFINE_SIMPLIFY_DEFAULT = AUTO_REVIEW_SIMPLIFY_DEFAULT
REFINE_SMOOTH_DEFAULT = 0
# Round corners: how many Chaikin passes the tick means. The Automatic review
# runs ONE. Each pass roughly doubles the point count, so the 5 Manual used to
# run left an outline about thirty times denser than the review's for the same
# object, and undid the point budget that had just thinned it.
REFINE_SMOOTH_ITERATIONS = 1
# Points (%): the share of its own points the outline keeps, the Manual twin of
# AUTO_REVIEW_POINTS_PCT_DEFAULT. Manual always ran the point budget; it just had
# no dial on it, so the class density was the only budget a user could get.
REFINE_POINTS_PCT_DEFAULT = AUTO_REVIEW_POINTS_PCT_DEFAULT
# Clean edges (morphological opening, px): OFF by default. A local Manual mask
# comes out clean already, so it is opt-in here (the Automatic review, whose
# cloud masks carry more attached fringe, starts it at AUTO_REVIEW_CLEAN_DEFAULT
# instead). 0 = off, raise to strip thin attached fringe. Manual twin of
# AUTO_REVIEW_CLEAN_DEFAULT.
REFINE_CLEAN_DEFAULT = 0.0
REFINE_EXPAND_DEFAULT = 0
REFINE_FILL_HOLES_DEFAULT = True
# Manual's own hole ceiling (ground m2, 0 = fill every hole). A Refine-in-Manual
# handoff seeds it from the review's value.
#
# Deliberately NOT the Automatic number: Manual clicks one object at a time and
# the user sees the result immediately, so it stays at the noise ceiling and
# leaves everything above it alone. Automatic reviews hundreds of objects at
# once and nobody inspects each, which is why it fills further.
#
# Starts AT the noise ceiling, not at 0. Manual runs no separate noise pass: the
# tick and this number are the whole decision, so a 0 here filled every hole at
# any size and a clicked courtyard came out solid. At the ceiling, Manual now
# does what the cloud path does on every tile: fill the pits, keep the structure.
REFINE_FILL_HOLES_MAX_M2_DEFAULT = HOLE_NOISE_CEILING_M2
REFINE_ORTHO_DEFAULT = False
REFINE_MIN_SIZE_M2_DEFAULT = 0.0
REFINE_MAX_SIZE_M2_DEFAULT = 0.0
# Seed for the auto-computed min polygon area (detection px2), before
# _compute_auto_min_area() x 2 replaces it. Not user-facing; named here so the
# two Manual seed sites do not carry their own copy.
REFINE_MIN_AREA_DEFAULT = 200
# The Manual panel's opening state is NOT server-tunable, on purpose. Manual is
# the offline mode: what a fresh session starts with is decided here and stays
# the same with no account, no network and no configuration fetch. Only the
# Automatic review's own floors and presets are served.

# ---------------------------------------------------------------------------
# The visible-set gate
#
# Which whole objects the user sees, by confidence and by ground size. Two
# threads ask it: the live stitcher while a run streams, and the review when it
# reslices. They must answer the same, or the preview shows a different set than
# the review opens on, so the decision lives here once instead of once per
# caller. Pure: dict in, bool out, no Qt, safe off the GUI thread.
# ---------------------------------------------------------------------------


def area_passes_size_gates(area: float, params: dict) -> bool:
    """The min/max ground-size half of the visible-set gate. 0 means off.

    Objects the user drew by hand bypass the confidence gate but never this one.
    """
    min_a = params.get("min_a", 0.0)
    max_a = params.get("max_a", 0.0)
    if min_a > 0 and area < min_a:
        return False
    return not (max_a > 0 and area > max_a)


def object_passes_review_gates(score: float, area: float, params: dict) -> bool:
    """Whole-object confidence plus min/max-size gate.

    ``score`` is the object's representative score (the max over the fragments
    stitched into it), never a fragment's, so a strong object survives even when
    one of its seam halves scored low.
    """
    if score < params.get("conf", 0.0):
        return False
    return area_passes_size_gates(area, params)


# ---------------------------------------------------------------------------
# Adaptive review starting confidence
#
# A fixed default cutoff assumes one score scale across every object class,
# which does not hold. Instead of a per-class table, the run's OWN score
# distribution decides: evidence that the low-score cohort is the same physical
# population as the high-score one (not noise) lowers the starting cutoff. It
# never raises above the default, so well-behaved classes keep the old cutoff.
# The tuned band lives server-side; these are the generic fallbacks.
# ---------------------------------------------------------------------------

# Below this many detections the distribution is too thin to trust.
_ADAPTIVE_MIN_OBJECTS = 30
# Only intervene when the default would hide MORE than this fraction of the
# run's detections (a hidden minority tail is normal and usually noise).
_ADAPTIVE_HIDDEN_TRIGGER = 0.35
# SEPARATE/count mode population-coherence gate: the low-score cohort's median
# footprint must sit within this band of the high-score cohort's median for the
# two to read as one physical population rather than noise.
_ADAPTIVE_AREA_RATIO_BAND = (0.35, 2.8)
# High-score cohort must be at least this big for its median to anchor the
# comparison.
_ADAPTIVE_MIN_ANCHOR = 10
# Never open the review below this cutoff: the bottom score tier next to the
# server recall floor is junk-dominated for every class.
_ADAPTIVE_FLOOR = 0.15


def _adaptive_params() -> tuple[int, float, float, float, int, float]:
    """The adaptive-confidence tuning scalars, server-overridable.

    The server policy's ``review.adaptive_confidence`` block (when present)
    overrides any of them, so this heuristic is re-tunable fleet-wide without
    a plugin release; the module constants above are the client fallbacks."""
    lo, hi = _ADAPTIVE_AREA_RATIO_BAND
    min_objects, hidden_trigger = _ADAPTIVE_MIN_OBJECTS, _ADAPTIVE_HIDDEN_TRIGGER
    min_anchor, floor = _ADAPTIVE_MIN_ANCHOR, _ADAPTIVE_FLOOR
    try:
        from .detection_policy import adaptive_confidence_policy

        pol = adaptive_confidence_policy()

        def _num(key: str, fb: float) -> float:
            v = pol.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
            return fb

        min_objects = int(_num("min_objects", min_objects))
        hidden_trigger = _num("hidden_trigger", hidden_trigger)
        lo = _num("area_ratio_lo", lo)
        hi = _num("area_ratio_hi", hi)
        min_anchor = int(_num("min_anchor", min_anchor))
        floor = _num("floor", floor)
    except Exception:  # noqa: BLE001 -- policy is best-effort  # nosec B110
        pass
    return min_objects, hidden_trigger, lo, hi, min_anchor, floor


def adaptive_review_confidence(
    scored: list[tuple[float, float]],
    default: float = AUTO_DEFAULT_CONFIDENCE,
    merge_separate: bool = True,
) -> float | None:
    """Data-driven starting cutoff for the post-run review, or None to keep
    ``default``.

    ``scored`` is one (score, ground_area) pair per detected object. In
    SEPARATE/count mode the cutoff is lowered only when the hidden (below-
    default) cohort passes the size-coherence gate against the confidently
    detected cohort; in MAP/coverage mode (merged keepers, heterogeneous areas
    by construction) the hidden-mass trigger alone decides. The returned value
    is the hidden cohort's 25th score percentile, floored, snapped down to the
    review slider's 5% grid, and always strictly below ``default``.
    """
    (min_objects, hidden_trigger, ratio_lo, ratio_hi,
     min_anchor, floor) = _adaptive_params()
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
    step = int(cutoff * 100 / 5.0) * 5  # snap DOWN to the slider's 5% grid
    step = min(step, int(round(default * 100)) - 5)
    if step < int(floor * 100):
        return None
    return step / 100.0
