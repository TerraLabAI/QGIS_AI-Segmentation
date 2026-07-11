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
# box; the user opts into more smoothing by raising the visible Simplify spinbox.
# Sub-pixel default, so its spinbox must be a QDoubleSpinBox.
AUTO_REVIEW_SIMPLIFY_DEFAULT = 0.4

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

# Fill interior holes: ON by default. Now that instances come back cleanly
# separated (per-tile NMS + select-duplicates merging), an interior hole in a
# single detection is almost always mask noise, not information, so filling
# reads as "clean" for the general case. The classes where holes ARE real keep
# it off explicitly in review_presets: natural_area (clearings, islands) and
# linear (road medians).
AUTO_REVIEW_FILL_HOLES_DEFAULT = True

# ---------------------------------------------------------------------------
# Manual refine-panel defaults ("Refine selection" in base Manual, "Shape
# settings" in a Refine-in-Manual handoff). Used to be hardcoded independently
# at 5 call sites (plugin __init__, session reset, saved-polygon-restore
# fallback, and the dock's widget setup + slider reset); a drift between them
# silently changed what a fresh session or a restored polygon lacking the
# field started with. Grep-proof: defined once here.
# ---------------------------------------------------------------------------
REFINE_SIMPLIFY_DEFAULT = 3
REFINE_SMOOTH_DEFAULT = 0
REFINE_EXPAND_DEFAULT = 0
REFINE_FILL_HOLES_DEFAULT = True
REFINE_ORTHO_DEFAULT = False
REFINE_MIN_SIZE_M2_DEFAULT = 0.0
REFINE_MAX_SIZE_M2_DEFAULT = 0.0

# ---------------------------------------------------------------------------
# Adaptive review starting confidence
#
# The fixed default cutoff assumes the model's score scale is the same for
# every object class; it is not. On some classes (small natural objects in
# dense scenes) most TRUE detections score under the default, so the review
# opened hiding the majority of what the user paid for. Instead of a per-class
# table, the run's OWN score distribution decides: evidence that the low-score
# cohort is the same physical population as the high-score one (not noise)
# lowers the starting cutoff. Never raises above the default, so classes whose
# scores behave keep exactly the old behaviour.
# ---------------------------------------------------------------------------

# Below this many detections the distribution is too thin to trust.
_ADAPTIVE_MIN_OBJECTS = 30
# Only intervene when the default would hide MORE than this fraction of the
# run's detections (a hidden minority tail is normal and usually noise).
_ADAPTIVE_HIDDEN_TRIGGER = 0.35
# SEPARATE/count mode population-coherence gate: the low-score cohort's median
# footprint must be within this band of the high-score cohort's median for the
# two to read as one physical population. Noise (slivers, texture fragments)
# is far smaller or far larger than the confidently-detected objects.
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
