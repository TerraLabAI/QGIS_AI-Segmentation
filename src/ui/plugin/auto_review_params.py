"""The review's filter and refine parameters: the run's smart preset, the
starting confidence, the params snapshots and the visible-set gates.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from ...core.review_defaults import (
    AUTO_REVIEW_CLEAN_DEFAULT as _AUTO_REVIEW_CLEAN_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_EXPAND_DEFAULT as _AUTO_REVIEW_EXPAND_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_FILL_HOLES_DEFAULT as _AUTO_REVIEW_FILL_HOLES_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_ORTHO_DEFAULT as _AUTO_REVIEW_ORTHO_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_POINTS_PCT_DEFAULT as _AUTO_REVIEW_POINTS_PCT_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_SIMPLIFY_DEFAULT as _AUTO_REVIEW_SIMPLIFY_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_SMOOTH_DEFAULT as _AUTO_REVIEW_SMOOTH_DEFAULT,
)
from ...core.review_defaults import (
    area_passes_size_gates,
    object_passes_review_gates,
)
from ...core.review_defaults import (
    fill_holes_max_m2_with_floor as _fill_holes_max_m2_with_floor,
)
from ...core.review_defaults import (
    min_size_noise_floor_m2 as _min_size_noise_floor_m2,
)


def _plan_vertex_spacing_m(review: dict) -> float:
    """The run plan's point spacing (ground metres per exported vertex), or the
    generic client value when the plan carries none. A plan from a server that
    predates the budget must not silently turn it off, so a missing key reads as
    "no opinion", not as zero."""
    val = review.get("vertex_spacing_m")
    if isinstance(val, (int, float)) and not isinstance(val, bool) and val >= 0:
        return float(val)
    from ...core.detection_policy import vertex_budget_settings

    return float(vertex_budget_settings()["spacing_m"])


class AutoReviewParamsMixin:
    """What the review filters and refines with, before any geometry is touched."""

    def _auto_review_preset(self) -> dict:
        """The run's smart review defaults: prompt-shaped regularizers (Right
        angles/Fill holes/Round corners per object kind) + the resolution-aware
        Min size floor. Recomputed per call from the run context, so every NEW
        run starts from ITS optimum (no cross-run memory by design); the user
        can still override any control in the review."""
        from ...core.review_presets import review_preset_for
        prompt = str((self._auto_run_ctx or {}).get("prompt") or "")
        # Meters per RETURNED-mask pixel: the run's meter GSD scaled by the
        # observed mask/render ratio (the mask grid is what the polygons
        # staircase on, so it is the real noise floor).
        gsd_m = getattr(self, "_auto_gsd_m", 0.0)
        mask_gsd = getattr(self, "_auto_mask_gsd", 0.0)
        if gsd_m > 0 and mask_gsd > 0 and self._auto_gsd > 0:
            gsd_m *= mask_gsd / self._auto_gsd
        # Prefer the server run plan's review block when it was fetched for this
        # run's prompt; else the blob/generic prompt-shaped preset.
        plan = self._active_run_plan(prompt)
        if plan is not None:
            preset = self._review_preset_from_plan(plan.get("review"), gsd_m)
            if preset is not None:
                return preset
        return review_preset_for(prompt, gsd_m)

    def _review_preset_from_plan(self, review: object, gsd_m: float) -> dict | None:
        """Build the review preset dict from a run plan's ``review`` block, or
        None when it is missing/malformed. The plan's ``min_size_m2`` is the
        OBJECT floor only; the client still maxes it with its own resolution
        noise floor, which stays client-side. A plan only ever exists for a
        prompted run, so the floor here is always the prompted one."""
        if not isinstance(review, dict):
            return None
        try:
            noise = _min_size_noise_floor_m2(gsd_m)
            object_floor = float(review.get("min_size_m2") or 0.0)
            return {
                "simplify_px": float(review.get("simplify_px", _AUTO_REVIEW_SIMPLIFY_DEFAULT)),
                "smooth": bool(review.get("smooth", _AUTO_REVIEW_SMOOTH_DEFAULT)),
                "expand_px": int(review.get("expand_px", _AUTO_REVIEW_EXPAND_DEFAULT)),
                # The plan does not get to turn hole filling off either (same
                # reason as the client preset: the classes that ask to keep
                # their holes are the ones that come back shredded). It only
                # decides how far above the client floor the filling goes.
                "fill_holes": _AUTO_REVIEW_FILL_HOLES_DEFAULT,
                "fill_holes_max_m2": _fill_holes_max_m2_with_floor(
                    review.get("fill_holes"), review.get("fill_holes_max_m2")),
                "clean_px": float(review.get("clean_px", _AUTO_REVIEW_CLEAN_DEFAULT)),
                "close_notches_m": float(review.get("close_notches_m", 0.0) or 0.0),
                "ortho": bool(review.get("ortho", _AUTO_REVIEW_ORTHO_DEFAULT)),
                "min_size_m2": round(max(object_floor, noise), 1),
                "vertex_spacing_m": _plan_vertex_spacing_m(review),
                "shape_class": str(review.get("shape_class", "server")),
            }
        except (TypeError, ValueError):
            return None

    def _confidence_default_for(self, prompt: str, is_exemplar_only: bool) -> float:
        """The starting confidence cutoff for one run, from its prompt alone.

        The run plan's value when the plan was fetched for this prompt, else the
        shared resolver (exemplar-only default, the prompt's shape-class value
        from the server policy, or the generic one). Object classes, and an
        exemplar-only run's lack of a text prior, score differently on the same
        true object, so the start adapts instead of one flat cutoff.

        Takes its two inputs as ARGUMENTS rather than reading run state, because
        the run start has to resolve the cutoff before ``_auto_run_ctx`` and
        ``_auto_is_exemplar_only`` exist. That is what lets the live preview and
        the review share one decision instead of each making its own."""
        plan = self._active_run_plan(prompt)
        if plan is not None:
            c = plan.get("confidence_default")
            if isinstance(c, (int, float)) and not isinstance(c, bool):
                return float(c)
        from ...core.review_presets import review_start_confidence_default

        return review_start_confidence_default(prompt, is_exemplar_only)

    def _effective_confidence_default(self) -> float:
        """The review's starting confidence default: the run's own cutoff, read
        back off the finished run's state. The live seed resolves the same value
        through _confidence_default_for at run start, so the live preview and
        the review open at the same cutoff. The auto_lowered comparisons all
        read this so they compare against the SAME default the review seeds
        from."""
        return self._confidence_default_for(
            str((self._auto_run_ctx or {}).get("prompt") or ""),
            bool(getattr(self, "_auto_is_exemplar_only", False)))

    def _fresh_review_params(self) -> dict:
        """Review filter/refine params for a fresh result: the pre-run
        confidence, no max size, and the run's smart preset for the shape
        refine + Min size (see _auto_review_preset). Used at finalize so a
        stale filter from a prior review never touches a fresh run before the
        review widgets are (re)seeded with the same preset."""
        from ...core.boundary_snap import snap_default_enabled
        preset = self._auto_review_preset()
        return {
            # Shared borders is a WHOLE-SET operation, not a per-object one:
            # it rides in the params so it travels with every reslice
            # snapshot, but it is applied once on the assembled visible set
            # (_apply_boundary_snap), never inside the per-object refine.
            "snap_boundaries": snap_default_enabled(),
            "conf": self._auto_confidence,
            "min_a": float(preset["min_size_m2"]),
            "max_a": 0.0,
            "simplify_px": float(preset["simplify_px"]),
            "smooth": bool(preset["smooth"]),
            "expand_px": int(preset["expand_px"]),
            "fill_holes": bool(preset["fill_holes"]),
            "fill_max_m2": float(preset.get("fill_holes_max_m2", 0.0) or 0.0),
            "open_px": float(preset["clean_px"]),
            # Not a widget either: closing an outside bite is a class decision,
            # not something the user dials, so it rides the preset like the
            # point budget does and survives every reslice snapshot.
            "close_notches_m": float(preset.get("close_notches_m", 0.0) or 0.0),
            "ortho": bool(preset["ortho"]),
            # Not a widget: the point budget travels with the run's preset so
            # every reslice snapshot carries it, like Min size does.
            "vertex_spacing_m": float(preset.get("vertex_spacing_m", 0.0) or 0.0),
            # The user's share-of-points dial opens at "keep them all", so a
            # fresh run is the class density alone.
            "points_pct": _AUTO_REVIEW_POINTS_PCT_DEFAULT,
        }

    def _widget_review_params(self) -> dict:
        """Current review filter/refine params read from the dock widgets (a
        reslice snapshot). Confidence comes from _auto_confidence (the confidence
        handler keeps it in sync)."""
        params = self._fresh_review_params()
        d = self.dock_widget
        if d is None:
            return params
        try:
            params["conf"] = self._auto_confidence
            params["min_a"] = d.get_auto_min_size()
            params["max_a"] = d.get_auto_max_size()
            params["fill_max_m2"] = d.get_auto_fill_holes_max()
            params["snap_boundaries"] = d.get_auto_boundary_snap()
            simplify, smooth, expand, fill, clean, ortho = d.get_auto_refine_params()
            params["simplify_px"] = simplify
            params["points_pct"] = d.get_auto_points_pct()
            params["smooth"] = smooth
            params["expand_px"] = expand
            params["fill_holes"] = fill
            params["open_px"] = clean
            params["ortho"] = ortho
        except (RuntimeError, AttributeError):
            pass
        return params

    def _object_is_manual(self, det_idx: int) -> bool:
        """Whether the object at this index is hand-drawn (Add a polygon) or a
        native split piece. Such objects skip the confidence and size gates, so
        the user's own geometry is never filtered off the map. Keyed by det_id,
        not index, so it holds across a reslice."""
        manual = getattr(self, "_auto_manual_object_ids", None)
        if not manual:
            return False
        return self._object_fid_for(det_idx) in manual

    def _passes_review_filters(self, score: float, area: float, params: dict) -> bool:
        """Whole-object confidence + min/max-size gate (the VISIBLE-set filter).

        The decision itself lives in core.review_defaults so the live stitcher
        thread applies the identical gate while the run streams.
        """
        return object_passes_review_gates(score, area, params)

    def _passes_size_filters(self, area: float, params: dict) -> bool:
        """The min/max-size half of the visible-set gate alone. Revealed
        correction objects bypass the confidence gate but never this one."""
        return area_passes_size_gates(area, params)

    def _review_removed_fids(self) -> set:
        """Union of every per-object removal source: Manual-refine deletions
        and review correction removals. Kept as separate sets (their undo
        paths must not fight); every visible-set consumer filters on the
        union. An explicitly removed object is never resurrected by a reveal
        or a re-detect batch without a new add gesture."""
        removed = set(getattr(self, "_auto_manual_removed", None) or ())
        removed |= set(getattr(self, "_auto_correction_removed", None) or ())
        return removed

    def _snap_review_start_confidence(self, conf: float) -> float:
        """Snap a starting review cutoff to the review slider's 5% grid, clamped
        to [0, review-max]. The slider can only rest on 5% steps, so snapping the
        stored _auto_confidence (and thus the histogram cutoff and the seeded
        widgets) to the same grid keeps all three on one value. 0 is preserved,
        so a run whose best score is under 5% still opens showing every object."""
        from ..dock.styles import _REVIEW_CONF_MAX, _REVIEW_CONF_STEP
        try:
            c = float(conf)
        except (TypeError, ValueError):
            return conf
        pct = int(round(c * 100.0 / _REVIEW_CONF_STEP)) * _REVIEW_CONF_STEP
        pct = max(0, min(_REVIEW_CONF_MAX, pct))
        return pct / 100.0

    def _review_start_confidence(self) -> float:
        """Starting review cutoff. The default (0.30) unless either

        - it would hide EVERY found object: then the highest 5% step in
          [0, 0.30] that shows at least one, or
        - the run's own score distribution says the hidden cohort is the same
          physical population as the confidently detected one (dense scenes
          where most true objects score under the default): then the adaptive
          cutoff from core/review_defaults.adaptive_review_confidence, always
          strictly below the default (classes whose scores behave keep the
          old start exactly).

        Headless/MCP runs keep the seeded value (stable API contract)."""
        default = self._effective_confidence_default()
        if self._auto_headless_run or not self._auto_objects:
            return default
        scores = [s for (_g, s, _a) in self._auto_objects]
        best = max(scores)
        if best < default:
            # Highest 5% step <= best. A best below 5% starts at 0 so the review
            # never opens on "0 shown" for a run that DID find something (the old
            # floor of 5 broke exactly the guarantee this function exists for).
            import math
            step = max(0, int(math.floor(best * 100 / 5.0)) * 5)
            return step / 100.0
        from ...core.review_defaults import adaptive_review_confidence
        adaptive = adaptive_review_confidence(
            [(s, a) for (_g, s, a) in self._auto_objects],
            default=default,
            merge_separate=self._auto_merge_separate,
        )
        return adaptive if adaptive is not None else default
