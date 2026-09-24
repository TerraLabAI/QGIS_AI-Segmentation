






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
from ...core.shape_policy_dials import auto_review_points_pct_default


def _plan_vertex_spacing_m(review: dict) -> float:




    val = review.get("vertex_spacing_m")
    if isinstance(val, (int, float)) and not isinstance(val, bool) and val >= 0:
        return float(val)
    from ...core.detection_policy import vertex_budget_settings

    return float(vertex_budget_settings()["spacing_m"])


class AutoReviewParamsMixin:


    def _auto_review_preset(self) -> dict:
















        from ...core.review_presets import review_preset_for
        prompt = str((self._auto_run_ctx or {}).get("prompt") or "")



        gsd_m = getattr(self, "_auto_gsd_m", 0.0)
        mask_gsd = getattr(self, "_auto_mask_gsd", 0.0)
        if gsd_m > 0 and mask_gsd > 0 and self._auto_gsd > 0:
            gsd_m *= mask_gsd / self._auto_gsd
        overrides = getattr(self, "_auto_review_preset_overrides", None)
        memo_key = (
            str(getattr(self, "_auto_run_id", "") or ""),
            prompt,
            round(float(gsd_m), 6),
            repr(sorted(overrides.items())) if isinstance(overrides, dict) else "",
        )
        memo = getattr(self, "_auto_review_preset_memo", None)
        if memo is not None and memo[0] == memo_key:
            return dict(memo[1])


        plan = self._active_run_plan(prompt)
        preset = None
        if plan is not None:
            preset = self._review_preset_from_plan(plan.get("review"), gsd_m)
        if preset is None:
            preset = review_preset_for(prompt, gsd_m)
        built = self._with_review_preset_overrides(preset)
        self._auto_review_preset_memo = (memo_key, dict(built))
        return built

    def _with_review_preset_overrides(self, preset: dict) -> dict:





        overrides = getattr(self, "_auto_review_preset_overrides", None)
        if not isinstance(overrides, dict) or not overrides:
            return preset
        merged = dict(preset)
        allowed = set(merged) | {"points_pct", "max_size_m2", "snap_boundaries"}
        for key, value in overrides.items():
            if key in allowed and value is not None:
                merged[key] = value
        return merged

    def _review_preset_from_plan(self, review: object, gsd_m: float) -> dict | None:





        if not isinstance(review, dict):
            return None
        try:
            noise = _min_size_noise_floor_m2(gsd_m)
            object_floor = float(review.get("min_size_m2") or 0.0)
            return {
                "simplify_px": float(review.get("simplify_px", _AUTO_REVIEW_SIMPLIFY_DEFAULT)),
                "smooth": bool(review.get("smooth", _AUTO_REVIEW_SMOOTH_DEFAULT)),
                "expand_px": int(review.get("expand_px", _AUTO_REVIEW_EXPAND_DEFAULT)),




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












        plan = self._active_run_plan(prompt)
        if plan is not None:
            c = plan.get("confidence_default")
            if isinstance(c, (int, float)) and not isinstance(c, bool):
                return float(c)
        from ...core.review_presets import review_start_confidence_default

        return review_start_confidence_default(prompt, is_exemplar_only)

    def _effective_confidence_default(self) -> float:






        return self._confidence_default_for(
            str((self._auto_run_ctx or {}).get("prompt") or ""),
            bool(getattr(self, "_auto_is_exemplar_only", False)))

    def _fresh_review_params(self) -> dict:





        from ...core.boundary_snap import snap_default_enabled
        preset = self._auto_review_preset()
        return {




            "snap_boundaries": bool(preset.get("snap_boundaries", snap_default_enabled())),
            "conf": self._auto_confidence,
            "min_a": float(preset["min_size_m2"]),
            "max_a": float(preset.get("max_size_m2", 0.0)),
            "simplify_px": float(preset["simplify_px"]),
            "smooth": bool(preset["smooth"]),
            "expand_px": int(preset["expand_px"]),
            "fill_holes": bool(preset["fill_holes"]),
            "fill_max_m2": float(preset.get("fill_holes_max_m2", 0.0) or 0.0),
            "open_px": float(preset["clean_px"]),



            "close_notches_m": float(preset.get("close_notches_m", 0.0) or 0.0),
            "ortho": bool(preset["ortho"]),


            "vertex_spacing_m": float(preset.get("vertex_spacing_m", 0.0) or 0.0),



            "points_pct": int(preset.get(
                "points_pct",
                auto_review_points_pct_default(_AUTO_REVIEW_POINTS_PCT_DEFAULT))),
        }

    def _widget_review_params(self) -> dict:








        params = self._fresh_review_params()
        d = self.dock_widget
        if d is None:
            return params
        try:
            conf = self._auto_confidence
            min_a = d.get_auto_min_size()
            max_a = d.get_auto_max_size()
            fill_max = d.get_auto_fill_holes_max()
            snap = d.get_auto_boundary_snap()
            simplify, smooth, expand, fill, clean, ortho = d.get_auto_refine_params()
            points_pct = d.get_auto_points_pct()
        except (RuntimeError, AttributeError):
            return params
        params["conf"] = conf
        params["min_a"] = min_a
        params["max_a"] = max_a
        params["fill_max_m2"] = fill_max
        params["snap_boundaries"] = snap
        params["simplify_px"] = simplify
        params["points_pct"] = points_pct
        params["smooth"] = smooth
        params["expand_px"] = expand
        params["fill_holes"] = fill
        params["open_px"] = clean
        params["ortho"] = ortho
        return params

    def _object_is_manual(self, det_idx: int) -> bool:




        manual = getattr(self, "_auto_manual_object_ids", None)
        if not manual:
            return False
        return self._object_fid_for(det_idx) in manual

    def _passes_review_filters(self, score: float, area: float, params: dict) -> bool:





        return object_passes_review_gates(score, area, params)

    def _passes_size_filters(self, area: float, params: dict) -> bool:


        return area_passes_size_gates(area, params)

    def _review_removed_fids(self) -> set:





        removed = set(getattr(self, "_auto_manual_removed", None) or ())
        removed |= set(getattr(self, "_auto_correction_removed", None) or ())
        return removed

    def _snap_review_start_confidence(self, conf: float) -> float:





        from ..dock.styles import review_conf_max, review_conf_step
        try:
            c = float(conf)
        except (TypeError, ValueError):
            return conf
        step = review_conf_step()
        pct = int(round(c * 100.0 / step)) * step
        pct = max(0, min(review_conf_max(), pct))
        return pct / 100.0

    def _run_scores_rank_objects(self) -> bool:















        if len(self._auto_objects) < 2:
            return True
        try:
            from ...core.server_dials import dial_in_range
            tolerance = dial_in_range("tuning.review.flat_score_tolerance", 0.005, 0.001, 0.05)
        except Exception:  # noqa: BLE001
            tolerance = 0.005


        lo = hi = None
        for (_g, s, _a) in self._auto_objects:
            if lo is None or s < lo:
                lo = s
            if hi is None or s > hi:
                hi = s
            if (hi - lo) > tolerance:
                return True
        return False

    def _review_start_confidence(self) -> float:












        default = self._effective_confidence_default()
        if self._auto_headless_run or not self._auto_objects:
            return default
        scores = [s for (_g, s, _a) in self._auto_objects]
        best = max(scores)
        if best < default:




            import math

            from ..dock.styles import review_conf_step
            step_pct = review_conf_step()
            step = max(0, int(math.floor(best * 100 / step_pct)) * step_pct)
            return step / 100.0
        from ...core.review_defaults import adaptive_review_confidence
        adaptive = adaptive_review_confidence(
            [(s, a) for (_g, s, a) in self._auto_objects],
            default=default,
            merge_separate=self._auto_merge_separate,
        )
        return adaptive if adaptive is not None else default
