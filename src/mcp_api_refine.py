









from __future__ import annotations

import contextlib
import math

from .mcp_api_guard import gui_thread_only




_REFINE_KEY_ALIASES = {
    "simplify": "simplify_px",
    "simplify_px": "simplify_px",
    "clean": "clean_px",
    "clean_px": "clean_px",
    "trim_spikes": "clean_px",
    "trim_spikes_px": "clean_px",
    "smooth": "smooth",
    "round_corners": "smooth",
    "ortho": "ortho",
    "right_angles": "ortho",
    "expand": "expand_px",
    "expand_px": "expand_px",
    "grow_shrink_px": "expand_px",
    "fill_holes": "fill_holes",
    "fill_holes_max": "fill_holes_max_m2",
    "fill_holes_max_m2": "fill_holes_max_m2",
    "min_size_m2": "min_size_m2",
    "max_size_m2": "max_size_m2",
    "shared_borders": "snap_boundaries",
    "snap_boundaries": "snap_boundaries",
    "points": "points_pct",
    "points_pct": "points_pct",
}

_REFINE_BOOL_KEYS = ("smooth", "ortho", "fill_holes", "snap_boundaries")
_REFINE_INT_KEYS = ("expand_px", "points_pct")


_REFINE_WIDGET_NAMES = {
    "simplify_px": "auto_simplify_spin",
    "clean_px": "auto_clean_spin",
    "max_size_m2": "auto_max_size_spin",
    "expand_px": "auto_expand_spin",
    "points_pct": "auto_points_spin",
    "fill_holes_max_m2": "auto_fill_max_spin",
    "smooth": "auto_round_corners_check",
    "ortho": "auto_ortho_check",
    "fill_holes": "auto_fill_holes_check",
    "min_size_m2": "auto_min_size_spin",
    "shared_borders": "auto_boundary_snap_check",
}


class SegmentationRefineMixin:


    def refine_settings(self) -> dict:
















        plugin = self._plugin
        out: dict = {"keys": _refine_key_help()}

        preset = None
        try:
            preset = plugin._fresh_review_params()
        except Exception:  # noqa: BLE001
            preset = None
        out["defaults"] = _refine_public_view(preset) if preset else None

        current = None
        try:
            if getattr(plugin, "_auto_review", None) is not None:
                current = _refine_public_view(plugin._widget_review_params())
                current["confidence"] = float(getattr(plugin, "_auto_confidence", 0.0))
        except Exception:  # noqa: BLE001
            current = None
        out["current"] = current
        return out

    @gui_thread_only
    def apply_refine(
        self,
        simplify_px: float | None = None,
        points_pct: int | None = None,
        round_corners: bool | None = None,
        expand_px: int | None = None,
        fill_holes: bool | None = None,
        fill_holes_max_m2: float | None = None,
        trim_spikes_px: float | None = None,
        right_angles: bool | None = None,
        min_size_m2: float | None = None,
        max_size_m2: float | None = None,
        shared_borders: bool | None = None,
    ) -> dict:














































        from .mcp_api import coerce_bool_param
        bool_args = {
            "round_corners": round_corners,
            "fill_holes": fill_holes,
            "right_angles": right_angles,
            "shared_borders": shared_borders,
        }
        for arg_name, value in bool_args.items():
            if value is None:
                continue
            coerced, bool_err = coerce_bool_param(arg_name, value)
            if bool_err:
                return bool_err
            bool_args[arg_name] = coerced
        round_corners = bool_args["round_corners"]
        fill_holes = bool_args["fill_holes"]
        right_angles = bool_args["right_angles"]
        shared_borders = bool_args["shared_borders"]

        plugin = self._plugin
        if getattr(plugin, "_auto_review", None) is None:
            return {"_error": _NO_REVIEW_MESSAGE}
        dock = getattr(plugin, "dock_widget", None)
        if dock is None:
            return {"_error": "The AI Segmentation panel is not open."}
        busy = self._review_mutation_error()
        if busy:
            return busy

        wanted = {
            "simplify_px": simplify_px,
            "points_pct": points_pct,
            "smooth": round_corners,
            "expand_px": expand_px,
            "fill_holes": fill_holes,
            "fill_holes_max_m2": fill_holes_max_m2,
            "clean_px": trim_spikes_px,
            "ortho": right_angles,
            "min_size_m2": min_size_m2,
            "max_size_m2": max_size_m2,
            "shared_borders": shared_borders,
        }
        for key, value in wanted.items():
            if value is None or key in (*_REFINE_BOOL_KEYS, "shared_borders"):
                continue
            try:
                wanted[key] = _refine_number(key, value)
            except (TypeError, ValueError, OverflowError):
                return {"_error": f"{key} must be a finite number."}
        applied = {}
        for key, value in wanted.items():
            if value is None:
                continue
            widget_name = _REFINE_WIDGET_NAMES.get(key)
            widget = getattr(dock, widget_name, None) if widget_name else None
            if widget is None:
                continue
            if _write_widget_value(widget, value):
                applied[key] = widget.isChecked() if hasattr(widget, "isChecked") else widget.value()

        if not applied:
            return {"_error": (
                "Nothing to apply. Pass at least one setting, and check the "
                "review is on its Shapes step.")}


        for sync_name in ("_sync_auto_right_angle_controls", "_sync_auto_fill_max_row"):
            with contextlib.suppress(RuntimeError, AttributeError):
                getattr(dock, sync_name)()
        params = plugin._widget_review_params()
        if "smooth" in applied:
            applied["smooth"] = params.get("smooth", applied["smooth"])
        if "clean_px" in applied:
            applied["clean_px"] = params.get("open_px", applied["clean_px"])
        if "shared_borders" in applied:
            applied["shared_borders"] = params.get("snap_boundaries", False)

        return self._reslice_open_review(applied)

    def _reslice_open_review(self, applied: dict) -> dict:





        plugin = self._plugin
        counted = self._count_review_kept()

        with contextlib.suppress(Exception):
            reslice = getattr(plugin, "_start_auto_reslice", None)
            if callable(reslice):
                reslice()
        result = {"applied": applied}
        result.update(counted)
        return result

    def _count_review_kept(self) -> dict:

        plugin = self._plugin
        objects = getattr(plugin, "_auto_objects", None) or []
        total = len(objects)
        try:
            params = plugin._widget_review_params()
            removed = plugin._review_removed_fids()
            kept = sum(
                1 for idx, (geom, score, area) in enumerate(objects)
                if idx not in removed and geom is not None and not geom.isEmpty()
                and (plugin._object_is_manual(idx)
                     or plugin._passes_review_filters(score, area, params))
            )
        except Exception:  # noqa: BLE001

            kept = total
        return {"kept_instances": kept, "total_found": total}

    def _refine_overrides_from(self, refine) -> dict | None:






        if not isinstance(refine, dict) or not refine:
            return None
        from .mcp_api import coerce_bool_param
        out: dict = {}
        for raw_key, value in refine.items():
            key = _REFINE_KEY_ALIASES.get(str(raw_key))
            if key is None or value is None:
                continue
            try:
                if key in _REFINE_BOOL_KEYS:


                    coerced, bool_err = coerce_bool_param(key, value)
                    if bool_err:
                        continue
                    out[key] = coerced
                else:
                    out[key] = _refine_number(key, value)
            except (TypeError, ValueError, OverflowError):
                continue
        return out or None


_NO_REVIEW_MESSAGE = (
    "No open detection review to change. A run started through this API saves "
    "itself and leaves nothing open, so pass refine= and confidence= to "
    "detect_auto instead. This call works on a run a person started in the "
    "panel and has not exported yet."
)



def _refine_key_help() -> dict:
    return {
        "simplify_px": "Straighten a staircased outline. Higher loses detail.",
        "points_pct": "Share of outline points kept, 1 to 100.",
        "round_corners": "Round the corners. For organic shapes, not buildings.",
        "expand_px": "Grow (positive) or shrink (negative) every outline.",
        "fill_holes": "Close holes inside an outline.",
        "fill_holes_max_m2": "Close only holes under this ground area. 0 = all.",
        "trim_spikes_px": "Cut thin spikes off an outline. 0 = leave them.",
        "right_angles": "Square the corners. For buildings, not for anything grown.",
        "min_size_m2": "Drop objects under this ground area. 0 = keep every size.",
        "max_size_m2": "Drop objects over this ground area. 0 = no limit.",
        "shared_borders": "Snap touching outlines onto one shared border.",
    }




def _refine_public_view(params: dict) -> dict:
    def _get(key, fallback=None):
        value = params.get(key, fallback)
        return fallback if value is None else value

    return {
        "simplify_px": float(_get("simplify_px", 0.0)),
        "points_pct": int(_get("points_pct", 100)),
        "round_corners": bool(_get("smooth", False)),
        "expand_px": int(_get("expand_px", 0)),
        "fill_holes": bool(_get("fill_holes", False)),
        "fill_holes_max_m2": float(
            _get("fill_max_m2", _get("fill_holes_max_m2", 0.0))),
        "trim_spikes_px": float(_get("open_px", _get("clean_px", 0.0))),
        "right_angles": bool(_get("ortho", False)),
        "min_size_m2": float(_get("min_a", _get("min_size_m2", 0.0))),
        "max_size_m2": float(_get("max_a", _get("max_size_m2", 0.0))),
        "shared_borders": bool(_get("snap_boundaries", False)),
    }


def _refine_number(key, value):

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Expected a finite number")
    return int(number) if key in _REFINE_INT_KEYS else number


def _write_widget_value(widget, value) -> bool:

    try:
        was_blocked = widget.blockSignals(True)
        try:
            if hasattr(widget, "setChecked"):
                widget.setChecked(bool(value))
            elif hasattr(widget, "setValue"):
                try:
                    widget.setValue(value)
                except TypeError:

                    widget.setValue(int(value))
            else:
                return False
        finally:
            widget.blockSignals(was_blocked)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return False
    return True
