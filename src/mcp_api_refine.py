"""Shape cleanup for the public API: the panel's refine controls, as arguments.

Part of `SegmentationMCPAPI` (see `mcp_api.py`), split out so one concern sits
in one file. Refining never re-detects anything and never costs anything: it
re-shapes outlines the run already produced.

Two ways in. Before a zone run, pass ``refine`` to ``detect_auto`` and the run
saves the shapes you asked for. After a run that a person left open in the
panel, call :meth:`apply_refine` to reshape what is on screen.
"""
from __future__ import annotations

import contextlib

# Every name a caller may use, mapped onto the key the refine pipeline reads.
# Both spellings of each control are accepted, because the panel labels them in
# plain words and the pipeline names them by unit.
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
    "points": "points_pct",
    "points_pct": "points_pct",
}

_REFINE_BOOL_KEYS = ("smooth", "ortho", "fill_holes")
_REFINE_INT_KEYS = ("expand_px", "points_pct")

# Widget on the dock that each refine key drives, for an open review.
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
    """Read the shape-cleanup settings, and apply them to an open review."""

    def refine_settings(self) -> dict:
        """Report the shape-cleanup settings in force, and what they mean.

        Returns
        -------
        dict with keys:
            "defaults"   -- the settings a fresh run of the current object
                            class starts from, as the same plain keys
                            :meth:`apply_refine` takes.
            "current"    -- the settings an open review is showing, or None
                            when no review is open.
            "keys"       -- one line per setting saying what it does, so a
                            caller can choose without reading this source.
            "_error"     -- present only on failure.

        Costs nothing and changes nothing.
        """
        plugin = self._plugin
        out: dict = {"keys": _refine_key_help()}

        preset = None
        try:
            preset = plugin._auto_review_preset()
        except Exception:  # noqa: BLE001 - reporting must not break on a half-built run
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
        """Reshape the objects of an open review, the way the panel's Shapes step does.

        Every argument left at None keeps what the review is already using, so
        a caller changes one thing without restating the rest.

        Parameters
        ----------
        simplify_px : float | None
            How hard to straighten a staircased outline. Higher removes more
            points and more true detail.
        points_pct : int | None
            Share of each outline's points to keep, 1 to 100.
        round_corners : bool | None
            Round the corners. Right for organic shapes, wrong for buildings.
        expand_px : int | None
            Grow (positive) or shrink (negative) every outline.
        fill_holes : bool | None
            Close holes inside an outline.
        fill_holes_max_m2 : float | None
            Only close holes smaller than this ground area. 0 closes them all.
        trim_spikes_px : float | None
            Cut thin spikes off an outline. 0 leaves them.
        right_angles : bool | None
            Square the corners. Right for buildings, wrong for anything grown.
            Turning this on switches off rounding and spike trimming, exactly
            as the panel does, so two cleanups never stack.
        min_size_m2 : float | None
            Drop objects smaller than this ground area. 0 keeps every size.
        max_size_m2 : float | None
            Drop objects larger than this ground area. 0 means no limit.
        shared_borders : bool | None
            Snap outlines that touch onto one shared border.

        Returns
        -------
        dict with "applied" (the settings now in force), "kept_instances" and
        "total_found", or "_error" when no review is open.

        Cost
        ----
        Free. It re-shapes objects the run already paid for and calls no
        server. It also does not save: call the panel's export, or run
        :meth:`detect_auto`, to write the result.
        """
        plugin = self._plugin
        if getattr(plugin, "_auto_review", None) is None:
            return {"_error": _NO_REVIEW_MESSAGE}
        dock = getattr(plugin, "dock_widget", None)
        if dock is None:
            return {"_error": "The AI Segmentation panel is not open."}

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
        applied = {}
        for key, value in wanted.items():
            if value is None:
                continue
            widget_name = _REFINE_WIDGET_NAMES.get(key)
            widget = getattr(dock, widget_name, None) if widget_name else None
            if widget is None:
                continue
            if _write_widget_value(widget, value):
                applied[key] = value

        if not applied:
            return {"_error": (
                "Nothing to apply. Pass at least one setting, and check the "
                "review is on its Shapes step.")}

        return self._reslice_open_review(applied)

    def _reslice_open_review(self, applied: dict) -> dict:
        """Re-derive the review's visible set and count it, after a settings change.

        The plugin's own reslice repaints the canvas but runs in slices, so the
        count is computed here in one pass and is the authoritative one.
        """
        plugin = self._plugin
        counted = self._count_review_kept()
        # The repaint is a bonus, the count is not.
        with contextlib.suppress(Exception):
            reslice = getattr(plugin, "_start_auto_reslice", None)
            if callable(reslice):
                reslice()
        result = {"applied": applied}
        result.update(counted)
        return result

    def _count_review_kept(self) -> dict:
        """How many of the review's objects the current settings keep."""
        plugin = self._plugin
        objects = getattr(plugin, "_auto_objects", None) or []
        total = len(objects)
        kept = total
        # A count is never worth an exception: the total stands in for it.
        with contextlib.suppress(Exception):
            params = plugin._widget_review_params()
            removed = plugin._review_removed_fids()
            kept = sum(
                1 for idx, (geom, score, area) in enumerate(objects)
                if idx not in removed and geom is not None
                and plugin._passes_review_filters(score, area, params)
            )
        return {"kept_instances": kept, "total_found": total}

    def _refine_overrides_from(self, refine) -> dict | None:
        """Normalize a caller's refine dict onto the keys the pipeline reads.

        Accepts either spelling of every control (``simplify`` or
        ``simplify_px``, ``right_angles`` or ``ortho``). Unknown keys and
        values that will not convert are dropped rather than passed on.
        """
        if not isinstance(refine, dict) or not refine:
            return None
        out: dict = {}
        for raw_key, value in refine.items():
            key = _REFINE_KEY_ALIASES.get(str(raw_key))
            if key is None or value is None:
                continue
            try:
                if key in _REFINE_BOOL_KEYS:
                    out[key] = bool(value)
                elif key in _REFINE_INT_KEYS:
                    out[key] = int(value)
                else:
                    out[key] = float(value)
            except (TypeError, ValueError):
                continue
        return out or None


_NO_REVIEW_MESSAGE = (
    "No open detection review to change. A run started through this API saves "
    "itself and leaves nothing open, so pass refine= and confidence= to "
    "detect_auto instead. This call works on a run a person started in the "
    "panel and has not exported yet."
)


# One line per setting, so a caller can choose without reading this source.
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


# The pipeline's own key names are units and abbreviations. This is the view a
# caller gets back, in the words apply_refine takes.
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
        "max_size_m2": float(_get("max_a", 0.0)),
        "shared_borders": bool(_get("snap_boundaries", False)),
    }


# Writes one value onto one Qt widget without waking the debounce that would
# otherwise re-enter the reslice the caller is about to start itself.
def _write_widget_value(widget, value) -> bool:
    try:
        widget.blockSignals(True)
        try:
            if hasattr(widget, "setChecked"):
                widget.setChecked(bool(value))
            elif hasattr(widget, "setValue"):
                try:
                    widget.setValue(value)
                except TypeError:
                    # A whole-number spin box refuses a float outright.
                    widget.setValue(int(value))
            else:
                return False
        finally:
            widget.blockSignals(False)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return False
    return True
