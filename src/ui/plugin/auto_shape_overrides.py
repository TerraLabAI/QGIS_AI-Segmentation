"""Per-shape settings for the Correct step's selected detection.

The Shapes step settles the whole set at once, which is the right default: a
run of buildings wants one Simplify and one Right angles. It leaves one gap.
A single odd building (a many-winged roof, a shape whose outline came back
with far more points than its neighbours) needs its own dial, and moving the
shared one to suit it wrecks the rest of the set.

So one object can carry its own values. They are stored per canonical index
(``_auto_objects`` rows are overwritten and appended, never deleted, so an
index is a stable key), merged over the shared params inside the ONE refine
chokepoint, and dropped on a new run. Points matters most here because it runs
BEFORE the squaring: lowering it leaves fewer points for Right angles to snap,
which is the way out of a staircased wall.
"""
from __future__ import annotations

try:
    from ...core.i18n import tr
except ImportError:  # no QGIS bindings: the store logic is plain dict work,
    # and keeping it importable without them lets the tests cover the merge
    # rule everywhere.
    def tr(text: str) -> str:
        return text

# The per-shape keys: every control the Shapes step carries that acts on ONE
# outline. Confidence and the size filters stay shared (they decide what is on
# the map, not what it looks like), and so does Shared borders, which needs
# every neighbour at once and therefore cannot mean anything for one object.
_OVERRIDE_KEYS = ("points_pct", "simplify_px", "open_px", "expand_px",
                  "smooth", "fill_holes", "ortho")

# How each key is stored, so a widget value can never land as the wrong type in
# the params dict the refine reads.
_OVERRIDE_COERCE = {
    "points_pct": lambda v: max(1, min(100, int(v))),
    "simplify_px": lambda v: max(0.0, float(v)),
    "open_px": lambda v: max(0.0, float(v)),
    "expand_px": lambda v: int(v),
    "smooth": bool,
    "fill_holes": bool,
    "ortho": bool,
}


class AutoShapeOverridesMixin:
    """Store, apply and publish the selected detection's own shape settings."""

    def _init_shape_override_state(self) -> None:
        """Fresh per-shape settings; called wherever a run's review state is
        reset, so a new run never inherits the last one's exceptions."""
        # canonical index -> {"points_pct": int, "simplify_px": float,
        # "ortho": bool}
        self._auto_shape_overrides: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def _shape_override_for(self, det_idx: int) -> dict | None:
        """This object's own settings, or None when it follows the set."""
        overrides = getattr(self, "_auto_shape_overrides", None)
        if not overrides:
            return None
        return overrides.get(int(det_idx))

    def _shape_params_for_object(self, det_idx: int, params: dict) -> dict:
        """``params`` with this object's own settings written over them.

        Returns the SAME dict when the object has no settings of its own, so
        the shared path allocates nothing.
        """
        override = self._shape_override_for(det_idx)
        if not override:
            return params
        merged = dict(params)
        for key in _OVERRIDE_KEYS:
            if key in override:
                merged[key] = override[key]
        return merged

    # ------------------------------------------------------------------
    # Dock handlers
    # ------------------------------------------------------------------

    def _on_shape_only_changed(self, values: dict) -> None:
        """A per-shape control moved: store the values for the selected object,
        drop only its cached geometry and re-derive the visible set.

        ``values`` is keyed by review param name (the dock's own map), so a
        control added on the dock side needs only a new key in _OVERRIDE_KEYS
        here. An unknown key is ignored rather than written blindly into the
        params the refine reads."""
        idx = getattr(self, "_correct_selected_idx", None)
        if idx is None or self._auto_review is None:
            return
        if idx < 0 or idx >= len(self._auto_objects):
            return
        overrides = getattr(self, "_auto_shape_overrides", None)
        if overrides is None:
            self._init_shape_override_state()
            overrides = self._auto_shape_overrides
        stored: dict = {}
        for key in _OVERRIDE_KEYS:
            if key not in values:
                continue
            try:
                stored[key] = _OVERRIDE_COERCE[key](values[key])
            except (TypeError, ValueError):
                continue
        if not stored:
            return
        overrides[int(idx)] = stored
        # These controls only exist in the AI method, so moving one is the user
        # asking for this outline to be shaped. That thaws a hand-made outline
        # the Manual method had frozen; there is no other way back except Undo.
        frozen_ids = getattr(self, "_auto_shape_frozen_ids", None)
        if frozen_ids:
            frozen_ids.discard(self._object_fid_for(int(idx)))
        self._after_shape_edit(changed=(int(idx),))
        self._push_shape_only_state()

    def _on_shape_only_reset(self) -> None:
        """Back to shared settings: forget this object's exception."""
        idx = getattr(self, "_correct_selected_idx", None)
        overrides = getattr(self, "_auto_shape_overrides", None)
        if idx is None or not overrides or int(idx) not in overrides:
            return
        overrides.pop(int(idx), None)
        self._after_shape_edit(changed=(int(idx),))
        self._push_shape_only_state()

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def _push_shape_only_state(self) -> None:
        """Send the selected object's facts and settings to the dock.

        With no override the controls show the SHARED values, so opening the
        panel starts from what the object already looks like instead of zero.
        """
        dock = self.dock_widget
        if dock is None:
            return
        idx = getattr(self, "_correct_selected_idx", None)
        try:
            if idx is None:
                dock.set_correct_selection_info("")
                return
            shared = self._widget_review_params()
            override = self._shape_override_for(idx) or {}
            values = {key: override.get(key, shared.get(key))
                      for key in _OVERRIDE_KEYS
                      if override.get(key, shared.get(key)) is not None}
            dock.set_shape_only_values(values, bool(override))
            dock.set_correct_selection_info(self._selected_shape_facts(idx))
        except (RuntimeError, AttributeError):
            pass

    def _selected_shape_facts(self, det_idx: int) -> str:
        """"412 m2 - 38 points": the two numbers that decide what to do next.

        The point count is measured on the geometry as DRAWN, so it answers the
        question Points raises ("how many points is this thing carrying?")
        and moves as the control moves. Returns the area alone when the outline
        cannot be counted, and an empty string when neither is available.
        """
        try:
            area = float(self._auto_objects[det_idx][2])
        except (IndexError, TypeError, ValueError):
            return ""
        area_txt = f"{area:,.0f} m²" if area >= 100 else f"{area:,.1f} m²"
        points = self._selected_shape_vertex_count(det_idx)
        if points is None:
            return area_txt
        return f"{area_txt} · " + tr("{count} vertices").format(count=points)

    def _selected_shape_vertex_count(self, det_idx: int) -> int | None:
        """Points in the outline currently drawn for this object, or None."""
        geom = None
        try:
            # Same resolver the yellow highlight uses, so the count and the
            # outline the user is looking at can never disagree.
            resolve = getattr(self, "_drawn_geom_for_index", None)
            if resolve is not None:
                geom = resolve(det_idx)
            if geom is None:
                geom = (getattr(self, "_shape_hit_geoms", None) or {}).get(det_idx)
            if geom is None:
                geom = self._auto_objects[det_idx][0]
        except (IndexError, TypeError, AttributeError):
            return None
        if geom is None:
            return None
        try:
            if geom.isEmpty():
                return None
            return sum(1 for _v in geom.vertices())
        except (RuntimeError, AttributeError, TypeError):
            return None
