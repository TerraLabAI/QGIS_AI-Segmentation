














from __future__ import annotations

from ..dock.manual_recap import format_ground_area

try:
    from ...core.i18n import tr
except ImportError:


    def tr(text: str) -> str:
        return text





_OVERRIDE_KEYS = ("points_pct", "simplify_px", "open_px", "expand_px",
                  "smooth", "fill_holes", "ortho")



_OVERRIDE_COERCE = {
    "points_pct": lambda v: max(1, min(100, int(v))),
    "simplify_px": lambda v: max(0.0, float(v)),
    "open_px": lambda v: max(0.0, float(v)),
    "expand_px": int,
    "smooth": bool,
    "fill_holes": bool,
    "ortho": bool,
}




_SHAPE_ONLY_APPLY_MS = 150


def shape_debounce_ms() -> int:



    from ...core.server_dials import dial_in_range
    return dial_in_range("tuning.review.shape_debounce_ms", _SHAPE_ONLY_APPLY_MS, 30, 1000)


class AutoShapeOverridesMixin:


    def _init_shape_override_state(self) -> None:




        self._auto_shape_overrides: dict[int, dict] = {}

        self._stop_shape_only_apply_timer()





    def _shape_override_for(self, det_idx: int) -> dict | None:

        overrides = getattr(self, "_auto_shape_overrides", None)
        if not overrides:
            return None
        return overrides.get(int(det_idx))

    def _shape_params_for_object(self, det_idx: int, params: dict) -> dict:





        override = self._shape_override_for(det_idx)
        if not override:
            return params
        merged = dict(params)
        for key in _OVERRIDE_KEYS:
            if key in override:
                merged[key] = override[key]
        return merged





    def _on_shape_only_changed(self, values: dict) -> None:









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
        self._auto_shape_only_pending_idx = int(idx)
        self._arm_shape_only_apply_timer()

    def _arm_shape_only_apply_timer(self) -> None:


        dock = getattr(self, "dock_widget", None)
        if dock is None:
            self._apply_shape_only_pending()
            return
        try:
            from .shared import _debounce_timer
            _debounce_timer(self, "_shape_only_apply_timer", dock,
                            shape_debounce_ms(),
                            self._apply_shape_only_pending)
        except (ImportError, RuntimeError, AttributeError, TypeError):



            self._shape_only_apply_timer = None
            self._apply_shape_only_pending()

    def _stop_shape_only_apply_timer(self) -> None:


        self._auto_shape_only_pending_idx = None
        timer = getattr(self, "_shape_only_apply_timer", None)
        if timer is None:
            return
        try:
            timer.stop()
        except (RuntimeError, AttributeError):
            pass

    def _apply_shape_only_pending(self) -> None:







        idx = getattr(self, "_auto_shape_only_pending_idx", None)
        self._auto_shape_only_pending_idx = None
        if idx is None or self._auto_review is None:
            return
        if idx < 0 or idx >= len(self._auto_objects):
            return
        applied = False
        try:
            applied = bool(self._apply_shape_only_to_session(int(idx)))
        except (RuntimeError, AttributeError):
            applied = False
        if not applied:
            self._after_shape_edit(changed=(int(idx),))
        self._push_shape_only_state()

    def _on_shape_only_reset(self) -> None:

        idx = getattr(self, "_correct_selected_idx", None)
        overrides = getattr(self, "_auto_shape_overrides", None)
        self._stop_shape_only_apply_timer()
        if idx is None or not overrides or int(idx) not in overrides:
            return
        overrides.pop(int(idx), None)
        self._after_shape_edit(changed=(int(idx),))
        self._push_shape_only_state()





    def _push_shape_only_state(self) -> None:





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








        try:
            area = float(self._auto_objects[det_idx][2])
        except (IndexError, TypeError, ValueError):
            return ""


        area_txt = format_ground_area(area)
        points = self._selected_shape_vertex_count(det_idx)
        if points is None:
            return area_txt


        points_txt = tr("{count} points").format(count=points)
        if not area_txt:
            return points_txt
        return f"{area_txt} · " + points_txt

    def _selected_shape_vertex_count(self, det_idx: int) -> int | None:

        geom = None
        try:


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



            abstract = geom.constGet()
            if abstract is not None:
                return int(abstract.nCoordinates())
            return sum(1 for _v in geom.vertices())
        except (RuntimeError, AttributeError, TypeError):
            return None
