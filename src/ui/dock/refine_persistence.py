



































from __future__ import annotations

from contextlib import suppress

from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtWidgets import QCheckBox, QDoubleSpinBox




_KEY_PREFIX = "AISegmentation/refine/"








_REMEMBERED: tuple[tuple[str, type], ...] = (
    ("round_corners", bool),
    ("fill_holes", bool),
    ("fill_holes_max_m2", float),
    ("points_pct", int),
    ("simplify_px", float),
    ("clean_px", float),
    ("expand_px", int),
)

_TYPES = dict(_REMEMBERED)


_WIDGETS: tuple[tuple[str, str], ...] = (
    ("right_angles", "right_angles_checkbox"),
    ("round_corners", "round_corners_checkbox"),
    ("fill_holes", "fill_holes_checkbox"),
    ("fill_holes_max_m2", "fill_holes_max_spinbox"),
    ("points_pct", "points_spinbox"),
    ("simplify_px", "simplify_spinbox"),
    ("clean_px", "clean_edges_spinbox"),
    ("expand_px", "expand_spinbox"),
)


def refine_memory_enabled() -> bool:

    try:
        from ...core.server_dials import dial_bool

        return bool(dial_bool("ui.remember_refine_settings", True))
    except Exception:  # noqa: BLE001
        return True


def remembered_refine_settings() -> dict:





    out: dict = {}
    if not refine_memory_enabled():
        return out
    try:
        settings = QSettings()
    except Exception:  # noqa: BLE001
        return out
    for name, kind in _REMEMBERED:
        key = _KEY_PREFIX + name
        with suppress(Exception):
            if not settings.contains(key):
                continue
            value = settings.value(key, type=kind)
            if value is not None:
                out[name] = kind(value)
    return out


def remember_refine_settings(values: dict) -> None:







    if not values or not refine_memory_enabled():
        return
    with suppress(Exception):
        settings = QSettings()
        for name, value in values.items():
            kind = _TYPES.get(name)
            if kind is None:
                continue
            with suppress(TypeError, ValueError):
                settings.setValue(_KEY_PREFIX + name, kind(value))


def refine_start_values(defaults: dict) -> dict:





    start = dict(defaults)
    with suppress(Exception):
        start.update(remembered_refine_settings())
    return start


def apply_refine_start_values(panel, start: dict) -> None:






    widgets = [getattr(panel, attr, None) for _name, attr in _WIDGETS]
    widgets = [w for w in widgets if w is not None]
    for widget in widgets:
        widget.blockSignals(True)
    try:
        for name, attr in _WIDGETS:
            widget = getattr(panel, attr, None)
            if widget is None or name not in start:
                continue
            with suppress(RuntimeError, TypeError, ValueError):
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(start[name]))
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(start[name]))
                else:
                    widget.setValue(int(start[name]))
    finally:
        for widget in widgets:
            widget.blockSignals(False)


def refine_setting_name_for(panel, widget) -> str | None:






    if widget is None:
        return None
    for name, attr in _WIDGETS:
        if getattr(panel, attr, None) is widget:
            return name
    return None


def capture_refine_settings(panel, only=None) -> dict:





    wanted = None if only is None else set(only)
    out: dict = {}
    for name, attr in _WIDGETS:
        if wanted is not None and name not in wanted:
            continue
        widget = getattr(panel, attr, None)
        if widget is None:
            continue
        with suppress(RuntimeError, TypeError, ValueError):
            if isinstance(widget, QCheckBox):
                out[name] = bool(widget.isChecked())
            else:
                out[name] = _TYPES[name](widget.value())
    return out
