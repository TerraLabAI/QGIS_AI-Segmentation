"""The refine panel's settings, remembered between QGIS sessions.

A user who works on one kind of object sets the same few controls at the start
of every session: Right angles for buildings, Round corners for vegetation, a
point budget for their own drawing standard. Those choices died with the
window, so the panel opened on the shipped defaults every time and the work of
setting it up was paid again.

What is remembered, and what is not
-----------------------------------
The shape and outline controls are remembered: they say how this user likes an
outline drawn, and that does not change with the imagery. Right angles is the
exception: it opens off and returns to off at the end of every session, so a
tick made for one run of buildings never reaches the next session.

The Min and Max size filters are NOT. They hide objects, they are read against
one dataset's ground areas, and a threshold carried into a different project
would drop detections with nothing on screen to say why. They stay a
per-session filter, which is what they already were.

How a served default still wins
-------------------------------
Only a control the user has actually moved is written here, so a setting
nobody touched carries no entry and keeps whatever default is in force,
served or shipped. The memory answers for the user's own choices and for
nothing else.

The panel says which controls those are: it marks a setting touched from the
widget that emitted the change, and every programmatic write to these controls
blocks its signals. One entry per touched control, under the same keys as
before, so a memory written by an earlier build reads back unchanged.

``ui.remember_refine_settings`` withdraws the whole memory fleet-wide if it
turns out to cost more than it saves, and the panel then opens on the defaults
exactly as before.
"""
from __future__ import annotations

from contextlib import suppress

from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtWidgets import QCheckBox, QDoubleSpinBox

#: QSettings group. Same "AISegmentation/" root as the dismissed hints and the
#: More settings state. Never rename these literals: they sit in the user's
#: QGIS profile, and a rename silently forgets everyone's panel.
_KEY_PREFIX = "AISegmentation/refine/"

#: One entry per remembered control: the settings key, and the type QSettings
#: has to read it back as. Order is the order the panel reads top to bottom.
#: Right angles is deliberately absent: it opens off and goes back to off at
#: the end of every session, so a tick lives only for the session that made
#: it. It is the one control whose answer belongs to the objects in front of
#: the user rather than to the user, and remembering it squared the trees of
#: the next session.
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

#: Each remembered control paired with the panel attribute that carries it.
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
    """Whether the panel remembers anything at all. Served, default on."""
    try:
        from ...core.server_dials import dial_bool

        return bool(dial_bool("ui.remember_refine_settings", True))
    except Exception:  # noqa: BLE001 -- a preference is best-effort
        return True


def remembered_refine_settings() -> dict:
    """The settings this user has moved, by name. Only what they touched.

    An unreadable or absent entry is simply not in the result, so every caller
    falls to the default in force for that control without a special case.
    """
    out: dict = {}
    if not refine_memory_enabled():
        return out
    try:
        settings = QSettings()
    except Exception:  # noqa: BLE001 -- no settings, no memory
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
    """Write the settings the user moved. Best-effort and silent.

    ``values`` carries the touched controls and no others: a name absent here
    leaves no entry, and the default in force answers for it at the next
    start. Called on every settled change, which is a keystroke's worth of
    work against a QSettings write Qt already buffers.
    """
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
    """``defaults``, with this user's own choices written over them.

    A control they never moved keeps whatever default was handed in, served or
    shipped, so the memory answers for their choices and for nothing else.
    """
    start = dict(defaults)
    with suppress(Exception):
        start.update(remembered_refine_settings())
    return start


def apply_refine_start_values(panel, start: dict) -> None:
    """Put ``start`` on the panel's controls, emitting nothing.

    The caller runs the panel's own sync passes afterwards: a restored
    Fill holes has a size row to show, and a restored Right angles has
    neighbours to disable.
    """
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
    """The remembered name of one of the panel's controls, or None.

    The panel calls this with the widget that emitted a change, so a setting
    is marked touched by the control the user actually moved and by nothing
    else. A widget that is not remembered (the size filters) answers None.
    """
    if widget is None:
        return None
    for name, attr in _WIDGETS:
        if getattr(panel, attr, None) is widget:
            return name
    return None


def capture_refine_settings(panel, only=None) -> dict:
    """Read the remembered controls off the panel. Empty on anything unread.

    ``only`` narrows the read to the named settings, which is how the panel
    keeps a control nobody touched out of the memory.
    """
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


def forget_refine_settings() -> None:
    """Drop every remembered control, so the panel opens on the defaults."""
    with suppress(Exception):
        settings = QSettings()
        for name, _kind in _REMEMBERED:
            settings.remove(_KEY_PREFIX + name)
