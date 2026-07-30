"""Whether "Right angles" can run at all, and the refusal when it cannot.

The control drives core.building_regularizer, whose geometry core is numpy +
shapely. The plugin installs neither: numpy comes with QGIS and with the
plugin's own env, and shapely comes with QGIS ALONE (venv_manager.
REQUIRED_PACKAGES does not list it), and only when that install carries it.
OSGeo4W ships shapely with the qgis-full meta package and the standalone
installer, not with the bare qgis package.

Without it, regularize_qgs_geometry_ex hands its input straight back and every
caller keeps it. The user ticks the box, waits for the slowest control in the
panel, and gets an outline nothing squared, with nothing to tell it apart from
the anti-distortion guard declining one shape.

So ask first and say no out loud: untick the box, grey it out, put the reason
in its tooltip, and warn once in the message bar. The other shape controls are
untouched, because none of them needs shapely.

The test is the import itself, the only one that catches a shapely whose
native GEOS library will not load. So it runs when the control is REACHED, the
user ticking it or a preset seeding it on, never at plugin load: a user who
leaves Right angles alone never pays for a shapely import. Importing THIS
module costs nothing either, which is why tr and the message bar are bound
inside the functions.
"""
from __future__ import annotations

# One warning per session. Repeating it on every toggle of every panel turns
# the answer into noise; the greyed control and its tooltip carry it after that.
_warned = False


def _tr(text: str) -> str:
    """tr(), and the plain text when there is no QGIS to translate with."""
    try:
        from ...core.i18n import tr

        return tr(text)
    except Exception:  # noqa: BLE001 -- untranslated beats no answer
        return text


def right_angles_available() -> bool:
    """True when the footprint regularizer's dependencies really import."""
    try:
        from ...core.building_regularizer import dependencies_available

        return bool(dependencies_available())
    except Exception:  # noqa: BLE001 -- an unreachable engine is an absent one
        return False


def unavailable_tooltip() -> str:
    """Why the control is greyed out, in the user's words."""
    return _tr(
        "Unavailable: this QGIS does not carry the shapely geometry library "
        "that squares the walls. A QGIS installed with its full package set "
        "carries it.")


def _warn_once() -> None:
    global _warned
    if _warned:
        return
    _warned = True
    try:
        from qgis.utils import iface as _iface

        _iface.messageBar().pushWarning(
            "AI Segmentation",
            _tr("Right angles is off: this QGIS does not carry the shapely "
                "geometry library it needs. Every other shape control still "
                "works."))
    except Exception:  # noqa: BLE001 -- no message bar outside the GUI
        pass  # nosec B110


def gate_right_angles(checkbox, *labels) -> bool:
    """Let "Right angles" through, or refuse it visibly. True = it can run.

    Called when the control is reached: the user ticking it, or a preset
    seeding it on. Costs nothing while the box is unticked, which is why every
    caller checks that first.

    On refusal the box is unticked with its signals blocked (the value getters
    read the widget, so an unticked box is also an unticked setting) and left
    disabled, with the reason on it and on the labels beside it. Nothing is
    touched when the engine is there, so an install that works never sees this.
    """
    if checkbox is None:
        return False
    if right_angles_available():
        return True
    reason = unavailable_tooltip()
    try:
        checkbox.blockSignals(True)
        checkbox.setChecked(False)
    except (RuntimeError, AttributeError):
        pass
    finally:
        try:
            checkbox.blockSignals(False)
        except (RuntimeError, AttributeError):
            pass
    for widget in (checkbox, *labels):
        if widget is None:
            continue
        try:
            widget.setEnabled(False)
            widget.setToolTip(reason)
        except (RuntimeError, AttributeError):
            pass
    _warn_once()
    return False
