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


def _tr_function():
    """tr(), or plain text when there is no QGIS to translate with.

    Returns the function rather than the translated string: the string
    extractor reads the literal written at the call site, so every sentence
    below has to sit inside a tr(...) call of its own.
    """
    try:
        from ...core.i18n import tr

        return tr
    except Exception:  # noqa: BLE001 -- untranslated beats no answer
        return lambda text: text


def right_angles_available() -> bool:
    """True when the footprint regularizer's dependencies really import."""
    try:
        from ...core.building_regularizer import dependencies_available

        return bool(dependencies_available())
    except Exception:  # noqa: BLE001 -- an unreachable engine is an absent one
        return False


def unavailable_tooltip() -> str:
    """Why the control is greyed out, in the user's words."""
    tr = _tr_function()
    return tr(
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

        tr = _tr_function()
        _iface.messageBar().pushWarning(
            "AI Segmentation",
            tr("Right angles is off: this QGIS does not carry the shapely "
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
    # Qt never shows a tooltip over a disabled widget, so the reason also goes
    # on the row behind it, which stays enabled and does show one. Only onto a
    # row that has nothing of its own to say.
    try:
        row = checkbox.parentWidget()
        if row is not None and row.isEnabled() and not row.toolTip():
            row.setToolTip(reason)
    except (RuntimeError, AttributeError):
        pass
    _warn_once()
    return False


# ---- The two dock-side rules that hang off this gate ------------------------
# Free functions taking the dock, so the review panel mixin keeps its two method
# names and delegates here. The panel was over its size band, and both rules are
# about this control, which is what this module already answers for.


def offer_right_angles_availability(dock) -> None:
    """Ask once, when a review opens, whether Right angles can run at all.

    The gate used to fire only on a TICKED box, so a QGIS without the
    geometry library behind it showed a live control that would hand the
    outline back unchanged. Asking here still keeps the import off plugin
    load (no review, no import) and answers before the user reaches the
    Shapes step. The refusal is written under the control as well as put in
    its tooltip.
    """
    ortho = getattr(dock, "auto_ortho_check", None)
    if ortho is None:
        return
    try:
        available = gate_right_angles(
            ortho, getattr(dock, "auto_ortho_label", None))
        label = getattr(dock, "auto_ortho_unavailable_label", None)
        if label is None:
            return
        if available:
            label.setVisible(False)
        else:
            label.setText(unavailable_tooltip())
            label.setVisible(True)
    except (RuntimeError, AttributeError, ImportError):
        pass


def apply_right_angle_conflicts(dock, check_name: str, label_name: str,
                                tooltips_name: str) -> bool:
    """Refuse Right angles when its engine is absent, then make the controls it
    contradicts unavailable. Returns whether those controls are free.

    Orthogonalizing needs a controlled de-staircase pass. Extra generic
    cleanup can erase narrow building parts, while corner rounding reverses
    the requested result. The same rule is also enforced by the value getter,
    so a disabled widget can never leave an old value active.

    Only a TICKED box is checked, which is what keeps the shapely import off
    plugin load: the seeded default is off, so a build-time call costs nothing.

    One body for the two panels that carry this pair (the review's Shapes step
    and the Manual outline panel). They name their widgets differently, which
    is what the three name arguments are for, and they differ only in what they
    do with Round corners afterwards, which stays with each caller.
    """
    ortho = getattr(dock, check_name, None)
    if ortho is not None and ortho.isChecked():
        gate_right_angles(ortho, getattr(dock, label_name, None))
    enabled = not bool(ortho is not None and ortho.isChecked())
    tr = _tr_function()
    blocked_tip = tr(
        "Unavailable while Right angles is on. Turn it off to adjust this "
        "setting.")
    for widget, normal_tip in getattr(dock, tooltips_name, ()):
        try:
            widget.setEnabled(enabled)
            widget.setToolTip(normal_tip if enabled else blocked_tip)
        except (RuntimeError, AttributeError):
            pass
    return enabled


def sync_right_angle_conflicts(dock) -> None:
    """The review's Shapes step: the shared rule, then its own Round corners
    handling (cleared outright, with no memory of the user's tick)."""
    enabled = apply_right_angle_conflicts(
        dock, "auto_ortho_check", "auto_ortho_label",
        "_auto_right_angle_conflict_tooltips")
    # Curving a footprint after it has been squared is contradictory. Clear
    # the state as well as disabling the control, so toggling Right angles
    # never leaves a hidden rounding pass in the preview.
    round_corners = getattr(dock, "auto_round_corners_check", None)
    if not enabled and round_corners is not None:
        try:
            round_corners.blockSignals(True)
            round_corners.setChecked(False)
        except (RuntimeError, AttributeError):
            pass
        finally:
            try:
                round_corners.blockSignals(False)
            except (RuntimeError, AttributeError):
                pass
