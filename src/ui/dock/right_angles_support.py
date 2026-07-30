
























from __future__ import annotations



_state = {"warned": False}


def _tr_function():






    try:
        from ...core.i18n import tr

        return tr
    except Exception:  # noqa: BLE001
        return lambda text: text


def right_angles_available() -> bool:

    try:
        from ...core.building_regularizer import dependencies_available

        return bool(dependencies_available())
    except Exception:  # noqa: BLE001
        return False


def unavailable_tooltip() -> str:

    tr = _tr_function()
    return tr(
        "Unavailable: this QGIS does not carry the shapely geometry library "
        "that squares the walls. A QGIS installed with its full package set "
        "carries it.")


def _warn_once() -> None:
    if _state["warned"]:
        return
    _state["warned"] = True
    try:
        from qgis.utils import iface as _iface

        tr = _tr_function()
        _iface.messageBar().pushWarning(
            "AI Segmentation",
            tr("Right angles is off: this QGIS does not carry the shapely "
               "geometry library it needs. Every other shape control still "
               "works."))
    except Exception:  # noqa: BLE001
        pass  # nosec B110


def gate_right_angles(checkbox, *labels) -> bool:











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



    try:
        row = checkbox.parentWidget()
        if row is not None and row.isEnabled() and not row.toolTip():
            row.setToolTip(reason)
    except (RuntimeError, AttributeError):
        pass
    _warn_once()
    return False








def offer_right_angles_availability(dock) -> None:









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


    enabled = apply_right_angle_conflicts(
        dock, "auto_ortho_check", "auto_ortho_label",
        "_auto_right_angle_conflict_tooltips")



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
