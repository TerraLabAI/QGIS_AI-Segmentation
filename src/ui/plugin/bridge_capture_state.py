










from __future__ import annotations

import time




_VERTEX_DOCK_RESCAN_S = 1.0


def live_capture_points(owner, read_points) -> int:










    try:
        canvas = owner.iface.mapCanvas()
        tool = canvas.mapTool() if canvas is not None else None
        if tool is not None:
            points = read_points(tool, str(tool.metaObject().className()))



            return int(points) if points is not None else 0
    except (RuntimeError, AttributeError, TypeError):
        pass
    return int(getattr(owner, "_qgis_bridge_prev_points", 0) or 0)


def vertex_editor_docks(owner) -> list:







    held = getattr(owner, "_qgis_bridge_vertex_docks", None)
    if held:
        return held
    from ...core.server_dials import dial_in_range
    rescan_s = dial_in_range(
        "tuning.agent.vertex_dock_rescan_s", _VERTEX_DOCK_RESCAN_S, 0.25, 5.0)
    now = time.monotonic()
    if now - getattr(owner, "_qgis_bridge_vertex_docks_at", 0.0) < rescan_s:
        return []
    owner._qgis_bridge_vertex_docks_at = now
    try:
        from qgis.PyQt.QtWidgets import QDockWidget

        docks = owner.iface.mainWindow().findChildren(QDockWidget)
    except (RuntimeError, AttributeError, TypeError):
        return []
    found = []
    for dock in docks:
        try:
            key = f"{dock.objectName()} {dock.windowTitle()}".lower()
        except (RuntimeError, AttributeError):
            continue
        if "vertex" in key and ("editor" in key or "dock" in key):
            found.append(dock)
    owner._qgis_bridge_vertex_docks = found
    return found


def report_editing_refused(owner, tr, reason: str) -> None:





    try:
        from qgis.core import Qgis, QgsMessageLog

        from ...core.server_dials import dial_in_range

        QgsMessageLog.logMessage(
            f"Manual edit bridge: the layer refused to open ({reason})",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        owner.iface.messageBar().pushMessage(
            "AI Segmentation",
            tr("QGIS would not open this layer for editing, so the manual "
               "tools could not start."),
            level=Qgis.MessageLevel.Warning,
            duration=dial_in_range("tuning.agent.editing_refused_notice_s", 8, 4, 10))
    except (RuntimeError, AttributeError):
        pass  # nosec B110
