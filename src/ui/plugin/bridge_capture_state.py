"""What the hand-edit bridge has to ask the canvas rather than remember.

Two of the bridge's questions cannot be answered from state it keeps. How many
points are in the line being traced changes between two ticks of its poll, so a
key pressed in that gap reads a count from before the point the user just
placed. And QGIS builds its Vertex Editor dock the first time a corner is
locked, so a look for it can only be held once it has found something.

Plain functions taking the plugin instance, so no name lands in the bridge's
mixin namespace and nothing has to be assembled into the class.
"""
from __future__ import annotations

import time

# How long a failed look for the QGIS Vertex Editor dock stands before it is
# tried again. QGIS builds that dock the first time a corner is locked, so a
# miss has to be retried, but not on every tick of the gesture poll.
_VERTEX_DOCK_RESCAN_S = 1.0


def live_capture_points(owner, read_points) -> int:
    """Points in the tool's open capture line, asked of the tool NOW.

    The bridge's poll samples a fifth of a second apart, so Escape and Undo
    pressed in that gap read the count from before the point that was just
    placed: Escape then ends the session instead of dropping the line. Falls
    back to the polled count when the tool cannot be asked.

    ``read_points`` is the bridge's own reader, handed in so this file needs no
    knowledge of which tools carry a capture line.
    """
    try:
        canvas = owner.iface.mapCanvas()
        tool = canvas.mapTool() if canvas is not None else None
        if tool is not None:
            points = read_points(tool, str(tool.metaObject().className()))
            if points is not None:
                return int(points)
    except (RuntimeError, AttributeError, TypeError):
        pass
    return int(getattr(owner, "_qgis_bridge_prev_points", 0) or 0)


def vertex_editor_docks(owner) -> list:
    """The QGIS Vertex Editor dock widgets, found without relying on locale.

    Held once found: the gesture poll asks several times a second, and walking
    every dock widget in the main window that often is work for nothing. QGIS
    builds that dock once and keeps it, so a hit never goes stale; a miss is
    retried a second later instead of on the next tick.
    """
    held = getattr(owner, "_qgis_bridge_vertex_docks", None)
    if held:
        return held
    now = time.monotonic()
    if now - getattr(owner, "_qgis_bridge_vertex_docks_at", 0.0) < _VERTEX_DOCK_RESCAN_S:
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
    """Say that QGIS would not open the review layer for editing.

    The entry unwinds cleanly either way, so without this the Edit button looks
    pressed and nothing at all happens.
    """
    try:
        from qgis.core import Qgis, QgsMessageLog

        QgsMessageLog.logMessage(
            f"Manual edit bridge: the layer refused to open ({reason})",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        owner.iface.messageBar().pushMessage(
            "AI Segmentation",
            tr("QGIS would not open this layer for editing, so the manual "
               "tools could not start."),
            level=Qgis.MessageLevel.Warning, duration=8)
    except (RuntimeError, AttributeError):
        pass  # nosec B110 -- the entry has already unwound
