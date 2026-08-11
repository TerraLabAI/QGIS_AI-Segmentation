"""Click-to-pick map tool for the review's merge gesture.

The other Automatic gestures point at an AREA (a drag rectangle for an example
or a correction box, a free polygon for the zone). Merging points at OBJECTS
instead: the user clicks the pieces of one thing that came back cut apart. So
this tool does the one thing the others do not, a plain click, and hands the
map point to its owner. Which detection sits under that point, and whether the
click picks it or un-picks it, is the owner's business; keeping that out of
here leaves the tool a leaf with no plugin or dock import.

Left click emits ``point_clicked``; a left drag pans the canvas. Enter/Return
emits ``confirmed`` (the same commit key Split
uses, so both review gestures confirm the same way). Escape emits ``cancelled``;
right-click emits ``point_right_clicked`` and is otherwise inert (like the zone
tool), so a context-click opens no menu and never abandons the picks. An owner
that leaves that signal unconnected keeps the old fully inert right-click. A
double-click leaves the object picked (the second click is absorbed) rather than
picking then un-picking it. Cursor motion is forwarded to the owner, which owns
the data-aware hover outline.
"""
from __future__ import annotations

from qgis.core import QgsPointXY
from qgis.gui import QgsMapTool
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QCursor

from ..core.qt_compat import event_pos
from .canvas_slide_pan import CanvasSlidePan


class PickObjectMapTool(QgsMapTool):
    """Emit the map point of every left click until it is disarmed.

    Signals:
      ``point_clicked(QgsPointXY)``: a left click, in the canvas map CRS.
      ``point_right_clicked(QgsPointXY)``: a right click, same CRS. Its only
        owner today is the review's delete gesture.
      ``confirmed()``: Enter/Return, meaning "commit the current picks".
      ``cancelled()``: the gesture was abandoned (Escape).
      ``tool_deactivated()``: the canvas dropped this tool without the owner
        asking (the user picked another QGIS map tool while it was armed). The
        owner wires it to reset its armed UI. Guarded so an owner-driven disarm
        (``suppress_deactivate_signal``) does not re-enter.
    """

    point_clicked = pyqtSignal(QgsPointXY)
    point_right_clicked = pyqtSignal(QgsPointXY)
    cursor_moved = pyqtSignal(QgsPointXY)
    confirmed = pyqtSignal()
    cancelled = pyqtSignal()
    tool_deactivated = pyqtSignal()

    # A press/release further apart than this in screen pixels is a drag (the
    # user grabbed the map), not a pick: ignored, so panning never picks.
    MAX_CLICK_DRIFT_PX: int = 4

    def __init__(self, canvas):
        super().__init__(canvas)
        self._canvas = canvas
        self._press_screen = None
        self._panning = False
        self._slide_pan = CanvasSlidePan(canvas)
        # B-F2 stale-armed guard: see tool_deactivated.
        self.suppress_deactivate_signal = False
        self._in_deactivate_emit = False
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    # -- Mouse --
    # Contract: right-click never cancels a gesture (like the zone tool); Escape
    # is the one explicit cancel. Shared by the cut, box and pick edit tools.
    def canvasPressEvent(self, event):  # noqa: N802 (Qt API)
        if event.button() == Qt.MouseButton.RightButton:
            # Absorb it so no context menu pops and no gesture is cancelled,
            # then hand the point to the owner. Emitted last: the slot may
            # rearrange the owner's canvas state, so nothing here touches the
            # tool afterwards.
            event.accept()
            self.point_right_clicked.emit(self.toMapCoordinates(event_pos(event)))
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_screen = event_pos(event)
            self._panning = False
            event.accept()

    def canvasMoveEvent(self, event):  # noqa: N802 (Qt API)
        pos = event_pos(event)
        press = self._press_screen
        if press is not None:
            drift_x = abs(pos.x() - press.x())
            drift_y = abs(pos.y() - press.y())
            if not self._panning and (drift_x > self.MAX_CLICK_DRIFT_PX or drift_y > self.MAX_CLICK_DRIFT_PX):
                self._panning = True
                # From the press point, not from here: the map has to follow
                # the whole drag, including the first few pixels that were
                # still ambiguous between a pick and a grab.
                self._slide_pan.begin_at(press)
            if self._panning:
                self._slide_pan.slide_to(pos)
                return
        self.cursor_moved.emit(self.toMapCoordinates(pos))

    def canvasReleaseEvent(self, event):  # noqa: N802 (Qt API)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        press = self._press_screen
        self._press_screen = None
        if press is None:
            return
        event.accept()
        if self._panning:
            self._panning = False
            self._slide_pan.commit(event_pos(event))
            return
        pos = event_pos(event)
        drift_x = abs(pos.x() - press.x())
        drift_y = abs(pos.y() - press.y())
        if drift_x > self.MAX_CLICK_DRIFT_PX or drift_y > self.MAX_CLICK_DRIFT_PX:
            return
        self.point_clicked.emit(self.toMapCoordinates(pos))

    def canvasDoubleClickEvent(self, event):  # noqa: N802 (Qt API)
        # Absorb the second click of a double-click. The first click already
        # picked the object; clearing the pending press stops the trailing
        # release from firing a second point_clicked that would un-pick it. So
        # a double-click leaves the object picked, not toggled back off.
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_screen = None
            event.accept()

    def wheelEvent(self, event):  # noqa: N802 (Qt API)
        """Leave wheel input to the canvas, including trackpad scrolling."""
        event.ignore()

    def gestureEvent(self, event):  # noqa: N802 (Qt API)
        """Do not consume macOS trackpad gestures owned by the canvas."""
        event.ignore()
        return False

    def keyPressEvent(self, event):  # noqa: N802 (Qt API)
        # Both keys are acted on here, so both are accepted: an unaccepted key
        # keeps travelling up the parent chain and fires a second handler.
        if event.key() == Qt.Key.Key_Escape:
            self._press_screen = None
            event.accept()
            self.cancelled.emit()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            self.confirmed.emit()
        else:
            super().keyPressEvent(event)

    def _emit_deactivated(self) -> None:
        # B-F2: tell the owner the canvas dropped us. Suppressed during an
        # owner-driven disarm, and re-entrancy-guarded so a slot that triggers
        # another deactivate cannot loop.
        if self.suppress_deactivate_signal or self._in_deactivate_emit:
            return
        self._in_deactivate_emit = True
        try:
            self.tool_deactivated.emit()
        finally:
            self._in_deactivate_emit = False

    def deactivate(self) -> None:  # noqa: N802 (Qt API)
        self._press_screen = None
        self._panning = False
        # A tool swap mid-drag is the end of that drag too: keep the map where
        # the user dragged it to rather than snapping it back.
        self._slide_pan.commit()
        # No rubber band on this tool, so nothing to remove from the scene.
        super().deactivate()
        self._emit_deactivated()
