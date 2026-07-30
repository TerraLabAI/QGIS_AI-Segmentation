"""Drag-rectangle map tool for object-designating gestures.

UNREACHABLE (2026-07-30): no caller in `src/`. The example and exclude draws
were its last ones and now trace the object point by point with
`PolygonZoneMapTool`, because users asked to outline the object itself rather
than a box around it. Kept because it is the ready-made rectangle gesture, it
is covered by tests, and nothing about it is wrong.

A drag rectangle is press, drag, release. The free-form polygon stays the
gesture for the search zone. This is a leaf tool with no plugin or dock
imports, so any caller can use it.

Press starts a rubber-band rectangle in the caller's colour; dragging resizes
it; releasing on a real (non-degenerate) rectangle emits ``box_selected`` with
the QgsRectangle in canvas-map coordinates and clears the band. A plain click,
or a drag too small on either screen axis to be intentional, is ignored (band
cleared, no signal). Escape emits ``cancelled`` and clears; Ctrl+Z (or
Backspace) takes back the rectangle being dragged and otherwise abandons the
draw, the same two stages as the zone tool; right-click is inert mid-gesture
(like the zone tool), so a context-click never abandons the draw.
"""
from __future__ import annotations

from qgis.core import QgsPointXY, QgsRectangle
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor

from ..core.qt_compat import PolygonGeometry, event_pos


class BoxDrawMapTool(QgsMapTool):
    """Draw a rectangle by dragging, then emit it as a QgsRectangle.

    Signals:
      ``box_selected(QgsRectangle)``: a finished, non-degenerate rectangle in
        the canvas map CRS (the rubber band is already cleared).
      ``cancelled()``: the draw was abandoned (Escape).
      ``tool_deactivated()``: the canvas dropped this tool without the owner
        asking (the user picked another QGIS map tool while it was armed). The
        owner wires it to reset its armed UI. Guarded so an owner-driven disarm
        (``suppress_deactivate_signal``) does not re-enter.
    """

    box_selected = pyqtSignal(QgsRectangle)
    cancelled = pyqtSignal()
    tool_deactivated = pyqtSignal()

    # A press/release closer than this in screen pixels on either axis is a
    # click or an accidental micro-drag, not a rectangle: ignored.
    MIN_DRAG_PX: int = 4

    def __init__(self, canvas, color: QColor):
        super().__init__(canvas)
        self._canvas = canvas
        self._drawing = False
        self._press_map: QgsPointXY | None = None
        self._press_screen = None
        # B-F2 stale-armed guard: see tool_deactivated.
        self.suppress_deactivate_signal = False
        self._in_deactivate_emit = False
        stroke = QColor(color)
        fill = QColor(color.red(), color.green(), color.blue(), 60)
        self._band = QgsRubberBand(canvas, PolygonGeometry)
        self._band.setColor(stroke)
        self._band.setFillColor(fill)
        self._band.setWidth(2)
        self._band.reset(PolygonGeometry)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    # -- Mouse --
    # Contract: right-click is inert mid-gesture (like the zone tool); Escape is
    # the one explicit cancel. Shared by the cut, box and pick edit tools.
    def canvasPressEvent(self, event):  # noqa: N802 (Qt API)
        if event.button() == Qt.MouseButton.RightButton:
            # Inert: absorb it so no context menu pops, but do not cancel.
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event_pos(event)
        self._press_screen = pos
        self._press_map = self.toMapCoordinates(pos)
        self._drawing = True
        self._band.reset(PolygonGeometry)

    def canvasMoveEvent(self, event):  # noqa: N802 (Qt API)
        if not self._drawing or self._press_map is None:
            return
        self._draw_rect(self._press_map, self.toMapCoordinates(event_pos(event)))

    def canvasReleaseEvent(self, event):  # noqa: N802 (Qt API)
        if event.button() == Qt.MouseButton.RightButton:
            # Inert: the press already absorbed the click. Cancelling here too
            # would emit ``cancelled`` a second time for one right-click.
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton or not self._drawing:
            return
        self._drawing = False
        pos = event_pos(event)
        start = self._press_map
        press_screen = self._press_screen
        self._press_map = None
        self._press_screen = None
        self._band.reset(PolygonGeometry)
        if start is None:
            return
        # Reject a click / sub-threshold drag on either screen axis: it is not
        # a deliberate rectangle. Silent (no signal), the band is already clear.
        if press_screen is not None:
            dx = abs(pos.x() - press_screen.x())
            dy = abs(pos.y() - press_screen.y())
            if dx < self.MIN_DRAG_PX or dy < self.MIN_DRAG_PX:
                return
        rect = QgsRectangle(start, self.toMapCoordinates(pos))
        rect.normalize()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        self.box_selected.emit(rect)

    def keyPressEvent(self, event):  # noqa: N802 (Qt API)
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self._cancel()
        elif key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete) or (
            key == Qt.Key.Key_Z and (event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        ):
            # Same two-stage Undo as the zone tool: take back the rectangle
            # being dragged, or (nothing under way) abandon the draw.
            if not self.undo_point():
                self._cancel()
        else:
            super().keyPressEvent(event)

    def undo_point(self) -> bool:
        """Take back the rectangle being dragged, from outside the tool.

        The platform Undo shortcut is bound at window level elsewhere, so it
        matches the key before the canvas ever sees the press and the draw
        would never undo from the keyboard. Returns True when a drag in
        progress was dropped (the tool stays armed for a new one), False when
        nothing was under way: the owner decides what Undo means then, exactly
        as the zone tool does.
        """
        if not self._drawing and self._press_map is None:
            return False
        self._drawing = False
        self._press_map = None
        self._press_screen = None
        self._band.reset(PolygonGeometry)
        return True

    # -- Internals --
    def _draw_rect(self, p1: QgsPointXY, p2: QgsPointXY) -> None:
        self._band.reset(PolygonGeometry)
        self._band.addPoint(QgsPointXY(p1.x(), p1.y()), False)
        self._band.addPoint(QgsPointXY(p2.x(), p1.y()), False)
        self._band.addPoint(QgsPointXY(p2.x(), p2.y()), False)
        self._band.addPoint(QgsPointXY(p1.x(), p2.y()), True)

    def _cancel(self) -> None:
        self._drawing = False
        self._press_map = None
        self._press_screen = None
        self._band.reset(PolygonGeometry)
        self.cancelled.emit()

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
        self._drawing = False
        self._press_map = None
        self._press_screen = None
        try:
            self._band.reset(PolygonGeometry)
        except (RuntimeError, AttributeError):
            pass
        # reset() leaves the empty QgsRubberBand on the scene; a fresh tool is
        # built on every arm, so remove it here or empty bands accumulate.
        try:
            self._canvas.scene().removeItem(self._band)
        except (RuntimeError, AttributeError):
            pass
        super().deactivate()
        self._emit_deactivated()
