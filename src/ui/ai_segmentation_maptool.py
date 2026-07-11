from __future__ import annotations

from qgis.core import QgsPointXY
from qgis.gui import QgsMapCanvas, QgsMapTool, QgsVertexMarker
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor

from ..core.qt_compat import event_pos


class AISegmentationMapTool(QgsMapTool):
    """Map tool for Manual AI segmentation with point prompts.

    Keyboard/mouse map while the tool is active (the dock's "?" menu >
    Keyboard shortcuts shows the full plugin map):
    - Left click: add area (positive point)
    - Right click: remove area (negative point)
    - Ctrl/Cmd+Z or Backspace: undo the last point
    - S: save the current polygon
    - Enter: export the saved polygons to a layer
    - Escape: stop the session (confirms when work would be discarded)
    - Delete (or Ctrl/Cmd+Backspace): delete the active object
    - Hold Space + move, arrow keys, or middle mouse drag: pan the map
    """

    positive_click = pyqtSignal(QgsPointXY)
    negative_click = pyqtSignal(QgsPointXY)
    # Second press of a double-click (the first press already emitted
    # positive/negative_click): the refine handoff opens the clicked detection
    # for editing on it.
    double_click = pyqtSignal(QgsPointXY)
    # Plain cursor motion (not while Space-panning): the refine handoff uses it
    # for the hover highlight. Cheap consumers only; emitted on every move.
    cursor_moved = pyqtSignal(QgsPointXY)
    tool_deactivated = pyqtSignal()
    undo_requested = pyqtSignal()
    save_polygon_requested = pyqtSignal()
    export_layer_requested = pyqtSignal()
    stop_segmentation_requested = pyqtSignal()

    POSITIVE_COLOR = QColor(0, 200, 0)  # Green for include
    NEGATIVE_COLOR = QColor(220, 0, 0)  # Red for exclude
    MARKER_SIZE = 10
    MARKER_PEN_WIDTH = 2

    def __init__(self, canvas: QgsMapCanvas):
        super().__init__(canvas)
        self.canvas = canvas
        self._active = False
        self._markers: list[QgsVertexMarker] = []
        self._space_panning = False
        self._pan_last_point = None
        # Modifiers of the LAST emitted click, read by the plugin handlers
        # (Ctrl+click = additive selection in the refine handoff). Stored as an
        # attribute so the click signal signatures stay stable.
        self.last_click_modifiers = Qt.KeyboardModifier.NoModifier

    def activate(self):
        super().activate()
        self._active = True
        self._space_panning = False
        self._pan_last_point = None
        self.canvas.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def deactivate(self):
        super().deactivate()
        self._active = False
        self._space_panning = False
        self._pan_last_point = None
        # Don't clear markers here - the plugin decides whether to keep them
        # (e.g. when user is asked to confirm leaving segmentation mode).
        self.tool_deactivated.emit()

    def add_marker(self, point: QgsPointXY, is_positive: bool) -> QgsVertexMarker:
        """Add a visual marker at the click location."""
        marker = QgsVertexMarker(self.canvas)
        marker.setCenter(point)
        marker.setIconSize(self.MARKER_SIZE)
        marker.setPenWidth(self.MARKER_PEN_WIDTH)

        if is_positive:
            marker.setIconType(QgsVertexMarker.IconType.ICON_CIRCLE)
            marker.setColor(self.POSITIVE_COLOR)
            marker.setFillColor(QColor(0, 200, 0, 100))
        else:
            marker.setIconType(QgsVertexMarker.IconType.ICON_X)
            marker.setColor(self.NEGATIVE_COLOR)
            marker.setFillColor(QColor(220, 0, 0, 100))

        self._markers.append(marker)
        return marker

    def remove_last_marker(self) -> bool:
        """Remove the last marker added."""
        if self._markers:
            marker = self._markers.pop()
            try:
                scene = self.canvas.scene()
                if scene is not None:
                    scene.removeItem(marker)
            except RuntimeError:
                pass
            try:
                self.canvas.refresh()
            except RuntimeError:
                pass
            return True
        return False

    def clear_markers(self):
        """Remove all markers from the canvas."""
        for marker in self._markers:
            try:
                scene = self.canvas.scene()
                if scene is not None:
                    scene.removeItem(marker)
            except RuntimeError:
                pass
        self._markers.clear()
        try:
            self.canvas.refresh()
        except RuntimeError:
            pass

    def canvasPressEvent(self, event):
        if not self._active:
            return

        # Ignore clicks while Space-panning
        if self._space_panning:
            return

        pt = event_pos(event)
        point = self.toMapCoordinates(pt)

        self.last_click_modifiers = event.modifiers()
        if event.button() == Qt.MouseButton.LeftButton:
            self.add_marker(point, is_positive=True)
            self.positive_click.emit(point)
        elif event.button() == Qt.MouseButton.RightButton:
            self.add_marker(point, is_positive=False)
            self.negative_click.emit(point)

    def canvasDoubleClickEvent(self, event):
        if not self._active or self._space_panning:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_click_modifiers = event.modifiers()
            self.double_click.emit(self.toMapCoordinates(event_pos(event)))

    def canvasMoveEvent(self, event):
        if not self._space_panning:
            try:
                self.cursor_moved.emit(self.toMapCoordinates(event_pos(event)))
            except RuntimeError:
                pass
            return

        current = event_pos(event)
        if self._pan_last_point is not None:
            start_map = self.toMapCoordinates(self._pan_last_point)
            end_map = self.toMapCoordinates(current)
            center = self.canvas.center()
            new_center = QgsPointXY(
                center.x() + (start_map.x() - end_map.x()),
                center.y() + (start_map.y() - end_map.y()),
            )
            self.canvas.setCenter(new_center)
            self.canvas.refresh()
        self._pan_last_point = current

    def start_space_pan(self):
        """Called when Space is pressed - enable temporary pan mode."""
        self._space_panning = True
        # Record current cursor position as pan reference
        self._pan_last_point = self.canvas.mapFromGlobal(QCursor.pos())
        self.canvas.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def stop_space_pan(self):
        """Called when Space is released - restore segmentation mode."""
        self._space_panning = False
        self._pan_last_point = None
        if self._active:
            self.canvas.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def wheelEvent(self, event):
        # Let the canvas handle wheel events for zoom
        # By not accepting the event, it propagates to the canvas
        event.ignore()

    def gestureEvent(self, event):
        # Let the canvas handle pinch-to-zoom and other gestures
        # This is essential for trackpad gestures on macOS
        # Return False so the event propagates (QGIS 3.44+/Qt6 requires bool return)
        event.ignore()
        return False

    def keyPressEvent(self, event):
        # Keyboard shortcuts are handled by the plugin's eventFilter on the
        # main window (works regardless of focus).  This method only exists
        # to prevent unhandled keys from propagating to QGIS defaults.
        event.ignore()

    def isActive(self) -> bool:
        return self._active
