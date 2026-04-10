from __future__ import annotations

from qgis.core import QgsPointXY
from qgis.gui import QgsMapCanvas, QgsMapTool, QgsVertexMarker
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor


class AISegmentationMapTool(QgsMapTool):
    """Map tool for AI segmentation with positive/negative point prompts.

    Shortcuts:
    - G: Start segmentation (when not active)
    - Left click: Select this element (add positive point)
    - Right click: Refine selection (add negative point)
    - Ctrl+Z: Undo last point
    - S: Save mask (Batch mode only)
    - Enter: Export to layer
    - Hold Space + move: Pan the map (release to resume)
    """

    positive_click = pyqtSignal(QgsPointXY)
    negative_click = pyqtSignal(QgsPointXY)
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
        # Don't clear markers here — the plugin decides whether to keep them
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

    def get_marker_count(self) -> int:
        return len(self._markers)

    def canvasPressEvent(self, event):
        if not self._active:
            return

        # Ignore clicks while Space-panning
        if self._space_panning:
            return

        point = self.toMapCoordinates(event.pos())

        if event.button() == Qt.MouseButton.LeftButton:
            self.add_marker(point, is_positive=True)
            self.positive_click.emit(point)
        elif event.button() == Qt.MouseButton.RightButton:
            self.add_marker(point, is_positive=False)
            self.negative_click.emit(point)

    def canvasMoveEvent(self, event):
        if not self._space_panning:
            return

        current = event.pos()
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
        """Called when Space is pressed — enable temporary pan mode."""
        self._space_panning = True
        # Record current cursor position as pan reference
        self._pan_last_point = self.canvas.mapFromGlobal(QCursor.pos())
        self.canvas.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def stop_space_pan(self):
        """Called when Space is released — restore segmentation mode."""
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
