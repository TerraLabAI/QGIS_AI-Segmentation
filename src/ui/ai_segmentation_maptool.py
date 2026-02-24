from typing import List

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QCursor, QColor
from qgis.gui import QgsMapTool, QgsMapCanvas, QgsVertexMarker
from qgis.core import QgsPointXY


class AISegmentationMapTool(QgsMapTool):
    """Map tool for AI segmentation with positive/negative point prompts.

    Shortcuts:
    - G: Start segmentation (when not active)
    - Left click: Select this element (add positive point)
    - Right click: Refine selection (add negative point)
    - Ctrl+Z: Undo last point
    - S: Save mask (Batch mode only)
    - Enter: Export to layer
    """

    positive_click = pyqtSignal(QgsPointXY)
    negative_click = pyqtSignal(QgsPointXY)
    tool_deactivated = pyqtSignal()
    undo_requested = pyqtSignal()
    save_polygon_requested = pyqtSignal()
    export_layer_requested = pyqtSignal()

    POSITIVE_COLOR = QColor(0, 200, 0)    # Green for include
    NEGATIVE_COLOR = QColor(220, 0, 0)    # Red for exclude
    MARKER_SIZE = 10
    MARKER_PEN_WIDTH = 2

    def __init__(self, canvas: QgsMapCanvas):
        super().__init__(canvas)
        self.canvas = canvas
        self._active = False
        self._markers: List[QgsVertexMarker] = []

    def activate(self):
        super().activate()
        self._active = True
        self.canvas.setCursor(QCursor(Qt.CrossCursor))

    def deactivate(self):
        super().deactivate()
        self._active = False
        self.clear_markers()
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

        point = self.toMapCoordinates(event.pos())

        if event.button() == Qt.LeftButton:
            # Positive point - include this area
            self.add_marker(point, is_positive=True)
            self.positive_click.emit(point)
        elif event.button() == Qt.RightButton:
            # Negative point - exclude this area
            self.add_marker(point, is_positive=False)
            self.negative_click.emit(point)

    def canvasMoveEvent(self, event):
        pass

    def canvasReleaseEvent(self, event):
        pass

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
        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key_Z and modifiers & Qt.ControlModifier:
            # Ctrl+Z: Undo last point
            self.undo_requested.emit()
            event.accept()
        elif key == Qt.Key_S and not (modifiers & (Qt.ControlModifier | Qt.AltModifier | Qt.ShiftModifier)):
            # Bare S only: Save polygon (add to collection)
            # With modifiers (Ctrl+S, etc.) let QGIS handle it
            self.save_polygon_requested.emit()
            event.accept()
        elif key == Qt.Key_Return or key == Qt.Key_Enter:
            # Enter: Export to layer
            self.export_layer_requested.emit()
            event.accept()
        else:
            event.ignore()

    def isActive(self) -> bool:
        return self._active
