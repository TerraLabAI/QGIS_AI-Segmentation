"""
Map Tool for AI Segmentation

Custom QGIS map tool that captures mouse clicks on the map canvas
for interactive segmentation prompts.
"""

from typing import List

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QCursor, QColor
from qgis.gui import QgsMapTool, QgsMapCanvas, QgsVertexMarker
from qgis.core import QgsPointXY


class AISegmentationMapTool(QgsMapTool):
    """
    Custom map tool for capturing segmentation prompts.

    Emits signals when the user clicks on the map:
    - Left click: positive prompt (include this region)
    - Right click: negative prompt (exclude this region)

    Provides visual feedback with QgsVertexMarker for each click.
    """

    # Signals
    positive_click = pyqtSignal(QgsPointXY)  # Left click - foreground
    negative_click = pyqtSignal(QgsPointXY)  # Right click - background
    tool_deactivated = pyqtSignal()
    undo_requested = pyqtSignal()  # Ctrl+Z pressed

    # Visual styling constants - smaller markers with outline only
    POSITIVE_COLOR = QColor(0, 200, 0)    # Green for positive points
    NEGATIVE_COLOR = QColor(220, 0, 0)    # Red for negative points
    MARKER_SIZE = 8  # Smaller size
    MARKER_PEN_WIDTH = 2

    def __init__(self, canvas: QgsMapCanvas):
        """
        Initialize the map tool.

        Args:
            canvas: QgsMapCanvas to attach the tool to
        """
        super().__init__(canvas)
        self.canvas = canvas
        self._active = False

        # Visual markers for clicked points
        self._markers: List[QgsVertexMarker] = []

    def activate(self):
        """Called when the tool is activated."""
        super().activate()
        self._active = True
        # Set crosshair cursor
        self.canvas.setCursor(QCursor(Qt.CrossCursor))

    def deactivate(self):
        """Called when the tool is deactivated."""
        super().deactivate()
        self._active = False
        self.clear_markers()  # Clean up visual markers
        self.tool_deactivated.emit()

    # ==================== Marker Management ====================

    def add_marker(self, point: QgsPointXY, is_positive: bool) -> QgsVertexMarker:
        """
        Add a visual marker at the clicked point.

        Args:
            point: Map coordinates of the click
            is_positive: True for positive/foreground (green), False for negative/background (red)

        Returns:
            The created QgsVertexMarker
        """
        marker = QgsVertexMarker(self.canvas)
        marker.setCenter(point)
        marker.setIconType(QgsVertexMarker.IconType.ICON_CIRCLE)
        marker.setIconSize(self.MARKER_SIZE)
        marker.setPenWidth(self.MARKER_PEN_WIDTH)

        if is_positive:
            marker.setColor(self.POSITIVE_COLOR)
            marker.setFillColor(QColor(0, 200, 0, 0))  # Transparent fill (outline only)
        else:
            marker.setColor(self.NEGATIVE_COLOR)
            marker.setFillColor(QColor(220, 0, 0, 0))  # Transparent fill (outline only)

        self._markers.append(marker)
        return marker

    def remove_last_marker(self) -> bool:
        """
        Remove the last added marker (for undo functionality).

        Returns:
            True if a marker was removed, False if no markers exist
        """
        if self._markers:
            marker = self._markers.pop()
            self.canvas.scene().removeItem(marker)
            self.canvas.refresh()
            return True
        return False

    def clear_markers(self):
        """Remove all visual markers from the canvas."""
        for marker in self._markers:
            self.canvas.scene().removeItem(marker)
        self._markers.clear()
        self.canvas.refresh()

    def get_marker_count(self) -> int:
        """Get the number of markers currently displayed."""
        return len(self._markers)

    def canvasPressEvent(self, event):
        """
        Handle mouse press events on the canvas.

        Args:
            event: QgsMapMouseEvent
        """
        if not self._active:
            return

        # Get the point in map coordinates
        point = self.toMapCoordinates(event.pos())

        # Check which button was pressed and add visual marker
        if event.button() == Qt.LeftButton:
            self.add_marker(point, is_positive=True)
            self.positive_click.emit(point)
        elif event.button() == Qt.RightButton:
            self.add_marker(point, is_positive=False)
            self.negative_click.emit(point)

    def canvasMoveEvent(self, event):
        """
        Handle mouse move events (for potential preview functionality).

        Args:
            event: QgsMapMouseEvent
        """
        # Could be used for real-time preview in future versions
        pass

    def canvasReleaseEvent(self, event):
        """
        Handle mouse release events.

        Args:
            event: QgsMapMouseEvent
        """
        pass

    def keyPressEvent(self, event):
        """
        Handle key press events.

        Args:
            event: QKeyEvent
        """
        if event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            # Ctrl+Z for undo - emit signal to main plugin
            self.undo_requested.emit()

    def isActive(self) -> bool:
        """Check if the tool is currently active."""
        return self._active
