"""
Map Tool for AI Segmentation

Custom QGIS map tool that captures mouse clicks on the map canvas
for interactive segmentation prompts.
"""

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QCursor
from qgis.gui import QgsMapTool, QgsMapCanvas
from qgis.core import QgsPointXY


class AISegmentationMapTool(QgsMapTool):
    """
    Custom map tool for capturing segmentation prompts.

    Emits signals when the user clicks on the map:
    - Left click: positive prompt (include this region)
    - Right click: negative prompt (exclude this region)
    """

    # Signals
    positive_click = pyqtSignal(QgsPointXY)  # Left click - foreground
    negative_click = pyqtSignal(QgsPointXY)  # Right click - background
    tool_deactivated = pyqtSignal()

    def __init__(self, canvas: QgsMapCanvas):
        """
        Initialize the map tool.

        Args:
            canvas: QgsMapCanvas to attach the tool to
        """
        super().__init__(canvas)
        self.canvas = canvas
        self._active = False

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
        self.tool_deactivated.emit()

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

        # Check which button was pressed
        if event.button() == Qt.LeftButton:
            self.positive_click.emit(point)
        elif event.button() == Qt.RightButton:
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
        if event.key() == Qt.Key_Escape:
            # Deactivate tool on Escape
            self.canvas.unsetMapTool(self)
        elif event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            # Ctrl+Z for undo - handled by the main plugin
            pass

    def isActive(self) -> bool:
        """Check if the tool is currently active."""
        return self._active
