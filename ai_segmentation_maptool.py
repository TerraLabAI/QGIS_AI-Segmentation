

from typing import List
from enum import Enum

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QCursor, QColor
from qgis.gui import QgsMapTool, QgsMapCanvas, QgsVertexMarker
from qgis.core import QgsPointXY


class ClickMode(Enum):
    """Click modes for segmentation workflow."""
    NEW = "new"           # Left click: create new standalone polygon
    ADD = "add"           # Ctrl+Left click: union/add to existing polygon  
    SUBTRACT = "subtract" # Shift+Left click: subtract from existing polygon


class AISegmentationMapTool(QgsMapTool):
    
    # Signal emitted with point and click mode (new, add, subtract)
    segmentation_click = pyqtSignal(QgsPointXY, str)
    
    # Legacy signals for compatibility
    positive_click = pyqtSignal(QgsPointXY)
    negative_click = pyqtSignal(QgsPointXY)
    
    tool_deactivated = pyqtSignal()
    undo_requested = pyqtSignal()
    save_polygon_requested = pyqtSignal()  

    # Marker colors
    NEW_COLOR = QColor(0, 120, 255)       # Blue for new polygon
    ADD_COLOR = QColor(0, 200, 0)         # Green for add/union
    SUBTRACT_COLOR = QColor(220, 0, 0)    # Red for subtract
    
    MARKER_SIZE = 8  
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

    def add_marker(self, point: QgsPointXY, mode: ClickMode) -> QgsVertexMarker:
        """Add a marker with color based on click mode."""
        marker = QgsVertexMarker(self.canvas)
        marker.setCenter(point)
        marker.setIconType(QgsVertexMarker.IconType.ICON_CIRCLE)
        marker.setIconSize(self.MARKER_SIZE)
        marker.setPenWidth(self.MARKER_PEN_WIDTH)

        if mode == ClickMode.NEW:
            marker.setColor(self.NEW_COLOR)
            marker.setFillColor(QColor(0, 120, 255, 80))
        elif mode == ClickMode.ADD:
            marker.setColor(self.ADD_COLOR)
            marker.setFillColor(QColor(0, 200, 0, 80))
        else:  # SUBTRACT
            marker.setColor(self.SUBTRACT_COLOR)
            marker.setFillColor(QColor(220, 0, 0, 80))

        self._markers.append(marker)
        return marker

    def remove_last_marker(self) -> bool:
        
        if self._markers:
            marker = self._markers.pop()
            self.canvas.scene().removeItem(marker)
            self.canvas.refresh()
            return True
        return False

    def clear_markers(self):
        
        for marker in self._markers:
            self.canvas.scene().removeItem(marker)
        self._markers.clear()
        self.canvas.refresh()

    def get_marker_count(self) -> int:
        
        return len(self._markers)

    def canvasPressEvent(self, event):
        
        if not self._active:
            return

        point = self.toMapCoordinates(event.pos())
        modifiers = event.modifiers()

        if event.button() == Qt.LeftButton:
            # Determine click mode based on modifiers
            if modifiers & Qt.ControlModifier:
                mode = ClickMode.ADD
            elif modifiers & Qt.ShiftModifier:
                mode = ClickMode.SUBTRACT
            else:
                mode = ClickMode.NEW
            
            self.add_marker(point, mode)
            self.segmentation_click.emit(point, mode.value)
            
            # Also emit legacy signals for compatibility
            if mode in (ClickMode.NEW, ClickMode.ADD):
                self.positive_click.emit(point)
            else:
                self.negative_click.emit(point)
                
        elif event.button() == Qt.RightButton:
            # Right click = undo last action
            self.undo_requested.emit()

    def canvasMoveEvent(self, event):
        
        pass

    def canvasReleaseEvent(self, event):
        
        pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo_requested.emit()
        elif event.key() == Qt.Key_S and not event.modifiers():
            self.save_polygon_requested.emit()
        elif event.key() == Qt.Key_Escape:
            # Escape to deactivate tool
            self.canvas.unsetMapTool(self)

    def isActive(self) -> bool:
        
        return self._active
