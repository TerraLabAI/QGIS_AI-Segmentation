

from typing import List

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QCursor, QColor
from qgis.gui import QgsMapTool, QgsMapCanvas, QgsVertexMarker
from qgis.core import QgsPointXY


class AISegmentationMapTool(QgsMapTool):
    

    positive_click = pyqtSignal(QgsPointXY)
    negative_click = pyqtSignal(QgsPointXY)
    tool_deactivated = pyqtSignal()
    undo_requested = pyqtSignal()
    save_polygon_requested = pyqtSignal()  

    POSITIVE_COLOR = QColor(0, 200, 0)    
    NEGATIVE_COLOR = QColor(220, 0, 0)    
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


    def add_marker(self, point: QgsPointXY, is_positive: bool) -> QgsVertexMarker:
        
        marker = QgsVertexMarker(self.canvas)
        marker.setCenter(point)
        marker.setIconType(QgsVertexMarker.IconType.ICON_CIRCLE)
        marker.setIconSize(self.MARKER_SIZE)
        marker.setPenWidth(self.MARKER_PEN_WIDTH)

        if is_positive:
            marker.setColor(self.POSITIVE_COLOR)
            marker.setFillColor(QColor(0, 200, 0, 0))  
        else:
            marker.setColor(self.NEGATIVE_COLOR)
            marker.setFillColor(QColor(220, 0, 0, 0))  

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

        if event.button() == Qt.LeftButton:
            self.add_marker(point, is_positive=True)
            self.positive_click.emit(point)
        elif event.button() == Qt.RightButton:
            self.add_marker(point, is_positive=False)
            self.negative_click.emit(point)

    def canvasMoveEvent(self, event):
        
        pass

    def canvasReleaseEvent(self, event):
        
        pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo_requested.emit()
        elif event.key() == Qt.Key_S and not event.modifiers():
            self.save_polygon_requested.emit()

    def isActive(self) -> bool:
        
        return self._active
