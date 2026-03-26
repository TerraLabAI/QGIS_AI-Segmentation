"""Map tool for drawing a rectangular zone selection on the QGIS canvas."""

from qgis.core import QgsPointXY, QgsRectangle, QgsWkbTypes
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor


class ZoneSelectionMapTool(QgsMapTool):
    """Allows the user to draw a rectangle on the map canvas.

    Emits zone_selected(QgsRectangle) when the user finishes drawing.
    Emits zone_cleared() when the selection is reset.
    """

    zone_selected = pyqtSignal(QgsRectangle)
    zone_cleared = pyqtSignal()

    def __init__(self, canvas):
        super().__init__(canvas)
        self._canvas = canvas
        self._rubber_band = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
        self._rubber_band.setColor(QColor(65, 105, 225, 80))
        self._rubber_band.setStrokeColor(QColor(65, 105, 225, 200))
        self._rubber_band.setWidth(2)
        self._start_point = None
        self._is_drawing = False

    def canvasPressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._start_point = self.toMapCoordinates(event.pos())
            self._is_drawing = True
            self._rubber_band.reset(QgsWkbTypes.PolygonGeometry)

    def canvasMoveEvent(self, event):
        if not self._is_drawing or self._start_point is None:
            return
        end_point = self.toMapCoordinates(event.pos())
        self._update_rubber_band(self._start_point, end_point)

    def canvasReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._is_drawing:
            self._is_drawing = False
            end_point = self.toMapCoordinates(event.pos())
            rect = QgsRectangle(self._start_point, end_point)
            rect.normalize()
            if rect.width() > 0 and rect.height() > 0:
                self.zone_selected.emit(rect)
            else:
                self.zone_cleared.emit()

    def _update_rubber_band(self, start, end):
        self._rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        self._rubber_band.addPoint(QgsPointXY(start.x(), start.y()), False)
        self._rubber_band.addPoint(QgsPointXY(end.x(), start.y()), False)
        self._rubber_band.addPoint(QgsPointXY(end.x(), end.y()), False)
        self._rubber_band.addPoint(QgsPointXY(start.x(), end.y()), True)

    def clear_selection(self):
        """Remove the rectangle from the canvas."""
        self._rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        self._start_point = None
        self.zone_cleared.emit()

    def deactivate(self):
        self._rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        super().deactivate()
