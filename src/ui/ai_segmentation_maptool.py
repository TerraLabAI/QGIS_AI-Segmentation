from __future__ import annotations

from qgis.core import QgsPointXY
from qgis.gui import QgsMapCanvas, QgsMapTool, QgsVertexMarker
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor

from ..core.qt_compat import event_pos
from .canvas_palette import EXCLUDE_STROKE, EXEMPLAR_STROKE
from .canvas_slide_pan import CanvasSlidePan





_MARKER_FILL_ALPHA = 100


def _marker_fill(stroke: QColor) -> QColor:
    fill = QColor(stroke)
    fill.setAlpha(_MARKER_FILL_ALPHA)
    return fill


class AISegmentationMapTool(QgsMapTool):


















    positive_click = pyqtSignal(QgsPointXY)
    negative_click = pyqtSignal(QgsPointXY)



    double_click = pyqtSignal(QgsPointXY)


    cursor_moved = pyqtSignal(QgsPointXY)
    tool_deactivated = pyqtSignal()

    POSITIVE_COLOR = QColor(EXEMPLAR_STROKE)
    NEGATIVE_COLOR = QColor(EXCLUDE_STROKE)
    MARKER_SIZE = 10
    MARKER_PEN_WIDTH = 2

    def __init__(self, canvas: QgsMapCanvas):
        super().__init__(canvas)
        self.canvas = canvas
        self._active = False
        self._markers: list[QgsVertexMarker] = []
        self._space_panning = False
        self._slide_pan = CanvasSlidePan(canvas)



        self.last_click_modifiers = Qt.KeyboardModifier.NoModifier

    def activate(self):
        super().activate()
        self._active = True
        self._space_panning = False
        self._slide_pan.abandon()
        self.canvas.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def deactivate(self):
        super().deactivate()
        self._active = False
        self._space_panning = False


        self._slide_pan.commit()



        try:
            self.canvas.unsetCursor()
        except RuntimeError:
            pass


        self.tool_deactivated.emit()

    def add_marker(self, point: QgsPointXY, is_positive: bool) -> QgsVertexMarker:

        marker = QgsVertexMarker(self.canvas)
        marker.setCenter(point)
        marker.setIconSize(self.MARKER_SIZE)
        marker.setPenWidth(self.MARKER_PEN_WIDTH)

        if is_positive:
            marker.setIconType(QgsVertexMarker.IconType.ICON_CIRCLE)
            marker.setColor(self.POSITIVE_COLOR)
            marker.setFillColor(_marker_fill(self.POSITIVE_COLOR))
        else:
            marker.setIconType(QgsVertexMarker.IconType.ICON_X)
            marker.setColor(self.NEGATIVE_COLOR)
            marker.setFillColor(_marker_fill(self.NEGATIVE_COLOR))

        self._markers.append(marker)
        return marker

    def remove_last_marker(self) -> bool:

        if self._markers:
            marker = self._markers.pop()
            try:
                scene = self.canvas.scene()
                if scene is not None:
                    scene.removeItem(marker)
            except RuntimeError:
                pass
            try:





                self.canvas.update()
            except RuntimeError:
                pass
            return True
        return False

    def clear_markers(self):

        for marker in self._markers:
            try:
                scene = self.canvas.scene()
                if scene is not None:
                    scene.removeItem(marker)
            except RuntimeError:
                pass
        self._markers.clear()
        try:


            self.canvas.update()
        except RuntimeError:
            pass

    def canvasPressEvent(self, event):
        if not self._active:
            return


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



        self._slide_pan.slide_to(event_pos(event))

    def start_space_pan(self):

        self._space_panning = True





        pos = self.canvas.mapFromGlobal(QCursor.pos())
        if self.canvas.rect().contains(pos):
            self._slide_pan.begin_at(pos)
        self.canvas.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def stop_space_pan(self):

        self._space_panning = False
        self._slide_pan.commit()
        if self._active:
            self.canvas.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def is_space_panning(self) -> bool:

        return self._space_panning

    def wheelEvent(self, event):


        event.ignore()

    def gestureEvent(self, event):



        event.ignore()
        return False

    def keyPressEvent(self, event):



        event.ignore()

    def isActive(self) -> bool:
        return self._active
