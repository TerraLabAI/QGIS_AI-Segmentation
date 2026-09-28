






















from __future__ import annotations

from qgis.core import QgsGeometry, QgsPointXY
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QCursor

from ..core.qt_compat import PolygonGeometry, event_pos
from .canvas_slide_pan import CanvasSlidePan


class PickObjectMapTool(QgsMapTool):

















    point_clicked = pyqtSignal(QgsPointXY)
    point_right_clicked = pyqtSignal(QgsPointXY)
    rect_right_dragged = pyqtSignal(QgsGeometry)
    cursor_moved = pyqtSignal(QgsPointXY)
    confirmed = pyqtSignal()
    cancelled = pyqtSignal()
    tool_deactivated = pyqtSignal()



    MAX_CLICK_DRIFT_PX: int = 4

    def __init__(self, canvas):
        super().__init__(canvas)
        self._canvas = canvas
        self._press_screen = None
        self._panning = False
        self._right_press = None
        self._right_band = None
        self._right_dragged = False
        self._slide_pan = CanvasSlidePan(canvas)

        self.suppress_deactivate_signal = False
        self._in_deactivate_emit = False
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))




    def canvasPressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.RightButton:



            event.accept()
            self._right_press = event_pos(event)
            self._right_dragged = False
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_screen = event_pos(event)
            self._panning = False
            event.accept()

    def canvasMoveEvent(self, event):  # noqa: N802
        pos = event_pos(event)
        rpress = self._right_press
        if rpress is not None and not (event.buttons() & Qt.MouseButton.RightButton):


            self._drop_right_drag()
            rpress = None
        if rpress is not None:
            if not self._right_dragged and (
                    abs(pos.x() - rpress.x()) > self.MAX_CLICK_DRIFT_PX
                    or abs(pos.y() - rpress.y()) > self.MAX_CLICK_DRIFT_PX):
                self._right_dragged = True

                if self.receivers(self.rect_right_dragged) > 0:
                    self._right_band = self._make_removal_band()
            if self._right_band is not None:
                self._right_band.setToGeometry(self._right_area(rpress, pos), None)
            if self._right_dragged:
                return
        press = self._press_screen
        if press is not None:
            drift_x = abs(pos.x() - press.x())
            drift_y = abs(pos.y() - press.y())
            if not self._panning and (drift_x > self.MAX_CLICK_DRIFT_PX or drift_y > self.MAX_CLICK_DRIFT_PX):
                self._panning = True



                self._slide_pan.begin_at(press)
            if self._panning:
                self._slide_pan.slide_to(pos)
                return
        try:
            self.cursor_moved.emit(self.toMapCoordinates(pos))
        except RuntimeError:
            pass

    def canvasReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.RightButton:
            self._release_right(event)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        press = self._press_screen
        self._press_screen = None
        if press is None:
            return
        event.accept()
        if self._panning:
            self._panning = False
            self._slide_pan.commit(event_pos(event))
            return
        pos = event_pos(event)
        drift_x = abs(pos.x() - press.x())
        drift_y = abs(pos.y() - press.y())
        if drift_x > self.MAX_CLICK_DRIFT_PX or drift_y > self.MAX_CLICK_DRIFT_PX:
            return
        self.point_clicked.emit(self.toMapCoordinates(pos))

    def canvasDoubleClickEvent(self, event):  # noqa: N802




        if event.button() == Qt.MouseButton.LeftButton:
            self._press_screen = None
            event.accept()

    def wheelEvent(self, event):  # noqa: N802

        event.ignore()

    def gestureEvent(self, event):  # noqa: N802

        event.ignore()
        return False

    def keyPressEvent(self, event):  # noqa: N802


        if event.key() == Qt.Key.Key_Escape:
            if self.cancel_right_drag():
                event.accept()
                return
            self._press_screen = None
            event.accept()
            self.cancelled.emit()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            self.confirmed.emit()
        else:
            super().keyPressEvent(event)


    def _make_removal_band(self):
        from .canvas_palette import EXCLUDE_FILL, EXCLUDE_STROKE
        band = QgsRubberBand(self._canvas, PolygonGeometry)
        band.setFillColor(EXCLUDE_FILL)
        band.setStrokeColor(EXCLUDE_STROKE)
        band.setWidth(2)
        return band

    def _right_area(self, a, b) -> QgsGeometry:

        from qgis.PyQt.QtCore import QPoint
        corners = [QPoint(a.x(), a.y()), QPoint(b.x(), a.y()),
                   QPoint(b.x(), b.y()), QPoint(a.x(), b.y())]
        ring = [QgsPointXY(self.toMapCoordinates(c)) for c in corners]
        return QgsGeometry.fromPolygonXY([ring + [ring[0]]])

    def right_drag_in_progress(self) -> bool:
        return self._right_press is not None and self._right_dragged

    def cancel_right_drag(self) -> bool:



        if not self.right_drag_in_progress():
            return False
        self._drop_right_drag()
        return True

    def _drop_right_drag(self) -> None:
        self._right_press = None
        self._right_dragged = False
        band = self._right_band
        self._right_band = None
        if band is not None:
            try:
                band.reset(PolygonGeometry)
                self._canvas.scene().removeItem(band)
            except (RuntimeError, AttributeError):
                pass

    def _release_right(self, event) -> None:
        press = self._right_press
        if press is None:
            return
        event.accept()
        dragged = self._right_dragged
        banded = self._right_band is not None
        pos = event_pos(event)
        self._drop_right_drag()


        if dragged:
            if banded:
                self.rect_right_dragged.emit(self._right_area(press, pos))
            return
        self.point_right_clicked.emit(self.toMapCoordinates(press))

    def _emit_deactivated(self) -> None:



        if self.suppress_deactivate_signal or self._in_deactivate_emit:
            return
        self._in_deactivate_emit = True
        try:
            self.tool_deactivated.emit()
        finally:
            self._in_deactivate_emit = False

    def deactivate(self) -> None:  # noqa: N802
        self._press_screen = None
        self._panning = False
        self._drop_right_drag()


        self._slide_pan.commit()
        super().deactivate()
        self._emit_deactivated()
