



















from __future__ import annotations

from qgis.core import QgsPointXY
from qgis.gui import QgsMapTool
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QCursor

from ..core.qt_compat import event_pos
from .canvas_slide_pan import CanvasSlidePan


class PickObjectMapTool(QgsMapTool):














    point_clicked = pyqtSignal(QgsPointXY)
    point_right_clicked = pyqtSignal(QgsPointXY)
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
        self._slide_pan = CanvasSlidePan(canvas)

        self.suppress_deactivate_signal = False
        self._in_deactivate_emit = False
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))




    def canvasPressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.RightButton:




            event.accept()
            self.point_right_clicked.emit(self.toMapCoordinates(event_pos(event)))
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_screen = event_pos(event)
            self._panning = False
            event.accept()

    def canvasMoveEvent(self, event):  # noqa: N802
        pos = event_pos(event)
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
            self._press_screen = None
            event.accept()
            self.cancelled.emit()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            self.confirmed.emit()
        else:
            super().keyPressEvent(event)

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


        self._slide_pan.commit()

        super().deactivate()
        self._emit_deactivated()
