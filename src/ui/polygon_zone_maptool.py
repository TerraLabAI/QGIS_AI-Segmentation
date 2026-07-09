






















from __future__ import annotations

import logging

from qgis.core import QgsGeometry, QgsPointXY
from qgis.gui import QgsMapTool, QgsRubberBand, QgsVertexMarker
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor

from ..core.qt_compat import LineGeometry, PolygonGeometry, VertexIconCircle, event_pos
from .canvas_palette import (
    BADGE_X,
    CHROME_BLUE,
    CLOSE_DOT_OK,
    CLOSE_DOT_WARN,
)
from .canvas_slide_pan import CanvasSlidePan

logger = logging.getLogger("AISegmentation")


_BLUE = CHROME_BLUE
_TRANSPARENT = QColor(0, 0, 0, 0)

_CLOSE_GREEN = CLOSE_DOT_OK


class PolygonZoneMapTool(QgsMapTool):







    zone_selected = pyqtSignal(QgsGeometry)
    zone_cleared = pyqtSignal()
    vertices_changed = pyqtSignal(int)





    tool_deactivated = pyqtSignal()



    back_requested = pyqtSignal()

    MIN_VERTICES: int = 3


    MIN_STEP_PX: int = 6

    CLOSE_PX: int = 14

    def __init__(self, canvas, color: QColor | None = None):
        super().__init__(canvas)
        self._canvas = canvas
        self._points: list[QgsPointXY] = []



        base = color or _BLUE
        self._c_main = base
        self._c_line = QColor(base.red(), base.green(), base.blue(), 235)
        self._c_fill = QColor(base.red(), base.green(), base.blue(), 55)



        self._c_close = _CLOSE_GREEN if color is None else CLOSE_DOT_WARN


        self._fill_band = QgsRubberBand(canvas, PolygonGeometry)
        self._fill_band.setColor(self._c_fill)
        self._fill_band.setStrokeColor(_TRANSPARENT)
        self._fill_band.setWidth(0)

        self._edges_band = QgsRubberBand(canvas, LineGeometry)
        self._edges_band.setColor(self._c_line)
        self._edges_band.setWidth(3)

        self._preview_band = QgsRubberBand(canvas, LineGeometry)
        self._preview_band.setColor(self._c_line)
        self._preview_band.setWidth(2)
        self._preview_band.setLineStyle(Qt.PenStyle.DashLine)
        self._markers: list[QgsVertexMarker] = []
        self._can_close = False
        self._space_panning = False

        self.suppress_deactivate_signal = False
        self._in_deactivate_emit = False
        self._slide_pan = CanvasSlidePan(canvas)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))


    def start_space_pan(self) -> None:
        self._space_panning = True





        pos = self._canvas.mapFromGlobal(QCursor.pos())
        if self._canvas.rect().contains(pos):
            self._slide_pan.begin_at(pos)
        self._canvas.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def stop_space_pan(self) -> None:
        self._space_panning = False
        self._slide_pan.commit()



        if self.isActive():
            self._canvas.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def is_space_panning(self) -> bool:

        return self._space_panning


    def canvasMoveEvent(self, event):  # noqa: N802
        if self._space_panning:


            self._slide_pan.slide_to(event_pos(event))
            return
        if not self._points:
            return
        pos = event_pos(event)
        self._can_close = self._near_first(pos)
        cursor = self._points[0] if self._can_close else self.toMapCoordinates(pos)

        self._preview_band.setToGeometry(
            QgsGeometry.fromPolylineXY([self._points[-1], cursor]), None)


        self._draw_fill(self._points + [cursor])
        self._highlight_first(self._can_close)

    def canvasReleaseEvent(self, event):  # noqa: N802
        if self._space_panning:
            return
        if event.button() == Qt.MouseButton.RightButton:



            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event_pos(event)



        self._can_close = self._near_first(pos)
        if self._can_close:
            self._finish()
            return
        self._add_point(self.toMapCoordinates(pos), pos)

    def canvasDoubleClickEvent(self, event):  # noqa: N802


        if not self._space_panning and event.button() == Qt.MouseButton.LeftButton:
            self._finish()

    def keyPressEvent(self, event):  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Escape:



            self._cancel_or_back()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish()
        elif key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete) or (
            key == Qt.Key.Key_Z and (event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        ):
            self._undo_or_back()
        else:
            super().keyPressEvent(event)

    def _undo_or_back(self) -> None:


        if self._points:
            self._undo_last()
        else:
            self.back_requested.emit()

    def _cancel_or_back(self) -> None:



        if self._points:
            self._cancel()
        else:
            self.back_requested.emit()

    def has_points(self) -> bool:

        return bool(self._points)

    def finish(self) -> bool:







        if len(self._points) < self.MIN_VERTICES:
            return False
        self._finish()
        return True

    def undo_point(self) -> bool:








        if not self._points:
            return False
        self._undo_last()
        return True


    def _add_point(self, map_pt: QgsPointXY, screen_pt) -> None:
        if self._points:
            last_screen = self.toCanvasCoordinates(self._points[-1])
            dx_small = abs(screen_pt.x() - last_screen.x()) < self.MIN_STEP_PX
            dy_small = abs(screen_pt.y() - last_screen.y()) < self.MIN_STEP_PX
            if dx_small and dy_small:
                return
        self._points.append(map_pt)
        self._add_marker(map_pt, first=len(self._points) == 1)
        self._draw_edges()
        self._draw_fill(self._points)
        self.vertices_changed.emit(len(self._points))

    def _undo_last(self) -> None:
        if not self._points:
            return
        self._points.pop()
        if self._markers:
            marker = self._markers.pop()
            try:
                self._canvas.scene().removeItem(marker)
            except (RuntimeError, AttributeError):



                pass
        self._restyle_markers()
        self._draw_edges()
        self._draw_fill(self._points)
        self._preview_band.reset(LineGeometry)
        self.vertices_changed.emit(len(self._points))

    def _finish(self) -> None:
        pts = list(self._points)
        if len(pts) < self.MIN_VERTICES:
            return
        geom = QgsGeometry.fromPolygonXY([pts])
        self._reset_visuals()
        self._points = []
        if geom.isEmpty():
            self.vertices_changed.emit(0)
            self.zone_cleared.emit()
            return
        self.vertices_changed.emit(0)
        self.zone_selected.emit(geom)

    def _cancel(self) -> None:
        self._reset_visuals()
        self._points = []
        self.vertices_changed.emit(0)
        self.zone_cleared.emit()

    def clear_selection(self) -> None:
        self._cancel()

    def _draw_edges(self) -> None:



        if len(self._points) >= 2:
            self._edges_band.setToGeometry(
                QgsGeometry.fromPolylineXY(self._points), None)
        else:
            self._edges_band.reset(LineGeometry)

    def _draw_fill(self, pts: list) -> None:
        if len(pts) >= 3:
            self._fill_band.setToGeometry(
                QgsGeometry.fromPolygonXY([list(pts)]), None)
        else:
            self._fill_band.reset(PolygonGeometry)

    def _add_marker(self, pt: QgsPointXY, first: bool) -> None:
        m = QgsVertexMarker(self._canvas)
        m.setCenter(pt)
        m.setIconType(VertexIconCircle)
        m.setColor(self._c_main)
        try:

            m.setFillColor(BADGE_X)
        except (AttributeError, TypeError):
            pass
        m.setPenWidth(3)

        m.setIconSize(13 if first else 10)
        m.setZValue(1000)
        self._markers.append(m)

    def _restyle_markers(self) -> None:
        for i, m in enumerate(self._markers):
            m.setIconSize(13 if i == 0 else 10)
            m.setColor(self._c_main)

    def _highlight_first(self, hot: bool) -> None:
        if not self._markers or len(self._points) < self.MIN_VERTICES:
            return
        first = self._markers[0]
        first.setColor(self._c_close if hot else self._c_main)
        first.setIconSize(18 if hot else 13)

    def _near_first(self, screen_pt) -> bool:
        if len(self._points) < self.MIN_VERTICES:
            return False
        first_screen = self.toCanvasCoordinates(self._points[0])
        dx_close = abs(screen_pt.x() - first_screen.x()) <= self.CLOSE_PX
        dy_close = abs(screen_pt.y() - first_screen.y()) <= self.CLOSE_PX
        return dx_close and dy_close

    def _reset_visuals(self) -> None:
        self._edges_band.reset(LineGeometry)
        self._fill_band.reset(PolygonGeometry)
        self._preview_band.reset(LineGeometry)
        for m in self._markers:
            try:
                self._canvas.scene().removeItem(m)
            except (RuntimeError, AttributeError):
                pass
        self._markers = []
        self._can_close = False

    def remove_bands_from_canvas(self) -> None:







        for band in (self._edges_band, self._fill_band, self._preview_band):
            try:
                self._canvas.scene().removeItem(band)
            except (RuntimeError, AttributeError):
                pass

    def _emit_deactivated(self) -> None:



        if self.suppress_deactivate_signal or self._in_deactivate_emit:
            return
        self._in_deactivate_emit = True
        try:
            self.tool_deactivated.emit()
        finally:
            self._in_deactivate_emit = False

    def deactivate(self) -> None:
        had_points = bool(self._points)
        self._reset_visuals()
        self._points = []
        self._space_panning = False


        self._slide_pan.commit()
        super().deactivate()



        if had_points:
            self.vertices_changed.emit(0)
        self._emit_deactivated()
