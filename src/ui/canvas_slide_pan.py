
























from __future__ import annotations

from qgis.core import QgsPointXY



_CANVAS_GONE = (RuntimeError, AttributeError, TypeError)


class CanvasSlidePan:


    def __init__(self, canvas) -> None:
        self._canvas = canvas
        self._anchor = None
        self._last = None

    @property
    def sliding(self) -> bool:

        return self._anchor is not None

    def begin_at(self, screen_point) -> None:

        self._anchor = screen_point
        self._last = screen_point

    def slide_to(self, screen_point) -> None:





        anchor = self._anchor
        if anchor is None:
            self.begin_at(screen_point)
            return
        self._last = screen_point
        try:
            size = self._canvas.viewport().size()
            self._canvas.setSceneRect(
                anchor.x() - screen_point.x(), anchor.y() - screen_point.y(),
                size.width(), size.height())
        except _CANVAS_GONE:
            pass

    def commit(self, screen_point=None) -> None:









        anchor = self._anchor
        last = screen_point if screen_point is not None else self._last
        self._anchor = None
        self._last = None
        if anchor is None:
            return
        self.square_the_scene()
        if last is None or (last.x() == anchor.x() and last.y() == anchor.y()):
            return
        try:
            transform = self._canvas.getCoordinateTransform()
            start = transform.toMapCoordinates(anchor.x(), anchor.y())
            end = transform.toMapCoordinates(last.x(), last.y())
            center = self._canvas.center()
            self._canvas.setCenter(QgsPointXY(
                center.x() + start.x() - end.x(),
                center.y() + start.y() - end.y(),
            ))
            self._canvas.refresh()
        except _CANVAS_GONE:
            pass

    def abandon(self) -> None:


        self._anchor = None
        self._last = None
        self.square_the_scene()

    def square_the_scene(self) -> None:



        try:
            size = self._canvas.viewport().size()
            self._canvas.setSceneRect(0, 0, size.width(), size.height())
        except _CANVAS_GONE:
            pass
