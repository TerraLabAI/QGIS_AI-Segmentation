



















from __future__ import annotations

from qgis.core import QgsGeometry, QgsRectangle
from qgis.gui import QgsMapCanvasItem
from qgis.PyQt.QtCore import QPointF, Qt
from qgis.PyQt.QtGui import QBrush, QPainter, QPainterPath, QPen, QPolygonF

from ..core.qt_compat import resolve_qt_enum
from .canvas_palette import (
    HOVER_PREVIEW_DASH_PATTERN,
    HOVER_PREVIEW_EDGE,
    HOVER_PREVIEW_FILL,
    HOVER_PREVIEW_OUTLINE_WIDTH,
)

_RENDER_ANTIALIASING = resolve_qt_enum(QPainter, "RenderHint", "Antialiasing")
_FLAT_CAP = resolve_qt_enum(Qt, "PenCapStyle", "FlatCap")




_GHOST_PEN: QPen | None = None
_GHOST_BRUSH: QBrush | None = None


def ghost_outline_pen() -> QPen:








    global _GHOST_PEN
    if _GHOST_PEN is None:
        pen = QPen(HOVER_PREVIEW_EDGE)
        pen.setWidthF(HOVER_PREVIEW_OUTLINE_WIDTH)
        pen.setCosmetic(True)
        pen.setCapStyle(_FLAT_CAP)
        pen.setDashPattern(list(HOVER_PREVIEW_DASH_PATTERN))
        _GHOST_PEN = pen
    return _GHOST_PEN


def ghost_fill_brush() -> QBrush:


    global _GHOST_BRUSH
    if _GHOST_BRUSH is None:
        _GHOST_BRUSH = QBrush(HOVER_PREVIEW_FILL)
    return _GHOST_BRUSH


def preview_polygon_rings(geometry: QgsGeometry) -> list:






    if geometry.isMultipart():
        polygons = geometry.asMultiPolygon()
    else:
        polygons = [geometry.asPolygon()]
    rings = []
    for polygon in polygons:
        for ring in polygon:
            if ring:
                rings.append(ring)
    return rings


class HoverPreviewOverlay(QgsMapCanvasItem):


    def __init__(self, canvas) -> None:
        super().__init__(canvas)
        self._polygon: QgsGeometry | None = None
        self._ground: QgsRectangle | None = None



        self._rings: list = []
        self._path: QPainterPath | None = None
        self._path_at: QPointF | None = None
        self.hide()

    def show_preview_polygon(self, geometry: QgsGeometry,
                             ground: QgsRectangle) -> None:






        try:
            if geometry is None or geometry.isEmpty() or ground is None \
                    or ground.isEmpty():
                self.clear_preview()
                return
            self._polygon = QgsGeometry(geometry)
            self._forget_path()


            self._rings = preview_polygon_rings(self._polygon)
            self._ground = QgsRectangle(ground)
            self.setRect(self._ground)
            self.show()
            self.update()
        except Exception:  # noqa: BLE001
            self.clear_preview()

    def _forget_path(self) -> None:

        self._path = None
        self._path_at = None

    def clear_preview(self) -> None:

        self._polygon = None
        self._rings = []
        self._forget_path()
        self._ground = None
        try:
            self.hide()
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def has_preview(self) -> bool:

        return self._polygon is not None

    def updatePosition(self) -> None:  # noqa: N802





        try:

            self._forget_path()
            if self._ground is not None:
                self.setRect(self._ground)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _ghost_item_path(self, here) -> QPainterPath | None:







        path = self._path
        at = self._path_at
        if path is not None and at is not None \
                and at.x() == here.x() and at.y() == here.y():
            return path
        built = QPainterPath()
        for ring in self._rings:
            points = QPolygonF([
                QPointF(c.x() - here.x(), c.y() - here.y())
                for c in (self.toCanvasCoordinates(p) for p in ring)])
            if len(points) < 3:
                continue
            built.addPolygon(points)
            built.closeSubpath()
        if built.isEmpty():
            return None
        self._path = built
        self._path_at = QPointF(here)
        return built

    def paint(self, painter, option=None, widget=None) -> None:






        try:
            if self._ground is None or painter is None \
                    or self._polygon is None:
                return


            path = self._ghost_item_path(self.pos())
            if path is None:
                return
            painter.setRenderHint(_RENDER_ANTIALIASING, True)
            painter.setPen(ghost_outline_pen())
            painter.setBrush(ghost_fill_brush())
            painter.drawPath(path)
        except Exception:  # noqa: BLE001  # nosec B110
            return
