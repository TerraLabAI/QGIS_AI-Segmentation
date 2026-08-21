"""The translucent shape that follows the cursor before any click.

A canvas ITEM holding one shape over one ground rectangle, not a rubber band
and not a layer. Three reasons, and each one is why the obvious version was not
built:

- No layer. Nothing here reaches the project, so a preview can never be saved,
  exported, styled or left behind by a session that ended.
- No repaint of its own. The item is told once per answer and never schedules
  a second update, because a producer faster than the compositor turns every
  drawn frame into a merged one and the shape starts to lag the cursor.
- One form only. The controller hands it the FINAL polygon a click would
  produce, the shaped outline, so what glows is what a click saves. When that
  shape cannot be built, nothing is drawn at all: a ghost that does not match
  the click is worse than no ghost. The stroked outline is dashed, because the
  shape is not saved yet and the broken line says so.

The paint must never raise. An exception on a paint path kills QGIS at start
up, so every step here is inside a guard that draws nothing rather than throw.
"""
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


# The pen and the brush the ghost is drawn with, built on first paint and kept.
# Both are constant, and a paint runs far more often than an answer lands.
_GHOST_PEN: QPen | None = None
_GHOST_BRUSH: QBrush | None = None


def ghost_outline_pen() -> QPen:
    """The dashed rim every vector ghost wears, in device pixels.

    Cosmetic so a zoom never fattens it, flat-capped so the gaps stay gaps
    (the default square cap grows each dash by half a width and closes them).

    One shared pen: it is read on a paint path and never edited, and building
    a pen plus its dash list on every frame is work the compositor pays for.
    """
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
    """The translucent body of the ghost. Shared, like the pen and for the
    same reason."""
    global _GHOST_BRUSH
    if _GHOST_BRUSH is None:
        _GHOST_BRUSH = QBrush(HOVER_PREVIEW_FILL)
    return _GHOST_BRUSH


def preview_polygon_rings(geometry: QgsGeometry) -> list:
    """Every ring of ``geometry`` as one flat list of QgsPointXY lists.

    Exterior rings and holes together: a QPainterPath's default odd-even fill
    turns the holes back into holes, so a courtyard the shaping kept stays
    open in the ghost too.
    """
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
    """One preview shape, pinned to the ground rectangle it describes."""

    def __init__(self, canvas) -> None:
        super().__init__(canvas)
        self._polygon: QgsGeometry | None = None
        self._ground: QgsRectangle | None = None
        # The shape's rings, converted once when the polygon is set, and the
        # path they were last drawn as with the item position it was built at.
        # A paint runs far more often than either moves.
        self._rings: list = []
        self._path: QPainterPath | None = None
        self._path_at: QPointF | None = None
        self.hide()

    def show_preview_polygon(self, geometry: QgsGeometry,
                             ground: QgsRectangle) -> None:
        """Draw ``geometry`` (canvas CRS) as the ghost, over the crop window
        ``ground`` it was read from.

        One update, and only one: the caller answers a preview at most once per
        request, so nothing here needs to schedule a repaint of its own.
        """
        try:
            if geometry is None or geometry.isEmpty() or ground is None \
                    or ground.isEmpty():
                self.clear_preview()
                return
            self._polygon = QgsGeometry(geometry)
            self._forget_path()
            # Converted here and not on the paint path: the ring list is the
            # same for every frame the shape is up.
            self._rings = preview_polygon_rings(self._polygon)
            self._ground = QgsRectangle(ground)
            self.setRect(self._ground)
            self.show()
            self.update()
        except Exception:  # noqa: BLE001 -- a preview never breaks the canvas
            self.clear_preview()

    def _forget_path(self) -> None:
        """Drop the drawn path, so the next paint builds it from the rings."""
        self._path = None
        self._path_at = None

    def clear_preview(self) -> None:
        """Take the shape off the map. Never raises: teardown calls it."""
        self._polygon = None
        self._rings = []
        self._forget_path()
        self._ground = None
        try:
            self.hide()
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def has_preview(self) -> bool:
        """Whether a shape is on the map right now."""
        return self._polygon is not None

    def updatePosition(self) -> None:  # noqa: N802 (Qt API)
        """Re-place the shape after a pan or a zoom.

        The rectangle is ground, so the shape stays on its object while the map
        moves under it, which is exactly what a rubber band would do.
        """
        try:
            # The map transform moved, so every vertex lands somewhere else.
            self._forget_path()
            if self._ground is not None:
                self.setRect(self._ground)
        except Exception:  # noqa: BLE001 -- a stale item simply stops moving  # nosec B110
            pass

    def _ghost_item_path(self, here) -> QPainterPath | None:
        """The ghost as one odd-even path in item coordinates, or None when
        nothing of it can be placed.

        Built from the cached rings and kept until the map transform or the
        item position moves. Rebuilding it re-projects every vertex, and a
        paint runs many times per answer.
        """
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
        """Draw the shaped outline over its ground.

        Wrapped whole. An exception raised on a paint reaches Qt as an abort,
        and on a canvas item that is drawn at start up it takes QGIS with it,
        so every failure here draws nothing instead.
        """
        try:
            if self._ground is None or painter is None \
                    or self._polygon is None:
                return
            # Item coordinates: the item sits at the rectangle's own corner, so
            # every canvas position below is moved back by that offset.
            path = self._ghost_item_path(self.pos())
            if path is None:
                return
            painter.setRenderHint(_RENDER_ANTIALIASING, True)
            painter.setPen(ghost_outline_pen())
            painter.setBrush(ghost_fill_brush())
            painter.drawPath(path)
        except Exception:  # noqa: BLE001 -- a paint that raises kills QGIS  # nosec B110
            return
