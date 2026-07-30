"""The Correct step's gesture plate: a drawn preview of what each fix method
does to a polygon.

AI and Manual used to differ by one 17px glyph and one sentence, on two pages
built from the same widgets, so the switch looked like it changed nothing. This
draws the gesture instead of describing it: the same outline under both
methods, marked the way the map will really mark it.

Every colour comes from ``canvas_palette``, and it is the colour the user is
about to see on the canvas: the AI plate carries the keep/trim point markers of
the on-device refine session, the Manual plate carries the amber vertex handles
of the QGIS edit session. So the plate is a preview, not an illustration, and
it introduces no hue of its own.
"""

from __future__ import annotations

from qgis.PyQt.QtCore import QPointF, QRectF, Qt
from qgis.PyQt.QtGui import QColor, QPainter, QPainterPath, QPen
from qgis.PyQt.QtWidgets import QWidget

from ..canvas_palette import (
    BADGE_FILL_HOVER as VERTEX_HELD,  # the palette's brand red
)
from ..canvas_palette import (
    CHROME_BLUE,
    MARKER_NEGATIVE,
    MARKER_POSITIVE,
    SHAPE_EDIT_MARKER_FILL,
)
from .font_scale import scale_px_length

# The mouse arrow, in its own 0..1 box with the tip at the origin. Both plates
# carry it: the gesture is something the USER does, and an outline with marks
# on it does not say who put them there.
_CURSOR: tuple[tuple[float, float], ...] = (
    (0.0, 0.0), (0.0, 0.90), (0.24, 0.69), (0.40, 1.0),
    (0.55, 0.93), (0.39, 0.63), (0.68, 0.60),
)
_CURSOR_PX: float = 15.0

# One parcel-shaped outline, in 0..1 of the plate box, drawn by both methods.
# An irregular hexagon rather than a rectangle: a rectangle reads as an icon,
# this reads as a detected object, which is what the user is about to click.
_OUTLINE: tuple[tuple[float, float], ...] = (
    (0.05, 0.30), (0.42, 0.06), (0.74, 0.26),
    (0.71, 0.63), (0.38, 0.92), (0.08, 0.70),
)

# Where the AI plate puts its two markers: one keep point in the bay the model
# under-cut, one trim point on the lobe it over-cut. Left as fractions of the
# same box so both plates stay registered on the one outline.
_KEEP_POINT: tuple[float, float] = (0.29, 0.45)
_TRIM_POINT: tuple[float, float] = (0.62, 0.60)

# The Manual plate drags this vertex (index into _OUTLINE) by this offset, and
# leaves a dashed tether back to where it was.
_DRAGGED_VERTEX: int = 2
_DRAG_OFFSET: tuple[float, float] = (0.12, -0.13)

_PLATE_W: int = 132
_PLATE_H: int = 62


class CorrectGestureArt(QWidget):
    """A fixed-height plate drawing the current fix method's gesture.

    ``paintEvent`` must never raise (an exception there can take QGIS down at
    startup), so every path here is arithmetic on constants with no lookup that
    can fail.
    """

    def __init__(self, method: str = "ai", parent=None) -> None:
        super().__init__(parent)
        self._method = "manual" if str(method) == "manual" else "ai"
        self.setFixedHeight(scale_px_length(_PLATE_H))
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def method(self) -> str:
        return self._method

    def set_method(self, method: str) -> None:
        method = "manual" if str(method) == "manual" else "ai"
        if method == self._method:
            return
        self._method = method
        self.update()

    # ------------------------------------------------------------------

    def _box(self) -> QRectF:
        """The square-ish drawing box, centred in the widget."""
        w = float(scale_px_length(_PLATE_W))
        h = float(self.height()) - 8.0
        x = (float(self.width()) - w) / 2.0
        return QRectF(max(0.0, x), 4.0, w, max(1.0, h))

    def _pt(self, box: QRectF, fx: float, fy: float) -> QPointF:
        return QPointF(box.x() + fx * box.width(), box.y() + fy * box.height())

    def _outline_path(self, box: QRectF, points) -> QPainterPath:
        path = QPainterPath()
        first = True
        for fx, fy in points:
            p = self._pt(box, fx, fy)
            if first:
                path.moveTo(p)
                first = False
            else:
                path.lineTo(p)
        path.closeSubpath()
        return path

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            box = self._box()
            if self._method == "manual":
                self._paint_manual(painter, box)
            else:
                self._paint_ai(painter, box)
            painter.end()
        except Exception:  # noqa: BLE001 - a paint fault must never reach Qt
            return

    # ------------------------------------------------------------------

    def _paint_outline(self, painter: QPainter, box: QRectF, points,
                       colour: QColor, width: float = 1.6) -> None:
        fill = QColor(colour)
        fill.setAlpha(26)
        painter.setPen(QPen(colour, width, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(fill)
        painter.drawPath(self._outline_path(box, points))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_cursor(self, painter: QPainter, tip: QPointF) -> None:
        """The mouse arrow, tip at ``tip``. White body, dark outline, so it
        reads on the tinted outline and on either QGIS theme."""
        path = QPainterPath()
        first = True
        for fx, fy in _CURSOR:
            p = QPointF(tip.x() + fx * _CURSOR_PX, tip.y() + fy * _CURSOR_PX)
            if first:
                path.moveTo(p)
                first = False
            else:
                path.lineTo(p)
        path.closeSubpath()
        painter.setPen(QPen(QColor(20, 20, 20, 230), 1.1,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                            Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(QColor(255, 255, 255))
        painter.drawPath(path)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_ai(self, painter: QPainter, box: QRectF) -> None:
        """Point at it: the outline, plus one keep marker and one trim marker,
        drawn as the refine session draws them on the canvas, with the cursor
        landing on the keep point."""
        self._paint_outline(painter, box, _OUTLINE, CHROME_BLUE)
        r = max(4.0, box.height() * 0.125)
        for (fx, fy), colour, sign in (
                (_KEEP_POINT, MARKER_POSITIVE, 1),
                (_TRIM_POINT, MARKER_NEGATIVE, -1)):
            c = self._pt(box, fx, fy)
            # A soft ring under the marker so it sits ON the outline rather
            # than floating over it.
            halo = QColor(colour)
            halo.setAlpha(60)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(halo)
            painter.drawEllipse(c, r * 1.55, r * 1.55)
            painter.setPen(QPen(QColor(255, 255, 255, 230), 1.4))
            painter.setBrush(QColor(colour))
            painter.drawEllipse(c, r, r)
            # The sign inside the marker, painted rather than typed: a glyph
            # this small depends on the user's font having it.
            bar = r * 0.52
            painter.setPen(QPen(QColor(255, 255, 255), 1.8,
                                Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(c.x() - bar, c.y()),
                             QPointF(c.x() + bar, c.y()))
            if sign > 0:
                painter.drawLine(QPointF(c.x(), c.y() - bar),
                                 QPointF(c.x(), c.y() + bar))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        keep = self._pt(box, *_KEEP_POINT)
        self._paint_cursor(painter, QPointF(keep.x() + r * 0.62,
                                            keep.y() + r * 0.58))

    def _paint_manual(self, painter: QPainter, box: QRectF) -> None:
        """Move the corners: the outline with one vertex dragged clear, every
        vertex wearing the square handle the QGIS edit session gives it."""
        moved = list(_OUTLINE)
        ox, oy = _OUTLINE[_DRAGGED_VERTEX]
        moved[_DRAGGED_VERTEX] = (ox + _DRAG_OFFSET[0], oy + _DRAG_OFFSET[1])

        # Where the vertex came from: a hollow ghost handle plus a dashed
        # tether, so the drag reads as a drag and not as a stray red mark.
        ghost = QColor(VERTEX_HELD)
        ghost.setAlpha(150)
        pen = QPen(ghost, 1.3, Qt.PenStyle.DashLine)
        pen.setDashPattern([2.6, 2.4])
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        origin = self._pt(box, ox, oy)
        painter.drawLine(origin, self._pt(box, *moved[_DRAGGED_VERTEX]))

        half = max(2.5, box.height() * 0.068)
        painter.setPen(QPen(ghost, 1.2))
        painter.drawRect(QRectF(origin.x() - half, origin.y() - half,
                                half * 2.0, half * 2.0))

        self._paint_outline(painter, box, moved, CHROME_BLUE)

        for i, (fx, fy) in enumerate(moved):
            c = self._pt(box, fx, fy)
            held = i == _DRAGGED_VERTEX
            painter.setPen(QPen(VERTEX_HELD if held else CHROME_BLUE, 1.4))
            painter.setBrush(VERTEX_HELD if held
                             else QColor(SHAPE_EDIT_MARKER_FILL))
            h = half * (1.35 if held else 1.0)
            painter.drawRect(QRectF(c.x() - h, c.y() - h, h * 2.0, h * 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        held_pt = self._pt(box, *moved[_DRAGGED_VERTEX])
        self._paint_cursor(painter, QPointF(held_pt.x() + half * 0.3,
                                            held_pt.y() + half * 0.3))
