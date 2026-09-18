# SPDX-FileCopyrightText: 2026 TerraLab <yvann.barbot@terra-lab.ai>
# SPDX-License-Identifier: GPL-2.0-or-later












from __future__ import annotations

import math
import os

from qgis.PyQt.QtCore import QPointF, QRectF, QSize, Qt
from qgis.PyQt.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QPolygonF,
)

from .dock.font_scale import widget_pixel_ratio


BOX = 20.0

_ICON_CACHE: dict = {}


_ICON_CACHE_MAX = 512


_PIXMAP_CACHE: dict = {}
_PIXMAP_CACHE_MAX = 512



_LOGO_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "resources", "icons", "mark@4x.png")


LOGO_ASPECT = 247 / 182
_LOGO_CACHE: dict = {}


def _relative_luminance(colour: QColor) -> float:
    def channel(value: int) -> float:
        v = value / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    return (0.2126 * channel(colour.red()) + 0.7152 * channel(colour.green())
            + 0.0722 * channel(colour.blue()))


def contrast_ratio(a: QColor, b: QColor) -> float:

    high, low = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)




_MIN_GLYPH_CONTRAST = 3.0


def _palette_ink(widget) -> QColor:
    try:
        return widget.palette().color(QPalette.ColorRole.WindowText)
    except (AttributeError, RuntimeError):
        from qgis.PyQt.QtWidgets import QApplication

        app = QApplication.instance()
        palette = app.palette() if app is not None else QPalette()
        return palette.color(QPalette.ColorRole.WindowText)


def ink_of(widget) -> QColor:










    colour = _palette_ink(widget)
    try:
        from .dock import styles as style

        if contrast_ratio(colour, QColor(style.SURFACE)) < _MIN_GLYPH_CONTRAST:
            return QColor(style.INK)
    except Exception:  # nosec B110
        pass
    return colour


def paper_of(widget) -> QColor:


    try:
        return widget.palette().color(QPalette.ColorRole.Base)
    except (AttributeError, RuntimeError):
        from qgis.PyQt.QtWidgets import QApplication

        app = QApplication.instance()
        palette = app.palette() if app is not None else QPalette()
        return palette.color(QPalette.ColorRole.Base)


def _pen(color: QColor, width: float = 1.8) -> QPen:
    pen = QPen(color)
    pen.setWidthF(width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen




def _draw_send(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 2.2))
    p.drawLine(QPointF(10, 16), QPointF(10, 5))
    p.drawPolyline(QPolygonF([QPointF(5, 10), QPointF(10, 5), QPointF(15, 10)]))


def _draw_stop(p: QPainter, c: QColor) -> None:

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawRoundedRect(QRectF(5.5, 5.5, 9, 9), 1.5, 1.5)


def _draw_plus(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 2.0))
    p.drawLine(QPointF(10, 4.5), QPointF(10, 15.5))
    p.drawLine(QPointF(4.5, 10), QPointF(15.5, 10))


def _draw_plus_circle(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QRectF(2.5, 2.5, 15, 15))
    p.drawLine(QPointF(10, 6.5), QPointF(10, 13.5))
    p.drawLine(QPointF(6.5, 10), QPointF(13.5, 10))


def _draw_clock(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QRectF(2.5, 2.5, 15, 15))
    p.drawPolyline(QPolygonF([QPointF(10, 5.5), QPointF(10, 10.5), QPointF(13.5, 12.5)]))


def _draw_kebab(p: QPainter, c: QColor) -> None:

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    for y in (4.5, 10.0, 15.5):
        p.drawEllipse(QPointF(10, y), 1.7, 1.7)


def _draw_lock(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.8))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(QRectF(6.0, 3.0, 8.0, 8.0), 0, 180 * 16)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    p.drawRoundedRect(QRectF(4.5, 8.5, 11.0, 8.5), 2.0, 2.0)


def _draw_chevron_down(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.8))
    p.drawPolyline(QPolygonF([QPointF(5.5, 8), QPointF(10, 12.5), QPointF(14.5, 8)]))


def _draw_chevron_up(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.8))
    p.drawPolyline(QPolygonF([QPointF(5.5, 12.5), QPointF(10, 8), QPointF(14.5, 12.5)]))


def _draw_chevron_right(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.8))
    p.drawPolyline(QPolygonF([QPointF(8, 5.5), QPointF(12.5, 10), QPointF(8, 14.5)]))


def _draw_chevron_left(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.8))
    p.drawPolyline(QPolygonF([QPointF(12, 5.5), QPointF(7.5, 10), QPointF(12, 14.5)]))


def _draw_check(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 2.2))
    p.drawPolyline(QPolygonF([QPointF(4.5, 10.5), QPointF(8.5, 14.5), QPointF(15.5, 6)]))


def _draw_close(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.drawLine(QPointF(5.5, 5.5), QPointF(14.5, 14.5))
    p.drawLine(QPointF(14.5, 5.5), QPointF(5.5, 14.5))


def _draw_warning(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(QPolygonF([QPointF(10, 3), QPointF(17.5, 16.5), QPointF(2.5, 16.5)]))
    p.setPen(_pen(c, 1.8))
    p.drawLine(QPointF(10, 8), QPointF(10, 11.5))
    p.drawPoint(QPointF(10, 14))


def _draw_circle(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QRectF(4, 4, 12, 12))


def _draw_dash(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 2.0))
    p.drawLine(QPointF(6, 10), QPointF(14, 10))


def _draw_checklist(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    for y, ticked in ((5.0, True), (10.0, True), (15.0, False)):
        p.drawLine(QPointF(8.5, y), QPointF(17, y))
        if ticked:
            p.drawPolyline(QPolygonF([QPointF(2.5, y), QPointF(4.3, y + 1.8), QPointF(6.8, y - 1.6)]))
        else:
            p.drawEllipse(QPointF(4.6, y), 1.6, 1.6)


def _draw_eye(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(2.5, 10)
    path.cubicTo(6, 4.5, 14, 4.5, 17.5, 10)
    path.cubicTo(14, 15.5, 6, 15.5, 2.5, 10)
    p.drawPath(path)
    p.setBrush(QBrush(c))
    p.drawEllipse(QPointF(10, 10), 2.4, 2.4)


def _draw_pencil(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(QPolygonF([
        QPointF(14, 3.5), QPointF(16.5, 6), QPointF(7, 15.5),
        QPointF(3.5, 16.5), QPointF(4.5, 13),
    ]))
    p.drawLine(QPointF(11.5, 6), QPointF(14, 8.5))


def _draw_copy(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(7, 7, 9.5, 9.5), 1.5, 1.5)
    path = QPainterPath()
    path.moveTo(13, 5)
    path.lineTo(5, 5)
    path.lineTo(5, 13)
    p.drawPath(path)


def _draw_paperclip(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(13.5, 6.5)
    path.lineTo(7.5, 12.5)
    path.arcTo(QRectF(5.5, 10.5, 4, 4), 90, 180)
    path.lineTo(14, 8)
    path.arcTo(QRectF(10, 4, 6, 6), 90, 180)
    path.lineTo(6.5, 14)
    p.drawPath(path)


def _draw_spark(p: QPainter, c: QColor) -> None:

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    path = QPainterPath()
    path.moveTo(10, 2.5)
    path.cubicTo(10.8, 7.5, 12.5, 9.2, 17.5, 10)
    path.cubicTo(12.5, 10.8, 10.8, 12.5, 10, 17.5)
    path.cubicTo(9.2, 12.5, 7.5, 10.8, 2.5, 10)
    path.cubicTo(7.5, 9.2, 9.2, 7.5, 10, 2.5)
    p.drawPath(path)


def _draw_gem(p: QPainter, c: QColor) -> None:


    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawPolygon(QPolygonF([
        QPointF(6.4, 3.8), QPointF(13.6, 3.8), QPointF(17.4, 8.4),
        QPointF(10.0, 17.2), QPointF(2.6, 8.4),
    ]))


def _draw_person(p: QPainter, c: QColor) -> None:









    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QRectF(6.9, 3.4, 6.2, 6.2))
    path = QPainterPath()
    path.moveTo(3.6, 16.9)
    path.arcTo(QRectF(3.6, 11.0, 12.8, 11.8), 180.0, -180.0)
    p.drawPath(path)


def _draw_undo(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.7))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(6.5, 8.5)
    path.lineTo(13, 8.5)
    path.arcTo(QRectF(9.5, 8.5, 7, 7), 90, -180)
    path.lineTo(8, 15.5)
    p.drawPath(path)
    p.drawPolyline(QPolygonF([QPointF(9, 5.5), QPointF(6, 8.5), QPointF(9, 11.5)]))


def _draw_file(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(5, 3)
    path.lineTo(12, 3)
    path.lineTo(15.5, 6.5)
    path.lineTo(15.5, 17)
    path.lineTo(5, 17)
    path.closeSubpath()
    p.drawPath(path)
    p.drawPolyline(QPolygonF([QPointF(12, 3), QPointF(12, 6.5), QPointF(15.5, 6.5)]))
    p.drawLine(QPointF(7.5, 10.5), QPointF(13, 10.5))
    p.drawLine(QPointF(7.5, 13.5), QPointF(13, 13.5))


def _draw_image(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(3, 4, 14, 12), 1.5, 1.5)
    p.drawPolyline(QPolygonF([QPointF(4, 14), QPointF(8, 9.5), QPointF(11, 12.5),
                              QPointF(13, 10.5), QPointF(16, 14)]))
    p.setBrush(QBrush(c))
    p.drawEllipse(QPointF(13, 7.5), 1.3, 1.3)


def _draw_arrow_up(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 2.2))
    p.drawLine(QPointF(10, 15.5), QPointF(10, 5))
    p.drawPolyline(QPolygonF([QPointF(5.5, 9.5), QPointF(10, 5), QPointF(14.5, 9.5)]))


def _draw_arrow_down(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 2.0))
    p.drawLine(QPointF(10, 4.5), QPointF(10, 15))
    p.drawPolyline(QPolygonF([QPointF(5.5, 10.5), QPointF(10, 15), QPointF(14.5, 10.5)]))


def _draw_gear(p: QPainter, c: QColor) -> None:







    centre = BOX / 2.0
    teeth = 6
    step = 2.0 * math.pi / teeth
    r_tip, r_root = BOX * 0.37, BOX * 0.27
    half_tip, half_root = step * 0.16, step * 0.32
    path = QPainterPath()
    first = True
    for i in range(teeth):
        angle = i * step
        for ang, radius in (
            (angle - half_root, r_root), (angle - half_tip, r_tip),
            (angle + half_tip, r_tip), (angle + half_root, r_root),
        ):
            point = QPointF(centre + radius * math.cos(ang), centre + radius * math.sin(ang))
            if first:
                path.moveTo(point)
                first = False
            else:
                path.lineTo(point)
    path.closeSubpath()
    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)
    p.drawEllipse(QPointF(centre, centre), BOX * 0.13, BOX * 0.13)


def _draw_camera(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(3, 7)
    path.lineTo(7, 7)
    path.lineTo(8.5, 4.5)
    path.lineTo(11.5, 4.5)
    path.lineTo(13, 7)
    path.lineTo(17, 7)
    path.lineTo(17, 16)
    path.lineTo(3, 16)
    path.closeSubpath()
    p.drawPath(path)
    p.drawEllipse(QPointF(10, 11.5), 2.8, 2.8)


def _draw_layers(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(QPolygonF([QPointF(10, 3.5), QPointF(17, 7.5), QPointF(10, 11.5), QPointF(3, 7.5)]))
    p.drawPolyline(QPolygonF([QPointF(3, 11), QPointF(10, 15), QPointF(17, 11)]))
    p.drawPolyline(QPolygonF([QPointF(3, 14), QPointF(10, 18), QPointF(17, 14)]))


def _draw_expand(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.8))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(QPolygonF([QPointF(11.5, 4), QPointF(16, 4), QPointF(16, 8.5)]))
    p.drawPolyline(QPolygonF([QPointF(8.5, 16), QPointF(4, 16), QPointF(4, 11.5)]))
    p.drawLine(QPointF(16, 4), QPointF(11.5, 8.5))
    p.drawLine(QPointF(4, 16), QPointF(8.5, 11.5))


def _shield_path() -> QPainterPath:
    path = QPainterPath()
    path.moveTo(10, 2.8)
    path.lineTo(16, 5.2)
    path.lineTo(16, 9.5)
    path.cubicTo(16, 13.2, 13.5, 16, 10, 17.4)
    path.cubicTo(6.5, 16, 4, 13.2, 4, 9.5)
    path.lineTo(4, 5.2)
    path.closeSubpath()
    return path


def _draw_shield(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(_shield_path())


def _draw_shield_check(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(_shield_path())
    p.setPen(_pen(c, 1.7))
    p.drawPolyline(QPolygonF([QPointF(7.2, 10), QPointF(9.2, 12), QPointF(13, 7.8)]))


def _draw_bolt(p: QPainter, c: QColor) -> None:

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawPolygon(QPolygonF([
        QPointF(11.2, 2.5), QPointF(4.8, 11.2), QPointF(9.4, 11.2),
        QPointF(8.4, 17.5), QPointF(15.2, 8.4), QPointF(10.6, 8.4),
    ]))


def _draw_globe(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)


    p.drawEllipse(QPointF(10, 10), 7.2, 7.2)
    p.drawEllipse(QPointF(10, 10), 3.1, 7.2)
    p.drawLine(QPointF(2.8, 10), QPointF(17.2, 10))


def _draw_pin(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(10, 17.5)
    path.cubicTo(10, 17.5, 4.5, 11.8, 4.5, 8.3)
    path.arcTo(QRectF(4.5, 2.8, 11, 11), 180, -180)
    path.cubicTo(15.5, 11.8, 10, 17.5, 10, 17.5)
    p.drawPath(path)
    p.drawEllipse(QPointF(10, 8.3), 2.1, 2.1)


def _draw_book(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    left = QPainterPath()
    left.moveTo(10, 16.2)
    left.cubicTo(8.5, 14.8, 6, 14.5, 3, 14.8)
    left.lineTo(3, 5)
    left.cubicTo(6, 4.6, 8.5, 5, 10, 6.4)
    p.drawPath(left)
    right = QPainterPath()
    right.moveTo(10, 16.2)
    right.cubicTo(11.5, 14.8, 14, 14.5, 17, 14.8)
    right.lineTo(17, 5)
    right.cubicTo(14, 4.6, 11.5, 5, 10, 6.4)
    p.drawPath(right)
    p.drawLine(QPointF(10, 6.4), QPointF(10, 16.2))


def _draw_play(p: QPainter, c: QColor) -> None:






    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(3, 4.5, 14, 11), 2.2, 2.2)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawPolygon(QPolygonF([QPointF(8.4, 7.4), QPointF(13.2, 10), QPointF(8.4, 12.6)]))


def _draw_new_chat(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    box = QPainterPath()
    box.moveTo(10.5, 3.5)
    box.lineTo(5.5, 3.5)
    box.quadTo(3.5, 3.5, 3.5, 5.5)
    box.lineTo(3.5, 14.5)
    box.quadTo(3.5, 16.5, 5.5, 16.5)
    box.lineTo(14.5, 16.5)
    box.quadTo(16.5, 16.5, 16.5, 14.5)
    box.lineTo(16.5, 9.5)
    p.drawPath(box)
    pencil = QPainterPath()
    pencil.moveTo(14.3, 3.4)
    pencil.lineTo(16.6, 5.7)
    pencil.lineTo(9.6, 12.7)
    pencil.lineTo(6.6, 13.4)
    pencil.lineTo(7.3, 10.4)
    pencil.closeSubpath()
    p.drawPath(pencil)
    p.drawLine(QPointF(12.6, 5.1), QPointF(14.9, 7.4))


def _draw_search(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(8.8, 8.8), 5.3, 5.3)
    p.setPen(_pen(c, 1.7))
    p.drawLine(QPointF(12.8, 12.8), QPointF(16.8, 16.8))


def _draw_chat_bubble(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(7.2, 14.5)
    path.lineTo(13.5, 14.5)
    path.quadTo(17, 14.5, 17, 11)
    path.lineTo(17, 7)
    path.quadTo(17, 3.5, 13.5, 3.5)
    path.lineTo(6.5, 3.5)
    path.quadTo(3, 3.5, 3, 7)
    path.lineTo(3, 17.2)
    path.closeSubpath()
    p.drawPath(path)


def _draw_trash(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawLine(QPointF(4, 6.2), QPointF(16, 6.2))
    p.drawPolyline(QPolygonF([QPointF(8, 6.2), QPointF(8, 4), QPointF(12, 4), QPointF(12, 6.2)]))
    body = QPainterPath()
    body.moveTo(5.5, 6.2)
    body.lineTo(6.3, 15.2)
    body.quadTo(6.4, 16.5, 7.7, 16.5)
    body.lineTo(12.3, 16.5)
    body.quadTo(13.6, 16.5, 13.7, 15.2)
    body.lineTo(14.5, 6.2)
    p.drawPath(body)
    p.drawLine(QPointF(8.6, 9), QPointF(8.9, 13.8))
    p.drawLine(QPointF(11.4, 9), QPointF(11.1, 13.8))


def _draw_terminal(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.8))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(QPolygonF([QPointF(4, 5.5), QPointF(9, 10), QPointF(4, 14.5)]))
    p.drawLine(QPointF(11, 14.8), QPointF(16.5, 14.8))


def _draw_code(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(QPolygonF([QPointF(6.5, 5.5), QPointF(2.5, 10), QPointF(6.5, 14.5)]))
    p.drawPolyline(QPolygonF([QPointF(13.5, 5.5), QPointF(17.5, 10), QPointF(13.5, 14.5)]))
    p.drawLine(QPointF(11.6, 3.8), QPointF(8.4, 16.2))








def _hexagon(cx: float, cy: float, r: float) -> QPolygonF:

    return QPolygonF([
        QPointF(cx + r * math.cos(math.radians(60 * i - 90)),
                cy + r * math.sin(math.radians(60 * i - 90)))
        for i in range(6)
    ])


def _dashed(c: QColor, width: float, dash: float = 2.4, gap: float = 1.9) -> QPen:

    pen = QPen(c)
    pen.setWidthF(width)
    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    pen.setStyle(Qt.PenStyle.CustomDashLine)
    pen.setDashPattern([dash, gap])
    return pen


def _draw_buffer(p: QPainter, c: QColor) -> None:

    p.setPen(_dashed(c, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(10, 10), 7.6, 7.6)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawEllipse(QPointF(10, 10), 3.4, 3.4)


def _draw_clip(p: QPainter, c: QColor) -> None:

    shape = QPainterPath()
    shape.addPolygon(QPolygonF([
        QPointF(2.8, 11.6), QPointF(6.2, 4.0), QPointF(14.4, 3.4),
        QPointF(17.2, 10.2), QPointF(10.6, 16.6),
    ]))
    shape.closeSubpath()
    cutter = QRectF(8.6, 7.4, 8.8, 9.2)
    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(shape)
    box = QPainterPath()
    box.addRect(cutter)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawPath(shape.intersected(box))
    p.setPen(_dashed(c, 1.3))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(cutter)


def _draw_intersect(p: QPainter, c: QColor) -> None:

    left = QPainterPath()
    left.addEllipse(QPointF(7.5, 10), 5.4, 5.4)
    right = QPainterPath()
    right.addEllipse(QPointF(12.5, 10), 5.4, 5.4)
    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(left)
    p.drawPath(right)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawPath(left.intersected(right))


def _draw_dissolve(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(3.2, 4.6, 13.6, 10.8), 1.8, 1.8)
    p.setPen(_dashed(c, 1.2, 1.9, 1.7))
    p.drawLine(QPointF(10, 4.6), QPointF(10, 15.4))
    p.drawLine(QPointF(3.2, 10), QPointF(16.8, 10))


def _draw_heatmap(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(10, 10), 7.4, 6.3)
    p.drawEllipse(QPointF(10, 10), 4.8, 4.1)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawEllipse(QPointF(10, 10), 2.3, 2.0)


def _draw_classify(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(3.2, 5.2, 13.6, 9.6))
    p.drawLine(QPointF(7.73, 5.2), QPointF(7.73, 14.8))
    p.drawLine(QPointF(12.27, 5.2), QPointF(12.27, 14.8))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawRect(QRectF(8.5, 11.3, 3.0, 2.75))
    p.drawRect(QRectF(13.05, 6.6, 3.0, 7.45))


def _draw_raster(p: QPainter, c: QColor) -> None:

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawRect(QRectF(7.83, 3.5, 4.34, 4.33))
    p.drawRect(QRectF(3.5, 12.17, 4.33, 4.33))
    p.drawRect(QRectF(12.17, 7.83, 4.33, 4.34))
    p.setPen(_pen(c, 1.3))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(3.5, 3.5, 13, 13))
    p.drawLine(QPointF(7.83, 3.5), QPointF(7.83, 16.5))
    p.drawLine(QPointF(12.17, 3.5), QPointF(12.17, 16.5))
    p.drawLine(QPointF(3.5, 7.83), QPointF(16.5, 7.83))
    p.drawLine(QPointF(3.5, 12.17), QPointF(16.5, 12.17))


def _draw_terrain(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(QPolygonF([
        QPointF(2.8, 15.6), QPointF(7.4, 6.2), QPointF(10.4, 11.4),
        QPointF(12.6, 8.2), QPointF(17.2, 15.6),
    ]))
    p.drawLine(QPointF(2.8, 15.6), QPointF(17.2, 15.6))


def _draw_contour(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    for cx, cy, rx, ry in ((9.4, 10.6, 7.2, 6.0), (10.0, 10.0, 4.6, 3.8), (10.7, 9.3, 2.1, 1.7)):
        p.drawEllipse(QPointF(cx, cy), rx, ry)


def _draw_join(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    for x in (2.6, 11.4):
        p.drawRect(QRectF(x, 5.4, 6.0, 9.2))
        p.drawLine(QPointF(x, 8.5), QPointF(x + 6.0, 8.5))
        p.drawLine(QPointF(x, 11.6), QPointF(x + 6.0, 11.6))
    p.setPen(_pen(c, 1.7))
    p.drawLine(QPointF(8.6, 10.05), QPointF(11.4, 10.05))


def _draw_table(p: QPainter, c: QColor) -> None:

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawRect(QRectF(3.2, 4.6, 13.6, 3.3))
    p.setPen(_pen(c, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(3.2, 4.6, 13.6, 10.8))
    p.drawLine(QPointF(3.2, 11.8), QPointF(16.8, 11.8))
    p.drawLine(QPointF(9.6, 7.9), QPointF(9.6, 15.4))


def _draw_route(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.7))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(4.6, 15.4)
    path.cubicTo(9.8, 15.4, 6.2, 8.6, 10.6, 8.6)
    path.cubicTo(14.2, 8.6, 13.2, 5.2, 15.4, 4.6)
    p.drawPath(path)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawEllipse(QPointF(4.6, 15.4), 2.0, 2.0)
    p.drawEllipse(QPointF(15.4, 4.6), 2.0, 2.0)


def _draw_isochrone(p: QPainter, c: QColor) -> None:

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawEllipse(QPointF(10, 10), 1.9, 1.9)
    p.setPen(_pen(c, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(10, 10), 4.4, 4.4)
    p.setPen(_dashed(c, 1.3, 2.2, 1.8))
    p.drawEllipse(QPointF(10, 10), 7.3, 7.3)


def _draw_hexgrid(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(_hexagon(10.0, 6.2, 3.4))
    p.drawPolygon(_hexagon(7.06, 11.3, 3.4))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawPolygon(_hexagon(12.94, 11.3, 3.4))


def _draw_chart(p: QPainter, c: QColor) -> None:

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawRoundedRect(QRectF(4.2, 10.4, 3.4, 4.7), 1.0, 1.0)
    p.drawRoundedRect(QRectF(8.3, 6.2, 3.4, 8.9), 1.0, 1.0)
    p.drawRoundedRect(QRectF(12.4, 8.6, 3.4, 6.5), 1.0, 1.0)
    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawLine(QPointF(3.2, 16.2), QPointF(16.8, 16.2))


def _draw_download(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawLine(QPointF(10, 3.2), QPointF(10, 11.8))
    p.drawPolyline(QPolygonF([QPointF(6.4, 8.4), QPointF(10, 12.0), QPointF(13.6, 8.4)]))
    p.drawPolyline(QPolygonF([
        QPointF(4.0, 13.4), QPointF(4.0, 16.6), QPointF(16.0, 16.6), QPointF(16.0, 13.4),
    ]))


def _draw_satellite(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(8.4, 8.6, 3.2, 3.2))
    p.drawRect(QRectF(2.6, 7.6, 4.6, 5.2))
    p.drawRect(QRectF(12.8, 7.6, 4.6, 5.2))
    p.drawLine(QPointF(7.2, 10.2), QPointF(8.4, 10.2))
    p.drawLine(QPointF(11.6, 10.2), QPointF(12.8, 10.2))
    p.drawLine(QPointF(10, 8.6), QPointF(10, 5.4))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawEllipse(QPointF(10, 4.2), 1.4, 1.4)
    p.drawRect(QRectF(9.1, 14.0, 1.8, 3.0))


def _draw_funnel(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(QPolygonF([
        QPointF(3.2, 4.4), QPointF(16.8, 4.4), QPointF(11.5, 10.6),
        QPointF(11.5, 16.6), QPointF(8.5, 14.6), QPointF(8.5, 10.6),
    ]))


def _draw_layout(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(3.4, 2.6, 13.2, 14.8))
    p.drawRect(QRectF(5.2, 7.4, 9.6, 5.4))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawRect(QRectF(5.2, 4.4, 6.4, 1.4))
    p.drawRect(QRectF(5.2, 14.2, 2.2, 1.2))
    p.drawRect(QRectF(8.4, 14.2, 6.4, 1.2))


def _draw_crs(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.3))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(3.4, 3.4, 13.2, 13.2))
    p.drawLine(QPointF(3.4, 10), QPointF(16.6, 10))
    p.drawLine(QPointF(10, 3.4), QPointF(10, 16.6))
    p.setPen(_pen(c, 1.7))
    p.drawEllipse(QPointF(10, 10), 2.6, 2.6)


def _draw_measure(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.save()
    try:
        p.translate(10, 10)
        p.rotate(-33)
        p.translate(-10, -10)
        p.drawRoundedRect(QRectF(2.4, 7.8, 15.2, 4.4), 1.2, 1.2)
        for x in (5.6, 8.4, 11.6, 14.4):
            p.drawLine(QPointF(x, 7.8), QPointF(x, 9.9))
    finally:
        p.restore()


def _draw_centroid(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(QPolygonF([
        QPointF(3.4, 8.0), QPointF(9.0, 3.2), QPointF(16.6, 7.4),
        QPointF(14.2, 16.4), QPointF(5.0, 15.0),
    ]))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawEllipse(QPointF(9.7, 10.3), 2.1, 2.1)


def _draw_points(p: QPainter, c: QColor) -> None:

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    for x, y, r in ((5.0, 6.2, 1.7), (11.6, 4.4, 1.5), (15.6, 8.8, 1.7),
                    (7.2, 12.6, 1.6), (13.2, 14.8, 1.8)):
        p.drawEllipse(QPointF(x, y), r, r)


def _vertex_handle(p: QPainter, c: QColor, point: QPointF, side: float = 3.2) -> None:

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    half = side / 2.0
    p.drawRoundedRect(QRectF(point.x() - half, point.y() - half, side, side), 0.7, 0.7)


def _knock_out(p: QPainter, path: QPainterPath, margin: float) -> None:



    p.save()
    try:
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        p.setPen(_pen(QColor(0, 0, 0), margin * 2.0))
        p.setBrush(QBrush(QColor(0, 0, 0)))
        p.drawPath(path)
    finally:
        p.restore()


def _pointer_path(tip: QPointF, length: float = 7.0) -> QPainterPath:

    x, y = tip.x(), tip.y()
    k = length / 7.0
    path = QPainterPath()
    path.moveTo(x, y)
    path.lineTo(x + 6.8 * k, y + 2.5 * k)
    path.lineTo(x + 3.6 * k, y + 3.6 * k)
    path.lineTo(x + 2.5 * k, y + 6.8 * k)
    path.closeSubpath()
    return path


def _draw_polygon(p: QPainter, c: QColor) -> None:




    corners = [QPointF(3.6, 4.0), QPointF(14.6, 3.4), QPointF(11.2, 11.2),
               QPointF(4.2, 15.0)]
    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(QPolygonF(corners))
    for corner in (corners[0], corners[1], corners[3]):
        _vertex_handle(p, c, corner)
    pointer = _pointer_path(QPointF(11.2, 11.2), 7.2)
    _knock_out(p, pointer, 1.5)
    p.setPen(_pen(c, 1.3))
    p.setBrush(QBrush(c))
    p.drawPath(pointer)


def _draw_merge(p: QPainter, c: QColor) -> None:






    first = QPainterPath()
    first.addRoundedRect(QRectF(2.6, 2.6, 10.2, 10.2), 2.0, 2.0)
    second = QPainterPath()
    second.addRoundedRect(QRectF(7.2, 7.2, 10.2, 10.2), 2.0, 2.0)
    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(first.united(second))
    faint = QColor(c)
    faint.setAlphaF(c.alphaF() * 0.45)
    p.setPen(_pen(faint, 1.1))
    p.drawPath(first.intersected(second))


def _draw_link(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.save()
    try:
        p.translate(10, 10)
        p.rotate(-45)
        p.translate(-10, -10)
        p.drawRoundedRect(QRectF(2.8, 7.4, 8.4, 5.2), 2.6, 2.6)
        p.drawRoundedRect(QRectF(8.8, 7.4, 8.4, 5.2), 2.6, 2.6)
    finally:
        p.restore()


def _draw_package(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(QPolygonF([
        QPointF(3.2, 6.4), QPointF(10, 3.0), QPointF(16.8, 6.4),
        QPointF(16.8, 13.6), QPointF(10, 17.0), QPointF(3.2, 13.6),
    ]))
    p.drawLine(QPointF(3.2, 6.4), QPointF(10, 9.8))
    p.drawLine(QPointF(16.8, 6.4), QPointF(10, 9.8))
    p.drawLine(QPointF(10, 9.8), QPointF(10, 17.0))


def _draw_server(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(3.2, 4.4, 13.6, 4.9), 1.4, 1.4)
    p.drawRoundedRect(QRectF(3.2, 10.7, 13.6, 4.9), 1.4, 1.4)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawEllipse(QPointF(6.2, 6.85), 1.15, 1.15)
    p.drawEllipse(QPointF(6.2, 13.15), 1.15, 1.15)


def _draw_cluster(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(10, 10), 7.2, 7.2)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    for x, y in ((7.6, 8.2), (12.3, 7.9), (8.9, 12.4), (12.9, 12.1), (10.3, 10.0)):
        p.drawEllipse(QPointF(x, y), 1.5, 1.5)


def _draw_mic(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(7.2, 2.8, 5.6, 9.4), 2.8, 2.8)
    arc = QPainterPath()
    arc.moveTo(4.6, 9.6)
    arc.cubicTo(4.6, 13.2, 7.0, 15.2, 10, 15.2)
    arc.cubicTo(13.0, 15.2, 15.4, 13.2, 15.4, 9.6)
    p.drawPath(arc)
    p.drawLine(QPointF(10, 15.2), QPointF(10, 17.6))


def _draw_home(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    roof = QPainterPath()
    roof.moveTo(3.4, 9.6)
    roof.lineTo(10, 3.6)
    roof.lineTo(16.6, 9.6)
    p.drawPath(roof)
    body = QPainterPath()
    body.moveTo(5.2, 8.4)
    body.lineTo(5.2, 16.2)
    body.lineTo(14.8, 16.2)
    body.lineTo(14.8, 8.4)
    p.drawPath(body)
    p.drawLine(QPointF(10, 16.2), QPointF(10, 11.8))


def _draw_sidebar(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(3.2, 4.2, 13.6, 11.6), 2.2, 2.2)
    p.drawLine(QPointF(8.2, 4.4), QPointF(8.2, 15.6))
    p.drawLine(QPointF(5.2, 7.4), QPointF(6.4, 7.4))
    p.drawLine(QPointF(5.2, 9.8), QPointF(6.4, 9.8))


def _draw_sparkles(p: QPainter, c: QColor) -> None:

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    big = QPainterPath()
    big.moveTo(8, 3.5)
    big.cubicTo(8.7, 7.6, 10, 8.9, 14.1, 9.6)
    big.cubicTo(10, 10.3, 8.7, 11.6, 8, 15.7)
    big.cubicTo(7.3, 11.6, 6, 10.3, 1.9, 9.6)
    big.cubicTo(6, 8.9, 7.3, 7.6, 8, 3.5)
    p.drawPath(big)
    small = QPainterPath()
    small.moveTo(15, 12)
    small.cubicTo(15.3, 13.9, 15.9, 14.5, 17.8, 14.8)
    small.cubicTo(15.9, 15.1, 15.3, 15.7, 15, 17.6)
    small.cubicTo(14.7, 15.7, 14.1, 15.1, 12.2, 14.8)
    small.cubicTo(14.1, 14.5, 14.7, 13.9, 15, 12)
    p.drawPath(small)


def _draw_polyline(p: QPainter, c: QColor) -> None:

    points = [QPointF(3.6, 14.6), QPointF(7.8, 8.4), QPointF(12.2, 11.6),
              QPointF(16.4, 4.6)]
    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(QPolygonF(points))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    for point in points:
        p.drawEllipse(point, 1.5, 1.5)


def _draw_puzzle(p: QPainter, c: QColor) -> None:



    left, top, right, bottom = 2.6, 6.4, 13.6, 17.4
    radius, reach = 2.3, 1.5
    top_knob = QRectF(8.1 - radius, top - reach - radius, 2 * radius, 2 * radius)
    right_knob = QRectF(right + reach - radius, 11.9 - radius, 2 * radius, 2 * radius)



    meet = math.degrees(math.asin(reach / radius))
    side = math.degrees(math.acos(reach / radius))
    path = QPainterPath()
    path.moveTo(left, top)

    path.arcTo(top_knob, 180.0 + meet, -(180.0 + 2 * meet))
    path.lineTo(right, top)
    path.arcTo(right_knob, 180.0 - side, 2 * side - 360.0)
    path.lineTo(right, bottom)
    path.lineTo(left, bottom)
    path.closeSubpath()
    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)


def _draw_thumbs_up(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(3.0, 9.5, 3.4, 7.0), 1.0, 1.0)
    path = QPainterPath()
    path.moveTo(7.4, 16.5)
    path.lineTo(7.4, 10.2)
    path.lineTo(10.3, 4.2)
    path.cubicTo(11.6, 3.5, 12.8, 4.6, 12.3, 6.0)
    path.lineTo(11.2, 9.2)
    path.lineTo(15.6, 9.2)
    path.cubicTo(16.7, 9.2, 17.3, 10.4, 16.7, 11.3)
    path.lineTo(14.7, 15.3)
    path.cubicTo(14.3, 16.1, 13.5, 16.5, 12.7, 16.5)
    path.closeSubpath()
    p.drawPath(path)


def _draw_thumbs_down(p: QPainter, c: QColor) -> None:

    p.save()
    p.translate(10.0, 10.0)
    p.rotate(180)
    p.translate(-10.0, -10.0)
    _draw_thumbs_up(p, c)
    p.restore()


def _draw_thumbs_up_filled(p: QPainter, c: QColor) -> None:


    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    p.drawRoundedRect(QRectF(2.6, 9.2, 3.4, 7.7), 1.0, 1.0)
    path = QPainterPath()
    path.moveTo(7.4, 16.5)
    path.lineTo(7.4, 10.2)
    path.lineTo(10.3, 4.2)
    path.cubicTo(11.6, 3.5, 12.8, 4.6, 12.3, 6.0)
    path.lineTo(11.2, 9.2)
    path.lineTo(15.6, 9.2)
    path.cubicTo(16.7, 9.2, 17.3, 10.4, 16.7, 11.3)
    path.lineTo(14.7, 15.3)
    path.cubicTo(14.3, 16.1, 13.5, 16.5, 12.7, 16.5)
    path.closeSubpath()
    p.setPen(_pen(c, 1.1))
    p.drawPath(path)


def _draw_thumbs_down_filled(p: QPainter, c: QColor) -> None:

    p.save()
    p.translate(10.0, 10.0)
    p.rotate(180)
    p.translate(-10.0, -10.0)
    _draw_thumbs_up_filled(p, c)
    p.restore()


_GLYPHS = {
    "send": _draw_send,
    "bolt": _draw_bolt,
    "stop": _draw_stop,
    "plus": _draw_plus,
    "plus_circle": _draw_plus_circle,
    "clock": _draw_clock,
    "kebab": _draw_kebab,
    "chevron_down": _draw_chevron_down,
    "chevron_up": _draw_chevron_up,
    "chevron_right": _draw_chevron_right,
    "chevron_left": _draw_chevron_left,
    "check": _draw_check,
    "close": _draw_close,
    "warning": _draw_warning,
    "circle": _draw_circle,
    "dash": _draw_dash,
    "checklist": _draw_checklist,
    "eye": _draw_eye,
    "pencil": _draw_pencil,
    "copy": _draw_copy,
    "paperclip": _draw_paperclip,
    "spark": _draw_spark,
    "undo": _draw_undo,
    "gear": _draw_gear,
    "file": _draw_file,
    "image": _draw_image,
    "arrow_down": _draw_arrow_down,
    "arrow_up": _draw_arrow_up,
    "camera": _draw_camera,
    "layers": _draw_layers,
    "expand": _draw_expand,
    "lock": _draw_lock,
    "shield": _draw_shield,
    "shield_check": _draw_shield_check,
    "globe": _draw_globe,
    "pin": _draw_pin,
    "book": _draw_book,
    "play": _draw_play,
    "new_chat": _draw_new_chat,
    "search": _draw_search,
    "chat_bubble": _draw_chat_bubble,
    "trash": _draw_trash,
    "gem": _draw_gem,
    "person": _draw_person,
    "terminal": _draw_terminal,
    "code": _draw_code,

    "buffer": _draw_buffer,
    "clip": _draw_clip,
    "intersect": _draw_intersect,
    "dissolve": _draw_dissolve,
    "heatmap": _draw_heatmap,
    "classify": _draw_classify,
    "raster": _draw_raster,
    "terrain": _draw_terrain,
    "contour": _draw_contour,
    "join": _draw_join,
    "table": _draw_table,
    "route": _draw_route,
    "isochrone": _draw_isochrone,
    "hexgrid": _draw_hexgrid,
    "chart": _draw_chart,
    "download": _draw_download,
    "satellite": _draw_satellite,
    "funnel": _draw_funnel,
    "layout": _draw_layout,
    "crs": _draw_crs,
    "measure": _draw_measure,
    "centroid": _draw_centroid,
    "points": _draw_points,
    "polygon": _draw_polygon,
    "merge": _draw_merge,
    "link": _draw_link,
    "package": _draw_package,
    "server": _draw_server,
    "cluster": _draw_cluster,
    "mic": _draw_mic,
    "home": _draw_home,
    "sidebar": _draw_sidebar,
    "sparkles": _draw_sparkles,
    "polyline": _draw_polyline,
    "puzzle": _draw_puzzle,
    "thumbs_up": _draw_thumbs_up,
    "thumbs_down": _draw_thumbs_down,
    "thumbs_up_filled": _draw_thumbs_up_filled,
    "thumbs_down_filled": _draw_thumbs_down_filled,
}




def _draw_help(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QRectF(2.8, 2.8, 14.4, 14.4))
    hook = QPainterPath()
    hook.moveTo(7.6, 8.0)
    hook.cubicTo(7.6, 6.4, 8.7, 5.7, 10.0, 5.7)
    hook.cubicTo(11.4, 5.7, 12.4, 6.6, 12.4, 7.9)
    hook.cubicTo(12.4, 9.5, 10.0, 9.8, 10.0, 11.6)
    p.drawPath(hook)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    p.drawEllipse(QPointF(10.0, 14.1), 1.05, 1.05)


def _draw_float_window(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(QPolygonF([QPointF(7.5, 5.5), QPointF(7.5, 4.8),
                              QPointF(15.2, 4.8), QPointF(15.2, 12.5),
                              QPointF(14.5, 12.5)]))
    p.drawRoundedRect(QRectF(4.8, 7.5, 7.7, 7.7), 1.4, 1.4)


_GLYPHS["help"] = _draw_help
_GLYPHS["float_window"] = _draw_float_window


def _draw_cloud(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(5.6, 15.2)
    path.cubicTo(3.4, 15.2, 2.2, 13.7, 2.2, 12.0)
    path.cubicTo(2.2, 10.2, 3.6, 8.9, 5.3, 8.9)
    path.cubicTo(5.8, 6.3, 7.8, 4.8, 10.2, 4.8)
    path.cubicTo(12.9, 4.8, 14.8, 6.7, 15.0, 9.1)
    path.cubicTo(16.7, 9.3, 17.8, 10.6, 17.8, 12.2)
    path.cubicTo(17.8, 13.9, 16.5, 15.2, 14.7, 15.2)
    path.closeSubpath()
    p.drawPath(path)


def _draw_laptop(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(4.2, 4.6, 11.6, 8.2), 1.4, 1.4)
    p.drawLine(QPointF(2.2, 15.6), QPointF(17.8, 15.6))


def _draw_map(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    fold = QPolygonF([
        QPointF(2.8, 5.2), QPointF(7.4, 3.4), QPointF(12.6, 5.2),
        QPointF(17.2, 3.4), QPointF(17.2, 14.8), QPointF(12.6, 16.6),
        QPointF(7.4, 14.8), QPointF(2.8, 16.6),
    ])
    p.drawPolygon(fold)
    p.drawLine(QPointF(7.4, 3.4), QPointF(7.4, 14.8))
    p.drawLine(QPointF(12.6, 5.2), QPointF(12.6, 16.6))


def _draw_dot(p: QPainter, c: QColor) -> None:

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    p.drawEllipse(QPointF(10.0, 10.0), 4.6, 4.6)


def _draw_lightbulb(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    bulb = QPainterPath()
    bulb.moveTo(7.6, 13.2)
    bulb.cubicTo(7.6, 11.6, 5.0, 10.6, 5.0, 7.8)
    bulb.cubicTo(5.0, 5.0, 7.2, 2.9, 10.0, 2.9)
    bulb.cubicTo(12.8, 2.9, 15.0, 5.0, 15.0, 7.8)
    bulb.cubicTo(15.0, 10.6, 12.4, 11.6, 12.4, 13.2)
    bulb.closeSubpath()
    p.drawPath(bulb)
    p.drawLine(QPointF(8.0, 15.6), QPointF(12.0, 15.6))
    p.drawLine(QPointF(8.8, 17.6), QPointF(11.2, 17.6))


_GLYPHS["cloud"] = _draw_cloud
_GLYPHS["laptop"] = _draw_laptop
_GLYPHS["map"] = _draw_map
_GLYPHS["dot"] = _draw_dot
_GLYPHS["lightbulb"] = _draw_lightbulb




def _draw_refresh(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.7))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.arcMoveTo(QRectF(4.0, 4.0, 12.0, 12.0), 60)
    path.arcTo(QRectF(4.0, 4.0, 12.0, 12.0), 60, 300)
    p.drawPath(path)
    p.drawPolyline(QPolygonF([QPointF(10.6, 2.6), QPointF(13.4, 4.8), QPointF(10.9, 7.4)]))


def _star_path() -> QPainterPath:
    import math as _math

    path = QPainterPath()
    for i in range(10):
        radius = 7.2 if i % 2 == 0 else 3.1
        angle = _math.radians(-90 + i * 36)
        point = QPointF(10.0 + radius * _math.cos(angle), 10.4 + radius * _math.sin(angle))
        if i == 0:
            path.moveTo(point)
        else:
            path.lineTo(point)
    path.closeSubpath()
    return path


def _draw_star(p: QPainter, c: QColor) -> None:

    pen = _pen(c, 1.4)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(_star_path())


def _draw_star_filled(p: QPainter, c: QColor) -> None:

    pen = _pen(c, 1.4)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(c)
    p.drawPath(_star_path())


_GLYPHS["refresh"] = _draw_refresh
_GLYPHS["star"] = _draw_star
_GLYPHS["star_filled"] = _draw_star_filled




def _draw_keyboard(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(2.6, 5.2, 14.8, 9.8), 2.0, 2.0)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    for row, y in enumerate((7.9, 10.3)):
        for i in range(4 if row == 0 else 3):
            x = 5.6 + i * 2.95 + (1.4 if row else 0.0)
            p.drawEllipse(QPointF(x, y), 0.8, 0.8)
    p.setPen(_pen(c, 1.5))
    p.drawLine(QPointF(7.4, 12.7), QPointF(12.6, 12.7))


def _draw_info(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QRectF(2.8, 2.8, 14.4, 14.4))
    p.drawLine(QPointF(10.0, 9.2), QPointF(10.0, 13.8))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    p.drawEllipse(QPointF(10.0, 6.5), 1.05, 1.05)


_GLYPHS["keyboard"] = _draw_keyboard
_GLYPHS["info"] = _draw_info




def _draw_polygon_add(p: QPainter, c: QColor) -> None:



    corners = [QPointF(3.4, 4.0), QPointF(13.8, 3.2), QPointF(11.0, 11.0),
               QPointF(4.2, 14.2)]
    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(QPolygonF(corners))
    for corner in (corners[0], corners[1], corners[3]):
        _vertex_handle(p, c, corner)
    badge = QPainterPath()
    badge.addEllipse(QPointF(14.2, 14.2), 3.9, 3.9)
    _knock_out(p, badge, 0.6)
    p.setPen(_pen(c, 1.7))
    p.drawLine(QPointF(14.2, 11.2), QPointF(14.2, 17.2))
    p.drawLine(QPointF(11.2, 14.2), QPointF(17.2, 14.2))


_GLYPHS["polygon_add"] = _draw_polygon_add










def _draw_right_angle(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(QPolygonF(
        [QPointF(4.2, 3.6), QPointF(4.2, 15.6), QPointF(16.2, 15.6)]))
    p.drawRect(QRectF(4.2, 11.6, 4.0, 4.0))


def _draw_round_corner(p: QPainter, c: QColor) -> None:




    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(4.2, 3.6)
    path.lineTo(4.2, 6.0)
    path.cubicTo(4.2, 11.6, 8.2, 15.6, 13.8, 15.6)
    path.lineTo(16.2, 15.6)
    p.drawPath(path)


def _draw_fill_holes(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(QPolygonF([
        QPointF(3.6, 7.4), QPointF(9.4, 3.2), QPointF(16.4, 6.6),
        QPointF(15.0, 14.6), QPointF(6.4, 16.4), QPointF(2.6, 12.0)]))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(c))
    p.drawEllipse(QPointF(9.6, 10.2), 2.6, 2.6)


def _draw_simplify(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(QPolygonF([
        QPointF(2.8, 6.4), QPointF(5.4, 8.8), QPointF(6.8, 5.8),
        QPointF(9.2, 9.2), QPointF(11.4, 6.0), QPointF(13.8, 8.6),
        QPointF(16.6, 6.2)]))
    p.setPen(_pen(c, 1.8))
    p.drawLine(QPointF(2.8, 14.2), QPointF(16.6, 14.2))


def _draw_trim(p: QPainter, c: QColor) -> None:

    p.setPen(_pen(c, 1.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(QPolygonF([
        QPointF(2.8, 13.2), QPointF(7.0, 13.2), QPointF(10.0, 3.6),
        QPointF(12.6, 13.2), QPointF(17.2, 13.2)]))
    p.setPen(_pen(c, 1.9))
    p.drawLine(QPointF(7.6, 10.2), QPointF(10.8, 12.6))


def _draw_grow_shrink(p: QPainter, c: QColor) -> None:


    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(6.2, 6.2, 7.6, 7.6), 1.2, 1.2)
    p.drawLine(QPointF(13.8, 6.2), QPointF(17.4, 2.6))
    p.drawLine(QPointF(17.4, 2.6), QPointF(14.9, 2.8))
    p.drawLine(QPointF(17.4, 2.6), QPointF(17.2, 5.1))
    p.drawLine(QPointF(6.2, 13.8), QPointF(2.6, 17.4))
    p.drawLine(QPointF(2.6, 17.4), QPointF(5.1, 17.2))
    p.drawLine(QPointF(2.6, 17.4), QPointF(2.8, 14.9))


def _draw_size_range(p: QPainter, c: QColor) -> None:



    p.setPen(_pen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(2.8, 7.4, 4.0, 4.0))
    p.drawRect(QRectF(10.0, 2.8, 7.2, 8.6))

    p.drawLine(QPointF(3.0, 15.6), QPointF(17.0, 15.6))
    p.drawPolyline(QPolygonF(
        [QPointF(5.2, 13.6), QPointF(3.0, 15.6), QPointF(5.2, 17.6)]))
    p.drawPolyline(QPolygonF(
        [QPointF(14.8, 13.6), QPointF(17.0, 15.6), QPointF(14.8, 17.6)]))


_GLYPHS["right_angle"] = _draw_right_angle
_GLYPHS["round_corner"] = _draw_round_corner
_GLYPHS["fill_holes"] = _draw_fill_holes
_GLYPHS["simplify"] = _draw_simplify
_GLYPHS["trim"] = _draw_trim
_GLYPHS["grow_shrink"] = _draw_grow_shrink
_GLYPHS["size_range"] = _draw_size_range

ICON_NAMES = tuple(_GLYPHS)





FALLBACK_GLYPH = "circle"
_UNKNOWN_REPORTED: set = set()


def _glyph(name: str):

    draw = _GLYPHS.get(name)
    if draw is not None:
        return draw
    if name not in _UNKNOWN_REPORTED:
        _UNKNOWN_REPORTED.add(name)
        try:
            from qgis.core import Qgis

            from ..core.logging_utils import log

            log(f"Unknown icon {name!r}; drawing {FALLBACK_GLYPH} instead",
                Qgis.MessageLevel.Warning)
        except Exception:  # nosec B110
            pass
    return _GLYPHS[FALLBACK_GLYPH]


def render_pixmap(name: str, color: QColor, size: int, ratio: float = 1.0) -> QPixmap:

    key = (name, color.rgba(), int(size), round(float(ratio), 2))
    cached = _PIXMAP_CACHE.get(key)
    if cached is not None:
        return cached
    pixmap = _render_pixmap(name, color, size, ratio)
    if len(_PIXMAP_CACHE) >= _PIXMAP_CACHE_MAX:
        _PIXMAP_CACHE.clear()
    _PIXMAP_CACHE[key] = pixmap
    return pixmap


def _render_pixmap(name: str, color: QColor, size: int, ratio: float) -> QPixmap:
    physical = max(1, int(round(size * ratio)))
    pixmap = QPixmap(physical, physical)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.scale(size / BOX, size / BOX)
        _glyph(name)(painter, color)
    finally:
        painter.end()
    return pixmap


def make_icon(name: str, color: QColor, size: int = 20, ratio: float = 1.0,
              disabled_color: QColor | None = None) -> QIcon:





    disabled_key = disabled_color.rgba() if disabled_color is not None else None
    key = (name, color.rgba(), int(size), round(float(ratio), 2), disabled_key)
    cached = _ICON_CACHE.get(key)
    if cached is not None:
        return cached
    icon = QIcon()
    ratios = {1.0, 2.0, round(float(ratio), 2)}
    for r in sorted(ratios):
        icon.addPixmap(render_pixmap(name, color, size, r))
        if disabled_color is not None:
            icon.addPixmap(render_pixmap(name, disabled_color, size, r), QIcon.Mode.Disabled)
    if len(_ICON_CACHE) >= _ICON_CACHE_MAX:
        _ICON_CACHE.clear()
    _ICON_CACHE[key] = icon
    return icon


def icon_for(widget, name: str, size: int = 20, color: QColor | None = None,
             disabled_color: QColor | None = None) -> QIcon:

    ink = color if color is not None else ink_of(widget)
    return make_icon(name, QColor(ink), size, widget_pixel_ratio(widget), disabled_color)


def icon_factory(name: str, size: int = 20, color: QColor | None = None):

    return lambda widget: icon_for(widget, name, size, color)


def pixmap_for(widget, name: str, size: int = 16, color: QColor | None = None) -> QPixmap:

    ink = color if color is not None else ink_of(widget)
    return render_pixmap(name, QColor(ink), size, widget_pixel_ratio(widget))





_THEME_CACHE: dict = {}


def theme_pixmap(widget, name: str, size: int = 18) -> QPixmap | None:





    ratio = widget_pixel_ratio(widget)
    key = (str(name), int(size), round(float(ratio), 2))
    if key in _THEME_CACHE:
        return _THEME_CACHE[key]
    pixmap = None
    try:
        from qgis.core import QgsApplication

        icon = QgsApplication.getThemeIcon(str(name))
        if icon is not None and not icon.isNull():
            physical = max(1, int(round(size * ratio)))
            candidate = icon.pixmap(physical, physical)
            if not candidate.isNull():
                candidate.setDevicePixelRatio(ratio)
                pixmap = candidate
    except Exception:  # noqa: BLE001
        pixmap = None
    _THEME_CACHE[key] = pixmap
    return pixmap


def logo_pixmap(widget, size: int = 18) -> QPixmap:








    ratio = widget_pixel_ratio(widget)
    physical = max(1, int(round(size * ratio)))
    key = (physical, round(float(ratio), 3))
    cached = _LOGO_CACHE.get(key)
    if cached is not None:
        return cached
    source = QPixmap(_LOGO_FILE)
    if source.isNull():
        pixmap = QPixmap(physical, physical)
        pixmap.fill(Qt.GlobalColor.transparent)
    else:
        pixmap = source.scaledToHeight(
            physical, Qt.TransformationMode.SmoothTransformation)
    pixmap.setDevicePixelRatio(ratio)
    _LOGO_CACHE[key] = pixmap
    return pixmap


def logo_size(height: int) -> QSize:





    return QSize(max(1, math.ceil(height * LOGO_ASPECT)), height)


def logo_icon(widget, size: int = 18) -> QIcon:
    return QIcon(logo_pixmap(widget, size))


def spinner_frames(color: QColor, size: int = 14, count: int = 12, ratio: float = 1.0) -> list:

    frames = []
    for i in range(count):
        physical = max(1, int(round(size * ratio)))
        pixmap = QPixmap(physical, physical)
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pen = _pen(color, max(1.5, size / 7.0))
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            margin = pen.widthF()
            rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
            angle = int(-(360 / count) * i * 16)
            painter.drawArc(rect, angle, 270 * 16)
        finally:
            painter.end()
        frames.append(pixmap)
    return frames


def icon_size(size: int = 20) -> QSize:
    return QSize(size, size)
