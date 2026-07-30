"""What the dock footer row paints for itself: the two vector glyphs, and the
cross-promo button that gives way when the panel is narrow.

Split out of widgets.py so the footer's own pieces sit together and neither
file grows past the size two agents can edit at once.
"""
from __future__ import annotations

import math

from qgis.PyQt.QtCore import QPointF, Qt
from qgis.PyQt.QtGui import (
    QFontMetrics,
    QFontMetricsF,
    QIcon,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
)
from qgis.PyQt.QtWidgets import QSizePolicy

from .font_scale import widget_pixel_ratio
from .widgets import _FooterIconButton

# Side of a footer glyph, in the box the button reserves for its icon.
#
# Everything below is AI Edit's footer glyph code, kept identical on purpose:
# the box, the pixmap, the tooth and page numbers, and the fact that the
# drawing is NOT scaled by the pixmap ratio. AI Edit's icons are the ones that
# look right on Windows, so they are the target, and a version of this file
# that improved on them looked worse. If you change a number here, change it
# there too, or the two footers stop matching.
FOOTER_GLYPH_PX = 20

# What the cross-promo button may shrink to. Enough for the first word plus the
# ellipsis, so a narrow panel still shows there is something to click.
FOOTER_CTA_MIN_PX = 56


def _footer_paint_ratio(widget) -> float:
    """Pixels the SCREEN puts behind one drawing unit, never zero.

    The widget's own ratio is not the same number: Qt rounds it to a whole one,
    so a 125% Windows display answers 1 on the widget and 1.25 on its screen.
    That is the number QIcon asks the glyph for, and a drawing painted at the
    widget's answer is stretched to reach it, which is what turned the book
    into a smudge.
    """
    ratio = 0.0
    try:
        screen = widget.screen()
        if screen is not None:
            ratio = float(screen.devicePixelRatio())
    except (AttributeError, RuntimeError):
        ratio = 0.0
    return max(ratio, widget_pixel_ratio(widget), 1.0)


def _footer_glyph_pixmap(widget) -> QPixmap:
    """Transparent square, painted at the screen's own pixel count and tagged
    with the ratio, so the glyph stays sharp at any display scale."""
    ratio = _footer_paint_ratio(widget)
    physical = max(1, int(round(FOOTER_GLYPH_PX * ratio)))
    pixmap = QPixmap(physical, physical)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    return pixmap


def footer_gear_icon(widget) -> QIcon:
    """Settings glyph: a vector gear in the palette text colour.

    Replaces the U+2699 character, which Windows draws through its emoji font
    as a colour bitmap at the full em box: one coloured picture in a row of
    flat marks, and taller than the buttons beside it.
    """
    size = FOOTER_GLYPH_PX
    ink = widget.palette().color(QPalette.ColorRole.WindowText)
    pixmap = _footer_glyph_pixmap(widget)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    centre = size / 2.0
    teeth = 8
    step = 2.0 * math.pi / teeth
    r_tip = size * 0.46
    r_root = size * 0.34
    half_tip = step * 0.18
    half_root = step * 0.30
    path = QPainterPath()
    for i in range(teeth):
        angle = i * step
        corners = (
            (angle - half_root, r_root),
            (angle - half_tip, r_tip),
            (angle + half_tip, r_tip),
            (angle + half_root, r_root),
        )
        for ang, radius in corners:
            point = QPointF(centre + radius * math.cos(ang),
                            centre + radius * math.sin(ang))
            if i == 0 and ang == corners[0][0]:
                path.moveTo(point)
            else:
                path.lineTo(point)
    path.closeSubpath()
    # Centre hole: OddEven fill subtracts it from the gear body.
    path.addEllipse(QPointF(centre, centre), size * 0.15, size * 0.15)
    path.setFillRule(Qt.FillRule.OddEvenFill)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(ink)
    painter.drawPath(path)
    painter.end()
    return QIcon(pixmap)


def footer_book_icon(widget) -> QIcon:
    """Tutorial glyph: two stroked pages meeting on a spine, in the palette
    text colour.

    Replaces U+1F4D6, which Windows draws as a full-colour bitmap. The stroke
    width is set so the painted weight matches the gear beside it rather than
    fading next to it.
    """
    size = FOOTER_GLYPH_PX
    ink = widget.palette().color(QPalette.ColorRole.WindowText)
    pixmap = _footer_glyph_pixmap(widget)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(ink)
    pen.setWidthF(2.2)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    centre = size / 2.0
    top, bottom = size * 0.18, size * 0.82
    edge = size * 0.10
    for outer in (edge, size - edge):
        page = QPainterPath()
        page.moveTo(QPointF(centre, top))
        # Both curves bow away from the spine, so the pages read as open.
        page.quadTo(
            QPointF((centre + outer) / 2.0, top - size * 0.09),
            QPointF(outer, top - size * 0.05),
        )
        page.lineTo(QPointF(outer, bottom - size * 0.05))
        page.quadTo(
            QPointF((centre + outer) / 2.0, bottom - size * 0.09),
            QPointF(centre, bottom),
        )
        # closeSubpath draws the spine back up to the start point.
        page.closeSubpath()
        painter.drawPath(page)
    painter.end()
    return QIcon(pixmap)


class _ElidingFooterButton(_FooterIconButton):
    """Footer button whose label gives way when the panel is narrow.

    The dock is a scroll area, so a child that refuses to go below its own text
    width sets the minimum width of the whole panel. A long label in the footer
    was enough to push the panel past a normal dock, and the panel then
    scrolled sideways with the gear and the help button off screen.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full_label = ""
        # Maximum, not the QToolButton default of Fixed. A layout asks a Fixed
        # widget for its sizeHint and uses that as the minimum too, so the
        # minimumSizeHint below would never be read. Maximum says: never wider
        # than the full label, free to shrink down to that minimum.
        self.setSizePolicy(
            QSizePolicy.Policy.Maximum, self.sizePolicy().verticalPolicy())

    def set_label(self, text: str) -> None:
        """The text to show whenever there is room for all of it."""
        self._full_label = text
        self._apply_elided_label()

    def sizeHint(self):  # noqa: N802 - Qt override
        """Preferred width is the FULL label, not the elided one, or the layout
        would keep handing back the width the last elision asked for."""
        hint = super().sizeHint()
        chrome = self._label_chrome()
        hint.setWidth(self._full_label_width() + chrome)
        return hint

    def _full_label_width(self) -> int:
        """Width the whole label needs, rounded UP.

        QFontMetrics reports whole pixels while elidedText measures in floats.
        A label that truly needs 208.4px asked the layout for 208, was given
        exactly that, and then elided two characters away over four tenths of a
        pixel. Rounding up here is what keeps the button honest: it elides
        because the dock is narrow, never because of the rounding.
        """
        return int(math.ceil(
            QFontMetricsF(self.font()).horizontalAdvance(self._full_label)))

    def minimumSizeHint(self):  # noqa: N802 - Qt override
        hint = super().minimumSizeHint()
        hint.setWidth(min(hint.width(), FOOTER_CTA_MIN_PX))
        return hint

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._apply_elided_label()

    def _apply_elided_label(self) -> None:
        room = max(0, self.width() - self._label_chrome())
        if room >= self._full_label_width():
            # There is room for all of it, so show all of it. Handing a width
            # equal to its own measurement to elidedText is exactly the case
            # that used to come back one ellipsis short.
            text = self._full_label
        else:
            text = QFontMetrics(self.font()).elidedText(
                self._full_label, Qt.TextElideMode.ElideRight, room)
        # Only on a real change: setText asks for a new layout, and a layout
        # pass that ends in another setText never settles.
        if text != self.text():
            super().setText(text)

    def _label_chrome(self) -> int:
        """Everything the button spends on a label besides the glyphs: the
        stylesheet padding and the frame, read back from Qt rather than
        repeated here."""
        metrics = QFontMetrics(self.font())
        return max(
            0, super().sizeHint().width() - metrics.horizontalAdvance(self.text()))
