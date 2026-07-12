"""Canvas helpers for the Automatic zone: the on-canvas delete "x" badge
(ZoneDeleteBadge) plus the event filters that make it clickable
(ZoneBadgeClickFilter) and let Escape clear the zone (ZoneEscapeFilter).

The zone itself is drawn point-by-point by PolygonZoneMapTool; the legacy
rectangle-drawing tool that used to live here was removed as dead code.
"""
from __future__ import annotations

import logging
from typing import Optional

from qgis.core import QgsPointXY
from qgis.gui import QgsMapCanvasItem
from qgis.PyQt.QtCore import QEvent, QObject, QPointF, QRectF, Qt
from qgis.PyQt.QtGui import QColor, QPainter, QPen

from ..core.i18n import tr
from ..core.qt_compat import event_pos
from .canvas_palette import (
    BADGE_FILL,
    BADGE_FILL_DISABLED,
    BADGE_FILL_HOVER,
    BADGE_X,
)

logger = logging.getLogger("AISegmentation")


class ZoneDeleteBadge(QgsMapCanvasItem):
    """Floating x badge anchored to a map point (ported from AI Edit).

    Lives in the canvas's QGraphicsScene so it follows the canvas during
    pan/zoom (the scene gets pixel-shifted during pan, so anything in it
    stays aligned with the rubber band - a plain widget parented to the
    viewport would not). Click handling is done externally via
    :meth:`hit_test` (see :class:`ZoneBadgeClickFilter`), because the
    active map tool eats mouse events before the scene items see them.
    """

    RADIUS = 12
    _BRAND_BLUE = BADGE_FILL
    _DISABLED_BG = BADGE_FILL_DISABLED
    _HOVER_BG = BADGE_FILL_HOVER

    def __init__(self, canvas):
        super().__init__(canvas)
        self._anchor: Optional[QgsPointXY] = None
        self._enabled = True
        self._hovered = False
        self.setZValue(10000)
        self.setAcceptHoverEvents(True)
        self.setToolTip(tr("Remove zone"))

    def set_anchor(self, point: QgsPointXY) -> None:
        self._anchor = point
        self.updatePosition()
        self.update()

    def set_enabled(self, enabled: bool) -> None:
        if self._enabled == enabled:
            return
        self._enabled = enabled
        if enabled:
            self.setToolTip(tr("Remove zone"))
        else:
            self.setToolTip(tr("Cancel the running detection first"))
        self.update()

    def hoverEnterEvent(self, event):  # noqa: N802 (Qt API)
        self._hovered = True
        self.update()

    def hoverLeaveEvent(self, event):  # noqa: N802 (Qt API)
        self._hovered = False
        self.update()

    def is_enabled(self) -> bool:
        return self._enabled

    def hit_test(self, canvas_pt) -> bool:
        """True when a canvas-pixel point lands inside the badge circle."""
        if self._anchor is None or not self.isVisible():
            return False
        center = self.toCanvasCoordinates(self._anchor)
        dx = canvas_pt.x() - center.x()
        dy = canvas_pt.y() - center.y()
        return (dx * dx + dy * dy) <= (self.RADIUS * self.RADIUS)

    def updatePosition(self) -> None:  # noqa: N802 (Qt API)
        if self._anchor is None:
            return
        self.setPos(self.toCanvasCoordinates(self._anchor))

    def boundingRect(self):  # noqa: N802 (Qt API)
        r = self.RADIUS + 2
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter, option, widget):
        if self._anchor is None:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Red on hover = a clear "this deletes the zone" affordance; disabled
        # keeps its faded blue (no hover feedback while a run is in flight).
        if self._enabled:
            bg = self._HOVER_BG if self._hovered else self._BRAND_BLUE
        else:
            bg = self._DISABLED_BG
        painter.setBrush(bg)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(0, 0), self.RADIUS, self.RADIUS)
        line_color = (
            BADGE_X if self._enabled else QColor(255, 255, 255, 153)
        )
        pen = QPen(line_color, 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        d = self.RADIUS * 0.45
        painter.drawLine(QPointF(-d, -d), QPointF(d, d))
        painter.drawLine(QPointF(-d, d), QPointF(d, -d))


class ZoneBadgeClickFilter(QObject):
    """Event filter on the canvas viewport that routes clicks to the badge.

    Whatever map tool is active (pan, zoom, ...) receives mouse events
    before scene items, so the badge can never see its own clicks. This
    filter intercepts the press at the viewport level, hit-tests the badge
    and consumes the whole press -> release gesture when it lands inside.

    The action fires on RELEASE, not press: the click handler clears the
    zone, which tears down this very filter, so if it ran on press the
    following release would escape to the active map tool. A polygon tool
    that adds a vertex on release would then drop a stray point right under
    the x. Firing on release (after both press and release are swallowed)
    avoids that. A whole press -> release -> double-click sequence started on
    the badge is consumed so no partial gesture ever reaches the map tool.
    """

    def __init__(self, badge: ZoneDeleteBadge, on_clicked, parent=None):
        super().__init__(parent)
        self._badge = badge
        self._on_clicked = on_clicked
        self._armed = False  # a press landed on the badge; swallow its release

    def eventFilter(self, _obj, event):  # noqa: N802 (Qt API)
        et = event.type()
        if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            if self._badge.hit_test(event_pos(event)):
                self._armed = True
                return True  # swallow even when disabled: never pan under the badge
            self._armed = False
            return False
        if et == QEvent.Type.MouseButtonRelease and self._armed:
            self._armed = False
            inside = self._badge.hit_test(event_pos(event))
            # Fire only on a clean click that both started and ended on the
            # badge; either way swallow the release so no point is dropped.
            if inside and self._badge.is_enabled():
                self._on_clicked()
            return True
        if et == QEvent.Type.MouseButtonDblClick and self._badge.hit_test(event_pos(event)):
            return True  # never let a double-click on the badge finish a polygon
        return False


class ZoneEscapeFilter(QObject):
    """Canvas-level filter: Escape drops the drawn zone, same as the x badge.

    Installed on the canvas itself (key events never reach the viewport).
    The callback returns True when it consumed the key (a zone existed and
    was cleared); False lets QGIS handle Escape normally.
    """

    def __init__(self, on_escape, parent=None):
        super().__init__(parent)
        self._on_escape = on_escape

    def eventFilter(self, _obj, event):  # noqa: N802 (Qt API)
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            return bool(self._on_escape())
        return False
