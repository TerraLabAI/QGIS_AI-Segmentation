






from __future__ import annotations

import logging

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










    RADIUS = 12
    _BRAND_BLUE = BADGE_FILL
    _DISABLED_BG = BADGE_FILL_DISABLED
    _HOVER_BG = BADGE_FILL_HOVER

    def __init__(self, canvas):
        super().__init__(canvas)
        self._canvas = canvas
        self._anchor: QgsPointXY | None = None
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

    def hoverEnterEvent(self, event):  # noqa: N802
        self._hovered = True
        self.update()

    def hoverLeaveEvent(self, event):  # noqa: N802
        self._hovered = False
        self.update()

    def is_enabled(self) -> bool:
        return self._enabled

    def _scene_offset(self) -> tuple[float, float]:






        try:
            rect = self._canvas.sceneRect()
            return rect.x(), rect.y()
        except (RuntimeError, AttributeError):
            return 0.0, 0.0

    def hit_test(self, canvas_pt) -> bool:

        if self._anchor is None or not self.isVisible():
            return False
        center = self.toCanvasCoordinates(self._anchor)
        off_x, off_y = self._scene_offset()
        dx = canvas_pt.x() - (center.x() - off_x)
        dy = canvas_pt.y() - (center.y() - off_y)
        return (dx * dx + dy * dy) <= (self.RADIUS * self.RADIUS)

    def updatePosition(self) -> None:  # noqa: N802
        if self._anchor is None:
            return
        self.setPos(self.toCanvasCoordinates(self._anchor))

    def boundingRect(self):  # noqa: N802
        r = self.RADIUS + 2
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter, option, widget):



        try:
            if self._anchor is None:
                return
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)



            if self._enabled:
                bg = self._HOVER_BG if self._hovered else self._BRAND_BLUE
            else:
                bg = self._DISABLED_BG
            painter.setBrush(bg)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(0, 0), self.RADIUS, self.RADIUS)
            line_color = (


                BADGE_X if self._enabled else QColor(
                    BADGE_X.red(), BADGE_X.green(), BADGE_X.blue(), 153)
            )
            pen = QPen(line_color, 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            d = self.RADIUS * 0.45
            painter.drawLine(QPointF(-d, -d), QPointF(d, d))
            painter.drawLine(QPointF(-d, d), QPointF(d, -d))
        except Exception:  # noqa: BLE001  # nosec B110
            return


class ZoneBadgeClickFilter(QObject):
















    def __init__(self, badge: ZoneDeleteBadge, on_clicked, parent=None):
        super().__init__(parent)
        self._badge = badge
        self._on_clicked = on_clicked
        self._armed = False

    def eventFilter(self, _obj, event):  # noqa: N802



        try:
            return self._route_mouse_event(event)
        except Exception:
            return False

    def _route_mouse_event(self, event) -> bool:
        et = event.type()
        if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            if self._badge.hit_test(event_pos(event)):
                self._armed = True
                return True
            self._armed = False
            return False
        if (et == QEvent.Type.MouseButtonRelease and self._armed
                and event.button() == Qt.MouseButton.LeftButton):
            self._armed = False
            inside = self._badge.hit_test(event_pos(event))


            if inside and self._badge.is_enabled():
                self._on_clicked()
            return True

        return et == QEvent.Type.MouseButtonDblClick and self._badge.hit_test(event_pos(event))


class ZoneEscapeFilter(QObject):







    def __init__(self, on_escape, parent=None):
        super().__init__(parent)
        self._on_escape = on_escape

    def eventFilter(self, _obj, event):  # noqa: N802




        try:
            if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                return bool(self._on_escape())
        except Exception:
            return False
        return False
