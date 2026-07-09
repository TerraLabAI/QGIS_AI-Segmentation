







from __future__ import annotations

from qgis.PyQt.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from qgis.PyQt.QtWidgets import QWidget

from ..core import qt_compat as QtC
from ..core.i18n import tr
from ..core.server_dials import dial_in_range
from .dock.font_scale import scale_point_size, widget_pixel_ratio


_AUTO_LOOP_PERIOD_MS = 5800


_FRAME_INTERVAL_MS = 33

_DIVIDER_COLOR = QColor("#FFFFFF")
_DIVIDER_SHADOW = QColor(0, 0, 0, 64)


_HANDLE_ARROW = QColor(32, 32, 32)
_HANDLE_RADIUS_PX = 14
_DIVIDER_LINE_PX = 2


_BADGE_BG_BEFORE = QColor(0, 0, 0, 158)
_BADGE_BG_AFTER = QColor(0, 0, 0, 158)
_BADGE_TEXT = QColor("#FFFFFF")

_PLACEHOLDER_BG = QColor(128, 128, 128, 36)
_PLACEHOLDER_TEXT = QColor(128, 128, 128, 230)


def _ease_in_out_inverse(y: float) -> float:

    y = max(0.0, min(1.0, float(y)))
    if y < 0.5:
        return (y / 4.0) ** (1.0 / 3.0)
    return 1.0 - ((2.0 * (1.0 - y)) ** (1.0 / 3.0)) / 2.0


def _ease_in_out(t: float) -> float:

    if t < 0.5:
        return 4 * t * t * t
    p = -2 * t + 2
    return 1 - p * p * p / 2


class BeforeAfterSlider(QWidget):






    clicked = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        auto_loop: bool = True,
        show_badges: bool = True,
        example_badge: str | None = None,
        handle_grab_only: bool = False,
    ):
        super().__init__(parent)



        self._handle_grab_only = handle_grab_only
        self._show_badges = show_badges



        self._badge_before_text = tr("Before")
        self._badge_after_text = tr("After")


        self._example_badge = example_badge or None


        self._placeholder_text = tr("Loading...")




        self._corner_radius = 10.0
        self._round_bottom = True
        self.setMinimumHeight(140)
        self.setMouseTracking(False)
        self._before: QPixmap | None = None
        self._after: QPixmap | None = None



        self._unavailable: set[str] = set()





        self._cover_cache: dict[str, tuple] = {}

        self._pos = 0.5
        self._dragging = False
        self._hovering = False
        self._elapsed_ms = 0



        self._press_x: float | None = None
        self._moved_far = False




        self._auto_loop = auto_loop
        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)


        self.setAccessibleName(tr("Before and after comparison"))






    def showEvent(self, ev):  # noqa: N802
        if self._auto_loop and not self._timer.isActive():
            self._timer.start()
        super().showEvent(ev)

    def hideEvent(self, ev):  # noqa: N802
        self._timer.stop()
        super().hideEvent(ev)

    def closeEvent(self, ev):  # noqa: N802
        self._timer.stop()
        super().closeEvent(ev)

    def resizeEvent(self, ev):  # noqa: N802

        self._cover_cache.clear()
        super().resizeEvent(ev)

    def deleteLater(self):
        self._timer.stop()
        super().deleteLater()



    def set_before(self, pixmap: QPixmap | None) -> None:
        self._before = pixmap if pixmap and not pixmap.isNull() else None
        if self._before is not None:
            self._unavailable.discard("before")
        self._cover_cache.pop("before", None)
        self.update()

    def set_after(self, pixmap: QPixmap | None) -> None:
        self._after = pixmap if pixmap and not pixmap.isNull() else None
        if self._after is not None:
            self._unavailable.discard("after")
        self._cover_cache.pop("after", None)
        self.update()

    def mark_unavailable(self, which: str) -> None:







        if which in ("before", "after"):
            self._unavailable.add(which)
            self.update()

    def has_images(self) -> bool:
        return self._before is not None and self._after is not None

    def _solo_pixmap(self) -> tuple[QPixmap, str] | None:

        if self._before is not None and self._after is None and "after" in self._unavailable:
            return self._before, "before"
        if self._after is not None and self._before is None and "before" in self._unavailable:
            return self._after, "after"
        return None

    def set_placeholder_text(self, text: str) -> None:

        self._placeholder_text = text or ""
        if self._before is None and self._after is None:
            self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(280, 160)



    def _on_tick(self) -> None:
        if not self._auto_loop or self._hovering or self._dragging:
            return
        loop_period_ms = dial_in_range(
            "tuning.ui.before_after_loop_ms", _AUTO_LOOP_PERIOD_MS, 2000, 20000)
        self._elapsed_ms = (self._elapsed_ms + _FRAME_INTERVAL_MS) % loop_period_ms

        half = loop_period_ms / 2
        t = self._elapsed_ms / half
        if t > 1.0:
            t = 2.0 - t
        self._pos = _ease_in_out(t)
        self.update()



    def enterEvent(self, ev):  # noqa: N802
        self._hovering = True
        self.setMouseTracking(True)
        super().enterEvent(ev)

    def leaveEvent(self, ev):  # noqa: N802




        self._hovering = False
        if not self._dragging:
            self.setMouseTracking(False)
            self._resume_loop_from_divider()
        self.unsetCursor()
        super().leaveEvent(ev)

    def _resume_loop_from_divider(self) -> None:







        if not self._auto_loop:
            return
        try:
            loop_period_ms = dial_in_range(
                "tuning.ui.before_after_loop_ms", _AUTO_LOOP_PERIOD_MS, 2000, 20000)
            half = loop_period_ms / 2
            self._elapsed_ms = int(_ease_in_out_inverse(self._pos) * half)
        except Exception:  # noqa: BLE001
            pass  # nosec B110



    _CLICK_DRAG_THRESHOLD_PX = 5


    _HANDLE_GRAB_PX = 16

    def mousePressEvent(self, ev):  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self._press_x = self._event_x(ev)
            self._moved_far = False
            if self._handle_grab_only:
                divider_x = self._pos * max(1, self.width())
                self._dragging = abs(self._press_x - divider_x) <= self._HANDLE_GRAB_PX
            else:
                self._dragging = True
            if self._dragging:
                self._update_pos_from_event(ev)
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            was_pressed = self._press_x is not None
            was_dragging = self._dragging
            moved_far = self._moved_far
            self._dragging = False
            self._moved_far = False
            self._press_x = None


            if not self._hovering:
                self.setMouseTracking(False)
                self._resume_loop_from_divider()



            if was_pressed and not moved_far and not (self._handle_grab_only and was_dragging):
                self.clicked.emit()
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):  # noqa: N802
        self._update_hover_cursor(ev)
        if self._dragging:
            if not self._moved_far and self._press_x is not None:
                if abs(self._event_x(ev) - self._press_x) > self._CLICK_DRAG_THRESHOLD_PX:
                    self._moved_far = True
            self._update_pos_from_event(ev)
        super().mouseMoveEvent(ev)

    def _update_hover_cursor(self, ev) -> None:



        try:
            if self._before is None or self._after is None:
                self.unsetCursor()
                return
            on_handle = (not self._handle_grab_only or self._dragging or abs(
                self._event_x(ev) - self._pos * max(1, self.width()))
                <= self._HANDLE_GRAB_PX)
            self.setCursor(Qt.CursorShape.SplitHCursor if on_handle
                           else Qt.CursorShape.PointingHandCursor)
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    @staticmethod
    def _event_x(ev) -> float:


        return QtC.event_pos(ev).x()

    def _update_pos_from_event(self, ev) -> None:
        x = QtC.event_pos(ev).x()
        w = max(1, self.width())
        self._pos = max(0.0, min(1.0, x / w))
        self.update()



    def set_card_corners(self, radius: float, top_only: bool = False) -> None:







        self._corner_radius = float(radius)
        self._round_bottom = not top_only
        self.update()

    def paintEvent(self, ev):  # noqa: N802



        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            rect = self.rect()
            radius = float(self._corner_radius)






            path = QPainterPath()
            path.addRoundedRect(QRectF(rect), radius, radius)
            if not self._round_bottom and rect.height() > radius:
                square = QPainterPath()
                square.addRect(QRectF(
                    rect.x(), rect.y() + rect.height() - radius,
                    rect.width(), radius))
                path = path.united(square)
            painter.setClipPath(path)


            if self._before is None and self._after is None:
                self._paint_placeholder(painter, rect)
                painter.end()
                return




            solo = self._solo_pixmap()
            if solo is not None:
                self._draw_pixmap_cover(painter, solo[0], rect, solo[1])
                if self._example_badge:
                    self._draw_example_badge(painter, rect, self._example_badge)
                painter.end()
                return


            split_x = int(rect.width() * self._pos)


            if self._before is not None:
                painter.save()
                painter.setClipRect(QRectF(0, 0, split_x, rect.height()))
                self._draw_pixmap_cover(painter, self._before, rect, "before")
                painter.restore()
            else:
                painter.save()
                painter.setClipRect(QRectF(0, 0, split_x, rect.height()))
                painter.fillRect(rect, _PLACEHOLDER_BG)
                painter.restore()


            if self._after is not None:
                painter.save()
                painter.setClipRect(QRectF(split_x, 0, rect.width() - split_x, rect.height()))
                self._draw_pixmap_cover(painter, self._after, rect, "after")
                painter.restore()
            else:
                painter.save()
                painter.setClipRect(QRectF(split_x, 0, rect.width() - split_x, rect.height()))
                painter.fillRect(rect, _PLACEHOLDER_BG)
                painter.restore()


            pen_shadow = QPen(_DIVIDER_SHADOW)
            pen_shadow.setWidth(_DIVIDER_LINE_PX + 2)
            painter.setPen(pen_shadow)
            painter.drawLine(split_x, 0, split_x, rect.height())
            pen = QPen(_DIVIDER_COLOR)
            pen.setWidth(_DIVIDER_LINE_PX)
            painter.setPen(pen)
            painter.drawLine(split_x, 0, split_x, rect.height())


            handle_y = rect.height() // 2
            painter.setPen(QPen(_DIVIDER_SHADOW, 1))
            painter.setBrush(QBrush(_DIVIDER_COLOR))
            painter.drawEllipse(
                QPointF(split_x, handle_y),
                _HANDLE_RADIUS_PX,
                _HANDLE_RADIUS_PX,
            )

            painter.setPen(QPen(_HANDLE_ARROW, 2))
            ay = handle_y
            painter.drawLine(split_x - 5, ay, split_x - 1, ay - 4)
            painter.drawLine(split_x - 5, ay, split_x - 1, ay + 4)
            painter.drawLine(split_x + 5, ay, split_x + 1, ay - 4)
            painter.drawLine(split_x + 5, ay, split_x + 1, ay + 4)


            if self._show_badges:
                self._draw_badge(painter, self._badge_before_text, y=8,
                                 bg=_BADGE_BG_BEFORE, x=8)
                self._draw_badge(
                    painter,
                    self._badge_after_text,
                    y=8,
                    bg=_BADGE_BG_AFTER,
                    right=rect.width() - 8,
                )

            if self._example_badge:
                self._draw_example_badge(painter, rect, self._example_badge)

            painter.end()
        except Exception:  # noqa: BLE001
            return

    def _draw_pixmap_cover(self, painter: QPainter, pm: QPixmap, rect, slot: str) -> None:




        if pm.isNull() or rect.width() <= 0 or rect.height() <= 0:
            return
        pw, ph = pm.width(), pm.height()
        if pw <= 0 or ph <= 0:
            return
        widget_ar = rect.width() / rect.height()
        pix_ar = pw / ph
        if pix_ar > widget_ar:

            scale_h = rect.height() / ph
            scaled_w = pw * scale_h
            offset_x = (scaled_w - rect.width()) / 2
            target = QRectF(-offset_x, 0, scaled_w, rect.height())
        else:
            scale_w = rect.width() / pw
            scaled_h = ph * scale_w
            offset_y = (scaled_h - rect.height()) / 2
            target = QRectF(0, -offset_y, rect.width(), scaled_h)








        ratio = widget_pixel_ratio(self)
        target_w = max(1, round(target.width() * ratio))
        target_h = max(1, round(target.height() * ratio))



        cache_key = (pm.cacheKey(), target_w, target_h)
        cached = self._cover_cache.get(slot)
        if cached is not None and cached[0] == cache_key:
            scaled = cached[1]
        else:
            scaled = pm.scaled(
                target_w, target_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            scaled.setDevicePixelRatio(ratio)
            self._cover_cache[slot] = (cache_key, scaled)
        painter.drawPixmap(QPointF(target.x(), target.y()), scaled)

    def _draw_example_badge(self, painter: QPainter, rect, text: str) -> None:



        f = painter.font()
        f.setPointSize(scale_point_size(9))
        f.setWeight(QFont.Weight.DemiBold)
        painter.setFont(f)
        metrics = painter.fontMetrics()
        bw = float(metrics.horizontalAdvance(text)) + 20.0



        bh = max(18.0, float(metrics.height()) + 4.0)
        bx = (rect.width() - bw) / 2.0
        badge = QRectF(bx, 8, bw, bh)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(_BADGE_BG_BEFORE))
        painter.drawRoundedRect(badge, bh / 2.0, bh / 2.0)
        painter.setPen(QPen(_BADGE_TEXT))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_badge(self, painter: QPainter, text: str, y: int, bg: QColor,
                    x: float | None = None, right: float | None = None) -> None:






        f = painter.font()
        f.setPointSize(scale_point_size(9))
        f.setWeight(QFont.Weight.DemiBold)
        painter.setFont(f)
        metrics = painter.fontMetrics()
        bw = max(44.0, float(metrics.horizontalAdvance(text)) + 18.0)
        bh = max(18.0, float(metrics.height()) + 4.0)
        if x is not None:
            bx = float(x)
        elif right is not None:
            bx = float(right) - bw
        else:
            bx = 0.0
        rect = QRectF(bx, float(y), bw, bh)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(rect, bh / 2.0, bh / 2.0)
        painter.setPen(QPen(_BADGE_TEXT))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_placeholder(self, painter: QPainter, rect) -> None:
        painter.fillRect(rect, _PLACEHOLDER_BG)
        painter.setPen(QPen(_PLACEHOLDER_TEXT))
        f = painter.font()
        f.setPointSize(scale_point_size(9))
        painter.setFont(f)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._placeholder_text)
