"""BeforeAfterSlider - custom Qt widget that mimics the dashboard's
react slider in PyQt. Two QPixmap layers, vertically split by an animated
divider. Idle: auto-loop animation. Hover: pause + accept mouse drag.

Ported verbatim from the AI Edit plugin so the segment library gallery shows
the same before/after preview. Cross-version: works on PyQt5 (QGIS 3 / Qt 5)
and PyQt6 (QGIS 4 / Qt 6).
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap
from qgis.PyQt.QtWidgets import QWidget

from ..core import qt_compat as QtC
from ..core.i18n import tr
from .dock.font_scale import scale_point_size, widget_pixel_ratio

# Auto-loop period (ms): full oscillation 0 -> 100 -> 0.
_AUTO_LOOP_PERIOD_MS = 5800
# Frame interval (ms): ~30 fps is plenty for the slow triangle-wave loop and
# halves the repaint load versus 60 fps.
_FRAME_INTERVAL_MS = 33
# Divider visuals.
_DIVIDER_COLOR = QColor("#FFFFFF")
_DIVIDER_SHADOW = QColor(0, 0, 0, 64)
_HANDLE_RADIUS_PX = 14
_DIVIDER_LINE_PX = 2
_BADGE_BG_BEFORE = QColor(20, 20, 20, 200)
_BADGE_BG_AFTER = QColor(139, 172, 39, 230)
_BADGE_TEXT = QColor("#FFFFFF")
_PLACEHOLDER_BG = QColor("#1E2A35")
_PLACEHOLDER_TEXT = QColor("#557080")


def _ease_in_out(t: float) -> float:
    """Smooth cubic ease-in-out on [0, 1]."""
    if t < 0.5:
        return 4 * t * t * t
    p = -2 * t + 2
    return 1 - p * p * p / 2


class BeforeAfterSlider(QWidget):
    """Two-image overlay slider with auto-loop + drag.

    Owners set ``before_pixmap`` and ``after_pixmap`` and the widget paints
    itself. Sliders that lack either pixmap show a tinted placeholder.
    """

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
        # Grid cards set this: a press only grabs the divider when it lands on
        # the handle, so the rest of the preview stays a click that opens the
        # card instead of swallowing the gesture as a drag.
        self._handle_grab_only = handle_grab_only
        self._show_badges = show_badges
        # Optional "Example" pill: marks a curated demo so the user reads the
        # before/after as a sample, not the exact result they will get.
        self._example_badge = example_badge or None
        # Text shown over the tinted backdrop while the slider has no images.
        # Owners flip it to "No preview" for cards that will never get one.
        self._placeholder_text = tr("Loading...")
        self.setMinimumHeight(140)
        self.setMouseTracking(False)
        self._before: QPixmap | None = None
        self._after: QPixmap | None = None
        # Sides the owner has given up on (404, decode failure). A side that is
        # merely still loading is None without being flagged here, which is what
        # separates "wait" from "there will never be a second image".
        self._unavailable: set[str] = set()
        # Pre-scaled "object-fit: cover" copies, keyed on (source identity,
        # target size), rebuilt only when the source pixmap or the widget size
        # changes. Without this, every repaint (the auto-loop animation redraws
        # ~30 times a second) would smooth-scale the full-resolution source
        # again for no visual change.
        self._cover_cache: dict[str, tuple] = {}
        # Divider position 0..1 (0 = fully before visible, 1 = fully after).
        self._pos = 0.5
        self._dragging = False
        self._hovering = False
        self._elapsed_ms = 0
        # Click-vs-drag tracking: a press emits `clicked` on release only if
        # the cursor stayed within the threshold; otherwise the gesture is
        # treated as a slider adjustment and no click fires.
        self._press_x: float | None = None
        self._moved_far = False
        # Auto-loop drives a calm idle animation when the slider is the only
        # thing the user looks at (hero, detail view). On a Top Picks grid the
        # caller passes auto_loop=False so 6 sliders don't all wiggle at once;
        # the divider stays at 50/50 until the user drags.
        self._auto_loop = auto_loop
        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)
        # The timer is started from showEvent and stopped on hide, so an
        # auto-loop slider scrolled off-screen or on a hidden tab does not keep
        # repainting at full frame rate.

    # ---- lifecycle -------------------------------------------------------

    def showEvent(self, ev):  # noqa: N802 - Qt signature
        if self._auto_loop and not self._timer.isActive():
            self._timer.start()
        super().showEvent(ev)

    def hideEvent(self, ev):  # noqa: N802 - Qt signature
        self._timer.stop()
        super().hideEvent(ev)

    def closeEvent(self, ev):  # noqa: N802 - Qt signature
        self._timer.stop()
        super().closeEvent(ev)

    def resizeEvent(self, ev):  # noqa: N802 - Qt signature
        # The cover-scaled cache is sized for the OLD widget rect.
        self._cover_cache.clear()
        super().resizeEvent(ev)

    def deleteLater(self):
        self._timer.stop()
        super().deleteLater()

    # ---- public API ------------------------------------------------------

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
        """Declare one side permanently absent.

        A run whose tiles found nothing has an archived input but no result
        overlay, so the comparison has only one side to show. Flagging it makes
        the widget paint that side full-bleed instead of holding half the card
        on a placeholder that will never fill in.
        """
        if which in ("before", "after"):
            self._unavailable.add(which)
            self.update()

    def has_images(self) -> bool:
        return self._before is not None and self._after is not None

    def _solo_pixmap(self) -> tuple[QPixmap, str] | None:
        """The one image to paint full-bleed, when the other side is settled empty."""
        if self._before is not None and self._after is None and "after" in self._unavailable:
            return self._before, "before"
        if self._after is not None and self._before is None and "before" in self._unavailable:
            return self._after, "after"
        return None

    def set_placeholder_text(self, text: str) -> None:
        """Override the empty-state caption (default 'Loading...')."""
        self._placeholder_text = text or ""
        if self._before is None and self._after is None:
            self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt signature
        return QSize(280, 160)

    # ---- animation tick --------------------------------------------------

    def _on_tick(self) -> None:
        if not self._auto_loop or self._hovering or self._dragging:
            return  # paused while user is engaging
        self._elapsed_ms = (self._elapsed_ms + _FRAME_INTERVAL_MS) % _AUTO_LOOP_PERIOD_MS
        # Triangle wave normalised to 0..1.
        half = _AUTO_LOOP_PERIOD_MS / 2
        t = self._elapsed_ms / half
        if t > 1.0:
            t = 2.0 - t
        self._pos = _ease_in_out(t)
        self.update()

    # ---- mouse handling --------------------------------------------------

    def enterEvent(self, ev):  # noqa: N802 - Qt signature
        self._hovering = True
        self.setMouseTracking(True)
        super().enterEvent(ev)

    def leaveEvent(self, ev):  # noqa: N802 - Qt signature
        # Keep an in-progress drag alive when the cursor leaves the widget: the
        # implicit mouse grab from the press keeps delivering move events, so the
        # divider stays draggable (and can be pulled back) until the button is
        # released. Only the hover state ends here.
        self._hovering = False
        if not self._dragging:
            self.setMouseTracking(False)
        super().leaveEvent(ev)

    # If the mouse moved more than this from the press point, treat the
    # interaction as a drag (slider adjust) rather than a click (select).
    _CLICK_DRAG_THRESHOLD_PX = 5
    # In handle_grab_only mode, how far from the divider a press may land and
    # still grab it (the painted handle's radius plus slack).
    _HANDLE_GRAB_PX = 16

    def mousePressEvent(self, ev):  # noqa: N802 - Qt signature
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

    def mouseReleaseEvent(self, ev):  # noqa: N802 - Qt signature
        if ev.button() == Qt.MouseButton.LeftButton:
            was_pressed = self._press_x is not None
            was_dragging = self._dragging
            moved_far = self._moved_far
            self._dragging = False
            self._moved_far = False
            self._press_x = None
            # If the drag ended with the cursor outside the widget, drop the
            # hover-tracking we kept alive during the drag.
            if not self._hovering:
                self.setMouseTracking(False)
            # Click only when the press barely moved; drag-to-adjust must
            # never accidentally select the preset. In handle-grab mode a press
            # that grabbed the handle is always a drag, never a click.
            if was_pressed and not moved_far and not (self._handle_grab_only and was_dragging):
                self.clicked.emit()
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):  # noqa: N802 - Qt signature
        if self._dragging:
            if not self._moved_far and self._press_x is not None:
                if abs(self._event_x(ev) - self._press_x) > self._CLICK_DRAG_THRESHOLD_PX:
                    self._moved_far = True
            self._update_pos_from_event(ev)
        super().mouseMoveEvent(ev)

    @staticmethod
    def _event_x(ev) -> float:
        # QtC.event_pos returns a QPoint that exposes .x() on both Qt5 and
        # Qt6, so callers stop branching on QT_VERSION themselves.
        return QtC.event_pos(ev).x()

    def _update_pos_from_event(self, ev) -> None:
        x = QtC.event_pos(ev).x()
        w = max(1, self.width())
        self._pos = max(0.0, min(1.0, x / w))
        self.update()

    # ---- paint -----------------------------------------------------------

    def paintEvent(self, ev):  # noqa: N802 - Qt signature
        # A paintEvent must never raise: on macOS an escaped exception here
        # hangs and then segfaults QGIS at launch (seen in production), so
        # any failure below just skips the paint for this frame.
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            rect = self.rect()
            radius = 10.0

            # Clip the whole widget to a rounded rect for a soft card look.
            path = QPainterPath()
            path.addRoundedRect(QRectF(rect), radius, radius)
            painter.setClipPath(path)

            # --- backdrop ------------------------------------------------
            if self._before is None and self._after is None:
                self._paint_placeholder(painter, rect)
                painter.end()
                return

            # One side settled empty: no comparison to make, so the image that
            # does exist takes the whole card rather than sharing it with a
            # placeholder. No divider, no BEFORE/AFTER badges.
            solo = self._solo_pixmap()
            if solo is not None:
                self._draw_pixmap_cover(painter, solo[0], rect, solo[1])
                if self._example_badge:
                    self._draw_example_badge(painter, rect, self._example_badge)
                painter.end()
                return

            # Compute divider X in widget coords.
            split_x = int(rect.width() * self._pos)

            # --- before layer (left of divider) ---------------------------
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

            # --- after layer (right of divider) ----------------------------
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

            # --- divider line + handle -------------------------------------
            pen_shadow = QPen(_DIVIDER_SHADOW)
            pen_shadow.setWidth(_DIVIDER_LINE_PX + 2)
            painter.setPen(pen_shadow)
            painter.drawLine(split_x, 0, split_x, rect.height())
            pen = QPen(_DIVIDER_COLOR)
            pen.setWidth(_DIVIDER_LINE_PX)
            painter.setPen(pen)
            painter.drawLine(split_x, 0, split_x, rect.height())

            # Handle circle in the middle.
            handle_y = rect.height() // 2
            painter.setPen(QPen(_DIVIDER_SHADOW, 1))
            painter.setBrush(QBrush(_DIVIDER_COLOR))
            painter.drawEllipse(
                QPointF(split_x, handle_y),
                _HANDLE_RADIUS_PX,
                _HANDLE_RADIUS_PX,
            )
            # Twin arrows inside handle (drawn as a simple ASCII-glyph approx).
            painter.setPen(QPen(QColor("#202020"), 2))
            ay = handle_y
            painter.drawLine(split_x - 5, ay, split_x - 1, ay - 4)
            painter.drawLine(split_x - 5, ay, split_x - 1, ay + 4)
            painter.drawLine(split_x + 5, ay, split_x + 1, ay - 4)
            painter.drawLine(split_x + 5, ay, split_x + 1, ay + 4)

            # --- badges ------------------------------------------------------
            if self._show_badges:
                self._draw_badge(painter, "BEFORE", y=8, bg=_BADGE_BG_BEFORE, x=8)
                self._draw_badge(
                    painter,
                    "AFTER",
                    y=8,
                    bg=_BADGE_BG_AFTER,
                    right=rect.width() - 8,
                )

            if self._example_badge:
                self._draw_example_badge(painter, rect, self._example_badge)

            painter.end()
        except Exception:  # noqa: BLE001 - paint must never raise
            return

    def _draw_pixmap_cover(self, painter: QPainter, pm: QPixmap, rect, slot: str) -> None:
        """Center-crop the pixmap to fully cover the widget rect (object-fit:cover).

        Draws a cached copy pre-scaled to the target size (see ``_cover_cache``)
        instead of smooth-scaling the full-resolution source on every call."""
        if pm.isNull() or rect.width() <= 0 or rect.height() <= 0:
            return
        pw, ph = pm.width(), pm.height()
        if pw <= 0 or ph <= 0:
            return
        widget_ar = rect.width() / rect.height()
        pix_ar = pw / ph
        if pix_ar > widget_ar:
            # Pixmap is wider than widget - fit to height, crop sides.
            scale_h = rect.height() / ph
            scaled_w = pw * scale_h
            offset_x = (scaled_w - rect.width()) / 2
            target = QRectF(-offset_x, 0, scaled_w, rect.height())
        else:
            scale_w = rect.width() / pw
            scaled_h = ph * scale_w
            offset_y = (scaled_h - rect.height()) / 2
            target = QRectF(0, -offset_y, rect.width(), scaled_h)

        # Scale to the pixels the screen really has, then tell Qt what that
        # ratio is so it draws the copy at the size asked for. Scaling to the
        # rectangle alone hands Qt a picture with a third fewer pixels than the
        # band it fills on a 150% display, and Qt stretches it: every preview
        # in the library comes out soft. The ratio is part of the cache key
        # because dragging the window to a screen with a different one has to
        # rebuild the copy.
        ratio = widget_pixel_ratio(self)
        target_w = max(1, round(target.width() * ratio))
        target_h = max(1, round(target.height() * ratio))
        # cacheKey(), not id(): CPython reuses an address once the old QPixmap is
        # collected, so a new pixmap allocated there would hit the previous
        # entry and paint the wrong thumbnail on a recycled card.
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
        """Small centered pill at the top marking the preview as a demo."""
        f = painter.font()
        f.setPointSize(scale_point_size(8))
        f.setBold(True)
        painter.setFont(f)
        metrics = painter.fontMetrics()
        bw = float(metrics.horizontalAdvance(text)) + 20.0
        # Sized from the font metrics, for the reason _draw_badge spells out:
        # a point size is DPI-relative, so a fixed pill height clips its own
        # text at a higher DPI.
        bh = max(20.0, float(metrics.height()) + 4.0)
        bx = (rect.width() - bw) / 2.0
        badge = QRectF(bx, 8, bw, bh)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
        painter.drawRoundedRect(badge, bh / 2.0, bh / 2.0)
        painter.setPen(QPen(_BADGE_TEXT))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_badge(self, painter: QPainter, text: str, y: int, bg: QColor,
                    x: float | None = None, right: float | None = None) -> None:
        """One corner pill. Give it a left edge or a right edge, not both.

        The pill is MEASURED, like _draw_example_badge above. A point size is
        DPI-relative, so the 8pt bold that fitted a fixed 52px at the 72 DPI
        this was designed on overflows it at Windows's 96 and clips "BEFORE".
        """
        f = painter.font()
        f.setPointSize(scale_point_size(8))
        f.setBold(True)
        painter.setFont(f)
        metrics = painter.fontMetrics()
        bw = max(52.0, float(metrics.horizontalAdvance(text)) + 16.0)
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
        painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(QPen(_BADGE_TEXT))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_placeholder(self, painter: QPainter, rect) -> None:
        painter.fillRect(rect, _PLACEHOLDER_BG)
        painter.setPen(QPen(_PLACEHOLDER_TEXT))
        f = painter.font()
        f.setPointSize(scale_point_size(9))
        painter.setFont(f)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._placeholder_text)
