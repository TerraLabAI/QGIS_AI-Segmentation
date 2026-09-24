











from __future__ import annotations

import math
import time

from qgis.PyQt.QtCore import QRectF, QSize, Qt, QTimer
from qgis.PyQt.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from ...core.server_dials import dial_in_range
from .font_scale import scale_point_size
from .styles import FONT_BODY, HUE_RUN, INK, INK_2, category_ink

GRID_PX = 15
_DOT_PX = 4.0
_DOT_GAP = 1.5
_DOT_RADIUS = 1.0
_CYCLE_S = 0.65
_REST = 0.15
_DELAYS = ((0.09, 0.18, 0.27), (0.0, 0.09, 0.18), (0.09, 0.18, 0.27))
_SHIMMER_S = 1.4
_TICK_MS = 40
_CLOCK_MS = 100
RUN_LINE_PX = 20
RUN_LINE_GAP_PX = 8


def _ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def dot_opacity_at(phase: float) -> float:

    phase %= 1.0
    if phase < 0.18:
        return _REST + (1.0 - _REST) * _ease_in_out(phase / 0.18)
    if phase < 0.42:
        return 1.0
    if phase < 0.62:
        return 1.0 - (1.0 - _REST) * _ease_in_out((phase - 0.42) / 0.20)
    return _REST


def format_run_elapsed(seconds) -> str:

    try:
        seconds = float(seconds)
    except (TypeError, ValueError, OverflowError):
        seconds = 0.0
    if math.isnan(seconds) or math.isinf(seconds):
        seconds = 0.0
    seconds = max(0.0, seconds)
    switch_s = dial_in_range("tuning.ui.run_elapsed_switch_s", 60, 30, 300)
    if seconds < switch_s:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    return f"{minutes}m {seconds - minutes * 60:.1f}s"


def _token_color(token: str) -> QColor:


    text = str(token or "").strip()
    if text.startswith("rgb"):
        parts = [p.strip() for p in text[text.index("(") + 1:text.rindex(")")].split(",")]
        try:
            r, g, b = (int(float(p)) for p in parts[:3])
            alpha = float(parts[3]) if len(parts) > 3 else 1.0
        except (ValueError, IndexError):
            return QColor()
        colour = QColor(r, g, b)
        colour.setAlphaF(max(0.0, min(1.0, alpha)))
        return colour
    return QColor(text)


class _RunTicker(QWidget):


    def __init__(self, interval_ms: int, parent=None):
        super().__init__(parent)
        self._wanted = False
        self._clock = time.monotonic()
        self._timer = QTimer(self)
        self._timer.setInterval(max(16, int(interval_ms)))
        self._timer.timeout.connect(self.update)

    def start(self) -> None:
        self._wanted = True
        self._sync_timer()

    def stop(self) -> None:
        self._wanted = False
        self._timer.stop()
        self.update()

    def _sync_timer(self) -> None:
        if self._wanted and self.isVisible():
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self._sync_timer()

    def hideEvent(self, event):  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    def _since_start(self) -> float:
        return time.monotonic() - self._clock


class RunDotsGrid(_RunTicker):


    def __init__(self, parent=None):
        super().__init__(_TICK_MS, parent)

        self._ink = _token_color(category_ink(HUE_RUN))
        self.setFixedSize(GRID_PX, GRID_PX)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def _dot_alpha(self, row: int, col: int) -> float:
        if not self._timer.isActive():
            return 1.0 if (row, col) == (1, 1) else _REST
        return dot_opacity_at((self._since_start() - _DELAYS[row][col]) / _CYCLE_S)

    def paintEvent(self, event):  # noqa: N802
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.PenStyle.NoPen)
            step = _DOT_PX + _DOT_GAP
            for row in range(3):
                for col in range(3):
                    colour = QColor(self._ink)
                    colour.setAlphaF(self._dot_alpha(row, col))
                    painter.setBrush(colour)
                    painter.drawRoundedRect(
                        QRectF(col * step, row * step, _DOT_PX, _DOT_PX),
                        _DOT_RADIUS, _DOT_RADIUS)
            painter.end()
        except Exception:  # noqa: BLE001
            return


class RunShimmerText(_RunTicker):


    def __init__(self, text: str = "", parent=None):
        super().__init__(_TICK_MS, parent)
        self._text = text or ""
        self._base = _token_color(INK_2)
        self._light = _token_color(INK)
        font = QFont(self.font())
        font.setPixelSize(scale_point_size(FONT_BODY))
        font.setWeight(QFont.Weight.Medium)
        self.setFont(font)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(RUN_LINE_PX)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def setText(self, text: str) -> None:  # noqa: N802
        self._text = text or ""
        self.setToolTip(self._text if len(self._text) > 40 else "")
        self.updateGeometry()
        self.update()

    def text(self) -> str:
        return self._text

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.fontMetrics().horizontalAdvance(self._text) + 2, RUN_LINE_PX)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(24, RUN_LINE_PX)

    def _pen_brush(self, width: int):
        if not self._timer.isActive() or width <= 0:
            return self._base
        centre = (1.5 - 2.0 * ((self._since_start() / _SHIMMER_S) % 1.0)) * width
        band = max(24.0, width / 3.0)
        gradient = QLinearGradient(centre - band, 0, centre + band, 0)
        gradient.setColorAt(0.0, self._base)
        gradient.setColorAt(0.5, self._light)
        gradient.setColorAt(1.0, self._base)
        return gradient

    def paintEvent(self, event):  # noqa: N802
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            rect = self.contentsRect()
            metrics = self.fontMetrics()
            text = metrics.elidedText(self._text, Qt.TextElideMode.ElideRight, rect.width())
            pen = QPen()
            pen.setBrush(self._pen_brush(metrics.horizontalAdvance(text)))
            painter.setPen(pen)
            painter.setFont(self.font())
            painter.drawText(
                rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text)
            painter.end()
        except Exception:  # noqa: BLE001
            return


class RunElapsedClock(QLabel):


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("autoStatusTime")



        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)





        self.setMinimumWidth(self.fontMetrics().horizontalAdvance("59.9s") + 2)
        self._started = time.monotonic()
        self._frozen = None
        self._timer = QTimer(self)
        self._timer.setInterval(_CLOCK_MS)
        self._timer.timeout.connect(self._tick_clock)
        self._tick_clock()

    def restart_clock(self, started: float | None = None) -> None:
        self._started = float(started) if started is not None else time.monotonic()
        self._frozen = None
        self._tick_clock()
        if self.isVisible():
            self._timer.start()

    def freeze_clock(self) -> None:
        self._timer.stop()
        self._frozen = max(0.0, time.monotonic() - self._started)
        self.setText(format_run_elapsed(self._frozen))

    def _tick_clock(self) -> None:
        if self._frozen is None:
            self.setText(format_run_elapsed(time.monotonic() - self._started))

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if self._frozen is None:
            self._tick_clock()
            self._timer.start()

    def hideEvent(self, event):  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)


class RunStatusLine(QWidget):




    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("autoStatusLine")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(RUN_LINE_GAP_PX)
        self.dots = RunDotsGrid(self)
        row.addWidget(self.dots, 0, Qt.AlignmentFlag.AlignVCenter)
        self.verb = RunShimmerText("", self)
        row.addWidget(self.verb, 1, Qt.AlignmentFlag.AlignVCenter)
        self.clock = RunElapsedClock(self)
        row.addWidget(self.clock, 0, Qt.AlignmentFlag.AlignVCenter)
        self.figure = QLabel("", self)
        self.figure.setObjectName("autoStatusFigure")
        self.figure.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.figure, 0, Qt.AlignmentFlag.AlignVCenter)
        self.setFixedHeight(RUN_LINE_PX)

    def set_verb(self, text: str) -> None:

        self.verb.setText((text or "").rstrip(". …"))

    def set_figure(self, text: str) -> None:
        self.figure.setText(text or "")

    def restart(self) -> None:
        self.clock.restart_clock()
        self.dots.start()
        self.verb.start()

    def stop(self) -> None:
        self.dots.stop()
        self.verb.stop()
        self.clock.freeze_clock()


__all__ = [
    "RUN_LINE_PX",
    "RunDotsGrid",
    "RunElapsedClock",
    "RunShimmerText",
    "RunStatusLine",
    "dot_opacity_at",
    "format_run_elapsed",
]
