"""Compact circular indicator that visualizes credits remaining.

Draws two concentric arcs: a neutral grey track plus a coloured progress
arc whose hue reflects subscription state and remaining ratio. Ported
from AI Edit so the two TerraLab plugins share the same footer language.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QRectF, QSize, Qt
from qgis.PyQt.QtGui import QColor, QPainter, QPen
from qgis.PyQt.QtWidgets import QWidget

# Brand palette - mirrors dockwidget constants. Duplicated here so the
# widget remains importable without circular dependency.
_BRAND_BLUE = "#1e88e5"
_BRAND_GREEN = "#8bac27"
_BRAND_RED = "#ef5350"
_LOW_ORANGE = "#ff8f00"  # free-tier low-credit nudge (amber, not alarm red)
_TRACK_RGBA = (128, 128, 128, 64)  # rgba(128,128,128,0.25)
_LOW_THRESHOLD = 0.20


def _cap_style(name: str):
    """Resolve Qt5 flat or Qt6 scoped enum for PenCapStyle."""
    scoped = getattr(Qt, "PenCapStyle", None)
    if scoped is not None:
        value = getattr(scoped, name, None)
        if value is not None:
            return value
    return getattr(Qt, name)


_FLAT_CAP = _cap_style("FlatCap")
_ROUND_CAP = _cap_style("RoundCap")


class CreditRing(QWidget):
    """Small circular gauge: track + progress arc.

    The widget defaults to 18x18 px which sits cleanly next to 11px footer
    text. Pass ``diameter`` to override.
    """

    def __init__(self, diameter: int = 18, parent: QWidget | None = None):
        super().__init__(parent)
        self._diameter = diameter
        self._stroke = 2.5
        self._used: int | None = None
        self._total: int | None = None
        self._free_tier = False
        self.setFixedSize(diameter, diameter)

    # -- public API -------------------------------------------------------

    def set_credits(
        self,
        used: int | None,
        total: int | None,
        free_tier: bool = False,
    ) -> None:
        self._used = used
        self._total = total
        self._free_tier = free_tier
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self._diameter, self._diameter)

    # -- internals --------------------------------------------------------

    def _ratio_remaining(self) -> float:
        """Fraction of credits still available, in [0, 1]."""
        if self._total is None or self._total <= 0 or self._used is None:
            return 0.0
        remaining = max(self._total - self._used, 0)
        return min(remaining / self._total, 1.0)

    def _progress_color(self) -> QColor:
        ratio = self._ratio_remaining()
        if ratio <= _LOW_THRESHOLD:
            # Free tier running low is a soft nudge (amber), paired with the
            # Start-page "Running low" note; subscribers keep the red at low.
            return QColor(_LOW_ORANGE) if self._free_tier else QColor(_BRAND_RED)
        return QColor(_BRAND_GREEN if self._free_tier else _BRAND_BLUE)

    def paintEvent(self, _event):  # noqa: N802
        # A paintEvent must never raise: on macOS an escaped exception here
        # hangs and then segfaults QGIS at launch (seen in production), so
        # any failure below just skips the paint for this frame.
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            # Inset by half the stroke so the arc stays fully inside the widget
            # rect - Qt strokes are centred on the geometric path.
            inset = self._stroke / 2.0
            rect = QRectF(
                inset, inset,
                self._diameter - 2 * inset,
                self._diameter - 2 * inset,
            )

            # Track: full ring, neutral grey, theme-agnostic via alpha.
            track_pen = QPen(QColor(*_TRACK_RGBA))
            track_pen.setWidthF(self._stroke)
            track_pen.setCapStyle(_FLAT_CAP)
            painter.setPen(track_pen)
            painter.drawArc(rect, 0, 360 * 16)

            # Progress arc: starts at 12 o'clock, sweeps clockwise.
            # Qt angle units: 1/16 degree, 0 deg at 3 o'clock, positive = CCW -
            # so a clockwise sweep needs a negative span.
            ratio = self._ratio_remaining()
            if ratio > 0:
                progress_pen = QPen(self._progress_color())
                progress_pen.setWidthF(self._stroke)
                progress_pen.setCapStyle(_ROUND_CAP)
                painter.setPen(progress_pen)
                start_angle = 90 * 16
                span_angle = int(-ratio * 360 * 16)
                painter.drawArc(rect, start_angle, span_angle)
        except Exception:  # noqa: BLE001 - paint must never raise
            return
