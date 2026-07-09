"""A tiny 10-bucket confidence-score distribution strip for the Auto review.

Purely visual: it shows how many detections sit at each confidence level and
dims the part below the current cutoff, so the user sees at a glance how much
raising or lowering Confidence would reveal or hide. No geometry, no filtering.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QPainter
from qgis.PyQt.QtWidgets import QSizePolicy, QWidget


class ConfidenceHistogram(QWidget):
    """10-bucket score distribution; the part below the cutoff renders dimmed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buckets = [0] * 10
        self._cutoff = 0.30
        self.setFixedHeight(14)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_scores(self, scores: list[float]) -> None:
        self._buckets = [0] * 10
        for s in scores:
            self._buckets[min(9, max(0, int(s * 10)))] += 1
        self.update()

    def set_cutoff(self, cutoff: float) -> None:
        self._cutoff = cutoff
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt signature
        # A paintEvent must never raise: on macOS an escaped exception here
        # hangs and then segfaults QGIS at launch (seen in production), so
        # any failure below just skips the paint for this frame.
        try:
            p = QPainter(self)
            p.setPen(Qt.PenStyle.NoPen)
            top = max(self._buckets) or 1
            w = self.width() / 10.0
            for i, n in enumerate(self._buckets):
                if n <= 0:
                    continue
                h = max(2, int((n / top) * (self.height() - 2)))
                kept = (i + 1) * 0.1 > self._cutoff + 1e-9
                p.setBrush(QColor(30, 136, 229, 220 if kept else 70))
                p.drawRect(int(i * w) + 1, self.height() - h, int(w) - 2, h)
            p.end()
        except Exception:  # noqa: BLE001 - paint must never raise
            return
