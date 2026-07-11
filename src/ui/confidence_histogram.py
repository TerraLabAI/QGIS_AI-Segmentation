"""A granular confidence-score distribution strip for the Auto review.

Purely visual: it shows how many detections sit at each confidence level across
the slider's usable range, and colours each small score packet by its relation
to the current cutoff, so the user sees at a glance how much raising or lowering
Confidence would reveal or hide. No geometry, no filtering, no text.
"""
from __future__ import annotations

import math

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QPainter
from qgis.PyQt.QtWidgets import QSizePolicy, QWidget


class ConfidenceHistogram(QWidget):
    """Fine score-distribution strip; packets below the cutoff render dimmed
    grey, packets at/above render brand blue. Self-explanatory by colour, so it
    carries no legend text."""

    # One score packet per bucket. Fine enough that small groups of detections
    # stay distinguishable while each bar keeps a few readable pixels.
    _BUCKET_W = 0.025

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lo = 0.05
        self._hi = 0.95
        self._scores: list[float] = []
        self._buckets = [0]
        self._cutoff = 0.30
        self.setFixedHeight(18)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_range(self, lo: float, hi: float) -> None:
        """Set the usable score range the bars span (the slider's floor..max)."""
        hi = max(hi, lo + self._BUCKET_W)
        self._lo = max(0.0, min(lo, hi - 1e-6))
        self._hi = hi
        self._rebucket()
        self.update()

    def set_scores(self, scores: list[float]) -> None:
        self._scores = [float(s) for s in scores]
        self._rebucket()
        self.update()

    def set_cutoff(self, cutoff: float) -> None:
        # Cheap recolour only: no rebucket, so a slider drag just repaints.
        self._cutoff = cutoff
        self.update()

    def _n_buckets(self) -> int:
        span = max(1e-6, self._hi - self._lo)
        return max(1, int(round(span / self._BUCKET_W)))

    def _rebucket(self) -> None:
        n = self._n_buckets()
        span = max(1e-6, self._hi - self._lo)
        buckets = [0] * n
        for s in self._scores:
            idx = int((s - self._lo) / span * n)
            buckets[min(n - 1, max(0, idx))] += 1
        self._buckets = buckets

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt signature
        # A paintEvent must never raise: on macOS an escaped exception here
        # hangs and then segfaults QGIS at launch (seen in production), so
        # any failure below just skips the paint for this frame.
        try:
            p = QPainter(self)
            p.setPen(Qt.PenStyle.NoPen)
            n = len(self._buckets)
            span = max(1e-6, self._hi - self._lo)
            # sqrt scaling so a small packet next to a tall one stays visible
            # (a raw linear scale flattens the small groups to nothing).
            top = math.sqrt(max(self._buckets) or 1)
            w = self.width() / float(n)
            avail = self.height() - 2
            kept = QColor(30, 136, 229, 235)
            dimmed = QColor(128, 128, 128, 90)
            for i, count in enumerate(self._buckets):
                if count <= 0:
                    continue
                h = max(2, int((math.sqrt(count) / top) * avail))
                bucket_top_score = self._lo + (i + 1) * span / n
                is_kept = bucket_top_score > self._cutoff + 1e-9
                p.setBrush(kept if is_kept else dimmed)
                x = int(i * w) + 1
                bar_w = max(1, int(w) - 1)
                p.drawRect(x, self.height() - h, bar_w, h)
            p.end()
        except Exception:  # noqa: BLE001 - paint must never raise
            return
