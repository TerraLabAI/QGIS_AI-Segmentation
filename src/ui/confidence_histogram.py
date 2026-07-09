






from __future__ import annotations

import math

from qgis.PyQt.QtCore import QEvent, Qt
from qgis.PyQt.QtGui import QColor, QPainter
from qgis.PyQt.QtWidgets import QSizePolicy, QWidget



_KEPT_RGB = (30, 136, 229)

_HIDDEN_BAR_ALPHA = 80


class ConfidenceHistogram(QWidget):






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

    def paintEvent(self, _event) -> None:  # noqa: N802





        try:
            p = QPainter(self)
        except Exception:  # noqa: BLE001
            return
        try:
            p.setPen(Qt.PenStyle.NoPen)
            n = len(self._buckets)
            span = max(1e-6, self._hi - self._lo)


            top = math.sqrt(max(self._buckets) or 1)
            width = float(self.width())
            avail = self.height() - 2




            kept = QColor(_KEPT_RGB[0], _KEPT_RGB[1], _KEPT_RGB[2], 235)


            dimmed = QColor(self.palette().color(self.foregroundRole()))
            dimmed.setAlpha(_HIDDEN_BAR_ALPHA)
            if not self.isEnabled():
                kept = QColor(128, 128, 128, 150)
            for i, count in enumerate(self._buckets):
                if count <= 0:
                    continue
                h = max(2, int((math.sqrt(count) / top) * avail))
                bucket_top_score = self._lo + (i + 1) * span / n
                is_kept = bucket_top_score > self._cutoff + 1e-9
                p.setBrush(kept if is_kept else dimmed)



                x0 = int(round(i * width / n))
                x1 = int(round((i + 1) * width / n))
                bar_w = max(1, x1 - x0 - 1)
                p.drawRect(x0, self.height() - h, bar_w, h)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        finally:
            try:
                p.end()
            except Exception:  # noqa: BLE001
                pass  # nosec B110

    def changeEvent(self, event) -> None:  # noqa: N802

        super().changeEvent(event)
        try:
            if event.type() == QEvent.Type.EnabledChange:
                self.update()
        except Exception:  # noqa: BLE001
            pass  # nosec B110
