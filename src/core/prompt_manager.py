"""Point prompt management for SAM segmentation.

Tracks positive/negative annotation points with undo history.
Pure data structure with no QGIS UI dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from qgis.core import QgsGeometry


@dataclass
class FrozenCropSession:
    """A frozen crop session whose mask has been converted to a geographic polygon."""
    polygon: QgsGeometry
    points_positive: list[tuple[float, float]] = field(default_factory=list)
    points_negative: list[tuple[float, float]] = field(default_factory=list)
    crop_info: dict | None = None


class PromptManager:
    def __init__(self):
        self.positive_points: list[tuple[float, float]] = []
        self.negative_points: list[tuple[float, float]] = []
        self._history: list[tuple[str, tuple[float, float]]] = []

    def add_positive_point(self, x: float, y: float):
        self.positive_points.append((x, y))
        self._history.append(("positive", (x, y)))

    def add_negative_point(self, x: float, y: float):
        self.negative_points.append((x, y))
        self._history.append(("negative", (x, y)))

    def undo(self) -> tuple[str, tuple[float, float]] | None:
        if not self._history:
            return None

        label, point = self._history.pop()
        if label == "positive":
            # Remove last occurrence (not first) to correctly undo duplicate coords
            for i in range(len(self.positive_points) - 1, -1, -1):
                if self.positive_points[i] == point:
                    self.positive_points.pop(i)
                    break
        elif label == "negative":
            for i in range(len(self.negative_points) - 1, -1, -1):
                if self.negative_points[i] == point:
                    self.negative_points.pop(i)
                    break

        return label, point

    def clear(self):
        self.positive_points = []
        self.negative_points = []
        self._history = []

    @property
    def point_count(self) -> tuple[int, int]:
        return len(self.positive_points), len(self.negative_points)
