"""Manages point prompts (positive/negative) for SAM segmentation."""

from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy


class PromptManager:
    def __init__(self):
        self.positive_points: List[Tuple[float, float]] = []
        self.negative_points: List[Tuple[float, float]] = []
        self._history: List[Tuple[str, Tuple[float, float]]] = []

    def add_positive_point(self, x: float, y: float):
        self.positive_points.append((x, y))
        self._history.append(("positive", (x, y)))

    def add_negative_point(self, x: float, y: float):
        self.negative_points.append((x, y))
        self._history.append(("negative", (x, y)))

    def undo(self) -> Optional[Tuple[str, Tuple[float, float]]]:
        if not self._history:
            return None

        label, point = self._history.pop()
        if label == "positive":
            # Remove last occurrence (not first) to correctly undo
            # duplicate coords
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
    def point_count(self) -> Tuple[int, int]:
        return len(self.positive_points), len(self.negative_points)

    def get_points_for_predictor(
        self, transform
    ) -> Tuple[Optional["numpy.ndarray"], Optional["numpy.ndarray"]]:
        import numpy as np
        from rasterio import transform as rio_transform

        all_points = self.positive_points + self.negative_points
        if not all_points:
            return None, None

        point_coords = []
        point_labels = []

        for x, y in self.positive_points:
            row, col = rio_transform.rowcol(transform, x, y)
            point_coords.append([col, row])
            point_labels.append(1)

        for x, y in self.negative_points:
            row, col = rio_transform.rowcol(transform, x, y)
            point_coords.append([col, row])
            point_labels.append(0)

        return np.array(point_coords), np.array(point_labels)
