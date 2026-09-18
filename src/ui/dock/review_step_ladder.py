




from __future__ import annotations

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import QWidget

from .auto_flow_look import repolish_widget
from .font_scale import scale_px_length





REVIEW_LADDER_ROW_PX = 32


class ReviewStepChip(QWidget):







    clicked = pyqtSignal(int)

    def __init__(self, step: int, parent=None):
        super().__init__(parent)
        self._step = int(step)
        self._navigable = False
        self.setObjectName("autoTaskRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setProperty("navigable", False)

    def set_navigable(self, on: bool) -> None:
        self._navigable = bool(on)
        self.setCursor(Qt.CursorShape.PointingHandCursor if self._navigable
                       else Qt.CursorShape.ArrowCursor)
        if bool(self.property("navigable")) != self._navigable:
            self.setProperty("navigable", self._navigable)
            repolish_widget(self)

    def mouseReleaseEvent(self, event):  # noqa: N802



        getter = (getattr(event, "position", None) or getattr(event, "localPos", None))
        hit = getter is not None and self._navigable and event.button() == Qt.MouseButton.LeftButton
        if hit and self.rect().contains(getter().toPoint()):
            self.clicked.emit(self._step)
        super().mouseReleaseEvent(event)









_LADDER_LABEL_MIN_PX = 360


class ReviewLadderStrip(QWidget):







    def __init__(self, parent=None):
        super().__init__(parent)
        self._labels: list = []
        self._current = 0
        self._compact = False



        self.on_change = None

    def register(self, labels: list) -> None:
        self._labels = labels
        self._apply()

    def set_current(self, index: int) -> None:
        self._current = int(index)
        self._apply()

    def is_compact(self) -> bool:



        return self._compact

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._apply()

    def _full_width(self) -> int:



        lay = self.layout()
        if lay is None:
            return scale_px_length(_LADDER_LABEL_MIN_PX)
        extra = 0
        spacing = 6
        for lab in self._labels:
            if lab.isHidden():
                extra += lab.sizeHint().width() + spacing

        rules = scale_px_length(10) * max(0, len(self._labels) - 1)
        return lay.sizeHint().width() + extra + rules

    def _apply(self) -> None:
        try:
            need = self._full_width()
        except RuntimeError:
            need = scale_px_length(_LADDER_LABEL_MIN_PX)
        self._compact = self.width() < need
        for i, lab in enumerate(self._labels):
            try:
                lab.setVisible((not self._compact) or i == self._current)
            except RuntimeError:
                pass
        if self.on_change is not None:
            try:
                self.on_change()
            except (RuntimeError, AttributeError):
                pass


__all__ = [
    "REVIEW_LADDER_ROW_PX",
    "ReviewLadderStrip",
    "ReviewStepChip",
]
