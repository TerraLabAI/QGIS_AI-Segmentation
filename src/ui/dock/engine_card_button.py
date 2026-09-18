






from __future__ import annotations

from qgis.PyQt.QtCore import QEvent, QSize
from qgis.PyQt.QtWidgets import QLabel, QPushButton, QSizePolicy


class EngineCardButton(QPushButton):


    def __init__(self, min_height: int, parent=None):
        super().__init__(parent)
        self._card_min_height = int(min_height)
        self.setMinimumHeight(self._card_min_height)

        policy = self.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Policy.Minimum)
        self.setSizePolicy(policy)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return self.layout() is not None

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        lay = self.layout()
        if lay is None:
            return super().heightForWidth(width)
        return max(self._card_min_height, lay.totalHeightForWidth(width))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        lay = self.layout()
        if lay is None:
            return super().minimumSizeHint()
        return QSize(lay.totalMinimumSize().width(), self._card_min_height)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._sync_min_height()

    def event(self, event):  # noqa: D401
        handled = super().event(event)




        if event.type() == QEvent.Type.LayoutRequest:
            self._sync_min_height()
        return handled

    def _sync_min_height(self) -> None:





        for label in self.findChildren(QLabel):
            if not label.wordWrap() or label.width() <= 0:
                continue
            needed = label.heightForWidth(label.width())
            if needed > 0 and needed != label.minimumHeight():
                label.setMinimumHeight(needed)
        wanted = self.heightForWidth(self.width())
        if wanted != self.minimumHeight():


            self.setMinimumHeight(wanted)
