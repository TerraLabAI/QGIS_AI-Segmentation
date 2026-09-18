








from __future__ import annotations

from qgis.PyQt.QtCore import QEvent, QSize, Qt
from qgis.PyQt.QtWidgets import QGridLayout, QHBoxLayout, QLayout, QWidget


class WrappingButtonRow(QWidget):


    def __init__(self, spacing: int = 8, parent: QWidget | None = None):
        super().__init__(parent)
        self._row_buttons: list[QWidget] = []
        self._row_stretch: dict[int, int] = {}
        self._row_stacked = False
        self._row_placed: list[int] = []
        self._row_grid = QGridLayout(self)
        self._row_grid.setContentsMargins(0, 0, 0, 0)
        self._row_grid.setSpacing(spacing)


        self._row_grid.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

    def add_row_item(self, button: QWidget, stretch: int = 1) -> None:




        self._row_stretch[id(button)] = int(stretch)
        self._row_buttons.append(button)
        button.installEventFilter(self)
        self._place_row_buttons()

    @staticmethod
    def _row_item_shown(button: QWidget) -> bool:


        return not (button.isHidden() and button.testAttribute(
            Qt.WidgetAttribute.WA_WState_ExplicitShowHide))

    def _shown_row_buttons(self) -> list[QWidget]:
        return [b for b in self._row_buttons if self._row_item_shown(b)]

    def _one_line_width(self) -> int:
        shown = self._shown_row_buttons()
        if not shown:
            return 0
        spacing = max(0, self._row_grid.horizontalSpacing())
        return (sum(b.sizeHint().width() for b in shown)
                + spacing * (len(shown) - 1))

    def _place_row_buttons(self) -> None:






        grid = self._row_grid
        while grid.count():
            grid.takeAt(0)
        for col in range(max(1, grid.columnCount())):
            grid.setColumnStretch(col, 0)
            if grid.columnMinimumWidth(col):
                grid.setColumnMinimumWidth(col, 0)
        shown = self._shown_row_buttons()
        self._row_placed = [id(b) for b in shown]
        for index, button in enumerate(shown):
            if self._row_stacked:
                grid.addWidget(button, index, 0)
            else:
                grid.addWidget(button, 0, index)
                grid.setColumnStretch(index, self._row_stretch.get(id(button), 1))



        hidden = [b for b in self._row_buttons if id(b) not in self._row_placed]
        for offset, button in enumerate(hidden, start=len(shown)):
            if self._row_stacked:
                grid.addWidget(button, offset, 0)
            else:
                grid.addWidget(button, 0, offset)
        if self._row_stacked:
            grid.setColumnStretch(0, 1)
        self.updateGeometry()

    def _sync_row_shape(self) -> None:
        stacked = self.width() < self._one_line_width()
        placed = [id(b) for b in self._shown_row_buttons()]
        if stacked != self._row_stacked or placed != self._row_placed:
            self._row_stacked = stacked
            self._place_row_buttons()
        self._pin_row_column_widths()

    def _pin_row_column_widths(self) -> None:





        grid = self._row_grid
        for index, button in enumerate(self._shown_row_buttons()):
            wanted = 0 if self._row_stacked else button.sizeHint().width()


            if grid.columnMinimumWidth(index) != wanted:
                grid.setColumnMinimumWidth(index, wanted)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        try:
            self._sync_row_shape()
        except RuntimeError:
            pass  # nosec B110

    def eventFilter(self, watched, event):  # noqa: N802




        try:
            if event.type() in (QEvent.Type.Show, QEvent.Type.Hide,
                                QEvent.Type.FontChange):
                self._sync_row_shape()
                self.updateGeometry()
        except RuntimeError:
            pass  # nosec B110
        return False

    def event(self, event):  # noqa: N802






        try:
            if event.type() == QEvent.Type.LayoutRequest:
                self._sync_row_shape()
            return super().event(event)
        except RuntimeError:
            return False

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        try:
            shown = self._shown_row_buttons()
            if not shown:
                return QSize(0, 0)
            width = max(b.minimumSizeHint().width() for b in shown)
            return QSize(width, self._row_grid.minimumSize().height())
        except RuntimeError:
            return QSize(0, 0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802



        try:
            return self._row_grid.hasHeightForWidth()
        except RuntimeError:
            return False

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        try:
            return self._row_grid.totalHeightForWidth(width)
        except RuntimeError:
            return -1


def labelled_field_pair(label: QWidget, field: QWidget) -> QWidget:

    pair = QWidget()
    row = QHBoxLayout(pair)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(label)
    row.addStretch()
    row.addWidget(field)
    return pair
