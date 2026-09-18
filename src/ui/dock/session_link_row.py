










from __future__ import annotations

from qgis.PyQt.QtCore import QEvent, QObject, QSize, Qt
from qgis.PyQt.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .styles import _HINT_LINE_QSS

_DOT = "·"
_GAP = 2


class SessionLinkRow(QWidget):



    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._session_links: list[QWidget] = []
        self._link_lines: list[list[QWidget]] = []
        self._line_dots: list[QLabel] = []
        self._dot_px: int | None = None
        self._links_col = QVBoxLayout(self)
        self._links_col.setContentsMargins(0, 0, 0, 0)
        self._links_col.setSpacing(0)


        self._links_col.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def add_session_link(self, button: QWidget) -> None:

        self._session_links.append(button)
        button.setParent(self)
        button.installEventFilter(self)
        self._flow_session_links(force=True)



    def _dot_width(self) -> int:
        if self._dot_px is None:
            probe = QLabel(_DOT, self)
            probe.setStyleSheet(_HINT_LINE_QSS)
            self._dot_px = probe.sizeHint().width()



            probe.hide()
            probe.deleteLater()
        return self._dot_px

    def _plan_link_lines(self) -> list[list[QWidget]]:

        room = max(1, self.width())
        dot_w = self._dot_width() + 2 * _GAP
        lines: list[list[QWidget]] = []
        used = 0
        for link in self._session_links:
            if link.isHidden():
                continue
            w = link.sizeHint().width()
            if lines and used + dot_w + w <= room:
                lines[-1].append(link)
                used += dot_w + w
            else:
                lines.append([link])
                used = w
        return lines

    def _flow_session_links(self, force: bool = False) -> None:
        lines = self._plan_link_lines()
        if not force and lines == self._link_lines:
            return
        self._link_lines = lines
        col = self._links_col
        while col.count():
            item = col.takeAt(0)
            inner = item.layout()
            if inner is not None:
                while inner.count():
                    inner.takeAt(0)
                inner.deleteLater()
        for dot in self._line_dots:
            dot.hide()
            dot.deleteLater()
        self._line_dots = []
        for line in lines:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(_GAP)
            row.addStretch(1)
            for i, link in enumerate(line):
                if i:
                    dot = QLabel(_DOT, self)
                    dot.setStyleSheet(_HINT_LINE_QSS)
                    self._line_dots.append(dot)
                    row.addWidget(dot, 0, Qt.AlignmentFlag.AlignVCenter)
                    dot.show()
                row.addWidget(link, 0, Qt.AlignmentFlag.AlignVCenter)
            row.addStretch(1)
            col.addLayout(row)
        self.updateGeometry()



    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        try:
            self._flow_session_links()
        except RuntimeError:
            pass  # nosec B110

    def eventFilter(self, watched, event):  # noqa: N802




        try:
            if event.type() in (QEvent.Type.ShowToParent, QEvent.Type.HideToParent,
                                QEvent.Type.FontChange):
                self._flow_session_links(force=True)
        except RuntimeError:
            pass  # nosec B110
        return False

    def minimumSizeHint(self) -> QSize:  # noqa: N802



        try:
            shown = [b for b in self._session_links if not b.isHidden()]
            if not shown:
                return QSize(0, 0)
            width = max(b.minimumSizeHint().width() for b in shown)
            return QSize(width, self._links_col.minimumSize().height())
        except RuntimeError:
            return QSize(0, 0)

    def sizeHint(self) -> QSize:  # noqa: N802
        hint = self._links_col.sizeHint()
        return QSize(min(hint.width(), max(1, self.width())), hint.height())

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return False


class EnabledChangeRelay(QObject):







    def __init__(self, callback, parent: QObject | None = None):
        super().__init__(parent)
        self._enabled_callback = callback

    def eventFilter(self, watched, event):  # noqa: N802
        try:
            if event.type() == QEvent.Type.EnabledChange:
                self._enabled_callback()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return False


__all__ = ["EnabledChangeRelay", "SessionLinkRow"]
