













from __future__ import annotations

from qgis.PyQt.QtCore import QSettings, QSize, Qt, pyqtSignal
from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy

from .font_scale import scale_px_length
from .styles import _FOLD_FACT_QSS, _FOLD_ROW_QSS, _FOLD_TITLE_QSS, INK_3


FOLD_ROW_PX = 32
_CHEVRON_PX = 14


def read_fold_open(key: str, default: bool) -> bool:

    if not key:
        return bool(default)
    try:
        return bool(QSettings().value(key, bool(default), type=bool))
    except Exception:  # noqa: BLE001
        return bool(default)


def write_fold_open(key: str, open_: bool) -> None:

    if not key:
        return
    try:
        QSettings().setValue(key, bool(open_))
    except Exception:  # noqa: BLE001  # nosec B110
        pass


class _FoldLabel(QLabel):







    def __init__(self, keep_chars: int, parent=None):
        super().__init__("", parent)
        self._fold_full = ""
        self._fold_keep = max(0, int(keep_chars))
        self.setMinimumWidth(0)

    def set_full_text(self, text: str) -> None:
        self._fold_full = text or ""
        super().setText(self._fold_full)
        self._elide_fold_label()
        self.updateGeometry()

    def full_text(self) -> str:
        return self._fold_full

    def sizeHint(self):  # noqa: N802
        hint = super().sizeHint()
        try:
            width = self.fontMetrics().horizontalAdvance(self._fold_full) + 2
            return QSize(width, hint.height())
        except (RuntimeError, AttributeError):
            return hint

    def minimumSizeHint(self):  # noqa: N802
        hint = super().minimumSizeHint()
        try:
            if not self._fold_keep:
                return QSize(0, hint.height())
            head = self._fold_full[:self._fold_keep]
            if len(self._fold_full) > self._fold_keep:
                head += "\u2026"
            return QSize(self.fontMetrics().horizontalAdvance(head) + 2, hint.height())
        except (RuntimeError, AttributeError):
            return hint

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._elide_fold_label()

    def _elide_fold_label(self) -> None:
        try:
            if self.width() <= 1:
                return
            shown = self.fontMetrics().elidedText(
                self._fold_full, Qt.TextElideMode.ElideRight, self.width())
            if shown != self.text():
                super().setText(shown)
        except (RuntimeError, AttributeError):
            pass  # nosec B110


class FoldRow(QPushButton):









    fold_toggled = pyqtSignal(bool)

    def __init__(self, title: str, settings_key: str = "",
                 default_open: bool = False, managed: bool = True,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("foldRow")
        self.setStyleSheet(_FOLD_ROW_QSS)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setMinimumHeight(scale_px_length(FOLD_ROW_PX))
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 4, 0)
        row.setSpacing(6)




        self.fold_title = _FoldLabel(keep_chars=4)
        self.fold_title.setStyleSheet(_FOLD_TITLE_QSS)
        self.fold_title.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.fold_title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        row.addWidget(self.fold_title, 0)
        self.fold_fact = _FoldLabel(keep_chars=0)
        self.fold_fact.setStyleSheet(_FOLD_FACT_QSS)
        self.fold_fact.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.fold_fact.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.fold_fact.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.fold_fact.setVisible(False)
        row.addWidget(self.fold_fact, 1)
        self.fold_chevron = QLabel("")
        self.fold_chevron.setFixedSize(QSize(_CHEVRON_PX, _CHEVRON_PX))
        self.fold_chevron.setStyleSheet("background: transparent; border: none;")
        self.fold_chevron.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        row.addWidget(self.fold_chevron, 0, Qt.AlignmentFlag.AlignVCenter)

        self._fold_key = settings_key
        self._fold_managed = bool(managed)
        self._fold_body = None
        self._fold_open = read_fold_open(settings_key, default_open)
        self.set_fold_title(title)
        if self._fold_managed:
            self.clicked.connect(self._on_fold_row_clicked)
        self._paint_fold_chevron()



    def is_fold_open(self) -> bool:
        return bool(self._fold_open)

    def bind_fold_body(self, body) -> None:

        self._fold_body = body
        self._show_fold_body()

    def set_fold_open(self, open_: bool) -> None:



        self._fold_open = bool(open_)
        self._paint_fold_chevron()
        self._show_fold_body()

    def set_fold_title(self, text: str) -> None:
        self.fold_title.set_full_text(text or "")
        self.setAccessibleName(text or "")

    def set_fold_fact(self, text: str) -> None:

        self.fold_fact.set_full_text(text or "")
        self.fold_fact.setVisible(bool(text))



    def _on_fold_row_clicked(self) -> None:
        self._fold_open = not self._fold_open
        write_fold_open(self._fold_key, self._fold_open)
        self._paint_fold_chevron()
        self._show_fold_body()
        self.fold_toggled.emit(self._fold_open)

    def _show_fold_body(self) -> None:
        body = self._fold_body
        if body is None:
            return
        try:
            body.setVisible(self._fold_open)
        except RuntimeError:
            self._fold_body = None

    def _paint_fold_chevron(self) -> None:
        try:
            from ..icons import pixmap_for
            from .auto_flow_look import token_qcolor

            name = "chevron_down" if self._fold_open else "chevron_right"
            self.fold_chevron.setPixmap(pixmap_for(
                self.fold_chevron, name, _CHEVRON_PX, token_qcolor(INK_3)))
        except Exception:  # noqa: BLE001  # nosec B110
            pass


__all__ = ["FOLD_ROW_PX", "FoldRow", "read_fold_open", "write_fold_open"]
