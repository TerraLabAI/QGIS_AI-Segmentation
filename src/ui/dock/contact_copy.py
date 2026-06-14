










from __future__ import annotations

import weakref
from typing import Callable

from qgis.PyQt import sip
from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtWidgets import QApplication, QLabel

from ...core.i18n import tr
from ...core.server_dials import dial_copy, dial_in_range
from .font_scale import scale_qss_font_px
from .styles import FONT_HINT, INK_2

_COPIED_MS = 2000


_LINE_QSS = f"font-size: {FONT_HINT}px; color: {INK_2}; background: transparent;"


def copy_cta_text() -> str:

    return dial_copy("pro_ceiling.copy_cta", tr("Copy email"))


def copied_text() -> str:

    return dial_copy("pro_ceiling.copied", tr("Copied!"))


def custom_needs_line(email: str) -> str:



    return dial_copy(
        "pro_ceiling.custom_needs",
        tr("Custom needs? Write to us: {email}"),
    ).replace("{email}", email)


def copy_to_clipboard(text: str) -> bool:

    try:
        QApplication.clipboard().setText(text)
        return True
    except (RuntimeError, AttributeError):
        return False


def flash_text(widget, text: str, revert: Callable[[], str],
               ms: int | None = None) -> None:






    try:
        widget.setText(text)
    except RuntimeError:
        return
    if ms is None:
        ms = dial_in_range("tuning.contact.copied_ms", _COPIED_MS, 500, 10000)
    ref = weakref.ref(widget)

    def _back() -> None:
        alive = ref()
        if alive is None:
            return
        try:
            if sip.isdeleted(alive):
                return
            alive.setText(revert())
        except RuntimeError:
            pass  # nosec B110

    QTimer.singleShot(ms, _back)


def copy_with_feedback(button, email: str) -> None:





    copy_to_clipboard(email)
    flash_text(button, copied_text().replace("&", "&&"),
               lambda: copy_cta_text().replace("&", "&&"))


class CopyEmailLabel(QLabel):







    def __init__(self, parent=None):
        super().__init__(parent)
        self._email = ""
        self._line = ""
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.PlainText)


        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(scale_qss_font_px(_LINE_QSS))
        self.setVisible(False)

    def set_email(self, email: str | None) -> None:

        self._email = (email or "").strip()
        self._line = custom_needs_line(self._email) if self._email else ""
        self.setText(self._line)
        self.setToolTip(copy_cta_text() if self._email else "")
        self.setVisible(bool(self._email))

    def mousePressEvent(self, event):  # noqa: N802
        if self._email and event.button() == Qt.MouseButton.LeftButton:
            copy_to_clipboard(self._email)
            flash_text(self, copied_text(), lambda: self._line)
        super().mousePressEvent(event)
