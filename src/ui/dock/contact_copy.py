"""Our address, shown in the UI and copied on click.

Every place that names the contact address goes through here: the Pro
ceiling cards, the free walls, the account dialog. The address is plain
text the user can read and select; the click copies it and the control says
so for a moment. No ``mailto:`` anywhere: a mail client that is not set up
swallows the click and the user learns nothing.

The two served strings live here too, so the cards and the walls quote the
same words. Nothing here raises on the paint path.
"""
from __future__ import annotations

import weakref
from typing import Callable

from qgis.PyQt import sip
from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtWidgets import QApplication, QLabel

from ...core.i18n import tr
from ...core.server_dials import dial_copy
from .font_scale import scale_qss_font_px

_COPIED_MS = 2000
_LINE_QSS = "font-size: 11px; color: rgba(128,128,128,0.95);"


def copy_cta_text() -> str:
    """Label of every button that copies the address."""
    return dial_copy("pro_ceiling.copy_cta", tr("Copy email"))


def copied_text() -> str:
    """What the control says for a moment after the copy."""
    return dial_copy("pro_ceiling.copied", tr("Copied!"))


def custom_needs_line(email: str) -> str:
    """The one-line invitation under a free wall, with the address in it.
    Filled with ``str.replace``: ``format()`` on a served sentence raises on
    a stray brace, and this line paints a wall."""
    return dial_copy(
        "pro_ceiling.custom_needs",
        tr("Custom needs? Write to us: {email}"),
    ).replace("{email}", email)


def copy_to_clipboard(text: str) -> bool:
    """Put ``text`` on the system clipboard. False when there is none."""
    try:
        QApplication.clipboard().setText(text)
        return True
    except (RuntimeError, AttributeError):
        return False


def flash_text(widget, text: str, revert: Callable[[], str],
               ms: int = _COPIED_MS) -> None:
    """Show ``text`` on a button or label, then put back ``revert()``.

    The timer holds a weak reference and checks the C++ side before
    touching it: the dock can be closed inside the two seconds, and a
    callback on a dead widget kills QGIS.
    """
    try:
        widget.setText(text)
    except RuntimeError:
        return
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
            pass  # nosec B110 -- gone between the check and the call

    QTimer.singleShot(ms, _back)


def copy_with_feedback(button, email: str) -> None:
    """Copy ``email`` and let ``button`` say so for a moment.

    The button's resting label is the served copy CTA, read again on the
    way back so a deploy between the click and the revert still lands.
    """
    copy_to_clipboard(email)
    flash_text(button, copied_text().replace("&", "&&"),
               lambda: copy_cta_text().replace("&", "&&"))


class CopyEmailLabel(QLabel):
    """A muted one-liner naming the address; a click copies it.

    Plain text, selectable by mouse, hidden while it has no address. The
    served sentence is read on every ``set_email`` so a refresh after the
    copy blob lands rewords it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._email = ""
        self._line = ""
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(scale_qss_font_px(_LINE_QSS))
        self.setVisible(False)

    def set_email(self, email: str | None) -> None:
        """Fill the line with ``email``, or hide it when there is none."""
        self._email = (email or "").strip()
        self._line = custom_needs_line(self._email) if self._email else ""
        self.setText(self._line)
        self.setToolTip(copy_cta_text() if self._email else "")
        self.setVisible(bool(self._email))

    def mousePressEvent(self, event):  # noqa: N802 -- Qt override
        if self._email and event.button() == Qt.MouseButton.LeftButton:
            copy_to_clipboard(self._email)
            flash_text(self, copied_text(), lambda: self._line)
        super().mousePressEvent(event)
