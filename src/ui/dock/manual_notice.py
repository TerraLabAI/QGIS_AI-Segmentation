"""The small in-panel notice for a click the user can fix themselves.

One line, on the instruction card the user is already reading, tinted with the
taxonomy's transient error pair. It replaces the guidance for as long as it is
up, because a panel that says "click the object you want to segment" under a
sentence explaining that this layer draws nothing here is two answers to one
question.

It exists so that picking the wrong basemap stops opening a copy-logs report
dialog. What the notice says, and which failures earn one, is decided in
``core/click_error_advice.py``; this file only puts a sentence on screen.
"""

from __future__ import annotations

from qgis.PyQt.QtCore import QTimer

from .styles import _msg_label_qss, _msg_text

# Long enough to read a line and act on it, short enough that a notice never
# outlives the state it describes. A new read clears it before this fires.
_NOTICE_MS = 12000


class DockManualNoticeMixin:
    """Transient one-line error notice on the Manual instruction card."""

    def show_manual_notice(self, text: str) -> None:
        """Put ``text`` on the instruction card until the next read, or 12s.

        Falls back to the QGIS message bar whenever the card is not on screen
        (a refine handoff, where the Correct card owns the panel). A notice
        that lands nowhere is worse than the dialog it replaced.
        """
        if not text:
            return
        label = getattr(self, "instructions_label", None)
        try:
            usable = label is not None and label.isVisibleTo(self)
        except RuntimeError:
            usable = False
        if not usable:
            self._push_notice_to_message_bar(text)
            return
        self._manual_notice_text = text
        self._paint_manual_notice(label, text)
        timer = self._manual_notice_timer()
        timer.start(_NOTICE_MS)

    def clear_manual_notice(self) -> None:
        """Drop a live notice and hand the card back to the guidance.

        Called when a new read starts: the user did the thing the notice asked
        for, so the sentence has stopped being true.
        """
        if not getattr(self, "_manual_notice_text", ""):
            return
        self._manual_notice_text = ""
        timer = getattr(self, "_manual_notice_qtimer", None)
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:  # nosec B110 -- teardown
                pass
        try:
            self._update_instructions()
        except (RuntimeError, AttributeError):  # nosec B110 -- teardown
            pass

    def manual_notice_is_live(self) -> bool:
        """A notice owns the instruction card right now."""
        return bool(getattr(self, "_manual_notice_text", ""))

    def _paint_manual_notice(self, label, text: str) -> None:
        """Dress the instruction card as the taxonomy's transient error."""
        self._instructions_style = "notice"
        label.setStyleSheet(_msg_label_qss("error_transient"))
        label.setMinimumHeight(0)
        label.setText(_msg_text("error", text))
        label.setVisible(True)

    def _manual_notice_timer(self) -> QTimer:
        """The one single-shot timer this dock uses for notices.

        Parented to the dock so it dies with it, and reused rather than
        recreated so a second failure restarts the countdown instead of
        stacking two timers that each clear a card they no longer own.
        """
        timer = getattr(self, "_manual_notice_qtimer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self.clear_manual_notice)
            self._manual_notice_qtimer = timer
        return timer

    def _push_notice_to_message_bar(self, text: str) -> None:
        """Last resort when the instruction card is off screen."""
        try:
            from qgis.core import Qgis
            from qgis.utils import iface as _iface

            _iface.messageBar().pushMessage(
                "AI Segmentation", text,
                level=Qgis.MessageLevel.Warning, duration=10)
        except Exception:  # noqa: BLE001 -- a notice must never break a click
            pass  # nosec B110
