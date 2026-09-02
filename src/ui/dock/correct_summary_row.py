"""The Correct step's journal summary row, and its "Clear all" confirm.

Split out of auto_correct_build.py, which had reached the top of its size
band. One row, one file: the label that counts the round, the Undo-last link,
and the Clear-all link with the guard that makes it ask.

"Clear all" undoes a whole round of corrections at once, with nothing to bring
it back, so it asks first. It asks the way the retry link does, inline: the
first click relabels the link, the second one clears. A modal for a quiet text
link would stop the whole dock for a question the link can carry itself.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from ...core.i18n import tr
from ...core.interaction_dials import confirm_reset_ms
from ...core.qt_compat import safe_single_shot
from ...core.review_corrections import RetryLinkState
from .styles import _BTN_LINK_MUTED

_SUMMARY_TEXT_QSS = (
    "font-size: 11px; color: rgba(128,128,128,0.95);"
    " background: transparent; border: none;")
_SUMMARY_DOT_QSS = (
    "font-size: 11px; color: rgba(128,128,128,0.9);"
    " background: transparent; border: none;")

# How long the armed confirm waits before it goes back to its resting label.
# Long enough to read the question, short enough that a link left armed is not
# still armed when the user comes back to the card. Served as ui.confirm_reset_ms.
CLEAR_CONFIRM_RESET_MS = 6000


def clear_all_rest_label(count: int) -> str:
    """The link at rest. It names what it takes, so the count is on the button
    rather than only in the summary beside it."""
    if int(count) > 1:
        return tr("Clear all {n}").format(n=int(count))
    return tr("Clear all")


class ClearAllConfirm:
    """Two-stage guard driving one "Clear all" button.

    Owns the state machine AND the label, so nothing else can leave the link
    reading "Confirm" over a guard that has gone cold.
    """

    def __init__(self, button) -> None:
        self._button = button
        self._state = RetryLinkState()
        self._count = 0

    @property
    def armed(self) -> bool:
        """True while the confirm question is on the link."""
        return self._state.armed

    def clicked(self) -> bool:
        """Answer one click: True means the user confirmed and the round may
        be cleared. The first click only arms the question."""
        if self._state.activate():
            self._apply_label()
            return True
        self._apply_label()
        safe_single_shot(confirm_reset_ms(CLEAR_CONFIRM_RESET_MS), self._button, self.reset)
        return False

    def reset(self) -> None:
        """Disarm and put the resting label back (a timeout, or any other
        interaction with the card)."""
        if not self._state.armed:
            return
        self._state.reset()
        self._apply_label()

    def set_count(self, count: int) -> None:
        """Follow the journal: a new correction lands, so the question the user
        was asked is no longer the one on screen. Disarms."""
        self._count = max(0, int(count))
        self._state.reset()
        self._apply_label()

    def _apply_label(self) -> None:
        button = self._button
        if button is None:
            return
        try:
            if self._state.armed:
                # Same warm confirm look as the retry link, imported here and
                # not at module level: auto_review_build imports the build file
                # that imports this one.
                from .auto_review_build import _BTN_LINK_CONFIRM
                button.setText(tr("Undo every correction? Confirm"))
                button.setStyleSheet(_BTN_LINK_CONFIRM)
            else:
                button.setText(clear_all_rest_label(self._count))
                button.setStyleSheet(_BTN_LINK_MUTED)
        except (RuntimeError, AttributeError, ImportError):
            pass


def build_correction_summary_row(dock, lay) -> None:
    """Build "N corrections this round . Undo last . Clear all" onto ``dock``.

    Hidden while the journal is empty (the dock's set_correction_summary drives
    it). Clear all goes through the dock's own click handler, which asks the
    guard built here before it emits.
    """
    dock.auto_correct_summary_row = QWidget()
    row = QHBoxLayout(dock.auto_correct_summary_row)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(2)
    dock.auto_correct_summary_label = QLabel("")
    dock.auto_correct_summary_label.setStyleSheet(_SUMMARY_TEXT_QSS)
    dock.auto_correct_undo_btn = QPushButton(tr("Undo last"))
    dock.auto_correct_undo_btn.setStyleSheet(_BTN_LINK_MUTED)
    dock.auto_correct_undo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    dock.auto_correct_undo_btn.clicked.connect(
        dock.auto_correction_undo_requested.emit)
    dock.auto_correct_clear_btn = QPushButton(clear_all_rest_label(0))
    dock.auto_correct_clear_btn.setStyleSheet(_BTN_LINK_MUTED)
    dock.auto_correct_clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    dock.auto_correct_clear_btn.setToolTip(tr(
        "Undo every correction of this round at once. The count is in the "
        "label, so you can see what goes. It asks once first."))
    # Undoing a whole round cannot be taken back, so the link asks inline
    # before it fires.
    dock._correct_clear_confirm = ClearAllConfirm(dock.auto_correct_clear_btn)
    dock.auto_correct_clear_btn.clicked.connect(dock._on_correct_clear_clicked)
    row.addWidget(dock.auto_correct_summary_label)
    for widget in (dock.auto_correct_undo_btn, dock.auto_correct_clear_btn):
        dot = QLabel("\u00b7")
        dot.setStyleSheet(_SUMMARY_DOT_QSS)
        row.addWidget(dot)
        row.addWidget(widget)
    row.addStretch(1)
    dock.auto_correct_summary_row.setVisible(False)
    lay.addWidget(dock.auto_correct_summary_row)
