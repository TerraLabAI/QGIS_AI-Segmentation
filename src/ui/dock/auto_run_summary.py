"""The in-run receipt card: what the user gave this run, and nothing else.

Part of the Automatic page (see auto_build.py). Its own widget rather than a
stripped-down copy of the setup cards, because a card whose control is empty
and unusable says less than no card at all.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...core.i18n import tr
from .styles import _CARD_MARGINS, _CARD_QSS


class AutoRunSummaryCard(QWidget):
    """What the run was given, shown while the tiles are in flight.

    A run takes no new word and no new example, so the two setup cards leave
    the screen and this one takes their place: the typed word, the drawn
    references, or both. One card, because the run holds the two inputs
    together; the "and / or" separator belongs to the choice, and the choice
    is behind the user by then.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("autoRunSummaryCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_CARD_QSS.format(name="autoRunSummaryCard"))
        _col = QVBoxLayout(self)
        _col.setContentsMargins(*_CARD_MARGINS)
        _col.setSpacing(6)
        # The 12px bold header every card on this page wears. It says the card
        # is a receipt, and it covers a word, a reference and an exclude alike,
        # which "Looking for" does not.
        self._summary_header = QLabel(tr("What you asked for"))
        self._summary_header.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: palette(text);"
            " background: transparent; border: none;")
        _col.addWidget(self._summary_header)
        # The typed word, one step larger than the header: the header names
        # the line, the line carries the answer.
        self._summary_word = QLabel("")
        self._summary_word.setWordWrap(True)
        self._summary_word.setStyleSheet(
            "font-size: 13px; color: palette(text);"
            " background: transparent; border: none;")
        self._summary_word.setVisible(False)
        _col.addWidget(self._summary_word)
        # The drawn references, as the same thumbnails the setup card shows
        # (built by the dock, minus the remove x). They are the only picture of
        # what was asked for, so they stay clickable to enlarge.
        self._summary_thumbs = QWidget()
        self._summary_thumbs_layout = QHBoxLayout(self._summary_thumbs)
        self._summary_thumbs_layout.setContentsMargins(0, 0, 0, 0)
        self._summary_thumbs_layout.setSpacing(6)
        self._summary_thumbs_layout.addStretch()
        self._summary_thumbs.setVisible(False)
        _col.addWidget(self._summary_thumbs)
        self.setVisible(False)

    def set_run_recipe(self, word: str, chips: list) -> None:
        """Show this run's typed word and reference chips, dropping whatever
        the previous run left. A line with nothing to say is hidden, never
        shown empty. ``chips`` are ready-made widgets: this card owns them
        from here and deletes them on the next call."""
        while self._summary_thumbs_layout.count() > 1:
            item = self._summary_thumbs_layout.takeAt(0)
            old = item.widget()
            if old is not None:
                old.setParent(None)
                old.deleteLater()
        word = (word or "").strip()
        self._summary_word.setText(word)
        self._summary_word.setVisible(bool(word))
        for chip in chips:
            self._summary_thumbs_layout.insertWidget(
                self._summary_thumbs_layout.count() - 1, chip)
        self._summary_thumbs.setVisible(bool(chips))
