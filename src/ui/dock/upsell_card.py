"""The one shape every Pro offer in the dock takes.

A premium-tinted card: a bold fact line (why the user is looking at this),
one line on what Pro removes, a blue button, and the free way out in muted
text. Three variants keep the family from reading as one stamped box:

- ``"full"``: the reference, the zone-cap card. Filled blue button across
  the card. For a refusal that blocks the current action.
- ``"compact"``: one bold line and the button on the SAME row. For a nudge
  the user can ignore (running low). The button is filled like every other
  Pro button: an outline one on a tinted card read as a disabled control,
  and the row already carries the "you may ignore this" weight.
- ``"star"``: the full card with a ★ headline above the fact line. For an
  offer that opens on the offer itself (the account dialog).
- ``"wall"``: the fact first, its muted note under it, THEN the ★ offer.
  For an end-of-allowance wall, where the user needs to read what happened
  before being sold the way out.

Copy is the caller's: it is filled through ``UpsellCard.set_text`` so a
served sentence and the shipped fallback go through one path. No networking,
no raise on the paint path.
"""
from __future__ import annotations

from typing import Callable

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from .font_scale import scale_qss_font_px
from .styles import (
    _BTN_BLUE,
    _CARD_CHILD_BTN_RESET_QSS,
    _PREMIUM_STAR,
    _SUBCARD_MARGINS,
    _msg_card_qss,
)

_TITLE_QSS = "font-size: 13px; font-weight: bold; color: palette(text);"
_BODY_QSS = "font-size: 12px; color: palette(text);"
_COMPACT_QSS = "font-size: 12px; font-weight: 600; color: palette(text);"
_MUTED_QSS = "font-size: 11px; color: rgba(128,128,128,0.95);"
_STAR_QSS = "font-size: 11px; font-weight: bold; color: palette(text);"


class UpsellCard(QFrame):
    """Premium-tinted offer card. Build once, fill with ``set_text``."""

    def __init__(self, name: str, variant: str = "full",
                 on_cta: Callable[[], None] | None = None, parent=None):
        super().__init__(parent)
        self.variant = variant
        self.setObjectName(name)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_msg_card_qss(name, "premium")
                           + _CARD_CHILD_BTN_RESET_QSS)
        self.star = QLabel()
        self.title = QLabel()
        self.note = QLabel()
        self.body = QLabel()
        self.button = QPushButton()
        self.escape = QLabel()
        for lbl in (self.star, self.title, self.note, self.body, self.escape):
            lbl.setWordWrap(True)
            # A served sentence never reaches a rich-text parser: plain text
            # only, so stray HTML-looking characters in a fallback never
            # render as markup.
            lbl.setTextFormat(Qt.TextFormat.PlainText)
        self.star.setStyleSheet(scale_qss_font_px(_STAR_QSS))
        self.body.setStyleSheet(scale_qss_font_px(_BODY_QSS))
        self.note.setStyleSheet(scale_qss_font_px(_MUTED_QSS))
        self.escape.setStyleSheet(scale_qss_font_px(_MUTED_QSS))
        self.note.setVisible(False)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        if on_cta is not None:
            self.button.clicked.connect(on_cta)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*_SUBCARD_MARGINS)
        layout.setSpacing(4)
        if variant == "compact":
            self.title.setStyleSheet(scale_qss_font_px(_COMPACT_QSS))
            self.button.setStyleSheet(_BTN_BLUE)
            self.button.setMinimumHeight(28)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            row.addWidget(self.title, 1)
            row.addWidget(self.button, 0, Qt.AlignmentFlag.AlignVCenter)
            layout.addLayout(row)
            layout.addWidget(self.body)
            self.star.setVisible(False)
            self.escape.setVisible(False)
        else:
            self.title.setStyleSheet(scale_qss_font_px(_TITLE_QSS))
            self.button.setStyleSheet(_BTN_BLUE)
            self.button.setMinimumHeight(34)
            self.escape.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if variant == "star":
                layout.addWidget(self.star)
            layout.addWidget(self.title)
            layout.addWidget(self.note)
            if variant == "wall":
                # The offer comes AFTER the fact and its note: a reader who
                # meets the price before knowing what stopped them reads the
                # card twice.
                layout.addSpacing(4)
                layout.addWidget(self.star)
            elif variant != "star":
                self.star.setVisible(False)
            layout.addWidget(self.body)
            layout.addSpacing(4)
            layout.addWidget(self.button)
            layout.addWidget(self.escape)

    def set_text(self, title: str, body: str | None, cta: str,
                 escape: str | None = None, star: str | None = None,
                 note: str | None = None) -> None:
        """Fill every line. ``None`` hides the optional ones. Plain text
        only: a served sentence never reaches a rich-text parser here."""
        self.title.setText(title)
        self.body.setText(body or "")
        self.body.setVisible(bool(body))
        # A QPushButton reads a single "&" as the mnemonic marker and eats
        # it, so a served or fallback sentence with one shows a missing
        # letter. Double it here, once, for every caller.
        self.button.setText(cta.replace("&", "&&"))
        if self.variant != "compact":
            self.escape.setText(escape or "")
            self.escape.setVisible(bool(escape))
            self.note.setText(note or "")
            self.note.setVisible(bool(note))
        if self.variant in ("star", "wall"):
            self.star.setText(f"{_PREMIUM_STAR}  {star}" if star else "")
            self.star.setVisible(bool(star))
