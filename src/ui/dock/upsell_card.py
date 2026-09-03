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

from ...core.i18n import tr
from .contact_copy import CopyEmailLabel
from .font_scale import scale_qss_font_px
from .styles import (
    _BTN_BLUE,
    _BTN_LINK_MUTED,
    _CARD_CHILD_BTN_RESET_QSS,
    _PREMIUM_STAR,
    _SUBCARD_MARGINS,
    _msg_card_qss,
)

_TITLE_QSS = "font-size: 13px; font-weight: bold; color: palette(text);"
_BODY_QSS = "font-size: 12px; color: palette(text);"
_DETAIL_QSS = "font-size: 12px; font-weight: bold; color: palette(text);"
_COMPACT_QSS = "font-size: 12px; font-weight: 600; color: palette(text);"
_MUTED_QSS = "font-size: 11px; color: rgba(128,128,128,0.95);"
_STAR_QSS = "font-size: 11px; font-weight: bold; color: palette(text);"


class UpsellCard(QFrame):
    """Premium-tinted offer card. Build once, fill with ``set_text``."""

    def __init__(self, name: str, variant: str = "full",
                 on_cta: Callable[[], None] | None = None, parent=None):
        super().__init__(parent)
        self.variant = variant
        self._tint = "premium"
        self.setObjectName(name)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_msg_card_qss(name, "premium")
                           + _CARD_CHILD_BTN_RESET_QSS)
        self.star = QLabel()
        self.title = QLabel()
        self.note = QLabel()
        self.body = QLabel()
        # The one fact the button acts on (an address to copy), between the
        # body and the button so it is read before the click. Selectable:
        # a user who prefers to drag it into a mail must be able to.
        self.detail = QLabel()
        self.button = QPushButton()
        # The quiet second door, under the blue button: the public plan list.
        # An organisation reading one price on one card has no way to know
        # there is a yearly plan and a team plan, and a card cannot hold them.
        # Hidden until a caller calls set_pro_offer.
        self.plans_link = QPushButton()
        self.escape = QLabel()
        # Under everything else: the invitation to write to us with a custom
        # need. A click copies the address. Hidden until a caller fills it.
        self.contact = CopyEmailLabel()
        for lbl in (self.star, self.title, self.note, self.body, self.detail,
                    self.escape):
            lbl.setWordWrap(True)
            # A served sentence never reaches a rich-text parser: plain text
            # only, so stray HTML-looking characters in a fallback never
            # render as markup.
            lbl.setTextFormat(Qt.TextFormat.PlainText)
        self.star.setStyleSheet(scale_qss_font_px(_STAR_QSS))
        self.body.setStyleSheet(scale_qss_font_px(_BODY_QSS))
        self.detail.setStyleSheet(scale_qss_font_px(_DETAIL_QSS))
        self.detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.note.setStyleSheet(scale_qss_font_px(_MUTED_QSS))
        self.escape.setStyleSheet(scale_qss_font_px(_MUTED_QSS))
        self.note.setVisible(False)
        self.detail.setVisible(False)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.plans_link.setCursor(Qt.CursorShape.PointingHandCursor)
        self.plans_link.setStyleSheet(_BTN_LINK_MUTED)
        self.plans_link.setVisible(False)
        self.plans_link.clicked.connect(self._on_plans_link_clicked)
        self._plans_cta_source = ""
        self._pro_offer_armed = False
        self._pro_offer_fallback: str | None = None
        # The line the price is added to, as the caller wrote it. Kept apart
        # from what the label shows, so re-applying the price never stacks a
        # second copy on top of the first.
        self._pro_offer_base: str | None = None
        self._on_cta = on_cta
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
            layout.addWidget(self.detail)
            layout.addWidget(self.plans_link, 0,
                             Qt.AlignmentFlag.AlignRight)
            self.star.setVisible(False)
            self.escape.setVisible(False)
            self.contact.setVisible(False)
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
            layout.addWidget(self.detail)
            layout.addSpacing(4)
            layout.addWidget(self.button)
            layout.addWidget(self.escape)
            layout.addWidget(self.plans_link, 0,
                             Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.contact)

    def set_tint(self, kind: str) -> None:
        """Repaint the card in another message tint.

        One card can carry more than one kind of news (see auto_run_block.py,
        where the same card names a refusal and a service outage). A premium
        tint on a card that sells nothing reads as an offer the reader cannot
        find, so the tint follows the news, not the widget. A no-op when the
        tint is already the one asked for.
        """
        if kind == self._tint:
            return
        self._tint = kind
        self.setStyleSheet(_msg_card_qss(self.objectName(), kind)
                           + _CARD_CHILD_BTN_RESET_QSS)

    def route_cta(self, on_cta: Callable[[], None]) -> None:
        """Point the button at another handler.

        The same card can sell Pro to a free account and, for a subscriber
        who spent the month, copy our address instead: one card, one
        button, and the handler follows the account rather than a second
        card fighting the first for the page. A no-op when the handler is
        already the one wired.
        """
        if on_cta == self._on_cta:
            return
        try:
            self.button.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass  # nosec B110 -- nothing was connected
        self.button.clicked.connect(on_cta)
        self._on_cta = on_cta

    def set_text(self, title: str, body: str | None, cta: str,
                 escape: str | None = None, star: str | None = None,
                 note: str | None = None, detail: str | None = None) -> None:
        """Fill every line. ``None`` hides the optional ones. Plain text
        only: a served sentence never reaches a rich-text parser here."""
        self.title.setText(title)
        self.body.setText(body or "")
        self.body.setVisible(bool(body))
        self.detail.setText(detail or "")
        self.detail.setVisible(bool(detail))
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
        # A refill replaces the line the price was added to, so the price is
        # taken from the new one. A no-op on a card that sells nothing.
        self._pro_offer_base = None
        self._apply_pro_offer()

    def set_pro_offer(self, cta_source: str,
                      price_fallback: str | None = None) -> None:
        """Put the served price on this card and show the See all plans link.

        Call it right after ``set_text`` on any card that sells Pro. Two rules
        keep the card the size it was. The price joins a line the card already
        has, the muted one under the button where there is one and the body
        line on a compact card, so no card grows a line to hold it. And the
        plan list is a small text link, never a second button: one card, one
        thing to press.

        ``price_fallback`` is the sentence to show when the server serves no
        price: the shipped or served hint the card carried before this. A
        served price always wins over it, so the number the website charges is
        the number the card shows.

        A compact card takes the link and no price: see _apply_pro_offer.
        """
        self._plans_cta_source = cta_source
        self._pro_offer_fallback = price_fallback
        self._pro_offer_armed = True
        self._pro_offer_base = None
        self._apply_pro_offer()
        self.plans_link.setText(tr("See all plans"))
        self.plans_link.setVisible(True)

    def _apply_pro_offer(self) -> None:
        """Write the price into the card's own line, from the config in force.

        Read late, never once. Two of these cards are filled while the plugin
        starts, before the first configuration has landed, so a price read at
        build time is the shipped fallback for the whole session. This runs
        again on every refill and every time the card is shown, which is when
        the number has to be right.
        """
        if not self._pro_offer_armed or self.variant == "compact":
            # A compact card is one line and a button on the same row. Measured
            # at the dock's real width, the price wraps that line in two and
            # pushes the card over the widget under it, so this nudge carries
            # the plan link alone and the price waits for the card that sells.
            return
        from ...core.pro_offer_copy import join_offer_line, pro_price_phrase

        if self._pro_offer_base is None:
            self._pro_offer_base = self.escape.text()
        phrase = pro_price_phrase() or (self._pro_offer_fallback or "")
        line = join_offer_line(self._pro_offer_base, phrase)
        self.escape.setText(line)
        self.escape.setVisible(bool(line))

    def showEvent(self, event):  # noqa: N802 -- Qt name
        """The last chance to get the price right before it is read."""
        self._apply_pro_offer()
        super().showEvent(event)

    def _on_plans_link_clicked(self) -> None:
        """The plan list, opened as it always was: a plain public page, no
        account, no server call, no waiting."""
        from ...core.activation_manager import get_plans_page_url
        from ..external_links import open_external_url
        open_external_url(
            get_plans_page_url(self._plans_cta_source or "plugin"),
            parent=self)

    def set_contact_email(self, email: str | None) -> None:
        """Show the custom-needs line under the card with ``email`` in it,
        or hide it. Compact cards have no room for it and ignore the call."""
        if self.variant == "compact":
            return
        self.contact.set_email(email)
