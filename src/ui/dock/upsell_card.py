
































from __future__ import annotations

from typing import Callable

from qgis.PyQt.QtCore import QSize, Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
)

from ...core.i18n import tr
from .contact_copy import CopyEmailLabel
from .font_scale import scale_px_length, scale_qss_font_px
from .styles import (
    _BTN_GHOST,
    _BTN_GREEN_STEP,
    _BTN_LINK_QUIET,
    _CARD_CHILD_BTN_RESET_QSS,
    ACCENT_BORDER,
    BTN_PILL_PX,
    BTN_PRIMARY_WIDE_PX,
    FONT_BASE,
    FONT_BODY,
    FONT_HINT,
    HOVER,
    HOVER_ON,
    INK,
    INK_2,
    MUTED,
    ON_ACCENT,
    RADIUS_CONTROL,
    SPACE_CARD,
    SPACE_TIGHT,
    _msg_card_qss,
    category_ink,
)

_TITLE_QSS = f"font-size: {FONT_BASE}px; font-weight: 600; color: {INK};"
_BODY_QSS = f"font-size: {FONT_BODY}px; color: {INK};"
_DETAIL_QSS = f"font-size: {FONT_BODY}px; font-weight: 600; color: {INK};"
_COMPACT_QSS = f"font-size: {FONT_BODY}px; font-weight: 600; color: {INK};"
_MUTED_QSS = f"font-size: {FONT_HINT}px; color: {MUTED};"

_OFFER_CARD_MARGINS = (16, 14, 16, 14)



_DISMISS_PX = 20
_DISMISS_GLYPH_PX = 12
_DISMISS_QSS = (
    "QToolButton#upsellDismiss { background: transparent; border: none;"
    f" padding: 0; border-radius: {RADIUS_CONTROL}px; }}"
    f"QToolButton#upsellDismiss:hover {{ background: {HOVER}; }}"
    f"QToolButton#upsellDismiss:pressed {{ background: {HOVER_ON}; }}"
    f"QToolButton#upsellDismiss:focus {{ border: 2px solid {ACCENT_BORDER}; }}"
)
_FLAT_QSS = (
    "QFrame#{name} {{ background: transparent; border: none; }}"
    "QLabel {{ background: transparent; border: none; }}"
)


class UpsellCard(QFrame):



    dismissed = pyqtSignal()

    def __init__(self, name: str, variant: str = "full",
                 on_cta: Callable[[], None] | None = None, parent=None,
                 flat: bool | None = None):
        super().__init__(parent)
        self.variant = variant


        self._flat = (variant == "wall") if flat is None else bool(flat)
        self._tint = "premium"
        self._ghost = False
        self.setObjectName(name)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._apply_card_qss()
        self.star = QLabel()
        self.title = QLabel()
        self.note = QLabel()
        self.body = QLabel()


        self.detail = QLabel()
        self.button = QPushButton()
        self.button.setAutoDefault(False)


        self.plans_link = QPushButton()
        self.escape = QLabel()


        self.contact = CopyEmailLabel()
        for lbl in (self.star, self.title, self.note, self.body, self.detail,
                    self.escape):
            lbl.setWordWrap(True)

            lbl.setTextFormat(Qt.TextFormat.PlainText)
        self.star.setStyleSheet(scale_qss_font_px(_BODY_QSS))
        self.body.setStyleSheet(scale_qss_font_px(_BODY_QSS))
        self.detail.setStyleSheet(scale_qss_font_px(_MUTED_QSS))
        self.detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.note.setStyleSheet(scale_qss_font_px(_MUTED_QSS))
        self.escape.setStyleSheet(scale_qss_font_px(_MUTED_QSS))
        self.note.setVisible(False)
        self.detail.setVisible(False)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.plans_link.setCursor(Qt.CursorShape.PointingHandCursor)
        self.plans_link.setStyleSheet(_BTN_LINK_QUIET)
        self.plans_link.setVisible(False)
        self.plans_link.clicked.connect(self._on_plans_link_clicked)
        self._plans_cta_source = ""
        self._on_cta = on_cta


        self._offer_cta = on_cta
        if on_cta is not None:
            self.button.clicked.connect(on_cta)


        self.dismiss_button = QToolButton()
        self.dismiss_button.setObjectName("upsellDismiss")
        self.dismiss_button.setStyleSheet(_DISMISS_QSS)
        self.dismiss_button.setFixedSize(
            scale_px_length(_DISMISS_PX), scale_px_length(_DISMISS_PX))
        self.dismiss_button.setIconSize(QSize(_DISMISS_GLYPH_PX, _DISMISS_GLYPH_PX))
        self.dismiss_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.dismiss_button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.dismiss_button.setAutoRaise(True)
        self.dismiss_button.setToolTip(tr("Dismiss"))
        self.dismiss_button.setAccessibleName(tr("Dismiss"))
        self.dismiss_button.clicked.connect(self._on_dismiss_clicked)
        self.dismiss_button.setVisible(False)

        layout = QVBoxLayout(self)
        if self._flat:
            layout.setContentsMargins(0, 0, 0, 0)
        else:
            layout.setContentsMargins(*_OFFER_CARD_MARGINS)
        layout.setSpacing(SPACE_TIGHT)
        if variant == "compact":
            self.title.setStyleSheet(scale_qss_font_px(_COMPACT_QSS))
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(SPACE_CARD + 2)
            row.addWidget(self.title, 1)
            row.addWidget(self.button, 0, Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(self.dismiss_button, 0, Qt.AlignmentFlag.AlignVCenter)
            layout.addLayout(row)
            layout.addWidget(self.body)
            layout.addWidget(self.detail)
            self.star.setVisible(False)
            self.escape.setVisible(False)
            self.contact.setVisible(False)
        else:
            self.title.setStyleSheet(scale_qss_font_px(_TITLE_QSS))
            title_row = QHBoxLayout()
            title_row.setContentsMargins(0, 0, 0, 0)
            title_row.setSpacing(SPACE_CARD)
            title_row.addWidget(self.title, 1)
            title_row.addWidget(self.dismiss_button, 0, Qt.AlignmentFlag.AlignTop)
            layout.addLayout(title_row)
            layout.addWidget(self.note)
            layout.addSpacing(SPACE_CARD)
            if variant in ("star", "wall"):
                layout.addWidget(self.star)
            else:
                self.star.setVisible(False)
            layout.addWidget(self.body)
            layout.addSpacing(SPACE_CARD)
            layout.addWidget(self.button)
            layout.addWidget(self.detail)
            layout.addWidget(self.escape)
            layout.addWidget(self.contact)
        self._apply_button_qss()



    def _apply_card_qss(self) -> None:
        name = self.objectName()
        if self._flat and self._tint in ("premium", "neutral"):
            qss = _FLAT_QSS.format(name=name)
        else:
            qss = _msg_card_qss(name, self._tint)
        self.setStyleSheet(qss + _CARD_CHILD_BTN_RESET_QSS)

    def _apply_button_qss(self) -> None:


        if self._ghost or self.variant == "compact":
            self.button.setStyleSheet(_BTN_GHOST)
            height = BTN_PILL_PX
        else:
            self.button.setStyleSheet(_BTN_GREEN_STEP)
            height = BTN_PRIMARY_WIDE_PX
        self.button.setFixedHeight(scale_px_length(height))

    def set_ghost_button(self, ghost: bool) -> None:

        ghost = bool(ghost)
        if ghost == self._ghost:
            return
        self._ghost = ghost
        self._apply_button_qss()
        self._paint_pro_gem()

        if self.variant != "compact":
            self.layout().setAlignment(
                self.button,
                Qt.AlignmentFlag.AlignLeft if ghost else Qt.AlignmentFlag(0))

    def enable_dismiss(self, enabled: bool = True) -> None:

        try:
            from ..icons import icon_for

            if enabled and self.dismiss_button.icon().isNull():
                self.dismiss_button.setIcon(icon_for(
                    self.dismiss_button, "close", _DISMISS_GLYPH_PX, QColor(INK_2)))
            self.dismiss_button.setVisible(bool(enabled))
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _on_dismiss_clicked(self) -> None:
        self.setVisible(False)
        self.dismissed.emit()

    def set_tint(self, kind: str) -> None:






        if kind == self._tint:
            return
        self._tint = kind
        self._apply_card_qss()



    def route_cta(self, on_cta: Callable[[], None]) -> None:






        if on_cta == self._on_cta:
            return
        try:
            self.button.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass  # nosec B110
        self.button.clicked.connect(on_cta)
        self._on_cta = on_cta
        self._paint_pro_gem()

    def set_text(self, title: str, body: str | None, cta: str,
                 escape: str | None = None, star: str | None = None,
                 note: str | None = None, detail: str | None = None) -> None:


        self.title.setText(title or "")
        self.title.setVisible(bool(title))
        self.body.setText(body or "")
        self.body.setVisible(bool(body))
        self.detail.setText(detail or "")
        self.detail.setVisible(bool(detail))

        self.button.setText((cta or "").replace("&", "&&"))
        if self.variant != "compact":
            self.escape.setText(escape or "")
            self.escape.setVisible(bool(escape))
            self.note.setText(note or "")
            self.note.setVisible(bool(note))
        if self.variant in ("star", "wall"):
            self.star.setText(star or "")
            self.star.setVisible(bool(star))

    def set_pro_offer(self, cta_source: str,
                      price_fallback: str | None = None) -> None:







        self._plans_cta_source = cta_source
        self._paint_pro_gem()

    def _paint_pro_gem(self) -> None:



        try:
            from qgis.PyQt.QtCore import QSize
            from qgis.PyQt.QtGui import QColor, QIcon

            from ..icons import icon_for

            selling = bool(self._plans_cta_source) and not self._ghost and (
                self._on_cta is None or self._on_cta == self._offer_cta)
            if selling:



                ink = (category_ink("amber") if self.variant == "compact"
                       else ON_ACCENT)
                self.button.setIcon(
                    icon_for(self.button, "gem", 14, QColor(ink)))
                self.button.setIconSize(QSize(14, 14))
            else:
                self.button.setIcon(QIcon())
        except Exception:  # noqa: BLE001
            return

    def _on_plans_link_clicked(self) -> None:

        from ...core.activation_manager import get_plans_page_url
        from ..external_links import open_external_url
        open_external_url(
            get_plans_page_url(self._plans_cta_source or "plugin"),
            parent=self)

    def set_contact_email(self, email: str | None) -> None:


        if self.variant == "compact":
            return
        self.contact.set_email(email)


def keep_working_cta() -> str:


    return tr("Get Pro")
