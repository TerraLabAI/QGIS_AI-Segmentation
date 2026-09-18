





from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QDialog, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout

from ...core.activation_manager import get_contact_call_url, get_support_email
from ...core.i18n import tr
from ...core.server_dials import dial_copy
from ..dock.contact_copy import copied_text, copy_cta_text, copy_to_clipboard, flash_text
from ..dock.font_scale import apply_font_scale_to_tree, scale_px_length, scale_qss_font_px
from ..dock.styles import (
    _BTN_SETTINGS_ACCENT,
    _BTN_SETTINGS_GHOST,
    FIELD,
    FONT_BASE,
    FONT_BODY,
    FONT_HINT,
    INK,
    INK_2,
    LINE,
    RADIUS_CONTROL,
)
from ..external_links import open_external_url
from .settings_widgets import TITLE_PX, settings_button

_CONTACT_QSS = scale_qss_font_px(
    "QDialog#contactDialog { background: palette(window); }"
    f"QLabel#contactTitle {{ font-size: {TITLE_PX}px; font-weight: 600; color: {INK}; }}"
    f"QLabel#contactSub {{ font-size: {FONT_BODY}px; color: {INK_2}; }}"
    f"QLabel#contactAddress {{ font-size: {FONT_BASE}px; font-weight: 600; color: {INK};"
    f" background: {FIELD}; border: 1px solid {LINE}; border-radius: {RADIUS_CONTROL}px;"
    " padding: 10px 12px; }"
    f"QLabel#contactOr {{ font-size: {FONT_HINT}px; color: {INK_2}; }}"
)


def show_contact_dialog(parent=None) -> None:
    dlg = build_contact_dialog(parent)
    dlg.exec()
    dlg.deleteLater()


def build_contact_dialog(parent=None) -> QDialog:
    support_email = get_support_email("yvann.barbot@terra-lab.ai")
    call_url = get_contact_call_url()

    dlg = QDialog(parent)
    dlg.setObjectName("contactDialog")
    dlg.setWindowTitle(tr("Contact us"))
    dlg.setStyleSheet(_CONTACT_QSS)
    dlg.setMinimumWidth(scale_px_length(360))

    dlg.setMaximumWidth(scale_px_length(600))
    lay = QVBoxLayout(dlg)
    lay.setContentsMargins(22, 20, 22, 18)
    lay.setSpacing(10)

    title = QLabel(tr("Contact us"), dlg)
    title.setObjectName("contactTitle")
    from .category_tile import category_icon_tile, tile_beside


    lay.addLayout(tile_beside(category_icon_tile("chat_bubble", "sky", dlg), title))
    sub = QLabel(tr("We read every message."), dlg)
    sub.setObjectName("contactSub")
    sub.setWordWrap(True)
    lay.addWidget(sub)

    address = QLabel(support_email, dlg)
    address.setObjectName("contactAddress")
    address.setTextFormat(Qt.TextFormat.PlainText)
    address.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    lay.addSpacing(4)
    lay.addWidget(address)



    copy_btn = settings_button(copy_cta_text().replace("&", "&&"),
                               _BTN_SETTINGS_ACCENT, dlg)
    copy_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    copy_btn.setDefault(True)

    def copy_address() -> None:
        if copy_to_clipboard(support_email):
            flash_text(copy_btn, copied_text().replace("&", "&&"),
                       lambda: copy_cta_text().replace("&", "&&"))

    copy_btn.clicked.connect(copy_address)
    lay.addWidget(copy_btn)

    if call_url:
        row = QHBoxLayout()
        row.setSpacing(8)
        or_label = QLabel(tr("or"), dlg)
        or_label.setObjectName("contactOr")
        row.addStretch(1)
        row.addWidget(or_label)
        row.addStretch(1)
        lay.addLayout(row)
        call_btn = settings_button(
            dial_copy("account.contact_call_cta", tr("Book a call")),
            _BTN_SETTINGS_GHOST, dlg)
        call_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        call_btn.clicked.connect(lambda: open_external_url(call_url, parent=dlg))
        lay.addWidget(call_btn)

    apply_font_scale_to_tree(dlg)
    return dlg
