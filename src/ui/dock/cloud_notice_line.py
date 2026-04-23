

















from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QLabel

from ...core.i18n import tr
from ...core.server_dials import dial_copy
from .font_scale import scale_qss_font_px
from .styles import FONT_HINT, INK_2, LINK_INK


_DOT = "·"




_CLOUD_NOTICE_QSS = scale_qss_font_px(f"font-size: {FONT_HINT}px; color: {INK_2};"
                                      " background: transparent;")


_CLOUD_NOTICE_LINK_COLOR = LINK_INK


def build_cloud_notice_line() -> QLabel:

    label = QLabel("")
    label.setObjectName("cloudNoticeLine")
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setOpenExternalLinks(True)
    label.setStyleSheet(_CLOUD_NOTICE_QSS)
    label.setVisible(False)
    return label


def cloud_notice_line_html() -> str:








    from ...core.activation_manager import get_privacy_url

    link = (f'<a href="{get_privacy_url()}"'
            f' style="color: {_CLOUD_NOTICE_LINK_COLOR}; text-decoration: none;">'
            f'{tr("Privacy")}</a>')
    text = dial_copy(
        "engine.privacy_line",
        tr("Your selection is sent to our servers in Europe {dot} {privacy}"))
    return text.replace("{dot}", _DOT).replace("{privacy}", link)
