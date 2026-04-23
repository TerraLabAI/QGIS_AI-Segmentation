











from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ...core import qt_compat as QtC
from ...core.activation_manager import get_consent_terms_url, get_privacy_url
from ...core.i18n import tr
from ..external_links import open_external_url

_DIALOG_W = 480
_SIDE_PAD = 24
_MARK_PX = 40


_ROW_CATEGORIES = {"globe": "sky", "layers": "teal", "chart": "green"}


def _row_category(glyph: str) -> str:
    return _ROW_CATEGORIES.get(glyph, "leaf")


class PrivacyNoticeDialog(QDialog):


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("privacyNoticeDialog")
        self.setWindowTitle(tr("Before you start"))
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self):


        from ..dock.font_scale import apply_font_scale_to_tree, scale_px_length
        from ..dock.styles import LINE

        self.setStyleSheet("QDialog#privacyNoticeDialog { background: palette(window); }")
        width = scale_px_length(_DIALOG_W)
        try:
            width = min(width, self.screen().availableGeometry().width() - 48)
        except (AttributeError, RuntimeError):
            pass  # nosec B110
        self.setFixedWidth(max(320, width))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(self._build_header())

        body = QVBoxLayout()
        body.setContentsMargins(_SIDE_PAD, 6, _SIDE_PAD, 18)
        body.setSpacing(14)
        rows = (
            ("globe", tr("The map area you detect on, and what you ask us to find in "
                         "it, go to our servers only to run the detection.")),
            ("layers", tr("The detections stay in your account until you delete them; "
                          "the polygons you keep are written into your own project.")),
            ("chart", tr("Usage statistics linked to your account help us fix bugs; "
                         "you can turn them off in Settings at any time.")),
        )
        for glyph, sentence in rows:
            body.addWidget(self._build_row(glyph, sentence))
        layout.addLayout(body)

        rule = QFrame(self)
        rule.setFixedHeight(1)
        rule.setStyleSheet(f"border: none; background: {LINE};")
        layout.addWidget(rule)

        footer = QHBoxLayout()
        footer.setContentsMargins(_SIDE_PAD, 13, _SIDE_PAD, 15)
        footer.setSpacing(14)
        self._fill_footer(footer)
        layout.addLayout(footer)
        apply_font_scale_to_tree(self)

    def _build_header(self) -> QHBoxLayout:
        from ..dock.font_scale import scale_qss_font_px
        from ..dock.styles import FONT_HERO, INK
        from ..icons import logo_pixmap, logo_size

        header = QHBoxLayout()
        header.setContentsMargins(_SIDE_PAD, 22, _SIDE_PAD, 10)
        header.setSpacing(14)
        mark = QLabel(self)
        mark.setFixedSize(logo_size(_MARK_PX))
        mark.setPixmap(logo_pixmap(mark, _MARK_PX))
        header.addWidget(mark, 0, Qt.AlignmentFlag.AlignVCenter)
        heading = QLabel(tr("How AI Segmentation uses your data"), self)
        heading.setWordWrap(True)
        heading.setStyleSheet(scale_qss_font_px(
            f"font-size: {FONT_HERO}px; font-weight: 600; color: {INK};"))
        header.addWidget(heading, 1, Qt.AlignmentFlag.AlignVCenter)
        return header

    def _build_row(self, glyph: str, sentence: str) -> QFrame:
        from ..dock.font_scale import scale_qss_font_px
        from ..dock.styles import FONT_BASE, INK
        from ..settings.category_tile import category_icon_tile

        row = QFrame(self)
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(14)


        badge = category_icon_tile(glyph, _row_category(glyph), row)
        line.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        text = QLabel(sentence, row)
        text.setWordWrap(True)
        text.setTextFormat(Qt.TextFormat.PlainText)
        text.setStyleSheet(scale_qss_font_px(f"font-size: {FONT_BASE}px; color: {INK};"))
        line.addWidget(text, 1)
        return row

    def _fill_footer(self, footer: QHBoxLayout) -> None:

        from ..dock.font_scale import scale_qss_font_px
        from ..dock.styles import _BTN_SETTINGS_ACCENT, FONT_HINT, INK_2
        from ..settings.a11y import make_accessible



        terms_link = (f'<a href="{get_consent_terms_url("consent_terms")}">'
                      f'{tr("Terms")}</a>')
        privacy_link = (f'<a href="{get_privacy_url("consent_privacy")}">'
                        f'{tr("Privacy Policy")}</a>')
        body = tr("By continuing you accept the {terms} and the {privacy}.")
        for token, value in (("{terms}", terms_link), ("{privacy}", privacy_link)):
            body = body.replace(token, value)
        accept_line = QLabel(body, self)
        accept_line.setObjectName("privacyNoticeAcceptLine")
        accept_line.setWordWrap(True)
        accept_line.setTextFormat(Qt.TextFormat.RichText)


        accept_line.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)


        accept_line.setOpenExternalLinks(False)
        accept_line.linkActivated.connect(lambda url: open_external_url(url, self))
        accept_line.setStyleSheet(scale_qss_font_px(f"font-size: {FONT_HINT}px; color: {INK_2};"))
        footer.addWidget(accept_line, 1)

        self._continue_btn = QPushButton(tr("Continue"))
        self._continue_btn.setObjectName("privacyNoticeContinue")
        make_accessible(self._continue_btn, _BTN_SETTINGS_ACCENT)
        self._continue_btn.setCursor(QtC.PointingHandCursor)
        self._continue_btn.setDefault(True)
        self._continue_btn.clicked.connect(self.accept)
        footer.addWidget(self._continue_btn, 0, Qt.AlignmentFlag.AlignVCenter)
