











from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ...core.i18n import tr
from ..dock.font_scale import apply_font_scale_to_tree, scale_px_length, scale_qss_font_px
from ..dock.styles import (
    _BTN_SETTINGS_DANGER,
    _BTN_SETTINGS_GHOST,
    _INPUT_THEME_QSS,
    FONT_BASE,
    FONT_BODY,
    FONT_HINT,
    INK,
    INK_2,
)
from ..settings.a11y import DANGER_INK, make_accessible
from ..settings.settings_widgets import TITLE_PX

_DELETE_QSS = scale_qss_font_px(
    "QDialog#deleteAccountDialog { background: palette(window); }"
    f"QLabel#deleteTitle {{ font-size: {TITLE_PX}px; font-weight: 600; color: {DANGER_INK}; }}"
    f"QLabel#deleteBody {{ font-size: {FONT_BODY}px; color: {INK_2}; }}"
    f"QLabel#deleteWho {{ font-size: {FONT_BASE}px; font-weight: 600; color: {INK}; }}"
    f"QLabel#deleteAsk {{ font-size: {FONT_HINT}px; color: {INK_2}; }}"
)


class DeleteAccountDialog(QDialog):







    def __init__(self, email: str, parent=None):
        super().__init__(parent)
        self._email = (email or "").strip()
        self.setObjectName("deleteAccountDialog")
        self.setStyleSheet(_DELETE_QSS)
        self.setWindowTitle(tr("Delete my account"))
        self.setModal(True)



        width = scale_px_length(460)
        try:
            width = min(width, self.screen().availableGeometry().width() - 48)
        except (AttributeError, RuntimeError):
            pass  # nosec B110
        self.setFixedWidth(max(320, width))

        lay = QVBoxLayout(self)



        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(10)

        title = QLabel(tr("Delete your TerraLab account"))
        title.setObjectName("deleteTitle")
        title.setWordWrap(True)
        from ..settings.category_tile import category_icon_tile, tile_beside


        title_tile = category_icon_tile("trash", "coral", self)
        lay.addLayout(tile_beside(title_tile, title))

        body = QLabel(
            tr("Your account and its data are erased. Every TerraLab plugin "
               "stops, and a paid plan stops renewing.")
            + "\n\n"
            + tr("To cancel, sign in on terra-lab.ai before the grace period ends."))
        body.setObjectName("deleteBody")
        body.setWordWrap(True)
        lay.addWidget(body)

        who = QLabel(tr("Signed in as {email}").format(email=self._email))
        who.setObjectName("deleteWho")
        who.setWordWrap(True)


        who.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addSpacing(4)
        lay.addWidget(who)

        prompt = QLabel(tr("Type that address to confirm."))
        prompt.setObjectName("deleteAsk")
        prompt.setWordWrap(True)
        lay.addWidget(prompt)

        self.email_edit = QLineEdit()
        self.email_edit.setStyleSheet(_INPUT_THEME_QSS)
        self.email_edit.setPlaceholderText(self._email)
        self.email_edit.textChanged.connect(self._sync_confirm_enabled)
        lay.addWidget(self.email_edit)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 6, 0, 0)
        btn_row.setSpacing(8)
        btn_row.addStretch()
        cancel = QPushButton(tr("Cancel"))
        make_accessible(cancel, _BTN_SETTINGS_GHOST)
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)




        cancel.setAutoDefault(False)
        cancel.setDefault(False)
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(cancel)

        self.confirm_btn = QPushButton(tr("Delete my account"))
        make_accessible(self.confirm_btn, _BTN_SETTINGS_DANGER)
        self.confirm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.confirm_btn.setAutoDefault(False)
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.confirm_btn)
        lay.addLayout(btn_row)

        apply_font_scale_to_tree(self)


        inner = self.width() - 44
        for label in (title, body, who):
            label.ensurePolished()
            width = inner - (title_tile.width() + 12 if label is title else 0)
            label.setMinimumHeight(label.heightForWidth(width))
        self.email_edit.setFocus()

    def typed_email(self) -> str:

        return self.email_edit.text().strip()

    def _matches(self) -> bool:






        return bool(self._email) and (
            self.email_edit.text().strip().lower() == self._email.lower())

    def _sync_confirm_enabled(self, _text: str = "") -> None:
        self.confirm_btn.setEnabled(self._matches())
