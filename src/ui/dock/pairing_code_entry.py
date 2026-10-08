







from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from .styles import (
    _BTN_GREEN,
    _CARD_TITLE_QSS,
    _HINT_LINE_QSS,
    _INPUT_THEME_QSS,
    BTN_PX,
    SPACE_CARD,
)


class DockPairingCodeMixin:


    def _build_pairing_code_entry(self, wait_layout) -> None:



        self._pairing_code_section = QWidget()
        code_layout = QVBoxLayout(self._pairing_code_section)
        code_layout.setContentsMargins(0, 0, 0, 0)
        code_layout.setSpacing(SPACE_CARD)
        prompt = QLabel(tr("Type the code shown in your browser:"))
        prompt.setWordWrap(True)
        prompt.setStyleSheet(_CARD_TITLE_QSS)
        code_layout.addWidget(prompt)

        row = QHBoxLayout()
        row.setSpacing(SPACE_CARD)
        self._pairing_code_input = QLineEdit()

        self._pairing_code_input.setMaxLength(12)
        self._pairing_code_input.setPlaceholderText("XXX-XXX")
        self._pairing_code_input.setMinimumHeight(BTN_PX)
        self._pairing_code_input.setStyleSheet(_INPUT_THEME_QSS)
        self._pairing_code_input.setAccessibleName(
            tr("Type the code shown in your browser:").rstrip(": "))
        self._pairing_code_input.returnPressed.connect(self._on_pairing_code_submit)
        row.addWidget(self._pairing_code_input, 1)
        self._pairing_code_btn = QPushButton(tr("Sign in"))
        self._pairing_code_btn.setMinimumHeight(BTN_PX)
        self._pairing_code_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pairing_code_btn.setStyleSheet(_BTN_GREEN)
        self._pairing_code_btn.setAutoDefault(False)
        self._pairing_code_btn.clicked.connect(self._on_pairing_code_submit)
        row.addWidget(self._pairing_code_btn)
        code_layout.addLayout(row)

        note = QLabel(tr("Only type a code you see on terra-lab.ai, on a page "
                         "you opened from this QGIS."))
        note.setWordWrap(True)
        note.setStyleSheet(_HINT_LINE_QSS)
        code_layout.addWidget(note)
        self._pairing_code_section.setVisible(False)
        wait_layout.addWidget(self._pairing_code_section)

    def _set_pairing_code_entry_visible(self, visible: bool) -> None:

        self._pairing_code_section.setVisible(visible)
        self._pairing_spinner.setVisible(not visible)
        self._pairing_status.setVisible(not visible)
        self._pairing_reopen_btn.setVisible(not visible)
        if not visible:
            self._pairing_code_input.clear()
            self.set_pairing_code_busy(False)

    def show_pairing_code_entry(self) -> None:

        if self._pairing_wait_section.isHidden():
            return
        self._pairing_anim_timer.stop()
        self._set_pairing_code_entry_visible(True)
        self._pairing_code_input.setFocus()

    def set_pairing_code_busy(self, busy: bool) -> None:

        self._pairing_code_input.setEnabled(not busy)
        self._pairing_code_btn.setEnabled(not busy)

    def _on_pairing_code_submit(self) -> None:
        typed = self._pairing_code_input.text().strip()
        if not typed or not self._pairing_code_btn.isEnabled():
            return
        self.activation_message_label.setVisible(False)
        self.pairing_code_entered.emit(typed)
