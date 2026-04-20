"""Account Settings dialog for AI Segmentation plugin."""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt, QThread, QTimer, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.i18n import tr

BRAND_BLUE = "#1976d2"
BRAND_GREEN = "#2e7d32"
BRAND_RED = "#c62828"

PRODUCT_ID = "ai-segmentation"
PRODUCT_NAME = "AI Segmentation"

_DASHBOARD_URL = (
    "https://terra-lab.ai/dashboard"
    "?utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation&utm_content=dashboard"
)

_STATUS_DISPLAY = {
    "active": (tr("Active"), BRAND_GREEN),
    "canceled": (tr("Canceled"), BRAND_RED),
}

_LINK_BTN = (
    f"QPushButton {{ border: none; color: {BRAND_BLUE}; font-size: 11px;"
    f" text-decoration: underline; padding: 2px 4px; background: transparent; }}"
    f"QPushButton:hover {{ color: #1565c0; }}"
)

_CARD_STYLE = (
    "QFrame { background: rgba(128,128,128,0.08);"
    " border: 1px solid rgba(128,128,128,0.2);"
    " border-radius: 6px; }"
    "QLabel { background: transparent; border: none; }"
    "QPushButton { background: transparent; }"
)

# Qt5/Qt6 compat helpers
try:
    _AlignCenter = Qt.AlignmentFlag.AlignCenter
except AttributeError:
    _AlignCenter = Qt.AlignCenter

try:
    _PointingHandCursor = Qt.CursorShape.PointingHandCursor
except AttributeError:
    _PointingHandCursor = Qt.PointingHandCursor

try:
    _TextSelectableByMouse = Qt.TextInteractionFlag.TextSelectableByMouse
except AttributeError:
    _TextSelectableByMouse = Qt.TextSelectableByMouse

try:
    _FrameHLine = QFrame.Shape.HLine
except AttributeError:
    _FrameHLine = QFrame.HLine


class _AccountLoaderWorker(QThread):

    loaded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, client, auth):
        super().__init__()
        self._client = client
        self._auth = auth

    def run(self):
        result = self._client.get_account(auth=self._auth)
        if "error" in result:
            self.failed.emit(result.get("error", "Unknown error"))
        else:
            self.loaded.emit(result)


class AccountSettingsDialog(QDialog):

    change_key_requested = pyqtSignal()

    def __init__(self, client, auth, activation_key, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Account Settings"))
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setMaximumWidth(500)

        self._activation_key = activation_key
        self._key_visible = False
        self._worker = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 16, 16, 12)
        self._layout.setSpacing(12)

        self._loading_label = QLabel(tr("Loading account info..."))
        self._loading_label.setAlignment(_AlignCenter)
        self._loading_label.setStyleSheet("color: palette(text); padding: 16px;")
        self._layout.addWidget(self._loading_label)

        self._error_widget = QWidget()
        error_layout = QVBoxLayout(self._error_widget)
        error_layout.setContentsMargins(0, 0, 0, 0)
        error_layout.setSpacing(8)
        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setAlignment(_AlignCenter)
        self._error_label.setStyleSheet(f"color: {BRAND_RED}; padding: 12px;")
        error_layout.addWidget(self._error_label)
        self._retry_btn = QPushButton(tr("Retry"))
        self._retry_btn.setMaximumWidth(100)
        self._retry_btn.clicked.connect(self._fetch_account)
        retry_row = QHBoxLayout()
        retry_row.addStretch()
        retry_row.addWidget(self._retry_btn)
        retry_row.addStretch()
        error_layout.addLayout(retry_row)
        self._error_widget.setVisible(False)
        self._layout.addWidget(self._error_widget)

        self._content_widget = QWidget()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(10)
        self._content_widget.setVisible(False)
        self._layout.addWidget(self._content_widget)

        self._layout.addStretch()

        self._client = client
        self._auth = auth
        self._fetch_account()

    def _fetch_account(self):
        self._loading_label.setVisible(True)
        self._error_widget.setVisible(False)
        self._content_widget.setVisible(False)

        self._worker = _AccountLoaderWorker(self._client, self._auth)
        self._worker.loaded.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_loaded(self, data: dict):
        self._loading_label.setVisible(False)
        self._error_widget.setVisible(False)

        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._content_layout.addWidget(self._build_account_card(data))

        sub = self._find_subscription(data)
        if sub:
            self._content_layout.addWidget(self._build_subscription_card(sub))

        manage_label = QLabel(
            f'<a href="{_DASHBOARD_URL}" style="color: {BRAND_BLUE};">'
            f'{tr("Manage subscription on terra-lab.ai")}</a>'
        )
        manage_label.setOpenExternalLinks(True)
        manage_label.setAlignment(_AlignCenter)
        manage_label.setStyleSheet("font-size: 11px; padding-top: 2px;")
        self._content_layout.addWidget(manage_label)

        self._content_widget.setVisible(True)
        self.adjustSize()

    def _on_failed(self, message: str):
        self._loading_label.setVisible(False)
        self._error_label.setText(message)
        self._error_widget.setVisible(True)
        self._content_widget.setVisible(False)

    @staticmethod
    def _find_subscription(data: dict) -> dict | None:
        for s in data.get("subscriptions", []):
            if s.get("product_id") == PRODUCT_ID:
                return s
        return None

    def _build_account_card(self, data: dict) -> QFrame:
        card = QFrame()
        card.setStyleSheet(_CARD_STYLE)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        email_row = QHBoxLayout()
        email_row.setSpacing(4)
        email_lbl = QLabel(f"<b>{tr('Email')}</b>")
        email_lbl.setStyleSheet("font-size: 12px; color: palette(text);")
        email_row.addWidget(email_lbl)
        email_row.addStretch()
        email_val = QLabel(data.get("email", "\u2014"))
        email_val.setTextInteractionFlags(_TextSelectableByMouse)
        email_val.setStyleSheet("font-size: 12px; color: palette(text);")
        email_row.addWidget(email_val)
        email_copy = QPushButton(tr("Copy"))
        email_copy.setStyleSheet(_LINK_BTN)
        email_copy.setCursor(_PointingHandCursor)
        email_copy.clicked.connect(
            lambda: self._copy_text(data.get("email", ""), email_copy)
        )
        email_row.addWidget(email_copy)
        layout.addLayout(email_row)

        sep = QFrame()
        sep.setFrameShape(_FrameHLine)
        sep.setStyleSheet("color: rgba(128,128,128,0.2);")
        sep.setFixedHeight(1)
        layout.addWidget(sep)

        key_row = QHBoxLayout()
        key_row.setSpacing(4)
        key_lbl = QLabel(f"<b>{tr('Key')}</b>")
        key_lbl.setStyleSheet("font-size: 12px; color: palette(text);")
        key_row.addWidget(key_lbl)
        key_row.addStretch()
        self._key_label = QLabel(self._masked_key())
        self._key_label.setTextInteractionFlags(_TextSelectableByMouse)
        self._key_label.setStyleSheet(
            "font-size: 11px; font-family: monospace; color: palette(text);"
        )
        key_row.addWidget(self._key_label)
        self._toggle_btn = QPushButton(tr("Show"))
        self._toggle_btn.setStyleSheet(_LINK_BTN)
        self._toggle_btn.setCursor(_PointingHandCursor)
        self._toggle_btn.clicked.connect(self._toggle_key_visibility)
        key_row.addWidget(self._toggle_btn)
        self._copy_key_btn = QPushButton(tr("Copy"))
        self._copy_key_btn.setStyleSheet(_LINK_BTN)
        self._copy_key_btn.setCursor(_PointingHandCursor)
        self._copy_key_btn.clicked.connect(
            lambda: self._copy_text(self._activation_key, self._copy_key_btn)
        )
        key_row.addWidget(self._copy_key_btn)
        layout.addLayout(key_row)

        change_row = QHBoxLayout()
        change_row.addStretch()
        change_btn = QPushButton(tr("Change activation key"))
        change_btn.setStyleSheet(_LINK_BTN)
        change_btn.setCursor(_PointingHandCursor)
        change_btn.clicked.connect(self._on_change_key)
        change_row.addWidget(change_btn)
        layout.addLayout(change_row)

        return card

    def _build_subscription_card(self, sub: dict) -> QFrame:
        """AI Segmentation is free — show plan status only, no credits."""
        card = QFrame()
        card.setStyleSheet(_CARD_STYLE)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(6)

        title = QLabel(f"<b>{PRODUCT_NAME}</b>")
        title.setStyleSheet("font-size: 13px; color: palette(text);")
        card_layout.addWidget(title)

        status = sub.get("status", "active")

        grid = QGridLayout()
        grid.setContentsMargins(0, 2, 0, 0)
        grid.setSpacing(4)
        grid.setColumnMinimumWidth(0, 70)

        grid.addWidget(self._field_label(tr("Plan")), 0, 0)
        status_text, status_color = _STATUS_DISPLAY.get(
            status, (status.title(), BRAND_RED)
        )
        plan_status = QLabel(
            f"{tr('Free')} \u00b7 <span style='color:{status_color};'>{status_text}</span>"
        )
        plan_status.setStyleSheet("font-size: 12px; color: palette(text);")
        grid.addWidget(plan_status, 0, 1)

        card_layout.addLayout(grid)

        return card

    # -- Helpers --

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-size: 11px; color: palette(text);")
        return label

    def _masked_key(self) -> str:
        key = self._activation_key
        if len(key) <= 8:
            return key[:3] + "****"
        return key[:6] + "****" + key[-4:]

    def _toggle_key_visibility(self):
        self._key_visible = not self._key_visible
        if self._key_visible:
            self._key_label.setText(self._activation_key)
            self._toggle_btn.setText(tr("Hide"))
        else:
            self._key_label.setText(self._masked_key())
            self._toggle_btn.setText(tr("Show"))

    def _on_change_key(self):
        self.change_key_requested.emit()
        self.accept()

    def _copy_text(self, text: str, btn: QPushButton):
        QApplication.clipboard().setText(text)
        original = btn.text()
        btn.setText(tr("Copied!"))
        QTimer.singleShot(1500, lambda: btn.setText(original))

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
        super().closeEvent(event)
