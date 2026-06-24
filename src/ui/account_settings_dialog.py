"""Account Settings dialog for AI Segmentation plugin.

Mirrors AI Edit's Settings: an account chip (avatar + email + status) with a
quiet inline Sign out link, a product card, a prominent Manage button, and a
discreet legal footer. The activation key lives only in the web dashboard, so
it is intentionally not shown here.
"""
from __future__ import annotations

import os

from qgis.PyQt.QtCore import Qt, QThread, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.activation_manager import (
    get_dashboard_url,
    get_privacy_url,
    get_terms_url,
)
from ..core.i18n import tr
from ..core.telemetry import is_telemetry_enabled, set_telemetry_enabled
from .ai_segmentation_dockwidget import (
    BRAND_BLUE,
    BRAND_BLUE_HOVER,
    BRAND_GREEN,
    BRAND_GREEN_TEXT,
    BRAND_RED,
)

PRODUCT_ID = "ai-segmentation"
PRODUCT_NAME = "AI Segmentation"

_STATUS_DISPLAY = {
    "active": (tr("Active"), BRAND_GREEN_TEXT),
    "trialing": (tr("Free Trial"), "#f57c00"),
    "canceled": (tr("Canceled"), BRAND_RED),
}

_LINK_BTN = (
    f"QPushButton {{ border: none; color: {BRAND_BLUE}; font-size: 11px;"
    f" text-decoration: underline; padding: 2px 4px; background: transparent; }}"
    f"QPushButton:hover {{ color: {BRAND_BLUE_HOVER}; }}"
)

# Prominent primary action: open the account dashboard on terra-lab.ai.
_MANAGE_BTN = (
    f"QPushButton {{ background-color: {BRAND_BLUE}; color: #ffffff;"
    f" border: none; border-radius: 8px; padding: 9px 16px;"
    f" font-size: 12px; font-weight: 600; }}"
    f"QPushButton:hover {{ background-color: {BRAND_BLUE_HOVER}; }}"
)

# Compact sign-out link (sits inside the account chip, not a full-width button).
_SIGNOUT_LINK = (
    "QPushButton { border: none; background: transparent; color: palette(text);"
    " font-size: 11px; text-decoration: underline; padding: 2px 4px; }"
    f"QPushButton:hover {{ color: {BRAND_RED}; }}"
)

# Small outlined secondary button for minor actions inside a card (e.g. "Open
# folder"), distinct from the prominent blue Manage button.
_SECONDARY_BTN = (
    "QPushButton { background: palette(base); color: palette(text);"
    " border: 1px solid rgba(128,128,128,0.35); border-radius: 6px;"
    " padding: 5px 12px; font-size: 11px; }"
    "QPushButton:hover { border-color: rgba(128,128,128,0.65); }"
)

_CARD_STYLE = (
    "QFrame { background: rgba(128,128,128,0.08);"
    " border: 1px solid rgba(128,128,128,0.2);"
    " border-radius: 6px; }"
    "QLabel { background: transparent; border: none; }"
    "QPushButton { background: transparent; }"
)


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

    sign_out_requested = pyqtSignal()

    def __init__(self, client, auth, activation_key, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Account Settings"))
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setMaximumWidth(500)

        # ``activation_key`` is accepted for signature compatibility but no
        # longer displayed; key management happens on the web dashboard.
        del activation_key
        self._worker = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 16, 16, 12)
        self._layout.setSpacing(12)

        self._loading_label = QLabel(tr("Loading account info..."))
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_label.setStyleSheet("color: palette(text); padding: 16px;")
        self._layout.addWidget(self._loading_label)

        self._error_widget = QWidget()
        error_layout = QVBoxLayout(self._error_widget)
        error_layout.setContentsMargins(0, 0, 0, 0)
        error_layout.setSpacing(8)
        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._error_label.setStyleSheet(f"color: {BRAND_RED}; padding: 12px;")
        error_layout.addWidget(self._error_label)
        self._retry_btn = QPushButton(tr("Retry"))
        self._retry_btn.setMaximumWidth(100)
        self._retry_btn.clicked.connect(self._fetch_account)
        self._error_sign_out_btn = QPushButton(tr("Sign out"))
        self._error_sign_out_btn.setStyleSheet(_LINK_BTN)
        self._error_sign_out_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._error_sign_out_btn.clicked.connect(self._on_sign_out)
        retry_row = QHBoxLayout()
        retry_row.addStretch()
        retry_row.addWidget(self._retry_btn)
        retry_row.addStretch()
        error_layout.addLayout(retry_row)
        signout_row = QHBoxLayout()
        signout_row.addStretch()
        signout_row.addWidget(self._error_sign_out_btn)
        signout_row.addStretch()
        error_layout.addLayout(signout_row)
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

        self._content_layout.addWidget(self._build_dependencies_card())

        self._content_layout.addWidget(self._build_privacy_card())

        # Discreet footer: thin top separator, small muted Terms / Privacy links.
        footer = QFrame()
        footer.setObjectName("legalFooter")
        footer.setStyleSheet(
            "QFrame#legalFooter { border: none;"
            " border-top: 1px solid rgba(127,127,127,0.18); }"
        )
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 8, 0, 0)
        footer_layout.setSpacing(0)
        footer_layout.addStretch()
        legal_label = QLabel(
            f'<a href="{get_terms_url()}" style="color: rgba(128,128,128,0.85);'
            f' text-decoration: none;">{tr("Terms")}</a>'
            f' <span style="color: rgba(128,128,128,0.5);">·</span> '
            f'<a href="{get_privacy_url()}" style="color: rgba(128,128,128,0.85);'
            f' text-decoration: none;">{tr("Privacy")}</a>'
        )
        legal_label.setOpenExternalLinks(True)
        legal_label.setStyleSheet("font-size: 10px;")
        footer_layout.addWidget(legal_label)
        footer_layout.addStretch()
        self._content_layout.addWidget(footer)

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
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        email = data.get("email", "-")

        # A real account "chip": round avatar (first letter) + email + status.
        chip = QFrame()
        chip.setStyleSheet(
            "QFrame { background: palette(base);"
            " border: 1px solid rgba(128,128,128,0.25); border-radius: 8px; }"
            "QLabel { background: transparent; border: none; }"
        )
        chip_row = QHBoxLayout(chip)
        chip_row.setContentsMargins(12, 10, 12, 10)
        chip_row.setSpacing(11)

        avatar = QLabel(email[:1].upper() if email and email != "-" else "?")
        avatar.setFixedSize(38, 38)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar.setStyleSheet(
            f"background: {BRAND_GREEN}; color: #14210A; border-radius: 19px;"
            " font-size: 17px; font-weight: 700;"
        )
        chip_row.addWidget(avatar)

        id_col = QVBoxLayout()
        id_col.setSpacing(2)
        email_val = QLabel(email)
        email_val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        email_val.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: palette(text);"
        )
        id_col.addWidget(email_val)
        status_lbl = QLabel("✓ " + tr("Connected"))
        status_lbl.setStyleSheet(
            f"font-size: 11px; font-weight: 600; color: {BRAND_GREEN_TEXT};"
        )
        id_col.addWidget(status_lbl)
        chip_row.addLayout(id_col, 1)

        # Sign out is a quiet inline link on the right of the chip, not a heavy
        # full-width button below the email.
        sign_out_btn = QPushButton(tr("Sign out"))
        sign_out_btn.setStyleSheet(_SIGNOUT_LINK)
        sign_out_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        sign_out_btn.clicked.connect(self._on_sign_out)
        chip_row.addWidget(sign_out_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(chip)
        return card

    def _open_dashboard(self):
        QDesktopServices.openUrl(QUrl(get_dashboard_url()))

    def _build_subscription_card(self, sub: dict) -> QFrame:
        """AI Segmentation is free - show plan status, no credits."""
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
            f"{tr('Free')} · <span style='color:{status_color};'>{status_text}</span>"
        )
        plan_status.setStyleSheet("font-size: 12px; color: palette(text);")
        grid.addWidget(plan_status, 0, 1)

        card_layout.addLayout(grid)
        card_layout.addSpacing(4)

        # Account management lives on the website (terra-lab.ai); the plugin
        # only points there. Keeps billing and key handling out of the plugin.
        manage_btn = QPushButton(tr("Manage account on terra-lab.ai") + "  ↗")
        manage_btn.setStyleSheet(_MANAGE_BTN)
        manage_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        manage_btn.setMinimumHeight(36)
        manage_btn.clicked.connect(self._open_dashboard)
        card_layout.addWidget(manage_btn)

        return card

    def _build_privacy_card(self) -> QFrame:
        """Anonymous usage telemetry with a clear, ON-by-default opt-out.

        QGIS users expect data collection to be transparent and switchable. The
        checkbox flips the shared TerraLab/telemetry_enabled flag (see
        telemetry.py), so turning it off here also silences the sibling AI Edit
        plugin. The metrics are anonymous (no email, no identifier that singles a
        user out) and carry no geospatial or project content.
        """
        card = QFrame()
        card.setStyleSheet(_CARD_STYLE)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        title = QLabel(f"<b>{tr('Privacy')}</b>")
        title.setStyleSheet("font-size: 13px; color: palette(text);")
        layout.addWidget(title)

        self._telemetry_checkbox = QCheckBox(tr("Share anonymous usage statistics"))
        self._telemetry_checkbox.setChecked(is_telemetry_enabled())
        self._telemetry_checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
        self._telemetry_checkbox.setStyleSheet(
            "font-size: 12px; color: palette(text);"
        )
        self._telemetry_checkbox.toggled.connect(set_telemetry_enabled)
        layout.addWidget(self._telemetry_checkbox)

        caption = QLabel(
            tr(
                "Anonymous metrics (durations, error codes, OS, QGIS version) "
                "help us fix issues."
            )
        )
        caption.setWordWrap(True)
        caption.setStyleSheet("font-size: 11px; color: palette(text);")
        layout.addWidget(caption)

        return card

    def _build_dependencies_card(self) -> QFrame:
        """Local AI dependencies: where they live, how big, and an Open button.

        The isolated venv and model weights (~1 GB) live outside QGIS under the
        user home. Surfacing the path + size + an Open button answers the common
        support needs (disk space, antivirus exclusions, manual cleanup) without
        the user having to hunt for a hidden folder.
        """
        from ..core.venv_manager import CACHE_DIR, VENV_DIR

        card = QFrame()
        card.setStyleSheet(_CARD_STYLE)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        title = QLabel(f"<b>{tr('Dependencies')}</b>")
        title.setStyleSheet("font-size: 13px; color: palette(text);")
        layout.addWidget(title)

        caption = QLabel(tr("Local AI model files stored on this computer."))
        caption.setWordWrap(True)
        caption.setStyleSheet("font-size: 11px; color: palette(text);")
        layout.addWidget(caption)

        installed = os.path.isdir(VENV_DIR)
        size_text = self._format_dir_size(CACHE_DIR) if installed else tr("Not installed")
        info = QLabel(f"{tr('On disk')}: {size_text}")
        info.setStyleSheet("font-size: 11px; color: rgba(128,128,128,0.9);")
        layout.addWidget(info)

        path_lbl = QLabel(CACHE_DIR)
        path_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        path_lbl.setWordWrap(True)
        # 11px + palette(text): the 10px muted grey was hard to read on dark.
        path_lbl.setStyleSheet("font-size: 11px; color: palette(text);")
        layout.addWidget(path_lbl)

        open_btn = QPushButton(tr("Open folder"))
        open_btn.setStyleSheet(_SECONDARY_BTN)
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.clicked.connect(lambda: self._open_install_folder(CACHE_DIR))
        row = QHBoxLayout()
        row.setContentsMargins(0, 2, 0, 0)
        row.addStretch()
        row.addWidget(open_btn)
        layout.addLayout(row)

        return card

    @staticmethod
    def _open_install_folder(path: str):
        # The folder may not exist yet (deps never installed); fall back to the
        # nearest existing parent so the button never silently does nothing.
        target = path
        while target and not os.path.isdir(target):
            parent = os.path.dirname(target)
            if parent == target:
                break
            target = parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    @staticmethod
    def _format_dir_size(path: str) -> str:
        total = 0
        try:
            for root, _dirs, files in os.walk(path):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:  # nosec B112
                        continue
        except OSError:
            return "-"
        mb = total / (1024 * 1024)
        if mb >= 1024:
            return f"{mb / 1024:.1f} GB"
        return f"{mb:.0f} MB"

    # -- Helpers --

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-size: 11px; color: palette(text);")
        return label

    def _on_sign_out(self):
        from qgis.PyQt.QtWidgets import QMessageBox

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(tr("Sign out"))
        box.setText(tr("Sign out of AI Segmentation?"))
        box.setInformativeText(tr("You can sign back in anytime from QGIS."))
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        box.setDefaultButton(QMessageBox.StandardButton.No)
        if box.exec() != QMessageBox.StandardButton.Yes:
            return
        self.sign_out_requested.emit()
        self.accept()

    def closeEvent(self, event):
        # The account loader thread has no event loop, so quit() is a no-op.
        # Wait gives the in-flight network call room to return; terminate is
        # the last-resort exit when the network stack itself is wedged,
        # otherwise QThread destruction would crash QGIS.
        if self._worker and self._worker.isRunning():
            if not self._worker.wait(6000):
                self._worker.terminate()
                self._worker.wait(1000)
        super().closeEvent(event)
