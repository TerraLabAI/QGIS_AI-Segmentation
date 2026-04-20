"""TerraLab sign-in dialog (v1.0.0 — paste key flow, #24).

Opens the TerraLab website in the browser so the user can sign in and
copy their activation key, then paste it back into the plugin.
"""
from pathlib import Path

from qgis.PyQt.QtCore import Qt, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices, QFont, QPixmap
from qgis.PyQt.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ..core.activation_manager import get_sign_in_url, validate_key_with_server
from ..core.i18n import tr


class ActivationDialog(QDialog):
    """Modal dialog: sign in on website, paste activation key."""

    activated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Sign in to TerraLab"))
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setMaximumWidth(480)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 24, 24, 24)

        banner_label = QLabel()
        banner_path = (
            Path(__file__).parent.parent.parent / "resources" / "icons" / "terralab-banner.png"
        )
        if banner_path.exists():
            pixmap = QPixmap(str(banner_path))
            scaled = pixmap.scaledToWidth(340, Qt.TransformationMode.SmoothTransformation)
            banner_label.setPixmap(scaled)
        else:
            banner_label.setText("TerraLab")
            font = banner_label.font()
            font.setPointSize(18)
            font.setBold(True)
            banner_label.setFont(font)
        banner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(banner_label)

        title = QLabel(tr("Sign in to get your key"))
        tfont = QFont()
        tfont.setPointSize(14)
        tfont.setBold(True)
        title.setFont(tfont)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: palette(text);")
        layout.addWidget(title)

        subtitle = QLabel(tr(
            "Create your free TerraLab account or sign in, "
            "then copy your activation key from the dashboard."))
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: palette(text); font-size: 11px;")
        layout.addWidget(subtitle)

        self.sign_in_button = QPushButton(tr("Sign in to TerraLab (free)"))
        self.sign_in_button.setMinimumHeight(40)
        self.sign_in_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sign_in_button.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: bold; font-size: 13px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1b5e20; }"
        )
        self.sign_in_button.clicked.connect(self._on_sign_in_clicked)
        layout.addWidget(self.sign_in_button)

        key_row = QHBoxLayout()
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("tl_...")
        self.key_input.setMinimumHeight(36)
        self.key_input.returnPressed.connect(self._on_activate_clicked)
        key_row.addWidget(self.key_input)

        self.activate_button = QPushButton(tr("Activate"))
        self.activate_button.setMinimumHeight(36)
        self.activate_button.setMinimumWidth(100)
        self.activate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.activate_button.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: white; "
            "font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1565c0; }"
            "QPushButton:disabled { background-color: #b0bec5; }"
        )
        self.activate_button.clicked.connect(self._on_activate_clicked)
        key_row.addWidget(self.activate_button)
        layout.addLayout(key_row)

        self.message_label = QLabel("")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setWordWrap(True)
        self.message_label.setVisible(False)
        layout.addWidget(self.message_label)

    def _on_sign_in_clicked(self):
        QDesktopServices.openUrl(QUrl(get_sign_in_url()))

    def _on_activate_clicked(self):
        key = self.key_input.text().strip()
        if not key:
            self._show_message(tr("Please enter your activation key."), is_error=True)
            return

        self.activate_button.setEnabled(False)
        self.activate_button.setText(tr("Checking..."))

        success, msg = validate_key_with_server(key)

        self.activate_button.setText(tr("Activate"))
        self.activate_button.setEnabled(True)

        if success:
            self._show_message(tr(msg), is_error=False)
            self.activated.emit()
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(600, self.accept)
        else:
            self._show_message(tr(msg), is_error=True)

    def _show_message(self, text: str, is_error: bool = False):
        self.message_label.setText(text)
        if is_error:
            self.message_label.setStyleSheet("color: #ef5350; font-size: 12px;")
        else:
            self.message_label.setStyleSheet("color: #66bb6a; font-size: 12px;")
        self.message_label.setVisible(True)
