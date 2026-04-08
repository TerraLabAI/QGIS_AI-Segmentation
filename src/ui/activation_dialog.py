"""
Activation dialog for the AI Segmentation plugin.
Shows during dependency installation and prompts user to get activation code.
"""

from pathlib import Path

from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFrame,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal, QUrl
from qgis.PyQt.QtGui import QPixmap, QDesktopServices, QFont

from ..core.activation_manager import (
    activate_plugin,
    get_shared_email,
    save_shared_email,
    get_newsletter_url_with_email,
)
from ..core.i18n import tr


class ActivationDialog(QDialog):
    """
    Modal dialog for plugin activation.
    Shows logo, explanation, and code input field.
    """

    activated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Setup AI Segmentation by TerraLab"))
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setMaximumWidth(500)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 24, 24, 24)

        # Banner section
        banner_label = QLabel()
        banner_path = Path(__file__).parent.parent.parent / "resources" / "icons" / "terralab-banner.png"
        if banner_path.exists():
            pixmap = QPixmap(str(banner_path))
            scaled_pixmap = pixmap.scaledToWidth(380, Qt.TransformationMode.SmoothTransformation)
            banner_label.setPixmap(scaled_pixmap)
        else:
            banner_label.setText("TerraLab")
            font = banner_label.font()
            font.setPointSize(18)
            font.setBold(True)
            banner_label.setFont(font)
        banner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(banner_label)

        # Title
        title_label = QLabel(tr("Unlock Plugin"))
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: palette(text);")
        layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel(tr("Enter your email to get a verification code."))
        subtitle_label.setWordWrap(True)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: palette(mid); font-size: 12px;")
        layout.addWidget(subtitle_label)

        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("your@email.com")
        self.email_input.setMinimumHeight(36)
        # Pre-fill from shared TerraLab email if available
        shared_email = get_shared_email()
        if shared_email:
            self.email_input.setText(shared_email)
        self.email_input.textChanged.connect(self._on_email_changed)
        layout.addWidget(self.email_input)

        # Get code button - disabled until email entered
        self.get_code_button = QPushButton(tr("Get my verification code"))
        self.get_code_button.setMinimumHeight(40)
        self.get_code_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.get_code_button.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: bold; font-size: 13px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1b5e20; }"
            "QPushButton:disabled { background-color: #b0bec5; }"
        )
        self.get_code_button.setEnabled(bool(shared_email and "@" in shared_email))
        self.get_code_button.clicked.connect(self._on_get_code_clicked)
        layout.addWidget(self.get_code_button)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        code_label = QLabel(tr("Then paste your code:"))
        code_label.setStyleSheet("color: palette(text); font-size: 12px;")
        layout.addWidget(code_label)

        # Code input section
        code_layout = QHBoxLayout()
        code_layout.setSpacing(8)

        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText(tr("Code"))
        self.code_input.setMinimumHeight(36)
        self.code_input.returnPressed.connect(self._on_activate_clicked)
        code_layout.addWidget(self.code_input)

        self.activate_button = QPushButton(tr("Unlock"))
        self.activate_button.setMinimumHeight(36)
        self.activate_button.setMinimumWidth(80)
        self.activate_button.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: white; "
            "font-weight: bold; font-size: 12px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1565c0; }"
            "QPushButton:disabled { background-color: #b0bec5; }"
        )
        self.activate_button.clicked.connect(self._on_activate_clicked)
        code_layout.addWidget(self.activate_button)

        layout.addLayout(code_layout)

        # Error/success message label
        self.message_label = QLabel("")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setWordWrap(True)
        self.message_label.setVisible(False)
        layout.addWidget(self.message_label)

    def _on_email_changed(self, text: str):
        """Enable/disable the get code button based on email validity."""
        self.get_code_button.setEnabled("@" in text.strip())

    def _on_get_code_clicked(self):
        """Save email and open the verification page with email pre-filled."""
        email = self.email_input.text().strip()
        if email and "@" in email:
            save_shared_email(email)
        url = get_newsletter_url_with_email(email)
        QDesktopServices.openUrl(QUrl(url))
        # Move focus to code input so Enter triggers Unlock, not this button
        self.code_input.setFocus()

    def _on_activate_clicked(self):
        """Attempt to activate the plugin with the entered code."""
        code = self.code_input.text().strip()

        if not code:
            self._show_message(tr("Enter your verification code"), is_error=True)
            return

        success, message = activate_plugin(code)

        if success:
            self._show_message(tr("Unlocked!"), is_error=False)
            self.activated.emit()
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(600, self.accept)
        else:
            self._show_message(tr("Invalid code"), is_error=True)
            self.code_input.selectAll()
            self.code_input.setFocus()

    def _show_message(self, text: str, is_error: bool = False):
        """Display a message to the user."""
        self.message_label.setText(text)
        if is_error:
            self.message_label.setStyleSheet("color: #ef5350; font-size: 12px;")
        else:
            self.message_label.setStyleSheet("color: #66bb6a; font-size: 12px;")
        self.message_label.setVisible(True)
