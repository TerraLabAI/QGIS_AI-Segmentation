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
    get_newsletter_url,
)


class ActivationDialog(QDialog):
    """
    Modal dialog for plugin activation.
    Shows logo, explanation, and code input field.
    """

    activated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup AI Segmentation by TerraLab")
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setMaximumWidth(500)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 24, 24, 24)

        # Banner section - using the TerraLab banner
        banner_label = QLabel()
        banner_path = Path(__file__).parent.parent.parent / "resources" / "icons" / "terralab-banner.png"
        if banner_path.exists():
            pixmap = QPixmap(str(banner_path))
            scaled_pixmap = pixmap.scaledToWidth(380, Qt.SmoothTransformation)
            banner_label.setPixmap(scaled_pixmap)
        else:
            banner_label.setText("TerraLab")
            font = banner_label.font()
            font.setPointSize(18)
            font.setBold(True)
            banner_label.setFont(font)
        banner_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(banner_label)

        # Title - friendly (uses palette for theme compatibility)
        title_label = QLabel("Thanks for trying our plugin!")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: palette(text);")
        layout.addWidget(title_label)

        # Description - clear about the email/code relationship
        desc_label = QLabel(
            "This plugin is in beta. We'd love to keep you updated\n"
            "when we release new versions and features."
        )
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        desc_font = QFont()
        desc_font.setPointSize(12)
        desc_label.setFont(desc_font)
        desc_label.setStyleSheet("color: palette(text);")
        layout.addWidget(desc_label)

        # Clear instruction about the flow
        flow_label = QLabel(
            "Enter your email and you'll get a<br>"
            "<b>Verification Code</b> to paste below."
        )
        flow_label.setWordWrap(True)
        flow_label.setAlignment(Qt.AlignCenter)
        flow_font = QFont()
        flow_font.setPointSize(12)
        flow_label.setFont(flow_font)
        flow_label.setStyleSheet("color: palette(text);")
        layout.addWidget(flow_label)

        # Get code button - clearer label
        get_code_button = QPushButton("Get my verification code")
        get_code_button.setMinimumHeight(40)
        get_code_button.setCursor(Qt.PointingHandCursor)
        get_code_button.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: bold; font-size: 13px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1b5e20; }"
        )
        get_code_button.clicked.connect(self._on_get_code_clicked)
        layout.addWidget(get_code_button)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # Code input label (uses palette for theme compatibility)
        code_label = QLabel("Paste your verification code:")
        code_label.setAlignment(Qt.AlignLeft)
        code_font = QFont()
        code_font.setPointSize(11)
        code_label.setFont(code_font)
        code_label.setStyleSheet("color: palette(text);")
        layout.addWidget(code_label)

        # Code input section
        code_layout = QHBoxLayout()
        code_layout.setSpacing(8)

        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText("Code")
        self.code_input.setMinimumHeight(36)
        self.code_input.returnPressed.connect(self._on_activate_clicked)
        code_layout.addWidget(self.code_input)

        self.activate_button = QPushButton("Unlock")
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
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setWordWrap(True)
        self.message_label.setVisible(False)
        layout.addWidget(self.message_label)

    def _on_get_code_clicked(self):
        """Open the newsletter signup page in the default browser."""
        QDesktopServices.openUrl(QUrl(get_newsletter_url()))
        # Move focus to code input so Enter triggers Unlock, not this button
        self.code_input.setFocus()

    def _on_activate_clicked(self):
        """Attempt to activate the plugin with the entered code."""
        code = self.code_input.text().strip()

        if not code:
            self._show_message("Enter your verification code", is_error=True)
            return

        success, message = activate_plugin(code)

        if success:
            self._show_message("Unlocked!", is_error=False)
            self.activated.emit()
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(600, self.accept)
        else:
            self._show_message("Invalid code", is_error=True)
            self.code_input.selectAll()
            self.code_input.setFocus()

    def _show_message(self, text: str, is_error: bool = False):
        """Display a message to the user."""
        self.message_label.setText(text)
        if is_error:
            self.message_label.setStyleSheet("color: #d32f2f; font-size: 12px;")
        else:
            self.message_label.setStyleSheet("color: #2e7d32; font-size: 12px;")
        self.message_label.setVisible(True)
