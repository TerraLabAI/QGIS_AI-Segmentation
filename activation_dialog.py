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

from .core.activation_manager import (
    activate_plugin,
    get_newsletter_url,
    get_website_url,
)


class ActivationDialog(QDialog):
    """
    Modal dialog for plugin activation.
    Shows logo, explanation, and code input field.
    """

    activated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quick Setup")
        self.setModal(True)
        self.setMinimumWidth(380)
        self.setMaximumWidth(450)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # Logo section
        logo_label = QLabel()
        logo_path = Path(__file__).parent / "resources" / "icons" / "logo.png"
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            scaled_pixmap = pixmap.scaledToWidth(160, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
        else:
            logo_label.setText("TerraLab")
            font = logo_label.font()
            font.setPointSize(16)
            font.setBold(True)
            logo_label.setFont(font)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        # Title - friendly and direct
        title_label = QLabel("Thanks for trying our plugin!")
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Description - authentic and minimal
        desc_label = QLabel(
            "We're a small team building open-source AI tools for QGIS.\n\n"
            "This is a beta version. Drop your email to get notified\n"
            "when we release updates and new plugins."
        )
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("color: palette(text); font-size: 11px;")
        layout.addWidget(desc_label)

        # Get code button
        get_code_button = QPushButton("Get my code")
        get_code_button.setMinimumHeight(36)
        get_code_button.setCursor(Qt.PointingHandCursor)
        get_code_button.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: bold; font-size: 12px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1b5e20; }"
        )
        get_code_button.clicked.connect(self._on_get_code_clicked)
        layout.addWidget(get_code_button)

        # Separator with text
        sep_layout = QHBoxLayout()
        sep_layout.setSpacing(8)
        left_line = QFrame()
        left_line.setFrameShape(QFrame.HLine)
        left_line.setFrameShadow(QFrame.Sunken)
        right_line = QFrame()
        right_line.setFrameShape(QFrame.HLine)
        right_line.setFrameShadow(QFrame.Sunken)
        then_label = QLabel("then")
        then_label.setStyleSheet("color: palette(mid); font-size: 10px;")
        sep_layout.addWidget(left_line, 1)
        sep_layout.addWidget(then_label)
        sep_layout.addWidget(right_line, 1)
        layout.addLayout(sep_layout)

        # Code input section
        code_layout = QHBoxLayout()
        code_layout.setSpacing(8)

        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText("Paste your code here")
        self.code_input.setMinimumHeight(34)
        self.code_input.returnPressed.connect(self._on_activate_clicked)
        code_layout.addWidget(self.code_input)

        self.activate_button = QPushButton("Unlock")
        self.activate_button.setMinimumHeight(34)
        self.activate_button.setMinimumWidth(70)
        self.activate_button.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: white; "
            "font-weight: bold; border-radius: 4px; }"
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
        self.message_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.message_label)

        # Footer note
        footer_label = QLabel("No spam, just updates. You can close this and enter the code later.")
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setWordWrap(True)
        footer_label.setStyleSheet("color: palette(mid); font-size: 10px;")
        layout.addWidget(footer_label)

    def _on_get_code_clicked(self):
        """Open the newsletter signup page in the default browser."""
        QDesktopServices.openUrl(QUrl(get_newsletter_url()))

    def _on_activate_clicked(self):
        """Attempt to activate the plugin with the entered code."""
        code = self.code_input.text().strip()

        if not code:
            self._show_message("Enter your code", is_error=True)
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
            self.message_label.setStyleSheet("color: #d32f2f; font-size: 11px;")
        else:
            self.message_label.setStyleSheet("color: #2e7d32; font-size: 11px;")
        self.message_label.setVisible(True)
