"""
Activation dialog for the AI Segmentation plugin.
Shows during dependency installation and prompts user to get activation code.
"""

import os
from pathlib import Path

from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFrame,
    QSizePolicy,
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

    activated = pyqtSignal()  # Emitted when plugin is successfully activated

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Activate AI Segmentation")
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setMaximumWidth(500)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Logo section
        logo_label = QLabel()
        logo_path = Path(__file__).parent / "resources" / "icons" / "logo.png"
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            # Scale to reasonable size while maintaining aspect ratio
            scaled_pixmap = pixmap.scaledToWidth(180, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
        else:
            logo_label.setText("TerraLab")
            font = logo_label.font()
            font.setPointSize(18)
            font.setBold(True)
            logo_label.setFont(font)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # Title
        title_label = QLabel("Activate Your Plugin")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(
            "This plugin is currently in beta.\n\n"
            "Sign up to receive updates on new features\n"
            "and upcoming AI plugins for QGIS by TerraLab."
        )
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)

        # Get code button - opens newsletter page
        get_code_button = QPushButton("Get Your Free Activation Code")
        get_code_button.setMinimumHeight(40)
        get_code_button.setCursor(Qt.PointingHandCursor)
        get_code_button.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: bold; font-size: 12px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1b5e20; }"
        )
        get_code_button.clicked.connect(self._on_get_code_clicked)
        layout.addWidget(get_code_button)

        # Instruction for after getting code
        instruction_label = QLabel(
            "After signing up, you'll receive a code.\nEnter it below:"
        )
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setStyleSheet("color: palette(mid);")
        layout.addWidget(instruction_label)

        # Code input section
        code_layout = QHBoxLayout()
        code_layout.setSpacing(8)

        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText("Enter your activation code")
        self.code_input.setMinimumHeight(36)
        self.code_input.returnPressed.connect(self._on_activate_clicked)
        code_layout.addWidget(self.code_input)

        self.activate_button = QPushButton("Activate")
        self.activate_button.setMinimumHeight(36)
        self.activate_button.setMinimumWidth(80)
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
        layout.addWidget(self.message_label)

        # Footer with website link
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 8, 0, 0)

        footer_label = QLabel("Questions? Visit ")
        footer_label.setStyleSheet("color: palette(mid);")

        website_link = QLabel(f'<a href="{get_website_url()}">terra-lab.ai</a>')
        website_link.setOpenExternalLinks(True)
        website_link.setStyleSheet("color: #1976d2;")

        footer_layout.addStretch()
        footer_layout.addWidget(footer_label)
        footer_layout.addWidget(website_link)
        footer_layout.addStretch()

        layout.addLayout(footer_layout)

    def _on_get_code_clicked(self):
        """Open the newsletter signup page in the default browser."""
        QDesktopServices.openUrl(QUrl(get_newsletter_url()))

    def _on_activate_clicked(self):
        """Attempt to activate the plugin with the entered code."""
        code = self.code_input.text().strip()

        if not code:
            self._show_message("Please enter your activation code.", is_error=True)
            return

        success, message = activate_plugin(code)

        if success:
            self._show_message(message, is_error=False)
            self.activated.emit()
            # Close dialog after short delay to show success message
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(800, self.accept)
        else:
            self._show_message(message, is_error=True)
            self.code_input.selectAll()
            self.code_input.setFocus()

    def _show_message(self, text: str, is_error: bool = False):
        """Display a message to the user."""
        self.message_label.setText(text)
        if is_error:
            self.message_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
        else:
            self.message_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
        self.message_label.setVisible(True)
