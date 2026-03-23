# SHARED MODULE v1.0 — keep in sync between AI Canvas and AI Segmentation
"""Shared activation dialog for TerraLab QGIS plugins."""

from qgis.PyQt.QtCore import Qt, QTimer, QUrl
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from .activation_manager import (
    activate_plugin,
    auto_activate_from_sibling,
    detect_sibling_activation,
    get_newsletter_url,
)
from .branding import (
    create_primary_button,
    create_secondary_button,
    create_terralab_title_bar,
    style_error_label,
    style_success_label,
)
from .constants import PRODUCTS


class ActivationDialog(QDialog):
    """Modal dialog for plugin activation.

    Checks for sibling activation first. If a sibling is already activated,
    auto-activates and shows a welcome message. Otherwise shows the
    email → code verification flow.
    """

    def __init__(self, parent, product_id: str, settings=None):
        super().__init__(parent)
        self._product_id = product_id
        self._settings = settings
        self._activated = False

        display_name = PRODUCTS[product_id]["display_name"]
        self.setWindowTitle(f"Unlock {display_name}")
        self.setMinimumWidth(320)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Check sibling activation first
        sibling = detect_sibling_activation(product_id, settings)
        if sibling:
            auto_activate_from_sibling(product_id, settings)
            self._activated = True
            sibling_name = PRODUCTS[sibling]["display_name"]

            welcome = QLabel(
                f"Welcome back! {display_name} has been automatically "
                f"unlocked because you already activated {sibling_name}."
            )
            welcome.setWordWrap(True)
            welcome.setAlignment(Qt.AlignCenter)
            layout.addWidget(welcome)

            close_btn = create_primary_button("Get Started")
            close_btn.clicked.connect(self.accept)
            layout.addWidget(close_btn)
            return

        # Title
        title = create_terralab_title_bar(display_name)
        layout.addWidget(title)

        # Step 1: Get verification code
        get_code_btn = create_primary_button("Get my verification code")
        get_code_btn.setCursor(Qt.PointingHandCursor)
        get_code_btn.clicked.connect(self._on_get_code)
        layout.addWidget(get_code_btn)

        # Step 2: Enter code
        paste_label = QLabel("Then paste your code:")
        paste_label.setStyleSheet("font-size: 11px; color: #666;")
        layout.addWidget(paste_label)

        code_row = QHBoxLayout()
        code_row.setSpacing(6)
        self._code_input = QLineEdit()
        self._code_input.setPlaceholderText("Code")
        self._code_input.setMinimumHeight(28)
        self._code_input.returnPressed.connect(self._on_unlock)
        code_row.addWidget(self._code_input)

        unlock_btn = create_secondary_button("Unlock")
        unlock_btn.setMinimumWidth(60)
        unlock_btn.clicked.connect(self._on_unlock)
        code_row.addWidget(unlock_btn)
        layout.addLayout(code_row)

        # Message label
        self._message = QLabel("")
        self._message.setAlignment(Qt.AlignCenter)
        self._message.setWordWrap(True)
        self._message.setVisible(False)
        layout.addWidget(self._message)

    @property
    def activated(self) -> bool:
        return self._activated

    def _on_get_code(self):
        url = get_newsletter_url(self._product_id)
        QDesktopServices.openUrl(QUrl(url))

    def _on_unlock(self):
        code = self._code_input.text().strip()
        if not code:
            return

        ok, msg = activate_plugin(self._product_id, code, self._settings)
        self._message.setVisible(True)
        if ok:
            self._activated = True
            style_success_label(self._message)
            self._message.setText("Plugin unlocked!")
            QTimer.singleShot(1500, self.accept)
        else:
            style_error_label(self._message)
            self._message.setText(msg)
