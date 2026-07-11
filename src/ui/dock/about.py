"""Update notification, about section, tutorial/report/shortcuts/contact.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QStyle,
    QToolButton,
    QWidget,
)


from ..credit_ring import CreditRing

from ...core.activation_manager import (
    get_tutorial_url,
)
from ...core.i18n import tr
from .styles import (
    BRAND_BLUE,
    BRAND_BLUE_HOVER,
    _BTN_BLUE,
    _BTN_GREEN,
    _FOOTER_CTA_BTN_STYLE,
    _FOOTER_ICON_BTN_STYLE,
    _FOOTER_MENU_STYLE,
    _HELP_ICON_BTN_STYLE,
    _msg_card_qss,
    _msg_label_qss,
)
from .widgets import (
    _FooterIconButton,
)


class DockAboutMixin:
    """Update notification, about section, tutorial/report/shortcuts/contact."""

    def _setup_update_notification(self):
        """Setup the update notification label (hidden by default)."""
        # Container just for right-alignment
        self._update_notif_container = QWidget()
        container_layout = QHBoxLayout(self._update_notif_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addStretch()

        self.update_notification_label = QLabel("")
        self.update_notification_label.setStyleSheet(_msg_label_qss("info"))
        self.update_notification_label.setOpenExternalLinks(False)
        self.update_notification_label.linkActivated.connect(
            self._on_open_plugin_manager)
        container_layout.addWidget(self.update_notification_label)

        self._update_notif_container.setVisible(False)
        self.update_notification_widget = self._update_notif_container

        self.main_layout.addWidget(self.update_notification_widget)

    def check_for_updates(self):
        """Check if a newer version is available in the QGIS plugin repository."""
        try:
            from pyplugin_installer.installer_data import plugins
            plugin_data = plugins.all().get("AI_Segmentation")
            if plugin_data and plugin_data.get("status") == "upgradeable":
                available_version = plugin_data.get(
                    "version_available", "?")
                message = tr("Version {version} is available.").format(
                    version=available_version)
                link_text = tr("Update now")
                text = (
                    f'{message} <a href="#update" style="color: {BRAND_BLUE};'
                    f' font-weight: bold;">{link_text}</a>'
                )
                self.update_notification_label.setText(text)
                self.update_notification_widget.setVisible(True)
        except Exception:
            pass  # nosec B110  No repo data yet, dev install, etc.

    def _on_open_plugin_manager(self, _link=None):
        """Open the QGIS Plugin Manager on the Upgradeable tab (index 3)."""
        try:
            from qgis.utils import iface
            iface.pluginManagerInterface().showPluginManager(3)
        except Exception:
            pass  # nosec B110

    def _setup_about_section(self):
        """Setup the info box and links section."""
        # Info box for segmentation mode (subtle blue style)
        self.batch_info_widget = QWidget()
        self.batch_info_widget.setObjectName("batchInfoCard")
        self.batch_info_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.batch_info_widget.setStyleSheet(_msg_card_qss("batchInfoCard", "info"))
        batch_info_layout = QHBoxLayout(self.batch_info_widget)
        batch_info_layout.setContentsMargins(8, 6, 8, 6)
        batch_info_layout.setSpacing(8)

        batch_info_icon = QLabel()
        style = self.batch_info_widget.style()
        _ico = style.pixelMetric(QStyle.PixelMetric.PM_SmallIconSize)
        batch_icon = style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
        batch_info_icon.setPixmap(batch_icon.pixmap(_ico, _ico))
        batch_info_icon.setFixedSize(_ico, _ico)
        batch_info_layout.addWidget(batch_info_icon, 0, Qt.AlignmentFlag.AlignTop)

        info_msg = "{}\n{}".format(
            tr("The AI model works best on one element at a time."),
            tr("Save your polygon before selecting the next element."))
        batch_info_text = QLabel(info_msg)
        batch_info_text.setWordWrap(True)
        batch_info_text.setStyleSheet("font-size: 11px; color: palette(text);")
        batch_info_layout.addWidget(batch_info_text, 1)

        self.batch_info_widget.setVisible(False)
        self.main_layout.addWidget(self.batch_info_widget)

        # Footer icon row - mirrors AI Edit. Gear opens Account Settings
        # (visible only when activated), help opens a popup with Tutorial /
        # Shortcuts / Contact us. The contact / tutorial / shortcuts links
        # previously sat as blue underlined labels but moved into the help
        # menu so the bar matches AI Edit's compact look.
        footer_widget = QWidget()
        footer_row = QHBoxLayout(footer_widget)
        footer_row.setContentsMargins(0, 4, 0, 4)
        footer_row.setSpacing(6)

        # Automatic mode only: compact credit gauge (ring + "remaining / total")
        # plus a discreet Subscribe pill, bottom-left like AI Edit. Replaces
        # the old always-on upsell card that ate half the Automatic page.
        self._credit_ring = CreditRing(diameter=16, parent=footer_widget)
        self._credit_ring.setVisible(False)
        footer_row.addWidget(self._credit_ring)

        self._footer_credits_label = QLabel()
        self._footer_credits_label.setStyleSheet(
            "QLabel { font-size: 11px; color: palette(text);"
            " background: transparent; border: none; }"
        )
        self._footer_credits_label.setVisible(False)
        footer_row.addWidget(self._footer_credits_label)

        self._subscribe_pill = QPushButton(tr("Subscribe to Pro"))
        self._subscribe_pill.setToolTip(tr("Upgrade to Pro"))
        self._subscribe_pill.setCursor(Qt.CursorShape.PointingHandCursor)
        # Filled brand-blue pill (stronger than the old ghost outline): white
        # text on a solid blue, lighter blue on hover. Kept small.
        self._subscribe_pill.setStyleSheet(
            f"QPushButton {{ border: none; color: #ffffff;"  # ui-ok: footer pill shape, documented one-off
            f" background: {BRAND_BLUE}; border-radius: 8px; padding: 2px 10px;"
            f" font-size: 11px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: {BRAND_BLUE_HOVER}; }}"
        )
        self._subscribe_pill.clicked.connect(self._on_upgrade_clicked)
        self._subscribe_pill.setVisible(False)
        footer_row.addWidget(self._subscribe_pill)

        # Cross-promo CTA, pinned bottom-left (before the stretch) so it sits
        # beside the gear/help icons without crowding them (#30). Always opens
        # the AI Edit product page in the browser.
        from ..cross_plugin_discovery import open_ai_edit_page
        self._ai_edit_btn = _FooterIconButton(footer_widget)
        # Decorative glyph kept out of the translatable string. The copy sells
        # AI Edit's promise (presentation and planning visuals) and deliberately
        # stays off AI Segmentation Pro's turf (no segmentation wording).
        self._ai_edit_btn.setText("🍌 " + tr("Make this map presentation-ready"))
        self._ai_edit_btn.setToolTip(tr(
            "AI Edit: turn your imagery into presentation and planning visuals"))
        self._ai_edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ai_edit_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._ai_edit_btn.setStyleSheet(_FOOTER_CTA_BTN_STYLE)
        self._ai_edit_btn.clicked.connect(lambda: open_ai_edit_page())
        footer_row.addWidget(self._ai_edit_btn)

        footer_row.addStretch()

        # Dedicated tutorial button: its own glyph next to the
        # gear and help, always visible (signed-in and signed-out), so a lost
        # user can reach the step-by-step guide from anywhere. Order: tutorial,
        # gear, help. Green hover groups it with the help "?" as a learn action.
        self._tutorial_btn = _FooterIconButton(footer_widget)
        self._tutorial_btn.setText("📖")  # U+1F4D6 OPEN BOOK
        self._tutorial_btn.setToolTip(tr("Open the step-by-step tutorial"))
        self._tutorial_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._tutorial_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._tutorial_btn.setStyleSheet(_HELP_ICON_BTN_STYLE)
        self._tutorial_btn.clicked.connect(self._on_open_guide_footer)
        footer_row.addWidget(self._tutorial_btn)

        self._settings_btn = _FooterIconButton(footer_widget)
        self._settings_btn.setText("⚙")  # U+2699 GEAR
        self._settings_btn.setToolTip(tr("Settings"))
        self._settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._settings_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._settings_btn.setStyleSheet(_FOOTER_ICON_BTN_STYLE)
        self._settings_btn.clicked.connect(lambda: self.settings_clicked.emit())
        self._settings_btn.setVisible(False)  # shown when activated
        footer_row.addWidget(self._settings_btn)

        self._help_btn = _FooterIconButton(footer_widget)
        self._help_btn.setText("?")
        self._help_btn.setToolTip(tr("Help / Report a problem"))
        self._help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._help_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._help_btn.setStyleSheet(_HELP_ICON_BTN_STYLE)
        self._help_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        help_menu = QMenu(self._help_btn)
        help_menu.setStyleSheet(_FOOTER_MENU_STYLE)
        help_menu.addAction(tr("Tutorial"), self._on_open_tutorial)
        help_menu.addAction(tr("Keyboard shortcuts"), self._on_show_shortcuts)
        help_menu.addAction(tr("Contact us"), self._on_contact_us)
        help_menu.addAction(tr("Report a problem"), self._on_report_problem)
        self._help_btn.setMenu(help_menu)
        # Force the hover tint off when the popup closes - Qt does not
        # synthesise a Leave event in this case. The green active tint stays
        # lit while the menu is open (mirrors AI Edit's footer buttons).
        help_menu.aboutToShow.connect(
            lambda btn=self._help_btn: btn.set_active(True)
        )
        help_menu.aboutToHide.connect(
            lambda btn=self._help_btn: (
                btn.setDown(False), btn.set_hovered(False), btn.set_active(False))
        )
        footer_row.addWidget(self._help_btn)

        self.main_layout.addWidget(footer_widget)

    def _on_open_tutorial(self):
        """Open the tutorial URL in the system browser."""
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(get_tutorial_url()))

    def _on_open_guide_footer(self):
        """Footer book button: open the step-by-step written guide."""
        from .guidance import open_guide
        open_guide("footer_tutorial")

    def _on_report_problem(self, _link=None):
        """User-initiated report: open the log-report dialog (collects the
        session logs and pre-fills the support email)."""
        from ..error_report_dialog import show_error_report
        show_error_report(
            self,
            tr("Report a problem"),
            "",
            error_code="user_reported",
        )

    def _on_show_shortcuts(self):
        """Keyboard shortcuts dialog: the full plugin keyboard map (K4)."""
        from qgis.PyQt.QtGui import QKeySequence
        from qgis.PyQt.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

        def native(seq) -> str:
            """Platform-native rendering of a key sequence (mirrors AI Edit):
            macOS shows symbols (⌘Z, ⌫, ⎋), Windows/Linux shows Ctrl+Z etc.
            Qt maps "Ctrl" to Cmd on macOS automatically."""
            return QKeySequence(seq).toString(
                QKeySequence.SequenceFormat.NativeText)

        undo_key = native(QKeySequence.StandardKey.Undo)
        backspace_key = native("Backspace")
        enter_key = native("Return")
        esc_key = native("Esc")
        # Delete the active object: Delete, or Ctrl/Cmd+Backspace (the big
        # key on Mac keyboards); matches shortcut_filter.py.
        delete_key = f"{native('Del')} / {native('Ctrl+Backspace')}"

        key_style = (
            "background-color: rgba(128,128,128,0.18);"
            "border: 1px solid rgba(128,128,128,0.35);"
            "border-radius: 3px;"
            "padding: 1px 5px;"
            "font-family: monospace;"
        )
        k = f"<span style='{key_style}'>{{}}</span>"

        def _row(key_html: str, action: str) -> str:
            return ("<tr><td style='padding-right:12px;'>"
                    f"{key_html}</td><td>{action}</td></tr>")

        def _section(title: str) -> str:
            return ("<tr><td colspan='2' style='padding-top:8px;"
                    f"padding-bottom:1px;'><b>{title}</b></td></tr>")

        rows = [
            "<table cellspacing='0' cellpadding='1'>",
            _section(tr("General")),
            _row(k.format("G"), tr("Start (the visible mode's Start button)")),
            _section(tr("Automatic - draw your zone")),
            _row(k.format(tr("Click")), tr("Add a point")),
            _row(k.format(tr("Double-click") + f" / {enter_key}"),
                 tr("Finish the zone")),
            _row(k.format(f"{backspace_key} / {undo_key}"), tr("Undo the last point")),
            _row(k.format(esc_key), tr("Cancel the drawing")),
            _section(tr("Automatic - detect and review")),
            _row(k.format(enter_key),
                 tr("Detect objects, or export the reviewed polygons")),
            _row(k.format(esc_key),
                 tr("Cancel the running detection, or exit the review")),
            _section(tr("Manual session")),
            _row(k.format(tr("Left-click")), tr("Add area")),
            _row(k.format(tr("Right-click")), tr("Remove area")),
            _row(k.format("S"), tr("Save polygon")),
            _row(k.format(f"{undo_key} / {backspace_key}"), tr("Undo last point")),
            _row(k.format(enter_key), tr("Export polygon to a layer")),
            _row(k.format(esc_key), tr("Stop segmentation")),
            _row(k.format(delete_key), tr("Delete the active object")),
            _section(tr("Navigation")),
            _row(k.format(tr("Space")), tr("Hold and move to pan the map")),
            _row(k.format(tr("Arrow keys")), tr("Pan the map")),
            _row(k.format(tr("Middle mouse button")),
                 tr("Click and drag to pan the map")),
            "</table>",
        ]
        shortcuts_html = "".join(rows)

        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Keyboard shortcuts"))
        dlg.setMaximumWidth(460)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)
        label = QLabel(shortcuts_html)
        label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(label)
        ok_btn = QPushButton(tr("OK"))
        ok_btn.setStyleSheet(_BTN_BLUE)
        ok_btn.setFixedWidth(80)
        ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        ok_btn.clicked.connect(dlg.accept)
        layout.addWidget(ok_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        dlg.exec()

    def _on_contact_us(self, _link=None):
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        from qgis.PyQt.QtWidgets import QApplication, QDialog
        from qgis.PyQt.QtWidgets import QVBoxLayout as _VBox

        calendly_url = "https://calendly.com/barbot-yvann/30min"
        support_email = "yvann.barbot@terra-lab.ai"

        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Contact us"))
        dlg.setMinimumWidth(350)
        dlg.setMaximumWidth(450)
        lay = _VBox(dlg)
        lay.setSpacing(10)
        lay.setContentsMargins(16, 16, 16, 16)

        msg = QLabel(
            tr("Bug, question, feature request?") + "\n"
            + tr("We read every message.")  # noqa: W503
        )
        msg.setWordWrap(True)
        msg.setStyleSheet("font-size: 12px; color: palette(text);")
        lay.addWidget(msg)

        email_label = QLabel(f"<b>{support_email}</b>")
        email_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        email_label.setStyleSheet("font-size: 12px; color: palette(text);")
        lay.addWidget(email_label)

        # Primary action: green filled (design-system CTA), like the dock's
        # own primary buttons. The click feedback swaps the label to "Copied".
        copy_btn = QPushButton(tr("Copy email address"))
        copy_btn.setStyleSheet(_BTN_GREEN)
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.clicked.connect(
            lambda: (
                QApplication.clipboard().setText(support_email),
                copy_btn.setText(tr("Copied")),
            )
        )
        lay.addWidget(copy_btn)

        or_label = QLabel(tr("or"))
        or_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        or_label.setStyleSheet("color: palette(text); font-size: 11px;")
        lay.addWidget(or_label)

        # Secondary action: blue filled, one step down from the green primary.
        call_btn = QPushButton(tr("Book a video call"))
        call_btn.setStyleSheet(_BTN_BLUE)
        call_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        call_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(calendly_url))
        )
        lay.addWidget(call_btn)

        dlg.exec()
