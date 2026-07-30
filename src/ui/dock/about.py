"""Update notification, about section, tutorial/report/shortcuts/contact.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

import html

from qgis.PyQt.QtCore import QSize, Qt
from qgis.PyQt.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QStyle,
    QToolButton,
    QWidget,
)

from ...core.activation_manager import (
    TUTORIAL_URL_FALLBACK,
    get_tutorial_url,
)
from ...core.i18n import tr
from ..credit_ring import CreditRing
from .font_scale import apply_font_scale_to_tree, scale_px_length
from .footer_bar import (
    FOOTER_GLYPH_PX,
    _ElidingFooterButton,
    footer_book_icon,
    footer_gear_icon,
)
from .styles import (
    _BTN_BLUE,
    _BTN_GREEN,
    _FOOTER_CTA_BTN_STYLE,
    _FOOTER_ICON_BTN_STYLE,
    _FOOTER_MENU_STYLE,
    _HELP_ICON_BTN_STYLE,
    BRAND_BLUE,
    BRAND_BLUE_HOVER,
    _msg_card_qss,
    _msg_label_qss,
)
from .widgets import (
    _ClickableFooterArea,
    _FooterIconButton,
)


def _web_url_or(candidate: object, fallback: str) -> str:
    """Return ``candidate`` only when it is a usable https web address.

    A URL that comes from the server (or from any other value we do not
    control) must never reach the desktop URL handler unchecked: another scheme
    would hand the string to a local application instead of the browser. The
    check itself lives in ``core.server_dials.safe_web_url``, shared with the
    rich-text label that shows the same address, so both places accept exactly
    the same strings.
    """
    from ...core.server_dials import safe_web_url

    return safe_web_url(candidate, fallback)


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
                # The version string comes from the plugin repository's XML and
                # lands in a rich-text label, so it is escaped like any other
                # remote value shown as markup.
                available_version = html.escape(
                    str(plugin_data.get("version_available") or "?"))
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
        # Two update banners at once would be noise. This one carries a real
        # version number, so it wins and the server-driven one steps aside.
        self.refresh_update_recommendation()

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

        # Review "View detections as" block, pinned here so it sits directly
        # above the footer credits row: a way to look at the results, always
        # available during a review, never the first item of the flow. Built
        # by DockAutoReviewBuildMixin (auto_review_build.py); it is a
        # main_layout sibling, so its show/hide travels with the review state,
        # not with the Automatic page.
        self._setup_auto_review_view_block(self.main_layout)

        # Footer icon row - mirrors AI Edit. Gear opens Account Settings
        # (visible only when activated), help opens a popup with Tutorial /
        # Shortcuts / Contact us. The contact / tutorial / shortcuts links
        # previously sat as blue underlined labels but moved into the help
        # menu so the bar matches AI Edit's compact look.
        footer_widget = QWidget()
        # One row: the promo link, the credit gauge, the Subscribe pill and the
        # icon buttons. The link elides when the dock is genuinely too narrow
        # for all of it, and only then (see _ElidingFooterButton).
        footer_row = QHBoxLayout(footer_widget)
        footer_row.setContentsMargins(0, 4, 0, 4)
        footer_row.setSpacing(6)

        # Automatic mode only: compact credit gauge (ring + "remaining / total")
        # plus a discreet Subscribe pill, bottom-left like AI Edit. Replaces
        # the old always-on upsell card that ate half the Automatic page. The
        # ring and its label sit in one clickable strip: the number is the
        # first place a user looks for their balance, so it is also the way to
        # the dashboard that explains it.
        self._credit_gauge = _ClickableFooterArea(footer_widget)
        gauge_row = QHBoxLayout(self._credit_gauge)
        gauge_row.setContentsMargins(0, 0, 0, 0)
        gauge_row.setSpacing(6)
        self._credit_gauge.clicked.connect(self._on_credit_gauge_clicked)
        footer_row.addWidget(self._credit_gauge)

        self._credit_ring = CreditRing(diameter=16, parent=self._credit_gauge)
        self._credit_ring.setVisible(False)
        gauge_row.addWidget(self._credit_ring)

        self._footer_credits_label = QLabel()
        self._footer_credits_label.setStyleSheet(
            "QLabel { font-size: 11px; color: palette(text);"
            " background: transparent; border: none; }"
        )
        self._footer_credits_label.setVisible(False)
        gauge_row.addWidget(self._footer_credits_label)

        self._subscribe_pill = QPushButton(tr("Upgrade to Pro"))
        self._subscribe_pill.setToolTip(tr("10,000 credits a month."))
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
        # Eliding, not plain: the label is the longest thing in the row, and a
        # button that refuses to shrink sets the minimum width of the whole
        # panel (see _ElidingFooterButton).
        self._ai_edit_btn = _ElidingFooterButton(footer_widget)
        # Decorative glyph kept out of the translatable string. The copy sells
        # AI Edit's promise (presentation and planning visuals) and deliberately
        # stays off AI Segmentation Pro's turf (no segmentation wording).
        self._ai_edit_btn.set_label(
            "🍌 " + tr("Make this map presentation-ready"))
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
        # A painted book, not U+1F4D6: Windows draws that character as a colour
        # emoji at the full em box, which broke the row's shared baseline and
        # made one glyph twice the size of its neighbours.
        self._tutorial_btn.setIcon(footer_book_icon(footer_widget))
        self._tutorial_btn.setIconSize(QSize(FOOTER_GLYPH_PX, FOOTER_GLYPH_PX))
        self._tutorial_btn.setToolTip(tr("Open the step-by-step tutorial"))
        self._tutorial_btn.setAccessibleName(tr("Tutorial"))
        self._tutorial_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._tutorial_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._tutorial_btn.setStyleSheet(_HELP_ICON_BTN_STYLE)
        self._tutorial_btn.clicked.connect(self._on_open_guide_footer)
        footer_row.addWidget(self._tutorial_btn)

        self._settings_btn = _FooterIconButton(footer_widget)
        self._settings_btn.setIcon(footer_gear_icon(footer_widget))
        self._settings_btn.setIconSize(QSize(FOOTER_GLYPH_PX, FOOTER_GLYPH_PX))
        self._settings_btn.setToolTip(tr("Settings"))
        self._settings_btn.setAccessibleName(tr("Settings"))
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

    def _on_credit_gauge_clicked(self):
        """Footer credit gauge: open the dashboard, where the balance, the plan
        and the invoices live. The address is a plugin constant, so it needs no
        https guard (that one is for server-supplied URLs)."""
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices

        from ...core.activation_manager import get_dashboard_url
        QDesktopServices.openUrl(QUrl(get_dashboard_url()))
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_pro_upsell_clicked(source="credit_gauge")
        except Exception:  # nosec B110 -- telemetry never blocks a click
            pass

    def _on_open_tutorial(self):
        """Open the tutorial URL in the system browser.

        The address is server-supplied, so it goes through the https guard and
        falls back to the built-in one when it is anything else.
        """
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        QDesktopServices.openUrl(
            QUrl(_web_url_or(get_tutorial_url(), TUTORIAL_URL_FALLBACK)))

    def _on_open_guide_footer(self):
        """Footer book button: open the step-by-step written guide.

        No guard needed here: guidance.guide_url() builds the address from a
        constant base, nothing server-supplied reaches the URL handler.
        """
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
        """Keyboard shortcuts dialog: the plugin keyboard map, grouped by the
        context where each key is actually live (K4).

        One group per context, in the order a user meets them: General, then
        Manual, then the Automatic steps (zone, detect, review, merge), then
        map navigation. Only keys a user can press in normal use are listed.
        The keys the native QGIS edit tools own during a Correct-step hand
        edit stay out: their banner buttons and tooltips carry them.
        """
        from qgis.PyQt.QtGui import QKeySequence
        from qgis.PyQt.QtWidgets import (
            QDialog,
            QFrame,
            QPushButton,
            QTextBrowser,
            QVBoxLayout,
        )

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
        del_key = native("Del")
        # Delete the active object: Delete, or Ctrl/Cmd+Backspace (the big
        # key on Mac keyboards); matches shortcut_filter.py.
        ctrl_backspace_key = native("Ctrl+Backspace")

        key_style = (
            "background-color: rgba(128,128,128,0.18);"
            "border: 1px solid rgba(128,128,128,0.35);"
            "border-radius: 3px;"
            "padding: 1px 5px;"
            # Named faces before the generic: Qt resolves the bare "monospace"
            # only where the font database aliases it, so Windows fell back to
            # a proportional face. Matches _KEY_BADGE_STYLE in dock/widgets.py.
            "font-family: Consolas, 'DejaVu Sans Mono', Menlo, monospace;"
        )

        def keys(*labels: str) -> str:
            """One key cap per label, joined by a slash when a context accepts
            several keys for the same action. Every label is a literal or a
            Qt-rendered key name, so there is no user text to escape here."""
            return " / ".join(
                f"<span style='{key_style}'>{label}</span>" for label in labels)

        def _row(key_html: str, action: str) -> str:
            return ("<tr><td style='padding-right:12px;'>"
                    f"{key_html}</td><td>{action}</td></tr>")

        def _section(title: str) -> str:
            return ("<tr><td colspan='2' style='padding-top:9px;"
                    f"padding-bottom:2px;'><b>{title}</b></td></tr>")

        rows = [
            "<table cellspacing='0' cellpadding='1'>",

            _section(tr("General")),
            _row(keys("G"), tr("Start (the visible mode's Start button)")),

            _section(tr("Manual")),
            _row(keys(tr("Left-click")), tr("Add area")),
            _row(keys(tr("Right-click")), tr("Remove area")),
            _row(keys(undo_key, backspace_key), tr("Undo last point")),
            _row(keys("S"), tr("Save polygon")),
            _row(keys("E"),
                 tr("Open the selected saved polygon for AI editing")),
            _row(keys(del_key, ctrl_backspace_key),
                 tr("Delete the active object")),
            _row(keys(enter_key), tr("Export polygon to a layer")),
            _row(keys(esc_key),
                 tr("Clear the selection, or stop the segmentation")),

            _section(tr("Automatic: draw the zone")),
            _row(keys(tr("Click")), tr("Add a point")),
            _row(keys(tr("Double-click")), tr("Finish the zone")),
            _row(keys(undo_key, backspace_key), tr("Undo the last point")),
            _row(keys(esc_key), tr("Clear the points, then exit Automatic")),

            _section(tr("Automatic: detect")),
            _row(keys(enter_key), tr("Run the detection")),
            _row(keys(esc_key),
                 tr("Cancel the example box, the detection, "
                    "or exit Automatic")),

            _section(tr("Automatic: review and Correct")),
            _row(keys(enter_key), tr("Export the polygons to a layer")),
            _row(keys(del_key, ctrl_backspace_key),
                 tr("Remove the selected detection")),
            _row(keys(undo_key), tr("Undo the last correction")),
            _row(keys("S"), tr("Save the fix and go back to the review")),
            _row(keys(esc_key),
                 tr("Close the fix, clear the selection, "
                    "or exit the review")),

            _section(tr("Automatic: merge with neighbours")),
            _row(keys(tr("Click")), tr("Pick or un-pick an object")),
            _row(keys(enter_key), tr("Confirm the merge")),
            _row(keys(esc_key), tr("Cancel the merge")),

            _section(tr("Navigation (while a tool is armed)")),
            _row(keys(tr("Space")), tr("Hold and move to pan the map")),
            _row(keys(tr("Arrow keys")), tr("Pan the map")),

            "</table>",
        ]
        shortcuts_html = "".join(rows)

        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Keyboard shortcuts"))
        # Same width as before; fixed, so the two columns keep their rhythm
        # whatever the platform key names are.
        dlg.setFixedWidth(scale_px_length(460))
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        # The map is long, so the rows scroll and the OK button stays on
        # screen. A read-only rich-text view, not a label in a scroll area:
        # it lays the table out at its real width, so a row that wraps can
        # never end up clipped at the bottom of the list.
        view = QTextBrowser(dlg)
        view.setHtml(shortcuts_html)
        view.setFrameShape(QFrame.Shape.NoFrame)
        view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setOpenExternalLinks(False)
        # Without this the viewport paints its own base color and the list
        # reads as a pale block dropped on the dialog, in both themes.
        view.setStyleSheet(
            "QTextBrowser { background: transparent; border: none;"
            " font-size: 12px; color: palette(text); }")
        list_height = 420
        try:
            available = dlg.screen().availableGeometry().height()
            list_height = max(240, min(list_height, available - 180))
        except (AttributeError, RuntimeError):
            pass
        view.setFixedHeight(list_height)
        layout.addWidget(view)

        ok_btn = QPushButton(tr("OK"))
        ok_btn.setStyleSheet(_BTN_BLUE)
        ok_btn.setFixedWidth(scale_px_length(80))
        ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        ok_btn.clicked.connect(dlg.accept)
        layout.addWidget(ok_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        apply_font_scale_to_tree(dlg)
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
        dlg.setMinimumWidth(scale_px_length(350))
        dlg.setMaximumWidth(scale_px_length(450))
        lay = _VBox(dlg)
        lay.setSpacing(10)
        lay.setContentsMargins(16, 16, 16, 16)

        msg = QLabel(tr("Bug, question, feature request?") + "\n" + tr("We read every message."))
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

        apply_font_scale_to_tree(dlg)
        dlg.exec()
