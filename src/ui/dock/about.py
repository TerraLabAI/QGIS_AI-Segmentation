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
    QVBoxLayout,
    QWidget,
)

from ...core.activation_manager import (
    TUTORIAL_URL_FALLBACK,
    get_contact_call_url,
    get_support_email,
    get_tutorial_url,
)
from ...core.i18n import tr
from ...core.server_dials import dial_copy
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
    _BTN_BLUE_PRIMARY,
    _BTN_GREEN,
    _FOOTER_CTA_BTN_STYLE,
    _FOOTER_ICON_BTN_STYLE,
    _FOOTER_MENU_STYLE,
    _HELP_ICON_BTN_STYLE,
    _UPDATE_CARD_STYLE,
    _UPDATE_LATER_STYLE,
    _UPDATE_NOTE_STYLE,
    _UPDATE_TITLE_STYLE,
    BRAND_BLUE,
    BRAND_BLUE_HOVER,
    _msg_card_qss,
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
        """The update card, full width at the top of the dock, hidden.

        An update the user never notices is an update they never install, so
        while one is pending this is the loudest thing on the panel. Built like
        the panel's other cards, so it reads as part of the dock rather than a
        banner dropped on top of it. Same card as AI Edit, on purpose: both
        plugins offer an update the same way.
        """
        import os

        from qgis.PyQt.QtGui import QPixmap

        self._update_notif_container = QWidget()
        self._update_notif_container.setObjectName("updateCard")
        self._update_notif_container.setStyleSheet(_UPDATE_CARD_STYLE)
        card = QVBoxLayout(self._update_notif_container)
        card.setContentsMargins(12, 12, 12, 12)
        card.setSpacing(8)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(10)
        icon_label = QLabel()
        icon_label.setFixedSize(36, 36)
        icon_label.setScaledContents(True)
        # about.py sits at src/ui/dock/, so the plugin root is four levels up.
        plugin_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        pixmap = QPixmap(os.path.join(plugin_root, "resources", "icons", "icon.png"))
        if not pixmap.isNull():
            icon_label.setPixmap(pixmap)
        head.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignTop)

        words = QVBoxLayout()
        words.setContentsMargins(0, 0, 0, 0)
        words.setSpacing(2)
        self._update_title_label = QLabel("")
        self._update_title_label.setWordWrap(True)
        self._update_title_label.setStyleSheet(_UPDATE_TITLE_STYLE)
        words.addWidget(self._update_title_label)
        # One line, and only one: the served release note. It says what the
        # release brings, which is the whole reason to press the button.
        self.update_notification_label = QLabel("")
        self.update_notification_label.setWordWrap(True)
        self.update_notification_label.setStyleSheet(_UPDATE_NOTE_STYLE)
        words.addWidget(self.update_notification_label)
        head.addLayout(words, 1)
        card.addLayout(head)

        self._update_now_btn = QPushButton(tr("Update now"))
        self._update_now_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_now_btn.setStyleSheet(_BTN_BLUE_PRIMARY)
        self._update_now_btn.setMinimumHeight(38)
        self._update_now_btn.setAutoDefault(False)
        self._update_now_btn.clicked.connect(self._on_open_plugin_manager)
        card.addWidget(self._update_now_btn)

        later_row = QHBoxLayout()
        later_row.setContentsMargins(0, 0, 0, 0)
        later_row.addStretch()
        self._update_later_btn = QPushButton(tr("Later"))
        self._update_later_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_later_btn.setStyleSheet(_UPDATE_LATER_STYLE)
        self._update_later_btn.setAutoDefault(False)
        self._update_later_btn.clicked.connect(self._on_dismiss_update_banner)
        later_row.addWidget(self._update_later_btn)
        later_row.addStretch()
        card.addLayout(later_row)

        self._update_notif_container.setVisible(False)
        self.update_notification_widget = self._update_notif_container
        self.main_layout.addWidget(self.update_notification_widget)

    def check_for_updates(self):
        """Offer the update only when QGIS can actually install it.

        The banner used to fire on the served latest_version alone. QGIS
        usually has not fetched the plugin repository yet at that point, so
        Update now opened a Plugin Manager whose Upgradeable tab was empty and
        the user could do nothing about it for ten minutes. The rule is now the
        installer's own verdict: status == "upgradeable", and nothing else.

        The served version is only the hint that a refresh is worth making.
        """
        version = self._upgradeable_version()
        if version:
            self._show_update_banner(version)
        else:
            self._maybe_refresh_plugin_repository()
        # The server-driven nudge no longer speaks for itself: one card, one
        # rule, and that rule is what QGIS can install.
        self.refresh_update_recommendation()

    def _show_update_banner(self, version: str) -> None:
        """Offer one version, with the served line about what it brings.
        Silent when the user already pressed Later for it."""
        from .guidance import HINT_UPDATE_RECOMMENDED, is_hint_dismissed_for_version

        try:
            if is_hint_dismissed_for_version(
                    HINT_UPDATE_RECOMMENDED, version, self._installed_version()):
                return
        except Exception:  # noqa: BLE001 -- a memory is best-effort
            pass  # nosec B110
        self._repo_offered_version = version
        self._update_title_label.setText(
            tr("AI Segmentation {version} is out").format(version=version))
        note = self._served_update_line()
        self.update_notification_label.setText(note)
        self.update_notification_label.setVisible(bool(note))
        self.update_notification_widget.setVisible(True)
        from .server_switches import UPDATE_TRIGGER_PLUGIN_REGISTRY

        self._track_update_prompt_shown(version, UPDATE_TRIGGER_PLUGIN_REGISTRY)

    def _plugin_installer_key(self) -> str:
        """The key pyplugin_installer files this install under: the folder
        name. A dev clone whose folder differs from the published id is never
        listed, which is why a dev install sees no banner."""
        import os

        return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))))

    def _upgradeable_version(self) -> str:
        """The version QGIS says is installable, or "" when there is none."""
        try:
            from pyplugin_installer.installer_data import plugins

            data = plugins.all().get(self._plugin_installer_key())
            if data and data.get("status") == "upgradeable":
                return str(data.get("version_available") or "")
        except Exception:  # noqa: BLE001
            pass  # nosec B110 -- no repo metadata yet, dev install, etc.
        return ""

    def _maybe_refresh_plugin_repository(self) -> None:
        """One non-blocking repository refresh per session, when the server
        says a newer version exists and QGIS has not caught up.

        fetchAvailablePlugins() is not usable here: it opens a modal fetching
        dialog and exec()s it, which freezes QGIS. requestFetching() does the
        same work without the dialog, and checkingDone says when to look again.
        """
        if getattr(self, "_update_refresh_requested", False):
            return
        served = self._served_newer_version()
        if not served:
            return
        self._update_refresh_requested = True
        try:
            from pyplugin_installer.installer_data import repositories
        except Exception:  # noqa: BLE001
            self._track_update_prompt_suppressed(served, "installer_unavailable")
            return
        try:
            enabled = list(repositories.allEnabled())
            if not enabled:
                self._track_update_prompt_suppressed(served, "no_repository")
                return
            repositories.checkingDone.connect(self._on_plugin_repository_checked)
            for key in enabled:
                repositories.requestFetching(key, force_reload=True)
        except Exception:  # noqa: BLE001
            self._track_update_prompt_suppressed(served, "refresh_failed")

    def _on_plugin_repository_checked(self) -> None:
        """The repository answered: rebuild the plugin list and look again."""
        served = self._served_newer_version()
        try:
            from pyplugin_installer.installer_data import plugins, repositories

            try:
                repositories.checkingDone.disconnect(
                    self._on_plugin_repository_checked)
            except (TypeError, RuntimeError):
                pass  # nosec B110 -- already disconnected
            plugins.rebuild()
        except Exception:  # noqa: BLE001
            if served:
                self._track_update_prompt_suppressed(served, "refresh_failed")
            return
        version = self._upgradeable_version()
        if version:
            self._show_update_banner(version)
        elif served:
            self._track_update_prompt_suppressed(served, "not_listed")

    def _served_newer_version(self) -> str:
        """The served version when it is ahead of the installed one, else "".

        Cache-only: the plugin makes no network call to answer this.
        """
        try:
            from ...core.activation_manager import (
                get_latest_version,
                is_update_available,
            )

            installed = self._installed_version()
            if not installed:
                return ""
            latest = get_latest_version()
            if latest and is_update_available(installed):
                return str(latest)
        except Exception:  # noqa: BLE001
            pass  # nosec B110 -- a bad nudge must never break the dock
        return ""

    def _track_update_prompt_suppressed(self, served_version: str,
                                        reason: str) -> None:
        """Report an offer we chose not to make, once per reason per session."""
        seen = getattr(self, "_update_suppressed_reported", None)
        if seen is None:
            seen = set()
            self._update_suppressed_reported = seen
        if reason in seen:
            return
        seen.add(reason)
        try:
            from ...core import telemetry_events as ev
            from ...core.telemetry import track

            track(ev.PLUGIN_UPDATE_PROMPT_SUPPRESSED,
                  {"served_version": served_version, "reason": reason})
        except Exception:  # noqa: BLE001
            pass  # nosec B110 -- telemetry must never break the dock

    @staticmethod
    def _served_update_line() -> str:
        """The one served sentence under the offer: the release note when
        there is one, the generic update line otherwise."""
        try:
            from ...core.activation_manager import get_release_notes_line

            return get_release_notes_line() or ""
        except Exception:  # noqa: BLE001
            return ""  # a bad line must never break the dock

    def _on_dismiss_update_banner(self) -> None:
        """Later: hide this version's offer and remember only this version."""
        version = getattr(self, "_repo_offered_version", "") or ""
        try:
            self.update_notification_widget.setVisible(False)
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- teardown
        try:
            from .guidance import HINT_UPDATE_RECOMMENDED, dismiss_hint_for_version

            if version:
                dismiss_hint_for_version(HINT_UPDATE_RECOMMENDED, version)
        except Exception:  # noqa: BLE001
            pass  # nosec B110 -- a memory is best-effort
        self._track_update_prompt_clicked(version, "dismissed")

    def _on_open_plugin_manager(self, _link=None):
        """Open the QGIS Plugin Manager on the Upgradeable tab (index 3)."""
        from .server_switches import open_plugin_manager_or_marketplace

        landed = open_plugin_manager_or_marketplace()
        self._track_update_prompt_clicked(
            getattr(self, "_repo_offered_version", ""),
            "plugin_manager" if landed else "marketplace_page")

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

        # Both modes: compact credit gauge (ring + "remaining / total")
        # plus a discreet Subscribe pill, bottom-left like AI Edit. The balance
        # belongs to the account, so it reads the same wherever the user
        # is, and it is hidden only while signed out. Replaces
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

        # The same served label every other Pro button in the dock wears, so
        # one deploy renames them together.
        self._subscribe_pill = QPushButton(
            dial_copy("upsell.cta", tr("Upgrade to Pro")))
        # The tooltip quotes both monthly envelopes, two commercial figures
        # that move without waiting for a plugin release.
        self.refresh_subscribe_pill_tooltip()
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
        # Read late, never once: this runs while the plugin starts, before the
        # first configuration has landed, so the refresh calls it again.
        footer_row.addWidget(self._subscribe_pill)

        # Cross-promo CTA, pinned bottom-left (before the stretch) so it sits
        # beside the gear/help icons without crowding them (#30). Always opens
        # the AI Edit product page in the browser.
        #
        # SURFACED NOWHERE: the credit gauge took the bottom-left
        # slot in both modes, and signed out the footer shows neither. Every
        # setVisible on this button is now False. It is built and wired so the
        # slot can be given back without rebuilding the footer.
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
        self._tutorial_btn.set_glyph_icon(footer_book_icon, FOOTER_GLYPH_PX)
        self._tutorial_btn.setToolTip(tr("Open the step-by-step tutorial"))
        self._tutorial_btn.setAccessibleName(tr("Tutorial"))
        self._tutorial_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._tutorial_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._tutorial_btn.setStyleSheet(_HELP_ICON_BTN_STYLE)
        self._tutorial_btn.clicked.connect(self._on_open_guide_footer)
        footer_row.addWidget(self._tutorial_btn)

        self._settings_btn = _FooterIconButton(footer_widget)
        self._settings_btn.set_glyph_icon(footer_gear_icon, FOOTER_GLYPH_PX)
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

    def refresh_subscribe_pill_tooltip(self) -> None:
        """Both monthly envelopes and, when the server serves one, the price.

        The pill is a 20-pixel strip in the dock footer: a line of text under
        it would push the credit gauge off the row, so this is the one Pro
        surface where the offer is read on hover rather than on the card.
        """
        pill = getattr(self, "_subscribe_pill", None)
        if pill is None:
            return
        from ...core.pro_offer_copy import join_offer_line, pro_price_phrase
        try:
            pill.setToolTip(join_offer_line(dial_copy(
                "upsell.pill_tooltip",
                tr("Pro: 500 cloud objects a month in Semi-Auto with Cloud AI, "
                   "and 200 km² of Automatic. Same AI, same free clicks and "
                   "corrections, every machine you work on.")),
                pro_price_phrase()))
        except RuntimeError:
            pass  # nosec B110 -- teardown

    def _on_credit_gauge_clicked(self):
        """Footer credit gauge: open the dashboard, where the balance, the plan
        and the invoices live. The address is a plugin constant, so it needs no
        https guard (that one is for server-supplied URLs)."""
        from ...core.activation_manager import get_dashboard_url
        from ..external_links import open_external_url
        open_external_url(get_dashboard_url(), parent=self)
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
        from ..external_links import open_external_url
        open_external_url(
            _web_url_or(get_tutorial_url(), TUTORIAL_URL_FALLBACK), parent=self)

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

            _section(tr("Semi-Auto")),
            _row(keys(tr("Left-click")), tr("Add area")),
            _row(keys(tr("Right-click")), tr("Remove area")),
            _row(keys(undo_key, backspace_key), tr("Undo last point")),
            _row(keys("S"), tr("Save polygon")),
            _row(keys("E"),
                 tr("Open the selected saved polygon for AI editing")),
            _row(keys(del_key, ctrl_backspace_key),
                 tr("Delete the active object")),
            _row(keys(enter_key), tr("Export polygon to a layer")),
            _row(keys("C"), tr("Clear the selection in progress")),
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
        # A floor, not a cap. The same rows run about a third longer in French
        # and in German, and a fixed width cut their right-hand column off.
        dlg.setMinimumWidth(scale_px_length(460))
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
        from qgis.PyQt.QtWidgets import QApplication, QDialog
        from qgis.PyQt.QtWidgets import QVBoxLayout as _VBox

        from ..external_links import open_external_url

        # Served like every other outbound link on this screen, so the
        # booking address can move without a plugin release.
        calendly_url = get_contact_call_url()
        support_email = get_support_email("yvann.barbot@terra-lab.ai")

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
            lambda: open_external_url(calendly_url, parent=dlg)
        )
        lay.addWidget(call_btn)

        apply_font_scale_to_tree(dlg)
        dlg.exec()
