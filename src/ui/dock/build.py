





from __future__ import annotations

import sys

from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtGui import QKeySequence
from qgis.PyQt.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...core.cache_paths import PLUGIN_CACHE_DIR
from ...core.i18n import tr
from ...core.model_config import _IS_MACOS_X86, USE_SAM2
from ...core.qt_compat import QShortcut
from ...core.server_dials import dial_in_range
from ..layer_tree_combobox import LayerTreeComboBox
from .guidance import (
    BLUE_TINT,
    HINT_PREVIEW_ZOOM,
    DismissibleHint,
)
from .styles import (
    _BTN_EXPORT_READY,
    _BTN_GREEN,
    _BTN_GREEN_AUTH,
    _BTN_GREEN_STEP,
    _BTN_LINK,
    _BTN_PAIR_CANCEL,
    _BTN_PAIR_NEUTRAL,
    _BTN_RED,
    _CARD_MARGINS,
    _CARD_TITLE_QSS,
    _FIELD_LABEL_QSS,
    _HERO_TITLE_QSS,
    _HINT_LINE_QSS,
    _MUTED_LINE_QSS,
    _QUIET_LINK_QSS,
    _SETUP_STATUS_QSS,
    BTN_PRIMARY_WIDE_PX,
    BTN_PX,
    BTN_SMALL_PX,
    FONT_HINT,
    HUE_HOWTO,
    HUE_LOCAL,
    HUE_RESULT,
    INK_2,
    INSET,
    LINE,
    RADIUS_CARD,
    RADIUS_CONTROL,
    SPACE_CARD,
    SPACE_OUTER,
    SPACE_STAGE,
    SPACE_TIGHT,
    _btn_start_qss,
    category_card_qss,
    category_label_qss,
    category_progress_qss,
    category_tile,
    category_wash,
    category_wash_line,
    combo_theme_qss,
)
from .widgets import (
    Mode,
    _ModeSwitch,
    _ShortcutArmingFilter,
    _Spinner,
    build_no_imagery_hero,
    set_key_chip,
)




_MODE_BLOCK_NOTICE_MS = 3000


_PAIRING_SPINNER_MS = 80


class DockBuildMixin:


    def _setup_title_bar(self):






        from .dock_header import DockHeader

        header = DockHeader(self)


        header.pro_pill_clicked.connect(self._on_header_pro_pill_clicked)
        header.settings_clicked.connect(lambda: self.settings_clicked.emit())
        header.float_clicked.connect(lambda: self.setFloating(not self.isFloating()))
        header.close_clicked.connect(self.close)
        self._settings_btn = header.settings_btn

        self._settings_btn.setVisible(False)

        self.setTitleBarWidget(header)
        self._custom_title_bar = header
        self._dock_header = header
        if sys.platform == "win32":



            self.topLevelChanged.connect(self._swap_title_bar_for_floating)

    def _swap_title_bar_for_floating(self, floating: bool) -> None:







        header = self._custom_title_bar
        try:
            if floating:
                self.setTitleBarWidget(None)
                header.float_btn.setVisible(False)
                header.close_btn.setVisible(False)
                self.main_layout.insertWidget(0, header)
                header.show()
            else:
                self.main_layout.removeWidget(header)
                header.float_btn.setVisible(True)
                header.close_btn.setVisible(True)
                self.setTitleBarWidget(header)
                header.show()
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _setup_ui(self):
        self._setup_mode_switch()
        self._setup_welcome_section()
        self._setup_setup_section()
        self._setup_activation_section()
        self._setup_segmentation_section()


        self._setup_manual_engine()
        self._setup_manual_credit_gate()
        self._setup_automatic_page()
        self.main_layout.addStretch()
        self._setup_low_credit_slot()


        self._setup_update_notification()


        self.main_layout.removeWidget(self.update_notification_widget)
        self.main_layout.insertWidget(0, self.update_notification_widget)
        self._setup_update_recommendation()
        self._setup_about_section()


        self.apply_server_feature_switches()

    def _setup_low_credit_slot(self):














        holder = QWidget()
        holder.setObjectName("lowCreditSlot")
        slot = QVBoxLayout(holder)
        slot.setContentsMargins(0, 0, 0, 0)
        slot.setSpacing(6)
        self.low_credit_slot = slot
        self.main_layout.addWidget(holder)


        line = getattr(self, "manual_engine_low_line", None)
        if line is not None:
            slot.addWidget(line)

    def _setup_mode_switch(self):

        self.mode_switch = _ModeSwitch(self._mode, self)
        self.mode_switch.mode_selected.connect(self._on_mode_selected)
        self.main_layout.addWidget(self.mode_switch)

    def _on_mode_selected(self, mode: Mode) -> bool:






        if mode == Mode.AUTOMATIC and self._segmentation_active:
            self.mode_switch.blockSignals(True)
            self.mode_switch.set_mode(Mode.INTERACTIVE)
            self.mode_switch.blockSignals(False)
            msg = tr("Stop the active segmentation before switching modes.")
            self.mode_switch.setToolTip(msg)


            try:


                self._set_instructions_style("card")
                self.instructions_label.setText(msg)
                QTimer.singleShot(dial_in_range(
                    "tuning.ui.mode_block_notice_ms",
                    _MODE_BLOCK_NOTICE_MS, 1000, 10000),
                    self._update_instructions)
            except (RuntimeError, AttributeError):
                pass
            return False
        if mode == Mode.INTERACTIVE and self._auto_run_active:
            self.mode_switch.blockSignals(True)
            self.mode_switch.set_mode(Mode.AUTOMATIC)
            self.mode_switch.blockSignals(False)
            msg = tr("Cancel the active detection before switching modes.")
            self.mode_switch.setToolTip(msg)


            try:
                self.set_auto_status("info", msg)
                QTimer.singleShot(dial_in_range(
                    "tuning.ui.mode_block_notice_ms",
                    _MODE_BLOCK_NOTICE_MS, 1000, 10000),
                    self._restore_auto_run_status)
            except (RuntimeError, AttributeError):
                pass
            return False
        self.mode_switch.setToolTip("")
        self.hide_pro_after_success()
        self.clear_manual_export_success()
        self._mode = mode




        self.mode_switch.set_mode(mode)
        self._auto_zone_too_large = False


        self.reset_manual_engine_to_cloud()

        self.mode_changed.emit(mode)
        self._update_full_ui()
        return True

    def _on_manual_try_example(self) -> None:








        self._on_mode_selected(Mode.AUTOMATIC)
        QTimer.singleShot(0, self.auto_demo_requested.emit)

    def _restore_auto_run_status(self) -> None:


        try:
            if self._auto_run_active:
                self.set_auto_status("progress")
                self._refresh_auto_progress_readout()
                self._set_auto_progress_visible(True)
            else:
                self.set_auto_status("idle")
        except (RuntimeError, AttributeError):
            pass

    def _setup_welcome_section(self):

        self.welcome_widget = QWidget()
        self.welcome_widget.setObjectName("welcomeCard")
        self.welcome_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.welcome_widget.setStyleSheet(category_card_qss("welcomeCard", HUE_LOCAL))
        layout = QVBoxLayout(self.welcome_widget)

        layout.setContentsMargins(*_CARD_MARGINS)
        layout.setSpacing(SPACE_TIGHT)

        self.welcome_title = QLabel(tr("Click Install to set up AI Segmentation"))
        self.welcome_title.setWordWrap(True)
        self.welcome_title.setStyleSheet(_CARD_TITLE_QSS)
        layout.addWidget(self.welcome_title)






        self.welcome_widget.setVisible(False)
        self.main_layout.addWidget(self.welcome_widget)

    def _setup_setup_section(self):





        self.setup_group = QGroupBox("")


        self.setup_group.setObjectName("setupCard")
        self.setup_group.setStyleSheet(
            f"QGroupBox#setupCard {{ background-color: {category_wash(HUE_LOCAL)};"
            f" border: 1px solid {category_wash_line(HUE_LOCAL)};"
            f" border-radius: {RADIUS_CARD}px; margin: 0; padding: 0; }}"
        )
        layout = QVBoxLayout(self.setup_group)
        layout.setContentsMargins(*_CARD_MARGINS)
        layout.setSpacing(SPACE_CARD)

        _setup_head = QHBoxLayout()
        _setup_head.setContentsMargins(0, 0, 0, 0)
        _setup_head.setSpacing(10)
        _setup_head.addWidget(
            category_tile("package", HUE_LOCAL, on_wash=True), 0,
            Qt.AlignmentFlag.AlignVCenter)
        self.setup_status_label = QLabel(tr("Checking..."))
        self.setup_status_label.setWordWrap(True)

        self.setup_status_label.setStyleSheet(_SETUP_STATUS_QSS)
        _setup_head.addWidget(self.setup_status_label, 1, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(_setup_head)

        self.setup_progress = QProgressBar()
        self.setup_progress.setRange(0, 100)
        self.setup_progress.setStyleSheet(category_progress_qss(HUE_LOCAL))
        self.setup_progress.setTextVisible(False)
        self.setup_progress.setVisible(False)
        layout.addWidget(self.setup_progress)

        self.setup_progress_label = QLabel("")
        self.setup_progress_label.setStyleSheet(_HINT_LINE_QSS)
        self.setup_progress_label.setVisible(False)
        layout.addWidget(self.setup_progress_label)

        self.install_button = QPushButton(tr("Install"))
        self.install_button.clicked.connect(self._on_install_clicked)
        self.install_button.setVisible(False)
        self.install_button.setMinimumHeight(BTN_PRIMARY_WIDE_PX)
        self.install_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.install_button.setStyleSheet(_BTN_GREEN)
        layout.addWidget(self.install_button)

        self.install_path_label = QLabel(
            tr("Install path: {}").format(PLUGIN_CACHE_DIR))
        self.install_path_label.setWordWrap(True)






        self.install_path_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.install_path_label.setToolTip(PLUGIN_CACHE_DIR)
        self.install_path_label.setStyleSheet(
            f"color: {INK_2};"
            f"font-size: {FONT_HINT}px;"
            "padding: 6px 8px;"
            f"background-color: {INSET};"
            f"border: 1px solid {LINE};"
            f"border-radius: {RADIUS_CONTROL}px;"
        )
        layout.addWidget(self.install_path_label)



        self.install_path_label.setVisible(False)


        self.setup_cloud_link = QPushButton(tr("Use Cloud AI instead"))
        self.setup_cloud_link.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setup_cloud_link.setStyleSheet(_BTN_LINK)
        self.setup_cloud_link.setMinimumHeight(BTN_SMALL_PX)
        self.setup_cloud_link.setVisible(False)
        self.setup_cloud_link.clicked.connect(
            lambda: self._on_manual_engine_picked(True))
        layout.addWidget(self.setup_cloud_link, 0, Qt.AlignmentFlag.AlignLeft)

        self.cancel_button = QPushButton(tr("Cancel"))
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.cancel_button.setVisible(False)
        self.cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_button.setMinimumHeight(BTN_PX)
        self.cancel_button.setStyleSheet(_BTN_RED)
        layout.addWidget(self.cancel_button)

        if not USE_SAM2:
            if _IS_MACOS_X86:
                sam1_text = tr("Intel Mac: using the older AI model.")
            else:
                sam1_text = tr("Update QGIS to 3.34+ for the latest AI model")
            sam1_info = QLabel(sam1_text)
            sam1_info.setWordWrap(True)
            sam1_info.setStyleSheet(_HINT_LINE_QSS)
            layout.addWidget(sam1_info)






        self.setup_group.setVisible(False)
        self.main_layout.addWidget(self.setup_group)

    def _setup_activation_section(self):


        self.activation_group = QGroupBox()
        self.activation_group.setStyleSheet(
            "QGroupBox { border: none; margin: 0; padding: 0; }"
        )
        layout = QVBoxLayout(self.activation_group)
        layout.setSpacing(SPACE_OUTER)


        layout.setContentsMargins(0, 0, 0, 0)

        layout.addSpacing(36)


        from ..icons import logo_pixmap, logo_size
        _logo_lbl = QLabel()
        _logo_lbl.setFixedSize(logo_size(40))
        _logo_lbl.setPixmap(logo_pixmap(_logo_lbl, 40))
        _logo_lbl.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(_logo_lbl, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(SPACE_TIGHT)
        self._setup_header = QLabel(tr("Segment your map with AI"))
        self._setup_header.setWordWrap(True)
        self._setup_header.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._setup_header.setStyleSheet(_HERO_TITLE_QSS)
        layout.addWidget(self._setup_header)


        self._connect_section = QWidget()
        connect_layout = QVBoxLayout(self._connect_section)
        connect_layout.setContentsMargins(0, 0, 0, 0)
        connect_layout.setSpacing(SPACE_OUTER)




        self._connect_hint_label = QLabel(
            tr("Free account - sign up takes 15 seconds in your browser."))
        self._connect_hint_label.setWordWrap(True)
        self._connect_hint_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._connect_hint_label.setStyleSheet(_MUTED_LINE_QSS)
        connect_layout.addWidget(self._connect_hint_label)
        connect_layout.addSpacing(SPACE_STAGE)

        self._connect_btn = QPushButton(tr("Sign in / Sign up to start"))
        self._connect_btn.setToolTip(
            tr("Sign in via your browser to start using AI Segmentation"))
        self._connect_btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)


        self._connect_btn.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._connect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._connect_btn.setStyleSheet(_BTN_GREEN_AUTH)
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        connect_layout.addWidget(self._connect_btn)



        from .tutorial_link import add_home_quiet_link, build_home_tutorial_link
        self._signed_out_links = build_home_tutorial_link(
            self._on_open_guide_footer, "signedOutLinksRow")
        add_home_quiet_link(self._signed_out_links, tr("Contact us"),
                            "chat_bubble", self._on_signed_out_contact)
        connect_layout.addWidget(self._signed_out_links)

        layout.addWidget(self._connect_section)


        self._pairing_wait_section = QWidget()
        wait_layout = QVBoxLayout(self._pairing_wait_section)
        wait_layout.setContentsMargins(0, SPACE_STAGE, 0, 0)
        wait_layout.setSpacing(SPACE_STAGE)




        self._pairing_spinner = _Spinner(16)
        wait_layout.addWidget(
            self._pairing_spinner, 0, Qt.AlignmentFlag.AlignHCenter)
        self._pairing_status = QLabel(tr("Waiting for your browser sign-in..."))
        self._pairing_status.setWordWrap(True)
        self._pairing_status.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._pairing_status.setStyleSheet(_MUTED_LINE_QSS)
        wait_layout.addWidget(self._pairing_status)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(SPACE_CARD)
        self._pairing_reopen_btn = QPushButton(tr("Open again"))
        self._pairing_reopen_btn.setToolTip(tr("Didn't open? Open the page again"))

        self._pairing_reopen_btn.setMinimumHeight(BTN_PX)
        self._pairing_reopen_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pairing_reopen_btn.setStyleSheet(_BTN_PAIR_NEUTRAL)
        self._pairing_reopen_btn.clicked.connect(self._on_pairing_reopen_clicked)
        btn_row.addWidget(self._pairing_reopen_btn)

        self._pairing_cancel_btn = QPushButton(tr("Cancel"))

        self._pairing_cancel_btn.setMinimumHeight(BTN_PX)
        self._pairing_cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pairing_cancel_btn.setStyleSheet(_BTN_PAIR_CANCEL)
        self._pairing_cancel_btn.clicked.connect(self._on_pairing_cancel_clicked)
        btn_row.addWidget(self._pairing_cancel_btn)
        wait_layout.addLayout(btn_row)

        self._pairing_wait_section.setVisible(False)
        layout.addWidget(self._pairing_wait_section)



        self._pairing_anim_timer = QTimer(self)
        self._pairing_anim_timer.setInterval(dial_in_range(
            "tuning.ui.pairing_spinner_ms", _PAIRING_SPINNER_MS, 20, 500))
        self._pairing_anim_timer.timeout.connect(self._pairing_spinner.advance)
        self._pending_pairing_code = ""





        self.activation_message_label = QLabel("")
        self.activation_message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.activation_message_label.setWordWrap(True)
        self.activation_message_label.setVisible(False)
        self.activation_message_label.setStyleSheet(_HINT_LINE_QSS)
        layout.addWidget(self.activation_message_label)

        self.activation_group.setVisible(False)
        self.main_layout.addWidget(self.activation_group)



    def _on_connect_clicked(self):


        import secrets
        self._pending_pairing_code = secrets.token_urlsafe(32)
        self.show_pairing_waiting()
        self.pairing_requested.emit(self._pending_pairing_code)

    def _on_pairing_reopen_clicked(self):

        if self._pending_pairing_code:
            self.pairing_requested.emit(self._pending_pairing_code)

    def _on_pairing_cancel_clicked(self):
        self.pairing_cancel_requested.emit(self._pending_pairing_code)
        self._pending_pairing_code = ""
        self.show_pairing_idle()

    def show_pairing_waiting(self):

        self._connect_section.setVisible(False)
        self.activation_message_label.setVisible(False)
        self._pairing_wait_section.setVisible(True)
        self._pairing_anim_timer.start()

    def _stop_pairing_wait(self):

        self._pairing_anim_timer.stop()
        self._pairing_wait_section.setVisible(False)

    def show_pairing_idle(self):

        self._stop_pairing_wait()
        self._connect_section.setVisible(True)

    def _on_start_shortcut_key(self):











        if self._dock_combo_has_focus():
            return
        self._on_start_shortcut()

    def _setup_segmentation_section(self):
        self.seg_widget = QWidget()
        layout = QVBoxLayout(self.seg_widget)
        layout.setContentsMargins(0, SPACE_OUTER, 0, 0)
        layout.setSpacing(SPACE_OUTER)






        layer_label = QLabel(tr("Image to segment"))
        layer_label.setStyleSheet(_FIELD_LABEL_QSS)
        layout.addWidget(layer_label)
        self.layer_label = layer_label

        self.layer_combo = LayerTreeComboBox()
        self.layer_combo.layerChanged.connect(self._on_layer_changed)
        self.layer_combo.setToolTip(tr("Select a raster layer (GeoTIFF, WMS, XYZ tiles, etc.)"))
        self.layer_combo.setStyleSheet(combo_theme_qss())
        self.layer_combo.setAccessibleName(
            tr("Select a raster layer to segment:").rstrip(": "))
        self.layer_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.layer_combo.setMinimumWidth(0)
        layout.addWidget(self.layer_combo)






        self.no_rasters_widget, self.manual_demo_btn = build_no_imagery_hero(
            on_demo=self._on_manual_try_example,
        )
        self.no_rasters_widget.setVisible(False)







        layout.addWidget(self.no_rasters_widget)


        self.instructions_label = QLabel("")
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setMinimumHeight(0)
        self.instructions_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.instructions_label.setStyleSheet(category_label_qss(HUE_HOWTO))

        self._instructions_style = "card"
        self.instructions_label.setVisible(False)
        layout.addWidget(self.instructions_label)







        self.preview_zoom_hint = DismissibleHint(
            HINT_PREVIEW_ZOOM,
            tr("Zoom in for a finer outline. The AI reads the image at "
               "your current zoom."),
            tint=BLUE_TINT,
            visibility_gate=self._preview_zoom_tip_wanted,
        )
        self.preview_zoom_hint.setVisible(False)

        self.preview_zoom_hint.dismissed.connect(self._sync_manual_session_tips)
        layout.addWidget(self.preview_zoom_hint)


        self.start_container = QWidget()
        start_layout = QVBoxLayout(self.start_container)
        start_layout.setContentsMargins(0, SPACE_TIGHT, 0, 0)
        start_layout.setSpacing(SPACE_OUTER)







        self.start_button = QPushButton(tr("Start Semi-Auto AI Segmentation"))
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._on_start_clicked)
        self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)


        self.start_button.setMinimumHeight(BTN_PRIMARY_WIDE_PX)


        self.start_button.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.start_button.setStyleSheet(
            _btn_start_qss(_BTN_GREEN_STEP, full_width=True))
        start_layout.addWidget(self.start_button)




        self.manual_export_success = QLabel()
        self.manual_export_success.setWordWrap(True)
        self.manual_export_success.setTextFormat(Qt.TextFormat.RichText)
        self.manual_export_success.setOpenExternalLinks(False)
        self.manual_export_success.linkActivated.connect(self._on_auto_recap_link)
        self.manual_export_success.setStyleSheet(category_label_qss(HUE_RESULT))
        self.manual_export_success.setVisible(False)
        start_layout.addWidget(self.manual_export_success)


        from .pro_nudges import build_pro_after_success_label
        self.manual_pro_after_success = build_pro_after_success_label(
            self._on_pro_after_success_link)
        start_layout.addWidget(self.manual_pro_after_success)


        from .tutorial_link import build_home_tutorial_link
        self.manual_tutorial_link = build_home_tutorial_link(
            self._on_open_guide_footer, "manualTutorialRow")
        start_layout.addWidget(self.manual_tutorial_link)






















        self.start_shortcut = QShortcut(QKeySequence("G"), self)
        self.start_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.start_shortcut.activated.connect(self._on_start_shortcut_key)







        self.auto_escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.auto_escape_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.auto_escape_shortcut.activated.connect(self._on_auto_escape_shortcut)
        self.auto_enter_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Return), self)
        self.auto_enter_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.auto_enter_shortcut.activated.connect(self._on_auto_enter_shortcut)
        self.auto_enter_shortcut_kp = QShortcut(QKeySequence(Qt.Key.Key_Enter), self)
        self.auto_enter_shortcut_kp.setContext(Qt.ShortcutContext.WindowShortcut)
        self.auto_enter_shortcut_kp.activated.connect(self._on_auto_enter_shortcut)












        self.auto_correct_remove_backspace_shortcut = QShortcut(
            QKeySequence("Ctrl+Backspace"), self)
        self.auto_correct_remove_backspace_shortcut.setContext(
            Qt.ShortcutContext.WindowShortcut)
        self.auto_correct_remove_backspace_shortcut.activated.connect(
            self._on_auto_correct_remove_shortcut)
        self.auto_correct_remove_delete_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Delete), self)
        self.auto_correct_remove_delete_shortcut.setContext(
            Qt.ShortcutContext.WindowShortcut)
        self.auto_correct_remove_delete_shortcut.activated.connect(
            self._on_auto_correct_remove_shortcut)
        self.auto_correct_undo_shortcut = QShortcut(
            QKeySequence.StandardKey.Undo, self)
        self.auto_correct_undo_shortcut.setContext(
            Qt.ShortcutContext.WindowShortcut)
        self.auto_correct_undo_shortcut.activated.connect(
            self._on_auto_correct_undo_shortcut)




        self._dock_shortcuts = [
            self.start_shortcut,
            self.auto_escape_shortcut,
            self.auto_enter_shortcut,
            self.auto_enter_shortcut_kp,
            self.auto_correct_remove_backspace_shortcut,
            self.auto_correct_remove_delete_shortcut,
            self.auto_correct_undo_shortcut,
        ]





        self.refresh_auto_shortcut_arming()
        self._shortcut_arming_filter = _ShortcutArmingFilter(self, self)


        self.installEventFilter(self._shortcut_arming_filter)


        self._shortcut_arming_window = None
        try:
            from qgis.utils import iface

            window = iface.mainWindow() if iface is not None else None
            if window is not None:
                window.installEventFilter(self._shortcut_arming_filter)
                self._shortcut_arming_window = window
        except (ImportError, AttributeError, RuntimeError):
            pass  # nosec B110

        layout.addWidget(self.start_container)


        self._setup_refine_panel(layout)






        self.save_mask_button = QPushButton(tr("Save polygon"))
        self.save_mask_button.clicked.connect(self._on_save_polygon_clicked)
        self.save_mask_button.setVisible(False)
        self.save_mask_button.setEnabled(False)
        self.save_mask_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_mask_button.setMinimumHeight(BTN_PRIMARY_WIDE_PX)


        self.save_mask_button.setStyleSheet(_BTN_GREEN)
        self.save_mask_button.setToolTip(
            tr("Click the object first. Save polygon keeps it in your "
               "session; Export writes all kept polygons to a layer.")
            + "\n" + tr("Shortcut: {key}").format(key="S")
        )
        set_key_chip(self.save_mask_button, "S", on_fill=True)


        from .session_link_row import EnabledChangeRelay, SessionLinkRow
        self._save_enabled_relay = EnabledChangeRelay(
            self._sync_manual_session_actions, self)
        self.save_mask_button.installEventFilter(self._save_enabled_relay)
        layout.addWidget(self.save_mask_button)

        self.export_button = QPushButton(tr("Export polygon to a layer"))
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setVisible(False)
        self.export_button.setEnabled(False)
        self.export_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_button.setMinimumHeight(BTN_PRIMARY_WIDE_PX)


        self.export_button.setStyleSheet(_BTN_EXPORT_READY)
        layout.addWidget(self.export_button)



        self.secondary_buttons_widget = SessionLinkRow()

        self.undo_button = QPushButton(tr("Undo last point"))
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip(
            tr("Removes the last point you placed on the object."))
        self.undo_button.clicked.connect(self._on_undo_clicked)
        self.undo_button.setVisible(False)
        self.undo_button.setMinimumHeight(BTN_SMALL_PX)
        self.undo_button.setStyleSheet(_QUIET_LINK_QSS)
        self.undo_button.setCursor(Qt.CursorShape.PointingHandCursor)


        from .widgets import native_key
        set_key_chip(self.undo_button, native_key(QKeySequence.StandardKey.Undo))
        self.secondary_buttons_widget.add_session_link(self.undo_button)

        self.clear_selection_button = QPushButton(tr("Clear selection"))
        self.clear_selection_button.setEnabled(False)
        self.clear_selection_button.setToolTip(
            tr("Removes the points and the shape you are working on. "
               "Saved polygons stay.")
            + "\n" + tr("Shortcut: {key}").format(key="C"))
        self.clear_selection_button.clicked.connect(
            self._on_clear_selection_clicked)
        self.clear_selection_button.setVisible(False)
        self.clear_selection_button.setMinimumHeight(BTN_SMALL_PX)
        self.clear_selection_button.setStyleSheet(_QUIET_LINK_QSS)
        self.clear_selection_button.setCursor(Qt.CursorShape.PointingHandCursor)
        set_key_chip(self.clear_selection_button, "C")
        self.secondary_buttons_widget.add_session_link(self.clear_selection_button)

        self.stop_button = QPushButton(tr("Stop segmentation"))
        self.stop_button.setToolTip(tr("End this segmentation session."))
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setVisible(False)
        self.stop_button.setMinimumHeight(BTN_SMALL_PX)
        self.stop_button.setStyleSheet(_QUIET_LINK_QSS)
        self.stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.secondary_buttons_widget.add_session_link(self.stop_button)

        self.secondary_buttons_widget.setVisible(False)
        layout.addWidget(self.secondary_buttons_widget)

        self.main_layout.addWidget(self.seg_widget)
