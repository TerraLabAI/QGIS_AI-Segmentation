"""Panel construction: title bar, welcome/setup/activation sections, pairing.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

import os

from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtGui import QKeySequence, QShortcut
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


from ..layer_tree_combobox import LayerTreeComboBox

from ...core.activation_manager import (
    has_tos_accepted,
    has_tos_locked,
)
from ...core.i18n import tr
from ...core.model_config import _IS_MACOS_X86, USE_SAM2
from ...core.venv_manager import CACHE_DIR
from .styles import (
    BRAND_BLUE,
    _BTN_BLUE,
    _BTN_EXPORT_DISABLED,
    _BTN_GHOST,
    _BTN_GRAY,
    _BTN_GREEN,
    _BTN_GREEN_AUTH,
    _BTN_PAIR_CANCEL,
    _BTN_PAIR_NEUTRAL,
    _BTN_RED,
    _INSTRUCTIONS_CARD_QSS,
)
from .guidance import (
    BLUE_TINT,
    GREEN_TINT,
    HINT_START_MANUAL,
    HINT_TRY_AUTOMATIC,
    DismissibleHint,
)
from .widgets import (
    Mode,
    _ModeSwitch,
    _Spinner,
    build_no_imagery_hero,
)


class DockBuildMixin:
    """Panel construction: title bar, welcome/setup/activation sections, pairing."""

    def _setup_title_bar(self):
        """Custom title bar with clickable TerraLab link and native buttons."""
        title_widget = QWidget()
        title_outer = QVBoxLayout(title_widget)
        title_outer.setContentsMargins(0, 0, 0, 0)
        title_outer.setSpacing(0)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(4, 0, 0, 0)
        title_row.setSpacing(0)

        title_label = QLabel(
            'AI Segmentation by '
            '<a href="https://terra-lab.ai?utm_source=qgis&utm_medium=plugin'
            '&utm_campaign=ai-segmentation&utm_content=title_link"'
            f' style="color: {BRAND_BLUE}; text-decoration: none;">TerraLab</a>'
        )
        title_label.setOpenExternalLinks(True)
        title_row.addWidget(title_label)
        title_row.addStretch()

        icon_size = self.style().pixelMetric(QStyle.PixelMetric.PM_SmallIconSize)

        float_btn = QToolButton()
        float_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarNormalButton))
        float_btn.setFixedSize(icon_size + 4, icon_size + 4)
        float_btn.setAutoRaise(True)
        float_btn.clicked.connect(lambda: self.setFloating(not self.isFloating()))
        title_row.addWidget(float_btn)

        close_btn = QToolButton()
        close_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton))
        close_btn.setFixedSize(icon_size + 4, icon_size + 4)
        close_btn.setAutoRaise(True)
        close_btn.clicked.connect(self.close)
        title_row.addWidget(close_btn)

        title_outer.addLayout(title_row)

        separator = QFrame()
        separator.setFrameShape(getattr(QFrame, "Shape", QFrame).HLine)
        separator.setFrameShadow(getattr(QFrame, "Shadow", QFrame).Sunken)
        title_outer.addWidget(separator)

        self.setTitleBarWidget(title_widget)

    def _setup_ui(self):
        self._setup_mode_switch()
        self._setup_welcome_section()
        self._setup_setup_section()
        self._setup_activation_section()
        self._setup_segmentation_section()
        self._setup_automatic_page()
        self.main_layout.addStretch()
        # Both bottom nudges live here, AFTER the stretch, so they sit pinned at
        # the bottom of the dock just above the footer CTAs instead of floating
        # under the Start button. They are mutually exclusive by mode: the
        # Try-Automatic band shows only on the Manual Start view
        # (_update_try_automatic_hint_visibility), the first-steps guide band only
        # on the Automatic Start step (_update_auto_tutorial_banner_visibility).
        self.main_layout.addWidget(self.try_automatic_hint)
        self.main_layout.addWidget(self.auto_tutorial_banner)
        self._setup_update_notification()
        self._setup_about_section()

    def _setup_mode_switch(self):
        """Create the Interactive / Automatic segmented control at the dock top."""
        self.mode_switch = _ModeSwitch(self._mode, self)
        self.mode_switch.mode_selected.connect(self._on_mode_selected)
        self.main_layout.addWidget(self.mode_switch)

    def _on_mode_selected(self, mode: Mode) -> None:
        """Handle mode-switch toggle. Blocks switch while a run is in progress."""
        if mode == Mode.AUTOMATIC and self._segmentation_active:
            self.mode_switch.blockSignals(True)
            self.mode_switch.set_mode(Mode.INTERACTIVE)
            self.mode_switch.blockSignals(False)
            msg = tr("Stop the active segmentation before switching modes.")
            self.mode_switch.setToolTip(msg)
            # T4: tooltips are hover-only, so also flash the reason as a 3 s
            # status line where the user is already looking.
            try:
                self.instructions_label.setText(msg)
                QTimer.singleShot(3000, self._update_instructions)
            except (RuntimeError, AttributeError):
                pass
            return
        if mode == Mode.INTERACTIVE and self._auto_run_active:
            self.mode_switch.blockSignals(True)
            self.mode_switch.set_mode(Mode.AUTOMATIC)
            self.mode_switch.blockSignals(False)
            msg = tr("Cancel the active detection before switching modes.")
            self.mode_switch.setToolTip(msg)
            # T4: same 3 s status line on the Automatic side; the restore
            # brings the live progress bar back if the run is still going.
            try:
                self.set_auto_status("info", msg)
                QTimer.singleShot(3000, self._restore_auto_run_status)
            except (RuntimeError, AttributeError):
                pass
            return
        self.mode_switch.setToolTip("")
        self._mode = mode
        # B1: programmatic callers (MCP set_mode, headless auto runs) enter
        # here directly, bypassing the segmented control's own toggle, so sync
        # its checked segment explicitly. set_mode blocks the button-group
        # signals and is a no-op for a real user click (already checked).
        self.mode_switch.set_mode(mode)
        self._auto_zone_too_large = False
        # No last-mode persistence: the dock always reopens on Automatic (D1).
        self.mode_changed.emit(mode)
        self._update_full_ui()

    def _on_manual_try_example(self) -> None:
        """Manual first-run hero 'Try it on an example'. The ready-made demo
        place is an Automatic showcase (basemap + zone + primed prompt, one
        click to Detect), so switch to Automatic and run it. The loaded basemap
        persists across the switch, so the user can flip back to Manual and
        click on the same imagery. Deferred a tick so the mode switch settles
        (mode_changed resets the Automatic flow to Start) before the demo loads."""
        self._on_mode_selected(Mode.AUTOMATIC)
        QTimer.singleShot(0, self.auto_demo_requested.emit)

    def _restore_auto_run_status(self) -> None:
        """Drop the 3 s blocked-mode-switch notice (T4): back to the live
        progress bar while a run is still in flight, else a clean idle line."""
        try:
            if self._auto_run_active:
                self.set_auto_status("progress")
                self._refresh_auto_progress_readout()
                self._set_auto_progress_visible(True)
            else:
                self.set_auto_status("idle")
        except (RuntimeError, AttributeError):
            pass

    def begin_refine_handoff(self, seed_total: int = 0) -> None:
        """Enter Manual mode to refine the Automatic results, preserving state.

        Drives the mode change by emitting mode_changed directly instead of going
        through the segmented control's mode_selected: a Manual session IS active
        during the round trip, so _on_mode_selected's run-active guard would
        otherwise refuse the switch. The plugin's _on_mode_changed handler sees the
        handoff flag and skips its usual destructive reset. ``seed_total`` is the
        detection count seeded into the session, for the banner progress counter.
        """
        self._refine_handoff = True
        self._handoff_seed_total = int(seed_total)
        self.refine_handoff_banner.setVisible(True)
        self.mode_switch.setEnabled(False)
        self.mode_switch.setToolTip(
            tr("Finish or go back to review to switch modes."))
        self._mode = Mode.INTERACTIVE
        self.mode_switch.blockSignals(True)
        self.mode_switch.set_mode(Mode.INTERACTIVE)
        self.mode_switch.blockSignals(False)
        self.mode_changed.emit(Mode.INTERACTIVE)
        self._update_full_ui()

    def end_refine_handoff(self, target_mode: Mode) -> None:
        """Leave the refine handoff, returning to target_mode (usually Automatic)."""
        self._refine_handoff = False
        self.set_refine_handoff_preparing(False)
        self.refine_handoff_banner.setVisible(False)
        self.mode_switch.setEnabled(True)
        self.mode_switch.setToolTip("")
        self._mode = target_mode
        self.mode_switch.blockSignals(True)
        self.mode_switch.set_mode(target_mode)
        self.mode_switch.blockSignals(False)
        self.mode_changed.emit(target_mode)
        self._update_full_ui()

    def set_protected_note(self, visible: bool) -> None:
        """Show/hide the truth note that hand-refined objects survive any
        confidence change.

        Confidence stays EDITABLE after a Manual refine: hand-edited objects are
        protected via geometry (_auto_protected_geoms), so re-filtering only
        re-derives the untouched auto detections. The old lock (disabling the
        slider) was dead; this replaces it with an honest note.
        """
        try:
            self._auto_conf_lock_note.setText(
                tr("Hand-refined objects are always kept, whatever the confidence."))
            self._auto_conf_lock_note.setVisible(bool(visible))
        except (RuntimeError, AttributeError):
            pass

    def set_refine_handoff_preparing(self, preparing: bool) -> None:
        """Reflect that the local model is still loading after a Refine click.

        The predictor loads asynchronously; while it comes up there is nothing to
        refine yet, so the banner says so and the Back button is disabled (there
        is no manual edit to harvest). The plugin completes the import and clears
        this state once the model is ready.
        """
        try:
            self._refine_banner_lbl.setText(
                tr("Preparing Manual mode, loading the local model...")
                if preparing else tr("Refining Automatic results"))
            # The purpose subtitle only makes sense once refining can begin;
            # hide it while the model is still loading (nothing to fine-tune yet).
            self._refine_banner_sub.setVisible(not preparing)
            # Back to review must stay clickable even while preparing: it only
            # needs _restore_auto_review_after_handoff (no predictor), so a stuck
            # or slow model load can never trap the user in the handoff.
        except (RuntimeError, AttributeError):
            pass

    def update_handoff_progress(self, kept: int) -> None:
        """Track how many seeded detections have been kept (turned green) so
        far. The count surfaces in the compact instructions hint ("N of M
        detections kept - click a blue detection to edit it"); the banner stays
        a plain "Refining Automatic results" (the old "· 8/43 kept" suffix read
        as cryptic)."""
        self._handoff_kept = int(kept)
        self._update_instructions()

    def _setup_welcome_section(self):
        """Setup the welcome section."""
        self.welcome_widget = QWidget()
        self.welcome_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(25, 118, 210, 0.08);
                border: 1px solid rgba(25, 118, 210, 0.2);
                border-radius: 4px;
            }
            QLabel { background: transparent; border: none; }
        """)
        layout = QVBoxLayout(self.welcome_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        self.welcome_title = QLabel(tr("Click Install to set up AI Segmentation"))
        self.welcome_title.setStyleSheet("font-weight: bold; font-size: 12px; color: palette(text);")
        layout.addWidget(self.welcome_title)

        # Hidden until the startup dependency check concludes an install is
        # actually needed (_update_full_ui gates it on the setup section being
        # on display). Showing it during the silent re-check made "Click
        # Install..." flash for half a second on every open of an already
        # installed plugin.
        self.welcome_widget.setVisible(False)
        self.main_layout.addWidget(self.welcome_widget)

    def _setup_setup_section(self):
        """Unified setup section: deps install + model download in one flow."""
        self.setup_group = QGroupBox("")
        self.setup_group.setStyleSheet(
            "QGroupBox { border: none; margin: 0; padding: 0; }"
        )
        layout = QVBoxLayout(self.setup_group)

        self.setup_status_label = QLabel(tr("Checking..."))
        self.setup_status_label.setStyleSheet("color: palette(text);")
        layout.addWidget(self.setup_status_label)

        self.setup_progress = QProgressBar()
        self.setup_progress.setRange(0, 100)
        self.setup_progress.setVisible(False)
        layout.addWidget(self.setup_progress)

        self.setup_progress_label = QLabel("")
        self.setup_progress_label.setStyleSheet("color: palette(text); font-size: 10px;")
        self.setup_progress_label.setVisible(False)
        layout.addWidget(self.setup_progress_label)

        self.install_button = QPushButton(tr("Install"))
        self.install_button.clicked.connect(self._on_install_clicked)
        self.install_button.setVisible(False)
        self.install_button.setMinimumHeight(34)
        self.install_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.install_button.setStyleSheet(_BTN_GREEN)
        layout.addWidget(self.install_button)

        self.install_path_label = QLabel(
            tr("Install path: {}").format(CACHE_DIR))
        self.install_path_label.setWordWrap(True)
        self.install_path_label.setStyleSheet(
            "color: palette(text);"
            "font-size: 10px;"
            "padding: 4px 6px;"
            "background-color: palette(base);"
            "border: 1px solid rgba(128, 128, 128, 0.35);"
            "border-radius: 3px;"
        )
        layout.addWidget(self.install_path_label)

        self.cancel_toggle = QToolButton()
        self.cancel_toggle.setText(tr("Cancel installation"))
        self.cancel_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.cancel_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.cancel_toggle.setStyleSheet(
            "color: palette(text); font-size: 10px; border: none;")
        self.cancel_toggle.setVisible(False)
        self.cancel_toggle.clicked.connect(self._toggle_cancel_button)
        layout.addWidget(self.cancel_toggle)

        self.cancel_button = QPushButton(tr("Cancel"))
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.cancel_button.setVisible(False)
        self.cancel_button.setStyleSheet(_BTN_RED)
        layout.addWidget(self.cancel_button)

        if not USE_SAM2:
            if _IS_MACOS_X86:
                sam1_text = tr("Intel Mac: using SAM1 (compatible with PyTorch 2.2)")
            else:
                sam1_text = tr("Update QGIS to 3.34+ for the latest AI model")
            sam1_info = QLabel(sam1_text)
            sam1_info.setStyleSheet("color: palette(text); font-size: 10px;")
            layout.addWidget(sam1_info)

        # Hidden until a state that needs it: set_dependency_status(ok=False)
        # and the model-download path show it; set_checkpoint_status(ok=True)
        # hides it. Starting visible made the "Checking..." / "Install path:"
        # lines flash on every open of an already installed plugin (the
        # startup quick-check hides them ~0.5 s later).
        self.setup_group.setVisible(False)
        self.main_layout.addWidget(self.setup_group)

    def _setup_activation_section(self):
        """One-click browser sign-in (mirrors AI Edit), with a discreet
        paste-a-key fallback for admins and browserless machines."""
        self.activation_group = QGroupBox()
        self.activation_group.setStyleSheet(
            "QGroupBox { border: none; margin: 0; padding: 0; }"
        )
        layout = QVBoxLayout(self.activation_group)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addSpacing(6)
        # Header row: copy + brand logo at the end (AI Edit ends its header
        # with the banana glyph; ours is the AI Segmentation mark).
        header_row = QHBoxLayout()
        header_row.setSpacing(7)
        header_row.addStretch(1)
        self._setup_header = QLabel(tr("Segment your map with AI"))
        self._setup_header.setStyleSheet(
            "font-weight: 600; font-size: 14px; color: palette(text);")
        header_row.addWidget(self._setup_header, 0, Qt.AlignmentFlag.AlignVCenter)
        from qgis.PyQt.QtGui import QPixmap
        _logo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))),
            "resources", "icons", "logo_title.png")
        _logo_pm = QPixmap(_logo_path)
        if not _logo_pm.isNull():
            _dpr = self.devicePixelRatioF()
            _logo_pm = _logo_pm.scaledToHeight(
                int(18 * _dpr), Qt.TransformationMode.SmoothTransformation)
            _logo_pm.setDevicePixelRatio(_dpr)
            _logo_lbl = QLabel()
            _logo_lbl.setPixmap(_logo_pm)
            header_row.addWidget(_logo_lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        header_row.addStretch(1)
        layout.addLayout(header_row)

        layout.addSpacing(14)

        # --- Primary: one tap to sign in (browser handoff, no copy-paste) ---
        self._connect_section = QWidget()
        connect_layout = QVBoxLayout(self._connect_section)
        connect_layout.setContentsMargins(0, 0, 0, 0)
        connect_layout.setSpacing(6)

        self._connect_btn = QPushButton(tr("Sign in / Sign up to start"))
        self._connect_btn.setToolTip(
            tr("Sign in via your browser to start using AI Segmentation"))
        self._connect_btn.setMinimumHeight(38)
        self._connect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._connect_btn.setStyleSheet(_BTN_GREEN_AUTH)
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        connect_layout.addWidget(self._connect_btn)

        # Signed-out is one neutral first step (before any mode concept), so the
        # hint stays mode-agnostic: no price, no local/cloud wording. Framed
        # card with two check rows.
        hint_card = QFrame()
        hint_card.setObjectName("signinHintCard")
        hint_card.setStyleSheet(
            "QFrame#signinHintCard {"
            " border: 1px solid rgba(128,128,128,0.35);"
            " border-radius: 6px;"
            " background-color: rgba(128,128,128,0.08); }")
        hint_card_layout = QVBoxLayout(hint_card)
        hint_card_layout.setContentsMargins(10, 8, 10, 8)
        hint_card_layout.setSpacing(5)

        def _hint_row(text):
            row = QHBoxLayout()
            row.setSpacing(7)
            check = QLabel("✓")
            check.setStyleSheet(
                "font-size: 11px; font-weight: 600; color: #43a047;"
                " border: none; background: transparent;")
            row.addWidget(check, 0, Qt.AlignmentFlag.AlignTop)
            label = QLabel(text)
            label.setWordWrap(True)
            label.setStyleSheet(
                "font-size: 11px; color: palette(text);"
                " border: none; background: transparent;")
            row.addWidget(label, 1)
            hint_card_layout.addLayout(row)
            return label

        self._connect_hint_label = _hint_row(
            tr("Free account - sign up takes 15 seconds in your browser."))
        _hint_row(
            tr("Then segment any imagery: point and click, or fully automatic."))
        connect_layout.addWidget(hint_card)

        layout.addWidget(self._connect_section)

        # --- Waiting state: shown while the browser handoff is in progress ---
        self._pairing_wait_section = QWidget()
        wait_layout = QVBoxLayout(self._pairing_wait_section)
        wait_layout.setContentsMargins(0, 4, 0, 0)
        wait_layout.setSpacing(12)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        status_row.addStretch(1)
        self._pairing_spinner = _Spinner(16)
        status_row.addWidget(self._pairing_spinner, 0, Qt.AlignmentFlag.AlignVCenter)
        self._pairing_status = QLabel(tr("Waiting for your browser sign-in..."))
        self._pairing_status.setWordWrap(True)
        self._pairing_status.setStyleSheet("font-size: 12px; color: palette(text);")
        status_row.addWidget(self._pairing_status, 0, Qt.AlignmentFlag.AlignVCenter)
        status_row.addStretch(1)
        wait_layout.addLayout(status_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._pairing_reopen_btn = QPushButton(tr("Open again"))
        self._pairing_reopen_btn.setToolTip(tr("Didn't open? Open the page again"))
        self._pairing_reopen_btn.setMinimumHeight(28)
        self._pairing_reopen_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pairing_reopen_btn.setStyleSheet(_BTN_PAIR_NEUTRAL)
        self._pairing_reopen_btn.clicked.connect(self._on_pairing_reopen_clicked)
        btn_row.addWidget(self._pairing_reopen_btn)

        self._pairing_cancel_btn = QPushButton(tr("Cancel"))
        self._pairing_cancel_btn.setMinimumHeight(28)
        self._pairing_cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pairing_cancel_btn.setStyleSheet(_BTN_PAIR_CANCEL)
        self._pairing_cancel_btn.clicked.connect(self._on_pairing_cancel_clicked)
        btn_row.addWidget(self._pairing_cancel_btn)
        wait_layout.addLayout(btn_row)

        self._pairing_wait_section.setVisible(False)
        layout.addWidget(self._pairing_wait_section)

        # One timer rotates the spinner while waiting. Parented to the dock
        # (segfault-safe) and stopped the moment the wait section hides.
        self._pairing_anim_timer = QTimer(self)
        self._pairing_anim_timer.setInterval(80)
        self._pairing_anim_timer.timeout.connect(self._pairing_spinner.advance)
        self._pending_pairing_code = ""

        # Browser pairing is the only sign-in path. (The old "Use an activation
        # key" manual-entry fallback was removed; keys still back the client
        # auth, but a human never types one into the UI.)

        self.activation_message_label = QLabel("")
        self.activation_message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.activation_message_label.setWordWrap(True)
        self.activation_message_label.setVisible(False)
        self.activation_message_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.activation_message_label)

        self.activation_group.setVisible(False)
        self.main_layout.addWidget(self.activation_group)

    # --- One-click connect (browser pairing handoff) ----------------------

    def _on_connect_clicked(self):
        """Start the one-click browser handoff. Mints a high-entropy pairing
        code; the plugin opens the browser and polls until it gets the key."""
        import secrets
        self._pending_pairing_code = secrets.token_urlsafe(32)
        self.show_pairing_waiting()
        self.pairing_requested.emit(self._pending_pairing_code)

    def _on_pairing_reopen_clicked(self):
        """Re-open the browser with the SAME code (do not mint a new one)."""
        if self._pending_pairing_code:
            self.pairing_requested.emit(self._pending_pairing_code)

    def _on_pairing_cancel_clicked(self):
        self.pairing_cancel_requested.emit(self._pending_pairing_code)
        self._pending_pairing_code = ""
        self.show_pairing_idle()

    def show_pairing_waiting(self):
        """Switch the onboarding into the 'waiting for browser' state."""
        self._connect_section.setVisible(False)
        self.activation_message_label.setVisible(False)
        self._pairing_wait_section.setVisible(True)
        self._pairing_anim_timer.start()

    def _stop_pairing_wait(self):
        """Hide the waiting section and stop its animation timer."""
        self._pairing_anim_timer.stop()
        self._pairing_wait_section.setVisible(False)

    def show_pairing_idle(self):
        """Return to the idle onboarding (Connect button visible)."""
        self._stop_pairing_wait()
        self._connect_section.setVisible(True)

    def _setup_segmentation_section(self):
        self.seg_widget = QWidget()
        layout = QVBoxLayout(self.seg_widget)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        # Refine-handoff banner: shown only while Manual mode is refining the
        # Automatic results. A quiet blue info-card (temporary = blue, matching
        # the pending-blue seeds on the canvas) with a clear title, a muted
        # one-line subtitle that carries the purpose, and a secondary (ghost)
        # Back-to-review action, so it informs without shouting the way a solid
        # box + bright button did. Hidden in a normal Manual-from-scratch session.
        self.refine_handoff_banner = QWidget()
        self.refine_handoff_banner.setObjectName("refineHandoffBanner")
        self.refine_handoff_banner.setStyleSheet(
            "QWidget#refineHandoffBanner { background-color: rgba(30,136,229,0.08);"
            " border: 1px solid rgba(30,136,229,0.35); border-radius: 4px; }"
            "QLabel { background: transparent; border: none; color: palette(text); }"
        )
        _banner_col = QVBoxLayout(self.refine_handoff_banner)
        _banner_col.setContentsMargins(10, 8, 8, 8)
        _banner_col.setSpacing(3)
        _banner_row = QHBoxLayout()
        _banner_row.setContentsMargins(0, 0, 0, 0)
        _banner_row.setSpacing(8)
        self._refine_banner_lbl = QLabel(tr("Refining Automatic results"))
        self._refine_banner_lbl.setStyleSheet(
            "font-weight: bold; font-size: 12px;")
        _banner_row.addWidget(self._refine_banner_lbl)
        _banner_row.addStretch()
        self.back_to_review_btn = QPushButton(tr("Back to review"))
        self.back_to_review_btn.setStyleSheet(_BTN_GHOST)
        self.back_to_review_btn.setMinimumHeight(28)
        self.back_to_review_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_to_review_btn.setToolTip(
            tr("Merges your edits back into the review. Nothing is exported yet."))
        self.back_to_review_btn.clicked.connect(self.back_to_review_requested.emit)
        _banner_row.addWidget(self.back_to_review_btn)
        _banner_col.addLayout(_banner_row)
        self._refine_banner_sub = QLabel(
            tr("Fine-tune the detections, then go back to review to export."))
        self._refine_banner_sub.setWordWrap(True)
        self._refine_banner_sub.setStyleSheet(
            "font-size: 11px; color: rgba(128,128,128,0.95);")
        _banner_col.addWidget(self._refine_banner_sub)
        self.refine_handoff_banner.setVisible(False)
        layout.addWidget(self.refine_handoff_banner)

        layer_label = QLabel(tr("Select a Raster Layer to Segment:"))
        layer_label.setStyleSheet("font-weight: bold; color: palette(text);")
        layout.addWidget(layer_label)
        self.layer_label = layer_label

        self.layer_combo = LayerTreeComboBox()
        self.layer_combo.layerChanged.connect(self._on_layer_changed)
        self.layer_combo.setToolTip(tr("Select a raster layer (GeoTIFF, WMS, XYZ tiles, etc.)"))
        self.layer_combo.setStyleSheet("QComboBox { color: palette(text); }")
        from qgis.PyQt.QtWidgets import QSizePolicy
        self.layer_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.layer_combo.setMinimumWidth(0)
        layout.addWidget(self.layer_combo)

        # No-rasters state = the SAME first-run hero the Automatic page shows
        # (shared helper), so both modes read identically: the imagery is the
        # user's to bring, with the demo place as the reassurance fallback. This
        # replaces the old yellow "No raster layer found" alarm. The Manual
        # example reuses the Automatic showcase (see _on_manual_try_example).
        self.no_rasters_widget, self.manual_demo_btn = build_no_imagery_hero(
            on_demo=self._on_manual_try_example,
        )
        self.no_rasters_widget.setVisible(False)
        layout.addWidget(self.no_rasters_widget, 1)

        # Dynamic instruction label - styled as a card (slightly darker gray than refine panel)
        self.instructions_label = QLabel("")
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setMinimumHeight(70)
        self.instructions_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.instructions_label.setStyleSheet(_INSTRUCTIONS_CARD_QSS)
        self._instructions_compact = False
        self.instructions_label.setVisible(False)
        layout.addWidget(self.instructions_label)

        # Container for start button
        self.start_container = QWidget()
        start_layout = QVBoxLayout(self.start_container)
        start_layout.setContentsMargins(0, 8, 0, 0)
        start_layout.setSpacing(6)

        # Terms + Privacy consent - required to run a segmentation.
        # Placed here (not on activation) so the user only sees it when they
        # are about to actually use the service. Hidden permanently once the
        # user has clicked Start for the first time (see lock_tos): at that
        # point consent is considered irrevocably given.
        _tos_terms_url = (
            "https://terra-lab.ai/terms-of-use"
            "?utm_source=qgis&utm_medium=plugin"
            "&utm_campaign=ai-segmentation&utm_content=consent_terms"
        )
        _tos_privacy_url = (
            "https://terra-lab.ai/privacy-policy"
            "?utm_source=qgis&utm_medium=plugin"
            "&utm_campaign=ai-segmentation&utm_content=consent_privacy"
        )
        self.tos_container = QWidget()
        tos_row = QHBoxLayout(self.tos_container)
        tos_row.setContentsMargins(0, 0, 0, 0)
        tos_row.setSpacing(4)
        # Use Qt's native QCheckBox rendering - the previous custom stylesheet
        # only toggled the indicator's background colour without rendering a
        # checkmark, so users couldn't tell at a glance whether the box was
        # checked. Native rendering matches QGIS's own checkbox conventions.
        self.tos_checkbox = QCheckBox()
        self.tos_checkbox.setChecked(has_tos_accepted())
        self.tos_checkbox.toggled.connect(self._on_tos_toggled)
        tos_row.addWidget(self.tos_checkbox, 0)
        self.tos_label = QLabel(
            tr('I agree to the <a href="{terms}">Terms</a> '
               'and <a href="{privacy}">Privacy Policy</a>').format(
                terms=_tos_terms_url, privacy=_tos_privacy_url
            )
        )
        self.tos_label.setOpenExternalLinks(True)
        self.tos_label.setWordWrap(True)
        self.tos_label.setStyleSheet("font-size: 11px; color: palette(text);")
        tos_row.addWidget(self.tos_label, 1)
        # If the user has already started a segmentation in the past, the
        # consent is sealed and the row disappears forever.
        if has_tos_locked():
            self.tos_container.setVisible(False)
        start_layout.addWidget(self.tos_container)

        self.start_button = QPushButton(tr("Start Manual AI Segmentation"))
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._on_start_clicked)
        self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        # Same prominence as AI Edit's "Launch AI Edit" button.
        self.start_button.setMinimumHeight(36)
        self.start_button.setStyleSheet(_BTN_GREEN)
        self.start_button.setToolTip(
            tr("Accept the Terms and Privacy Policy to enable segmentation.")
        )
        start_layout.addWidget(self.start_button)

        # "What is this mode for" caption: a quiet framed card
        # under the Start button. One plain sentence; no free/paid wording,
        # no cloud/local wording. Dismissible (small x), so a
        # returning user can clear the guidance; re-enable from Account Settings.
        # Lives inside start_container so it hides with the button once a
        # session starts.
        self.manual_start_caption = DismissibleHint(
            HINT_START_MANUAL,
            tr("Click an object on the map and the AI outlines it. "
               "You go one object at a time, checking and saving each "
               "polygon yourself."),
            tint=GREEN_TINT,
        )
        start_layout.addWidget(self.manual_start_caption)

        # Last-session value recap (mirrors the Automatic Start page card):
        # one quiet line under the caption summarizing what
        # the last Manual export produced, so the value does not vanish when
        # the session closes. When the caption is dismissed Qt collapses it and
        # the recap sits right under the Start button. Lives in start_container
        # so it hides with the button during a session. Session only; filled by
        # set_manual_last_run_recap().
        self.manual_last_run_recap = QLabel()
        self.manual_last_run_recap.setWordWrap(True)
        self.manual_last_run_recap.setTextFormat(Qt.TextFormat.PlainText)
        self.manual_last_run_recap.setStyleSheet(
            "font-size: 11px; color: palette(text);"
            " border: 1px solid rgba(120, 144, 156, 90);"
            " border-radius: 6px; padding: 6px 8px;"
            " background: rgba(120, 144, 156, 22);"
        )
        self.manual_last_run_recap.setVisible(False)
        start_layout.addWidget(self.manual_last_run_recap)

        # Cross-sell nudge: a compact dismissible blue band pointing
        # at Automatic mode. Built here (with the rest of the Manual page) but NOT
        # added to start_container: it is pinned to the very BOTTOM of the dock,
        # just above the footer CTAs, via main_layout after the stretch.
        # Its visibility is driven by
        # _update_try_automatic_hint_visibility so it only shows on the Manual
        # Start view, never during a session or in Automatic mode.
        self.try_automatic_hint = DismissibleHint(
            HINT_TRY_AUTOMATIC,
            tr("New: Automatic mode finds every object in a zone at once."),
            tint=BLUE_TINT,
            action_text=tr("Try Automatic"),
            visibility_gate=self._should_show_try_automatic,
        )
        self.try_automatic_hint.action.connect(
            lambda: self._on_mode_selected(Mode.AUTOMATIC))

        # Keyboard shortcut G to start segmentation (scoped to dock + children)
        self.start_shortcut = QShortcut(QKeySequence("G"), self)
        self.start_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.start_shortcut.activated.connect(self._on_start_shortcut)

        # Automatic-flow keyboard parity (mirrors AI Edit): Escape exits the
        # selection view back to Start; Enter detects. WindowShortcut so they
        # work whether focus is in the dock or on the canvas; a focus/mode guard
        # (_is_auto_for_us) keeps them from hijacking unrelated QGIS tools.
        self.auto_escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.auto_escape_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.auto_escape_shortcut.activated.connect(self._on_auto_escape_shortcut)
        self.auto_enter_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Return), self)
        self.auto_enter_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.auto_enter_shortcut.activated.connect(self._on_auto_enter_shortcut)
        self.auto_enter_shortcut_kp = QShortcut(QKeySequence(Qt.Key.Key_Enter), self)
        self.auto_enter_shortcut_kp.setContext(Qt.ShortcutContext.WindowShortcut)
        self.auto_enter_shortcut_kp.activated.connect(self._on_auto_enter_shortcut)

        layout.addWidget(self.start_container)

        # Collapsible Refine mask panel
        self._setup_refine_panel(layout)

        # Primary action buttons (reduced height)
        self.save_mask_button = QPushButton(tr("Save polygon (S)"))
        self.save_mask_button.clicked.connect(self._on_save_polygon_clicked)
        self.save_mask_button.setVisible(False)
        self.save_mask_button.setEnabled(False)
        self.save_mask_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_mask_button.setMinimumHeight(34)
        self.save_mask_button.setStyleSheet(_BTN_BLUE)
        self.save_mask_button.setToolTip(
            tr("Keeps this polygon in your session. Export writes all kept "
               "polygons to a layer.")
        )
        layout.addWidget(self.save_mask_button)

        self.export_button = QPushButton(tr("Export polygon to a layer"))
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setVisible(False)
        self.export_button.setEnabled(False)
        self.export_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_button.setMinimumHeight(34)
        self.export_button.setStyleSheet(_BTN_EXPORT_DISABLED)
        layout.addWidget(self.export_button)

        # Secondary action buttons (small, horizontal)
        secondary_layout = QHBoxLayout()
        secondary_layout.setSpacing(8)

        self.undo_button = QPushButton(tr("Undo last point"))
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._on_undo_clicked)
        self.undo_button.setVisible(False)  # Hidden until segmentation starts
        self.undo_button.setMinimumHeight(30)
        self.undo_button.setStyleSheet(_BTN_GRAY)
        secondary_layout.addWidget(self.undo_button, 1)  # stretch factor 1

        self.stop_button = QPushButton(tr("Stop segmentation"))
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setVisible(False)  # Hidden until segmentation starts
        self.stop_button.setMinimumHeight(30)
        self.stop_button.setStyleSheet(_BTN_GRAY)
        secondary_layout.addWidget(self.stop_button, 1)  # stretch factor 1 for same width

        self.secondary_buttons_widget = QWidget()
        self.secondary_buttons_widget.setLayout(secondary_layout)
        self.secondary_buttons_widget.setVisible(False)
        layout.addWidget(self.secondary_buttons_widget)

        self.main_layout.addWidget(self.seg_widget)
