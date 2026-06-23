from __future__ import annotations

import os
import sys

from qgis.core import QgsProject
from qgis.PyQt.QtCore import QT_VERSION, Qt, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QKeySequence
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .layer_tree_combobox import LayerTreeComboBox

# Collapsed height for refine panel title (just enough to show the arrow + label)
_REFINE_COLLAPSED_HEIGHT = 25

# Brand colors (Material Design 2 - shared with AI Edit, same values).
# Primary CTA buttons keep the material green; it reads as THE action color.
# Every other green accent uses the TerraLab leaf green below.
BTN_GREEN = "#43a047"
BTN_GREEN_HOVER = "#2e7d32"
BTN_GREEN_DISABLED = "#c8e6c9"

# Brand accent green = the QGIS green (terralab-website --qgis-green). Lime
# fills use BRAND_GREEN; green text on light backgrounds uses BRAND_GREEN_TEXT.
BRAND_GREEN = "#8bac27"
BRAND_GREEN_TEXT = "#4d7c0f"
BRAND_BLUE = "#1e88e5"
BRAND_BLUE_HOVER = "#1976d2"
BRAND_RED = "#d32f2f"
BRAND_RED_HOVER = "#b71c1c"
BRAND_GRAY = "#757575"
BRAND_GRAY_HOVER = "#616161"
BRAND_DISABLED = "#b0bec5"
DISABLED_TEXT = "#666666"
ERROR_TEXT = "#ef5350"
SUCCESS_TEXT = "#66bb6a"

# Design-system QSS constants, identical to AI Edit (dock_widget.py).
# border: none kills the native frame on dark themes; black text on the
# mid-tone fills keeps AA contrast on both light and dark QGIS themes.
_BTN_GREEN = (
    f"QPushButton {{ background-color: {BTN_GREEN}; color: #000000;"
    f" padding: 8px 16px; border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: {BTN_GREEN_HOVER}; color: #000000; }}"
    f"QPushButton:disabled {{ background-color: {BTN_GREEN_DISABLED};"
    f" color: {DISABLED_TEXT}; }}"
)

_BTN_GREEN_AUTH = (
    f"QPushButton {{ background-color: {BTN_GREEN}; color: #000000;"
    f" border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: {BTN_GREEN_HOVER}; }}"
    f"QPushButton:disabled {{ background-color: {BRAND_DISABLED};"
    f" color: {DISABLED_TEXT}; }}"
)

_BTN_BLUE = (
    f"QPushButton {{ background-color: {BRAND_BLUE}; color: #000000;"
    f" padding: 6px 12px; border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: {BRAND_BLUE_HOVER}; color: #000000; }}"
    f"QPushButton:disabled {{ background-color: {BRAND_DISABLED};"
    f" color: {DISABLED_TEXT}; }}"
)

_BTN_BLUE_AUTH = (
    f"QPushButton {{ background-color: {BRAND_BLUE}; color: #000000;"
    f" border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: {BRAND_BLUE_HOVER}; }}"
    f"QPushButton:disabled {{ background-color: {BRAND_DISABLED}; }}"
)

_BTN_GRAY = (
    f"QPushButton {{ background-color: {BRAND_GRAY}; color: #000000;"
    f" padding: 4px 8px; border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: {BRAND_GRAY_HOVER}; color: #000000; }}"
    f"QPushButton:disabled {{ background-color: {BRAND_DISABLED}; color: {DISABLED_TEXT}; }}"
)

_BTN_RED = (
    f"QPushButton {{ background-color: rgba(211,47,47,0.12); color: {BRAND_RED};"
    f" padding: 6px 12px; border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: rgba(211,47,47,0.22); }}"
)

_BTN_EXPORT_READY = (
    f"QPushButton {{ background-color: {BTN_GREEN}; color: #000000;"
    f" padding: 6px 12px; border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: {BTN_GREEN_HOVER}; color: #000000; }}"
)

_BTN_EXPORT_DISABLED = (
    f"QPushButton {{ background-color: {BRAND_DISABLED}; color: {DISABLED_TEXT};"
    f" padding: 6px 12px; border: none; border-radius: 4px; }}"
)

# Compact filled buttons for the browser-handoff waiting state. Both carry a
# soft tint (never transparent): neutral for "open again", red for "cancel".
_BTN_PAIR_NEUTRAL = (
    "QPushButton { background-color: rgba(128,128,128,0.16); color: palette(text);"
    " border: none; border-radius: 4px; }"
    "QPushButton:hover { background-color: rgba(128,128,128,0.28); }"
)
_BTN_PAIR_CANCEL = (
    f"QPushButton {{ background-color: rgba(211,47,47,0.12); color: {BRAND_RED};"
    f" border: none; border-radius: 4px; }}"
    f"QPushButton:hover {{ background-color: rgba(211,47,47,0.22); }}"
)

# Footer icon buttons (gear / question mark) — slim toolbuttons that mirror
# AI Edit. Hover state is driven by a dynamic `hover` property rather than
# Qt's :hover pseudo, because with InstantPopup menus Qt fails to fire a
# Leave event once the menu closes, so the button stays tinted until the
# next real mouse move. ``_FooterIconButton.set_hovered(False)`` resets it.
# The TerraLab leaf-green ``[active]`` tint marks "this menu is open".
_FOOTER_ICON_BTN_STYLE = (
    "QToolButton { background: transparent; border: none; padding: 6px 10px;"
    " font-size: 22px; font-weight: 600;"
    " color: palette(text); border-radius: 4px; }"
    'QToolButton[hover="true"] { background: rgba(128,128,128,0.15); }'
    'QToolButton[active="true"] { background: rgba(139, 172, 39, 0.55); }'
    'QToolButton[active="true"][hover="true"] { background: rgba(139, 172, 39, 0.75); }'
    "QToolButton::menu-indicator { image: none; width: 0; }"
)

# Help (question mark) hovers green — the leaf tint invites the user toward
# Tutorial / Report a problem instead of reading as a neutral icon.
_HELP_ICON_BTN_STYLE = (
    "QToolButton { background: transparent; border: none; padding: 6px 10px;"
    " font-size: 22px; font-weight: 600;"
    " color: palette(text); border-radius: 4px; }"
    'QToolButton[hover="true"] { background: rgba(139, 172, 39, 0.35); }'
    'QToolButton[active="true"] { background: rgba(139, 172, 39, 0.55); }'
    'QToolButton[active="true"][hover="true"] { background: rgba(139, 172, 39, 0.75); }'
    "QToolButton::menu-indicator { image: none; width: 0; }"
)

_FOOTER_MENU_STYLE = (
    "QMenu { background: palette(base); border: 1px solid rgba(128,128,128,0.35);"
    " border-radius: 6px; padding: 4px; }"
    "QMenu::item { background: transparent; padding: 6px 14px; border-radius: 4px;"
    " color: palette(text); }"
    "QMenu::item:selected { background: rgba(128,128,128,0.18); }"
)

# Footer cross-promo CTA — same flat/transparent + hover-tint look as the gear
# and help buttons, but sized for a label (11px) instead of a 22px glyph so the
# text reads as a small button rather than dwarfing the icons beside it.
_FOOTER_CTA_BTN_STYLE = (
    "QToolButton { background: transparent; border: none; padding: 6px 10px;"
    " font-size: 11px; font-weight: 600;"
    " color: palette(text); border-radius: 4px; }"
    'QToolButton[hover="true"] { background: rgba(128,128,128,0.15); }'
)


from ..core.activation_manager import (  # noqa: E402
    get_tutorial_url,
    has_tos_accepted,
    has_tos_locked,
    is_plugin_activated,
    lock_tos,
    set_tos_accepted,
)
from ..core.i18n import tr  # noqa: E402
from ..core.model_config import _IS_MACOS_X86, USE_SAM2  # noqa: E402
from ..core.venv_manager import CACHE_DIR  # noqa: E402


class _FooterIconButton(QToolButton):
    """QToolButton whose hover tint is driven by an explicit ``hover``
    dynamic property rather than Qt's :hover pseudo-state.

    With InstantPopup menus, Qt fails to fire the synthetic Leave event
    after the menu closes, so the button stays visually pressed/hovered
    until the next real mouse move. Tracking hover ourselves lets us
    force-reset it on ``menu.aboutToHide``.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("hover", False)
        self.setProperty("active", False)

    def _repolish(self) -> None:
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_hovered(self, hovered: bool) -> None:
        if bool(self.property("hover")) == hovered:
            return
        self.setProperty("hover", hovered)
        self._repolish()

    def set_active(self, active: bool) -> None:
        """Leaf-green tint while the attached menu is open (mirrors AI Edit)."""
        if bool(self.property("active")) == active:
            return
        self.setProperty("active", active)
        self._repolish()

    def enterEvent(self, event):  # noqa: N802
        self.set_hovered(True)
        super().enterEvent(event)

    def leaveEvent(self, event):  # noqa: N802
        self.set_hovered(False)
        super().leaveEvent(event)


class _Spinner(QWidget):
    """A small rotating arc, the conventional 'busy' indicator. Driven by an
    external QTimer calling ``advance()`` so one timer can be paused with the
    section it belongs to. Mirrors AI Edit's pairing spinner."""

    def __init__(self, diameter: int = 16, parent=None):
        super().__init__(parent)
        self._angle = 0
        self._d = diameter
        self.setFixedSize(diameter, diameter)

    def advance(self):
        self._angle = (self._angle + 30) % 360
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt signature
        from qgis.PyQt.QtCore import QRectF
        from qgis.PyQt.QtGui import QColor, QPainter, QPen
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        margin = 2.0
        rect = QRectF(margin, margin, self._d - 2 * margin, self._d - 2 * margin)
        pen = QPen(QColor(BRAND_GREEN))
        pen.setWidthF(2.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, int(-self._angle * 16), 270 * 16)
        painter.end()


class AISegmentationDockWidget(QDockWidget):

    install_requested = pyqtSignal()
    cancel_install_requested = pyqtSignal()
    start_segmentation_requested = pyqtSignal(object)
    undo_requested = pyqtSignal()
    save_polygon_requested = pyqtSignal()
    settings_clicked = pyqtSignal()
    export_layer_requested = pyqtSignal()
    stop_segmentation_requested = pyqtSignal()
    pairing_requested = pyqtSignal(str)        # one-click connect: emits the minted pairing code
    pairing_cancel_requested = pyqtSignal(str)  # user cancelled the browser handoff (emits the code)
    # simplify, smooth, expand, fill_holes
    # (min_area is auto-computed server-side and no longer in the UI)
    refine_settings_changed = pyqtSignal(int, int, int, bool)

    def __init__(self, parent=None):
        super().__init__(tr("AI Segmentation by TerraLab"), parent)
        # A stable objectName lets QGIS persist and restore the dock's
        # open/closed state and position across sessions (same as AI Edit).
        self.setObjectName("AISegmentationDockWidget")

        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.setMinimumWidth(260)

        self._setup_title_bar()

        # Initialize state variables that are needed during UI setup
        self._refine_expanded = True  # Refine panel expanded by default

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setSpacing(8)
        self.main_layout.setContentsMargins(8, 8, 8, 8)

        self._setup_ui()

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.main_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidget(scroll_area)

        self._dependencies_ok = False
        self._checkpoint_ok = False
        self._segmentation_active = False
        self._has_mask = False
        self._saved_polygon_count = 0
        self._positive_count = 0
        self._negative_count = 0
        self._plugin_activated = is_plugin_activated()
        self._activation_popup_shown = False  # Track if popup was shown
        self._segmentation_layer_id = None  # Track which layer we're segmenting
        # Note: _refine_expanded is initialized before _setup_ui() call

        # Smooth progress animation timer for long-running installs
        self._progress_timer = QTimer(self)
        self._progress_timer.timeout.connect(self._on_progress_tick)
        self._current_progress = 0
        self._target_progress = 0
        self._install_start_time = None
        self._last_percent = 0
        self._last_percent_time = None
        self._creep_counter = 0
        # Debounce timer for refinement sliders
        self._refine_debounce_timer = QTimer(self)
        self._refine_debounce_timer.setSingleShot(True)
        self._refine_debounce_timer.timeout.connect(self._emit_refine_changed)

        # Debounce timer for layer visibility changes (fires per-node in groups)
        self._visibility_debounce_timer = QTimer(self)
        self._visibility_debounce_timer.setSingleShot(True)
        self._visibility_debounce_timer.timeout.connect(self._update_ui_state)

        # Connect to project layer signals for dynamic updates
        QgsProject.instance().layersAdded.connect(self._on_layers_added)
        QgsProject.instance().layersRemoved.connect(self._on_layers_removed)
        # Refresh layer dropdown when layer visibility is toggled (debounced)
        QgsProject.instance().layerTreeRoot().visibilityChanged.connect(
            self._on_layer_visibility_changed)

        # Update UI state
        self._update_full_ui()

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
        self._setup_welcome_section()
        self._setup_setup_section()
        self._setup_activation_section()
        self._setup_segmentation_section()
        self.main_layout.addStretch()
        self._setup_update_notification()
        self._setup_about_section()

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
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
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

        # "Unlimited free plan" (not "free forever"): keeps the promise honest
        # once the paid AI Segmentation Pro tier ships alongside it.
        connect_hint = QLabel(tr("Unlimited free plan, runs locally"))
        connect_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        connect_hint.setWordWrap(True)
        connect_hint.setStyleSheet("font-size: 11px; color: palette(text);")
        connect_layout.addWidget(connect_hint)

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
        self._pairing_status = QLabel(tr("Waiting for you to sign in in your browser"))
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

        # Copy the connect link so the user can finish sign-in in a different
        # browser (e.g. their default has no Google session). Standard CLI
        # device-flow fallback ("open browser, or copy this link").
        self._pairing_copy_btn = QPushButton(tr("Link not opening? Copy link"))
        self._pairing_copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pairing_copy_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none;"
            " color: palette(text); font-size: 11px; padding: 2px;"
            " text-decoration: underline; }"
        )
        self._pairing_copy_btn.clicked.connect(self._on_pairing_copy_clicked)
        wait_layout.addWidget(self._pairing_copy_btn, 0, Qt.AlignmentFlag.AlignCenter)

        self._pairing_wait_section.setVisible(False)
        layout.addWidget(self._pairing_wait_section)

        # One timer rotates the spinner while waiting. Parented to the dock
        # (segfault-safe) and stopped the moment the wait section hides.
        self._pairing_anim_timer = QTimer(self)
        self._pairing_anim_timer.setInterval(80)
        self._pairing_anim_timer.timeout.connect(self._pairing_spinner.advance)
        self._pending_pairing_code = ""
        self._pairing_link = ""
        # Explicit pairing state. _update_full_ui() must key off this, NOT off
        # the wait section's visibility, which races UI refreshes.
        self._pairing_active = False

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

    def set_pairing_link(self, url):
        """Store the connect URL so the copy-link button can offer it (the URL
        is built plugin-side; the dock only displays it)."""
        self._pairing_link = url or ""

    def _on_pairing_copy_clicked(self):
        """Copy the connect link so the user can finish sign-in in another
        browser. Brief 'Copied!' feedback, then restore the label."""
        if not self._pairing_link:
            return
        from qgis.PyQt.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return
        clipboard.setText(self._pairing_link)
        self._pairing_copy_btn.setText(tr("Copied!"))
        QTimer.singleShot(
            1400, lambda: self._pairing_copy_btn.setText(tr("Link not opening? Copy link"))
        )

    def _on_pairing_cancel_clicked(self):
        self.pairing_cancel_requested.emit(self._pending_pairing_code)
        self._pending_pairing_code = ""
        self.show_pairing_idle()

    def show_pairing_waiting(self):
        """Switch the onboarding into the 'waiting for browser' state."""
        self._pairing_active = True
        self._pairing_status.setText(tr("Waiting for you to sign in in your browser"))
        self._connect_section.setVisible(False)
        self.activation_message_label.setVisible(False)
        self._pairing_wait_section.setVisible(True)
        self._pairing_anim_timer.start()

    def show_pairing_browser_seen(self):
        """The server saw the browser reach /connect: reassure the user."""
        if self._pairing_active:
            self._pairing_status.setText(
                tr("Browser page open. Finish signing in to connect."))

    def show_pairing_stalled_hint(self):
        """Long wait and the browser was never seen server-side: surface the
        recovery paths instead of an endless spinner."""
        if self._pairing_active:
            self._pairing_status.setText(tr(
                "Still waiting. If the page did not open or shows an error, "
                "click Open again or copy the link into another browser."))

    def _stop_pairing_wait(self):
        """Hide the waiting section and stop its animation timer."""
        self._pairing_active = False
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

        # Warning container with icon and text - yellow background with dark text
        self.no_rasters_widget = QWidget()
        self.no_rasters_widget.setStyleSheet(
            "QWidget { background-color: rgb(255, 230, 150); "
            "border: 1px solid rgba(255, 152, 0, 0.6); border-radius: 4px; }"
            "QLabel { background: transparent; border: none; color: #333333; }"
        )
        no_rasters_layout = QHBoxLayout(self.no_rasters_widget)
        no_rasters_layout.setContentsMargins(8, 8, 8, 8)
        no_rasters_layout.setSpacing(8)

        # Warning icon from Qt standard icons
        warning_icon_label = QLabel()
        style = self.no_rasters_widget.style()
        warning_icon = style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
        _ico = style.pixelMetric(QStyle.PixelMetric.PM_SmallIconSize)
        warning_icon_label.setPixmap(warning_icon.pixmap(_ico, _ico))
        warning_icon_label.setFixedSize(_ico, _ico)
        no_rasters_layout.addWidget(warning_icon_label, 0, Qt.AlignmentFlag.AlignTop)

        self.no_rasters_label = QLabel(
            tr("No raster layer found. Add a GeoTIFF, image file, "
               "or online layer (WMS, XYZ) to your project."))
        self.no_rasters_label.setWordWrap(True)
        no_rasters_layout.addWidget(self.no_rasters_label, 1)

        self.no_rasters_widget.setVisible(False)
        layout.addWidget(self.no_rasters_widget)

        # Dynamic instruction label - styled as a card (slightly darker gray than refine panel)
        self.instructions_label = QLabel("")
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setMinimumHeight(70)
        self.instructions_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.instructions_label.setStyleSheet("""
            QLabel {
                background-color: rgba(128, 128, 128, 0.12);
                border: 1px solid rgba(128, 128, 128, 0.25);
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
                color: palette(text);
            }
        """)
        self.instructions_label.setVisible(False)
        layout.addWidget(self.instructions_label)

        # Container for start button
        self.start_container = QWidget()
        start_layout = QVBoxLayout(self.start_container)
        start_layout.setContentsMargins(0, 8, 0, 0)
        start_layout.setSpacing(6)

        # Terms + Privacy consent — required to run a segmentation.
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
        # Use Qt's native QCheckBox rendering — the previous custom stylesheet
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

        self.start_button = QPushButton(tr("Start AI Segmentation"))
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

        # Keyboard shortcut G to start segmentation (scoped to dock + children)
        self.start_shortcut = QShortcut(QKeySequence("G"), self)
        self.start_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.start_shortcut.activated.connect(self._on_start_shortcut)

        layout.addWidget(self.start_container)

        # Collapsible Refine mask panel
        self._setup_refine_panel(layout)

        # Primary action buttons (reduced height)
        self.save_mask_button = QPushButton(tr("Save polygon") + "  (Shortcut: S)")
        self.save_mask_button.clicked.connect(self._on_save_polygon_clicked)
        self.save_mask_button.setVisible(False)
        self.save_mask_button.setEnabled(False)
        self.save_mask_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_mask_button.setMinimumHeight(34)
        self.save_mask_button.setStyleSheet(_BTN_BLUE)
        self.save_mask_button.setToolTip(
            tr("Save current polygon to your session")
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

    def _setup_refine_panel(self, parent_layout):
        """Setup the collapsible Refine mask panel (collapsible via click on title)."""
        self.refine_group = QGroupBox("▼ " + tr("Refine selection"))
        self.refine_group.setCheckable(False)  # No checkbox, just clickable title
        self.refine_group.setVisible(False)  # Hidden until segmentation active
        self.refine_group.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refine_group.mousePressEvent = self._on_refine_group_clicked
        # Remove all QGroupBox styling - make it look like a simple collapsible section
        self.refine_group.setStyleSheet("""
            QGroupBox {
                background-color: transparent;
                border: none;
                border-radius: 0px;
                margin: 0px;
                padding: 0px;
                padding-top: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: padding;
                subcontrol-position: top left;
                padding: 2px 4px;
                background-color: transparent;
                border: none;
            }
        """)
        refine_layout = QVBoxLayout(self.refine_group)
        refine_layout.setSpacing(0)
        refine_layout.setContentsMargins(0, 0, 0, 0)

        # Content widget to show/hide - styled as a subtle bordered box
        self.refine_content_widget = QWidget()
        self.refine_content_widget.setObjectName("refineContentWidget")

        # Build checkbox indicator images at runtime so they match the
        # current theme regardless of light/dark mode or Qt version.
        from qgis.PyQt.QtGui import QColor, QPainter, QPen, QPixmap
        _sz = 18
        _bg = self.refine_content_widget.palette().color(
            self.refine_content_widget.palette().currentColorGroup(),
            self.refine_content_widget.backgroundRole(),
        )
        # Unchecked: plain background-colored square (invisible)
        _pm_off = QPixmap(_sz, _sz)
        _pm_off.fill(_bg)
        # Checked: background + blue checkmark
        _pm_on = QPixmap(_sz, _sz)
        _pm_on.fill(_bg)
        _painter = QPainter(_pm_on)
        _painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _pen = QPen(QColor("#1976d2"))
        _pen.setWidthF(2.4)
        _pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        _pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        _painter.setPen(_pen)
        _painter.drawLine(3, 10, 7, 14)
        _painter.drawLine(7, 14, 15, 4)
        _painter.end()
        # Save to temp files for the stylesheet
        import tempfile
        # Kept on the instance so cleanup_signals can delete it on unload
        self._checkbox_icon_dir = tempfile.mkdtemp(prefix="qgis_ai_seg_")
        _dir = self._checkbox_icon_dir
        _path_off = os.path.join(_dir, "cb_off.png").replace("\\", "/")
        _path_on = os.path.join(_dir, "cb_on.png").replace("\\", "/")
        _pm_off.save(_path_off, "PNG")
        _pm_on.save(_path_on, "PNG")

        self.refine_content_widget.setStyleSheet(f"""
            QWidget#refineContentWidget {{
                background-color: rgba(128, 128, 128, 0.08);
                border: 1px solid rgba(128, 128, 128, 0.2);
                border-radius: 4px;
            }}
            QLabel {{
                background: transparent;
                border: none;
            }}
            QCheckBox {{
                background: transparent;
            }}
            QCheckBox::indicator {{
                width: {_sz}px;
                height: {_sz}px;
                border: none;
                image: url({_path_off});
            }}
            QCheckBox::indicator:checked {{
                border: none;
                image: url({_path_on});
            }}
        """)
        refine_content_layout = QVBoxLayout(self.refine_content_widget)
        refine_content_layout.setContentsMargins(10, 10, 10, 10)
        refine_content_layout.setSpacing(6)

        # ── Outline section ──
        outline_label = QLabel(tr("Outline").upper())
        outline_label.setStyleSheet(
            "font-size: 10px; color: palette(text); font-weight: bold; "
            "background: transparent; border: none; border-bottom: 1px solid rgba(128, 128, 128, 0.35); "
            "padding: 8px 0px 4px 0px; margin-bottom: 4px; letter-spacing: 1px;")
        refine_content_layout.addWidget(outline_label)

        # 1. Simplify outline: SpinBox (0 to 1000) - reduces small variations
        simplify_layout = QHBoxLayout()
        simplify_label = QLabel(tr("Simplify outline:"))
        simplify_label.setToolTip(tr("Reduce small variations in the outline (0 = no change)"))
        self.simplify_spinbox = QSpinBox()
        self.simplify_spinbox.setRange(0, 1000)
        self.simplify_spinbox.setValue(3)
        self.simplify_spinbox.setMinimumWidth(55)
        self.simplify_spinbox.setMaximumWidth(70)
        simplify_layout.addWidget(simplify_label)
        simplify_layout.addStretch()
        simplify_layout.addWidget(self.simplify_spinbox)
        refine_content_layout.addLayout(simplify_layout)

        # 2. Round corners: Label + Checkbox aligned right (like spinbox rows)
        round_layout = QHBoxLayout()
        round_label = QLabel(tr("Round corners:"))
        round_label.setToolTip(
            tr("Round corners for natural shapes like trees and bushes. "
               "Increase 'Simplify outline' for smoother results."))
        self.round_corners_checkbox = QCheckBox()
        self.round_corners_checkbox.setToolTip(round_label.toolTip())
        self.round_corners_checkbox.setChecked(False)
        round_layout.addWidget(round_label)
        round_layout.addStretch()
        round_layout.addWidget(self.round_corners_checkbox)
        refine_content_layout.addLayout(round_layout)

        # ── Selection section ──
        selection_label = QLabel(tr("Selection").upper())
        selection_label.setStyleSheet(
            "font-size: 10px; color: palette(text); font-weight: bold; "
            "background: transparent; border: none; border-bottom: 1px solid rgba(128, 128, 128, 0.35); "
            "padding: 14px 0px 4px 0px; margin-bottom: 4px; letter-spacing: 1px;")
        refine_content_layout.addWidget(selection_label)

        # 3. Expand/Contract: SpinBox with +/- buttons (-1000 to +1000)
        expand_layout = QHBoxLayout()
        expand_label = QLabel(tr("Expand/Contract:"))
        expand_label.setToolTip(tr("Positive = expand outward, Negative = shrink inward"))
        self.expand_spinbox = QSpinBox()
        self.expand_spinbox.setRange(-1000, 1000)
        self.expand_spinbox.setValue(0)
        self.expand_spinbox.setSuffix(" px")
        self.expand_spinbox.setMinimumWidth(55)
        self.expand_spinbox.setMaximumWidth(70)
        expand_layout.addWidget(expand_label)
        expand_layout.addStretch()
        expand_layout.addWidget(self.expand_spinbox)
        refine_content_layout.addLayout(expand_layout)

        # 4. Fill holes: Label + Checkbox aligned right (like spinbox rows)
        fill_layout = QHBoxLayout()
        fill_label = QLabel(tr("Fill holes:"))
        fill_label.setToolTip(tr("Fill interior holes in the selection"))
        self.fill_holes_checkbox = QCheckBox()
        self.fill_holes_checkbox.setChecked(True)
        self.fill_holes_checkbox.setToolTip(fill_label.toolTip())
        fill_layout.addWidget(fill_label)
        fill_layout.addStretch()
        fill_layout.addWidget(self.fill_holes_checkbox)
        refine_content_layout.addLayout(fill_layout)

        # Min area was previously exposed here but is now auto-computed based
        # on the crop scale (see plugin._compute_auto_min_area). Removing the
        # spinbox keeps the refine panel focused on the controls users tune.

        refine_layout.addWidget(self.refine_content_widget)
        self.refine_content_widget.setVisible(self._refine_expanded)
        # Set initial max height constraint (collapsed by default)
        if not self._refine_expanded:
            self.refine_group.setMaximumHeight(_REFINE_COLLAPSED_HEIGHT)

        # Connect signals
        self.simplify_spinbox.valueChanged.connect(self._on_refine_changed)
        self.round_corners_checkbox.stateChanged.connect(self._on_refine_changed)
        self.expand_spinbox.valueChanged.connect(self._on_refine_changed)
        self.fill_holes_checkbox.stateChanged.connect(self._on_refine_changed)

        parent_layout.addWidget(self.refine_group)

    def _on_refine_group_clicked(self, event):
        """Toggle the refine panel expanded/collapsed state (only when clicking on title)."""
        # Only toggle if click is in the title area
        # This prevents collapsing when clicking spinbox arrows at min/max values
        pt = event.position().toPoint() if QT_VERSION >= 0x060000 else event.pos()
        if pt.y() > _REFINE_COLLAPSED_HEIGHT:
            return  # Click was on content, not title - ignore

        self._refine_expanded = not self._refine_expanded
        arrow = "▼" if self._refine_expanded else "▶"
        self.refine_group.setTitle(f"{arrow} " + tr("Refine selection"))
        # Defer visibility and height changes to next event loop tick to avoid layout glitch
        expanded = self._refine_expanded
        QTimer.singleShot(0, lambda: self._apply_refine_toggle(expanded))

    def _apply_refine_toggle(self, expanded):
        """Apply refine panel expand/collapse after event loop tick."""
        self.refine_content_widget.setVisible(expanded)
        if expanded:
            self.refine_group.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
        else:
            self.refine_group.setMaximumHeight(_REFINE_COLLAPSED_HEIGHT)

    def _on_refine_changed(self, value=None):
        """Handle refine control changes with debounce."""
        self._refine_debounce_timer.start(150)

    def _emit_refine_changed(self):
        """Emit the refine settings changed signal after debounce."""
        self.refine_settings_changed.emit(
            self.simplify_spinbox.value(),
            5 if self.round_corners_checkbox.isChecked() else 0,
            self.expand_spinbox.value(),
            self.fill_holes_checkbox.isChecked(),
        )

    def reset_refine_sliders(self):
        """Reset refinement controls to default values without emitting signals."""
        for w in (self.simplify_spinbox, self.round_corners_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox):
            w.blockSignals(True)

        self.simplify_spinbox.setValue(3)
        self.round_corners_checkbox.setChecked(False)
        self.expand_spinbox.setValue(0)
        self.fill_holes_checkbox.setChecked(True)

        for w in (self.simplify_spinbox, self.round_corners_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox):
            w.blockSignals(False)

    def set_refine_values(self, simplify: int, smooth: int, expand: int,
                          fill_holes: bool, min_area: int | None = None):
        """Set refine slider values without emitting signals.

        min_area is kept in the signature for backward compatibility with
        stored polygon metadata but no longer touches the UI.
        """
        del min_area  # unused since the spinbox was removed
        for w in (self.simplify_spinbox, self.round_corners_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox):
            w.blockSignals(True)

        self.simplify_spinbox.setValue(simplify)
        self.round_corners_checkbox.setChecked(smooth > 0)
        self.expand_spinbox.setValue(expand)
        self.fill_holes_checkbox.setChecked(fill_holes)

        for w in (self.simplify_spinbox, self.round_corners_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox):
            w.blockSignals(False)

    def _setup_update_notification(self):
        """Setup the update notification label (hidden by default)."""
        # Container just for right-alignment
        self._update_notif_container = QWidget()
        container_layout = QHBoxLayout(self._update_notif_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addStretch()

        self.update_notification_label = QLabel("")
        self.update_notification_label.setStyleSheet(
            "background-color: rgba(25, 118, 210, 0.15); "
            "border: 2px solid rgba(25, 118, 210, 0.4); border-radius: 6px; "
            "padding: 6px 12px; font-size: 12px; font-weight: bold; color: palette(text);"
        )
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
                text = '{} <a href="#update" style="color: #1976d2; font-weight: bold; font-size: 13px;">{}</a>'.format(
                    tr("Big update dropped — v{version} is here!").format(
                        version=available_version),
                    tr("Grab it now"))
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
        self.batch_info_widget.setStyleSheet(
            "QWidget { background-color: rgba(100, 149, 237, 0.15); "
            "border: 1px solid rgba(100, 149, 237, 0.3); border-radius: 4px; }"
            "QLabel { background: transparent; border: none; }"
        )
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

        # Footer icon row — mirrors AI Edit. Gear opens Account Settings
        # (visible only when activated), help opens a popup with Tutorial /
        # Shortcuts / Contact us. The contact / tutorial / shortcuts links
        # previously sat as blue underlined labels but moved into the help
        # menu so the bar matches AI Edit's compact look.
        footer_widget = QWidget()
        footer_row = QHBoxLayout(footer_widget)
        footer_row.setContentsMargins(0, 4, 0, 4)
        footer_row.setSpacing(6)

        # Cross-promo CTA, pinned bottom-left (before the stretch) so it sits
        # beside the gear/help icons without crowding them (#30). Opens the Plugin
        # Manager pre-filtered to AI Edit (falls back to the product page).
        from .cross_plugin_discovery import open_ai_edit_page
        self._ai_edit_btn = _FooterIconButton(footer_widget)
        # Decorative glyph kept out of the translatable string. The copy sells
        # AI Edit's own promise (transform imagery) and deliberately stays off
        # AI Segmentation Pro's turf (no segmentation/vectorization wording).
        self._ai_edit_btn.setText("🍌 " + tr("Edit this map with AI"))
        self._ai_edit_btn.setToolTip(tr(
            "Transform imagery with AI Edit: remove clouds, add vegetation, change seasons"))
        self._ai_edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ai_edit_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._ai_edit_btn.setStyleSheet(_FOOTER_CTA_BTN_STYLE)
        self._ai_edit_btn.clicked.connect(lambda: open_ai_edit_page())
        footer_row.addWidget(self._ai_edit_btn)

        footer_row.addStretch()

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
        help_menu.addAction(tr("Shortcuts"), self._on_show_shortcuts)
        help_menu.addAction(tr("Contact us"), self._on_contact_us)
        help_menu.addAction(tr("Report a problem"), self._on_report_problem)
        self._help_btn.setMenu(help_menu)
        # Force the hover tint off when the popup closes — Qt does not
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

    def _on_report_problem(self, _link=None):
        """User-initiated report: open the log-report dialog (collects the
        session logs and pre-fills the support email)."""
        from .error_report_dialog import show_error_report
        show_error_report(
            self,
            tr("Report a problem"),
            "",
            error_code="user_reported",
        )

    def _on_show_shortcuts(self):
        """Show keyboard shortcuts in a dialog."""
        from qgis.PyQt.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout
        undo_key = "Cmd+Z" if sys.platform == "darwin" else "Ctrl+Z"

        key_style = (
            "background-color: rgba(128,128,128,0.18);"
            "border: 1px solid rgba(128,128,128,0.35);"
            "border-radius: 3px;"
            "padding: 1px 5px;"
            "font-family: monospace;"
        )
        k = f"<span style='{key_style}'>{{}}</span>"

        shortcuts_html = (
            "<table cellspacing='4' cellpadding='2'>"
            "<tr><td colspan='2' style='padding-bottom:2px;'>"
            "<b>{seg_title}</b></td></tr>"
            "<tr><td>{g}</td><td>{start}</td></tr>"
            "<tr><td>{s}</td><td>{save}</td></tr>"
            "<tr><td>{enter}</td><td>{export}</td></tr>"
            "<tr><td>{undo}</td><td>{undo_text}</td></tr>"
            "<tr><td>{esc}</td><td>{stop}</td></tr>"
            "<tr><td colspan='2' style='padding-top:6px;padding-bottom:2px;'>"
            "<b>{nav_title}</b></td></tr>"
            "<tr><td>{space}</td><td>{space_desc}</td></tr>"
            "<tr><td>{mouse}</td><td>{mouse_desc}</td></tr>"
            "</table>"
        ).format(
            seg_title=tr("Segmentation"),
            g=k.format("G"),
            start=tr("Start AI Segmentation"),
            s=k.format("S"),
            save=tr("Save polygon"),
            enter=k.format("Enter"),
            export=tr("Export polygon to a layer"),
            undo=k.format(undo_key),
            undo_text=tr("Undo last point"),
            esc=k.format("Esc"),
            stop=tr("Stop segmentation"),
            nav_title=tr("Navigation"),
            space=k.format(tr("Space")),
            space_desc=tr("Hold and move to pan the map"),
            mouse=k.format(tr("Middle mouse button")),
            mouse_desc=tr("Click and drag to pan the map"),
        )

        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Shortcuts"))
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 16, 16, 12)
        label = QLabel(shortcuts_html)
        label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(label)
        ok_btn = QPushButton("OK")
        ok_btn.setFixedWidth(80)
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
            + tr("We'd love to hear from you!")  # noqa: W503
        )
        msg.setWordWrap(True)
        lay.addWidget(msg)

        email_label = QLabel(f"<b>{support_email}</b>")
        email_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(email_label)

        copy_btn = QPushButton(tr("Copy email address"))
        copy_btn.clicked.connect(
            lambda: (
                QApplication.clipboard().setText(support_email),
                copy_btn.setText(tr("Copied!")),
            )
        )
        lay.addWidget(copy_btn)

        or_label = QLabel(tr("or"))
        or_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        or_label.setStyleSheet("color: palette(text);")
        lay.addWidget(or_label)

        call_btn = QPushButton(tr("Book a video call"))
        call_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(calendly_url))
        )
        lay.addWidget(call_btn)

        dlg.exec()

    def set_activated_state(self, activated: bool):
        """Flip the signed-in state and refresh every dependent section."""
        self._plugin_activated = activated
        if activated:
            # A stale pairing spinner must never survive a successful activation.
            self._stop_pairing_wait()
            self._pending_pairing_code = ""
        else:
            self.show_pairing_idle()
            self.activation_message_label.setVisible(False)
        self._update_full_ui()

    def set_activation_message(self, text: str, is_error: bool = False):
        """Public alias used by the plugin's pairing handlers."""
        self._show_activation_message(text, is_error)

    def _show_activation_message(self, text: str, is_error: bool = False):
        """Display a message in the activation section."""
        self.activation_message_label.setText(text)
        if is_error:
            self.activation_message_label.setStyleSheet("color: #ef5350; font-size: 11px;")
        else:
            self.activation_message_label.setStyleSheet("color: #66bb6a; font-size: 11px;")
        self.activation_message_label.setVisible(True)

    def _update_full_ui(self):
        """Update the full UI based on current state."""
        setup_complete = self._dependencies_ok and self._checkpoint_ok

        # Segmentation section: only show if fully set up + activated
        show_segmentation = setup_complete and self._plugin_activated
        self.seg_widget.setVisible(show_segmentation)

        # Welcome section: hide when setup is complete
        self.welcome_widget.setVisible(not setup_complete)

        # Activation section: show after setup complete when not yet activated
        not_activated = not self._plugin_activated
        show_activation = setup_complete and not_activated
        self.activation_group.setVisible(show_activation)

        if show_activation:
            self.welcome_widget.setVisible(False)

        self._settings_btn.setVisible(self._plugin_activated)

        # Sign-in page: hide the cross-promo CTA, and make sure no session-only
        # widget (batch tip) leaks onto it.
        self._ai_edit_btn.setVisible(not show_activation)
        if not show_segmentation:
            self.batch_info_widget.setVisible(False)

        self._update_ui_state()

    def _on_install_clicked(self):
        # Log on click receipt so users stuck on the install screen with no
        # apparent reaction (observed on macOS 26) can prove the click event
        # was actually delivered to Qt — separating UI-event regressions from
        # signal-propagation or background-worker bugs.
        from ..core.logging_utils import log as _log
        _log("Install button clicked")
        self.install_button.setEnabled(False)
        self.install_requested.emit()

    def _toggle_cancel_button(self):
        visible = not self.cancel_button.isVisible()
        self.cancel_button.setVisible(visible)
        arrow = Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow
        self.cancel_toggle.setArrowType(arrow)

    def _on_cancel_clicked(self):
        from qgis.PyQt.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            tr("Cancel installation"),
            tr("Are you sure you want to cancel the installation?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.cancel_install_requested.emit()

    def _on_layer_changed(self, layer):
        # Just update UI state - layer change handling is done by the plugin
        self._update_ui_state()

    def _on_layers_added(self, layers):
        """Handle new layers added to project - auto-select if none selected."""
        # Update UI state first (includes layer filter)
        self._update_ui_state()

        if self.layer_combo.currentLayer() is not None:
            return

        for layer in layers:
            if layer.type() == layer.RasterLayer:
                # Auto-select: prefer local georeferenced, then online, then any raster
                if self._is_online_layer(layer):
                    self.layer_combo.setLayer(layer)
                    break
                if self._is_layer_georeferenced(layer):
                    self.layer_combo.setLayer(layer)
                    break

    def _on_layers_removed(self, _layer_ids):
        """Handle layers removed from project."""
        self._update_ui_state()

    def _on_layer_visibility_changed(self, node):
        """Handle layer visibility toggle in the layer tree (debounced)."""
        self._visibility_debounce_timer.start(100)

    def _on_tos_toggled(self, checked: bool):
        """Persist the Terms + Privacy acceptance and refresh gating."""
        set_tos_accepted(checked)
        self._update_ui_state()

    def _on_start_clicked(self):
        layer = self.layer_combo.currentLayer()
        if not layer:
            return
        # First successful Start click seals the consent forever. Subsequent
        # sessions (or plugin updates) will not re-display the checkbox.
        if not has_tos_locked():
            lock_tos()
            if hasattr(self, "tos_container"):
                self.tos_container.setVisible(False)
        self.start_segmentation_requested.emit(layer)

    def _on_start_shortcut(self):
        """Handle G shortcut to start segmentation."""
        if self.start_button.isEnabled() and self.start_button.isVisible():
            self._on_start_clicked()

    def _on_undo_clicked(self):
        self.undo_requested.emit()

    def _on_save_polygon_clicked(self):
        self.save_polygon_requested.emit()

    def _on_export_clicked(self):
        self.export_layer_requested.emit()

    def _on_stop_clicked(self):
        self.stop_segmentation_requested.emit()

    def set_dependency_status(self, ok: bool, message: str):
        self._dependencies_ok = ok

        if ok:
            self.setup_status_label.setText(message)
            self.setup_status_label.setVisible(True)
            self.setup_status_label.setStyleSheet("font-weight: bold; color: palette(text);")
            self.install_button.setVisible(False)
            self.cancel_toggle.setVisible(False)
            self.cancel_button.setVisible(False)
            self.setup_progress.setVisible(False)
            self.setup_progress_label.setVisible(False)
        else:
            is_update = "updating" in message.lower() or "upgrading" in message.lower()
            is_dll_error = "dll" in message.lower() and "failed" in message.lower()
            if is_dll_error:
                short_msg = tr(
                    "Missing Visual C++ Redistributable. "
                    "Install it, restart your computer, then click Retry.")
                self.setup_status_label.setText(short_msg)
                self.setup_status_label.setStyleSheet(
                    f"font-weight: bold; color: {ERROR_TEXT};")
                self.setup_status_label.setVisible(True)
                self.install_button.setText(tr("Retry"))
            else:
                self.setup_status_label.setVisible(False)
                if is_update:
                    self.install_button.setText(tr("Update"))
                else:
                    self.install_button.setText(tr("Install"))
            self.install_button.setVisible(True)
            self.install_button.setEnabled(True)
            self.setup_group.setVisible(True)

        self._update_full_ui()

    def set_install_progress(self, percent: int, message: str):
        """Unified progress for deps install + model download."""
        import time

        self._target_progress = percent

        time_info = ""
        now = time.time()
        if percent > 10 and percent < 100 and self._install_start_time:
            elapsed = now - self._install_start_time
            if elapsed > 5:
                overall_speed = percent / elapsed
                remaining_pct = 100 - percent

                has_prev = self._last_percent_time is not None
                pct_increased = percent > self._last_percent
                time_increased = now > self._last_percent_time if has_prev else False
                if has_prev and pct_increased and time_increased:
                    dt = now - self._last_percent_time
                    dp = percent - self._last_percent
                    recent_speed = dp / dt
                    blended_speed = 0.7 * recent_speed + 0.3 * overall_speed
                else:
                    blended_speed = overall_speed

                if blended_speed > 0:
                    remaining = remaining_pct / blended_speed
                    max_remaining = 480
                    remaining = min(remaining, max_remaining)
                    if remaining > 60:
                        time_info = f" (~{int(remaining / 60)} min left)"
                    elif remaining > 10:
                        time_info = f" (~{int(remaining)} sec left)"

        if percent > self._last_percent:
            self._last_percent_time = now
            self._last_percent = percent

        self.setup_progress_label.setText(f"{message}{time_info}")

        is_update = self.install_button.text() in (
            tr("Update"), tr("Updating..."))

        if percent == 0:
            self._install_start_time = time.time()
            self._current_progress = 0
            self._last_percent = 0
            self._last_percent_time = None
            self._creep_counter = 0
            self.setup_progress.setValue(0)
            self.setup_progress.setVisible(True)
            self.setup_progress_label.setVisible(True)
            self.cancel_toggle.setVisible(True)
            self.cancel_toggle.setArrowType(Qt.ArrowType.RightArrow)
            self.cancel_button.setVisible(False)
            self.install_button.setVisible(False)
            self.setup_status_label.setVisible(False)
            self.welcome_title.setText(tr("Installing AI Segmentation..."))
            self._progress_timer.start(500)
        elif percent >= 100 or "cancel" in message.lower() or "failed" in message.lower():
            self._progress_timer.stop()
            self._install_start_time = None
            self.setup_progress.setValue(percent)
            self.setup_progress.setVisible(False)
            self.setup_progress_label.setVisible(False)
            self.cancel_toggle.setVisible(False)
            self.cancel_button.setVisible(False)
            self.install_button.setVisible(True)
            self.install_button.setEnabled(True)
            if is_update:
                self.install_button.setText(tr("Update"))
            else:
                self.install_button.setText(tr("Install"))
            if "cancel" in message.lower():
                self.setup_status_label.setVisible(True)
                self.setup_status_label.setText(tr("Installation cancelled"))
                self.welcome_title.setText(tr("Click Install to set up AI Segmentation"))
            elif "failed" in message.lower():
                self.setup_status_label.setVisible(True)
                self.setup_status_label.setText(tr("Installation failed"))
                self.welcome_title.setText(tr("Click Install to set up AI Segmentation"))
            else:
                self.welcome_title.setText(tr("Click Install to set up AI Segmentation"))
        else:
            # Intermediate tick (1..99). The deps phase hides the bar via
            # set_dependency_status once deps are validated; the very next
            # callback then enters the model-download/load phase but never
            # re-shows the progress UI, so users sat on "Dependencies ready"
            # for several minutes with no visible activity. Re-asserting
            # visibility here keeps the hand-off seamless.
            self.setup_progress.setVisible(True)
            self.setup_progress_label.setVisible(True)
            if self._current_progress < percent:
                self._current_progress = percent
                self.setup_progress.setValue(percent)
            msg_lower = message.lower() if message else ""
            if "loading" in msg_lower and "model" in msg_lower:
                self.welcome_title.setText(tr("Loading AI model..."))
            elif "downloading" in msg_lower and "model" in msg_lower:
                self.welcome_title.setText(tr("Downloading AI model..."))
            elif "verifying" in msg_lower:
                self.welcome_title.setText(tr("Verifying installation..."))

    def _on_progress_tick(self):
        """Animate progress bar smoothly between updates."""
        if self._current_progress < self._target_progress:
            step = max(1, (self._target_progress - self._current_progress) // 3)
            self._current_progress = min(
                self._current_progress + step, self._target_progress)
            self._creep_counter = 0
        elif self._current_progress < 99 and self._target_progress > 0:
            self._creep_counter += 1
            if self._creep_counter >= 4:
                self._creep_counter = 0
                if self._current_progress < self._target_progress + 5:
                    self._current_progress += 1

        self.setup_progress.setValue(self._current_progress)

    def set_checkpoint_status(self, ok: bool, message: str):
        self._checkpoint_ok = ok
        if ok:
            self.setup_status_label.setText(message)
            self.setup_status_label.setStyleSheet("font-weight: bold; color: palette(text);")
            self.setup_group.setVisible(False)
        self._update_full_ui()

    def set_segmentation_active(self, active: bool):
        self._segmentation_active = active

        # Track which layer we're segmenting
        if active:
            layer = self.layer_combo.currentLayer()
            self._segmentation_layer_id = layer.id() if layer else None
        else:
            self._segmentation_layer_id = None

        self._update_button_visibility()
        self._update_ui_state()
        if active:
            self._update_instructions()

    def _update_button_visibility(self):
        if self._segmentation_active:
            # Hide label, lock combo (grayed out, no dropdown arrow)
            self.layer_label.setVisible(False)
            self.layer_combo.setEnabled(False)
            self.layer_combo.setStyleSheet(
                "QComboBox { color: palette(text); }"
                "QComboBox::drop-down { width: 0px; border: none; }"
            )

            self.start_container.setVisible(False)
            self.instructions_label.setVisible(True)
            self._update_instructions()

            # Refine panel visibility
            self._update_refine_panel_visibility()

            # Save mask button: always visible during segmentation
            self.save_mask_button.setVisible(True)
            self.save_mask_button.setEnabled(self._has_mask)

            # Export button: visible during segmentation
            self.export_button.setVisible(True)
            self._update_export_button_style()

            # Secondary buttons: visible during segmentation
            self.secondary_buttons_widget.setVisible(True)
            self.undo_button.setVisible(True)
            has_points = self._positive_count > 0 or self._negative_count > 0
            can_undo_saved = self._saved_polygon_count > 0
            self.undo_button.setEnabled(has_points or can_undo_saved)
            self.stop_button.setVisible(True)
            self.stop_button.setEnabled(True)

            # Info box
            self.batch_info_widget.setVisible(True)
        else:
            # Not segmenting - show label, unlock combo, restore dropdown arrow
            self.layer_label.setVisible(True)
            self.layer_combo.setEnabled(True)
            self.layer_combo.setStyleSheet("QComboBox { color: palette(text); }")

            self.start_container.setVisible(True)
            self.instructions_label.setVisible(False)
            self.refine_group.setVisible(False)
            self.save_mask_button.setVisible(False)
            self.export_button.setVisible(False)
            self.undo_button.setVisible(False)
            self.stop_button.setVisible(False)
            self.secondary_buttons_widget.setVisible(False)
            self.batch_info_widget.setVisible(False)

    def _update_refine_panel_visibility(self):
        """Update refine panel visibility based on mask state."""
        if not self._segmentation_active:
            self.refine_group.setVisible(False)
            return

        self.refine_group.setVisible(self._has_mask)

    def _update_export_button_style(self):
        count = self._saved_polygon_count
        if count > 1:
            self.export_button.setText(
                tr("Export {count} polygons to a layer").format(count=count)
            )
        else:
            self.export_button.setText(tr("Export polygon to a layer"))

        if count > 0:
            self.export_button.setEnabled(True)
            self.export_button.setStyleSheet(_BTN_EXPORT_READY)
        else:
            self.export_button.setEnabled(False)
            self.export_button.setStyleSheet(_BTN_EXPORT_DISABLED)

    def set_point_count(self, positive: int, negative: int):
        self._positive_count = positive
        self._negative_count = negative
        total = positive + negative
        has_points = total > 0
        old_has_mask = self._has_mask
        self._has_mask = has_points

        # Undo enabled if: has points OR has saved masks
        can_undo_saved = self._saved_polygon_count > 0
        self.undo_button.setEnabled((has_points or can_undo_saved) and self._segmentation_active)
        self.save_mask_button.setEnabled(has_points)

        if self._segmentation_active:
            self._update_instructions()
            self._update_export_button_style()
            if old_has_mask != self._has_mask:
                self._update_refine_panel_visibility()

    def _update_instructions(self):
        """Update instruction text based on current segmentation state."""
        total = self._positive_count + self._negative_count

        if total == 0 and self._saved_polygon_count > 0:
            # Already saved polygon(s), encourage next or export
            text = (
                tr("Polygon saved! Click on another element to segment, "
                   "or export your polygons.") + "\n\n"
                "\U0001F7E2 " + tr("Left-click to select")
            )
        elif total == 0:
            text = (
                tr("Click on the element you want to segment:") + "\n\n"
                "\U0001F7E2 " + tr("Left-click to select")
            )
        else:
            text = (
                "\U0001F7E2 " + tr("Left-click to add more") + "\n"
                "\u274C " + tr("Right-click to exclude from selection")
            )

        self.instructions_label.setText(text)

    def closeEvent(self, event):
        # Route the close-button (X) through the existing Stop flow when a
        # session is active, so the user gets the discard-warning dialog
        # instead of silently leaving the map tool armed without a panel.
        if self._segmentation_active:
            self.stop_segmentation_requested.emit()
            if self._segmentation_active:
                event.ignore()
                return
        super().closeEvent(event)

    def reset_session(self):
        self._has_mask = False
        self._segmentation_active = False
        self._segmentation_layer_id = None
        self._saved_polygon_count = 0
        self._positive_count = 0
        self._negative_count = 0
        self.reset_refine_sliders()
        self._update_button_visibility()
        self._update_ui_state()

    def set_saved_polygon_count(self, count: int):
        self._saved_polygon_count = count
        self._update_refine_panel_visibility()
        self._update_export_button_style()
        # Update undo button state
        if self._segmentation_active:
            has_points = self._positive_count > 0 or self._negative_count > 0
            can_undo_saved = count > 0
            self.undo_button.setEnabled(has_points or can_undo_saved)

    @staticmethod
    def _is_online_layer(layer) -> bool:
        """Check if a raster layer is an online/remote service."""
        if layer is None or layer.type() != layer.RasterLayer:
            return False
        provider = layer.dataProvider()
        if provider is None:
            return False
        return provider.name() in ("wms", "wmts", "xyz", "arcgismapserver", "wcs")

    def _is_layer_georeferenced(self, layer) -> bool:
        """Check if a raster layer is properly georeferenced."""
        if layer is None or layer.type() != layer.RasterLayer:
            return False

        # Check file extension for compatible formats
        source = layer.source().lower()

        # PNG, JPG, BMP etc. without world files are not georeferenced
        non_georef_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

        has_non_georef_ext = any(source.endswith(ext) for ext in non_georef_extensions)

        # If it's a known non-georeferenced format, check if it has a valid CRS
        if has_non_georef_ext:
            # Check if the layer has a valid CRS (not just default)
            if not layer.crs().isValid():
                return False
            # Check if extent looks like pixel coordinates (0,0 to width,height)
            extent = layer.extent()
            if extent.xMinimum() == 0 and extent.yMinimum() == 0:
                # Likely not georeferenced - just pixel dimensions
                return False

        return True

    def _update_ui_state(self):
        layer = self.layer_combo.currentLayer()
        has_layer = layer is not None

        has_rasters_available = self.layer_combo.count_layers() > 0
        self.no_rasters_widget.setVisible(not has_rasters_available and not self._segmentation_active)
        self.layer_combo.setVisible(has_rasters_available)
        self.layer_label.setVisible(not self._segmentation_active)

        deps_ok = self._dependencies_ok
        checkpoint_ok = self._checkpoint_ok
        activated = self._plugin_activated
        # Once the ToS lock is set, consent is permanent — skip the accepted check.
        tos_ok = has_tos_locked() or has_tos_accepted()
        can_start = deps_ok and checkpoint_ok and has_layer and activated and tos_ok
        self.start_button.setEnabled(can_start and not self._segmentation_active)

    def cleanup_signals(self):
        """Disconnect project signals and clean up shortcuts/timers on plugin reload."""
        try:
            self.layer_combo.cleanup()
        except (TypeError, RuntimeError, AttributeError):
            pass
        try:
            QgsProject.instance().layersAdded.disconnect(self._on_layers_added)
        except (TypeError, RuntimeError):
            pass
        try:
            QgsProject.instance().layersRemoved.disconnect(self._on_layers_removed)
        except (TypeError, RuntimeError):
            pass
        try:
            QgsProject.instance().layerTreeRoot().visibilityChanged.disconnect(
                self._on_layer_visibility_changed)
        except (TypeError, RuntimeError, AttributeError):
            pass
        # Clean up QShortcut to prevent stale callbacks
        try:
            self.start_shortcut.activated.disconnect()
            self.start_shortcut.deleteLater()
        except (TypeError, RuntimeError, AttributeError):
            pass
        # Stop timers first, then disconnect to avoid race conditions
        try:
            self._progress_timer.blockSignals(True)
            self._progress_timer.stop()
            self._progress_timer.timeout.disconnect()
        except (TypeError, RuntimeError, AttributeError):
            pass
        try:
            self._refine_debounce_timer.blockSignals(True)
            self._refine_debounce_timer.stop()
            self._refine_debounce_timer.timeout.disconnect()
        except (TypeError, RuntimeError, AttributeError):
            pass
        try:
            self._visibility_debounce_timer.blockSignals(True)
            self._visibility_debounce_timer.stop()
            self._visibility_debounce_timer.timeout.disconnect()
        except (TypeError, RuntimeError, AttributeError):
            pass
        try:
            self._pairing_anim_timer.blockSignals(True)
            self._pairing_anim_timer.stop()
            self._pairing_anim_timer.timeout.disconnect()
        except (TypeError, RuntimeError, AttributeError):
            pass
        # Remove the temp dir holding the generated checkbox icons
        if getattr(self, "_checkbox_icon_dir", None):
            import shutil
            shutil.rmtree(self._checkbox_icon_dir, ignore_errors=True)
            self._checkbox_icon_dir = None

    def is_activated(self) -> bool:
        """Check if the plugin is activated."""
        return self._plugin_activated

    def showEvent(self, event):
        """Emit plugin_opened once per session when the dock becomes visible.

        Only fires post-activation (telemetry._send is a no-op otherwise),
        so QGIS-open-the-app-but-never-click-the-dock never creates events.
        We only flip the "emitted" flag on a real queued send; that way a
        first-run user who activates + opts in later still gets one
        plugin_opened for this session.
        """
        super().showEvent(event)
        if getattr(self, "_plugin_opened_emitted", False):
            return
        try:
            from ..core.telemetry import track_plugin_opened
            if track_plugin_opened():
                self._plugin_opened_emitted = True
        except Exception:
            pass  # nosec B110
