import os
import sys

from qgis.core import QgsProject
from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QKeySequence
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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


from ..core.activation_manager import (  # noqa: E402
    get_sign_in_url,
    get_tutorial_url,
    is_plugin_activated,
    validate_key_with_server,
)
from ..core.i18n import tr  # noqa: E402
from ..core.model_config import _IS_MACOS_X86, USE_SAM2  # noqa: E402
from ..core.venv_manager import CACHE_DIR  # noqa: E402


class AISegmentationDockWidget(QDockWidget):

    install_requested = pyqtSignal()
    cancel_install_requested = pyqtSignal()
    start_segmentation_requested = pyqtSignal(object)
    undo_requested = pyqtSignal()
    save_polygon_requested = pyqtSignal()
    settings_clicked = pyqtSignal()
    export_layer_requested = pyqtSignal()
    stop_segmentation_requested = pyqtSignal()
    # simplify, smooth, expand, fill_holes, min_area
    refine_settings_changed = pyqtSignal(int, int, int, bool, int)

    def __init__(self, parent=None):
        super().__init__(tr("AI Segmentation by TerraLab"), parent)

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
            ' style="color: #1976d2; text-decoration: none;">TerraLab</a>'
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
        layout.addWidget(self.install_button)

        self.install_path_label = QLabel(
            tr("Install path: {}").format(CACHE_DIR))
        self.install_path_label.setWordWrap(True)
        self.install_path_label.setStyleSheet(
            "color: palette(text);"
            "font-size: 10px;"
            "padding: 4px 6px;"
            "background-color: palette(base);"
            "border: 1px solid palette(mid);"
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
        self.cancel_button.setStyleSheet(
            "QPushButton { background-color: #d32f2f; color: #ffffff; }"
            "QPushButton:hover { background-color: #b71c1c; color: #ffffff; }"
        )
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
        """Setup the activation section with step-based layout (matches AI Edit)."""
        self.activation_group = QGroupBox()
        self.activation_group.setStyleSheet(
            "QGroupBox { border: none; margin: 0; padding: 0; }"
        )
        layout = QVBoxLayout(self.activation_group)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)

        step1_label = QLabel(tr("1. Create your free account"))
        step1_label.setStyleSheet(
            "font-weight: bold; font-size: 12px; color: palette(text);")
        layout.addWidget(step1_label)

        sign_in_btn = QPushButton(tr("Create account (free)"))
        sign_in_btn.setMinimumHeight(36)
        sign_in_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        sign_in_btn.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white;"
            " font-weight: bold; }"
            "QPushButton:hover { background-color: #1b5e20; }"
            "QPushButton:disabled { background-color: #b0bec5; }"
        )
        sign_in_btn.clicked.connect(self._on_panel_sign_in_clicked)
        layout.addWidget(sign_in_btn)

        hint_label = QLabel(tr("Free forever — runs locally"))
        hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("font-size: 11px; color: palette(text);")
        layout.addWidget(hint_label)

        step2_label = QLabel(tr("2. Paste your AI Segmentation activation key"))
        step2_label.setStyleSheet(
            "font-weight: bold; font-size: 12px; color: palette(text);")
        layout.addWidget(step2_label)

        key_row = QHBoxLayout()
        self.panel_key_input = QLineEdit()
        self.panel_key_input.setPlaceholderText("tl_...")
        self.panel_key_input.setMinimumHeight(28)
        self.panel_key_input.returnPressed.connect(self._on_panel_activate_clicked)
        key_row.addWidget(self.panel_key_input)

        self.panel_activate_button = QPushButton(tr("Activate"))
        self.panel_activate_button.setMinimumHeight(28)
        self.panel_activate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.panel_activate_button.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: white;"
            " font-weight: bold; }"
            "QPushButton:hover { background-color: #1565c0; }"
            "QPushButton:disabled { background-color: #b0bec5; }"
        )
        self.panel_activate_button.clicked.connect(self._on_panel_activate_clicked)
        key_row.addWidget(self.panel_activate_button)
        layout.addLayout(key_row)

        self.activation_message_label = QLabel("")
        self.activation_message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.activation_message_label.setWordWrap(True)
        self.activation_message_label.setVisible(False)
        self.activation_message_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.activation_message_label)

        # Optional diagnostics consent. Unchecked by default — must be an
        # active opt-in. Decision persists via core.telemetry.set_consent.
        self.panel_consent_checkbox = QCheckBox(
            tr("Help improve the plugin by sending anonymous diagnostics "
               "and error reports. No images, prompts, or clicks are ever sent.")
        )
        self.panel_consent_checkbox.setStyleSheet("font-size: 11px;")
        self.panel_consent_checkbox.setChecked(False)
        layout.addWidget(self.panel_consent_checkbox)

        self.activation_group.setVisible(False)
        self.main_layout.addWidget(self.activation_group)

    def _setup_segmentation_section(self):
        self.seg_widget = QWidget()
        layout = QVBoxLayout(self.seg_widget)
        layout.setContentsMargins(0, 8, 0, 0)

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

        self.start_button = QPushButton(tr("Start AI Segmentation"))
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._on_start_clicked)
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: #000000; padding: 8px 16px; }"
            "QPushButton:hover { background-color: #1b5e20; color: #000000; }"
            "QPushButton:disabled { background-color: #c8e6c9; color: #666666; }"
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
        self.save_mask_button.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: #000000; padding: 6px 12px; }"
            "QPushButton:hover { background-color: #1565c0; color: #000000; }"
            "QPushButton:disabled { background-color: #b0bec5; color: #666666; }"
        )
        self.save_mask_button.setToolTip(
            tr("Save current polygon to your session")
        )
        layout.addWidget(self.save_mask_button)

        self.export_button = QPushButton(tr("Export polygon to a layer"))
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setVisible(False)
        self.export_button.setEnabled(False)
        self.export_button.setStyleSheet(
            "QPushButton { background-color: #b0bec5; color: #000000; padding: 6px 12px; }"
            "QPushButton:hover { background-color: #90a4ae; color: #000000; }"
            "QPushButton:disabled { background-color: #b0bec5; color: #666666; }"
        )
        layout.addWidget(self.export_button)

        # Secondary action buttons (small, horizontal)
        secondary_layout = QHBoxLayout()
        secondary_layout.setSpacing(8)

        self.undo_button = QPushButton(tr("Undo last point"))
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._on_undo_clicked)
        self.undo_button.setVisible(False)  # Hidden until segmentation starts
        self.undo_button.setStyleSheet(
            "QPushButton { background-color: #757575; color: #000000; padding: 4px 8px; }"
            "QPushButton:hover { background-color: #616161; color: #000000; }"
        )
        secondary_layout.addWidget(self.undo_button, 1)  # stretch factor 1

        self.stop_button = QPushButton(tr("Stop segmentation"))
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setVisible(False)  # Hidden until segmentation starts
        self.stop_button.setStyleSheet(
            "QPushButton { background-color: #757575; color: #000000; padding: 4px 8px; }"
            "QPushButton:hover { background-color: #616161; color: #000000; }"
        )
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
        _dir = tempfile.mkdtemp(prefix="qgis_ai_seg_")
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
            "background: transparent; border: none; border-bottom: 1px solid palette(mid); "
            "padding: 8px 0px 4px 0px; margin-bottom: 4px; letter-spacing: 1px;")
        refine_content_layout.addWidget(outline_label)

        # 1. Simplify outline: SpinBox (0 to 1000) - reduces small variations
        simplify_layout = QHBoxLayout()
        simplify_label = QLabel(tr("Simplify outline:"))
        simplify_label.setToolTip(tr("Reduce small variations in the outline (0 = no change)"))
        self.simplify_spinbox = QSpinBox()
        self.simplify_spinbox.setRange(0, 1000)
        self.simplify_spinbox.setValue(10)
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
            "background: transparent; border: none; border-bottom: 1px solid palette(mid); "
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

        # 5. Min area: remove small artifacts / noise regions (#12)
        min_area_layout = QHBoxLayout()
        min_area_label = QLabel(tr("Min area:"))
        min_area_label.setToolTip(tr("Remove polygons smaller than this area (in pixels)"))
        self.min_area_spinbox = QSpinBox()
        self.min_area_spinbox.setRange(0, 100000)
        self.min_area_spinbox.setSingleStep(50)
        self.min_area_spinbox.setValue(200)
        self.min_area_spinbox.setSuffix(" px")
        self.min_area_spinbox.setMinimumWidth(55)
        self.min_area_spinbox.setMaximumWidth(80)
        self.min_area_spinbox.setToolTip(min_area_label.toolTip())
        min_area_layout.addWidget(min_area_label)
        min_area_layout.addStretch()
        min_area_layout.addWidget(self.min_area_spinbox)
        refine_content_layout.addLayout(min_area_layout)

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
        self.min_area_spinbox.valueChanged.connect(self._on_refine_changed)

        parent_layout.addWidget(self.refine_group)

    def _on_refine_group_clicked(self, event):
        """Toggle the refine panel expanded/collapsed state (only when clicking on title)."""
        # Only toggle if click is in the title area
        # This prevents collapsing when clicking spinbox arrows at min/max values
        if event.pos().y() > _REFINE_COLLAPSED_HEIGHT:
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
            self.min_area_spinbox.value(),
        )

    def reset_refine_sliders(self):
        """Reset refinement controls to default values without emitting signals."""
        for w in (self.simplify_spinbox, self.round_corners_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox,
                  self.min_area_spinbox):
            w.blockSignals(True)

        self.simplify_spinbox.setValue(10)
        self.round_corners_checkbox.setChecked(False)
        self.expand_spinbox.setValue(0)
        self.fill_holes_checkbox.setChecked(True)
        self.min_area_spinbox.setValue(200)

        for w in (self.simplify_spinbox, self.round_corners_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox,
                  self.min_area_spinbox):
            w.blockSignals(False)

    def set_refine_values(self, simplify: int, smooth: int, expand: int,
                          fill_holes: bool, min_area: int | None = None):
        """Set refine slider values without emitting signals."""
        for w in (self.simplify_spinbox, self.round_corners_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox,
                  self.min_area_spinbox):
            w.blockSignals(True)

        self.simplify_spinbox.setValue(simplify)
        self.round_corners_checkbox.setChecked(smooth > 0)
        self.expand_spinbox.setValue(expand)
        self.fill_holes_checkbox.setChecked(fill_holes)
        if min_area is not None:
            self.min_area_spinbox.setValue(min_area)

        for w in (self.simplify_spinbox, self.round_corners_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox,
                  self.min_area_spinbox):
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

        # Simple horizontal layout for links, aligned right with larger font
        links_widget = QWidget()
        links_layout = QHBoxLayout(links_widget)
        links_layout.setContentsMargins(0, 4, 0, 4)
        links_layout.setSpacing(16)

        links_layout.addStretch()  # Push links to the right

        # Settings link (only visible when activated)
        self._settings_link = QLabel(
            '<a href="#" style="color: #1976d2;">' + tr("Settings") + "</a>"
        )
        self._settings_link.setStyleSheet("font-size: 13px;")
        self._settings_link.setCursor(Qt.CursorShape.PointingHandCursor)
        self._settings_link.linkActivated.connect(lambda _: self.settings_clicked.emit())
        self._settings_link.setVisible(False)
        links_layout.addWidget(self._settings_link)

        # Contact us link
        contact_link = QLabel(
            '<a href="#" style="color: #1976d2;">' + tr("Contact us") + "</a>"
        )
        contact_link.setStyleSheet("font-size: 13px;")
        contact_link.setCursor(Qt.CursorShape.PointingHandCursor)
        contact_link.linkActivated.connect(self._on_contact_us)
        links_layout.addWidget(contact_link)

        # Tutorial link (server-driven URL)
        docs_link = QLabel(
            '<a href="' + get_tutorial_url() + '" style="color: #1976d2;">' + tr("Tutorial") + "</a>"
        )
        docs_link.setStyleSheet("font-size: 13px;")
        docs_link.setOpenExternalLinks(True)
        docs_link.setCursor(Qt.CursorShape.PointingHandCursor)
        links_layout.addWidget(docs_link)

        # Shortcuts link
        shortcuts_link = QLabel(
            '<a href="#" style="color: #1976d2;">' + tr("Shortcuts") + "</a>"
        )
        shortcuts_link.setStyleSheet("font-size: 13px;")
        shortcuts_link.setCursor(Qt.CursorShape.PointingHandCursor)
        shortcuts_link.linkActivated.connect(self._on_show_shortcuts)
        links_layout.addWidget(shortcuts_link)

        self.main_layout.addWidget(links_widget)

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

    def _on_panel_sign_in_clicked(self):
        """Open TerraLab sign-in page in browser."""
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(get_sign_in_url()))

    def _on_panel_activate_clicked(self):
        """Validate the pasted activation key."""
        key = self.panel_key_input.text().strip()
        if not key:
            self._show_activation_message(
                tr("Please enter your activation key."), is_error=True)
            return

        self.panel_activate_button.setEnabled(False)
        self.panel_activate_button.setText(tr("Checking..."))

        success, msg = validate_key_with_server(key)

        self.panel_activate_button.setText(tr("Activate"))
        self.panel_activate_button.setEnabled(True)

        if success:
            self._plugin_activated = True
            self._show_activation_message(tr(msg), is_error=False)
            self._update_full_ui()
            # Persist the consent choice captured below the key input, then
            # backfill plugin_opened for this session if the dock was shown
            # before the user activated.
            try:
                from ..core.telemetry import (
                    set_consent,
                    track_plugin_activated,
                    track_plugin_opened,
                )
                consent_checked = bool(
                    getattr(self, "panel_consent_checkbox", None)
                    and self.panel_consent_checkbox.isChecked()
                )
                set_consent(consent_checked)
                track_plugin_activated()
                if not getattr(self, "_plugin_opened_emitted", False):
                    if track_plugin_opened():
                        self._plugin_opened_emitted = True
            except Exception:
                pass  # nosec B110
        else:
            self._show_activation_message(tr(msg), is_error=True)

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

        self._settings_link.setVisible(self._plugin_activated)

        self._update_ui_state()

    def _on_install_clicked(self):
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

    def _on_start_clicked(self):
        layer = self.layer_combo.currentLayer()
        if layer:
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
                    "font-weight: bold; color: #c62828;")
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
            if self._current_progress < percent:
                self._current_progress = percent
                self.setup_progress.setValue(percent)

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
            self.export_button.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: #000000; padding: 6px 12px; }"
                "QPushButton:hover { background-color: #388E3C; color: #000000; }"
            )
        else:
            self.export_button.setEnabled(False)
            self.export_button.setStyleSheet(
                "QPushButton { background-color: #b0bec5; color: #666666; padding: 6px 12px; }"
            )

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
        can_start = deps_ok and checkpoint_ok and has_layer and activated
        self.start_button.setEnabled(can_start and not self._segmentation_active)

    def show_activation_dialog(self):
        """Show the activation dialog (popup). Only shown once per session."""
        if self._activation_popup_shown:
            return
        from .activation_dialog import ActivationDialog

        self._activation_popup_shown = True
        dialog = ActivationDialog(self)
        dialog.activated.connect(self._on_dialog_activated)
        dialog.exec()

        # If dialog was closed without activation, show the panel section
        if not self._plugin_activated:
            self._update_full_ui()

    def _on_dialog_activated(self):
        """Handle activation from dialog."""
        self._plugin_activated = True
        self._update_full_ui()

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
