import os
import sys

from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QProgressBar,
    QFrame,
    QLineEdit,
    QSpinBox,
    QCheckBox,
    QToolButton,
    QStyle,
    QSizePolicy,
    QScrollArea,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal, QTimer, QUrl
from qgis.PyQt.QtGui import QDesktopServices, QKeySequence
from qgis.PyQt.QtWidgets import QShortcut
from qgis.core import QgsMapLayerProxyModel, QgsProject

from qgis.gui import QgsMapLayerComboBox

# Collapsed height for refine panel title (just enough to show the arrow + label)
_REFINE_COLLAPSED_HEIGHT = 25

from ..core.activation_manager import (  # noqa: E402
    is_plugin_activated,
    activate_plugin,
    get_newsletter_url,
)
from ..core.i18n import tr  # noqa: E402
from ..core.model_config import CHECKPOINT_SIZE_LABEL, USE_SAM2  # noqa: E402


class AISegmentationDockWidget(QDockWidget):

    install_dependencies_requested = pyqtSignal()
    cancel_deps_install_requested = pyqtSignal()
    download_checkpoint_requested = pyqtSignal()
    start_segmentation_requested = pyqtSignal(object)
    clear_points_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    save_polygon_requested = pyqtSignal()
    export_layer_requested = pyqtSignal()
    stop_segmentation_requested = pyqtSignal()
    refine_settings_changed = pyqtSignal(int, int, bool, int)  # expand, simplify, fill_holes, min_area
    batch_mode_changed = pyqtSignal(bool)  # Batch mode is always on

    def __init__(self, parent=None):
        super().__init__(tr("AI Segmentation by TerraLab"), parent)

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self._setup_title_bar()

        # Initialize state variables that are needed during UI setup
        self._refine_expanded = False  # Refine panel collapsed state persisted in session

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setSpacing(8)
        self.main_layout.setContentsMargins(8, 8, 8, 8)

        self._setup_ui()

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.main_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
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
        self._batch_mode = True  # Batch mode is now the only mode
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
        self._is_cuda_install = False

        # Debounce timer for refinement sliders
        self._refine_debounce_timer = QTimer(self)
        self._refine_debounce_timer.setSingleShot(True)
        self._refine_debounce_timer.timeout.connect(self._emit_refine_changed)

        # Connect to project layer signals for dynamic updates
        QgsProject.instance().layersAdded.connect(self._on_layers_added)
        QgsProject.instance().layersRemoved.connect(self._on_layers_removed)

        # Update UI state
        self._update_full_ui()

    def _setup_title_bar(self):
        """Custom title bar with clickable TerraLab link and native buttons."""
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(4, 0, 0, 0)
        title_layout.setSpacing(0)

        title_label = QLabel(
            'AI Segmentation by '
            '<a href="https://terra-lab.ai" style="color: #1976d2; text-decoration: none;">TerraLab</a>'
        )
        title_label.setOpenExternalLinks(True)
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        icon_size = self.style().pixelMetric(QStyle.PM_SmallIconSize)

        float_btn = QToolButton()
        float_btn.setIcon(self.style().standardIcon(QStyle.SP_TitleBarNormalButton))
        float_btn.setFixedSize(icon_size + 4, icon_size + 4)
        float_btn.setAutoRaise(True)
        float_btn.clicked.connect(lambda: self.setFloating(not self.isFloating()))
        title_layout.addWidget(float_btn)

        close_btn = QToolButton()
        close_btn.setIcon(self.style().standardIcon(QStyle.SP_TitleBarCloseButton))
        close_btn.setFixedSize(icon_size + 4, icon_size + 4)
        close_btn.setAutoRaise(True)
        close_btn.clicked.connect(self.close)
        title_layout.addWidget(close_btn)

        self.setTitleBarWidget(title_widget)

    def _setup_ui(self):
        self._setup_welcome_section()
        self._setup_dependencies_section()
        self._setup_checkpoint_section()
        self._setup_activation_section()
        self._setup_segmentation_section()
        self.main_layout.addStretch()
        self._setup_update_notification()
        self._setup_about_section()

    def _setup_welcome_section(self):
        """Setup the welcome/intro section explaining the 2-step setup."""
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

        welcome_title = QLabel(tr("Welcome! Two quick steps to get started:"))
        welcome_title.setStyleSheet("font-weight: bold; font-size: 12px; color: palette(text);")
        layout.addWidget(welcome_title)

        step1_label = QLabel("1. " + tr("Install AI dependencies"))
        step1_label.setStyleSheet("font-size: 11px; color: palette(text); margin-left: 8px;")
        layout.addWidget(step1_label)

        step2_label = QLabel("2. " + tr("Download the segmentation model"))
        step2_label.setStyleSheet("font-size: 11px; color: palette(text); margin-left: 8px;")
        layout.addWidget(step2_label)

        self.main_layout.addWidget(self.welcome_widget)

    def _setup_dependencies_section(self):
        self.deps_group = QGroupBox(tr("Step 1: AI Dependencies"))
        layout = QVBoxLayout(self.deps_group)

        self.deps_status_label = QLabel(tr("Checking if dependencies are installed..."))
        self.deps_status_label.setStyleSheet("color: palette(text);")
        layout.addWidget(self.deps_status_label)

        self.deps_progress = QProgressBar()
        self.deps_progress.setRange(0, 100)
        self.deps_progress.setVisible(False)
        layout.addWidget(self.deps_progress)

        self.deps_progress_label = QLabel("")
        self.deps_progress_label.setStyleSheet("color: palette(text); font-size: 10px;")
        self.deps_progress_label.setVisible(False)
        layout.addWidget(self.deps_progress_label)

        self.install_button = QPushButton(tr("Install Dependencies"))
        self.install_button.clicked.connect(self._on_install_clicked)
        self.install_button.setVisible(False)
        self.install_button.setToolTip("")
        layout.addWidget(self.install_button)

        self.cancel_deps_button = QPushButton(tr("Cancel"))
        self.cancel_deps_button.clicked.connect(self._on_cancel_deps_clicked)
        self.cancel_deps_button.setVisible(False)
        self.cancel_deps_button.setStyleSheet("background-color: #d32f2f;")
        layout.addWidget(self.cancel_deps_button)

        self.gpu_info_box = QLabel("")
        self.gpu_info_box.setWordWrap(True)
        self.gpu_info_box.setStyleSheet(
            "background-color: rgba(46, 125, 50, 0.08);"
            "border: 1px solid rgba(46, 125, 50, 0.25);"
            "border-radius: 4px;"
            "padding: 8px;"
            "font-size: 11px;"
            "color: palette(text);"
        )
        self.gpu_info_box.setVisible(False)
        layout.addWidget(self.gpu_info_box)

        self.main_layout.addWidget(self.deps_group)

    def _setup_checkpoint_section(self):
        self.checkpoint_group = QGroupBox(tr("Step 2: Segmentation Model"))
        layout = QVBoxLayout(self.checkpoint_group)

        self.checkpoint_status_label = QLabel(tr("Waiting for Step 1..."))
        self.checkpoint_status_label.setStyleSheet("color: palette(text);")
        layout.addWidget(self.checkpoint_status_label)

        self.checkpoint_progress = QProgressBar()
        self.checkpoint_progress.setRange(0, 100)
        self.checkpoint_progress.setVisible(False)
        layout.addWidget(self.checkpoint_progress)

        self.checkpoint_progress_label = QLabel("")
        self.checkpoint_progress_label.setStyleSheet("color: palette(text); font-size: 10px;")
        self.checkpoint_progress_label.setVisible(False)
        layout.addWidget(self.checkpoint_progress_label)

        self.download_button = QPushButton(
            tr("Download AI Segmentation Model ({size})").format(size=CHECKPOINT_SIZE_LABEL))
        self.download_button.clicked.connect(self._on_download_clicked)
        self.download_button.setVisible(False)
        self.download_button.setToolTip(tr("Download the SAM checkpoint for segmentation"))
        layout.addWidget(self.download_button)

        if not USE_SAM2:
            sam1_info = QLabel(tr("Update QGIS to 3.34+ for the latest AI model"))
            sam1_info.setStyleSheet("color: palette(text); font-size: 10px;")
            layout.addWidget(sam1_info)

        self.main_layout.addWidget(self.checkpoint_group)

    def _setup_activation_section(self):
        """Setup the activation section - only shown if popup was closed without activating."""
        self.activation_group = QGroupBox()
        self.activation_group.setStyleSheet(
            "QGroupBox { border: none; margin: 0; padding: 0; }"
        )
        layout = QVBoxLayout(self.activation_group)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)

        # TerraLab banner
        banner_label = QLabel()
        banner_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            "resources", "icons", "terralab-banner.png")
        if os.path.exists(banner_path):
            from qgis.PyQt.QtGui import QPixmap
            pixmap = QPixmap(banner_path)
            scaled = pixmap.scaledToWidth(280, Qt.SmoothTransformation)
            banner_label.setPixmap(scaled)
        else:
            banner_label.setText("TerraLab")
            banner_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: palette(text);")
        banner_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(banner_label)

        # Title
        title_label = QLabel(tr("Unlock Plugin"))
        title_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: palette(text);")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Get code button
        get_code_button = QPushButton(tr("Get my verification code"))
        get_code_button.setMinimumHeight(30)
        get_code_button.setCursor(Qt.PointingHandCursor)
        get_code_button.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1b5e20; }"
        )
        get_code_button.clicked.connect(self._on_get_code_clicked)
        layout.addWidget(get_code_button)

        # Code input section - compact
        code_label = QLabel(tr("Then paste your code:"))
        code_label.setStyleSheet(
            "font-size: 11px; margin-top: 2px; color: palette(text);")
        layout.addWidget(code_label)

        code_layout = QHBoxLayout()
        code_layout.setSpacing(6)

        self.activation_code_input = QLineEdit()
        self.activation_code_input.setPlaceholderText(tr("Code"))
        self.activation_code_input.setMinimumHeight(28)
        self.activation_code_input.returnPressed.connect(self._on_activate_clicked)
        code_layout.addWidget(self.activation_code_input)

        self.activate_button = QPushButton(tr("Unlock"))
        self.activate_button.setMinimumHeight(28)
        self.activate_button.setMinimumWidth(60)
        self.activate_button.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: white; "
            "font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1565c0; }"
        )
        self.activate_button.clicked.connect(self._on_activate_clicked)
        code_layout.addWidget(self.activate_button)

        layout.addLayout(code_layout)

        # Error message label
        self.activation_message_label = QLabel("")
        self.activation_message_label.setAlignment(Qt.AlignCenter)
        self.activation_message_label.setWordWrap(True)
        self.activation_message_label.setVisible(False)
        self.activation_message_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.activation_message_label)

        # Hidden by default - only shown if popup closed without activation
        self.activation_group.setVisible(False)
        self.main_layout.addWidget(self.activation_group)

    def _setup_segmentation_section(self):
        self.seg_separator = QFrame()
        self.seg_separator.setFrameShape(QFrame.HLine)
        self.seg_separator.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(self.seg_separator)

        self.seg_widget = QWidget()
        layout = QVBoxLayout(self.seg_widget)
        layout.setContentsMargins(0, 8, 0, 0)

        layer_label = QLabel(tr("Select a Raster Layer to Segment:"))
        layer_label.setStyleSheet("font-weight: bold; color: palette(text);")
        layout.addWidget(layer_label)
        self.layer_label = layer_label

        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.layer_combo.setAllowEmptyLayer(False)
        self.layer_combo.setShowCrs(False)
        self.layer_combo.layerChanged.connect(self._on_layer_changed)
        self.layer_combo.setToolTip(tr("Select a raster layer (GeoTIFF, WMS, XYZ tiles, etc.)"))
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
        warning_icon = style.standardIcon(style.SP_MessageBoxWarning)
        warning_icon_label.setPixmap(warning_icon.pixmap(16, 16))
        warning_icon_label.setFixedSize(16, 16)
        no_rasters_layout.addWidget(warning_icon_label, 0, Qt.AlignTop)

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
        self.instructions_label.setMinimumHeight(0)
        self.instructions_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Minimum)
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
        start_layout.setContentsMargins(0, 0, 0, 0)
        start_layout.setSpacing(6)

        self.start_button = QPushButton(tr("Start AI Segmentation"))
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._on_start_clicked)
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #2e7d32; padding: 8px 16px; }"
            "QPushButton:disabled { background-color: #c8e6c9; }"
        )
        start_layout.addWidget(self.start_button)

        # Keyboard shortcut G to start segmentation
        self.start_shortcut = QShortcut(QKeySequence("G"), self)
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
            "QPushButton { background-color: #1976d2; padding: 6px 12px; }"
            "QPushButton:disabled { background-color: #b0bec5; }"
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
            "QPushButton { background-color: #b0bec5; padding: 6px 12px; }"
        )
        layout.addWidget(self.export_button)

        # Secondary action buttons (small, horizontal)
        secondary_layout = QHBoxLayout()
        secondary_layout.setSpacing(8)

        self.undo_button = QPushButton(tr("Undo last point"))
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._on_undo_clicked)
        self.undo_button.setVisible(False)  # Hidden until segmentation starts
        self.undo_button.setStyleSheet("QPushButton { padding: 4px 8px; }")
        secondary_layout.addWidget(self.undo_button, 1)  # stretch factor 1

        self.stop_button = QPushButton(tr("Stop segmentation"))
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setVisible(False)  # Hidden until segmentation starts
        self.stop_button.setStyleSheet(
            "QPushButton { background-color: #757575; padding: 4px 8px; }"
        )
        secondary_layout.addWidget(self.stop_button, 1)  # stretch factor 1 for same width

        self.secondary_buttons_widget = QWidget()
        self.secondary_buttons_widget.setLayout(secondary_layout)
        self.secondary_buttons_widget.setVisible(False)
        layout.addWidget(self.secondary_buttons_widget)

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

        # Info icon
        batch_info_icon = QLabel()
        style = self.batch_info_widget.style()
        batch_icon = style.standardIcon(style.SP_MessageBoxInformation)
        batch_info_icon.setPixmap(batch_icon.pixmap(14, 14))
        batch_info_icon.setFixedSize(14, 14)
        batch_info_layout.addWidget(batch_info_icon, 0, Qt.AlignTop)

        # Info text
        info_msg = "{}\n{}".format(
            tr("The AI model works best on one element at a time."),
            tr("Save your polygon before selecting the next element."))
        batch_info_text = QLabel(info_msg)
        batch_info_text.setWordWrap(True)
        batch_info_text.setStyleSheet("font-size: 11px; color: palette(text);")
        batch_info_layout.addWidget(batch_info_text, 1)

        self.batch_info_widget.setVisible(False)
        layout.addWidget(self.batch_info_widget)

        # Warning box for disjoint regions (yellow/orange style)
        self.disjoint_warning_widget = QWidget()
        self.disjoint_warning_widget.setStyleSheet(
            "QWidget { background-color: rgba(255, 180, 50, 0.20); "
            "border: 1px solid rgba(255, 180, 50, 0.4); border-radius: 4px; }"
            "QLabel { background: transparent; border: none; }"
        )
        disjoint_layout = QHBoxLayout(self.disjoint_warning_widget)
        disjoint_layout.setContentsMargins(8, 6, 8, 6)
        disjoint_layout.setSpacing(8)

        disjoint_icon = QLabel()
        warn_icon = style.standardIcon(style.SP_MessageBoxWarning)
        disjoint_icon.setPixmap(warn_icon.pixmap(14, 14))
        disjoint_icon.setFixedSize(14, 14)
        disjoint_layout.addWidget(disjoint_icon, 0, Qt.AlignTop)

        disjoint_msg = "{}\n{}".format(
            tr("Disconnected parts detected in your polygon."),
            tr("For best accuracy, segment one element at a time."))
        disjoint_text = QLabel(disjoint_msg)
        disjoint_text.setWordWrap(True)
        disjoint_text.setStyleSheet("font-size: 11px; color: palette(text);")
        disjoint_layout.addWidget(disjoint_text, 1)

        self.disjoint_warning_widget.setVisible(False)
        layout.addWidget(self.disjoint_warning_widget)

        # Collapsible shortcuts section
        self._setup_shortcuts_section(layout)

        self.main_layout.addWidget(self.seg_widget)

    def _setup_refine_panel(self, parent_layout):
        """Setup the collapsible Refine mask panel (collapsible via click on title)."""
        self.refine_group = QGroupBox("▶ " + tr("Refine selection"))
        self.refine_group.setCheckable(False)  # No checkbox, just clickable title
        self.refine_group.setVisible(False)  # Hidden until segmentation active
        self.refine_group.setCursor(Qt.PointingHandCursor)
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
        self.refine_content_widget.setStyleSheet("""
            QWidget#refineContentWidget {
                background-color: rgba(128, 128, 128, 0.08);
                border: 1px solid rgba(128, 128, 128, 0.2);
                border-radius: 4px;
            }
            QLabel {
                background: transparent;
                border: none;
            }
        """)
        refine_content_layout = QVBoxLayout(self.refine_content_widget)
        refine_content_layout.setContentsMargins(10, 10, 10, 10)
        refine_content_layout.setSpacing(8)

        # 1. Expand/Contract: SpinBox with +/- buttons (-1000 to +1000)
        expand_layout = QHBoxLayout()
        expand_label = QLabel(tr("Expand/Contract:"))
        expand_label.setToolTip(tr("Positive = expand outward, Negative = shrink inward"))
        self.expand_spinbox = QSpinBox()
        self.expand_spinbox.setRange(-1000, 1000)
        self.expand_spinbox.setValue(0)
        self.expand_spinbox.setSuffix(" px")
        self.expand_spinbox.setMinimumWidth(80)
        expand_layout.addWidget(expand_label)
        expand_layout.addStretch()
        expand_layout.addWidget(self.expand_spinbox)
        refine_content_layout.addLayout(expand_layout)

        # 2. Simplify outline: SpinBox (0 to 1000) - reduces small variations in the outline
        simplify_layout = QHBoxLayout()
        simplify_label = QLabel(tr("Simplify outline:"))
        simplify_label.setToolTip(tr("Reduce small variations in the outline (0 = no change)"))
        self.simplify_spinbox = QSpinBox()
        self.simplify_spinbox.setRange(0, 1000)
        self.simplify_spinbox.setValue(3)  # Default to 3 for smoother outlines
        self.simplify_spinbox.setMinimumWidth(80)
        simplify_layout.addWidget(simplify_label)
        simplify_layout.addStretch()
        simplify_layout.addWidget(self.simplify_spinbox)
        refine_content_layout.addLayout(simplify_layout)

        # 3. Fill holes: Checkbox - fills interior holes in the mask
        fill_holes_layout = QHBoxLayout()
        self.fill_holes_checkbox = QCheckBox(tr("Fill holes"))
        self.fill_holes_checkbox.setChecked(False)  # Default: no fill holes
        self.fill_holes_checkbox.setToolTip(tr("Fill interior holes in the selection"))
        fill_holes_layout.addWidget(self.fill_holes_checkbox)
        fill_holes_layout.addStretch()
        refine_content_layout.addLayout(fill_holes_layout)

        # 4. Remove small artifacts: SpinBox - minimum area threshold
        min_area_layout = QHBoxLayout()
        min_area_label = QLabel(tr("Min. region size:"))
        min_area_label.setToolTip(
            "{}\n{}\n{}".format(
                tr("Remove disconnected regions smaller than this area (in pixels²)."),
                tr("Example: 100 = ~10x10 pixel regions, 900 = ~30x30."),
                tr("0 = keep all.")))
        self.min_area_spinbox = QSpinBox()
        self.min_area_spinbox.setRange(0, 100000)
        self.min_area_spinbox.setValue(100)  # Default: remove small artifacts
        self.min_area_spinbox.setSuffix(" px²")
        self.min_area_spinbox.setSingleStep(50)
        self.min_area_spinbox.setMinimumWidth(80)
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
        self.expand_spinbox.valueChanged.connect(self._on_refine_changed)
        self.simplify_spinbox.valueChanged.connect(self._on_refine_changed)
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
        self.refine_content_widget.setVisible(self._refine_expanded)
        arrow = "▼" if self._refine_expanded else "▶"
        self.refine_group.setTitle(f"{arrow} " + tr("Refine selection"))
        # Adjust size constraints to eliminate empty rectangle when collapsed
        if self._refine_expanded:
            self.refine_group.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX  # Reset to default
        else:
            self.refine_group.setMaximumHeight(_REFINE_COLLAPSED_HEIGHT)  # Just enough for the title

    def _on_refine_changed(self, value=None):
        """Handle refine control changes with debounce."""
        self._refine_debounce_timer.start(150)

    def _emit_refine_changed(self):
        """Emit the refine settings changed signal after debounce."""
        self.refine_settings_changed.emit(
            self.expand_spinbox.value(),
            self.simplify_spinbox.value(),
            self.fill_holes_checkbox.isChecked(),
            self.min_area_spinbox.value()
        )

    def reset_refine_sliders(self):
        """Reset refinement controls to default values without emitting signals."""
        self.expand_spinbox.blockSignals(True)
        self.simplify_spinbox.blockSignals(True)
        self.fill_holes_checkbox.blockSignals(True)
        self.min_area_spinbox.blockSignals(True)

        self.expand_spinbox.setValue(0)
        self.simplify_spinbox.setValue(3)  # Default to 3
        self.fill_holes_checkbox.setChecked(False)  # Default: no fill holes
        self.min_area_spinbox.setValue(100)  # Default: remove small artifacts

        self.expand_spinbox.blockSignals(False)
        self.simplify_spinbox.blockSignals(False)
        self.fill_holes_checkbox.blockSignals(False)
        self.min_area_spinbox.blockSignals(False)

    def set_refine_values(self, expand: int, simplify: int, fill_holes: bool = False, min_area: int = 0):
        """Set refine slider values without emitting signals."""
        self.expand_spinbox.blockSignals(True)
        self.simplify_spinbox.blockSignals(True)
        self.fill_holes_checkbox.blockSignals(True)
        self.min_area_spinbox.blockSignals(True)

        self.expand_spinbox.setValue(expand)
        self.simplify_spinbox.setValue(simplify)
        self.fill_holes_checkbox.setChecked(fill_holes)
        self.min_area_spinbox.setValue(min_area)

        self.expand_spinbox.blockSignals(False)
        self.simplify_spinbox.blockSignals(False)
        self.fill_holes_checkbox.blockSignals(False)
        self.min_area_spinbox.blockSignals(False)

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
            "background-color: rgba(25, 118, 210, 0.08); "
            "border: 1px solid rgba(25, 118, 210, 0.2); border-radius: 4px; "
            "padding: 2px 8px; font-size: 10px; color: palette(text);"
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
            plugin_data = plugins.all().get('AI_Segmentation')
            if plugin_data and plugin_data.get('status') == 'upgradeable':
                available_version = plugin_data.get(
                    'version_available', '?')
                text = '{} <a href="#update" style="color: #1976d2; font-weight: bold;">{}</a>'.format(
                    tr("A new version is available (v{version}).").format(
                        version=available_version),
                    tr("Update now"))
                self.update_notification_label.setText(text)
                self.update_notification_widget.setVisible(True)
        except Exception:
            pass  # No repo data yet, dev install, etc.

    def _on_open_plugin_manager(self, link=None):
        """Open the QGIS Plugin Manager on the Upgradeable tab (index 3)."""
        try:
            from qgis.utils import iface
            iface.pluginManagerInterface().showPluginManager(3)
        except Exception:
            pass

    def _setup_shortcuts_section(self, parent_layout):
        """Setup the collapsible keyboard shortcuts section."""
        self._shortcuts_expanded = False

        self._shortcuts_toggle = QLabel(
            '<a href="#" style="color: palette(text); '
            'text-decoration: none; font-size: 11px;">'
            '&#9654; ' + tr("Shortcuts") + '</a>'
        )
        self._shortcuts_toggle.setAlignment(Qt.AlignRight)
        self._shortcuts_toggle.setCursor(Qt.PointingHandCursor)
        self._shortcuts_toggle.linkActivated.connect(self._on_shortcuts_toggle)
        parent_layout.addWidget(self._shortcuts_toggle)

        undo_key = "Cmd+Z" if sys.platform == "darwin" else "Ctrl+Z"
        self._shortcuts_content = QLabel(
            "G : {start}\n"
            "S : {save}\n"
            "Enter : {export}\n"
            "{undo_key} : {undo}\n"
            "Esc : {stop}".format(
                start=tr("Start AI Segmentation"),
                save=tr("Save polygon"),
                export=tr("Export polygon to a layer"),
                undo_key=undo_key,
                undo=tr("Undo last point"),
                stop=tr("Stop segmentation"))
        )
        self._shortcuts_content.setStyleSheet(
            "font-size: 11px; color: palette(text); "
            "padding: 4px 8px; margin: 0;"
        )
        self._shortcuts_content.setAlignment(Qt.AlignRight)
        self._shortcuts_content.setVisible(False)
        parent_layout.addWidget(self._shortcuts_content)

    def _on_shortcuts_toggle(self):
        """Toggle shortcuts section visibility."""
        self._shortcuts_expanded = not self._shortcuts_expanded
        self._shortcuts_content.setVisible(self._shortcuts_expanded)
        arrow = "&#9660;" if self._shortcuts_expanded else "&#9654;"
        self._shortcuts_toggle.setText(
            '<a href="#" style="color: palette(text); '
            'text-decoration: none; font-size: 11px;">'
            '{} '.format(arrow) + tr("Shortcuts") + '</a>'
        )

    def _setup_about_section(self):
        """Setup the links section."""
        # Simple horizontal layout for links, aligned right with larger font
        links_widget = QWidget()
        links_layout = QHBoxLayout(links_widget)
        links_layout.setContentsMargins(0, 4, 0, 4)
        links_layout.setSpacing(16)

        links_layout.addStretch()  # Push links to the right

        # Report a bug button (styled as link)
        report_link = QLabel(
            '<a href="#" style="color: #1976d2;">' + tr("Report a bug") + '</a>'
        )
        report_link.setStyleSheet("font-size: 13px;")
        report_link.setCursor(Qt.PointingHandCursor)
        report_link.linkActivated.connect(self._on_report_bug)
        links_layout.addWidget(report_link)

        # Suggest feature button (styled as link)
        suggest_link = QLabel(
            '<a href="#" style="color: #1976d2;">' + tr("Suggest feature") + '</a>'
        )
        suggest_link.setStyleSheet("font-size: 13px;")
        suggest_link.setCursor(Qt.PointingHandCursor)
        suggest_link.linkActivated.connect(self._on_suggest_feature)
        links_layout.addWidget(suggest_link)

        # Tutorial link
        docs_link = QLabel(
            '<a href="https://terra-lab.ai/docs/ai-segmentation" style="color: #1976d2;">' + tr("Tutorial") + '</a>'
        )
        docs_link.setStyleSheet("font-size: 13px;")
        docs_link.setOpenExternalLinks(True)
        docs_link.setCursor(Qt.PointingHandCursor)
        links_layout.addWidget(docs_link)

        # About link
        about_link = QLabel(
            '<a href="https://terra-lab.ai/about" style="color: #1976d2;">' + tr("About us") + '</a>'
        )
        about_link.setStyleSheet("font-size: 13px;")
        about_link.setOpenExternalLinks(True)
        about_link.setCursor(Qt.PointingHandCursor)
        links_layout.addWidget(about_link)

        self.main_layout.addWidget(links_widget)

    def _on_report_bug(self):
        """Open the bug report dialog."""
        from .error_report_dialog import show_bug_report
        show_bug_report(self)

    def _on_suggest_feature(self):
        """Open the suggest a feature dialog."""
        from .error_report_dialog import show_suggest_feature
        show_suggest_feature(self)

    def is_batch_mode(self) -> bool:
        """Return whether batch mode is active (always True)."""
        return True

    def set_batch_mode(self, batch: bool):
        """Set batch mode programmatically. Batch is always on."""
        pass

    def _on_get_code_clicked(self):
        """Open the newsletter signup page in the default browser."""
        QDesktopServices.openUrl(QUrl(get_newsletter_url()))

    def _on_activate_clicked(self):
        """Attempt to activate the plugin with the entered code."""
        code = self.activation_code_input.text().strip()

        if not code:
            self._show_activation_message(tr("Enter your code"), is_error=True)
            return

        success, message = activate_plugin(code)

        if success:
            self._plugin_activated = True
            self._show_activation_message(tr("Unlocked!"), is_error=False)
            self._update_full_ui()
        else:
            self._show_activation_message(tr("Invalid code"), is_error=True)
            self.activation_code_input.selectAll()
            self.activation_code_input.setFocus()

    def _show_activation_message(self, text: str, is_error: bool = False):
        """Display a message in the activation section."""
        self.activation_message_label.setText(text)
        if is_error:
            self.activation_message_label.setStyleSheet("color: #d32f2f; font-size: 11px;")
        else:
            self.activation_message_label.setStyleSheet("color: #2e7d32; font-size: 11px;")
        self.activation_message_label.setVisible(True)

    def _update_full_ui(self):
        """Update the full UI based on current state."""
        # Segmentation section visibility: only show if deps + model + activated
        show_segmentation = self._dependencies_ok and self._checkpoint_ok and self._plugin_activated
        self.seg_widget.setVisible(show_segmentation)
        self.seg_separator.setVisible(show_segmentation)

        # Welcome section: hide when both deps and model are installed
        setup_complete = self._dependencies_ok and self._checkpoint_ok
        self.welcome_widget.setVisible(not setup_complete)

        # Checkpoint section: disable/grey out if dependencies not installed
        if not self._dependencies_ok:
            self.checkpoint_group.setEnabled(False)
            self.checkpoint_status_label.setText(tr("Waiting for Step 1..."))
            self.checkpoint_status_label.setStyleSheet("color: palette(text);")
            self.download_button.setVisible(False)
        else:
            self.checkpoint_group.setEnabled(True)

        # Activation section: show ONLY after deps+model ready, not activated, popup shown
        deps_ok = self._dependencies_ok
        checkpoint_ok = self._checkpoint_ok
        not_activated = not self._plugin_activated
        popup_shown = self._activation_popup_shown
        show_activation = deps_ok and checkpoint_ok and not_activated and popup_shown
        self.activation_group.setVisible(show_activation)

        # When showing activation panel, hide setup sections
        if show_activation:
            self.welcome_widget.setVisible(False)
            self.checkpoint_group.setVisible(False)

        self._update_ui_state()

    def _on_install_clicked(self):
        self.install_button.setEnabled(False)
        self.install_dependencies_requested.emit()

    def _on_cancel_deps_clicked(self):
        self.cancel_deps_install_requested.emit()

    def _on_download_clicked(self):
        self.download_button.setEnabled(False)
        self.download_checkpoint_requested.emit()

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

    def _on_layers_removed(self, layer_ids):
        """Handle layers removed from project."""
        self._update_ui_state()

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

    def update_gpu_info(self):
        """GPU info display disabled (CPU-only mode)."""
        return

    def get_cuda_enabled(self) -> bool:
        """Always CPU-only. GPU code kept for future reactivation."""
        return False

    def set_dependency_status(self, ok: bool, message: str):
        self._dependencies_ok = ok
        # Use "Not installed yet" for clearer messaging
        if not ok and message == tr("Dependencies not installed"):
            message = tr("Not installed yet")
        self.deps_status_label.setText(message)

        if ok:
            self.deps_status_label.setStyleSheet("font-weight: bold; color: palette(text);")
            self.install_button.setVisible(False)
            self.cancel_deps_button.setVisible(False)
            self.deps_progress.setVisible(False)
            self.deps_progress_label.setVisible(False)
            self.gpu_info_box.setVisible(False)
            self.deps_group.setVisible(False)
        else:
            self.deps_status_label.setStyleSheet("color: palette(text);")
            self.install_button.setVisible(True)
            self.install_button.setEnabled(True)
            # Detect update mode (deps exist but specs changed)
            is_update = "updating" in message.lower() or "upgrading" in message.lower()
            if is_update:
                self.install_button.setText(tr("Update Dependencies"))
            else:
                self.install_button.setText(tr("Install Dependencies"))
            self.deps_group.setVisible(True)

        self._update_full_ui()

    def set_deps_install_progress(self, percent: int, message: str):
        import time

        self._target_progress = percent

        time_info = ""
        now = time.time()
        if percent > 10 and percent < 100 and self._install_start_time:
            elapsed = now - self._install_start_time
            if elapsed > 5:
                overall_speed = percent / elapsed
                remaining_pct = 100 - percent

                # Weighted average: 70% recent speed, 30% overall
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
                    # Cap: 15 min for GPU, 8 min for CPU
                    max_remaining = 900 if self._is_cuda_install else 480
                    remaining = min(remaining, max_remaining)
                    if remaining > 60:
                        time_info = " (~{} min left)".format(int(remaining / 60))
                    elif remaining > 10:
                        time_info = " (~{} sec left)".format(int(remaining))

        # Track last percent for recent speed calculation
        if percent > self._last_percent:
            self._last_percent_time = now
            self._last_percent = percent

        self.deps_progress_label.setText("{}{}".format(message, time_info))

        # Detect update mode from button text set by set_dependency_status()
        is_update = self.install_button.text() in (
            tr("Update Dependencies"), tr("Updating..."))

        if percent == 0:
            self._install_start_time = time.time()
            self._current_progress = 0
            self._last_percent = 0
            self._last_percent_time = None
            self._creep_counter = 0
            self._is_cuda_install = False
            self.deps_progress.setValue(0)
            self.deps_progress.setVisible(True)
            self.deps_progress_label.setVisible(True)
            self.cancel_deps_button.setVisible(True)
            self.install_button.setEnabled(False)
            if is_update:
                self.install_button.setText(tr("Updating..."))
                self.deps_status_label.setText(tr("Updating dependencies..."))
            else:
                self.install_button.setText(tr("Installing..."))
                self.deps_status_label.setText(tr("Installing dependencies..."))
            self._progress_timer.start(500)
        elif percent >= 100 or "cancel" in message.lower() or "failed" in message.lower():
            self._progress_timer.stop()
            self._install_start_time = None
            self.deps_progress.setValue(percent)
            self.deps_progress.setVisible(False)
            self.deps_progress_label.setVisible(False)
            self.cancel_deps_button.setVisible(False)
            self.install_button.setEnabled(True)
            if is_update:
                self.install_button.setText(tr("Update Dependencies"))
            else:
                self.install_button.setText(tr("Install Dependencies"))
            if "cancel" in message.lower():
                self.deps_status_label.setText(tr("Installation cancelled"))
            elif "failed" in message.lower():
                self.deps_status_label.setText(tr("Installation failed"))
        else:
            if self._current_progress < percent:
                self._current_progress = percent
                self.deps_progress.setValue(percent)

    def _on_progress_tick(self):
        """Animate progress bar smoothly between updates."""
        if self._current_progress < self._target_progress:
            step = max(1, (self._target_progress - self._current_progress) // 3)
            self._current_progress = min(
                self._current_progress + step, self._target_progress)
            self._creep_counter = 0
        elif self._current_progress < 99 and self._target_progress > 0:
            # Slow creep during stalls: +1 every 4 ticks (2 sec)
            self._creep_counter += 1
            if self._creep_counter >= 4:
                self._creep_counter = 0
                if self._current_progress < self._target_progress + 5:
                    self._current_progress += 1

        self.deps_progress.setValue(self._current_progress)

    def set_checkpoint_status(self, ok: bool, message: str):
        self._checkpoint_ok = ok
        # Use "Not installed yet" for clearer messaging
        if not ok and message == tr("Model not found"):
            message = tr("Not installed yet")
        self.checkpoint_status_label.setText(message)

        if ok:
            self.checkpoint_status_label.setStyleSheet("font-weight: bold; color: palette(text);")
            self.download_button.setVisible(False)
            self.checkpoint_progress.setVisible(False)
            self.checkpoint_progress_label.setVisible(False)
            self.checkpoint_group.setVisible(False)
        else:
            self.checkpoint_status_label.setStyleSheet("color: palette(text);")
            self.download_button.setVisible(True)
            self.download_button.setEnabled(True)
            self.checkpoint_group.setVisible(True)

        self._update_full_ui()

    def set_download_progress(self, percent: int, message: str):
        self.checkpoint_progress.setValue(percent)
        self.checkpoint_progress_label.setText(message)

        if percent == 0:
            self.checkpoint_progress.setVisible(True)
            self.checkpoint_progress_label.setVisible(True)
            self.download_button.setEnabled(False)
            self.download_button.setText(tr("Downloading..."))
            self.checkpoint_status_label.setText(tr("Model downloading..."))
        elif percent >= 100:
            self.checkpoint_progress.setVisible(False)
            self.checkpoint_progress_label.setVisible(False)
            self.download_button.setEnabled(True)
            self.download_button.setText(
                tr("Download AI Segmentation Model ({size})").format(size=CHECKPOINT_SIZE_LABEL))

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
            # Not segmenting - hide all segmentation buttons, show start controls
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
        elif count == 1:
            self.export_button.setText(tr("Export polygon to a layer"))
        else:
            self.export_button.setText(tr("Export polygon to a layer"))

        if count > 0:
            self.export_button.setEnabled(True)
            self.export_button.setStyleSheet(
                "QPushButton { background-color: #4CAF50; padding: 6px 12px; }"
            )
        else:
            self.export_button.setEnabled(False)
            self.export_button.setStyleSheet(
                "QPushButton { background-color: #b0bec5; padding: 6px 12px; }"
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

        if total == 0:
            text = (
                tr("Click on the element you want to segment:") + "\n\n"
                "🟢 " + tr("Left-click to select")
            )
        else:
            text = (
                "🟢 " + tr("Left-click to add more") + "\n"
                "❌ " + tr("Right-click to exclude from selection")
            )

        self.instructions_label.setText(text)

    def set_disjoint_warning(self, visible: bool):
        self.disjoint_warning_widget.setVisible(visible)
        if self._segmentation_active:
            self.batch_info_widget.setVisible(not visible)

    def reset_session(self):
        self._has_mask = False
        self._segmentation_active = False
        self._segmentation_layer_id = None
        self._saved_polygon_count = 0
        self._positive_count = 0
        self._negative_count = 0
        self.disjoint_warning_widget.setVisible(False)
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
        return provider.name() in ('wms', 'wmts', 'xyz', 'arcgismapserver', 'wcs')

    def _is_layer_georeferenced(self, layer) -> bool:
        """Check if a raster layer is properly georeferenced."""
        if layer is None or layer.type() != layer.RasterLayer:
            return False

        # Check file extension for compatible formats
        source = layer.source().lower()

        # PNG, JPG, BMP etc. without world files are not georeferenced
        non_georef_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

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

    def _update_layer_filter(self):
        """Update the layer combo to exclude non-georeferenced rasters."""
        from qgis.core import QgsProject

        excluded_layers = []
        all_raster_count = 0

        for layer in QgsProject.instance().mapLayers().values():
            if layer.type() != layer.RasterLayer:
                continue
            all_raster_count += 1

        self.layer_combo.setExceptedLayerList(excluded_layers)

        # Update warning message
        if all_raster_count == 0:
            self.no_rasters_label.setText(
                tr("No layer found. Add a raster or online layer to your project.")
            )

    def _update_ui_state(self):
        # Update layer filter first
        self._update_layer_filter()

        layer = self.layer_combo.currentLayer()
        has_layer = layer is not None

        has_rasters_available = self.layer_combo.count() > 0
        self.no_rasters_widget.setVisible(not has_rasters_available and not self._segmentation_active)
        self.layer_combo.setVisible(has_rasters_available)

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
        dialog.exec_()

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

    def is_activated(self) -> bool:
        """Check if the plugin is activated."""
        return self._plugin_activated
