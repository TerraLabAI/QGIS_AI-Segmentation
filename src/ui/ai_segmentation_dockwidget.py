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
    QMessageBox,
    QLineEdit,
    QSpinBox,
    QCheckBox,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal, QTimer, QUrl
from qgis.PyQt.QtGui import QDesktopServices, QKeySequence
from qgis.PyQt.QtWidgets import QShortcut
from qgis.core import QgsMapLayerProxyModel, QgsProject

from qgis.gui import QgsMapLayerComboBox

from ..core.activation_manager import (
    is_plugin_activated,
    activate_plugin,
    get_newsletter_url,
)
from ..core.i18n import tr


class AISegmentationDockWidget(QDockWidget):

    install_dependencies_requested = pyqtSignal()
    cancel_deps_install_requested = pyqtSignal()
    download_checkpoint_requested = pyqtSignal()
    cancel_download_requested = pyqtSignal()
    cancel_preparation_requested = pyqtSignal()
    start_segmentation_requested = pyqtSignal(object)
    clear_points_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    save_polygon_requested = pyqtSignal()
    export_layer_requested = pyqtSignal()
    stop_segmentation_requested = pyqtSignal()
    refine_settings_changed = pyqtSignal(int, int, bool, int)  # expand, simplify, fill_holes, min_area
    batch_mode_changed = pyqtSignal(bool)  # True = batch mode, False = simple mode

    def __init__(self, parent=None):
        super().__init__("AI Segmentation by TerraLab", parent)

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Initialize state variables that are needed during UI setup
        self._refine_expanded = False  # Refine panel collapsed state persisted in session

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setSpacing(8)
        self.main_layout.setContentsMargins(8, 8, 8, 8)

        self._setup_ui()

        self.setWidget(self.main_widget)

        self._dependencies_ok = False
        self._checkpoint_ok = False
        self._segmentation_active = False
        self._has_mask = False
        self._saved_polygon_count = 0
        self._encoding_start_time = None
        self._positive_count = 0
        self._negative_count = 0
        self._plugin_activated = is_plugin_activated()
        self._activation_popup_shown = False  # Track if popup was shown
        self._batch_mode = False  # Simple mode by default
        self._segmentation_layer_id = None  # Track which layer we're segmenting
        # Note: _refine_expanded is initialized before _setup_ui() call

        # Smooth progress animation timer for long-running installs
        self._progress_timer = QTimer(self)
        self._progress_timer.timeout.connect(self._on_progress_tick)
        self._current_progress = 0
        self._target_progress = 0
        self._install_start_time = None

        # Debounce timer for refinement sliders
        self._refine_debounce_timer = QTimer(self)
        self._refine_debounce_timer.setSingleShot(True)
        self._refine_debounce_timer.timeout.connect(self._emit_refine_changed)

        # Connect to project layer signals for dynamic updates
        QgsProject.instance().layersAdded.connect(self._on_layers_added)
        QgsProject.instance().layersRemoved.connect(self._on_layers_removed)

        # Update UI state
        self._update_full_ui()

    def _setup_ui(self):
        self._setup_dependencies_section()
        self._setup_checkpoint_section()
        self._setup_activation_section()
        self._setup_segmentation_section()
        self.main_layout.addStretch()
        self._setup_about_section()

    def _setup_dependencies_section(self):
        self.deps_group = QGroupBox(tr("Dependencies"))
        layout = QVBoxLayout(self.deps_group)

        self.deps_status_label = QLabel(tr("Checking dependencies..."))
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
        self.install_button.setToolTip(
            tr("Create isolated virtual environment and install required packages") + "\n" +
            "(PyTorch, Segment Anything, pandas, rasterio)\n" +
            tr("Download size: ~800MB")
        )
        layout.addWidget(self.install_button)

        self.cancel_deps_button = QPushButton(tr("Cancel"))
        self.cancel_deps_button.clicked.connect(self._on_cancel_deps_clicked)
        self.cancel_deps_button.setVisible(False)
        self.cancel_deps_button.setStyleSheet("background-color: #d32f2f;")
        layout.addWidget(self.cancel_deps_button)

        self.main_layout.addWidget(self.deps_group)

    def _setup_checkpoint_section(self):
        self.checkpoint_group = QGroupBox(tr("AI Segmentation Model"))
        layout = QVBoxLayout(self.checkpoint_group)

        self.checkpoint_status_label = QLabel(tr("Checking model..."))
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

        self.download_button = QPushButton(tr("Download SAM Model (~375MB)"))
        self.download_button.clicked.connect(self._on_download_clicked)
        self.download_button.setVisible(False)
        self.download_button.setToolTip(tr("Download the SAM checkpoint for segmentation"))
        layout.addWidget(self.download_button)

        self.cancel_download_button = QPushButton(tr("Cancel"))
        self.cancel_download_button.clicked.connect(self._on_cancel_download_clicked)
        self.cancel_download_button.setVisible(False)
        self.cancel_download_button.setStyleSheet("background-color: #d32f2f;")
        layout.addWidget(self.cancel_download_button)

        self.main_layout.addWidget(self.checkpoint_group)

    def _setup_activation_section(self):
        """Setup the minimal activation section - only shown if popup was closed without activating."""
        self.activation_group = QGroupBox(tr("Unlock Plugin"))
        layout = QVBoxLayout(self.activation_group)

        # Explanation about why we need the email
        desc_label = QLabel(tr("Enter your email to receive updates and get a verification code."))
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 11px; color: palette(text);")
        layout.addWidget(desc_label)

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

        # Code input label
        code_label = QLabel(tr("Then paste your code:"))
        code_label.setStyleSheet("font-size: 11px; margin-top: 6px; color: palette(text);")
        layout.addWidget(code_label)

        # Code input section - compact
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
        self.layer_combo.setExcludedProviders(['wms', 'wmts', 'xyz', 'arcgismapserver', 'wcs'])
        self.layer_combo.setAllowEmptyLayer(False)
        self.layer_combo.setShowCrs(False)
        self.layer_combo.layerChanged.connect(self._on_layer_changed)
        self.layer_combo.setToolTip(tr("Select a file-based raster layer (GeoTIFF, etc.)"))
        layout.addWidget(self.layer_combo)

        # Warning container with icon and text - yellow background with white text
        self.no_rasters_widget = QWidget()
        self.no_rasters_widget.setStyleSheet(
            "QWidget { background-color: rgba(255, 193, 7, 0.4); "
            "border: 1px solid rgba(255, 152, 0, 0.6); border-radius: 4px; }"
            "QLabel { background: transparent; border: none; color: white; }"
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

        self.no_rasters_label = QLabel(tr("No compatible raster found. Add a GeoTIFF or local image to your project."))
        self.no_rasters_label.setWordWrap(True)
        no_rasters_layout.addWidget(self.no_rasters_label, 1)

        self.no_rasters_widget.setVisible(False)
        layout.addWidget(self.no_rasters_widget)

        # Mode indicator - shows current mode during segmentation
        self.mode_indicator_label = QLabel("")
        self.mode_indicator_label.setVisible(False)
        layout.addWidget(self.mode_indicator_label)

        # Dynamic instruction label - styled as a card
        self.instructions_label = QLabel("")
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setStyleSheet("""
            QLabel {
                background-color: rgba(25, 118, 210, 0.08);
                border: 1px solid rgba(25, 118, 210, 0.2);
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
                color: palette(text);
            }
        """)
        self.instructions_label.setToolTip(
            tr("Shortcuts: S (save mask) · Enter (export to layer) · Ctrl+Z (undo) · Escape (clear)")
        )
        self.instructions_label.setVisible(False)
        layout.addWidget(self.instructions_label)

        # Encoding progress section - green background with theme-compatible text
        self.encoding_info_label = QLabel("")
        self.encoding_info_label.setStyleSheet(
            "background-color: rgba(46, 125, 50, 0.15); padding: 8px; "
            "border-radius: 4px; font-size: 11px; border: 1px solid rgba(46, 125, 50, 0.3); "
            "color: palette(text);"
        )
        self.encoding_info_label.setWordWrap(True)
        self.encoding_info_label.setVisible(False)
        layout.addWidget(self.encoding_info_label)

        self.prep_progress = QProgressBar()
        self.prep_progress.setRange(0, 100)
        self.prep_progress.setVisible(False)
        layout.addWidget(self.prep_progress)

        self.prep_status_label = QLabel("")
        self.prep_status_label.setStyleSheet("color: palette(text); font-size: 11px;")
        self.prep_status_label.setVisible(False)
        layout.addWidget(self.prep_status_label)

        self.cancel_prep_button = QPushButton(tr("Cancel"))
        self.cancel_prep_button.clicked.connect(self._on_cancel_prep_clicked)
        self.cancel_prep_button.setVisible(False)
        self.cancel_prep_button.setMaximumHeight(26)
        self.cancel_prep_button.setStyleSheet(
            "QPushButton { background-color: #d32f2f; font-size: 10px; }"
        )
        layout.addWidget(self.cancel_prep_button)

        # Container for start button and batch mode checkbox
        self.start_container = QWidget()
        start_layout = QVBoxLayout(self.start_container)
        start_layout.setContentsMargins(0, 0, 0, 0)
        start_layout.setSpacing(6)

        self.start_button = QPushButton(tr("Start AI Segmentation"))
        self.start_button.setEnabled(False)
        self.start_button.setMinimumHeight(36)
        self.start_button.clicked.connect(self._on_start_clicked)
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #2e7d32; font-weight: bold; font-size: 12px; }"
            "QPushButton:disabled { background-color: #c8e6c9; }"
        )
        self.start_button.setToolTip(tr("Start segmentation (G)"))
        start_layout.addWidget(self.start_button)

        # Keyboard shortcut G to start segmentation
        self.start_shortcut = QShortcut(QKeySequence("G"), self)
        self.start_shortcut.activated.connect(self._on_start_shortcut)

        # Batch mode checkbox row - aligned right
        checkbox_row = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_row)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        checkbox_layout.addStretch()  # Push checkbox to the right

        self.batch_mode_checkbox = QCheckBox(tr("Batch mode"))
        self.batch_mode_checkbox.setChecked(False)
        self.batch_mode_checkbox.setToolTip(
            tr("Simple mode: One element per export.") + "\n" +
            tr("Batch mode: Save multiple masks, then export all together.") + "\n\n" +
            tr("Mode can only be changed when segmentation is stopped.")
        )
        self.batch_mode_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 12px;
                color: palette(text);
                padding: 2px 4px;
            }
            QCheckBox:disabled {
                color: palette(mid);
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
        self.batch_mode_checkbox.stateChanged.connect(self._on_batch_mode_checkbox_changed)
        checkbox_layout.addWidget(self.batch_mode_checkbox)

        start_layout.addWidget(checkbox_row)

        layout.addWidget(self.start_container)

        # Collapsible Refine mask panel
        self._setup_refine_panel(layout)

        # Primary action buttons (reduced height)
        self.save_mask_button = QPushButton(tr("Save mask"))
        self.save_mask_button.clicked.connect(self._on_save_polygon_clicked)
        self.save_mask_button.setVisible(False)
        self.save_mask_button.setEnabled(False)
        self.save_mask_button.setMinimumHeight(32)
        self.save_mask_button.setStyleSheet(
            "QPushButton { background-color: #1976d2; font-weight: bold; }"
            "QPushButton:disabled { background-color: #b0bec5; }"
        )
        self.save_mask_button.setToolTip(
            tr("Save current mask to your session (S)")
        )
        layout.addWidget(self.save_mask_button)

        self.export_button = QPushButton(tr("Export to layer"))
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setVisible(False)
        self.export_button.setEnabled(False)
        self.export_button.setMinimumHeight(32)
        self.export_button.setStyleSheet(
            "QPushButton { background-color: #b0bec5; }"
        )
        self.export_button.setToolTip(
            tr("Export all saved masks as a new vector layer (Enter)")
        )
        layout.addWidget(self.export_button)

        # Secondary action buttons (small, horizontal)
        secondary_layout = QHBoxLayout()
        secondary_layout.setSpacing(8)

        self.undo_button = QPushButton(tr("Undo last point"))
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._on_undo_clicked)
        self.undo_button.setVisible(False)  # Hidden until segmentation starts
        self.undo_button.setMaximumHeight(28)
        self.undo_button.setToolTip(tr("Remove last point (Ctrl+Z)"))
        secondary_layout.addWidget(self.undo_button)

        self.stop_button = QPushButton(tr("Stop segmentation"))
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setVisible(False)  # Hidden until segmentation starts
        self.stop_button.setMaximumHeight(28)
        self.stop_button.setStyleSheet(
            "QPushButton { background-color: #757575; }"
        )
        self.stop_button.setToolTip(tr("Exit segmentation without saving (Escape)"))
        secondary_layout.addWidget(self.stop_button)

        self.secondary_buttons_widget = QWidget()
        self.secondary_buttons_widget.setLayout(secondary_layout)
        self.secondary_buttons_widget.setVisible(False)
        layout.addWidget(self.secondary_buttons_widget)

        # Info box explaining one element per segmentation
        self.one_element_info_widget = QWidget()
        self.one_element_info_widget.setStyleSheet(
            "QWidget { background-color: rgba(255, 193, 7, 0.25); "
            "border: 1px solid rgba(255, 152, 0, 0.5); border-radius: 4px; }"
            "QLabel { background: transparent; border: none; }"
        )
        info_layout = QHBoxLayout(self.one_element_info_widget)
        info_layout.setContentsMargins(8, 6, 8, 6)
        info_layout.setSpacing(8)

        # Info icon
        info_icon_label = QLabel()
        style = self.one_element_info_widget.style()
        info_icon = style.standardIcon(style.SP_MessageBoxInformation)
        info_icon_label.setPixmap(info_icon.pixmap(14, 14))
        info_icon_label.setFixedSize(14, 14)
        info_layout.addWidget(info_icon_label, 0, Qt.AlignTop)

        # Info text
        info_text = QLabel(
            tr("One element per segmentation (e.g., one building, one car).") + "\n" +
            tr("For multiple elements in one layer, use Batch Mode.")
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("font-size: 10px; color: palette(text);")
        info_layout.addWidget(info_text, 1)

        self.one_element_info_widget.setVisible(False)
        layout.addWidget(self.one_element_info_widget)

        self.main_layout.addWidget(self.seg_widget)

    def _setup_refine_panel(self, parent_layout):
        """Setup the collapsible Refine mask panel (collapsible via click on title)."""
        self.refine_group = QGroupBox("▶ " + tr("Refine mask"))
        self.refine_group.setCheckable(False)  # No checkbox, just clickable title
        self.refine_group.setVisible(False)  # Hidden until segmentation active
        self.refine_group.setCursor(Qt.PointingHandCursor)
        self.refine_group.mousePressEvent = self._on_refine_group_clicked
        refine_layout = QVBoxLayout(self.refine_group)
        refine_layout.setSpacing(8)
        refine_layout.setContentsMargins(8, 8, 8, 8)

        # Content widget to show/hide
        self.refine_content_widget = QWidget()
        refine_content_layout = QVBoxLayout(self.refine_content_widget)
        refine_content_layout.setContentsMargins(0, 0, 0, 0)
        refine_content_layout.setSpacing(8)

        # 1. Expand/Contract: SpinBox with +/- buttons (-30 to +30)
        expand_layout = QHBoxLayout()
        expand_label = QLabel(tr("Expand/Contract:"))
        expand_label.setToolTip(tr("Positive = expand outward, Negative = shrink inward"))
        self.expand_spinbox = QSpinBox()
        self.expand_spinbox.setRange(-30, 30)
        self.expand_spinbox.setValue(0)
        self.expand_spinbox.setSuffix(" px")
        self.expand_spinbox.setMinimumWidth(80)
        expand_layout.addWidget(expand_label)
        expand_layout.addStretch()
        expand_layout.addWidget(self.expand_spinbox)
        refine_content_layout.addLayout(expand_layout)

        # 2. Simplify outline: SpinBox (0 to 20) - reduces small variations in the outline
        simplify_layout = QHBoxLayout()
        simplify_label = QLabel(tr("Simplify outline:"))
        simplify_label.setToolTip(tr("Reduce small variations in the outline (0 = no change)"))
        self.simplify_spinbox = QSpinBox()
        self.simplify_spinbox.setRange(0, 20)
        self.simplify_spinbox.setValue(2)  # Default to 2 for smoother outlines
        self.simplify_spinbox.setMinimumWidth(80)
        simplify_layout.addWidget(simplify_label)
        simplify_layout.addStretch()
        simplify_layout.addWidget(self.simplify_spinbox)
        refine_content_layout.addLayout(simplify_layout)

        # 3. Fill holes: Checkbox - fills interior holes in the mask
        fill_holes_layout = QHBoxLayout()
        self.fill_holes_checkbox = QCheckBox(tr("Fill holes"))
        self.fill_holes_checkbox.setChecked(False)  # Default: no fill holes
        self.fill_holes_checkbox.setToolTip(tr("Fill interior holes in the mask (holes completely surrounded by the selection)"))
        fill_holes_layout.addWidget(self.fill_holes_checkbox)
        fill_holes_layout.addStretch()
        refine_content_layout.addLayout(fill_holes_layout)

        # 4. Remove small artifacts: SpinBox - minimum area threshold
        min_area_layout = QHBoxLayout()
        min_area_label = QLabel(tr("Min. region size:"))
        min_area_label.setToolTip(tr("Remove disconnected regions smaller than this area (in pixels²).") + "\n" + tr("Example: 100 = ~10x10 pixel regions, 900 = ~30x30.") + "\n" + tr("0 = keep all."))
        self.min_area_spinbox = QSpinBox()
        self.min_area_spinbox.setRange(0, 10000)
        self.min_area_spinbox.setValue(200)  # Default: remove small artifacts
        self.min_area_spinbox.setSuffix(" px²")
        self.min_area_spinbox.setSingleStep(50)
        self.min_area_spinbox.setMinimumWidth(80)
        min_area_layout.addWidget(min_area_label)
        min_area_layout.addStretch()
        min_area_layout.addWidget(self.min_area_spinbox)
        refine_content_layout.addLayout(min_area_layout)

        refine_layout.addWidget(self.refine_content_widget)
        self.refine_content_widget.setVisible(self._refine_expanded)

        # Connect signals
        self.expand_spinbox.valueChanged.connect(self._on_refine_changed)
        self.simplify_spinbox.valueChanged.connect(self._on_refine_changed)
        self.fill_holes_checkbox.stateChanged.connect(self._on_refine_changed)
        self.min_area_spinbox.valueChanged.connect(self._on_refine_changed)

        parent_layout.addWidget(self.refine_group)

    def _on_refine_group_clicked(self, event):
        """Toggle the refine panel expanded/collapsed state (only when clicking on title)."""
        # Only toggle if click is in the title area (top ~25 pixels)
        # This prevents collapsing when clicking spinbox arrows at min/max values
        if event.pos().y() > 25:
            return  # Click was on content, not title - ignore

        self._refine_expanded = not self._refine_expanded
        self.refine_content_widget.setVisible(self._refine_expanded)
        arrow = "▼" if self._refine_expanded else "▶"
        self.refine_group.setTitle(f"{arrow} " + tr("Refine mask"))

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
        """Reset refinement controls to default values."""
        self.expand_spinbox.setValue(0)
        self.simplify_spinbox.setValue(2)  # Default to 2
        self.fill_holes_checkbox.setChecked(False)  # Default: no fill holes
        self.min_area_spinbox.setValue(200)  # Default: remove small artifacts

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

    def _setup_about_section(self):
        """Setup the links section."""
        # Simple horizontal layout for links, aligned right with larger font
        links_widget = QWidget()
        links_layout = QHBoxLayout(links_widget)
        links_layout.setContentsMargins(0, 4, 0, 4)
        links_layout.setSpacing(16)

        links_layout.addStretch()  # Push links to the right

        # Tutorial link
        tutorial_link = QLabel(
            '<a href="https://youtube.com/playlist?list=PL4hCF043nAUW2iIxALNUzy1fKHcCWwDsv&si=KO8kECsezunLe09p" style="color: #1976d2;">' + tr("Tutorials") + '</a>'
        )
        tutorial_link.setStyleSheet("font-size: 13px;")
        tutorial_link.setOpenExternalLinks(True)
        tutorial_link.setCursor(Qt.PointingHandCursor)
        links_layout.addWidget(tutorial_link)

        # Documentation link
        docs_link = QLabel(
            '<a href="https://terra-lab.ai/docs/ai-segmentation" style="color: #1976d2;">' + tr("Documentation") + '</a>'
        )
        docs_link.setStyleSheet("font-size: 13px;")
        docs_link.setOpenExternalLinks(True)
        docs_link.setCursor(Qt.PointingHandCursor)
        links_layout.addWidget(docs_link)

        # Contact link
        contact_link = QLabel(
            '<a href="https://terra-lab.ai/about" style="color: #1976d2;">' + tr("Contact Us") + '</a>'
        )
        contact_link.setStyleSheet("font-size: 13px;")
        contact_link.setOpenExternalLinks(True)
        contact_link.setCursor(Qt.PointingHandCursor)
        links_layout.addWidget(contact_link)

        self.main_layout.addWidget(links_widget)

    def _on_batch_mode_checkbox_changed(self, state: int):
        """Handle batch mode checkbox change."""
        checked = state == Qt.Checked
        self._batch_mode = checked
        self._update_ui_for_mode()
        self.batch_mode_changed.emit(checked)

    def _update_ui_for_mode(self):
        """Update UI elements based on current mode (simple vs batch)."""
        if self._batch_mode:
            # Batch mode: show Save mask button
            self.save_mask_button.setVisible(self._segmentation_active)
        else:
            # Simple mode: hide Save mask button
            self.save_mask_button.setVisible(False)

        self._update_button_visibility()

    def is_batch_mode(self) -> bool:
        """Return whether batch mode is active."""
        return self._batch_mode

    def set_batch_mode(self, batch: bool):
        """Set batch mode programmatically."""
        self._batch_mode = batch
        self.batch_mode_checkbox.setChecked(batch)
        self._update_ui_for_mode()

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

        # Activation section: show if deps OK but not activated AND popup was shown/closed
        deps_ok = self._dependencies_ok
        not_activated = not self._plugin_activated
        popup_shown = self._activation_popup_shown
        show_activation = deps_ok and not_activated and popup_shown
        self.activation_group.setVisible(show_activation)

        self._update_ui_state()

    def _on_install_clicked(self):
        self.install_button.setEnabled(False)
        self.install_dependencies_requested.emit()

    def _on_cancel_deps_clicked(self):
        self.cancel_deps_install_requested.emit()

    def _on_download_clicked(self):
        self.download_button.setEnabled(False)
        self.download_checkpoint_requested.emit()

    def _on_cancel_download_clicked(self):
        self.cancel_download_requested.emit()

    def _on_cancel_prep_clicked(self):
        reply = QMessageBox.question(
            self,
            tr("Cancel Encoding?"),
            tr("Are you sure you want to cancel?") + "\n\n" +
            tr("Once encoding is complete, it's cached permanently.") + "\n" +
            tr("You'll never need to wait for this image again."),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.cancel_preparation_requested.emit()

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
                provider = layer.dataProvider()
                if provider and provider.name() not in ['wms', 'wmts', 'xyz', 'arcgismapserver', 'wcs']:
                    # Only auto-select if georeferenced
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

    def set_dependency_status(self, ok: bool, message: str):
        self._dependencies_ok = ok
        self.deps_status_label.setText(message)

        if ok:
            self.deps_status_label.setStyleSheet("font-weight: bold; color: palette(text);")
            self.install_button.setVisible(False)
            self.cancel_deps_button.setVisible(False)
            self.deps_progress.setVisible(False)
            self.deps_progress_label.setVisible(False)
            self.deps_group.setVisible(False)
        else:
            self.deps_status_label.setStyleSheet("color: palette(text);")
            self.install_button.setVisible(True)
            self.install_button.setEnabled(True)
            self.deps_group.setVisible(True)

        self._update_full_ui()

    def set_deps_install_progress(self, percent: int, message: str):
        import time

        self._target_progress = percent

        time_info = ""
        if percent > 5 and percent < 100 and self._install_start_time:
            elapsed = time.time() - self._install_start_time
            if elapsed > 5:
                estimated_total = elapsed / (percent / 100)
                remaining = estimated_total - elapsed
                if remaining > 60:
                    time_info = f" (~{int(remaining / 60)} min left)"
                elif remaining > 10:
                    time_info = f" (~{int(remaining)} sec left)"

        self.deps_progress_label.setText(f"{message}{time_info}")

        if percent == 0:
            self._install_start_time = time.time()
            self._current_progress = 0
            self.deps_progress.setValue(0)
            self.deps_progress.setVisible(True)
            self.deps_progress_label.setVisible(True)
            self.cancel_deps_button.setVisible(True)
            self.install_button.setEnabled(False)
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
            self._current_progress = min(self._current_progress + step, self._target_progress)
        elif self._current_progress < 99 and self._target_progress > 0:
            if self._current_progress < self._target_progress + 3:
                self._current_progress += 1

        self.deps_progress.setValue(self._current_progress)

    def set_checkpoint_status(self, ok: bool, message: str):
        self._checkpoint_ok = ok
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
            self.cancel_download_button.setVisible(True)
            self.download_button.setEnabled(False)
            self.download_button.setText(tr("Downloading..."))
            self.checkpoint_status_label.setText(tr("Model downloading..."))
        elif percent >= 100 or "cancel" in message.lower():
            self.checkpoint_progress.setVisible(False)
            self.checkpoint_progress_label.setVisible(False)
            self.cancel_download_button.setVisible(False)
            self.download_button.setEnabled(True)
            self.download_button.setText(tr("Download SAM Model (~375MB)"))
            if "cancel" in message.lower():
                self.checkpoint_status_label.setText(tr("Download cancelled"))

    def set_preparation_progress(self, percent: int, message: str, cache_path: str = None):
        import time

        self.prep_progress.setValue(percent)

        time_info = ""
        if percent > 5 and percent < 100 and self._encoding_start_time:
            elapsed = time.time() - self._encoding_start_time
            if elapsed > 2:
                estimated_total = elapsed / (percent / 100)
                remaining = estimated_total - elapsed
                if remaining > 60:
                    time_info = f" (~{int(remaining / 60)} min left)"
                elif remaining > 10:
                    time_info = f" (~{int(remaining)} sec left)"

        self.prep_status_label.setText(f"{message}{time_info}")

        if percent == 0:
            self._encoding_start_time = time.time()
            self.prep_progress.setVisible(True)
            self.prep_status_label.setVisible(True)
            self.start_button.setVisible(False)
            self.cancel_prep_button.setVisible(True)
            self.encoding_info_label.setText(
                "⏳ " + tr("Encoding this image for AI segmentation...") + "\n" +
                tr("This is stored permanently, no waiting next time :)")
            )
            self.encoding_info_label.setVisible(True)
        elif percent >= 100 or "cancel" in message.lower():
            self.prep_progress.setVisible(False)
            self.prep_status_label.setVisible(False)
            self.cancel_prep_button.setVisible(False)
            self.encoding_info_label.setVisible(False)
            self.start_button.setVisible(True)
            self._encoding_start_time = None
            self._update_ui_state()

    def set_encoding_cache_path(self, cache_path: str):
        """Hide the encoding info label when encoding completes (no longer show cache path)."""
        # Simply hide the label when ready - no need to show cache path to user
        self.encoding_info_label.setVisible(False)

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
            self.start_container.setVisible(False)  # Hide start button and mode checkbox
            self.encoding_info_label.setVisible(False)
            self.instructions_label.setVisible(True)
            self._update_instructions()

            # Show mode indicator
            self._update_mode_indicator()

            # Refine panel visibility
            self._update_refine_panel_visibility()

            # Save mask button: only in batch mode
            if self._batch_mode:
                self.save_mask_button.setVisible(True)
                self.save_mask_button.setEnabled(self._has_mask)
            else:
                self.save_mask_button.setVisible(False)

            # Export button: visible during segmentation
            self.export_button.setVisible(True)
            self._update_export_button_style()

            # Secondary buttons: visible during segmentation
            self.secondary_buttons_widget.setVisible(True)
            self.undo_button.setVisible(True)
            # Undo enabled if: has points OR (batch mode AND has saved masks)
            has_points = self._positive_count > 0 or self._negative_count > 0
            can_undo_saved = self._batch_mode and self._saved_polygon_count > 0
            self.undo_button.setEnabled(has_points or can_undo_saved)
            self.stop_button.setVisible(True)
            self.stop_button.setEnabled(True)

            # Info box: only visible in Simple mode
            self.one_element_info_widget.setVisible(not self._batch_mode)

            # Batch mode checkbox: disabled during segmentation (can't change mode mid-session)
            self.batch_mode_checkbox.setEnabled(False)
        else:
            # Not segmenting - hide all segmentation buttons, show start controls
            self.start_container.setVisible(True)
            self.batch_mode_checkbox.setEnabled(True)  # Can change mode when not segmenting
            self.mode_indicator_label.setVisible(False)
            self.instructions_label.setVisible(False)
            self.refine_group.setVisible(False)
            self.save_mask_button.setVisible(False)
            self.export_button.setVisible(False)
            self.undo_button.setVisible(False)
            self.stop_button.setVisible(False)
            self.secondary_buttons_widget.setVisible(False)
            self.one_element_info_widget.setVisible(False)

    def _update_mode_indicator(self):
        """Update the mode indicator label based on current mode."""
        if self._batch_mode:
            self.mode_indicator_label.setText(tr("Batch mode"))
            self.mode_indicator_label.setStyleSheet("""
                QLabel {
                    background-color: rgba(25, 118, 210, 0.15);
                    border: 1px solid rgba(25, 118, 210, 0.3);
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 11px;
                    font-weight: bold;
                    color: palette(text);
                }
            """)
        else:
            self.mode_indicator_label.setText(tr("Simple mode"))
            self.mode_indicator_label.setStyleSheet("""
                QLabel {
                    background-color: rgba(46, 125, 50, 0.15);
                    border: 1px solid rgba(46, 125, 50, 0.3);
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 11px;
                    font-weight: bold;
                    color: palette(text);
                }
            """)
        self.mode_indicator_label.setVisible(True)

    def _update_refine_panel_visibility(self):
        """Update refine panel visibility based on mode and mask state."""
        if not self._segmentation_active:
            self.refine_group.setVisible(False)
            return

        if self._batch_mode:
            # Batch mode: show when current mask exists (refine only affects current blue mask)
            show_refine = self._has_mask
        else:
            # Simple mode: show when current mask exists (points placed)
            show_refine = self._has_mask
        self.refine_group.setVisible(show_refine)

    def _update_export_button_style(self):
        if self._batch_mode:
            # Batch mode: need saved polygons to export
            if self._saved_polygon_count > 0:
                self.export_button.setEnabled(True)
                self.export_button.setStyleSheet(
                    "QPushButton { background-color: #4CAF50; font-weight: bold; }"
                )
                self.export_button.setToolTip(
                    tr("Export {count} mask(s) as a new layer (Enter)").format(count=self._saved_polygon_count)
                )
            else:
                self.export_button.setEnabled(False)
                self.export_button.setStyleSheet(
                    "QPushButton { background-color: #b0bec5; }"
                )
                self.export_button.setToolTip(
                    tr("Save at least one mask first (S)")
                )
        else:
            # Simple mode: export current mask directly
            if self._has_mask:
                self.export_button.setEnabled(True)
                self.export_button.setStyleSheet(
                    "QPushButton { background-color: #4CAF50; font-weight: bold; }"
                )
                self.export_button.setToolTip(
                    tr("Export current mask as a new layer (Enter)")
                )
            else:
                self.export_button.setEnabled(False)
                self.export_button.setStyleSheet(
                    "QPushButton { background-color: #b0bec5; }"
                )
                self.export_button.setToolTip(
                    tr("Place points to create a mask first")
                )

    def set_point_count(self, positive: int, negative: int):
        self._positive_count = positive
        self._negative_count = negative
        total = positive + negative
        has_points = total > 0
        old_has_mask = self._has_mask
        self._has_mask = has_points

        # Undo enabled if: has points OR (batch mode AND has saved masks)
        can_undo_saved = self._batch_mode and self._saved_polygon_count > 0
        self.undo_button.setEnabled((has_points or can_undo_saved) and self._segmentation_active)
        self.save_mask_button.setEnabled(has_points)

        if self._segmentation_active:
            self._update_instructions()
            self._update_export_button_style()  # Update export button for simple mode
            # Update refine panel visibility when mask state changes (for simple mode)
            if old_has_mask != self._has_mask:
                self._update_refine_panel_visibility()

    def _update_instructions(self):
        """Update instruction text based on current segmentation state."""
        total = self._positive_count + self._negative_count

        if total == 0:
            # No points yet - only show green option
            text = (
                tr("Click on the element you want to segment:") + "\n\n"
                "🟢 " + tr("Left-click to select")
            )
        elif self._positive_count > 0 and self._negative_count == 0:
            # Has green points but no red yet - show both options
            counts = "🟢 " + tr("{count} point(s)").format(count=self._positive_count)
            text = (
                f"{counts}\n\n"
                "🟢 " + tr("Left-click to add more") + "\n"
                "❌ " + tr("Right-click to exclude from selection")
            )
        else:
            # Has both types of points
            counts = "🟢 " + tr("{count} point(s)").format(count=self._positive_count) + " · ❌ " + tr("{count} adjustment(s)").format(count=self._negative_count)
            if self._saved_polygon_count > 0:
                state = tr("{count} mask(s) saved").format(count=self._saved_polygon_count)
            else:
                state = tr("Refine selection or save mask")
            text = f"{counts}\n{state}"

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
        # Update undo button state (may be enabled/disabled based on saved count in batch mode)
        if self._segmentation_active:
            has_points = self._positive_count > 0 or self._negative_count > 0
            can_undo_saved = self._batch_mode and count > 0
            self.undo_button.setEnabled(has_points or can_undo_saved)

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
            # Skip web services
            provider = layer.dataProvider()
            if provider and provider.name() in ['wms', 'wmts', 'xyz', 'arcgismapserver', 'wcs']:
                continue

            all_raster_count += 1
            if not self._is_layer_georeferenced(layer):
                excluded_layers.append(layer)

        self.layer_combo.setExceptedLayerList(excluded_layers)

        # Update warning message based on situation
        compatible_count = all_raster_count - len(excluded_layers)

        if compatible_count == 0 and all_raster_count > 0:
            # Has rasters but none are georeferenced
            self.no_rasters_label.setText(
                tr("No compatible raster found. {count} layer(s) excluded (PNG/JPG without georeferencing). Use GeoTIFF format.").format(count=all_raster_count)
            )
        else:
            # Default message
            self.no_rasters_label.setText(
                tr("No compatible raster found. Add a GeoTIFF or georeferenced image to your project.")
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
        """Show the activation dialog (called from plugin during install)."""
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

    def is_activated(self) -> bool:
        """Check if the plugin is activated."""
        return self._plugin_activated
