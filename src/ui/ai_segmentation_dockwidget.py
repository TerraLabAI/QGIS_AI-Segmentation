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
)
from qgis.PyQt.QtCore import Qt, pyqtSignal, QTimer, QUrl
from qgis.PyQt.QtGui import QDesktopServices

from qgis.core import QgsMapLayerProxyModel, QgsProject

from qgis.gui import QgsMapLayerComboBox

from ..core.activation_manager import (
    is_plugin_activated,
    activate_plugin,
    get_newsletter_url,
)


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
    refine_settings_changed = pyqtSignal(int, int)  # expand, simplify

    def __init__(self, parent=None):
        super().__init__("AI Segmentation by TerraLab", parent)

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

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
        self.deps_group = QGroupBox("Dependencies")
        layout = QVBoxLayout(self.deps_group)

        self.deps_status_label = QLabel("Checking dependencies...")
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

        self.install_button = QPushButton("Install Dependencies")
        self.install_button.clicked.connect(self._on_install_clicked)
        self.install_button.setVisible(False)
        self.install_button.setToolTip(
            "Create isolated virtual environment and install required packages\n"
            "(PyTorch, Segment Anything, pandas, rasterio)\n"
            "Download size: ~2.5GB"
        )
        layout.addWidget(self.install_button)

        self.cancel_deps_button = QPushButton("Cancel")
        self.cancel_deps_button.clicked.connect(self._on_cancel_deps_clicked)
        self.cancel_deps_button.setVisible(False)
        self.cancel_deps_button.setStyleSheet("background-color: #d32f2f;")
        layout.addWidget(self.cancel_deps_button)

        self.main_layout.addWidget(self.deps_group)

    def _setup_checkpoint_section(self):
        self.checkpoint_group = QGroupBox("AI Segmentation Model")
        layout = QVBoxLayout(self.checkpoint_group)

        self.checkpoint_status_label = QLabel("Checking model...")
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

        self.download_button = QPushButton("Download SAM Model (~375MB)")
        self.download_button.clicked.connect(self._on_download_clicked)
        self.download_button.setVisible(False)
        self.download_button.setToolTip("Download the SAM vit_b checkpoint for segmentation")
        layout.addWidget(self.download_button)

        self.cancel_download_button = QPushButton("Cancel")
        self.cancel_download_button.clicked.connect(self._on_cancel_download_clicked)
        self.cancel_download_button.setVisible(False)
        self.cancel_download_button.setStyleSheet("background-color: #d32f2f;")
        layout.addWidget(self.cancel_download_button)

        self.main_layout.addWidget(self.checkpoint_group)

    def _setup_activation_section(self):
        """Setup the minimal activation section - only shown if popup was closed without activating."""
        self.activation_group = QGroupBox("Unlock Plugin")
        layout = QVBoxLayout(self.activation_group)

        # Explanation about why we need the email
        desc_label = QLabel("Enter your email to receive updates and get a verification code.")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 11px; color: palette(text);")
        layout.addWidget(desc_label)

        # Get code button
        get_code_button = QPushButton("Get my verification code")
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
        code_label = QLabel("Then paste your code:")
        code_label.setStyleSheet("font-size: 11px; margin-top: 6px; color: palette(text);")
        layout.addWidget(code_label)

        # Code input section - compact
        code_layout = QHBoxLayout()
        code_layout.setSpacing(6)

        self.activation_code_input = QLineEdit()
        self.activation_code_input.setPlaceholderText("Code")
        self.activation_code_input.setMinimumHeight(28)
        self.activation_code_input.returnPressed.connect(self._on_activate_clicked)
        code_layout.addWidget(self.activation_code_input)

        self.activate_button = QPushButton("Unlock")
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

        layer_label = QLabel("Select a Raster Layer to Segment :")
        layer_label.setStyleSheet("font-weight: bold; color: palette(text);")
        layout.addWidget(layer_label)
        self.layer_label = layer_label

        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.layer_combo.setExcludedProviders(['wms', 'wmts', 'xyz', 'arcgismapserver', 'wcs'])
        self.layer_combo.setAllowEmptyLayer(False)
        self.layer_combo.setShowCrs(False)
        self.layer_combo.layerChanged.connect(self._on_layer_changed)
        self.layer_combo.setToolTip("Select a file-based raster layer (GeoTIFF, etc.)")
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

        self.no_rasters_label = QLabel("No compatible raster found. Add a GeoTIFF or local image to your project.")
        self.no_rasters_label.setWordWrap(True)
        no_rasters_layout.addWidget(self.no_rasters_label, 1)

        self.no_rasters_widget.setVisible(False)
        layout.addWidget(self.no_rasters_widget)

        # Dynamic instruction label - changes based on user state
        self.instructions_label = QLabel("")
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setStyleSheet(
            "font-size: 12px; padding: 8px 0px; color: palette(text);"
        )
        self.instructions_label.setToolTip(
            "Shortcuts: S (save mask) · Enter (export to layer) · Ctrl+Z (undo) · Escape (clear)"
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

        self.cancel_prep_button = QPushButton("Cancel")
        self.cancel_prep_button.clicked.connect(self._on_cancel_prep_clicked)
        self.cancel_prep_button.setVisible(False)
        self.cancel_prep_button.setMaximumHeight(26)
        self.cancel_prep_button.setStyleSheet(
            "QPushButton { background-color: #d32f2f; font-size: 10px; }"
        )
        layout.addWidget(self.cancel_prep_button)

        self.start_button = QPushButton("Start AI Segmentation")
        self.start_button.setEnabled(False)
        self.start_button.setMinimumHeight(36)
        self.start_button.clicked.connect(self._on_start_clicked)
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #2e7d32; font-weight: bold; font-size: 12px; }"
            "QPushButton:disabled { background-color: #c8e6c9; }"
        )
        self.start_button.setToolTip(
            "Click to segment objects on the image.\n"
            "Left-click = Select this element\n"
            "Right-click = Refine selection\n"
            "Multiple points refine the segmentation"
        )
        layout.addWidget(self.start_button)

        # Collapsible Refine mask panel
        self._setup_refine_panel(layout)

        # Primary action buttons (reduced height)
        self.save_mask_button = QPushButton("Save mask")
        self.save_mask_button.clicked.connect(self._on_save_polygon_clicked)
        self.save_mask_button.setVisible(False)
        self.save_mask_button.setEnabled(False)
        self.save_mask_button.setMinimumHeight(32)
        self.save_mask_button.setStyleSheet(
            "QPushButton { background-color: #1976d2; font-weight: bold; }"
            "QPushButton:disabled { background-color: #b0bec5; }"
        )
        self.save_mask_button.setToolTip(
            "Save current mask to your session (S)"
        )
        layout.addWidget(self.save_mask_button)

        self.export_button = QPushButton("Export to layer")
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setVisible(False)
        self.export_button.setEnabled(False)
        self.export_button.setMinimumHeight(32)
        self.export_button.setStyleSheet(
            "QPushButton { background-color: #b0bec5; }"
        )
        self.export_button.setToolTip(
            "Export all saved masks as a new vector layer (Enter)"
        )
        layout.addWidget(self.export_button)

        # Secondary action buttons (small, horizontal)
        secondary_layout = QHBoxLayout()
        secondary_layout.setSpacing(8)

        self.undo_button = QPushButton("Undo")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._on_undo_clicked)
        self.undo_button.setVisible(False)
        self.undo_button.setMaximumHeight(28)
        self.undo_button.setToolTip("Remove last point (Ctrl+Z)")
        secondary_layout.addWidget(self.undo_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setVisible(False)
        self.stop_button.setMaximumHeight(28)
        self.stop_button.setStyleSheet(
            "QPushButton { background-color: #757575; }"
        )
        self.stop_button.setToolTip("Exit segmentation without saving (Escape)")
        secondary_layout.addWidget(self.stop_button)

        self.secondary_buttons_widget = QWidget()
        self.secondary_buttons_widget.setLayout(secondary_layout)
        self.secondary_buttons_widget.setVisible(False)
        layout.addWidget(self.secondary_buttons_widget)

        self.main_layout.addWidget(self.seg_widget)

    def _setup_refine_panel(self, parent_layout):
        """Setup the collapsible Refine mask panel."""
        self.refine_group = QGroupBox("Refine mask")
        self.refine_group.setCheckable(True)
        self.refine_group.setChecked(False)  # Collapsed by default
        self.refine_group.setVisible(False)  # Hidden until segmentation active
        refine_layout = QVBoxLayout(self.refine_group)
        refine_layout.setSpacing(8)
        refine_layout.setContentsMargins(8, 8, 8, 8)

        # 1. Expand/Contract: SpinBox with +/- buttons (-20 to +20)
        expand_layout = QHBoxLayout()
        expand_label = QLabel("Expand/Contract:")
        expand_label.setToolTip("Positive = expand outward, Negative = shrink inward")
        self.expand_spinbox = QSpinBox()
        self.expand_spinbox.setRange(-20, 20)
        self.expand_spinbox.setValue(0)
        self.expand_spinbox.setSuffix(" px")
        self.expand_spinbox.setMinimumWidth(80)
        expand_layout.addWidget(expand_label)
        expand_layout.addStretch()
        expand_layout.addWidget(self.expand_spinbox)
        refine_layout.addLayout(expand_layout)

        # 2. Simplify outline: SpinBox (0 to 10) - reduces small variations in the outline
        simplify_layout = QHBoxLayout()
        simplify_label = QLabel("Simplify outline:")
        simplify_label.setToolTip("Reduce small variations in the outline (0 = no change)")
        self.simplify_spinbox = QSpinBox()
        self.simplify_spinbox.setRange(0, 10)
        self.simplify_spinbox.setValue(0)
        self.simplify_spinbox.setMinimumWidth(80)
        simplify_layout.addWidget(simplify_label)
        simplify_layout.addStretch()
        simplify_layout.addWidget(self.simplify_spinbox)
        refine_layout.addLayout(simplify_layout)

        # Connect signals
        self.expand_spinbox.valueChanged.connect(self._on_refine_changed)
        self.simplify_spinbox.valueChanged.connect(self._on_refine_changed)

        parent_layout.addWidget(self.refine_group)

    def _on_refine_changed(self, value=None):
        """Handle refine control changes with debounce."""
        self._refine_debounce_timer.start(150)

    def _emit_refine_changed(self):
        """Emit the refine settings changed signal after debounce."""
        self.refine_settings_changed.emit(
            self.expand_spinbox.value(),
            self.simplify_spinbox.value()
        )

    def reset_refine_sliders(self):
        """Reset refinement controls to default values."""
        self.expand_spinbox.setValue(0)
        self.simplify_spinbox.setValue(0)

    def _setup_about_section(self):
        """Setup the links section - minimal, just links."""
        # Simple horizontal layout for links, no group box
        links_widget = QWidget()
        links_layout = QHBoxLayout(links_widget)
        links_layout.setContentsMargins(0, 4, 0, 4)
        links_layout.setSpacing(16)

        # Documentation link
        docs_link = QLabel(
            '<a href="https://terra-lab.ai/docs/ai-segmentation" style="color: #1976d2;">Documentation</a>'
        )
        docs_link.setOpenExternalLinks(True)
        docs_link.setCursor(Qt.PointingHandCursor)
        links_layout.addWidget(docs_link)

        # Contact link
        contact_link = QLabel(
            '<a href="https://terra-lab.ai/contact" style="color: #1976d2;">Contact Us</a>'
        )
        contact_link.setOpenExternalLinks(True)
        contact_link.setCursor(Qt.PointingHandCursor)
        links_layout.addWidget(contact_link)

        links_layout.addStretch()

        self.main_layout.addWidget(links_widget)

    def _on_get_code_clicked(self):
        """Open the newsletter signup page in the default browser."""
        QDesktopServices.openUrl(QUrl(get_newsletter_url()))

    def _on_activate_clicked(self):
        """Attempt to activate the plugin with the entered code."""
        code = self.activation_code_input.text().strip()

        if not code:
            self._show_activation_message("Enter your code", is_error=True)
            return

        success, message = activate_plugin(code)

        if success:
            self._plugin_activated = True
            self._show_activation_message("Unlocked!", is_error=False)
            self._update_full_ui()
        else:
            self._show_activation_message("Invalid code", is_error=True)
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
            "Cancel Encoding?",
            "Are you sure you want to cancel?\n\n"
            "Once encoding is complete, it's cached permanently.\n"
            "You'll never need to wait for this image again.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.cancel_preparation_requested.emit()

    def _on_layer_changed(self, _layer):
        self._update_ui_state()

        if self._segmentation_active:
            self._segmentation_active = False
            self._update_button_visibility()
            QMessageBox.warning(
                self,
                "Layer Changed",
                "Segmentation cancelled because the layer was changed."
            )

    def _on_layers_added(self, layers):
        """Handle new layers added to project - auto-select if none selected."""
        if self.layer_combo.currentLayer() is not None:
            return

        for layer in layers:
            if layer.type() == layer.RasterLayer:
                provider = layer.dataProvider()
                if provider and provider.name() not in ['wms', 'wmts', 'xyz', 'arcgismapserver', 'wcs']:
                    self.layer_combo.setLayer(layer)
                    break

        self._update_ui_state()

    def _on_layers_removed(self, layer_ids):
        """Handle layers removed from project."""
        self._update_ui_state()

    def _on_start_clicked(self):
        layer = self.layer_combo.currentLayer()
        if layer:
            self.start_segmentation_requested.emit(layer)

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
            self.install_button.setText("Installing...")
            self.deps_status_label.setText("Installing dependencies...")
            self._progress_timer.start(500)
        elif percent >= 100 or "cancel" in message.lower() or "failed" in message.lower():
            self._progress_timer.stop()
            self._install_start_time = None
            self.deps_progress.setValue(percent)
            self.deps_progress.setVisible(False)
            self.deps_progress_label.setVisible(False)
            self.cancel_deps_button.setVisible(False)
            self.install_button.setEnabled(True)
            self.install_button.setText("Install Dependencies")
            if "cancel" in message.lower():
                self.deps_status_label.setText("Installation cancelled")
            elif "failed" in message.lower():
                self.deps_status_label.setText("Installation failed")
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
            self.download_button.setText("Downloading...")
            self.checkpoint_status_label.setText("Model downloading...")
        elif percent >= 100 or "cancel" in message.lower():
            self.checkpoint_progress.setVisible(False)
            self.checkpoint_progress_label.setVisible(False)
            self.cancel_download_button.setVisible(False)
            self.download_button.setEnabled(True)
            self.download_button.setText("Download SAM Model (~375MB)")
            if "cancel" in message.lower():
                self.checkpoint_status_label.setText("Download cancelled")

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
                "⏳ Encoding this image for AI segmentation...\n"
                "This is stored permanently, no waiting next time (:"
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
        """Show the cache path after encoding completes."""
        if cache_path:
            display_path = cache_path
            if len(display_path) > 45:
                display_path = "..." + display_path[-42:]
            self.encoding_info_label.setText(f"✓ Cached at:\n{display_path}")
            self.encoding_info_label.setStyleSheet(
                "background-color: rgba(46, 125, 50, 0.15); padding: 8px; "
                "border-radius: 4px; font-size: 10px; border: 1px solid rgba(46, 125, 50, 0.3); "
                "color: palette(text);"
            )
            self.encoding_info_label.setVisible(True)

    def set_segmentation_active(self, active: bool):
        self._segmentation_active = active
        self._update_button_visibility()
        self._update_ui_state()
        if active:
            self._update_instructions()

    def _update_button_visibility(self):
        if self._segmentation_active:
            self.start_button.setVisible(False)
            self.encoding_info_label.setVisible(False)
            self.instructions_label.setVisible(True)
            self._update_instructions()
            # Refine panel only visible after at least one mask is saved
            self.refine_group.setVisible(self._saved_polygon_count > 0)
            self.save_mask_button.setVisible(True)
            self.save_mask_button.setEnabled(self._has_mask)
            self.export_button.setVisible(True)
            self._update_export_button_style()
            self.undo_button.setVisible(True)
            self.stop_button.setVisible(True)
            self.secondary_buttons_widget.setVisible(True)
        else:
            self.start_button.setVisible(True)
            self.instructions_label.setVisible(False)
            self.refine_group.setVisible(False)
            self.save_mask_button.setVisible(False)
            self.export_button.setVisible(False)
            self.undo_button.setVisible(False)
            self.stop_button.setVisible(False)
            self.secondary_buttons_widget.setVisible(False)

    def _update_export_button_style(self):
        if self._saved_polygon_count > 0:
            self.export_button.setEnabled(True)
            self.export_button.setStyleSheet(
                "QPushButton { background-color: #4CAF50; font-weight: bold; }"
            )
            self.export_button.setToolTip(
                f"Export {self._saved_polygon_count} mask(s) as a new layer (Enter)"
            )
        else:
            self.export_button.setEnabled(False)
            self.export_button.setStyleSheet(
                "QPushButton { background-color: #b0bec5; }"
            )
            self.export_button.setToolTip(
                "Save at least one mask first (S)"
            )

    def set_point_count(self, positive: int, negative: int):
        self._positive_count = positive
        self._negative_count = negative
        total = positive + negative
        has_points = total > 0
        self._has_mask = has_points

        self.undo_button.setEnabled(has_points and self._segmentation_active)
        self.save_mask_button.setEnabled(has_points)

        if self._segmentation_active:
            self._update_instructions()

    def _update_instructions(self):
        """Update instruction text based on current segmentation state."""
        total = self._positive_count + self._negative_count

        if total == 0:
            # No points yet - only show green option
            text = (
                "Click on the element you want to segment:\n\n"
                "🟢 Left-click to select"
            )
        elif self._positive_count > 0 and self._negative_count == 0:
            # Has green points but no red yet - show both options
            counts = f"🟢 {self._positive_count} point(s)"
            text = (
                f"{counts}\n\n"
                "🟢 Left-click to add more\n"
                "❌ Right-click to exclude from selection"
            )
        else:
            # Has both types of points
            counts = f"🟢 {self._positive_count} point(s) · ❌ {self._negative_count} adjustment(s)"
            if self._saved_polygon_count > 0:
                state = f"{self._saved_polygon_count} mask(s) saved"
            else:
                state = "Refine selection or save mask"
            text = f"{counts}\n{state}"

        self.instructions_label.setText(text)

    def reset_session(self):
        self._has_mask = False
        self._segmentation_active = False
        self._saved_polygon_count = 0
        self._positive_count = 0
        self._negative_count = 0
        self.reset_refine_sliders()
        self._update_button_visibility()
        self._update_ui_state()

    def set_saved_polygon_count(self, count: int):
        self._saved_polygon_count = count
        self._update_button_visibility()
        self._update_export_button_style()

    def _update_ui_state(self):
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
