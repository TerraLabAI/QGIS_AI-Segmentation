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
    QSizePolicy,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal, QTimer

from qgis.core import QgsMapLayerProxyModel, QgsProject
from qgis.gui import QgsMapLayerComboBox


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

        # Smooth progress animation timer for long-running installs
        self._progress_timer = QTimer(self)
        self._progress_timer.timeout.connect(self._on_progress_tick)
        self._current_progress = 0
        self._target_progress = 0
        self._install_start_time = None

        # Connect to project layer signals for dynamic updates
        QgsProject.instance().layersAdded.connect(self._on_layers_added)
        QgsProject.instance().layersRemoved.connect(self._on_layers_removed)

    def _setup_ui(self):
        self._setup_dependencies_section()
        self._setup_checkpoint_section()
        self._setup_segmentation_section()
        self.main_layout.addStretch()
        self._setup_status_bar()

    def _setup_dependencies_section(self):
        self.deps_group = QGroupBox("Dependencies")
        layout = QVBoxLayout(self.deps_group)

        self.deps_status_label = QLabel("Checking dependencies...")
        layout.addWidget(self.deps_status_label)

        self.deps_progress = QProgressBar()
        self.deps_progress.setRange(0, 100)
        self.deps_progress.setVisible(False)
        layout.addWidget(self.deps_progress)

        self.deps_progress_label = QLabel("")
        self.deps_progress_label.setStyleSheet("color: #666; font-size: 10px;")
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
        layout.addWidget(self.checkpoint_status_label)

        self.checkpoint_progress = QProgressBar()
        self.checkpoint_progress.setRange(0, 100)
        self.checkpoint_progress.setVisible(False)
        layout.addWidget(self.checkpoint_progress)

        self.checkpoint_progress_label = QLabel("")
        self.checkpoint_progress_label.setStyleSheet("color: #666; font-size: 10px;")
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

    def _setup_segmentation_section(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(sep)

        seg_widget = QWidget()
        layout = QVBoxLayout(seg_widget)
        layout.setContentsMargins(0, 8, 0, 0)

        layer_label = QLabel("Select a Raster Layer to Segment")
        layer_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(layer_label)

        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.layer_combo.setExcludedProviders(['wms', 'wmts', 'xyz', 'arcgismapserver', 'wcs'])
        self.layer_combo.setAllowEmptyLayer(False)
        self.layer_combo.setShowCrs(False)
        self.layer_combo.layerChanged.connect(self._on_layer_changed)
        self.layer_combo.setToolTip("Select a file-based raster layer (GeoTIFF, etc.)")
        layout.addWidget(self.layer_combo)

        self.no_rasters_label = QLabel("No compatible raster found. Add a GeoTIFF or local image to your project.")
        self.no_rasters_label.setStyleSheet(
            "background-color: #fff3cd; padding: 8px; "
            "border-radius: 4px; font-size: 11px;"
        )
        self.no_rasters_label.setWordWrap(True)
        self.no_rasters_label.setVisible(False)
        layout.addWidget(self.no_rasters_label)

        # Dynamic instruction label - changes based on user state
        self.instructions_label = QLabel("")
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setStyleSheet(
            "font-size: 12px; padding: 8px 0px;"
        )
        self.instructions_label.setToolTip(
            "Shortcuts: S (save polygon) · Enter (export to layer) · Ctrl+Z (undo) · Escape (clear)"
        )
        self.instructions_label.setVisible(False)
        layout.addWidget(self.instructions_label)

        # Encoding progress section
        self.encoding_info_label = QLabel("")
        self.encoding_info_label.setStyleSheet(
            "background-color: #e8f5e9; padding: 8px; "
            "border-radius: 4px; font-size: 11px;"
        )
        self.encoding_info_label.setWordWrap(True)
        self.encoding_info_label.setVisible(False)
        layout.addWidget(self.encoding_info_label)

        self.prep_progress = QProgressBar()
        self.prep_progress.setRange(0, 100)
        self.prep_progress.setVisible(False)
        layout.addWidget(self.prep_progress)

        self.prep_status_label = QLabel("")
        self.prep_status_label.setStyleSheet("color: #555; font-size: 11px;")
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
            "Left-click = Include area (add positive points)\n"
            "Right-click = Exclude area (add negative points)\n"
            "Multiple points refine the segmentation"
        )
        layout.addWidget(self.start_button)

        # Primary action buttons (large)
        self.save_polygon_button = QPushButton("Add Polygon")
        self.save_polygon_button.clicked.connect(self._on_save_polygon_clicked)
        self.save_polygon_button.setVisible(False)
        self.save_polygon_button.setEnabled(False)
        self.save_polygon_button.setMinimumHeight(36)
        self.save_polygon_button.setStyleSheet(
            "QPushButton { background-color: #1976d2; font-weight: bold; }"
            "QPushButton:disabled { background-color: #b0bec5; }"
        )
        self.save_polygon_button.setToolTip("Add current polygon to collection (S)")
        layout.addWidget(self.save_polygon_button)

        self.export_button = QPushButton("Save as Layer")
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setVisible(False)
        self.export_button.setEnabled(False)
        self.export_button.setMinimumHeight(36)
        self.export_button.setStyleSheet(
            "QPushButton { background-color: #b0bec5; }"
        )
        self.export_button.setToolTip("Add polygons first, then save as layer (Enter)")
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

        self.main_layout.addWidget(seg_widget)

    def _setup_status_bar(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(sep)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(
            "background-color: #424242; font-size: 12px; "
            "padding: 8px; border-radius: 4px;"
        )
        self.status_label.setWordWrap(True)
        self.main_layout.addWidget(self.status_label)

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
        # Only auto-select if no layer is currently selected
        if self.layer_combo.currentLayer() is not None:
            return

        # Check if any of the new layers are compatible rasters
        for layer in layers:
            if layer.type() == layer.RasterLayer:
                provider = layer.dataProvider()
                if provider and provider.name() not in ['wms', 'wmts', 'xyz', 'arcgismapserver', 'wcs']:
                    # Found a compatible raster, select it
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
            self.deps_status_label.setStyleSheet("font-weight: bold;")
            self.install_button.setVisible(False)
            self.cancel_deps_button.setVisible(False)
            self.deps_progress.setVisible(False)
            self.deps_progress_label.setVisible(False)
            self.deps_group.setVisible(False)
        else:
            self.deps_status_label.setStyleSheet("")
            self.install_button.setVisible(True)
            self.install_button.setEnabled(True)
            self.deps_group.setVisible(True)

        self._update_ui_state()

    def set_deps_install_progress(self, percent: int, message: str):
        import time

        # Store target progress for smooth animation
        self._target_progress = percent

        # Calculate time estimate
        time_info = ""
        if percent > 5 and percent < 100 and self._install_start_time:
            elapsed = time.time() - self._install_start_time
            if elapsed > 5:  # Only show after 5 seconds
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
            # Start smooth progress animation
            self._progress_timer.start(500)  # Tick every 500ms
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
            # Update progress smoothly - jump to actual progress if we're behind
            if self._current_progress < percent:
                self._current_progress = percent
                self.deps_progress.setValue(percent)

    def _on_progress_tick(self):
        """Animate progress bar smoothly between updates."""
        # Only animate if we haven't reached the target yet
        if self._current_progress < self._target_progress:
            # Catch up to target
            step = max(1, (self._target_progress - self._current_progress) // 3)
            self._current_progress = min(self._current_progress + step, self._target_progress)
        elif self._current_progress < 99 and self._target_progress > 0:
            # Slowly advance toward target to show activity (max 1% per tick)
            # But don't exceed target by more than a small buffer
            if self._current_progress < self._target_progress + 3:
                self._current_progress += 1

        self.deps_progress.setValue(self._current_progress)


    def set_checkpoint_status(self, ok: bool, message: str):
        self._checkpoint_ok = ok
        self.checkpoint_status_label.setText(message)

        if ok:
            self.checkpoint_status_label.setStyleSheet("font-weight: bold;")
            self.download_button.setVisible(False)
            self.checkpoint_progress.setVisible(False)
            self.checkpoint_progress_label.setVisible(False)
            self.checkpoint_group.setVisible(False)
        else:
            self.checkpoint_status_label.setStyleSheet("")
            self.download_button.setVisible(True)
            self.download_button.setEnabled(True)
            self.checkpoint_group.setVisible(True)

        self._update_ui_state()

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

        # Calculate time estimate
        time_info = ""
        if percent > 5 and percent < 100 and self._encoding_start_time:
            elapsed = time.time() - self._encoding_start_time
            if elapsed > 2:  # Only show after 2 seconds
                estimated_total = elapsed / (percent / 100)
                remaining = estimated_total - elapsed
                if remaining > 60:
                    time_info = f" (~{int(remaining / 60)} min left)"
                elif remaining > 10:
                    time_info = f" (~{int(remaining)} sec left)"

        self.prep_status_label.setText(f"{message}{time_info}")

        if percent == 0:
            import time
            self._encoding_start_time = time.time()
            self.prep_progress.setVisible(True)
            self.prep_status_label.setVisible(True)
            self.start_button.setVisible(False)
            self.cancel_prep_button.setVisible(True)
            self.encoding_info_label.setText(
                "⏳ Preparing image for segmentation...\n"
                "One-time only - results are cached."
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
                "background-color: #e8f5e9; padding: 8px; "
                "border-radius: 4px; font-size: 10px;"
            )
            self.encoding_info_label.setVisible(True)

    def set_segmentation_active(self, active: bool):
        self._segmentation_active = active
        self._update_button_visibility()
        self._update_ui_state()

    def _update_button_visibility(self):
        if self._segmentation_active:
            self.start_button.setVisible(False)
            self.encoding_info_label.setVisible(False)
            self.instructions_label.setVisible(True)
            self._update_instructions()
            self.save_polygon_button.setVisible(True)
            self.save_polygon_button.setEnabled(self._has_mask)
            self.export_button.setVisible(True)
            self._update_export_button_style()
            self.undo_button.setVisible(True)
            self.stop_button.setVisible(True)
            self.secondary_buttons_widget.setVisible(True)
        else:
            self.start_button.setVisible(True)
            self.instructions_label.setVisible(False)
            self.save_polygon_button.setVisible(False)
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
            self.export_button.setToolTip(f"Save {self._saved_polygon_count} polygon(s) as a new layer (Enter)")
        else:
            self.export_button.setEnabled(False)
            self.export_button.setStyleSheet(
                "QPushButton { background-color: #b0bec5; }"
            )
            self.export_button.setToolTip("Add polygons first, then save as layer (Enter)")

    def set_point_count(self, positive: int, negative: int):
        self._positive_count = positive
        self._negative_count = negative
        total = positive + negative
        has_points = total > 0
        self._has_mask = has_points

        self.undo_button.setEnabled(has_points and self._segmentation_active)
        self.save_polygon_button.setEnabled(has_points)

        # Update dynamic instructions
        if self._segmentation_active:
            self._update_instructions()
            self._update_status_hint()

    def set_status(self, message: str):
        self.status_label.setText(message)

    def _update_instructions(self):
        """Update instruction text based on current segmentation state."""
        total = self._positive_count + self._negative_count

        if total == 0:
            # No points yet - invite user to click
            text = "Click on the element you want to segment"
        else:
            # Show click counts with simple indicators
            # ● = include (green marker on map), ✕ = exclude (red marker on map)
            text = f"● {self._positive_count} include · ✕ {self._negative_count} exclude"

        self.instructions_label.setText(text)

    def _update_status_hint(self):
        """Show contextual hints in status bar (shortcuts, tips)."""
        total = self._positive_count + self._negative_count

        if total == 0:
            # No points - hint about clicking
            hint = "Left-click: include · Right-click: exclude"
        elif total == 1 and self._saved_polygon_count == 0:
            # First point - hint about saving
            hint = "S: save polygon · Ctrl+Z: undo"
        elif self._saved_polygon_count > 0:
            # Has saved polygons - hint about exporting
            hint = f"{self._saved_polygon_count} polygon(s) ready · Enter: export to layer"
        else:
            # Multiple points - hint about actions
            hint = "S: save · Enter: export · Ctrl+Z: undo"

        self.status_label.setText(hint)

    def reset_session(self):
        self._has_mask = False
        self._segmentation_active = False
        self._saved_polygon_count = 0
        self._positive_count = 0
        self._negative_count = 0
        self._update_button_visibility()
        self._update_ui_state()

    def set_saved_polygon_count(self, count: int):
        self._saved_polygon_count = count
        self._update_button_visibility()
        self._update_export_button_style()

    def _update_ui_state(self):
        layer = self.layer_combo.currentLayer()
        has_layer = layer is not None

        # Check if there are any rasters available
        has_rasters_available = self.layer_combo.count() > 0
        self.no_rasters_label.setVisible(not has_rasters_available and not self._segmentation_active)

        # Hide layer combo if no rasters available
        self.layer_combo.setVisible(has_rasters_available)

        can_start = (
            self._dependencies_ok and
            self._checkpoint_ok and
            has_layer
        )
        self.start_button.setEnabled(can_start and not self._segmentation_active)
