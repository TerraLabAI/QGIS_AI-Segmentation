"""
Dock Widget for AI Segmentation

Main user interface panel for the segmentation plugin.
Shows clear status for each step: Dependencies → Models → Ready
"""

from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QProgressBar,
    QFileDialog,
    QFrame,
    QTextEdit,
    QMessageBox,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QFont

from qgis.core import QgsRasterLayer, QgsMapLayerProxyModel
from qgis.gui import QgsMapLayerComboBox


class AISegmentationDockWidget(QDockWidget):
    """
    Main dock widget for the AI Segmentation plugin.

    Clear step-by-step interface:
    1. Check/Install Dependencies
    2. Download Models
    3. Select Layer → Start Segmentation
    4. Export Results
    """

    # Signals
    install_dependencies_requested = pyqtSignal()
    download_models_requested = pyqtSignal()
    cancel_download_requested = pyqtSignal()
    cancel_preparation_requested = pyqtSignal()
    start_segmentation_requested = pyqtSignal(object)  # QgsRasterLayer
    clear_points_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    save_mask_requested = pyqtSignal()
    export_requested = pyqtSignal(str)  # output path
    stop_segmentation_requested = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the dock widget."""
        super().__init__("AI Segmentation", parent)

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Create main widget and layout
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setSpacing(8)

        self._setup_ui()

        self.setWidget(self.main_widget)

        # State
        self._dependencies_ok = False
        self._models_ok = False
        self._segmentation_active = False

    def _setup_ui(self):
        """Set up the user interface."""
        # Title
        title_label = QLabel("AI Segmentation")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(title_label)

        self._add_separator()

        # Step 1: Dependencies
        self._setup_dependencies_section()

        # Step 2: Models
        self._setup_models_section()

        # Step 3: Segmentation
        self._setup_segmentation_section()

        # Step 4: Export
        self._setup_export_section()

        # Stretch at bottom
        self.main_layout.addStretch()

        # Status bar
        self._setup_status_bar()

    def _add_separator(self):
        """Add a horizontal separator line."""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(line)

    def _setup_dependencies_section(self):
        """Set up the dependencies section."""
        group = QGroupBox("Step 1: Dependencies")
        layout = QVBoxLayout(group)

        # Status label
        self.deps_status_label = QLabel("Checking...")
        layout.addWidget(self.deps_status_label)

        # Progress bar (hidden by default)
        self.deps_progress = QProgressBar()
        self.deps_progress.setRange(0, 100)
        self.deps_progress.setVisible(False)
        layout.addWidget(self.deps_progress)

        # Progress message
        self.deps_progress_label = QLabel("")
        self.deps_progress_label.setStyleSheet("color: gray; font-size: 10px;")
        self.deps_progress_label.setVisible(False)
        layout.addWidget(self.deps_progress_label)

        # Install button
        self.install_button = QPushButton("Install Dependencies")
        self.install_button.clicked.connect(self._on_install_clicked)
        self.install_button.setVisible(False)
        layout.addWidget(self.install_button)

        self.main_layout.addWidget(group)

    def _setup_models_section(self):
        """Set up the models section."""
        group = QGroupBox("Step 2: AI Models")
        layout = QVBoxLayout(group)

        # Status label
        self.models_status_label = QLabel("Waiting for dependencies...")
        layout.addWidget(self.models_status_label)

        # Progress bar (hidden by default)
        self.models_progress = QProgressBar()
        self.models_progress.setRange(0, 100)
        self.models_progress.setVisible(False)
        layout.addWidget(self.models_progress)

        # Progress message
        self.models_progress_label = QLabel("")
        self.models_progress_label.setStyleSheet("color: gray; font-size: 10px;")
        self.models_progress_label.setVisible(False)
        layout.addWidget(self.models_progress_label)

        # Download button
        self.download_button = QPushButton("Download Models (~109 MB)")
        self.download_button.clicked.connect(self._on_download_clicked)
        self.download_button.setVisible(False)
        self.download_button.setEnabled(False)
        layout.addWidget(self.download_button)

        # Cancel download button (hidden by default)
        self.cancel_download_button = QPushButton("Cancel Download")
        self.cancel_download_button.clicked.connect(self._on_cancel_download_clicked)
        self.cancel_download_button.setVisible(False)
        self.cancel_download_button.setStyleSheet("background-color: #d32f2f; color: white;")
        layout.addWidget(self.cancel_download_button)

        self.main_layout.addWidget(group)

    def _setup_segmentation_section(self):
        """Set up the segmentation controls section."""
        group = QGroupBox("Step 3: Segmentation")
        layout = QVBoxLayout(group)

        # Layer selection
        layer_label = QLabel("Select raster layer:")
        layout.addWidget(layer_label)

        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.layer_combo.setAllowEmptyLayer(True)
        self.layer_combo.setShowCrs(True)
        self.layer_combo.layerChanged.connect(self._on_layer_changed)
        layout.addWidget(self.layer_combo)

        # Instructions (shown when segmentation is not active)
        self.instructions_label = QLabel(
            "Select a raster layer and click 'Start Point Mode'.\n"
            "Supports GeoTIFF, XYZ tiles, and other raster formats."
        )
        self.instructions_label.setStyleSheet("color: #666; font-size: 10px;")
        self.instructions_label.setWordWrap(True)
        layout.addWidget(self.instructions_label)

        # Active mode instructions (shown when segmentation is active)
        self.active_instructions_label = QLabel(
            "🎯 POINT MODE ACTIVE\n"
            "• Left-click: Add foreground point (include)\n"
            "• Right-click: Add background point (exclude)\n"
            "• Ctrl+Z: Undo last point\n"
            "• Escape: Stop segmentation"
        )
        self.active_instructions_label.setStyleSheet(
            "color: #1976d2; font-size: 10px; font-weight: bold; "
            "background-color: #e3f2fd; padding: 8px; border-radius: 4px;"
        )
        self.active_instructions_label.setWordWrap(True)
        self.active_instructions_label.setVisible(False)
        layout.addWidget(self.active_instructions_label)

        # Preparation progress (hidden by default)
        self.prep_progress = QProgressBar()
        self.prep_progress.setRange(0, 100)
        self.prep_progress.setVisible(False)
        layout.addWidget(self.prep_progress)

        self.prep_status_label = QLabel("")
        self.prep_status_label.setStyleSheet("color: gray; font-size: 10px;")
        self.prep_status_label.setVisible(False)
        layout.addWidget(self.prep_status_label)

        # Cancel preparation button (hidden by default)
        self.cancel_prep_button = QPushButton("Cancel Preparation")
        self.cancel_prep_button.clicked.connect(self._on_cancel_prep_clicked)
        self.cancel_prep_button.setVisible(False)
        self.cancel_prep_button.setStyleSheet("background-color: #d32f2f; color: white;")
        layout.addWidget(self.cancel_prep_button)

        # Point counter
        self.point_counter_label = QLabel("Points: 0 positive, 0 negative")
        layout.addWidget(self.point_counter_label)

        # Start/Stop button - this activates the map tool for clicking
        self.start_button = QPushButton("▶ Start Point Mode")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._on_start_clicked)
        self.start_button.setToolTip(
            "Click to activate point placement mode.\n"
            "Your cursor will change to a crosshair,\n"
            "allowing you to click on the map to add points."
        )
        layout.addWidget(self.start_button)

        # Button row
        button_layout = QHBoxLayout()

        self.clear_button = QPushButton("Clear")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self._on_clear_clicked)
        button_layout.addWidget(self.clear_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._on_undo_clicked)
        button_layout.addWidget(self.undo_button)

        layout.addLayout(button_layout)

        self.main_layout.addWidget(group)

    def _setup_export_section(self):
        """Set up the export section."""
        group = QGroupBox("Step 4: Export")
        layout = QVBoxLayout(group)

        # Save current mask button
        self.save_mask_button = QPushButton("Save Current Mask")
        self.save_mask_button.setEnabled(False)
        self.save_mask_button.clicked.connect(self._on_save_mask_clicked)
        layout.addWidget(self.save_mask_button)

        # Export to file button
        self.export_button = QPushButton("Export to GeoPackage...")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export_clicked)
        layout.addWidget(self.export_button)

        self.main_layout.addWidget(group)

    def _setup_status_bar(self):
        """Set up the status bar at the bottom."""
        self._add_separator()
        self.status_label = QLabel("Open panel to start...")
        self.status_label.setStyleSheet("color: #666; font-size: 11px;")
        self.status_label.setWordWrap(True)
        self.main_layout.addWidget(self.status_label)

    # ==================== Slots ====================

    def _on_install_clicked(self):
        """Handle install button click."""
        self.install_button.setEnabled(False)
        self.install_dependencies_requested.emit()

    def _on_download_clicked(self):
        """Handle download button click."""
        self.download_button.setEnabled(False)
        self.download_models_requested.emit()

    def _on_cancel_download_clicked(self):
        """Handle cancel download button click."""
        self.cancel_download_requested.emit()

    def _on_cancel_prep_clicked(self):
        """Handle cancel preparation button click."""
        self.cancel_preparation_requested.emit()

    def _on_layer_changed(self, layer):
        """Handle layer selection change."""
        # Always update UI state when layer changes (to enable/disable Start button)
        self._update_ui_state()

        # If segmentation was active, stop it and warn user
        if self._segmentation_active:
            self.stop_segmentation_requested.emit()
            QMessageBox.warning(
                self,
                "Layer Changed",
                "Segmentation stopped because the layer was changed.\n\n"
                "Please click 'Start Segmentation' again to continue with the new layer."
            )

    def _on_start_clicked(self):
        """Handle start/stop button click."""
        if self._segmentation_active:
            self.stop_segmentation_requested.emit()
        else:
            layer = self.layer_combo.currentLayer()
            if layer:
                self.start_segmentation_requested.emit(layer)

    def _on_clear_clicked(self):
        """Handle clear button click."""
        self.clear_points_requested.emit()

    def _on_undo_clicked(self):
        """Handle undo button click."""
        self.undo_requested.emit()

    def _on_save_mask_clicked(self):
        """Handle save mask button click."""
        self.save_mask_requested.emit()

    def _on_export_clicked(self):
        """Handle export button click."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to GeoPackage",
            "",
            "GeoPackage (*.gpkg)"
        )
        if file_path:
            self.export_requested.emit(file_path)

    # ==================== Public Methods ====================

    def set_dependency_status(self, ok: bool, message: str):
        """Update dependency status display."""
        self._dependencies_ok = ok
        self.deps_status_label.setText(message)

        if ok:
            self.deps_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.install_button.setVisible(False)
            self.deps_progress.setVisible(False)
            self.deps_progress_label.setVisible(False)
            # Enable models section
            self.download_button.setEnabled(True)
        else:
            self.deps_status_label.setStyleSheet("color: orange;")
            self.install_button.setVisible(True)
            self.install_button.setEnabled(True)

        self._update_ui_state()

    def set_install_progress(self, percent: int, message: str):
        """Update installation progress."""
        self.deps_progress.setValue(percent)
        self.deps_progress_label.setText(message)

        if percent == 0:
            self.deps_progress.setVisible(True)
            self.deps_progress_label.setVisible(True)
            self.install_button.setEnabled(False)
        elif percent >= 100:
            self.deps_progress.setVisible(False)
            self.deps_progress_label.setVisible(False)

    def set_models_status(self, ok: bool, message: str):
        """Update model status display."""
        self._models_ok = ok
        self.models_status_label.setText(message)

        if ok:
            self.models_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.download_button.setVisible(False)
            self.models_progress.setVisible(False)
            self.models_progress_label.setVisible(False)
        else:
            self.models_status_label.setStyleSheet("color: orange;")
            if self._dependencies_ok:
                self.download_button.setVisible(True)
                self.download_button.setEnabled(True)

        self._update_ui_state()

    def set_download_progress(self, percent: int, message: str):
        """Update download progress."""
        self.models_progress.setValue(percent)
        self.models_progress_label.setText(message)

        if percent == 0:
            self.models_progress.setVisible(True)
            self.models_progress_label.setVisible(True)
            self.download_button.setEnabled(False)
            self.cancel_download_button.setVisible(True)
        elif percent >= 100 or "cancel" in message.lower():
            self.models_progress.setVisible(False)
            self.models_progress_label.setVisible(False)
            self.cancel_download_button.setVisible(False)

    def set_preparation_progress(self, percent: int, message: str):
        """Update layer preparation progress."""
        self.prep_progress.setValue(percent)
        self.prep_status_label.setText(message)

        if percent == 0:
            self.prep_progress.setVisible(True)
            self.prep_status_label.setVisible(True)
            self.start_button.setEnabled(False)
            self.cancel_prep_button.setVisible(True)
        elif percent >= 100 or "cancel" in message.lower():
            self.prep_progress.setVisible(False)
            self.prep_status_label.setVisible(False)
            self.cancel_prep_button.setVisible(False)
            self._update_ui_state()

    def set_segmentation_active(self, active: bool):
        """Set segmentation active state."""
        self._segmentation_active = active

        if active:
            self.start_button.setText("■ Stop Point Mode")
            self.start_button.setStyleSheet("background-color: #d32f2f; color: white;")
            self.start_button.setEnabled(True)
            # Show active instructions, hide inactive
            self.instructions_label.setVisible(False)
            self.active_instructions_label.setVisible(True)
        else:
            self.start_button.setText("▶ Start Point Mode")
            self.start_button.setStyleSheet("")
            # Show inactive instructions, hide active
            self.instructions_label.setVisible(True)
            self.active_instructions_label.setVisible(False)

        self._update_ui_state()

    def set_point_count(self, positive: int, negative: int):
        """Update the point counter display."""
        self.point_counter_label.setText(
            f"Points: {positive} positive, {negative} negative"
        )
        has_points = positive > 0 or negative > 0
        self.clear_button.setEnabled(has_points and self._segmentation_active)
        self.undo_button.setEnabled(has_points and self._segmentation_active)
        self.save_mask_button.setEnabled(has_points and self._segmentation_active)

    def set_status(self, message: str):
        """Set the status bar message."""
        self.status_label.setText(message)

    def _update_ui_state(self):
        """Update UI element enabled states."""
        has_layer = self.layer_combo.currentLayer() is not None

        # Start button enabled when:
        # - Dependencies OK
        # - Models OK
        # - Layer selected
        # OR segmentation is active (to allow stopping)
        can_start = self._dependencies_ok and self._models_ok and has_layer
        self.start_button.setEnabled(can_start or self._segmentation_active)

        # Export enabled when models are OK
        self.export_button.setEnabled(self._models_ok)
