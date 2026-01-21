"""
Dock Widget for AI Segmentation

Main user interface panel for the segmentation plugin.
Simplified interface - no manual encoding required.
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
)
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QFont

from qgis.core import QgsRasterLayer, QgsMapLayerProxyModel
from qgis.gui import QgsMapLayerComboBox


class AISegmentationDockWidget(QDockWidget):
    """
    Main dock widget for the AI Segmentation plugin.

    Simplified interface:
    - Select layer
    - Start Segmentation (auto-downloads models, auto-encodes)
    - Click to segment
    - Export results
    """

    # Signals
    download_models_requested = pyqtSignal()
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
        self.main_layout.setSpacing(10)

        self._setup_ui()

        self.setWidget(self.main_widget)

        # State
        self._models_ready = False
        self._segmentation_active = False
        self._layer_ready = False

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

        # Separator
        self._add_separator()

        # Model Status Section
        self._setup_model_section()

        # Source Layer Section
        self._setup_source_section()

        # Segmentation Section
        self._setup_segmentation_section()

        # Export Section
        self._setup_export_section()

        # Add stretch at the bottom
        self.main_layout.addStretch()

        # Status bar
        self._setup_status_bar()

    def _add_separator(self):
        """Add a horizontal separator line."""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(line)

    def _setup_model_section(self):
        """Set up the model status section."""
        group = QGroupBox("Model")
        layout = QVBoxLayout(group)

        # Model status label
        self.model_status_label = QLabel("Checking models...")
        layout.addWidget(self.model_status_label)

        # Progress bar (hidden by default)
        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 100)
        self.model_progress.setValue(0)
        self.model_progress.setVisible(False)
        layout.addWidget(self.model_progress)

        # Download button (hidden when models are ready)
        self.download_button = QPushButton("Download Models (~109 MB)")
        self.download_button.clicked.connect(self._on_download_clicked)
        self.download_button.setVisible(False)
        layout.addWidget(self.download_button)

        self.main_layout.addWidget(group)

    def _setup_source_section(self):
        """Set up the source layer selection section."""
        group = QGroupBox("Source Layer")
        layout = QVBoxLayout(group)

        # Layer combo box
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.layer_combo.setAllowEmptyLayer(True)
        self.layer_combo.setShowCrs(True)
        self.layer_combo.layerChanged.connect(self._on_layer_changed)
        layout.addWidget(self.layer_combo)

        self.main_layout.addWidget(group)

    def _setup_segmentation_section(self):
        """Set up the segmentation controls section."""
        group = QGroupBox("Segmentation")
        layout = QVBoxLayout(group)

        # Instructions
        instructions = QLabel(
            "Left-click: Add point (include)\n"
            "Right-click: Add point (exclude)"
        )
        instructions.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(instructions)

        # Preparation progress (hidden by default)
        self.prep_progress = QProgressBar()
        self.prep_progress.setRange(0, 100)
        self.prep_progress.setValue(0)
        self.prep_progress.setVisible(False)
        layout.addWidget(self.prep_progress)

        # Preparation status label
        self.prep_status_label = QLabel("")
        self.prep_status_label.setStyleSheet("color: gray;")
        self.prep_status_label.setVisible(False)
        layout.addWidget(self.prep_status_label)

        # Point counter
        self.point_counter_label = QLabel("Points: 0 positive, 0 negative")
        layout.addWidget(self.point_counter_label)

        # Start/Stop button
        self.start_button = QPushButton("Start Segmentation")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._on_start_clicked)
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
        group = QGroupBox("Export")
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
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        self.main_layout.addWidget(self.status_label)

    # Slots

    def _on_download_clicked(self):
        """Handle download button click."""
        self.download_models_requested.emit()

    def _on_layer_changed(self, layer):
        """Handle layer selection change."""
        self._layer_ready = False
        self._update_ui_state()

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

    # Public methods

    def get_selected_layer(self):
        """Get the currently selected raster layer."""
        return self.layer_combo.currentLayer()

    def set_models_status(self, ready: bool, message: str):
        """Update model status display."""
        self._models_ready = ready
        self.model_status_label.setText(message)

        if ready:
            self.model_status_label.setStyleSheet("color: green;")
            self.download_button.setVisible(False)
        else:
            self.model_status_label.setStyleSheet("color: orange;")
            self.download_button.setVisible(True)

        self._update_ui_state()

    def set_download_progress(self, percent: int, message: str):
        """Update download progress."""
        self.model_progress.setValue(percent)
        self.model_status_label.setText(message)

        if percent == 0:
            self.model_progress.setVisible(True)
            self.download_button.setEnabled(False)
        elif percent >= 100:
            self.model_progress.setVisible(False)
            self.download_button.setVisible(False)

    def set_preparation_progress(self, percent: int, message: str):
        """Update layer preparation (encoding) progress."""
        self.prep_progress.setValue(percent)
        self.prep_status_label.setText(message)

        if percent == 0:
            self.prep_progress.setVisible(True)
            self.prep_status_label.setVisible(True)
            self.start_button.setEnabled(False)
        elif percent >= 100:
            self.prep_progress.setVisible(False)
            self.prep_status_label.setVisible(False)
            self._layer_ready = True
            self._update_ui_state()

    def set_segmentation_active(self, active: bool):
        """Set segmentation active state."""
        self._segmentation_active = active

        if active:
            self.start_button.setText("Stop Segmentation")
            self.start_button.setStyleSheet("background-color: #f44336; color: white;")
        else:
            self.start_button.setText("Start Segmentation")
            self.start_button.setStyleSheet("")

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
        # - Models are ready
        # - Layer is selected
        # OR segmentation is active (to allow stopping)
        can_start = self._models_ready and has_layer
        self.start_button.setEnabled(can_start or self._segmentation_active)

        # Export enabled when layer is ready
        self.export_button.setEnabled(self._layer_ready)
