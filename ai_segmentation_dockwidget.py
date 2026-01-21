"""
Dock Widget for AI Segmentation

Main user interface panel for the segmentation plugin.
"""

import os
from pathlib import Path

from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QFrame,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QFont

from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsMapLayerProxyModel,
)
from qgis.gui import QgsMapLayerComboBox


class AISegmentationDockWidget(QDockWidget):
    """
    Main dock widget for the AI Segmentation plugin.

    Provides controls for:
    - Selecting source raster layer
    - Encoding images
    - Managing segmentation prompts
    - Exporting results
    """

    # Signals
    encode_requested = pyqtSignal(object)  # QgsRasterLayer
    clear_points_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    save_mask_requested = pyqtSignal()
    export_requested = pyqtSignal(str)  # output path
    tool_activation_requested = pyqtSignal(bool)  # activate/deactivate

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
        self._encoding_in_progress = False
        self._features_loaded = False

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

        # Source Layer Section
        self._setup_source_section()

        # Encoding Section
        self._setup_encoding_section()

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

    def _setup_encoding_section(self):
        """Set up the image encoding section."""
        group = QGroupBox("Image Encoding")
        layout = QVBoxLayout(group)

        # Status label
        self.encoding_status_label = QLabel("Status: No layer selected")
        layout.addWidget(self.encoding_status_label)

        # Progress bar
        self.encoding_progress = QProgressBar()
        self.encoding_progress.setRange(0, 100)
        self.encoding_progress.setValue(0)
        self.encoding_progress.setVisible(False)
        layout.addWidget(self.encoding_progress)

        # Encode button
        self.encode_button = QPushButton("Encode Image")
        self.encode_button.setEnabled(False)
        self.encode_button.clicked.connect(self._on_encode_clicked)
        layout.addWidget(self.encode_button)

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

        # Point counter
        self.point_counter_label = QLabel("Points: 0 positive, 0 negative")
        layout.addWidget(self.point_counter_label)

        # Tool activation button
        self.activate_tool_button = QPushButton("Start Segmentation")
        self.activate_tool_button.setCheckable(True)
        self.activate_tool_button.setEnabled(False)
        self.activate_tool_button.clicked.connect(self._on_tool_activation_clicked)
        layout.addWidget(self.activate_tool_button)

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

    def _on_layer_changed(self, layer):
        """Handle layer selection change."""
        if layer and isinstance(layer, QgsRasterLayer):
            self.encoding_status_label.setText("Status: Ready to encode")
            self.encode_button.setEnabled(True)
            self._features_loaded = False
            self._update_ui_state()
        else:
            self.encoding_status_label.setText("Status: No layer selected")
            self.encode_button.setEnabled(False)
            self._features_loaded = False
            self._update_ui_state()

    def _on_encode_clicked(self):
        """Handle encode button click."""
        layer = self.layer_combo.currentLayer()
        if layer:
            self.encode_requested.emit(layer)

    def _on_tool_activation_clicked(self, checked):
        """Handle tool activation toggle."""
        self.tool_activation_requested.emit(checked)
        self._update_tool_button_state(checked)

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

    def set_encoding_progress(self, value: int, message: str = ""):
        """Update encoding progress."""
        self.encoding_progress.setValue(value)
        if message:
            self.encoding_status_label.setText(f"Status: {message}")

        if value == 0:
            self.encoding_progress.setVisible(True)
            self._encoding_in_progress = True
        elif value >= 100:
            self.encoding_progress.setVisible(False)
            self._encoding_in_progress = False
            self._features_loaded = True

        self._update_ui_state()

    def set_point_count(self, positive: int, negative: int):
        """Update the point counter display."""
        self.point_counter_label.setText(
            f"Points: {positive} positive, {negative} negative"
        )
        has_points = positive > 0 or negative > 0
        self.clear_button.setEnabled(has_points and self._features_loaded)
        self.undo_button.setEnabled(has_points and self._features_loaded)
        self.save_mask_button.setEnabled(has_points and self._features_loaded)

    def set_status(self, message: str):
        """Set the status bar message."""
        self.status_label.setText(message)

    def set_features_loaded(self, loaded: bool):
        """Set whether features are loaded."""
        self._features_loaded = loaded
        self._update_ui_state()

    def _update_ui_state(self):
        """Update UI element enabled states."""
        has_layer = self.layer_combo.currentLayer() is not None
        can_segment = has_layer and self._features_loaded and not self._encoding_in_progress

        self.encode_button.setEnabled(has_layer and not self._encoding_in_progress)
        self.activate_tool_button.setEnabled(can_segment)
        self.export_button.setEnabled(can_segment)

        if self._features_loaded:
            self.encoding_status_label.setText("Status: Ready for segmentation")

    def _update_tool_button_state(self, active: bool):
        """Update the tool activation button state."""
        if active:
            self.activate_tool_button.setText("Stop Segmentation")
            self.activate_tool_button.setStyleSheet("background-color: #4CAF50; color: white;")
        else:
            self.activate_tool_button.setText("Start Segmentation")
            self.activate_tool_button.setStyleSheet("")

    def set_tool_active(self, active: bool):
        """Set the tool active state from external source."""
        self.activate_tool_button.setChecked(active)
        self._update_tool_button_state(active)
