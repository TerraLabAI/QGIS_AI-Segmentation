"""
Dock Widget for AI Segmentation

Main user interface panel for the segmentation plugin.
Clean, minimal interface with session-based workflow.
"""

from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QProgressBar,
    QFrame,
    QMessageBox,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal

from qgis.core import QgsMapLayerProxyModel
from qgis.gui import QgsMapLayerComboBox


class AISegmentationDockWidget(QDockWidget):
    """
    Main dock widget for the AI Segmentation plugin.

    Clean session-based interface:
    1. Select Layer -> Start AI Segmentation
    2. Click to segment (left=include, right=exclude)
    3. Finish Segmentation -> Creates layer automatically
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
    finish_segmentation_requested = pyqtSignal()  # Finish and auto-export to layer

    def __init__(self, parent=None):
        """Initialize the dock widget."""
        super().__init__("AI Segmentation by Terralab", parent)

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Create main widget and layout
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setSpacing(8)
        self.main_layout.setContentsMargins(8, 8, 8, 8)

        self._setup_ui()

        self.setWidget(self.main_widget)

        # State
        self._dependencies_ok = False
        self._models_ok = False
        self._segmentation_active = False
        self._has_mask = False  # Track if there's a current mask (points placed)

    def _setup_ui(self):
        """Set up the user interface."""
        # Setup section (Dependencies + Models) - hidden when ready
        self._setup_setup_section()

        # Segmentation section
        self._setup_segmentation_section()

        # Stretch at bottom
        self.main_layout.addStretch()

        # Status bar
        self._setup_status_bar()

    def _setup_setup_section(self):
        """Set up the combined setup section (dependencies + models)."""
        self.setup_group = QGroupBox("Setup")
        layout = QVBoxLayout(self.setup_group)

        # Dependencies status
        self.deps_status_label = QLabel("Checking dependencies...")
        layout.addWidget(self.deps_status_label)

        # Dependencies progress (hidden by default)
        self.deps_progress = QProgressBar()
        self.deps_progress.setRange(0, 100)
        self.deps_progress.setVisible(False)
        layout.addWidget(self.deps_progress)

        self.deps_progress_label = QLabel("")
        self.deps_progress_label.setStyleSheet("color: #666; font-size: 10px;")
        self.deps_progress_label.setVisible(False)
        self.deps_progress_label.setWordWrap(True)
        layout.addWidget(self.deps_progress_label)

        # Install button
        self.install_button = QPushButton("Install Dependencies")
        self.install_button.clicked.connect(self._on_install_clicked)
        self.install_button.setVisible(False)
        self.install_button.setToolTip("Install required Python packages (onnxruntime, numpy)")
        layout.addWidget(self.install_button)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # Models status
        self.models_status_label = QLabel("Waiting for dependencies...")
        layout.addWidget(self.models_status_label)

        # Models progress (hidden by default)
        self.models_progress = QProgressBar()
        self.models_progress.setRange(0, 100)
        self.models_progress.setVisible(False)
        layout.addWidget(self.models_progress)

        self.models_progress_label = QLabel("")
        self.models_progress_label.setStyleSheet("color: #666; font-size: 10px;")
        self.models_progress_label.setVisible(False)
        layout.addWidget(self.models_progress_label)

        # Download button
        self.download_button = QPushButton("Download Models (~109 MB)")
        self.download_button.clicked.connect(self._on_download_clicked)
        self.download_button.setVisible(False)
        self.download_button.setEnabled(False)
        self.download_button.setToolTip("Download AI models from HuggingFace (one-time)")
        layout.addWidget(self.download_button)

        # Cancel download button
        self.cancel_download_button = QPushButton("Cancel")
        self.cancel_download_button.clicked.connect(self._on_cancel_download_clicked)
        self.cancel_download_button.setVisible(False)
        self.cancel_download_button.setStyleSheet("background-color: #d32f2f; color: white;")
        layout.addWidget(self.cancel_download_button)

        self.main_layout.addWidget(self.setup_group)

    def _setup_segmentation_section(self):
        """Set up the segmentation controls section."""
        # Separator line instead of group box title
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(sep)

        # Container widget for segmentation controls
        seg_widget = QWidget()
        layout = QVBoxLayout(seg_widget)
        layout.setContentsMargins(0, 8, 0, 0)

        # Layer selection
        layer_label = QLabel("Raster layer:")
        layout.addWidget(layer_label)

        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.layer_combo.setAllowEmptyLayer(True)
        self.layer_combo.setShowCrs(True)
        self.layer_combo.layerChanged.connect(self._on_layer_changed)
        self.layer_combo.setToolTip("Select a raster layer (GeoTIFF, XYZ tiles, WMS, etc.)")
        layout.addWidget(self.layer_combo)

        # Active mode instructions (shown when segmentation is active)
        self.active_instructions_label = QLabel(
            "CLICK ON THE MAP\n\n"
            "Left-click = INCLUDE\n"
            "Right-click = EXCLUDE"
        )
        self.active_instructions_label.setStyleSheet(
            "color: #1976d2; font-size: 11px; font-weight: bold; "
            "background-color: #e3f2fd; padding: 10px; border-radius: 4px;"
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
        self.prep_status_label.setStyleSheet("color: #666; font-size: 10px;")
        self.prep_status_label.setVisible(False)
        layout.addWidget(self.prep_status_label)

        # Cancel preparation button
        self.cancel_prep_button = QPushButton("Cancel")
        self.cancel_prep_button.clicked.connect(self._on_cancel_prep_clicked)
        self.cancel_prep_button.setVisible(False)
        self.cancel_prep_button.setStyleSheet("background-color: #d32f2f; color: white;")
        layout.addWidget(self.cancel_prep_button)

        # Start button (shown when NOT active)
        self.start_button = QPushButton("Start AI Segmentation")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._on_start_clicked)
        self.start_button.setToolTip(
            "Activate point mode to click on the map.\n"
            "Left-click to include, right-click to exclude."
        )
        layout.addWidget(self.start_button)

        # Undo button (shown when active, above Finish)
        self.undo_button = QPushButton("Undo Last Point")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._on_undo_clicked)
        self.undo_button.setVisible(False)
        self.undo_button.setToolTip("Remove the last point (Ctrl+Z)")
        layout.addWidget(self.undo_button)

        # Finish button (shown when active, replaces Start)
        self.finish_button = QPushButton("Finish Segmentation")
        self.finish_button.clicked.connect(self._on_finish_clicked)
        self.finish_button.setVisible(False)
        self.finish_button.setStyleSheet("background-color: #388e3c; color: white;")
        self.finish_button.setToolTip("Save the segmentation as a new layer and start fresh")
        layout.addWidget(self.finish_button)

        self.main_layout.addWidget(seg_widget)

    def _setup_status_bar(self):
        """Set up the status bar at the bottom."""
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(sep)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; font-size: 10px;")
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
        self._update_ui_state()

        # If segmentation was active, cancel it (layer changed)
        if self._segmentation_active:
            self._segmentation_active = False
            self._update_button_visibility()
            QMessageBox.warning(
                self,
                "Layer Changed",
                "Segmentation cancelled because the layer was changed."
            )

    def _on_start_clicked(self):
        """Handle start button click."""
        layer = self.layer_combo.currentLayer()
        if layer:
            self.start_segmentation_requested.emit(layer)

    def _on_undo_clicked(self):
        """Handle undo button click."""
        self.undo_requested.emit()

    def _on_finish_clicked(self):
        """Handle finish button click - export to layer and reset."""
        self.finish_segmentation_requested.emit()

    # ==================== Public Methods ====================

    def set_dependency_status(self, ok: bool, message: str):
        """Update dependency status display."""
        self._dependencies_ok = ok
        self.deps_status_label.setText(message)

        if ok:
            self.deps_status_label.setStyleSheet("color: #388e3c; font-weight: bold;")
            self.install_button.setVisible(False)
            self.deps_progress.setVisible(False)
            self.deps_progress_label.setVisible(False)
            self.download_button.setEnabled(True)
        else:
            self.deps_status_label.setStyleSheet("color: #f57c00;")
            self.install_button.setVisible(True)
            self.install_button.setEnabled(True)

        self._update_setup_visibility()
        self._update_ui_state()

    def set_install_progress(self, percent: int, message: str):
        """Update installation progress."""
        self.deps_progress.setValue(percent)
        self.deps_progress_label.setText(message)

        if percent == 0:
            self.deps_progress.setVisible(True)
            self.deps_progress_label.setVisible(True)
            self.install_button.setEnabled(False)
            self.install_button.setText("Installing...")
        elif percent >= 100:
            self.deps_progress.setVisible(False)
            if "failed" in message.lower():
                self.install_button.setEnabled(True)
                self.install_button.setText("Retry")

    def set_models_status(self, ok: bool, message: str):
        """Update model status display."""
        self._models_ok = ok
        self.models_status_label.setText(message)

        if ok:
            self.models_status_label.setStyleSheet("color: #388e3c; font-weight: bold;")
            self.download_button.setVisible(False)
            self.models_progress.setVisible(False)
            self.models_progress_label.setVisible(False)
        else:
            self.models_status_label.setStyleSheet("color: #f57c00;")
            if self._dependencies_ok:
                self.download_button.setVisible(True)
                self.download_button.setEnabled(True)

        self._update_setup_visibility()
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
        self._update_button_visibility()
        self._update_ui_state()

    def _update_button_visibility(self):
        """Update button visibility based on segmentation state."""
        if self._segmentation_active:
            # Active: show instructions, undo, finish; hide start
            self.start_button.setVisible(False)
            self.active_instructions_label.setVisible(True)
            self.undo_button.setVisible(True)
            self.finish_button.setVisible(True)
            # Enable finish only if we have points
            self.finish_button.setEnabled(self._has_mask)
        else:
            # Inactive: show start; hide undo, finish, instructions
            self.start_button.setVisible(True)
            self.active_instructions_label.setVisible(False)
            self.undo_button.setVisible(False)
            self.finish_button.setVisible(False)

    def set_point_count(self, positive: int, negative: int):
        """Update the point counter display in the status bar."""
        total = positive + negative
        has_points = total > 0
        self._has_mask = has_points

        # Update undo button state
        self.undo_button.setEnabled(has_points and self._segmentation_active)
        # Update finish button state (enabled only if we have points)
        self.finish_button.setEnabled(has_points)

        # Show point count in status bar
        if has_points:
            self.status_label.setText(f"Points: {positive} + / {negative} -")

    def set_status(self, message: str):
        """Set the status bar message."""
        self.status_label.setText(message)

    def reset_session(self):
        """Reset the session state (called after finish/export)."""
        self._has_mask = False
        self._segmentation_active = False
        self._update_button_visibility()
        self._update_ui_state()

    def _update_setup_visibility(self):
        """Hide setup section when both dependencies and models are ready."""
        if self._dependencies_ok and self._models_ok:
            self.setup_group.setVisible(False)
        else:
            self.setup_group.setVisible(True)

    def _update_ui_state(self):
        """Update UI element enabled states."""
        has_layer = self.layer_combo.currentLayer() is not None

        # Start button enabled when ready and layer selected
        can_start = self._dependencies_ok and self._models_ok and has_layer
        self.start_button.setEnabled(can_start or self._segmentation_active)
