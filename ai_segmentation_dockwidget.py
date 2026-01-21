"""
Dock Widget for AI Segmentation

Main user interface panel for the segmentation plugin.
Clean, minimal interface with session-based workflow.
Supports multiple AI models with easy switching and installation.
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
    QFrame,
    QMessageBox,
    QComboBox,
    QToolButton,
    QSizePolicy,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QStandardItem, QStandardItemModel, QFont

from qgis.core import QgsMapLayerProxyModel
from qgis.gui import QgsMapLayerComboBox

from typing import List


class AISegmentationDockWidget(QDockWidget):
    """
    Main dock widget for the AI Segmentation plugin.

    Clean session-based interface:
    1. Select Model -> Select Layer -> Start AI Segmentation
    2. Click to segment (left=include, right=exclude)
    3. Finish Segmentation -> Creates layer automatically

    Supports multiple AI models with dropdown selector and install section.
    """

    # Signals
    install_dependencies_requested = pyqtSignal()
    download_models_requested = pyqtSignal()
    install_model_requested = pyqtSignal(str)  # model_id
    cancel_download_requested = pyqtSignal()
    cancel_preparation_requested = pyqtSignal()
    start_segmentation_requested = pyqtSignal(object)  # QgsRasterLayer
    clear_points_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    save_mask_requested = pyqtSignal()
    finish_segmentation_requested = pyqtSignal()  # Finish and auto-export to layer
    model_changed = pyqtSignal(str)  # model_id

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
        self._installed_models: List[str] = []
        self._current_model_id: str = None
        self._downloading_model_id: str = None  # Track which model is being downloaded

    def _setup_ui(self):
        """Set up the user interface."""
        # Setup section (Dependencies) - hidden when ready
        self._setup_dependencies_section()

        # Model install section (collapsible)
        self._setup_model_install_section()

        # Segmentation section
        self._setup_segmentation_section()

        # Stretch at bottom
        self.main_layout.addStretch()

        # Status bar
        self._setup_status_bar()

    def _setup_dependencies_section(self):
        """Set up the dependencies section."""
        self.deps_group = QGroupBox("Dependencies")
        layout = QVBoxLayout(self.deps_group)

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

        self.main_layout.addWidget(self.deps_group)

    def _setup_model_install_section(self):
        """Set up the collapsible model installation section."""
        # Container for the collapsible section
        self.model_install_container = QWidget()
        container_layout = QVBoxLayout(self.model_install_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(4)

        # Header with toggle button
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.model_install_toggle = QToolButton()
        self.model_install_toggle.setArrowType(Qt.RightArrow)
        self.model_install_toggle.setCheckable(True)
        self.model_install_toggle.setChecked(False)
        self.model_install_toggle.clicked.connect(self._on_model_install_toggle)
        self.model_install_toggle.setStyleSheet("QToolButton { border: none; }")
        header_layout.addWidget(self.model_install_toggle)

        header_label = QLabel("Install AI Models")
        header_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        container_layout.addWidget(header_widget)

        # Collapsible content
        self.model_install_content = QWidget()
        self.model_install_content.setVisible(False)
        content_layout = QVBoxLayout(self.model_install_content)
        content_layout.setContentsMargins(20, 4, 0, 4)
        content_layout.setSpacing(6)

        # Model list will be populated dynamically
        self.model_rows = {}  # model_id -> dict of widgets

        # Create rows for each model (will be populated in update_model_list)
        self._create_model_rows(content_layout)

        container_layout.addWidget(self.model_install_content)

        # Download progress (shown during download)
        self.models_progress = QProgressBar()
        self.models_progress.setRange(0, 100)
        self.models_progress.setVisible(False)
        container_layout.addWidget(self.models_progress)

        self.models_progress_label = QLabel("")
        self.models_progress_label.setStyleSheet("color: #666; font-size: 10px;")
        self.models_progress_label.setVisible(False)
        container_layout.addWidget(self.models_progress_label)

        # Cancel download button
        self.cancel_download_button = QPushButton("Cancel")
        self.cancel_download_button.clicked.connect(self._on_cancel_download_clicked)
        self.cancel_download_button.setVisible(False)
        self.cancel_download_button.setStyleSheet("background-color: #d32f2f; color: white;")
        container_layout.addWidget(self.cancel_download_button)

        self.main_layout.addWidget(self.model_install_container)

    def _create_model_rows(self, layout):
        """Create the model installation rows."""
        # Import here to avoid circular dependency
        from .core.model_registry import get_all_models

        for config in get_all_models():
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 2, 0, 2)

            # Model name and size
            info_label = QLabel(f"{config.display_name} - {config.total_size_mb}MB")
            info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            row_layout.addWidget(info_label)

            # Status/Install button
            status_label = QLabel("")
            status_label.setVisible(False)
            row_layout.addWidget(status_label)

            install_btn = QPushButton("Install")
            install_btn.setFixedWidth(70)
            install_btn.clicked.connect(lambda checked, mid=config.model_id: self._on_model_install_clicked(mid))
            row_layout.addWidget(install_btn)

            self.model_rows[config.model_id] = {
                "row": row,
                "info_label": info_label,
                "status_label": status_label,
                "install_btn": install_btn,
            }

            layout.addWidget(row)

    def _setup_segmentation_section(self):
        """Set up the segmentation controls section."""
        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(sep)

        # Container widget for segmentation controls
        seg_widget = QWidget()
        layout = QVBoxLayout(seg_widget)
        layout.setContentsMargins(0, 8, 0, 0)

        # Model selector dropdown
        model_label = QLabel("AI Model:")
        layout.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.setToolTip(
            "Select the AI model for segmentation.\n"
            "Grayed out models are not installed yet."
        )
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        layout.addWidget(self.model_combo)

        # Populate model combo (will be updated when models are checked)
        self._populate_model_combo()

        # Small spacing
        layout.addSpacing(8)

        # Layer selection
        layer_label = QLabel("Raster layer to segment:")
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

    def _populate_model_combo(self):
        """Populate the model selector combo box."""
        from .core.model_registry import get_all_models

        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        # Use a custom model to control item enabled state
        model = QStandardItemModel()
        for config in get_all_models():
            item = QStandardItem(config.display_name)
            item.setData(config.model_id, Qt.UserRole)
            # Initially all disabled until we check what's installed
            item.setEnabled(False)
            model.appendRow(item)

        self.model_combo.setModel(model)
        self.model_combo.blockSignals(False)

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

    def _on_model_install_toggle(self, checked):
        """Handle model install section toggle."""
        self.model_install_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.model_install_content.setVisible(checked)

    def _on_model_install_clicked(self, model_id: str):
        """Handle install button click for a specific model."""
        self._downloading_model_id = model_id
        # Disable all install buttons during download
        for mid, widgets in self.model_rows.items():
            widgets["install_btn"].setEnabled(False)
        self.install_model_requested.emit(model_id)

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

    def _on_model_combo_changed(self, index):
        """Handle model selection change."""
        if index < 0:
            return

        model = self.model_combo.model()
        item = model.item(index)
        model_id = item.data(Qt.UserRole)

        # Check if the item is enabled (model is installed)
        if not item.isEnabled():
            # Revert to current model
            self._select_model_in_combo(self._current_model_id)
            QMessageBox.information(
                self,
                "Model Not Installed",
                f"Please install this model first using the 'Install AI Models' section above."
            )
            return

        # Check if segmentation is active
        if self._segmentation_active:
            # Revert and warn
            self._select_model_in_combo(self._current_model_id)
            QMessageBox.warning(
                self,
                "Segmentation Active",
                "Please finish or cancel the current segmentation before switching models."
            )
            return

        # Emit model change signal
        if model_id != self._current_model_id:
            self._current_model_id = model_id
            self.model_changed.emit(model_id)

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
            # Hide entire deps section when OK
            self.deps_group.setVisible(False)
        else:
            self.deps_status_label.setStyleSheet("color: #f57c00;")
            self.install_button.setVisible(True)
            self.install_button.setEnabled(True)
            self.deps_group.setVisible(True)

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

    def set_models_status(self, installed_models: List[str], current_model_id: str = None):
        """
        Update model status display.

        Args:
            installed_models: List of installed model IDs
            current_model_id: Currently active model ID
        """
        self._installed_models = installed_models
        self._models_ok = len(installed_models) > 0

        if current_model_id:
            self._current_model_id = current_model_id

        # Update model rows
        for model_id, widgets in self.model_rows.items():
            is_installed = model_id in installed_models
            if is_installed:
                widgets["status_label"].setText("Installed")
                widgets["status_label"].setStyleSheet("color: #388e3c; font-weight: bold;")
                widgets["status_label"].setVisible(True)
                widgets["install_btn"].setVisible(False)
            else:
                widgets["status_label"].setVisible(False)
                widgets["install_btn"].setVisible(True)
                widgets["install_btn"].setEnabled(True)

        # Update model combo
        self._update_model_combo_state()

        # Select current model in combo
        if current_model_id:
            self._select_model_in_combo(current_model_id)

        # Collapse install section if at least one model is installed
        if self._models_ok:
            self.model_install_toggle.setChecked(False)
            self.model_install_content.setVisible(False)
            self.model_install_toggle.setArrowType(Qt.RightArrow)
        else:
            # Expand if no models installed
            self.model_install_toggle.setChecked(True)
            self.model_install_content.setVisible(True)
            self.model_install_toggle.setArrowType(Qt.DownArrow)

        self._update_ui_state()

    def _update_model_combo_state(self):
        """Update which items in the model combo are enabled/disabled."""
        model = self.model_combo.model()
        for i in range(model.rowCount()):
            item = model.item(i)
            model_id = item.data(Qt.UserRole)
            is_installed = model_id in self._installed_models
            item.setEnabled(is_installed)

            # Update display to show installed/not installed
            from .core.model_registry import get_model_config
            config = get_model_config(model_id)
            if is_installed:
                item.setText(config.display_name)
            else:
                item.setText(f"{config.display_name} (not installed)")

    def _select_model_in_combo(self, model_id: str):
        """Select a model in the combo box by ID."""
        if not model_id:
            return

        model = self.model_combo.model()
        self.model_combo.blockSignals(True)
        for i in range(model.rowCount()):
            item = model.item(i)
            if item.data(Qt.UserRole) == model_id:
                self.model_combo.setCurrentIndex(i)
                break
        self.model_combo.blockSignals(False)

    def set_download_progress(self, percent: int, message: str):
        """Update download progress."""
        self.models_progress.setValue(percent)
        self.models_progress_label.setText(message)

        if percent == 0:
            self.models_progress.setVisible(True)
            self.models_progress_label.setVisible(True)
            self.cancel_download_button.setVisible(True)
            # Expand install section to show progress
            self.model_install_toggle.setChecked(True)
            self.model_install_content.setVisible(True)
            self.model_install_toggle.setArrowType(Qt.DownArrow)
        elif percent >= 100 or "cancel" in message.lower():
            self.models_progress.setVisible(False)
            self.models_progress_label.setVisible(False)
            self.cancel_download_button.setVisible(False)
            self._downloading_model_id = None
            # Re-enable install buttons
            for mid, widgets in self.model_rows.items():
                if mid not in self._installed_models:
                    widgets["install_btn"].setEnabled(True)

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
            # Disable model switching during segmentation
            self.model_combo.setEnabled(False)
        else:
            # Inactive: show start; hide undo, finish, instructions
            self.start_button.setVisible(True)
            self.active_instructions_label.setVisible(False)
            self.undo_button.setVisible(False)
            self.finish_button.setVisible(False)
            # Re-enable model switching
            self.model_combo.setEnabled(True)

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

    def get_selected_model_id(self) -> str:
        """Get the currently selected model ID."""
        return self._current_model_id

    def _update_ui_state(self):
        """Update UI element enabled states."""
        has_layer = self.layer_combo.currentLayer() is not None

        # Start button enabled when ready and layer selected and model selected
        can_start = (
            self._dependencies_ok and
            self._models_ok and
            has_layer and
            self._current_model_id is not None
        )
        self.start_button.setEnabled(can_start or self._segmentation_active)
