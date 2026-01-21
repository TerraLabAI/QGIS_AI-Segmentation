"""
AI Segmentation - AI-powered segmentation plugin for QGIS

Main plugin class that coordinates all components.
"""

import os
from pathlib import Path
from typing import Optional

from qgis.PyQt.QtWidgets import QAction, QMessageBox, QProgressDialog
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal

from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsMessageLog,
    Qgis,
    QgsApplication,
)
from qgis.gui import QgisInterface

from .ai_segmentation_dockwidget import AISegmentationDockWidget
from .ai_segmentation_maptool import AISegmentationMapTool
from .core.sam_decoder import SAMDecoder, PromptManager


class EncodingWorker(QThread):
    """
    Worker thread for image encoding.

    Runs the heavy encoding process in a separate thread
    to keep the UI responsive.
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, dict)  # features, transform_info
    error = pyqtSignal(str)

    def __init__(self, encoder, layer, parent=None):
        super().__init__(parent)
        self.encoder = encoder
        self.layer = layer

    def run(self):
        """Run the encoding process."""
        try:
            features, transform_info = self.encoder.encode_raster_layer(
                self.layer,
                progress_callback=lambda p, m: self.progress.emit(p, m)
            )

            if features is not None:
                self.finished.emit(features, transform_info)
            else:
                self.error.emit("Encoding failed - check the log for details")

        except Exception as e:
            self.error.emit(str(e))


class AISegmentationPlugin:
    """
    Main plugin class for AI Segmentation.

    Handles:
    - Plugin initialization and UI setup
    - Dependency checking and installation
    - Coordination between encoder, decoder, map tool, and UI
    """

    def __init__(self, iface: QgisInterface):
        """
        Initialize the plugin.

        Args:
            iface: QGIS interface instance
        """
        self.iface = iface
        self.plugin_dir = Path(__file__).parent

        # Components (initialized in initGui)
        self.dock_widget: Optional[AISegmentationDockWidget] = None
        self.map_tool: Optional[AISegmentationMapTool] = None
        self.action: Optional[QAction] = None

        # SAM components
        self.encoder = None
        self.decoder: Optional[SAMDecoder] = None
        self.prompt_manager: Optional[PromptManager] = None

        # State
        self.current_features = None
        self.current_transform_info = None
        self.current_mask = None
        self.current_score = 0.0
        self.output_layer = None

        # Worker thread
        self.encoding_worker = None

        # Paths to ONNX models
        self.encoder_model_path = self.plugin_dir / "models" / "sam_vit_b_encoder.onnx"
        self.decoder_model_path = self.plugin_dir / "models" / "sam_vit_b_decoder.onnx"

    def initGui(self):
        """
        Initialize the plugin GUI.

        Called by QGIS when the plugin is loaded.
        """
        # Create action for the plugin
        icon_path = str(self.plugin_dir / "resources" / "icons" / "ai_segmentation_icon.png")
        if not os.path.exists(icon_path):
            icon = QIcon()
        else:
            icon = QIcon(icon_path)

        self.action = QAction(
            icon,
            "AI Segmentation",
            self.iface.mainWindow()
        )
        self.action.setCheckable(True)
        self.action.triggered.connect(self.toggle_dock_widget)

        # Add to menu and toolbar
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu("&AI Segmentation", self.action)

        # Create dock widget
        self.dock_widget = AISegmentationDockWidget(self.iface.mainWindow())
        self.dock_widget.setVisible(False)
        self.dock_widget.visibilityChanged.connect(self._on_dock_visibility_changed)

        # Connect dock widget signals
        self.dock_widget.encode_requested.connect(self._on_encode_requested)
        self.dock_widget.clear_points_requested.connect(self._on_clear_points)
        self.dock_widget.undo_requested.connect(self._on_undo)
        self.dock_widget.save_mask_requested.connect(self._on_save_mask)
        self.dock_widget.export_requested.connect(self._on_export)
        self.dock_widget.tool_activation_requested.connect(self._on_tool_activation)

        # Add dock widget to QGIS
        self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)

        # Create map tool
        self.map_tool = AISegmentationMapTool(self.iface.mapCanvas())
        self.map_tool.positive_click.connect(self._on_positive_click)
        self.map_tool.negative_click.connect(self._on_negative_click)
        self.map_tool.tool_deactivated.connect(self._on_tool_deactivated)

        # Initialize prompt manager
        self.prompt_manager = PromptManager()

        # Check dependencies on startup
        self._check_dependencies()

    def unload(self):
        """
        Unload the plugin.

        Called by QGIS when the plugin is unloaded.
        """
        # Remove menu and toolbar items
        self.iface.removePluginRasterMenu("&AI Segmentation", self.action)
        self.iface.removeToolBarIcon(self.action)

        # Remove dock widget
        if self.dock_widget:
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget.deleteLater()
            self.dock_widget = None

        # Deactivate map tool
        if self.map_tool:
            if self.iface.mapCanvas().mapTool() == self.map_tool:
                self.iface.mapCanvas().unsetMapTool(self.map_tool)
            self.map_tool = None

        # Clean up workers
        if self.encoding_worker and self.encoding_worker.isRunning():
            self.encoding_worker.terminate()
            self.encoding_worker.wait()

    def toggle_dock_widget(self, checked: bool):
        """Toggle the dock widget visibility."""
        if self.dock_widget:
            self.dock_widget.setVisible(checked)

    def _on_dock_visibility_changed(self, visible: bool):
        """Handle dock widget visibility changes."""
        if self.action:
            self.action.setChecked(visible)

    def _check_dependencies(self):
        """Check and install required dependencies."""
        try:
            from .core.dependency_manager import (
                get_missing_dependencies,
                install_all_dependencies,
                verify_installation
            )

            missing = get_missing_dependencies()

            if missing:
                reply = QMessageBox.question(
                    self.iface.mainWindow(),
                    "AI Segmentation - Missing Dependencies",
                    f"AI Segmentation requires the following packages:\n"
                    f"{', '.join([f'{name} {ver}' for name, ver in missing])}\n\n"
                    f"Would you like to install them now?",
                    QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    self._install_dependencies()
                else:
                    self.dock_widget.set_status("Dependencies missing - functionality limited")
            else:
                # Verify installation
                success, msg = verify_installation()
                if success:
                    QgsMessageLog.logMessage(msg, "AI Segmentation", level=Qgis.Info)
                    self._initialize_models()
                else:
                    QgsMessageLog.logMessage(msg, "AI Segmentation", level=Qgis.Warning)

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Dependency check failed: {str(e)}",
                "AI Segmentation",
                level=Qgis.Warning
            )

    def _install_dependencies(self):
        """Install missing dependencies with a progress dialog."""
        from .core.dependency_manager import (
            install_all_dependencies,
            verify_installation
        )

        progress = QProgressDialog(
            "Installing dependencies...",
            "Cancel",
            0, 100,
            self.iface.mainWindow()
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        def update_progress(current, total, message):
            percent = int((current / total) * 100) if total > 0 else 0
            progress.setValue(percent)
            progress.setLabelText(message)
            QgsApplication.processEvents()

        success, messages = install_all_dependencies(update_progress)
        progress.close()

        if success:
            QMessageBox.information(
                self.iface.mainWindow(),
                "Installation Complete",
                "Dependencies installed successfully.\n"
                "Please restart QGIS for changes to take effect."
            )
        else:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Installation Failed",
                "Some dependencies could not be installed:\n" +
                "\n".join(messages)
            )

    def _initialize_models(self):
        """Initialize SAM encoder and decoder models."""
        # Check if models exist
        if not self.encoder_model_path.exists():
            QgsMessageLog.logMessage(
                f"Encoder model not found at: {self.encoder_model_path}\n"
                "Please download the SAM ViT-B ONNX models.",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self.dock_widget.set_status("Models not found - see log")
            return

        if not self.decoder_model_path.exists():
            QgsMessageLog.logMessage(
                f"Decoder model not found at: {self.decoder_model_path}\n"
                "Please download the SAM ViT-B ONNX models.",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self.dock_widget.set_status("Models not found - see log")
            return

        try:
            from .core.sam_encoder import SAMEncoder

            # Initialize encoder
            self.encoder = SAMEncoder()
            if not self.encoder.load_model(str(self.encoder_model_path)):
                self.dock_widget.set_status("Failed to load encoder")
                return

            # Initialize decoder
            self.decoder = SAMDecoder()
            if not self.decoder.load_model(str(self.decoder_model_path)):
                self.dock_widget.set_status("Failed to load decoder")
                return

            self.dock_widget.set_status("Ready")
            QgsMessageLog.logMessage(
                "SAM models loaded successfully",
                "AI Segmentation",
                level=Qgis.Success
            )

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to initialize models: {str(e)}",
                "AI Segmentation",
                level=Qgis.Critical
            )
            self.dock_widget.set_status("Model initialization failed")

    def _on_encode_requested(self, layer: QgsRasterLayer):
        """Handle encoding request from UI."""
        if not self.encoder:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Encoder Not Ready",
                "The SAM encoder is not initialized.\n"
                "Please check that the ONNX models are installed."
            )
            return

        # Check for cached features
        from .core.sam_encoder import has_cached_features, load_features, get_features_path

        if has_cached_features(layer):
            reply = QMessageBox.question(
                self.iface.mainWindow(),
                "Cached Features Found",
                "This image has been encoded before.\n"
                "Use cached features?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                features_path = get_features_path(layer)
                self.current_features, self.current_transform_info = load_features(features_path)

                if self.current_features is not None:
                    self.dock_widget.set_encoding_progress(100, "Features loaded from cache")
                    self.dock_widget.set_features_loaded(True)
                    self._create_output_layer(layer)
                    return

        # Start encoding in a worker thread
        self.dock_widget.set_encoding_progress(0, "Starting encoding...")

        self.encoding_worker = EncodingWorker(self.encoder, layer)
        self.encoding_worker.progress.connect(self._on_encoding_progress)
        self.encoding_worker.finished.connect(
            lambda f, t: self._on_encoding_finished(f, t, layer)
        )
        self.encoding_worker.error.connect(self._on_encoding_error)
        self.encoding_worker.start()

    def _on_encoding_progress(self, percent: int, message: str):
        """Handle encoding progress updates."""
        self.dock_widget.set_encoding_progress(percent, message)

    def _on_encoding_finished(self, features, transform_info, layer):
        """Handle encoding completion."""
        self.current_features = features
        self.current_transform_info = transform_info

        # Save features to cache
        from .core.sam_encoder import save_features, get_features_path
        features_path = get_features_path(layer)
        save_features(features, transform_info, features_path)

        self.dock_widget.set_encoding_progress(100, "Encoding complete!")
        self.dock_widget.set_features_loaded(True)
        self._create_output_layer(layer)

    def _on_encoding_error(self, error_message: str):
        """Handle encoding error."""
        self.dock_widget.set_encoding_progress(0, "Encoding failed")
        QMessageBox.critical(
            self.iface.mainWindow(),
            "Encoding Error",
            f"Failed to encode image:\n{error_message}"
        )

    def _create_output_layer(self, source_layer: QgsRasterLayer):
        """Create or get the output vector layer."""
        from .core.polygon_exporter import create_output_layer

        if self.output_layer is None or not self.output_layer.isValid():
            self.output_layer = create_output_layer(
                source_layer.crs(),
                "AI_Segmentation_Output"
            )
            QgsProject.instance().addMapLayer(self.output_layer)

    def _on_tool_activation(self, activate: bool):
        """Handle tool activation request."""
        if activate:
            if self.current_features is None:
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "No Features",
                    "Please encode an image first."
                )
                self.dock_widget.set_tool_active(False)
                return

            self.iface.mapCanvas().setMapTool(self.map_tool)
            self.dock_widget.set_status("Click on the map to segment")
        else:
            self.iface.mapCanvas().unsetMapTool(self.map_tool)
            self.dock_widget.set_status("Ready")

    def _on_tool_deactivated(self):
        """Handle map tool deactivation."""
        self.dock_widget.set_tool_active(False)

    def _on_positive_click(self, point):
        """Handle positive (foreground) click."""
        self.prompt_manager.add_positive(point.x(), point.y())
        self._update_segmentation()

    def _on_negative_click(self, point):
        """Handle negative (background) click."""
        self.prompt_manager.add_negative(point.x(), point.y())
        self._update_segmentation()

    def _update_segmentation(self):
        """Update segmentation based on current prompts."""
        if self.current_features is None or self.decoder is None:
            return

        points, labels = self.prompt_manager.get_all_points()

        # Update UI
        pos_count, neg_count = self.prompt_manager.point_count
        self.dock_widget.set_point_count(pos_count, neg_count)

        if not points:
            return

        # Run decoder
        self.current_mask, self.current_score = self.decoder.predict_mask(
            self.current_features,
            points,
            labels,
            self.current_transform_info
        )

        if self.current_mask is not None:
            self.dock_widget.set_status(f"Mask score: {self.current_score:.3f}")
            # TODO: Display mask preview on canvas

    def _on_clear_points(self):
        """Clear all prompt points."""
        self.prompt_manager.clear()
        self.current_mask = None
        self.current_score = 0.0
        self.dock_widget.set_point_count(0, 0)
        self.dock_widget.set_status("Points cleared")
        # TODO: Clear mask preview

    def _on_undo(self):
        """Undo the last point."""
        if self.prompt_manager.undo():
            self._update_segmentation()

    def _on_save_mask(self):
        """Save the current mask to the output layer."""
        if self.current_mask is None:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "No Mask",
                "No segmentation mask to save.\n"
                "Click on the map to create a segmentation first."
            )
            return

        if self.output_layer is None:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "No Output Layer",
                "Output layer not created."
            )
            return

        from .core.polygon_exporter import add_mask_to_layer

        count = add_mask_to_layer(
            self.output_layer,
            self.current_mask,
            self.current_transform_info,
            self.current_score
        )

        if count > 0:
            self.dock_widget.set_status(f"Saved {count} polygon(s)")
            # Clear for next segmentation
            self._on_clear_points()
        else:
            self.dock_widget.set_status("Failed to save mask")

    def _on_export(self, output_path: str):
        """Export the output layer to a file."""
        if self.output_layer is None or self.output_layer.featureCount() == 0:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Nothing to Export",
                "No segmentation results to export.\n"
                "Save some masks first."
            )
            return

        from .core.polygon_exporter import export_to_geopackage

        success, message = export_to_geopackage(self.output_layer, output_path)

        if success:
            QMessageBox.information(
                self.iface.mainWindow(),
                "Export Complete",
                message
            )
        else:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Export Failed",
                message
            )
