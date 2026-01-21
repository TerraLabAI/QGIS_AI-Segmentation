"""
AI Segmentation - AI-powered segmentation plugin for QGIS

Main plugin class that coordinates all components.
Designed for stability - silent startup, checks only when panel opens.
Supports multiple AI models (SAM, SAM2).
"""

import os
from pathlib import Path
from typing import Optional

from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal

from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsMessageLog,
    Qgis,
    QgsWkbTypes,
    QgsGeometry,
    QgsFeature,
    QgsFillSymbol,
)
from qgis.gui import QgisInterface, QgsRubberBand
from qgis.PyQt.QtGui import QColor

from .ai_segmentation_dockwidget import AISegmentationDockWidget
from .ai_segmentation_maptool import AISegmentationMapTool


class InstallWorker(QThread):
    """Worker thread for dependency installation.

    Runs pip installation in background thread to avoid blocking QGIS UI.
    Emits progress updates for real-time feedback in the dock widget.
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, list)  # success, messages

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_message = ""

    def run(self):
        """Run the installation process."""
        try:
            from .core.dependency_manager import install_all_dependencies

            def callback(current, total, msg):
                # Calculate percentage (handle edge cases)
                if total > 0:
                    # Use floating point for smoother progress
                    percent = int((current / total) * 100)
                else:
                    percent = 0

                # Only emit if message changed (avoid UI flicker)
                if msg != self._last_message:
                    self._last_message = msg
                    self.progress.emit(percent, msg)

            # Emit initial progress
            self.progress.emit(0, "Starting installation...")

            success, messages = install_all_dependencies(callback)

            # Emit completion
            if success:
                self.progress.emit(100, "Installation complete!")
            else:
                self.progress.emit(100, "Installation failed")

            self.finished.emit(success, messages)

        except Exception as e:
            self.progress.emit(100, f"Error: {str(e)[:50]}")
            self.finished.emit(False, [str(e)])


class DownloadWorker(QThread):
    """Worker thread for model downloading."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, str)  # success, message, model_id

    def __init__(self, model_id: str, parent=None):
        super().__init__(parent)
        self.model_id = model_id

    def run(self):
        """Run the download process."""
        try:
            from .core.model_manager import download_model

            success, message = download_model(
                self.model_id,
                progress_callback=lambda p, m: self.progress.emit(p, m)
            )
            self.finished.emit(success, message, self.model_id)

        except Exception as e:
            self.finished.emit(False, str(e), self.model_id)


class PreparationWorker(QThread):
    """Worker thread for layer preparation (encoding)."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, sam_model, layer, parent=None):
        super().__init__(parent)
        self.sam_model = sam_model
        self.layer = layer

    def run(self):
        """Run the preparation process."""
        try:
            success, message = self.sam_model.prepare_layer(
                self.layer,
                progress_callback=lambda p, m: self.progress.emit(p, m)
            )
            self.finished.emit(success, message)

        except Exception as e:
            self.finished.emit(False, str(e))


class AISegmentationPlugin:
    """
    Main plugin class for AI Segmentation.

    Key design principles:
    - SILENT startup: No popups or checks at QGIS startup
    - Lazy initialization: Only check dependencies when panel opens
    - Clear feedback: Show all status in the plugin panel
    - Stability first: Never crash QGIS
    - Multi-model support: SAM and SAM2 variants
    """

    def __init__(self, iface: QgisInterface):
        """Initialize the plugin."""
        self.iface = iface
        self.plugin_dir = Path(__file__).parent

        # Components (initialized in initGui)
        self.dock_widget: Optional[AISegmentationDockWidget] = None
        self.map_tool: Optional[AISegmentationMapTool] = None
        self.action: Optional[QAction] = None

        # SAM model (lazy loaded)
        self.sam_model = None
        self._current_model_id = None

        # State
        self.current_mask = None
        self.current_score = 0.0
        self._initialized = False  # Track if we've done first-time setup
        self._current_layer_name = ""  # Name of current raster layer for output naming
        self._segmentation_counter = 0  # Counter for naming exported layers

        # Worker threads
        self.install_worker = None
        self.download_worker = None
        self.prep_worker = None

        # Visual feedback (RubberBand only - no preview layer)
        self.mask_rubber_band: Optional[QgsRubberBand] = None

    def initGui(self):
        """
        Initialize the plugin GUI.

        Called by QGIS when the plugin is loaded.
        NOTE: We do NOT check dependencies here - only when panel opens.
        """
        # Create action for the plugin with SVG icon
        icon_path = str(self.plugin_dir / "resources" / "icons" / "ai_segmentation_icon.svg")
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

        # Add to Plugins menu (not Raster menu) and toolbar
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&AI Segmentation", self.action)

        # Create dock widget
        self.dock_widget = AISegmentationDockWidget(self.iface.mainWindow())
        self.dock_widget.setVisible(False)
        self.dock_widget.visibilityChanged.connect(self._on_dock_visibility_changed)

        # Connect dock widget signals
        self.dock_widget.install_dependencies_requested.connect(self._on_install_requested)
        self.dock_widget.install_model_requested.connect(self._on_install_model_requested)
        self.dock_widget.cancel_download_requested.connect(self._on_cancel_download)
        self.dock_widget.cancel_preparation_requested.connect(self._on_cancel_preparation)
        self.dock_widget.start_segmentation_requested.connect(self._on_start_segmentation)
        self.dock_widget.finish_segmentation_requested.connect(self._on_finish_segmentation)
        self.dock_widget.clear_points_requested.connect(self._on_clear_points)
        self.dock_widget.undo_requested.connect(self._on_undo)
        self.dock_widget.model_changed.connect(self._on_model_changed)

        # Add dock widget to QGIS
        self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)

        # Create map tool
        self.map_tool = AISegmentationMapTool(self.iface.mapCanvas())
        self.map_tool.positive_click.connect(self._on_positive_click)
        self.map_tool.negative_click.connect(self._on_negative_click)
        self.map_tool.tool_deactivated.connect(self._on_tool_deactivated)
        self.map_tool.undo_requested.connect(self._on_undo)

        # Create rubber band for mask visualization
        self.mask_rubber_band = QgsRubberBand(
            self.iface.mapCanvas(),
            QgsWkbTypes.PolygonGeometry
        )
        self.mask_rubber_band.setColor(QColor(0, 120, 255, 100))  # Semi-transparent blue fill
        self.mask_rubber_band.setStrokeColor(QColor(0, 80, 200))   # Darker blue stroke
        self.mask_rubber_band.setWidth(2)

        # Log that plugin loaded (but don't do any heavy work)
        QgsMessageLog.logMessage(
            "AI Segmentation plugin loaded (checks deferred until panel opens)",
            "AI Segmentation",
            level=Qgis.Info
        )

    def unload(self):
        """Unload the plugin."""
        # Remove menu and toolbar items
        self.iface.removePluginMenu("&AI Segmentation", self.action)
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

        # Remove rubber band
        if self.mask_rubber_band:
            self.iface.mapCanvas().scene().removeItem(self.mask_rubber_band)
            self.mask_rubber_band = None

        # Clean up workers
        for worker in [self.install_worker, self.download_worker, self.prep_worker]:
            if worker and worker.isRunning():
                worker.terminate()
                worker.wait()

        # Unload SAM model
        if self.sam_model:
            self.sam_model.unload()
            self.sam_model = None

    def toggle_dock_widget(self, checked: bool):
        """Toggle the dock widget visibility."""
        if self.dock_widget:
            self.dock_widget.setVisible(checked)

    def _on_dock_visibility_changed(self, visible: bool):
        """Handle dock widget visibility changes."""
        if self.action:
            self.action.setChecked(visible)

        # First time panel opens: do initialization
        if visible and not self._initialized:
            self._initialized = True
            self._do_first_time_setup()

    def _do_first_time_setup(self):
        """
        First-time setup when panel opens.

        This is where we check dependencies and models - NOT at startup.
        """
        QgsMessageLog.logMessage(
            "Panel opened - checking dependencies...",
            "AI Segmentation",
            level=Qgis.Info
        )

        # Step 1: Check dependencies
        try:
            from .core.dependency_manager import all_dependencies_installed, get_missing_dependencies

            if all_dependencies_installed():
                # Dependencies OK, check models
                self.dock_widget.set_dependency_status(True, "Dependencies OK")
                self._check_models()
            else:
                # Show what's missing
                missing = get_missing_dependencies()
                missing_str = ", ".join([f"{name}" for name, _ in missing])
                self.dock_widget.set_dependency_status(
                    False,
                    f"Missing: {missing_str}"
                )
                self.dock_widget.set_status("Click 'Install Dependencies' to continue")

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Dependency check error: {str(e)}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self.dock_widget.set_dependency_status(False, f"Error: {str(e)[:50]}")

    def _check_models(self):
        """Check which models are installed and update UI."""
        try:
            from .core.model_manager import get_installed_models, get_first_installed_model

            installed = get_installed_models()

            if installed:
                # At least one model installed - load the first one
                first_model = get_first_installed_model()
                self._load_models(first_model)
                self.dock_widget.set_models_status(installed, first_model)
                self.dock_widget.set_status("Ready - select a layer and start!")
            else:
                # No models installed
                self.dock_widget.set_models_status([], None)
                self.dock_widget.set_status("Install an AI model to get started")

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Model check error: {str(e)}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self.dock_widget.set_models_status([], None)

    def _load_models(self, model_id: str):
        """Load SAM models for a specific model ID."""
        try:
            from .core.sam_model import SAMModel

            self.sam_model = SAMModel(model_id=model_id)
            success, message = self.sam_model.load()

            if success:
                self._current_model_id = model_id
                QgsMessageLog.logMessage(
                    f"SAM models loaded successfully: {model_id}",
                    "AI Segmentation",
                    level=Qgis.Info
                )
            else:
                QgsMessageLog.logMessage(
                    f"Failed to load models: {message}",
                    "AI Segmentation",
                    level=Qgis.Warning
                )
                self.dock_widget.set_status(f"Error: {message[:50]}")

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to load models: {str(e)}",
                "AI Segmentation",
                level=Qgis.Warning
            )

    # ==================== Installation ====================

    def _on_install_requested(self):
        """Handle dependency installation request."""
        self.dock_widget.set_install_progress(0, "Starting installation...")

        self.install_worker = InstallWorker()
        self.install_worker.progress.connect(self._on_install_progress)
        self.install_worker.finished.connect(self._on_install_finished)
        self.install_worker.start()

    def _on_install_progress(self, percent: int, message: str):
        """Handle installation progress."""
        self.dock_widget.set_install_progress(percent, message)

    def _on_install_finished(self, success: bool, messages: list):
        """Handle installation completion."""
        if success:
            self.dock_widget.set_install_progress(100, "Installed! Restart QGIS.")
            self.dock_widget.set_dependency_status(True, "Restart QGIS to use")
            self.dock_widget.set_status("Please restart QGIS to complete setup")

            QMessageBox.information(
                self.iface.mainWindow(),
                "Installation Complete",
                "Dependencies installed successfully!\n\n"
                "Please restart QGIS for changes to take effect."
            )
        else:
            # Show error with manual instructions
            from .core.dependency_manager import get_manual_install_instructions

            error_text = "\n".join(messages)
            manual_instructions = get_manual_install_instructions()

            self.dock_widget.set_install_progress(0, "Installation failed")
            self.dock_widget.set_dependency_status(False, "Install failed")
            self.dock_widget.set_status("See error details below")

            QMessageBox.warning(
                self.iface.mainWindow(),
                "Installation Failed",
                f"Automatic installation failed:\n{error_text}\n\n"
                f"Please install manually:\n\n{manual_instructions}"
            )

    # ==================== Model Download ====================

    def _on_install_model_requested(self, model_id: str):
        """Handle model installation request for a specific model."""
        from .core.model_registry import get_model_config
        config = get_model_config(model_id)

        self.dock_widget.set_download_progress(0, f"Downloading {config.display_name}...")

        self.download_worker = DownloadWorker(model_id)
        self.download_worker.progress.connect(self._on_download_progress)
        self.download_worker.finished.connect(self._on_download_finished)
        self.download_worker.start()

    def _on_download_progress(self, percent: int, message: str):
        """Handle download progress."""
        self.dock_widget.set_download_progress(percent, message)

    def _on_download_finished(self, success: bool, message: str, model_id: str):
        """Handle download completion."""
        from .core.model_manager import get_installed_models

        if success:
            self.dock_widget.set_download_progress(100, "Download complete!")

            # Refresh installed models list
            installed = get_installed_models()

            # If this is the first model, load it
            if self.sam_model is None or not self.sam_model.is_loaded:
                self._load_models(model_id)
                self.dock_widget.set_models_status(installed, model_id)
            else:
                # Keep current model, just update UI
                self.dock_widget.set_models_status(installed, self._current_model_id)

            self.dock_widget.set_status(f"Model installed: {model_id}")
        else:
            self.dock_widget.set_download_progress(0, "")
            self.dock_widget.set_status(f"Error: {message[:50]}")

            QMessageBox.warning(
                self.iface.mainWindow(),
                "Download Failed",
                f"Failed to download model:\n{message}\n\n"
                "Please check your internet connection and try again."
            )

    def _on_cancel_download(self):
        """Handle cancel download request."""
        if self.download_worker and self.download_worker.isRunning():
            self.download_worker.terminate()
            self.download_worker.wait()
            self.dock_widget.set_download_progress(0, "Cancelled")
            self.dock_widget.set_status("Download cancelled by user")
            QgsMessageLog.logMessage(
                "Model download cancelled by user",
                "AI Segmentation",
                level=Qgis.Info
            )

    def _on_cancel_preparation(self):
        """Handle cancel preparation request."""
        if self.prep_worker and self.prep_worker.isRunning():
            self.prep_worker.terminate()
            self.prep_worker.wait()
            self.dock_widget.set_preparation_progress(0, "Cancelled")
            self.dock_widget.set_status("Layer preparation cancelled by user")
            QgsMessageLog.logMessage(
                "Layer preparation cancelled by user",
                "AI Segmentation",
                level=Qgis.Info
            )

    # ==================== Model Selection ====================

    def _on_model_changed(self, model_id: str):
        """Handle model selection change from the UI."""
        if model_id == self._current_model_id:
            return

        QgsMessageLog.logMessage(
            f"Switching model from {self._current_model_id} to {model_id}",
            "AI Segmentation",
            level=Qgis.Info
        )

        # Clear visual markers and mask visualization when switching models
        if self.map_tool:
            self.map_tool.clear_markers()
        self._clear_mask_visualization()
        self.current_mask = None
        self.current_score = 0.0
        self.dock_widget.set_point_count(0, 0)

        # Switch model
        if self.sam_model:
            success, message = self.sam_model.switch_model(model_id)
            if success:
                self._current_model_id = model_id
                self.dock_widget.set_status(f"Switched to {message}")
            else:
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "Model Switch Failed",
                    f"Failed to switch model:\n{message}"
                )
        else:
            # No model loaded yet, load the new one
            self._load_models(model_id)

    # ==================== Segmentation ====================

    def _on_start_segmentation(self, layer: QgsRasterLayer):
        """Handle start segmentation request."""
        if self.sam_model is None or not self.sam_model.is_loaded:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Not Ready",
                "Please install and select a model first."
            )
            return

        # Store layer name for export naming
        self._current_layer_name = layer.name().replace(" ", "_")

        # Check if layer is already prepared
        if self.sam_model.is_ready and self.sam_model.get_current_layer() == layer:
            self._activate_segmentation_tool()
            return

        # Need to prepare layer
        self.dock_widget.set_preparation_progress(0, "Preparing layer...")

        self.prep_worker = PreparationWorker(self.sam_model, layer)
        self.prep_worker.progress.connect(self._on_prep_progress)
        self.prep_worker.finished.connect(
            lambda s, m: self._on_prep_finished(s, m, layer)
        )
        self.prep_worker.start()

    def _on_prep_progress(self, percent: int, message: str):
        """Handle preparation progress."""
        self.dock_widget.set_preparation_progress(percent, message)

    def _on_prep_finished(self, success: bool, message: str, layer: QgsRasterLayer):
        """Handle preparation completion."""
        if success:
            self.dock_widget.set_preparation_progress(100, "Ready!")
            self._activate_segmentation_tool()
        else:
            self.dock_widget.set_preparation_progress(0, "")
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Preparation Failed",
                f"Failed to prepare layer:\n{message}"
            )

    def _activate_segmentation_tool(self):
        """Activate the segmentation map tool."""
        self.iface.mapCanvas().setMapTool(self.map_tool)
        self.dock_widget.set_segmentation_active(True)
        self.dock_widget.set_status("Click on the map to segment")

    def _on_finish_segmentation(self):
        """Handle finish segmentation - create layer and reset session."""
        # Deactivate map tool first
        self.iface.mapCanvas().unsetMapTool(self.map_tool)

        # If no mask (no points clicked), just reset silently
        if self.current_mask is None:
            self._reset_session()
            self.dock_widget.reset_session()
            self.dock_widget.set_status("Segmentation cancelled")
            return

        # Generate layer name based on raster name and counter
        self._segmentation_counter += 1
        layer_name = f"{self._current_layer_name}_segmentation_{self._segmentation_counter}"

        # Create memory vector layer with the segmentation result
        from .core.polygon_exporter import mask_to_polygons

        transform_info = self.sam_model.get_transform_info()
        geometries = mask_to_polygons(self.current_mask, transform_info)

        if not geometries:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "No Polygons",
                "Could not generate polygons from the segmentation."
            )
            self.dock_widget.set_segmentation_active(False)
            return

        # Get CRS from transform info
        crs = transform_info.get('crs')
        if crs is None:
            # Fallback to project CRS
            crs = QgsProject.instance().crs()

        # Create memory layer
        result_layer = QgsVectorLayer(
            f"Polygon?crs={crs.authid()}&field=id:integer&field=score:double&field=area:double",
            layer_name,
            "memory"
        )

        if not result_layer.isValid():
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Layer Creation Failed",
                "Could not create the output layer."
            )
            self.dock_widget.set_segmentation_active(False)
            return

        # Add features to layer
        result_layer.startEditing()
        for i, geom in enumerate(geometries):
            feature = QgsFeature()
            feature.setGeometry(geom)
            feature.setAttributes([i + 1, self.current_score, geom.area()])
            result_layer.addFeature(feature)
        result_layer.commitChanges()

        # Style with semi-transparent fill
        symbol = QgsFillSymbol.createSimple({
            'color': '0,120,255,100',
            'outline_color': '0,80,200,255',
            'outline_width': '0.5'
        })
        result_layer.renderer().setSymbol(symbol)

        # Add to project
        QgsProject.instance().addMapLayer(result_layer)

        QgsMessageLog.logMessage(
            f"Created segmentation layer: {layer_name}",
            "AI Segmentation",
            level=Qgis.Info
        )

        # Reset session
        self._reset_session()
        self.dock_widget.reset_session()
        self.dock_widget.set_status(f"Saved: {layer_name}")

    def _on_tool_deactivated(self):
        """Handle map tool deactivation."""
        self.dock_widget.set_segmentation_active(False)

    # ==================== Click Handling ====================

    def _on_positive_click(self, point):
        """Handle positive click - adds foreground point (include this area)."""
        if self.sam_model is None or not self.sam_model.is_ready:
            QgsMessageLog.logMessage(
                f"Model not ready: sam_model={self.sam_model is not None}, is_ready={self.sam_model.is_ready if self.sam_model else 'N/A'}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self.dock_widget.set_status("Model not ready - please wait")
            return

        QgsMessageLog.logMessage(
            "───────────────────────────────────────────────────────────",
            "AI Segmentation",
            level=Qgis.Info
        )
        QgsMessageLog.logMessage(
            f"LEFT-CLICK (POSITIVE/INCLUDE) at map coords ({point.x():.2f}, {point.y():.2f})",
            "AI Segmentation",
            level=Qgis.Info
        )

        self.current_mask, self.current_score = self.sam_model.add_positive_point(
            point.x(), point.y()
        )
        self._update_ui_after_click()

    def _on_negative_click(self, point):
        """
        Handle negative click - adds background point (exclude this area).

        NOTE: SAM negative points tell the model "this point is NOT part of
        the object I want". They work best when placed:
        - On regions that were incorrectly included in the mask
        - Near the boundary of the desired object
        - On nearby objects that should be excluded

        They do NOT create "exclusion zones" - they're hints to refine the segmentation.
        """
        if self.sam_model is None or not self.sam_model.is_ready:
            self.dock_widget.set_status("Model not ready - please wait")
            return

        QgsMessageLog.logMessage(
            "───────────────────────────────────────────────────────────",
            "AI Segmentation",
            level=Qgis.Info
        )
        QgsMessageLog.logMessage(
            f"RIGHT-CLICK (NEGATIVE/EXCLUDE) at map coords ({point.x():.2f}, {point.y():.2f})",
            "AI Segmentation",
            level=Qgis.Info
        )
        QgsMessageLog.logMessage(
            "   TIP: Negative points work best when placed ON incorrectly included areas",
            "AI Segmentation",
            level=Qgis.Info
        )

        self.current_mask, self.current_score = self.sam_model.add_negative_point(
            point.x(), point.y()
        )
        self._update_ui_after_click()

    def _update_ui_after_click(self):
        """Update UI and visualizations after click."""
        pos_count, neg_count = self.sam_model.prompts.point_count
        self.dock_widget.set_point_count(pos_count, neg_count)

        if self.current_mask is not None:
            mask_pixels = int(self.current_mask.sum())
            QgsMessageLog.logMessage(
                f"Segmentation result: score={self.current_score:.3f}, mask_pixels={mask_pixels}",
                "AI Segmentation",
                level=Qgis.Info
            )
            self.dock_widget.set_status(f"Segmented - Score: {self.current_score:.2f}")
            self._update_mask_visualization()
        else:
            QgsMessageLog.logMessage(
                "Segmentation returned None - check model and coordinates",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self.dock_widget.set_status("No mask generated - try clicking elsewhere")
            self._clear_mask_visualization()

    def _update_mask_visualization(self):
        """Update the rubber band and preview layer to show the current segmentation mask."""
        if self.mask_rubber_band is None:
            QgsMessageLog.logMessage("No rubber band available", "AI Segmentation", level=Qgis.Warning)
            return

        if self.current_mask is None or self.sam_model is None:
            self._clear_mask_visualization()
            return

        try:
            # Import here to avoid circular dependency
            from .core.polygon_exporter import mask_to_polygons

            # Convert mask to polygon geometries
            transform_info = self.sam_model.get_transform_info()
            QgsMessageLog.logMessage(
                f"Transform info keys: {list(transform_info.keys())}",
                "AI Segmentation",
                level=Qgis.Info
            )

            geometries = mask_to_polygons(self.current_mask, transform_info)
            QgsMessageLog.logMessage(
                f"Generated {len(geometries)} polygons from mask",
                "AI Segmentation",
                level=Qgis.Info
            )

            if geometries:
                # Combine all geometries into one for the rubber band
                combined = QgsGeometry.unaryUnion(geometries)
                if combined and not combined.isEmpty():
                    QgsMessageLog.logMessage(
                        f"Displaying mask: area={combined.area():.2f}",
                        "AI Segmentation",
                        level=Qgis.Info
                    )
                    self.mask_rubber_band.setToGeometry(combined, None)
                else:
                    QgsMessageLog.logMessage("Combined geometry is empty", "AI Segmentation", level=Qgis.Warning)
                    self._clear_mask_visualization()
            else:
                QgsMessageLog.logMessage("No geometries generated from mask", "AI Segmentation", level=Qgis.Warning)
                self._clear_mask_visualization()

        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Failed to visualize mask: {str(e)}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self._clear_mask_visualization()

    def _clear_mask_visualization(self):
        """Clear the mask rubber band."""
        if self.mask_rubber_band:
            self.mask_rubber_band.reset(QgsWkbTypes.PolygonGeometry)

    def _on_clear_points(self):
        """Clear all points, markers, and mask visualization."""
        if self.sam_model:
            self.sam_model.clear_points()

        # Clear visual markers
        if self.map_tool:
            self.map_tool.clear_markers()

        # Clear mask visualization
        self._clear_mask_visualization()

        self.current_mask = None
        self.current_score = 0.0
        self.dock_widget.set_point_count(0, 0)
        self.dock_widget.set_status("Points cleared")

    def _on_undo(self):
        """Undo last point and its visual marker."""
        if self.sam_model is None:
            return

        # Remove the last visual marker
        if self.map_tool:
            self.map_tool.remove_last_marker()

        # Undo in SAM model and update mask
        self.current_mask, self.current_score = self.sam_model.undo_point()
        self._update_ui_after_click()

    def _reset_session(self):
        """Reset the current segmentation session."""
        # Clear points in model
        if self.sam_model:
            self.sam_model.clear_points()

        # Clear visual markers
        if self.map_tool:
            self.map_tool.clear_markers()

        # Clear mask visualization
        self._clear_mask_visualization()

        # Reset state
        self.current_mask = None
        self.current_score = 0.0
        self.dock_widget.set_point_count(0, 0)
