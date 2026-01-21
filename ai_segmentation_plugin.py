"""
AI Segmentation - AI-powered segmentation plugin for QGIS

Main plugin class that coordinates all components.
Designed for stability - silent startup, checks only when panel opens.
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
    QgsMessageLog,
    Qgis,
    QgsWkbTypes,
    QgsGeometry,
)
from qgis.gui import QgisInterface, QgsRubberBand
from qgis.PyQt.QtGui import QColor

from .ai_segmentation_dockwidget import AISegmentationDockWidget
from .ai_segmentation_maptool import AISegmentationMapTool


class InstallWorker(QThread):
    """Worker thread for dependency installation."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, list)  # success, messages

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        """Run the installation process."""
        try:
            from .core.dependency_manager import install_all_dependencies

            def callback(current, total, msg):
                percent = int((current / total) * 100) if total > 0 else 0
                self.progress.emit(percent, msg)

            success, messages = install_all_dependencies(callback)
            self.finished.emit(success, messages)

        except Exception as e:
            self.finished.emit(False, [str(e)])


class DownloadWorker(QThread):
    """Worker thread for model downloading."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        """Run the download process."""
        try:
            from .core.model_manager import download_models

            success, message = download_models(
                progress_callback=lambda p, m: self.progress.emit(p, m)
            )
            self.finished.emit(success, message)

        except Exception as e:
            self.finished.emit(False, str(e))


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

        # State
        self.current_mask = None
        self.current_score = 0.0
        self.output_layer = None
        self._initialized = False  # Track if we've done first-time setup

        # Worker threads
        self.install_worker = None
        self.download_worker = None
        self.prep_worker = None

        # Visual feedback
        self.mask_rubber_band: Optional[QgsRubberBand] = None

    def initGui(self):
        """
        Initialize the plugin GUI.

        Called by QGIS when the plugin is loaded.
        NOTE: We do NOT check dependencies here - only when panel opens.
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
        self.dock_widget.install_dependencies_requested.connect(self._on_install_requested)
        self.dock_widget.download_models_requested.connect(self._on_download_requested)
        self.dock_widget.start_segmentation_requested.connect(self._on_start_segmentation)
        self.dock_widget.stop_segmentation_requested.connect(self._on_stop_segmentation)
        self.dock_widget.clear_points_requested.connect(self._on_clear_points)
        self.dock_widget.undo_requested.connect(self._on_undo)
        self.dock_widget.save_mask_requested.connect(self._on_save_mask)
        self.dock_widget.export_requested.connect(self._on_export)

        # Add dock widget to QGIS
        self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)

        # Create map tool
        self.map_tool = AISegmentationMapTool(self.iface.mapCanvas())
        self.map_tool.positive_click.connect(self._on_positive_click)
        self.map_tool.negative_click.connect(self._on_negative_click)
        self.map_tool.tool_deactivated.connect(self._on_tool_deactivated)

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
                self.dock_widget.set_dependency_status(True, "Dependencies OK ✓")
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
        """Check if models are downloaded and load them."""
        try:
            from .core.model_manager import models_exist

            if models_exist():
                self._load_models()
            else:
                self.dock_widget.set_models_status(False, "Models not downloaded")
                self.dock_widget.set_status("Click 'Download Models' to continue")

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Model check error: {str(e)}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self.dock_widget.set_models_status(False, f"Error: {str(e)[:50]}")

    def _load_models(self):
        """Load SAM models."""
        try:
            from .core.sam_model import SAMModel

            self.sam_model = SAMModel()
            success, message = self.sam_model.load()

            if success:
                self.dock_widget.set_models_status(True, "Models ready ✓")
                self.dock_widget.set_status("Ready - select a layer and start!")
                QgsMessageLog.logMessage(
                    "SAM models loaded successfully",
                    "AI Segmentation",
                    level=Qgis.Info
                )
            else:
                self.dock_widget.set_models_status(False, f"Load failed")
                self.dock_widget.set_status(f"Error: {message[:50]}")

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to load models: {str(e)}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self.dock_widget.set_models_status(False, "Load failed")

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

    def _on_download_requested(self):
        """Handle model download request."""
        self.dock_widget.set_download_progress(0, "Starting download...")

        self.download_worker = DownloadWorker()
        self.download_worker.progress.connect(self._on_download_progress)
        self.download_worker.finished.connect(self._on_download_finished)
        self.download_worker.start()

    def _on_download_progress(self, percent: int, message: str):
        """Handle download progress."""
        self.dock_widget.set_download_progress(percent, message)

    def _on_download_finished(self, success: bool, message: str):
        """Handle download completion."""
        if success:
            self.dock_widget.set_download_progress(100, "Download complete!")
            self._load_models()
        else:
            self.dock_widget.set_models_status(False, "Download failed")
            self.dock_widget.set_status(f"Error: {message[:50]}")

            QMessageBox.warning(
                self.iface.mainWindow(),
                "Download Failed",
                f"Failed to download models:\n{message}\n\n"
                "Please check your internet connection and try again."
            )

    # ==================== Segmentation ====================

    def _on_start_segmentation(self, layer: QgsRasterLayer):
        """Handle start segmentation request."""
        if self.sam_model is None or not self.sam_model.is_loaded:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Not Ready",
                "Please download models first."
            )
            return

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
            self._create_output_layer(layer)
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

    def _on_stop_segmentation(self):
        """Handle stop segmentation request."""
        # Clear visual feedback
        self._clear_mask_visualization()

        self.iface.mapCanvas().unsetMapTool(self.map_tool)
        self.dock_widget.set_segmentation_active(False)
        self.dock_widget.set_status("Ready")

    def _on_tool_deactivated(self):
        """Handle map tool deactivation."""
        self.dock_widget.set_segmentation_active(False)

    def _create_output_layer(self, source_layer: QgsRasterLayer):
        """Create output vector layer."""
        from .core.polygon_exporter import create_output_layer

        if self.output_layer is None or not self.output_layer.isValid():
            self.output_layer = create_output_layer(
                source_layer.crs(),
                "AI_Segmentation_Output"
            )
            QgsProject.instance().addMapLayer(self.output_layer)

    # ==================== Click Handling ====================

    def _on_positive_click(self, point):
        """Handle positive click."""
        if self.sam_model is None or not self.sam_model.is_ready:
            return

        self.current_mask, self.current_score = self.sam_model.add_positive_point(
            point.x(), point.y()
        )
        self._update_ui_after_click()

    def _on_negative_click(self, point):
        """Handle negative click."""
        if self.sam_model is None or not self.sam_model.is_ready:
            return

        self.current_mask, self.current_score = self.sam_model.add_negative_point(
            point.x(), point.y()
        )
        self._update_ui_after_click()

    def _update_ui_after_click(self):
        """Update UI and visualizations after click."""
        pos_count, neg_count = self.sam_model.prompts.point_count
        self.dock_widget.set_point_count(pos_count, neg_count)

        if self.current_mask is not None:
            self.dock_widget.set_status(f"Mask score: {self.current_score:.3f}")
            self._update_mask_visualization()
        else:
            self._clear_mask_visualization()

    def _update_mask_visualization(self):
        """Update the rubber band to show the current segmentation mask."""
        if self.mask_rubber_band is None:
            return

        if self.current_mask is None or self.sam_model is None:
            self._clear_mask_visualization()
            return

        try:
            # Import here to avoid circular dependency
            from .core.polygon_exporter import mask_to_polygons

            # Convert mask to polygon geometries
            transform_info = self.sam_model.get_transform_info()
            geometries = mask_to_polygons(self.current_mask, transform_info)

            if geometries:
                # Combine all geometries into one for the rubber band
                combined = QgsGeometry.unaryUnion(geometries)
                if combined and not combined.isEmpty():
                    self.mask_rubber_band.setToGeometry(combined, None)
                else:
                    self._clear_mask_visualization()
            else:
                self._clear_mask_visualization()

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to visualize mask: {str(e)}",
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

    def _on_save_mask(self):
        """Save current mask."""
        if self.current_mask is None:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "No Mask",
                "Click on the map to create a segmentation first."
            )
            return

        if self.output_layer is None:
            return

        from .core.polygon_exporter import add_mask_to_layer

        count = add_mask_to_layer(
            self.output_layer,
            self.current_mask,
            self.sam_model.get_transform_info(),
            self.current_score
        )

        if count > 0:
            self.dock_widget.set_status(f"Saved {count} polygon(s)")
            self._on_clear_points()
        else:
            self.dock_widget.set_status("Failed to save mask")

    def _on_export(self, output_path: str):
        """Export to GeoPackage."""
        if self.output_layer is None or self.output_layer.featureCount() == 0:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Nothing to Export",
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
