import os
from pathlib import Path
from typing import Optional, List, Tuple
import sys

from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal, QVariant

from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsMessageLog,
    Qgis,
    QgsWkbTypes,
    QgsGeometry,
    QgsFeature,
    QgsField,
    QgsFillSymbol,
    QgsSingleSymbolRenderer,
    QgsRectangle,
    QgsCoordinateReferenceSystem,
)
from qgis.gui import QgisInterface, QgsRubberBand
from qgis.PyQt.QtGui import QColor

from .ai_segmentation_dockwidget import AISegmentationDockWidget
from .ai_segmentation_maptool import AISegmentationMapTool




class DepsInstallWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from ..core.venv_manager import create_venv_and_install
            success, message = create_venv_and_install(
                progress_callback=lambda percent, msg: self.progress.emit(percent, msg),
                cancel_check=lambda: self._cancelled
            )
            self.finished.emit(success, message)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)


class DownloadWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        try:
            from ..core.checkpoint_manager import download_checkpoint
            success, message = download_checkpoint(
                progress_callback=lambda p, m: self.progress.emit(p, m)
            )
            self.finished.emit(success, message)
        except Exception as e:
            self.finished.emit(False, str(e))


class EncodingWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, raster_path: str, output_dir: str, checkpoint_path: str, layer_crs_wkt: str = None, layer_extent: tuple = None, parent=None):
        super().__init__(parent)
        self.raster_path = raster_path
        self.output_dir = output_dir
        self.checkpoint_path = checkpoint_path
        self.layer_crs_wkt = layer_crs_wkt
        self.layer_extent = layer_extent  # (xmin, ymin, xmax, ymax)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from ..core.feature_encoder import encode_raster_to_features
            success, message = encode_raster_to_features(
                self.raster_path,
                self.output_dir,
                self.checkpoint_path,
                layer_crs_wkt=self.layer_crs_wkt,
                layer_extent=self.layer_extent,
                progress_callback=lambda p, m: self.progress.emit(p, m),
                cancel_check=lambda: self._cancelled,
            )
            self.finished.emit(success, message)
        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Encoding error: {str(e)}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self.finished.emit(False, str(e))


class PromptManager:
    def __init__(self):
        self.positive_points: List[Tuple[float, float]] = []
        self.negative_points: List[Tuple[float, float]] = []
        self._history: List[Tuple[str, Tuple[float, float]]] = []

    def add_positive_point(self, x: float, y: float):
        self.positive_points.append((x, y))
        self._history.append(("positive", (x, y)))

    def add_negative_point(self, x: float, y: float):
        self.negative_points.append((x, y))
        self._history.append(("negative", (x, y)))

    def undo(self) -> Optional[Tuple[str, Tuple[float, float]]]:
        if not self._history:
            return None

        label, point = self._history.pop()
        if label == "positive" and point in self.positive_points:
            self.positive_points.remove(point)
        elif label == "negative" and point in self.negative_points:
            self.negative_points.remove(point)

        return label, point

    def clear(self):
        self.positive_points = []
        self.negative_points = []
        self._history = []

    @property
    def point_count(self) -> Tuple[int, int]:
        return len(self.positive_points), len(self.negative_points)

    def get_points_for_predictor(self, transform) -> Tuple[Optional['np.ndarray'], Optional['np.ndarray']]:
        import numpy as np
        from rasterio import transform as rio_transform

        all_points = self.positive_points + self.negative_points
        if not all_points:
            return None, None

        point_coords = []
        point_labels = []

        QgsMessageLog.logMessage(
            f"DEBUG get_points_for_predictor - Transform: {transform}",
            "AI Segmentation",
            level=Qgis.Info
        )

        for x, y in self.positive_points:
            row, col = rio_transform.rowcol(transform, x, y)
            QgsMessageLog.logMessage(
                f"DEBUG - Positive point geo ({x:.2f}, {y:.2f}) -> pixel (row={row}, col={col})",
                "AI Segmentation",
                level=Qgis.Info
            )
            point_coords.append([col, row])
            point_labels.append(1)

        for x, y in self.negative_points:
            row, col = rio_transform.rowcol(transform, x, y)
            QgsMessageLog.logMessage(
                f"DEBUG - Negative point geo ({x:.2f}, {y:.2f}) -> pixel (row={row}, col={col})",
                "AI Segmentation",
                level=Qgis.Info
            )
            point_coords.append([col, row])
            point_labels.append(0)

        return np.array(point_coords), np.array(point_labels)


class AISegmentationPlugin:

    def __init__(self, iface: QgisInterface):
        self.iface = iface
        self.plugin_dir = Path(__file__).parent.parent.parent

        self.dock_widget: Optional[AISegmentationDockWidget] = None
        self.map_tool: Optional[AISegmentationMapTool] = None
        self.action: Optional[QAction] = None

        self.predictor = None
        self.feature_dataset = None
        self.prompts = PromptManager()

        self.current_mask = None
        self.current_score = 0.0
        self.current_transform_info = None
        self.saved_polygons = []

        self._initialized = False
        self._current_layer = None
        self._current_layer_name = ""
        self._segmentation_counter = 0

        # Refinement settings
        self._refine_expand = 0
        self._refine_simplify = 0

        self.deps_install_worker = None
        self.download_worker = None
        self.encoding_worker = None

        self.mask_rubber_band: Optional[QgsRubberBand] = None
        self.saved_rubber_bands: List[QgsRubberBand] = []

        self._previous_map_tool = None  # Store the tool active before segmentation
        self._stopping_segmentation = False  # Flag to track if we're stopping programmatically

    def initGui(self):
        icon_path = str(self.plugin_dir / "resources" / "icons" / "icon.png")
        if not os.path.exists(icon_path):
            icon = QIcon()
        else:
            icon = QIcon(icon_path)

        self.action = QAction(
            icon,
            "AI Segmentation by TerraLab",
            self.iface.mainWindow()
        )
        self.action.setCheckable(True)
        self.action.setToolTip(
            "AI Segmentation by TerraLab\n"
            "Segment objects on raster images using AI"
        )
        self.action.triggered.connect(self.toggle_dock_widget)

        self.iface.addToolBarIcon(self.action)

        # Add directly to Plugins menu (no submenu)
        plugins_menu = self.iface.pluginMenu()
        plugins_menu.addAction(self.action)

        self.dock_widget = AISegmentationDockWidget(self.iface.mainWindow())
        self.dock_widget.setVisible(False)
        self.dock_widget.visibilityChanged.connect(self._on_dock_visibility_changed)

        self.dock_widget.install_dependencies_requested.connect(self._on_install_requested)
        self.dock_widget.cancel_deps_install_requested.connect(self._on_cancel_deps_install)
        self.dock_widget.download_checkpoint_requested.connect(self._on_download_checkpoint_requested)
        self.dock_widget.cancel_download_requested.connect(self._on_cancel_download)
        self.dock_widget.cancel_preparation_requested.connect(self._on_cancel_preparation)
        self.dock_widget.start_segmentation_requested.connect(self._on_start_segmentation)
        self.dock_widget.save_polygon_requested.connect(self._on_save_polygon)
        self.dock_widget.export_layer_requested.connect(self._on_export_layer)
        self.dock_widget.clear_points_requested.connect(self._on_clear_points)
        self.dock_widget.undo_requested.connect(self._on_undo)
        self.dock_widget.stop_segmentation_requested.connect(self._on_stop_segmentation)
        self.dock_widget.refine_settings_changed.connect(self._on_refine_settings_changed)

        self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)

        self.map_tool = AISegmentationMapTool(self.iface.mapCanvas())
        self.map_tool.positive_click.connect(self._on_positive_click)
        self.map_tool.negative_click.connect(self._on_negative_click)
        self.map_tool.tool_deactivated.connect(self._on_tool_deactivated)
        self.map_tool.undo_requested.connect(self._on_undo)
        self.map_tool.save_polygon_requested.connect(self._on_save_polygon)
        self.map_tool.export_layer_requested.connect(self._on_export_layer)
        self.map_tool.clear_requested.connect(self._on_clear_points)

        self.mask_rubber_band = QgsRubberBand(
            self.iface.mapCanvas(),
            QgsWkbTypes.PolygonGeometry
        )
        self.mask_rubber_band.setColor(QColor(0, 120, 255, 100))
        self.mask_rubber_band.setStrokeColor(QColor(0, 80, 200))
        self.mask_rubber_band.setWidth(2)

        QgsMessageLog.logMessage(
            "AI Segmentation plugin loaded (checks deferred until panel opens)",
            "AI Segmentation",
            level=Qgis.Info
        )

    def unload(self):
        if self.predictor:
            try:
                self.predictor.cleanup()
            except Exception:
                pass
            self.predictor = None

        # Remove from Plugins menu
        plugins_menu = self.iface.pluginMenu()
        plugins_menu.removeAction(self.action)
        self.iface.removeToolBarIcon(self.action)

        if self.dock_widget:
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget.deleteLater()
            self.dock_widget = None

        if self.map_tool:
            if self.iface.mapCanvas().mapTool() == self.map_tool:
                self.iface.mapCanvas().unsetMapTool(self.map_tool)
            self.map_tool = None

        if self.mask_rubber_band:
            self.iface.mapCanvas().scene().removeItem(self.mask_rubber_band)
            self.mask_rubber_band = None

        for rb in self.saved_rubber_bands:
            self.iface.mapCanvas().scene().removeItem(rb)
        self.saved_rubber_bands = []

        for worker in [self.deps_install_worker, self.download_worker, self.encoding_worker]:
            if worker and worker.isRunning():
                worker.terminate()
                worker.wait()

    def toggle_dock_widget(self, checked: bool):
        if self.dock_widget:
            self.dock_widget.setVisible(checked)

    def _on_dock_visibility_changed(self, visible: bool):
        if self.action:
            self.action.setChecked(visible)

        if visible and not self._initialized:
            self._initialized = True
            self._do_first_time_setup()

    def _do_first_time_setup(self):
        QgsMessageLog.logMessage(
            "Panel opened - checking dependencies...",
            "AI Segmentation",
            level=Qgis.Info
        )

        try:
            from ..core.venv_manager import get_venv_status, cleanup_old_libs

            cleanup_old_libs()

            is_ready, message = get_venv_status()

            if is_ready:
                self.dock_widget.set_dependency_status(True, "✓ Virtual environment ready")
                QgsMessageLog.logMessage(
                    "✓ Virtual environment verified successfully",
                    "AI Segmentation",
                    level=Qgis.Success
                )
                self._check_checkpoint()
            else:
                self.dock_widget.set_dependency_status(False, message)
                QgsMessageLog.logMessage(
                    f"Virtual environment status: {message}",
                    "AI Segmentation",
                    level=Qgis.Info
                )

        except Exception as e:
            import traceback
            error_msg = f"Dependency check error: {str(e)}\n{traceback.format_exc()}"
            QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.Warning)
            self.dock_widget.set_dependency_status(False, f"Error: {str(e)[:50]}")

    def _check_checkpoint(self):
        try:
            from ..core.checkpoint_manager import checkpoint_exists

            if checkpoint_exists():
                self.dock_widget.set_checkpoint_status(True, "SAM model ready")
                self._load_predictor()
            else:
                self.dock_widget.set_checkpoint_status(False, "Model not downloaded")

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Checkpoint check error: {str(e)}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            self.dock_widget.set_checkpoint_status(False, f"Error: {str(e)[:50]}")

    def _load_predictor(self):
        try:
            from ..core.checkpoint_manager import get_checkpoint_path
            from ..core.sam_predictor import build_sam_vit_b_no_encoder, SamPredictorNoImgEncoder

            checkpoint_path = get_checkpoint_path()
            sam_config = build_sam_vit_b_no_encoder(checkpoint=checkpoint_path)
            self.predictor = SamPredictorNoImgEncoder(sam_config)

            QgsMessageLog.logMessage(
                f"SAM predictor initialized (subprocess mode)",
                "AI Segmentation",
                level=Qgis.Info
            )
            self.dock_widget.set_checkpoint_status(True, "SAM ready (subprocess)")

        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Failed to initialize predictor: {str(e)}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.Warning
            )

    def _verify_venv(self):
        """Verify virtual environment status."""
        try:
            from ..core.venv_manager import verify_venv
            is_valid, message = verify_venv()

            if is_valid:
                QgsMessageLog.logMessage(
                    "✓ Virtual environment verified successfully",
                    "AI Segmentation",
                    level=Qgis.Success
                )
            else:
                QgsMessageLog.logMessage(
                    f"⚠ Virtual environment verification failed: {message}",
                    "AI Segmentation",
                    level=Qgis.Warning
                )
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Isolation verification error: {e}",
                "AI Segmentation",
                level=Qgis.Warning
            )

    def _on_install_requested(self):
        from ..core.venv_manager import get_venv_status

        is_ready, message = get_venv_status()
        if is_ready:
            self.dock_widget.set_dependency_status(True, "✓ Virtual environment ready")
            self._check_checkpoint()
            return

        reply = QMessageBox.question(
            self.iface.mainWindow(),
            "Install Dependencies",
            "Download ~2.5GB of AI dependencies?\n\n"
            "This may take 3-5 minutes.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            self.dock_widget.install_button.setEnabled(True)
            return

        QgsMessageLog.logMessage(
            "Starting virtual environment creation and dependency installation...",
            "AI Segmentation",
            level=Qgis.Info
        )
        QgsMessageLog.logMessage(
            f"Platform: {sys.platform}, Python: {sys.version}",
            "AI Segmentation",
            level=Qgis.Info
        )

        self.dock_widget.set_deps_install_progress(0, "Preparing installation...")

        self.deps_install_worker = DepsInstallWorker()
        self.deps_install_worker.progress.connect(self._on_deps_install_progress)
        self.deps_install_worker.finished.connect(self._on_deps_install_finished)
        self.deps_install_worker.start()

        # Show activation popup 2 seconds after install starts (if not already activated)
        from ..core.activation_manager import is_plugin_activated
        if not is_plugin_activated():
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(2000, self._show_activation_popup_if_needed)

    def _show_activation_popup_if_needed(self):
        """Show activation popup during installation if not already activated."""
        from ..core.activation_manager import is_plugin_activated
        if not is_plugin_activated() and not self.dock_widget.is_activated():
            self.dock_widget.show_activation_dialog()

    def _on_deps_install_progress(self, percent: int, message: str):
        self.dock_widget.set_deps_install_progress(percent, message)

    def _on_deps_install_finished(self, success: bool, message: str):
        self.dock_widget.set_deps_install_progress(100, "Done")

        if success:
            from ..core.venv_manager import verify_venv
            is_valid, verify_msg = verify_venv()

            if is_valid:
                self.dock_widget.set_dependency_status(True, "✓ Virtual environment ready")
                self._verify_venv()
                self._check_checkpoint()
            else:
                self.dock_widget.set_dependency_status(False, f"Verification failed: {verify_msg}")

                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "Verification Failed",
                    f"Virtual environment was created but verification failed:\n\n{verify_msg}\n\n"
                    "Please check the logs or try reinstalling."
                )
        else:
            error_msg = message[:300] if message else "Unknown error"
            self.dock_widget.set_dependency_status(False, "Installation failed")

            QMessageBox.warning(
                self.iface.mainWindow(),
                "Installation Failed",
                f"Failed to install dependencies:\n\n{error_msg}\n\n"
                "Check the QGIS log panel (View → Panels → Log Messages) "
                "for detailed error information."
            )

    def _on_cancel_deps_install(self):
        if self.deps_install_worker and self.deps_install_worker.isRunning():
            self.deps_install_worker.cancel()
            QgsMessageLog.logMessage(
                "Dependency installation cancelled by user",
                "AI Segmentation",
                level=Qgis.Warning
            )

    def _on_download_checkpoint_requested(self):
        self.dock_widget.set_download_progress(0, "Downloading SAM checkpoint...")

        self.download_worker = DownloadWorker()
        self.download_worker.progress.connect(self._on_download_progress)
        self.download_worker.finished.connect(self._on_download_finished)
        self.download_worker.start()

    def _on_download_progress(self, percent: int, message: str):
        self.dock_widget.set_download_progress(percent, message)

    def _on_download_finished(self, success: bool, message: str):
        if success:
            self.dock_widget.set_download_progress(100, "Download complete!")
            self.dock_widget.set_checkpoint_status(True, "SAM model ready")
            self._load_predictor()
        else:
            self.dock_widget.set_download_progress(0, "")

            QMessageBox.warning(
                self.iface.mainWindow(),
                "Download Failed",
                f"Failed to download model:\n{message}\n\n"
                "Please check your internet connection and try again."
            )

    def _on_cancel_download(self):
        if self.download_worker and self.download_worker.isRunning():
            self.download_worker.terminate()
            self.download_worker.wait()
            self.dock_widget.set_download_progress(0, "Cancelled")

    def _on_cancel_preparation(self):
        if self.encoding_worker and self.encoding_worker.isRunning():
            self.encoding_worker.cancel()
            # The worker will emit finished signal with cancelled message
            # Clean up will happen in _on_encoding_finished

    def _on_start_segmentation(self, layer: QgsRasterLayer):
        if self.predictor is None:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Not Ready",
                "Please wait for the SAM model to load."
            )
            return

        self._reset_session()

        self._current_layer = layer
        self._current_layer_name = layer.name().replace(" ", "_")
        raster_path = layer.source()

        from ..core.checkpoint_manager import has_features_for_raster, get_raster_features_dir, get_checkpoint_path

        if has_features_for_raster(raster_path):
            self._load_features_and_activate(raster_path)
        else:
            output_dir = get_raster_features_dir(raster_path)
            checkpoint_path = get_checkpoint_path()

            layer_crs_wkt = None
            if layer.crs().isValid():
                layer_crs_wkt = layer.crs().toWkt()

            # Get layer extent for rasters without embedded georeferencing (e.g., PNG)
            layer_extent = None
            ext = layer.extent()
            if ext and not ext.isEmpty():
                layer_extent = (ext.xMinimum(), ext.yMinimum(), ext.xMaximum(), ext.yMaximum())

            QgsMessageLog.logMessage(
                f"Starting encoding - Layer extent: {layer_extent}, Layer CRS valid: {layer.crs().isValid()}, CRS: {layer.crs().authid() if layer.crs().isValid() else 'None'}",
                "AI Segmentation",
                level=Qgis.Info
            )

            self.dock_widget.set_preparation_progress(0, "Encoding raster (first time)...")

            self.encoding_worker = EncodingWorker(raster_path, output_dir, checkpoint_path, layer_crs_wkt, layer_extent)
            self.encoding_worker.progress.connect(self._on_encoding_progress)
            self.encoding_worker.finished.connect(
                lambda s, m: self._on_encoding_finished(s, m, raster_path)
            )
            self.encoding_worker.start()

    def _on_encoding_progress(self, percent: int, message: str):
        self.dock_widget.set_preparation_progress(percent, message)

    def _on_encoding_finished(self, success: bool, message: str, raster_path: str):
        if success:
            from ..core.checkpoint_manager import get_raster_features_dir
            cache_dir = get_raster_features_dir(raster_path)
            self.dock_widget.set_preparation_progress(100, "Done!")
            self.dock_widget.set_encoding_cache_path(str(cache_dir))
            self._load_features_and_activate(raster_path)
        else:
            # Check if this was a user cancellation
            is_cancelled = "cancelled" in message.lower() or "canceled" in message.lower()

            # Clean up partial cache files
            from ..core.checkpoint_manager import clear_features_for_raster
            try:
                clear_features_for_raster(raster_path)
                QgsMessageLog.logMessage(
                    f"Cleaned up partial cache for: {raster_path}",
                    "AI Segmentation",
                    level=Qgis.Info
                )
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Failed to clean up partial cache: {str(e)}",
                    "AI Segmentation",
                    level=Qgis.Warning
                )

            # Reset UI to default state
            self.dock_widget.set_preparation_progress(100, "Cancelled")
            self.dock_widget.reset_session()

            if not is_cancelled:
                # Actual error - show warning dialog
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "Encoding Failed",
                    f"Failed to encode raster:\n{message}"
                )

    def _load_features_and_activate(self, raster_path: str):
        try:
            from ..core.checkpoint_manager import get_raster_features_dir
            from ..core.feature_dataset import FeatureDataset

            features_dir = get_raster_features_dir(raster_path)
            self.feature_dataset = FeatureDataset(features_dir, cache=True)

            bounds = self.feature_dataset.bounds
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            raster_crs = self._current_layer.crs() if self._current_layer else None
            raster_extent = self._current_layer.extent() if self._current_layer else None

            QgsMessageLog.logMessage(
                f"Loaded {len(self.feature_dataset)} feature tiles",
                "AI Segmentation",
                level=Qgis.Info
            )
            QgsMessageLog.logMessage(
                f"Feature dataset bounds: minx={bounds[0]:.2f}, maxx={bounds[1]:.2f}, "
                f"miny={bounds[2]:.2f}, maxy={bounds[3]:.2f}, CRS={self.feature_dataset.crs}",
                "AI Segmentation",
                level=Qgis.Info
            )
            QgsMessageLog.logMessage(
                f"DEBUG - Canvas CRS: {canvas_crs.authid() if canvas_crs else 'None'}, "
                f"Raster CRS: {raster_crs.authid() if raster_crs else 'None'}",
                "AI Segmentation",
                level=Qgis.Info
            )
            if raster_extent:
                QgsMessageLog.logMessage(
                    f"DEBUG - Raster layer extent (in layer CRS): xmin={raster_extent.xMinimum():.2f}, "
                    f"xmax={raster_extent.xMaximum():.2f}, ymin={raster_extent.yMinimum():.2f}, ymax={raster_extent.yMaximum():.2f}",
                    "AI Segmentation",
                    level=Qgis.Info
                )

            self._activate_segmentation_tool()

        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Failed to load features: {str(e)}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Load Failed",
                f"Failed to load feature data:\n{str(e)}"
            )

    def _activate_segmentation_tool(self):
        # Save the current map tool to restore it later
        current_tool = self.iface.mapCanvas().mapTool()
        if current_tool and current_tool != self.map_tool:
            self._previous_map_tool = current_tool

        self.iface.mapCanvas().setMapTool(self.map_tool)
        self.dock_widget.set_segmentation_active(True)
        # Status bar hint will be set by _update_status_hint via set_point_count

    def _on_save_polygon(self):
        """Save current mask as polygon."""
        if self.current_mask is None:
            return

        from ..core.polygon_exporter import mask_to_polygons

        geometries = mask_to_polygons(self.current_mask, self.current_transform_info)

        if not geometries:
            return

        combined = QgsGeometry.unaryUnion(geometries)
        if combined and not combined.isEmpty():
            # Store WKT, score, transform info, AND raw mask for later refinement
            self.saved_polygons.append({
                'geometry_wkt': combined.asWkt(),
                'score': self.current_score,
                'transform_info': self.current_transform_info.copy() if self.current_transform_info else None,
                'raw_mask': self.current_mask.copy(),  # Store raw mask for refinement
            })

            saved_rb = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
            saved_rb.setColor(QColor(0, 200, 100, 120))
            saved_rb.setFillColor(QColor(0, 200, 100, 80))
            saved_rb.setWidth(2)
            saved_rb.setToGeometry(QgsGeometry(combined), None)
            self.saved_rubber_bands.append(saved_rb)

            QgsMessageLog.logMessage(
                f"Saved mask #{len(self.saved_polygons)}",
                "AI Segmentation",
                level=Qgis.Info
            )

            self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))

            # Reset refinement sliders when a new mask is saved
            self._refine_expand = 0
            self._refine_fill = 0
            self._refine_smooth = 0
            if self.dock_widget:
                self.dock_widget.reset_refine_sliders()

        # Clear current state for next polygon
        self.prompts.clear()
        if self.map_tool:
            self.map_tool.clear_markers()
        self._clear_mask_visualization()
        self.current_mask = None
        self.current_score = 0.0
        self.dock_widget.set_point_count(0, 0)

    def _on_export_layer(self):
        # Only allow export if at least one polygon is saved
        if not self.saved_polygons:
            # Silently ignore - Enter should only work when polygons are saved
            return

        from ..core.polygon_exporter import mask_to_polygons

        # Copy saved_polygons - note that the last one's geometry_wkt may have been
        # updated by refinement sliders via _update_last_saved_mask_visualization
        polygons_to_export = list(self.saved_polygons)

        # Include current unsaved mask if exists (no refinement - refinement is for saved masks only)
        if self.current_mask is not None:
            geometries = mask_to_polygons(self.current_mask, self.current_transform_info)
            if geometries:
                combined = QgsGeometry.unaryUnion(geometries)
                if combined and not combined.isEmpty():
                    polygons_to_export.append({
                        'geometry_wkt': combined.asWkt(),
                        'score': self.current_score,
                        'transform_info': self.current_transform_info.copy() if self.current_transform_info else None,
                    })

        self._stopping_segmentation = True
        self.iface.mapCanvas().unsetMapTool(self.map_tool)
        self._restore_previous_map_tool()
        self._stopping_segmentation = False

        self._segmentation_counter += 1
        layer_name = f"{self._current_layer_name}_segmentation_{self._segmentation_counter}"

        # Determine CRS (same logic as before)
        crs_str = None
        for pg in polygons_to_export:
            ti = pg.get('transform_info')
            if ti:
                crs_str = ti.get('crs', None)
                if crs_str and not (isinstance(crs_str, float) and str(crs_str) == 'nan'):
                    break
        if crs_str is None or (isinstance(crs_str, float) and str(crs_str) == 'nan'):
            if self.current_transform_info:
                crs_str = self.current_transform_info.get('crs', None)
        if crs_str is None or (isinstance(crs_str, float) and str(crs_str) == 'nan'):
            crs_str = self._current_layer.crs().authid() if self._current_layer and self._current_layer.crs().isValid() else 'EPSG:4326'
        if isinstance(crs_str, str) and crs_str.strip():
            crs = QgsCoordinateReferenceSystem(crs_str)
        else:
            crs = self._current_layer.crs() if self._current_layer else QgsCoordinateReferenceSystem('EPSG:4326')

        # Determine output directory for GeoPackage file
        # Priority: 1) Project directory (if project is saved), 2) Raster source directory
        output_dir = None
        project_path = QgsProject.instance().absolutePath()
        if project_path:
            output_dir = project_path
        elif self._current_layer:
            raster_source = self._current_layer.source()
            if raster_source and os.path.exists(raster_source):
                output_dir = os.path.dirname(raster_source)

        if not output_dir:
            # Fallback to user's home directory
            output_dir = str(Path.home())

        # Create unique GeoPackage filename
        gpkg_path = os.path.join(output_dir, f"{layer_name}.gpkg")
        counter = 1
        while os.path.exists(gpkg_path):
            gpkg_path = os.path.join(output_dir, f"{layer_name}_{counter}.gpkg")
            counter += 1

        # Create a temporary memory layer to build features
        temp_layer = QgsVectorLayer("MultiPolygon", layer_name, "memory")
        if not temp_layer.isValid():
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Layer Creation Failed",
                "Could not create the output layer."
            )
            return

        temp_layer.setCrs(crs)

        # Add attributes
        pr = temp_layer.dataProvider()
        pr.addAttributes([
            QgsField("id", QVariant.Int),
            QgsField("score", QVariant.Double),
            QgsField("area", QVariant.Double)
        ])
        temp_layer.updateFields()

        # Add features to temp layer
        features_to_add = []
        for i, polygon_data in enumerate(polygons_to_export):
            feature = QgsFeature(temp_layer.fields())

            # Reconstruct geometry from WKT
            geom_wkt = polygon_data.get('geometry_wkt')
            if not geom_wkt:
                QgsMessageLog.logMessage(
                    f"Polygon {i+1} has no WKT data",
                    "AI Segmentation",
                    level=Qgis.Warning
                )
                continue

            geom = QgsGeometry.fromWkt(geom_wkt)

            if geom and not geom.isEmpty():
                # Ensure geometry is MultiPolygon
                if not geom.isMultipart():
                    geom.convertToMultiType()

                feature.setGeometry(geom)
                area = geom.area()
                feature.setAttributes([i + 1, polygon_data['score'], area])
                features_to_add.append(feature)

        if not features_to_add:
            return

        pr.addFeatures(features_to_add)
        temp_layer.updateExtents()

        # Save to GeoPackage file
        from qgis.core import QgsVectorFileWriter

        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.fileEncoding = "UTF-8"

        error = QgsVectorFileWriter.writeAsVectorFormatV3(
            temp_layer,
            gpkg_path,
            QgsProject.instance().transformContext(),
            options
        )

        if error[0] != QgsVectorFileWriter.NoError:
            QgsMessageLog.logMessage(
                f"Failed to save GeoPackage: {error[1]}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Export Failed",
                f"Could not save layer to file:\n{error[1]}"
            )
            return

        # Load the saved GeoPackage as a permanent layer
        result_layer = QgsVectorLayer(gpkg_path, layer_name, "ogr")
        if not result_layer.isValid():
            QgsMessageLog.logMessage(
                f"Failed to load saved GeoPackage: {gpkg_path}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Load Failed",
                f"Layer was saved but could not be loaded:\n{gpkg_path}"
            )
            return

        # Add to project
        QgsProject.instance().addMapLayer(result_layer)

        # Log the layer extent for debugging
        layer_extent = result_layer.extent()
        QgsMessageLog.logMessage(
            f"Exported layer extent: xmin={layer_extent.xMinimum():.2f}, ymin={layer_extent.yMinimum():.2f}, "
            f"xmax={layer_extent.xMaximum():.2f}, ymax={layer_extent.yMaximum():.2f}",
            "AI Segmentation",
            level=Qgis.Info
        )
        QgsMessageLog.logMessage(
            f"Layer CRS: {result_layer.crs().authid()}",
            "AI Segmentation",
            level=Qgis.Info
        )
        QgsMessageLog.logMessage(
            f"Saved to: {gpkg_path}",
            "AI Segmentation",
            level=Qgis.Info
        )

        # Set renderer - red thin outline, transparent fill
        symbol = QgsFillSymbol.createSimple({
            'color': '0,0,0,0',
            'outline_color': '220,0,0,255',
            'outline_width': '0.5'
        })
        renderer = QgsSingleSymbolRenderer(symbol)
        result_layer.setRenderer(renderer)
        result_layer.triggerRepaint()
        self.iface.mapCanvas().refresh()

        QgsMessageLog.logMessage(
            f"Created segmentation layer: {layer_name} with {len(features_to_add)} polygons",
            "AI Segmentation",
            level=Qgis.Info
        )

        self._reset_session()
        self.dock_widget.reset_session()

    def _on_tool_deactivated(self):
        if self.dock_widget:
            self.dock_widget.set_segmentation_active(False)
        # Only clear the previous tool if the user manually switched tools
        # (not when we're stopping programmatically via Stop/Export)
        if not self._stopping_segmentation:
            self._previous_map_tool = None

    def _restore_previous_map_tool(self):
        """Restore the map tool that was active before segmentation started."""
        if self._previous_map_tool:
            try:
                self.iface.mapCanvas().setMapTool(self._previous_map_tool)
            except RuntimeError:
                # The previous tool may have been deleted
                pass
        self._previous_map_tool = None

    def _on_stop_segmentation(self):
        """Exit segmentation mode without saving."""
        # Count polygons that will be lost
        polygon_count = len(self.saved_polygons)
        if self.current_mask is not None:
            polygon_count += 1  # Include current unsaved polygon

        # Show warning if there are polygons
        if polygon_count > 0:
            reply = QMessageBox.warning(
                self.iface.mainWindow(),
                "Stop Segmentation?",
                f"This will discard {polygon_count} mask(s).\n\n"
                "Use 'Export to layer' to keep them.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        self._stopping_segmentation = True
        self.iface.mapCanvas().unsetMapTool(self.map_tool)
        self._restore_previous_map_tool()
        self._stopping_segmentation = False
        self._reset_session()
        self.dock_widget.reset_session()

    def _on_refine_settings_changed(self, expand: int, simplify: int):
        """Handle refinement control changes - affects last saved mask only."""
        QgsMessageLog.logMessage(
            f"Refine settings changed: expand={expand}, simplify={simplify}",
            "AI Segmentation",
            level=Qgis.Info
        )
        self._refine_expand = expand
        self._refine_simplify = simplify

        # Update the last saved mask's rubber band
        self._update_last_saved_mask_visualization()

    def _update_last_saved_mask_visualization(self):
        """Update the last saved mask's rubber band with current refinement settings."""
        QgsMessageLog.logMessage(
            f"_update_last_saved_mask_visualization called. saved_polygons={len(self.saved_polygons)}, saved_rubber_bands={len(self.saved_rubber_bands)}",
            "AI Segmentation",
            level=Qgis.Info
        )

        if not self.saved_polygons or not self.saved_rubber_bands:
            QgsMessageLog.logMessage("No saved polygons or rubber bands", "AI Segmentation", level=Qgis.Warning)
            return

        last_polygon = self.saved_polygons[-1]
        last_rubber_band = self.saved_rubber_bands[-1]

        raw_mask = last_polygon.get('raw_mask')
        transform_info = last_polygon.get('transform_info')

        QgsMessageLog.logMessage(
            f"raw_mask is None: {raw_mask is None}, transform_info is None: {transform_info is None}",
            "AI Segmentation",
            level=Qgis.Info
        )

        if raw_mask is None or transform_info is None:
            QgsMessageLog.logMessage("raw_mask or transform_info is None, returning", "AI Segmentation", level=Qgis.Warning)
            return

        try:
            from ..core.polygon_exporter import mask_to_polygons, apply_mask_refinement

            QgsMessageLog.logMessage(
                f"Applying refinement: expand={self._refine_expand}, simplify={self._refine_simplify}",
                "AI Segmentation",
                level=Qgis.Info
            )

            # Apply mask refinement (expand/contract only)
            mask_to_display = raw_mask
            if self._refine_expand != 0:
                QgsMessageLog.logMessage("Calling apply_mask_refinement...", "AI Segmentation", level=Qgis.Info)
                mask_to_display = apply_mask_refinement(raw_mask, self._refine_expand)
                QgsMessageLog.logMessage(f"Refinement done, mask shape: {mask_to_display.shape}", "AI Segmentation", level=Qgis.Info)

            geometries = mask_to_polygons(mask_to_display, transform_info)
            QgsMessageLog.logMessage(f"Generated {len(geometries)} geometries", "AI Segmentation", level=Qgis.Info)

            if geometries:
                combined = QgsGeometry.unaryUnion(geometries)
                if combined and not combined.isEmpty():
                    # Apply simplification if enabled
                    if self._refine_simplify > 0:
                        # Calculate tolerance based on image resolution
                        bbox = transform_info.get("bbox", [0, 1, 0, 1])
                        img_shape = transform_info.get("img_shape", (1024, 1024))
                        pixel_size = (bbox[1] - bbox[0]) / img_shape[1]  # map units per pixel
                        tolerance = pixel_size * self._refine_simplify * 0.5  # 0.5-5 pixels worth
                        QgsMessageLog.logMessage(f"Simplifying with tolerance={tolerance:.4f}", "AI Segmentation", level=Qgis.Info)
                        combined = combined.simplify(tolerance)

                    last_rubber_band.setToGeometry(combined, None)
                    # Also update the stored WKT for export
                    last_polygon['geometry_wkt'] = combined.asWkt()
                    QgsMessageLog.logMessage("Updated rubber band geometry", "AI Segmentation", level=Qgis.Info)
                else:
                    last_rubber_band.reset(QgsWkbTypes.PolygonGeometry)
                    QgsMessageLog.logMessage("Combined geometry was empty", "AI Segmentation", level=Qgis.Warning)
            else:
                last_rubber_band.reset(QgsWkbTypes.PolygonGeometry)
                QgsMessageLog.logMessage("No geometries generated", "AI Segmentation", level=Qgis.Warning)

        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Failed to update last saved mask: {str(e)}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.Warning
            )

    def _on_positive_click(self, point):
        """Handle left-click: add positive point (select this element)."""
        if self.predictor is None or self.feature_dataset is None:
            # Remove the marker that was added by the maptool
            if self.map_tool:
                self.map_tool.remove_last_marker()
            return

        QgsMessageLog.logMessage(
            f"POSITIVE POINT at ({point.x():.6f}, {point.y():.6f})",
            "AI Segmentation",
            level=Qgis.Info
        )

        self.prompts.add_positive_point(point.x(), point.y())
        self._run_prediction()

    def _on_negative_click(self, point):
        """Handle right-click: add negative point (exclude this area)."""
        if self.predictor is None or self.feature_dataset is None:
            # Remove the marker that was added by the maptool
            if self.map_tool:
                self.map_tool.remove_last_marker()
            return

        # Block negative points until at least one positive point exists
        if len(self.prompts.positive_points) == 0:
            # Remove the marker that was added by the maptool
            if self.map_tool:
                self.map_tool.remove_last_marker()
            QgsMessageLog.logMessage(
                "Negative point ignored - need at least one positive point first",
                "AI Segmentation",
                level=Qgis.Info
            )
            return

        QgsMessageLog.logMessage(
            f"NEGATIVE POINT at ({point.x():.6f}, {point.y():.6f})",
            "AI Segmentation",
            level=Qgis.Info
        )

        self.prompts.add_negative_point(point.x(), point.y())
        self._run_prediction()

    def _run_prediction(self):
        """Run SAM prediction using all positive and negative points."""
        import numpy as np
        from rasterio.transform import from_bounds as transform_from_bounds
        from ..core.feature_dataset import FeatureSampler

        all_points = self.prompts.positive_points + self.prompts.negative_points
        if not all_points:
            return

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        bounds = self.feature_dataset.bounds

        roi = (
            min(xs), max(xs),
            min(ys), max(ys),
            bounds[4], bounds[5]
        )

        sampler = FeatureSampler(self.feature_dataset, roi)
        if len(sampler) == 0:
            QgsMessageLog.logMessage(
                "No feature found for click location",
                "AI Segmentation",
                level=Qgis.Warning
            )
            return

        for query in sampler:
            sample = self.feature_dataset[query]
            break

        bbox = sample["bbox"]
        features = sample["image"]

        img_size = self.predictor.model.image_encoder.img_size
        img_height = img_width = img_size
        input_height = input_width = img_size

        if "img_shape" in sample:
            img_height = sample["img_shape"][0]
            img_width = sample["img_shape"][1]
            input_height = sample["input_shape"][0]
            input_width = sample["input_shape"][1]

        if hasattr(features, 'cpu'):
            features_np = features.cpu().numpy()
        else:
            features_np = features.numpy() if hasattr(features, 'numpy') else features

        self.predictor.set_image_feature(
            img_features=features_np,
            img_size=(img_height, img_width),
            input_size=(input_height, input_width),
        )

        minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
        img_clip_transform = transform_from_bounds(minx, miny, maxx, maxy, img_width, img_height)

        point_coords, point_labels = self.prompts.get_points_for_predictor(img_clip_transform)

        if point_coords is None:
            return

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

        self.current_mask = masks[0]
        self.current_score = float(scores[0])

        # Use layer CRS as fallback if feature_dataset.crs is None or empty
        crs_value = self.feature_dataset.crs
        if not crs_value or (isinstance(crs_value, str) and not crs_value.strip()):
            if self._current_layer and self._current_layer.crs().isValid():
                crs_value = self._current_layer.crs().authid()
                QgsMessageLog.logMessage(
                    f"Using layer CRS as fallback: {crs_value}",
                    "AI Segmentation",
                    level=Qgis.Info
                )

        self.current_transform_info = {
            "bbox": bbox,
            "img_shape": (img_height, img_width),
            "crs": crs_value,
        }

        self._update_ui_after_prediction()

    def _update_ui_after_prediction(self):
        pos_count, neg_count = self.prompts.point_count
        self.dock_widget.set_point_count(pos_count, neg_count)

        if self.current_mask is not None:
            mask_pixels = int(self.current_mask.sum())
            QgsMessageLog.logMessage(
                f"Segmentation result: score={self.current_score:.3f}, mask_pixels={mask_pixels}",
                "AI Segmentation",
                level=Qgis.Info
            )
            self._update_mask_visualization()
        else:
            self._clear_mask_visualization()

    def _update_mask_visualization(self):
        if self.mask_rubber_band is None:
            return

        if self.current_mask is None:
            self._clear_mask_visualization()
            return

        try:
            from ..core.polygon_exporter import mask_to_polygons

            # No refinement on current preview - refinement only applies to saved masks
            geometries = mask_to_polygons(self.current_mask, self.current_transform_info)

            if geometries:
                combined = QgsGeometry.unaryUnion(geometries)
                if combined and not combined.isEmpty():
                    self.mask_rubber_band.setToGeometry(combined, None)
                else:
                    self._clear_mask_visualization()
            else:
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
        if self.mask_rubber_band:
            self.mask_rubber_band.reset(QgsWkbTypes.PolygonGeometry)

    def _on_clear_points(self):
        self.prompts.clear()

        if self.map_tool:
            self.map_tool.clear_markers()

        self._clear_mask_visualization()

        self.current_mask = None
        self.current_score = 0.0
        self.dock_widget.set_point_count(0, 0)

    def _on_undo(self):
        """Undo last point added."""
        result = self.prompts.undo()
        if result is None:
            return

        if self.map_tool:
            self.map_tool.remove_last_marker()

        # Re-run prediction with remaining points
        if self.prompts.point_count[0] + self.prompts.point_count[1] > 0:
            self._run_prediction()
        else:
            self.current_mask = None
            self.current_score = 0.0
            self._clear_mask_visualization()
            self.dock_widget.set_point_count(0, 0)

    def _reset_session(self):
        self.prompts.clear()
        self.saved_polygons = []

        for rb in self.saved_rubber_bands:
            self.iface.mapCanvas().scene().removeItem(rb)
        self.saved_rubber_bands = []

        if self.map_tool:
            self.map_tool.clear_markers()

        self._clear_mask_visualization()

        self.current_mask = None
        self.current_score = 0.0
        self.current_transform_info = None

        # Reset refinement settings
        self._refine_expand = 0
        self._refine_simplify = 0

        self.dock_widget.set_point_count(0, 0)
        self.dock_widget.set_saved_polygon_count(0)
