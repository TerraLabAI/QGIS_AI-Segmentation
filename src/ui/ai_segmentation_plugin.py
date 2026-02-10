import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy

from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal, QVariant, QSettings
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
    QgsCoordinateReferenceSystem,
    QgsPointXY,
)
from qgis.gui import QgisInterface, QgsRubberBand
from qgis.PyQt.QtGui import QColor

from .ai_segmentation_dockwidget import AISegmentationDockWidget
from .ai_segmentation_maptool import AISegmentationMapTool
from .error_report_dialog import show_error_report, start_log_collector
from ..core.i18n import tr

# QSettings keys for tutorial flags
SETTINGS_KEY_TUTORIAL_SIMPLE = "AI_Segmentation/tutorial_simple_shown"


class DepsInstallWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, cuda_enabled: bool = False, parent=None):
        super().__init__(parent)
        self._cancelled = False
        self._cuda_enabled = cuda_enabled

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from ..core.venv_manager import create_venv_and_install
            success, message = create_venv_and_install(
                progress_callback=lambda percent, msg: self.progress.emit(percent, msg),
                cancel_check=lambda: self._cancelled,
                cuda_enabled=self._cuda_enabled
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
        if label == "positive":
            # Remove last occurrence (not first) to correctly undo duplicate coords
            for i in range(len(self.positive_points) - 1, -1, -1):
                if self.positive_points[i] == point:
                    self.positive_points.pop(i)
                    break
        elif label == "negative":
            for i in range(len(self.negative_points) - 1, -1, -1):
                if self.negative_points[i] == point:
                    self.negative_points.pop(i)
                    break

        return label, point

    def clear(self):
        self.positive_points = []
        self.negative_points = []
        self._history = []

    @property
    def point_count(self) -> Tuple[int, int]:
        return len(self.positive_points), len(self.negative_points)

    def get_points_for_predictor(self, transform) -> Tuple[Optional['numpy.ndarray'], Optional['numpy.ndarray']]:
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
        self.current_low_res_mask = None  # For iterative refinement with negative points
        self.saved_polygons = []

        self._initialized = False
        self._current_layer = None
        self._current_layer_name = ""

        # Refinement settings
        self._refine_expand = 0
        self._refine_simplify = 4  # Default: matches dockwidget spinbox
        self._refine_fill_holes = False  # Default: matches dockwidget checkbox
        self._refine_min_area = 100  # Default: matches dockwidget spinbox

        # Per-raster mask counters (used for both Simple and Batch modes)
        self._mask_counters = {}  # {raster_name: counter}

        # Mode flag (simple mode by default)
        self._batch_mode = False
        self._batch_tutorial_shown_this_session = False  # Reset each QGIS launch

        self.deps_install_worker = None
        self.download_worker = None
        self.encoding_worker = None

        self.mask_rubber_band: Optional[QgsRubberBand] = None
        self.saved_rubber_bands: List[QgsRubberBand] = []

        self._previous_map_tool = None  # Store the tool active before segmentation
        self._stopping_segmentation = False  # Flag to track if we're stopping programmatically

    @staticmethod
    def _safe_remove_rubber_band(rb):
        """Remove a rubber band from the canvas scene, handling C++ deletion."""
        if rb is None:
            return
        try:
            # QgsRubberBand doesn't expose parentWidget; use scene directly
            scene = rb.scene()
            if scene is not None:
                scene.removeItem(rb)
        except RuntimeError:
            pass  # C++ object already deleted (QGIS shutdown)

    def _is_layer_valid(self, layer=None) -> bool:
        """Check if a layer's C++ object is still alive."""
        if layer is None:
            layer = self._current_layer
        if layer is None:
            return False
        try:
            layer.id()
            return True
        except RuntimeError:
            return False

    def _ensure_polygon_rubberband_sync(self):
        """Recover from polygon/rubber band list mismatch by truncating to shorter."""
        n_polygons = len(self.saved_polygons)
        n_bands = len(self.saved_rubber_bands)
        if n_polygons == n_bands:
            return
        QgsMessageLog.logMessage(
            "Polygon/rubber band mismatch: {} polygons vs {} bands, syncing".format(
                n_polygons, n_bands),
            "AI Segmentation",
            level=Qgis.Warning
        )
        target = min(n_polygons, n_bands)
        # Remove orphaned rubber bands
        while len(self.saved_rubber_bands) > target:
            rb = self.saved_rubber_bands.pop()
            self._safe_remove_rubber_band(rb)
        # Trim polygons
        self.saved_polygons = self.saved_polygons[:target]
        if self.dock_widget:
            self.dock_widget.set_saved_polygon_count(target)

    @staticmethod
    def _compute_simplification_tolerance(transform_info, simplify_value):
        """Compute simplification tolerance from transform_info and slider value.

        Returns 0 if inputs are invalid or simplify_value is 0.
        """
        if simplify_value <= 0 or transform_info is None:
            return 0
        bbox = transform_info.get("bbox", [0, 1, 0, 1])
        img_shape = transform_info.get("img_shape", (1024, 1024))
        width_pixels = max(img_shape[1], 1)
        bbox_width = bbox[1] - bbox[0]
        if bbox_width == 0:
            return 0
        pixel_size = bbox_width / width_pixels
        return pixel_size * simplify_value * 0.5

    def initGui(self):
        start_log_collector()

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
            "AI Segmentation by TerraLab\n{}".format(
                tr("Segment objects on raster images using AI"))
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
        self.dock_widget.batch_mode_changed.connect(self._on_batch_mode_changed)
        self.dock_widget.layer_combo.layerChanged.connect(self._on_layer_combo_changed)

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
        # 1. Disconnect ALL signals FIRST to prevent callbacks on partially-cleaned state
        try:
            QgsProject.instance().layersAdded.disconnect(self.dock_widget._on_layers_added)
        except (TypeError, RuntimeError, AttributeError):
            pass
        try:
            QgsProject.instance().layersRemoved.disconnect(self.dock_widget._on_layers_removed)
        except (TypeError, RuntimeError, AttributeError):
            pass
        try:
            if self.dock_widget:
                self.dock_widget.layer_combo.layerChanged.disconnect(self._on_layer_combo_changed)
        except (TypeError, RuntimeError, AttributeError):
            pass
        try:
            if self.map_tool:
                self.map_tool.positive_click.disconnect(self._on_positive_click)
                self.map_tool.negative_click.disconnect(self._on_negative_click)
                self.map_tool.tool_deactivated.disconnect(self._on_tool_deactivated)
                self.map_tool.undo_requested.disconnect(self._on_undo)
                self.map_tool.save_polygon_requested.disconnect(self._on_save_polygon)
                self.map_tool.export_layer_requested.disconnect(self._on_export_layer)
                self.map_tool.clear_requested.disconnect(self._on_clear_points)
        except (TypeError, RuntimeError, AttributeError):
            pass

        # 2. Cleanup predictor subprocess
        if self.predictor:
            try:
                self.predictor.cleanup()
            except Exception:
                pass
            self.predictor = None

        # 3. Terminate workers with timeout
        for worker in [self.deps_install_worker, self.download_worker, self.encoding_worker]:
            if worker and worker.isRunning():
                try:
                    worker.terminate()
                    if not worker.wait(5000):
                        QgsMessageLog.logMessage(
                            "Worker did not terminate within 5s",
                            "AI Segmentation",
                            level=Qgis.Warning
                        )
                except RuntimeError:
                    pass

        # 4. Remove menu/toolbar
        plugins_menu = self.iface.pluginMenu()
        plugins_menu.removeAction(self.action)
        self.iface.removeToolBarIcon(self.action)

        # 5. Remove dock widget
        if self.dock_widget:
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget.deleteLater()
            self.dock_widget = None

        # 6. Unset map tool
        if self.map_tool:
            try:
                if self.iface.mapCanvas().mapTool() == self.map_tool:
                    self.iface.mapCanvas().unsetMapTool(self.map_tool)
            except RuntimeError:
                pass
            self.map_tool = None

        # 7. Remove rubber bands safely
        self._safe_remove_rubber_band(self.mask_rubber_band)
        self.mask_rubber_band = None

        for rb in self.saved_rubber_bands:
            self._safe_remove_rubber_band(rb)
        self.saved_rubber_bands = []

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
                self._show_device_info()
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
                self._show_activation_popup_if_needed()
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
                "SAM predictor initialized (subprocess mode)",
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

    def _show_device_info(self):
        """Detect and display which compute device will be used."""
        try:
            from ..core.venv_manager import ensure_venv_packages_available
            ensure_venv_packages_available()

            from ..core.device_manager import get_device_info
            info = get_device_info()
            self.dock_widget.set_device_info(info)
            QgsMessageLog.logMessage(
                f"Device info: {info}",
                "AI Segmentation",
                level=Qgis.Info
            )
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Could not determine device info: {e}",
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

        cuda_enabled = self.dock_widget.get_cuda_enabled()
        self.deps_install_worker = DepsInstallWorker(cuda_enabled=cuda_enabled)
        self.deps_install_worker.progress.connect(self._on_deps_install_progress)
        self.deps_install_worker.finished.connect(self._on_deps_install_finished)
        self.deps_install_worker.start()

    def _show_activation_popup_if_needed(self):
        """Show activation popup if not already activated (after deps+model ready)."""
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
                self.dock_widget.set_dependency_status(True, "✓ " + tr("Virtual environment ready"))
                self._show_device_info()
                self._verify_venv()
                self._check_checkpoint()
            else:
                self.dock_widget.set_dependency_status(False, tr("Verification failed:") + f" {verify_msg}")

                show_error_report(
                    self.iface.mainWindow(),
                    tr("Verification Failed"),
                    "{}\n{}".format(
                        tr("Virtual environment was created but verification failed:"),
                        verify_msg)
                )
        else:
            error_msg = message[:300] if message else tr("Unknown error")
            self.dock_widget.set_dependency_status(False, tr("Installation failed"))

            # Determine specific error title based on failure type
            error_title = tr("Installation Failed")
            msg_lower = message.lower() if message else ""
            if any(p in msg_lower for p in [
                "ssl", "certificate verify", "sslerror",
                "unable to get local issuer",
            ]):
                error_title = tr("SSL Certificate Error")
            elif any(p in msg_lower for p in [
                "access is denied", "winerror 5", "winerror 225",
                "permission denied", "blocked",
            ]):
                error_title = tr("Installation Blocked")

            show_error_report(
                self.iface.mainWindow(),
                error_title,
                error_msg
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
            self._show_activation_popup_if_needed()
        else:
            self.dock_widget.set_download_progress(0, "")

            show_error_report(
                self.iface.mainWindow(),
                tr("Download Failed"),
                "{}\n{}".format(
                    tr("Failed to download model:"),
                    message)
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
                tr("Not Ready"),
                tr("Please wait for the SAM model to load.")
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
                # Actual error - show error report dialog
                show_error_report(
                    self.iface.mainWindow(),
                    tr("Encoding Failed"),
                    "{}\n{}".format(tr("Failed to encode raster:"), message)
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
            show_error_report(
                self.iface.mainWindow(),
                tr("Load Failed"),
                "{}\n{}".format(tr("Failed to load feature data:"), str(e))
            )

    def _activate_segmentation_tool(self):
        # Save the current map tool to restore it later
        current_tool = self.iface.mapCanvas().mapTool()
        if current_tool and current_tool != self.map_tool:
            self._previous_map_tool = current_tool

        self.iface.mapCanvas().setMapTool(self.map_tool)
        self.dock_widget.set_segmentation_active(True)
        # Status bar hint will be set by _update_status_hint via set_point_count

        # Show tutorial notification for first-time users (simple mode)
        if not self._batch_mode:
            self._show_tutorial_notification("simple")

    def _get_next_mask_counter(self) -> int:
        """Increment and return the mask counter for the current raster."""
        name = self._current_layer_name
        if name not in self._mask_counters:
            self._mask_counters[name] = 0
        self._mask_counters[name] += 1
        return self._mask_counters[name]

    def _show_tutorial_notification(self, mode: str):
        """Show YouTube tutorial notification."""
        # TODO: Replace with actual tutorial URLs when available
        tutorial_url = "https://www.youtube.com/playlist?list=PL4hCF043nAUW2iIxALNUzy1fKHcCWwDsv&jct=GTA3Fx8pJzuTLPPivC9RRQ"

        if mode == "batch":
            # Batch mode: show once per QGIS session (first activation)
            if self._batch_tutorial_shown_this_session:
                return
            self._batch_tutorial_shown_this_session = True
            message = tr('Batch mode activated.') + f' <a href="{tutorial_url}">' + tr('Watch the tutorial') + '</a> ' + tr('to learn how to use it.')
        else:
            # Simple mode: show only once ever (persisted in QSettings)
            settings = QSettings()
            if settings.value(SETTINGS_KEY_TUTORIAL_SIMPLE, False, type=bool):
                return
            settings.setValue(SETTINGS_KEY_TUTORIAL_SIMPLE, True)
            message = tr('New to AI Segmentation?') + f' <a href="{tutorial_url}">' + tr('Watch our tutorial') + '</a>'

        self.iface.messageBar().pushMessage(
            "AI Segmentation",
            message,
            level=Qgis.Info,
            duration=10
        )

    def _on_batch_mode_changed(self, batch: bool):
        """Handle batch mode toggle from the dock widget."""
        # Check if we have unsaved masks when switching from batch to simple
        if not batch and len(self.saved_polygons) > 0:
            reply = QMessageBox.question(
                self.dock_widget,
                tr("Unsaved Polygons"),
                "{}\n{}".format(
                    tr("You have {count} saved polygon(s).").format(count=len(self.saved_polygons)),
                    tr("Export before changing mode?")),
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Cancel
            )

            if reply == QMessageBox.Yes:
                self._on_export_layer()
                # Continue to set the mode after exporting
            elif reply == QMessageBox.Cancel:
                # Revert the toggle
                self.dock_widget.set_batch_mode(True)
                return
            # If No, continue and lose the masks

        self._batch_mode = batch

        # Show tutorial for batch mode on first activation
        if batch:
            self._show_tutorial_notification("batch")

    def _on_layer_combo_changed(self, layer):
        """Handle layer selection change in the combo box."""
        # Only care if we're currently in segmentation mode
        if not self._current_layer:
            return

        # Check if it's actually a different layer
        # Handle case where the C++ layer object was deleted
        try:
            new_layer_id = layer.id() if layer else None
            current_layer_id = self._current_layer.id() if self._current_layer else None
        except RuntimeError:
            # Layer was deleted, reset our reference
            self._current_layer = None
            return

        if new_layer_id == current_layer_id:
            # Same layer selected, do nothing
            return

        # Different layer selected while segmenting - stop segmentation silently
        if self.iface.mapCanvas().mapTool() == self.map_tool:
            self._stopping_segmentation = True
            self.iface.mapCanvas().unsetMapTool(self.map_tool)
            self._restore_previous_map_tool()
            self._stopping_segmentation = False
            self._reset_session()
            self.dock_widget.reset_session()

    def _confirm_exit_segmentation(self) -> bool:
        """Show confirmation dialog if user has placed points."""
        has_points = (self.prompts.positive_points or self.prompts.negative_points)

        if not has_points:
            return True  # No confirmation needed

        reply = QMessageBox.question(
            self.dock_widget,
            tr("Exit Segmentation"),
            tr("Exit segmentation?") + "\n" + tr("The current selection will be lost."),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        return reply == QMessageBox.Yes

    def _on_save_polygon(self):
        """Save current mask as polygon. Only works in Batch mode."""
        # S shortcut only works in Batch mode
        if not self._batch_mode:
            return

        if self.current_mask is None or self.current_transform_info is None:
            return

        self._ensure_polygon_rubberband_sync()

        from ..core.polygon_exporter import mask_to_polygons, apply_mask_refinement

        # Apply all mask-level refinements for display (green shows with effects)
        mask_for_display = self.current_mask
        if self._refine_fill_holes or self._refine_min_area > 0 or self._refine_expand != 0:
            mask_for_display = apply_mask_refinement(
                self.current_mask,
                expand_value=self._refine_expand,
                fill_holes=self._refine_fill_holes,
                min_area=self._refine_min_area
            )

        geometries = mask_to_polygons(mask_for_display, self.current_transform_info)

        if not geometries:
            return

        combined = QgsGeometry.unaryUnion(geometries)
        if combined and not combined.isEmpty():
            # Apply simplification if enabled
            tolerance = self._compute_simplification_tolerance(
                self.current_transform_info, self._refine_simplify
            )
            if tolerance > 0:
                combined = combined.simplify(tolerance)

            # Store WKT (with effects), score, transform info, raw mask, points, and refine settings
            self.saved_polygons.append({
                'geometry_wkt': combined.asWkt(),
                'score': self.current_score,
                'transform_info': self.current_transform_info.copy() if self.current_transform_info else None,
                'raw_mask': self.current_mask.copy(),  # Store RAW mask for re-applying different settings
                'points_positive': list(self.prompts.positive_points),  # Points for undo restoration
                'points_negative': list(self.prompts.negative_points),  # Points for undo restoration
                'refine_expand': self._refine_expand,  # Refine settings at save time
                'refine_simplify': self._refine_simplify,  # Refine settings at save time
                'refine_fill_holes': self._refine_fill_holes,  # Refine settings at save time
                'refine_min_area': self._refine_min_area,  # Refine settings at save time
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

            # Note: We keep refinement settings in batch mode so the user can
            # apply the same expand/simplify to multiple masks

        # Clear current state for next polygon
        self.prompts.clear()
        if self.map_tool:
            self.map_tool.clear_markers()
        self._clear_mask_visualization()
        self.current_mask = None
        self.current_score = 0.0
        self.current_low_res_mask = None
        self.dock_widget.set_point_count(0, 0)

    def _on_export_layer(self):
        """Export masks to a new layer. Behavior depends on mode."""
        from ..core.polygon_exporter import mask_to_polygons, apply_mask_refinement

        self._ensure_polygon_rubberband_sync()

        if self._batch_mode:
            # Batch mode: export all saved polygons + current unsaved mask
            if not self.saved_polygons and self.current_mask is None:
                return  # Nothing to export

            polygons_to_export = list(self.saved_polygons)

            # Include current unsaved mask if exists (apply current refine settings)
            if self.current_mask is not None and self.current_transform_info is not None:
                mask_to_export = self.current_mask
                if self._refine_fill_holes or self._refine_min_area > 0 or self._refine_expand != 0:
                    mask_to_export = apply_mask_refinement(
                        self.current_mask,
                        expand_value=self._refine_expand,
                        fill_holes=self._refine_fill_holes,
                        min_area=self._refine_min_area
                    )
                geometries = mask_to_polygons(mask_to_export, self.current_transform_info)
                if geometries:
                    combined = QgsGeometry.unaryUnion(geometries)
                    if combined and not combined.isEmpty():
                        tolerance = self._compute_simplification_tolerance(
                            self.current_transform_info, self._refine_simplify
                        )
                        if tolerance > 0:
                            combined = combined.simplify(tolerance)
                        polygons_to_export.append({
                            'geometry_wkt': combined.asWkt(),
                            'score': self.current_score,
                            'transform_info': self.current_transform_info.copy() if self.current_transform_info else None,
                        })
        else:
            # Simple mode: export only current mask with refinement applied
            if self.current_mask is None or self.current_transform_info is None:
                return  # Nothing to export

            # Apply all mask-level refinement settings
            mask_to_export = self.current_mask
            if self._refine_fill_holes or self._refine_min_area > 0 or self._refine_expand != 0:
                mask_to_export = apply_mask_refinement(
                    self.current_mask,
                    expand_value=self._refine_expand,
                    fill_holes=self._refine_fill_holes,
                    min_area=self._refine_min_area
                )

            geometries = mask_to_polygons(mask_to_export, self.current_transform_info)
            if not geometries:
                return

            combined = QgsGeometry.unaryUnion(geometries)
            if not combined or combined.isEmpty():
                return

            # Apply simplification if enabled
            tolerance = self._compute_simplification_tolerance(
                self.current_transform_info, self._refine_simplify
            )
            if tolerance > 0:
                combined = combined.simplify(tolerance)

            polygons_to_export = [{
                'geometry_wkt': combined.asWkt(),
                'score': self.current_score,
                'transform_info': self.current_transform_info.copy() if self.current_transform_info else None,
            }]

        self._stopping_segmentation = True
        self.iface.mapCanvas().unsetMapTool(self.map_tool)
        self._restore_previous_map_tool()
        self._stopping_segmentation = False

        # Generate layer name: {RasterName}_mask_{number} (same for both modes)
        mask_num = self._get_next_mask_counter()
        layer_name = f"{self._current_layer_name}_mask_{mask_num}"

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
            show_error_report(
                self.iface.mainWindow(),
                tr("Layer Creation Failed"),
                tr("Could not create the output layer.")
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
                    f"Polygon {i + 1} has no WKT data",
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
            show_error_report(
                self.iface.mainWindow(),
                tr("Export Failed"),
                "{}\n{}".format(tr("Could not save layer to file:"), error[1])
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
            show_error_report(
                self.iface.mainWindow(),
                tr("Load Failed"),
                "{}\n{}".format(tr("Layer was saved but could not be loaded:"), gpkg_path)
            )
            return

        # Add to project in a group for this raster
        self._add_layer_to_raster_group(result_layer)

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

    def _add_layer_to_raster_group(self, layer):
        """Add the exported layer to a group named after the source raster."""
        group_name = f"{self._current_layer_name} (AI Segmentation)"

        root = QgsProject.instance().layerTreeRoot()

        # Find or create the group
        group = root.findGroup(group_name)
        if group is None:
            # Create the group at the top of the layer tree
            group = root.insertGroup(0, group_name)

        # Add layer to project without adding to root
        QgsProject.instance().addMapLayer(layer, False)

        # Add layer to the group
        group.addLayer(layer)

    def _on_tool_deactivated(self):
        # If we're stopping programmatically (via Stop/Export), just update UI
        if self._stopping_segmentation:
            if self.dock_widget:
                self.dock_widget.set_segmentation_active(False)
            return

        # Check if QGIS is closing - don't show popup in that case
        # When QGIS closes, the main window is being destroyed or hidden
        main_window = self.iface.mainWindow()
        if main_window is None or not main_window.isVisible():
            # QGIS is closing, just cleanup silently
            self._reset_session()
            if self.dock_widget:
                self.dock_widget.set_segmentation_active(False)
            return

        # Check if there's unsaved work
        has_unsaved_mask = self.current_mask is not None
        has_saved_polygons = len(self.saved_polygons) > 0

        if has_unsaved_mask or has_saved_polygons:
            # Defer dialog out of signal handler to avoid re-entrant event loop
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._show_deactivation_dialog)
            return

        if self.dock_widget:
            self.dock_widget.set_segmentation_active(False)
        self._previous_map_tool = None

    def _show_deactivation_dialog(self):
        """Deferred dialog for tool deactivation (called outside signal handler)."""
        # Re-check state: it may have changed since the deferred call was scheduled
        main_window = self.iface.mainWindow()
        if main_window is None or not main_window.isVisible():
            self._reset_session()
            if self.dock_widget:
                self.dock_widget.set_segmentation_active(False)
            return

        has_unsaved_mask = self.current_mask is not None
        has_saved_polygons = len(self.saved_polygons) > 0

        if not has_unsaved_mask and not has_saved_polygons:
            # Nothing to warn about anymore
            if self.dock_widget:
                self.dock_widget.set_segmentation_active(False)
            self._previous_map_tool = None
            return

        # Show warning popup
        if self._batch_mode:
            polygon_count = len(self.saved_polygons)
            if has_unsaved_mask:
                polygon_count += 1
            message = tr("You have {count} unsaved polygon(s).").format(count=polygon_count) + "\n\n" + tr("Discard and exit segmentation?")
        else:
            message = tr("You have an unsaved selection.") + "\n\n" + tr("Discard and exit segmentation?")

        reply = QMessageBox.warning(
            main_window,
            tr("Exit Segmentation?"),
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            # User wants to continue - re-activate the segmentation tool
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, lambda: self.iface.mapCanvas().setMapTool(self.map_tool))
            return

        # User confirmed exit - reset session
        self._reset_session()
        if self.dock_widget:
            self.dock_widget.reset_session()

        if self.dock_widget:
            self.dock_widget.set_segmentation_active(False)
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
        if self._batch_mode:
            # Batch mode: warn about saved polygons
            polygon_count = len(self.saved_polygons)
            if self.current_mask is not None:
                polygon_count += 1

            if polygon_count > 0:
                reply = QMessageBox.warning(
                    self.iface.mainWindow(),
                    tr("Stop Segmentation?"),
                    "{}\n\n{}".format(
                        tr("This will discard {count} polygon(s).").format(count=polygon_count),
                        tr("Use 'Export to layer' to keep them.")),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
        else:
            # Simple mode: use the standard confirmation dialog
            if not self._confirm_exit_segmentation():
                return

        self._stopping_segmentation = True
        self.iface.mapCanvas().unsetMapTool(self.map_tool)
        self._restore_previous_map_tool()
        self._stopping_segmentation = False
        self._reset_session()
        self.dock_widget.reset_session()

    def _on_refine_settings_changed(self, expand: int, simplify: int, fill_holes: bool, min_area: int):
        """Handle refinement control changes."""
        QgsMessageLog.logMessage(
            f"Refine settings changed: expand={expand}, simplify={simplify}, fill_holes={fill_holes}, min_area={min_area}",
            "AI Segmentation",
            level=Qgis.Info
        )
        self._refine_expand = expand
        self._refine_simplify = simplify
        self._refine_fill_holes = fill_holes
        self._refine_min_area = min_area

        # In both modes: update current mask preview only
        # Saved masks (green) keep their own refine settings from when they were saved
        self._update_mask_visualization()

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
                    tolerance = self._compute_simplification_tolerance(
                        transform_info, self._refine_simplify
                    )
                    if tolerance > 0:
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

    def _update_all_saved_masks_visualization(self):
        """Update all saved masks' rubber bands with current refinement settings."""
        if not self.saved_polygons or not self.saved_rubber_bands:
            return

        self._ensure_polygon_rubberband_sync()

        if not self.saved_polygons:
            return

        try:
            from ..core.polygon_exporter import mask_to_polygons, apply_mask_refinement

            for i, (polygon_data, rubber_band) in enumerate(zip(self.saved_polygons, self.saved_rubber_bands)):
                raw_mask = polygon_data.get('raw_mask')
                transform_info = polygon_data.get('transform_info')

                if raw_mask is None or transform_info is None:
                    continue

                # Apply mask refinement (expand/contract)
                mask_to_display = raw_mask
                if self._refine_expand != 0:
                    mask_to_display = apply_mask_refinement(raw_mask, self._refine_expand)

                geometries = mask_to_polygons(mask_to_display, transform_info)

                if geometries:
                    combined = QgsGeometry.unaryUnion(geometries)
                    if combined and not combined.isEmpty():
                        tolerance = self._compute_simplification_tolerance(
                            transform_info, self._refine_simplify
                        )
                        if tolerance > 0:
                            combined = combined.simplify(tolerance)

                        rubber_band.setToGeometry(combined, None)
                        # Update stored WKT for export
                        polygon_data['geometry_wkt'] = combined.asWkt()
                    else:
                        rubber_band.reset(QgsWkbTypes.PolygonGeometry)
                else:
                    rubber_band.reset(QgsWkbTypes.PolygonGeometry)

            QgsMessageLog.logMessage(
                f"Updated {len(self.saved_polygons)} saved masks with refinement settings",
                "AI Segmentation",
                level=Qgis.Info
            )

        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Failed to update saved masks: {str(e)}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.Warning
            )

    def _is_point_in_encoded_area(self, point):
        """Check if a point falls within the encoded raster extent."""
        bounds = self.feature_dataset.bounds
        # bounds = (minx, maxx, miny, maxy, mint, maxt)
        in_x = bounds[0] <= point.x() <= bounds[1]
        in_y = bounds[2] <= point.y() <= bounds[3]
        return in_x and in_y

    def _on_positive_click(self, point):
        """Handle left-click: add positive point (select this element)."""
        if self.predictor is None or self.feature_dataset is None:
            # Remove the marker that was added by the maptool
            if self.map_tool:
                self.map_tool.remove_last_marker()
            return

        if not self._is_point_in_encoded_area(point):
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Point is outside the encoded image. Click inside the raster."),
                level=Qgis.Warning,
                duration=4
            )
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

        if not self._is_point_in_encoded_area(point):
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Point is outside the encoded image. Click inside the raster."),
                level=Qgis.Warning,
                duration=4
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

        # Use previous low_res_mask for iterative refinement when we have negative points
        # This helps SAM understand the context and improves negative point behavior
        mask_input = None
        if len(self.prompts.negative_points) > 0 and self.current_low_res_mask is not None:
            mask_input = self.current_low_res_mask

        masks, scores, low_res_masks = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            multimask_output=False,
        )

        self.current_mask = masks[0]
        self.current_score = float(scores[0])
        self.current_low_res_mask = low_res_masks  # Store for next refinement

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

        if self.current_mask is None or self.current_transform_info is None:
            self._clear_mask_visualization()
            return

        try:
            from ..core.polygon_exporter import mask_to_polygons, apply_mask_refinement

            # Apply refinement to preview in both modes (refine affects current mask only)
            mask_to_display = self.current_mask
            # Apply all mask-level refinements (fill holes, min area, expand/contract)
            if self._refine_fill_holes or self._refine_min_area > 0 or self._refine_expand != 0:
                mask_to_display = apply_mask_refinement(
                    self.current_mask,
                    expand_value=self._refine_expand,
                    fill_holes=self._refine_fill_holes,
                    min_area=self._refine_min_area
                )

            geometries = mask_to_polygons(mask_to_display, self.current_transform_info)

            if geometries:
                combined = QgsGeometry.unaryUnion(geometries)
                if combined and not combined.isEmpty():
                    # Apply simplification to preview in both modes
                    tolerance = self._compute_simplification_tolerance(
                        self.current_transform_info, self._refine_simplify
                    )
                    if tolerance > 0:
                        combined = combined.simplify(tolerance)

                    self.mask_rubber_band.setToGeometry(combined, None)
                else:
                    self._clear_mask_visualization()
            else:
                self._clear_mask_visualization()

        except (ValueError, TypeError, RuntimeError) as e:
            QgsMessageLog.logMessage(
                "Mask visualization error ({}): {}".format(
                    type(e).__name__, str(e)),
                "AI Segmentation",
                level=Qgis.Warning
            )
            self._clear_mask_visualization()
        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                "Unexpected mask visualization error ({}): {}\n{}".format(
                    type(e).__name__, str(e), traceback.format_exc()),
                "AI Segmentation",
                level=Qgis.Critical
            )
            self._clear_mask_visualization()

    def _clear_mask_visualization(self):
        if self.mask_rubber_band:
            self.mask_rubber_band.reset(QgsWkbTypes.PolygonGeometry)

    def _on_clear_points(self):
        """Handle Escape key - clear current mask or warn about saved masks."""
        if self._batch_mode and len(self.saved_polygons) > 0:
            # Batch mode with saved masks: warn user before clearing everything
            reply = QMessageBox.warning(
                self.iface.mainWindow(),
                tr("Delete all saved polygons?"),
                "{}\n{}".format(
                    tr("This will delete all saved polygons."),
                    tr("Do you want to continue?")),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            # User confirmed: reset entire session
            self._reset_session()
            self.dock_widget.set_point_count(0, 0)
            self.dock_widget.set_saved_polygon_count(0)
            return

        # Normal clear: just clear current mask points
        self.prompts.clear()

        if self.map_tool:
            self.map_tool.clear_markers()

        self._clear_mask_visualization()

        self.current_mask = None
        self.current_score = 0.0
        self.current_low_res_mask = None
        self.dock_widget.set_point_count(0, 0)

    def _on_undo(self):
        """Undo last point added, or restore last saved mask in batch mode."""
        # Check if we have points in current mask
        current_point_count = self.prompts.point_count[0] + self.prompts.point_count[1]

        if current_point_count > 0:
            # Normal undo: remove last point from current mask
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
        elif self._batch_mode and len(self.saved_polygons) > 0:
            # Batch mode: no points in current mask, but have saved masks
            # Ask user if they want to restore the last saved mask
            reply = QMessageBox.warning(
                self.iface.mainWindow(),
                tr("Edit saved polygon"),
                "{}\n{}".format(
                    tr("Warning: you are about to edit an already saved polygon."),
                    tr("Do you want to continue?")),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._restore_last_saved_mask()

    def _restore_last_saved_mask(self):
        """Restore the last saved mask for editing in batch mode."""
        self._ensure_polygon_rubberband_sync()

        if not self.saved_polygons or not self.saved_rubber_bands:
            return

        # Pop the last saved polygon data
        last_polygon = self.saved_polygons.pop()

        # Remove the corresponding rubber band (green)
        if self.saved_rubber_bands:
            last_rb = self.saved_rubber_bands.pop()
            self._safe_remove_rubber_band(last_rb)

        # Clear current state first
        self.prompts.clear()
        if self.map_tool:
            self.map_tool.clear_markers()

        # Restore points
        points_positive = last_polygon.get('points_positive', [])
        points_negative = last_polygon.get('points_negative', [])

        # Rebuild prompts and history
        for pt in points_positive:
            self.prompts.add_positive_point(pt[0], pt[1])
            if self.map_tool:
                self.map_tool.add_marker(QgsPointXY(pt[0], pt[1]), is_positive=True)

        for pt in points_negative:
            self.prompts.add_negative_point(pt[0], pt[1])
            if self.map_tool:
                self.map_tool.add_marker(QgsPointXY(pt[0], pt[1]), is_positive=False)

        # Restore mask data
        self.current_mask = last_polygon.get('raw_mask')
        self.current_score = last_polygon.get('score', 0.0)
        self.current_transform_info = last_polygon.get('transform_info')

        # Restore refine settings
        self._refine_expand = last_polygon.get('refine_expand', 0)
        self._refine_simplify = last_polygon.get('refine_simplify', 4)
        self._refine_fill_holes = last_polygon.get('refine_fill_holes', False)
        self._refine_min_area = last_polygon.get('refine_min_area', 100)

        # Update UI sliders without emitting signals
        self.dock_widget.set_refine_values(
            self._refine_expand,
            self._refine_simplify,
            self._refine_fill_holes,
            self._refine_min_area
        )

        # Update visualization
        self._update_mask_visualization()

        # Update UI counters
        pos_count, neg_count = self.prompts.point_count
        self.dock_widget.set_point_count(pos_count, neg_count)
        self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))

        QgsMessageLog.logMessage(
            f"Restored mask with {pos_count} positive, {neg_count} negative points. "
            f"Refine: expand={self._refine_expand}, simplify={self._refine_simplify}, "
            f"fill_holes={self._refine_fill_holes}, min_area={self._refine_min_area}",
            "AI Segmentation",
            level=Qgis.Info
        )

    def _reset_session(self):
        self.prompts.clear()
        self.saved_polygons = []

        for rb in self.saved_rubber_bands:
            self._safe_remove_rubber_band(rb)
        self.saved_rubber_bands = []

        if self.map_tool:
            self.map_tool.clear_markers()

        self._clear_mask_visualization()

        self.current_mask = None
        self.current_score = 0.0
        self.current_transform_info = None
        self.current_low_res_mask = None

        # Reset refinement settings to defaults
        self._refine_expand = 0
        self._refine_simplify = 4   # Default: matches dockwidget spinbox
        self._refine_fill_holes = False  # Default: matches dockwidget checkbox
        self._refine_min_area = 100  # Default: matches dockwidget spinbox

        if self.dock_widget:
            self.dock_widget.set_point_count(0, 0)
            self.dock_widget.set_saved_polygon_count(0)
