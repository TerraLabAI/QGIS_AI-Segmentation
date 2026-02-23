import math
import os
import sys
import time
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
    QgsCoordinateTransform,
    QgsPointXY,
)
from qgis.gui import QgisInterface, QgsRubberBand
from qgis.PyQt.QtGui import QColor

from .ai_segmentation_dockwidget import AISegmentationDockWidget
from .ai_segmentation_maptool import AISegmentationMapTool
from .error_report_dialog import show_error_report, start_log_collector
from ..core.i18n import tr

# QSettings keys for tutorial flags
SETTINGS_KEY_TUTORIAL_SHOWN = "AI_Segmentation/tutorial_simple_shown"


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
        self.visible_extent = None  # (xmin, ymin, xmax, ymax) in raster CRS - set externally
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
                visible_extent=self.visible_extent,
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

        for x, y in self.positive_points:
            row, col = rio_transform.rowcol(transform, x, y)
            point_coords.append([col, row])
            point_labels.append(1)

        for x, y in self.negative_points:
            row, col = rio_transform.rowcol(transform, x, y)
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

        # Per-raster mask counters
        self._mask_counters = {}  # {raster_name: counter}

        self._visible_area_mode = False  # Encode full raster by default
        self._visible_extent = None  # Captured canvas extent (raster CRS) when visible area mode is on
        self._is_non_georeferenced_mode = False  # Track if current layer is non-georeferenced

        self.deps_install_worker = None
        self.download_worker = None
        self.encoding_worker = None

        self.mask_rubber_band: Optional[QgsRubberBand] = None
        self.saved_rubber_bands: List[QgsRubberBand] = []
        self._encoded_area_rubber_band: Optional[QgsRubberBand] = None

        self._previous_map_tool = None  # Store the tool active before segmentation
        self._stopping_segmentation = False  # Flag to track if we're stopping programmatically

        # CRS transforms (canvas CRS <-> raster CRS), created when features load.
        # None when both CRS are the same (no transform needed).
        self._canvas_to_raster_xform = None  # type: Optional[QgsCoordinateTransform]
        self._raster_to_canvas_xform = None  # type: Optional[QgsCoordinateTransform]

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
        except (RuntimeError, AttributeError):
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

    def _is_layer_georeferenced(self, layer) -> bool:
        """Check if a raster layer is properly georeferenced."""
        if layer is None or layer.type() != layer.RasterLayer:
            return False

        try:
            source = layer.source().lower()
        except RuntimeError:
            return False

        # PNG, JPG, BMP etc. without world files are not georeferenced
        non_georef_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        has_non_georef_ext = any(source.endswith(ext) for ext in non_georef_extensions)

        # If it's a known non-georeferenced format, check if it has a valid CRS
        if has_non_georef_ext:
            try:
                # Check if the layer has a valid CRS (not just default)
                if not layer.crs().isValid():
                    return False
                # Check if extent looks like pixel coordinates (0,0 to width,height)
                extent = layer.extent()
                if extent.xMinimum() == 0 and extent.yMinimum() == 0:
                    # Likely not georeferenced - just pixel dimensions
                    return False
            except RuntimeError:
                return False

        return True

    def _ensure_polygon_rubberband_sync(self):
        """Check polygon/rubber band list consistency. Log and preserve on mismatch."""
        n_polygons = len(self.saved_polygons)
        n_bands = len(self.saved_rubber_bands)
        if n_polygons != n_bands:
            QgsMessageLog.logMessage(
                "BUG: polygon/rubber band mismatch: {} vs {}. "
                "Data preserved, please report.".format(n_polygons, n_bands),
                "AI Segmentation",
                level=Qgis.Critical
            )

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
                tr("Segment elements on raster images using AI"))
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
        self.dock_widget.cancel_preparation_requested.connect(self._on_cancel_preparation)
        self.dock_widget.start_segmentation_requested.connect(self._on_start_segmentation)
        self.dock_widget.save_polygon_requested.connect(self._on_save_polygon)
        self.dock_widget.export_layer_requested.connect(self._on_export_layer)
        self.dock_widget.clear_points_requested.connect(self._on_clear_points)
        self.dock_widget.undo_requested.connect(self._on_undo)
        self.dock_widget.stop_segmentation_requested.connect(self._on_stop_segmentation)
        self.dock_widget.refine_settings_changed.connect(self._on_refine_settings_changed)
        self.dock_widget.batch_mode_changed.connect(self._on_batch_mode_changed)
        self.dock_widget.visible_area_changed.connect(self._on_visible_area_changed)
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

        # Log plugin version and environment for diagnostics (no personal paths)
        try:
            metadata_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "metadata.txt"
            )
            plugin_version = "unknown"
            if os.path.exists(metadata_path):
                with open(metadata_path, encoding="utf-8") as mf:
                    for mline in mf:
                        if mline.startswith("version="):
                            plugin_version = mline.strip().split("=", 1)[1]
                            break
            qgis_version = Qgis.version() if hasattr(Qgis, 'version') else "unknown"
            QgsMessageLog.logMessage(
                "AI Segmentation v{} | QGIS {} | Python {}.{}.{} | {}".format(
                    plugin_version, qgis_version,
                    sys.version_info.major, sys.version_info.minor,
                    sys.version_info.micro, sys.platform
                ),
                "AI Segmentation",
                level=Qgis.Info
            )
        except Exception:
            QgsMessageLog.logMessage(
                "AI Segmentation plugin loaded",
                "AI Segmentation",
                level=Qgis.Info
            )

    def unload(self):
        # 1. Disconnect ALL signals FIRST to prevent callbacks on partially-cleaned state
        if self.dock_widget:
            try:
                self.dock_widget.cleanup_signals()
            except (TypeError, RuntimeError, AttributeError):
                pass
            try:
                self.dock_widget.visibilityChanged.disconnect(self._on_dock_visibility_changed)
            except (TypeError, RuntimeError, AttributeError):
                pass
            try:
                self.dock_widget.layer_combo.layerChanged.disconnect(self._on_layer_combo_changed)
            except (TypeError, RuntimeError, AttributeError):
                pass
            # Stop timers before disconnection
            try:
                self.dock_widget._progress_timer.stop()
                self.dock_widget._refine_debounce_timer.stop()
            except (AttributeError, RuntimeError):
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

        # 3. Disconnect worker signals before termination to prevent callbacks on deleted UI
        for worker in [self.deps_install_worker, self.download_worker, self.encoding_worker]:
            if worker:
                try:
                    worker.progress.disconnect()
                except (TypeError, RuntimeError):
                    pass
                try:
                    worker.finished.disconnect()
                except (TypeError, RuntimeError):
                    pass

        # 4. Terminate workers with timeout
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

        # 5. Remove menu/toolbar
        plugins_menu = self.iface.pluginMenu()
        plugins_menu.removeAction(self.action)
        self.iface.removeToolBarIcon(self.action)

        # 6. Remove dock widget
        if self.dock_widget:
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget.deleteLater()
            self.dock_widget = None

        # 7. Unset map tool
        if self.map_tool:
            try:
                if self.iface.mapCanvas().mapTool() == self.map_tool:
                    self.iface.mapCanvas().unsetMapTool(self.map_tool)
            except RuntimeError:
                pass
            self.map_tool = None

        # 8. Remove rubber bands safely
        self._safe_remove_rubber_band(self.mask_rubber_band)
        self.mask_rubber_band = None

        self._safe_remove_rubber_band(self._encoded_area_rubber_band)
        self._encoded_area_rubber_band = None

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

        # Show GPU info in dependencies section (uses cached result, no extra overhead)
        self.dock_widget.update_gpu_info()

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
                # Auto-trigger install for upgrades and CUDA upgrades
                if "GPU acceleration" in message or "need updating" in message:
                    self._on_install_requested()

        except Exception as e:
            import traceback
            error_msg = f"Dependency check error: {str(e)}\n{traceback.format_exc()}"
            QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.Warning)
            self.dock_widget.set_dependency_status(False, f"Error: {str(e)[:50]}")

        # Check for plugin updates after a delay (gives QGIS time to fetch repo metadata)
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(5000, self._check_for_plugin_update)

    def _check_for_plugin_update(self):
        """Trigger the update check on the dock widget."""
        if self.dock_widget:
            self.dock_widget.check_for_updates()

    def _check_checkpoint(self):
        try:
            from ..core.checkpoint_manager import checkpoint_exists

            if checkpoint_exists():
                self.dock_widget.set_checkpoint_status(True, "SAM model ready")
                self._load_predictor()
                self._update_raster_encoded_status()
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
            QgsMessageLog.logMessage(
                f"Device info: {info}",
                "AI Segmentation",
                level=Qgis.Info
            )
        except RuntimeError as e:
            error_str = str(e)
            # Check if this is a PyTorch DLL error (Windows)
            if "DLL" in error_str or "shm.dll" in error_str:
                QgsMessageLog.logMessage(
                    f"PyTorch DLL error: {error_str}",
                    "AI Segmentation",
                    level=Qgis.Critical
                )
                # Show user-friendly error dialog
                from qgis.PyQt.QtWidgets import QMessageBox
                msg_box = QMessageBox(self.iface.mainWindow())
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setWindowTitle(tr("PyTorch Error"))
                msg_box.setText(tr("PyTorch cannot load on Windows"))
                msg_box.setInformativeText(
                    tr("The plugin requires Visual C++ Redistributables to run PyTorch.\n\n"
                       "Please download and install:\n"
                       "https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"
                       "After installation, restart QGIS and try again."))
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.exec_()
            else:
                raise
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Could not determine device info: {e}",
                "AI Segmentation",
                level=Qgis.Warning
            )

    def _on_install_requested(self):
        # Guard: prevent concurrent installs
        if self.deps_install_worker is not None and self.deps_install_worker.isRunning():
            QgsMessageLog.logMessage(
                "Install already in progress, ignoring duplicate request",
                "AI Segmentation",
                level=Qgis.Warning
            )
            return

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

        # Show activation popup 2 seconds after install starts
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(2000, self._show_activation_popup_if_needed)

    def _show_activation_popup_if_needed(self):
        """Show activation popup if not already activated (after deps+model ready)."""
        from ..core.activation_manager import is_plugin_activated
        if not is_plugin_activated() and not self.dock_widget.is_activated():
            self.dock_widget.show_activation_dialog()

    def _on_deps_install_progress(self, percent: int, message: str):
        if not self.dock_widget:
            return
        self.dock_widget.set_deps_install_progress(percent, message)

    def _on_deps_install_finished(self, success: bool, message: str):
        if not self.dock_widget:
            return
        self.dock_widget.set_deps_install_progress(100, "Done")

        if success:
            # GPU driver too old: just show a simple message bar, no error dialog
            if "DRIVER_TOO_OLD" in message:
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("Using CPU mode (GPU driver needs update)."),
                    level=Qgis.Info,
                    duration=10,
                )
            # CUDA install actually failed: show error dialog with log option
            elif "CUDA_FALLBACK" in message:
                # Disable install button during dialog to prevent re-entrant installs
                self.dock_widget.install_button.setEnabled(False)
                fallback_msg = "{}\n\n{}".format(
                    tr("Your GPU was detected but CUDA installation didn't work."),
                    tr("No worries, the plugin now uses CPU mode and everything works fine :) "
                       "If you'd like us to fix GPU support for your setup, send us your logs!"))
                show_error_report(
                    self.iface.mainWindow(),
                    tr("GPU mode failed, using CPU"),
                    fallback_msg
                )

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
        if not self.dock_widget:
            return
        self.dock_widget.set_download_progress(percent, message)

    def _on_download_finished(self, success: bool, message: str):
        if not self.dock_widget:
            return
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

        # Validate layer is still alive before accessing C++ properties
        if not self._is_layer_valid(layer):
            QgsMessageLog.logMessage(
                "Layer was deleted before segmentation could start",
                "AI Segmentation", level=Qgis.Warning)
            return

        self._current_layer = layer
        try:
            self._current_layer_name = layer.name().replace(" ", "_")
            raster_path = os.path.normcase(layer.source())
        except RuntimeError:
            QgsMessageLog.logMessage(
                "Layer deleted during segmentation start",
                "AI Segmentation", level=Qgis.Warning)
            return

        # Detect if layer is non-georeferenced (pixel coordinate mode)
        self._is_non_georeferenced_mode = not self._is_layer_georeferenced(layer)
        if self._is_non_georeferenced_mode:
            QgsMessageLog.logMessage(
                "Non-georeferenced image detected - using pixel coordinate mode. "
                "Polygons will be created in pixel coordinates.",
                "AI Segmentation",
                level=Qgis.Info
            )

        from ..core.checkpoint_manager import has_features_for_raster, get_raster_features_dir, get_checkpoint_path

        # Capture visible extent if visible area mode is enabled
        self._visible_extent = None
        if self._visible_area_mode:
            self._visible_extent = self._capture_visible_extent(layer)
            if self._visible_extent is None:
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("Could not determine visible area. Encoding full raster instead."),
                    level=Qgis.Warning, duration=5
                )

        # Check if full raster is already encoded (no visible extent restriction)
        full_raster_encoded = has_features_for_raster(raster_path, None)
        if self.dock_widget:
            self.dock_widget.set_raster_fully_encoded(full_raster_encoded)

        if has_features_for_raster(raster_path, self._visible_extent):
            self._load_features_and_activate(raster_path)
        else:
            output_dir = get_raster_features_dir(raster_path, self._visible_extent)
            checkpoint_path = get_checkpoint_path()

            layer_crs_wkt = None
            try:
                if layer.crs().isValid():
                    layer_crs_wkt = layer.crs().toWkt()
            except RuntimeError:
                pass

            # Get layer extent for rasters without embedded georeferencing (e.g., PNG)
            layer_extent = None
            try:
                ext = layer.extent()
                if ext and not ext.isEmpty():
                    coords = (ext.xMinimum(), ext.yMinimum(), ext.xMaximum(), ext.yMaximum())
                    if any(math.isnan(c) or math.isinf(c) for c in coords):
                        show_error_report(
                            self.iface.mainWindow(),
                            tr("Invalid Layer"),
                            tr("Layer extent contains invalid coordinates (NaN/Inf). Check the raster file.")
                        )
                        return
                    layer_extent = coords
            except RuntimeError:
                pass

            QgsMessageLog.logMessage(
                "Starting encoding - Layer extent: {}, Visible extent: {}, "
                "CRS valid: {}, CRS: {}".format(
                    layer_extent, self._visible_extent,
                    layer.crs().isValid(),
                    layer.crs().authid() if layer.crs().isValid() else 'None'),
                "AI Segmentation",
                level=Qgis.Info
            )

            self.dock_widget.set_preparation_progress(0, "Encoding raster (first time)...")
            self._encoding_start_time = time.monotonic()

            self.encoding_worker = EncodingWorker(
                raster_path, output_dir, checkpoint_path,
                layer_crs_wkt, layer_extent)
            self.encoding_worker.visible_extent = self._visible_extent
            self.encoding_worker.progress.connect(self._on_encoding_progress)
            self.encoding_worker.finished.connect(
                lambda s, m: self._on_encoding_finished(s, m, raster_path)
            )
            self.encoding_worker.start()

    def _on_encoding_progress(self, percent: int, message: str):
        if not self.dock_widget:
            return
        self.dock_widget.set_preparation_progress(percent, message)

    def _on_encoding_finished(self, success: bool, message: str, raster_path: str):
        if not self.dock_widget:
            return

        # Log encoding duration
        encoding_duration = 0
        if hasattr(self, '_encoding_start_time'):
            encoding_duration = time.monotonic() - self._encoding_start_time
            QgsMessageLog.logMessage(
                "Encoding {}: {:.1f}s".format(
                    "completed" if success else "failed", encoding_duration),
                "AI Segmentation",
                level=Qgis.Info
            )

        if success:
            self.dock_widget.set_preparation_progress(100, "Done!")
            self.dock_widget.encoding_info_label.setVisible(False)
            # Update fully encoded flag if this was a full raster encoding
            if self._visible_extent is None:
                self.dock_widget.set_raster_fully_encoded(True)
            self._load_features_and_activate(raster_path)
        else:
            # Check if this was a user cancellation
            is_cancelled = "cancelled" in message.lower() or "canceled" in message.lower()

            try:
                # Clean up partial cache files
                from ..core.checkpoint_manager import clear_features_for_raster
                try:
                    clear_features_for_raster(raster_path, self._visible_extent)
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

                if not is_cancelled:
                    # Actual error - show error report dialog
                    show_error_report(
                        self.iface.mainWindow(),
                        tr("Encoding Failed"),
                        "{}\n{}".format(tr("Failed to encode raster:"), message)
                    )
            finally:
                # Always reset UI to default state, even if error dialog fails
                self.dock_widget.set_preparation_progress(100, "Cancelled")
                self.dock_widget.reset_session()

    def _load_features_and_activate(self, raster_path: str):
        try:
            from ..core.checkpoint_manager import get_raster_features_dir
            from ..core.feature_dataset import FeatureDataset

            features_dir = get_raster_features_dir(raster_path, self._visible_extent)
            self.feature_dataset = FeatureDataset(features_dir, cache=True)

            bounds = self.feature_dataset.bounds
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            raster_crs = self._current_layer.crs() if self._current_layer else None

            # Build CRS transforms when canvas and raster CRS differ.
            # All click coordinates arrive in canvas CRS and must be
            # transformed to raster CRS before bounds checks, tile lookup,
            # and point-to-pixel conversion. Geometries produced from masks
            # (in raster CRS) must be transformed back for rubberband display.
            self._canvas_to_raster_xform = None
            self._raster_to_canvas_xform = None
            if raster_crs and canvas_crs.isValid() and raster_crs.isValid():
                if canvas_crs != raster_crs:
                    self._canvas_to_raster_xform = QgsCoordinateTransform(
                        canvas_crs, raster_crs, QgsProject.instance())
                    self._raster_to_canvas_xform = QgsCoordinateTransform(
                        raster_crs, canvas_crs, QgsProject.instance())
                    QgsMessageLog.logMessage(
                        "CRS transform enabled: {} -> {}".format(
                            canvas_crs.authid(), raster_crs.authid()),
                        "AI Segmentation",
                        level=Qgis.Info
                    )

            QgsMessageLog.logMessage(
                f"Loaded {len(self.feature_dataset)} feature tiles",
                "AI Segmentation",
                level=Qgis.Info
            )
            QgsMessageLog.logMessage(
                "Feature dataset bounds: minx={:.2f}, maxx={:.2f}, "
                "miny={:.2f}, maxy={:.2f}, CRS={}".format(
                    bounds[0], bounds[1], bounds[2], bounds[3],
                    self.feature_dataset.crs),
                "AI Segmentation",
                level=Qgis.Info
            )
            QgsMessageLog.logMessage(
                "DEBUG - Canvas CRS: {}, Raster CRS: {}".format(
                    canvas_crs.authid(),
                    raster_crs.authid() if raster_crs else "None"),
                "AI Segmentation",
                level=Qgis.Info
            )
            # Show encoded area rectangle if visible area mode is active
            if self._visible_extent is not None:
                self._show_encoded_area_rectangle()

            self._activate_segmentation_tool()

        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                "Failed to load features: {}\n{}".format(str(e), traceback.format_exc()),
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

        # Show tutorial notification for first-time users
        self._show_tutorial_notification()

    def _get_next_mask_counter(self) -> int:
        """Increment and return the mask counter for the current raster."""
        name = self._current_layer_name
        if name not in self._mask_counters:
            self._mask_counters[name] = 0
        self._mask_counters[name] += 1
        return self._mask_counters[name]

    def _show_tutorial_notification(self):
        """Show YouTube tutorial notification (once ever, persisted in QSettings)."""
        settings = QSettings()
        if settings.value(SETTINGS_KEY_TUTORIAL_SHOWN, False, type=bool):
            return
        settings.setValue(SETTINGS_KEY_TUTORIAL_SHOWN, True)

        tutorial_url = "https://www.youtube.com/playlist?list=PL4hCF043nAUW2iIxALNUzy1fKHcCWwDsv&jct=GTA3Fx8pJzuTLPPivC9RRQ"
        message = '{} <a href="{}">{}</a>'.format(
            tr('New to AI Segmentation?'),
            tutorial_url,
            tr('Watch our tutorial'))

        self.iface.messageBar().pushMessage(
            "AI Segmentation",
            message,
            level=Qgis.Info,
            duration=10
        )

    def _on_batch_mode_changed(self, batch: bool):
        """Handle batch mode toggle (no-op, batch mode is always on)."""
        pass

    def _on_visible_area_changed(self, visible: bool):
        """Handle visible area mode toggle from the dock widget."""
        self._visible_area_mode = visible

    def _capture_visible_extent(self, layer):
        """Capture the current canvas visible extent in the raster's CRS.

        Returns (xmin, ymin, xmax, ymax) tuple in raster CRS, or None on failure.
        """
        try:
            canvas = self.iface.mapCanvas()
            canvas_extent = canvas.extent()
            canvas_crs = canvas.mapSettings().destinationCrs()
            raster_crs = layer.crs()

            if not canvas_crs.isValid() or not raster_crs.isValid():
                return None

            # Transform canvas extent to raster CRS if different
            if canvas_crs != raster_crs:
                xform = QgsCoordinateTransform(
                    canvas_crs, raster_crs, QgsProject.instance())
                canvas_extent = xform.transformBoundingBox(canvas_extent)

            # Intersect with layer extent to avoid encoding outside the raster
            layer_extent = layer.extent()
            intersected = canvas_extent.intersect(layer_extent)

            if intersected.isEmpty():
                QgsMessageLog.logMessage(
                    "Visible area does not intersect with raster extent",
                    "AI Segmentation", level=Qgis.Warning)
                return None

            result = (
                intersected.xMinimum(), intersected.yMinimum(),
                intersected.xMaximum(), intersected.yMaximum()
            )

            QgsMessageLog.logMessage(
                "Captured visible extent (raster CRS): {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
                    *result),
                "AI Segmentation", level=Qgis.Info
            )
            return result

        except Exception as e:
            QgsMessageLog.logMessage(
                "Failed to capture visible extent: {}".format(e),
                "AI Segmentation", level=Qgis.Warning)
            return None

    def _on_layer_combo_changed(self, layer):
        """Handle layer selection change in the combo box."""
        # Always check if the selected raster is fully encoded (for visible area checkbox)
        self._update_raster_encoded_status(layer)

        # Only care about segmentation reset if we're currently segmenting
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

    def _update_raster_encoded_status(self, layer=None):
        """Check if a raster layer is fully encoded and update the dock widget."""
        if not self.dock_widget:
            return
        if layer is None:
            layer = self.dock_widget.layer_combo.currentLayer()
        if layer is None:
            self.dock_widget.set_raster_fully_encoded(False)
            return
        try:
            raster_path = os.path.normcase(layer.source())
        except RuntimeError:
            self.dock_widget.set_raster_fully_encoded(False)
            return
        try:
            from ..core.checkpoint_manager import has_features_for_raster
            fully_encoded = has_features_for_raster(raster_path, None)
            self.dock_widget.set_raster_fully_encoded(fully_encoded)
        except Exception:
            self.dock_widget.set_raster_fully_encoded(False)

    def _on_save_polygon(self):
        """Save current mask as polygon."""
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
            # Geometry is in raster CRS; transform to canvas CRS for display
            display_geom = QgsGeometry(combined)
            self._transform_geometry_to_canvas_crs(display_geom)
            saved_rb.setToGeometry(display_geom, None)
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
        """Export all saved polygons + current unsaved mask to a new layer."""
        from ..core.polygon_exporter import mask_to_polygons, apply_mask_refinement

        self._ensure_polygon_rubberband_sync()

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

        self._stopping_segmentation = True
        self.iface.mapCanvas().unsetMapTool(self.map_tool)
        self._restore_previous_map_tool()
        self._stopping_segmentation = False

        # Generate layer name: {RasterName}_mask_{number} (same for both modes)
        mask_num = self._get_next_mask_counter()
        layer_name = f"{self._current_layer_name}_mask_{mask_num}"

        # Determine CRS
        # For non-georeferenced images, use a local pixel-based CRS
        if self._is_non_georeferenced_mode:
            # Use EPSG:3857 (Web Mercator) with pixel coordinates
            # This allows visualization while being clear it's not true geographic data
            crs = QgsCoordinateReferenceSystem('EPSG:3857')
            QgsMessageLog.logMessage(
                "Non-georeferenced mode: Using EPSG:3857 with pixel coordinates",
                "AI Segmentation",
                level=Qgis.Info
            )
        else:
            # Normal georeferenced mode
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
                try:
                    crs_str = self._current_layer.crs().authid() if self._is_layer_valid() and self._current_layer.crs().isValid() else 'EPSG:4326'
                except RuntimeError:
                    crs_str = 'EPSG:4326'
            if isinstance(crs_str, str) and crs_str.strip():
                crs = QgsCoordinateReferenceSystem(crs_str)
            else:
                try:
                    crs = self._current_layer.crs() if self._is_layer_valid() else QgsCoordinateReferenceSystem('EPSG:4326')
                except RuntimeError:
                    crs = QgsCoordinateReferenceSystem('EPSG:4326')

        # Determine output directory for GeoPackage file
        # Priority: 1) Project directory (if project is saved), 2) Raster source directory
        output_dir = None
        project_path = QgsProject.instance().absolutePath()
        if project_path:
            output_dir = project_path
        elif self._is_layer_valid():
            try:
                raster_source = self._current_layer.source()
                if raster_source and os.path.exists(raster_source):
                    output_dir = os.path.dirname(raster_source)
            except RuntimeError:
                pass

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
        if not self.dock_widget:
            return
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
        polygon_count = len(self.saved_polygons)
        if has_unsaved_mask:
            polygon_count += 1
        message = "{}\n\n{}".format(
            tr("You have {count} unsaved polygon(s).").format(count=polygon_count),
            tr("Discard and exit segmentation?"))

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

    def _transform_to_raster_crs(self, point):
        """Transform a QgsPointXY from canvas CRS to raster CRS.

        Returns the original point unchanged when both CRS are identical.
        """
        if self._canvas_to_raster_xform is not None:
            return self._canvas_to_raster_xform.transform(point)
        return point

    def _transform_geometry_to_canvas_crs(self, geometry):
        """Transform a QgsGeometry from raster CRS to canvas CRS (in-place).

        Does nothing when both CRS are identical.
        """
        if self._raster_to_canvas_xform is not None:
            geometry.transform(self._raster_to_canvas_xform)

    def _transform_to_canvas_crs(self, point):
        """Transform a QgsPointXY from raster CRS to canvas CRS.

        Returns the original point unchanged when both CRS are identical.
        """
        if self._raster_to_canvas_xform is not None:
            return self._raster_to_canvas_xform.transform(point)
        return point

    def _is_point_in_encoded_area(self, point):
        """Check if a point (in raster CRS) falls within the encoded extent."""
        bounds = self.feature_dataset.bounds
        # bounds = (minx, maxx, miny, maxy, mint, maxt)
        in_x = bounds[0] <= point.x() <= bounds[1]
        in_y = bounds[2] <= point.y() <= bounds[3]
        return in_x and in_y

    def _get_out_of_bounds_message(self):
        """Return an appropriate message when clicking outside encoded area."""
        if self._visible_extent is not None:
            return tr("Click is outside the encoded visible area. Restart segmentation to encode a different area.")
        return tr("Point is outside the encoded image. Click inside the raster.")

    def _on_positive_click(self, point):
        """Handle left-click: add positive point (select this element)."""
        if self.predictor is None or self.feature_dataset is None:
            # Remove the marker that was added by the maptool
            if self.map_tool:
                self.map_tool.remove_last_marker()
            return

        # Transform click from canvas CRS to raster CRS for all downstream use
        raster_pt = self._transform_to_raster_crs(point)

        if not self._is_point_in_encoded_area(raster_pt):
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                self._get_out_of_bounds_message(),
                level=Qgis.Warning,
                duration=4
            )
            return

        QgsMessageLog.logMessage(
            "POSITIVE POINT at ({:.6f}, {:.6f})".format(
                raster_pt.x(), raster_pt.y()),
            "AI Segmentation",
            level=Qgis.Info
        )

        self.prompts.add_positive_point(raster_pt.x(), raster_pt.y())
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

        # Transform click from canvas CRS to raster CRS for all downstream use
        raster_pt = self._transform_to_raster_crs(point)

        if not self._is_point_in_encoded_area(raster_pt):
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                self._get_out_of_bounds_message(),
                level=Qgis.Warning,
                duration=4
            )
            return

        QgsMessageLog.logMessage(
            "NEGATIVE POINT at ({:.6f}, {:.6f})".format(
                raster_pt.x(), raster_pt.y()),
            "AI Segmentation",
            level=Qgis.Info
        )

        self.prompts.add_negative_point(raster_pt.x(), raster_pt.y())
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

        try:
            masks, scores, low_res_masks = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                multimask_output=False,
            )
        except RuntimeError as e:
            error_str = str(e)
            QgsMessageLog.logMessage(
                "Prediction failed: {}".format(error_str),
                "AI Segmentation",
                level=Qgis.Critical
            )
            # Check if this is a DLL error
            if "DLL" in error_str or "Visual C++" in error_str:
                from qgis.PyQt.QtWidgets import QMessageBox
                msg_box = QMessageBox(self.iface.mainWindow())
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setWindowTitle(tr("Prediction Error"))
                msg_box.setText(tr("Segmentation failed"))
                msg_box.setInformativeText(tr("{}").format(error_str))
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.exec_()
            return
        except Exception as e:
            QgsMessageLog.logMessage(
                "Unexpected prediction error: {}".format(str(e)),
                "AI Segmentation",
                level=Qgis.Critical
            )
            return

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

                    # Geometry is in raster CRS; transform to canvas CRS
                    self._transform_geometry_to_canvas_crs(combined)
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

    def _show_encoded_area_rectangle(self):
        """Show a blue semi-transparent rectangle on the map for the encoded area."""
        self._clear_encoded_area_rectangle()

        if self.feature_dataset is None:
            return

        bounds = self.feature_dataset.bounds
        # bounds = (minx, maxx, miny, maxy, mint, maxt)
        minx, maxx, miny, maxy = bounds[0], bounds[1], bounds[2], bounds[3]

        rb = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
        rb.setColor(QColor(30, 120, 220, 160))
        rb.setFillColor(QColor(0, 0, 0, 0))
        rb.setWidth(2)

        # Build rectangle geometry in raster CRS
        rect_geom = QgsGeometry.fromRect(
            self._make_qgs_rectangle(minx, miny, maxx, maxy))

        # Transform to canvas CRS if needed
        if self._raster_to_canvas_xform:
            rect_geom.transform(self._raster_to_canvas_xform)

        rb.setToGeometry(rect_geom, None)
        self._encoded_area_rubber_band = rb

    def _clear_encoded_area_rectangle(self):
        """Remove the encoded area rectangle from the map."""
        if self._encoded_area_rubber_band is not None:
            self._safe_remove_rubber_band(self._encoded_area_rubber_band)
            self._encoded_area_rubber_band = None

    @staticmethod
    def _make_qgs_rectangle(minx, miny, maxx, maxy):
        """Create a QgsRectangle from coordinates."""
        from qgis.core import QgsRectangle
        return QgsRectangle(minx, miny, maxx, maxy)

    def _on_clear_points(self):
        """Handle Escape key - clear current mask or warn about saved masks."""
        if len(self.saved_polygons) > 0:
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
        elif len(self.saved_polygons) > 0:
            # No points in current mask, but have saved masks
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

        # Rebuild prompts (stored in raster CRS) and markers (displayed in canvas CRS)
        for pt in points_positive:
            self.prompts.add_positive_point(pt[0], pt[1])
            if self.map_tool:
                canvas_pt = self._transform_to_canvas_crs(QgsPointXY(pt[0], pt[1]))
                self.map_tool.add_marker(canvas_pt, is_positive=True)

        for pt in points_negative:
            self.prompts.add_negative_point(pt[0], pt[1])
            if self.map_tool:
                canvas_pt = self._transform_to_canvas_crs(QgsPointXY(pt[0], pt[1]))
                self.map_tool.add_marker(canvas_pt, is_positive=False)

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
        self._clear_encoded_area_rectangle()

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
