import math
import os
import sys
from pathlib import Path
from typing import List, Optional

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFillSymbol,
    QgsGeometry,
    QgsMessageLog,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.gui import QgisInterface, QgsRubberBand
from qgis.PyQt.QtCore import (
    QSettings,
    Qt,
    QVariant,
)
from qgis.PyQt.QtGui import QColor, QIcon
from qgis.PyQt.QtWidgets import (
    QAction,
    QApplication,
    QMenu,
    QMessageBox,
)

from ..core.i18n import tr
from ..core.prompt_manager import PromptManager
from ..core.tile_manager import TileManager
from ..workers.setup_workers import DepsInstallWorker, DownloadWorker, VerifyWorker
from .ai_segmentation_dockwidget import AISegmentationDockWidget
from .ai_segmentation_maptool import AISegmentationMapTool
from .ai_segmentation_pro_dockwidget import AISegmentationProDockWidget
from .error_report_dialog import (
    show_error_report,
    start_log_collector,
    stop_log_collector,
)
from .shortcut_filter import ShortcutFilter
from .zone_selection_maptool import ZoneSelectionMapTool

# QSettings keys for tutorial flags
SETTINGS_KEY_TUTORIAL_SHOWN = "AI_Segmentation/tutorial_simple_shown"


def _get_change_path_instructions():
    """Return platform-specific instructions for changing the install path."""
    if sys.platform == "win32":
        steps = tr(
            "1. Open Windows Settings > System > Advanced system settings\n"
            "2. Click 'Environment Variables'\n"
            "3. Under 'User variables', click 'New'\n"
            "4. Variable name: AI_SEGMENTATION_CACHE_DIR\n"
            "5. Variable value: the folder path you want to use\n"
            "6. Click OK and restart QGIS"
        )
    elif sys.platform == "darwin":
        steps = tr(
            "Run this command in Terminal, then restart QGIS:\n\n"
            "launchctl setenv AI_SEGMENTATION_CACHE_DIR /your/path"
        )
    else:
        steps = tr(
            "Add this line to your ~/.bashrc or ~/.profile, "
            "then restart QGIS:\n\n"
            "export AI_SEGMENTATION_CACHE_DIR=/your/path"
        )
    return "{}\n\n{}".format(
        tr(
            "To install in a different folder, set the environment "
            "variable AI_SEGMENTATION_CACHE_DIR:"
        ),
        steps,
    )


class AISegmentationPlugin:
    def __init__(self, iface: QgisInterface):
        self.iface = iface
        self.plugin_dir = Path(__file__).parent.parent.parent

        self.dock_widget: Optional[AISegmentationDockWidget] = None
        self.pro_dock_widget: Optional[AISegmentationProDockWidget] = None
        self.map_tool: Optional[AISegmentationMapTool] = None
        self.action: Optional[QAction] = None
        self.pro_action: Optional[QAction] = None
        self._terralab_menu: Optional[QMenu] = None
        self._active_mode: str = "standard"  # "standard" | "pro"

        self.predictor = None
        self.prompts = PromptManager()

        self.current_mask = None
        self.current_score = 0.0
        self.current_transform_info = None
        self.current_low_res_mask = (
            None  # For iterative refinement with negative points
        )
        self.saved_polygons = []

        self._initialized = False
        self._current_layer = None
        self._current_layer_name = ""

        # Refinement settings
        self._refine_expand = 0
        self._refine_simplify = 3  # Default: matches dockwidget spinbox
        self._refine_fill_holes = False  # Default: matches dockwidget checkbox
        self._refine_min_area = 100  # Default: matches dockwidget spinbox

        self._is_non_georeferenced_mode = (
            False  # Track if current layer is non-georeferenced
        )
        self._is_online_layer = (
            False  # Track if current layer is online (WMS, XYZ, etc.)
        )

        # On-demand encoding state
        self._current_crop_info = None  # dict with 'bounds', 'img_shape'
        self._current_raster_path = None
        self._encoding_in_progress = False  # Guard against concurrent clicks
        self._shortcut_filter = None  # Event filter for keyboard shortcuts
        self._current_crop_canvas_mupp = (
            None  # canvas mupp at encode time (zoom detection)
        )
        self._current_crop_actual_mupp = (
            None  # actual mupp used for the crop (may differ if zoomed out)
        )
        self._current_crop_scale_factor = None  # scale_factor used for file-based crop

        self.deps_install_worker = None
        self.download_worker = None
        self._verify_worker = None

        self.mask_rubber_band: Optional[QgsRubberBand] = None
        self.saved_rubber_bands: List[QgsRubberBand] = []

        self._previous_map_tool = None  # Store the tool active before segmentation
        self._stopping_segmentation = (
            False  # Flag to track if we're stopping programmatically
        )
        self._exporting_in_progress = False  # Guard against double-click on export

        # CRS transforms (canvas CRS <-> raster CRS), created when features load.
        # None when both CRS are the same (no transform needed).
        self._canvas_to_raster_xform = None  # type: Optional[QgsCoordinateTransform]
        self._raster_to_canvas_xform = None  # type: Optional[QgsCoordinateTransform]

        # PRO mode state
        self._pro_pending_detections = []  # list of {mask, score, transform_info, rb}
        self._pro_detection_batches = []  # list of int (size of each batch)

        # Tiling / zone selection state
        self._tile_manager = TileManager(
            tile_size=1024, overlap_fraction=0.15, max_tiles=50
        )
        self._zone_selection_tool = None
        self._selected_zone = None  # QgsRectangle or None (full image)
        self._tiled_worker = None

    @property
    def _active_dock(self):
        """Return the dock widget corresponding to the current active mode."""
        if self._active_mode == "pro" and self.pro_dock_widget:
            return self.pro_dock_widget
        return self.dock_widget

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
        non_georef_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
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

    @staticmethod
    def _is_online_provider(layer) -> bool:
        """Check if a raster layer uses an online data provider."""
        if layer is None:
            return False
        try:
            provider = layer.dataProvider()
            if provider is None:
                return False
            from ..core.feature_encoder import ONLINE_PROVIDERS

            return provider.name() in ONLINE_PROVIDERS
        except (RuntimeError, AttributeError):
            return False

    def _ensure_polygon_rubberband_sync(self):
        """Check polygon/rubber band list consistency. Repair on mismatch."""
        n_polygons = len(self.saved_polygons)
        n_bands = len(self.saved_rubber_bands)
        if n_polygons != n_bands:
            QgsMessageLog.logMessage(
                "BUG: polygon/rubber band mismatch: {} vs {}. "
                "Truncating to min. Please report.".format(n_polygons, n_bands),
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical,
            )
            min_len = min(n_polygons, n_bands)
            while len(self.saved_rubber_bands) > min_len:
                rb = self.saved_rubber_bands.pop()
                self._safe_remove_rubber_band(rb)
            self.saved_polygons = self.saved_polygons[:min_len]

    @staticmethod
    def _compute_simplification_tolerance(transform_info, simplify_value):
        """Compute simplification tolerance from transform_info and slider value.

        Returns 0 if inputs are invalid or simplify_value is 0.
        """
        if simplify_value <= 0 or transform_info is None:
            return 0
        bbox = transform_info.get("bbox", [0, 0, 1, 1])
        img_shape = transform_info.get("img_shape", (1024, 1024))
        width_pixels = max(img_shape[1], 1)
        bbox_width = bbox[2] - bbox[0]  # maxx - minx
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

        self.action = QAction(icon, "AI Segmentation", self.iface.mainWindow())
        self.action.setCheckable(True)
        self.action.setToolTip(
            "AI Segmentation by TerraLab\n{}".format(
                tr("Segment elements on raster images using AI")
            )
        )
        self.action.triggered.connect(self.toggle_dock_widget)

        from .terralab_menu import (
            add_plugin_to_menu,
            add_to_plugins_menu,
            get_or_create_terralab_menu,
        )

        main_window = self.iface.mainWindow()
        self._terralab_menu = get_or_create_terralab_menu(main_window)
        add_plugin_to_menu(self._terralab_menu, self.action, "ai-segmentation")
        self.iface.addToolBarIcon(self.action)
        add_to_plugins_menu(self.iface, self.action)

        self.dock_widget = AISegmentationDockWidget(self.iface.mainWindow())

        self.dock_widget.install_requested.connect(self._on_install_requested)
        self.dock_widget.cancel_install_requested.connect(self._on_cancel_install)
        self.dock_widget.start_segmentation_requested.connect(
            self._on_start_segmentation
        )
        self.dock_widget.save_polygon_requested.connect(self._on_save_polygon)
        self.dock_widget.export_layer_requested.connect(self._on_export_layer)
        self.dock_widget.clear_points_requested.connect(self._on_clear_points)
        self.dock_widget.undo_requested.connect(self._on_undo)
        self.dock_widget.stop_segmentation_requested.connect(self._on_stop_segmentation)
        self.dock_widget.refine_settings_changed.connect(
            self._on_refine_settings_changed
        )
        self.dock_widget.batch_mode_changed.connect(self._on_batch_mode_changed)
        self.dock_widget.layer_combo.layerChanged.connect(self._on_layer_combo_changed)

        self.iface.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.dock_widget
        )
        self.dock_widget.setVisible(False)
        self.dock_widget.visibilityChanged.connect(self._on_dock_visibility_changed)

        # Project signals owned by plugin (not dock widget) for clean lifecycle
        QgsProject.instance().layersAdded.connect(self.dock_widget._on_layers_added)
        QgsProject.instance().layersRemoved.connect(self.dock_widget._on_layers_removed)

        # --- PRO dock widget ---
        self.pro_action = QAction(icon, "AI Segmentation Pro", self.iface.mainWindow())
        self.pro_action.setCheckable(True)
        self.pro_action.triggered.connect(self._toggle_pro_dock_widget)
        add_plugin_to_menu(self._terralab_menu, self.pro_action, "ai-segmentation-pro")
        self.iface.addToolBarIcon(self.pro_action)
        add_to_plugins_menu(self.iface, self.pro_action)

        self.pro_dock_widget = AISegmentationProDockWidget(self.iface.mainWindow())
        self.pro_dock_widget.start_pro_segmentation_requested.connect(
            self._on_start_pro_segmentation
        )
        self.pro_dock_widget.save_polygon_requested.connect(self._on_save_polygon)
        self.pro_dock_widget.export_layer_requested.connect(self._on_export_layer)
        self.pro_dock_widget.clear_points_requested.connect(self._on_clear_points)
        self.pro_dock_widget.undo_requested.connect(self._on_undo)
        self.pro_dock_widget.stop_segmentation_requested.connect(
            self._on_stop_segmentation
        )
        self.pro_dock_widget.pro_detect_requested.connect(self._run_fal_detection)
        self.pro_dock_widget.zone_select_requested.connect(self._on_zone_select)
        self.pro_dock_widget.zone_clear_requested.connect(self._on_zone_clear)
        self.pro_dock_widget.refine_settings_changed.connect(
            self._on_refine_settings_changed
        )
        self.pro_dock_widget.layer_combo.layerChanged.connect(
            self._on_pro_layer_combo_changed
        )
        self.iface.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.pro_dock_widget
        )
        self.pro_dock_widget.setVisible(False)
        self.pro_dock_widget.visibilityChanged.connect(
            self._on_pro_dock_visibility_changed
        )

        QgsProject.instance().layersAdded.connect(
            self.pro_dock_widget._on_layers_changed
        )
        QgsProject.instance().layersRemoved.connect(
            self.pro_dock_widget._on_layers_changed
        )

        self.map_tool = AISegmentationMapTool(self.iface.mapCanvas())
        self.map_tool.positive_click.connect(self._on_positive_click)
        self.map_tool.negative_click.connect(self._on_negative_click)
        self.map_tool.tool_deactivated.connect(self._on_tool_deactivated)
        self.map_tool.undo_requested.connect(self._on_undo)
        self.map_tool.save_polygon_requested.connect(self._on_save_polygon)
        self.map_tool.export_layer_requested.connect(self._on_export_layer)
        self.map_tool.stop_segmentation_requested.connect(self._on_stop_segmentation)

        self.mask_rubber_band = QgsRubberBand(
            self.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry
        )
        self.mask_rubber_band.setColor(QColor(0, 120, 255, 100))
        self.mask_rubber_band.setStrokeColor(QColor(0, 80, 200))
        self.mask_rubber_band.setWidth(2)

        # Log plugin version and environment for diagnostics (no personal paths)
        try:
            metadata_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "metadata.txt",
            )
            plugin_version = "unknown"
            if os.path.exists(metadata_path):
                with open(metadata_path, encoding="utf-8") as mf:
                    for mline in mf:
                        if mline.startswith("version="):
                            plugin_version = mline.strip().split("=", 1)[1]
                            break
            qgis_version = Qgis.version() if hasattr(Qgis, "version") else "unknown"
            QgsMessageLog.logMessage(
                "AI Segmentation v{} | QGIS {} | Python {}.{}.{} | {}".format(
                    plugin_version,
                    qgis_version,
                    sys.version_info.major,
                    sys.version_info.minor,
                    sys.version_info.micro,
                    sys.platform,
                ),
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )
        except Exception:
            QgsMessageLog.logMessage(
                "AI Segmentation plugin loaded",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )

    def unload(self):
        # 1. Remove keyboard shortcut filter
        if self._shortcut_filter is not None:
            app = QApplication.instance()
            if app:
                app.removeEventFilter(self._shortcut_filter)
            self._shortcut_filter = None

        # 2. Cancel and terminate workers
        for worker in [
            self.deps_install_worker,
            self.download_worker,
            self._verify_worker,
            self._tiled_worker,
        ]:
            if worker and worker.isRunning():
                if hasattr(worker, "cancel"):
                    worker.cancel()
                worker.wait(5000)
        self.deps_install_worker = None
        self.download_worker = None
        self._verify_worker = None

        # 3. Cleanup predictor subprocess
        if self.predictor:
            try:
                self.predictor.cleanup()
            except Exception:
                pass
            self.predictor = None

        # 4. Disconnect QgsProject signals (only ones needing explicit disconnect)
        if self.dock_widget:
            for sig, slot in [
                (
                    QgsProject.instance().layersAdded,
                    self.dock_widget._on_layers_added,
                ),
                (
                    QgsProject.instance().layersRemoved,
                    self.dock_widget._on_layers_removed,
                ),
            ]:
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
        if self.pro_dock_widget:
            for sig, slot in [
                (
                    QgsProject.instance().layersAdded,
                    self.pro_dock_widget._on_layers_changed,
                ),
                (
                    QgsProject.instance().layersRemoved,
                    self.pro_dock_widget._on_layers_changed,
                ),
            ]:
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass

        # 6. Remove rubber bands + PRO pending detections
        self._safe_remove_rubber_band(self.mask_rubber_band)
        self.mask_rubber_band = None
        for rb in self.saved_rubber_bands:
            self._safe_remove_rubber_band(rb)
        self.saved_rubber_bands = []
        for det in self._pro_pending_detections:
            self._safe_remove_rubber_band(det.get("rb"))
        self._pro_pending_detections = []

        # 7. Remove dock widgets (deleteLater cascades all widget signals)
        if self.dock_widget:
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget.deleteLater()
            self.dock_widget = None
        if self.pro_dock_widget:
            self.iface.removeDockWidget(self.pro_dock_widget)
            self.pro_dock_widget.deleteLater()
            self.pro_dock_widget = None

        # 7. Unset map tool
        if self.map_tool:
            try:
                if self.iface.mapCanvas().mapTool() == self.map_tool:
                    self.iface.mapCanvas().unsetMapTool(self.map_tool)
            except RuntimeError:
                pass
            self.map_tool = None

        # 8. Remove menu/toolbar
        from .terralab_menu import (
            remove_from_plugins_menu,
            remove_plugin_from_menu,
        )

        for action_ref in (self.action, self.pro_action):
            if action_ref:
                remove_from_plugins_menu(self.iface, action_ref)
                if self._terralab_menu:
                    remove_plugin_from_menu(
                        self._terralab_menu,
                        action_ref,
                        self.iface.mainWindow(),
                    )
                self.iface.removeToolBarIcon(action_ref)
        self.action = None
        self.pro_action = None
        self._terralab_menu = None

        # 9. Stop log collector
        stop_log_collector()

    def toggle_dock_widget(self, checked: bool):
        if self.dock_widget:
            self.dock_widget.setVisible(checked)

    def _on_dock_visibility_changed(self, visible: bool):
        if self.action:
            self.action.setChecked(visible)

        if visible and not self._initialized:
            self._initialized = True
            self._do_first_time_setup()

    def _toggle_pro_dock_widget(self, checked: bool):
        if self.pro_dock_widget:
            self.pro_dock_widget.setVisible(checked)

    def _on_pro_dock_visibility_changed(self, visible: bool):
        if self.pro_action:
            self.pro_action.setChecked(visible)

    def _do_first_time_setup(self):
        QgsMessageLog.logMessage(
            "Panel opened - checking dependencies...",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        # Clean up legacy SAM1 data (old checkpoint + features cache)
        try:
            from ..core.checkpoint_manager import cleanup_legacy_sam1_data

            cleanup_legacy_sam1_data()
        except Exception:
            pass  # Logged internally, never block startup

        try:
            from ..core.venv_manager import cleanup_old_libs, get_venv_status

            cleanup_old_libs()

            is_ready, message = get_venv_status()

            if is_ready:
                self.dock_widget.set_dependency_status(
                    True, "✓ " + tr("Dependencies ready")
                )
                self._show_device_info()
                QgsMessageLog.logMessage(
                    "✓ Virtual environment verified successfully",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Success,
                )
                self._check_checkpoint()
            else:
                self.dock_widget.set_dependency_status(False, message)
                QgsMessageLog.logMessage(
                    f"Virtual environment status: {message}",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )
                # Auto-trigger install for upgrades and CUDA upgrades
                if "GPU acceleration" in message or "need updating" in message:
                    self._on_install_requested()

        except Exception as e:
            import traceback

            error_msg = f"Dependency check error: {str(e)}\n{traceback.format_exc()}"
            QgsMessageLog.logMessage(
                error_msg, "AI Segmentation", level=Qgis.MessageLevel.Warning
            )
            self.dock_widget.set_dependency_status(False, f"Error: {str(e)[:50]}")

        # Check for plugin updates with retries - QGIS repo metadata may not
        # be ready after just a few seconds, especially on slower connections.
        from qgis.PyQt.QtCore import QTimer

        self._update_check_delays = [5000, 30000, 60000, 120000]
        self._update_check_index = 0
        QTimer.singleShot(self._update_check_delays[0], self._check_for_plugin_update)

    def _check_for_plugin_update(self):
        """Trigger the update check on the dock widget, retrying if needed."""
        if not self.dock_widget:
            return
        self.dock_widget.check_for_updates()

        # If notification is still hidden and we have retries left, schedule next
        if not self.dock_widget.update_notification_widget.isVisible() and hasattr(
            self, "_update_check_delays"
        ):
            self._update_check_index += 1
            if self._update_check_index < len(self._update_check_delays):
                from qgis.PyQt.QtCore import QTimer

                delay = self._update_check_delays[self._update_check_index]
                QTimer.singleShot(delay, self._check_for_plugin_update)

    def _check_checkpoint(self):
        try:
            from ..core.checkpoint_manager import checkpoint_exists

            if checkpoint_exists():
                self.dock_widget.set_checkpoint_status(True, "SAM model ready")
                self._load_predictor()
                self._show_activation_popup_if_needed()
            else:
                # Model missing but deps ok: show install button for model download
                self.dock_widget.set_dependency_status(
                    True, tr("Dependencies ready, model not downloaded")
                )
                self.dock_widget.install_button.setVisible(True)
                self.dock_widget.install_button.setEnabled(True)
                self.dock_widget.install_button.setText(tr("Download Model"))
                self.dock_widget.setup_group.setVisible(True)

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Checkpoint check error: {str(e)}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )

    def _load_predictor(self):
        try:
            from ..core.checkpoint_manager import get_checkpoint_path
            from ..core.sam_predictor import SamPredictor, build_sam_predictor_config

            checkpoint_path = get_checkpoint_path()
            sam_config = build_sam_predictor_config(checkpoint=checkpoint_path)
            self.predictor = SamPredictor(sam_config)

            QgsMessageLog.logMessage(
                "SAM predictor initialized (subprocess mode)",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )
            self.dock_widget.set_checkpoint_status(True, "SAM ready (subprocess)")

        except Exception as e:
            import traceback

            QgsMessageLog.logMessage(
                f"Failed to initialize predictor: {str(e)}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )

    def _show_device_info(self):
        """Detect and display which compute device will be used."""
        try:
            from ..core.venv_manager import ensure_venv_packages_available

            ensure_venv_packages_available()

            from ..core.device_manager import get_device_info

            info = get_device_info()
            QgsMessageLog.logMessage(
                f"Device info: {info}", "AI Segmentation", level=Qgis.MessageLevel.Info
            )
        except RuntimeError as e:
            error_str = str(e)
            # Check if this is a PyTorch DLL error (Windows)
            if "DLL" in error_str or "shm.dll" in error_str:
                QgsMessageLog.logMessage(
                    f"PyTorch DLL error: {error_str}",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Critical,
                )
                # Show user-friendly error dialog
                from qgis.PyQt.QtWidgets import QMessageBox

                msg_box = QMessageBox(self.iface.mainWindow())
                msg_box.setIcon(QMessageBox.Icon.Critical)
                msg_box.setWindowTitle(tr("PyTorch Error"))
                msg_box.setText(tr("PyTorch cannot load on Windows"))
                msg_box.setInformativeText(
                    tr(
                        "The plugin requires Visual C++ Redistributables to run PyTorch.\n\n"
                        "Please download and install:\n"
                        "https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"
                        "After installation, restart QGIS and try again."
                    )
                )
                msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg_box.exec()
            else:
                raise
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Could not determine device info: {e}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )

    def _on_install_requested(self):
        # Guard: prevent concurrent installs
        if (
            self.deps_install_worker is not None
            and self.deps_install_worker.isRunning()
        ):
            QgsMessageLog.logMessage(
                "Install already in progress, ignoring duplicate request",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            return

        from ..core.venv_manager import get_venv_status

        is_ready, message = get_venv_status()
        if is_ready:
            # Deps already installed, just need model download
            self.dock_widget.set_dependency_status(
                True, "✓ " + tr("Dependencies ready")
            )
            self._auto_download_checkpoint()
            return

        QgsMessageLog.logMessage(
            "Starting virtual environment creation and dependency installation...",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
        QgsMessageLog.logMessage(
            f"Platform: {sys.platform}, Python: {sys.version}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        self.dock_widget.set_install_progress(0, "Preparing installation...")

        self.deps_install_worker = DepsInstallWorker(cuda_enabled=False)
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
        # Scale deps progress to 0-80% (model download gets 80-100%)
        scaled = int(percent * 0.8)
        self.dock_widget.set_install_progress(scaled, message)

    def _on_deps_install_finished(self, success: bool, message: str):
        if not self.dock_widget:
            return

        if success:
            # GPU driver too old: just show a simple message bar, no error dialog
            if "DRIVER_TOO_OLD" in message:
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("Using CPU mode (GPU driver needs update)."),
                    level=Qgis.MessageLevel.Info,
                    duration=10,
                )
            elif "CUDA_FALLBACK" in message:
                self.dock_widget.install_button.setEnabled(False)
                fallback_msg = "{}\n\n{}".format(
                    tr("Your GPU was detected but CUDA installation didn't work."),
                    tr(
                        "No worries, the plugin now uses CPU mode and everything works fine :) "
                        "If you'd like us to fix GPU support for your setup, send us your logs!"
                    ),
                )
                show_error_report(
                    self.iface.mainWindow(),
                    tr("GPU mode failed, using CPU"),
                    fallback_msg,
                )

            # Run verification + device detection off main thread
            self.dock_widget.set_install_progress(80, tr("Verifying installation..."))
            self._verify_worker = VerifyWorker()
            self._verify_worker.progress.connect(self._on_verify_progress)
            self._verify_worker.finished.connect(self._on_verify_finished)
            self._verify_worker.start()
        else:
            self.dock_widget.set_install_progress(100, "Failed")
            error_msg = message[:300] if message else tr("Unknown error")
            self.dock_widget.set_dependency_status(False, tr("Installation failed"))

            error_title = tr("Installation Failed")
            msg_lower = message.lower() if message else ""
            if any(
                p in msg_lower
                for p in [
                    "ssl",
                    "certificate verify",
                    "sslerror",
                    "unable to get local issuer",
                ]
            ):
                error_title = tr("SSL Certificate Error")
            elif any(
                p in msg_lower
                for p in [
                    "access is denied",
                    "winerror 5",
                    "winerror 225",
                    "permission denied",
                    "blocked",
                    "cannot write to install",
                ]
            ):
                error_title = tr("Installation Blocked")
                error_msg = "{}\n\n{}".format(
                    error_msg, _get_change_path_instructions()
                )

            show_error_report(self.iface.mainWindow(), error_title, error_msg)

    def _on_verify_progress(self, percent: int, message: str):
        if not self.dock_widget:
            return
        # Scale verify progress (0-100%) into the 80-95% range
        scaled = 80 + int(percent * 0.15)
        self.dock_widget.set_install_progress(scaled, message)

    def _on_verify_finished(self, is_valid: bool, message: str):
        if not self.dock_widget:
            return
        if is_valid:
            self.dock_widget.set_dependency_status(
                True, "✓ " + tr("Dependencies ready")
            )
            if message and not message.startswith("device_error"):
                QgsMessageLog.logMessage(
                    "Device info: {}".format(message),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )
            elif message.startswith("device_error"):
                QgsMessageLog.logMessage(
                    "Could not determine device info: {}".format(
                        message.replace("device_error: ", "")
                    ),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Warning,
                )
            try:
                self._auto_download_checkpoint()
            except Exception as e:
                QgsMessageLog.logMessage(
                    "Auto-download checkpoint failed: {}".format(e),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Warning,
                )
                self.dock_widget.set_install_progress(100, "Failed")
                self.dock_widget.set_dependency_status(
                    True, tr("Dependencies ready, model download failed")
                )
        else:
            self.dock_widget.set_install_progress(100, "Failed")
            self.dock_widget.set_dependency_status(
                False, "{} {}".format(tr("Verification failed:"), message)
            )
            show_error_report(
                self.iface.mainWindow(),
                tr("Verification Failed"),
                "{}\n{}".format(
                    tr("Virtual environment was created but verification failed:"),
                    message,
                ),
            )

    def _on_cancel_install(self):
        if self.deps_install_worker and self.deps_install_worker.isRunning():
            self.deps_install_worker.cancel()
            QgsMessageLog.logMessage(
                "Installation cancelled by user",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )

    def _auto_download_checkpoint(self):
        """Auto-download model after deps install if not already present."""
        from ..core.checkpoint_manager import checkpoint_exists

        try:
            if checkpoint_exists():
                self.dock_widget.set_install_progress(100, "Done")
                self.dock_widget.set_checkpoint_status(True, "SAM model ready")
                self._load_predictor()
                self._show_activation_popup_if_needed()
                return
        except Exception:
            pass

        self.dock_widget.set_install_progress(80, tr("Downloading AI model..."))
        try:
            self.download_worker = DownloadWorker()
            self.download_worker.progress.connect(self._on_download_progress)
            self.download_worker.finished.connect(self._on_download_finished)
            self.download_worker.start()
        except Exception as e:
            QgsMessageLog.logMessage(
                "Failed to start model download: {}".format(e),
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            self.dock_widget.set_install_progress(100, "Failed")
            self.dock_widget.set_dependency_status(
                True, tr("Dependencies ready, model download failed")
            )

    def _on_download_progress(self, percent: int, message: str):
        if not self.dock_widget:
            return
        # Scale model download to 80-100% of the unified progress
        scaled = 80 + int(percent * 0.2)
        self.dock_widget.set_install_progress(scaled, message)

    def _on_download_finished(self, success: bool, message: str):
        if not self.dock_widget:
            return
        if success:
            self.dock_widget.set_install_progress(100, "Done")
            self.dock_widget.set_checkpoint_status(True, "SAM model ready")
            self._load_predictor()
            self._show_activation_popup_if_needed()
        else:
            self.dock_widget.set_install_progress(100, "Failed")
            self.dock_widget.set_dependency_status(
                True, tr("Dependencies ready, model download failed")
            )

            show_error_report(
                self.iface.mainWindow(),
                tr("Download Failed"),
                "{}\n{}".format(tr("Failed to download model:"), message),
            )

    def _on_start_segmentation(self, layer: QgsRasterLayer):
        if self.predictor is None:
            QMessageBox.warning(
                self.iface.mainWindow(),
                tr("Not Ready"),
                tr("Please wait for the SAM model to load."),
            )
            return

        # Validate layer BEFORE resetting session to avoid leaving broken state
        if not self._is_layer_valid(layer):
            QgsMessageLog.logMessage(
                "Layer was deleted before segmentation could start",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            return

        try:
            layer_name = layer.name().replace(" ", "_")
            raster_path = os.path.normcase(layer.source())
        except RuntimeError:
            QgsMessageLog.logMessage(
                "Layer deleted during segmentation start",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            return

        self._reset_session()

        self._current_layer = layer
        self._current_layer_name = layer_name

        # Detect online layer (WMS, XYZ, WMTS, WCS, ArcGIS)
        self._is_online_layer = self._is_online_provider(layer)

        # Detect if layer is non-georeferenced (pixel coordinate mode)
        self._is_non_georeferenced_mode = (
            not self._is_online_layer and not self._is_layer_georeferenced(layer)
        )
        if self._is_non_georeferenced_mode:
            QgsMessageLog.logMessage(
                "Non-georeferenced image detected - using pixel coordinate mode. "
                "Polygons will be created in pixel coordinates.",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )

        if self._is_online_layer:
            QgsMessageLog.logMessage(
                "Online layer detected ({})".format(layer.dataProvider().name()),
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )

        # Validate layer extent
        if not self._is_online_layer:
            try:
                ext = layer.extent()
                if ext and not ext.isEmpty():
                    coords = (
                        ext.xMinimum(),
                        ext.yMinimum(),
                        ext.xMaximum(),
                        ext.yMaximum(),
                    )
                    if any(math.isnan(c) or math.isinf(c) for c in coords):
                        show_error_report(
                            self.iface.mainWindow(),
                            tr("Invalid Layer"),
                            tr(
                                "Layer extent contains invalid coordinates "
                                "(NaN/Inf). Check the raster file."
                            ),
                        )
                        return
            except RuntimeError:
                pass

        # Store raster path for on-demand crop extraction
        self._current_raster_path = raster_path

        # Set up CRS transforms (canvas CRS <-> raster CRS)
        canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        raster_crs = layer.crs() if layer else None
        self._canvas_to_raster_xform = None
        self._raster_to_canvas_xform = None
        if raster_crs and canvas_crs.isValid() and raster_crs.isValid():
            if canvas_crs != raster_crs:
                self._canvas_to_raster_xform = QgsCoordinateTransform(
                    canvas_crs, raster_crs, QgsProject.instance()
                )
                self._raster_to_canvas_xform = QgsCoordinateTransform(
                    raster_crs, canvas_crs, QgsProject.instance()
                )
                QgsMessageLog.logMessage(
                    "CRS transform enabled: {} -> {}".format(
                        canvas_crs.authid(), raster_crs.authid()
                    ),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )

        # Pre-warm the worker subprocess so SAM model loads while the
        # user positions their first click (reduces first-click latency)
        self.predictor.warm_up()

        # Activate segmentation tool immediately (no pre-encoding)
        self._activate_segmentation_tool()

    def _activate_segmentation_tool(self):
        # Save the current map tool to restore it later
        current_tool = self.iface.mapCanvas().mapTool()
        if current_tool and current_tool != self.map_tool:
            self._previous_map_tool = current_tool

        self.iface.mapCanvas().setMapTool(self.map_tool)
        self._active_dock.set_segmentation_active(True)
        QgsMessageLog.logMessage(
            "_activate_segmentation_tool: mode={}, dock={}".format(
                self._active_mode, type(self._active_dock).__name__
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
        # Status bar hint will be set by _update_status_hint via set_point_count

        # Install keyboard shortcut filter on the main window so shortcuts
        # work regardless of which widget has focus (the canvas loses focus
        # after encoding/prediction because dock widget updates steal it).
        if self._shortcut_filter is None:
            self._shortcut_filter = ShortcutFilter(self)
        app = QApplication.instance()
        if app:
            app.installEventFilter(self._shortcut_filter)

        # Show tutorial notification for first-time users
        self._show_tutorial_notification()

    def _get_next_mask_counter(self) -> int:
        """Return the next available mask number, checking existing project layers."""
        existing_numbers = set()
        for layer in QgsProject.instance().mapLayers().values():
            lname = layer.name()
            if lname.startswith("mask_"):
                try:
                    existing_numbers.add(int(lname.split("_", 1)[1]))
                except (ValueError, IndexError):
                    pass
        counter = 1
        while counter in existing_numbers:
            counter += 1
        return counter

    def _show_tutorial_notification(self):
        """Show YouTube tutorial notification (once ever, persisted in QSettings)."""
        settings = QSettings()
        if settings.value(SETTINGS_KEY_TUTORIAL_SHOWN, False, type=bool):
            return
        settings.setValue(SETTINGS_KEY_TUTORIAL_SHOWN, True)

        tutorial_url = (
            "https://www.youtube.com/playlist"
            "?list=PL4hCF043nAUW2iIxALNUzy1fKHcCWwDsv&jct=GTA3Fx8pJzuTLPPivC9RRQ"
        )
        message = '{} <a href="{}">{}</a>'.format(
            tr("New to AI Segmentation?"), tutorial_url, tr("Watch our tutorial")
        )

        self.iface.messageBar().pushMessage(
            "AI Segmentation", message, level=Qgis.MessageLevel.Info, duration=10
        )

    def _on_batch_mode_changed(self, batch: bool):
        """Handle batch mode toggle (no-op, batch mode is always on)."""
        pass

    def _on_layer_combo_changed(self, layer):
        """Handle layer selection change in the combo box."""
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
            return

        # Different layer selected while segmenting
        if self.iface.mapCanvas().mapTool() == self.map_tool:
            has_unsaved_mask = self.current_mask is not None
            has_saved_polygons = len(self.saved_polygons) > 0

            if has_unsaved_mask or has_saved_polygons:
                polygon_count = len(self.saved_polygons)
                if has_unsaved_mask:
                    polygon_count += 1
                message = "{}\n\n{}".format(
                    tr("You have {count} unsaved polygon(s).").format(
                        count=polygon_count
                    ),
                    tr(
                        "Changing layer will discard your current segmentation. Continue?"
                    ),
                )

                reply = QMessageBox.warning(
                    self.iface.mainWindow(),
                    tr("Change Layer?"),
                    message,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )

                if reply != QMessageBox.StandardButton.Yes:
                    self.dock_widget.layer_combo.blockSignals(True)
                    self.dock_widget.layer_combo.setLayer(self._current_layer)
                    self.dock_widget.layer_combo.blockSignals(False)
                    return

            self._stopping_segmentation = True
            self.iface.mapCanvas().unsetMapTool(self.map_tool)
            self._restore_previous_map_tool()
            self._stopping_segmentation = False
            self._reset_session()
            self.dock_widget.reset_session()

    def _on_save_polygon(self):
        """Save current mask as polygon."""
        if self._encoding_in_progress:
            return
        if self.current_mask is None or self.current_transform_info is None:
            return

        self._ensure_polygon_rubberband_sync()

        from ..core.polygon_exporter import apply_mask_refinement, mask_to_polygons

        # Apply all mask-level refinements for display (green shows with effects)
        mask_for_display = self.current_mask
        if (
            self._refine_fill_holes
            or self._refine_min_area > 0
            or self._refine_expand != 0
        ):
            mask_for_display = apply_mask_refinement(
                self.current_mask,
                expand_value=self._refine_expand,
                fill_holes=self._refine_fill_holes,
                min_area=self._refine_min_area,
            )

        geometries = mask_to_polygons(mask_for_display, self.current_transform_info)

        if geometries:
            combined = QgsGeometry.unaryUnion(geometries)
        else:
            combined = None

        if combined and not combined.isEmpty():
            # Apply simplification if enabled
            tolerance = self._compute_simplification_tolerance(
                self.current_transform_info, self._refine_simplify
            )
            if tolerance > 0:
                combined = combined.simplify(tolerance)

            # Store WKT (with effects), score, transform info, raw mask, points, and refine settings
            self.saved_polygons.append(
                {
                    "geometry_wkt": combined.asWkt(),
                    "score": self.current_score,
                    "transform_info": self.current_transform_info.copy()
                    if self.current_transform_info
                    else None,
                    "raw_mask": self.current_mask.copy(),  # Store RAW mask for re-applying different settings
                    "points_positive": list(
                        self.prompts.positive_points
                    ),  # Points for undo restoration
                    "points_negative": list(
                        self.prompts.negative_points
                    ),  # Points for undo restoration
                    "refine_expand": self._refine_expand,  # Refine settings at save time
                    "refine_simplify": self._refine_simplify,  # Refine settings at save time
                    "refine_fill_holes": self._refine_fill_holes,  # Refine settings at save time
                    "refine_min_area": self._refine_min_area,  # Refine settings at save time
                }
            )

            saved_rb = QgsRubberBand(
                self.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry
            )
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
                level=Qgis.MessageLevel.Info,
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

        # Keep crop info so clicks in the same area reuse the encoding
        # (re-encode only happens when point falls outside current crop bounds)

    def _on_export_layer(self):
        """Export all saved polygons + current unsaved mask to a new layer."""
        if self._exporting_in_progress:
            return
        self._exporting_in_progress = True
        try:
            self._on_export_layer_impl()
        finally:
            self._exporting_in_progress = False

    def _on_export_layer_impl(self):
        """Internal export implementation."""
        from ..core.polygon_exporter import apply_mask_refinement, mask_to_polygons

        self._ensure_polygon_rubberband_sync()

        if not self.saved_polygons and self.current_mask is None:
            return  # Nothing to export

        polygons_to_export = list(self.saved_polygons)

        # Include current unsaved mask if exists (apply current refine settings)
        if self.current_mask is not None and self.current_transform_info is not None:
            mask_to_export = self.current_mask
            if (
                self._refine_fill_holes
                or self._refine_min_area > 0
                or self._refine_expand != 0
            ):
                mask_to_export = apply_mask_refinement(
                    self.current_mask,
                    expand_value=self._refine_expand,
                    fill_holes=self._refine_fill_holes,
                    min_area=self._refine_min_area,
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
                    polygons_to_export.append(
                        {
                            "geometry_wkt": combined.asWkt(),
                            "score": self.current_score,
                            "transform_info": self.current_transform_info.copy()
                            if self.current_transform_info
                            else None,
                        }
                    )

        self._stopping_segmentation = True
        self.iface.mapCanvas().unsetMapTool(self.map_tool)
        self._restore_previous_map_tool()
        self._stopping_segmentation = False

        # Generate layer name: mask_{number}
        mask_num = self._get_next_mask_counter()
        layer_name = f"mask_{mask_num}"

        # Determine CRS
        # For non-georeferenced images, use a local pixel-based CRS
        if self._is_non_georeferenced_mode:
            # Use EPSG:3857 (Web Mercator) with pixel coordinates
            # This allows visualization while being clear it's not true geographic data
            crs = QgsCoordinateReferenceSystem("EPSG:3857")
            QgsMessageLog.logMessage(
                "Non-georeferenced mode: Using EPSG:3857 with pixel coordinates",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )
        else:
            # Normal georeferenced mode
            crs_str = None
            for pg in polygons_to_export:
                ti = pg.get("transform_info")
                if ti:
                    crs_str = ti.get("crs", None)
                    if isinstance(crs_str, str) and crs_str.strip():
                        break
                    crs_str = None
            if crs_str is None and self.current_transform_info:
                val = self.current_transform_info.get("crs", None)
                if isinstance(val, str) and val.strip():
                    crs_str = val
            if crs_str is None:
                try:
                    if self._is_layer_valid() and self._current_layer.crs().isValid():
                        crs_str = self._current_layer.crs().authid()
                except RuntimeError:
                    pass
            if isinstance(crs_str, str) and crs_str.strip():
                crs = QgsCoordinateReferenceSystem(crs_str)
            else:
                crs = QgsCoordinateReferenceSystem("EPSG:4326")

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
                tr("Could not create the output layer."),
            )
            return

        temp_layer.setCrs(crs)

        # Add attributes
        pr = temp_layer.dataProvider()
        pr.addAttributes(
            [
                QgsField("id", QVariant.Int),
                QgsField("score", QVariant.Double),
                QgsField("area", QVariant.Double),
            ]
        )
        temp_layer.updateFields()

        # Add features to temp layer
        features_to_add = []
        for i, polygon_data in enumerate(polygons_to_export):
            feature = QgsFeature(temp_layer.fields())

            # Reconstruct geometry from WKT
            geom_wkt = polygon_data.get("geometry_wkt")
            if not geom_wkt:
                QgsMessageLog.logMessage(
                    f"Polygon {i + 1} has no WKT data",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Warning,
                )
                continue

            geom = QgsGeometry.fromWkt(geom_wkt)

            if geom and not geom.isEmpty():
                # Ensure geometry is MultiPolygon
                if not geom.isMultipart():
                    geom.convertToMultiType()

                feature.setGeometry(geom)
                area = geom.area()
                feature.setAttributes([i + 1, polygon_data["score"], area])
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
            temp_layer, gpkg_path, QgsProject.instance().transformContext(), options
        )

        if error[0] != QgsVectorFileWriter.NoError:
            QgsMessageLog.logMessage(
                f"Failed to save GeoPackage: {error[1]}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            show_error_report(
                self.iface.mainWindow(),
                tr("Export Failed"),
                "{}\n{}".format(tr("Could not save layer to file:"), error[1]),
            )
            return

        # Load the saved GeoPackage as a permanent layer
        result_layer = QgsVectorLayer(gpkg_path, layer_name, "ogr")
        if not result_layer.isValid():
            QgsMessageLog.logMessage(
                f"Failed to load saved GeoPackage: {gpkg_path}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            show_error_report(
                self.iface.mainWindow(),
                tr("Load Failed"),
                "{}\n{}".format(
                    tr("Layer was saved but could not be loaded:"), gpkg_path
                ),
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
            level=Qgis.MessageLevel.Info,
        )
        QgsMessageLog.logMessage(
            f"Layer CRS: {result_layer.crs().authid()}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
        QgsMessageLog.logMessage(
            f"Saved to: {gpkg_path}", "AI Segmentation", level=Qgis.MessageLevel.Info
        )

        # Set renderer - red thin outline, transparent fill
        symbol = QgsFillSymbol.createSimple(
            {"color": "0,0,0,0", "outline_color": "220,0,0,255", "outline_width": "0.5"}
        )
        renderer = QgsSingleSymbolRenderer(symbol)
        result_layer.setRenderer(renderer)
        result_layer.triggerRepaint()
        self.iface.mapCanvas().refresh()

        QgsMessageLog.logMessage(
            f"Created segmentation layer: {layer_name} with {len(features_to_add)} polygons",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
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
        # Remove keyboard shortcut filter
        try:
            if self._shortcut_filter is not None:
                app = QApplication.instance()
                if app:
                    app.removeEventFilter(self._shortcut_filter)
        except (RuntimeError, AttributeError):
            pass

        if self._stopping_segmentation:
            if self.dock_widget:
                self.dock_widget.set_segmentation_active(False)
            return

        # In PRO mode, ask before discarding pending detections
        if self._active_mode == "pro":
            if self._pro_pending_detections:
                reply = QMessageBox.warning(
                    self.iface.mainWindow(),
                    tr("Stop Segmentation?"),
                    "{}\n\n{}".format(
                        tr("This will discard {count} polygon(s).").format(
                            count=len(self._pro_pending_detections)
                        ),
                        tr("Use 'Export to layer' to keep them."),
                    ),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    # Re-activate the segmentation tool to stay in PRO mode
                    if self.map_tool:
                        self._stopping_segmentation = True
                        self.iface.mapCanvas().setMapTool(self.map_tool)
                        self._stopping_segmentation = False
                    return
                # User confirmed: fall through to reset
            else:
                return  # No detections, just keep PRO mode alive

        # Silently reset session and deactivate
        self._reset_session()
        if self.dock_widget:
            self.dock_widget.reset_session()
        if self.pro_dock_widget:
            self.pro_dock_widget.reset_session()
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
        polygon_count += len(self._pro_pending_detections)

        if polygon_count > 0:
            reply = QMessageBox.warning(
                self.iface.mainWindow(),
                tr("Stop Segmentation?"),
                "{}\n\n{}".format(
                    tr("This will discard {count} polygon(s).").format(
                        count=polygon_count
                    ),
                    tr("Use 'Export to layer' to keep them."),
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Cancel any running tiled detection
        if self._tiled_worker and self._tiled_worker.isRunning():
            self._tiled_worker.cancel()
            self._tiled_worker.wait(5000)

        if self._shortcut_filter is not None:
            app = QApplication.instance()
            if app:
                app.removeEventFilter(self._shortcut_filter)
        self._stopping_segmentation = True
        self.iface.mapCanvas().unsetMapTool(self.map_tool)
        self._restore_previous_map_tool()
        self._stopping_segmentation = False
        self._reset_session()
        self.dock_widget.reset_session()
        if self.pro_dock_widget:
            self.pro_dock_widget.reset_session()

    def _on_refine_settings_changed(
        self, expand: int, simplify: int, fill_holes: bool, min_area: int
    ):
        """Handle refinement control changes."""
        QgsMessageLog.logMessage(
            "Refine settings: expand={}, simplify={}, fill_holes={}, min_area={}".format(
                expand, simplify, fill_holes, min_area
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
        self._refine_expand = expand
        self._refine_simplify = simplify
        self._refine_fill_holes = fill_holes
        self._refine_min_area = min_area

        # Standard mode: update current mask preview (yellow rubber band)
        self._update_mask_visualization()
        # PRO mode: re-render all pending detection rubber bands in real-time
        if self._active_mode == "pro" and self._pro_pending_detections:
            self._update_pro_detections_visualization()

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

    def _is_point_in_raster_extent(self, point):
        """Check if a point (in raster CRS) falls within the layer extent."""
        if not self._is_layer_valid():
            return False
        try:
            ext = self._current_layer.extent()
            # Transform extent to raster CRS if needed
            if self._canvas_to_raster_xform is not None:
                # Layer extent is in layer CRS, point is already in raster CRS
                pass
            in_x = ext.xMinimum() <= point.x() <= ext.xMaximum()
            in_y = ext.yMinimum() <= point.y() <= ext.yMaximum()
            return in_x and in_y
        except RuntimeError:
            return False

    def _check_crop_status(self, point):
        """Check if a point (in raster CRS) is usable in the current crop.

        Returns a reason code:
        - "ok": point is inside the crop
        - "no_crop": no crop has been encoded yet
        - "outside_bounds": point is geographically outside the crop
        - "zoom_changed": user zoomed in significantly, crop should be re-encoded
        """
        if self._current_crop_info is None:
            return "no_crop"
        bounds = self._current_crop_info["bounds"]
        in_x = bounds[0] <= point.x() <= bounds[2]
        in_y = bounds[1] <= point.y() <= bounds[3]
        if not (in_x and in_y):
            return "outside_bounds"

        # Detect significant zoom-in requiring higher resolution
        if self._is_online_layer:
            canvas = self.iface.mapCanvas()
            current_canvas_mupp = canvas.mapUnitsPerPixel()
            if self._current_crop_canvas_mupp and current_canvas_mupp > 0:
                if current_canvas_mupp < 0.7 * self._current_crop_canvas_mupp:
                    return "zoom_changed"
                if current_canvas_mupp > 1.5 * self._current_crop_canvas_mupp:
                    return "zoom_changed"
        else:
            if self._current_crop_canvas_mupp is not None:
                canvas = self.iface.mapCanvas()
                current_mupp = canvas.mapUnitsPerPixel()
                if current_mupp > 0:
                    if current_mupp < 0.7 * self._current_crop_canvas_mupp:
                        return "zoom_changed"
                    if current_mupp > 1.5 * self._current_crop_canvas_mupp:
                        return "zoom_changed"

        return "ok"

    def _is_point_in_current_crop(self, point):
        """Check if a point is usable in the current crop (backward compat)."""
        return self._check_crop_status(point) == "ok"

    @staticmethod
    def _resize_nearest(arr, target_h, target_w):
        """Resize a 2D numpy array using nearest-neighbor interpolation."""
        import numpy as np

        src_h, src_w = arr.shape
        row_idx = (np.arange(target_h) * src_h / target_h).astype(int)
        col_idx = (np.arange(target_w) * src_w / target_w).astype(int)
        np.clip(row_idx, 0, src_h - 1, out=row_idx)
        np.clip(col_idx, 0, src_w - 1, out=col_idx)
        return arr[row_idx[:, None], col_idx[None, :]]

    def _build_mask_input_from_previous(
        self, old_mask, old_bounds, old_shape, new_bounds, new_shape
    ):
        """Transfer a binary mask from old crop space to new crop as SAM logits.

        Computes geographic overlap between old and new crops, maps the
        overlapping region, converts to logits, and resizes to (1, 256, 256).
        Returns None if there is no overlap.
        """
        import numpy as np

        old_minx, old_miny, old_maxx, old_maxy = old_bounds
        new_minx, new_miny, new_maxx, new_maxy = new_bounds

        # Geographic overlap
        ovlp_minx = max(old_minx, new_minx)
        ovlp_miny = max(old_miny, new_miny)
        ovlp_maxx = min(old_maxx, new_maxx)
        ovlp_maxy = min(old_maxy, new_maxy)
        if ovlp_minx >= ovlp_maxx or ovlp_miny >= ovlp_maxy:
            return None

        old_h, old_w = old_shape
        new_h, new_w = new_shape

        def geo_to_pixel(gx, gy, bminx, bminy, bmaxx, bmaxy, pw, ph):
            col = (gx - bminx) / (bmaxx - bminx) * pw
            row = (bmaxy - gy) / (bmaxy - bminy) * ph
            return int(round(col)), int(round(row))

        # Overlap region in old pixel coords
        o_c0, o_r0 = geo_to_pixel(
            ovlp_minx, ovlp_maxy, old_minx, old_miny, old_maxx, old_maxy, old_w, old_h
        )
        o_c1, o_r1 = geo_to_pixel(
            ovlp_maxx, ovlp_miny, old_minx, old_miny, old_maxx, old_maxy, old_w, old_h
        )
        o_r0 = max(0, min(o_r0, old_h))
        o_r1 = max(0, min(o_r1, old_h))
        o_c0 = max(0, min(o_c0, old_w))
        o_c1 = max(0, min(o_c1, old_w))
        if o_r0 >= o_r1 or o_c0 >= o_c1:
            return None

        patch = old_mask[o_r0:o_r1, o_c0:o_c1]

        # Overlap region in new pixel coords
        n_c0, n_r0 = geo_to_pixel(
            ovlp_minx, ovlp_maxy, new_minx, new_miny, new_maxx, new_maxy, new_w, new_h
        )
        n_c1, n_r1 = geo_to_pixel(
            ovlp_maxx, ovlp_miny, new_minx, new_miny, new_maxx, new_maxy, new_w, new_h
        )
        n_r0 = max(0, min(n_r0, new_h))
        n_r1 = max(0, min(n_r1, new_h))
        n_c0 = max(0, min(n_c0, new_w))
        n_c1 = max(0, min(n_c1, new_w))
        target_h = n_r1 - n_r0
        target_w = n_c1 - n_c0
        if target_h < 1 or target_w < 1:
            return None

        resized_patch = self._resize_nearest(patch, target_h, target_w)

        # Place patch into full-size new crop mask
        new_mask = np.zeros((new_h, new_w), dtype=np.float32)
        new_mask[n_r0:n_r1, n_c0:n_c1] = resized_patch

        # Convert binary mask to logits: foreground=+6, background=-6
        logits = (new_mask * 2.0 - 1.0) * 6.0

        # Resize to SAM's low-res mask size (256x256), shape (1, 1, 256, 256)
        logits_256 = self._resize_nearest(logits, 256, 256)
        return logits_256[None, None, :, :]

    def _compute_crop_center_and_mupp(self, all_points_geo, crop_size=1024):
        """Compute optimal crop center and mupp to fit all points.

        When points span more than crop_size pixels at the current resolution,
        this zooms out (increases mupp) so all points fit in one image.

        Args:
            all_points_geo: list of (x, y) tuples in raster CRS
            crop_size: target image size in pixels (1024)

        Returns:
            (center_x, center_y, mupp_or_scale) where the third value is:
            - For online layers: the mupp needed (at least canvas mupp)
            - For file-based layers: the scale_factor (>= 1.0)
        """
        xs = [p[0] for p in all_points_geo]
        ys = [p[1] for p in all_points_geo]
        bbox_width = max(xs) - min(xs)
        bbox_height = max(ys) - min(ys)
        center_x = (min(xs) + max(xs)) / 2.0
        center_y = (min(ys) + max(ys)) / 2.0

        # Add 20% margin on each side (1.4x total)
        needed_geo_size = max(bbox_width, bbox_height) * 1.4

        if self._is_online_layer:
            canvas_mupp = self.iface.mapCanvas().mapUnitsPerPixel()
            # mupp must cover at least the needed area, but not less than canvas
            needed_mupp = needed_geo_size / crop_size
            mupp = max(canvas_mupp, needed_mupp)
            return center_x, center_y, mupp
        else:
            # For file-based layers, compute scale_factor from native pixel size
            native_pixel_size = self._get_native_pixel_size()
            if native_pixel_size > 0:
                needed_mupp = needed_geo_size / crop_size
                scale_factor = max(1.0, needed_mupp / native_pixel_size)
            else:
                scale_factor = 1.0
            return center_x, center_y, scale_factor

    def _get_native_pixel_size(self):
        """Get the native pixel size of the current file-based raster layer."""
        try:
            ext = self._current_layer.extent()
            w = self._current_layer.width()
            h = self._current_layer.height()
            if w > 0 and h > 0:
                px = (ext.xMaximum() - ext.xMinimum()) / w
                py = (ext.yMaximum() - ext.yMinimum()) / h
                return max(px, py)
        except (RuntimeError, AttributeError):
            pass
        return 0.0

    def _compute_initial_scale_factor(self):
        """Compute initial scale_factor from canvas zoom for file-based rasters.

        For high-res imagery where the user is zoomed out, the crop should cover
        a proportionally larger geographic area instead of just 1024 native pixels.

        Uses canvas extent in raster CRS to avoid unit mismatches (e.g. canvas
        in meters vs raster in degrees).
        """
        if self._is_online_layer:
            return None
        native_pixel_size = self._get_native_pixel_size()
        if native_pixel_size <= 0:
            return None

        canvas = self.iface.mapCanvas()
        canvas_extent = canvas.extent()
        # Transform canvas extent to raster CRS if needed
        if self._canvas_to_raster_xform is not None:
            try:
                canvas_extent = self._canvas_to_raster_xform.transformBoundingBox(
                    canvas_extent
                )
            except Exception:
                return None

        # Compute canvas mupp in raster CRS units
        canvas_width_px = canvas.width()
        if canvas_width_px <= 0:
            return None
        canvas_geo_width = canvas_extent.xMaximum() - canvas_extent.xMinimum()
        canvas_mupp_raster_crs = canvas_geo_width / canvas_width_px

        ratio = canvas_mupp_raster_crs / native_pixel_size
        return max(0.25, min(ratio, 8.0))

    def _extract_and_encode_crop(self, center_point, mupp_override=None):
        """Extract a crop centered on the point and encode it with SAM.

        Args:
            center_point: QgsPointXY center in raster CRS
            mupp_override: For online layers, override mupp (zoom-out).
                For file-based layers, this is the scale_factor [0.25, 8.0].

        Returns True on success, False on error.
        """
        if self._encoding_in_progress:
            return False
        self._encoding_in_progress = True

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            return self._do_extract_and_encode(center_point, mupp_override)
        finally:
            QApplication.restoreOverrideCursor()

    def _do_extract_and_encode(self, center_point, mupp_override):
        """Internal: does the actual crop extraction + SAM encoding."""
        from ..core.feature_encoder import (
            extract_crop_from_online_layer,
            extract_crop_from_raster,
        )

        raster_pt_x = center_point.x()
        raster_pt_y = center_point.y()

        if self._is_online_layer:
            canvas = self.iface.mapCanvas()
            canvas_mupp = canvas.mapUnitsPerPixel()
            # When canvas CRS != raster CRS, the MUPP is in canvas units
            # (e.g. degrees) but crop is in raster units (e.g. meters).
            # Convert by measuring a small canvas-pixel offset in raster CRS.
            if self._canvas_to_raster_xform is not None:
                canvas_center = canvas.center()
                cx, cy = canvas_center.x(), canvas_center.y()
                p1 = self._canvas_to_raster_xform.transform(QgsPointXY(cx, cy))
                p2 = self._canvas_to_raster_xform.transform(
                    QgsPointXY(cx + canvas_mupp, cy)
                )
                raster_mupp = math.sqrt((p2.x() - p1.x()) ** 2 + (p2.y() - p1.y()) ** 2)
            else:
                raster_mupp = canvas_mupp
            self._current_crop_canvas_mupp = canvas_mupp
            actual_mupp = mupp_override if mupp_override else raster_mupp
            self._current_crop_actual_mupp = actual_mupp
            skip_stretch = self._active_mode == "pro"
            image_np, crop_info, error = extract_crop_from_online_layer(
                self._current_layer,
                raster_pt_x,
                raster_pt_y,
                actual_mupp,
                crop_size=1024,
                skip_stretch=skip_stretch,
            )
        elif not self._current_raster_path:
            self._encoding_in_progress = False
            show_error_report(
                self.iface.mainWindow(),
                tr("Crop Error"),
                tr("No raster file path available. Please restart segmentation."),
            )
            return False
        else:
            layer_crs_wkt = None
            layer_extent = None
            try:
                if self._current_layer.crs().isValid():
                    layer_crs_wkt = self._current_layer.crs().toWkt()
                ext = self._current_layer.extent()
                if ext and not ext.isEmpty():
                    layer_extent = (
                        ext.xMinimum(),
                        ext.yMinimum(),
                        ext.xMaximum(),
                        ext.yMaximum(),
                    )
            except RuntimeError:
                pass

            scale_factor = mupp_override if mupp_override else 1.0
            self._current_crop_scale_factor = scale_factor
            self._current_crop_canvas_mupp = self.iface.mapCanvas().mapUnitsPerPixel()
            skip_stretch = self._active_mode == "pro"
            image_np, crop_info, error = extract_crop_from_raster(
                self._current_raster_path,
                raster_pt_x,
                raster_pt_y,
                crop_size=1024,
                layer_crs_wkt=layer_crs_wkt,
                layer_extent=layer_extent,
                scale_factor=scale_factor,
                skip_stretch=skip_stretch,
            )

        if error:
            self._encoding_in_progress = False
            QgsMessageLog.logMessage(
                "Crop extraction failed: {}".format(error),
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical,
            )
            show_error_report(self.iface.mainWindow(), tr("Crop Error"), error)
            return False

        # Encode the crop with SAM (this blocks for ~3-8s on CPU)
        try:
            self.predictor.set_image(image_np)
        except Exception as e:
            self._encoding_in_progress = False
            QgsMessageLog.logMessage(
                "Image encoding failed: {}".format(str(e)),
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical,
            )
            show_error_report(self.iface.mainWindow(), tr("Encoding Error"), str(e))
            return False

        self._current_crop_info = crop_info
        self._encoding_in_progress = False

        QgsMessageLog.logMessage(
            "Encoded crop: bounds={}, shape={}".format(
                crop_info["bounds"], crop_info["img_shape"]
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
        return True

    def _handle_reencode(self, crop_status, raster_pt):
        """Handle re-encoding based on crop status. Returns True on success."""
        if crop_status == "no_crop":
            self.current_low_res_mask = None
            initial_scale = self._compute_initial_scale_factor()
            return self._extract_and_encode_crop(raster_pt, mupp_override=initial_scale)

        # outside_bounds: save old state, re-encode to fit all points
        old_crop_info = self._current_crop_info
        old_mask = self.current_mask

        all_pts = [
            (p[0], p[1])
            for p in self.prompts.positive_points + self.prompts.negative_points
        ]
        if len(all_pts) > 1:
            center_x, center_y, mupp_or_scale = self._compute_crop_center_and_mupp(
                all_pts
            )
            new_center = QgsPointXY(center_x, center_y)
        else:
            new_center = raster_pt
            mupp_or_scale = None
        self.current_low_res_mask = None
        if not self._extract_and_encode_crop(new_center, mupp_override=mupp_or_scale):
            return False

        # Transfer previous mask as context to the new crop
        if old_mask is not None and old_crop_info is not None:
            transferred = self._build_mask_input_from_previous(
                old_mask.astype(float),
                old_crop_info["bounds"],
                old_crop_info["img_shape"],
                self._current_crop_info["bounds"],
                self._current_crop_info["img_shape"],
            )
            if transferred is not None:
                self.current_low_res_mask = transferred

        return True

    def _on_positive_click(self, point):
        """Handle left-click: add positive point (select this element)."""
        if self.predictor is None:
            if self.map_tool:
                self.map_tool.remove_last_marker()
            return

        # Transform click from canvas CRS to raster CRS for all downstream use
        raster_pt = self._transform_to_raster_crs(point)

        if not self._is_point_in_raster_extent(raster_pt):
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Point is outside the raster image. Click inside the raster."),
                level=Qgis.MessageLevel.Warning,
                duration=4,
            )
            return

        # Register point in prompts immediately so it stays in sync with
        # the marker already added by canvasPressEvent.  This lets Cmd+Z
        # work at any time, even during the (blocking) encoding below.
        self.prompts.add_positive_point(raster_pt.x(), raster_pt.y())

        QgsMessageLog.logMessage(
            "POSITIVE POINT at ({:.6f}, {:.6f})".format(raster_pt.x(), raster_pt.y()),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        # On-demand encoding: encode crop if needed
        crop_status = self._check_crop_status(raster_pt)

        if crop_status != "ok":
            if not self._handle_reencode(crop_status, raster_pt):
                self.prompts.undo()
                if self.map_tool:
                    self.map_tool.remove_last_marker()
                return

        self._run_prediction()

    def _on_negative_click(self, point):
        """Handle right-click: add negative point (exclude this area)."""
        if self.predictor is None:
            if self.map_tool:
                self.map_tool.remove_last_marker()
            return

        # Block negative points until at least one positive point exists
        if len(self.prompts.positive_points) == 0:
            if self.map_tool:
                self.map_tool.remove_last_marker()
            QgsMessageLog.logMessage(
                "Negative point ignored - need at least one positive point first",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )
            return

        # Transform click from canvas CRS to raster CRS for all downstream use
        raster_pt = self._transform_to_raster_crs(point)

        if not self._is_point_in_raster_extent(raster_pt):
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Point is outside the raster image. Click inside the raster."),
                level=Qgis.MessageLevel.Warning,
                duration=4,
            )
            return

        # Register point in prompts immediately (sync with marker)
        self.prompts.add_negative_point(raster_pt.x(), raster_pt.y())

        QgsMessageLog.logMessage(
            "NEGATIVE POINT at ({:.6f}, {:.6f})".format(raster_pt.x(), raster_pt.y()),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        # Re-encode if needed (zoom changed or point outside crop)
        crop_status = self._check_crop_status(raster_pt)

        if crop_status != "ok":
            if not self._handle_reencode(crop_status, raster_pt):
                self.prompts.undo()
                if self.map_tool:
                    self.map_tool.remove_last_marker()
                return

        self._run_prediction()

    def _run_prediction(self):
        """Run SAM prediction using all positive and negative points."""
        from rasterio.transform import from_bounds as transform_from_bounds

        all_points = self.prompts.positive_points + self.prompts.negative_points
        if not all_points:
            return

        if self._current_crop_info is None:
            QgsMessageLog.logMessage(
                "No crop encoded yet - cannot predict",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            return

        crop_bounds = self._current_crop_info["bounds"]
        img_shape = self._current_crop_info["img_shape"]
        img_height, img_width = img_shape

        minx, miny, maxx, maxy = crop_bounds
        img_clip_transform = transform_from_bounds(
            minx, miny, maxx, maxy, img_width, img_height
        )

        point_coords, point_labels = self.prompts.get_points_for_predictor(
            img_clip_transform
        )

        if point_coords is None:
            return

        # Use previous low_res_mask for iterative refinement (includes
        # transferred mask context after zoom re-encode)
        mask_input = None
        if self.current_low_res_mask is not None:
            mask_input = self.current_low_res_mask

        # Use multimask only on the very first point of a new polygon
        # (more accurate initial selection). For subsequent points or
        # re-encoded crops with transferred mask, use single mask.
        one_positive = len(self.prompts.positive_points) == 1
        no_negatives = len(self.prompts.negative_points) == 0
        is_first_point = one_positive and no_negatives and mask_input is None
        use_multimask = is_first_point

        try:
            masks, scores, low_res_masks = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                multimask_output=use_multimask,
            )
        except RuntimeError as e:
            error_str = str(e)
            QgsMessageLog.logMessage(
                "Prediction failed: {}".format(error_str),
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical,
            )
            if "DLL" in error_str or "Visual C++" in error_str:
                msg_box = QMessageBox(self.iface.mainWindow())
                msg_box.setIcon(QMessageBox.Icon.Critical)
                msg_box.setWindowTitle(tr("Prediction Error"))
                msg_box.setText(tr("Segmentation failed"))
                msg_box.setInformativeText(error_str)
                msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg_box.exec()
            return
        except Exception as e:
            QgsMessageLog.logMessage(
                "Unexpected prediction error: {}".format(str(e)),
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical,
            )
            return

        if use_multimask:
            total_pixels = masks[0].shape[0] * masks[0].shape[1]
            mask_areas = [int(m.sum()) for m in masks]

            # Avoid selecting the whole crop when clicking on small elements
            # in repetitive patterns (e.g. trees in an orchard). SAM's highest
            # score often goes to the "all similar elements" interpretation.
            small_enough = [
                i for i in range(len(scores)) if mask_areas[i] < 0.8 * total_pixels
            ]
            if small_enough:
                best_idx = max(small_enough, key=lambda i: scores[i])
            else:
                best_idx = min(range(len(scores)), key=lambda i: mask_areas[i])

            QgsMessageLog.logMessage(
                "Multimask: areas={}, scores={}, picked={}".format(
                    mask_areas, [round(float(s), 3) for s in scores], best_idx
                ),
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )
            self.current_mask = masks[best_idx]
            self.current_score = float(scores[best_idx])
            self.current_low_res_mask = low_res_masks[best_idx : best_idx + 1]
        else:
            self.current_mask = masks[0]
            self.current_score = float(scores[0])
            self.current_low_res_mask = low_res_masks

        # Get CRS from layer
        crs_value = None
        try:
            if self._current_layer and self._current_layer.crs().isValid():
                crs_value = self._current_layer.crs().authid()
        except RuntimeError:
            pass

        self.current_transform_info = {
            "bbox": (minx, maxx, miny, maxy),
            "img_shape": (img_height, img_width),
            "crs": crs_value,
        }

        self._update_ui_after_prediction()

    def _update_ui_after_prediction(self):
        if not self.dock_widget:
            return
        pos_count, neg_count = self.prompts.point_count
        self.dock_widget.set_point_count(pos_count, neg_count)

        if self.current_mask is not None:
            mask_pixels = int(self.current_mask.sum())
            QgsMessageLog.logMessage(
                f"Segmentation result: score={self.current_score:.3f}, mask_pixels={mask_pixels}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
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
            from ..core.polygon_exporter import (
                apply_mask_refinement,
                count_significant_regions,
                mask_to_polygons,
            )

            # Apply refinement to preview in both modes (refine affects current mask only)
            mask_to_display = self.current_mask
            # Apply all mask-level refinements (fill holes, min area, expand/contract)
            if (
                self._refine_fill_holes
                or self._refine_min_area > 0
                or self._refine_expand != 0
            ):
                mask_to_display = apply_mask_refinement(
                    self.current_mask,
                    expand_value=self._refine_expand,
                    fill_holes=self._refine_fill_holes,
                    min_area=self._refine_min_area,
                )

            # Detect disjoint regions and show/hide warning
            region_count = count_significant_regions(mask_to_display)
            self.dock_widget.set_disjoint_warning(region_count > 1)

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
                "Mask visualization error ({}): {}".format(type(e).__name__, str(e)),
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            self._clear_mask_visualization()
        except Exception as e:
            import traceback

            QgsMessageLog.logMessage(
                "Unexpected mask visualization error ({}): {}\n{}".format(
                    type(e).__name__, str(e), traceback.format_exc()
                ),
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical,
            )
            self._clear_mask_visualization()

    def _clear_mask_visualization(self):
        if self.mask_rubber_band:
            try:
                self.mask_rubber_band.reset(QgsWkbTypes.PolygonGeometry)
            except RuntimeError:
                self.mask_rubber_band = None
        if self.dock_widget:
            self.dock_widget.set_disjoint_warning(False)

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
                    tr("Do you want to continue?"),
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
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

            # Clear prior mask logits so SAM predicts fresh from remaining points
            self.current_low_res_mask = None

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
                    tr("Do you want to continue?"),
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._restore_last_saved_mask()

    def _restore_last_saved_mask(self):
        """Restore the last saved mask for editing in batch mode."""
        if not self.dock_widget:
            return
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
        points_positive = last_polygon.get("points_positive", [])
        points_negative = last_polygon.get("points_negative", [])

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
        self.current_mask = last_polygon.get("raw_mask")
        self.current_score = last_polygon.get("score", 0.0)
        self.current_transform_info = last_polygon.get("transform_info")

        # Restore refine settings
        self._refine_expand = last_polygon.get("refine_expand", 0)
        self._refine_simplify = last_polygon.get("refine_simplify", 3)
        self._refine_fill_holes = last_polygon.get("refine_fill_holes", False)
        self._refine_min_area = last_polygon.get("refine_min_area", 100)

        # Update UI sliders without emitting signals
        self.dock_widget.set_refine_values(
            self._refine_expand,
            self._refine_simplify,
            self._refine_fill_holes,
            self._refine_min_area,
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
            level=Qgis.MessageLevel.Info,
        )

    def _reset_session(self):
        import traceback

        QgsMessageLog.logMessage(
            "_reset_session called! _selected_zone was: {}\nStack: {}".format(
                self._selected_zone,
                "".join(traceback.format_stack()[-4:-1]),
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Warning,
        )
        self.prompts.clear()
        self.saved_polygons = []

        for rb in self.saved_rubber_bands:
            self._safe_remove_rubber_band(rb)
        self.saved_rubber_bands = []

        # Clean up PRO pending detections
        for det in self._pro_pending_detections:
            self._safe_remove_rubber_band(det.get("rb"))
        self._pro_pending_detections = []
        self._pro_detection_batches = []

        if self.map_tool:
            self.map_tool.clear_markers()

        self._clear_mask_visualization()

        self.current_mask = None
        self.current_score = 0.0
        self.current_transform_info = None
        self.current_low_res_mask = None

        # Reset on-demand encoding state
        self._current_crop_info = None
        self._current_raster_path = None
        self._current_crop_canvas_mupp = None
        self._current_crop_actual_mupp = None
        self._current_crop_scale_factor = None

        # Reset online layer state
        self._is_online_layer = False

        # Reset tiling worker (but preserve zone selection — it's user config, not session state)
        if self._tiled_worker and self._tiled_worker.isRunning():
            self._tiled_worker.cancel()
            self._tiled_worker.wait(3000)
        self._tiled_worker = None
        # NOTE: self._selected_zone is NOT cleared here — zone selection
        # persists across detection runs until the user explicitly clears it

        # Reset refinement settings to defaults
        self._refine_expand = 0
        self._refine_simplify = 3  # Default: matches dockwidget spinbox
        self._refine_fill_holes = False  # Default: matches dockwidget checkbox
        self._refine_min_area = 100  # Default: matches dockwidget spinbox

        if self._active_dock:
            self._active_dock.set_point_count(0, 0)
            self._active_dock.set_saved_polygon_count(0)
        self._active_mode = "standard"

    # ── PRO mode methods ─────────────────────────────────────────────────

    def _on_start_pro_segmentation(self, layer: QgsRasterLayer):
        """Start PRO (SAM 3) segmentation via serverless inference."""
        import pathlib

        QgsMessageLog.logMessage(
            "_on_start_pro_segmentation: called with layer={}".format(
                layer.name() if layer else "None"
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        from ..core.venv_manager import ensure_venv_packages_available

        ensure_venv_packages_available()

        from ..core.pro_predictor import FalPredictor

        # Read FAL_KEY from .env at plugin root
        env_path = pathlib.Path(__file__).parent.parent.parent / ".env"
        fal_key = ""
        if env_path.exists():
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("FAL_KEY="):
                        fal_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

        QgsMessageLog.logMessage(
            "_on_start_pro_segmentation: .env found={}, fal_key_len={}".format(
                env_path.exists(), len(fal_key)
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        if not fal_key:
            QMessageBox.warning(
                self.iface.mainWindow(),
                tr("AI Segmentation PRO"),
                tr(
                    "No API key configured.\n\n"
                    "Add FAL_KEY=your_key to the .env file\n"
                    "at the plugin root directory."
                ),
            )
            return

        # Clean up previous predictor
        if self.predictor:
            try:
                self.predictor.cleanup()
            except Exception:
                pass
        self.predictor = FalPredictor(fal_key=fal_key)

        # Validate layer
        if not self._is_layer_valid(layer):
            QgsMessageLog.logMessage(
                "Layer was deleted before PRO segmentation could start",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            self.predictor.cleanup()
            self.predictor = None
            return

        try:
            layer_name = layer.name().replace(" ", "_")
            raster_path = os.path.normcase(layer.source())
        except RuntimeError:
            self.predictor.cleanup()
            self.predictor = None
            return

        self._reset_session()
        self._current_layer = layer
        self._current_layer_name = layer_name
        self._is_online_layer = self._is_online_provider(layer)
        self._is_non_georeferenced_mode = (
            not self._is_online_layer and not self._is_layer_georeferenced(layer)
        )
        self._current_raster_path = raster_path

        # Set up CRS transforms
        canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        raster_crs = layer.crs() if layer else None
        self._canvas_to_raster_xform = None
        self._raster_to_canvas_xform = None
        if raster_crs and canvas_crs.isValid() and raster_crs.isValid():
            if canvas_crs != raster_crs:
                self._canvas_to_raster_xform = QgsCoordinateTransform(
                    canvas_crs, raster_crs, QgsProject.instance()
                )
                self._raster_to_canvas_xform = QgsCoordinateTransform(
                    raster_crs, canvas_crs, QgsProject.instance()
                )

        self._active_mode = "pro"
        QgsMessageLog.logMessage(
            "_on_start_pro_segmentation: _active_mode set to 'pro', "
            "calling _activate_segmentation_tool",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
        self._activate_segmentation_tool()
        QgsMessageLog.logMessage(
            "PRO mode activated — predictor={}, layer={}, dock={}".format(
                type(self.predictor).__name__,
                self._current_layer_name,
                type(self._active_dock).__name__,
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

    def _update_pro_detections_visualization(self):
        """Re-render all PRO pending detection rubber bands with current refine settings."""
        from ..core.polygon_exporter import apply_mask_refinement, mask_to_polygons

        for det in self._pro_pending_detections:
            mask = det["mask"]
            ti = det["transform_info"]
            rb = det.get("rb")
            if rb is None or mask is None:
                continue

            # Apply mask refinement
            refined_mask = mask
            if (
                self._refine_fill_holes
                or self._refine_min_area > 0
                or self._refine_expand != 0
            ):
                refined_mask = apply_mask_refinement(
                    mask,
                    expand_value=self._refine_expand,
                    fill_holes=self._refine_fill_holes,
                    min_area=self._refine_min_area,
                )

            polys = mask_to_polygons(refined_mask, ti)
            if not polys:
                rb.reset(QgsWkbTypes.PolygonGeometry)
                continue

            geom = QgsGeometry.unaryUnion(polys)
            if not geom or geom.isEmpty():
                rb.reset(QgsWkbTypes.PolygonGeometry)
                continue

            # Apply simplification
            tolerance = self._compute_simplification_tolerance(
                ti, self._refine_simplify
            )
            if tolerance > 0:
                geom = geom.simplify(tolerance)

            display_geom = QgsGeometry(geom)
            self._transform_geometry_to_canvas_crs(display_geom)
            rb.setToGeometry(display_geom, None)

    def _run_fal_detection(self):
        """Detect objects matching the text prompt on the current canvas view."""
        from ..core.pro_predictor import decode_rle_to_mask

        QgsMessageLog.logMessage(
            "_run_fal_detection: called — dock={}, predictor={}, layer={}".format(
                type(self._active_dock).__name__ if self._active_dock else "None",
                type(self.predictor).__name__ if self.predictor else "None",
                self._current_layer_name,
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        if not self._active_dock or not self.predictor:
            QgsMessageLog.logMessage(
                "_run_fal_detection: ABORT — missing dock or predictor",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            return
        text_prompt = self._active_dock.get_pro_text_prompt()
        if not text_prompt:
            QgsMessageLog.logMessage(
                "_run_fal_detection: ABORT — empty text prompt",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            return

        QgsMessageLog.logMessage(
            "_run_fal_detection: prompt='{}', threshold={}".format(
                text_prompt,
                self._active_dock.get_score_threshold() if self._active_dock else "N/A",
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        raster_layer = self._current_layer
        if raster_layer is None:
            QgsMessageLog.logMessage(
                "_run_fal_detection: ABORT — no current layer",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            return

        # Clear any rubber bands from a previous detection run
        for det in self._pro_pending_detections:
            self._safe_remove_rubber_band(det.get("rb"))
        self._pro_pending_detections = []
        self._pro_detection_batches = []

        # ── Tiling branch: check if we need multi-tile detection ──
        QgsMessageLog.logMessage(
            "_run_fal_detection: _selected_zone={}, type={}".format(
                self._selected_zone,
                type(self._selected_zone).__name__ if self._selected_zone else "None",
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
        if self._selected_zone:
            zone_in_layer_crs = self._reproject_zone_to_layer_crs(
                self._selected_zone, raster_layer
            )
            pixel_w, pixel_h = self._zone_to_pixels(zone_in_layer_crs, raster_layer)
        else:
            pixel_w, pixel_h = raster_layer.width(), raster_layer.height()

        tiles = self._tile_manager.compute_grid(pixel_w, pixel_h)

        if tiles is None:
            self.iface.messageBar().pushWarning(
                "AI Segmentation",
                tr("Zone too large. Please reduce the selection."),
            )
            return

        if len(tiles) > 1 or self._selected_zone is not None:
            # Multi-tile or zone-selected flow — use tiled worker
            self._run_tiled_detection(text_prompt, tiles, raster_layer)
            return

        # ── Single tile, no zone — use existing flow unchanged ──
        canvas = self.iface.mapCanvas()
        canvas_extent = canvas.extent()
        canvas_center = canvas_extent.center()

        # Convert canvas center to raster CRS if needed
        canvas_crs = canvas.mapSettings().destinationCrs()
        layer_crs = raster_layer.crs()
        if canvas_crs != layer_crs:
            xform = QgsCoordinateTransform(canvas_crs, layer_crs, QgsProject.instance())
            raster_center = xform.transform(canvas_center)
        else:
            raster_center = canvas_center

        # Compute scale factor so the crop covers the visible canvas area
        initial_scale = self._compute_initial_scale_factor()
        QgsMessageLog.logMessage(
            "_run_fal_detection: scale_factor={}, online={}".format(
                initial_scale, self._is_online_layer
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        # Extract and store the image (calls predictor.set_image internally)
        if not self._extract_and_encode_crop(
            raster_center, mupp_override=initial_scale
        ):
            QgsMessageLog.logMessage(
                "_run_fal_detection: ABORT — _extract_and_encode_crop failed",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            return

        if self._current_crop_info is None:
            QgsMessageLog.logMessage(
                "_run_fal_detection: ABORT — _current_crop_info is None",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            return

        crop_bounds = self._current_crop_info["bounds"]
        img_shape = self._current_crop_info["img_shape"]
        img_height, img_width = img_shape
        # Tell the predictor to crop to native size before upload.
        # This discards reflect-padding so SAM-3 won't detect mirror objects.
        # RLE offsets from the response will be in native (img_height x img_width) space.
        if self.predictor:
            self.predictor.set_native_size(img_height, img_width)
        sent_h = img_height
        sent_w = img_width
        QgsMessageLog.logMessage(
            "_run_fal_detection: crop OK — native={}x{}, sent={}x{}, bounds={}".format(
                img_width, img_height, sent_w, sent_h, crop_bounds
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
        minx, miny, maxx, maxy = crop_bounds

        crs_value = None
        try:
            if self._current_layer and self._current_layer.crs().isValid():
                crs_value = self._current_layer.crs().authid()
        except RuntimeError:
            pass

        transform_info = {
            "bbox": (minx, miny, maxx, maxy),
            "img_shape": (img_height, img_width),
            "crs": crs_value,
        }
        QgsMessageLog.logMessage(
            "_run_fal_detection: transform_info — native_shape=({},{}), "
            "sent_shape=({},{}), bbox=({:.2f},{:.2f},{:.2f},{:.2f}), crs={}".format(
                img_height,
                img_width,
                sent_h,
                sent_w,
                minx,
                miny,
                maxx,
                maxy,
                crs_value,
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        score_threshold = (
            self._active_dock.get_score_threshold() if self._active_dock else 0.0
        )

        self.iface.messageBar().pushMessage(
            tr("AI Segmentation"),
            tr("Detecting '{prompt}'...").format(prompt=text_prompt),
            level=Qgis.MessageLevel.Info,
            duration=0,
        )
        QApplication.processEvents()

        max_masks = (
            self._active_dock.get_max_objects()
            if hasattr(self._active_dock, "get_max_objects")
            else 10
        )

        try:
            QgsMessageLog.logMessage(
                "_run_fal_detection: calling predict_text (max_masks={})...".format(
                    max_masks
                ),
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )
            result = self.predictor.predict_text(text_prompt, max_masks=max_masks)
            QgsMessageLog.logMessage(
                "_run_fal_detection: predict_text returned — keys={}".format(
                    list(result.keys()) if isinstance(result, dict) else type(result)
                ),
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )
        except RuntimeError as e:
            self.iface.messageBar().clearWidgets()
            QgsMessageLog.logMessage(
                "_run_fal_detection: predict_text ERROR — {}".format(e),
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical,
            )
            QMessageBox.warning(
                self.iface.mainWindow(),
                tr("Detection Error"),
                str(e),
            )
            return

        self.iface.messageBar().clearWidgets()

        # Normalize rle to list of strings
        rle_list = result.get("rle", [])
        if isinstance(rle_list, str):
            rle_list = [rle_list]
        scores_list = result.get("scores") or []
        QgsMessageLog.logMessage(
            "_run_fal_detection: {} RLE masks, {} scores, threshold={}".format(
                len(rle_list), len(scores_list), score_threshold
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        all_detections = []
        for i, rle_str in enumerate(rle_list):
            score = float(scores_list[i]) if i < len(scores_list) else 1.0
            if score < score_threshold:
                QgsMessageLog.logMessage(
                    "  mask {}: score={:.3f} < threshold={:.3f}, SKIP".format(
                        i, score, score_threshold
                    ),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )
                continue
            try:
                mask = decode_rle_to_mask(rle_str, sent_h, sent_w)
            except Exception as e:
                QgsMessageLog.logMessage(
                    "RLE decode error for mask {}: {}".format(i, e),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Warning,
                )
                continue
            QgsMessageLog.logMessage(
                "  mask {}: score={:.3f}, rle_tokens={}, pixels={}, "
                "decode_shape={}".format(
                    i,
                    score,
                    len(rle_str.split()),
                    int(mask.sum()),
                    mask.shape,
                ),
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )
            all_detections.append((mask, score, transform_info))

        QgsMessageLog.logMessage(
            "_run_fal_detection: {} detections passed score filter".format(
                len(all_detections)
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        if not all_detections:
            self.iface.messageBar().pushMessage(
                tr("AI Segmentation"),
                tr("No objects found for '{prompt}'.").format(prompt=text_prompt),
                level=Qgis.MessageLevel.Info,
                duration=5,
            )
            return

        from ..core.polygon_exporter import (
            _iou,
            apply_mask_refinement,
            mask_to_polygons,
        )

        all_detections.sort(key=lambda x: x[1], reverse=True)

        IOU_THRESHOLD = 0.3

        accepted_geoms = []
        batch_count = 0

        for det_idx, (mask, score, ti) in enumerate(all_detections):
            # Apply mask refinement for filtering/display, keep raw mask for storage
            refined_mask = mask
            if (
                self._refine_fill_holes
                or self._refine_min_area > 0
                or self._refine_expand != 0
            ):
                refined_mask = apply_mask_refinement(
                    mask,
                    expand_value=self._refine_expand,
                    fill_holes=self._refine_fill_holes,
                    min_area=self._refine_min_area,
                )
            px_count = int(refined_mask.sum())
            if px_count < 20:
                QgsMessageLog.logMessage(
                    "  det {}: SKIP — only {} fg pixels (<20)".format(
                        det_idx, px_count
                    ),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )
                continue
            polys = mask_to_polygons(refined_mask, ti)
            if not polys:
                QgsMessageLog.logMessage(
                    "  det {}: SKIP — mask_to_polygons returned empty".format(det_idx),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )
                continue
            geom = QgsGeometry.unaryUnion(polys)
            if not geom or geom.isEmpty():
                QgsMessageLog.logMessage(
                    "  det {}: SKIP — unaryUnion empty".format(det_idx),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )
                continue
            max_iou = 0.0
            for ag in accepted_geoms:
                iou_val = _iou(geom, ag)
                if iou_val > max_iou:
                    max_iou = iou_val
            if max_iou > IOU_THRESHOLD:
                QgsMessageLog.logMessage(
                    "  det {}: SKIP — IoU={:.3f} > {:.1f} (duplicate)".format(
                        det_idx, max_iou, IOU_THRESHOLD
                    ),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )
                continue
            QgsMessageLog.logMessage(
                "  det {}: ACCEPTED — score={:.3f}, {}px, {} polys, "
                "area={:.1f}, max_iou={:.3f}".format(
                    det_idx,
                    score,
                    px_count,
                    len(polys),
                    geom.area(),
                    max_iou,
                ),
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )
            # Apply simplification
            tolerance = self._compute_simplification_tolerance(
                ti, self._refine_simplify
            )
            if tolerance > 0:
                geom = geom.simplify(tolerance)

            accepted_geoms.append(geom)

            rb = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
            rb.setColor(QColor(0, 200, 100, 120))
            rb.setFillColor(QColor(0, 200, 100, 80))
            rb.setWidth(2)
            display_geom = QgsGeometry(geom)
            self._transform_geometry_to_canvas_crs(display_geom)
            rb.setToGeometry(display_geom, None)

            self._pro_pending_detections.append(
                {
                    "mask": mask,
                    "score": score,
                    "transform_info": ti.copy(),
                    "rb": rb,
                }
            )
            batch_count += 1

        self._clear_mask_visualization()
        self.current_mask = None
        self.current_transform_info = None

        if batch_count > 0:
            self._pro_detection_batches.append(batch_count)
            if self._active_dock:
                pos = len(self._pro_detection_batches)
                self._active_dock.set_point_count(pos, 0)
                self._active_dock.set_batch_done(batch_count)
                self._active_dock.set_mask_available(True)
            self.iface.messageBar().pushMessage(
                tr("AI Segmentation"),
                tr("{n} object(s) detected. Review and save.").format(n=batch_count),
                level=Qgis.MessageLevel.Success,
                duration=5,
            )

    # ── Zone selection and tiled detection ────────────────────────────────

    def _on_zone_select(self):
        """Activate rectangle drawing tool on canvas."""
        QgsMessageLog.logMessage(
            "Zone selection tool activated",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
        canvas = self.iface.mapCanvas()
        self._zone_selection_tool = ZoneSelectionMapTool(canvas)
        self._zone_selection_tool.zone_selected.connect(self._on_zone_drawn)
        canvas.setMapTool(self._zone_selection_tool)

    def _on_zone_drawn(self, rect):
        """User finished drawing a zone rectangle."""
        self._selected_zone = rect
        QgsMessageLog.logMessage(
            "Zone drawn: {:.2f},{:.2f} to {:.2f},{:.2f}".format(
                rect.xMinimum(),
                rect.yMinimum(),
                rect.xMaximum(),
                rect.yMaximum(),
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
        self.pro_dock_widget.set_zone_active(True)
        self._update_credit_estimate()
        # Restore previous map tool
        self.iface.mapCanvas().unsetMapTool(self._zone_selection_tool)

    def _on_zone_clear(self):
        """Reset to full image."""
        QgsMessageLog.logMessage(
            "Zone selection cleared — using full image",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
        self._selected_zone = None
        if self._zone_selection_tool:
            self._zone_selection_tool.clear_selection()
        self.pro_dock_widget.set_zone_active(False)
        self._update_credit_estimate()

    def _update_credit_estimate(self):
        """Compute and display credit estimate based on current zone."""
        layer = self.pro_dock_widget.layer_combo.currentLayer()
        if not layer:
            return

        if self._selected_zone:
            zone_in_layer_crs = self._reproject_zone_to_layer_crs(
                self._selected_zone, layer
            )
            pixel_width, pixel_height = self._zone_to_pixels(zone_in_layer_crs, layer)
        else:
            pixel_width = layer.width()
            pixel_height = layer.height()

        credits = self._tile_manager.estimate_credits(pixel_width, pixel_height)
        QgsMessageLog.logMessage(
            "Credit estimate: {}px x {}px = {} credits (tiles)".format(
                pixel_width,
                pixel_height,
                credits,
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
        self.pro_dock_widget.set_credit_estimate(credits)

    def _reproject_zone_to_layer_crs(self, zone_rect, layer):
        """Reproject a zone rectangle from canvas CRS to layer CRS if needed.

        Returns a QgsRectangle in layer CRS coordinates.
        """
        canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        layer_crs = layer.crs()
        if canvas_crs != layer_crs and canvas_crs.isValid() and layer_crs.isValid():
            xform = QgsCoordinateTransform(canvas_crs, layer_crs, QgsProject.instance())
            zone_rect = xform.transformBoundingBox(zone_rect)
            QgsMessageLog.logMessage(
                "Zone reprojected from {} to {}: {:.2f},{:.2f} to {:.2f},{:.2f}".format(
                    canvas_crs.authid(),
                    layer_crs.authid(),
                    zone_rect.xMinimum(),
                    zone_rect.yMinimum(),
                    zone_rect.xMaximum(),
                    zone_rect.yMaximum(),
                ),
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )
        return zone_rect

    def _zone_to_pixels(self, zone_rect, layer):
        """Convert a geographic QgsRectangle to pixel dimensions for a raster layer.

        zone_rect must be in layer CRS coordinates.
        """
        extent = layer.extent()
        if extent.width() == 0 or extent.height() == 0:
            return 1, 1
        px_w = extent.width() / layer.width()
        px_h = extent.height() / layer.height()
        pixel_width = int(zone_rect.width() / px_w)
        pixel_height = int(zone_rect.height() / px_h)
        return max(1, pixel_width), max(1, pixel_height)

    def _run_tiled_detection(self, text_prompt, tiles, layer):
        """Run tiled detection for large images or selected zones."""
        from ..workers.tiled_detection_worker import TiledDetectionWorker

        QgsMessageLog.logMessage(
            "Starting tiled detection: prompt='{}', tiles={}, layer='{}'".format(
                text_prompt,
                len(tiles),
                layer.name(),
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
        for i, (x, y, w, h) in enumerate(tiles):
            QgsMessageLog.logMessage(
                "  Tile {}: offset=({},{}) size={}x{}".format(i, x, y, w, h),
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )

        full_image, geo_transform = self._extract_full_zone_image(layer)
        if full_image is None or geo_transform is None:
            self.iface.messageBar().pushCritical(
                "AI Segmentation",
                tr("Failed to extract zone image"),
            )
            return

        self._tiled_worker = TiledDetectionWorker(
            predictor=self.predictor,
            tiles=tiles,
            full_image=full_image,
            text_prompt=text_prompt,
            max_masks=(
                self._active_dock.get_max_objects()
                if hasattr(self._active_dock, "get_max_objects")
                else 10
            ),
            score_threshold=(
                self._active_dock.get_score_threshold() if self._active_dock else 0.0
            ),
            geo_transform_info=geo_transform,
            tile_manager=self._tile_manager,
        )
        self._tiled_worker.tile_completed.connect(self._on_tile_completed)
        self._tiled_worker.all_completed.connect(self._on_all_tiles_completed)
        self._tiled_worker.progress.connect(self._on_tile_progress)
        self._tiled_worker.error.connect(self._on_tile_error)
        self._tiled_worker.cancelled.connect(self._on_tile_cancelled)

        self.pro_dock_widget.set_tile_progress(0, len(tiles))
        self.iface.messageBar().pushMessage(
            tr("AI Segmentation"),
            tr("Detecting '{prompt}' across {n} tile(s)...").format(
                prompt=text_prompt, n=len(tiles)
            ),
            level=Qgis.MessageLevel.Info,
            duration=0,
        )
        self._tiled_worker.start()

    def _on_tile_completed(self, tile_idx, detections):
        """Progressive display: show detections from this tile immediately."""
        QgsMessageLog.logMessage(
            "Tile {} completed: {} raw detections".format(tile_idx, len(detections)),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
        from ..core.polygon_exporter import (
            _iou,
            apply_mask_refinement,
            mask_to_polygons,
        )

        for mask, score, ti in detections:
            refined = apply_mask_refinement(
                mask,
                expand_value=self._refine_expand,
                fill_holes=self._refine_fill_holes,
                min_area=self._refine_min_area,
            )
            polys = mask_to_polygons(refined, ti)
            if not polys:
                continue
            geom = QgsGeometry.unaryUnion(polys)
            if not geom or geom.isEmpty():
                continue

            # Check IoU against existing accepted detections
            existing_geoms = [
                d["geom"] for d in self._pro_pending_detections if "geom" in d
            ]
            max_iou = max((_iou(geom, ag) for ag in existing_geoms), default=0.0)
            if max_iou > 0.3:
                QgsMessageLog.logMessage(
                    "  Tile {}: detection skipped (IoU={:.2f} > 0.3, duplicate)".format(
                        tile_idx,
                        max_iou,
                    ),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )
                continue

            # Apply simplification
            tolerance = self._compute_simplification_tolerance(
                ti, self._refine_simplify
            )
            if tolerance > 0:
                geom = geom.simplify(tolerance)

            # Create rubber band for progressive display
            rb = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
            rb.setColor(QColor(0, 200, 100, 120))
            rb.setFillColor(QColor(0, 200, 100, 80))
            rb.setWidth(2)
            display_geom = QgsGeometry(geom)
            self._transform_geometry_to_canvas_crs(display_geom)
            rb.setToGeometry(display_geom, None)

            self._pro_pending_detections.append(
                {
                    "mask": mask,
                    "score": score,
                    "transform_info": ti.copy(),
                    "rb": rb,
                    "geom": geom,
                }
            )
            QgsMessageLog.logMessage(
                "  Tile {}: detection accepted (score={:.3f}, area={:.1f}, total={})".format(
                    tile_idx,
                    score,
                    geom.area(),
                    len(self._pro_pending_detections),
                ),
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )

    def _on_tile_progress(self, current, total):
        """Update tile progress in dock."""
        self.pro_dock_widget.set_tile_progress(current, total)

    def _merge_overlapping_detections(self):
        """Merge overlapping polygons from different tiles into single detections.

        Uses a union-find approach: any two polygons that intersect get merged.
        After merging, rubber bands are recreated for the merged geometries.
        """
        detections = self._pro_pending_detections
        if len(detections) < 2:
            return

        n = len(detections)
        # Union-Find to group overlapping polygons
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb_ = find(a), find(b)
            if ra != rb_:
                parent[ra] = rb_

        # Find all pairs that overlap
        for i in range(n):
            gi = detections[i].get("geom")
            if not gi or gi.isEmpty():
                continue
            for j in range(i + 1, n):
                gj = detections[j].get("geom")
                if not gj or gj.isEmpty():
                    continue
                if gi.intersects(gj):
                    union(i, j)

        # Group detections by their root
        from collections import defaultdict

        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        # Check if any merging is needed
        merge_count = sum(1 for g in groups.values() if len(g) > 1)
        if merge_count == 0:
            QgsMessageLog.logMessage(
                "Merge pass: no overlapping polygons found",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )
            return

        QgsMessageLog.logMessage(
            "Merge pass: {} groups with overlaps, merging...".format(merge_count),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        # Build merged detections
        merged = []
        for root, indices in groups.items():
            # Take the highest-score detection as the base
            best_idx = max(indices, key=lambda i: detections[i].get("score", 0))
            base = detections[best_idx]

            if len(indices) == 1:
                # No merge needed, keep as-is
                merged.append(base)
                continue

            # Merge all geometries in this group
            geoms = [
                detections[i]["geom"]
                for i in indices
                if detections[i].get("geom") and not detections[i]["geom"].isEmpty()
            ]
            merged_geom = QgsGeometry.unaryUnion(geoms)

            # Remove old rubber bands for all entries in this group
            for i in indices:
                self._safe_remove_rubber_band(detections[i].get("rb"))

            # Create new rubber band for merged geometry
            rb = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
            rb.setColor(QColor(0, 200, 100, 120))
            rb.setFillColor(QColor(0, 200, 100, 80))
            rb.setWidth(2)
            display_geom = QgsGeometry(merged_geom)
            self._transform_geometry_to_canvas_crs(display_geom)
            rb.setToGeometry(display_geom, None)

            merged.append(
                {
                    "mask": base["mask"],
                    "score": base["score"],
                    "transform_info": base["transform_info"],
                    "rb": rb,
                    "geom": merged_geom,
                }
            )

            QgsMessageLog.logMessage(
                "  Merged {} polygons into 1 (area={:.1f})".format(
                    len(indices),
                    merged_geom.area(),
                ),
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )

        self._pro_pending_detections = merged

    def _on_all_tiles_completed(self, all_detections):
        """All tiles processed."""
        QgsMessageLog.logMessage(
            "All tiles completed: {} raw detections, {} accepted after dedup".format(
                len(all_detections),
                len(self._pro_pending_detections),
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        # Merge overlapping polygons from adjacent tiles
        before_merge = len(self._pro_pending_detections)
        self._merge_overlapping_detections()
        after_merge = len(self._pro_pending_detections)
        if before_merge != after_merge:
            QgsMessageLog.logMessage(
                "Merge reduced {} detections to {}".format(
                    before_merge,
                    after_merge,
                ),
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )

        self.iface.messageBar().clearWidgets()
        self.pro_dock_widget.hide_tile_progress()
        count = len(self._pro_pending_detections)
        if self._active_dock:
            self._active_dock.set_point_count(count, 0)
            if count > 0:
                self._pro_detection_batches.append(count)
                self._active_dock.set_batch_done(count)
                self._active_dock.set_mask_available(True)
        if count > 0:
            self.iface.messageBar().pushMessage(
                tr("AI Segmentation"),
                tr("{n} object(s) detected. Review and save.").format(n=count),
                level=Qgis.MessageLevel.Success,
                duration=5,
            )
        else:
            self.iface.messageBar().pushMessage(
                tr("AI Segmentation"),
                tr("No objects found."),
                level=Qgis.MessageLevel.Info,
                duration=5,
            )

    def _on_tile_error(self, msg):
        """Tiled detection failed."""
        self.iface.messageBar().clearWidgets()
        self.pro_dock_widget.hide_tile_progress()
        QgsMessageLog.logMessage(
            "Tiled detection failed: {}".format(msg),
            "AI Segmentation",
            level=Qgis.MessageLevel.Critical,
        )
        self.iface.messageBar().pushCritical("AI Segmentation", msg)

    def _on_tile_cancelled(self):
        """User cancelled tiled detection — discard partial results."""
        QgsMessageLog.logMessage(
            "Tiled detection cancelled — discarding {} detections".format(
                len(self._pro_pending_detections),
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Warning,
        )
        self.iface.messageBar().clearWidgets()
        self.pro_dock_widget.hide_tile_progress()
        for det in self._pro_pending_detections:
            self._safe_remove_rubber_band(det.get("rb"))
        self._pro_pending_detections.clear()
        if self._active_dock:
            self._active_dock.set_point_count(0, 0)

    def _extract_full_zone_image(self, layer):
        """Read the selected zone at native resolution using rasterio.

        Uses the rasterio affine transform for correct pixel-to-geo mapping,
        which handles rotated rasters properly (unlike layer.extent()).

        Returns:
            tuple: (image_array, geo_info) where image_array is (H, W, 3) uint8
                   and geo_info is a dict with 'bbox', 'img_shape', 'crs'.
            Returns (None, None) on error.
        """
        import numpy as np

        try:
            import rasterio
            from rasterio.windows import Window
        except ImportError:
            QgsMessageLog.logMessage(
                "rasterio not available for zone extraction",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical,
            )
            return None, None

        raster_path = layer.source()

        if self._selected_zone:
            zone = self._reproject_zone_to_layer_crs(self._selected_zone, layer)
        else:
            zone = None  # Full image

        try:
            with rasterio.open(raster_path) as src:
                raster_transform = src.transform
                # Use rasterio's affine transform for pixel size (handles rotation)
                px_size_x = abs(raster_transform.a)
                px_size_y = abs(raster_transform.e)

                if zone is not None:
                    # Convert zone geographic coords to pixel coords using
                    # the rasterio transform (inverse)
                    inv_transform = ~raster_transform
                    # Geographic corners → pixel coords
                    col_min, row_min = inv_transform * (
                        zone.xMinimum(),
                        zone.yMaximum(),
                    )
                    col_max, row_max = inv_transform * (
                        zone.xMaximum(),
                        zone.yMinimum(),
                    )

                    col_off = max(0, int(round(min(col_min, col_max))))
                    row_off = max(0, int(round(min(row_min, row_max))))
                    col_end = min(src.width, int(round(max(col_min, col_max))))
                    row_end = min(src.height, int(round(max(row_min, row_max))))
                    win_w = col_end - col_off
                    win_h = row_end - row_off
                else:
                    # Full image
                    col_off = 0
                    row_off = 0
                    win_w = src.width
                    win_h = src.height

                QgsMessageLog.logMessage(
                    "Zone extraction: raster {}x{}, "
                    "px_size={:.6f}x{:.6f}, "
                    "window=({},{} {}x{}), bands={}, "
                    "transform.a={:.6f} transform.e={:.6f}".format(
                        src.width,
                        src.height,
                        px_size_x,
                        px_size_y,
                        col_off,
                        row_off,
                        win_w,
                        win_h,
                        src.count,
                        raster_transform.a,
                        raster_transform.e,
                    ),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )

                if win_w <= 0 or win_h <= 0:
                    QgsMessageLog.logMessage(
                        "Zone extraction: invalid window {}x{}".format(win_w, win_h),
                        "AI Segmentation",
                        level=Qgis.MessageLevel.Warning,
                    )
                    return None, None

                window = Window(col_off, row_off, win_w, win_h)

                band_count = min(src.count, 3)
                bands = src.read(
                    list(range(1, band_count + 1)),
                    window=window,
                )

                if band_count == 1:
                    rgb = np.stack([bands[0]] * 3, axis=-1)
                elif band_count == 2:
                    rgb = np.stack([bands[0], bands[1], bands[0]], axis=-1)
                else:
                    rgb = np.transpose(bands[:3], (1, 2, 0))

                result = rgb.astype(np.uint8)

                # Compute the actual geographic bbox of the window using
                # the rasterio affine transform (handles rotation correctly)
                win_transform = rasterio.windows.transform(window, raster_transform)
                # Geographic coordinates of the 4 corners of the window
                corners_geo = [
                    win_transform * (0, 0),  # top-left
                    win_transform * (win_w, 0),  # top-right
                    win_transform * (win_w, win_h),  # bottom-right
                    win_transform * (0, win_h),  # bottom-left
                ]
                xs = [c[0] for c in corners_geo]
                ys = [c[1] for c in corners_geo]
                geo_minx = min(xs)
                geo_miny = min(ys)
                geo_maxx = max(xs)
                geo_maxy = max(ys)

                crs_value = None
                try:
                    if layer.crs().isValid():
                        crs_value = layer.crs().authid()
                except RuntimeError:
                    pass

                geo_info = {
                    "bbox": (geo_minx, geo_miny, geo_maxx, geo_maxy),
                    "img_shape": (win_h, win_w),
                    "crs": crs_value,
                }

                QgsMessageLog.logMessage(
                    "Zone extraction complete: shape={}, "
                    "geo_bbox=({:.2f},{:.2f},{:.2f},{:.2f}), crs={}".format(
                        result.shape,
                        geo_minx,
                        geo_miny,
                        geo_maxx,
                        geo_maxy,
                        crs_value,
                    ),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )
                return result, geo_info

        except Exception as e:
            QgsMessageLog.logMessage(
                "Zone extraction failed: {}".format(str(e)),
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical,
            )
            return None, None

    def _on_pro_layer_combo_changed(self, layer):
        """Handle layer selection change in the PRO dock combo box."""
        if not self._current_layer:
            return

        try:
            new_layer_id = layer.id() if layer else None
            current_layer_id = self._current_layer.id() if self._current_layer else None
        except RuntimeError:
            self._current_layer = None
            return

        if new_layer_id == current_layer_id:
            return

        if self.iface.mapCanvas().mapTool() == self.map_tool:
            has_unsaved_mask = self.current_mask is not None
            has_saved_polygons = len(self.saved_polygons) > 0

            if has_unsaved_mask or has_saved_polygons:
                polygon_count = len(self.saved_polygons)
                if has_unsaved_mask:
                    polygon_count += 1
                message = "{}\n\n{}".format(
                    tr("You have {count} unsaved polygon(s).").format(
                        count=polygon_count
                    ),
                    tr(
                        "Changing layer will discard your current "
                        "segmentation. Continue?"
                    ),
                )

                reply = QMessageBox.warning(
                    self.iface.mainWindow(),
                    tr("Change Layer?"),
                    message,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )

                if reply != QMessageBox.StandardButton.Yes:
                    self.pro_dock_widget.layer_combo.blockSignals(True)
                    self.pro_dock_widget.layer_combo.setLayer(self._current_layer)
                    self.pro_dock_widget.layer_combo.blockSignals(False)
                    return

            self._stopping_segmentation = True
            self.iface.mapCanvas().unsetMapTool(self.map_tool)
            self._restore_previous_map_tool()
            self._stopping_segmentation = False
            self._reset_session()
            if self.pro_dock_widget:
                self.pro_dock_widget.reset_session()
