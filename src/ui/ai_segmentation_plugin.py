from __future__ import annotations

import math
import os
import sys
from datetime import datetime
from pathlib import Path

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
from qgis.PyQt.QtCore import QSettings, Qt
from qgis.PyQt.QtGui import QColor, QIcon
from qgis.PyQt.QtWidgets import (
    QAction,
    QApplication,
    QMenu,
    QMessageBox,
)

from ..core.i18n import tr
from ..core.prompt_manager import FrozenCropSession, PromptManager
from .ai_segmentation_dockwidget import AISegmentationDockWidget
from .ai_segmentation_maptool import AISegmentationMapTool
from .background_workers import DepsInstallWorker, DownloadWorker, VerifyWorker
from .error_report_dialog import show_error_report, start_log_collector, stop_log_collector
from .shortcut_filter import ShortcutFilter

# QGIS 4 rejects raw int for QgsField type arg (#25, #36); pick by version.
try:
    from qgis.core import Qgis as _Qgis
    _QGIS_VERSION_INT = getattr(_Qgis, "QGIS_VERSION_INT", 0)
except ImportError:
    _QGIS_VERSION_INT = 0

if _QGIS_VERSION_INT >= 40000:
    from qgis.PyQt.QtCore import QMetaType as _QMetaType
    # Qt6/PyQt6 scoped enums: QMetaType.Type.QString
    _FIELD_TYPE_STRING = _QMetaType.Type.QString
    _FIELD_TYPE_DOUBLE = _QMetaType.Type.Double
else:
    from qgis.PyQt.QtCore import QVariant as _QVariant
    _FIELD_TYPE_STRING = _QVariant.String
    _FIELD_TYPE_DOUBLE = _QVariant.Double

# QSettings keys for tutorial flags
SETTINGS_KEY_TUTORIAL_SHOWN = "AISegmentation/tutorial_simple_shown"


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
        tr("To install in a different folder, set the environment "
           "variable AI_SEGMENTATION_CACHE_DIR:"),
        steps)


def _probe_writable(directory: str) -> bool:
    probe = os.path.join(directory, f".ai_seg_write_probe_{os.getpid()}")
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


class AISegmentationPlugin:

    def __init__(self, iface: QgisInterface):
        self.iface = iface
        self.plugin_dir = Path(__file__).parent.parent.parent

        self.dock_widget: AISegmentationDockWidget | None = None
        self._dock_created = False
        self.map_tool: AISegmentationMapTool | None = None
        self.action: QAction | None = None
        self.terralab_menu: QMenu | None = None
        self.terralab_toolbar = None

        self.predictor = None
        self.prompts = PromptManager()

        self.current_mask = None
        self.current_score = 0.0
        self.current_transform_info = None
        self.current_low_res_mask = None  # For iterative refinement with negative points
        self.saved_polygons = []
        self._mask_state_history: list = []  # Stack of mask states for per-point undo (capped at 30)
        self._frozen_sessions: list[FrozenCropSession] = []  # Frozen crop polygons
        self._active_crop_points_positive: list[tuple[float, float]] = []
        self._active_crop_points_negative: list[tuple[float, float]] = []

        self._initialized = False
        self._setup_done = False
        self._current_layer = None
        self._current_layer_name = ""

        # MCP/headless mode: when True, skip modal dialogs and cursors
        self._headless = False
        self._headless_error = None

        # Refinement settings (#12, #23: defaults tuned for ease-of-use)
        self._refine_simplify = 10
        self._refine_smooth = 0
        self._refine_expand = 0
        self._refine_fill_holes = True
        self._refine_min_area = 200  # overridden by _compute_auto_min_area() × 2

        self._is_non_georeferenced_mode = False  # Track if current layer is non-georeferenced
        self._is_online_layer = False  # Track if current layer is online (WMS, XYZ, etc.)
        self._disjoint_warning_shown = False

        # On-demand encoding state
        self._current_crop_info = None  # dict with 'bounds', 'img_shape'
        self._current_raster_path = None
        self._encoding_in_progress = False  # Guard against concurrent clicks (main/UI thread only)
        self._shortcut_filter = None  # Event filter for keyboard shortcuts
        self._current_crop_canvas_mupp = None  # canvas mupp at encode time (zoom detection)
        self._current_crop_actual_mupp = None  # actual mupp used for the crop (may differ if zoomed out)
        self._current_crop_scale_factor = None  # scale_factor used for file-based crop
        self.deps_install_worker = None
        self.download_worker = None
        self._verify_worker = None
        self._predictor_worker = None

        self.mask_rubber_band: QgsRubberBand | None = None
        self.saved_rubber_bands: list[QgsRubberBand] = []

        self._previous_map_tool = None  # Store the tool active before segmentation
        self._stopping_segmentation = False  # Flag to track if we're stopping programmatically
        self._exporting_in_progress = False  # Guard against double-click on export

        # CRS transforms (canvas CRS <-> raster CRS), created when features load.
        # None when both CRS are the same (no transform needed).
        self._canvas_to_raster_xform: QgsCoordinateTransform | None = None
        self._raster_to_canvas_xform: QgsCoordinateTransform | None = None

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
                f"BUG: polygon/rubber band mismatch: {n_polygons} vs {n_bands}. "
                "Truncating to min. Please report.",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
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
        bbox = transform_info.get("bbox", [0, 1, 0, 1])
        img_shape = transform_info.get("img_shape", (1024, 1024))
        width_pixels = max(img_shape[1], 1)
        bbox_width = bbox[1] - bbox[0]
        if bbox_width == 0:
            return 0
        pixel_size = bbox_width / width_pixels
        return pixel_size * simplify_value * 0.5

    def initGui(self):
        from ..mcp_api import SegmentationMCPAPI
        self.mcp_api = SegmentationMCPAPI(self)

        start_log_collector()

        icon_path = str(self.plugin_dir / "resources" / "icons" / "icon.png")
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
        self.action.setToolTip(
            "AI Segmentation by TerraLab\n{}".format(
                tr("Segment elements on raster images using AI"))
        )
        self.action.triggered.connect(self.toggle_dock_widget)

        from .terralab_toolbar import add_action_to_toolbar, get_or_create_terralab_toolbar
        self.terralab_toolbar = get_or_create_terralab_toolbar(self.iface)
        add_action_to_toolbar(self.terralab_toolbar, self.action, "ai-segmentation")

        from .terralab_menu import (
            _UTILITY_SEPARATOR,
            add_plugin_to_menu,
            add_to_plugins_menu,
            get_or_create_terralab_menu,
        )
        self.terralab_menu = get_or_create_terralab_menu(self.iface.mainWindow())
        add_plugin_to_menu(self.terralab_menu, self.action, "ai-segmentation")
        add_to_plugins_menu(self.iface, self.action)

        # Add "Settings" to TerraLab menu (deduplicated with AI Edit via objectName)
        self._settings_action = None
        self._owns_settings_action = False
        existing = self.terralab_menu.findChild(QAction, "_terralab_settings_action")
        if existing:
            existing.triggered.connect(self._on_settings_clicked)
            self._settings_action = existing
        else:
            settings_icon = QIcon(":/images/themes/default/mActionOptions.svg")
            self._settings_action = QAction(settings_icon, tr("Settings"), self.iface.mainWindow())
            self._settings_action.setObjectName("_terralab_settings_action")
            self._settings_action.setMenuRole(QAction.MenuRole.NoRole)
            self._settings_action.triggered.connect(self._on_settings_clicked)
            insert_before = None
            found_sep = False
            for a in self.terralab_menu.actions():
                if a.objectName() == _UTILITY_SEPARATOR:
                    found_sep = True
                    continue
                if found_sep:
                    insert_before = a
                    break
            if insert_before:
                self.terralab_menu.insertAction(insert_before, self._settings_action)
            else:
                self.terralab_menu.addAction(self._settings_action)
            self._owns_settings_action = True

        # Cross-plugin discovery: show AI Edit entry even when it's not installed (#30).
        from .cross_plugin_discovery import make_ai_edit_action
        ai_edit_icon_path = str(self.plugin_dir / "resources" / "icons" / "ai_edit_icon.png")
        ai_edit_icon = QIcon(ai_edit_icon_path) if os.path.exists(ai_edit_icon_path) else None
        self.ai_edit_action = make_ai_edit_action(
            self.iface.mainWindow(),
            self.iface,
            tr("AI Edit"),
            tr("Generate imagery with AI on map zones (opens AI Edit plugin)"),
            icon=ai_edit_icon,
        )
        add_action_to_toolbar(self.terralab_toolbar, self.ai_edit_action, "ai-edit")
        add_plugin_to_menu(self.terralab_menu, self.ai_edit_action, "ai-edit")
        add_to_plugins_menu(self.iface, self.ai_edit_action)

        # Defer dock widget creation to first toggle for fast plugin load
        self.dock_widget = None
        self._dock_created = False

        self.map_tool = AISegmentationMapTool(self.iface.mapCanvas())
        self.map_tool.positive_click.connect(self._on_positive_click)
        self.map_tool.negative_click.connect(self._on_negative_click)
        self.map_tool.tool_deactivated.connect(self._on_tool_deactivated)
        self.map_tool.undo_requested.connect(self._on_undo)
        self.map_tool.save_polygon_requested.connect(self._on_save_polygon)
        self.map_tool.export_layer_requested.connect(self._on_export_layer)
        self.map_tool.stop_segmentation_requested.connect(self._on_stop_segmentation)

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
            qgis_version = Qgis.version() if hasattr(Qgis, "version") else "unknown"
            QgsMessageLog.logMessage(
                f"AI Segmentation v{plugin_version} | QGIS {qgis_version} | "
                f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} | {sys.platform}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )
        except Exception:
            QgsMessageLog.logMessage(
                "AI Segmentation plugin loaded",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )

    def unload(self):
        # 0. Remove keyboard shortcut filter
        try:
            if self._shortcut_filter is not None:
                try:
                    self.iface.mainWindow().removeEventFilter(self._shortcut_filter)
                    canvas = self.iface.mapCanvas()
                    canvas.viewport().removeEventFilter(self._shortcut_filter)
                    canvas.removeEventFilter(self._shortcut_filter)
                except RuntimeError:
                    pass
                self._shortcut_filter = None
        except (RuntimeError, AttributeError):
            pass

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
            # Disconnect all dock widget signals connected in initGui()
            _dock_signals = [
                (self.dock_widget.install_requested, self._on_install_requested),
                (self.dock_widget.cancel_install_requested, self._on_cancel_install),
                (self.dock_widget.start_segmentation_requested, self._on_start_segmentation),
                (self.dock_widget.save_polygon_requested, self._on_save_polygon),
                (self.dock_widget.export_layer_requested, self._on_export_layer),
                (self.dock_widget.undo_requested, self._on_undo),
                (self.dock_widget.stop_segmentation_requested, self._on_stop_segmentation),
                (self.dock_widget.refine_settings_changed, self._on_refine_settings_changed),
                (self.dock_widget.settings_clicked, self._on_settings_clicked),
            ]
            for sig, slot in _dock_signals:
                try:
                    sig.disconnect(slot)
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
                self.map_tool.stop_segmentation_requested.disconnect(self._on_stop_segmentation)
        except (TypeError, RuntimeError, AttributeError):
            pass

        # 2. Cleanup predictor subprocess (with timeout to avoid blocking unload)
        if self.predictor:
            import threading
            pred = self.predictor
            self.predictor = None
            t = threading.Thread(target=lambda: pred.cleanup(), daemon=True)
            t.start()
            t.join(timeout=8)
            if t.is_alive():
                QgsMessageLog.logMessage(
                    "Predictor cleanup did not finish within 8s",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Warning
                )

        # 3. Disconnect worker signals before termination to prevent callbacks on deleted UI
        for worker in [self.deps_install_worker, self.download_worker, self._verify_worker,
                       getattr(self, "_predictor_worker", None)]:
            if worker:
                try:
                    if hasattr(worker, "progress"):
                        worker.progress.disconnect()
                except (TypeError, RuntimeError):
                    pass
                try:
                    worker.finished.disconnect()
                except (TypeError, RuntimeError):
                    pass

        # 4. Terminate workers with timeout
        for worker in [self.deps_install_worker, self.download_worker, self._verify_worker,
                       getattr(self, "_predictor_worker", None)]:
            if worker and worker.isRunning():
                try:
                    worker.terminate()
                    if not worker.wait(5000):
                        QgsMessageLog.logMessage(
                            "Worker did not terminate within 5s",
                            "AI Segmentation",
                            level=Qgis.MessageLevel.Warning
                        )
                except RuntimeError:
                    pass
        self.deps_install_worker = None
        self.download_worker = None
        self._verify_worker = None
        self._predictor_worker = None

        # 5. Disconnect action signal and remove menu/toolbar
        try:
            self.action.triggered.disconnect(self.toggle_dock_widget)
        except (TypeError, RuntimeError, AttributeError):
            pass

        # Remove settings action from menu (only if we created it)
        if self._owns_settings_action and self._settings_action and self.terralab_menu:
            try:
                self.terralab_menu.removeAction(self._settings_action)
            except (RuntimeError, AttributeError):
                pass
        self._settings_action = None

        from .terralab_menu import remove_from_plugins_menu, remove_plugin_from_menu
        try:
            remove_from_plugins_menu(self.iface, self.action)
        except (RuntimeError, AttributeError):
            pass
        ai_edit_action = getattr(self, "ai_edit_action", None)
        if ai_edit_action is not None:
            try:
                remove_from_plugins_menu(self.iface, ai_edit_action)
            except (RuntimeError, AttributeError):
                pass
        if self.terralab_menu:
            try:
                remove_plugin_from_menu(
                    self.terralab_menu, self.action, self.iface.mainWindow())
            except (RuntimeError, AttributeError):
                pass
            ai_edit_action = getattr(self, "ai_edit_action", None)
            if ai_edit_action is not None:
                try:
                    remove_plugin_from_menu(
                        self.terralab_menu, ai_edit_action, self.iface.mainWindow())
                except (RuntimeError, AttributeError):
                    pass
            self.terralab_menu = None

        from .terralab_toolbar import remove_action_from_toolbar
        if self.terralab_toolbar:
            try:
                remove_action_from_toolbar(
                    self.terralab_toolbar, self.action, self.iface.mainWindow())
            except (RuntimeError, AttributeError):
                pass
            ai_edit_action = getattr(self, "ai_edit_action", None)
            if ai_edit_action is not None:
                try:
                    remove_action_from_toolbar(
                        self.terralab_toolbar, ai_edit_action, self.iface.mainWindow())
                except (RuntimeError, AttributeError):
                    pass
            self.terralab_toolbar = None
        self.ai_edit_action = None

        # 6. Remove dock widget
        if self.dock_widget:
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget.deleteLater()
            self.dock_widget = None

        # 7. Clear markers and unset map tool
        if self.map_tool:
            try:
                self.map_tool.clear_markers()
            except (RuntimeError, AttributeError):
                pass
            try:
                if self.iface.mapCanvas().mapTool() == self.map_tool:
                    self.iface.mapCanvas().unsetMapTool(self.map_tool)
            except RuntimeError:
                pass
            self.map_tool = None

        # 8. Remove rubber bands safely
        self._safe_remove_rubber_band(self.mask_rubber_band)
        self.mask_rubber_band = None

        for rb in self.saved_rubber_bands:
            self._safe_remove_rubber_band(rb)
        self.saved_rubber_bands = []

        # 9. Disconnect log collector signal
        stop_log_collector()

    def _ensure_dock_widget(self):
        """Create the dock widget on first use (deferred from initGui for speed)."""
        if self._dock_created:
            return
        self._dock_created = True

        self.dock_widget = AISegmentationDockWidget(self.iface.mainWindow())

        self.dock_widget.install_requested.connect(self._on_install_requested)
        self.dock_widget.cancel_install_requested.connect(self._on_cancel_install)
        self.dock_widget.start_segmentation_requested.connect(self._on_start_segmentation)
        self.dock_widget.save_polygon_requested.connect(self._on_save_polygon)
        self.dock_widget.export_layer_requested.connect(self._on_export_layer)
        self.dock_widget.undo_requested.connect(self._on_undo)
        self.dock_widget.stop_segmentation_requested.connect(self._on_stop_segmentation)
        self.dock_widget.refine_settings_changed.connect(self._on_refine_settings_changed)
        self.dock_widget.settings_clicked.connect(self._on_settings_clicked)
        self.dock_widget.layer_combo.layerChanged.connect(self._on_layer_combo_changed)

        self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_widget)
        self.dock_widget.visibilityChanged.connect(self._on_dock_visibility_changed)
        self._initialized = True

    def toggle_dock_widget(self, checked=None):
        self._ensure_dock_widget()
        if self.dock_widget:
            if checked is None:
                checked = not self.dock_widget.isVisible()
            if checked:
                self.dock_widget.show()
                self.dock_widget.raise_()
            else:
                self.dock_widget.hide()

    def _on_dock_visibility_changed(self, visible: bool):
        if self.action:
            self.action.setChecked(visible)

        if visible and not self._setup_done:
            self._setup_done = True
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._do_first_time_setup)

    def _do_first_time_setup(self):
        QgsMessageLog.logMessage(
            "Panel opened - checking dependencies...",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
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
                self.dock_widget.set_dependency_status(True, "✓ " + tr("Dependencies ready"))
                self._show_device_info()
                QgsMessageLog.logMessage(
                    "✓ Virtual environment verified successfully",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Success
                )
                self._check_checkpoint()
            else:
                self.dock_widget.set_dependency_status(False, message)
                QgsMessageLog.logMessage(
                    f"Virtual environment status: {message}",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info
                )
                # Auto-trigger install for upgrades
                if "need updating" in message:
                    self._on_install_requested()

        except Exception as e:
            import traceback
            error_msg = f"Dependency check error: {str(e)}\n{traceback.format_exc()}"
            QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self.dock_widget.set_dependency_status(False, f"Error: {str(e)[:50]}")

        # Check for plugin updates with retries - QGIS repo metadata may not
        # be ready after just a few seconds, especially on slower connections.
        from qgis.PyQt.QtCore import QTimer
        self._update_check_delays = [5000, 30000, 60000, 120000]
        self._update_check_index = 0
        QTimer.singleShot(
            self._update_check_delays[0], self._check_for_plugin_update)

    def _check_for_plugin_update(self):
        """Trigger the update check on the dock widget, retrying if needed."""
        if not self.dock_widget:
            return
        self.dock_widget.check_for_updates()

        # If notification is still hidden and we have retries left, schedule next
        if (not self.dock_widget.update_notification_widget.isVisible()
                and hasattr(self, "_update_check_delays")):  # noqa: W503
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
                    True, tr("Dependencies ready, model not downloaded"))
                self.dock_widget.install_button.setVisible(True)
                self.dock_widget.install_button.setEnabled(True)
                self.dock_widget.install_button.setText(tr("Download Model"))
                self.dock_widget.setup_group.setVisible(True)

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Checkpoint check error: {str(e)}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )

    def _load_predictor(self):
        """Kick off predictor initialization in the background (#34)."""
        from .background_workers import PredictorLoadWorker
        if self.dock_widget:
            self.dock_widget.set_checkpoint_status(True, tr("Loading AI model..."))
        self._predictor_worker = PredictorLoadWorker()
        self._predictor_worker.finished.connect(self._on_predictor_loaded)
        self._predictor_worker.start()

    def _on_predictor_loaded(self, predictor, err_msg: str):
        if predictor is None:
            QgsMessageLog.logMessage(
                f"Failed to initialize predictor: {err_msg}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
            if self.dock_widget:
                self.dock_widget.set_checkpoint_status(
                    False, tr("Model load failed"))
            return
        self.predictor = predictor
        QgsMessageLog.logMessage(
            "SAM predictor initialized (subprocess mode)",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )
        if self.dock_widget:
            self.dock_widget.set_checkpoint_status(True, tr("SAM model ready"))
            self.dock_widget.set_install_progress(100, tr("Ready"))

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
                level=Qgis.MessageLevel.Info
            )
        except RuntimeError as e:
            error_str = str(e)
            # Check if this is a PyTorch DLL error (Windows)
            if "DLL" in error_str or "shm.dll" in error_str:
                QgsMessageLog.logMessage(
                    f"PyTorch DLL error: {error_str}",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Critical
                )
                # Show user-friendly error dialog
                from qgis.PyQt.QtWidgets import QMessageBox
                msg_box = QMessageBox(self.iface.mainWindow())
                msg_box.setIcon(QMessageBox.Icon.Critical)
                msg_box.setWindowTitle(tr("PyTorch Error"))
                msg_box.setText(tr("PyTorch cannot load on Windows"))
                msg_box.setInformativeText(
                    tr("The plugin requires Visual C++ Redistributables to run PyTorch.\n\n"
                       "Please download and install:\n"
                       "https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"
                       "After installation, restart QGIS and try again."))
                msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg_box.exec()
            else:
                raise
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Could not determine device info: {e}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )

    def _on_install_requested(self):
        # Guard: prevent concurrent installs
        if self.deps_install_worker is not None and self.deps_install_worker.isRunning():
            QgsMessageLog.logMessage(
                "Install already in progress, ignoring duplicate request",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
            return

        from ..core.venv_manager import get_venv_status

        is_ready, message = get_venv_status()
        if is_ready:
            # Deps already installed, just need model download
            self.dock_widget.set_dependency_status(True, "✓ " + tr("Dependencies ready"))
            self._auto_download_checkpoint()
            return

        QgsMessageLog.logMessage(
            "Starting virtual environment creation and dependency installation...",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )
        QgsMessageLog.logMessage(
            f"Platform: {sys.platform}, Python: {sys.version}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        self.dock_widget.set_install_progress(0, "Preparing installation...")

        self.deps_install_worker = DepsInstallWorker()
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

    def _on_settings_clicked(self):
        from ..core.activation_manager import get_auth_header, get_auth_token, is_plugin_activated
        if not is_plugin_activated():
            return
        from ..api.terralab_client import TerraLabClient
        from .account_settings_dialog import AccountSettingsDialog
        client = TerraLabClient()
        dlg = AccountSettingsDialog(
            client=client,
            auth=get_auth_header(),
            activation_key=get_auth_token(),
            parent=self.iface.mainWindow(),
        )
        dlg.change_key_requested.connect(self._on_change_key_requested)
        dlg.exec()

    def _on_change_key_requested(self):
        from ..core.activation_manager import clear_auth
        clear_auth()
        if self.dock_widget:
            self.dock_widget._plugin_activated = False
            self.dock_widget._update_full_ui()

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
            if any(p in msg_lower for p in [
                "ssl", "certificate verify", "sslerror",
                "unable to get local issuer",
            ]):
                error_title = tr("SSL Certificate Error")
            elif any(p in msg_lower for p in [
                "access is denied", "winerror 5", "winerror 225",
                "permission denied", "blocked",
                "cannot write to install",
            ]):
                error_title = tr("Installation Blocked")
                error_msg = f"{error_msg}\n\n{_get_change_path_instructions()}"
            elif any(p in msg_lower for p in [
                "network error", "connection aborted", "connection reset",
                "timed out", "timeout", "network connection failed",
                "connection broken", "could not resolve",
            ]):
                error_title = tr("Network Connection Problem")
                error_msg = "{}\n\n{}".format(
                    error_msg,
                    tr(
                        "Your connection appears unstable or blocked. "
                        "Check: (1) your internet is working, "
                        "(2) QGIS > Settings > Options > Network has a proxy "
                        "configured if you are on a corporate network, "
                        "(3) your firewall allows connections to pypi.org "
                        "and files.pythonhosted.org."
                    ),
                )

            show_error_report(
                self.iface.mainWindow(),
                error_title,
                error_msg
            )

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
            self.dock_widget.set_dependency_status(True, "✓ " + tr("Dependencies ready"))
            if message and not message.startswith("device_error"):
                QgsMessageLog.logMessage(
                    f"Device info: {message}",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
            elif message.startswith("device_error"):
                QgsMessageLog.logMessage(
                    "Could not determine device info: {}".format(
                        message.replace("device_error: ", "")),
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
            try:
                self._auto_download_checkpoint()
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Auto-download checkpoint failed: {e}",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                self.dock_widget.set_install_progress(100, "Failed")
                self.dock_widget.set_dependency_status(
                    True, tr("Dependencies ready, model download failed"))
        else:
            self.dock_widget.set_install_progress(100, "Failed")
            self.dock_widget.set_dependency_status(
                False, "{} {}".format(tr("Verification failed:"), message))
            show_error_report(
                self.iface.mainWindow(),
                tr("Verification Failed"),
                "{}\n{}".format(
                    tr("Virtual environment was created but verification failed:"),
                    message))

    def _on_cancel_install(self):
        if self.deps_install_worker and self.deps_install_worker.isRunning():
            self.deps_install_worker.cancel()
            QgsMessageLog.logMessage(
                "Installation cancelled by user",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )

    def _auto_download_checkpoint(self):
        """Auto-download model after deps install if not already present."""
        from ..core.checkpoint_manager import checkpoint_exists
        try:
            if checkpoint_exists():
                self.dock_widget.set_install_progress(95, tr("Loading AI model..."))
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
                f"Failed to start model download: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self.dock_widget.set_install_progress(100, "Failed")
            self.dock_widget.set_dependency_status(
                True, tr("Dependencies ready, model download failed"))

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
            self.dock_widget.set_install_progress(95, tr("Loading AI model..."))
            self._load_predictor()
            self._show_activation_popup_if_needed()
        else:
            self.dock_widget.set_install_progress(100, "Failed")
            self.dock_widget.set_dependency_status(
                True, tr("Dependencies ready, model download failed"))

            show_error_report(
                self.iface.mainWindow(),
                tr("Download Failed"),
                "{}\n{}".format(
                    tr("Failed to download model:"),
                    message)
            )

    def _on_start_segmentation(self, layer: QgsRasterLayer):
        if self.predictor is None:
            QMessageBox.warning(
                self.iface.mainWindow(),
                tr("Not Ready"),
                tr("Please wait for the SAM model to load.")
            )
            return

        # Validate layer BEFORE resetting session to avoid leaving broken state
        if not self._is_layer_valid(layer):
            QgsMessageLog.logMessage(
                "Layer was deleted before segmentation could start",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return

        try:
            layer_name = layer.name().replace(" ", "_")
            raster_path = os.path.normcase(layer.source())
        except RuntimeError:
            QgsMessageLog.logMessage(
                "Layer deleted during segmentation start",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
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
                level=Qgis.MessageLevel.Info
            )

        if self._is_online_layer:
            QgsMessageLog.logMessage(
                f"Online layer detected ({layer.dataProvider().name()})",
                "AI Segmentation", level=Qgis.MessageLevel.Info
            )

        # Validate layer extent
        if not self._is_online_layer:
            try:
                ext = layer.extent()
                if ext and not ext.isEmpty():
                    coords = (ext.xMinimum(), ext.yMinimum(),
                              ext.xMaximum(), ext.yMaximum())
                    if any(math.isnan(c) or math.isinf(c) for c in coords):
                        show_error_report(
                            self.iface.mainWindow(),
                            tr("Invalid Layer"),
                            tr("Layer extent contains invalid coordinates "
                               "(NaN/Inf). Check the raster file.")
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
                    canvas_crs, raster_crs, QgsProject.instance())
                self._raster_to_canvas_xform = QgsCoordinateTransform(
                    raster_crs, canvas_crs, QgsProject.instance())
                QgsMessageLog.logMessage(
                    f"CRS transform enabled: {canvas_crs.authid()} -> {raster_crs.authid()}",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info
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
        self.dock_widget.set_segmentation_active(True)

        # Install keyboard shortcut filter on both mainWindow and the canvas
        # viewport.  mainWindow catches keys when focus is elsewhere (e.g.
        # after dock widget updates steal focus).  The canvas viewport is
        # needed to intercept ShortcutOverride for Space *before* QGIS
        # activates its built-in pan-tool shortcut.
        if self._shortcut_filter is None:
            self._shortcut_filter = ShortcutFilter(self)
        self.iface.mainWindow().installEventFilter(self._shortcut_filter)
        canvas = self.iface.mapCanvas()
        canvas.viewport().installEventFilter(self._shortcut_filter)
        canvas.installEventFilter(self._shortcut_filter)

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

        from ..core.activation_manager import get_tutorial_url
        tutorial_url = get_tutorial_url()
        message = '{} <a href="{}">{}</a>'.format(
            tr("New here?"),
            tutorial_url,
            tr("Watch the tutorial"))

        self.iface.messageBar().pushMessage(
            "AI Segmentation",
            message,
            level=Qgis.MessageLevel.Info,
            duration=10
        )

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
                        count=polygon_count),
                    tr("Changing layer will discard your current segmentation. Continue?"))

                reply = QMessageBox.warning(
                    self.iface.mainWindow(),
                    tr("Change Layer?"),
                    message,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
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
        """Save current mask as polygon (including any frozen crop sessions)."""
        if self._encoding_in_progress:
            return
        # Allow save if we have frozen sessions even without active mask
        has_active = (self.current_mask is not None
                      and self.current_transform_info is not None)  # noqa: W503
        if not has_active and not self._frozen_sessions:
            return

        self._ensure_polygon_rubberband_sync()

        from ..core.polygon_exporter import apply_mask_refinement, mask_to_polygons

        # Collect all geometry parts: frozen polygons + active mask
        all_geoms = [s.polygon for s in self._frozen_sessions]

        if has_active:
            # Apply all mask-level refinements for display
            mask_for_display = self.current_mask
            if self._refine_fill_holes or self._refine_min_area > 0 or self._refine_expand != 0:
                mask_for_display = apply_mask_refinement(
                    self.current_mask,
                    expand_value=self._refine_expand,
                    fill_holes=self._refine_fill_holes,
                    min_area=self._refine_min_area
                )

            geometries = mask_to_polygons(mask_for_display, self.current_transform_info)
            if geometries:
                active_combined = QgsGeometry.unaryUnion(geometries)
                if active_combined and not active_combined.isEmpty():
                    # Apply simplification if enabled
                    tolerance = self._compute_simplification_tolerance(
                        self.current_transform_info, self._refine_simplify
                    )
                    if tolerance > 0:
                        active_combined = active_combined.simplify(tolerance)
                    # Apply corner rounding (QGIS native C++ Chaikin)
                    if self._refine_smooth > 0:
                        active_combined = active_combined.smooth(
                            self._refine_smooth, 0.25)
                    all_geoms.append(active_combined)

        if all_geoms:
            combined = QgsGeometry.unaryUnion(all_geoms)
        else:
            combined = None

        if combined and not combined.isEmpty():
            # Store WKT (with effects), transform info, raw mask, points, and refine settings
            self.saved_polygons.append({
                "geometry_wkt": combined.asWkt(),
                "transform_info": self.current_transform_info.copy() if self.current_transform_info else None,
                "raw_mask": self.current_mask.copy() if self.current_mask is not None else None,
                "points_positive": list(self.prompts.positive_points),
                "points_negative": list(self.prompts.negative_points),
                "refine_simplify": self._refine_simplify,
                "refine_smooth": self._refine_smooth,
                "refine_expand": self._refine_expand,
                "refine_fill_holes": self._refine_fill_holes,
                "refine_min_area": self._refine_min_area,
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
                level=Qgis.MessageLevel.Info
            )

            self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))

            # Note: We keep refinement settings in batch mode so the user can
            # apply the same expand/simplify to multiple masks

        # Clear current state for next polygon (including frozen sessions)
        self.prompts.clear()
        self._mask_state_history = []
        self._frozen_sessions = []
        self._active_crop_points_positive = []
        self._active_crop_points_negative = []
        if self.map_tool:
            self.map_tool.clear_markers()
        self._clear_mask_visualization()
        self.current_mask = None
        self.current_score = 0.0
        self.current_low_res_mask = None
        self.dock_widget.set_point_count(0, 0)

        # Keep crop info so clicks in the same area reuse the encoding.

    def _on_export_layer(self):
        """Export all saved polygons + current unsaved mask to a new layer."""
        if self._exporting_in_progress:
            return
        self._exporting_in_progress = True
        try:
            self._on_export_layer_impl()
        except Exception:
            import traceback
            QgsMessageLog.logMessage(
                traceback.format_exc(),
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            show_error_report(
                self.iface.mainWindow(),
                tr("Export Failed"),
                tr("An unexpected error occurred during export. Please check the logs.")
            )
        finally:
            self._exporting_in_progress = False

    def _on_export_layer_impl(self):
        """Internal export implementation."""
        from ..core.polygon_exporter import apply_mask_refinement, mask_to_polygons

        self._ensure_polygon_rubberband_sync()

        has_active = (self.current_mask is not None
                      and self.current_transform_info is not None)  # noqa: W503
        if not self.saved_polygons and not has_active and not self._frozen_sessions:
            return  # Nothing to export

        polygons_to_export = list(self.saved_polygons)

        # Build current unsaved geometry: frozen sessions + active mask
        current_geoms = [s.polygon for s in self._frozen_sessions]

        if has_active:
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
                active_combined = QgsGeometry.unaryUnion(geometries)
                if active_combined and not active_combined.isEmpty():
                    tolerance = self._compute_simplification_tolerance(
                        self.current_transform_info, self._refine_simplify
                    )
                    if tolerance > 0:
                        active_combined = active_combined.simplify(tolerance)
                    if self._refine_smooth > 0:
                        active_combined = active_combined.smooth(
                            self._refine_smooth, 0.25)
                    current_geoms.append(active_combined)

        if current_geoms:
            combined = QgsGeometry.unaryUnion(current_geoms)
            if combined and not combined.isEmpty():
                polygons_to_export.append({
                    "geometry_wkt": combined.asWkt(),
                    "transform_info": self.current_transform_info.copy() if self.current_transform_info else None,
                })

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
                level=Qgis.MessageLevel.Info
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
                QgsMessageLog.logMessage(
                    "CRS could not be determined, falling back to EPSG:4326",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)

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

        # OGR/GDAL cannot create intermediate directories; do it here (#27).
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            show_error_report(
                self.iface.mainWindow(),
                tr("Cannot Write Export"),
                tr("Cannot create export directory '{path}': {reason}").format(
                    path=output_dir, reason=str(e))
            )
            return

        if not _probe_writable(output_dir):
            show_error_report(
                self.iface.mainWindow(),
                tr("Cannot Write Export"),
                tr("The export directory '{path}' is not writable. "
                   "Choose a different location.").format(path=output_dir)
            )
            return

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
            QgsField("label", _FIELD_TYPE_STRING),
            QgsField("area", _FIELD_TYPE_DOUBLE),
            QgsField("raster_source", _FIELD_TYPE_STRING),
            QgsField("created_at", _FIELD_TYPE_STRING),
        ])
        temp_layer.updateFields()

        # Resolve metadata for all features
        raster_name = ""
        try:
            if self._is_layer_valid() and self._current_layer:
                raster_name = self._current_layer.name()
        except RuntimeError:
            pass
        timestamp = datetime.now().isoformat(timespec="seconds")

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
                    level=Qgis.MessageLevel.Warning
                )
                continue

            geom = QgsGeometry.fromWkt(geom_wkt)

            if geom and not geom.isEmpty():
                # Ensure geometry is MultiPolygon
                if not geom.isMultipart():
                    geom.convertToMultiType()

                feature.setGeometry(geom)
                area = geom.area()
                feature.setAttributes(["", area, raster_name, timestamp])
                features_to_add.append(feature)

        if not features_to_add:
            QgsMessageLog.logMessage(
                "Export aborted: no valid geometries produced from mask",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
            show_error_report(
                self.iface.mainWindow(),
                tr("Export Failed"),
                tr("No valid polygons could be created from the selection. "
                   "Try adjusting the refine settings or making a new selection.")
            )
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
                level=Qgis.MessageLevel.Warning
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
                level=Qgis.MessageLevel.Warning
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
            level=Qgis.MessageLevel.Info
        )
        QgsMessageLog.logMessage(
            f"Layer CRS: {result_layer.crs().authid()}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )
        QgsMessageLog.logMessage(
            f"Saved to: {gpkg_path}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        # Set renderer - red thin outline, transparent fill
        symbol = QgsFillSymbol.createSimple({
            "color": "0,0,0,0",
            "outline_color": "220,0,0,255",
            "outline_width": "0.5"
        })
        renderer = QgsSingleSymbolRenderer(symbol)
        result_layer.setRenderer(renderer)
        result_layer.triggerRepaint()
        self.iface.mapCanvas().refresh()

        QgsMessageLog.logMessage(
            f"Created segmentation layer: {layer_name} with {len(features_to_add)} polygons",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
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

        # Add layer to the group and ensure it is visible
        node = group.addLayer(layer)
        if node is not None:
            node.setItemVisibilityChecked(True)
        group.setItemVisibilityChecked(True)
        group.setExpanded(True)

    def _on_tool_deactivated(self):
        # Remove keyboard shortcut filter from all targets
        try:
            if self._shortcut_filter is not None:
                self.iface.mainWindow().removeEventFilter(self._shortcut_filter)
                canvas = self.iface.mapCanvas()
                canvas.viewport().removeEventFilter(self._shortcut_filter)
                canvas.removeEventFilter(self._shortcut_filter)
        except (RuntimeError, AttributeError):
            pass

        if self._stopping_segmentation:
            if self.dock_widget:
                self.dock_widget.set_segmentation_active(False)
            return

        # User switched to another tool (pan, etc.) while segmenting.
        # Re-activate segmentation tool silently to prevent accidental exits.
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(0, self._return_to_segmentation)

    def _return_to_segmentation(self):
        if self._stopping_segmentation:
            return
        if not self.map_tool or not self.dock_widget:
            return
        self._activate_segmentation_tool()

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
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        if self._shortcut_filter is not None:
            try:
                self.iface.mainWindow().removeEventFilter(self._shortcut_filter)
                canvas = self.iface.mapCanvas()
                canvas.viewport().removeEventFilter(self._shortcut_filter)
                canvas.removeEventFilter(self._shortcut_filter)
            except RuntimeError:
                pass
        self._stopping_segmentation = True
        self.iface.mapCanvas().unsetMapTool(self.map_tool)
        self._restore_previous_map_tool()
        self._stopping_segmentation = False
        self._reset_session()
        self.dock_widget.reset_session()

    def _on_refine_settings_changed(self, simplify: int, smooth: int, expand: int,
                                    fill_holes: bool, min_area: int):
        """Handle refinement control changes."""
        QgsMessageLog.logMessage(
            f"Refine settings: simplify={simplify}, smooth={smooth}, "
            f"expand={expand}, fill_holes={fill_holes}, min_area={min_area}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )
        self._refine_simplify = simplify
        self._refine_smooth = smooth
        self._refine_expand = expand
        self._refine_fill_holes = fill_holes
        self._refine_min_area = min_area

        # In both modes: update current mask preview only
        # Saved masks (green) keep their own refine settings from when they were saved
        self._update_mask_visualization()

        # Restore canvas focus so keyboard shortcuts (S, Ctrl+Z, etc.)
        # work immediately after changing refine parameters.
        try:
            self.iface.mapCanvas().setFocus()
        except RuntimeError:
            pass

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

        # Detect significant zoom-in requiring higher resolution.
        # Skip zoom re-encode when there are active points — re-encoding
        # destroys the current mask via lossy 64x64 logit transfer.
        # The existing crop is still valid (point is in bounds), so SAM
        # can predict just fine on the current encoding.
        has_active_points = (self._active_crop_points_positive
                             or self._active_crop_points_negative)  # noqa: W503
        if not has_active_points:
            # No active points — always use tight thresholds so any
            # meaningful zoom change triggers re-encode at the correct
            # resolution.  Loose thresholds (old 0.7/1.5) caused SAM to
            # reuse a closer-zoom encoding when the user zoomed out,
            # segmenting a small element instead of the full object.
            zoom_in_thresh = 0.85
            zoom_out_thresh = 1.15

            if self._is_online_layer:
                canvas = self.iface.mapCanvas()
                current_canvas_mupp = canvas.mapUnitsPerPixel()
                if self._current_crop_canvas_mupp and current_canvas_mupp > 0:
                    ratio = current_canvas_mupp / self._current_crop_canvas_mupp
                    if ratio < zoom_in_thresh or ratio > zoom_out_thresh:
                        return "zoom_changed"
            else:
                if self._current_crop_canvas_mupp is not None:
                    canvas = self.iface.mapCanvas()
                    current_mupp = canvas.mapUnitsPerPixel()
                    if current_mupp > 0:
                        ratio = current_mupp / self._current_crop_canvas_mupp
                        if ratio < zoom_in_thresh or ratio > zoom_out_thresh:
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
            ovlp_minx, ovlp_maxy, old_minx, old_miny, old_maxx, old_maxy,
            old_w, old_h)
        o_c1, o_r1 = geo_to_pixel(
            ovlp_maxx, ovlp_miny, old_minx, old_miny, old_maxx, old_maxy,
            old_w, old_h)
        o_r0 = max(0, min(o_r0, old_h))
        o_r1 = max(0, min(o_r1, old_h))
        o_c0 = max(0, min(o_c0, old_w))
        o_c1 = max(0, min(o_c1, old_w))
        if o_r0 >= o_r1 or o_c0 >= o_c1:
            return None

        patch = old_mask[o_r0:o_r1, o_c0:o_c1]

        # Overlap region in new pixel coords
        n_c0, n_r0 = geo_to_pixel(
            ovlp_minx, ovlp_maxy, new_minx, new_miny, new_maxx, new_maxy,
            new_w, new_h)
        n_c1, n_r1 = geo_to_pixel(
            ovlp_maxx, ovlp_miny, new_minx, new_miny, new_maxx, new_maxy,
            new_w, new_h)
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

    def _compute_auto_min_area(self):
        """Compute min_area for artifact removal based on current crop scale.

        SAM artifacts are small disconnected blobs (1-25 pixels) that appear
        regardless of input content.  They get slightly larger when the input
        image is heavily downsampled (high scale_factor = zoomed out).

        Uses sqrt scaling for a gentle progression that stays well below the
        size of any intentionally selected object (~50+ pixels).

        Returns pixel count in the 1024x1024 SAM mask.
        """
        scale = self._current_crop_scale_factor
        if scale is None or scale <= 0:
            # Online layers or unknown: use the MUPP ratio as proxy
            if (self._current_crop_actual_mupp
                    and self._current_crop_canvas_mupp  # noqa: W503
                    and self._current_crop_canvas_mupp > 0):  # noqa: W503
                scale = max(1.0, self._current_crop_actual_mupp
                            / self._current_crop_canvas_mupp * 2.0)  # noqa: W503
            else:
                scale = 1.0
        # Power curve centered on 200 (bumped ×2 per #12 for cleaner defaults).
        return max(100, int(200 * max(0.6, scale) ** 0.3))

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
                    canvas_extent)
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

        if not self._headless:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            QApplication.processEvents()

        try:
            return self._do_extract_and_encode(center_point, mupp_override)
        finally:
            if not self._headless:
                QApplication.restoreOverrideCursor()

    def _do_extract_and_encode(self, center_point, mupp_override):
        """Internal: does the actual crop extraction + SAM encoding."""
        from ..core.feature_encoder import extract_crop_from_online_layer, extract_crop_from_raster

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
                p1 = self._canvas_to_raster_xform.transform(
                    QgsPointXY(cx, cy))
                p2 = self._canvas_to_raster_xform.transform(
                    QgsPointXY(cx + canvas_mupp, cy))
                raster_mupp = math.sqrt(
                    (p2.x() - p1.x()) ** 2 + (p2.y() - p1.y()) ** 2)
            else:
                raster_mupp = canvas_mupp
            self._current_crop_canvas_mupp = canvas_mupp
            actual_mupp = mupp_override or raster_mupp
            self._current_crop_actual_mupp = actual_mupp
            image_np, crop_info, error = extract_crop_from_online_layer(
                self._current_layer, raster_pt_x, raster_pt_y,
                actual_mupp, crop_size=1024
            )
        elif not self._current_raster_path:
            self._encoding_in_progress = False
            if self._headless:
                self._headless_error = tr("No raster file path available. Please restart segmentation.")
                return False
            show_error_report(
                self.iface.mainWindow(),
                tr("Crop Error"),
                tr("No raster file path available. Please restart segmentation.")
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
                    layer_extent = (ext.xMinimum(), ext.yMinimum(),
                                    ext.xMaximum(), ext.yMaximum())
            except RuntimeError:
                pass

            scale_factor = mupp_override or 1.0
            self._current_crop_scale_factor = scale_factor
            self._current_crop_canvas_mupp = self.iface.mapCanvas().mapUnitsPerPixel()
            image_np, crop_info, error = extract_crop_from_raster(
                self._current_raster_path, raster_pt_x, raster_pt_y,
                crop_size=1024,
                layer_crs_wkt=layer_crs_wkt,
                layer_extent=layer_extent,
                scale_factor=scale_factor,
            )

        if error:
            self._encoding_in_progress = False
            QgsMessageLog.logMessage(
                f"Crop extraction failed: {error}",
                "AI Segmentation", level=Qgis.MessageLevel.Critical
            )
            if self._headless:
                self._headless_error = error
                return False
            show_error_report(
                self.iface.mainWindow(),
                tr("Crop Error"),
                error
            )
            return False

        # Encode the crop with SAM (this blocks for ~3-8s on CPU)
        try:
            self.predictor.set_image(image_np)
        except Exception as e:
            self._encoding_in_progress = False
            QgsMessageLog.logMessage(
                f"Image encoding failed: {str(e)}",
                "AI Segmentation", level=Qgis.MessageLevel.Critical
            )
            if self._headless:
                self._headless_error = str(e)
                return False
            show_error_report(
                self.iface.mainWindow(),
                tr("Encoding Error"),
                str(e)
            )
            return False

        self._current_crop_info = crop_info
        self._encoding_in_progress = False

        # Auto-compute min_area based on current crop scale
        self._refine_min_area = self._compute_auto_min_area()
        if self.dock_widget and hasattr(self.dock_widget, "min_area_spinbox"):
            self.dock_widget.min_area_spinbox.blockSignals(True)
            self.dock_widget.min_area_spinbox.setValue(self._refine_min_area)
            self.dock_widget.min_area_spinbox.blockSignals(False)

        # Restore keyboard focus to canvas after encoding (dock widget
        # updates during encoding can steal focus, breaking shortcuts).
        try:
            self.iface.mapCanvas().setFocus()
        except RuntimeError:
            pass

        QgsMessageLog.logMessage(
            "Encoded crop: bounds={}, shape={}, auto_min_area={}".format(
                crop_info["bounds"], crop_info["img_shape"],
                self._refine_min_area),
            "AI Segmentation", level=Qgis.MessageLevel.Info
        )
        return True

    def _freeze_active_crop(self, crop_info_override=None):
        """Freeze the current active crop's mask as a geographic polygon.

        The polygon is stored in raster CRS (same as save/export) and added
        to _frozen_sessions for composite display.

        Args:
            crop_info_override: If provided, use this instead of
                self._current_crop_info (needed when the caller has already
                overwritten _current_crop_info with a new crop).
        """
        if self.current_mask is None or self.current_transform_info is None:
            return
        try:
            from ..core.polygon_exporter import mask_to_polygons
            geometries = mask_to_polygons(
                self.current_mask, self.current_transform_info)
            if geometries:
                combined = QgsGeometry.unaryUnion(geometries)
                if combined and not combined.isEmpty():
                    session = FrozenCropSession(
                        polygon=combined,
                        points_positive=list(self._active_crop_points_positive),
                        points_negative=list(self._active_crop_points_negative),
                        crop_info=crop_info_override if crop_info_override is not None else self._current_crop_info,
                    )
                    self._frozen_sessions.append(session)
                    QgsMessageLog.logMessage(
                        f"Froze crop session #{len(self._frozen_sessions)} "
                        f"with {len(session.points_positive) + len(session.points_negative)} points",
                        "AI Segmentation", level=Qgis.MessageLevel.Info)
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to freeze active crop: {str(e)}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)

        # Reset active crop tracking (mask/low_res cleared by caller)
        self._active_crop_points_positive = []
        self._active_crop_points_negative = []
        self._mask_state_history = []

    def _handle_reencode(self, crop_status, raster_pt):
        """Handle re-encoding based on crop status. Returns True on success."""
        if crop_status == "no_crop":
            self.current_low_res_mask = None
            initial_scale = self._compute_initial_scale_factor()
            return self._extract_and_encode_crop(raster_pt, mupp_override=initial_scale)

        if crop_status == "outside_bounds":
            old_crop_info = self._current_crop_info
            old_history = list(self._mask_state_history)
            initial_scale = self._compute_initial_scale_factor()
            if not self._extract_and_encode_crop(
                    raster_pt, mupp_override=initial_scale):
                return False

            self._freeze_active_crop(crop_info_override=old_crop_info)
            # Clear stale prompts from old crop (#11, #35). The click handler
            # that called us will re-add the new point with correct polarity.
            self.prompts.clear()
            self._active_crop_points_positive = []
            self._active_crop_points_negative = []
            # Preserve history so the empty-mask rollback path still works.
            self._mask_state_history = old_history
            self.current_mask = None
            self.current_low_res_mask = None
            return True

        # zoom_changed: re-encode same crop at new resolution, keep all points
        old_crop_info = self._current_crop_info
        old_mask = self.current_mask

        all_pts = [
            (p[0], p[1]) for p in
            self.prompts.positive_points + self.prompts.negative_points
        ]
        if len(all_pts) > 1:
            center_x, center_y, mupp_or_scale = (
                self._compute_crop_center_and_mupp(all_pts))
            new_center = QgsPointXY(center_x, center_y)
        else:
            new_center = raster_pt
            mupp_or_scale = self._compute_initial_scale_factor()
        self.current_low_res_mask = None
        if not self._extract_and_encode_crop(
                new_center, mupp_override=mupp_or_scale):
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
            layer_name = ""
            sel = self.dock_widget.layer_combo.currentLayer()
            if sel:
                layer_name = sel.name()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Click is outside the '{layer}' raster. To segment another raster, stop the current segmentation first.").format(layer=layer_name),  # noqa: E501
                level=Qgis.MessageLevel.Warning,
                duration=8
            )
            return

        # Check crop status BEFORE adding to active points, so the zoom
        # detection sees the true "no active points" state after a save.
        crop_status = self._check_crop_status(raster_pt)


        # Save current mask state for undo before modifying anything
        # Cap at 30 entries (~30MB) to prevent unbounded memory growth.
        if len(self._mask_state_history) >= 30:
            self._mask_state_history.pop(0)
        self._mask_state_history.append({
            "mask": self.current_mask.copy() if self.current_mask is not None else None,
            "score": self.current_score,
            "transform_info": self.current_transform_info,
        })

        # For outside_bounds, _handle_reencode now owns prompt registration.
        # For every other status, register the point immediately so undo can
        # find it even while a (blocking) encoding runs.
        if crop_status != "outside_bounds":
            self.prompts.add_positive_point(raster_pt.x(), raster_pt.y())
            self._active_crop_points_positive.append((raster_pt.x(), raster_pt.y()))

        QgsMessageLog.logMessage(
            f"POSITIVE POINT at ({raster_pt.x():.6f}, {raster_pt.y():.6f})",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        # On-demand encoding: encode crop if needed
        if crop_status != "ok":
            if not self._handle_reencode(crop_status, raster_pt):
                if crop_status != "outside_bounds":
                    self.prompts.undo()
                    if self._active_crop_points_positive:
                        self._active_crop_points_positive.pop()
                if self._mask_state_history:
                    self._mask_state_history.pop()
                if self.map_tool:
                    self.map_tool.remove_last_marker()
                return

            # After outside_bounds reencode, add the new point (prompts were cleared).
            if crop_status == "outside_bounds":
                self.prompts.add_positive_point(raster_pt.x(), raster_pt.y())
                self._active_crop_points_positive.append((raster_pt.x(), raster_pt.y()))

        self._run_prediction()

        # Auto-revert if prediction produced an empty mask (no element detected)
        if (self.current_mask is not None
                and self.current_mask.sum() == 0  # noqa: W503
                and self._mask_state_history):  # noqa: W503
            self.prompts.undo()
            if self._active_crop_points_positive:
                self._active_crop_points_positive.pop()
            state = self._mask_state_history.pop()
            self.current_mask = state["mask"]
            self.current_score = state["score"]
            self.current_transform_info = state["transform_info"]
            self.current_low_res_mask = None
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self._update_ui_after_prediction()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("No element detected at this point. Try clicking on a different area."),
                level=Qgis.MessageLevel.Info,
                duration=4
            )
            return

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
                level=Qgis.MessageLevel.Info
            )
            return

        # Transform click from canvas CRS to raster CRS for all downstream use
        raster_pt = self._transform_to_raster_crs(point)

        if not self._is_point_in_raster_extent(raster_pt):
            if self.map_tool:
                self.map_tool.remove_last_marker()
            layer_name = ""
            sel = self.dock_widget.layer_combo.currentLayer()
            if sel:
                layer_name = sel.name()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Click is outside the '{layer}' raster. To segment another raster, stop the current segmentation first.").format(layer=layer_name),  # noqa: E501
                level=Qgis.MessageLevel.Warning,
                duration=8
            )
            return

        crop_status = self._check_crop_status(raster_pt)

        # Negative points outside the current crop don't make sense — they're
        # meant to refine the current selection, not start a new one.
        if crop_status == "outside_bounds":
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Right-click must be inside the current selection area."),
                level=Qgis.MessageLevel.Info,
                duration=4
            )
            return

        if len(self._mask_state_history) >= 30:
            self._mask_state_history.pop(0)
        self._mask_state_history.append({
            "mask": self.current_mask.copy() if self.current_mask is not None else None,
            "score": self.current_score,
            "transform_info": self.current_transform_info,
        })

        self.prompts.add_negative_point(raster_pt.x(), raster_pt.y())
        self._active_crop_points_negative.append((raster_pt.x(), raster_pt.y()))

        QgsMessageLog.logMessage(
            f"NEGATIVE POINT at ({raster_pt.x():.6f}, {raster_pt.y():.6f})",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        # Re-encode if needed (zoom changed)
        if crop_status != "ok":
            if not self._handle_reencode(crop_status, raster_pt):
                self.prompts.undo()
                if self._active_crop_points_negative:
                    self._active_crop_points_negative.pop()
                if self._mask_state_history:
                    self._mask_state_history.pop()
                if self.map_tool:
                    self.map_tool.remove_last_marker()
                return

        self._run_prediction()

        # Auto-revert if prediction produced an empty mask (no element detected)
        if (self.current_mask is not None
                and self.current_mask.sum() == 0  # noqa: W503
                and self._mask_state_history):  # noqa: W503
            self.prompts.undo()
            if self._active_crop_points_negative:
                self._active_crop_points_negative.pop()
            state = self._mask_state_history.pop()
            self.current_mask = state["mask"]
            self.current_score = state["score"]
            self.current_transform_info = state["transform_info"]
            self.current_low_res_mask = None
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self._update_ui_after_prediction()
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("No element detected at this point. Try clicking on a different area."),
                level=Qgis.MessageLevel.Info,
                duration=4
            )
            return

    def _run_prediction(self):
        """Run SAM prediction using active crop points only.

        When frozen sessions exist, only the active crop's points are sent
        to SAM (frozen polygons are composited during visualization).
        """
        import numpy as np
        from rasterio.transform import from_bounds as transform_from_bounds

        # Use only active crop points for prediction (not frozen points)
        active_pos = self._active_crop_points_positive
        active_neg = self._active_crop_points_negative
        all_active = active_pos + active_neg
        if not all_active:
            return

        if self._current_crop_info is None:
            QgsMessageLog.logMessage(
                "No crop encoded yet - cannot predict",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )
            return

        crop_bounds = self._current_crop_info["bounds"]
        img_shape = self._current_crop_info["img_shape"]
        img_height, img_width = img_shape

        minx, miny, maxx, maxy = crop_bounds
        img_clip_transform = transform_from_bounds(
            minx, miny, maxx, maxy, img_width, img_height)

        # Build point arrays from active crop points only
        from rasterio import transform as rio_transform
        point_coords_list = []
        point_labels_list = []
        for x, y in active_pos:
            row, col = rio_transform.rowcol(img_clip_transform, x, y)
            point_coords_list.append([col, row])
            point_labels_list.append(1)
        for x, y in active_neg:
            row, col = rio_transform.rowcol(img_clip_transform, x, y)
            point_coords_list.append([col, row])
            point_labels_list.append(0)

        point_coords = np.array(point_coords_list)
        point_labels = np.array(point_labels_list)

        # Use previous low_res_mask for iterative refinement (includes
        # transferred mask context after zoom re-encode)
        mask_input = None
        if self.current_low_res_mask is not None:
            mask_input = self.current_low_res_mask

        # Use multimask only on the very first point of a new polygon/crop
        # (more accurate initial selection). For subsequent points or
        # re-encoded crops with transferred mask, use single mask.
        one_positive = len(active_pos) == 1
        no_negatives = len(active_neg) == 0
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
                f"Prediction failed: {error_str}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            if self._headless:
                self._headless_error = error_str
                return
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
                f"Unexpected prediction error: {str(e)}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            return

        if use_multimask:
            total_pixels = masks[0].shape[0] * masks[0].shape[1]
            mask_areas = [int(m.sum()) for m in masks]

            # Avoid selecting the whole crop when clicking on small elements
            # in repetitive patterns (e.g. trees in an orchard). SAM's highest
            # score often goes to the "all similar elements" interpretation.
            small_enough = [
                i for i in range(len(scores))
                if mask_areas[i] < 0.8 * total_pixels
            ]
            if small_enough:
                best_idx = max(small_enough, key=lambda i: scores[i])
            else:
                best_idx = min(range(len(scores)), key=lambda i: mask_areas[i])

            QgsMessageLog.logMessage(
                f"Multimask: areas={mask_areas}, scores={[round(float(s), 3) for s in scores]}, picked={best_idx}",
                "AI Segmentation", level=Qgis.MessageLevel.Info
            )
            self.current_mask = masks[best_idx]
            self.current_score = float(scores[best_idx])
            self.current_low_res_mask = low_res_masks[best_idx:best_idx + 1]
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
                level=Qgis.MessageLevel.Info
            )
            self._update_mask_visualization()
        else:
            self._clear_mask_visualization()

        # Restore focus to canvas after prediction (dock widget updates
        # above can steal keyboard focus, breaking S/Ctrl+Z shortcuts).
        try:
            self.iface.mapCanvas().setFocus()
        except RuntimeError:
            pass

    def _update_mask_visualization(self):
        if self.mask_rubber_band is None:
            # Recreate rubber band if it was lost (e.g. after RuntimeError)
            try:
                self.mask_rubber_band = QgsRubberBand(
                    self.iface.mapCanvas(),
                    QgsWkbTypes.PolygonGeometry
                )
                self.mask_rubber_band.setColor(QColor(0, 120, 255, 100))
                self.mask_rubber_band.setStrokeColor(QColor(0, 80, 200))
                self.mask_rubber_band.setWidth(2)
            except Exception:
                return

        if self.current_mask is None or self.current_transform_info is None:
            # No active mask — but may have frozen sessions to display
            if self._frozen_sessions:
                self._display_frozen_composite_with_extra()
            else:
                self._clear_mask_visualization()
            return

        try:
            from ..core.polygon_exporter import apply_mask_refinement, count_significant_regions, mask_to_polygons

            # Apply refinement to preview in both modes (refine affects current mask only)
            mask_to_display = self.current_mask
            # Apply mask-level refinements (fill holes, expand/contract, min region)
            if (self._refine_fill_holes or self._refine_expand != 0
                    or self._refine_min_area > 0):  # noqa: W503
                mask_to_display = apply_mask_refinement(
                    self.current_mask,
                    expand_value=self._refine_expand,
                    fill_holes=self._refine_fill_holes,
                    min_area=self._refine_min_area,
                )

            # Detect disjoint regions and show message bar warning
            region_count = count_significant_regions(mask_to_display)
            is_disjoint = region_count > 1
            has_multiple_positive = len(self._active_crop_points_positive) >= 2
            if is_disjoint and not self._disjoint_warning_shown and has_multiple_positive:
                from qgis.core import Qgis
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("Disconnected parts detected. For best accuracy, segment one element at a time."),
                    level=Qgis.MessageLevel.Warning,
                    duration=6
                )
                self._disjoint_warning_shown = True

            geometries = mask_to_polygons(mask_to_display, self.current_transform_info)

            # Build composite: frozen polygons + active mask polygons
            all_geoms = [s.polygon for s in self._frozen_sessions]

            if geometries:
                active_combined = QgsGeometry.unaryUnion(geometries)
                if active_combined and not active_combined.isEmpty():
                    # Apply simplification to active mask preview
                    tolerance = self._compute_simplification_tolerance(
                        self.current_transform_info, self._refine_simplify
                    )
                    if tolerance > 0:
                        active_combined = active_combined.simplify(tolerance)
                    # Apply corner rounding (QGIS native C++ Chaikin)
                    if self._refine_smooth > 0:
                        active_combined = active_combined.smooth(
                            self._refine_smooth, 0.25)
                    all_geoms.append(active_combined)

            if all_geoms:
                combined = QgsGeometry.unaryUnion(all_geoms)
                if combined and not combined.isEmpty():
                    # Geometry is in raster CRS; transform to canvas CRS
                    self._transform_geometry_to_canvas_crs(combined)
                    self.mask_rubber_band.setToGeometry(combined, None)
                else:
                    self._clear_mask_visualization()
            else:
                self._clear_mask_visualization()

        except (ValueError, TypeError, RuntimeError) as e:
            QgsMessageLog.logMessage(
                f"Mask visualization error ({type(e).__name__}): {str(e)}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
            self._clear_mask_visualization()
        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Unexpected mask visualization error ({type(e).__name__}): {str(e)}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            self._clear_mask_visualization()

    def _clear_mask_visualization(self):
        if self.mask_rubber_band:
            try:
                self.mask_rubber_band.reset(QgsWkbTypes.PolygonGeometry)
            except RuntimeError:
                self.mask_rubber_band = None

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

            # Restore the exact mask state from before this point was added
            state = (self._mask_state_history.pop()
                     if self._mask_state_history else None)
            if state:
                self.current_mask = state["mask"]
                self.current_score = state["score"]
                self.current_transform_info = state["transform_info"]

            # Invalidate SAM logits so next click re-predicts from scratch.
            # Keep _current_crop_info so the next click in the same area
            # can reuse the encoding (avoids expensive 3-8s re-encode).
            self.current_low_res_mask = None

            # Also remove from per-crop point tracking
            if result[0] == "positive" and self._active_crop_points_positive:
                self._active_crop_points_positive.pop()
            elif result[0] == "negative" and self._active_crop_points_negative:
                self._active_crop_points_negative.pop()

            if self.prompts.point_count[0] + self.prompts.point_count[1] > 0:
                # Update visualization with restored mask (no re-prediction)
                self._update_ui_after_prediction()
            else:
                # Active crop is empty — check if we can unfreeze a previous crop
                if self._frozen_sessions:
                    self._unfreeze_last_session()
                else:
                    self.current_mask = None
                    self.current_score = 0.0
                    self._clear_mask_visualization()
                    self.dock_widget.set_point_count(0, 0)
        elif self._frozen_sessions:
            # No active points but have frozen sessions — unfreeze last one
            self._unfreeze_last_session()
        elif len(self.saved_polygons) > 0:
            # No points in current mask, but have saved masks
            # Ask user if they want to restore the last saved mask
            reply = QMessageBox.warning(
                self.iface.mainWindow(),
                tr("Edit saved polygon"),
                "{}\n{}".format(
                    tr("Warning: you are about to edit an already saved polygon."),
                    tr("Do you want to continue?")),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._restore_last_saved_mask()
            # Restore canvas focus after dialog (prevents shortcuts from breaking)
            try:
                self.iface.mapCanvas().setFocus()
            except RuntimeError:
                pass

    def _unfreeze_last_session(self):
        """Unfreeze the last frozen crop session back to active display.

        The frozen polygon is displayed as the active mask. No re-encode is
        performed — SAM state is invalidated and will re-encode on next click.
        """
        if not self._frozen_sessions:
            return

        session = self._frozen_sessions.pop()

        # Clear active crop state
        self.current_mask = None
        self.current_score = 0.0
        self.current_low_res_mask = None
        self._current_crop_info = None  # Force re-encode on next click
        self._mask_state_history = []

        # Restore only the unfrozen session's points to prompts and markers.
        # Frozen sessions' points are NOT added to prompts — they are already
        # baked into frozen polygons and should not inflate the point count.
        self.prompts.clear()
        if self.map_tool:
            self.map_tool.clear_markers()

        self._active_crop_points_positive = list(session.points_positive)
        self._active_crop_points_negative = list(session.points_negative)
        for pt in session.points_positive:
            self.prompts.add_positive_point(pt[0], pt[1])
            if self.map_tool:
                canvas_pt = self._transform_to_canvas_crs(QgsPointXY(pt[0], pt[1]))
                self.map_tool.add_marker(canvas_pt, is_positive=True)
        for pt in session.points_negative:
            self.prompts.add_negative_point(pt[0], pt[1])
            if self.map_tool:
                canvas_pt = self._transform_to_canvas_crs(QgsPointXY(pt[0], pt[1]))
                self.map_tool.add_marker(canvas_pt, is_positive=False)

        # Display: frozen polygons + unfrozen session polygon as rubberband
        # The unfrozen polygon becomes a "display-only" active state
        # (no numpy mask — will re-encode on next click)
        self._display_frozen_composite_with_extra(session.polygon)

        pos_count, neg_count = self.prompts.point_count
        self.dock_widget.set_point_count(pos_count, neg_count)

        QgsMessageLog.logMessage(
            f"Unfroze crop session, {len(self._frozen_sessions)} frozen remaining",
            "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _display_frozen_composite_with_extra(self, extra_polygon=None):
        """Display all frozen polygons (+ optional extra) as the rubberband."""
        if self.mask_rubber_band is None:
            return

        all_geoms = [s.polygon for s in self._frozen_sessions]
        if extra_polygon is not None:
            all_geoms.append(extra_polygon)

        if not all_geoms:
            self._clear_mask_visualization()
            return

        combined = QgsGeometry.unaryUnion(all_geoms)
        if combined and not combined.isEmpty():
            self._transform_geometry_to_canvas_crs(combined)
            self.mask_rubber_band.setToGeometry(combined, None)
        else:
            self._clear_mask_visualization()

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
        self._mask_state_history = []
        self._frozen_sessions = []
        self._active_crop_points_positive = []
        self._active_crop_points_negative = []
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
        self._refine_simplify = last_polygon.get("refine_simplify", 10)
        self._refine_smooth = last_polygon.get("refine_smooth", 0)
        self._refine_expand = last_polygon.get("refine_expand", 0)
        self._refine_fill_holes = last_polygon.get("refine_fill_holes", True)
        self._refine_min_area = last_polygon.get("refine_min_area", 200)

        # Update UI sliders without emitting signals
        self.dock_widget.set_refine_values(
            self._refine_simplify,
            self._refine_smooth,
            self._refine_expand,
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
            f"Refine: simplify={self._refine_simplify}, smooth={self._refine_smooth}, "
            f"expand={self._refine_expand}, fill_holes={self._refine_fill_holes}, "
            f"min_area={self._refine_min_area}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

    def _reset_session(self):
        self.prompts.clear()
        self._mask_state_history = []
        self._frozen_sessions = []
        self._active_crop_points_positive = []
        self._active_crop_points_negative = []
        self._disjoint_warning_shown = False
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

        # Reset on-demand encoding state
        self._current_crop_info = None
        self._current_raster_path = None
        self._current_crop_canvas_mupp = None
        self._current_crop_actual_mupp = None
        self._current_crop_scale_factor = None

        # Reset online layer state
        self._is_online_layer = False

        # Reset refinement settings to defaults (#12, #23)
        self._refine_simplify = 10
        self._refine_smooth = 0
        self._refine_expand = 0
        self._refine_fill_holes = True
        self._refine_min_area = 200  # overridden by _compute_auto_min_area() × 2

        if self.dock_widget:
            self.dock_widget.set_point_count(0, 0)
            self.dock_widget.set_saved_polygon_count(0)
