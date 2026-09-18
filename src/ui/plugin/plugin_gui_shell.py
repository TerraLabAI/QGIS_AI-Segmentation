







from __future__ import annotations

import os
import sys
from functools import lru_cache

from qgis.core import Qgis, QgsMessageLog, QgsProject
from qgis.gui import QgsRubberBand
from qgis.PyQt.QtCore import QSettings, Qt
from qgis.PyQt.QtGui import QIcon

from ...core.i18n import tr
from ...core.log_scrub import start_log_collector
from ...core.qt_compat import PolygonGeometry, QAction
from ..ai_segmentation_maptool import AISegmentationMapTool
from ..canvas_palette import PENDING_FILL, PENDING_STROKE


class GuiShellMixin:





    def _build_gui(self):
        from ...mcp_api import SegmentationMCPAPI
        self.mcp_api = SegmentationMCPAPI(self)




        try:
            from ...agent_bridge import register_product
            register_product("segmentation", self.mcp_api)
        except Exception as err:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Agent bridge not published: {err}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )




        self.processing_provider = None
        try:
            self._register_processing_provider()
        except Exception as err:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Processing provider not registered: {err}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )

        start_log_collector()



        try:
            from ...core.activation_manager import migrate_legacy_key
            migrate_legacy_key()
        except Exception:  # nosec B110
            pass

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
        self.action.setToolTip(
            "AI Segmentation by TerraLab\n{}".format(
                tr("Segment elements on raster images using AI"))
        )
        self.action.triggered.connect(self.toggle_dock_widget)

        from ..terralab_toolbar import add_action_to_toolbar, get_or_create_terralab_toolbar
        self.terralab_toolbar = get_or_create_terralab_toolbar(self.iface)
        add_action_to_toolbar(self.terralab_toolbar, self.action, "ai-segmentation")

        from ..terralab_menu import add_plugin_to_menu, add_to_plugins_menu, get_or_create_terralab_menu
        self.terralab_menu = get_or_create_terralab_menu(self.iface.mainWindow())
        add_plugin_to_menu(self.terralab_menu, self.action, "ai-segmentation")
        add_to_plugins_menu(self.iface, self.action)





        for a in list(self.terralab_menu.actions()):
            if a.objectName() == "_terralab_settings_action":
                self.terralab_menu.removeAction(a)
                break


        from ..cross_plugin_discovery import make_ai_edit_action
        ai_edit_icon_path = str(self.plugin_dir / "resources" / "icons" / "ai_edit_icon.png")
        ai_edit_icon = QIcon(ai_edit_icon_path) if os.path.exists(ai_edit_icon_path) else None
        self.ai_edit_action = make_ai_edit_action(
            self.iface.mainWindow(),
            self.iface,


            "AI Edit",
            tr("Generate imagery with AI on map zones (opens AI Edit plugin)"),
            icon=ai_edit_icon,
        )
        add_action_to_toolbar(self.terralab_toolbar, self.ai_edit_action, "ai-edit", is_cross_promo=True)
        add_plugin_to_menu(self.terralab_menu, self.ai_edit_action, "ai-edit", is_cross_promo=True)
        add_to_plugins_menu(self.iface, self.ai_edit_action)

        self.map_tool = AISegmentationMapTool(self.iface.mapCanvas())
        self.map_tool.positive_click.connect(self._on_positive_click)
        self.map_tool.negative_click.connect(self._on_negative_click)



        self.map_tool.double_click.connect(self._on_canvas_double_click)
        self.map_tool.cursor_moved.connect(self._on_handoff_cursor_moved)



        self.map_tool.cursor_moved.connect(self._on_hover_cursor_moved)
        self.map_tool.tool_deactivated.connect(self._on_tool_deactivated)

        self.iface.mapCanvas().extentsChanged.connect(self._on_manual_view_changed)



        QgsProject.instance().layersWillBeRemoved.connect(
            self._on_layers_will_be_removed)




        try:
            from ...core.output_store import sweep_stale_temp_layers
            sweep_stale_temp_layers()
            QgsProject.instance().readProject.connect(
                self._on_project_read_sweep_temp)
        except Exception:  # nosec B110
            pass

        self.mask_rubber_band = QgsRubberBand(
            self.iface.mapCanvas(),
            PolygonGeometry
        )
        self.mask_rubber_band.setColor(PENDING_FILL)
        self.mask_rubber_band.setStrokeColor(PENDING_STROKE)
        self.mask_rubber_band.setWidth(2)


        try:
            plugin_version = self._read_plugin_version()
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








        self._ensure_dock_widget()



        try:
            from ...core.run_autosave import log_and_clear_stale_pending
            log_and_clear_stale_pending(self._auto_run_id)
        except Exception:  # nosec B110
            pass






        settings = QSettings()
        settings.remove("AISegmentation/dock_shown_once")
        current_version = self._read_plugin_version()
        last_shown_version = settings.value(
            "AISegmentation/dock_shown_version", "", type=str)
        if last_shown_version != current_version:
            settings.setValue(
                "AISegmentation/dock_shown_version", current_version)
            first_install = not last_shown_version
            if self.dock_widget and (first_install or self.dock_widget.isVisible()):
                self.dock_widget.show()
                self.dock_widget.raise_()
                self._ensure_dock_height()

    def _register_processing_provider(self):






        from qgis.core import QgsApplication

        from ...processing.segmentation_provider import TERRALAB_PROVIDER_ID, TerraLabProcessingProvider
        registry = QgsApplication.processingRegistry()
        provider = TerraLabProcessingProvider()



        if not registry.addProvider(provider):






            stale = registry.providerById(TERRALAB_PROVIDER_ID)
            if stale is not None:
                registry.removeProvider(stale)
            provider = TerraLabProcessingProvider()
            if not registry.addProvider(provider):
                self.processing_provider = None
                QgsMessageLog.logMessage(
                    "Processing provider not registered: the id "
                    f"'{TERRALAB_PROVIDER_ID}' is already taken.",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning
                )
                return
        self.processing_provider = provider

    def _unregister_processing_provider(self):

        provider = getattr(self, "processing_provider", None)
        if provider is None:
            return
        from qgis.core import QgsApplication
        QgsApplication.processingRegistry().removeProvider(provider)
        self.processing_provider = None

    @staticmethod
    @lru_cache(maxsize=1)
    def _read_plugin_version() -> str:






        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        metadata_path = os.path.join(plugin_dir, "metadata.txt")
        try:
            with open(metadata_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("version="):
                        return line.split("=", 1)[1].strip()
        except OSError:
            pass
        return "unknown"

    def _ensure_dock_height(self):




        def _apply():
            try:
                dock = self.dock_widget
                mw = self.iface.mainWindow()
                if dock is None or mw is None or not dock.isVisible():
                    return
                from ...core.server_dials import dial_in_range
                fraction = dial_in_range("tuning.ui.dock_height_fraction", 0.85, 0.5, 1.0)
                target = int(mw.height() * fraction)
                if dock.height() >= target:
                    return
                mw.resizeDocks([dock], [target], Qt.Orientation.Vertical)
            except Exception:  # nosec B110
                pass
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(0, _apply)
