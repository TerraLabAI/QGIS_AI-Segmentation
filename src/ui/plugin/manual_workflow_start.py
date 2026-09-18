







from __future__ import annotations

import math

from qgis.core import Qgis, QgsCoordinateTransform, QgsCsException, QgsMessageLog, QgsProject, QgsRasterLayer
from qgis.PyQt.QtCore import QSettings

from ...core.i18n import tr
from ..dialogs.confirm_dialog import question, warning_box
from ..shortcut_filter import ShortcutFilter
from .shared import SETTINGS_KEY_LAST_MANUAL_SESSION_TS, SETTINGS_KEY_TUTORIAL_SHOWN, looks_like_pixel_image


class ManualWorkflowStartMixin:





    def _on_manual_engine_changed(self, cloud: bool) -> None:








        if not cloud:
            return
        self._maybe_warmup_auto()

    def _warmup_if_manual_cloud(self) -> None:










        try:
            from ...core.manual_cloud_route import (
                manual_cloud_route_enabled,
                manual_cloud_route_offered,
            )

            if not (manual_cloud_route_enabled() and manual_cloud_route_offered()):
                return
        except Exception:  # noqa: BLE001
            return
        self._maybe_warmup_auto()

    def _on_start_segmentation(self, layer: QgsRasterLayer):




        if not getattr(self, "_refine_handoff_active", False):
            self._drop_cloud_correct_predictor()
            if self._ensure_manual_cloud_predictor():










                self._maybe_warmup_auto()
        if self.predictor is None:
            if getattr(self, "_local_ai_load_failed", False):




                warning_box(
                    self.iface.mainWindow(),
                    tr("AI not available"),
                    tr("The offline AI did not load, so this session cannot "
                       "start. Use the Install button in the panel to set it "
                       "up again.")
                )
                return




            self._load_predictor()
            if self._arm_manual_start_when_ready(layer):
                from ...core.server_dials import dial_in_range
                duration = dial_in_range(
                    "tuning.manual.ai_loading_notice_s", 6, 3, 15)
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("The AI is still loading. This session starts on its "
                       "own as soon as it is ready."),
                    level=Qgis.MessageLevel.Info, duration=duration)
                return
            warning_box(
                self.iface.mainWindow(),
                tr("Not Ready"),
                tr("The AI is still loading. Try again in a few seconds.")
            )
            return


        if not self._is_layer_valid(layer):
            QgsMessageLog.logMessage(
                "Layer was deleted before segmentation could start",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return

        try:
            layer_name = layer.name().replace(" ", "_")





            raster_path = layer.source()
        except RuntimeError:
            QgsMessageLog.logMessage(
                "Layer deleted during segmentation start",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return





        is_online = self._needs_canvas_render(layer)
        non_georeferenced = (
            not is_online and not self._is_layer_georeferenced(layer))


        if not is_online:
            try:
                ext = layer.extent()
                if ext and not ext.isEmpty():
                    coords = (ext.xMinimum(), ext.yMinimum(),
                              ext.xMaximum(), ext.yMaximum())
                    if any(math.isnan(c) or math.isinf(c) for c in coords):
                        self._refuse_manual_start(
                            tr("Invalid Layer"),
                            tr("This layer has no usable position on the map. "
                               "Open it in QGIS and check its extent."),
                            "invalid_layer")
                        return
            except RuntimeError:
                pass






        if not is_online and not non_georeferenced and self._raster_is_rotated(layer):
            self._refuse_manual_start(
                tr("Rotated raster"),
                tr("This raster is rotated. Run Warp (Reproject) on it to "
                   "straighten it before segmenting."),
                "rotated_raster")
            return

        self._reset_session()




        try:
            self.dock_widget.publish_refine_settings()
        except (RuntimeError, AttributeError):
            pass  # nosec B110



        self._speculative_manual_crop = False

        self._current_layer = layer
        self._current_layer_name = layer_name



        self._is_online_layer = is_online


        self._is_non_georeferenced_mode = non_georeferenced
        if self._is_non_georeferenced_mode:
            QgsMessageLog.logMessage(
                "Non-georeferenced image detected - using pixel coordinate mode. "
                "Polygons will be created in pixel coordinates.",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )




            if not looks_like_pixel_image(layer):
                self.iface.messageBar().pushInfo(
                    "AI Segmentation",
                    tr("This raster has no coordinate reference system, so "
                       "polygons will use pixel coordinates. Set a CRS in "
                       "Layer Properties for georeferenced output."))

        if self._is_online_layer:
            QgsMessageLog.logMessage(
                f"Layer read through the QGIS renderer ({layer.dataProvider().name()})",
                "AI Segmentation", level=Qgis.MessageLevel.Info
            )







        self._current_raster_path = raster_path



        if not self._is_online_layer and not self._is_non_georeferenced_mode:
            self._offer_zoom_to_off_screen_raster(layer)







        self._start_manual_credit_session()



        self._rebuild_manual_crs_transforms()
        self._start_canvas_crs_watch()



        self.predictor.warm_up()


        import time as _time
        self._segmentation_start_ts = _time.time()





        try:
            QSettings().setValue(
                SETTINGS_KEY_LAST_MANUAL_SESSION_TS,
                int(self._segmentation_start_ts))
        except Exception:  # noqa: BLE001
            pass  # nosec B110

        self._activate_segmentation_tool()







        if not self._headless and not self._refine_handoff_active:
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._prewarm_manual_encode)

    def _refuse_manual_start(self, title: str, body: str, code: str) -> None:







        try:
            from ...core.telemetry_errors import track_plugin_error

            track_plugin_error(stage="other", error_code=code, message=code)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        warning_box(self.iface.mainWindow(), title, body)

    def _arm_manual_start_when_ready(self, layer) -> bool:





        worker = getattr(self, "_predictor_worker", None)
        try:
            if worker is None or not worker.isRunning():
                return False
            self._start_manual_when_ready = layer
            if not getattr(self, "_manual_start_when_ready_wired", False):
                worker.done.connect(self._on_manual_start_when_ready)
                self._manual_start_when_ready_wired = True
            return True
        except (RuntimeError, AttributeError):
            return False

    def _on_manual_start_when_ready(self, predictor, err_msg: str) -> None:

        self._manual_start_when_ready_wired = False
        layer = getattr(self, "_start_manual_when_ready", None)
        self._start_manual_when_ready = None
        if layer is None or self.predictor is None:
            return
        dock = self.dock_widget
        if dock is None or getattr(dock, "_segmentation_active", False):
            return
        try:
            if not self._is_layer_valid(layer):
                return
        except RuntimeError:
            return
        self._on_start_segmentation(layer)

    def _offer_zoom_to_off_screen_raster(self, layer) -> None:





        try:
            canvas = self.iface.mapCanvas()
            view = canvas.extent()
            extent = layer.extent()
            if view.isEmpty() or extent.isEmpty():
                return
            canvas_crs = canvas.mapSettings().destinationCrs()
            layer_crs = layer.crs()
            if (canvas_crs.isValid() and layer_crs.isValid()
                    and canvas_crs != layer_crs):
                xform = QgsCoordinateTransform(
                    layer_crs, canvas_crs, QgsProject.instance())
                extent = xform.transformBoundingBox(extent)
            if view.intersects(extent):
                return
        except (QgsCsException, RuntimeError, AttributeError):
            return
        if not question(
            self.iface.mainWindow(),
            tr("Zoom to the layer?"),
            tr("It is outside the current map view."),
            default_yes=True, yes_label=tr("Zoom"),
        ):
            return
        try:
            canvas.setExtent(extent)
            canvas.refresh()
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _activate_segmentation_tool(self):





        try:
            self._warmup_if_manual_cloud()
        except Exception:  # noqa: BLE001  # nosec B110
            pass

        current_tool = self.iface.mapCanvas().mapTool()
        if current_tool and current_tool != self.map_tool:
            self._previous_map_tool = current_tool

        self.iface.mapCanvas().setMapTool(self.map_tool)



        self.dock_widget.set_segmentation_active(True, layer=self._current_layer)






        if self._shortcut_filter is None:
            self._shortcut_filter = ShortcutFilter(self)
        self.iface.mainWindow().installEventFilter(self._shortcut_filter)
        canvas = self.iface.mapCanvas()
        canvas.viewport().installEventFilter(self._shortcut_filter)
        canvas.installEventFilter(self._shortcut_filter)


        self._show_tutorial_notification()

    def _show_tutorial_notification(self):







        settings = QSettings()
        if settings.value(SETTINGS_KEY_TUTORIAL_SHOWN, False, type=bool):
            return
        settings.setValue(SETTINGS_KEY_TUTORIAL_SHOWN, True)

        from ...core.activation_manager import TUTORIAL_URL_FALLBACK, get_tutorial_url
        from ...core.server_dials import safe_web_url
        tutorial_url = safe_web_url(get_tutorial_url(), TUTORIAL_URL_FALLBACK)
        message = '{} <a href="{}">{}</a>'.format(
            tr("New here?"),
            tutorial_url,
            tr("Watch the tutorial"))

        from ...core.server_dials import dial_in_range
        duration = dial_in_range("tuning.manual.tutorial_notice_s", 10, 5, 20)
        self.iface.messageBar().pushMessage(
            "AI Segmentation",
            message,
            level=Qgis.MessageLevel.Info,
            duration=duration
        )
