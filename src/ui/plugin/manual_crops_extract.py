









from __future__ import annotations

import os

from qgis.core import (
    Qgis,
    QgsMessageLog,
    QgsPointXY,
)

from ...core.i18n import tr
from ..error_report_dialog import show_error_report


class ManualCropsExtractMixin:


    def _extract_and_encode_crop(self, center_point, mupp_override=None, *,
                                 on_encoded=None, show_busy=True, quiet=False):












































        self._ensure_manual_encode_state()
        if self._encoding_in_progress:




            if self._headless:
                return False
            self._queued_crop_request = {
                "center": center_point,
                "mupp": mupp_override,
                "on_encoded": on_encoded,
                "show_busy": bool(show_busy),
                "quiet": bool(quiet),
            }
            return True

        if self._headless:
            ok = self._encode_crop_blocking(center_point, mupp_override)
            if ok and on_encoded is not None:
                on_encoded()
            return ok

        if self._is_online_layer:






            return self._begin_online_crop_fetch(
                center_point, mupp_override, on_encoded,
                show_busy=show_busy, quiet=quiet)








        return self._begin_file_crop_read(
            center_point, mupp_override, on_encoded,
            show_busy=show_busy, quiet=quiet)

    def _extract_crop_only(self, center_point, mupp_override, quiet=False):










        from ...core.feature_encoder import extract_crop_from_online_layer, extract_crop_from_raster

        raster_pt_x = center_point.x()
        raster_pt_y = center_point.y()

        if self._is_online_layer:
            actual_mupp = self._online_crop_mupp(mupp_override)
            image_np, crop_info, error, error_code_from_crop = extract_crop_from_online_layer(
                self._current_layer, raster_pt_x, raster_pt_y,
                actual_mupp, crop_size=1024
            )
        else:
            args = self._file_crop_read_args(center_point, mupp_override, quiet)
            if args is None:
                return None, None
            image_np, crop_info, error, error_code_from_crop = extract_crop_from_raster(**args)

        if error:
            self._report_crop_error(error, error_code_from_crop, quiet)
            return None, None

        return image_np, crop_info

    def _file_crop_read_args(self, center_point, mupp_override, quiet=False):









        if not self._current_raster_path:
            if quiet:
                return None
            if self._crop_error_went_to_panel("crop_error_no_path"):
                return None
            message = tr("This layer has no file to read. Pick another layer at "
                         "the top of the panel, then start again.")
            if self._headless:
                self._headless_error = message
                return None
            show_error_report(
                self.iface.mainWindow(),
                tr("Crop Error"),
                message,
                error_code="crop_error_no_path",
            )
            return None

        from ...core.layer_conventions import ground_unit_aspect

        layer_crs_wkt = None
        layer_extent = None




        ground_aspect = 1.0
        try:
            if self._current_layer.crs().isValid():
                layer_crs_wkt = self._current_layer.crs().toWkt()
                ground_aspect = ground_unit_aspect(
                    self._current_layer.crs(), center_point.x(), center_point.y())
            ext = self._current_layer.extent()
            if ext and not ext.isEmpty():
                layer_extent = (ext.xMinimum(), ext.yMinimum(),
                                ext.xMaximum(), ext.yMaximum())
        except RuntimeError:
            pass

        scale_factor = mupp_override or 1.0






        self._pending_crop_zoom_baseline = (
            scale_factor, self.iface.mapCanvas().mapUnitsPerPixel(), None)
        return {
            "raster_path": self._current_raster_path,
            "center_x": center_point.x(),
            "center_y": center_point.y(),
            "crop_size": 1024,
            "layer_crs_wkt": layer_crs_wkt,
            "layer_extent": layer_extent,
            "scale_factor": scale_factor,
            "ground_aspect": ground_aspect,
        }

    def _crop_error_went_to_panel(self, error_code, center_point=None) -> bool:








        from ...core.click_error_advice import click_error_notice

        selected = ""
        try:
            selected = self._current_layer.name()
        except (RuntimeError, AttributeError):  # nosec B110
            pass
        notice = click_error_notice(
            error_code or "", selected,
            self._visible_raster_under_click(center_point))
        if not notice:
            return False
        if self._headless:
            self._headless_error = notice
        else:
            try:
                self.dock_widget.show_manual_notice(notice)
            except (RuntimeError, AttributeError):


                return False




        try:
            from ...core.telemetry_errors import track_plugin_error
            track_plugin_error(
                stage="segment", error_code=error_code or "crop_error_unknown",
                message="")
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        return True

    def _report_crop_error(self, error, error_code_from_crop, quiet=False) -> None:




        QgsMessageLog.logMessage(
            f"Crop extraction failed: {error}",
            "AI Segmentation", level=Qgis.MessageLevel.Critical
        )
        if quiet:
            return



        if self._crop_error_went_to_panel(error_code_from_crop):
            return
        if error_code_from_crop == "crop_error_rasterio_unavailable":












            if not self._rasterio_repair_attempted:
                self._rasterio_repair_attempted = True





                try:
                    from ...core.telemetry_errors import track_plugin_error
                    track_plugin_error(
                        stage="segment",
                        error_code="crop_error_rasterio_unavailable",
                        message=error,
                        module="manual_crops",
                    )
                except Exception:
                    pass  # nosec B110
                from ...core.venv_manager import purge_package_from_venv
                purge_package_from_venv("rasterio")
                self._recover_broken_venv(error)
                return
            QgsMessageLog.logMessage(
                "rasterio still unavailable after a repair; not repairing "
                "again this session",
                "AI Segmentation", level=Qgis.MessageLevel.Critical,
            )
            if self._headless:
                self._headless_error = error
                return





            report_key = ("", "crop_error_rasterio_unavailable")
            if report_key in self._crop_errors_reported:
                return
            self._crop_errors_reported.add(report_key)
            show_error_report(
                self.iface.mainWindow(),
                tr("Crop Error"),
                tr("The imagery reader could not be loaded, and repairing "
                   "the installation did not fix it. Please report this "
                   "so we can look into it.\n\n{details}").format(
                    details=error),
                error_code="crop_error_rasterio_unavailable",
                track=False,
            )
            return
        if self._headless:
            self._headless_error = error
            return




        report_key = (
            os.path.normcase(self._current_raster_path or ""),
            error_code_from_crop or "crop_error_unknown",
        )
        if report_key in self._crop_errors_reported:
            QgsMessageLog.logMessage(
                "Same crop error already reported this session; "
                "not showing the dialog again",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return
        self._crop_errors_reported.add(report_key)
        show_error_report(
            self.iface.mainWindow(),
            tr("Crop Error"),
            error,
            error_code=error_code_from_crop or "crop_error_unknown",
        )

    def _prewarm_manual_encode(self) -> None:













        self._ensure_manual_encode_state()
        if self._headless or self._encoding_in_progress or self.predictor is None or self._refine_handoff_active:
            return
        try:
            canvas = self.iface.mapCanvas()



            if canvas.mapTool() is not self.map_tool:
                return
            center = self._transform_to_raster_crs(QgsPointXY(canvas.center()))
        except Exception:  # noqa: BLE001
            return
        if not self._is_point_in_raster_extent(center):
            return
        if self._is_online_layer:
            self._prewarm_online_manual_encode(center)
            return
        QgsMessageLog.logMessage(
            "Prewarming first crop at view center",
            "AI Segmentation", level=Qgis.MessageLevel.Info
        )





        from ...core.crop_window import snap_center_to_grid
        scale = self._compute_initial_scale_factor()


        cx, cy = snap_center_to_grid(
            center.x(), center.y(), scale or 1.0, self._get_native_pixel_size())
        self._begin_file_crop_read(
            QgsPointXY(cx, cy), scale, None, quiet=True, show_busy=False)

    def _prewarm_online_manual_encode(self, center) -> None:














        from ...core.online_layer_twin import online_prewarm_enabled

        if not online_prewarm_enabled():
            return
        QgsMessageLog.logMessage(
            "Prewarming first crop at view center through a private copy "
            "of the rendered layer",
            "AI Segmentation", level=Qgis.MessageLevel.Info
        )
        self._extract_and_encode_crop(
            center, mupp_override=None, show_busy=False, quiet=True)
