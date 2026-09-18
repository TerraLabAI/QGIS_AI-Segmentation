







from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsMessageLog,
)

from ...core.i18n import tr


class AutoRunPreflightMixin:


    def _auto_raster_guard_message(self, layer) -> str | None:











        self._auto_raster_guard_reason = "raster_shape"




        try:
            crs_valid = layer.crs().isValid()
        except (RuntimeError, AttributeError):
            crs_valid = False
        if not crs_valid:
            self._auto_raster_guard_reason = "raster_no_crs"
            return tr(
                "This layer has no valid coordinate reference system. "
                "Set one in Layer Properties before detecting."
            )





        from ...core.layer_conventions import crs_disagrees_with_extent
        try:
            mismatch = crs_disagrees_with_extent(layer.crs(), layer.extent())
        except (RuntimeError, AttributeError):
            mismatch = ""
        if mismatch == "degrees_in_metric_crs":
            self._auto_raster_guard_reason = "crs_mismatch"
            return tr(
                "This layer is filed under {crs}, which counts in metres, but "
                "its coordinates are longitude and latitude. Detection would "
                "measure the whole image as under a millimetre of ground and "
                "return nothing. Set the layer's CRS to the one the pixels "
                "are really in (EPSG:4326 for plain longitude and latitude) "
                "in Layer Properties, or reproject it."
            ).format(crs=self._auto_layer_crs_label(layer))
        if mismatch == "metres_in_geographic_crs":
            self._auto_raster_guard_reason = "crs_mismatch"
            return tr(
                "This layer is filed under {crs}, which counts in degrees, "
                "but its coordinates are projected metres. Set the layer's "
                "CRS to the projected one the pixels are really in, in Layer "
                "Properties, before detecting."
            ).format(crs=self._auto_layer_crs_label(layer))
        try:
            online = self._needs_canvas_render(layer)
        except (RuntimeError, AttributeError):
            online = False
        if not online:
            try:
                georef = self._is_layer_georeferenced(layer)
            except (RuntimeError, AttributeError):
                georef = True
            if not georef:
                self._auto_raster_guard_reason = "raster_not_georeferenced"




                return tr(
                    "This image has no position on the map, so Automatic "
                    "cannot place what it finds. Give it one with the QGIS "
                    "Georeferencer, or use Semi-Auto mode on it as is."
                )
            if self._raster_is_rotated(layer):
                self._auto_raster_guard_reason = "raster_rotated"





                return tr(
                    "This raster is rotated. Run Warp (Reproject) on it to "
                    "straighten it first. Semi-Auto mode cannot read it "
                    "either."
                )
        return None

    @staticmethod
    def _auto_layer_crs_label(layer) -> str:






        try:
            crs = layer.crs()
            return str(crs.authid() or crs.description() or "").strip() or tr("its CRS")
        except (RuntimeError, AttributeError):
            return tr("its CRS")






    _DRAWN_MAP_BASEMAPS = ("OSM", "Carto")

    def _warn_drawn_map_basemap(self, layer) -> None:







        try:
            from ...core.basemap_label import detect_basemap_label
            label = detect_basemap_label(layer)
        except Exception:  # noqa: BLE001
            return
        if label not in self._DRAWN_MAP_BASEMAPS:
            return
        try:
            self.iface.messageBar().pushWarning(
                "AI Segmentation",
                tr("{basemap} is a drawn map, not aerial imagery, so detection "
                   "usually finds nothing on it and the tiles are still "
                   "charged. Switch the layer to a satellite basemap "
                   "(Google, Esri, Bing) or to your own raster first."
                   ).format(basemap=label))
        except (RuntimeError, AttributeError):
            pass

    def _warn_local_raster_quality(self, layer) -> None:














        try:
            if self._needs_canvas_render(layer):
                return
        except (RuntimeError, AttributeError):
            return
        try:
            if layer.crs().isValid() and layer.crs().isGeographic():
                zone = self._auto_zone
                if zone is None:






                    run_crs = layer.crs()
                else:
                    run_crs = self._run_crs_for_layer(
                        layer, self._zone_in_layer_crs(zone, layer))
                if run_crs is None or run_crs == layer.crs():
                    self.iface.messageBar().pushInfo(
                        "AI Segmentation",
                        tr("This raster uses a geographic CRS (degrees), which "
                           "distorts the imagery sent to the AI. For best "
                           "results, reproject it to a projected CRS (e.g. UTM)."))
                    return
        except (RuntimeError, AttributeError):
            pass


        try:
            source = layer.source() or ""
            low = source.lower()
            if low.startswith("/vsi") or "://" in low:
                return
            import os
            if not os.path.isfile(source):
                return
            from ...core.server_dials import dial_copy, dial_in_range
            tip_floor_bytes = dial_in_range(
                "tuning.processing.overview_tip_min_bytes", 512 * 1024 * 1024,
                1024 * 1024, 10 * 1024 * 1024 * 1024)
            if os.path.getsize(source) < tip_floor_bytes:
                return
            from ...core.raster_dataset_cache import acquire_gdal_dataset
            ds = acquire_gdal_dataset(source)
            if ds is None or ds.RasterCount < 1:
                return
            has_overviews = ds.GetRasterBand(1).GetOverviewCount() > 0
            if not has_overviews:
                self.iface.messageBar().pushInfo(
                    "AI Segmentation",
                    dial_copy("copy.auto.overview_tip", tr(
                        "Tip: this raster has no overviews (pyramids). "
                        "Build them (Raster menu, Miscellaneous, Build "
                        "Overviews) to make detection much faster.")))
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    @staticmethod
    def _raster_is_rotated(layer) -> bool:






        try:
            source = layer.source()
        except (RuntimeError, AttributeError):
            return False
        if not source:
            return False


        low = source.lower()
        if low.startswith("/vsi") or "://" in low:
            return False
        try:


            from ...core.raster_dataset_cache import acquire_gdal_dataset
            ds = acquire_gdal_dataset(source)
            if ds is None:
                return False
            gt = ds.GetGeoTransform()
            if gt is None:
                return False
            scale = max(abs(gt[1]), abs(gt[5]), 1e-9)
            return (abs(gt[2]) / scale > 1e-3) or (abs(gt[4]) / scale > 1e-3)
        except Exception:  # noqa: BLE001
            return False

    def _offer_automatic_setup(self, reason: str) -> None:







        from ..dialogs.confirm_dialog import (
            PRIMARY,
            SECONDARY,
            ChoiceButton,
            ask_choice,
        )

        if ask_choice(
            self.iface.mainWindow(), tr("One-time setup"), reason,
            [ChoiceButton("later", tr("Not now"), SECONDARY),
             ChoiceButton("setup", tr("Set up now"), PRIMARY)],
            default="setup", escape="later",
        ) != "setup":
            return



        self._on_install_requested(include_local_model=False)

    def _push_auto_warning(self, message: str) -> None:






        if self._auto_headless_run:
            return
        try:
            self.iface.messageBar().pushWarning("AI Segmentation", message)
        except (RuntimeError, AttributeError):
            pass

    def _probe_imagery_behind_banner(self, layer, grid) -> tuple[float, str | None]:








        banner = None
        if not self._auto_headless_run and self.dock_widget is not None:
            try:
                banner = self.dock_widget.auto_status_banner
                banner.setText(tr("Preparing your zone..."))
                banner.setVisible(True)
            except (RuntimeError, AttributeError):
                banner = None
        if banner is not None:
            from qgis.PyQt.QtCore import QEventLoop
            from qgis.PyQt.QtWidgets import QApplication



            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        try:
            return self._online_imagery_verdict(layer, grid)
        finally:
            if banner is not None:
                try:
                    banner.setVisible(False)
                except RuntimeError:
                    pass

    def _abort_zone_outside_layer(self) -> None:



        msg = tr(
            "The zone is outside the selected raster layer. "
            "Pick the right layer or redraw the zone."
        )
        self._headless_error = msg
        try:
            self.dock_widget.set_auto_status("error", msg)
        except (RuntimeError, AttributeError):
            pass
        self._push_auto_warning(msg)
        QgsMessageLog.logMessage(
            "Auto detection: zone covers no ground on this layer; aborting",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
