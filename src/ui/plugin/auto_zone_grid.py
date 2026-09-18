







from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsGeometry,
    QgsMessageLog,
    QgsRectangle,
)
from qgis.gui import QgsRubberBand
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor

from ...core.qt_compat import PolygonGeometry
from ..canvas_palette import GRID_LINE
from .shared import zone_too_large_message



_ZONE_GRID_CACHE_SIZE = 8


class AutoZoneGridMixin:





    def _tile_grid_allowed(self) -> bool:
























        if getattr(self, "_auto_grid_suppressed", False):
            return False
        if (getattr(self, "_auto_worker", None) is not None or getattr(self, "_auto_review", None) is not None):
            return False
        dock = self.dock_widget
        if dock is None:
            return True
        return not (getattr(dock, "_auto_run_active", False) or getattr(dock, "_auto_review_active", False))

    def _tile_grid_revealed(self) -> bool:











        dock = self.dock_widget
        if dock is None:
            return True
        return bool(getattr(dock, "_auto_advanced_open", False))

    def _on_auto_advanced_toggled(self, opened: bool) -> None:


        if opened:
            self._update_credit_estimate()
        else:
            self._clear_zone_tile_grid()

    def _restore_tile_grid_after_run(self) -> None:










        self._auto_grid_suppressed = False
        self._update_credit_estimate()

    def _update_credit_estimate(self) -> None:

        if self._tile_manager is None:
            return




        self._auto_zone_tile_cap()


        if not self._tile_grid_allowed():
            return
        layer = self._get_active_raster_layer()
        if layer is None:
            self._clear_zone_tile_grid()
            if self.dock_widget:
                self.dock_widget.set_auto_detail_visible(False)
                self._hide_auto_cost_label()
            return








        _rejecting = getattr(self, "_auto_zone_cap_rejecting", False)
        if self._auto_zone is not None and not _rejecting:
            self._auto_zone_cap_rejecting = True
            try:
                fit = self._refit_stored_zone_for_free()
                if fit is not None and fit.geom is None:

                    self._reject_zone_over_free_cap(fit.requested_km2)
                    return
            finally:
                self._auto_zone_cap_rejecting = False



        if self.dock_widget:
            self.dock_widget.set_auto_detail_visible(self._auto_zone is not None)









            if self._auto_zone is not None:
                zone_in_layer = self._reproject_zone_to_run_crs(self._auto_zone, layer)
                machine_max = self._max_useful_detail(layer, zone_in_layer)
                low, high = self._detail_window_for_object(
                    layer, zone_in_layer, self._resolved_auto_object_class())
                self.dock_widget.set_auto_detail_range(
                    low, high, object_bound=high < machine_max)


                if getattr(self, "_auto_detail_seeded", None) is not None:
                    self._auto_detail_seeded = max(
                        low, min(high, self._auto_detail_seeded))

        grid = self._compute_auto_grid(layer)
        if grid is None:

            self._clear_zone_tile_grid()
            self._hide_auto_cost_label()
            return

        pixel_w = grid["pixel_w"]
        pixel_h = grid["pixel_h"]










        tiles_list = self._tile_manager.compute_grid(
            pixel_w, pixel_h, apply_cap=False)
        if tiles_list is not None:


            tiles_list = self._tiles_in_polygon(
                tiles_list, grid["bbox"], pixel_w, pixel_h, layer,
                grid.get("crs"))
            if len(tiles_list) > self._auto_zone_tile_cap():
                tiles_list = None
        credit_count = len(tiles_list) if tiles_list is not None else -1

        self._auto_est_tiles = credit_count

        QgsMessageLog.logMessage(
            f"Credit estimate: {pixel_w}x{pixel_h}px -> {credit_count} tile(s)",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )



        if self.dock_widget and self._auto_zone is not None:





            self.dock_widget.set_auto_zone_surface(self._auto_zone_area_km2())




            self.dock_widget.set_auto_zone_fit_available(
                self._coarsest_detail_fits(layer, zone_in_layer)
                if credit_count < 0 else False)
            self.dock_widget.set_auto_credit_estimate(credit_count)



            self._refresh_rerun_guard()


            try:
                zone_in_layer = self._reproject_zone_to_run_crs(self._auto_zone, layer)
                sized = self._grid_for_detail(
                    layer, zone_in_layer, self._get_auto_detail_level())
                if sized is not None:
                    ground_mupp = self._mupp_to_meters(layer, zone_in_layer, sized[2])






                    from ...core.detection_policy import (
                        gsd_warn_max_mupp,
                        object_tile_ceiling_m,
                    )
                    from ...core.tile_manager import TILE_SIZE
                    object_class = self._resolved_auto_object_class()
                    floor_m = self._detail_window_profile(object_class)[1]
                    tile_ground_m = TILE_SIZE * ground_mupp
                    wide_view = floor_m > 0 and tile_ground_m >= floor_m







                    ceiling_m = (object_tile_ceiling_m(object_class)
                                 if object_class else 0.0)
                    too_coarse = (tile_ground_m > ceiling_m if ceiling_m > 0
                                  else ground_mupp >= gsd_warn_max_mupp(0.5))
                    self.dock_widget.set_auto_detail_gsd_warning(
                        too_coarse and not wide_view,
                        can_improve=self._detail_max_clears_coarse(
                            layer, zone_in_layer, ceiling_m),
                    )


                    self._push_detail_feedback(layer, zone_in_layer, ground_mupp)
            except (RuntimeError, AttributeError):
                pass






            try:
                self._refresh_exemplar_size_warning()
            except (RuntimeError, AttributeError, TypeError, ValueError):

                pass

        if credit_count == -1:

            self._clear_zone_tile_grid()
            QgsMessageLog.logMessage(
                zone_too_large_message(self._tile_manager.max_tiles),
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            return

        if credit_count > 0 and self._auto_zone is not None:
            self._show_zone_tile_grid(layer, grid)

    def _detail_max_clears_coarse(
        self, layer, zone_in_layer, ceiling_m: float
    ) -> bool:











        try:
            from ...core.detection_policy import gsd_warn_max_mupp
            from ...core.tile_manager import TILE_SIZE

            top = int(self.dock_widget.auto_detail_slider.maximum())
            sized = self._grid_for_detail(layer, zone_in_layer, top)
            if sized is None:
                return False
            mupp = self._mupp_to_meters(layer, zone_in_layer, sized[2])
            if mupp <= 0:
                return False
            if ceiling_m > 0:
                return TILE_SIZE * mupp <= ceiling_m
            return mupp < gsd_warn_max_mupp(0.5)
        except (RuntimeError, AttributeError, ValueError, ZeroDivisionError):
            return False

    def _hide_auto_cost_label(self) -> None:









        if not self.dock_widget:
            return
        try:
            self.dock_widget.set_auto_zone_surface(None)
        except (RuntimeError, AttributeError):
            pass

    def _show_zone_tile_grid(self, layer, grid: dict, force: bool = False) -> None:


















        if not force and not self._tile_grid_allowed():
            return
        self._clear_zone_tile_grid()



        if not force and not self._tile_grid_revealed():
            return
        if self._tile_manager is None or self._auto_zone is None:
            return







        tiles = self._tile_manager.compute_grid(
            grid["pixel_w"], grid["pixel_h"], apply_cap=False)
        if not tiles or len(tiles) <= 1:
            return
        minx, miny, maxx, maxy = grid["bbox"]
        try:
            cols = len({tx for tx, _ty, _tw, _th in tiles})
            rows = len({ty for _tx, ty, _tw, _th in tiles})
            if cols < 1 or rows < 1:
                return



            grid_crs = QgsCoordinateReferenceSystem(grid.get("crs") or "") \
                if grid.get("crs") else layer.crs()
            poly = self._polygon_in_run_crs(layer)






            cache = getattr(self, "_zone_grid_geom_cache", None)
            cache_key = None
            try:
                if cache is not None:
                    zone_id = bytes(poly.asWkb()) if poly is not None else b"rect"
                    cache_key = (zone_id, cols, rows, minx, miny, maxx, maxy)
            except (RuntimeError, AttributeError, TypeError):
                cache, cache_key = None, None
            if cache_key is not None:
                cached = cache.get(cache_key)
                if cached is not None:
                    self._zone_grid_rubber_band = self._new_zone_grid_band(
                        cached, grid_crs)
                    return



            is_rect_zone = poly is None or poly.isGeosEqual(
                QgsGeometry.fromRect(poly.boundingBox()))




            engine = None
            if not is_rect_zone:
                engine = QgsGeometry.createGeometryEngine(poly.constGet())
                engine.prepareGeometry()







            zone_rect = None
            if is_rect_zone:
                try:
                    zr = self._reproject_zone_to_run_crs(self._auto_zone, layer)
                    if zr is not None and (zr.xMaximum() < maxx or zr.yMinimum() > miny):
                        zone_rect = QgsGeometry.fromRect(zr)
                except (RuntimeError, AttributeError, TypeError):
                    zone_rect = None
            step_x = (maxx - minx) / cols
            step_y = (maxy - miny) / rows
            cells = []
            for i in range(cols):
                for j in range(rows):
                    cx0 = minx + i * step_x
                    cy0 = miny + j * step_y
                    cell = QgsGeometry.fromRect(
                        QgsRectangle(cx0, cy0, cx0 + step_x, cy0 + step_y))
                    if not is_rect_zone:
                        if not engine.contains(cell.constGet()):
                            cell = cell.intersection(poly)
                            if cell.isEmpty():
                                continue
                    elif zone_rect is not None:
                        cell = cell.intersection(zone_rect)
                        if cell.isEmpty():
                            continue
                    cells.append(cell)
            if not cells:
                return








            parts = []
            for cell in cells:
                if cell.isMultipart():
                    parts.extend(cell.asGeometryCollection())
                else:
                    parts.append(cell)
            collected = QgsGeometry.collectGeometry(parts)
            if cache is not None and cache_key is not None:



                if len(cache) >= _ZONE_GRID_CACHE_SIZE:
                    cache.pop(next(iter(cache)))
                cache[cache_key] = collected
            self._zone_grid_rubber_band = self._new_zone_grid_band(collected, grid_crs)
        except (RuntimeError, AttributeError, ZeroDivisionError):
            pass

    def _new_zone_grid_band(self, geom, crs):



        rb = QgsRubberBand(self.iface.mapCanvas(), PolygonGeometry)







        rb.setColor(QColor(0, 0, 0, 0))




        rb.setStrokeColor(GRID_LINE)
        rb.setSecondaryStrokeColor(QColor(0, 0, 0, 0))
        rb.setLineStyle(Qt.PenStyle.DashLine)
        rb.setWidth(2)




        rb.setToGeometry(geom, crs)
        return rb

    def _clear_zone_tile_grid(self) -> None:

        if self._zone_grid_rubber_band is not None:
            self._safe_remove_rubber_band(self._zone_grid_rubber_band)
            self._zone_grid_rubber_band = None

    def _coarsest_detail_fits(self, layer, zone_in_layer) -> bool:








        try:
            slider_min = int(self.dock_widget.auto_detail_slider.minimum())
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return False
        try:
            sized = self._grid_for_detail(layer, zone_in_layer, slider_min)
            if not sized:
                return False
            tiles = int(sized[3])
            return 0 <= tiles <= int(self._auto_zone_tile_cap())
        except Exception:  # noqa: BLE001
            return False
