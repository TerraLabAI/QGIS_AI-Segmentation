





from __future__ import annotations

import math
import re

from qgis.core import (
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProject,
    QgsRectangle,
)
from qgis.gui import QgsRubberBand
from qgis.PyQt.QtCore import Qt

from ...core.exemplar_size import exemplar_at_max_detail, exemplar_too_small
from ...core.i18n import tr
from ...core.qt_compat import PolygonGeometry
from ..canvas_palette import (
    EXCLUDE_FILL,
    EXCLUDE_STROKE,
    EXEMPLAR_FILL,
    EXEMPLAR_STROKE,
)


class ExemplarsMixin:






    def _on_add_exemplar_requested(self, label: int) -> None:












        if self._exemplar_maptool is not None:
            self._restore_maptool_after_exemplar()
            return


        if self._auto_worker is not None or self._auto_review is not None:
            return
        if self._auto_zone is None:
            return
        self._sync_exemplar_store_tier()
        store = self._auto_exemplar_store
        if store.is_full_for(int(label)):


            if store.is_full_on_free_only(int(label)) and self.dock_widget:
                try:
                    self.dock_widget.show_auto_exemplar_upsell()
                except (RuntimeError, AttributeError):
                    pass
            return
        from ..polygon_zone_maptool import PolygonZoneMapTool
        canvas = self.iface.mapCanvas()

        self._maptool_before_exemplar = canvas.mapTool()
        self._pending_exemplar_label = int(label)
        color = EXEMPLAR_STROKE if label == 1 else EXCLUDE_STROKE



        tool = PolygonZoneMapTool(canvas, color)
        tool.zone_selected.connect(self._on_exemplar_polygon_drawn)




        tool.zone_cleared.connect(self._on_exemplar_cancelled)
        tool.back_requested.connect(self._on_exemplar_cancelled)


        try:
            tool.tool_deactivated.connect(self._on_exemplar_tool_deactivated)
        except (RuntimeError, AttributeError):
            pass
        self._exemplar_maptool = tool


        try:
            self.dock_widget.set_auto_shortcuts_enabled(False)
        except (RuntimeError, AttributeError):
            pass
        self._connect_exemplar_undo_shortcut()



        self._suspend_exemplar_deactivate = True
        try:
            canvas.setMapTool(tool)
        finally:
            self._suspend_exemplar_deactivate = False





        self._auto_exemplar_arming = True
        try:
            self.dock_widget.set_auto_exemplar_armed(int(label))
        except (RuntimeError, AttributeError):
            pass

    def _on_exemplar_polygon_drawn(self, geom) -> None:











        label = int(getattr(self, "_pending_exemplar_label", 1))











        canvas_geom = geom
        geom = self._exemplar_geom_in_zone_crs(geom)
        zone_poly = self._auto_zone_polygon
        if (zone_poly is not None and geom is not None and not geom.isEmpty()):
            try:
                clipped = geom.intersection(zone_poly)
            except (RuntimeError, AttributeError):
                clipped = None
            if clipped is not None:
                if clipped.isEmpty():


                    self.iface.messageBar().pushWarning(
                        "AI Segmentation",
                        tr("Draw your example inside the selected zone."))
                    self._restore_maptool_after_exemplar()
                    return
                if not self._geom_centroid_in(geom, zone_poly):
                    geom = clipped
        if geom is None or geom.isEmpty():
            self._restore_maptool_after_exemplar()
            return
        rect = geom.boundingBox()


        canvas_geom = self._exemplar_geom_in_canvas_crs(geom, canvas_geom)
        eid = self._auto_exemplar_store.add(
            QgsRectangle(rect), label, thumbnail=None, polygon=geom)
        if eid is not None:
            band = self._make_exemplar_band_poly(canvas_geom, label)
            if band is not None:
                self._exemplar_bands[eid] = band





            self._prebuild_exemplar_stamp(eid)
            entry = self._auto_exemplar_store.get(eid)
            if entry is not None and (entry.thumbnail is None
                                      or entry.thumbnail.isNull()):
                entry.thumbnail = self._capture_exemplar_thumbnail(
                    canvas_geom.boundingBox(), polygon=canvas_geom)
                self._refresh_exemplar_chips()
            try:
                from ...core import telemetry_run_events
                telemetry_run_events.track_exemplar_added(
                    count_after=self._auto_exemplar_store.count(),
                    label="include" if label == 1 else "exclude")
            except Exception:
                pass  # nosec B110
        self._restore_maptool_after_exemplar()






        self._refresh_exemplar_size_warning()



        self._refresh_detail_band_after_exemplars()


        self._refresh_rerun_guard()

    def _exemplar_canvas_crs_now(self):

        try:
            return self.iface.mapCanvas().mapSettings().destinationCrs()
        except (RuntimeError, AttributeError):
            return None

    def _exemplar_geom_converted(self, geom, source_crs, target_crs):








        try:
            if (geom is None or source_crs is None or target_crs is None
                    or not source_crs.isValid() or not target_crs.isValid()
                    or source_crs == target_crs):
                return geom
            out = QgsGeometry(geom)
            out.transform(QgsCoordinateTransform(
                source_crs, target_crs, QgsProject.instance()))
            return out
        except Exception:  # noqa: BLE001
            return geom

    def _exemplar_geom_in_zone_crs(self, geom):

        return self._exemplar_geom_converted(
            geom, self._exemplar_canvas_crs_now(),
            getattr(self, "_auto_zone_crs", None))

    def _exemplar_geom_in_canvas_crs(self, geom, fallback):





        zone_crs = getattr(self, "_auto_zone_crs", None)
        if zone_crs is None:
            return fallback
        return self._exemplar_geom_converted(
            geom, zone_crs, self._exemplar_canvas_crs_now())

    def _refresh_detail_band_after_exemplars(self) -> None:




        try:
            self._update_credit_estimate()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass



        if self._auto_worker is not None or self._auto_review is not None:
            return
        try:
            size = self._exemplar_size_for_plan()







            if size != getattr(self, "_auto_exemplar_seed_m", None):
                self._auto_exemplar_seed_m = size
                self._reseed_auto_detail_from_blob(
                    self._resolved_auto_object_class())
            rp = getattr(self, "_auto_run_plan", None)
            seen = rp.get("exemplar_size_m") if isinstance(rp, dict) else None
            if size != seen:
                self._fetch_auto_run_plan(self._resolved_auto_object_class())
        except (RuntimeError, AttributeError, TypeError, ValueError):

            pass

    @staticmethod
    def _geom_centroid_in(geom, zone_poly) -> bool:



        try:
            c = geom.centroid()
            if c is None or c.isEmpty():
                return True
            return bool(zone_poly.contains(c))
        except (RuntimeError, AttributeError):
            return True

    def _refresh_exemplar_size_warning(self) -> None:



















        if not self.dock_widget or getattr(self, "_auto_exemplar_arming", False):
            return
        try:
            store = self._auto_exemplar_store
        except AttributeError:
            return
        if store.count() == 0:
            try:
                self.dock_widget.clear_auto_exemplar_size_warning()
            except (RuntimeError, AttributeError):
                pass
            return
        layer = self._get_active_raster_layer()
        if layer is None:
            return
        try:
            run_mupp = self._exemplar_run_mupp(layer)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return
        if run_mupp <= 0:
            return
        from ...core.detection_policy import exemplar_render_abs_min_side_px
        abs_min_side = exemplar_render_abs_min_side_px(self._STAMP_ABS_MIN_SIDE)
        any_valid = False
        too_small = False
        for ex in store.list():
            try:
                rect = self._reproject_zone_to_run_crs(ex.map_rect, layer)
            except (RuntimeError, AttributeError):
                continue
            any_valid = True
            if exemplar_too_small(
                    rect.width(), rect.height(), run_mupp, abs_min_side):
                too_small = True
                break
        try:
            if any_valid and too_small:






                at_max_detail = self._auto_detail_slider_at_max()
                self.dock_widget.show_auto_exemplar_size_warning(at_max_detail=at_max_detail)
            else:
                self.dock_widget.clear_auto_exemplar_size_warning()
        except (RuntimeError, AttributeError):
            pass

    def _auto_detail_slider_at_max(self) -> bool:





        try:
            slider = self.dock_widget.auto_detail_slider
            return exemplar_at_max_detail(int(slider.value()), int(slider.maximum()))
        except (RuntimeError, AttributeError):
            return False

    def _make_exemplar_band_poly(self, geom, label: int):

        try:
            canvas = self.iface.mapCanvas()
            band = QgsRubberBand(canvas, PolygonGeometry)
            col = EXEMPLAR_STROKE if label == 1 else EXCLUDE_STROKE
            band.setColor(col)
            band.setFillColor(EXEMPLAR_FILL if label == 1 else EXCLUDE_FILL)
            band.setWidth(2)
            band.setToGeometry(geom, None)
            return band
        except (RuntimeError, AttributeError):
            return None

    def _capture_exemplar_thumbnail(self, rect, polygon=None):







        try:
            from qgis.PyQt.QtCore import QRect

            from ...core.server_dials import dial_in_range

            side_px = dial_in_range("tuning.ui.exemplar_thumbnail_px", 240, 120, 480)
            canvas = self.iface.mapCanvas()
            m2p = canvas.getCoordinateTransform()

            top_left = m2p.transform(rect.xMinimum(), rect.yMaximum())
            bot_right = m2p.transform(rect.xMaximum(), rect.yMinimum())
            try:
                dpr = float(canvas.devicePixelRatioF())
            except (AttributeError, TypeError):
                dpr = 1.0
            x = int(min(top_left.x(), bot_right.x()) * dpr)
            y = int(min(top_left.y(), bot_right.y()) * dpr)
            w = int(abs(bot_right.x() - top_left.x()) * dpr)
            h = int(abs(bot_right.y() - top_left.y()) * dpr)
            if w < 4 or h < 4:
                return None
            crop = canvas.grab().copy(QRect(x, y, w, h)).toImage()
            if crop.isNull():
                return None






            return crop.scaled(
                side_px, side_px,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
        except Exception:  # noqa: BLE001
            return None

    def _on_exemplar_cancelled(self) -> None:

        self._restore_maptool_after_exemplar()

    def _on_exemplar_remove_requested(self, exemplar_id: str) -> None:

        self._auto_exemplar_store.remove(exemplar_id)
        band = self._exemplar_bands.pop(exemplar_id, None)
        self._remove_rubber_band(band)
        self._refresh_exemplar_chips()


        self._refresh_exemplar_size_warning()
        self._refresh_detail_band_after_exemplars()
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_exemplar_removed(count_after=self._auto_exemplar_store.count())
        except Exception:
            pass  # nosec B110

        self._refresh_rerun_guard()

    def _clear_exemplars(self) -> None:

        self._auto_exemplar_store.clear()
        for band in self._exemplar_bands.values():
            self._remove_rubber_band(band)
        self._exemplar_bands.clear()


        self._auto_exemplar_seed_m = None

        self._restore_maptool_after_exemplar()
        self._refresh_exemplar_chips()

    def _set_exemplar_bands_visible(self, visible: bool) -> None:



        for band in self._exemplar_bands.values():
            if band is None:
                continue
            try:
                band.setVisible(visible)
            except (RuntimeError, AttributeError):
                pass










    def _connect_exemplar_undo_shortcut(self) -> None:

        shortcut = getattr(self.dock_widget, "auto_correct_undo_shortcut", None)
        if shortcut is None or getattr(self, "_exemplar_undo_connected", False):
            return
        try:
            shortcut.activated.connect(self._on_exemplar_undo_shortcut)
        except (RuntimeError, AttributeError, TypeError):
            return
        self._exemplar_undo_connected = True

    def _disconnect_exemplar_undo_shortcut(self) -> None:

        if not getattr(self, "_exemplar_undo_connected", False):
            return
        self._exemplar_undo_connected = False
        shortcut = getattr(self.dock_widget, "auto_correct_undo_shortcut", None)
        if shortcut is None:
            return
        try:
            shortcut.activated.disconnect(self._on_exemplar_undo_shortcut)
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _on_exemplar_undo_shortcut(self) -> bool:







        tool = self._exemplar_maptool
        if tool is None:
            return False
        try:
            if self.iface.mapCanvas().mapTool() is not tool:
                return False
            if tool.undo_point():
                return True
        except (RuntimeError, AttributeError):
            return False
        self._restore_maptool_after_exemplar()
        return True

    def _restore_maptool_after_exemplar(self) -> None:

        tool = self._exemplar_maptool



        self._auto_exemplar_arming = False
        self._disconnect_exemplar_undo_shortcut()


        if self.dock_widget:
            try:
                self.dock_widget.set_auto_shortcuts_enabled(True)
            except (RuntimeError, AttributeError):
                pass
            try:
                self.dock_widget.set_auto_exemplar_armed(None)
            except (RuntimeError, AttributeError):
                pass
        if tool is None:
            return
        self._exemplar_maptool = None


        try:
            tool.suppress_deactivate_signal = True
        except (RuntimeError, AttributeError):
            pass
        try:
            canvas = self.iface.mapCanvas()
            prev = self._maptool_before_exemplar
            if prev is not None:
                canvas.setMapTool(prev)
            elif canvas.mapTool() == tool:
                canvas.unsetMapTool(tool)
        except (RuntimeError, AttributeError):
            pass



        try:
            tool.remove_bands_from_canvas()
        except (RuntimeError, AttributeError):
            pass
        try:
            tool.deleteLater()
        except (RuntimeError, AttributeError):
            pass
        self._maptool_before_exemplar = None

    def _on_exemplar_tool_deactivated(self) -> None:




        if getattr(self, "_suspend_exemplar_deactivate", False):
            return
        tool = self._exemplar_maptool
        if tool is None:
            return
        self._exemplar_maptool = None




        try:
            tool.remove_bands_from_canvas()
        except (RuntimeError, AttributeError):
            pass
        try:
            tool.deleteLater()
        except (RuntimeError, AttributeError):
            pass
        self._maptool_before_exemplar = None
        self._auto_exemplar_arming = False
        self._disconnect_exemplar_undo_shortcut()
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_shortcuts_enabled(True)
                self.dock_widget.set_auto_exemplar_armed(None)
            except (RuntimeError, AttributeError):
                pass

    def _restore_maptool_after_zone(self) -> None:








        tool = self._zone_selection_tool
        prev = self._maptool_before_zone
        self._maptool_before_zone = None
        if tool is None:
            return
        try:
            canvas = self.iface.mapCanvas()
            if canvas.mapTool() != tool:
                return
            if prev is not None and prev is not tool:
                canvas.setMapTool(prev)
            else:
                canvas.unsetMapTool(tool)
        except (RuntimeError, AttributeError):
            pass

    def _remove_rubber_band(self, band) -> None:

        if band is None:
            return
        try:
            self.iface.mapCanvas().scene().removeItem(band)
        except (RuntimeError, AttributeError):
            pass

    def _sync_exemplar_store_tier(self) -> None:







        dock = self.dock_widget
        try:
            known_free = (dock is not None
                          and getattr(dock, "_auto_credits", None) is not None
                          and not getattr(dock, "_auto_is_subscriber", False))
        except (RuntimeError, AttributeError):
            known_free = False
        self._auto_exemplar_store.free_tier = bool(known_free)

    def _refresh_exemplar_chips(self) -> None:

        if not self.dock_widget:
            return
        try:
            items = [(e.id, e.label, e.thumbnail)
                     for e in self._auto_exemplar_store.list()]
            self.dock_widget.set_exemplars(items)
        except (RuntimeError, AttributeError):
            pass

    def _compute_exemplar_pixel_boxes(
        self, layer, geo_bbox: tuple, pixel_w: int, pixel_h: int
    ) -> list[dict]:


        payload: list[dict] = []
        ext_minx, ext_miny, ext_maxx, ext_maxy = geo_bbox
        ext_w = ext_maxx - ext_minx
        ext_h = ext_maxy - ext_miny
        if ext_w <= 0 or ext_h <= 0 or pixel_w <= 0 or pixel_h <= 0:
            return payload

        def _clamp(v, lo, hi):
            return max(lo, min(hi, v))

        for ex in self._auto_exemplar_store.list():
            raw = self._exemplar_full_pixel_box(
                ex, layer, geo_bbox, pixel_w, pixel_h)
            if raw is None:
                continue
            box = [
                _clamp(raw[0], 0, pixel_w), _clamp(raw[1], 0, pixel_h),
                _clamp(raw[2], 0, pixel_w), _clamp(raw[3], 0, pixel_h),
            ]
            if box[2] - box[0] >= 1 and box[3] - box[1] >= 1:
                entry = {"box": [round(v, 1) for v in box], "label": int(ex.label)}
                ring = self._exemplar_polygon_px(
                    ex, layer, geo_bbox, pixel_w, pixel_h)
                if ring:
                    entry["polygon_px"] = ring
                payload.append(entry)
        return payload

    def _exemplar_full_pixel_box(
        self, ex, layer, geo_bbox: tuple, pixel_w: int, pixel_h: int
    ) -> list[float] | None:






        ext_minx, ext_miny, ext_maxx, ext_maxy = geo_bbox
        ext_w = ext_maxx - ext_minx
        ext_h = ext_maxy - ext_miny
        if ext_w <= 0 or ext_h <= 0 or pixel_w <= 0 or pixel_h <= 0:
            return None
        try:
            rect_run = self._reproject_zone_to_run_crs(ex.map_rect, layer)
        except (RuntimeError, AttributeError):
            return None
        x0 = (rect_run.xMinimum() - ext_minx) / ext_w * pixel_w
        x1 = (rect_run.xMaximum() - ext_minx) / ext_w * pixel_w

        y0 = (ext_maxy - rect_run.yMaximum()) / ext_h * pixel_h
        y1 = (ext_maxy - rect_run.yMinimum()) / ext_h * pixel_h
        return [x0, y0, x1, y1]

    def _exemplar_polygon_px(self, ex, layer, geo_bbox, pixel_w, pixel_h):



        poly = getattr(ex, "polygon", None)
        if poly is None or poly.isEmpty():
            return []
        ext_minx, ext_miny, ext_maxx, ext_maxy = geo_bbox
        ext_w = ext_maxx - ext_minx
        ext_h = ext_maxy - ext_miny
        if ext_w <= 0 or ext_h <= 0:
            return []
        g = QgsGeometry(poly)
        try:



            canvas_crs = self._zone_source_crs(getattr(ex, "map_rect", None))
            if canvas_crs is None:
                return []






            run_crs = self._run_crs_now(layer)
            if run_crs is None:
                run_crs = layer.crs()
            if canvas_crs.isValid() and run_crs.isValid() and canvas_crs != run_crs:
                g.transform(QgsCoordinateTransform(
                    canvas_crs, run_crs, QgsProject.instance()))
        except Exception:  # noqa: BLE001
            return []
        try:
            rings = g.asPolygon()
            if not rings:
                mp = g.asMultiPolygon()
                rings = mp[0] if mp else []
            ring = rings[0] if rings else []
        except (AttributeError, IndexError):
            return []
        out = []
        for pt in ring:
            px = (pt.x() - ext_minx) / ext_w * pixel_w
            py = (ext_maxy - pt.y()) / ext_h * pixel_h
            out.append([round(px, 1), round(py, 1)])
        return out












    _STAMP_MIN_SIDE = 96
    _STAMP_MAX_SIDE = 512






    _STAMP_ABS_MIN_SIDE = 32


    _STAMP_FALLBACK_GSD_M = 0.15


    _WEBMERC_GSD0_M = 156543.03392

    def _exemplar_stamp_longest_side(self, layer, padded, run_mupp: float = 0.0) -> int:















        from ...core.detection_policy import (
            exemplar_render_abs_min_side_px,
            exemplar_render_side_bounds,
        )
        min_side, max_side = exemplar_render_side_bounds(
            self._STAMP_MIN_SIDE, self._STAMP_MAX_SIDE)
        gsd_m, longest_m, run_matched = self._exemplar_render_gsd(
            layer, padded, run_mupp)
        if gsd_m <= 0 or longest_m <= 0:
            return max_side
        side = int(round(longest_m / gsd_m))



        floor = exemplar_render_abs_min_side_px(
            self._STAMP_ABS_MIN_SIDE) if run_matched else min_side
        return max(floor, min(max_side, side))

    def _exemplar_render_gsd(
        self, layer, padded, run_mupp: float = 0.0
    ) -> tuple[float, float, bool]:





















        if not self._layer_is_online_tiled(layer):
            try:
                rupp_x = float(layer.rasterUnitsPerPixelX())
                rupp_y = float(layer.rasterUnitsPerPixelY())
                if rupp_x > 0 and rupp_y > 0:





                    to_run_x, to_run_y = self._layer_units_to_run_units(layer, padded)
                    src_gsd = max(rupp_x * to_run_x, rupp_y * to_run_y)
                    gsd = max(src_gsd, run_mupp) if run_mupp > 0 else src_gsd
                    longest = max(padded.width(), padded.height())
                    return gsd, longest, gsd > src_gsd
            except (AttributeError, RuntimeError, ValueError):
                pass
        else:
            zmax = self._layer_max_zoom(layer)
            if zmax > 0:
                lat = self._extent_centre_latitude(layer, padded)
                cos_lat = max(math.cos(math.radians(lat)), 1e-6)


                src_gsd = self._WEBMERC_GSD0_M / (2 ** zmax) * cos_lat
                run_m = run_mupp * cos_lat if run_mupp > 0 else 0.0
                gsd = max(src_gsd, run_m)
                longest = max(padded.width(), padded.height()) * cos_lat
                return gsd, longest, gsd > src_gsd


        if run_mupp > 0:
            return run_mupp, max(padded.width(), padded.height()), True
        from ...core.detection_policy import exemplar_render_fallback_gsd_m
        fallback_gsd = exemplar_render_fallback_gsd_m(self._STAMP_FALLBACK_GSD_M)
        return fallback_gsd, max(padded.width(), padded.height()), False

    @staticmethod
    def _layer_is_online_tiled(layer) -> bool:




        name = ""
        try:
            prov = layer.dataProvider()
            if prov is not None:
                name = (prov.name() or "").lower()
        except (AttributeError, RuntimeError):
            name = ""
        if not name:
            try:
                name = (layer.providerType() or "").lower()
            except (AttributeError, RuntimeError):
                name = ""
        if name in ("wms", "wmts", "xyz"):
            return True
        try:
            uri = (layer.dataProvider().dataSourceUri() or "").lower()
        except (AttributeError, RuntimeError):
            uri = ""
        return "type=xyz" in uri or "zmax=" in uri or "tiles=" in uri

    def _layer_max_zoom(self, layer) -> int:



        try:
            uri = layer.dataProvider().dataSourceUri() or ""
        except (AttributeError, RuntimeError):
            return 19
        m = re.search(r"zmax=(\d+)", uri)
        if m:
            try:
                z = int(m.group(1))
                if 0 < z <= 30:
                    return z
            except ValueError:
                pass
        return 19

    def _extent_centre_latitude(self, layer, padded) -> float:




        cx = (padded.xMinimum() + padded.xMaximum()) / 2.0
        cy = (padded.yMinimum() + padded.yMaximum()) / 2.0
        try:
            from qgis.core import QgsCoordinateReferenceSystem, QgsPointXY
            layer_crs = layer.crs()
            wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
            if layer_crs.isValid() and wgs84.isValid() and layer_crs == wgs84:
                return cy
            if layer_crs.isValid() and wgs84.isValid() and layer_crs != wgs84:
                xform = QgsCoordinateTransform(layer_crs, wgs84, QgsProject.instance())
                lat = xform.transform(QgsPointXY(cx, cy)).y()
                if -89.9 <= lat <= 89.9:
                    return lat
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        try:
            r = 6378137.0
            return math.degrees(2.0 * math.atan(math.exp(cy / r)) - math.pi / 2.0)
        except (ValueError, OverflowError):
            return 0.0

    def _exemplar_run_mupp(self, layer) -> float:









        try:
            if self._auto_zone is not None and layer is not None:
                zone_in_layer = self._reproject_zone_to_run_crs(
                    self._auto_zone, layer)
                sized = self._grid_for_detail(
                    layer, zone_in_layer, self._get_auto_detail_level())
                if sized is not None and sized[2] > 0:
                    return float(sized[2])
        except (RuntimeError, AttributeError, ValueError, TypeError):

            pass
        gsd = getattr(self, "_auto_gsd", 0.0)
        return float(gsd) if gsd and gsd > 0 else 0.0

    @staticmethod
    def _stamp_gsd_matches(cached: float, current: float) -> bool:




        if current <= 0:
            return True
        if cached <= 0:
            return False
        return abs(cached - current) / max(cached, current) <= 0.05

    def _render_exemplar_stamp(self, ex, layer, run_mupp: float = 0.0, max_side: int = 0):

















        from ...core.cloud_detection import render_zone_to_image
        built = self._exemplar_padded_extent(ex, layer, run_mupp)
        if built is None:
            return None
        rect, padded = built
        fw, fh = padded.width(), padded.height()
        if fw <= 0 or fh <= 0:
            return None




        target = self._exemplar_stamp_longest_side(layer, padded, run_mupp)



        if max_side > 0:
            target = min(target, int(max_side))
        if fw >= fh:
            pw, ph = target, max(16, int(round(target * fh / fw)))
        else:
            ph, pw = target, max(16, int(round(target * fw / fh)))
        try:





            img, act = render_zone_to_image(
                layer, padded, pw, ph, resample_local=True,
                render_crs=self._run_crs_now(layer))
        except Exception:  # noqa: BLE001
            img, act = None, None
        if img is None or img.isNull() or act is None:
            return None




        aw = act.xMaximum() - act.xMinimum()
        ah = act.yMaximum() - act.yMinimum()
        iw, ih = img.width(), img.height()
        obj_box = None
        if aw > 0 and ah > 0:
            ox0 = (rect.xMinimum() - act.xMinimum()) / aw * iw
            ox1 = (rect.xMaximum() - act.xMinimum()) / aw * iw

            oy0 = (act.yMaximum() - rect.yMaximum()) / ah * ih
            oy1 = (act.yMaximum() - rect.yMinimum()) / ah * ih
            obj_box = [
                max(0.0, ox0), max(0.0, oy0),
                min(float(iw), ox1), min(float(ih), oy1),
            ]
        return img, obj_box

    def _exemplar_padded_extent(self, ex, layer, run_mupp: float = 0.0):





        from ...core.detection_policy import exemplar_context_pad, exemplar_context_pad_px_cap
        try:
            rect = self._reproject_zone_to_run_crs(ex.map_rect, layer)
        except (RuntimeError, AttributeError):
            return None
        ew, eh = rect.width(), rect.height()
        if ew <= 0 or eh <= 0:
            return None
        context_pad = exemplar_context_pad()
        pad_x = ew * context_pad
        pad_y = eh * context_pad
        if run_mupp > 0:
            pad_cap = exemplar_context_pad_px_cap() * run_mupp
            pad_x = min(pad_x, pad_cap)
            pad_y = min(pad_y, pad_cap)
        padded = QgsRectangle(
            rect.xMinimum() - pad_x, rect.yMinimum() - pad_y,
            rect.xMaximum() + pad_x, rect.yMaximum() + pad_y)
        return rect, padded

    def _exemplar_runscale_side(self, ex, layer, run_mupp: float) -> int:



        built = self._exemplar_padded_extent(ex, layer, run_mupp)
        if built is None:
            return 0
        _rect, padded = built
        return self._exemplar_stamp_longest_side(layer, padded, run_mupp)

    def _prebuild_exemplar_stamp(self, eid: str) -> None:






        ex = self._auto_exemplar_store.get(eid)
        if ex is None:
            return
        layer = self._get_active_raster_layer()
        if layer is None:
            return



        from ...core.cloud_detection import stamp_size_cap
        run_mupp = self._exemplar_run_mupp(layer)


        cap = stamp_size_cap(self._auto_exemplar_store.count())
        built = self._render_exemplar_stamp(ex, layer, run_mupp, max_side=cap)
        if built is None:
            return
        ex.stamp_img, ex.stamp_obj_box = built
        ex.stamp_layer_id = layer.id()
        ex.stamp_gsd = run_mupp
        ex.stamp_side = max(ex.stamp_img.width(), ex.stamp_img.height())





        if ex.stamp_img is not None and not ex.stamp_img.isNull():
            ex.thumbnail = ex.stamp_img
            self._refresh_exemplar_chips()

    def _build_exemplar_stamps(self, layer, geo_bbox=None, pixel_w=0, pixel_h=0,
                               has_prompt: bool = False):




















        from ...core.cloud_detection import should_paste_stamp, stamp_size_cap
        from ...core.detection_policy import exemplar_min_paste_scale
        stamps = []
        layer_id = layer.id() if layer is not None else None



        run_mupp = self._exemplar_run_mupp(layer)



        cap = stamp_size_cap(self._auto_exemplar_store.count())
        min_scale = exemplar_min_paste_scale()
        skipped_paste = 0
        for ex in self._auto_exemplar_store.list():
            full_box = (
                self._exemplar_full_pixel_box(ex, layer, geo_bbox, pixel_w, pixel_h)
                if geo_bbox is not None else None)
            if getattr(ex, "region", False):



                if full_box is not None:
                    stamps.append((None, int(ex.label), None, full_box, True))
                continue
            true_side = self._exemplar_runscale_side(ex, layer, run_mupp)
            if not should_paste_stamp(true_side, cap, has_prompt, min_scale):
                if full_box is not None:
                    stamps.append((None, int(ex.label), None, full_box))
                skipped_paste += 1
                continue
            stamp_valid = ex.stamp_img is not None and ex.stamp_layer_id == layer_id
            stamp_valid = stamp_valid and self._stamp_gsd_matches(ex.stamp_gsd, run_mupp)
            stamp_valid = stamp_valid and getattr(ex, "stamp_side", 0) <= cap
            if stamp_valid:
                stamps.append(
                    (ex.stamp_img, int(ex.label), ex.stamp_obj_box, full_box))
                continue
            built = self._render_exemplar_stamp(ex, layer, run_mupp, max_side=cap)
            if built is None:
                continue
            ex.stamp_img, ex.stamp_obj_box = built
            ex.stamp_layer_id = layer_id
            ex.stamp_gsd = run_mupp
            ex.stamp_side = max(ex.stamp_img.width(), ex.stamp_img.height())
            stamps.append(
                (ex.stamp_img, int(ex.label), ex.stamp_obj_box, full_box))
        if skipped_paste:
            from qgis.core import Qgis, QgsMessageLog
            QgsMessageLog.logMessage(
                f"Auto detection: {skipped_paste} example(s) larger than the paste band at "
                "run scale; sent in-situ only",
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )
        return stamps
