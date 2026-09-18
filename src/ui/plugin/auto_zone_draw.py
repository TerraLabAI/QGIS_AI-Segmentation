







from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
)

from ...core.i18n import tr
from ...core.qt_compat import geometry_op_succeeded, safe_disconnect
from ..shortcut_filter import ShortcutFilter
from .shared import max_tiles_per_run_cap


class AutoZoneDrawMixin:







    def _get_active_raster_layer(self) -> QgsRasterLayer | None:



        if not self.dock_widget:
            return None
        try:
            from ..ai_segmentation_dockwidget import Mode
            if self.dock_widget._mode == Mode.AUTOMATIC:
                layer = self.dock_widget.auto_layer_combo.currentLayer()
            else:
                layer = self.dock_widget.layer_combo.currentLayer()
        except (RuntimeError, AttributeError):
            return None
        if layer is None:
            return None



        if not isinstance(layer, QgsRasterLayer):
            return None
        return layer

    def _setup_auto_mode(self) -> None:




        if self._zone_selection_tool is not None:
            return

        from ...core.tile_manager import OVERLAP_FRACTION, TILE_SIZE, TileManager
        from ..polygon_zone_maptool import PolygonZoneMapTool

        canvas = self.iface.mapCanvas()
        self._tile_manager = TileManager(
            tile_size=TILE_SIZE,
            overlap_fraction=OVERLAP_FRACTION,
            max_tiles=max_tiles_per_run_cap(),
        )


        self._zone_selection_tool = PolygonZoneMapTool(canvas)
        self._zone_selection_tool.zone_selected.connect(self._on_zone_polygon_drawn)
        self._zone_selection_tool.zone_cleared.connect(self._on_zone_cleared)
        self._zone_selection_tool.vertices_changed.connect(self._on_zone_vertices_changed)

        self._zone_selection_tool.back_requested.connect(self._on_auto_exit_clicked)

        self._zone_selection_tool.tool_deactivated.connect(
            self._on_zone_tool_deactivated)


        QgsProject.instance().cleared.connect(self._on_project_cleared_auto)
        QgsProject.instance().readProject.connect(self._on_project_cleared_auto)

    def _teardown_auto_mode(self) -> None:




        if self._zone_selection_tool is not None:



            self._restore_maptool_after_zone()
            try:
                self.iface.mapCanvas().unsetMapTool(self._zone_selection_tool)
            except (RuntimeError, AttributeError):
                pass



            tool = self._zone_selection_tool
            safe_disconnect(tool, "zone_selected", self._on_zone_polygon_drawn)
            safe_disconnect(tool, "zone_cleared", self._on_zone_cleared)
            safe_disconnect(tool, "vertices_changed", self._on_zone_vertices_changed)
            safe_disconnect(tool, "back_requested", self._on_auto_exit_clicked)
            safe_disconnect(tool, "tool_deactivated", self._on_zone_tool_deactivated)


            try:
                tool.remove_bands_from_canvas()
            except (RuntimeError, AttributeError):
                pass
            self._zone_selection_tool = None

        self._remove_zone_shortcut_filter()
        self._clear_auto_canvas()




        self._stop_auto_stall_watchdog()
        self._auto_stall_timer = None




        proj = QgsProject.instance()
        safe_disconnect(proj, "cleared", self._on_project_cleared_auto)
        safe_disconnect(proj, "readProject", self._on_project_cleared_auto)





        pending_run_ctx = self._auto_run_ctx
        self._store_auto_zone(None)
        self._auto_zone_polygon = None
        self._tile_manager = None


        self._auto_last_run_sig = None
        self._rerun_guard_emitted_sig = None


        self._stop_auto_detection()



        self._auto_run_ctx = pending_run_ctx
        self._autosave_pending_auto_review(exit_path="unload")


        self._auto_run_ctx = None
        self._auto_review = None



        self._set_review_busy(False)
        self._clear_free_zone_review_outline()
        self._remove_auto_selection_layer()

    def _activate_zone_drawing(self) -> None:




        if self._auto_worker is not None or self._auto_review is not None:
            return
        if self._zone_selection_tool is None:
            self._setup_auto_mode()

        self._clear_auto_canvas()
        try:
            canvas = self.iface.mapCanvas()



            current = canvas.mapTool()
            if current is not self._zone_selection_tool:
                self._maptool_before_zone = current
            canvas.setMapTool(self._zone_selection_tool)


            canvas.setFocus()
        except (RuntimeError, AttributeError):
            pass



        if self._shortcut_filter is None:
            self._shortcut_filter = ShortcutFilter(self)
        self.iface.mainWindow().installEventFilter(self._shortcut_filter)
        canvas = self.iface.mapCanvas()
        canvas.viewport().installEventFilter(self._shortcut_filter)
        canvas.installEventFilter(self._shortcut_filter)
        if self.dock_widget:
            self.dock_widget.set_auto_zone_state("drawing")
            self.dock_widget.set_zone_draw_progress(0)

    def _remove_zone_shortcut_filter(self) -> None:








        if self._shortcut_filter is None:
            return
        try:
            self.iface.mainWindow().removeEventFilter(self._shortcut_filter)
        except (RuntimeError, AttributeError):
            pass
        try:
            self.iface.mapCanvas().viewport().removeEventFilter(
                self._shortcut_filter)
        except (RuntimeError, AttributeError):
            pass
        try:
            self.iface.mapCanvas().removeEventFilter(self._shortcut_filter)
        except (RuntimeError, AttributeError):
            pass

    def _on_zone_tool_deactivated(self) -> None:







        if (self._auto_zone is not None or self._auto_worker is not None
                or self._auto_review is not None):
            return
        if not self.dock_widget:
            return
        try:
            self.dock_widget.set_zone_draw_progress(0)
        except (RuntimeError, AttributeError):
            pass

    def _on_zone_polygon_drawn(self, geom: QgsGeometry) -> None:











        geom = self._zone_single_area(geom)
        if geom is None:
            self._reject_zone_self_crossing()
            return




        overlap = self._zone_layer_overlap_verdict(geom)
        if overlap == "outside":
            self._reject_zone_outside_layer()
            return




        drawn_geom = geom
        free_fit = self._fit_zone_to_free_budget(geom)
        if free_fit is not None:
            if free_fit.geom is None:
                self._reject_zone_over_free_cap(free_fit.requested_km2)
                return
            geom = QgsGeometry(free_fit.geom)
        self._auto_zone_polygon = QgsGeometry(geom)
        rect = QgsRectangle(geom.boundingBox())



        self._store_auto_zone(rect)
        if free_fit is not None:
            self._record_free_zone_fit(free_fit)


        self._maybe_warmup_auto()
        self._show_zone_polygon_band(geom)


        self._apply_default_detail(rect)
        self._update_credit_estimate()



        self._restore_maptool_after_zone()
        if self.dock_widget:
            self.dock_widget.set_auto_zone_state("zone_set")
        try:
            from ...core import telemetry_run_events

            try:
                vtx = int(drawn_geom.constGet().vertexCount())
            except Exception:
                vtx = 0
            telemetry_run_events.track_zone_drawn(
                vertices=vtx,
                area_km2=self._zone_geodesic_area_km2(drawn_geom),
            )
        except Exception:
            pass  # nosec B110
        if overlap == "partial":
            layer = self._get_active_raster_layer()
            if layer is not None:
                self.iface.messageBar().pushInfo(
                    "AI Segmentation",
                    tr(
                        'Part of your zone is outside "{layer}" - only the '
                        "overlapping area will return objects."
                    ).format(layer=layer.name()),
                )



    _ZONE_MAIN_PART_MIN_SHARE = 0.98

    def _zone_single_area(self, geom: QgsGeometry) -> QgsGeometry | None:







        try:
            if geom is None or geom.isEmpty():
                return None
            if geom.isGeosValid() and not geom.isMultipart():
                return geom
            from ...core.layer_conventions import repair_polygon
            repaired = repair_polygon(QgsGeometry(geom))
            if repaired is None or repaired.isEmpty():
                return None
            parts = (repaired.asGeometryCollection()
                     if repaired.isMultipart() else [repaired])
            areas = [p.area() for p in parts]
            total = sum(areas)
            if total <= 0:
                return None
            main = max(range(len(parts)), key=lambda i: areas[i])
            if areas[main] / total < self._ZONE_MAIN_PART_MIN_SHARE:
                return None
            return QgsGeometry(parts[main])
        except Exception:  # nosec B110
            return None

    def _reject_zone_self_crossing(self) -> None:



        self._on_zone_cleared()
        msg = tr("Your zone crosses itself. Draw it again without crossing lines.")
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_zone_rejected(None)
                self.dock_widget.set_auto_zone_refusal(msg)
                return
            except (RuntimeError, AttributeError):
                pass
        self.iface.messageBar().pushWarning("AI Segmentation", msg)


    _ZONE_OUTSIDE_INFO_FRACTION = 0.5

    def _zone_layer_overlap_verdict(self, geom: QgsGeometry) -> str:












        layer = self._get_active_raster_layer()
        if layer is None:
            return "ok"
        try:
            if self._needs_canvas_render(layer):
                return "ok"
            extent = layer.extent()
            if extent.isEmpty() or extent.width() <= 0 or extent.height() <= 0:
                return "ok"
            zone = QgsGeometry(geom)
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            layer_crs = layer.crs()
            if canvas_crs != layer_crs:
                if not canvas_crs.isValid() or not layer_crs.isValid():
                    return "ok"
                xform = QgsCoordinateTransform(
                    canvas_crs, layer_crs, QgsProject.instance())



                if not geometry_op_succeeded(zone.transform(xform)):
                    return "ok"
                if zone.isEmpty():
                    return "ok"
            extent_geom = QgsGeometry.fromRect(extent)
            if not zone.intersects(extent_geom):
                return "outside"
            zone_area = zone.area()
            if zone_area <= 0:
                return "ok"
            inside_fraction = zone.intersection(extent_geom).area() / zone_area
            if inside_fraction < self._ZONE_OUTSIDE_INFO_FRACTION:
                return "partial"
        except Exception:  # nosec B110
            return "ok"
        return "ok"

    def _reject_zone_outside_layer(self) -> None:






        layer = self._get_active_raster_layer()
        name = layer.name() if layer is not None else ""


        self._on_zone_cleared()
        msg = tr(
            'Your zone is outside "{layer}". Pick the right layer '
            "or draw inside it."
        ).format(layer=name)


        if self.dock_widget:
            try:
                self.dock_widget.set_auto_zone_rejected(None)
                self.dock_widget.set_auto_zone_refusal(msg)
                return
            except (RuntimeError, AttributeError):
                pass
        self.iface.messageBar().pushWarning("AI Segmentation", msg)

    def _zone_geodesic_area_km2(self, geom, crs=None) -> float:












        try:



            from ...core.layer_conventions import make_area_measurer
            if crs is None:
                crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            if isinstance(geom, QgsRectangle):
                geom = QgsGeometry.fromRect(geom)
            da = make_area_measurer(crs)
            return max(0.0, da.measureArea(geom) / 1_000_000.0)
        except (TypeError, ValueError, RuntimeError, AttributeError):
            return 0.0

    def _zone_billable_shape(self, geom, crs=None):



        if isinstance(geom, QgsRectangle):
            geom = QgsGeometry.fromRect(geom)
        if crs is None:
            try:
                crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            except (RuntimeError, AttributeError):
                return geom
        if crs is None or not crs.isValid():
            return geom
        return self._zone_clipped_to_data(geom, crs)

    def _reject_zone_over_free_cap(self, area_km2: float) -> None:




        self._on_zone_cleared()
        if self.dock_widget:
            try:

                self.dock_widget.set_auto_zone_refusal(None)
            except (RuntimeError, AttributeError):
                pass
            try:
                self.dock_widget.set_auto_zone_rejected(area_km2)
            except (RuntimeError, AttributeError):
                pass
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_auto_zone_too_large(area_km2=area_km2)
        except Exception:
            pass  # nosec B110
        try:
            self._recheck_plan_after_free_wall()
        except (RuntimeError, AttributeError):
            pass

    def _on_zone_vertices_changed(self, count: int) -> None:


        if self.dock_widget:
            try:
                self.dock_widget.set_zone_draw_progress(count)
            except (RuntimeError, AttributeError):
                pass

    def _on_zone_cleared(self) -> None:





        self._store_auto_zone(None)
        self._auto_zone_polygon = None


        self._clear_auto_canvas()
        self._update_credit_estimate()
        still_drawing = False
        try:
            still_drawing = self._zone_selection_tool is not None
            still_drawing = still_drawing and self.iface.mapCanvas().mapTool() == self._zone_selection_tool
        except (RuntimeError, AttributeError):
            pass
        if self.dock_widget:
            self.dock_widget.set_auto_zone_state(
                "drawing" if still_drawing else "idle")

    def _on_project_cleared_auto(self) -> None:








        try:
            if getattr(self, "_refine_handoff_active", False) and self.saved_polygons:
                self._collect_manual_refine_into_review()
        except Exception as exc:  # noqa: BLE001

            try:
                from ...core.telemetry_errors import track_plugin_error
                track_plugin_error(stage="segment",
                                   error_code="review_fold_edits_failed",
                                   message=type(exc).__name__)
            except Exception:  # noqa: BLE001  # nosec B110
                pass





        self._autosave_pending_auto_review()
        self._auto_review = None


        self._set_review_busy(False)
        self._clear_free_zone_review_outline()
        self._stop_auto_detection()
        self._refine_handoff_active = False





        try:
            self._exit_ai_add_mode()
        except Exception:  # noqa: BLE001
            self._refine_add_mode_active = False
        self._ai_add_install_pending = False
        self._drop_cloud_correct_predictor()
        if self.dock_widget and getattr(self.dock_widget, "_refine_handoff", False):
            try:
                self.dock_widget.leave_ai_reshape_state()
            except (RuntimeError, AttributeError):
                pass
        try:
            self._teardown_manual_session()
        except Exception:
            pass  # nosec B110
        self._store_auto_zone(None)
        self._auto_zone_polygon = None
        self._clear_auto_canvas()
        if self.dock_widget:
            try:
                self.dock_widget.reset_auto_to_start()
            except (RuntimeError, AttributeError):
                pass
        QgsMessageLog.logMessage(
            "Project closed: automatic run stopped.",
            "AI Segmentation", level=Qgis.MessageLevel.Info,
        )
