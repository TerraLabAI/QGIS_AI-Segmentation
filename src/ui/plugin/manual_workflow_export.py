






from __future__ import annotations

import os

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsMessageLog,
    QgsVectorLayer,
)

from ...core.i18n import tr
from ..error_report_dialog import show_error_report
from .shared import _FIELD_TYPE_DOUBLE, _FIELD_TYPE_STRING, _add_features_fast, pixel_grid_crs


class ManualWorkflowExportMixin:





    def _on_export_layer(self) -> bool:




        if self._refine_handoff_active:
            self._on_reshape_done()
            return False
        if self._exporting_in_progress:
            return False
        self._exporting_in_progress = True
        try:
            return bool(self._on_export_layer_impl())
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
                tr("The export did not finish. Your polygons are still on the "
                   "map, so you can try again."),
                error_code="export_failed",
            )
            return False
        finally:
            self._exporting_in_progress = False

    def _on_export_layer_impl(self) -> bool:

        import time as _time
        _t_start = _time.perf_counter()
        self._ensure_polygon_rubberband_sync()

        has_active = self.current_mask is not None and self.current_transform_info is not None
        should_skip_export = not self.saved_polygons and not has_active
        should_skip_export = should_skip_export and not self._frozen_sessions
        should_skip_export = should_skip_export and self._unfrozen_display_polygon is None
        if should_skip_export:
            from ...core.server_dials import dial_in_range
            duration = dial_in_range(
                "tuning.manual.export_nothing_notice_s", 5, 3, 15)
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Nothing to export yet. Click an object and save it first."),
                level=Qgis.MessageLevel.Info, duration=duration)
            return False

        polygons_to_export = list(self.saved_polygons)



        current_geoms = [s.polygon for s in self._frozen_sessions]
        if not has_active and self._unfrozen_display_polygon is not None:
            current_geoms.append(self._unfrozen_display_polygon)

        if has_active:








            active_combined = self._refined_active_mask_geometry()
            if active_combined is not None and not active_combined.isEmpty():
                current_geoms.append(active_combined)
        _t_shape = _time.perf_counter()





        live_billing_id = None
        live_billing_geom = None
        live_export_index = None
        if current_geoms:
            combined = QgsGeometry.unaryUnion(current_geoms)
            if combined and not combined.isEmpty():
                origin = self._active_refine_origin_entry or {}
                origin_id = origin.get("det_id")
                live_billing_id = (int(origin_id) if origin_id is not None
                                   else self._next_handoff_det_id())
                if self._manual_save_refused_for_credits(live_billing_id):







                    return False


                live_billing_geom = combined



                live_export_index = len(polygons_to_export)
                polygons_to_export.append({
                    "det_id": live_billing_id,
                    "geometry_wkt": combined.asWkt(),
                    "geom_obj": combined,
                    "score": origin.get("score"),
                    "transform_info": self.current_transform_info.copy() if self.current_transform_info else None,
                })

        from ...core import output_store



        layer_name = output_store.friendly_layer_name("")





        if self._is_non_georeferenced_mode:
            crs = pixel_grid_crs()
            QgsMessageLog.logMessage(
                "Non-georeferenced mode: writing pixel coordinates on a local grid",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )
        else:

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
                        layer_crs = self._current_layer.crs()
                        crs_str = layer_crs.authid() or layer_crs.toWkt()
                except RuntimeError:
                    pass
            crs = None
            if isinstance(crs_str, str) and crs_str.strip():
                crs = QgsCoordinateReferenceSystem(crs_str)
            if crs is None or not crs.isValid():




                try:
                    if self._is_layer_valid() and self._current_layer.crs().isValid():
                        crs = self._current_layer.crs()
                except RuntimeError:
                    crs = None
            if crs is None or not crs.isValid():
                crs = QgsCoordinateReferenceSystem("EPSG:4326")
                QgsMessageLog.logMessage(
                    "CRS could not be determined, falling back to EPSG:4326",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)


        temp_layer = QgsVectorLayer("MultiPolygon", layer_name, "memory")
        if not temp_layer.isValid():
            show_error_report(
                self.iface.mainWindow(),
                tr("Layer Creation Failed"),
                tr("Could not create the output layer."),
                error_code="layer_creation_failed",
            )
            return False

        temp_layer.setCrs(crs)

        from ...core.layer_conventions import (
            apply_output_conventions,
            make_area_measurer,
            make_committed_renderer,
            measure_field,
            repair_polygon,
            round_measure,
            to_multipolygon,
        )






        pr = temp_layer.dataProvider()
        pr.addAttributes([
            QgsField("det_id", _FIELD_TYPE_STRING),
            QgsField("class", _FIELD_TYPE_STRING),
            QgsField("confidence", _FIELD_TYPE_DOUBLE),
            measure_field("area_m2"),
            measure_field("perimeter_m"),
        ])
        temp_layer.updateFields()

        raster_name = ""
        try:
            if self._is_layer_valid() and self._current_layer:
                raster_name = self._current_layer.name()
        except RuntimeError:
            pass






        measurer = None if self._is_non_georeferenced_mode else make_area_measurer(crs)
        features_to_add = []
        live_reached_the_layer = False
        for i, polygon_data in enumerate(polygons_to_export):
            feature = QgsFeature(temp_layer.fields())



            geom = polygon_data.get("geom_obj")
            if geom is not None:
                geom = QgsGeometry(geom)
            else:
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


                geom = to_multipolygon(repair_polygon(geom) or geom)
                if geom is None or geom.isEmpty():
                    continue
                feature.setGeometry(geom)

                score = polygon_data.get("score")
                if measurer is None:


                    area, perimeter = None, None
                else:
                    area = measurer.measureArea(geom)
                    perimeter = measurer.measurePerimeter(geom)
                det_id = polygon_data.get("det_id")
                feature.setAttributes([
                    str(det_id) if det_id is not None else None,


                    "manual",
                    round(float(score), 3) if score is not None else None,
                    round_measure(area),
                    round_measure(perimeter),
                ])
                features_to_add.append(feature)
                if i == live_export_index:
                    live_reached_the_layer = True
            else:
                QgsMessageLog.logMessage(
                    f"Polygon {i + 1} could not be read back from its saved "
                    "outline and was left out",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Warning
                )

        if live_billing_id is not None and not live_reached_the_layer:







            QgsMessageLog.logMessage(
                "Export: the object still on screen produced no writable "
                "geometry and was not charged for",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            live_billing_id = None
            live_billing_geom = None

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
                   "Try adjusting the outline settings or making a new selection."),
                error_code="export_failed",
            )
            return False

        if not _add_features_fast(pr, features_to_add):


            QgsMessageLog.logMessage(
                f"Export aborted: the layer refused {len(features_to_add)} polygon(s)",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
            show_error_report(
                self.iface.mainWindow(),
                tr("Export Failed"),
                tr("The polygons could not be put into the new layer, so "
                   "nothing was saved. They are still on the map, so you can "
                   "try again."),
                error_code="export_failed",
            )
            return False
        temp_layer.updateExtents()

        try:
            source_layer = self._current_layer if self._is_layer_valid() else None
        except RuntimeError:
            source_layer = None


        try:
            source_name = source_layer.name() if source_layer is not None else ""
        except RuntimeError:
            source_name = ""





        result = output_store.write_run_table(
            temp_layer,
            prompt="",
            source_layer=source_layer,
            fallback_stem="segmentation",
        )
        _t_write = _time.perf_counter()

        if result is None:


            temp_layer.setRenderer(make_committed_renderer(
                color=output_store.committed_color_for_prompt("")))
            output_store.add_committed_layer(temp_layer, source_name=source_name)
            show_error_report(
                self.iface.mainWindow(),
                tr("Export Failed"),
                "{}\n\n{}".format(
                    tr("Could not save layer to file:"),
                    tr("Your polygons were added as a temporary layer so "
                       "nothing is lost.")),
                error_code="export_failed",
            )
            return False





        self._stopping_segmentation = True
        try:
            self.iface.mapCanvas().unsetMapTool(self.map_tool)
            self._restore_previous_map_tool()
        finally:


            self._stopping_segmentation = False






        if live_billing_id is not None:
            self._charge_manual_saved_object(
                live_billing_id, geom=live_billing_geom)

        result_layer = result.layer
        gpkg_path = result.gpkg_path
        layer_name = result_layer.name()







        result_layer.setRenderer(make_committed_renderer(
            color=output_store.committed_color_for_prompt("")))


        from datetime import datetime
        try:
            plugin_version = self._read_plugin_version()
        except Exception:  # nosec B110
            plugin_version = ""
        apply_output_conventions(
            result_layer, raster_name,
            created_iso=datetime.now().astimezone().isoformat(timespec="seconds"),
            plugin_version=plugin_version,
        )




        try:
            from .canvas_redraw_handover import hold_map_picture_during_redraw
            hold_map_picture_during_redraw(self.iface.mapCanvas())
        except (RuntimeError, AttributeError):  # nosec B110
            pass



        output_store.add_committed_layer(result_layer, source_name=source_name)

        if result.used_fallback:
            from ...core.server_dials import dial_in_range
            duration = dial_in_range(
                "tuning.manual.export_fallback_notice_s", 8, 4, 15)
            msg = tr(
                "Could not write to {name}. Saved to a separate file instead."
            ).format(name=os.path.basename(
                result.intended_path or output_store.GPKG_FILENAME))
            self.iface.messageBar().pushMessage(
                "AI Segmentation", msg,
                level=Qgis.MessageLevel.Warning, duration=duration)




        self.iface.mapCanvas().refresh()





        _t_end = _time.perf_counter()
        _ms_shape = int((_t_shape - _t_start) * 1000)
        _ms_write = int((_t_write - _t_shape) * 1000)
        _ms_layer = int((_t_end - _t_write) * 1000)
        _ms_total = int((_t_end - _t_start) * 1000)
        _extent = result_layer.extent()
        QgsMessageLog.logMessage(
            f"Export: {len(features_to_add)} polygon(s) to {layer_name} "
            f"[{result_layer.crs().authid()}] in {_ms_total} ms "
            f"(shape {_ms_shape}, write {_ms_write}, layer {_ms_layer}); "
            f"extent {_extent.xMinimum():.1f},{_extent.yMinimum():.1f} to "
            f"{_extent.xMaximum():.1f},{_extent.yMaximum():.1f}; file {gpkg_path}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

        try:
            from ...core import telemetry_session_events
            from ...core.review_defaults import (
                REFINE_POINTS_PCT_DEFAULT,
                REFINE_SIMPLIFY_DEFAULT,
            )


            refine_shape_changed = abs(
                float(self._refine_simplify) - REFINE_SIMPLIFY_DEFAULT) > 1e-6
            refine_shape_changed = refine_shape_changed or (
                int(self._refine_points_pct) != REFINE_POINTS_PCT_DEFAULT)
            refine_shape_changed = refine_shape_changed or self._refine_smooth or self._refine_expand
            refine_fill_or_ortho_changed = (
                not self._refine_fill_holes or self._refine_ortho)
            refine_used = bool(refine_shape_changed or refine_fill_or_ortho_changed)
            telemetry_session_events.track_manual_export_done(
                polygon_count=len(features_to_add),
                refine_used=refine_used,
                destination="new",
            )
            telemetry_session_events.track_first_generation_milestone(mode="manual")
        except Exception:
            pass  # nosec B110

        exported_count = len(features_to_add)
        try:
            exported_layer_id = result_layer.id()
        except (RuntimeError, AttributeError):
            exported_layer_id = ""




        self._end_manual_credit_session()
        self._stop_canvas_crs_watch()
        self._reset_session()
        self.dock_widget.reset_session()




        try:
            self.dock_widget.show_export_success_line(
                "manual", exported_count, layer_name,
                layer_id=exported_layer_id)
        except (RuntimeError, AttributeError):  # nosec B110
            pass
        return True
