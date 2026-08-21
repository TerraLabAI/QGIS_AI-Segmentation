





from __future__ import annotations

import os

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsProject,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsWkbTypes,
)

from .core.qt_compat import PolygonGeometry, field_type_string



_FIELD_TYPE_STRING = field_type_string()



_HAND_DRAWN_CLASS = "manual"


def _plain_folder_text(given: str) -> str:






    text = given.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if text.lower().startswith("file:"):
        from qgis.PyQt.QtCore import QUrl

        local = QUrl(text).toLocalFile()
        if local:
            text = local
    return text


class SegmentationExportMixin:


    def export_polygon(
        self,
        geometry_wkt: str,
        crs: str,
        raster_name: str,
        output_dir: str | None = None,
    ) -> dict:





















        provenance_note = self._raster_name_note(raster_name)
        if provenance_note is None:
            return {"_error": (
                "raster_name must be a non-empty string naming the imagery this "
                "polygon was read from. It files the result under that imagery, "
                "and it is not a name for the layer being created."
            )}
        try:
            crs_obj = QgsCoordinateReferenceSystem(crs)
            if not crs_obj.isValid():
                return {"_error": f"Invalid CRS '{crs}'."}
            geom = QgsGeometry.fromWkt(geometry_wkt)
            if geom is None or geom.isEmpty():
                return {"_error": "Invalid geometry WKT"}


            if geom.type() != PolygonGeometry:
                return {"_error": "Geometry must be a POLYGON or a MULTIPOLYGON."}





            dropped_z = bool(QgsWkbTypes.hasZ(geom.wkbType()))
            if dropped_z:
                try:
                    geom.get().dropZValue()
                except (RuntimeError, AttributeError, TypeError):


                    pass






            named_dir = None
            if output_dir:
                named_dir, dir_err = self._resolve_output_dir(output_dir)
                if dir_err:
                    return dir_err


            seg_group_name = f"{raster_name} (AI Segmentation)"
            root = QgsProject.instance().layerTreeRoot()
            existing_layer = self._append_target_layer(root, seg_group_name, named_dir)

            from .core.layer_conventions import (
                apply_output_conventions,
                attribute_values_for_fields,
                make_area_measurer,
                make_committed_renderer,
                measure_field,
                repair_polygon,
                round_measure,
                to_multipolygon,
            )
            from .core.output_group_order import keep_group_above_imagery
            from .core.output_metadata import (
                output_timestamp_iso,
                refresh_detection_count,
            )
            from .core.output_store import committed_color_for_prompt
            from .core.qt_compat import geometry_op_succeeded
            from .ui.plugin.shared import _add_features_fast




            timestamp = output_timestamp_iso()

            if existing_layer and existing_layer.dataProvider():
                try:
                    g = QgsGeometry(geom)



                    target_crs = existing_layer.crs()
                    if (crs_obj.isValid() and target_crs.isValid() and crs_obj != target_crs):
                        transformed = g.transform(QgsCoordinateTransform(
                            crs_obj, target_crs, QgsProject.instance()))
                        if not geometry_op_succeeded(transformed):
                            return {"_error": "Could not transform the polygon into the output layer CRS."}
                    g = repair_polygon(g) or g


                    g = to_multipolygon(g) or g
                    feature = QgsFeature(existing_layer.fields())
                    feature.setGeometry(g)


                    feature.setAttributes(attribute_values_for_fields(
                        existing_layer.fields(), g, existing_layer.crs(),
                        raster_name, timestamp,
                        det_id=self._next_det_id(existing_layer),
                        object_class=_HAND_DRAWN_CLASS,
                    ))
                    added = _add_features_fast(existing_layer.dataProvider(), [feature])
                    existing_layer.updateExtents()
                    existing_layer.triggerRepaint()
                    if not added:


                        return {
                            "_error": "Could not append the polygon to layer "
                                      f"'{existing_layer.name()}'.",
                            "appended": False,
                        }



                    refresh_detection_count(existing_layer)
                    answer = {
                        "layer_name": existing_layer.name(),
                        "file_path": existing_layer.source().split("|")[0],
                        "appended": True,
                    }
                    if dropped_z:
                        answer["dropped_z"] = True
                    return answer
                except Exception as e:
                    from qgis.core import QgsMessageLog
                    QgsMessageLog.logMessage(
                        f"Failed to append mask to existing layer, creating a new one: {e}",
                        "AI Segmentation", level=Qgis.MessageLevel.Warning
                    )



            out_dir, dir_err = self._resolve_output_dir(output_dir)
            if dir_err:
                return dir_err

            mask_num = 1
            for lyr in QgsProject.instance().mapLayers().values():
                if lyr.name().startswith("mask_"):
                    try:
                        num = int(lyr.name().split("_")[1])
                        mask_num = max(mask_num, num + 1)
                    except (IndexError, ValueError):
                        pass

            layer_name = f"mask_{mask_num}"
            gpkg_path = os.path.join(out_dir, f"{layer_name}.gpkg")
            counter = 1
            while os.path.exists(gpkg_path):
                gpkg_path = os.path.join(out_dir, f"{layer_name}_{counter}.gpkg")
                counter += 1

            temp_layer = QgsVectorLayer("MultiPolygon", layer_name, "memory")
            temp_layer.setCrs(crs_obj)





            pr = temp_layer.dataProvider()
            pr.addAttributes([
                QgsField("det_id", _FIELD_TYPE_STRING),
                QgsField("class", _FIELD_TYPE_STRING),
                measure_field("confidence", decimals=3),
                measure_field("area_m2"),
                measure_field("perimeter_m"),
            ])
            temp_layer.updateFields()

            g = QgsGeometry(geom)
            g = repair_polygon(g) or g


            g = to_multipolygon(g) or g
            feature = QgsFeature(temp_layer.fields())
            feature.setGeometry(g)



            area = perimeter = None
            try:
                measurer = make_area_measurer(crs_obj)
                area = measurer.measureArea(g)
                perimeter = measurer.measurePerimeter(g)
            except (RuntimeError, AttributeError):
                from qgis.core import QgsMessageLog
                QgsMessageLog.logMessage(
                    "Export: the ellipsoidal measure refused this CRS, so "
                    "area_m2 and perimeter_m are written empty",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning,
                )
            feature.setAttributes([
                "1",
                _HAND_DRAWN_CLASS,
                None,
                round_measure(area),
                round_measure(perimeter),
            ])
            if not _add_features_fast(pr, [feature]):
                return {"_error": "Could not add the polygon to the new layer."}
            temp_layer.updateExtents()

            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = "GPKG"
            options.fileEncoding = "UTF-8"



            out_xform = self._output_crs_transform(crs_obj, temp_layer.extent())
            if out_xform is not None:
                options.ct = out_xform
            from .core.layer_conventions import write_vector_layer
            error = write_vector_layer(
                temp_layer, gpkg_path, options,
                QgsProject.instance().transformContext(),
            )
            if error[0] != QgsVectorFileWriter.WriterError.NoError:
                return {"_error": f"Failed to save GeoPackage: {error[1]}"}





            table = os.path.splitext(os.path.basename(gpkg_path))[0]
            result_layer = QgsVectorLayer(
                f"{gpkg_path}|layername={table}", layer_name, "ogr")
            if not result_layer.isValid():
                result_layer = QgsVectorLayer(gpkg_path, layer_name, "ogr")
            if not result_layer.isValid():
                return {"_error": "Created GeoPackage but layer is invalid"}




            result_layer.setRenderer(make_committed_renderer(
                color=committed_color_for_prompt(raster_name or "")))

            apply_output_conventions(
                result_layer, raster_name,
                created_iso=timestamp,
                source_crs_authid=str(crs_obj.authid() or ""),
            )

            group = root.findGroup(seg_group_name)
            if group is None:
                group = root.insertGroup(0, seg_group_name)

            QgsProject.instance().addMapLayer(result_layer, False)



            group.insertLayer(0, result_layer)




            keep_group_above_imagery(group)

            answer = {"layer_name": layer_name, "file_path": gpkg_path}
            if provenance_note:
                answer["raster_name_note"] = provenance_note
            if dropped_z:
                answer["dropped_z"] = True
            return answer

        except Exception as e:
            return {"_error": f"Export failed: {str(e)}"}

    @staticmethod
    def _append_target_layer(root, seg_group_name: str, output_dir: str | None = None):









        group = root.findGroup(seg_group_name)
        if group is None:
            return None
        candidates = []
        try:
            for node in group.findLayers():
                layer = node.layer()
                if (isinstance(layer, QgsVectorLayer)
                        and layer.isValid() and not layer.isEditable()
                        and layer.name().startswith("mask_")):
                    if output_dir is not None:
                        source_dir = os.path.dirname(layer.source().split("|", 1)[0])
                        if os.path.normcase(os.path.realpath(source_dir)) != os.path.normcase(
                                os.path.realpath(output_dir)):
                            continue
                    candidates.append(layer)
        except (RuntimeError, AttributeError):
            return None
        if not candidates:
            return None

        def _mask_number(layer) -> int:
            try:
                return int(layer.name().split("_")[1])
            except (IndexError, ValueError):
                return -1

        candidates.sort(key=lambda lyr: (_mask_number(lyr), lyr.name()))
        return candidates[-1]

    @staticmethod
    def _next_det_id(layer) -> str:






        try:
            index = layer.fields().indexOf("det_id")
            if index < 0:
                return str(int(layer.featureCount()) + 1)
            highest = 0
            for value in layer.uniqueValues(index):
                try:
                    highest = max(highest, int(str(value)))
                except (TypeError, ValueError):
                    continue
            return str(highest + 1)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return "1"

    @staticmethod
    def _raster_name_note(raster_name) -> str | None:








        text = str(raster_name or "").strip()
        if not text:
            return None
        try:
            from qgis.core import QgsRasterLayer
            for layer in QgsProject.instance().mapLayers().values():
                if isinstance(layer, QgsRasterLayer) and layer.name() == text:
                    return ""
        except (RuntimeError, AttributeError):
            return ""
        return (
            f"No raster layer called '{text}' is in the project, so the result "
            "is filed under a group of that name. raster_name names the imagery "
            "the polygon was read from, not the layer to create."
        )

    def _resolve_output_dir(self, output_dir: str | None):










        if output_dir:
            given = str(output_dir)
            path = os.path.abspath(os.path.expanduser(_plain_folder_text(given)))
            if not os.path.isdir(path):
                return None, {"_error": (
                    f"Output directory '{given}' does not exist, or is not a "
                    "folder. Create it first, or pass one that is already "
                    f"there. (Resolved to: {path})"
                )}
            return path, None

        project_dir = QgsProject.instance().absolutePath()
        if project_dir:


            return os.path.normpath(project_dir), None
        return None, {"_error": (
            "This project has never been saved, so there is no folder to write "
            "to. Save the project, or pass output_dir."
        )}

    def _output_crs_transform(self, source_crs, extent):





        try:
            from .core.layer_conventions import pick_output_crs

            target = pick_output_crs(source_crs, extent)
            if target is None or not target.isValid() or target == source_crs:
                return None
            return QgsCoordinateTransform(source_crs, target, QgsProject.instance())
        except (RuntimeError, AttributeError, TypeError):
            return None
