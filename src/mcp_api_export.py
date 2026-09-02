"""Writing a polygon to disk, for the public API.

Part of `SegmentationMCPAPI` (see `mcp_api.py`), split out so one concern sits
in one file. Methods here are plain mixin members: state lives on the assembled
instance (`self._plugin`).
"""
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

# QgsField type args (QGIS 4 rejects raw int, #25/#36): resolved once in
# qt_compat (QVariant on QGIS 3, QMetaType on QGIS 4).
_FIELD_TYPE_STRING = field_type_string()

# What the class column says for a polygon a caller drew or traced by hand.
# The same word the Semi-Auto save writes, so the two are one column.
_HAND_DRAWN_CLASS = "manual"


class SegmentationExportMixin:
    """Turn a WKT polygon into a saved GeoPackage layer in the project."""

    def export_polygon(
        self,
        geometry_wkt: str,
        crs: str,
        raster_name: str,
        output_dir: str | None = None,
    ) -> dict:
        """Export a polygon to a GeoPackage layer in the project.

        Parameters
        ----------
        geometry_wkt : str
            The outline to save, as well-known text. POLYGON or MULTIPOLYGON.
        crs : str
            The CRS that WKT is expressed in, as an authid such as "EPSG:2154".
        raster_name : str
            Name of the imagery the polygon was read from. It names the layer
            group the result is filed under, and it groups this polygon with
            every other one taken from the same imagery. It is NOT a name for
            the layer being created, and nothing is read from that raster here.
            A name no raster in the project carries still works, and the answer
            then says so under ``raster_name_note``.
        output_dir : str | None
            Folder the GeoPackage goes in. It has to exist already. None keeps
            the project folder, and a project that was never saved has none, so
            the call refuses rather than dropping the file where nobody looks
            for it. Additive, default None.
        """
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
            # A POINT or a LINESTRING passes the empty test and stores nothing,
            # so it has to be refused by type rather than reported as saved.
            if geom.type() != PolygonGeometry:
                return {"_error": "Geometry must be a POLYGON or a MULTIPOLYGON."}

            # The layer this writes into is a plain MultiPolygon, so a Z
            # coordinate cannot be stored. Drop it here and say so in the
            # answer, rather than letting the provider flatten it in silence
            # and hand back a file whose heights vanished without a word.
            dropped_z = bool(QgsWkbTypes.hasZ(geom.wkbType()))
            if dropped_z:
                try:
                    geom.get().dropZValue()
                except (RuntimeError, AttributeError, TypeError):
                    # The provider flattens it anyway. The answer still says
                    # the height did not reach the file.
                    pass

            # A folder that is not there is a mistake, whichever path the
            # export then takes. Only the new-layer path writes a file, but a
            # caller who named a folder meant it, and appending elsewhere while
            # saying nothing hides the typo. A folder nobody named is resolved
            # further down, where the project one still stands in.
            if output_dir:
                _named_dir, dir_err = self._resolve_output_dir(output_dir)
                if dir_err:
                    return dir_err

            # Find existing segmentation layer to append to
            seg_group_name = f"{raster_name} (AI Segmentation)"
            root = QgsProject.instance().layerTreeRoot()
            existing_layer = self._append_target_layer(root, seg_group_name)

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

            # One offset-aware ISO 8601 stamp, the same shape every export
            # path writes. A naive local time cannot be placed on a clock by
            # anyone who did not run the export.
            timestamp = output_timestamp_iso()

            if existing_layer and existing_layer.dataProvider():
                try:
                    g = QgsGeometry(geom)
                    # The WKT arrives in the caller's CRS; the layer we append
                    # to has its own. Reproject or the polygon lands somewhere
                    # else entirely (and its area is measured in the wrong CRS).
                    target_crs = existing_layer.crs()
                    if (crs_obj.isValid() and target_crs.isValid() and crs_obj != target_crs):
                        g.transform(QgsCoordinateTransform(
                            crs_obj, target_crs, QgsProject.instance()))
                    g = repair_polygon(g) or g
                    # Coerce to polygon-only MultiPolygon so a collection can
                    # never reach the layer provider (it would be rejected).
                    g = to_multipolygon(g) or g
                    feature = QgsFeature(existing_layer.fields())
                    feature.setGeometry(g)
                    # Match the layer's schema by field name so appending
                    # works on layers created by any plugin version.
                    feature.setAttributes(attribute_values_for_fields(
                        existing_layer.fields(), g, existing_layer.crs(),
                        raster_name, timestamp,
                        det_id=self._next_det_id(existing_layer),
                        object_class=_HAND_DRAWN_CLASS,
                    ))
                    added = existing_layer.dataProvider().addFeatures([feature])
                    existing_layer.updateExtents()
                    existing_layer.triggerRepaint()
                    if not added:
                        # The provider refused the row. Saying "appended" here
                        # tells the caller its object is on disk when it is not.
                        return {
                            "_error": "Could not append the polygon to layer "
                                      f"'{existing_layer.name()}'.",
                            "appended": False,
                        }
                    # The abstract carries a detection count, and an append
                    # made it a lie. Recount it here rather than leave a
                    # provenance block that contradicts the table.
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

            # Create new layer. Only this path writes a file, so only this path
            # needs a folder to write it in.
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
            # The same five columns every export path writes, in the same
            # order, so a layer made here appends to a layer made by a run and
            # both open the same way. Confidence stays NULL: this call is
            # handed an outline and no model scored it. Run-level provenance
            # goes in the layer metadata, not per row.
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
            # Coerce to polygon-only MultiPolygon so a collection can never
            # reach the layer provider (it would be rejected).
            g = to_multipolygon(g) or g
            feature = QgsFeature(temp_layer.fields())
            feature.setGeometry(g)
            # Both measures come from one ellipsoidal measurer, and both stay
            # empty when it refuses the CRS. A planar number written under a
            # column named area_m2 says square metres and means square degrees.
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
            if not pr.addFeatures([feature]):
                return {"_error": "Could not add the polygon to the new layer."}
            temp_layer.updateExtents()

            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = "GPKG"
            options.fileEncoding = "UTF-8"
            # Saved layers are written in ground metres, like every other export
            # path. Without it, a length read off a file saved over a web
            # basemap is wrong by the latitude factor.
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

            # Open the table by its explicit name (a GPKG table defaults to
            # the file stem): a bare path leaves the sublayer choice to the
            # provider, which some GDAL/QGIS builds resolve differently and
            # then report the freshly written file as invalid.
            table = os.path.splitext(os.path.basename(gpkg_path))[0]
            result_layer = QgsVectorLayer(
                f"{gpkg_path}|layername={table}", layer_name, "ogr")
            if not result_layer.isValid():
                result_layer = QgsVectorLayer(gpkg_path, layer_name, "ogr")
            if not result_layer.isValid():
                return {"_error": "Created GeoPackage but layer is invalid"}

            # The per-imagery committed colour, the same one the dock gives a
            # saved run. Without it every layer this path makes is the plain
            # red outline and two of them are indistinguishable on the canvas.
            result_layer.setRenderer(make_committed_renderer(
                color=committed_color_for_prompt(raster_name or "")))
            # Style + provenance stored with the .gpkg (survives reloads).
            apply_output_conventions(
                result_layer, raster_name,
                created_iso=timestamp,
                source_crs_authid=str(crs_obj.authid() or ""),
            )

            group = root.findGroup(seg_group_name)
            if group is None:
                group = root.insertGroup(0, seg_group_name)

            QgsProject.instance().addMapLayer(result_layer, False)
            # Newest first, the same order the dock files a saved run in. With
            # addLayer the newest result landed at the bottom of the group,
            # under everything the caller made before it.
            group.insertLayer(0, result_layer)
            # Same rule as the dock: results paint above the imagery they were
            # made from. A headless caller has no eyes on the canvas, so a
            # group left under an opaque basemap goes unnoticed for longer.
            # The node is destroyed by the move, so never touch `group` after.
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
    def _append_target_layer(root, seg_group_name: str):
        """The one layer in a group that an append should go into, or None.

        Walking the project's layer registry gave whichever layer happened to
        register first, which changes between sessions and between project
        loads: two calls in a row could file two polygons into two different
        layers, in two different CRSs. The group's own children are ordered,
        and the highest "mask_N" in it is the newest, so the choice is the
        same every time.
        """
        group = root.findGroup(seg_group_name)
        if group is None:
            return None
        candidates = []
        try:
            for node in group.findLayers():
                layer = node.layer()
                if (isinstance(layer, QgsVectorLayer)
                        and layer.name().startswith("mask_")):
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
        """The det_id one more row on this layer should carry.

        One past the highest already there, so two appends never write the
        same id. Falls back to the row count when the column cannot be read,
        which is a layer written before the column existed.
        """
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
        """One sentence when the named imagery is not in the project, else "".

        None means the argument itself is unusable. A name that matches no
        raster is allowed: a caller may be filing a polygon read from imagery
        that has since been removed. It is worth saying, because the usual
        cause is a caller passing the name it wanted the OUTPUT layer to carry,
        and that files the result under a group nobody recognises.
        """
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
        """Folder the GeoPackage goes in, as (path, error_dict_or_None).

        A project that was never saved has no folder of its own. Writing to the
        user's home folder instead puts the file where nobody looks for it, so
        the caller has to name one.

        The folder has to exist already. Creating it would let a caller build a
        tree anywhere on the disk from one string, and a typo would write the
        result somewhere nobody thinks to look instead of refusing.
        """
        if output_dir:
            given = str(output_dir)
            path = os.path.abspath(os.path.expanduser(given))
            if not os.path.isdir(path):
                return None, {"_error": (
                    f"Output directory '{given}' does not exist, or is not a "
                    "folder. Create it first, or pass one that is already "
                    f"there. (Resolved to: {path})"
                )}
            return path, None

        project_dir = QgsProject.instance().absolutePath()
        if project_dir:
            return project_dir, None
        return None, {"_error": (
            "This project has never been saved, so there is no folder to write "
            "to. Save the project, or pass output_dir."
        )}

    def _output_crs_transform(self, source_crs, extent):
        """Transform onto the CRS a saved layer is written in, or None.

        None when the source CRS already measures in ground metres, which is
        the common case. See layer_conventions.pick_output_crs.
        """
        try:
            from .core.layer_conventions import pick_output_crs

            target = pick_output_crs(source_crs, extent)
            if target is None or not target.isValid() or target == source_crs:
                return None
            return QgsCoordinateTransform(source_crs, target, QgsProject.instance())
        except (RuntimeError, AttributeError, TypeError):
            return None
