

















from __future__ import annotations

import json
import os
import tempfile
import time

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsVectorFileWriter,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QSettings

from . import output_store
from .layer_conventions import make_area_measurer, measure_field, round_measure
from .qt_compat import WkbMultiPolygon, field_type_string, geometry_op_succeeded

_LOG_TAG = "AI Segmentation"


def _autosave_output_crs(source_crs, merged_ided):







    from qgis.core import QgsCoordinateTransform, QgsRectangle

    from .layer_conventions import pick_output_crs

    try:
        extent = QgsRectangle()
        for _fid, geom, _score in merged_ided:
            if geom is not None and not geom.isEmpty():
                extent.combineExtentWith(geom.boundingBox())
        if extent.isEmpty():
            return source_crs, None
        target = pick_output_crs(source_crs, extent)
        if target is None or not target.isValid() or target == source_crs:
            return source_crs, None
        return target, QgsCoordinateTransform(
            source_crs, target, QgsProject.instance())
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return source_crs, None




_PENDING_KEY = "AISegmentation/pending_run_autosave"


AUTOSAVE_TABLE_MARK = "autosave"






_ARMED_TABLES: dict[str, dict] = {}
_MAX_ARMED_TABLES = 16


def _autosave_fields() -> QgsFields:








    fields = QgsFields()
    fields.append(QgsField("det_id", field_type_string()))
    fields.append(QgsField("class", field_type_string()))
    fields.append(measure_field("confidence", decimals=3))
    fields.append(measure_field("area_m2"))
    fields.append(measure_field("perimeter_m"))
    return fields


def _open_writer(path: str, table: str, fields: QgsFields, crs,
                 transform_context=None):






    from .output_gpkg_rollover import file_size

    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.fileEncoding = "UTF-8"
    options.layerName = table



    options.actionOnExistingFile = (
        QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
        if file_size(path) == 0
        else QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
    )
    if transform_context is None:
        transform_context = QgsProject.instance().transformContext()
    try:
        writer = QgsVectorFileWriter.create(
            path, fields, WkbMultiPolygon, crs,
            transform_context, options)
    except Exception:  # noqa: BLE001
        return None
    if writer is None:
        return None
    if writer.hasError() != QgsVectorFileWriter.WriterError.NoError:
        del writer
        return None
    return writer


def prepare_autosave(merged_ided: list, crs_authid: str, prompt: str,
                     run_id: str, source_layer=None) -> dict | None:









    try:
        if not merged_ided:
            return None
        crs = QgsCoordinateReferenceSystem(crs_authid or "EPSG:4326")





        crs, ground_metre_xform = _autosave_output_crs(crs, merged_ided)
        stem = (prompt or "").strip() or "detection"
        gpkg_path = output_store.project_gpkg_path(source_layer)
        table = output_store.snake_table_name(
            f"{stem} {AUTOSAVE_TABLE_MARK}", gpkg_path)
        context = QgsProject.instance().transformContext()


        rows = []
        for fid, geom, score in merged_ided:
            try:
                if geom is None or geom.isEmpty():
                    continue
                rows.append((str(fid), bytes(geom.asWkb()),
                             None if score is None else float(score)))
            except Exception:  # nosec B112
                continue
        if not rows:
            return None
        return {
            "rows": rows,
            "crs": crs,
            "xform": ground_metre_xform,
            "context": context,
            "gpkg_path": gpkg_path,
            "table": table,
            "stem": stem,
            "object_class": (prompt or "").strip() or None,
            "run_id": run_id or "",
            "fallback_dirs": list(
                output_store.output_directory_candidates(source_layer)),




            "measurer": make_area_measurer(crs, transform_context=context),
        }
    except Exception:  # noqa: BLE001
        return None


def write_prepared_autosave(job: dict | None) -> dict | None:











    try:
        if not job or not job.get("rows"):
            return None
        crs = job["crs"]
        context = job["context"]
        fields = _autosave_fields()
        gpkg_path = job["gpkg_path"]
        table = job["table"]
        writer = _open_writer(gpkg_path, table, fields, crs, context)
        if writer is None:




            for directory in job.get("fallback_dirs") or []:


                try:
                    with tempfile.NamedTemporaryFile(
                            dir=os.path.normpath(directory), prefix=f"{table}_",
                            suffix=".gpkg", delete=False) as reserved:
                        fallback = reserved.name
                except OSError:
                    continue
                writer = _open_writer(fallback, table, fields, crs, context)
                if writer is not None:
                    gpkg_path = fallback
                    break
                try:
                    if os.path.getsize(fallback) == 0:
                        os.unlink(fallback)
                except OSError:  # nosec B110
                    pass
            if writer is None:
                QgsMessageLog.logMessage(
                    "Run autosave: could not open a GeoPackage writer",
                    _LOG_TAG, level=Qgis.MessageLevel.Warning)
                return None

        from .layer_conventions import to_multipolygon
        object_class = job.get("object_class")
        ground_metre_xform = job.get("xform")
        measurer = job["measurer"]
        count = 0
        for fid, wkb, score in job["rows"]:
            try:
                geom = QgsGeometry()
                geom.fromWkb(wkb)
                if geom.isEmpty():
                    continue
                multi = to_multipolygon(geom)
                if multi is None or multi.isEmpty():
                    continue
                if ground_metre_xform is not None:






                    if not geometry_op_succeeded(multi.transform(ground_metre_xform)):
                        return None
                feat = QgsFeature(fields)
                feat.setGeometry(multi)


                feat.setAttributes([
                    fid,
                    object_class,
                    None if score is None else round(float(score), 3),
                    round_measure(measurer.measureArea(multi)),
                    round_measure(measurer.measurePerimeter(multi)),
                ])
                if not writer.addFeature(feat):
                    return None
                count += 1
            except Exception:  # nosec B112
                continue
        had_error = writer.hasError() != QgsVectorFileWriter.WriterError.NoError
        del writer


        if count != len(job["rows"]) or had_error:
            return None
        return {
            "path": gpkg_path,
            "table": table,
            "layer_name": output_store.friendly_layer_name(job["stem"]),
            "prompt": object_class or "",
            "run_id": job.get("run_id") or "",
            "count": count,
            "ts": time.time(),
        }
    except Exception:  # noqa: BLE001
        try:
            QgsMessageLog.logMessage(
                "Run autosave: write failed", _LOG_TAG,
                level=Qgis.MessageLevel.Warning)
        except Exception:  # nosec B110
            pass
        return None


def write_autosave(merged_ided: list, crs_authid: str, prompt: str,
                   run_id: str, source_layer=None) -> dict | None:






    return write_prepared_autosave(
        prepare_autosave(merged_ided, crs_authid, prompt, run_id,
                         source_layer=source_layer))


def repaint_layers_over(path: str) -> None:







    try:
        if not path:
            return
        target = os.path.normcase(os.path.abspath(path))
        for layer in QgsProject.instance().mapLayers().values():
            try:
                source = layer.source() or ""
                source = source.split("|", 1)[0]
                if not source:
                    continue
                if os.path.normcase(os.path.abspath(source)) == target:
                    layer.triggerRepaint()
            except (RuntimeError, AttributeError, ValueError, OSError):
                continue
    except Exception:  # nosec B110
        pass


def record_pending(info: dict) -> None:



    try:
        QSettings().setValue(_PENDING_KEY, json.dumps(info))
    except Exception:  # nosec B110
        pass
    try:
        from .server_dials import dial_in_range
        max_armed_tables = dial_in_range(
            "tuning.auto.max_armed_autosave_tables", _MAX_ARMED_TABLES, 2, 200)
        run_id = str(info.get("run_id") or "")
        if len(_ARMED_TABLES) >= max_armed_tables:
            _ARMED_TABLES.pop(next(iter(_ARMED_TABLES)), None)
        _ARMED_TABLES[run_id] = dict(info)
    except Exception:  # nosec B110
        pass


def _pointer_raw() -> dict | None:



    try:
        raw = QSettings().value(_PENDING_KEY, "", type=str)
        if not raw:
            return None
        info = json.loads(raw)
        return info if isinstance(info, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _forget_pointer() -> None:







    try:
        QSettings().remove(_PENDING_KEY)
    except Exception:  # nosec B110
        pass


def read_pending(check_file: bool = True) -> dict | None:








    try:
        info = _pointer_raw()
        if info is None:

            if QSettings().value(_PENDING_KEY, "", type=str):
                _forget_pointer()
            return None
        if not info.get("path") or not info.get("table"):
            _forget_pointer()
            return None
        if check_file:
            from .output_gpkg_rollover import file_size

            size = file_size(str(info["path"]))
            if size == 0:
                _forget_pointer()
                return None
            if size is None:
                return None
        return info
    except Exception:  # noqa: BLE001
        return None


def drop_autosave_table(path: str, table: str) -> None:







    if not path or not table or not os.path.exists(path):
        return
    try:
        from qgis.core import QgsProviderRegistry

        metadata = QgsProviderRegistry.instance().providerMetadata("ogr")
        if metadata is None:
            return
        connection = metadata.createConnection(path, {})
        if connection is None:
            return
        connection.dropVectorTable("", table)
    except Exception:  # noqa: BLE001
        pass  # nosec B110


def clear_pending(run_id: str | None = None, drop_table: bool = False) -> None:




















    pointer = _pointer_raw()
    mine = pointer if (run_id and (pointer or {}).get("run_id") == run_id) else None
    if pointer is not None and mine is None:



        if drop_table:
            _drop_run_table(run_id, protected=pointer)
        return
    _forget_pointer()
    if drop_table:
        _drop_run_table(run_id, fallback=mine, protected=None)
    elif run_id:
        _ARMED_TABLES.pop(str(run_id), None)


def _drop_run_table(run_id: str | None, fallback: dict | None = None,
                    protected: dict | None = None) -> None:









    key = str(run_id or "")
    info = fallback
    if info is None:
        info = _ARMED_TABLES.get(key)
    if not info:
        return
    path = str(info.get("path") or "")
    table = str(info.get("table") or "")
    if protected is not None and _same_table(protected, path, table):
        return
    drop_autosave_table(path, table)
    _ARMED_TABLES.pop(key, None)


def _path_key(path: str) -> str:

    return os.path.normcase(os.path.normpath(path))




def _same_table(info: dict, path: str, table: str) -> bool:

    try:
        other_path = str(info.get("path") or "")
        same_file = (other_path and path
                     and _path_key(other_path) == _path_key(path))
        return bool(same_file) and str(info.get("table") or "").lower() == table.lower()
    except (AttributeError, TypeError, ValueError, OSError):
        return True


def log_and_clear_stale_pending(current_run_id: str | None = None) -> None:




    try:
        info = read_pending()
        if not info:
            return
        if current_run_id and str(info.get("run_id") or "") == current_run_id:
            return
        QgsMessageLog.logMessage(
            "Auto detection: a previous session left {n} autosaved object(s) "
            "at {path} (table {table}). Not loaded.".format(
                n=int(info.get("count", 0) or 0),
                path=str(info.get("path") or ""),
                table=str(info.get("table") or "")),
            _LOG_TAG, level=Qgis.MessageLevel.Info)



        _forget_pointer()
    except Exception:  # nosec B110
        pass


def _measures_already_written(layer) -> bool:

    try:
        for feat in layer.getFeatures():
            return feat["area_m2"] is not None
    except (RuntimeError, AttributeError, KeyError, TypeError, ValueError):
        return False
    return False


def _fill_measure_fields(layer) -> None:








    provider = layer.dataProvider()
    have = {f.name().lower() for f in layer.fields()}
    missing = [n for n in ("area_m2", "perimeter_m") if n not in have]
    if not missing and _measures_already_written(layer):
        return
    if missing:
        added = provider.addAttributes([measure_field(n) for n in missing])
        layer.updateFields()
        if not added:
            return
    fields = layer.fields()
    idx_area = fields.indexOf("area_m2")
    idx_perimeter = fields.indexOf("perimeter_m")
    if idx_area < 0 and idx_perimeter < 0:
        return
    measurer = make_area_measurer(layer.crs())
    changes = {}
    for feat in layer.getFeatures():
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue
        values = {}
        if idx_area >= 0:
            values[idx_area] = round_measure(measurer.measureArea(geom))
        if idx_perimeter >= 0:
            values[idx_perimeter] = round_measure(measurer.measurePerimeter(geom))
        changes[feat.id()] = values
    if changes and not provider.changeAttributeValues(changes):


        QgsMessageLog.logMessage(
            "Run autosave: recovered run kept empty area/perimeter columns",
            _LOG_TAG, level=Qgis.MessageLevel.Warning)


def load_pending_layer(info: dict) -> str | None:



    try:
        path = str(info.get("path") or "")
        table = str(info.get("table") or "")
        if not path or not table:
            return None
        prompt = str(info.get("prompt") or "")
        display = str(info.get("layer_name") or "") or (
            output_store.friendly_layer_name(prompt))
        layer = QgsVectorLayer(f"{path}|layername={table}", display, "ogr")
        if not layer.isValid():
            return None





        try:
            layer.dataProvider().reloadData()
            layer.updateExtents()
        except (RuntimeError, AttributeError):  # nosec B110
            pass
        if layer.featureCount() == 0:
            return None
        from .layer_conventions import apply_output_conventions, make_committed_renderer
        layer.setRenderer(make_committed_renderer(
            color=output_store.committed_color_for_prompt(prompt)))


        try:
            _fill_measure_fields(layer)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        try:
            ts = float(info.get("ts") or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        from .output_metadata import timestamp_iso_from_epoch

        apply_output_conventions(
            layer, "",
            prompt=prompt,
            created_iso=timestamp_iso_from_epoch(ts) if ts else "",
        )
        output_store.add_committed_layer(layer)
        layer.triggerRepaint()
        QgsMessageLog.logMessage(
            f"Run autosave: recovered {layer.featureCount()} object(s) "
            f"from table {table}",
            _LOG_TAG, level=Qgis.MessageLevel.Info)
        return layer.name()
    except Exception:  # noqa: BLE001
        return None
