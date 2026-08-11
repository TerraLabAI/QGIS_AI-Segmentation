"""Crash-net autosave of a finished run's merged detections.

The moment a run's merger is read at finalize, the merged scored set is
streamed to a GeoPackage table, BEFORE the review-entry tail (sweep, build,
filter, review) runs on it. If QGIS dies anywhere past that point, the billed
detections are still on disk. Nothing on screen offers them back (removed
2026-07-30): the same-session finalize error path reloads them silently, and a
pointer left by a dead session is written to the log with its path, then
dropped.

Storage reuses the committed-run conventions from ``output_store`` (the same
per-project ``ai_segmentation.gpkg``, the same table/name shapes), so a
recovered layer looks exactly like one the review's own autosave would have
produced. A small QSettings pointer marks the one run whose autosave was
never exported; exporting or discarding the run clears it.

All functions are best-effort: a failure returns None / no-op, never an
exception at the call site. Main-thread only (QgsProject access).
"""
from __future__ import annotations

import json
import os
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
from .qt_compat import WkbMultiPolygon, field_type_double, field_type_string

_LOG_TAG = "AI Segmentation"


def _autosave_output_crs(source_crs, merged_ided):
    """``(crs, transform)`` the autosave should be written in.

    Mirrors ``output_store._ground_metre_transform``, which reads a memory
    layer this path never builds, so the extent is taken off the geometries
    themselves. ``(source_crs, None)`` when the run CRS already measures in
    ground metres, which leaves the common case untouched.
    """
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


# QSettings pointer to the single run autosave that was never exported.
# JSON: {path, table, layer_name, prompt, run_id, count, ts}.
_PENDING_KEY = "AISegmentation/pending_run_autosave"


def _autosave_fields() -> QgsFields:
    """The lean autosave schema: the user annotation column, the object class
    (the run prompt) and the per-object confidence score. Measures (area,
    perimeter) are recomputed by the normal export; the crash net stays
    write-cheap."""
    fields = QgsFields()
    fields.append(QgsField("label", field_type_string()))
    fields.append(QgsField("class", field_type_string()))
    fields.append(QgsField("score", field_type_double()))
    return fields


def _open_writer(path: str, table: str, fields: QgsFields, crs):
    """One QgsVectorFileWriter over ``path``/``table``, or None."""
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.fileEncoding = "UTF-8"
    options.layerName = table
    options.actionOnExistingFile = (
        QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
        if os.path.exists(path)
        else QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
    )
    try:
        writer = QgsVectorFileWriter.create(
            path, fields, WkbMultiPolygon, crs,
            QgsProject.instance().transformContext(), options)
    except Exception:  # noqa: BLE001 -- a broken writer means no autosave here
        return None
    if writer is None:
        return None
    if writer.hasError() != QgsVectorFileWriter.WriterError.NoError:
        del writer
        return None
    return writer


def write_autosave(merged_ided: list, crs_authid: str, prompt: str,
                   run_id: str, source_layer=None) -> dict | None:
    """Stream the merged scored set ((fid, geom, score) triples) to a
    GeoPackage table in one pass, no temp memory layer. A None score is written
    as NULL, for a polygon no model scored.

    Lands in the project's shared GeoPackage next to the committed runs; if
    that file cannot be written, falls back to a standalone per-run file in
    the same output directory. Returns the pending-pointer dict on success
    (file written, at least one feature), None otherwise. Never raises.
    """
    try:
        if not merged_ided:
            return None
        crs = QgsCoordinateReferenceSystem(crs_authid or "EPSG:4326")
        # A recovered run is a deliverable like any other, so it gets the same
        # ground-metre move the two committed write paths apply. Without it a
        # crash hands back a file in degrees or in Mercator, where the columns
        # stay right but $area and any downstream buffer are off by
        # 1/cos(latitude).
        crs, ground_metre_xform = _autosave_output_crs(crs, merged_ided)
        fields = _autosave_fields()
        stem = (prompt or "").strip() or "detection"
        gpkg_path = output_store.project_gpkg_path(source_layer)
        table = output_store.snake_table_name(stem + " autosave", gpkg_path)
        writer = _open_writer(gpkg_path, table, fields, crs)
        if writer is None:
            # Shared file locked/corrupt: a standalone file still saves the run.
            # Walk the directories, since the folder itself can be what refuses
            # the write (a share or a WSL mount takes the file and then denies
            # the SQLite lock), and a new name there fails the same way.
            for directory in output_store.output_directory_candidates(source_layer):
                fallback = os.path.join(
                    directory, f"{table}_{time.strftime('%H%M%S')}.gpkg")
                writer = _open_writer(fallback, table, fields, crs)
                if writer is not None:
                    gpkg_path = fallback
                    break
            if writer is None:
                QgsMessageLog.logMessage(
                    "Run autosave: could not open a GeoPackage writer",
                    _LOG_TAG, level=Qgis.MessageLevel.Warning)
                return None

        from .layer_conventions import to_multipolygon
        object_class = (prompt or "").strip()
        count = 0
        for _fid, geom, score in merged_ided:
            try:
                if geom is None or geom.isEmpty():
                    continue
                multi = to_multipolygon(geom)
                if multi is None or multi.isEmpty():
                    continue
                if ground_metre_xform is not None:
                    # Reproject a geometry this loop OWNS, never the caller's.
                    # to_multipolygon hands the argument straight back when it
                    # is already multipart, and these come from the merger's
                    # keepers, so transforming in place moved the run's own
                    # objects into the output CRS behind its back. Everything
                    # downstream still measures in the render CRS the run
                    # declares, so an object that got moved read about 400x too
                    # small and the review's min-size floor deleted it.
                    # QgsGeometry(other) is a shallow copy and shares the same
                    # abstract geometry: the inner clone is what detaches.
                    inner = multi.constGet()
                    multi = (QgsGeometry(inner.clone()) if inner is not None
                             else QgsGeometry(multi))
                    multi.transform(ground_metre_xform)
                feat = QgsFeature(fields)
                feat.setGeometry(multi)
                # No score means no model scored this polygon (a hand save), so
                # the column stays NULL rather than claiming a confidence.
                feat.setAttributes([
                    "", object_class,
                    None if score is None else round(float(score), 3)])
                writer.addFeature(feat)
                count += 1
            except Exception:  # nosec B112 -- one bad geometry never stops the save
                continue
        had_error = writer.hasError() != QgsVectorFileWriter.WriterError.NoError
        del writer  # flush + close
        if count == 0 or had_error:
            return None
        return {
            "path": gpkg_path,
            "table": table,
            "layer_name": output_store.friendly_layer_name(stem),
            "prompt": object_class,
            "run_id": run_id or "",
            "count": count,
            "ts": time.time(),
        }
    except Exception:  # noqa: BLE001 -- the crash net must never break finalize
        try:
            QgsMessageLog.logMessage(
                "Run autosave: write failed", _LOG_TAG,
                level=Qgis.MessageLevel.Warning)
        except Exception:  # nosec B110
            pass
        return None


def record_pending(info: dict) -> None:
    """Mark ``info`` (a write_autosave result) as the run autosave that has
    not been exported yet. One slot: a newer run supersedes an older pointer
    (its table stays on disk either way)."""
    try:
        QSettings().setValue(_PENDING_KEY, json.dumps(info))
    except Exception:  # nosec B110 -- best-effort marker
        pass


def read_pending(check_file: bool = True) -> dict | None:
    """The pending autosave pointer, or None. With ``check_file`` (default) a
    pointer whose GeoPackage no longer exists is dropped and cleared."""
    try:
        raw = QSettings().value(_PENDING_KEY, "", type=str)
        if not raw:
            return None
        info = json.loads(raw)
        if not isinstance(info, dict) or not info.get("path") or not info.get("table"):
            clear_pending()
            return None
        if check_file and not os.path.exists(str(info["path"])):
            clear_pending()
            return None
        return info
    except Exception:  # noqa: BLE001 -- unreadable pointer = no pending
        return None


def clear_pending(run_id: str | None = None) -> None:
    """Drop the pending pointer. With ``run_id``, only when the stored pointer
    belongs to that run, so finishing today's run never consumes a previous
    session's still-unrecovered autosave."""
    try:
        if run_id:
            info = read_pending(check_file=False)
            if not info or info.get("run_id") != run_id:
                return
        QSettings().remove(_PENDING_KEY)
    except Exception:  # nosec B110
        pass


def log_and_clear_stale_pending(current_run_id: str | None = None) -> None:
    """A pointer left armed by a session that died with a review open reaches
    nobody: write its table path to the QGIS log so the file can still be
    opened by hand, then drop the pointer so it stops outliving its run.
    ``current_run_id`` guards the live run's own pointer. Never raises."""
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
        clear_pending()
    except Exception:  # nosec B110 -- a log line never breaks a start
        pass


def _fill_measure_fields(layer) -> None:
    """Add and fill the area_m2 / perimeter_m columns a normal export writes.

    The autosave itself stays write-cheap and stores neither, so a recovered
    run would otherwise hand the user a narrower attribute table than the same
    run exported the normal way. Best-effort: a read-only file simply keeps the
    lean schema.
    """
    from .layer_conventions import make_area_measurer, round_measure

    provider = layer.dataProvider()
    have = {f.name().lower() for f in layer.fields()}
    missing = [n for n in ("area_m2", "perimeter_m") if n not in have]
    if missing:
        added = provider.addAttributes(
            [QgsField(n, field_type_double()) for n in missing])
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
        # The columns exist but stayed empty, which reads as "this run has no
        # measures" rather than "the file would not take them".
        QgsMessageLog.logMessage(
            "Run autosave: recovered run kept empty area/perimeter columns",
            _LOG_TAG, level=Qgis.MessageLevel.Warning)


def load_pending_layer(info: dict) -> str | None:
    """Load a pending autosave table into the project with the committed-layer
    conventions (per-prompt color, the AI Segmentation group). Returns the
    layer name on success, None otherwise. Never raises."""
    try:
        path = str(info.get("path") or "")
        table = str(info.get("table") or "")
        if not path or not table:
            return None
        prompt = str(info.get("prompt") or "")
        display = str(info.get("layer_name") or "") or (
            output_store.friendly_layer_name(prompt))
        layer = QgsVectorLayer(f"{path}|layername={table}", display, "ogr")
        if not layer.isValid() or layer.featureCount() == 0:
            return None
        from .layer_conventions import apply_output_conventions, make_committed_renderer
        layer.setRenderer(make_committed_renderer(
            color=output_store.committed_color_for_prompt(prompt)))
        # A recovered run must look like a normally exported one: same columns,
        # same style, same provenance.
        try:
            _fill_measure_fields(layer)
        except Exception:  # noqa: BLE001 -- measures are a nicety, not the save
            pass  # nosec B110
        try:
            ts = float(info.get("ts") or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        apply_output_conventions(
            layer, "",
            prompt=prompt,
            created_iso=(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
                         if ts else ""),
        )
        output_store.add_committed_layer(layer)
        layer.triggerRepaint()
        QgsMessageLog.logMessage(
            f"Run autosave: recovered {layer.featureCount()} object(s) "
            f"from table {table}",
            _LOG_TAG, level=Qgis.MessageLevel.Info)
        return layer.name()
    except Exception:  # noqa: BLE001 -- a failed load leaves the pointer armed
        return None
