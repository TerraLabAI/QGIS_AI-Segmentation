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
import re
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

# The word every autosave table name carries, between the prompt slug and the
# date snake_table_name appends. Written once here, read by the writer, by the
# orphan report and by the drop guard, so the naming rule has one home.
AUTOSAVE_TABLE_MARK = "autosave"
_AUTOSAVE_TABLE_RE = re.compile(
    rf"_{AUTOSAVE_TABLE_MARK}_[0-9]{{8}}(?:_[0-9]+)?$")

# Where this session armed an autosave, keyed by run id. The QSettings pointer
# holds ONE slot, so a second run overwrites it and the first run's table then
# had nothing left naming it: that is how a project file collects autosave
# tables nobody drops. Bounded, session-only, and never the authority on what
# is still pending: only the pointer is.
_ARMED_TABLES: dict[str, dict] = {}
_MAX_ARMED_TABLES = 16


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
        table = output_store.snake_table_name(
            f"{stem} {AUTOSAVE_TABLE_MARK}", gpkg_path)
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
    try:
        run_id = str(info.get("run_id") or "")
        if len(_ARMED_TABLES) >= _MAX_ARMED_TABLES:
            _ARMED_TABLES.pop(next(iter(_ARMED_TABLES)), None)
        _ARMED_TABLES[run_id] = dict(info)
    except Exception:  # nosec B110 -- the registry is a convenience
        pass


def _pointer_raw() -> dict | None:
    """The stored pointer exactly as written, with no repair and no side
    effect. ``read_pending`` and ``clear_pending`` both need to look without
    one calling the other."""
    try:
        raw = QSettings().value(_PENDING_KEY, "", type=str)
        if not raw:
            return None
        info = json.loads(raw)
        return info if isinstance(info, dict) else None
    except Exception:  # noqa: BLE001 -- unreadable pointer = no pending
        return None


def read_pending(check_file: bool = True) -> dict | None:
    """The pending autosave pointer, or None. With ``check_file`` (default) a
    pointer whose GeoPackage no longer exists is dropped and cleared."""
    try:
        info = _pointer_raw()
        if info is None:
            # Stored but unreadable: drop it, or it is re-parsed every start.
            if QSettings().value(_PENDING_KEY, "", type=str):
                clear_pending()
            return None
        if not info.get("path") or not info.get("table"):
            clear_pending()
            return None
        if check_file and not os.path.exists(str(info["path"])):
            clear_pending()
            return None
        return info
    except Exception:  # noqa: BLE001 -- unreadable pointer = no pending
        return None


def drop_autosave_table(path: str, table: str) -> None:
    """Delete one autosave table from its GeoPackage. Never raises.

    Only the table: the shared project file holds every committed run beside
    it, and a standalone fallback file is left in place empty rather than
    removed, since deleting a file the user may have opened is not this
    module's call to make.
    """
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
    except Exception:  # noqa: BLE001 -- a table left behind changes nothing
        pass  # nosec B110


def is_autosave_table(table: str) -> bool:
    """Whether this table name was produced by the autosave naming rule.

    The rule is ``<prompt slug> autosave`` through ``snake_table_name``, so the
    name ends in ``_autosave_<yyyymmdd>``, plus the ``_2`` the deduper appends.
    A prompt long enough to push the mark past the 40-character cut answers no,
    which costs a table left on disk and never risks one that is not ours.
    """
    return bool(table) and _AUTOSAVE_TABLE_RE.search(str(table)) is not None


def orphan_autosave_tables(gpkg_path: str) -> list[dict]:
    """Autosave tables in this GeoPackage that no armed pointer claims.

    REPORTS, never deletes. Each entry is ``{"table", "rows", "bytes"}``, where
    ``bytes`` sums the stored geometry blobs: the bulk of an autosave table,
    and an estimate, since SQLite hands pages back to the file's free list
    rather than to the disk until the file is vacuumed.

    Nothing in the plugin calls this on a schedule, and nothing should: one of
    these tables is the only unfiltered copy of a past run, held exactly for
    the session that died before its export. Read straight from SQLite, so a
    report costs no OGR layer opens. Never raises.
    """
    out: list[dict] = []
    if not gpkg_path or not os.path.exists(gpkg_path):
        return out
    pointer = _pointer_raw() or {}
    connection = None
    try:
        import sqlite3
        from pathlib import Path

        connection = sqlite3.connect(
            f"{Path(gpkg_path).as_uri()}?mode=ro", uri=True, timeout=0.5)
        geometry_column = {
            str(name): str(column)
            for name, column in connection.execute(
                "SELECT table_name, column_name FROM gpkg_geometry_columns")
        }
        names = [str(row[0]) for row in connection.execute(
            "SELECT table_name FROM gpkg_contents")]
        for name in sorted(names):
            if not is_autosave_table(name):
                continue
            if _same_table(pointer, gpkg_path, name):
                continue  # still pending: this copy is the only one left
            quoted = '"{}"'.format(name.replace('"', '""'))
            # The name comes from gpkg_contents and is quoted above.
            rows = _scalar(
                connection,
                f"SELECT COUNT(*) FROM {quoted}") or 0  # nosec B608
            column = geometry_column.get(name)
            size = 0
            if column:
                quoted_column = '"{}"'.format(column.replace('"', '""'))
                size = _scalar(
                    connection,
                    # Both names come from gpkg metadata and are quoted above.
                    f"SELECT SUM(LENGTH({quoted_column})) FROM {quoted}"  # nosec B608
                ) or 0
            out.append({"table": name, "rows": int(rows), "bytes": int(size)})
    except Exception:  # noqa: BLE001 -- a report that fails reports nothing
        return out
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:  # nosec B110
                pass
    return out


def _scalar(connection, statement: str):
    """First column of the first row, or None. Never raises."""
    try:
        row = connection.execute(statement).fetchone()  # nosec B608
    except Exception:  # noqa: BLE001 -- a table that will not answer counts 0
        return None
    return row[0] if row else None


def drop_listed_autosave_tables(gpkg_path: str, tables) -> list[str]:
    """Drop the autosave tables NAMED by the caller. Returns what went.

    The list is the caller's, always. Nothing here scans a file and deletes
    what it finds: pair it with ``orphan_autosave_tables`` and a human who
    read the report, because every table it names holds objects that exist
    nowhere else. Two names are refused whatever the caller asked: one that
    the autosave rule did not produce, and one an armed pointer still claims.
    """
    dropped: list[str] = []
    if not gpkg_path or not tables:
        return dropped
    pointer = _pointer_raw() or {}
    for table in tables:
        name = str(table or "")
        if not is_autosave_table(name) or _same_table(pointer, gpkg_path, name):
            continue
        drop_autosave_table(gpkg_path, name)
        dropped.append(name)
    return dropped


def clear_pending(run_id: str | None = None, drop_table: bool = False) -> None:
    """Drop the pending pointer. With ``run_id``, only when the stored pointer
    belongs to that run, so finishing today's run never consumes a previous
    session's still-unrecovered autosave.

    ``drop_table`` also deletes that run's autosave table. It is for the caller
    that just wrote the same objects to a real layer: the crash-net copy is
    then a duplicate, and without this every finished run leaves one more table
    in the project's GeoPackage for good. Every other caller keeps the file,
    because that is where it is the only copy left.

    The table is found from the pointer, and from this session's own arming
    record when the pointer is gone, was never armed, or carries no run id. It
    used to be found from the pointer alone, under a run id test that a caller
    with an empty run id could not pass, so those runs kept their table for
    good. A table an armed pointer for ANOTHER run claims is never dropped:
    there it is still the only copy of that run.
    """
    pointer = _pointer_raw()
    mine = pointer if (not run_id or (pointer or {}).get("run_id") == run_id) else None
    if pointer is not None and mine is None:
        # Someone else's run is still pending: leave its pointer armed. This
        # run's own table can still go, when this session recorded where it is.
        if drop_table:
            _drop_run_table(run_id, protected=pointer)
        return
    try:
        QSettings().remove(_PENDING_KEY)
    except Exception:  # nosec B110
        pass
    if drop_table:
        _drop_run_table(run_id, fallback=mine, protected=None)
    elif run_id:
        _ARMED_TABLES.pop(str(run_id), None)


def _drop_run_table(run_id: str | None, fallback: dict | None = None,
                    protected: dict | None = None) -> None:
    """Delete the autosave table of ``run_id``, when this session can name it.

    ``fallback`` is the pointer that already names it. ``protected`` is a
    pointer still armed for another run, whose table stays whatever happens.
    """
    info = fallback
    if info is None and run_id:
        info = _ARMED_TABLES.get(str(run_id))
    if not info:
        return
    path = str(info.get("path") or "")
    table = str(info.get("table") or "")
    if protected is not None and _same_table(protected, path, table):
        return
    drop_autosave_table(path, table)
    if run_id:
        _ARMED_TABLES.pop(str(run_id), None)


def _same_table(info: dict, path: str, table: str) -> bool:
    """Whether ``info`` points at this very table, on this very file."""
    try:
        other_path = str(info.get("path") or "")
        same_file = (other_path and path
                     and os.path.normcase(other_path) == os.path.normcase(path))
        return bool(same_file) and str(info.get("table") or "").lower() == table.lower()
    except (AttributeError, TypeError, ValueError):
        return True  # cannot tell them apart: keep the table


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
