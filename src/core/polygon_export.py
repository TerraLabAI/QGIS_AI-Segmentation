







from __future__ import annotations

import contextlib
import threading
from typing import Any

from qgis.core import Qgis, QgsFeature, QgsField, QgsMessageLog
from qgis.PyQt.QtCore import QMetaObject, QObject, pyqtSlot

from .polygon_geometry import (
    count_overlapping_pairs,
)
from .qt_compat import field_type_string








EXPORT_DRIVERS = ("GPKG", "GeoJSON", "ESRI Shapefile", "KML")

_DRIVER_EXTENSIONS = {
    "GPKG": ".gpkg",
    "GeoJSON": ".geojson",
    "ESRI Shapefile": ".shp",
    "KML": ".kml",
}




_WGS84_ONLY_DRIVERS = ("GeoJSON", "KML")








_DRIVER_LAYER_OPTIONS = {
    "GeoJSON": ["RFC7946=YES", "WRITE_BBOX=YES", "COORDINATE_PRECISION=7"],
    "ESRI Shapefile": ["ENCODING=UTF-8"],
}




_SHAPEFILE_MEASURE_NAMES = ("area_m2", "perim_m")


def measure_field_names(driver: str) -> tuple[str, str]:

    if driver == "ESRI Shapefile":
        return _SHAPEFILE_MEASURE_NAMES
    return ("area_m2", "perimeter_m")


def driver_layer_options(driver: str) -> list[str]:

    return list(_DRIVER_LAYER_OPTIONS.get(driver, ()))


def driver_extension(driver: str) -> str:

    return _DRIVER_EXTENSIONS.get(driver, ".gpkg")


def _uri_layer_name(source: str) -> str:





    for part in str(source or "").split("|")[1:]:
        key, sep, value = part.partition("=")
        if sep and key.strip().lower() == "layername":
            return value.strip()
    return ""


def _tables_in_file(path: str) -> list[str]:





    from qgis.core import QgsProviderRegistry

    try:
        details = QgsProviderRegistry.instance().querySublayers(path)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return []
    numbered: dict[int, str] = {}
    for detail in details or []:
        try:
            numbered[int(detail.layerNumber())] = str(detail.name() or "")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
    if not numbered:
        return []
    return [numbered.get(i, "") for i in range(max(numbered) + 1)]


def _layer_table(layer: Any, source: str, path: str, cache: dict) -> str:







    named = _uri_layer_name(source)
    if named:
        return named
    try:
        if str(layer.providerType() or "").lower() != "ogr":
            return ""
    except (AttributeError, RuntimeError):
        return ""
    parts: dict = {}
    with contextlib.suppress(Exception):
        from qgis.core import QgsProviderRegistry

        parts = QgsProviderRegistry.instance().decodeUri("ogr", source) or {}
    decoded = parts.get("layerName")
    if decoded is not None and str(decoded) and str(decoded) != "NULL":
        return str(decoded)
    if path not in cache:
        cache[path] = _tables_in_file(path)
    tables = cache[path]
    if not tables:
        return ""
    try:
        index = int(parts.get("layerId") or 0)
    except (TypeError, ValueError):
        index = 0
    if 0 <= index < len(tables):
        return tables[index]
    return ""


def _release_project_layers_on_gui(path: str, table: str = "") -> int:














    import os

    from qgis.core import QgsProject

    try:
        target = os.path.normcase(os.path.abspath(path))
    except (OSError, ValueError):
        return 0



    wanted = str(table or "").strip().casefold()
    tables_by_file: dict[str, list[str]] = {}
    doomed = []
    for layer_id, layer in QgsProject.instance().mapLayers().items():
        try:


            source = layer.source() or ""
            file_part = source.split("|")[0]
            if not file_part:
                continue
            if os.path.normcase(os.path.abspath(file_part)) != target:
                continue
            if wanted and str(_layer_table(
                    layer, source, file_part,
                    tables_by_file) or "").casefold() != wanted:
                continue
            doomed.append(layer_id)
        except (AttributeError, RuntimeError, OSError, ValueError):
            continue
    if doomed:
        QgsProject.instance().removeMapLayers(doomed)
    return len(doomed)





_RELEASE_LAYERS_TIMEOUT_S = 5.0


class _LayerReleaseCall(QObject):







    def __init__(self, path: str, table: str = "") -> None:
        super().__init__()
        self._path = path
        self._table = table
        self.count = 0
        self.finished = threading.Event()

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.count = _release_project_layers_on_gui(self._path, self._table)
        except Exception:  # noqa: BLE001
            self.count = 0
        finally:
            self.finished.set()
            _pending_layer_releases.discard(self)
            self.deleteLater()





_pending_layer_releases: set[_LayerReleaseCall] = set()


def _release_project_layers_at(path: str, table: str = "") -> int:












    from qgis.core import QgsApplication
    from qgis.PyQt.QtCore import Qt, QThread

    from .qt_compat import resolve_qt_enum

    try:
        app = QgsApplication.instance()
        if app is None or QThread.currentThread() == app.thread():
            return _release_project_layers_on_gui(path, table)
    except (RuntimeError, AttributeError):
        return 0
    call = _LayerReleaseCall(path, table)
    try:
        call.moveToThread(app.thread())
        _pending_layer_releases.add(call)
        queued = resolve_qt_enum(Qt, "ConnectionType", "QueuedConnection")



        if QMetaObject.invokeMethod(call, "run", queued) is False:
            _pending_layer_releases.discard(call)
            return 0
    except (RuntimeError, AttributeError, TypeError):
        _pending_layer_releases.discard(call)
        return 0
    from .server_dials import dial_in_range
    release_timeout_s = dial_in_range(
        "tuning.export.release_layers_timeout_s", _RELEASE_LAYERS_TIMEOUT_S, 1.0, 30.0)
    if not call.finished.wait(release_timeout_s):
        QgsMessageLog.logMessage(
            "Export: gave up waiting for the layers holding the target file",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        return 0
    return call.count


def _on_gui_thread() -> bool:





    from qgis.core import QgsApplication
    from qgis.PyQt.QtCore import QThread

    try:
        app = QgsApplication.instance()
        return app is None or QThread.currentThread() == app.thread()
    except (RuntimeError, AttributeError):
        return True


def _batch_area_measurer(crs, transform_context=None, ellipsoid: str = ""):










    from .layer_conventions import make_area_measurer

    if transform_context is None:
        if not _on_gui_thread():
            QgsMessageLog.logMessage(
                "Export: no project measurement context on this thread, so "
                "area_m2 and perimeter_m are written empty",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return None


        return make_area_measurer(crs, None, str(ellipsoid) or None)



    return make_area_measurer(crs, transform_context, str(ellipsoid or ""))


def export_geometries_to_file(
    geoms: list,
    crs,
    output_path: str,
    driver: str = "GPKG",
    source_layer_name: str = "",
    layer_name: str | None = None,
    stats: dict | None = None,
    project_crs=None,
    transform_context=None,
    ellipsoid: str = "",
    scores: list | None = None,
    det_ids: list | None = None,
    object_class: str = "",
    prompt: str = "",
    detail: int | None = None,
    confidence: float | None = None,
):






















































    import os

    from qgis.core import (
        QgsCoordinateReferenceSystem,
        QgsCoordinateTransform,
        QgsProject,
        QgsVectorFileWriter,
        QgsVectorLayer,
    )

    from .layer_conventions import (
        apply_output_conventions,
        make_committed_renderer,
        measure_field,
        pick_output_crs,
        repair_polygon,
        round_measure,
        to_multipolygon,
    )

    if not geoms:
        return None
    if driver not in EXPORT_DRIVERS:
        driver = "GPKG"
    if not str(output_path or "").strip():
        return None



    output_path = os.path.abspath(output_path)

    stem = os.path.splitext(os.path.basename(output_path))[0]
    name = layer_name or stem or "detections"

    temp_layer = QgsVectorLayer("MultiPolygon", name, "memory")
    if not temp_layer.isValid():
        return None
    temp_layer.setCrs(crs)
    pr = temp_layer.dataProvider()


    area_name, perimeter_name = measure_field_names(driver)
    if not pr.addAttributes([
        QgsField("det_id", field_type_string()),
        QgsField("class", field_type_string()),


        measure_field("confidence", decimals=3),



        measure_field(area_name),
        measure_field(perimeter_name),
    ]):
        QgsMessageLog.logMessage(
            "Export: could not build the attribute schema",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        return None
    temp_layer.updateFields()

    measurer = _batch_area_measurer(crs, transform_context, ellipsoid)
    if scores is not None and len(scores) != len(geoms):
        scores = None
    if det_ids is not None and len(det_ids) != len(geoms):
        det_ids = None
    row_class = str(object_class or "").strip() or None
    feats = []
    skipped = 0
    written_geoms = []
    for index, geom in enumerate(geoms):
        if geom is None or geom.isEmpty():
            skipped += 1
            continue
        geom = to_multipolygon(repair_polygon(geom) or geom)
        if geom is None or geom.isEmpty():
            skipped += 1
            continue
        feat = QgsFeature(temp_layer.fields())
        feat.setGeometry(geom)
        if measurer is None:


            area, perimeter = None, None
        else:
            try:
                area = measurer.measureArea(geom)
                perimeter = measurer.measurePerimeter(geom)
            except (RuntimeError, AttributeError):





                area, perimeter = None, None
        score = scores[index] if scores is not None else None
        raw_id = det_ids[index] if det_ids is not None else index
        feat.setAttributes([
            str(raw_id),
            row_class,
            None if score is None else round(float(score), 3),
            round_measure(area),
            round_measure(perimeter),
        ])
        feats.append(feat)
        written_geoms.append(geom)



    if isinstance(stats, dict):
        stats["written"] = len(feats)
        stats["skipped"] = skipped
    if skipped:
        QgsMessageLog.logMessage(
            f"Export: {skipped} polygon(s) no repair could save, left out of "
            "the file",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
    if not feats:
        return None



    if not pr.addFeatures(feats):
        QgsMessageLog.logMessage(
            f"Export: could not stage {len(feats)} polygon(s) for writing",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        return None
    temp_layer.updateExtents()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as err:
            QgsMessageLog.logMessage(
                f"Export: cannot create directory: {err}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return None

    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = driver
    options.fileEncoding = "UTF-8"
    options.layerName = name
    driver_options = driver_layer_options(driver)
    if driver_options:
        options.layerOptions = driver_options







    from .output_gpkg_rollover import file_size

    options.actionOnExistingFile = (
        QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
        if driver == "GPKG" and file_size(output_path) != 0
        else QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
    )
    if os.path.exists(output_path):














        released = _release_project_layers_at(
            output_path, name if driver == "GPKG" else "")
        if released:
            QgsMessageLog.logMessage(
                f"Export: released {released} project layer(s) holding the "
                f"target file open before overwriting it",
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )


        from .export_file_lock import held_export_files

        held = held_export_files(output_path, driver)
        if held:
            QgsMessageLog.logMessage(
                f"Export: {len(held)} file(s) of the target are open in another "
                "program, so the existing file was left as it is. Close it "
                "there, or export under another name.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return None





    write_context = transform_context
    write_ellipsoid = str(ellipsoid or "")
    if write_context is None:
        if _on_gui_thread():
            write_context = QgsProject.instance().transformContext()
            if not write_ellipsoid:
                write_ellipsoid = str(QgsProject.instance().ellipsoid() or "")
        else:




            from qgis.core import QgsCoordinateTransformContext
            write_context = QgsCoordinateTransformContext()
    if driver in _WGS84_ONLY_DRIVERS:




        target = QgsCoordinateReferenceSystem("EPSG:4326")
    else:





        target = pick_output_crs(
            crs, temp_layer.extent(), project_crs,
            transform_context=write_context,
            ellipsoid=write_ellipsoid)
    if (crs is not None and crs.isValid() and target is not None and target.isValid() and target != crs):
        options.ct = QgsCoordinateTransform(crs, target, write_context)

    error = QgsVectorFileWriter.writeAsVectorFormatV3(
        temp_layer, output_path, write_context, options,
    )
    if error[0] != QgsVectorFileWriter.WriterError.NoError:
        QgsMessageLog.logMessage(
            f"Export failed ({driver}): {error[1]}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        return None






    result_layer = QgsVectorLayer(
        f"{output_path}|layername={name}", name, "ogr")
    if not result_layer.isValid():
        result_layer = QgsVectorLayer(output_path, name, "ogr")
    if not result_layer.isValid():
        QgsMessageLog.logMessage(
            "Export: file saved but could not be loaded back",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        return None






    try:
        result_layer.dataProvider().reloadData()
        result_layer.updateExtents()
    except (RuntimeError, AttributeError):  # nosec B110
        pass

    result_layer.setRenderer(make_committed_renderer())



    source_authid = ""
    try:
        if crs is not None and crs.isValid():
            source_authid = str(crs.authid() or "")
    except (RuntimeError, AttributeError):
        source_authid = ""
    apply_output_conventions(
        result_layer, source_layer_name,
        prompt=prompt or object_class,
        detail=detail,
        confidence=confidence,
        source_crs_authid=source_authid,
        overlapping_pairs=count_overlapping_pairs(written_geoms),
        store_style=(driver == "GPKG"),
    )
    return result_layer
