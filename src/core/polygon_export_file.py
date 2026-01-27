





from __future__ import annotations

import contextlib
import os
import shutil
import stat
import uuid
from collections import Counter
from typing import Callable

from qgis.core import Qgis, QgsCoordinateReferenceSystem, QgsGeometry, QgsMessageLog, QgsWkbTypes

from .file_replace_retry import replace_file_with_retry
from .qt_compat import PolygonGeometry


def _staging_path(output_path: str, driver: str) -> str:

    extension = {"GeoJSON": ".geojson", "KML": ".kml"}[driver]
    folder = os.path.dirname(output_path)
    length = 8
    if os.name == "nt":


        length = min(length, 259 - len(folder) - 1 - len(extension))
    if length < 1:
        raise OSError("No room for a staging filename beside the target")
    for _ in range(32):
        path = os.path.join(folder, uuid.uuid4().hex[:length] + extension)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue
        os.close(descriptor)
        return path
    raise OSError("Could not reserve a staging file beside the target")


def _validate_staged_file(path: str, expected_layer, *, table_name: str = "",
                          expected_crs=None) -> None:

    from osgeo import gdal

    dataset = table = feature = definition = spatial_ref = None
    try:
        dataset = gdal.OpenEx(path, gdal.OF_VECTOR | gdal.OF_READONLY)
        if dataset is None or (not table_name and dataset.GetLayerCount() != 1):
            raise ValueError("Staged export cannot be read as one vector layer")
        table = dataset.GetLayerByName(table_name) if table_name else dataset.GetLayer(0)
        if table is None:
            raise ValueError("Staged export table cannot be read")
        spatial_ref = table.GetSpatialRef()
        target = expected_crs if expected_crs is not None else QgsCoordinateReferenceSystem("EPSG:4326")
        if spatial_ref is None or QgsCoordinateReferenceSystem.fromWkt(
                spatial_ref.ExportToWkt()) != target:
            raise ValueError("Staged export has lost its coordinate system")
        definition = table.GetLayerDefn()
        if any(definition.GetFieldIndex(field.name()) < 0 for field in expected_layer.fields()):
            raise ValueError("Staged export has lost an attribute column")
        expected = Counter(str(row["det_id"]) for row in expected_layer.getFeatures())
        actual: Counter = Counter()
        for feature in table:
            geometry = feature.GetGeometryRef()
            shape = QgsGeometry()
            if geometry is None:
                raise ValueError("Staged export has lost an object geometry")
            shape.fromWkb(bytes(geometry.ExportToWkb()))
            if (shape.isEmpty() or not shape.isGeosValid()
                    or QgsWkbTypes.geometryType(shape.wkbType()) != PolygonGeometry):
                raise ValueError("Staged export contains an invalid geometry")
            actual[feature.GetFieldAsString("det_id")] += 1
        if actual != expected:
            raise ValueError("Staged export has lost or changed object identities")
    finally:



        geometry = feature = definition = spatial_ref = table = None
        dataset = None


def write_single_file(layer, output_path: str, context, options,
                      expected_layer, release_target: Callable[[], bool]) -> bool:

    from qgis.core import QgsVectorFileWriter

    staging = ""
    try:
        staging = _staging_path(output_path, options.driverName)
        error = QgsVectorFileWriter.writeAsVectorFormatV3(layer, staging, context, options)
        if error[0] != QgsVectorFileWriter.WriterError.NoError:
            raise RuntimeError(str(error[1]))
        _validate_staged_file(staging, expected_layer)



        with open(staging, "rb+") as saved:
            os.fsync(saved.fileno())
        if not release_target():
            return False
        if os.path.exists(output_path):



            shutil.copymode(output_path, staging)
        replace_file_with_retry(staging, output_path)
        staging = ""
        return True
    except (OSError, RuntimeError, ValueError) as error:
        QgsMessageLog.logMessage(
            f"Export failed ({options.driverName}): {error}. "
            "The existing file was left unchanged.",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return False
    finally:
        if staging:
            if os.name == "nt":


                with contextlib.suppress(OSError):
                    os.chmod(staging, os.stat(staging).st_mode | stat.S_IWRITE)
            with contextlib.suppress(OSError):
                os.remove(staging)
