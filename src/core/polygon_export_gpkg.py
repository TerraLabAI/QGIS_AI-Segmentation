








from __future__ import annotations

import contextlib
import uuid
from typing import Callable

from qgis.core import Qgis, QgsMessageLog

from .output_gpkg_lock import gpkg_write_lock
from .output_gpkg_rollover import file_size
from .polygon_export_file import _validate_staged_file


def _rename_staged_table(dataset, stage: str, target: str) -> None:

    from osgeo import gdal

    source_identifier = '"' + stage.replace('"', '""') + '"'
    target_identifier = '"' + target.replace('"', '""') + '"'
    gdal.ErrorReset()
    dataset.ExecuteSQL(f"ALTER TABLE {source_identifier} RENAME TO {target_identifier}")
    if gdal.GetLastErrorType() >= gdal.CE_Failure:
        raise RuntimeError(gdal.GetLastErrorMsg())
    layer = dataset.GetLayerByName(target)
    if layer is None:
        raise RuntimeError("GeoPackage driver did not publish the staging table")
    layer = None


def _publish_staged_table(path: str, stage: str, target: str) -> None:

    from osgeo import gdal, ogr

    dataset = None
    transaction = False
    try:
        dataset = gdal.OpenEx(path, gdal.OF_VECTOR | gdal.OF_UPDATE)
        if dataset is None or not dataset.TestCapability(ogr.ODsCTransactions):
            raise RuntimeError("GeoPackage driver does not support native transactions")
        if dataset.StartTransaction() != ogr.OGRERR_NONE:
            raise RuntimeError("Could not start the GeoPackage publication transaction")
        transaction = True


        old_layer = dataset.GetLayerByName(target)
        old_name = old_layer.GetName() if old_layer is not None else None
        old_layer = None
        if old_name == stage:
            raise RuntimeError("The staging table cannot replace itself")
        previous = next((index for index in range(dataset.GetLayerCount())
                         if dataset.GetLayer(index).GetName() == old_name), None)
        if previous is not None and dataset.DeleteLayer(previous) != ogr.OGRERR_NONE:
            raise RuntimeError("Could not replace the previous GeoPackage table")
        _rename_staged_table(dataset, stage, target)
        if dataset.CommitTransaction() != ogr.OGRERR_NONE:
            raise RuntimeError("Could not commit the GeoPackage publication transaction")
        transaction = False
    finally:
        if dataset is not None and transaction:
            if dataset.RollbackTransaction() != ogr.OGRERR_NONE:
                QgsMessageLog.logMessage(
                    "Export: GeoPackage publication rollback failed",
                    "AI Segmentation", level=Qgis.MessageLevel.Critical)
        dataset = None


def _remove_staged_table(path: str, stage: str) -> None:

    from osgeo import gdal, ogr

    dataset = None
    try:
        dataset = gdal.OpenEx(path, gdal.OF_VECTOR | gdal.OF_UPDATE)
        if dataset is None:
            raise RuntimeError("Could not open the GeoPackage to remove its staging table")
        index = next((index for index in range(dataset.GetLayerCount())
                      if dataset.GetLayer(index).GetName() == stage), None)
        if index is not None and dataset.DeleteLayer(index) != ogr.OGRERR_NONE:
            raise RuntimeError("Could not remove the unpublished GeoPackage table")
    finally:
        dataset = None


def write_gpkg_table(layer, output_path: str, context, options, expected_layer,
                     expected_crs, release_target: Callable[[], bool]) -> bool:

    from qgis.core import QgsVectorFileWriter

    stage = "__aiseg_export_" + uuid.uuid4().hex
    target = options.layerName
    action = options.actionOnExistingFile
    published = False
    with gpkg_write_lock(output_path) as acquired:
        if not acquired:
            QgsMessageLog.logMessage(
                "Export: another writer is using this output folder. Try again when it finishes.",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return False
        try:
            options.layerName = stage


            options.actionOnExistingFile = (
                QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
                if file_size(output_path) != 0
                else QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile)
            result = QgsVectorFileWriter.writeAsVectorFormatV3(layer, output_path, context, options)
            options.layerName = target
            options.actionOnExistingFile = action
            if result[0] != QgsVectorFileWriter.WriterError.NoError:
                raise RuntimeError(str(result[1]))
            _validate_staged_file(output_path, expected_layer,
                                  table_name=stage, expected_crs=expected_crs)
            if not release_target():
                return False
            _publish_staged_table(output_path, stage, target)
            published = True
            return True
        except (OSError, RuntimeError, ValueError) as error:
            QgsMessageLog.logMessage(
                f"Export failed (GPKG): {error}", "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return False
        finally:
            options.layerName = target
            options.actionOnExistingFile = action
            if not published:
                with contextlib.suppress(OSError, RuntimeError, ValueError):
                    _remove_staged_table(output_path, stage)
