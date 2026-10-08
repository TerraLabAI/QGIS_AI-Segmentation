









# ruff: noqa: E402


from __future__ import annotations

from .venv_manager import ensure_venv_packages_available

ensure_venv_packages_available()

import contextlib

import numpy as np
from qgis.core import Qgis, QgsGeometry, QgsMessageLog

from .polygon_packing import pack_disjoint_crops
from .polygon_refine import (
    _mask_bounding_window,
)
from .polygon_trace import (
    gdal_vertex_rule,
    north_up_geotransform,
    polygon_wkb,
    trace_crop_rings,
    vertex_tables_many,
)





POLYGONIZERS = ("gdal", "tracer", "fallback", "fallback_fast")


def polygonize_label_raster(raster: np.ndarray,
                            geotransform: tuple) -> list[tuple[int, bytes]]:









    from osgeo import gdal, ogr

    height, width = int(raster.shape[0]), int(raster.shape[1])
    if raster.dtype == np.uint8:
        data_type = gdal.GDT_Byte
    else:
        data_type = gdal.GDT_Int32
        raster = raster.astype(np.int32, copy=False)
    with _gdal_errors_raise(gdal, ogr):
        source = gdal.GetDriverByName("MEM").Create("", width, height, 1, data_type)
        if source is None:
            raise RuntimeError("GDAL could not create the polygonize raster")
        source.SetGeoTransform(geotransform)
        band = source.GetRasterBand(1)
        if band.WriteRaster(0, 0, width, height,
                            np.ascontiguousarray(raster).tobytes()) != 0:
            raise RuntimeError("GDAL could not fill the polygonize raster")
        target = _memory_vector_dataset(gdal)
        layer = (target.CreateLayer("polygons", geom_type=ogr.wkbPolygon)
                 if target is not None else None)
        if layer is None or layer.CreateField(ogr.FieldDefn("value", ogr.OFTInteger)) != 0:
            raise RuntimeError("GDAL could not create the polygon layer")

        if gdal.Polygonize(band, band, layer, 0, [], callback=None) != 0:
            raise RuntimeError("GDAL polygonize failed")
        out = []
        feature = layer.GetNextFeature()
        while feature is not None:
            out.append((feature.GetFieldAsInteger(0),
                        bytes(feature.GetGeometryRef().ExportToIsoWkb(ogr.wkbNDR))))
            feature = layer.GetNextFeature()
    return out


def rasterize_touched_mask(geometry_wkb: bytes, height: int, width: int,
                           geotransform: tuple) -> np.ndarray:



    from osgeo import gdal, ogr

    with _gdal_errors_raise(gdal, ogr):
        raster = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
        if raster is None:
            raise RuntimeError("GDAL could not create the rasterize grid")
        raster.SetGeoTransform(geotransform)
        shape = ogr.CreateGeometryFromWkb(geometry_wkb)
        source = _memory_vector_dataset(gdal)
        layer = source.CreateLayer("shape") if source is not None else None
        if shape is None or layer is None:
            raise RuntimeError("GDAL could not hold the geometry to rasterize")
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(shape)
        if (layer.CreateFeature(feature) != 0
                or gdal.RasterizeLayer(raster, [1], layer, burn_values=[1],
                                       options=["ALL_TOUCHED=TRUE"]) != 0):
            raise RuntimeError("GDAL rasterize failed")
        cells = raster.GetRasterBand(1).ReadRaster(0, 0, width, height)
    return np.frombuffer(cells, dtype=np.uint8).reshape(height, width).astype(np.bool_)


def _memory_vector_dataset(gdal):


    driver = gdal.GetDriverByName(
        "MEM" if int(gdal.VersionInfo("VERSION_NUM")) >= 3110000 else "Memory")
    return driver.Create("", 0, 0, 0, gdal.GDT_Unknown)


def _gdal_errors_raise(gdal, ogr):




    if not (hasattr(gdal, "ExceptionMgr") and hasattr(ogr, "ExceptionMgr")):
        return contextlib.nullcontext()
    scope = contextlib.ExitStack()
    scope.enter_context(gdal.ExceptionMgr(useExceptions=True))
    scope.enter_context(ogr.ExceptionMgr(useExceptions=True))
    return scope


def _polygons_from_wkb(polygonized: list, simplify_tolerance: float) -> list[QgsGeometry]:

    geometries = []
    for _value, wkb in polygonized:
        geom = QgsGeometry()
        geom.fromWkb(wkb)
        geom = _finish_traced_geometry(geom, simplify_tolerance)
        if geom is not None:
            geometries.append(geom)
    return geometries


def _finish_traced_geometry(geom: QgsGeometry | None,
                            simplify_tolerance: float) -> QgsGeometry | None:


    if geom is None or geom.isEmpty():
        return None
    if simplify_tolerance <= 0:
        return geom if geom.isGeosValid() else None





    simplified = geom.simplify(simplify_tolerance)
    if (simplified is not None and not simplified.isEmpty()
            and simplified.isGeosValid()):
        return simplified
    if not geom.isGeosValid():
        return None
    if simplified is None:
        return geom
    return _repaired_simplification(geom, simplified)


def simplify_and_revalidate(geom: QgsGeometry,
                            tolerance: float) -> QgsGeometry:








    if tolerance <= 0:
        return geom
    simplified = geom.simplify(tolerance)
    if simplified is None:
        return geom
    if not simplified.isEmpty() and simplified.isGeosValid():
        return simplified
    return _repaired_simplification(geom, simplified)


def _repaired_simplification(geom: QgsGeometry,
                             simplified: QgsGeometry) -> QgsGeometry:


    fixed = simplified.makeValid()
    if fixed is not None and not fixed.isEmpty() and fixed.isGeosValid():
        return fixed
    return geom


def masks_to_polygons_packed(
    crops: list,
    transform_info: dict,
    full_shape: tuple[int, int],
    simplify_tolerance: float = 0.0,
    max_side: int | None = None,
    outlines: list | None = None,
    skip_below_area: float = 0.0,
    path_counts: dict | None = None,
) -> list[list[QgsGeometry]]:













































    out: list[list[QgsGeometry]] = [[] for _ in crops]
    if not crops:
        return out
    if outlines is not None and len(outlines) < len(crops):


        outlines = [*outlines, *([None] * (len(crops) - len(outlines)))]
    bbox = transform_info.get("bbox")
    if not bbox:
        return out



    use_tracer = _fast_ring_tracer_enabled()




    minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
    full_h, full_w = int(full_shape[0]), int(full_shape[1])
    px_w = (maxx - minx) / max(full_w, 1)
    px_h = (maxy - miny) / max(full_h, 1)

    boxes = []
    for crop, (row0, col0) in crops:
        h, w = int(crop.shape[0]), int(crop.shape[1])
        boxes.append((int(row0), int(row0) + h - 1, int(col0), int(col0) + w - 1))

    try:


        rule = gdal_vertex_rule() if use_tracer else None
        packs = []
        for indices, (br0, br1, bc0, bc1) in pack_disjoint_crops(boxes, max_side):
            pack_h, pack_w = br1 - br0 + 1, bc1 - bc0 + 1
            pack_minx = minx + bc0 * px_w
            pack_maxy = maxy - br0 * px_h
            geotransform = north_up_geotransform(
                pack_minx, pack_maxy - pack_h * px_h,
                pack_minx + pack_w * px_w, pack_maxy,
                pack_w, pack_h,
            )



            px_area = abs(geotransform[1] * geotransform[5])
            skip_box = (skip_below_area * (1.0 - 1e-6) / px_area
                        if skip_below_area > 0.0 and px_area > 0.0 else 0.0)
            planned = (_plan_pack_traced(crops, indices, outlines, skip_box)
                       if rule is not None else None)
            packs.append((indices, br0, bc0, pack_h, pack_w, geotransform, planned))
        traced = [p for p in packs if p[6] is not None]
        tables = vertex_tables_many(
            [(p[5], p[3], p[4]) for p in traced], rule) if traced else []
        for (_indices, br0, bc0, _h, _w, _gt, planned), table in zip(traced, tables):
            for i, polygons in planned:
                r0, _r1, c0, _c1 = boxes[i]
                for polygon in polygons:
                    geom = QgsGeometry()
                    geom.fromWkb(polygon_wkb(polygon, table, r0 - br0, c0 - bc0))
                    geom = _finish_traced_geometry(geom, simplify_tolerance)
                    if geom is not None:
                        out[i].append(geom)
        by_gdal = 0
        for indices, br0, bc0, pack_h, pack_w, geotransform, planned in packs:
            if planned is not None:
                continue
            by_gdal += len(indices)
            lab = np.zeros((pack_h, pack_w), np.int32)
            for label, i in enumerate(indices, start=1):
                crop, _origin = crops[i]
                outline = outlines[i] if outlines is not None else None
                if outline is not None and outline.pixels is not None:

                    crop = outline.pixels()
                r0, _r1, c0, _c1 = boxes[i]
                ro, co = r0 - br0, c0 - bc0



                stamp = crop if crop.dtype == np.bool_ else crop != 0
                np.copyto(lab[ro:ro + crop.shape[0], co:co + crop.shape[1]],
                          np.int32(label), where=stamp)
            for label, wkb in polygonize_label_raster(lab, geotransform):
                if label <= 0 or label > len(indices):
                    continue
                out[indices[label - 1]].extend(
                    _polygons_from_wkb([(label, wkb)], simplify_tolerance))
        if path_counts is not None:
            _count_polygonizer(path_counts, "gdal", by_gdal)
            _count_polygonizer(path_counts, "tracer", len(crops) - by_gdal)
        return out
    except Exception as exc:  # noqa: BLE001
        QgsMessageLog.logMessage(
            f"Packed polygonize failed ({exc}); falling back to per-mask",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        crops = _filled_crops(crops, outlines)
        if path_counts is not None:
            _count_polygonizer(path_counts, "gdal", len(crops))
        return [
            mask_to_polygons(
                crop, transform_info, simplify_tolerance=simplify_tolerance,
                pixel_offset=(int(col0), int(row0)), full_shape=full_shape,
            )
            for crop, (row0, col0) in crops
        ]


def _fast_ring_tracer_enabled() -> bool:

    try:
        from .server_dials import feature_enabled

        return feature_enabled("fast_ring_tracer")
    except Exception:  # noqa: BLE001  # nosec B110
        return True


def _count_polygonizer(path_counts: dict, name: str, crops: int) -> None:

    if crops:
        path_counts[name] = path_counts.get(name, 0) + crops


def _filled_crops(crops: list, outlines: list | None) -> list:



    if outlines is None:
        return crops
    filled = []
    for i, (crop, origin) in enumerate(crops):
        outline = outlines[i] if i < len(outlines) else None
        filled.append((outline.pixels(), origin)
                      if outline is not None and outline.pixels is not None
                      else (crop, origin))
    return filled


def _plan_pack_traced(crops: list, indices: list, outlines: list | None,
                      skip_box: float = 0.0) -> list | None:



    planned = []
    try:
        for i in indices:
            outline = outlines[i] if outlines is not None else None
            if outline is None:
                outline = trace_crop_rings(crops[i][0])
            polygons = outline.polygons(skip_box) if outline is not None else None
            if polygons is None:
                return None
            planned.append((i, polygons))
    except Exception:  # noqa: BLE001
        return None
    return planned


def _mask_grid_info(mask: np.ndarray, transform_info: dict,
                    pixel_offset: tuple[int, int] | None = None,
                    full_shape: tuple[int, int] | None = None) -> dict:






    info = dict(transform_info)
    sub_h, sub_w = int(mask.shape[0]), int(mask.shape[1])
    info["img_shape"] = (sub_h, sub_w)
    bbox = info.get("bbox")
    if not bbox:
        extent = info.get("extent")
        if extent:
            minx, miny, maxx, maxy = extent
            bbox = (minx, maxx, miny, maxy)
    if bbox:
        minx, maxx, miny, maxy = bbox
        if pixel_offset is not None and full_shape is not None:
            full_h, full_w = int(full_shape[0]), int(full_shape[1])
            if full_h <= 0 or full_w <= 0:
                raise ValueError("Mask grid must have positive dimensions")
            col0, row0 = int(pixel_offset[0]), int(pixel_offset[1])
            px_w = (maxx - minx) / full_w
            px_h = (maxy - miny) / full_h
            minx = minx + col0 * px_w
            maxy = maxy - row0 * px_h
            maxx = minx + sub_w * px_w
            miny = maxy - sub_h * px_h
        info["bbox"] = (minx, maxx, miny, maxy)
    return info


def mask_to_polygons(
    mask: np.ndarray,
    transform_info: dict,
    simplify_tolerance: float = 0.0,
    pixel_offset: tuple[int, int] | None = None,
    full_shape: tuple[int, int] | None = None,
) -> list[QgsGeometry]:




    if mask is None or not mask.any():
        return []

    try:
        bbox = _mask_grid_info(mask, transform_info, pixel_offset, full_shape).get("bbox")

        if not bbox:
            return []





        minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]







        height, width = int(mask.shape[0]), int(mask.shape[1])
        gt = north_up_geotransform(minx, miny, maxx, maxy, width, height)





        window = _mask_bounding_window(mask)
        if window is None:
            return []
        row0, row1, col0, col1 = window
        gt = (gt[1] * col0 + gt[0], gt[1], 0.0, gt[5] * row0 + gt[3], 0.0, gt[5])


        sub = np.ascontiguousarray(mask[row0:row1, col0:col1] != 0, dtype=np.uint8)
        return _polygons_from_wkb(polygonize_label_raster(sub, gt), simplify_tolerance)
    except Exception as e:
        import traceback
        QgsMessageLog.logMessage(
            f"mask_to_polygons error: {str(e)}\n{traceback.format_exc()}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Warning
        )


        try:
            from .telemetry_errors import track_plugin_error_once

            track_plugin_error_once("segment", "mask_convert_failed",
                                    f"{type(e).__name__}: {e}")
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return []


def polygonal_part_of(geom: QgsGeometry | None) -> QgsGeometry | None:







    from .qt_compat import PolygonGeometry

    if geom is None or geom.isEmpty():
        return None
    if geom.type() == PolygonGeometry:
        return geom
    parts = []
    pending = list(reversed(geom.asGeometryCollection()))
    while pending:
        part = pending.pop()
        if part is None or part.isEmpty():
            continue
        if part.type() == PolygonGeometry:
            parts.append(part)
        elif part.isMultipart():
            pending.extend(reversed(part.asGeometryCollection()))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    collected = QgsGeometry.collectGeometry(parts)
    if collected is None or collected.isEmpty():
        return parts[0]
    return collected
