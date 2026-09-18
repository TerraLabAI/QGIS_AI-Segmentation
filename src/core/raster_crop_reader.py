








from __future__ import annotations

import os
import re

import numpy as np
from qgis.core import Qgis, QgsMessageLog

from .i18n import tr



_GDAL_ONLY_FORMATS = {
    ".ecw", ".sid", ".jp2", ".j2k", ".j2c",
    ".nitf", ".ntf", ".img", ".hdf", ".hdf5", ".he5", ".nc",
    ".gpkg",
    ".pdf",
}


def _normalize_to_uint8(bands, nodata_value=None):












    if bands.ndim != 3 or bands.shape[0] == 0:
        raise ValueError("raster bands must have shape (channels, height, width)")
    num_bands = bands.shape[0]
    if num_bands == 1:
        band_order = (0, 0, 0)
    elif num_bands == 2:
        band_order = (0, 1, 0)
    else:
        band_order = (0, 1, 2)

    is_float = np.issubdtype(bands.dtype, np.floating)






    if nodata_value is not None:
        valid_mask = np.zeros(bands.shape[1:], dtype=bool)
        for b in range(num_bands):
            valid_mask |= (bands[b] != nodata_value)
    else:
        valid_mask = None




    if is_float:
        for b in sorted(set(band_order)):
            finite = np.isfinite(bands[b])
            valid_mask = finite if valid_mask is None else (valid_mask & finite)

    is_uint8 = (bands.dtype == np.uint8)
    result = np.zeros((3, bands.shape[1], bands.shape[2]), dtype=np.uint8)




    done: dict[int, int] = {}

    for b in range(3):
        src = band_order[b]
        if src in done:
            result[b] = result[done[src]]
            continue
        done[src] = b

        band = bands[src]



        if is_float:
            band = band.astype(np.float64, copy=False)
        valid_pixels = band if valid_mask is None else band[valid_mask]

        if valid_pixels.size == 0:
            continue

        p2, p98 = _percentiles_2_98(valid_pixels)

        if is_uint8:

            if p98 - p2 < 220:
                if p98 > p2:
                    result[b] = _stretch_lookup(p2, p98, 255)[band]
                else:
                    result[b] = band
            else:
                result[b] = band
        else:
            if (band.dtype.kind == "u" and band.dtype.itemsize <= 2
                    and p98 > p2):


                result[b] = _stretch_lookup(p2, p98, int(band.max()))[band]
                continue
            if p98 > p2:
                stretched = np.clip(
                    band if is_float else band.astype(np.float64), p2, p98)
                stretched -= p2
                stretched /= p98 - p2
                stretched *= 255
                if is_float:






                    stretched = np.nan_to_num(
                        stretched, copy=False, nan=0.0, posinf=255.0, neginf=0.0)
                result[b] = stretched.astype(np.uint8)
            else:





                result[b] = 128


    if valid_mask is not None:
        nodata_mask = ~valid_mask
        if np.any(nodata_mask):
            for b in range(3):
                result[b][nodata_mask] = 0





    return np.ascontiguousarray(np.transpose(result, (1, 2, 0)))


def _stretch_lookup(p2, p98, max_value):





    values = np.clip(np.arange(max_value + 1, dtype=np.float64), p2, p98)
    return ((values - p2) / (p98 - p2) * 255).astype(np.uint8)





_HISTOGRAM_PERCENTILE_MAX_RANGE = 65536


def _percentiles_2_98(values):











    try:
        if (values.dtype.kind != "u"
                or values.size == 0
                or int(values.max()) >= _HISTOGRAM_PERCENTILE_MAX_RANGE):
            return np.percentile(values, [2, 98])
        counts = np.bincount(values.ravel())
    except (ValueError, TypeError, MemoryError):
        return np.percentile(values, [2, 98])

    total = int(values.size)
    if total <= 0:
        return np.percentile(values, [2, 98])
    cumulative = np.cumsum(counts)
    out = []
    for q in (2.0, 98.0):


        h = (total - 1) * q / 100.0
        low = int(np.floor(h))
        high = min(low + 1, total - 1)
        value_low = int(np.searchsorted(cumulative, low + 1))
        value_high = int(np.searchsorted(cumulative, high + 1))
        out.append(value_low + (h - low) * (value_high - value_low))
    return np.array(out, dtype=np.float64)


def _apply_colormap(indices, colormap):
















    idx = np.asarray(indices)
    if (idx.ndim != 2 or idx.size == 0 or not colormap
            or idx.dtype.kind not in "iu"):
        return None
    palette = {int(i): color for i, color in colormap.items()
               if int(i) >= 0 and color is not None and len(color) >= 3}
    max_i = min(max(palette, default=0), max(0, int(idx.max())))
    if max_i + 1 > idx.size + len(palette):

        keys = np.array(sorted(i for i in palette if i <= int(idx.max())),
                        dtype=idx.dtype)
        result = np.zeros(idx.shape + (3,), dtype=np.uint8)
        if keys.size == 0:
            return result
        colors = np.array([[int(v) & 0xFF for v in palette[int(i)][:3]]
                           for i in keys], dtype=np.uint8)
        positions = np.searchsorted(keys, idx)
        np.minimum(positions, keys.size - 1, out=positions)
        valid = keys[positions] == idx
        result[valid] = colors[positions[valid]]
        return result
    lut = np.zeros((max_i + 1, 3), dtype=np.uint8)
    for i, color in palette.items():
        if i <= max_i:
            lut[i] = [int(v) & 0xFF for v in color[:3]]
    if int(idx.min()) >= 0 and int(idx.max()) <= max_i:
        return lut[idx]
    valid = (idx >= 0) & (idx <= max_i)
    result = np.zeros(idx.shape + (3,), dtype=np.uint8)
    result[valid] = lut[idx[valid].astype(np.intp)]
    return result


def _rasterio_source_is_paletted(src) -> bool:







    try:
        from rasterio.enums import ColorInterp

        interp = src.colorinterp
        if not interp or interp[0] != ColorInterp.palette:
            return False
        return bool(src.colormap(1))
    except Exception:  # noqa: BLE001
        return False


def _read_palette_rgb_rasterio(src, window, out_h, out_w):




    try:
        from rasterio.enums import ColorInterp, Resampling

        interp = src.colorinterp
        if not interp or interp[0] != ColorInterp.palette:
            return None
        cmap = src.colormap(1)
        if not cmap:
            return None
        idx = src.read(
            1, window=window, out_shape=(out_h, out_w),
            resampling=Resampling.nearest,
        )
        return _apply_colormap(idx, cmap)
    except Exception:  # noqa: BLE001
        return None


def _read_palette_rgb_gdal(ds, col_off, row_off, actual_w, actual_h, out_w, out_h):



    try:
        from osgeo import gdal

        band1 = ds.GetRasterBand(1)
        ctable = band1.GetColorTable()
        if ctable is None or band1.GetColorInterpretation() != gdal.GCI_PaletteIndex:
            return None
        idx = band1.ReadAsArray(
            col_off, row_off, actual_w, actual_h,
            buf_xsize=out_w, buf_ysize=out_h,
        )
        cmap = {ci: ctable.GetColorEntry(ci) for ci in range(ctable.GetCount())}
        return _apply_colormap(idx, cmap)
    except Exception:  # noqa: BLE001
        return None


def _needs_gdal_conversion(raster_path):

    ext = os.path.splitext(raster_path)[1].lower()
    return ext in _GDAL_ONLY_FORMATS




_DRIVER_PREFIX_RE = re.compile(r"^[A-Za-z]{2,}:")


def vsi_normalized_path(raster_path):







    path = (raster_path or "").strip()
    if path.lower().startswith(("http://", "https://")):
        return "/vsicurl/" + path
    return raster_path


def source_file_is_missing(raster_path):








    path = (raster_path or "").strip()
    if not path:
        return False
    if path.startswith("/vsi") or "://" in path or _DRIVER_PREFIX_RE.match(path):
        return False
    return not os.path.exists(path.split("|")[0])


def _gdal_has_thread_local_config(gdal):








    return (hasattr(gdal, "SetThreadLocalConfigOption") and hasattr(gdal, "GetThreadLocalConfigOption"))


def crop_read_is_thread_safe(raster_path):













    try:
        from osgeo import gdal
    except ImportError:

        return True
    return _gdal_has_thread_local_config(gdal)


def _ground_square_row_ratio(ground_aspect: float, pixel_size_x: float,
                             pixel_size_y: float) -> float:











    try:
        aspect = float(ground_aspect)
    except (TypeError, ValueError):
        return 1.0
    if not (aspect > 0.0):
        return 1.0
    if not (pixel_size_x > 0.0 and pixel_size_y > 0.0):
        return 1.0
    ratio = (pixel_size_x / pixel_size_y) / aspect
    if not (0.0 < ratio < 1e6):
        return 1.0
    return ratio


def _ground_matched_out_rows(out_w: int, actual_width: int, actual_height: int,
                             row_ratio: float, crop_size: int) -> int:






    if actual_width <= 0 or actual_height <= 0 or not (row_ratio > 0.0):
        return max(1, int(out_w))
    ground_ratio = (float(actual_height) / float(actual_width)) / row_ratio
    return max(1, min(int(crop_size), int(round(out_w * ground_ratio))))


def _read_crop_with_gdal(raster_path, center_x, center_y, crop_size,
                         scale_factor, layer_extent, ground_aspect=1.0):












    ext = os.path.splitext(raster_path)[1].upper()

    try:
        from osgeo import gdal
    except ImportError:
        return None, None, tr(
            "{ext} format is not directly supported. "
            "GDAL is not available.\n"
            "Please convert your raster to GeoTIFF (.tif) before using "
            "AI Segmentation."
        ).format(ext=ext), "crop_error_gdal_unavailable"

    ds = None











    if _gdal_has_thread_local_config(gdal):
        _set_option = gdal.SetThreadLocalConfigOption
        _get_option = gdal.GetThreadLocalConfigOption
    else:
        _set_option = gdal.SetConfigOption
        _get_option = gdal.GetConfigOption
    _proj_data_backup = _get_option("PROJ_DATA")
    _proj_lib_backup = _get_option("PROJ_LIB")
    _gdal_data_backup = _get_option("GDAL_DATA")
    _set_option("PROJ_DATA", "")
    _set_option("PROJ_LIB", "")
    _set_option("GDAL_DATA", "")
    try:


        from .raster_dataset_cache import acquire_gdal_dataset
        ds = acquire_gdal_dataset(raster_path)
        if ds is None:
            return None, None, tr(
                "Cannot open {ext} file. The format may not be supported "
                "by your QGIS installation.\n"
                "Please convert your raster to GeoTIFF (.tif) before using "
                "AI Segmentation."
            ).format(ext=ext), "crop_error_unsupported_format"

        raster_width = ds.RasterXSize
        raster_height = ds.RasterYSize
        gt = ds.GetGeoTransform()

        use_layer_extent = False
        if layer_extent:
            if gt is None or gt == (0, 1, 0, 0, 0, 1):
                use_layer_extent = True
            else:
                left_near = abs(gt[0]) < 10
                top_near = abs(gt[3]) < 10
                right_near = abs(gt[0] + gt[1] * raster_width) < 10
                bottom_near = abs(gt[3] + gt[5] * raster_height) < 10
                if left_near and top_near and right_near and bottom_near:
                    use_layer_extent = True

        if use_layer_extent and layer_extent:
            xmin_le, ymin_le, xmax_le, ymax_le = layer_extent
            pixel_size_x = (xmax_le - xmin_le) / raster_width
            pixel_size_y = (ymax_le - ymin_le) / raster_height
            bounds_left = xmin_le
            bounds_top = ymax_le
        else:
            pixel_size_x = abs(gt[1])
            pixel_size_y = abs(gt[5])
            bounds_left = gt[0]
            bounds_top = gt[3]

        col_center = (center_x - bounds_left) / pixel_size_x
        row_center = (bounds_top - center_y) / pixel_size_y
        if not (0 <= col_center < raster_width and 0 <= row_center < raster_height):
            return None, None, "Click is outside the raster bounds", "crop_error_outside_bounds"



        row_ratio = _ground_square_row_ratio(
            ground_aspect, pixel_size_x, pixel_size_y)
        read_cols = int(crop_size * scale_factor)
        read_rows = (read_cols if row_ratio == 1.0
                     else max(1, int(round(read_cols * row_ratio))))
        col_off = max(0, int(round(col_center - read_cols // 2)))
        row_off = max(0, int(round(row_center - read_rows // 2)))

        actual_width = min(read_cols, raster_width - col_off)
        actual_height = min(read_rows, raster_height - row_off)

        if actual_width <= 0 or actual_height <= 0:
            return None, None, "Click is outside the raster bounds", "crop_error_outside_bounds"

        num_bands = min(ds.RasterCount, 3)
        if num_bands == 0:
            return None, None, "Raster has no bands", "crop_error_no_bands"

        if scale_factor > 1.0:
            out_h = max(1, min(crop_size, int(actual_height / scale_factor)))
            out_w = max(1, min(crop_size, int(actual_width / scale_factor)))
        elif scale_factor < 1.0:
            out_h = min(crop_size, max(1, int(actual_height / scale_factor)))
            out_w = min(crop_size, max(1, int(actual_width / scale_factor)))
        else:
            out_h = actual_height
            out_w = actual_width




        if row_ratio != 1.0:
            out_h = _ground_matched_out_rows(
                out_w, actual_width, actual_height, row_ratio, crop_size)



        palette_rgb = _read_palette_rgb_gdal(
            ds, col_off, row_off, actual_width, actual_height, out_w, out_h)
        if palette_rgb is not None:
            image_np = palette_rgb
            del ds
        else:







            read_kwargs = {}
            if out_w != actual_width or out_h != actual_height:
                bilinear = getattr(gdal, "GRIORA_Bilinear", None)
                if bilinear is not None:
                    read_kwargs["resample_alg"] = bilinear
            bands = []
            for b_idx in range(1, num_bands + 1):
                band = ds.GetRasterBand(b_idx)
                data = band.ReadAsArray(
                    col_off, row_off, actual_width, actual_height,
                    buf_xsize=out_w, buf_ysize=out_h, **read_kwargs
                )




                if data is None:
                    return None, None, tr(
                        "Could not read pixels from this {ext} file. The file may "
                        "be corrupt, truncated, or use a compression your GDAL "
                        "build cannot decode.\n"
                        "Try opening it in QGIS to confirm it displays, or convert "
                        "it to GeoTIFF (.tif) before using AI Segmentation."
                    ).format(ext=ext), "crop_error_read_failed"
                bands.append(data)

            nodata = ds.GetRasterBand(1).GetNoDataValue()




            del band, ds

            tile_data = np.stack(bands, axis=0)
            image_np = _normalize_to_uint8(tile_data, nodata_value=nodata)

        if out_h < crop_size or out_w < crop_size:
            pad_bottom = crop_size - out_h
            pad_right = crop_size - out_w
            image_np = np.pad(
                image_np,
                ((0, pad_bottom), (0, pad_right), (0, 0)),
                mode="reflect"
            )

        crop_minx = bounds_left + col_off * pixel_size_x
        crop_maxx = bounds_left + (col_off + actual_width) * pixel_size_x
        crop_maxy = bounds_top - row_off * pixel_size_y
        crop_miny = bounds_top - (row_off + actual_height) * pixel_size_y

        crop_info = {
            "bounds": (crop_minx, crop_miny, crop_maxx, crop_maxy),
            "img_shape": (out_h, out_w),
            "col_off": col_off,
            "row_off": row_off,
        }

        QgsMessageLog.logMessage(
            f"Read {ext} crop directly via GDAL: {out_w}x{out_h} at ({col_off}, {row_off})",
            "AI Segmentation", level=Qgis.MessageLevel.Info
        )
        return image_np, crop_info, None, None

    except Exception as e:
        return None, None, tr(
            "Failed to read {ext} file: {error}\n"
            "Please convert your raster to GeoTIFF (.tif) manually."
        ).format(ext=ext, error=str(e)), "crop_error_read_failed"

    finally:


        _set_option("PROJ_DATA", _proj_data_backup)
        _set_option("PROJ_LIB", _proj_lib_backup)
        _set_option("GDAL_DATA", _gdal_data_backup)


def extract_crop_from_raster(raster_path, center_x, center_y, crop_size=1024,
                             layer_crs_wkt=None, layer_extent=None,
                             scale_factor=1.0, ground_aspect=1.0):

























    raster_path = vsi_normalized_path(raster_path)





    if source_file_is_missing(raster_path):
        return (None, None, tr(
            "The raster file could not be found:\n{path}\n\n"
            "It may have been moved or renamed, or the drive or network "
            "share it is on may be disconnected. Reload the layer from "
            "where the file is now, then start again."
        ).format(path=raster_path), "crop_error_file_missing")




    if _needs_gdal_conversion(raster_path):
        return _read_crop_with_gdal(
            raster_path, center_x, center_y, crop_size,
            scale_factor, layer_extent, ground_aspect
        )





    try:
        import rasterio  # noqa: F401
        from rasterio.enums import Resampling
        from rasterio.windows import Window
    except ImportError:





        try:
            from .venv_manager import ensure_venv_packages_available
            ensure_venv_packages_available()
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        try:
            import rasterio  # noqa: F401
            from rasterio.enums import Resampling
            from rasterio.windows import Window
        except ImportError as err:












            QgsMessageLog.logMessage(
                f"rasterio is not available ({str(err)}), "
                f"reading the crop through GDAL instead...",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )
            return _read_crop_with_gdal(
                raster_path, center_x, center_y, crop_size,
                scale_factor, layer_extent, ground_aspect
            )

    try:



        from .raster_dataset_cache import borrow_rasterio_dataset
        with borrow_rasterio_dataset(raster_path) as src:
            raster_width = src.width
            raster_height = src.height
            raster_transform = src.transform


            use_layer_extent = False
            if layer_extent:
                if src.crs is None:
                    use_layer_extent = True
                else:
                    rb = src.bounds
                    left_near = abs(rb.left) < 10
                    bottom_near = abs(rb.bottom) < 10
                    right_near = abs(rb.right - raster_width) < 10
                    top_near = abs(rb.top - raster_height) < 10
                    if left_near and bottom_near and right_near and top_near:
                        use_layer_extent = True

            if use_layer_extent and layer_extent:
                xmin_le, ymin_le, xmax_le, ymax_le = layer_extent
                pixel_size_x = (xmax_le - xmin_le) / raster_width
                pixel_size_y = (ymax_le - ymin_le) / raster_height
                bounds_left = xmin_le
                bounds_top = ymax_le
            else:
                pixel_size_x = abs(raster_transform.a)
                pixel_size_y = abs(raster_transform.e)
                bounds_left = src.bounds.left
                bounds_top = src.bounds.top


            col_center = (center_x - bounds_left) / pixel_size_x
            row_center = (bounds_top - center_y) / pixel_size_y
            if not (0 <= col_center < raster_width and 0 <= row_center < raster_height):
                return None, None, "Click is outside the raster bounds", "crop_error_outside_bounds"




            row_ratio = _ground_square_row_ratio(
                ground_aspect, pixel_size_x, pixel_size_y)
            read_cols = int(crop_size * scale_factor)
            read_rows = (read_cols if row_ratio == 1.0
                         else max(1, int(round(read_cols * row_ratio))))
            col_off = max(0, int(round(col_center - read_cols // 2)))
            row_off = max(0, int(round(row_center - read_rows // 2)))

            actual_width = min(read_cols, raster_width - col_off)
            actual_height = min(read_rows, raster_height - row_off)

            if actual_width <= 0 or actual_height <= 0:
                return None, None, "Click is outside the raster bounds", "crop_error_outside_bounds"

            window = Window(col_off, row_off, actual_width, actual_height)





            read_bands = list(range(1, min(max(int(src.count), 1), 3) + 1))
            n_read = len(read_bands)

            if scale_factor > 1.0:
                out_h = min(crop_size, int(actual_height / scale_factor))
                out_w = min(crop_size, int(actual_width / scale_factor))
                out_h = max(1, out_h)
                out_w = max(1, out_w)
            elif scale_factor < 1.0:
                out_h = min(crop_size, max(1, int(actual_height / scale_factor)))
                out_w = min(crop_size, max(1, int(actual_width / scale_factor)))
            else:
                out_h = actual_height
                out_w = actual_width




            if row_ratio != 1.0:
                out_h = _ground_matched_out_rows(
                    out_w, actual_width, actual_height, row_ratio, crop_size)





            palette_rgb = None
            if _rasterio_source_is_paletted(src):
                palette_rgb = _read_palette_rgb_rasterio(
                    src, window, out_h, out_w)

            if palette_rgb is not None:
                image_np = palette_rgb
            else:
                if (scale_factor == 1.0 and out_h == actual_height
                        and out_w == actual_width):
                    tile_data = src.read(indexes=read_bands, window=window)
                else:
                    tile_data = src.read(
                        indexes=read_bands,
                        window=window,
                        out_shape=(n_read, out_h, out_w),
                        resampling=Resampling.bilinear
                    )
                image_np = _normalize_to_uint8(tile_data, nodata_value=src.nodata)




            if out_h < crop_size or out_w < crop_size:
                pad_bottom = crop_size - out_h
                pad_right = crop_size - out_w
                image_np = np.pad(
                    image_np,
                    ((0, pad_bottom), (0, pad_right), (0, 0)),
                    mode="reflect"
                )


            crop_minx = bounds_left + col_off * pixel_size_x
            crop_maxx = bounds_left + (col_off + actual_width) * pixel_size_x
            crop_maxy = bounds_top - row_off * pixel_size_y
            crop_miny = bounds_top - (row_off + actual_height) * pixel_size_y

            crop_info = {
                "bounds": (crop_minx, crop_miny, crop_maxx, crop_maxy),
                "img_shape": (out_h, out_w),
                "col_off": col_off,
                "row_off": row_off,
            }

            return image_np, crop_info, None, None

    except Exception as e:

        QgsMessageLog.logMessage(
            f"rasterio failed ({str(e)}), trying GDAL fallback...",
            "AI Segmentation", level=Qgis.MessageLevel.Warning
        )
        return _read_crop_with_gdal(
            raster_path, center_x, center_y, crop_size,
            scale_factor, layer_extent, ground_aspect
        )
