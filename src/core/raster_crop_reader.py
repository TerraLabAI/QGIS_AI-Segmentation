








from __future__ import annotations

import math
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


def _normalize_to_uint8(bands, nodata_value=None, valid_mask=None):













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






    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool).copy()
        if valid_mask.shape != bands.shape[1:]:
            raise ValueError("raster validity mask does not match the colour grid")
    if nodata_value is not None:
        nodata_valid = np.zeros(bands.shape[1:], dtype=bool)
        for b in range(num_bands):
            nodata_valid |= (bands[b] != nodata_value)
        valid_mask = nodata_valid if valid_mask is None else (valid_mask & nodata_valid)




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
        if is_uint8 and num_bands == 3:



            result[b] = band
            continue



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


_GDAL_NUMPY_TYPES = {
    "Byte": np.uint8, "Int8": np.int8, "UInt16": np.uint16, "Int16": np.int16,
    "UInt32": np.uint32, "Int32": np.int32, "UInt64": np.uint64,
    "Int64": np.int64, "Float32": np.float32, "Float64": np.float64,
}


def _band_array(band, xoff, yoff, xsize, ysize, buf_xsize, buf_ysize, **kwargs):







    try:
        return band.ReadAsArray(xoff, yoff, xsize, ysize,
                                buf_xsize=buf_xsize, buf_ysize=buf_ysize, **kwargs)
    except ImportError:
        pass
    from osgeo import gdal

    dtype = _GDAL_NUMPY_TYPES.get(gdal.GetDataTypeName(band.DataType))
    buf_type = band.DataType
    if dtype is None:
        dtype, buf_type = np.float32, gdal.GDT_Float32
    raw = band.ReadRaster(xoff, yoff, xsize, ysize, buf_xsize, buf_ysize,
                          buf_type, **kwargs)
    if raw is None:
        return None
    return np.frombuffer(raw, dtype=dtype).reshape(buf_ysize, buf_xsize).copy()


def _read_palette_rgb_gdal(ds, col_off, row_off, actual_w, actual_h, out_w, out_h):



    try:
        from osgeo import gdal

        band1 = ds.GetRasterBand(1)
        ctable = band1.GetColorTable()
        if ctable is None or band1.GetColorInterpretation() != gdal.GCI_PaletteIndex:
            return None
        idx = _band_array(band1, col_off, row_off, actual_w, actual_h, out_w, out_h)
        if idx is None:
            return None
        cmap = {ci: ctable.GetColorEntry(ci) for ci in range(ctable.GetCount())}
        return _apply_colormap(idx, cmap)
    except Exception:  # noqa: BLE001
        return None


def _declared_rgb_bands_rasterio(src):

    try:
        from rasterio.enums import ColorInterp

        return [src.colorinterp.index(colour) + 1 for colour in
                (ColorInterp.red, ColorInterp.green, ColorInterp.blue)]
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return None


def _declared_rgb_bands_gdal(ds):

    try:
        from osgeo import gdal

        colours = (gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand)
        found = {}
        for index in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(index)
            colour = band.GetColorInterpretation()
            if colour in colours:
                found.setdefault(colour, index)
            if len(found) == 3:
                return [found[colour] for colour in colours]
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return None


def _declared_gray_bands_rasterio(src):

    from rasterio.enums import ColorInterp

    roles = src.colorinterp
    if len(roles) == 2 and set(roles) == {ColorInterp.gray, ColorInterp.alpha}:
        return [roles.index(ColorInterp.gray) + 1]
    return None


def _declared_gray_bands_gdal(ds):
    from osgeo import gdal

    if ds.RasterCount == 2:
        roles = [ds.GetRasterBand(i).GetColorInterpretation() for i in (1, 2)]
        if set(roles) == {gdal.GCI_GrayIndex, gdal.GCI_AlphaBand}:
            return [roles.index(gdal.GCI_GrayIndex) + 1]
    return None


def _window_validity_gdal(ds, band_index, xoff, yoff, width, height,
                          out_w, out_h, palette=False):

    from osgeo import gdal

    band = ds.GetRasterBand(band_index)
    flags = band.GetMaskFlags()
    mask = None
    if not flags & gdal.GMF_ALL_VALID and (palette or not flags & gdal.GMF_NODATA):
        mask = band.GetMaskBand()
    else:
        for index in range(1, ds.RasterCount + 1):
            candidate = ds.GetRasterBand(index)
            if candidate.GetColorInterpretation() == gdal.GCI_AlphaBand:
                mask = candidate
                break
    if mask is None:
        return None
    values = _band_array(mask, xoff, yoff, width, height, out_w, out_h)
    if values is None:
        raise ValueError("Cannot read the raster validity mask")
    return values > 0


def _window_validity_rasterio(src, band_index, window, out_h, out_w, palette=False):
    from rasterio.enums import ColorInterp, MaskFlags, Resampling

    flags = src.mask_flag_enums[band_index - 1]
    if MaskFlags.all_valid not in flags and (palette or MaskFlags.nodata not in flags):
        values = src.read_masks(band_index, window=window, out_shape=(out_h, out_w),
                                resampling=Resampling.nearest)
    elif ColorInterp.alpha in src.colorinterp:
        index = src.colorinterp.index(ColorInterp.alpha) + 1
        values = src.read(index, window=window, out_shape=(out_h, out_w),
                          resampling=Resampling.nearest)
    else:
        return None
    return values > 0


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


def _window_center_on_raster(col_center, row_center, read_cols, read_rows,
                             raster_width, raster_height):










    values = (col_center, row_center, read_cols, read_rows)
    if not all(math.isfinite(v) for v in values):
        return None
    half_cols = max(read_cols, 1) / 2.0
    half_rows = max(read_rows, 1) / 2.0
    if (col_center + half_cols <= 0 or col_center - half_cols >= raster_width
            or row_center + half_rows <= 0 or row_center - half_rows >= raster_height):
        return None
    return (min(max(col_center, 0.0), raster_width - 1.0),
            min(max(row_center, 0.0), raster_height - 1.0))


def _is_pixel_coordinate_frame(transform, height):






    return transform is None or tuple(transform) in (
        (0, 1, 0, 0, 0, 1),
        (0, 1, 0, height, 0, -1),
    )


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

        use_layer_extent = bool(layer_extent) and _is_pixel_coordinate_frame(
            gt, raster_height)




        if not use_layer_extent and gt is not None and gt != (0, 1, 0, 0, 0, 1) and (
                gt[2] != 0 or gt[4] != 0 or gt[1] < 0 or gt[5] > 0):



            rgb_bands = _declared_rgb_bands_gdal(ds)
            if rgb_bands is not None and (
                    ds.RasterCount != 3 or rgb_bands != [1, 2, 3]):


                source_mask_flags = ds.GetRasterBand(rgb_bands[0]).GetMaskFlags()
                mask_kwargs = {}
                if not source_mask_flags & (gdal.GMF_ALL_VALID | gdal.GMF_NODATA):
                    mask_kwargs["maskBand"] = f"mask,{rgb_bands[0]}"
                ds = gdal.Translate("", ds, format="VRT", bandList=rgb_bands, **mask_kwargs)
                if ds is None:
                    raise ValueError("Cannot select the raster colour bands")
            first_band = ds.GetRasterBand(1)
            resampling = (gdal.GRA_NearestNeighbour
                          if first_band.GetColorInterpretation() == gdal.GCI_PaletteIndex
                          else gdal.GRA_Bilinear)
            source_step_x = math.hypot(gt[1], gt[4])
            source_step_y = math.hypot(gt[2], gt[5])
            step = min(source_step_x, source_step_y)
            corners = [
                (gt[0] + col * gt[1] + row * gt[2],
                 gt[3] + col * gt[4] + row * gt[5])
                for col in (0, raster_width) for row in (0, raster_height)
            ]
            xs, ys = zip(*corners)
            warp_kwargs = {}
            if rgb_bands is not None:



                warp_kwargs = {
                    "warpOptions": ["UNIFIED_SRC_NODATA=YES", "INIT_DEST=0"],
                    "dstNodata": "None",
                }
            if not first_band.GetMaskFlags() & (gdal.GMF_ALL_VALID | gdal.GMF_NODATA):
                warp_kwargs["dstAlpha"] = True
            ds = gdal.Warp(
                "", ds, format="VRT", resampleAlg=resampling,
                outputBounds=(min(xs), min(ys), max(xs), max(ys)),
                xRes=step, yRes=step, **warp_kwargs)
            if ds is None:
                raise ValueError("Cannot orient the raster crop")
            raster_width, raster_height = ds.RasterXSize, ds.RasterYSize
            gt = ds.GetGeoTransform()



            scale_factor *= source_step_x / gt[1]

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



        row_ratio = _ground_square_row_ratio(
            ground_aspect, pixel_size_x, pixel_size_y)
        read_cols = int(crop_size * scale_factor)
        read_rows = (read_cols if row_ratio == 1.0
                     else max(1, int(round(read_cols * row_ratio))))
        on_raster = _window_center_on_raster(
            col_center, row_center, read_cols, read_rows,
            raster_width, raster_height)
        if on_raster is None:
            return None, None, "Click is outside the raster bounds", "crop_error_outside_bounds"
        col_center, row_center = on_raster


        col_off = min(max(0, int(round(col_center - read_cols // 2))),
                      max(0, raster_width - read_cols))
        row_off = min(max(0, int(round(row_center - read_rows // 2))),
                      max(0, raster_height - read_rows))

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
            validity = _window_validity_gdal(
                ds, 1, col_off, row_off, actual_width, actual_height, out_w, out_h, palette=True)
            if validity is not None:
                image_np[~validity] = 0
            del ds
        else:







            read_kwargs = {}
            if out_w != actual_width or out_h != actual_height:
                bilinear = getattr(gdal, "GRIORA_Bilinear", None)
                if bilinear is not None:
                    read_kwargs["resample_alg"] = bilinear
            rgb_bands = _declared_rgb_bands_gdal(ds)
            read_bands = rgb_bands or _declared_gray_bands_gdal(ds) or list(range(1, num_bands + 1))
            bands = []
            for b_idx in read_bands:
                band = ds.GetRasterBand(b_idx)
                data = _band_array(
                    band, col_off, row_off, actual_width, actual_height,
                    out_w, out_h, **read_kwargs
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

            nodata = ds.GetRasterBand(read_bands[0]).GetNoDataValue()
            validity = _window_validity_gdal(
                ds, read_bands[0], col_off, row_off, actual_width, actual_height, out_w, out_h)




            del band, ds

            tile_data = np.stack(bands, axis=0)
            image_np = _normalize_to_uint8(
                tile_data, nodata_value=nodata, valid_mask=validity)

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
        ).format(ext=ext, error=str(e) or type(e).__name__), "crop_error_read_failed"

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


            use_layer_extent = bool(layer_extent) and _is_pixel_coordinate_frame(
                raster_transform.to_gdal(), raster_height)

            if not use_layer_extent and not raster_transform.is_identity and (
                    raster_transform.b != 0 or raster_transform.d != 0
                    or raster_transform.a < 0 or raster_transform.e > 0):


                return _read_crop_with_gdal(
                    raster_path, center_x, center_y, crop_size,
                    scale_factor, layer_extent, ground_aspect)

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




            row_ratio = _ground_square_row_ratio(
                ground_aspect, pixel_size_x, pixel_size_y)
            read_cols = int(crop_size * scale_factor)
            read_rows = (read_cols if row_ratio == 1.0
                         else max(1, int(round(read_cols * row_ratio))))
            on_raster = _window_center_on_raster(
                col_center, row_center, read_cols, read_rows,
                raster_width, raster_height)
            if on_raster is None:
                return None, None, "Click is outside the raster bounds", "crop_error_outside_bounds"
            col_center, row_center = on_raster


            col_off = min(max(0, int(round(col_center - read_cols // 2))),
                          max(0, raster_width - read_cols))
            row_off = min(max(0, int(round(row_center - read_rows // 2))),
                          max(0, raster_height - read_rows))

            actual_width = min(read_cols, raster_width - col_off)
            actual_height = min(read_rows, raster_height - row_off)

            if actual_width <= 0 or actual_height <= 0:
                return None, None, "Click is outside the raster bounds", "crop_error_outside_bounds"

            window = Window(col_off, row_off, actual_width, actual_height)




            rgb_bands = _declared_rgb_bands_rasterio(src)
            read_bands = (rgb_bands or _declared_gray_bands_rasterio(src)
                          or list(range(1, min(max(int(src.count), 1), 3) + 1)))
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
                validity = _window_validity_rasterio(src, 1, window, out_h, out_w, palette=True)
                if validity is not None:
                    image_np[~validity] = 0
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
                nodata = (src.nodatavals[read_bands[0] - 1]
                          if rgb_bands is not None else src.nodata)
                validity = _window_validity_rasterio(src, read_bands[0], window, out_h, out_w)
                image_np = _normalize_to_uint8(
                    tile_data, nodata_value=nodata, valid_mask=validity)




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
