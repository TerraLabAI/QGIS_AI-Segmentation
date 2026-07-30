from __future__ import annotations

import os
import re
import time

import numpy as np
from qgis.core import Qgis, QgsMessageLog

from .i18n import tr

# Raster formats that pip-installed rasterio may not support reliably.
# These go through GDAL windowed read instead.
_GDAL_ONLY_FORMATS = {
    ".ecw", ".sid", ".jp2", ".j2k", ".j2c",
    ".nitf", ".ntf", ".img", ".hdf", ".hdf5", ".he5", ".nc",
    ".gpkg",
    ".pdf",
}

# Online/remote raster providers that need rendering before encoding. QGIS
# serves WMTS and XYZ through the "wms" provider, so those two names never
# come back from provider.name(); they are kept because a build or a future
# QGIS could report them, and an extra name here costs nothing.
ONLINE_PROVIDERS = frozenset(["wms", "wmts", "xyz", "arcgismapserver", "wcs"])

# Local providers that serve pixels but have no file behind them, so the
# windowed file read has nothing to open. A PostGIS raster lives in a
# database, a virtual raster is an expression over other layers. Both are
# valid QgsRasterLayers, both are offered in the layer picker, and both used
# to reach the file reader and fail there.
FILELESS_LOCAL_PROVIDERS = frozenset(["postgresraster", "virtualraster"])

# The real question every caller asks: must this layer be rendered through
# QGIS instead of read from its own file? Online or fileless, the answer and
# the code path are the same.
CANVAS_RENDERED_PROVIDERS = ONLINE_PROVIDERS | FILELESS_LOCAL_PROVIDERS


def _normalize_to_uint8(bands, nodata_value=None):
    """Normalize a multi-band array to (H, W, 3) uint8 using per-band percentile stretch.

    Args:
        bands: numpy array of shape (C, H, W), any numeric dtype
        nodata_value: optional nodata value to mask before computing percentiles

    Returns:
        numpy array of shape (H, W, 3) uint8
    """
    # Which source band feeds each output channel. Indexing beats materializing
    # a repeated/stacked copy: a single-band crop no longer triples its own data
    # just to run the same stretch three times.
    num_bands = bands.shape[0]
    if num_bands == 1:
        band_order = (0, 0, 0)
    elif num_bands == 2:
        band_order = (0, 1, 0)
    else:
        band_order = (0, 1, 2)

    is_float = np.issubdtype(bands.dtype, np.floating)

    # Build nodata mask: True where pixel is valid
    if nodata_value is not None:
        valid_mask = bands[0] != nodata_value
    else:
        valid_mask = None

    # Also mask NaN AND the infinities for float types. An inf survives
    # np.isnan, so it used to reach np.percentile, where it drags p2/p98 to
    # inf and turns the whole stretch below into NaN.
    if is_float:
        for b in sorted(set(band_order)):
            finite = np.isfinite(bands[b])
            valid_mask = finite if valid_mask is None else (valid_mask & finite)

    is_uint8 = (bands.dtype == np.uint8)
    result = np.zeros((3, bands.shape[1], bands.shape[2]), dtype=np.uint8)

    # An output channel repeats a source band only when the crop has fewer than
    # 3 bands, and the same source gives the same percentiles and the same
    # stretch, so the work is done once and the row copied.
    done: dict[int, int] = {}

    for b in range(3):
        src = band_order[b]
        if src in done:
            result[b] = result[done[src]]
            continue
        done[src] = b

        band = bands[src]
        # Only cast to float64 where the percentile needs it: numpy already
        # returns float64 percentiles for integer input, so a full-size float64
        # copy of an 8-bit crop buys nothing.
        if is_float:
            band = band.astype(np.float64)
        valid_pixels = band if valid_mask is None else band[valid_mask]

        if valid_pixels.size == 0:
            continue

        p2, p98 = np.percentile(valid_pixels, [2, 98])

        if is_uint8:
            # Only stretch if histogram is compressed
            if p98 - p2 < 220:
                if p98 > p2:
                    stretched = np.clip(band.astype(np.float64), p2, p98)
                    stretched = (stretched - p2) / (p98 - p2) * 255
                    result[b] = stretched.astype(np.uint8)
                else:
                    result[b] = band
            else:
                result[b] = band
        else:
            if p98 > p2:
                stretched = np.clip(
                    band if is_float else band.astype(np.float64), p2, p98)
                stretched = (stretched - p2) / (p98 - p2) * 255
                if is_float:
                    # np.clip propagates NaN, so a nodata pixel excluded from
                    # the percentiles is still NaN here, and NaN -> uint8 is
                    # undefined ("invalid value encountered in cast", which a
                    # numpy error state turns into a hard read failure). The
                    # nodata zeroing below only runs after this cast, so the
                    # scrub has to happen first.
                    stretched = np.nan_to_num(
                        stretched, nan=0.0, posinf=255.0, neginf=0.0)
                result[b] = stretched.astype(np.uint8)
            # else: stays zeros

    # Zero out nodata pixels
    if valid_mask is not None:
        nodata_mask = ~valid_mask
        if np.any(nodata_mask):
            for b in range(3):
                result[b][nodata_mask] = 0

    # CHW -> HWC, returned C-contiguous. tobytes() in the worker handshake is
    # order-correct even on a transpose view, but a contiguous buffer keeps the
    # SAM predictor (which may take .data / buffer views internally) on the
    # well-tested path and avoids a latent footgun for future callers.
    return np.ascontiguousarray(np.transpose(result, (1, 2, 0)))


def _apply_colormap(indices, colormap):
    """Map a (H, W) palette-index array to (H, W, 3) uint8 RGB.

    A paletted raster (land cover, thematic classes) stores class INDICES, not
    colours. Reading the raw index band and stretching it (the default path)
    produces a meaningless grey ramp instead of the coloured classes the user
    sees. Expanding through the colour table restores the real classes.

    Args:
        indices: (H, W) integer array of palette indices.
        colormap: {index: (r, g, b[, a])} mapping (GDAL colour table or
                  rasterio ``src.colormap``).

    Returns:
        (H, W, 3) uint8 RGB, or None when the input cannot be mapped. Indices
        outside the table render black.
    """
    idx = np.asarray(indices)
    if idx.ndim != 2 or idx.size == 0 or not colormap:
        return None
    max_i = int(idx.max(initial=0))
    lut = np.zeros((max_i + 1, 3), dtype=np.uint8)
    for i, color in colormap.items():
        if 0 <= int(i) <= max_i and color is not None and len(color) >= 3:
            lut[int(i), 0] = int(color[0]) & 0xFF
            lut[int(i), 1] = int(color[1]) & 0xFF
            lut[int(i), 2] = int(color[2]) & 0xFF
    flat = np.clip(idx.astype(np.int64), 0, max_i)
    return lut[flat]


def _read_palette_rgb_rasterio(src, window, out_h, out_w):
    """Return (H, W, 3) uint8 for a paletted rasterio source, or None when the
    source is not paletted / has no usable colour table. Reads band 1 with
    NEAREST resampling so class indices are never interpolated (bilinear on
    indices would invent nonexistent classes at edges)."""
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
    except Exception:  # noqa: BLE001 - fall back to raw normalization on any issue
        return None


def _read_palette_rgb_gdal(ds, col_off, row_off, actual_w, actual_h, out_w, out_h):
    """Return (H, W, 3) uint8 for a paletted GDAL dataset, or None when the
    band is not paletted. GDAL windowed reads with buf_x/ysize use nearest
    resampling by default, so class indices are preserved."""
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
    except Exception:  # noqa: BLE001 - fall back to raw normalization on any issue
        return None


def _fetch_online_bands(provider, extent, width, height):
    """Fetch raw band data from an online raster provider.

    Handles ARGB32 formats directly and fetches individual bands for
    all other data types (Byte, UInt16, Int16, UInt32, Int32, Float32, Float64).

    Args:
        provider: QgsRasterDataProvider
        extent: QgsRectangle for the area to fetch
        width: pixel width to request
        height: pixel height to request

    Returns:
        (bands_array, is_argb, error) where:
        - bands_array: numpy array (C, H, W) or (H, W, 3) uint8 if ARGB
        - is_argb: True if result is already RGB uint8 (H, W, 3)
        - error: error string or None
    """
    block = provider.block(1, extent, width, height)
    if block is None or not block.isValid():
        return None, False, "Provider block fetch failed"

    block_w = block.width()
    block_h = block.height()
    if block_w == 0 or block_h == 0:
        return None, False, "Provider returned empty block"

    raw_bytes = block.data()
    if raw_bytes is None or len(raw_bytes) == 0:
        return None, False, "Provider returned empty data"

    raw_data = bytes(raw_bytes)
    dt = block.dataType()

    # ARGB32 formats: direct extraction
    is_argb32 = dt == Qgis.DataType.ARGB32
    is_argb32_pre = dt == Qgis.DataType.ARGB32_Premultiplied
    is_argb = is_argb32 or is_argb32_pre
    if is_argb:
        arr = np.frombuffer(raw_data, dtype=np.uint8).reshape(
            block_h, block_w, 4)
        # Qt BGRA byte order -> RGB. ascontiguousarray already allocates the
        # output, so copying the 4-channel buffer first only to drop its alpha
        # would be a whole extra pass over the crop.
        image_np = np.ascontiguousarray(arr[:, :, 2::-1])
        return image_np, True, None

    # Map Qgis data types to numpy dtypes
    dtype_map = {
        Qgis.DataType.Byte: np.uint8,
        Qgis.DataType.UInt16: np.uint16,
        Qgis.DataType.Int16: np.int16,
        Qgis.DataType.UInt32: np.uint32,
        Qgis.DataType.Int32: np.int32,
        Qgis.DataType.Float32: np.float32,
        Qgis.DataType.Float64: np.float64,
    }

    np_dtype = dtype_map.get(dt)
    if np_dtype is None:
        # Unknown type: try ARGB32 interpretation as fallback
        if len(raw_data) == block_w * block_h * 4:
            arr = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                block_h, block_w, 4)
            image_np = np.ascontiguousarray(arr[:, :, 2::-1])
            return image_np, True, None
        return None, False, f"Unsupported data type: {dt}"

    band_count = min(provider.bandCount(), 3)
    bands = []

    # First band already fetched
    band1 = np.frombuffer(raw_data, dtype=np_dtype).reshape(
        block_h, block_w).copy()
    bands.append(band1)

    # Fetch remaining bands
    for band_idx in range(2, band_count + 1):
        b = provider.block(band_idx, extent, block_w, block_h)
        if b is not None and b.isValid():
            b_data = bytes(b.data())
            if len(b_data) > 0:
                band_arr = np.frombuffer(
                    b_data, dtype=np_dtype
                ).reshape(block_h, block_w).copy()
                bands.append(band_arr)

    bands_array = np.stack(bands, axis=0)
    return bands_array, False, None


# Hard cap on the renderer fallback below. It is reached exactly when a tile
# host is slow, it waits on the calling thread, and that thread is the GUI one,
# so the wait needs its own stop and cannot rely on the provider's.
_RENDER_FALLBACK_TIMEOUT_MS: int = 30000


def _render_layer_to_image(layer, extent, width, height,
                           timeout_ms=_RENDER_FALLBACK_TIMEOUT_MS):
    """Render a layer to an RGB image using QGIS map renderer (fallback).

    Works with any layer type (WMS, WMTS, XYZ, WCS, vector tiles, etc.)

    The render runs as a parallel job and the wait is a local event loop with a
    timer, so a tile host that never answers cannot hold the caller past
    ``timeout_ms``. The loop holds back user input: a click or a window close
    delivered inside this frame would reenter the caller while it waits.

    MAIN THREAD ONLY (QgsMapRendererParallelJob).

    Args:
        layer: QgsMapLayer to render
        extent: QgsRectangle for the area
        width: pixel width
        height: pixel height
        timeout_ms: hard cap on the wait

    Returns:
        (image_np, error) where image_np is (H, W, 3) uint8 or None
    """
    try:
        from qgis.core import QgsMapRendererParallelJob, QgsMapSettings
        from qgis.PyQt.QtCore import QEventLoop, QSize, QTimer
        from qgis.PyQt.QtGui import QColor, QImage

        from .qt_compat import resolve_qt_enum

        settings = QgsMapSettings()
        settings.setOutputSize(QSize(width, height))
        settings.setExtent(extent)
        settings.setLayers([layer])
        settings.setDestinationCrs(layer.crs())
        settings.setBackgroundColor(QColor(0, 0, 0))

        job = QgsMapRendererParallelJob(settings)
        loop = QEventLoop()
        job.finished.connect(loop.quit)
        QTimer.singleShot(int(timeout_ms), loop.quit)
        job.start()
        loop.exec(resolve_qt_enum(
            QEventLoop, "ProcessEventsFlag", "ExcludeUserInputEvents"))

        if job.isActive():
            job.cancelWithoutBlocking()
            return None, f"Renderer fallback timed out after {int(timeout_ms)} ms"

        img = job.renderedImage()
        if img is None or img.isNull():
            return None, "Renderer fallback produced no image"

        # QImage -> numpy, on the image's own size (the job owns the buffer).
        img = img.convertToFormat(QImage.Format.Format_RGB32)
        out_h = img.height()
        out_w = img.width()
        ptr = img.bits()
        ptr.setsize(out_h * out_w * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
            out_h, out_w, 4)
        # BGRA -> RGB. The reversed slice is not contiguous, so this always
        # allocates its own buffer and never keeps the QImage alive.
        image_np = np.ascontiguousarray(arr[:, :, 2::-1])
        return image_np, None

    except Exception as e:
        return None, f"Renderer fallback failed: {str(e)}"


def _needs_gdal_conversion(raster_path):
    """Check if raster format requires GDAL conversion for rasterio."""
    ext = os.path.splitext(raster_path)[1].lower()
    return ext in _GDAL_ONLY_FORMATS


# A GDAL driver prefix: two or more letters then a colon, which rules out a
# Windows drive letter (always one). GPKG:, NETCDF:, PDF:, JPEG:, HDF5: ...
_DRIVER_PREFIX_RE = re.compile(r"^[A-Za-z]{2,}:")


def vsi_normalized_path(raster_path):
    """A source both readers can open: a bare http(s) URL gains /vsicurl/.

    QGIS normally stores a remote COG already prefixed, but a layer built from
    a plain URL (a hand-written project file, a script, an MCP call) keeps the
    bare form, which rasterio and GDAL both reject as a missing filename.
    Anything else is returned untouched.
    """
    path = (raster_path or "").strip()
    if path.lower().startswith(("http://", "https://")):
        return "/vsicurl/" + path
    return raster_path


def source_file_is_missing(raster_path):
    """True only when the source is a plain path with nothing at the end of it.

    Answered for plain filesystem paths and nothing else. A GDAL URI
    (``/vsizip/``, ``GPKG:...:layer``, ``NETCDF:"...":var``, an http source)
    names a dataset inside a container, so os.path.exists says nothing useful
    about it and those go straight to the normal reader, which reports its own
    error. A False answer here only means the usual path runs.
    """
    path = (raster_path or "").strip()
    if not path:
        return False
    if path.startswith("/vsi") or "://" in path or _DRIVER_PREFIX_RE.match(path):
        return False
    return not os.path.exists(path.split("|")[0])


def _gdal_has_thread_local_config(gdal):
    """True when this GDAL binding can scope a config option to one thread.

    ``SetConfigOption`` writes the PROCESS-global table, so a value shadowed
    there is missing for every other thread until it is put back. The
    thread-local pair keeps the shadow on the calling thread. Both names are
    checked: an old binding that exposes only one of them cannot do the
    round trip.
    """
    return (hasattr(gdal, "SetThreadLocalConfigOption") and hasattr(gdal, "GetThreadLocalConfigOption"))


def crop_read_is_thread_safe(raster_path):
    """True when this raster's windowed read may run off the GUI thread.

    The GDAL path blanks PROJ's and GDAL's data paths while it reads (see
    _read_crop_with_gdal). Thread-local config options keep that shadow on the
    reading thread. Without them the shadow is process-wide, so a canvas
    render would look for proj.db while it is blanked and come back
    unprojected or empty: such a read stays on the GUI thread instead.
    """
    if not _needs_gdal_conversion(raster_path):
        return True
    try:
        from osgeo import gdal
    except ImportError:
        # No GDAL, so nothing is shadowed: the read returns its own error.
        return True
    return _gdal_has_thread_local_config(gdal)


def _ground_square_row_ratio(ground_aspect: float, pixel_size_x: float,
                             pixel_size_y: float) -> float:
    """Rows per column that make a read window square on the GROUND.

    ``ground_aspect`` is the ground distance one y unit of the raster CRS
    covers divided by the ground distance one x unit covers, at the crop
    centre. It is above 1 in a geographic CRS, where a degree of longitude
    covers less ground than a degree of latitude, so a ground-square window
    there needs more columns than rows and this comes back below 1.

    Exactly 1.0 whenever nothing needs correcting: on an aspect of 1.0 (every
    projected CRS), on an unusable aspect, and on a pixel size that cannot be
    divided. A window built on 1.0 is the square-in-coordinates window this
    reader has always used.
    """
    try:
        aspect = float(ground_aspect)
    except (TypeError, ValueError):
        return 1.0
    if aspect == 1.0 or not (aspect > 0.0):
        return 1.0
    if not (pixel_size_x > 0.0 and pixel_size_y > 0.0):
        return 1.0
    ratio = (pixel_size_x / pixel_size_y) / aspect
    if not (0.0 < ratio < 1e6):
        return 1.0
    return ratio


def _ground_matched_out_rows(out_w: int, actual_width: int, actual_height: int,
                             row_ratio: float, crop_size: int) -> int:
    """Output rows whose proportions match the ground the window covers.

    Measured on the pixels actually read rather than on the pixels requested,
    so a crop clipped at the raster edge still reaches the model in the
    proportions of the ground behind it.
    """
    if actual_width <= 0 or actual_height <= 0 or not (row_ratio > 0.0):
        return max(1, int(out_w))
    ground_ratio = (float(actual_height) / float(actual_width)) / row_ratio
    return max(1, min(int(crop_size), int(round(out_w * ground_ratio))))


def _read_crop_with_gdal(raster_path, center_x, center_y, crop_size,
                         scale_factor, layer_extent, ground_aspect=1.0):
    """Read a windowed crop directly from a GDAL-supported raster (JP2, ECW, etc.).

    Instead of converting the entire file to GeoTIFF, reads only the needed
    pixels using GDAL windowed access.

    ``ground_aspect`` is measured by the caller (see extract_crop_from_raster);
    1.0 reads the window exactly as this function always has.

    Returns (image_np, crop_info, error_msg, error_code). On success, error_msg
    and error_code are both None. On failure error_code is a stable English
    snake_case identifier safe for analytics aggregation across locales.
    """
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
    # rasterio bundles its own proj.db in site-packages. If anything in the
    # process points PROJ_DATA/GDAL_DATA at it, GDAL (from osgeo) hits a
    # "DATABASE.LAYOUT.VERSION.MINOR = N whereas a number >= M is expected"
    # error when forced to read rasterio's older proj.db. Shadow the vars via
    # GDAL's THREAD-LOCAL config table instead of os.environ, so the shadow
    # lasts only for this read AND only on this thread: the plain setter is
    # process-global, and this read runs on a worker while the user keeps
    # panning, so a global blank would leave the canvas render looking for a
    # proj.db that is not there. A binding without the thread-local pair falls
    # back to the global one, which is why crop_read_is_thread_safe keeps that
    # case on the GUI thread with nothing else running.
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
        ds = gdal.Open(raster_path)
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
            xmin_le, ymin_le, xmax_le, ymax_le = layer_extent
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

        # Square on the ground, not square in coordinates. read_rows equals
        # read_cols on every CRS whose two axes cover the same ground distance,
        # and drops below it on one where a y unit covers more ground than an
        # x unit.
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

        # The image the model sees carries the proportions of the ground under
        # it. Only the rows move: every column the file holds is kept, and the
        # rows it never held are spread, not invented.
        if row_ratio != 1.0:
            out_h = _ground_matched_out_rows(
                out_w, actual_width, actual_height, row_ratio, crop_size)

        # Paletted (colour-table) raster: expand class indices to their true
        # colours instead of stretching raw indices to grey.
        palette_rgb = _read_palette_rgb_gdal(
            ds, col_off, row_off, actual_width, actual_height, out_w, out_h)
        if palette_rgb is not None:
            image_np = palette_rgb
            ds = None
        else:
            # A buffer smaller than the read window (the geographic-CRS ground
            # square, see ground_aspect above) makes GDAL resample on read.
            # With no resample_alg that resample is nearest neighbour, blocky
            # on the stretched axis. Bilinear matches the rasterio path above.
            # Resolved defensively (not a bare attribute access) and only
            # passed when the buffer actually differs from the window, so a
            # projected raster's 1:1 read stays byte-identical to before.
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
                # GDAL returns None (not an exception) when the block cannot be
                # decoded: corrupt tile, missing codec (JPEG-in-TIFF), broken
                # overview. np.stack on a None would raise a cryptic error that
                # surfaces as crop_error_read_failed. Catch it here with guidance.
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
            # Release the Band before the Dataset. A live Band holds its parent
            # Dataset open, so `ds = None` on its own left the user's raster
            # open through the stack and normalise below, and Windows refuses
            # to overwrite or delete a file it still has a handle on.
            band = None
            ds = None

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
        ds = None
        _set_option("PROJ_DATA", _proj_data_backup)
        _set_option("PROJ_LIB", _proj_lib_backup)
        _set_option("GDAL_DATA", _gdal_data_backup)


def extract_crop_from_raster(raster_path, center_x, center_y, crop_size=1024,
                             layer_crs_wkt=None, layer_extent=None,
                             scale_factor=1.0, ground_aspect=1.0):
    """Extract a crop_size x crop_size RGB crop centered on (center_x, center_y).

    Args:
        raster_path: Path to the raster file
        center_x, center_y: Center of crop in geo/pixel coordinates
        crop_size: Size of the crop in pixels (default 1024)
        layer_crs_wkt: Optional CRS WKT for non-georeferenced rasters
        layer_extent: Optional (xmin, ymin, xmax, ymax) for non-georeferenced rasters
        scale_factor: Controls native read size relative to crop_size.
            When > 1.0, reads more native pixels and downsamples to crop_size.
            When < 1.0, reads fewer native pixels and upsamples to crop_size.
            Clamped to [0.25, 8.0] by the caller.
        ground_aspect: Ground distance one y unit of the raster CRS covers,
            divided by the ground distance one x unit covers, at the crop
            centre. 1.0 (the default) means the two axes agree and the crop is
            read exactly as it always was. The CALLER measures it, with
            layer_conventions.ground_unit_aspect: that measurement is
            main-thread only and this read is not.

    Returns:
        (image_np, crop_info, error) where:
        - image_np: (H, W, 3) uint8 numpy array
        - crop_info: dict with 'bounds' (minx, miny, maxx, maxy) and 'img_shape' (H, W)
        - error: error string or None on success
    """
    raster_path = vsi_normalized_path(raster_path)

    # A file that is not there is not a format problem. Both readers below end
    # on "convert your raster to GeoTIFF", which is useless advice for a .tif
    # whose drive or network share went away, and it sent users converting a
    # file they could no longer open.
    if source_file_is_missing(raster_path):
        return (None, None, tr(
            "The raster file could not be found:\n{path}\n\n"
            "It may have been moved or renamed, or the drive or network "
            "share it is on may be disconnected. Reload the layer from "
            "where the file is now, then start again."
        ).format(path=raster_path), "crop_error_file_missing")

    # Handle GDAL-only formats (JP2, ECW, etc.) with direct windowed read.
    # Checked BEFORE the rasterio import: this path reads through QGIS's own
    # GDAL, so it must keep working when the venv rasterio is broken/missing.
    if _needs_gdal_conversion(raster_path):
        return _read_crop_with_gdal(
            raster_path, center_x, center_y, crop_size,
            scale_factor, layer_extent, ground_aspect
        )

    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import Window
    except ImportError:
        # Nothing on this path puts the venv on sys.path. The calls that do it
        # live in the startup diagnostics workers, so a session where those
        # never ran, or ran after the first click, reaches this import with
        # rasterio installed and still unimportable ("No module named
        # 'rasterio'"). Add the path and retry before calling it missing.
        try:
            from .venv_manager import ensure_venv_packages_available
            ensure_venv_packages_available()
        except Exception:  # noqa: BLE001 - the retry below reports the real cause
            pass  # nosec B110
        try:
            import rasterio
            from rasterio.enums import Resampling
            from rasterio.windows import Window
        except ImportError as err:
            # Keep the real reason. A missing package and a package whose native
            # extension will not load both land here, and only the second one
            # says why (a dlopen failure, an ABI mismatch against the host numpy).
            # The caller's automatic repair only helps the first, so discarding
            # the text left every one of these undiagnosable. Telemetry scrubs
            # paths out of the message before it leaves the machine.
            return (None, None, f"rasterio is not available: {err}",
                    "crop_error_rasterio_unavailable")

    try:
        with rasterio.open(raster_path) as src:
            raster_width = src.width
            raster_height = src.height
            raster_transform = src.transform

            # Detect non-georeferenced mode
            use_layer_extent = False
            if layer_extent:
                xmin_le, ymin_le, xmax_le, ymax_le = layer_extent
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

            # Convert geo coords to pixel coords
            col_center = (center_x - bounds_left) / pixel_size_x
            row_center = (bounds_top - center_y) / pixel_size_y

            # When scale_factor > 1, read a larger native window. Square on the
            # ground, not square in coordinates: read_rows equals read_cols on
            # every CRS whose two axes cover the same ground distance, and
            # drops below it on one where a y unit covers more ground than an
            # x unit.
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

            # Read at most the first 3 bands: the normalizer drops everything
            # past band 3 anyway, so decoding a 12-band stack in full would
            # allocate several times the crop for nothing. Mirrors the GDAL
            # path, which has always read min(band count, 3).
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

            # The image the model sees carries the proportions of the ground
            # under it. Only the rows move: every column the file holds is
            # kept, and the rows it never held are spread, not invented.
            if row_ratio != 1.0:
                out_h = _ground_matched_out_rows(
                    out_w, actual_width, actual_height, row_ratio, crop_size)

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

            nodata = src.nodata
            # Paletted (colour-table) raster: expand class indices to their true
            # colours (read band 1 with nearest so indices are never blended).
            palette_rgb = _read_palette_rgb_rasterio(src, window, out_h, out_w)
            if palette_rgb is not None:
                image_np = palette_rgb
            else:
                image_np = _normalize_to_uint8(tile_data, nodata_value=nodata)

            # Pad to full crop_size if crop was clipped at raster edge.
            # Uses reflect padding instead of black borders for better
            # SAM context at image boundaries.
            if out_h < crop_size or out_w < crop_size:
                pad_bottom = crop_size - out_h
                pad_right = crop_size - out_w
                image_np = np.pad(
                    image_np,
                    ((0, pad_bottom), (0, pad_right), (0, 0)),
                    mode="reflect"
                )

            # Compute geo bounds for this crop (covers the full read area)
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
        # Fallback to GDAL if rasterio fails (unsupported driver, etc.)
        QgsMessageLog.logMessage(
            f"rasterio failed ({str(e)}), trying GDAL fallback...",
            "AI Segmentation", level=Qgis.MessageLevel.Warning
        )
        return _read_crop_with_gdal(
            raster_path, center_x, center_y, crop_size,
            scale_factor, layer_extent, ground_aspect
        )


class OnlineCropFetcher:
    """Step-driven online-tile crop fetch.

    One provider block read + stabilization check is one ``step()``. The
    caller drives the retry back-off between steps: the interactive path
    schedules them with QTimer so the waits never block the GUI thread, while
    extract_crop_from_online_layer drives the same steps in a plain blocking
    loop for the headless/recovery contract. The retry policy (attempt count,
    progressive delays, blank-tile handling) lives here once, so both paths
    behave identically.

    The provider is the user's LIVE layer provider. The resampling state the
    fetch forces (provider resampling on, bilinear zoomed-in/out) is
    snapshotted by begin() and reverted by restore(); the caller MUST call
    restore() on every exit so the user's layer is never left mutated.
    """

    _MAX_RETRIES = 8
    _RETRY_DELAY = 1.0
    # Ceiling on the time one fetch may spend INSIDE provider.block(). On an
    # online provider that call is a synchronous network read on the calling
    # thread, and it carries the network stack's own timeout, so the retry
    # ladder would otherwise let one unresponsive host hold that thread for
    # minutes. Only the reads are charged here: the back-off waits between them
    # belong to the caller, and the interactive path takes them off the event
    # loop. A read already running cannot be interrupted, so the real bound is
    # this budget plus the one read in flight when it runs out.
    _READ_BUDGET_S = 8.0

    def __init__(self, layer, center_x, center_y, canvas_mupp, crop_size=1024):
        from qgis.core import QgsRectangle

        from .layer_conventions import ground_unit_aspect

        self._layer = layer
        self._crop_size = crop_size
        self.error = None
        self.error_code = None
        self._mutated = False
        self._orig_in = None
        self._orig_out = None
        self._orig_enabled = None
        self._attempt = 0
        self._prev_data = None
        self._reload_pending = False
        self._read_seconds = 0.0

        provider = layer.dataProvider()
        self._provider = provider
        if provider is None:
            self.error = tr("Layer data provider is not available.")
            self.error_code = "crop_error_online_provider_unavailable"
            return

        # The crop covers a square piece of GROUND, then fills a square pixel
        # image, so the two agree. canvas_mupp is measured along x, and one y
        # unit of a geographic CRS covers more ground than one x unit, so the
        # rectangle is shorter in y by exactly that factor. On a CRS whose axes
        # already agree the factor is 1.0 and the rectangle is the square it
        # has always been. Main-thread only, like the rest of this class
        # (finish() renders through the map job).
        half_x = crop_size * canvas_mupp / 2.0
        half_y = half_x / ground_unit_aspect(layer.crs(), center_x, center_y)
        self._extent = QgsRectangle(
            center_x - half_x, center_y - half_y,
            center_x + half_x, center_y + half_y
        )
        QgsMessageLog.logMessage(
            f"Online crop request: center=({center_x:.6f}, {center_y:.6f}), "
            f"mupp={canvas_mupp:.6f}, extent=({self._extent.xMinimum():.2f}, "
            f"{self._extent.yMinimum():.2f}, {self._extent.xMaximum():.2f}, "
            f"{self._extent.yMaximum():.2f}), CRS={layer.crs().authid()}",
            "AI Segmentation", level=Qgis.MessageLevel.Info
        )

    def begin(self):
        """Snapshot the live provider's resampling state, then force bilinear
        for the fetch. Safe to call once; restore() reverts whatever was
        captured, even if a setter below raises partway through."""
        if self._provider is None:
            return
        provider = self._provider
        try:
            self._orig_in = provider.zoomedInResamplingMethod()
            self._orig_out = provider.zoomedOutResamplingMethod()
        except (AttributeError, RuntimeError):
            # Provider without the resampling API: fetch at native resampling,
            # nothing to revert.
            return
        # isProviderResamplingEnabled() is the getter paired with
        # enableProviderResampling(); on a build that lacks it the enabled flag
        # stays unknown (None) and restore() leaves it on.
        if hasattr(provider, "isProviderResamplingEnabled"):
            try:
                self._orig_enabled = provider.isProviderResamplingEnabled()
            except (AttributeError, RuntimeError):
                self._orig_enabled = None
        # Past this point a revert is owed regardless of what the setters do.
        self._mutated = True
        try:
            provider.enableProviderResampling(True)
            provider.setZoomedInResamplingMethod(
                provider.ResamplingMethod.Bilinear)
            provider.setZoomedOutResamplingMethod(
                provider.ResamplingMethod.Bilinear)
        except (AttributeError, RuntimeError):
            pass

    def restore(self):
        """Revert the provider resampling state captured by begin().
        Idempotent and best-effort (the provider may be mid-teardown when a
        superseded async fetch unwinds)."""
        if not self._mutated:
            return
        self._mutated = False
        provider = self._provider
        try:
            provider.setZoomedInResamplingMethod(self._orig_in)
            provider.setZoomedOutResamplingMethod(self._orig_out)
            if self._orig_enabled is not None:
                provider.enableProviderResampling(self._orig_enabled)
        except (AttributeError, RuntimeError):
            pass

    def read_budget_spent(self) -> bool:
        """True once the provider reads have used their whole time ceiling
        (_READ_BUDGET_S). A driver may read this to stop early; step() and
        finish() already refuse to start another read once it is True."""
        return self._read_seconds >= self._READ_BUDGET_S

    def _timed_block(self, band):
        """One provider.block() read, charged to the read budget."""
        started = time.monotonic()
        try:
            return self._provider.block(
                band, self._extent, self._crop_size, self._crop_size)
        finally:
            self._read_seconds += time.monotonic() - started

    def step(self):
        """Perform one block read + stabilization check. Returns (action,
        delay_seconds):

        - ("stabilized", 0.0): two equal non-blank reads; go to finish().
        - ("refetch", 0.5): first valid non-blank read; reload issued, wait
          then step() again.
        - ("retry", delay): not stable yet; wait ``delay`` then step() again
          (the reload is issued at the START of the next step, matching the
          blocking loop's wait-then-reload order).
        - ("exhausted", 0.0): retry budget or read budget spent; go to finish()
          with whatever the provider returned.
        """
        provider = self._provider
        if self.read_budget_spent():
            # The reads have held the calling thread long enough. Stop the
            # ladder here rather than start another synchronous fetch.
            return ("exhausted", 0.0)
        if self._reload_pending:
            # The prior retry has served its back-off; refresh the tiles now,
            # exactly as the blocking loop reloaded before each re-read.
            provider.reloadData()
            self._reload_pending = False
        attempt = self._attempt
        block = self._timed_block(1)
        if block is not None and block.isValid():
            cur_data = bytes(block.data())
            # An all-zero block means the tiles have not downloaded yet (or the
            # area has no coverage). Two blank reads in a row must NOT count as
            # "stabilized": a slow network would trip the equality check and
            # abandon the retry budget after ~1s, shipping a blank crop. Only
            # real pixel content can end the loop early; blank reads always
            # fall through to the progressive retry branch.
            if cur_data.strip(b"\x00"):
                if self._prev_data is not None and cur_data == self._prev_data:
                    return ("stabilized", 0.0)
                self._prev_data = cur_data
                if attempt == 0:
                    # First valid fetch: reload once and re-check whether the
                    # tiles are still loading/updating.
                    provider.reloadData()
                    self._attempt = attempt + 1
                    return ("refetch", 0.5)
        if attempt < self._MAX_RETRIES - 1:
            delay = self._RETRY_DELAY * (1 + attempt * 0.5)  # 1.0, 1.5, 2.0, ...
            QgsMessageLog.logMessage(
                f"Online tile fetch attempt {attempt + 1} - "
                f"retrying in {delay:.1f}s...",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )
            self._reload_pending = True
            self._attempt = attempt + 1
            return ("retry", delay)
        return ("exhausted", 0.0)

    def finish(self):
        """Read the band data for the stabilized extent and build the crop.
        Returns the extract_crop_from_raster-style 4-tuple (image_np,
        crop_info, error, error_code)."""
        provider = self._provider
        if self.read_budget_spent():
            # The reads already held the calling thread for their whole
            # ceiling, so the provider is not answering usably: report the
            # network failure instead of starting one more synchronous fetch.
            return None, None, tr(
                "Failed to fetch tiles from the online layer. "
                "Check your network connection."
            ), "crop_error_online_fetch_failed"
        # Fetch bands using unified helper
        read_started = time.monotonic()
        bands_result, is_argb, fetch_err = _fetch_online_bands(
            provider, self._extent, self._crop_size, self._crop_size)
        self._read_seconds += time.monotonic() - read_started

        # If provider fetch failed, try canvas renderer fallback
        from_renderer = False
        if fetch_err is not None:
            QgsMessageLog.logMessage(
                f"Provider fetch failed ({fetch_err}), trying renderer "
                "fallback...",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )
            image_np, render_err = _render_layer_to_image(
                self._layer, self._extent, self._crop_size, self._crop_size)
            if render_err is not None:
                return None, None, tr(
                    "Failed to fetch tiles from the online layer. "
                    "Check your network connection."
                ), "crop_error_online_fetch_failed"
            from_renderer = True
        elif is_argb:
            # Already RGB uint8 (H, W, 3) from ARGB32 path
            image_np = bands_result
        else:
            # Raw bands (C, H, W) - normalize with nodata
            nodata = None
            try:
                nodata = provider.sourceNoDataValue(1)
            except Exception:
                pass  # nosec B110
            image_np = _normalize_to_uint8(bands_result, nodata_value=nodata)

        height = image_np.shape[0]
        width = image_np.shape[1]

        # Check for blank tiles
        total_sum = int(image_np.sum())
        if total_sum == 0:
            # Tiles fetched but all-zero: stale provider cache, zoom outside
            # the service's range, or genuinely no coverage. The canvas
            # renderer reads what QGIS actually displays, so one render-path
            # retry rescues the stale-cache case before giving up. A crop that
            # already came from the renderer has nothing left to try, so it
            # never pays for a second identical render.
            rendered = None
            if not from_renderer:
                rendered, render_err = _render_layer_to_image(
                    self._layer, self._extent, self._crop_size, self._crop_size)
                if render_err is not None:
                    rendered = None
            if rendered is not None and int(rendered.sum()) > 0:
                QgsMessageLog.logMessage(
                    "Blank tiles from provider, rescued by renderer fallback",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning
                )
                image_np = rendered
                height = image_np.shape[0]
                width = image_np.shape[1]
            else:
                return None, None, tr(
                    "Online layer returned blank tiles for this area. The "
                    "current zoom level may be outside the service's range, "
                    "or this area has no coverage. Zoom to a level where the "
                    "layer is visible on the map, then try again."
                ), "crop_error_online_blank_tiles"

        crop_info = {
            "bounds": (self._extent.xMinimum(), self._extent.yMinimum(),
                       self._extent.xMaximum(), self._extent.yMaximum()),
            "img_shape": (height, width),
        }
        return image_np, crop_info, None, None


# Wall-clock ceiling for the whole synchronous online fetch. The retry policy
# already bounds it, but this driver blocks the calling thread, so it carries
# its own stop so no future policy change can stretch that block.
_SYNC_FETCH_BUDGET_S: float = 25.0


def _blocking_wait(seconds, cancel_check=None):
    """Block the calling thread for ``seconds`` while pumping the event loop.
    Used only by the synchronous (headless/recovery) online fetch driver; the
    interactive path schedules its waits off the event loop instead.

    ``cancel_check`` is an optional callable polled between pumps; the wait
    stops as soon as it returns True. Returns False when the wait was cut
    short that way, True when it ran to its deadline.
    """
    from qgis.core import QgsApplication

    deadline = time.monotonic() + seconds
    while True:
        if cancel_check is not None and cancel_check():
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return True
        QgsApplication.processEvents()
        time.sleep(min(0.05, remaining))


def extract_crop_from_online_layer(layer, center_x, center_y, canvas_mupp,
                                   crop_size=1024, cancel_check=None):
    """Extract a crop_size x crop_size RGB crop from an online layer.

    SYNCHRONOUS and blocking (up to ~18s of tile-fetch retries): used by the
    headless/MCP and recovery callers, whose contract is a blocking call. The
    interactive path drives the same OnlineCropFetcher steps asynchronously so
    the GUI never freezes.

    Args:
        layer: QgsRasterLayer (WMS, WMTS, XYZ, WCS, ArcGIS)
        center_x, center_y: Center of crop in layer CRS coordinates
        canvas_mupp: Map units per pixel
        crop_size: Size of the crop in pixels (default 1024)
        cancel_check: optional callable polled during every back-off wait. When
            it returns True the fetch gives up at once instead of sitting out
            the rest of the retry budget.

    Returns:
        (image_np, crop_info, error, error_code) - same format as
        extract_crop_from_raster.
    """
    fetcher = OnlineCropFetcher(layer, center_x, center_y, canvas_mupp, crop_size)
    if fetcher.error is not None:
        return None, None, fetcher.error, fetcher.error_code
    try:
        fetcher.begin()
        stop_by = time.monotonic() + _SYNC_FETCH_BUDGET_S
        while True:
            action, delay = fetcher.step()
            if action in ("stabilized", "exhausted"):
                break
            if not _blocking_wait(delay, cancel_check):
                return (None, None, tr("Crop fetch was cancelled."),
                        "crop_error_online_cancelled")
            if time.monotonic() >= stop_by:
                break
        return fetcher.finish()
    except Exception as e:  # noqa: BLE001
        return None, None, str(e), "crop_error_online_exception"
    finally:
        fetcher.restore()
