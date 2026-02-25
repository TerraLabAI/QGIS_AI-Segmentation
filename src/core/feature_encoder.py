import os
import time

import numpy as np

from qgis.core import QgsMessageLog, Qgis

from .i18n import tr

# Raster formats that pip-installed rasterio may not support reliably.
# These go through GDAL windowed read instead.
_GDAL_ONLY_FORMATS = {
    '.ecw', '.sid', '.jp2', '.j2k', '.j2c',
    '.nitf', '.ntf', '.img', '.hdf', '.hdf5', '.he5', '.nc',
    '.gpkg',
}

# Online/remote raster providers that need rendering before encoding
ONLINE_PROVIDERS = frozenset(['wms', 'wmts', 'xyz', 'arcgismapserver', 'wcs'])


def _normalize_to_uint8(bands, nodata_value=None):
    """Normalize a multi-band array to (H, W, 3) uint8 using per-band percentile stretch.

    Args:
        bands: numpy array of shape (C, H, W), any numeric dtype
        nodata_value: optional nodata value to mask before computing percentiles

    Returns:
        numpy array of shape (H, W, 3) uint8
    """
    num_bands = bands.shape[0]
    if num_bands == 1:
        bands = np.repeat(bands, 3, axis=0)
    elif num_bands == 2:
        bands = np.stack([bands[0], bands[1], bands[0]], axis=0)
    elif num_bands > 3:
        bands = bands[:3, :, :]

    # Build nodata mask: True where pixel is valid
    if nodata_value is not None:
        valid_mask = bands[0] != nodata_value
    else:
        valid_mask = np.ones(bands.shape[1:], dtype=bool)

    # Also mask NaN for float types
    if np.issubdtype(bands.dtype, np.floating):
        for b in range(bands.shape[0]):
            valid_mask = valid_mask & ~np.isnan(bands[b])

    is_uint8 = (bands.dtype == np.uint8)
    result = np.zeros((3, bands.shape[1], bands.shape[2]), dtype=np.uint8)

    for b in range(3):
        band = bands[b].astype(np.float64)
        valid_pixels = band[valid_mask]

        if valid_pixels.size == 0:
            continue

        p2, p98 = np.percentile(valid_pixels, [2, 98])

        if is_uint8:
            # Only stretch if histogram is compressed
            if p98 - p2 < 220:
                if p98 > p2:
                    stretched = np.clip(band, p2, p98)
                    stretched = (stretched - p2) / (p98 - p2) * 255
                    result[b] = stretched.astype(np.uint8)
                else:
                    result[b] = band.astype(np.uint8)
            else:
                result[b] = bands[b]
        else:
            if p98 > p2:
                stretched = np.clip(band, p2, p98)
                stretched = (stretched - p2) / (p98 - p2) * 255
                result[b] = stretched.astype(np.uint8)
            # else: stays zeros

    # Zero out nodata pixels
    if nodata_value is not None or np.issubdtype(bands.dtype, np.floating):
        nodata_mask = ~valid_mask
        if np.any(nodata_mask):
            for b in range(3):
                result[b][nodata_mask] = 0

    # CHW -> HWC
    return np.transpose(result, (1, 2, 0))


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
            block_h, block_w, 4).copy()
        # Qt BGRA byte order -> RGB
        image_np = np.stack(
            [arr[:, :, 2], arr[:, :, 1], arr[:, :, 0]], axis=-1)
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
                block_h, block_w, 4).copy()
            image_np = np.stack(
                [arr[:, :, 2], arr[:, :, 1], arr[:, :, 0]], axis=-1)
            return image_np, True, None
        return None, False, "Unsupported data type: {}".format(dt)

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


def _render_layer_to_image(layer, extent, width, height):
    """Render a layer to an RGB image using QGIS map renderer (fallback).

    Works with any layer type (WMS, WMTS, XYZ, WCS, vector tiles, etc.)

    Args:
        layer: QgsMapLayer to render
        extent: QgsRectangle for the area
        width: pixel width
        height: pixel height

    Returns:
        (image_np, error) where image_np is (H, W, 3) uint8 or None
    """
    try:
        from qgis.core import QgsMapSettings, QgsMapRendererCustomPainterJob
        from qgis.PyQt.QtGui import QImage, QPainter
        from qgis.PyQt.QtCore import QSize

        img = QImage(QSize(width, height), QImage.Format_RGB32)
        img.fill(0)

        settings = QgsMapSettings()
        settings.setOutputSize(QSize(width, height))
        settings.setExtent(extent)
        settings.setLayers([layer])
        settings.setDestinationCrs(layer.crs())
        settings.setBackgroundColor(img.pixelColor(0, 0))

        painter = QPainter(img)
        job = QgsMapRendererCustomPainterJob(settings, painter)
        job.start()
        job.waitForFinished()
        painter.end()

        # QImage -> numpy
        img = img.convertToFormat(QImage.Format_RGB32)
        ptr = img.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
            height, width, 4).copy()
        # BGRA -> RGB
        image_np = np.stack(
            [arr[:, :, 2], arr[:, :, 1], arr[:, :, 0]], axis=-1)
        return image_np, None

    except Exception as e:
        return None, "Renderer fallback failed: {}".format(str(e))


def _needs_gdal_conversion(raster_path):
    """Check if raster format requires GDAL conversion for rasterio."""
    ext = os.path.splitext(raster_path)[1].lower()
    return ext in _GDAL_ONLY_FORMATS


def _read_crop_with_gdal(raster_path, center_x, center_y, crop_size,
                         scale_factor, layer_extent):
    """Read a windowed crop directly from a GDAL-supported raster (JP2, ECW, etc.).

    Instead of converting the entire file to GeoTIFF, reads only the needed
    pixels using GDAL windowed access.

    Returns (image_np, crop_info, error).
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
        ).format(ext=ext)

    try:
        ds = gdal.Open(raster_path)
        if ds is None:
            return None, None, tr(
                "Cannot open {ext} file. The format may not be supported "
                "by your QGIS installation.\n"
                "Please convert your raster to GeoTIFF (.tif) before using "
                "AI Segmentation."
            ).format(ext=ext)

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

        read_size = int(crop_size * scale_factor)
        half = read_size // 2
        col_off = max(0, int(round(col_center - half)))
        row_off = max(0, int(round(row_center - half)))

        actual_width = min(read_size, raster_width - col_off)
        actual_height = min(read_size, raster_height - row_off)

        if actual_width <= 0 or actual_height <= 0:
            ds = None
            return None, None, "Click is outside the raster bounds"

        num_bands = min(ds.RasterCount, 3)
        if num_bands == 0:
            ds = None
            return None, None, "Raster has no bands"

        if scale_factor > 1.0:
            out_h = max(1, min(crop_size, int(actual_height / scale_factor)))
            out_w = max(1, min(crop_size, int(actual_width / scale_factor)))
        else:
            out_h = actual_height
            out_w = actual_width

        bands = []
        for b_idx in range(1, num_bands + 1):
            band = ds.GetRasterBand(b_idx)
            data = band.ReadAsArray(
                col_off, row_off, actual_width, actual_height,
                buf_xsize=out_w, buf_ysize=out_h
            )
            bands.append(data)

        nodata = ds.GetRasterBand(1).GetNoDataValue()
        ds = None

        tile_data = np.stack(bands, axis=0)
        image_np = _normalize_to_uint8(tile_data, nodata_value=nodata)

        if out_h < crop_size or out_w < crop_size:
            pad_bottom = crop_size - out_h
            pad_right = crop_size - out_w
            image_np = np.pad(
                image_np,
                ((0, pad_bottom), (0, pad_right), (0, 0)),
                mode='reflect'
            )

        crop_minx = bounds_left + col_off * pixel_size_x
        crop_maxx = bounds_left + (col_off + actual_width) * pixel_size_x
        crop_maxy = bounds_top - row_off * pixel_size_y
        crop_miny = bounds_top - (row_off + actual_height) * pixel_size_y

        crop_info = {
            'bounds': (crop_minx, crop_miny, crop_maxx, crop_maxy),
            'img_shape': (out_h, out_w),
            'col_off': col_off,
            'row_off': row_off,
        }

        QgsMessageLog.logMessage(
            "Read {} crop directly via GDAL: {}x{} at ({}, {})".format(
                ext, out_w, out_h, col_off, row_off),
            "AI Segmentation", level=Qgis.Info
        )
        return image_np, crop_info, None

    except Exception as e:
        return None, None, tr(
            "Failed to read {ext} file: {error}\n"
            "Please convert your raster to GeoTIFF (.tif) manually."
        ).format(ext=ext, error=str(e))


def extract_crop_from_raster(raster_path, center_x, center_y, crop_size=1024,
                             layer_crs_wkt=None, layer_extent=None,
                             scale_factor=1.0):
    """Extract a crop_size x crop_size RGB crop centered on (center_x, center_y).

    Args:
        raster_path: Path to the raster file
        center_x, center_y: Center of crop in geo/pixel coordinates
        crop_size: Size of the crop in pixels (default 1024)
        layer_crs_wkt: Optional CRS WKT for non-georeferenced rasters
        layer_extent: Optional (xmin, ymin, xmax, ymax) for non-georeferenced rasters
        scale_factor: Read a larger window and downsample to crop_size.
            When > 1.0, reads crop_size * scale_factor native pixels and
            resamples down to crop_size using bilinear interpolation.

    Returns:
        (image_np, crop_info, error) where:
        - image_np: (H, W, 3) uint8 numpy array
        - crop_info: dict with 'bounds' (minx, miny, maxx, maxy) and 'img_shape' (H, W)
        - error: error string or None on success
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import Window
    except ImportError:
        return None, None, "rasterio is not available"

    # Handle GDAL-only formats (JP2, ECW, etc.) with direct windowed read
    if _needs_gdal_conversion(raster_path):
        return _read_crop_with_gdal(
            raster_path, center_x, center_y, crop_size,
            scale_factor, layer_extent
        )

    try:
        with rasterio.open(raster_path) as src:
            raster_width = src.width
            raster_height = src.height
            raster_transform = src.transform

            # Detect non-georeferenced mode (same logic as encoding_worker.py)
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

            # When scale_factor > 1, read a larger native window
            read_size = int(crop_size * scale_factor)
            half = read_size // 2
            col_off = max(0, int(round(col_center - half)))
            row_off = max(0, int(round(row_center - half)))

            actual_width = min(read_size, raster_width - col_off)
            actual_height = min(read_size, raster_height - row_off)

            if actual_width <= 0 or actual_height <= 0:
                return None, None, "Click is outside the raster bounds"

            window = Window(col_off, row_off, actual_width, actual_height)

            if scale_factor > 1.0:
                # Downsample to crop_size using rasterio resampling
                out_h = min(crop_size, int(actual_height / scale_factor))
                out_w = min(crop_size, int(actual_width / scale_factor))
                out_h = max(1, out_h)
                out_w = max(1, out_w)
                tile_data = src.read(
                    window=window,
                    out_shape=(src.count, out_h, out_w),
                    resampling=Resampling.bilinear
                )
            else:
                tile_data = src.read(window=window)
                out_h = actual_height
                out_w = actual_width

            nodata = src.nodata
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
                    mode='reflect'
                )

            # Compute geo bounds for this crop (covers the full read area)
            crop_minx = bounds_left + col_off * pixel_size_x
            crop_maxx = bounds_left + (col_off + actual_width) * pixel_size_x
            crop_maxy = bounds_top - row_off * pixel_size_y
            crop_miny = bounds_top - (row_off + actual_height) * pixel_size_y

            crop_info = {
                'bounds': (crop_minx, crop_miny, crop_maxx, crop_maxy),
                'img_shape': (out_h, out_w),
                'col_off': col_off,
                'row_off': row_off,
            }

            return image_np, crop_info, None

    except Exception as e:
        # Fallback to GDAL if rasterio fails (unsupported driver, etc.)
        QgsMessageLog.logMessage(
            "rasterio failed ({}), trying GDAL fallback...".format(str(e)),
            "AI Segmentation", level=Qgis.Warning
        )
        return _read_crop_with_gdal(
            raster_path, center_x, center_y, crop_size,
            scale_factor, layer_extent
        )


def extract_crop_from_online_layer(layer, center_x, center_y, canvas_mupp,
                                   crop_size=1024):
    """Extract a crop_size x crop_size RGB crop from an online layer.

    Args:
        layer: QgsRasterLayer (WMS, WMTS, XYZ, WCS, ArcGIS)
        center_x, center_y: Center of crop in layer CRS coordinates
        canvas_mupp: Map units per pixel
        crop_size: Size of the crop in pixels (default 1024)

    Returns:
        (image_np, crop_info, error) - same format as extract_crop_from_raster
    """
    from qgis.core import QgsRectangle

    provider = layer.dataProvider()
    if provider is None:
        return None, None, tr("Layer data provider is not available.")

    half_size = crop_size * canvas_mupp / 2.0
    extent = QgsRectangle(
        center_x - half_size, center_y - half_size,
        center_x + half_size, center_y + half_size
    )

    try:
        provider.enableProviderResampling(True)
        original_method = provider.zoomedInResamplingMethod()
        provider.setZoomedInResamplingMethod(
            provider.ResamplingMethod.Bilinear)
        provider.setZoomedOutResamplingMethod(
            provider.ResamplingMethod.Bilinear)

        # Retry fetching tiles: when the user pans to a new area, the
        # provider cache may not have the tiles yet.  A short delay
        # between attempts gives QGIS time to download them.
        # We also re-fetch after getting a valid block to detect
        # mixed-resolution tiles (stale cache from different zoom).
        from qgis.core import QgsApplication
        max_retries = 5
        retry_delay = 0.8
        block = None
        prev_data = None
        for attempt in range(max_retries):
            block = provider.block(1, extent, crop_size, crop_size)
            if block is not None and block.isValid():
                cur_data = bytes(block.data())
                if prev_data is not None and cur_data == prev_data:
                    # Image stabilized - tiles are consistent
                    break
                prev_data = cur_data
                if attempt == 0:
                    # First valid fetch - always re-fetch once to
                    # check if tiles are still loading/updating
                    provider.reloadData()
                    deadline = time.monotonic() + 0.5
                    while time.monotonic() < deadline:
                        QgsApplication.processEvents()
                        time.sleep(0.05)
                    continue
            if attempt < max_retries - 1:
                QgsMessageLog.logMessage(
                    "Online tile fetch attempt {} - "
                    "retrying in {:.1f}s...".format(
                        attempt + 1, retry_delay),
                    "AI Segmentation", level=Qgis.Warning
                )
                deadline = time.monotonic() + retry_delay
                while time.monotonic() < deadline:
                    QgsApplication.processEvents()
                    time.sleep(0.05)
                provider.reloadData()

        provider.setZoomedInResamplingMethod(original_method)

        # Fetch bands using unified helper
        bands_result, is_argb, fetch_err = _fetch_online_bands(
            provider, extent, crop_size, crop_size)

        # If provider fetch failed, try canvas renderer fallback
        if fetch_err is not None:
            QgsMessageLog.logMessage(
                "Provider fetch failed ({}), trying renderer "
                "fallback...".format(fetch_err),
                "AI Segmentation", level=Qgis.Warning
            )
            image_np, render_err = _render_layer_to_image(
                layer, extent, crop_size, crop_size)
            if render_err is not None:
                return None, None, tr(
                    "Failed to fetch tiles from the online layer. "
                    "Check your network connection."
                )
        elif is_argb:
            # Already RGB uint8 (H, W, 3) from ARGB32 path
            image_np = bands_result
        else:
            # Raw bands (C, H, W) - normalize with nodata
            nodata = None
            try:
                nodata = provider.sourceNoDataValue(1)
            except Exception:
                pass
            image_np = _normalize_to_uint8(bands_result, nodata_value=nodata)

        height = image_np.shape[0]
        width = image_np.shape[1]

        # Check for blank tiles
        total_sum = int(image_np.sum())
        if total_sum == 0:
            return None, None, tr(
                "Online layer returned blank tiles for this area. "
                "Try panning to an area with data coverage."
            )

        crop_info = {
            'bounds': (extent.xMinimum(), extent.yMinimum(),
                       extent.xMaximum(), extent.yMaximum()),
            'img_shape': (height, width),
        }

        return image_np, crop_info, None

    except Exception as e:
        return None, None, str(e)
