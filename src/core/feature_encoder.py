import os
import time

import numpy as np

from qgis.core import QgsMessageLog, Qgis

from .i18n import tr

# Raster formats known to require GDAL conversion (not supported by pip-installed rasterio)
_GDAL_ONLY_FORMATS = {
    '.ecw', '.sid', '.jp2', '.j2k', '.j2c',
    '.nitf', '.ntf', '.img', '.hdf', '.hdf5', '.he5', '.nc',
}

# Online/remote raster providers that need rendering before encoding
ONLINE_PROVIDERS = frozenset(['wms', 'wmts', 'xyz', 'arcgismapserver', 'wcs'])


def _needs_gdal_conversion(raster_path):
    """Check if raster format requires GDAL conversion for rasterio."""
    ext = os.path.splitext(raster_path)[1].lower()
    return ext in _GDAL_ONLY_FORMATS


def _convert_with_gdal(raster_path, output_dir, progress_callback=None):
    """Convert raster to GeoTIFF using QGIS's GDAL.

    Returns (converted_path, error_message). On success error_message is None.
    """
    ext = os.path.splitext(raster_path)[1].upper()

    try:
        from osgeo import gdal
    except ImportError:
        return None, tr(
            "{ext} format is not directly supported. "
            "GDAL is not available for automatic conversion.\n"
            "Please convert your raster to GeoTIFF (.tif) before using "
            "AI Segmentation."
        ).format(ext=ext)

    if progress_callback:
        progress_callback(0, tr("Converting {ext} to GeoTIFF...").format(ext=ext))

    try:
        ds = gdal.Open(raster_path)
        if ds is None:
            return None, tr(
                "Cannot open {ext} file. The format may not be supported "
                "by your QGIS installation.\n"
                "Please convert your raster to GeoTIFF (.tif) before using "
                "AI Segmentation."
            ).format(ext=ext)

        # Use .tmp extension so FeatureDataset's *.tif glob never picks it up
        converted_path = os.path.join(output_dir, '_converted_source.tmp')

        def gdal_progress(complete, message, data):
            if progress_callback:
                pct = int(complete * 5)
                progress_callback(
                    pct,
                    tr("Converting {ext} to GeoTIFF ({pct}%)...").format(
                        ext=ext, pct=int(complete * 100))
                )
            return 1

        result = gdal.Translate(
            converted_path, ds,
            format='GTiff',
            creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER'],
            callback=gdal_progress
        )
        ds = None
        if result is None:
            return None, tr(
                "Failed to convert {ext} file to GeoTIFF."
            ).format(ext=ext)
        result = None

        if not os.path.exists(converted_path):
            return None, tr(
                "Failed to convert {ext} file to GeoTIFF."
            ).format(ext=ext)

        QgsMessageLog.logMessage(
            "Converted {} to GeoTIFF for encoding: {}".format(ext, converted_path),
            "AI Segmentation",
            level=Qgis.Info
        )
        return converted_path, None

    except Exception as e:
        return None, tr(
            "Failed to convert {ext} file to GeoTIFF: {error}\n"
            "Please convert your raster to GeoTIFF (.tif) manually."
        ).format(ext=ext, error=str(e))


def extract_crop_from_raster(raster_path, center_x, center_y, crop_size=1024,
                             layer_crs_wkt=None, layer_extent=None):
    """Extract a crop_size x crop_size RGB crop centered on (center_x, center_y).

    Args:
        raster_path: Path to the raster file
        center_x, center_y: Center of crop in geo/pixel coordinates
        crop_size: Size of the crop in pixels (default 1024)
        layer_crs_wkt: Optional CRS WKT for non-georeferenced rasters
        layer_extent: Optional (xmin, ymin, xmax, ymax) for non-georeferenced rasters

    Returns:
        (image_np, crop_info, error) where:
        - image_np: (H, W, 3) uint8 numpy array
        - crop_info: dict with 'bounds' (minx, miny, maxx, maxy) and 'img_shape' (H, W)
        - error: error string or None on success
    """
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError:
        return None, None, "rasterio is not available"

    # Handle GDAL-only formats
    converted_path = None
    tmpdir_obj = None
    encoding_raster_path = raster_path
    if _needs_gdal_conversion(raster_path):
        import tempfile
        tmpdir_obj = tempfile.TemporaryDirectory(prefix="ai_seg_crop_")
        converted_path, conv_error = _convert_with_gdal(raster_path, tmpdir_obj.name)
        if conv_error:
            tmpdir_obj.cleanup()
            return None, None, conv_error
        encoding_raster_path = converted_path

    try:
        with rasterio.open(encoding_raster_path) as src:
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

            # Compute window clamped to raster bounds
            half = crop_size // 2
            col_off = max(0, int(round(col_center - half)))
            row_off = max(0, int(round(row_center - half)))

            actual_width = min(crop_size, raster_width - col_off)
            actual_height = min(crop_size, raster_height - row_off)

            if actual_width <= 0 or actual_height <= 0:
                return None, None, "Click is outside the raster bounds"

            window = Window(col_off, row_off, actual_width, actual_height)
            tile_data = src.read(window=window)

            # Handle bands: 1->RGB, 4->RGB, normalize non-uint8
            if tile_data.shape[0] == 1:
                tile_data = np.repeat(tile_data, 3, axis=0)
            elif tile_data.shape[0] >= 4:
                tile_data = tile_data[:3, :, :]
            elif tile_data.shape[0] > 3:
                tile_data = tile_data[:3, :, :]

            if tile_data.dtype != np.uint8:
                # Percentile stretch: clip outliers for better contrast
                flat = tile_data.astype(np.float64).ravel()
                p2, p98 = np.percentile(flat, [2, 98])
                if p98 > p2:
                    tile_data = np.clip(tile_data.astype(np.float64), p2, p98)
                    tile_data = (
                        (tile_data - p2) / (p98 - p2) * 255
                    ).astype(np.uint8)
                else:
                    tile_data = np.zeros_like(tile_data, dtype=np.uint8)
            else:
                # Percentile stretch on uint8 for under/over-exposed images
                flat = tile_data.ravel()
                p2, p98 = np.percentile(flat, [2, 98])
                if p98 - p2 < 220:
                    # Only stretch if histogram is compressed (not already full range)
                    p2_f, p98_f = float(p2), float(p98)
                    if p98_f > p2_f:
                        tile_data = np.clip(
                            tile_data.astype(np.float64), p2_f, p98_f)
                        tile_data = (
                            (tile_data - p2_f) / (p98_f - p2_f) * 255
                        ).astype(np.uint8)

            # CHW -> HWC
            image_np = np.transpose(tile_data, (1, 2, 0))

            # Pad to full crop_size if crop was clipped at raster edge.
            # Uses reflect padding instead of black borders for better
            # SAM context at image boundaries.
            if actual_height < crop_size or actual_width < crop_size:
                pad_bottom = crop_size - actual_height
                pad_right = crop_size - actual_width
                image_np = np.pad(
                    image_np,
                    ((0, pad_bottom), (0, pad_right), (0, 0)),
                    mode='reflect'
                )

            # Compute geo bounds for this crop
            crop_minx = bounds_left + col_off * pixel_size_x
            crop_maxx = bounds_left + (col_off + actual_width) * pixel_size_x
            crop_maxy = bounds_top - row_off * pixel_size_y
            crop_miny = bounds_top - (row_off + actual_height) * pixel_size_y

            crop_info = {
                'bounds': (crop_minx, crop_miny, crop_maxx, crop_maxy),
                'img_shape': (actual_height, actual_width),
                'col_off': col_off,
                'row_off': row_off,
            }

            return image_np, crop_info, None

    except Exception as e:
        return None, None, str(e)

    finally:
        if tmpdir_obj is not None:
            try:
                tmpdir_obj.cleanup()
            except Exception:
                pass


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
    from qgis.core import QgsRectangle, Qgis

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
        max_retries = 3
        retry_delay = 1.0
        block = None
        for attempt in range(max_retries):
            block = provider.block(1, extent, crop_size, crop_size)
            if block is not None and block.isValid():
                break
            if attempt < max_retries - 1:
                QgsMessageLog.logMessage(
                    "Online tile fetch attempt {} failed, "
                    "retrying in {:.1f}s...".format(
                        attempt + 1, retry_delay),
                    "AI Segmentation", level=Qgis.Warning
                )
                # Sleep while keeping UI responsive
                from qgis.core import QgsApplication
                deadline = time.monotonic() + retry_delay
                while time.monotonic() < deadline:
                    QgsApplication.processEvents()
                    time.sleep(0.05)
                # Refresh the provider to trigger a new tile request
                provider.reloadData()

        provider.setZoomedInResamplingMethod(original_method)

        if block is None or not block.isValid():
            return None, None, tr(
                "Failed to fetch tiles from the online layer. "
                "Check your network connection."
            )

        block_w = block.width()
        block_h = block.height()
        if block_w == 0 or block_h == 0:
            return None, None, tr(
                "Online layer returned empty data. "
                "The area may not have coverage."
            )
        width = block_w
        height = block_h

        raw_bytes = block.data()
        if raw_bytes is None or len(raw_bytes) == 0:
            return None, None, tr(
                "Online layer returned empty data. "
                "The area may not have coverage."
            )

        raw_data = bytes(raw_bytes)
        dt = block.dataType()

        if dt == Qgis.DataType.ARGB32 or dt == Qgis.DataType.ARGB32_Premultiplied:
            arr = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                height, width, 4).copy()
            # Qt uses BGRA byte order (ARGB32 in big-endian notation)
            red = arr[:, :, 2]
            green = arr[:, :, 1]
            blue = arr[:, :, 0]
        elif dt == Qgis.DataType.Byte:
            band_count = provider.bandCount()
            red = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                height, width)
            if band_count >= 3:
                b2 = provider.block(2, extent, width, height)
                b3 = provider.block(3, extent, width, height)
                green = np.frombuffer(
                    bytes(b2.data()), dtype=np.uint8
                ).reshape(height, width)
                blue = np.frombuffer(
                    bytes(b3.data()), dtype=np.uint8
                ).reshape(height, width)
            else:
                green = blue = red
        elif dt == Qgis.DataType.UInt16:
            band_arr = np.frombuffer(raw_data, dtype=np.uint16)
            red = (band_arr / 256).astype(np.uint8).reshape(height, width)
            green = blue = red
        elif dt == Qgis.DataType.Float32:
            band_arr = np.frombuffer(raw_data, dtype=np.float32)
            bmin, bmax = band_arr.min(), band_arr.max()
            if bmax > bmin:
                scaled = ((band_arr - bmin) / (bmax - bmin) * 255)
                red = scaled.astype(np.uint8).reshape(height, width)
            else:
                red = np.zeros((height, width), dtype=np.uint8)
            green = blue = red
        else:
            if len(raw_data) == width * height * 4:
                arr = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                    height, width, 4).copy()
                red = arr[:, :, 2]
                green = arr[:, :, 1]
                blue = arr[:, :, 0]
            else:
                return None, None, tr(
                    "Unexpected data format from online layer "
                    "(dataType={dt}, {size} bytes for {w}x{h})."
                ).format(dt=dt, size=len(raw_data), w=width, h=height)

        total_sum = int(red.sum()) + int(green.sum()) + int(blue.sum())
        if total_sum == 0:
            return None, None, tr(
                "Online layer returned blank tiles for this area. "
                "Try panning to an area with data coverage."
            )

        image_np = np.stack([red, green, blue], axis=-1)

        # Compute actual geo bounds from provider extent
        crop_info = {
            'bounds': (extent.xMinimum(), extent.yMinimum(),
                       extent.xMaximum(), extent.yMaximum()),
            'img_shape': (height, width),
        }

        return image_np, crop_info, None

    except Exception as e:
        return None, None, str(e)

