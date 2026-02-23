import os
import json
import time
import subprocess
import tempfile
from typing import Tuple, Optional, Callable

import numpy as np

from qgis.core import QgsMessageLog, Qgis

from .subprocess_utils import get_clean_env_for_venv, get_subprocess_kwargs
from .i18n import tr

# Global timeout for the encoding stdout reading loop (45 minutes)
_ENCODING_GLOBAL_TIMEOUT = 2700

# Stall detection: if no progress update for this many seconds, abort
_ENCODING_STALL_TIMEOUT = 300  # 5 minutes

# Windows NTSTATUS crash codes (reuse from venv_manager pattern)
_WINDOWS_ENCODING_CRASH_CODES = {
    3221225477,   # 0xC0000005 unsigned - ACCESS_VIOLATION
    -1073741819,  # 0xC0000005 signed   - ACCESS_VIOLATION
    3221225725,   # 0xC00000FD unsigned - STACK_OVERFLOW
    -1073741571,  # 0xC00000FD signed   - STACK_OVERFLOW
    3221225781,   # 0xC0000135 unsigned - DLL_NOT_FOUND
    -1073741515,  # 0xC0000135 signed   - DLL_NOT_FOUND
}

# Raster formats that require GDAL conversion (not supported by pip-installed rasterio)
_GDAL_ONLY_FORMATS = {'.ecw', '.sid', '.jp2', '.j2k', '.j2c'}

# Online/remote raster providers that need rendering before encoding
ONLINE_PROVIDERS = frozenset(['wms', 'wmts', 'xyz', 'arcgismapserver', 'wcs'])

# Maximum pixel dimensions for online layer rendering (prevents OOM)
_MAX_ONLINE_PIXELS = 10000


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


def _render_online_layer_to_geotiff(layer, extent, canvas_mupp,
                                    output_dir, progress_callback=None):
    """Render an online raster layer to a GeoTIFF using dataProvider().block().

    Follows the same approach as the deepness plugin: fetch a block, check
    dataType() for ARGB32 vs standalone bands, enable bilinear resampling.

    Args:
        layer: QgsRasterLayer (WMS, WMTS, XYZ, WCS, ArcGIS)
        extent: QgsRectangle in the layer's CRS
        canvas_mupp: map units per pixel (resolution from the canvas)
        output_dir: directory where the .tmp GeoTIFF will be written
        progress_callback: optional (percent, message) callback

    Returns:
        (geotiff_path, error_message) - on success error_message is None
    """
    try:
        from osgeo import gdal, osr
    except ImportError:
        return None, tr(
            "GDAL is not available. Cannot render online layer to GeoTIFF."
        )

    if progress_callback:
        progress_callback(0, tr("Rendering online layer..."))

    provider = layer.dataProvider()
    if provider is None:
        return None, tr("Layer data provider is not available.")

    # Compute pixel dimensions from extent and canvas resolution
    width = int(round(extent.width() / canvas_mupp))
    height = int(round(extent.height() / canvas_mupp))

    if width <= 0 or height <= 0:
        return None, tr("Visible area is too small to render.")

    if width > _MAX_ONLINE_PIXELS or height > _MAX_ONLINE_PIXELS:
        return None, tr(
            "The visible area is too large ({w}x{h} pixels). "
            "Zoom in to reduce the area below {max}x{max} pixels."
        ).format(w=width, h=height, max=_MAX_ONLINE_PIXELS)

    QgsMessageLog.logMessage(
        "Rendering online layer: {}x{} pixels, mupp={:.6f}".format(
            width, height, canvas_mupp),
        "AI Segmentation", level=Qgis.Info
    )

    try:
        # Enable bilinear resampling for better quality (like deepness)
        provider.enableProviderResampling(True)
        original_method = provider.zoomedInResamplingMethod()
        provider.setZoomedInResamplingMethod(
            provider.ResamplingMethod.Bilinear)
        provider.setZoomedOutResamplingMethod(
            provider.ResamplingMethod.Bilinear)

        # Fetch band 1 block - works for both composite and standalone
        block = provider.block(1, extent, width, height)

        # Restore original resampling
        provider.setZoomedInResamplingMethod(original_method)

        if block is None or not block.isValid():
            return None, tr(
                "Failed to fetch tiles from the online layer. "
                "Check your network connection."
            )

        block_w = block.width()
        block_h = block.height()
        if block_w == 0 or block_h == 0:
            return None, tr(
                "Online layer returned empty data. "
                "The area may not have coverage."
            )
        # Use actual block dimensions (provider may adjust)
        width = block_w
        height = block_h

        if progress_callback:
            progress_callback(30, tr("Processing tiles..."))

        raw_bytes = block.data()
        if raw_bytes is None or len(raw_bytes) == 0:
            return None, tr(
                "Online layer returned empty data. "
                "The area may not have coverage."
            )

        raw_data = bytes(raw_bytes)
        dt = block.dataType()

        QgsMessageLog.logMessage(
            "Block: {}x{}, dataType={}, bytes={}".format(
                width, height, dt, len(raw_data)),
            "AI Segmentation", level=Qgis.Info
        )

        # Check dataType to decide how to interpret the bytes
        # (same approach as deepness plugin)
        if dt == Qgis.DataType.ARGB32 or dt == Qgis.DataType.ARGB32_Premultiplied:
            # ARGB32: 4 bytes per pixel, little-endian = B, G, R, A
            arr = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                height, width, 4)
            red = arr[:, :, 2].copy()
            green = arr[:, :, 1].copy()
            blue = arr[:, :, 0].copy()
        elif dt == Qgis.DataType.Byte:
            # Single-byte band - fetch remaining bands for RGB
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
            # Fallback: try as ARGB32 based on data size
            if len(raw_data) == width * height * 4:
                arr = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                    height, width, 4)
                red = arr[:, :, 2].copy()
                green = arr[:, :, 1].copy()
                blue = arr[:, :, 0].copy()
            else:
                return None, tr(
                    "Unexpected data format from online layer "
                    "(dataType={dt}, {size} bytes for {w}x{h})."
                ).format(dt=dt, size=len(raw_data), w=width, h=height)

        # Check for all-zero (blank) tiles
        total_sum = int(red.sum()) + int(green.sum()) + int(blue.sum())
        if total_sum == 0:
            return None, tr(
                "Online layer returned blank tiles for this area. "
                "Try panning to an area with data coverage."
            )

        if progress_callback:
            progress_callback(50, tr("Writing GeoTIFF..."))

        # Write GeoTIFF using GDAL
        output_path = os.path.join(output_dir, '_online_rendered.tmp')
        driver = gdal.GetDriverByName('GTiff')
        if driver is None:
            return None, tr("GDAL GTiff driver is not available.")

        ds = driver.Create(
            output_path, width, height, 3, gdal.GDT_Byte,
            options=['COMPRESS=LZW', 'TILED=YES']
        )
        if ds is None:
            return None, tr("Failed to create GeoTIFF file.")

        # Set geotransform: top-left x, pixel width, 0, top-left y, 0, pixel height (negative)
        pixel_width = extent.width() / width
        pixel_height = extent.height() / height
        ds.SetGeoTransform([
            extent.xMinimum(), pixel_width, 0,
            extent.yMaximum(), 0, -pixel_height
        ])

        # Set CRS from layer
        crs = layer.crs()
        if crs.isValid():
            srs = osr.SpatialReference()
            srs.ImportFromWkt(crs.toWkt())
            ds.SetProjection(srs.ExportToWkt())

        ds.GetRasterBand(1).WriteArray(red)
        ds.GetRasterBand(2).WriteArray(green)
        ds.GetRasterBand(3).WriteArray(blue)
        ds.FlushCache()
        ds = None

        if progress_callback:
            progress_callback(70, tr("GeoTIFF ready for encoding."))

        if not os.path.exists(output_path):
            return None, tr("Failed to write rendered GeoTIFF.")

        QgsMessageLog.logMessage(
            "Rendered online layer to GeoTIFF: {}x{} pixels at {}".format(
                width, height, output_path),
            "AI Segmentation", level=Qgis.Info
        )
        return output_path, None

    except Exception as e:
        return None, tr(
            "Failed to render online layer: {error}"
        ).format(error=str(e))


def _terminate_process(process):
    """Safely terminate a subprocess: terminate -> wait -> kill."""
    if process is None:
        return
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    except Exception:
        pass


def encode_raster_to_features(
    raster_path: str,
    output_dir: str,
    checkpoint_path: str,
    layer_crs_wkt: Optional[str] = None,
    layer_extent: Optional[Tuple[float, float, float, float]] = None,
    visible_extent: Optional[Tuple[float, float, float, float]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[bool, str]:
    process = None
    stderr_file = None
    converted_path = None

    try:
        from .venv_manager import get_venv_python_path, get_venv_dir

        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        venv_python = get_venv_python_path(get_venv_dir())
        worker_script = os.path.join(plugin_dir, 'workers', 'encoding_worker.py')

        if not os.path.exists(venv_python):
            error_msg = f"Virtual environment Python not found: {venv_python}"
            QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.Critical)
            return False, error_msg

        if not os.path.exists(worker_script):
            error_msg = f"Worker script not found: {worker_script}"
            QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.Critical)
            return False, error_msg

        # Convert unsupported formats (ECW, MrSID) to GeoTIFF using QGIS's GDAL
        encoding_raster_path = raster_path
        if _needs_gdal_conversion(raster_path):
            converted_path, conv_error = _convert_with_gdal(
                raster_path, output_dir, progress_callback
            )
            if conv_error:
                return False, conv_error
            encoding_raster_path = converted_path

        config = {
            'raster_path': encoding_raster_path,
            'output_dir': output_dir,
            'checkpoint_path': checkpoint_path,
            'layer_crs_wkt': layer_crs_wkt,
            'layer_extent': layer_extent,
            'visible_extent': visible_extent,
        }

        QgsMessageLog.logMessage(
            f"Starting encoding worker subprocess: {venv_python}",
            "AI Segmentation",
            level=Qgis.Info
        )

        cmd = [venv_python, worker_script]

        env = get_clean_env_for_venv()
        subprocess_kwargs = get_subprocess_kwargs()

        # Create stderr temp file with fallback to DEVNULL
        try:
            stderr_file = tempfile.TemporaryFile(mode='w+', encoding='utf-8')
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Could not create stderr temp file, using DEVNULL: {e}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            stderr_file = None

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file if stderr_file is not None else subprocess.DEVNULL,
            text=True,
            env=env,
            **subprocess_kwargs
        )

        process.stdin.write(json.dumps(config))
        process.stdin.close()

        tiles_processed = 0
        start_time = time.monotonic()
        last_progress_time = time.monotonic()

        for line in process.stdout:
            # Global timeout check
            elapsed = time.monotonic() - start_time
            if elapsed > _ENCODING_GLOBAL_TIMEOUT:
                QgsMessageLog.logMessage(
                    "Encoding worker exceeded global timeout (45 min), terminating",
                    "AI Segmentation",
                    level=Qgis.Warning
                )
                _terminate_process(process)
                process = None
                return False, tr(
                    "Encoding timed out after 45 minutes. "
                    "Try reducing the image size or closing other applications."
                )

            # Stall detection: no progress for 5 minutes
            stall_elapsed = time.monotonic() - last_progress_time
            if stall_elapsed > _ENCODING_STALL_TIMEOUT:
                QgsMessageLog.logMessage(
                    "Encoding worker stalled (no output for 5 min), terminating",
                    "AI Segmentation",
                    level=Qgis.Warning
                )
                _terminate_process(process)
                process = None
                return False, tr(
                    "Encoding stalled (no progress for 5 minutes). "
                    "Try restarting QGIS and running again."
                )

            try:
                update = json.loads(line.strip())

                if update.get("type") == "progress":
                    last_progress_time = time.monotonic()
                    percent = update.get("percent", 0)
                    message = update.get("message", "")
                    try:
                        if progress_callback:
                            progress_callback(percent, message)
                    except Exception:
                        pass  # Don't let callback errors kill the reader loop

                elif update.get("type") == "success":
                    tiles_processed = update.get("tiles_processed", 0)
                    QgsMessageLog.logMessage(
                        f"Encoding completed successfully: {tiles_processed} tiles",
                        "AI Segmentation",
                        level=Qgis.Success
                    )

                elif update.get("type") == "error":
                    error_msg = update.get("message", "Unknown error")
                    QgsMessageLog.logMessage(
                        f"Encoding worker error: {error_msg}",
                        "AI Segmentation",
                        level=Qgis.Critical
                    )
                    return False, error_msg

            except json.JSONDecodeError:
                QgsMessageLog.logMessage(
                    f"Failed to parse worker output: {line}",
                    "AI Segmentation",
                    level=Qgis.Warning
                )

            try:
                cancelled = cancel_check and cancel_check()
            except Exception:
                cancelled = False
            if cancelled:
                QgsMessageLog.logMessage(
                    "Encoding cancelled by user, terminating worker",
                    "AI Segmentation",
                    level=Qgis.Info
                )
                _terminate_process(process)
                process = None
                return False, "Encoding cancelled by user"

        try:
            process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            QgsMessageLog.logMessage(
                "Encoding worker timed out (5 minutes), terminating",
                "AI Segmentation",
                level=Qgis.Warning
            )
            _terminate_process(process)
            process = None
            return False, "Encoding timed out"

        if process.returncode == 0:
            return True, f"Encoded {tiles_processed} tiles"
        else:
            stderr_output = ""
            if stderr_file is not None:
                try:
                    stderr_file.seek(0)
                    stderr_output = stderr_file.read()
                except Exception:
                    pass
            error_msg = f"Worker process failed with return code {process.returncode}"
            if stderr_output:
                error_msg += f"\nStderr: {stderr_output[:500]}"
            QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.Critical)

            # Detect OOM and show a clearer message
            oom_markers = ["not enough memory", "OutOfMemoryError", "MemoryError"]
            if any(m in stderr_output for m in oom_markers):
                return False, tr(
                    "Out of memory: your raster is too large for available RAM. "
                    "Try a smaller area or close other applications."
                )

            # Detect PROJ library conflict
            if "PROJ" in stderr_output or "proj_create" in stderr_output:
                return False, tr(
                    "PROJ library conflict detected. "
                    "Try updating QGIS to the latest version."
                )

            # Detect Windows DLL errors
            stderr_upper = stderr_output.upper()
            if "DLL" in stderr_upper or "WINERROR 1114" in stderr_upper:
                return False, tr(
                    "Windows DLL error detected. "
                    "Please install Visual C++ Redistributables: "
                    "https://aka.ms/vs/17/release/vc_redist.x64.exe"
                )

            # Detect Windows crash codes
            if process.returncode in _WINDOWS_ENCODING_CRASH_CODES:
                return False, tr(
                    "The encoding process crashed. "
                    "Try closing other applications, "
                    "reinstalling dependencies, "
                    "or running QGIS as administrator."
                )

            return False, error_msg

    except Exception as e:
        import traceback
        error_msg = f"Failed to start encoding worker: {str(e)}\n{traceback.format_exc()}"
        QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.Critical)
        return False, str(e)

    finally:
        # Ensure process is always cleaned up
        if process is not None:
            _terminate_process(process)
        if stderr_file is not None:
            try:
                stderr_file.close()
            except Exception:
                pass
        # Clean up temporary converted raster
        if converted_path and os.path.exists(converted_path):
            try:
                os.remove(converted_path)
            except Exception:
                pass
