import os
import json
import time
import subprocess
import tempfile
from typing import Tuple, Optional, Callable

from qgis.core import QgsMessageLog, Qgis

from .subprocess_utils import get_clean_env_for_venv, get_subprocess_kwargs
from .i18n import tr

# Global timeout for the encoding stdout reading loop (45 minutes)
_ENCODING_GLOBAL_TIMEOUT = 2700

# Raster formats that require GDAL conversion (not supported by pip-installed rasterio)
_GDAL_ONLY_FORMATS = {'.ecw', '.sid'}


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
            'layer_extent': layer_extent
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
                return False, "Encoding timed out (45 minutes)"

            try:
                update = json.loads(line.strip())

                if update.get("type") == "progress":
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
                return False, (
                    "Out of memory: your raster is too large for available RAM. "
                    "Try a smaller area or close other applications."
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
