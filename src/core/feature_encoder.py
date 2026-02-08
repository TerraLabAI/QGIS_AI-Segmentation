import os
import sys
import json
import subprocess
import tempfile
from typing import Tuple, Optional, Callable

from qgis.core import QgsMessageLog, Qgis


def _get_clean_env_for_venv() -> dict:
    env = os.environ.copy()
    vars_to_remove = [
        'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV',
        'QGIS_PREFIX_PATH', 'QGIS_PLUGINPATH',
    ]
    for var in vars_to_remove:
        env.pop(var, None)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _get_subprocess_kwargs() -> dict:
    kwargs = {}
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs['startupinfo'] = startupinfo
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
    return kwargs


def encode_raster_to_features(
    raster_path: str,
    output_dir: str,
    checkpoint_path: str,
    layer_crs_wkt: Optional[str] = None,
    layer_extent: Optional[Tuple[float, float, float, float]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[bool, str]:
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

        config = {
            'raster_path': raster_path,
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

        env = _get_clean_env_for_venv()
        subprocess_kwargs = _get_subprocess_kwargs()

        stderr_file = tempfile.TemporaryFile(mode='w+', encoding='utf-8')
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            env=env,
            **subprocess_kwargs
        )

        process.stdin.write(json.dumps(config))
        process.stdin.close()

        tiles_processed = 0

        for line in process.stdout:
            try:
                update = json.loads(line.strip())

                if update.get("type") == "progress":
                    percent = update.get("percent", 0)
                    message = update.get("message", "")
                    if progress_callback:
                        progress_callback(percent, message)

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

            if cancel_check and cancel_check():
                QgsMessageLog.logMessage(
                    "Encoding cancelled by user, terminating worker",
                    "AI Segmentation",
                    level=Qgis.Info
                )
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                stderr_file.close()
                return False, "Encoding cancelled by user"

        try:
            process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            QgsMessageLog.logMessage(
                "Encoding worker timed out (5 minutes), terminating",
                "AI Segmentation",
                level=Qgis.Warning
            )
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stderr_file.close()
            return False, "Encoding timed out"

        if process.returncode == 0:
            stderr_file.close()
            return True, f"Encoded {tiles_processed} tiles"
        else:
            stderr_file.seek(0)
            stderr_output = stderr_file.read()
            stderr_file.close()
            error_msg = f"Worker process failed with return code {process.returncode}"
            if stderr_output:
                error_msg += f"\nStderr: {stderr_output[:500]}"
            QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.Critical)
            return False, error_msg

    except Exception as e:
        import traceback
        error_msg = f"Failed to start encoding worker: {str(e)}\n{traceback.format_exc()}"
        QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.Critical)
        return False, str(e)
