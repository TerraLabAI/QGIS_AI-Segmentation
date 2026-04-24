from __future__ import annotations

import os
import sys

from qgis.core import Qgis, QgsMessageLog

from .venv_manager import ensure_venv_packages_available

ensure_venv_packages_available()

_cached_device = None
_device_info = None


def get_optimal_device():  # -> torch.device
    global _cached_device, _device_info

    if _cached_device is not None:
        return _cached_device

    try:
        import torch  # noqa: F811
    except OSError as e:
        if "shm.dll" in str(e) or "DLL" in str(e).upper():
            error_msg = (
                "PyTorch DLL loading failed on Windows. "
                "This usually means Visual C++ Redistributables are missing. "
                "Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                f"Error: {str(e)}"
            )
            QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.MessageLevel.Critical)
            _cached_device = None
            _device_info = "Error: PyTorch DLL failed"
            raise RuntimeError(error_msg) from e
        raise
    except ImportError as e:
        error_msg = f"Failed to import PyTorch: {str(e)}"
        QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.MessageLevel.Critical)
        raise

    if sys.platform == "darwin":
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                test = torch.zeros(1, device="mps")
                _ = test + 1
                torch.mps.synchronize()
                del test

                _cached_device = torch.device("mps")
                _device_info = "Apple Silicon GPU (MPS)"
                _configure_mps_optimizations()
                QgsMessageLog.logMessage(
                    "Using MPS acceleration (Apple Silicon GPU)",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info
                )
                return _cached_device
        except Exception as e:
            QgsMessageLog.logMessage(
                f"MPS check failed: {e}, falling back to CPU",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )

    _cached_device = torch.device("cpu")
    _device_info = f"CPU ({os.cpu_count()} cores)"
    _configure_cpu_optimizations()
    QgsMessageLog.logMessage(
        "Using CPU inference", "AI Segmentation", level=Qgis.MessageLevel.Info)
    return _cached_device


def _configure_mps_optimizations():
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def _configure_cpu_optimizations():
    import torch

    num_cores = os.cpu_count() or 4

    if sys.platform == "darwin":
        optimal_threads = max(4, num_cores // 2)
    else:
        optimal_threads = num_cores

    torch.set_num_threads(optimal_threads)

    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(2, optimal_threads // 2))
        except RuntimeError:
            pass  # Already set or torch parallelism already started

    QgsMessageLog.logMessage(
        f"CPU optimizations: {optimal_threads} threads",
        "AI Segmentation",
        level=Qgis.MessageLevel.Info
    )


def get_device_info() -> str:
    if _device_info is None:
        get_optimal_device()
    return _device_info or "Unknown"
