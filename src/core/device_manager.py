from __future__ import annotations

import os
import sys

from qgis.core import Qgis, QgsMessageLog

from .venv_manager import ensure_venv_packages_available


_probe: dict = {"device": None, "info": None}


def probe_inprocess_torch_device():
    if _probe["device"] is not None:
        return _probe["device"]





    ensure_venv_packages_available()

    try:
        import torch  # noqa: F811
    except (OSError, ImportError) as e:



        if sys.platform == "win32" and (
                "shm.dll" in str(e) or "DLL" in str(e).upper()):
            error_msg = (
                "PyTorch DLL loading failed on Windows. "
                "This usually means Visual C++ Redistributables are missing. "
                "Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                f"Error: {str(e)}"
            )
            QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.MessageLevel.Critical)
            _probe["device"] = None
            _probe["info"] = "Error: PyTorch DLL failed"
            raise RuntimeError(error_msg) from e
        if not isinstance(e, ImportError):
            raise
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

                _probe["device"] = torch.device("mps")
                _probe["info"] = "Apple Silicon (MPS)"
                _configure_mps_optimizations()
                QgsMessageLog.logMessage(
                    "Using MPS acceleration (Apple Silicon)",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info
                )
                return _probe["device"]
        except Exception as e:
            QgsMessageLog.logMessage(
                f"MPS check failed: {e}, falling back to CPU",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )

    _probe["device"] = torch.device("cpu")
    _probe["info"] = f"CPU ({os.cpu_count()} cores)"
    _configure_cpu_optimizations()
    QgsMessageLog.logMessage(
        "Using CPU inference", "AI Segmentation", level=Qgis.MessageLevel.Info)
    return _probe["device"]


def _configure_mps_optimizations():
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def _configure_cpu_optimizations():
    import torch

    num_cores = os.cpu_count() or 4



    optimal_threads = max(1, min(num_cores - 1, max(2, num_cores // 2)))

    torch.set_num_threads(optimal_threads)

    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(2, optimal_threads // 2))
        except RuntimeError:
            pass

    QgsMessageLog.logMessage(
        f"CPU optimizations: {optimal_threads} threads",
        "AI Segmentation",
        level=Qgis.MessageLevel.Info
    )


def get_device_info() -> str:
    if _probe["info"] is None:
        probe_inprocess_torch_device()
    return _probe["info"] or "Unknown"
