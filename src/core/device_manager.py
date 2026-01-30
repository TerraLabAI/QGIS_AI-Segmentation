import sys
import os
from typing import Optional, TYPE_CHECKING
from qgis.core import QgsMessageLog, Qgis

from .venv_manager import ensure_venv_packages_available
ensure_venv_packages_available()

if TYPE_CHECKING:
    import torch

_cached_device = None
_device_info = None


def get_optimal_device() -> "torch.device":
    global _cached_device, _device_info

    if _cached_device is not None:
        return _cached_device

    import torch

    if torch.cuda.is_available():
        _cached_device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        _device_info = f"NVIDIA GPU ({gpu_name}, {gpu_memory:.1f}GB)"
        _configure_cuda_optimizations()
        QgsMessageLog.logMessage(
            f"Using CUDA acceleration: {gpu_name}",
            "AI Segmentation",
            level=Qgis.Info
        )
        return _cached_device

    if sys.platform == "darwin":
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                _cached_device = torch.device("mps")
                _device_info = "Apple Silicon GPU (MPS)"
                _configure_mps_optimizations()
                QgsMessageLog.logMessage(
                    "Using MPS acceleration (Apple Silicon GPU)",
                    "AI Segmentation",
                    level=Qgis.Info
                )
                return _cached_device
        except Exception as e:
            QgsMessageLog.logMessage(
                f"MPS check failed: {e}",
                "AI Segmentation",
                level=Qgis.Warning
            )

    _cached_device = torch.device("cpu")
    _device_info = f"CPU ({os.cpu_count()} cores)"
    _configure_cpu_optimizations()
    QgsMessageLog.logMessage(
        "Using CPU (no GPU acceleration available)",
        "AI Segmentation",
        level=Qgis.Info
    )
    return _cached_device


def _configure_cuda_optimizations():
    import torch

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

    QgsMessageLog.logMessage(
        "CUDA optimizations enabled: cudnn.benchmark=True",
        "AI Segmentation",
        level=Qgis.Info
    )


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

    if hasattr(torch, 'set_num_interop_threads'):
        torch.set_num_interop_threads(max(2, optimal_threads // 2))

    QgsMessageLog.logMessage(
        f"CPU optimizations: {optimal_threads} threads",
        "AI Segmentation",
        level=Qgis.Info
    )


def get_device_info() -> str:
    if _device_info is None:
        get_optimal_device()
    return _device_info or "Unknown"


def is_mps_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import torch
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    except Exception:
        return False


def is_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def get_device_capabilities() -> dict:
    import torch

    caps = {
        "platform": sys.platform,
        "mps_available": False,
        "mps_built": False,
        "cuda_available": False,
        "cpu_cores": os.cpu_count() or 1,
        "recommended_device": "cpu",
        "pytorch_version": torch.__version__,
    }

    if sys.platform == "darwin":
        try:
            caps["mps_built"] = torch.backends.mps.is_built()
            caps["mps_available"] = torch.backends.mps.is_available()
            if caps["mps_available"]:
                caps["recommended_device"] = "mps"
        except Exception:
            pass

    caps["cuda_available"] = torch.cuda.is_available()
    if caps["cuda_available"]:
        caps["cuda_device_count"] = torch.cuda.device_count()
        caps["cuda_device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        caps["cuda_memory_gb"] = props.total_memory / (1024**3)
        caps["cuda_compute_capability"] = f"{props.major}.{props.minor}"
        caps["recommended_device"] = "cuda"

    return caps


def reset_device_cache():
    global _cached_device, _device_info
    _cached_device = None
    _device_info = None


def move_to_device(tensor_or_model, device: Optional["torch.device"] = None):
    if device is None:
        device = get_optimal_device()
    return tensor_or_model.to(device)


def synchronize_device(device: Optional["torch.device"] = None):
    import torch

    if device is None:
        device = get_optimal_device()

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def clear_gpu_memory():
    import torch
    import gc

    gc.collect()

    device = get_optimal_device()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        QgsMessageLog.logMessage(
            "CUDA memory cache cleared",
            "AI Segmentation",
            level=Qgis.Info
        )
    elif device.type == "mps":
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        QgsMessageLog.logMessage(
            "MPS memory cache cleared",
            "AI Segmentation",
            level=Qgis.Info
        )


def get_gpu_memory_usage() -> Optional[dict]:
    import torch

    device = get_optimal_device()

    if device.type == "cuda":
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        }
    return None
