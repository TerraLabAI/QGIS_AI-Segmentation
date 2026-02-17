import sys
import os
from qgis.core import QgsMessageLog, Qgis

from .venv_manager import ensure_venv_packages_available
ensure_venv_packages_available()

_cached_device = None
_device_info = None


def get_optimal_device() -> "torch.device":
    global _cached_device, _device_info

    if _cached_device is not None:
        return _cached_device

    try:
        import torch
    except OSError as e:
        # Windows DLL loading error (shm.dll, etc.)
        if "shm.dll" in str(e) or "DLL" in str(e).upper():
            error_msg = (
                "PyTorch DLL loading failed on Windows. "
                "This usually means Visual C++ Redistributables are missing. "
                "Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                "Error: {}".format(str(e))
            )
            QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.Critical)
            _cached_device = None
            _device_info = "Error: PyTorch DLL failed"
            raise RuntimeError(error_msg)
        else:
            raise
    except ImportError as e:
        error_msg = "Failed to import PyTorch: {}".format(str(e))
        QgsMessageLog.logMessage(error_msg, "AI Segmentation", level=Qgis.Critical)
        raise

    try:
        cuda_available = torch.cuda.is_available()
    except Exception as e:
        QgsMessageLog.logMessage(
            "torch.cuda.is_available() failed ({}), skipping CUDA".format(e),
            "AI Segmentation",
            level=Qgis.Warning
        )
        cuda_available = False

    if cuda_available:
        best_idx = -1
        best_mem = 0.0
        best_name = None

        try:
            count = torch.cuda.device_count()
            for i in range(count):
                try:
                    props = torch.cuda.get_device_properties(i)
                    mem_gb = props.total_memory / (1024**3)
                    if mem_gb >= 2.0 and mem_gb > best_mem:
                        best_mem = mem_gb
                        best_idx = i
                        best_name = props.name
                except Exception:
                    continue
        except Exception as e:
            QgsMessageLog.logMessage(
                "Cannot query CUDA device info ({}), falling back to CPU".format(e),
                "AI Segmentation",
                level=Qgis.Warning
            )

        if best_idx < 0:
            if best_mem > 0:
                QgsMessageLog.logMessage(
                    "No GPU with >=2GB memory found, falling back to CPU",
                    "AI Segmentation",
                    level=Qgis.Warning
                )
        else:
            # Test that GPU kernels actually work with a small allocation
            cuda_dev = "cuda:{}".format(best_idx)
            try:
                test = torch.zeros(1, device=cuda_dev)
                _ = test + 1
                torch.cuda.synchronize(best_idx)
                del test
                torch.cuda.empty_cache()

                _cached_device = torch.device(cuda_dev)
                _device_info = "NVIDIA GPU ({}, {:.1f}GB)".format(
                    best_name, best_mem)
                _configure_cuda_optimizations()
                QgsMessageLog.logMessage(
                    "Using CUDA acceleration: {} (device {})".format(
                        best_name, best_idx),
                    "AI Segmentation",
                    level=Qgis.Info
                )
                return _cached_device
            except RuntimeError as e:
                QgsMessageLog.logMessage(
                    "CUDA test failed ({}), falling back to CPU".format(str(e)),
                    "AI Segmentation",
                    level=Qgis.Warning
                )
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    if sys.platform == "darwin":
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                # Verify MPS actually works with a test allocation
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

    try:
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
    except Exception as e:
        QgsMessageLog.logMessage(
            "Failed to configure CUDA optimizations: {}".format(e),
            "AI Segmentation",
            level=Qgis.Warning
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
        try:
            torch.set_num_interop_threads(max(2, optimal_threads // 2))
        except RuntimeError:
            pass  # Already set or torch parallelism already started

    QgsMessageLog.logMessage(
        f"CPU optimizations: {optimal_threads} threads",
        "AI Segmentation",
        level=Qgis.Info
    )


def get_device_info() -> str:
    if _device_info is None:
        get_optimal_device()
    return _device_info or "Unknown"
