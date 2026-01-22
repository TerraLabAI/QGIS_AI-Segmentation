
import platform
import subprocess
from typing import List, Tuple, Dict, Any, Union, Optional

from qgis.core import QgsMessageLog, Qgis

ProviderConfig = Union[str, Tuple[str, Dict[str, Any]]]

def get_force_cpu_from_settings() -> bool:
    try:
        from .debug_settings import get_settings
        return get_settings().force_cpu
    except Exception:
        return True


def get_available_providers() -> List[str]:
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except ImportError:
        return ['CPUExecutionProvider']


def get_optimal_providers(force_cpu: bool = None) -> List[ProviderConfig]:
    if force_cpu is None:
        force_cpu = get_force_cpu_from_settings()

    if force_cpu:
        QgsMessageLog.logMessage(
            "Using CPU provider (forced)",
            "AI Segmentation",
            level=Qgis.Info
        )
        return ['CPUExecutionProvider']

    available = set(get_available_providers())
    providers: List[ProviderConfig] = []

    if 'CUDAExecutionProvider' in available:
        providers.append(('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
        }))

    if platform.system() == 'Darwin' and 'CoreMLExecutionProvider' in available:
        providers.append(('CoreMLExecutionProvider', {
            'MLComputeUnits': 'ALL',
        }))

    providers.append('CPUExecutionProvider')
    return providers


def is_gpu_available() -> bool:
    available = get_available_providers()
    return 'CUDAExecutionProvider' in available or 'CoreMLExecutionProvider' in available


def get_hardware_tier() -> str:
    available = get_available_providers()
    if 'CUDAExecutionProvider' in available:
        return "high"
    if 'CoreMLExecutionProvider' in available:
        return "medium"
    return "low"


def get_active_provider_name(session) -> str:
    try:
        providers = session.get_providers()
        if providers:
            return providers[0]
    except Exception:
        pass
    return "Unknown"


def get_gpu_install_suggestion() -> Optional[str]:
    if is_gpu_available():
        return None
    if _nvidia_gpu_exists():
        return "pip install onnxruntime-gpu"
    return None


def _nvidia_gpu_exists() -> bool:
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def log_provider_info():
    available = get_available_providers()
    tier = get_hardware_tier()

    QgsMessageLog.logMessage(
        f"Available ONNX providers: {available}",
        "AI Segmentation",
        level=Qgis.Info
    )
    QgsMessageLog.logMessage(
        f"Hardware tier: {tier}",
        "AI Segmentation",
        level=Qgis.Info
    )

    suggestion = get_gpu_install_suggestion()
    if suggestion:
        QgsMessageLog.logMessage(
            f"GPU detected but onnxruntime-gpu not installed. For faster processing: {suggestion}",
            "AI Segmentation",
            level=Qgis.Info
        )
