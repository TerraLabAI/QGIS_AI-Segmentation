from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional


@dataclass
class ModelConfig:
    model_id: str
    display_name: str
    description: str

    backend: str = "onnx"

    huggingface_repo: str = ""
    encoder_file: str = ""
    decoder_file: str = ""

    ultralytics_model: str = ""

    encoder_size_mb: int = 0
    decoder_size_mb: int = 0
    total_size_mb_override: int = 0

    input_size: int = 1024
    feature_shape: Tuple[int, int, int, int] = (1, 256, 64, 64)
    mask_input_shape: Tuple[int, int, int, int] = (1, 1, 256, 256)

    encoder_input_name: str = ""
    encoder_output_names: Dict[str, str] = field(default_factory=dict)
    decoder_input_names: Dict[str, str] = field(default_factory=dict)
    decoder_output_names: Dict[str, str] = field(default_factory=dict)
    decoder_input_ranks: Dict[str, int] = field(default_factory=dict)

    is_sam2: bool = False
    mask_threshold: float = 0.0
    use_sam2_preprocessing: bool = False
    use_sam2_coord_transform: bool = False
    recommended_tier: str = "low"

    @property
    def total_size_mb(self) -> int:
        if self.total_size_mb_override > 0:
            return self.total_size_mb_override
        return self.encoder_size_mb + self.decoder_size_mb

    @property
    def local_encoder_name(self) -> str:
        return "encoder.onnx"

    @property
    def local_decoder_name(self) -> str:
        return "decoder.onnx"

    @property
    def is_ultralytics(self) -> bool:
        return self.backend == "ultralytics"

    @property
    def is_onnx(self) -> bool:
        return self.backend == "onnx"



SAM_VIT_B = ModelConfig(
    model_id="sam_vit_b",
    display_name="Fast (SAM ViT-B)",
    description="Original SAM model, quantized. Good balance of speed and quality.",
    backend="onnx",

    huggingface_repo="visheratin/segment-anything-vit-b",
    encoder_file="encoder-quant.onnx",
    decoder_file="decoder-quant.onnx",

    encoder_size_mb=100,
    decoder_size_mb=9,

    input_size=1024,
    feature_shape=(1, 256, 64, 64),
    mask_input_shape=(1, 1, 256, 256),

    encoder_input_name="x",
    encoder_output_names={
        "image_embeddings": "image_embeddings",
    },

    decoder_input_names={
        "image_embeddings": "image_embeddings",
        "point_coords": "point_coords",
        "point_labels": "point_labels",
        "mask_input": "mask_input",
        "has_mask_input": "has_mask_input",
        "orig_im_size": "orig_im_size",
    },
    decoder_output_names={
        "masks": "masks",
        "iou_predictions": "iou_predictions",
    },

    decoder_input_ranks={
        "image_embeddings": 4,
        "point_coords": 3,
        "point_labels": 2,
        "mask_input": 4,
        "has_mask_input": 1,
        "orig_im_size": 1,
    },

    is_sam2=False,
)

SAM2_BASE_PLUS = ModelConfig(
    model_id="sam2_base_plus",
    display_name="Balanced (SAM2 Base)",
    description="SAM2 Base model via Ultralytics. Better quality, requires PyTorch.",
    backend="ultralytics",

    ultralytics_model="sam2_b.pt",
    total_size_mb_override=155,

    input_size=1024,
    is_sam2=True,
    mask_threshold=0.0,
    use_sam2_preprocessing=True,
    use_sam2_coord_transform=True,
    recommended_tier="medium",
)

SAM2_LARGE = ModelConfig(
    model_id="sam2_large",
    display_name="Precise (SAM2 Large)",
    description="SAM2 Large model via Ultralytics. Highest quality, requires PyTorch.",
    backend="ultralytics",

    ultralytics_model="sam2_l.pt",
    total_size_mb_override=390,

    input_size=1024,
    is_sam2=True,
    mask_threshold=0.0,
    use_sam2_preprocessing=True,
    use_sam2_coord_transform=True,
    recommended_tier="high",
)



MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "sam_vit_b": SAM_VIT_B,
    "sam2_base_plus": SAM2_BASE_PLUS,
    "sam2_large": SAM2_LARGE,
}

MODEL_ORDER: List[str] = ["sam_vit_b", "sam2_base_plus", "sam2_large"]

DEFAULT_MODEL_ID = "sam_vit_b"


def get_model_config(model_id: str) -> ModelConfig:
    
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_id}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_id]


def get_all_models() -> List[ModelConfig]:
    
    return [MODEL_REGISTRY[model_id] for model_id in MODEL_ORDER]


def get_model_ids() -> List[str]:

    return MODEL_ORDER.copy()


def get_recommended_model(hardware_tier: str) -> str:
    tier_map = {"high": "sam2_large", "medium": "sam2_base_plus", "low": "sam_vit_b"}
    return tier_map.get(hardware_tier, "sam_vit_b")


def model_requires_sam2_deps(model_id: str) -> bool:
    config = get_model_config(model_id)
    return config.is_ultralytics


def get_ultralytics_models() -> List[ModelConfig]:
    return [config for config in get_all_models() if config.is_ultralytics]


def get_onnx_models() -> List[ModelConfig]:
    return [config for config in get_all_models() if config.is_onnx]


def is_ultralytics_model_downloaded(model_id: str, verbose: bool = False) -> bool:
    from pathlib import Path
    from qgis.core import QgsMessageLog, Qgis

    config = get_model_config(model_id)
    if not config.is_ultralytics:
        return False

    model_name = config.ultralytics_model
    expected_size = config.total_size_mb * 1024 * 1024 * 0.8

    plugin_dir = Path(__file__).parent.parent
    possible_paths = [
        plugin_dir / model_name,
        plugin_dir / "models" / model_name,
        Path.home() / ".cache" / "ultralytics" / model_name,
    ]

    for path in possible_paths:
        if path.exists() and path.stat().st_size > expected_size:
            if verbose:
                QgsMessageLog.logMessage(
                    f"[MODEL_CHECK] {model_id}: found at {path}",
                    "AI Segmentation",
                    level=Qgis.Info
                )
            return True

    if verbose:
        QgsMessageLog.logMessage(
            f"[MODEL_CHECK] {model_id}: NOT found",
            "AI Segmentation",
            level=Qgis.Info
        )
    return False


def can_use_ultralytics_model(model_id: str) -> bool:
    config = get_model_config(model_id)
    if not config.is_ultralytics:
        return False

    if not is_ultralytics_model_downloaded(model_id):
        return False

    try:
        import torch
        from ultralytics import SAM
        return True
    except ImportError:
        return False


def get_ultralytics_model_path(model_id: str) -> Optional[str]:
    from pathlib import Path

    config = get_model_config(model_id)
    if not config.is_ultralytics:
        return None

    model_name = config.ultralytics_model
    expected_size = config.total_size_mb * 1024 * 1024 * 0.8

    plugin_dir = Path(__file__).parent.parent
    possible_paths = [
        plugin_dir / model_name,
        plugin_dir / "models" / model_name,
        Path.home() / ".cache" / "ultralytics" / model_name,
    ]

    for path in possible_paths:
        if path.exists() and path.stat().st_size > expected_size:
            return str(path)

    return None
