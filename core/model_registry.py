"""
Model Registry for AI Segmentation

Centralized configuration for all supported segmentation models.
Each model has specific tensor names, shapes, and download sources.

Supported Models:
- SAM ViT-B (Fast): Original SAM, quantized for CPU
- SAM2 Hiera Base+ (Balanced): SAM2 with better quality
- SAM2 Hiera Large (Precise): SAM2 highest quality
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List


@dataclass
class ModelConfig:
    """Configuration for a single segmentation model."""

    # Identity
    model_id: str
    display_name: str
    description: str

    # HuggingFace source
    huggingface_repo: str
    encoder_file: str
    decoder_file: str

    # File sizes (for download progress)
    encoder_size_mb: int
    decoder_size_mb: int

    # Model architecture parameters
    input_size: int  # SAM input size (typically 1024)
    feature_shape: Tuple[int, int, int, int]  # (batch, channels, height, width)
    mask_input_shape: Tuple[int, int, int, int]  # Shape for mask input tensor

    # Tensor name mappings (different between SAM and SAM2)
    # Maps logical name -> actual ONNX tensor name
    encoder_input_name: str
    encoder_output_name: str

    # Decoder tensor names - key is logical name, value is ONNX name
    decoder_input_names: Dict[str, str]
    decoder_output_names: Dict[str, str]

    # Expected tensor ranks for validation
    decoder_input_ranks: Dict[str, int]

    @property
    def total_size_mb(self) -> int:
        """Total download size in MB."""
        return self.encoder_size_mb + self.decoder_size_mb

    @property
    def local_encoder_name(self) -> str:
        """Local filename for encoder (standardized)."""
        return "encoder.onnx"

    @property
    def local_decoder_name(self) -> str:
        """Local filename for decoder (standardized)."""
        return "decoder.onnx"


# ============================================================================
# Model Configurations
# ============================================================================

SAM_VIT_B = ModelConfig(
    model_id="sam_vit_b",
    display_name="Fast (SAM ViT-B)",
    description="Original SAM model, quantized. Good balance of speed and quality.",

    huggingface_repo="visheratin/segment-anything-vit-b",
    encoder_file="encoder-quant.onnx",
    decoder_file="decoder-quant.onnx",

    encoder_size_mb=100,
    decoder_size_mb=9,

    input_size=1024,
    feature_shape=(1, 256, 64, 64),
    mask_input_shape=(1, 1, 256, 256),

    encoder_input_name="input_image",
    encoder_output_name="image_embeddings",

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
        "image_embeddings": 4,  # (1, 256, 64, 64)
        "point_coords": 3,      # (1, N, 2)
        "point_labels": 2,      # (1, N)
        "mask_input": 4,        # (1, 1, 256, 256)
        "has_mask_input": 1,    # (1,)
        "orig_im_size": 1,      # (2,)
    },
)

SAM2_BASE_PLUS = ModelConfig(
    model_id="sam2_base_plus",
    display_name="Balanced (SAM2 Base+)",
    description="SAM2 Hiera Base+. Better quality than SAM ViT-B.",

    huggingface_repo="shubham0204/sam2-onnx-models",
    encoder_file="sam2_hiera_base_plus_encoder.onnx",
    decoder_file="sam2_hiera_base_plus_decoder.onnx",

    encoder_size_mb=277,
    decoder_size_mb=17,

    input_size=1024,
    feature_shape=(1, 256, 64, 64),
    mask_input_shape=(1, 1, 256, 256),

    encoder_input_name="image",
    encoder_output_name="image_embeddings",

    # SAM2 has slightly different tensor names
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
)

SAM2_LARGE = ModelConfig(
    model_id="sam2_large",
    display_name="Precise (SAM2 Large)",
    description="SAM2 Hiera Large. Highest quality, slower.",

    huggingface_repo="SharpAI/sam2-hiera-large-onnx",
    encoder_file="encoder.onnx",
    decoder_file="decoder.onnx",

    encoder_size_mb=889,
    decoder_size_mb=17,

    input_size=1024,
    feature_shape=(1, 256, 64, 64),
    mask_input_shape=(1, 1, 256, 256),

    encoder_input_name="image",
    encoder_output_name="image_embeddings",

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
)


# ============================================================================
# Registry
# ============================================================================

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "sam_vit_b": SAM_VIT_B,
    "sam2_base_plus": SAM2_BASE_PLUS,
    "sam2_large": SAM2_LARGE,
}

# Order for display in UI (top to bottom)
MODEL_ORDER: List[str] = ["sam_vit_b", "sam2_base_plus", "sam2_large"]

# Default model for new users
DEFAULT_MODEL_ID = "sam_vit_b"


def get_model_config(model_id: str) -> ModelConfig:
    """
    Get configuration for a specific model.

    Args:
        model_id: Model identifier (e.g., "sam_vit_b")

    Returns:
        ModelConfig for the specified model

    Raises:
        KeyError: If model_id is not found in registry
    """
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_id}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_id]


def get_all_models() -> List[ModelConfig]:
    """
    Get all available models in display order.

    Returns:
        List of ModelConfig objects
    """
    return [MODEL_REGISTRY[model_id] for model_id in MODEL_ORDER]


def get_model_ids() -> List[str]:
    """
    Get all model IDs in display order.

    Returns:
        List of model ID strings
    """
    return MODEL_ORDER.copy()
