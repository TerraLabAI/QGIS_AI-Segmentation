

from dataclasses import dataclass
from typing import Dict, Tuple, List


@dataclass
class ModelConfig:

    model_id: str
    display_name: str
    description: str

    huggingface_repo: str
    encoder_file: str
    decoder_file: str

    encoder_size_mb: int
    decoder_size_mb: int

    input_size: int
    feature_shape: Tuple[int, int, int, int]
    mask_input_shape: Tuple[int, int, int, int]

    encoder_input_name: str
    encoder_output_names: Dict[str, str]

    decoder_input_names: Dict[str, str]
    decoder_output_names: Dict[str, str]

    decoder_input_ranks: Dict[str, int]

    is_sam2: bool = False

    mask_threshold: float = 0.0

    use_sam2_preprocessing: bool = False

    use_sam2_coord_transform: bool = False

    @property
    def total_size_mb(self) -> int:
        
        return self.encoder_size_mb + self.decoder_size_mb

    @property
    def local_encoder_name(self) -> str:
        
        return "encoder.onnx"

    @property
    def local_decoder_name(self) -> str:
        
        return "decoder.onnx"



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
        "image_embeddings": 4,  # (1, 256, 64, 64)
        "point_coords": 3,      # (1, N, 2)
        "point_labels": 2,      # (1, N)
        "mask_input": 4,        # (1, 1, 256, 256)
        "has_mask_input": 1,    # (1,)
        "orig_im_size": 1,      # (2,)
    },

    is_sam2=False,
)

SAM2_BASE_PLUS = ModelConfig(
    model_id="sam2_base_plus",
    display_name="Balanced (SAM2 Base+)",
    description="SAM2 Hiera Base+. Better quality than SAM ViT-B.",

    huggingface_repo="vietanhdev/segment-anything-2-onnx-models",
    encoder_file="sam2_hiera_base_plus.encoder.onnx",
    decoder_file="sam2_hiera_base_plus.decoder.onnx",

    encoder_size_mb=340,
    decoder_size_mb=21,

    input_size=1024,
    feature_shape=(1, 256, 64, 64),
    mask_input_shape=(1, 1, 256, 256),

    encoder_input_name="image",
    encoder_output_names={
        "high_res_feats_0": "high_res_feats_0",
        "high_res_feats_1": "high_res_feats_1",
        "image_embed": "image_embed",
    },

    decoder_input_names={
        "image_embed": "image_embed",
        "high_res_feats_0": "high_res_feats_0",
        "high_res_feats_1": "high_res_feats_1",
        "point_coords": "point_coords",
        "point_labels": "point_labels",
        "mask_input": "mask_input",
        "has_mask_input": "has_mask_input",
    },
    decoder_output_names={
        "masks": "masks",
        "iou_predictions": "scores",
    },

    decoder_input_ranks={
        "image_embed": 4,
        "high_res_feats_0": 4,
        "high_res_feats_1": 4,
        "point_coords": 3,
        "point_labels": 2,
        "mask_input": 4,
        "has_mask_input": 1,
    },

    is_sam2=True,
    mask_threshold=0.0,
    use_sam2_preprocessing=True,
    use_sam2_coord_transform=True,
)

SAM2_LARGE = ModelConfig(
    model_id="sam2_large",
    display_name="Precise (SAM2 Large)",
    description="SAM2 Hiera Large. Highest quality, slower.",

    huggingface_repo="vietanhdev/segment-anything-2-onnx-models",
    encoder_file="sam2_hiera_large.encoder.onnx",
    decoder_file="sam2_hiera_large.decoder.onnx",

    encoder_size_mb=889,
    decoder_size_mb=21,

    input_size=1024,
    feature_shape=(1, 256, 64, 64),
    mask_input_shape=(1, 1, 256, 256),

    encoder_input_name="image",
    encoder_output_names={
        "high_res_feats_0": "high_res_feats_0",
        "high_res_feats_1": "high_res_feats_1",
        "image_embed": "image_embed",
    },

    decoder_input_names={
        "image_embed": "image_embed",
        "high_res_feats_0": "high_res_feats_0",
        "high_res_feats_1": "high_res_feats_1",
        "point_coords": "point_coords",
        "point_labels": "point_labels",
        "mask_input": "mask_input",
        "has_mask_input": "has_mask_input",
    },
    decoder_output_names={
        "masks": "masks",
        "iou_predictions": "scores",
    },

    decoder_input_ranks={
        "image_embed": 4,
        "high_res_feats_0": 4,
        "high_res_feats_1": 4,
        "point_coords": 3,
        "point_labels": 2,
        "mask_input": 4,
        "has_mask_input": 1,
    },

    is_sam2=True,
    mask_threshold=0.0,
    use_sam2_preprocessing=True,
    use_sam2_coord_transform=True,
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
