"""Central registry of all supported segmentation models.

Each model entry contains metadata needed for download, loading,
encoding, and prediction. Helper functions provide access by model_id.
"""

import os

from .checkpoint_manager import get_checkpoints_dir


MODEL_REGISTRY = {
    "sam_vit_b": {
        "family": "sam1",
        "display_name": "SAM ViT-B",
        "registry_key": "vit_b",
        "feature_suffix": "vit_b",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
        "size_mb": 375,
        # SHA256 hash for checkpoint verification (not a secret - public checksum)
        "sha256": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",  # noqa: S105  # pragma: allowlist secret
        # SAM1 ViT-B/ViT-L share decoder architecture parameters
        "prompt_embed_dim": 256,
        "image_size": 1024,
        "vit_patch_size": 16,
    },
    "sam_vit_l": {
        "family": "sam1",
        "display_name": "SAM ViT-L",
        "registry_key": "vit_l",
        "feature_suffix": "vit_l",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
        "size_mb": 1200,
        "sha256": None,  # Will verify on first download or skip
        "prompt_embed_dim": 256,
        "image_size": 1024,
        "vit_patch_size": 16,
    },
    "sam2_large": {
        "family": "sam2",
        "display_name": "SAM2 Large",
        "feature_suffix": "sam2_large",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "filename": "sam2.1_hiera_large.pt",
        "size_mb": 225,
        "sha256": None,  # Will verify on first download or skip
        "sam2_model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
    },
}


def get_model_info(model_id):
    """Return the registry dict for a model_id, or None if unknown."""
    return MODEL_REGISTRY.get(model_id)


def get_all_model_ids():
    """Return list of all model IDs in display order."""
    return list(MODEL_REGISTRY.keys())


def get_checkpoint_path(model_id):
    """Return the full path to a model's checkpoint file."""
    info = MODEL_REGISTRY.get(model_id)
    if info is None:
        return None
    return os.path.join(get_checkpoints_dir(), info["filename"])


def model_checkpoint_exists(model_id):
    """Check if a model's checkpoint file exists on disk."""
    path = get_checkpoint_path(model_id)
    if path is None:
        return False
    return os.path.exists(path)


def get_installed_models():
    """Return list of model_ids whose checkpoints exist on disk."""
    return [
        model_id for model_id in MODEL_REGISTRY
        if model_checkpoint_exists(model_id)
    ]


def get_display_name_with_size(model_id):
    """Return e.g. 'SAM ViT-B (~375MB)' for UI display."""
    info = MODEL_REGISTRY.get(model_id)
    if info is None:
        return model_id
    size = info["size_mb"]
    if size >= 1000:
        size_str = "{:.1f}GB".format(size / 1000)
    else:
        size_str = "{}MB".format(size)
    return "{} (~{})".format(info["display_name"], size_str)
