"""Central configuration for version-dependent SAM model constants.

SAM2 (sam2 package) requires Python >= 3.10 and is unavailable on macOS
x86_64 (PyTorch dropped Intel Mac wheels after 2.2.2). On unsupported
platforms or Python 3.9, we fall back to SAM1 (segment-anything / ViT-B).
"""
import platform
import sys

_IS_MACOS_X86 = (sys.platform == "darwin" and platform.machine() == "x86_64")

USE_SAM2 = sys.version_info >= (3, 10) and not _IS_MACOS_X86

if USE_SAM2:
    SAM_PACKAGE = ("sam2", ">=1.0")
    TORCH_MIN = ">=2.5.1"
    TORCHVISION_MIN = ">=0.20.1"
    CHECKPOINT_URL = (
        "https://dl.fbaipublicfiles.com/segment_anything_2"
        "/092824/sam2.1_hiera_base_plus.pt"
    )
    CHECKPOINT_FILENAME = "sam2.1_hiera_base_plus.pt"
    # SHA256 hash for checkpoint verification (not a secret)
    CHECKPOINT_SHA256 = ""  # noqa: S105  # pragma: allowlist secret
    CHECKPOINT_SIZE_LABEL = "~323MB"
    MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
else:
    SAM_PACKAGE = ("segment-anything", ">=1.0")
    if _IS_MACOS_X86:
        TORCH_MIN = ">=2.0.0,<=2.2.2"
        TORCHVISION_MIN = ">=0.15.0,<=0.17.2"
    else:
        TORCH_MIN = ">=2.0.0"
        TORCHVISION_MIN = ">=0.15.0"
    CHECKPOINT_URL = (
        "https://dl.fbaipublicfiles.com/segment_anything"
        "/sam_vit_b_01ec64.pth"
    )
    CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"
    # SHA256 hash for checkpoint verification (not a secret)
    CHECKPOINT_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"  # noqa: S105, E501  # pragma: allowlist secret
    CHECKPOINT_SIZE_LABEL = "~375MB"
    MODEL_CFG = None  # SAM1 uses registry, no config file
