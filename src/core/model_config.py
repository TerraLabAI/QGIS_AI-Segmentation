








from __future__ import annotations

import platform
import subprocess  # nosec B404
import sys


def _is_rosetta() -> bool:






    if sys.platform != "darwin" or platform.machine() != "x86_64":
        return False
    try:
        result = subprocess.run(  # nosec B603
            ["sysctl", "-n", "sysctl.proc_translated"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=5,
        )
        return result.stdout.strip() == "1"
    except Exception:
        return False


IS_ROSETTA = _is_rosetta()

_IS_MACOS_X86 = sys.platform == "darwin" and platform.machine() == "x86_64" and not IS_ROSETTA


USE_SAM2 = (sys.version_info >= (3, 10) or IS_ROSETTA) and not _IS_MACOS_X86






MACOS_X86_NO_LOCAL_INFERENCE = _IS_MACOS_X86 and sys.version_info >= (3, 13)

if USE_SAM2:
    SAM_PACKAGE = ("sam2", ">=1.0")
    TORCH_MIN = ">=2.5.1"
    TORCHVISION_MIN = ">=0.20.1"
    CHECKPOINT_URL = (
        "https://dl.fbaipublicfiles.com/segment_anything_2"
        "/092824/sam2.1_hiera_base_plus.pt"
    )
    CHECKPOINT_FILENAME = "sam2.1_hiera_base_plus.pt"

    CHECKPOINT_SHA256 = "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"  # noqa: S105, E501  # pragma: allowlist secret
else:
    SAM_PACKAGE = ("segment-anything", ">=1.0")
    if _IS_MACOS_X86:
        TORCH_MIN = ">=2.0.0,<=2.2.2"
        TORCHVISION_MIN = ">=0.15.0,<=0.17.2"
    elif sys.version_info < (3, 10):

        TORCH_MIN = ">=2.0.0,<2.6.0"
        TORCHVISION_MIN = ">=0.15.0,<0.21.0"
    else:
        TORCH_MIN = ">=2.0.0"
        TORCHVISION_MIN = ">=0.15.0"
    CHECKPOINT_URL = (
        "https://dl.fbaipublicfiles.com/segment_anything"
        "/sam_vit_b_01ec64.pth"
    )
    CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"

    CHECKPOINT_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"  # noqa: S105, E501  # pragma: allowlist secret


def macos_intel_unsupported_python() -> bool:











    return _IS_MACOS_X86 and sys.version_info >= (3, 13)






if sys.platform == "win32":
    TORCH_WINDOWS_FALLBACK = "==2.5.1"
    TORCHVISION_WINDOWS_FALLBACK = "==0.20.1"
else:
    TORCH_WINDOWS_FALLBACK = None
    TORCHVISION_WINDOWS_FALLBACK = None
