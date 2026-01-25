import os
import hashlib
import urllib.request
from typing import Tuple, Optional, Callable

from qgis.core import QgsMessageLog, Qgis


CACHE_DIR = os.path.expanduser("~/.qgis_ai_segmentation")
CHECKPOINTS_DIR = os.path.join(CACHE_DIR, "checkpoints")
FEATURES_DIR = os.path.join(CACHE_DIR, "features")

SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"
SAM_CHECKPOINT_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"


def get_checkpoints_dir() -> str:
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    return CHECKPOINTS_DIR


def get_features_dir() -> str:
    os.makedirs(FEATURES_DIR, exist_ok=True)
    return FEATURES_DIR


def get_raster_features_dir(raster_path: str) -> str:
    import re

    # Get the raster filename without extension
    raster_filename = os.path.splitext(os.path.basename(raster_path))[0]

    # Sanitize: keep only alphanumeric, underscore, hyphen (replace others with underscore)
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raster_filename)
    # Limit length to avoid too long folder names
    sanitized_name = sanitized_name[:40]
    # Remove trailing underscores
    sanitized_name = sanitized_name.rstrip('_')

    # Add short hash suffix for uniqueness (based on full path)
    raster_hash = hashlib.md5(raster_path.encode()).hexdigest()[:8]

    # Combine: rastername_abc12345
    folder_name = f"{sanitized_name}_{raster_hash}"

    features_path = os.path.join(FEATURES_DIR, folder_name)
    os.makedirs(features_path, exist_ok=True)
    return features_path


def get_checkpoint_path() -> str:
    return os.path.join(get_checkpoints_dir(), SAM_CHECKPOINT_FILENAME)


def checkpoint_exists() -> bool:
    return os.path.exists(get_checkpoint_path())


def verify_checkpoint_hash(filepath: str) -> bool:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == SAM_CHECKPOINT_SHA256


def download_checkpoint(
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[bool, str]:
    checkpoint_path = get_checkpoint_path()

    if checkpoint_exists():
        QgsMessageLog.logMessage(
            "Checkpoint already exists, verifying...",
            "AI Segmentation",
            level=Qgis.Info
        )
        if verify_checkpoint_hash(checkpoint_path):
            return True, "Checkpoint verified"
        else:
            QgsMessageLog.logMessage(
                "Checkpoint hash mismatch, re-downloading...",
                "AI Segmentation",
                level=Qgis.Warning
            )
            os.remove(checkpoint_path)

    if progress_callback:
        progress_callback(0, f"Downloading SAM checkpoint (~375MB)...")

    try:
        temp_path = checkpoint_path + ".tmp"

        def _report_progress(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                percent = min(percent, 99)
                if progress_callback:
                    mb_downloaded = (count * block_size) / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    progress_callback(percent, f"Downloading: {mb_downloaded:.1f}/{mb_total:.1f} MB")

        urllib.request.urlretrieve(
            SAM_CHECKPOINT_URL,
            temp_path,
            reporthook=_report_progress
        )

        if progress_callback:
            progress_callback(99, "Verifying download...")

        if not verify_checkpoint_hash(temp_path):
            os.remove(temp_path)
            return False, "Download verification failed - hash mismatch"

        os.rename(temp_path, checkpoint_path)

        if progress_callback:
            progress_callback(100, "Checkpoint downloaded successfully!")

        QgsMessageLog.logMessage(
            f"Checkpoint downloaded to: {checkpoint_path}",
            "AI Segmentation",
            level=Qgis.Success
        )
        return True, "Checkpoint downloaded and verified"

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        QgsMessageLog.logMessage(
            f"Checkpoint download failed: {str(e)}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return False, f"Download failed: {str(e)}"


def has_features_for_raster(raster_path: str) -> bool:
    features_dir = get_raster_features_dir(raster_path)
    csv_path = os.path.join(features_dir, os.path.basename(features_dir) + ".csv")
    if not os.path.exists(csv_path):
        return False
    tif_files = [f for f in os.listdir(features_dir) if f.endswith('.tif')]
    return len(tif_files) > 0


def clear_features_for_raster(raster_path: str) -> bool:
    features_dir = get_raster_features_dir(raster_path)
    if os.path.exists(features_dir):
        import shutil
        shutil.rmtree(features_dir)
        os.makedirs(features_dir, exist_ok=True)
        return True
    return False
