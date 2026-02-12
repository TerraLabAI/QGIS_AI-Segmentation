import os
import hashlib
from typing import Tuple, Optional, Callable

from qgis.core import QgsMessageLog, Qgis
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest


CACHE_DIR = os.path.expanduser("~/.qgis_ai_segmentation")
CHECKPOINTS_DIR = os.path.join(CACHE_DIR, "checkpoints")
FEATURES_DIR = os.path.join(CACHE_DIR, "features")

SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"
# SHA256 hash for checkpoint verification (not a secret - this is a public checksum)
SAM_CHECKPOINT_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"  # noqa: S105  # pragma: allowlist secret


def get_checkpoints_dir() -> str:
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    return CHECKPOINTS_DIR


def get_features_dir() -> str:
    os.makedirs(FEATURES_DIR, exist_ok=True)
    return FEATURES_DIR


def get_raster_features_dir(raster_path: str, feature_suffix: Optional[str] = None) -> str:
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
    # MD5 is used here only for generating a short identifier, not for security
    raster_hash = hashlib.md5(raster_path.encode(), usedforsecurity=False).hexdigest()[:8]

    # Combine: rastername_abc12345
    folder_name = f"{sanitized_name}_{raster_hash}"

    features_path = os.path.join(FEATURES_DIR, folder_name)

    # When feature_suffix is given, return model-specific subdirectory
    if feature_suffix:
        features_path = os.path.join(features_path, feature_suffix)

    os.makedirs(features_path, exist_ok=True)
    return features_path


def get_checkpoint_path(model_id: str = "sam_vit_b") -> str:
    from .model_registry import get_checkpoint_path as _registry_path
    path = _registry_path(model_id)
    if path is not None:
        return path
    # Fallback for legacy callers
    return os.path.join(get_checkpoints_dir(), SAM_CHECKPOINT_FILENAME)


def checkpoint_exists(model_id: Optional[str] = None) -> bool:
    if model_id is not None:
        return os.path.exists(get_checkpoint_path(model_id))
    # Legacy: check if ANY model is installed
    from .model_registry import get_installed_models
    return len(get_installed_models()) > 0


def verify_checkpoint_hash(filepath: str, expected_hash: Optional[str] = None) -> bool:
    if expected_hash is None:
        expected_hash = SAM_CHECKPOINT_SHA256
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_hash


def download_checkpoint(
    progress_callback: Optional[Callable[[int, str], None]] = None,
    model_id: str = "sam_vit_b"
) -> Tuple[bool, str]:
    """
    Download a model checkpoint using QGIS network manager with progress reporting.

    Uses QNetworkAccessManager with a local event loop to provide real-time
    download progress updates instead of blocking without feedback.
    """
    from qgis.PyQt.QtCore import QEventLoop, QTimer
    from qgis.PyQt.QtNetwork import QNetworkReply
    from qgis.core import QgsNetworkAccessManager
    from .model_registry import get_model_info

    model_info = get_model_info(model_id)
    if model_info is None:
        return False, "Unknown model: {}".format(model_id)

    download_url = model_info["url"]
    expected_sha256 = model_info.get("sha256")

    checkpoint_path = get_checkpoint_path(model_id)
    temp_path = checkpoint_path + ".tmp"

    if os.path.exists(checkpoint_path):
        QgsMessageLog.logMessage(
            "Checkpoint already exists for {}, verifying...".format(model_id),
            "AI Segmentation",
            level=Qgis.Info
        )
        if expected_sha256 and verify_checkpoint_hash(checkpoint_path, expected_sha256):
            return True, "Checkpoint verified"
        elif expected_sha256:
            QgsMessageLog.logMessage(
                "Checkpoint hash mismatch, re-downloading...",
                "AI Segmentation",
                level=Qgis.Warning
            )
            os.remove(checkpoint_path)
        else:
            # No hash to verify, assume OK
            return True, "Checkpoint exists (no hash verification)"

    if progress_callback:
        progress_callback(0, "Connecting to download server...")

    # State container for progress tracking
    download_state = {
        'bytes_received': 0,
        'bytes_total': 0,
        'error': None,
        'data': bytearray()
    }

    def on_download_progress(received: int, total: int):
        """Handle download progress updates."""
        download_state['bytes_received'] = received
        download_state['bytes_total'] = total

        if total > 0 and progress_callback:
            # Calculate percentage (reserve 5% for verification)
            percent = int((received / total) * 90) + 5
            mb_received = received / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            progress_callback(
                min(percent, 95),
                f"Downloading: {mb_received:.1f} / {mb_total:.1f} MB"
            )
        elif progress_callback and received > 0:
            # Unknown total size, show bytes downloaded
            mb_received = received / (1024 * 1024)
            progress_callback(50, f"Downloading: {mb_received:.1f} MB...")

    def on_ready_read():
        """Accumulate downloaded data chunks."""
        data = reply.readAll()
        download_state['data'].extend(data.data())

    def on_error(error_code):
        """Handle download errors."""
        download_state['error'] = reply.errorString()

    try:
        # Use QGIS network manager for proxy-aware downloads
        manager = QgsNetworkAccessManager.instance()
        qurl = QUrl(download_url)
        request = QNetworkRequest(qurl)

        # Start the download
        reply = manager.get(request)

        # Connect progress signals
        reply.downloadProgress.connect(on_download_progress)
        reply.readyRead.connect(on_ready_read)
        reply.errorOccurred.connect(on_error)

        # Create event loop to wait for download completion
        loop = QEventLoop()
        reply.finished.connect(loop.quit)

        # Dynamic timeout based on model size (1min per 100MB, minimum 20min)
        timeout_ms = max(1200000, model_info["size_mb"] * 600 * 10)
        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit)
        timeout.start(timeout_ms)

        if progress_callback:
            progress_callback(5, "Download started...")

        # Run the event loop until download completes
        loop.exec_()

        # Check for timeout
        if timeout.isActive():
            timeout.stop()
        else:
            reply.abort()
            timeout_min = timeout_ms // 60000
            return False, "Download timed out after {} minutes".format(timeout_min)

        # Check for errors
        if reply.error() != QNetworkReply.NoError:
            error_msg = download_state['error'] or reply.errorString()
            QgsMessageLog.logMessage(
                f"Checkpoint download failed: {error_msg}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            reply.deleteLater()
            return False, f"Download failed: {error_msg}"

        # Read any remaining data
        remaining = reply.readAll()
        if remaining:
            download_state['data'].extend(remaining.data())

        content = bytes(download_state['data'])
        reply.deleteLater()

        if len(content) == 0:
            return False, "Download failed: empty response"

        if progress_callback:
            mb_total = len(content) / (1024 * 1024)
            progress_callback(92, f"Downloaded {mb_total:.1f} MB, saving...")

        # Write content to temp file
        with open(temp_path, 'wb') as f:
            f.write(content)

        if progress_callback:
            progress_callback(95, "Verifying download...")

        if expected_sha256:
            if not verify_checkpoint_hash(temp_path, expected_sha256):
                os.remove(temp_path)
                return False, "Download verification failed - hash mismatch"

        os.replace(temp_path, checkpoint_path)

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
            try:
                os.remove(temp_path)
            except OSError:
                pass
        QgsMessageLog.logMessage(
            f"Checkpoint download failed: {str(e)}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return False, f"Download failed: {str(e)}"


def has_features_for_raster(raster_path: str, feature_suffix: Optional[str] = None) -> bool:
    """Check if cached features exist for a raster.

    When feature_suffix is given, check only that model's subdirectory.
    When None, check if ANY model subdirectory has features.
    """
    if feature_suffix:
        features_dir = get_raster_features_dir(raster_path, feature_suffix)
        csv_name = "{}.csv".format(feature_suffix)
        csv_path = os.path.join(features_dir, csv_name)
        if not os.path.exists(csv_path):
            return False
        tif_files = [f for f in os.listdir(features_dir) if f.endswith('.tif')]
        return len(tif_files) > 0

    # No suffix: check if any model subdirectory has features
    base_dir = get_raster_features_dir(raster_path)
    if not os.path.exists(base_dir):
        return False
    for entry in os.listdir(base_dir):
        subdir = os.path.join(base_dir, entry)
        if not os.path.isdir(subdir):
            continue
        csv_path = os.path.join(subdir, "{}.csv".format(entry))
        if os.path.exists(csv_path):
            tif_files = [f for f in os.listdir(subdir) if f.endswith('.tif')]
            if len(tif_files) > 0:
                return True
    return False


def clear_features_for_raster(raster_path: str, feature_suffix: Optional[str] = None) -> bool:
    """Clear cached features for a raster.

    When feature_suffix is given, only clear that model's subdirectory.
    When None, clear all model subdirectories (whole raster cache).
    """
    import shutil
    if feature_suffix:
        features_dir = get_raster_features_dir(raster_path, feature_suffix)
        if os.path.exists(features_dir):
            shutil.rmtree(features_dir)
            os.makedirs(features_dir, exist_ok=True)
            return True
        return False

    base_dir = get_raster_features_dir(raster_path)
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        os.makedirs(base_dir, exist_ok=True)
        return True
    return False
