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
    # MD5 is used here only for generating a short identifier, not for security
    raster_hash = hashlib.md5(raster_path.encode(), usedforsecurity=False).hexdigest()[:8]

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
    """
    Download SAM checkpoint using QGIS network manager with progress reporting.

    Uses QNetworkAccessManager with a local event loop to provide real-time
    download progress updates instead of blocking without feedback.
    """
    from qgis.PyQt.QtCore import QEventLoop, QTimer
    from qgis.PyQt.QtNetwork import QNetworkReply
    from qgis.core import QgsNetworkAccessManager

    checkpoint_path = get_checkpoint_path()
    temp_path = checkpoint_path + ".tmp"

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

    max_retries = 3
    last_error = ""

    for attempt in range(1, max_retries + 1):
        # Reset state for each attempt
        download_state['bytes_received'] = 0
        download_state['bytes_total'] = 0
        download_state['error'] = None
        download_state['data'] = bytearray()

        try:
            # Use QGIS network manager for proxy-aware downloads
            manager = QgsNetworkAccessManager.instance()
            qurl = QUrl(SAM_CHECKPOINT_URL)
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

            # Add timeout (20 minutes for ~375MB file on slow connections)
            timeout = QTimer()
            timeout.setSingleShot(True)
            timeout.timeout.connect(loop.quit)
            timeout.start(1200000)  # 20 minutes

            if progress_callback:
                retry_msg = "" if attempt == 1 else " (retry {}/{})".format(
                    attempt, max_retries)
                progress_callback(5, "Download started...{}".format(retry_msg))

            # Run the event loop until download completes
            loop.exec_()

            # Check for timeout
            if timeout.isActive():
                timeout.stop()
            else:
                reply.abort()
                reply.deleteLater()
                last_error = "Download timed out after 20 minutes"
                continue

            # Check for errors
            if reply.error() != QNetworkReply.NoError:
                last_error = download_state['error'] or reply.errorString()
                QgsMessageLog.logMessage(
                    "Checkpoint download attempt {}/{} failed: {}".format(
                        attempt, max_retries, last_error),
                    "AI Segmentation",
                    level=Qgis.Warning
                )
                reply.deleteLater()
                if attempt < max_retries:
                    import time
                    time.sleep(2 * attempt)
                continue

            # Read any remaining data
            remaining = reply.readAll()
            if remaining:
                download_state['data'].extend(remaining.data())

            content = bytes(download_state['data'])
            reply.deleteLater()

            if len(content) == 0:
                last_error = "Download failed: empty response"
                continue

            if progress_callback:
                mb_total = len(content) / (1024 * 1024)
                progress_callback(92, f"Downloaded {mb_total:.1f} MB, saving...")

            # Write content to temp file
            with open(temp_path, 'wb') as f:
                f.write(content)

            if progress_callback:
                progress_callback(95, "Verifying download...")

            if not verify_checkpoint_hash(temp_path):
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
            last_error = str(e)
            QgsMessageLog.logMessage(
                "Checkpoint download attempt {}/{} exception: {}".format(
                    attempt, max_retries, last_error),
                "AI Segmentation",
                level=Qgis.Warning
            )
            if attempt < max_retries:
                import time
                time.sleep(2 * attempt)

    # All retries exhausted
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass
    return False, "Download failed after {} attempts: {}".format(
        max_retries, last_error)


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
