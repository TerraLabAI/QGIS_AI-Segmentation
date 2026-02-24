import os
import hashlib
import threading
from typing import Tuple, Optional, Callable

from qgis.core import QgsMessageLog, Qgis
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from .model_config import (
    CHECKPOINT_URL, CHECKPOINT_FILENAME, CHECKPOINT_SHA256,
    USE_SAM2,
)

CACHE_DIR = os.path.expanduser("~/.qgis_ai_segmentation")
CHECKPOINTS_DIR = os.path.join(CACHE_DIR, "checkpoints")
FEATURES_DIR = os.path.join(CACHE_DIR, "features")

SAM_CHECKPOINT_URL = CHECKPOINT_URL
SAM_CHECKPOINT_FILENAME = CHECKPOINT_FILENAME
SAM_CHECKPOINT_SHA256 = CHECKPOINT_SHA256
OLD_CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"


def get_checkpoints_dir() -> str:
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    return CHECKPOINTS_DIR


def get_features_dir() -> str:
    os.makedirs(FEATURES_DIR, exist_ok=True)
    return FEATURES_DIR


def _migrate_old_cache_layout():
    """Migrate old flat cache dirs into the new raster/full/ subfolder layout.

    Old layout: features/rastername_hash/*.tif + rastername_hash.csv
    New layout: features/rastername_hash/full/*.tif + full.csv

    Only runs once per session (idempotent).
    """
    if not os.path.exists(FEATURES_DIR):
        return

    try:
        for entry in os.listdir(FEATURES_DIR):
            entry_path = os.path.join(FEATURES_DIR, entry)
            if not os.path.isdir(entry_path):
                continue
            # Skip already-migrated dirs (contain only subdirs like full/, visible_*)
            has_tif = any(f.endswith('.tif') for f in os.listdir(entry_path))
            if not has_tif:
                continue
            # Old format detected: tif files directly in the raster folder
            full_dir = os.path.join(entry_path, "full")
            os.makedirs(full_dir, exist_ok=True)
            for fname in os.listdir(entry_path):
                if fname == "full":
                    continue
                src = os.path.join(entry_path, fname)
                if os.path.isfile(src):
                    # Rename old CSV to full.csv
                    if fname.endswith('.csv'):
                        dst = os.path.join(full_dir, "full.csv")
                    else:
                        dst = os.path.join(full_dir, fname)
                    os.replace(src, dst)
            QgsMessageLog.logMessage(
                "Migrated old cache folder: {}".format(entry),
                "AI Segmentation", level=Qgis.Info)
    except Exception as e:
        QgsMessageLog.logMessage(
            "Cache migration warning: {}".format(e),
            "AI Segmentation", level=Qgis.Warning)


_cache_migrated = False
_cache_migration_lock = threading.Lock()


def _get_raster_base_dir(raster_path: str) -> str:
    """Get the base cache directory for a raster (parent of full/ and visible_*/).

    Structure: features/{sanitized_name}_{path_hash}/
    The hash is based on the raster path only (not the extent).
    """
    import re

    raster_filename = os.path.splitext(os.path.basename(raster_path))[0]

    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raster_filename)
    sanitized_name = sanitized_name[:40]
    sanitized_name = sanitized_name.rstrip('_')

    # MD5 is used here only for generating a short identifier, not for security
    path_hash = hashlib.md5(
        raster_path.encode(), usedforsecurity=False
    ).hexdigest()[:8]

    return os.path.join(FEATURES_DIR, "{}_{}".format(sanitized_name, path_hash))


def get_raster_features_dir(raster_path: str, visible_extent: tuple = None) -> str:
    """Get the encoding cache directory for a raster.

    Structure:
      features/{rastername}_{path_hash}/full/          (full raster)
      features/{rastername}_{path_hash}/visible_{hash}/ (visible area)

    Each raster gets one parent folder, with subfolders for each encoding type.
    """
    global _cache_migrated
    if not _cache_migrated:
        with _cache_migration_lock:
            if not _cache_migrated:
                _migrate_old_cache_layout()
                _cache_migrated = True

    base_dir = _get_raster_base_dir(raster_path)

    if visible_extent is None:
        features_path = os.path.join(base_dir, "full")
    else:
        extent_str = "{:.2f}_{:.2f}_{:.2f}_{:.2f}".format(
            visible_extent[0], visible_extent[1],
            visible_extent[2], visible_extent[3])
        extent_hash = hashlib.md5(
            extent_str.encode(), usedforsecurity=False
        ).hexdigest()[:8]
        features_path = os.path.join(base_dir, "visible_{}".format(extent_hash))

    os.makedirs(features_path, exist_ok=True)
    return features_path


def get_checkpoint_path() -> str:
    return os.path.join(get_checkpoints_dir(), SAM_CHECKPOINT_FILENAME)


def checkpoint_exists() -> bool:
    return os.path.exists(get_checkpoint_path())


def verify_checkpoint_hash(filepath: str) -> bool:
    if not SAM_CHECKPOINT_SHA256:
        # Hash not yet computed, skip verification
        return True
    if not os.path.isfile(filepath):
        QgsMessageLog.logMessage(
            "Checkpoint file not found for hash verification: {}".format(filepath),
            "AI Segmentation", level=Qgis.Warning)
        return False
    try:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() == SAM_CHECKPOINT_SHA256
    except (OSError, IOError) as e:
        QgsMessageLog.logMessage(
            "Failed to verify checkpoint hash: {}".format(e),
            "AI Segmentation", level=Qgis.Warning)
        return False


def download_checkpoint(
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[bool, str]:
    """
    Download SAM checkpoint using QGIS network manager with progress reporting.

    Uses QNetworkAccessManager with a local event loop to provide real-time
    download progress updates. Supports resuming partial downloads via HTTP
    Range requests (issue #129).
    """
    from qgis.PyQt.QtCore import QEventLoop, QTimer, QByteArray
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

    max_retries = 5
    last_error = ""

    for attempt in range(1, max_retries + 1):
        # Check for existing partial download to resume
        resume_offset = 0
        if os.path.exists(temp_path):
            resume_offset = os.path.getsize(temp_path)
            if resume_offset > 0:
                QgsMessageLog.logMessage(
                    "Resuming download from {:.1f} MB".format(
                        resume_offset / (1024 * 1024)),
                    "AI Segmentation", level=Qgis.Info)

        # State container for progress tracking
        download_state = {
            'bytes_received': 0,
            'bytes_total': 0,
            'error': None,
            'file': None,
            'resume_offset': resume_offset,
        }

        def on_download_progress(received, total):
            download_state['bytes_received'] = received
            download_state['bytes_total'] = total
            if progress_callback:
                actual_received = resume_offset + received
                actual_total = resume_offset + total if total > 0 else 0
                if actual_total > 0:
                    percent = int((actual_received / actual_total) * 90) + 5
                    mb_recv = actual_received / (1024 * 1024)
                    mb_tot = actual_total / (1024 * 1024)
                    progress_callback(
                        min(percent, 95),
                        "Downloading: {:.1f} / {:.1f} MB".format(
                            mb_recv, mb_tot))
                elif actual_received > 0:
                    mb_recv = actual_received / (1024 * 1024)
                    progress_callback(
                        50, "Downloading: {:.1f} MB...".format(mb_recv))

        def on_ready_read():
            data = reply.readAll()
            if download_state['file'] is not None:
                download_state['file'].write(data.data())

        def on_error(error_code):
            download_state['error'] = reply.errorString()

        try:
            manager = QgsNetworkAccessManager.instance()
            qurl = QUrl(SAM_CHECKPOINT_URL)
            request = QNetworkRequest(qurl)

            # Resume from partial download using HTTP Range header
            if resume_offset > 0:
                range_header = "bytes={}-".format(resume_offset)
                request.setRawHeader(
                    QByteArray(b"Range"),
                    QByteArray(range_header.encode("ascii")))

            # Open temp file in append mode for resume, write mode for fresh
            try:
                if resume_offset > 0:
                    download_state['file'] = open(temp_path, 'ab')
                else:
                    download_state['file'] = open(temp_path, 'wb')
            except (OSError, IOError) as file_err:
                last_error = "Cannot open download file: {}".format(file_err)
                QgsMessageLog.logMessage(
                    last_error, "AI Segmentation", level=Qgis.Warning)
                if attempt < max_retries:
                    import time
                    time.sleep(2 * attempt)
                continue

            reply = manager.get(request)

            reply.downloadProgress.connect(on_download_progress)
            reply.readyRead.connect(on_ready_read)
            reply.errorOccurred.connect(on_error)

            loop = QEventLoop()
            reply.finished.connect(loop.quit)

            timeout = QTimer()
            timeout.setSingleShot(True)
            timeout.timeout.connect(loop.quit)
            timeout.start(1200000)  # 20 minutes

            if progress_callback:
                retry_msg = ""
                if attempt > 1:
                    retry_msg = " (retry {}/{})".format(attempt, max_retries)
                if resume_offset > 0:
                    progress_callback(
                        5, "Resuming download...{}".format(retry_msg))
                else:
                    progress_callback(
                        5, "Download started...{}".format(retry_msg))

            loop.exec_()

            # Check for timeout
            if timeout.isActive():
                timeout.stop()
            else:
                reply.abort()
                reply.deleteLater()
                download_state['file'].close()
                download_state['file'] = None
                last_error = "Download timed out after 20 minutes"
                if attempt < max_retries:
                    import time
                    time.sleep(2 * attempt)
                continue

            # Check if server returned 416 (range not satisfiable)
            # This means the partial file is invalid, restart from scratch
            status_code = reply.attribute(QNetworkRequest.HttpStatusCodeAttribute)
            if status_code == 416:
                QgsMessageLog.logMessage(
                    "Server rejected range request, restarting download",
                    "AI Segmentation", level=Qgis.Warning)
                download_state['file'].close()
                download_state['file'] = None
                reply.deleteLater()
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                if attempt < max_retries:
                    import time
                    time.sleep(1)
                continue

            # Check for errors
            if reply.error() != QNetworkReply.NoError:
                last_error = download_state['error'] or reply.errorString()
                QgsMessageLog.logMessage(
                    "Checkpoint download attempt {}/{} failed: {}".format(
                        attempt, max_retries, last_error),
                    "AI Segmentation", level=Qgis.Warning)
                reply.deleteLater()
                download_state['file'].close()
                download_state['file'] = None
                if attempt < max_retries:
                    import time
                    time.sleep(2 * attempt)
                continue

            # Flush remaining data
            remaining = reply.readAll()
            if remaining:
                download_state['file'].write(remaining.data())
            download_state['file'].close()
            download_state['file'] = None
            reply.deleteLater()

            # Verify file is not empty
            file_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
            if file_size == 0:
                last_error = "Download failed: empty file"
                if attempt < max_retries:
                    import time
                    time.sleep(2 * attempt)
                continue

            if progress_callback:
                mb_total = file_size / (1024 * 1024)
                progress_callback(
                    95, "Verifying {:.1f} MB download...".format(mb_total))

            if not verify_checkpoint_hash(temp_path):
                # Hash mismatch after full download: partial file was
                # corrupted, delete and retry from scratch
                QgsMessageLog.logMessage(
                    "Hash mismatch, deleting partial file and retrying",
                    "AI Segmentation", level=Qgis.Warning)
                os.remove(temp_path)
                last_error = "Download verification failed - hash mismatch"
                if attempt < max_retries:
                    import time
                    time.sleep(2 * attempt)
                continue

            os.replace(temp_path, checkpoint_path)

            # Clean up old SAM1 checkpoint if present (only on SAM2 path)
            if USE_SAM2:
                old_checkpoint = os.path.join(
                    get_checkpoints_dir(), OLD_CHECKPOINT_FILENAME)
                if os.path.exists(old_checkpoint):
                    try:
                        os.remove(old_checkpoint)
                        QgsMessageLog.logMessage(
                            "Removed old checkpoint: {}".format(
                                OLD_CHECKPOINT_FILENAME),
                            "AI Segmentation", level=Qgis.Info)
                    except OSError:
                        pass

            if progress_callback:
                progress_callback(100, "Checkpoint downloaded successfully!")

            QgsMessageLog.logMessage(
                "Checkpoint downloaded to: {}".format(checkpoint_path),
                "AI Segmentation", level=Qgis.Success)
            return True, "Checkpoint downloaded and verified"

        except Exception as e:
            last_error = str(e)
            QgsMessageLog.logMessage(
                "Checkpoint download attempt {}/{} exception: {}".format(
                    attempt, max_retries, last_error),
                "AI Segmentation", level=Qgis.Warning)
            if download_state.get('file') is not None:
                try:
                    download_state['file'].close()
                except Exception:
                    pass
                download_state['file'] = None
            if attempt < max_retries:
                import time
                time.sleep(2 * attempt)

    # All retries exhausted - keep partial file for next resume attempt
    partial_mb = 0
    if os.path.exists(temp_path):
        partial_mb = os.path.getsize(temp_path) / (1024 * 1024)
    if partial_mb > 0:
        return False, (
            "Download failed after {} attempts: {}. "
            "Partial file ({:.1f} MB) saved, will resume on next try."
        ).format(max_retries, last_error, partial_mb)
    return False, "Download failed after {} attempts: {}".format(
        max_retries, last_error)


def cleanup_legacy_sam1_data():
    """Remove legacy SAM1 data: old checkpoint and features cache.

    Called once at plugin startup. Errors are logged silently
    to avoid disturbing the user.

    On Python 3.9 (USE_SAM2=False), the SAM1 checkpoint is still needed,
    so only features cache is cleaned up.
    """
    import shutil

    # Remove old SAM1 checkpoint only when using SAM2 (Python 3.10+)
    if USE_SAM2:
        old_checkpoint = os.path.join(CHECKPOINTS_DIR, OLD_CHECKPOINT_FILENAME)
        if os.path.exists(old_checkpoint):
            try:
                os.remove(old_checkpoint)
                QgsMessageLog.logMessage(
                    "Removed old SAM1 checkpoint: {}".format(
                        OLD_CHECKPOINT_FILENAME),
                    "AI Segmentation", level=Qgis.Info)
            except OSError as e:
                QgsMessageLog.logMessage(
                    "Could not remove old checkpoint: {}".format(e),
                    "AI Segmentation", level=Qgis.Warning)

    # Remove old features cache (SAM2 does on-demand encoding, no cache needed)
    if os.path.exists(FEATURES_DIR):
        try:
            shutil.rmtree(FEATURES_DIR)
            QgsMessageLog.logMessage(
                "Removed legacy features cache",
                "AI Segmentation", level=Qgis.Info)
        except OSError as e:
            QgsMessageLog.logMessage(
                "Could not remove features cache: {}".format(e),
                "AI Segmentation", level=Qgis.Warning)


def has_features_for_raster(raster_path: str, visible_extent: tuple = None) -> bool:
    features_dir = get_raster_features_dir(raster_path, visible_extent)
    csv_path = os.path.join(features_dir, os.path.basename(features_dir) + ".csv")
    if not os.path.exists(csv_path):
        return False
    tif_files = [f for f in os.listdir(features_dir) if f.endswith('.tif')]
    return len(tif_files) > 0


def clear_features_for_raster(raster_path: str, visible_extent: tuple = None) -> bool:
    features_dir = get_raster_features_dir(raster_path, visible_extent)
    if os.path.exists(features_dir):
        import shutil
        shutil.rmtree(features_dir)
        os.makedirs(features_dir, exist_ok=True)
        return True
    return False
