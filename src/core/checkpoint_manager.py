from __future__ import annotations

import hashlib
import os
import shutil
import time
from typing import Callable

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from .model_config import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_SHA256,
    CHECKPOINT_URL,
    USE_SAM2,
)

CACHE_DIR = os.environ.get("AI_SEGMENTATION_CACHE_DIR") or os.path.expanduser("~/.qgis_ai_segmentation")
CHECKPOINTS_DIR = os.path.join(CACHE_DIR, "checkpoints")
FEATURES_DIR = os.path.join(CACHE_DIR, "features")

SAM_CHECKPOINT_URL = CHECKPOINT_URL
SAM_CHECKPOINT_FILENAME = CHECKPOINT_FILENAME
SAM_CHECKPOINT_SHA256 = CHECKPOINT_SHA256
OLD_CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"


def get_checkpoints_dir() -> str:
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    return CHECKPOINTS_DIR


def get_checkpoint_path() -> str:
    return os.path.join(get_checkpoints_dir(), SAM_CHECKPOINT_FILENAME)


def checkpoint_exists() -> bool:
    return os.path.exists(get_checkpoint_path())


def verify_checkpoint_hash(filepath: str) -> bool:
    if not SAM_CHECKPOINT_SHA256:
        # Fail closed: with no expected hash we cannot prove integrity, so we
        # refuse to treat the file as verified rather than silently trusting it.
        # (Both shipped checkpoints define a hash; an empty value here would be
        # a packaging mistake that must surface, not pass through.)
        QgsMessageLog.logMessage(
            "No expected checkpoint hash is configured; cannot verify integrity.",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return False
    if not os.path.isfile(filepath):
        QgsMessageLog.logMessage(
            f"Checkpoint file not found for hash verification: {filepath}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return False
    try:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            # Checkpoints are 150-320 MB; a 4 KiB block would take tens of
            # thousands of Python-level loop iterations, so read in 1 MiB
            # chunks instead.
            for byte_block in iter(lambda: f.read(1024 * 1024), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() == SAM_CHECKPOINT_SHA256
    except OSError as e:
        QgsMessageLog.logMessage(
            f"Failed to verify checkpoint hash: {e}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return False


def is_corrupt_checkpoint_error(error_message: str | None) -> bool:
    """Return True when an error indicates a corrupt checkpoint file.

    The checkpoint is a torch ``.pt`` (a zip archive). When the file on disk
    is truncated or damaged, ``torch.load`` raises errors whose text contains
    one of the signatures below. We match on these so the plugin can delete
    the bad file and re-download it instead of failing forever.
    """
    if not error_message:
        return False
    lowered = error_message.lower()
    signatures = (
        "pytorchstreamreader",
        "failed finding central directory",
        "checkpoint file is corrupted",
        "central directory",
        "not a zip archive",
        "invalid load key",
    )
    return any(sig in lowered for sig in signatures)


def delete_checkpoint() -> bool:
    """Delete the on-disk checkpoint (e.g. when found corrupt).

    Returns True if the file is gone afterwards (deleted or already absent),
    False if it could not be removed.
    """
    path = get_checkpoint_path()
    try:
        if os.path.exists(path):
            os.remove(path)
            QgsMessageLog.logMessage(
                "Removed corrupt checkpoint, will re-download",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return True
    except OSError as e:
        QgsMessageLog.logMessage(
            f"Could not remove corrupt checkpoint: {e}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return False


def _disk_space_preflight_hint(dest_dir: str, min_free_mb: float = 1024.0) -> str | None:
    """Return an error message if dest_dir has less than min_free_mb free.

    Mirrors the venv install's 4GB preflight (venv_manager.py) so a full disk
    fails clearly here too instead of an opaque mid-write error. The SAM
    checkpoint is a few hundred MB; 1GB headroom covers it plus the ".tmp"
    partial-download file that coexists with it during the swap.
    """
    try:
        os.makedirs(dest_dir, exist_ok=True)
        free_mb = shutil.disk_usage(dest_dir).free / (1024 ** 2)
    except OSError:
        # Let the actual write report the real error instead of guessing here.
        return None
    if free_mb < min_free_mb:
        return (
            f"Not enough free disk space to download the AI model: "
            f"{free_mb:.0f} MB available at {dest_dir}, "
            f"at least {min_free_mb:.0f} MB is required.\n\n"
            "Free up disk space, or set the AI_SEGMENTATION_CACHE_DIR "
            "environment variable to a directory on a larger drive, "
            "then restart QGIS."
        )
    return None


def _replace_with_retry(src: str, dst: str, max_attempts: int = 5, delay: float = 2.0):
    """Rename src to dst, retrying on PermissionError (Windows antivirus lock)."""
    import gc
    gc.collect()
    for attempt in range(1, max_attempts + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == max_attempts:
                raise
            QgsMessageLog.logMessage(
                f"File locked, retry {attempt}/{max_attempts} in {delay}s...",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            time.sleep(delay)


def download_checkpoint(
    progress_callback: Callable[[int, str], None] | None = None
) -> tuple[bool, str]:
    """
    Download SAM checkpoint using QGIS network manager with progress reporting.

    Uses QNetworkAccessManager with a local event loop to provide real-time
    download progress updates. Supports resuming partial downloads via HTTP
    Range requests (issue #129).
    """
    from qgis.core import QgsNetworkAccessManager
    from qgis.PyQt.QtCore import QByteArray, QEventLoop, QTimer
    from qgis.PyQt.QtNetwork import QNetworkReply

    checkpoint_path = get_checkpoint_path()
    temp_path = checkpoint_path + ".tmp"

    if checkpoint_exists():
        QgsMessageLog.logMessage(
            "Checkpoint already exists, verifying...",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )
        if verify_checkpoint_hash(checkpoint_path):
            return True, "Checkpoint verified"
        QgsMessageLog.logMessage(
            "Checkpoint hash mismatch, re-downloading...",
            "AI Segmentation",
            level=Qgis.MessageLevel.Warning
        )
        os.remove(checkpoint_path)

    disk_hint = _disk_space_preflight_hint(os.path.dirname(checkpoint_path))
    if disk_hint:
        QgsMessageLog.logMessage(disk_hint, "AI Segmentation", level=Qgis.MessageLevel.Critical)
        return False, disk_hint

    if progress_callback:
        progress_callback(0, "Connecting to download server...")

    max_retries = 5
    last_error = ""

    for attempt in range(1, max_retries + 1):
        # Check for existing partial download to resume
        resume_offset = 0
        if os.path.exists(temp_path):
            resume_offset = os.path.getsize(temp_path)
            if 0 < resume_offset < 1024 * 1024:
                # Tiny partials are more likely an aborted handshake or a
                # proxy error page than real data: restart from scratch.
                try:
                    os.remove(temp_path)
                except OSError:
                    pass  # nosec B110
                resume_offset = 0
            elif resume_offset > 0:
                QgsMessageLog.logMessage(
                    f"Resuming download from {resume_offset / (1024 * 1024):.1f} MB",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)

        # State container for progress tracking
        download_state = {
            "bytes_received": 0,
            "bytes_total": 0,
            "error": None,
            "file": None,
            "resume_offset": resume_offset,
            "start_time": time.monotonic(),
        }

        def on_download_progress(received, total):
            download_state["bytes_received"] = received
            download_state["bytes_total"] = total
            idle_timer = download_state.get("idle_timer")
            if idle_timer is not None:
                idle_timer.start()
            if not progress_callback:
                return
            # Read the base from the state dict, not the closure: a resume the
            # server answered with 200 resets it to 0 (see on_ready_read).
            base = download_state["resume_offset"]
            actual_received = base + received
            actual_total = base + total if total > 0 else 0
            elapsed = max(0.1, time.monotonic() - download_state["start_time"])
            speed_mbs = (received / (1024 * 1024)) / elapsed if received > 0 else 0.0
            retry_suffix = f" (retry {attempt}/{max_retries})" if attempt > 1 else ""

            if actual_total > 0:
                percent = int((actual_received / actual_total) * 90) + 5
                mb_recv = actual_received / (1024 * 1024)
                mb_tot = actual_total / (1024 * 1024)
                remaining_bytes = max(0, (total - received))
                eta_s = int(remaining_bytes / max(1.0, received / elapsed)) if received > 0 else 0
                if eta_s >= 60:
                    eta_str = f"~{eta_s // 60}m {eta_s % 60}s left"
                else:
                    eta_str = f"~{eta_s}s left"
                progress_callback(
                    min(percent, 95),
                    f"Downloading: {mb_recv:.1f} / {mb_tot:.1f} MB "
                    f"({speed_mbs:.1f} MB/s, {eta_str}){retry_suffix}")
            elif actual_received > 0:
                mb_recv = actual_received / (1024 * 1024)
                progress_callback(
                    50,
                    f"Downloading: {mb_recv:.1f} MB ({speed_mbs:.1f} MB/s){retry_suffix}")

        def on_ready_read():
            data = reply.readAll()
            if download_state["file"] is None:
                return
            # A resume answered with HTTP 200 (not 206 Partial Content) means
            # the server or a proxy ignored the Range header and is sending
            # the FULL body; appending it to the partial corrupts the file,
            # caught only by the SHA256 check after a wasted full download
            # (common behind caching proxies that strip Range). Restart the
            # temp file as a fresh write instead, once, on first data.
            if download_state["resume_offset"] > 0 and not download_state.get("status_checked"):
                download_state["status_checked"] = True
                try:
                    status = reply.attribute(
                        QNetworkRequest.Attribute.HttpStatusCodeAttribute)
                except (RuntimeError, AttributeError):
                    status = None
                if status == 200:
                    QgsMessageLog.logMessage(
                        "Server ignored the resume range (HTTP 200): "
                        "restarting the file from scratch",
                        "AI Segmentation", level=Qgis.MessageLevel.Warning)
                    try:
                        download_state["file"].close()
                        download_state["file"] = open(temp_path, "wb")
                    except OSError as reset_err:
                        download_state["error"] = (
                            f"Cannot restart download file: {reset_err}")
                        download_state["file"] = None
                        try:
                            reply.abort()
                        except (RuntimeError, AttributeError):
                            pass
                        return
                    download_state["resume_offset"] = 0
            download_state["file"].write(data.data())

        def on_error(_error_code):
            download_state["error"] = reply.errorString()

        try:
            manager = QgsNetworkAccessManager.instance()
            qurl = QUrl(SAM_CHECKPOINT_URL)
            request = QNetworkRequest(qurl)

            # Resume from partial download using HTTP Range header
            if resume_offset > 0:
                range_header = f"bytes={resume_offset}-"
                request.setRawHeader(
                    QByteArray(b"Range"),
                    QByteArray(range_header.encode("ascii")))

            # Open temp file in append mode for resume, write mode for fresh
            try:
                if resume_offset > 0:
                    download_state["file"] = open(temp_path, "ab")
                else:
                    download_state["file"] = open(temp_path, "wb")
            except OSError as file_err:
                last_error = f"Cannot open download file: {file_err}"
                QgsMessageLog.logMessage(
                    last_error, "AI Segmentation", level=Qgis.MessageLevel.Warning)
                if attempt < max_retries:
                    time.sleep(min(5 * (2 ** (attempt - 1)), 120))
                continue

            reply = manager.get(request)

            reply.downloadProgress.connect(on_download_progress)
            reply.readyRead.connect(on_ready_read)
            reply.errorOccurred.connect(on_error)

            loop = QEventLoop()
            reply.finished.connect(loop.quit)

            # Two watchdogs instead of one absolute 20 min cutoff: a stalled
            # server (connection open, zero bytes) aborts after 2 minutes of
            # silence, while a slow but live connection gets up to 60 minutes.
            def on_idle_timeout():
                download_state["timeout_reason"] = "no data received for 2 minutes"
                loop.quit()

            def on_hard_timeout():
                download_state["timeout_reason"] = "exceeded 60 minutes"
                loop.quit()

            idle_timeout = QTimer()
            idle_timeout.setSingleShot(True)
            idle_timeout.setInterval(120000)
            idle_timeout.timeout.connect(on_idle_timeout)
            download_state["idle_timer"] = idle_timeout
            idle_timeout.start()

            hard_timeout = QTimer()
            hard_timeout.setSingleShot(True)
            hard_timeout.setInterval(3600000)
            hard_timeout.timeout.connect(on_hard_timeout)
            hard_timeout.start()

            if progress_callback:
                retry_msg = ""
                if attempt > 1:
                    retry_msg = f" (retry {attempt}/{max_retries})"
                if resume_offset > 0:
                    progress_callback(
                        5, f"Resuming download...{retry_msg}")
                else:
                    progress_callback(
                        5, f"Download started...{retry_msg}")

            loop.exec()

            idle_timeout.stop()
            hard_timeout.stop()
            download_state["idle_timer"] = None

            if download_state.get("timeout_reason"):
                reply.abort()
                reply.deleteLater()
                if download_state["file"] is not None:
                    download_state["file"].close()
                download_state["file"] = None
                last_error = f"Download timed out ({download_state['timeout_reason']})"
                if attempt < max_retries:
                    time.sleep(min(5 * (2 ** (attempt - 1)), 120))
                continue

            # Check if server returned 416 (range not satisfiable)
            status_code = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
            if status_code == 416:
                if download_state["file"] is not None:
                    download_state["file"].close()
                download_state["file"] = None
                reply.deleteLater()
                # A range at/after the file end means the temp file is already
                # complete. If it verifies, only the final rename is left (a
                # prior _replace_with_retry lost to a file lock); retry that
                # instead of discarding and re-downloading a good, verified file.
                if os.path.exists(temp_path) and verify_checkpoint_hash(temp_path):
                    try:
                        _replace_with_retry(temp_path, checkpoint_path)
                    except OSError as replace_err:
                        last_error = f"Could not finalize checkpoint: {replace_err}"
                        if attempt < max_retries:
                            time.sleep(min(5 * (2 ** (attempt - 1)), 120))
                        continue
                    if progress_callback:
                        progress_callback(100, "Checkpoint downloaded successfully!")
                    QgsMessageLog.logMessage(
                        f"Checkpoint downloaded to: {checkpoint_path}",
                        "AI Segmentation", level=Qgis.MessageLevel.Success)
                    return True, "Checkpoint downloaded and verified"
                # The partial file is invalid: restart from scratch.
                QgsMessageLog.logMessage(
                    "Server rejected range request, restarting download",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                if attempt < max_retries:
                    time.sleep(1)
                continue

            # Check for errors
            if reply.error() != QNetworkReply.NetworkError.NoError:
                last_error = download_state["error"] or reply.errorString()
                QgsMessageLog.logMessage(
                    f"Checkpoint download attempt {attempt}/{max_retries} failed: {last_error}",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                reply.deleteLater()
                if download_state["file"] is not None:
                    download_state["file"].close()
                download_state["file"] = None
                if attempt < max_retries:
                    wait = min(5 * (2 ** (attempt - 1)), 120)
                    if progress_callback:
                        progress_callback(
                            5, f"Retry {attempt + 1}/{max_retries} in {wait}s...")
                    time.sleep(wait)
                continue

            # Flush remaining data
            remaining = reply.readAll()
            if remaining and download_state["file"] is not None:
                download_state["file"].write(remaining.data())
            if download_state["file"] is not None:
                download_state["file"].close()
            download_state["file"] = None
            reply.deleteLater()

            # Verify file is not empty
            file_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
            if file_size == 0:
                last_error = "Download failed: empty file"
                if attempt < max_retries:
                    time.sleep(min(5 * (2 ** (attempt - 1)), 120))
                continue

            if progress_callback:
                mb_total = file_size / (1024 * 1024)
                progress_callback(
                    95, f"Verifying {mb_total:.1f} MB download...")

            if not verify_checkpoint_hash(temp_path):
                # Hash mismatch after full download: partial file was
                # corrupted, delete and retry from scratch
                QgsMessageLog.logMessage(
                    "Hash mismatch, deleting partial file and retrying",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                os.remove(temp_path)
                last_error = "Download verification failed - hash mismatch"
                if attempt < max_retries:
                    time.sleep(min(5 * (2 ** (attempt - 1)), 120))
                continue

            _replace_with_retry(temp_path, checkpoint_path)

            # Clean up old SAM1 checkpoint if present (only on SAM2 path)
            if USE_SAM2:
                old_checkpoint = os.path.join(
                    get_checkpoints_dir(), OLD_CHECKPOINT_FILENAME)
                if os.path.exists(old_checkpoint):
                    try:
                        os.remove(old_checkpoint)
                        QgsMessageLog.logMessage(
                            f"Removed old checkpoint: {OLD_CHECKPOINT_FILENAME}",
                            "AI Segmentation", level=Qgis.MessageLevel.Info)
                    except OSError:
                        pass

            if progress_callback:
                progress_callback(100, "Checkpoint downloaded successfully!")

            QgsMessageLog.logMessage(
                f"Checkpoint downloaded to: {checkpoint_path}",
                "AI Segmentation", level=Qgis.MessageLevel.Success)
            return True, "Checkpoint downloaded and verified"

        except Exception as e:
            last_error = str(e)
            QgsMessageLog.logMessage(
                f"Checkpoint download attempt {attempt}/{max_retries} exception: {last_error}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            if download_state.get("file") is not None:
                try:
                    download_state["file"].close()
                except Exception:
                    pass  # nosec B110
                download_state["file"] = None
            if attempt < max_retries:
                time.sleep(min(5 * (2 ** (attempt - 1)), 120))

    # All retries exhausted - keep partial file for next resume attempt
    partial_mb = 0
    if os.path.exists(temp_path):
        partial_mb = os.path.getsize(temp_path) / (1024 * 1024)
    firewall_hint = (
        " A firewall or proxy may be blocking the download. "
        "Check your network settings in QGIS (Settings > Options > Network)."
    )
    if partial_mb > 0:
        return False, (
            f"Download failed after {max_retries} attempts: {last_error}. "
            f"Partial file ({partial_mb:.1f} MB) saved, will resume on next try.{firewall_hint}"
        )
    return False, f"Download failed after {max_retries} attempts: {last_error}{firewall_hint}"


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
                    f"Removed old SAM1 checkpoint: {OLD_CHECKPOINT_FILENAME}",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
            except OSError as e:
                QgsMessageLog.logMessage(
                    f"Could not remove old checkpoint: {e}",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)

    # Remove old features cache (SAM2 does on-demand encoding, no cache needed)
    if os.path.exists(FEATURES_DIR):
        try:
            shutil.rmtree(FEATURES_DIR)
            QgsMessageLog.logMessage(
                "Removed legacy features cache",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
        except OSError as e:
            QgsMessageLog.logMessage(
                f"Could not remove features cache: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
