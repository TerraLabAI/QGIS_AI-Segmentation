from __future__ import annotations

import hashlib
import os
import shutil
import stat
import sys
import time
from typing import Any, BinaryIO, Callable

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from .cache_paths import PLUGIN_CACHE_DIR, remove_tree_quietly
from .model_config import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_SHA256,
    CHECKPOINT_URL,
    USE_SAM2,
)
from .qt_compat import NoLessSafeRedirectPolicy, RedirectPolicyAttribute


def tr(text: str) -> str:





    try:
        from .i18n import tr as translate

        return translate(text)
    except Exception:  # noqa: BLE001
        return text


CHECKPOINTS_DIR = os.path.join(PLUGIN_CACHE_DIR, "checkpoints")
FEATURES_DIR = os.path.join(PLUGIN_CACHE_DIR, "features")

SAM_CHECKPOINT_URL = CHECKPOINT_URL
SAM_CHECKPOINT_FILENAME = CHECKPOINT_FILENAME
SAM_CHECKPOINT_SHA256 = CHECKPOINT_SHA256
OLD_CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"



DOWNLOAD_MAX_RETRIES = 5
DOWNLOAD_IDLE_TIMEOUT_MS = 120_000
DOWNLOAD_HARD_TIMEOUT_MS = 3_600_000


REPLACE_ATTEMPTS = 5
REPLACE_DELAY_S = 2.0



PARTIAL_DOWNLOAD_MAX_AGE_S = 7 * 24 * 60 * 60

RETRY_BACKOFF_BASE_S = 5
RETRY_BACKOFF_MAX_S = 120

DIGEST_MISMATCH_LIMIT = 2


def _retry_wait_s(attempt: int) -> int:

    from .server_dials import dial_in_range

    base = dial_in_range("tuning.install.checkpoint_backoff_base_s",
                         RETRY_BACKOFF_BASE_S, 1, 60)
    ceiling = dial_in_range("tuning.install.checkpoint_backoff_max_s",
                            RETRY_BACKOFF_MAX_S, 5, 600)
    return min(base * (2 ** (attempt - 1)), ceiling)


def resolved_checkpoint() -> tuple[str, str, tuple[str, ...]]:









    try:
        from .install_config import checkpoint_source

        return checkpoint_source(
            SAM_CHECKPOINT_FILENAME, SAM_CHECKPOINT_URL, SAM_CHECKPOINT_SHA256)
    except Exception:  # noqa: BLE001
        return SAM_CHECKPOINT_URL, SAM_CHECKPOINT_SHA256, ()


def get_checkpoints_dir() -> str:
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    return CHECKPOINTS_DIR


def get_checkpoint_path() -> str:
    return os.path.join(get_checkpoints_dir(), SAM_CHECKPOINT_FILENAME)


def checkpoint_exists() -> bool:







    return os.path.exists(os.path.join(CHECKPOINTS_DIR, SAM_CHECKPOINT_FILENAME))


def verify_checkpoint_hash(filepath: str) -> bool:
    expected_sha256 = resolved_checkpoint()[1]
    if not expected_sha256:




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



            for byte_block in iter(lambda: f.read(1024 * 1024), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() == expected_sha256
    except OSError as e:
        QgsMessageLog.logMessage(
            f"Failed to verify checkpoint hash: {e}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return False


def is_corrupt_checkpoint_error(error_message: str | None) -> bool:







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












    path = get_checkpoint_path()
    if not os.path.exists(path):
        return True



    try:
        os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE)
    except OSError:
        pass  # nosec B110
    if _remove_with_retry(path):
        QgsMessageLog.logMessage(
            "Removed corrupt checkpoint, will re-download",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return True
    try:
        os.replace(path, path + ".corrupt")
    except OSError as e:
        QgsMessageLog.logMessage(
            f"Could not remove corrupt checkpoint: {e}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return False
    QgsMessageLog.logMessage(
        "Moved the corrupt checkpoint aside, will re-download",
        "AI Segmentation", level=Qgis.MessageLevel.Warning)
    return True





CHECKPOINT_MIN_FREE_MB = 1024.0


def resolved_checkpoint_min_free_mb() -> float:

    try:
        from .server_dials import dial_in_range

        return float(dial_in_range(
            "install.disk.min_free_mb_checkpoint",
            CHECKPOINT_MIN_FREE_MB, 256.0, 20480.0))
    except Exception:  # noqa: BLE001
        return CHECKPOINT_MIN_FREE_MB


def _disk_space_preflight_hint(
        dest_dir: str, min_free_mb: float | None = None) -> str | None:





    if min_free_mb is None:
        min_free_mb = resolved_checkpoint_min_free_mb()
    try:
        os.makedirs(dest_dir, exist_ok=True)
        free_mb = shutil.disk_usage(dest_dir).free / (1024 ** 2)
    except OSError:

        return None
    if free_mb < min_free_mb:
        return (
            tr("Not enough free disk space to download the AI model: "
               "{free} MB available, at least {needed} MB is required.").format(
                   free=f"{free_mb:.0f}", needed=f"{min_free_mb:.0f}")
            + "\n\n"
            + tr("Free up disk space, or set the AI_SEGMENTATION_CACHE_DIR "
                 "environment variable to a directory on a larger drive, "
                 "then restart QGIS.")
        )
    return None







_cancel_requested = False

DOWNLOAD_CANCELLED_MESSAGE = "Download cancelled"


def request_download_cancel() -> None:





    global _cancel_requested
    _cancel_requested = True


def download_cancel_requested() -> bool:

    return _cancel_requested


def _consume_download_cancel() -> bool:






    global _cancel_requested
    if not _cancel_requested:
        return False
    _cancel_requested = False
    return True


def _wait_or_cancel(seconds: float) -> bool:






    deadline = time.monotonic() + seconds
    while True:
        if _cancel_requested:
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(0.25, remaining))


def _resolved_retry_ladder(max_attempts: int | None,
                           delay: float | None) -> tuple[int, float]:







    if max_attempts is not None and delay is not None:
        return max_attempts, delay
    try:
        from .install_config import replace_attempts, replace_delay_s

        served_attempts = replace_attempts(REPLACE_ATTEMPTS)
        served_delay = replace_delay_s(REPLACE_DELAY_S)
    except Exception:  # noqa: BLE001
        served_attempts, served_delay = REPLACE_ATTEMPTS, REPLACE_DELAY_S
    return (
        served_attempts if max_attempts is None else max_attempts,
        served_delay if delay is None else delay,
    )


def _replace_with_retry(src: str, dst: str, max_attempts: int | None = None,
                        delay: float | None = None) -> bool:





    import gc

    max_attempts, delay = _resolved_retry_ladder(max_attempts, delay)
    gc.collect()
    for attempt in range(1, max_attempts + 1):
        try:
            os.replace(src, dst)
            return True
        except PermissionError:
            if attempt == max_attempts:
                raise
            QgsMessageLog.logMessage(
                f"File locked, retry {attempt}/{max_attempts} in {delay}s...",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)



            if not _wait_or_cancel(delay):
                return False
    return False


def _remove_with_retry(path: str, max_attempts: int | None = None,
                       delay: float | None = None) -> bool:







    max_attempts, delay = _resolved_retry_ladder(max_attempts, delay)
    last_error: OSError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except OSError as err:
            last_error = err
            if sys.platform == "win32" and getattr(err, "winerror", None) == 5:


                try:
                    os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE)
                    os.remove(path)
                    return True
                except OSError as retry_err:
                    last_error = retry_err
            if attempt == max_attempts:
                break
            if not _wait_or_cancel(delay):
                return False
    QgsMessageLog.logMessage(
        f"Could not remove {os.path.basename(path)}: {last_error}",
        "AI Segmentation", level=Qgis.MessageLevel.Warning)
    return False


def _adopt_legacy_partial(legacy_path: str, temp_path: str) -> None:







    if os.path.exists(temp_path) or not os.path.exists(legacy_path):
        return
    try:
        os.replace(legacy_path, temp_path)
    except OSError:
        return  # nosec B110
    QgsMessageLog.logMessage(
        "Resuming a partial model download left by an earlier version",
        "AI Segmentation", level=Qgis.MessageLevel.Info)


def _discard_partial_download(temp_path: str) -> None:








    if _remove_with_retry(temp_path):
        return
    try:
        with open(temp_path, "wb"):
            pass
    except OSError as err:
        QgsMessageLog.logMessage(
            f"Could not clear the partial download: {err}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)






_CHECKPOINT_LOCK_BASENAME = "checkpoint.lock"


def checkpoint_lock_path() -> str:

    return os.path.join(CHECKPOINTS_DIR, _CHECKPOINT_LOCK_BASENAME)


def download_checkpoint(
    progress_callback: Callable[[int, str], None] | None = None
) -> tuple[bool, str]:







    from .install_lock import InstallBusyError, acquire_install_lock

    try:
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    except OSError:
        pass  # nosec B110
    try:
        lock = acquire_install_lock(checkpoint_lock_path())
    except InstallBusyError:
        QgsMessageLog.logMessage(
            "Another process is already downloading the model",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        return False, tr(
            "Another QGIS window is downloading the AI model. Wait for it to "
            "finish, then try again.")
    try:
        return _download_checkpoint(progress_callback)
    finally:
        lock.release()


class _DownloadFileHandle:








    handle: BinaryIO | None = None


def _download_checkpoint(
    progress_callback: Callable[[int, str], None] | None = None
) -> tuple[bool, str]:







    from qgis.core import QgsNetworkAccessManager
    from qgis.PyQt.QtCore import QByteArray, QEventLoop, QTimer
    from qgis.PyQt.QtNetwork import QNetworkReply

    from .server_dials import dial_in_range



    global _cancel_requested
    _cancel_requested = False

    checkpoint_path = get_checkpoint_path()

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




        _remove_with_retry(checkpoint_path)

    disk_hint = _disk_space_preflight_hint(os.path.dirname(checkpoint_path))
    if disk_hint:
        QgsMessageLog.logMessage(disk_hint, "AI Segmentation", level=Qgis.MessageLevel.Critical)
        return False, disk_hint

    if progress_callback:
        progress_callback(0, tr("Connecting to download server..."))

    from .install_config import (
        checkpoint_hard_timeout_ms,
        checkpoint_idle_timeout_ms,
        checkpoint_max_retries,
    )

    download_url, expected_sha256, mirrors = resolved_checkpoint()




    temp_path = f"{checkpoint_path}.{expected_sha256[:12]}.tmp"
    _adopt_legacy_partial(f"{checkpoint_path}.tmp", temp_path)



    download_urls = (download_url,) + mirrors
    max_retries = max(checkpoint_max_retries(DOWNLOAD_MAX_RETRIES), len(download_urls))
    idle_timeout_ms = checkpoint_idle_timeout_ms(DOWNLOAD_IDLE_TIMEOUT_MS)
    hard_timeout_ms = checkpoint_hard_timeout_ms(DOWNLOAD_HARD_TIMEOUT_MS)
    last_error = ""

    digest_mismatches = 0



    refused_urls: set[str] = set()

    for attempt in range(1, max_retries + 1):
        if _consume_download_cancel():
            return False, DOWNLOAD_CANCELLED_MESSAGE



        attempt_url = download_urls[(attempt - 1) % len(download_urls)]


        resume_offset = 0
        if os.path.exists(temp_path):
            resume_offset = os.path.getsize(temp_path)
            if 0 < resume_offset < 1024 * 1024:


                try:
                    os.remove(temp_path)
                except OSError:
                    pass  # nosec B110
                resume_offset = 0
            elif resume_offset > 0:
                QgsMessageLog.logMessage(
                    f"Resuming download from {resume_offset / (1024 * 1024):.1f} MB",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)






        download_state: dict[str, Any] = {
            "bytes_received": 0,
            "bytes_total": 0,
            "error": None,
            "reply": None,
            "resume_offset": resume_offset,
            "start_time": time.monotonic(),
        }
        file_slot = _DownloadFileHandle()

        def on_download_progress(received, total):
            download_state["bytes_received"] = received
            download_state["bytes_total"] = total
            idle_timer = download_state.get("idle_timer")
            if idle_timer is not None:
                idle_timer.start()
            if not progress_callback:
                return


            base = download_state["resume_offset"]
            actual_received = base + received
            actual_total = base + total if total > 0 else 0
            elapsed = max(0.1, time.monotonic() - download_state["start_time"])
            speed_mbs = (received / (1024 * 1024)) / elapsed if received > 0 else 0.0
            retry_suffix = ""
            if attempt > 1:
                retry_suffix = " " + tr("(retry {done}/{total})").format(
                    done=attempt, total=max_retries)

            if actual_total > 0:
                percent = int((actual_received / actual_total) * 90) + 5
                mb_recv = actual_received / (1024 * 1024)
                mb_tot = actual_total / (1024 * 1024)
                remaining_bytes = max(0, (total - received))
                eta_s = int(remaining_bytes / max(1.0, received / elapsed)) if received > 0 else 0
                if eta_s >= 60:
                    eta_str = tr("~{minutes}m {seconds}s left").format(
                        minutes=eta_s // 60, seconds=eta_s % 60)
                else:
                    eta_str = tr("~{seconds}s left").format(seconds=eta_s)
                progress_callback(
                    min(percent, 95),
                    tr("Downloading: {done} / {total} MB ({speed} MB/s, {eta})").format(
                        done=f"{mb_recv:.1f}", total=f"{mb_tot:.1f}",
                        speed=f"{speed_mbs:.1f}", eta=eta_str) + retry_suffix)
            elif actual_received > 0:
                mb_recv = actual_received / (1024 * 1024)
                progress_callback(
                    50,
                    tr("Downloading: {done} MB ({speed} MB/s)").format(
                        done=f"{mb_recv:.1f}", speed=f"{speed_mbs:.1f}") + retry_suffix)

        def on_ready_read():
            attempt_reply = download_state["reply"]
            if attempt_reply is None or file_slot.handle is None:
                return
            data = attempt_reply.readAll()






            if download_state["resume_offset"] > 0 and not download_state.get("status_checked"):
                download_state["status_checked"] = True
                try:
                    status = attempt_reply.attribute(
                        QNetworkRequest.Attribute.HttpStatusCodeAttribute)
                except (RuntimeError, AttributeError):
                    status = None
                if status == 200:
                    QgsMessageLog.logMessage(
                        "Server ignored the resume range (HTTP 200): "
                        "restarting the file from scratch",
                        "AI Segmentation", level=Qgis.MessageLevel.Warning)
                    try:
                        file_slot.handle.close()
                        file_slot.handle = open(temp_path, "wb")
                    except OSError as reset_err:
                        download_state["error"] = (
                            f"Cannot restart download file: {reset_err}")
                        file_slot.handle = None
                        try:
                            attempt_reply.abort()
                        except (RuntimeError, AttributeError):
                            pass
                        return
                    download_state["resume_offset"] = 0




            try:
                file_slot.handle.write(data.data())
            except OSError as write_err:
                download_state["error"] = f"Cannot write download file: {write_err}"
                try:
                    file_slot.handle.close()
                except OSError:
                    pass  # nosec B110
                file_slot.handle = None
                try:
                    attempt_reply.abort()
                except (RuntimeError, AttributeError):
                    pass  # nosec B110

        def on_error(_error_code):
            attempt_reply = download_state["reply"]
            if attempt_reply is None:
                return
            download_state["error"] = attempt_reply.errorString()

        try:
            manager = QgsNetworkAccessManager.instance()
            qurl = QUrl(attempt_url)
            request = QNetworkRequest(qurl)



            request.setAttribute(RedirectPolicyAttribute, NoLessSafeRedirectPolicy)


            if resume_offset > 0:
                range_header = f"bytes={resume_offset}-"
                request.setRawHeader(
                    QByteArray(b"Range"),
                    QByteArray(range_header.encode("ascii")))


            try:
                if resume_offset > 0:
                    file_slot.handle = open(temp_path, "ab")
                else:
                    file_slot.handle = open(temp_path, "wb")
            except OSError as file_err:
                last_error = f"Cannot open download file: {file_err}"
                QgsMessageLog.logMessage(
                    last_error, "AI Segmentation", level=Qgis.MessageLevel.Warning)
                if attempt < max_retries:
                    _wait_or_cancel(_retry_wait_s(attempt))
                continue

            reply = manager.get(request)
            download_state["reply"] = reply

            reply.downloadProgress.connect(on_download_progress)
            reply.readyRead.connect(on_ready_read)
            reply.errorOccurred.connect(on_error)

            loop = QEventLoop()
            reply.finished.connect(loop.quit)




            def on_idle_timeout():
                download_state["timeout_reason"] = (
                    f"no data received for {idle_timeout_ms // 60000} minutes")
                loop.quit()

            def on_hard_timeout():
                download_state["timeout_reason"] = (
                    f"exceeded {hard_timeout_ms // 60000} minutes")
                loop.quit()

            idle_timeout = QTimer()
            idle_timeout.setSingleShot(True)
            idle_timeout.setInterval(idle_timeout_ms)
            idle_timeout.timeout.connect(on_idle_timeout)
            download_state["idle_timer"] = idle_timeout
            idle_timeout.start()

            hard_timeout = QTimer()
            hard_timeout.setSingleShot(True)
            hard_timeout.setInterval(hard_timeout_ms)
            hard_timeout.timeout.connect(on_hard_timeout)
            hard_timeout.start()





            def on_cancel_poll():
                if _cancel_requested:
                    loop.quit()

            cancel_timer = QTimer()
            cancel_timer.setInterval(500)
            cancel_timer.timeout.connect(on_cancel_poll)
            cancel_timer.start()

            if progress_callback:
                retry_msg = ""
                if attempt > 1:
                    retry_msg = " " + tr("(retry {done}/{total})").format(
                        done=attempt, total=max_retries)
                if resume_offset > 0:
                    progress_callback(5, tr("Resuming download...") + retry_msg)
                else:
                    progress_callback(5, tr("Download started...") + retry_msg)

            loop.exec()

            idle_timeout.stop()
            hard_timeout.stop()
            cancel_timer.stop()
            download_state["idle_timer"] = None

            if _consume_download_cancel():


                try:
                    reply.abort()
                except (RuntimeError, AttributeError):
                    pass  # nosec B110
                try:
                    reply.deleteLater()
                except (RuntimeError, AttributeError):
                    pass  # nosec B110
                if file_slot.handle is not None:
                    try:
                        file_slot.handle.close()
                    except OSError:
                        pass  # nosec B110
                    file_slot.handle = None
                QgsMessageLog.logMessage(
                    "Model download cancelled",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
                return False, DOWNLOAD_CANCELLED_MESSAGE

            if download_state.get("timeout_reason"):
                reply.abort()
                reply.deleteLater()
                if file_slot.handle is not None:
                    file_slot.handle.close()
                file_slot.handle = None
                last_error = f"Download timed out ({download_state['timeout_reason']})"
                if attempt < max_retries:
                    _wait_or_cancel(_retry_wait_s(attempt))
                continue

            status_code = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)


            if status_code == 416:
                if file_slot.handle is not None:
                    file_slot.handle.close()
                file_slot.handle = None
                reply.deleteLater()




                if os.path.exists(temp_path) and verify_checkpoint_hash(temp_path):
                    try:
                        finalized = _replace_with_retry(temp_path, checkpoint_path)
                    except OSError as replace_err:
                        last_error = f"Could not finalize checkpoint: {replace_err}"
                        if attempt < max_retries:
                            _wait_or_cancel(_retry_wait_s(attempt))
                        continue
                    if not finalized:
                        _consume_download_cancel()
                        return False, DOWNLOAD_CANCELLED_MESSAGE
                    if progress_callback:
                        progress_callback(100, tr("Model downloaded."))
                    QgsMessageLog.logMessage(
                        f"Checkpoint downloaded to: {checkpoint_path}",
                        "AI Segmentation", level=Qgis.MessageLevel.Success)
                    return True, "Checkpoint downloaded and verified"

                QgsMessageLog.logMessage(
                    "Server rejected range request, restarting download",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                _discard_partial_download(temp_path)
                if attempt < max_retries:
                    _wait_or_cancel(1)
                continue


            if reply.error() != QNetworkReply.NetworkError.NoError:
                last_error = download_state["error"] or reply.errorString()
                QgsMessageLog.logMessage(
                    f"Checkpoint download attempt {attempt}/{max_retries} failed: {last_error}",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                reply.deleteLater()
                if file_slot.handle is not None:
                    file_slot.handle.close()
                file_slot.handle = None




                if status_code in (403, 404, 410):
                    _discard_partial_download(temp_path)
                    refused_urls.add(attempt_url)
                    refusal = {
                        403: tr("the server refused access to the model file"),
                        404: tr("the model file is not at that address"),
                        410: tr("the model file has been removed from that address"),
                    }[status_code]
                    if len(refused_urls) < len(download_urls):



                        QgsMessageLog.logMessage(
                            f"Model download: {refusal} (HTTP {status_code}); "
                            "trying another address",
                            "AI Segmentation", level=Qgis.MessageLevel.Warning)
                        continue
                    QgsMessageLog.logMessage(
                        f"Model download stopped: {refusal} (HTTP {status_code})",
                        "AI Segmentation", level=Qgis.MessageLevel.Critical)
                    return False, tr(
                        "Model download failed: {reason}. Retrying will not "
                        "help. Update the plugin, or ask your IT administrator "
                        "whether the download is being filtered."
                    ).format(reason=refusal)
                if attempt < max_retries:
                    wait = _retry_wait_s(attempt)
                    if progress_callback:
                        progress_callback(
                            5, tr("Retry {done}/{total} in {seconds}s...").format(
                                done=attempt + 1, total=max_retries, seconds=wait))
                    _wait_or_cancel(wait)
                continue


            remaining = reply.readAll()
            if remaining and file_slot.handle is not None:
                file_slot.handle.write(remaining.data())
            if file_slot.handle is not None:
                file_slot.handle.close()
            file_slot.handle = None
            reply.deleteLater()


            file_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
            if file_size == 0:
                last_error = "Download failed: empty file"
                if attempt < max_retries:
                    _wait_or_cancel(_retry_wait_s(attempt))
                continue

            if progress_callback:
                mb_total = file_size / (1024 * 1024)
                progress_callback(
                    95, tr("Checking the {size} MB download...").format(
                        size=f"{mb_total:.1f}"))

            if not verify_checkpoint_hash(temp_path):


                QgsMessageLog.logMessage(
                    "Hash mismatch, deleting partial file and retrying",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                _discard_partial_download(temp_path)
                last_error = "Download verification failed - hash mismatch"
                declared_total = download_state.get("bytes_total") or 0
                base = download_state["resume_offset"]
                if declared_total <= 0 or file_size >= base + declared_total:




                    digest_mismatches += 1
                else:


                    digest_mismatches = 0
                if digest_mismatches >= dial_in_range(
                        "tuning.install.checkpoint_digest_mismatch_limit",
                        DIGEST_MISMATCH_LIMIT, 1, 5):
                    return False, tr(
                        "The model file arrived complete twice and did not "
                        "match its checksum either time. Something between "
                        "this computer and the download is altering the file, "
                        "usually a proxy or a security appliance. Ask your IT "
                        "administrator to let the download through untouched."
                    )
                if attempt < max_retries:
                    _wait_or_cancel(_retry_wait_s(attempt))
                continue

            if not _replace_with_retry(temp_path, checkpoint_path):
                _consume_download_cancel()
                return False, DOWNLOAD_CANCELLED_MESSAGE


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
                progress_callback(100, tr("Model downloaded."))

            QgsMessageLog.logMessage(
                f"Checkpoint downloaded to: {checkpoint_path}",
                "AI Segmentation", level=Qgis.MessageLevel.Success)
            return True, "Checkpoint downloaded and verified"

        except Exception as e:




            last_error = str(e) or type(e).__name__
            QgsMessageLog.logMessage(
                f"Checkpoint download attempt {attempt}/{max_retries} exception: {last_error}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            if file_slot.handle is not None:
                try:
                    file_slot.handle.close()
                except Exception:
                    pass  # nosec B110
                file_slot.handle = None
            if attempt < max_retries:
                _wait_or_cancel(_retry_wait_s(attempt))


    partial_mb = 0.0
    if os.path.exists(temp_path):
        partial_mb = os.path.getsize(temp_path) / (1024 * 1024)
    firewall_hint = " " + tr(
        "A firewall or proxy may be blocking the download. Check your network "
        "settings in QGIS (Settings > Options > Network).")
    failed = tr("Download failed after {attempts} attempts: {reason}").format(
        attempts=max_retries, reason=last_error)
    if partial_mb > 0:
        resume = tr(
            "Partial file ({size} MB) saved, it will resume on the next try."
        ).format(size=f"{partial_mb:.1f}")
        return False, f"{failed}. {resume}{firewall_hint}"
    return False, f"{failed}{firewall_hint}"


def cleanup_legacy_sam1_data():









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


    if os.path.exists(FEATURES_DIR):




        if not remove_tree_quietly(FEATURES_DIR):
            QgsMessageLog.logMessage(
                "Legacy features cache only partly removed",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        else:
            QgsMessageLog.logMessage(
                "Removed legacy features cache",
                "AI Segmentation", level=Qgis.MessageLevel.Info)

    cleanup_stale_partial_downloads()


def cleanup_stale_partial_downloads() -> None:








    try:
        names = os.listdir(CHECKPOINTS_DIR)
    except OSError:
        return
    from .server_dials import dial_in_range

    cutoff = time.time() - dial_in_range(
        "tuning.install.partial_download_max_age_s",
        PARTIAL_DOWNLOAD_MAX_AGE_S, 86_400, 90 * 86_400)
    for name in names:
        if not name.endswith(".tmp"):
            continue
        path = os.path.join(CHECKPOINTS_DIR, name)
        try:
            if not os.path.isfile(path) or os.path.getmtime(path) > cutoff:
                continue
        except OSError:
            continue


        if _remove_with_retry(path, max_attempts=1, delay=0.0):
            QgsMessageLog.logMessage(
                f"Removed an abandoned partial download: {name}",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
