




















from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from contextlib import suppress
from typing import Any, BinaryIO, Callable, NamedTuple





def tr(text: str) -> str:






    try:
        from .i18n import tr as translate

        return translate(text)
    except Exception:  # noqa: BLE001
        return text




_CHUNK_BYTES = 256 * 1024


_CANCEL_POLL_MS = 400



_MIN_RESUME_BYTES = 1024 * 1024


class StreamedDownload(NamedTuple):


    ok: bool
    error: str
    http_status: int | None
    bytes_written: int
    cancelled: bool


def sleep_unless_cancelled(seconds: float, cancel_check, *, slice_s: float = 0.25, pump=None) -> bool:








    deadline = time.monotonic() + max(0.0, float(seconds))
    while True:
        if cancel_check and cancel_check():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if pump is not None:
            pump()
        time.sleep(min(slice_s, remaining))


def _part_path(dest_path: str) -> str:
    path = dest_path + ".part"
    if os.name == "nt" and len(os.path.abspath(path)) >= 260:
        return _short_sibling_path(dest_path, ".p")
    return path


def _short_sibling_path(dest_path: str, suffix: str) -> str:
    key = os.path.normcase(os.path.abspath(dest_path)).encode("utf-8")
    return os.path.join(os.path.dirname(dest_path), ".d-" + hashlib.sha256(key).hexdigest()[:16] + suffix)


def _resume_path(dest_path: str) -> str:


    return _short_sibling_path(dest_path, ".r")


def _strong_etag(value) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("latin-1")
    if not isinstance(value, str):
        return None
    value = value.strip()
    if len(value) > 512 or not re.fullmatch(r'"[\x21\x23-\x7e\x80-\xff]*"', value):
        return None
    return value


def _resume_etag(dest_path: str, url: str) -> str | None:
    try:
        with open(_resume_path(dest_path), encoding="utf-8") as handle:
            payload = handle.read(2049)
        if len(payload) > 2048:
            return None
        identity = json.loads(payload)
        if (not isinstance(identity, dict)
                or identity.get("url_sha256") != hashlib.sha256(url.encode("utf-8")).hexdigest()):
            return None
        return _strong_etag(identity.get("etag"))
    except (OSError, TypeError, ValueError, RecursionError):
        return None


def _save_resume_identity(dest_path: str, url: str, etag: str | None) -> None:
    path = _resume_path(dest_path)
    if etag is None:
        with suppress(OSError):
            os.unlink(path)
        return
    temporary = None
    try:
        handle, temporary = tempfile.mkstemp(prefix=".r-", dir=os.path.dirname(os.path.abspath(path)))
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump({"url_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(), "etag": etag}, output)
        from .file_replace_retry import replace_file_with_retry
        replace_file_with_retry(temporary, path)
    except (OSError, TypeError, ValueError):


        with suppress(OSError):
            os.unlink(path)
    finally:
        if temporary is not None:
            with suppress(OSError):
                os.unlink(temporary)


class _FileSlot:








    handle: BinaryIO | None = None


def stream_url_to_file(
    url: str,
    dest_path: str,
    timeout_ms: int,
    idle_timeout_ms: int,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> StreamedDownload:







    from qgis.core import QgsNetworkAccessManager
    from qgis.PyQt.QtCore import QEventLoop, QTimer, QUrl
    from qgis.PyQt.QtNetwork import QNetworkRequest

    from .qt_compat import (
        NoLessSafeRedirectPolicy,
        RedirectPolicyAttribute,
        reply_http_status,
    )



    if cancel_check is not None:
        with suppress(Exception):
            if cancel_check():
                return StreamedDownload(False, "", None, 0, True)
    if any(isinstance(value, bool) or not isinstance(value, int)
           or not 0 < value <= 2_147_483_647 for value in (timeout_ms, idle_timeout_ms)):
        return StreamedDownload(False, tr("Invalid download timeout"), None, 0, False)

    part_path = _part_path(dest_path)


    try:
        on_disk = os.path.getsize(part_path)
    except OSError:
        on_disk = 0
    resume_etag = _resume_etag(dest_path, url)
    resume_offset = on_disk if on_disk >= _MIN_RESUME_BYTES and resume_etag else 0
    if on_disk and not resume_offset:
        with suppress(OSError):
            os.unlink(part_path)
    state: dict[str, Any] = {"error": "", "cancelled": False, "written": 0,
                             "resume": resume_offset, "status_checked": False,
                             "expected_total": None}
    file_slot = _FileSlot()

    try:
        file_slot.handle = open(part_path, "ab" if resume_offset else "wb")
    except OSError as err:
        return StreamedDownload(
            False,
            tr("Cannot open download file: {error}").format(error=err),
            None, 0, False)

    try:
        request = QNetworkRequest(QUrl(url))
        request.setAttribute(RedirectPolicyAttribute, NoLessSafeRedirectPolicy)
        if resume_offset:
            from qgis.PyQt.QtCore import QByteArray
            request.setRawHeader(
                QByteArray(b"Range"),
                QByteArray(f"bytes={resume_offset}-".encode("ascii")))
            request.setRawHeader(b"If-Range", resume_etag.encode("latin-1"))


        request.setRawHeader(b"Accept-Encoding", b"identity")
        if hasattr(request, "setTransferTimeout"):
            request.setTransferTimeout(max(1000, int(timeout_ms)))
        reply = QgsNetworkAccessManager.instance().get(request)
    except (AttributeError, TypeError, ValueError, OverflowError, RuntimeError) as err:
        with suppress(OSError):
            file_slot.handle.close()
        file_slot.handle = None
        return StreamedDownload(False, _network_error(err), None, resume_offset, False)
    loop = QEventLoop()

    def _check_status_once() -> bool:

        if state["status_checked"]:
            return not state["error"]
        status = reply_http_status(reply)
        if status is None:
            return False
        state["status_checked"] = True
        coding = bytes(reply.rawHeader(b"Content-Encoding")).strip().lower()
        if status in (200, 206) and coding not in (b"", b"identity"):




            state["error"] = tr("The server returned an unsupported download encoding. Please try again.")
            _abort(reply)
            return False
        if status == 206:
            response_etag = bytes(reply.rawHeader(b"ETag"))
            raw = bytes(reply.rawHeader(b"Content-Range")).decode("ascii", "replace")
            match = re.fullmatch(r"bytes (\d{1,20})-(\d{1,20})/(\d{1,20})", raw.strip())
            if match:
                start, end, total = map(int, match.groups())
                if (state["resume"] and (not response_etag or _strong_etag(response_etag) == resume_etag)
                        and start == state["resume"] and start <= end < total
                        and end + 1 == total):
                    state["expected_total"] = total
                    return True
            state["error"] = tr("The server returned an invalid download range. Please try again.")
            _abort(reply)
            return False
        if status == 200:
            if state["resume"]:
                try:
                    file_slot.handle.close()
                    file_slot.handle = open(part_path, "wb")
                except OSError as err:
                    state["error"] = tr("Cannot restart the download: {error}").format(error=err)
                    file_slot.handle = None
                    _abort(reply)
                    return False
                state["resume"] = 0
            _save_resume_identity(dest_path, url, _strong_etag(bytes(reply.rawHeader(b"ETag"))))
            return True


        state["error"] = tr("Download failed") + f" (HTTP {status})"
        _abort(reply)
        return False

    def check_status() -> bool:


        try:
            return _check_status_once()
        except Exception:  # noqa: BLE001
            state["error"] = state["error"] or tr("Download failed")
            _abort(reply)
            return False

    def drain() -> None:


        if file_slot.handle is None or not check_status():
            return
        handle = file_slot.handle
        try:
            while reply.bytesAvailable() > 0:
                chunk = bytes(reply.read(_CHUNK_BYTES))
                if not chunk:
                    break
                handle.write(chunk)
                state["written"] += len(chunk)
        except (OSError, RuntimeError) as err:
            state["error"] = tr(
                "Cannot write download file: {error}").format(error=err) + " " + tr(
                "Check disk space and folder permissions, then try again.")
            _abort(reply)

    def on_progress(received: int, total: int) -> None:
        idle.start()
        check_status()
        if progress_callback is None:
            return
        with suppress(Exception):
            offset = int(state["resume"])
            progress_callback(int(received) + offset, int(total) + offset if total > 0 else 0)

    def on_error(_code) -> None:
        if state["error"] or state["cancelled"]:
            return
        try:
            state["error"] = _network_error(reply.errorString())
        except (RuntimeError, AttributeError):
            state["error"] = tr("Download failed")

    def on_idle() -> None:
        state["error"] = tr("the download stalled, no data was received")
        _abort(reply)
        loop.quit()

    def on_hard_timeout() -> None:
        state["error"] = tr("the download did not finish in time")
        _abort(reply)
        loop.quit()

    def on_cancel_poll() -> None:
        if cancel_check is None:
            return
        try:
            wants_stop = bool(cancel_check())
        except Exception:  # noqa: BLE001
            return
        if wants_stop:
            state["cancelled"] = True
            _abort(reply)
            loop.quit()

    idle = QTimer()
    idle.setSingleShot(True)
    idle.setInterval(max(1000, int(idle_timeout_ms)))
    idle.timeout.connect(on_idle)

    hard = QTimer()
    hard.setSingleShot(True)
    hard.setInterval(max(1000, int(timeout_ms)))
    hard.timeout.connect(on_hard_timeout)

    from .server_dials import dial_in_range

    poll = QTimer()
    poll.setInterval(dial_in_range(
        "tuning.install.download_cancel_poll_ms", _CANCEL_POLL_MS, 100, 5000))
    poll.timeout.connect(on_cancel_poll)

    reply.readyRead.connect(drain)
    reply.downloadProgress.connect(on_progress)
    if hasattr(reply, "errorOccurred"):
        reply.errorOccurred.connect(on_error)
    reply.finished.connect(loop.quit)

    idle.start()
    hard.start()
    poll.start()
    try:
        loop.exec()
    finally:
        for timer in (idle, hard, poll):
            timer.stop()

    drain()
    check_status()
    on_cancel_poll()
    status = reply_http_status(reply)
    if not state["error"] and not state["cancelled"]:




        with suppress(RuntimeError, AttributeError, TypeError):
            from qgis.PyQt.QtNetwork import QNetworkReply

            from .qt_compat import resolve_qt_enum
            no_error = resolve_qt_enum(QNetworkReply, "NetworkError", "NoError")
            if reply.error() != no_error:
                state["error"] = _network_error(reply.errorString())
    with suppress(RuntimeError, AttributeError):
        reply.deleteLater()

    handle = file_slot.handle
    file_slot.handle = None
    try:
        if handle is not None:
            handle.flush()
            handle.close()
    except OSError as err:
        if not state["error"]:
            state["error"] = tr(
                "Cannot close download file: {error}").format(error=err)



    written = int(state["written"]) + int(state["resume"])
    incomplete = (written == 0 or (
        state["expected_total"] is not None and written != state["expected_total"]))
    if not state["error"] and not state["cancelled"] and incomplete:
        state["error"] = tr("The download is incomplete. Please try again.")
    if state["cancelled"] or state["error"]:
        return StreamedDownload(
            False, state["error"], status, written, bool(state["cancelled"]))

    try:


        from .file_replace_retry import replace_file_with_retry

        replace_file_with_retry(part_path, dest_path)
    except OSError as err:
        return StreamedDownload(
            False,
            tr("Cannot save download: {error}").format(error=err),
            status, written, False)
    with suppress(OSError):
        os.unlink(_resume_path(dest_path))
    return StreamedDownload(True, "", status, written, False)


def _network_error(raw) -> str:








    return tr("the network reported: {error}").format(error=str(raw or ""))


def _abort(reply) -> None:

    with suppress(RuntimeError, AttributeError):
        reply.abort()


def discard_part_file(dest_path: str) -> None:

    with suppress(OSError):
        os.unlink(_part_path(dest_path))
    with suppress(OSError):
        os.unlink(_resume_path(dest_path))
