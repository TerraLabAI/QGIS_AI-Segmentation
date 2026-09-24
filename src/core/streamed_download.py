




















from __future__ import annotations

import os
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


def sleep_unless_cancelled(seconds: float, cancel_check) -> bool:







    deadline = time.monotonic() + max(0.0, float(seconds))
    while True:
        if cancel_check and cancel_check():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.25, remaining))


def _part_path(dest_path: str) -> str:
    return dest_path + ".part"


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
        HttpStatusCodeAttribute,
        NoLessSafeRedirectPolicy,
        RedirectPolicyAttribute,
    )

    part_path = _part_path(dest_path)


    try:
        on_disk = os.path.getsize(part_path)
    except OSError:
        on_disk = 0
    resume_offset = on_disk if on_disk >= _MIN_RESUME_BYTES else 0
    if on_disk and not resume_offset:
        with suppress(OSError):
            os.unlink(part_path)
    state: dict[str, Any] = {"error": "", "cancelled": False, "written": 0,
                             "resume": resume_offset, "status_checked": False}
    file_slot = _FileSlot()

    try:
        file_slot.handle = open(part_path, "ab" if resume_offset else "wb")
    except OSError as err:
        return StreamedDownload(
            False,
            tr("Cannot open download file: {error}").format(error=err),
            None, 0, False)

    request = QNetworkRequest(QUrl(url))
    request.setAttribute(RedirectPolicyAttribute, NoLessSafeRedirectPolicy)
    if resume_offset:
        from qgis.PyQt.QtCore import QByteArray
        request.setRawHeader(
            QByteArray(b"Range"),
            QByteArray(f"bytes={resume_offset}-".encode("ascii")))
    if hasattr(request, "setTransferTimeout"):
        request.setTransferTimeout(max(1000, int(timeout_ms)))

    reply = QgsNetworkAccessManager.instance().get(request)
    loop = QEventLoop()

    def drain() -> None:


        handle = file_slot.handle
        if handle is None:
            return




        if state["resume"] and not state["status_checked"]:
            state["status_checked"] = True
            status_now = None
            if HttpStatusCodeAttribute is not None:
                with suppress(RuntimeError, AttributeError):
                    status_now = reply.attribute(HttpStatusCodeAttribute)
            if status_now == 200:
                try:
                    handle.close()
                    handle = open(part_path, "wb")
                    file_slot.handle = handle
                except OSError as err:
                    state["error"] = tr(
                        "Cannot restart the download: {error}").format(error=err)
                    file_slot.handle = None
                    _abort(reply)
                    return
                state["resume"] = 0
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
        if progress_callback is None:
            return
        with suppress(Exception):
            progress_callback(int(received), int(total))

    def on_error(_code) -> None:
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
    status = None
    if HttpStatusCodeAttribute is not None:
        with suppress(RuntimeError, AttributeError):
            status = reply.attribute(HttpStatusCodeAttribute)
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
    return StreamedDownload(True, "", status, written, False)


def _network_error(raw) -> str:








    return tr("the network reported: {error}").format(error=str(raw or ""))


def _abort(reply) -> None:

    with suppress(RuntimeError, AttributeError):
        reply.abort()


def discard_part_file(dest_path: str) -> None:

    with suppress(OSError):
        os.unlink(_part_path(dest_path))
