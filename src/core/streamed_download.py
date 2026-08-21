"""One large file off the network and onto disk, without holding it in memory.

QgsBlockingNetworkRequest buffers the whole response body before a caller sees
a byte of it. That is fine for a JSON answer and wrong for an archive of tens
of megabytes: the peak is the file twice over, once in the reply and once in
the write, on the machines least likely to have the headroom for it.

This writes what arrives, as it arrives, in bounded chunks, into a ``.part``
beside the destination and moves it into place only once the transfer
finished. A cancelled or broken transfer leaves the part file and never a
half-written destination that a later run would mistake for a good one.

Redirects are followed as long as they do not downgrade the connection: a
release asset answers with a 30x to the storage host, and a request that does
not follow it comes back with an empty body and a checksum failure to explain
it.

Two watchdogs, because they catch different failures: a connection that goes
quiet stops on the idle timeout, while a slow but live one runs to the hard
one. Cancel is polled, so it is answered during the transfer and not after it.
"""
from __future__ import annotations

import os
from contextlib import suppress
from typing import Callable, NamedTuple


def tr(text: str) -> str:
    """Translate one user-facing line, falling back to English.

    The translator is imported inside the call, so this module keeps importing
    with no translator loaded and a lookup can never break a download.
    """
    try:
        from .i18n import tr as translate

        return translate(text)
    except Exception:  # noqa: BLE001 -- English is a fine answer here
        return text


#: Bytes read out of the reply at a time. Big enough that the loop is not the
#: cost, small enough that no single read is a spike of its own.
_CHUNK_BYTES = 256 * 1024

#: How often Cancel is looked at while bytes are moving.
_CANCEL_POLL_MS = 400


class StreamedDownload(NamedTuple):
    """What a transfer did. ``ok`` alone decides; the rest explains."""

    ok: bool
    error: str
    http_status: int | None
    bytes_written: int
    cancelled: bool


def _part_path(dest_path: str) -> str:
    return dest_path + ".part"


def stream_url_to_file(
    url: str,
    dest_path: str,
    timeout_ms: int,
    idle_timeout_ms: int,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> StreamedDownload:
    """Fetch ``url`` into ``dest_path``. Never raises.

    ``progress_callback`` is called with (bytes received, bytes total), total
    being 0 while the server has not said. It runs as a Qt slot, so it must be
    cheap and must not raise; anything it does raise is swallowed here rather
    than crossing back into C++.
    """
    from qgis.core import QgsNetworkAccessManager
    from qgis.PyQt.QtCore import QEventLoop, QTimer, QUrl
    from qgis.PyQt.QtNetwork import QNetworkRequest

    from .qt_compat import (
        HttpStatusCodeAttribute,
        NoLessSafeRedirectPolicy,
        RedirectPolicyAttribute,
    )

    part_path = _part_path(dest_path)
    state = {"error": "", "cancelled": False, "written": 0, "file": None}

    try:
        state["file"] = open(part_path, "wb")
    except OSError as err:
        return StreamedDownload(
            False,
            tr("Cannot open download file: {error}").format(error=err),
            None, 0, False)

    request = QNetworkRequest(QUrl(url))
    request.setAttribute(RedirectPolicyAttribute, NoLessSafeRedirectPolicy)
    if hasattr(request, "setTransferTimeout"):
        request.setTransferTimeout(max(1000, int(timeout_ms)))

    reply = QgsNetworkAccessManager.instance().get(request)
    loop = QEventLoop()

    def drain() -> None:
        # A Qt slot: an exception here crosses back into C++ and takes the
        # application with it, so a full disk ends the transfer instead.
        handle = state["file"]
        if handle is None:
            return
        try:
            while reply.bytesAvailable() > 0:
                chunk = bytes(reply.read(_CHUNK_BYTES))
                if not chunk:
                    break
                handle.write(chunk)
                state["written"] += len(chunk)
        except (OSError, RuntimeError) as err:
            state["error"] = tr(
                "Cannot write download file: {error}").format(error=err)
            _abort(reply)

    def on_progress(received: int, total: int) -> None:
        idle.start()
        if progress_callback is None:
            return
        with suppress(Exception):  # a progress line is best-effort
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
        except Exception:  # noqa: BLE001 -- an unreadable flag is not a cancel
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

    poll = QTimer()
    poll.setInterval(_CANCEL_POLL_MS)
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
        # NoError is zero in every Qt binding, so the falsy test needs no enum
        # literal and cannot be flagged by the Qt6 checker.
        with suppress(RuntimeError, AttributeError, TypeError):
            if reply.error():
                state["error"] = _network_error(reply.errorString())
    with suppress(RuntimeError, AttributeError):
        reply.deleteLater()

    handle = state["file"]
    state["file"] = None
    try:
        if handle is not None:
            handle.flush()
            handle.close()
    except OSError as err:
        if not state["error"]:
            state["error"] = tr(
                "Cannot close download file: {error}").format(error=err)

    written = int(state["written"])
    if state["cancelled"] or state["error"]:
        return StreamedDownload(
            False, state["error"], status, written, bool(state["cancelled"]))

    try:
        os.replace(part_path, dest_path)
    except OSError as err:
        return StreamedDownload(
            False,
            tr("Cannot save download: {error}").format(error=err),
            status, written, False)
    return StreamedDownload(True, "", status, written, False)


def _network_error(raw) -> str:
    """One Qt error string in a sentence the reader's own language carries.

    Qt writes ``errorString()`` in English whatever the interface language is,
    so it reached the user as the only untranslated line of a failed install.
    The raw text is kept whole inside the sentence, because it is the only
    thing that names the address and the HTTP answer, and because the caller
    reads it to tell "this archive is not published" from "try again".
    """
    return tr("the network reported: {error}").format(error=str(raw or ""))


def _abort(reply) -> None:
    """Stop a reply without ever raising out of a Qt slot."""
    with suppress(RuntimeError, AttributeError):
        reply.abort()


def discard_part_file(dest_path: str) -> None:
    """Remove the leftover of a transfer that did not finish."""
    with suppress(OSError):
        os.unlink(_part_path(dest_path))
