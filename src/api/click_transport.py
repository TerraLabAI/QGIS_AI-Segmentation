"""One round trip for the click path, taken without freezing the window.

The plugin's general client blocks its whole thread for the length of a
request. On a background task that is the right shape. On a click it is not: a
click is answered on the thread that draws, so a wait there stops every
repaint, every cursor change and every animation until the answer lands. A
fifth of a second of that reads as the plugin hanging, and it is exactly what
the remote click path spends on a good connection.

This takes the same request with a local event loop instead. Drawing and
timers keep running; mouse and key events are held back until the click being
served is finished, so nothing can start a second click on top of the first.

It is not a second HTTP client. The request itself, the redirect rule, the
deadline and the error wording all stay in ``terralab_client``; this only
changes how the caller waits.

Two outcomes, and the caller must keep them apart. Returning None means
nothing was ever sent, so the caller is free to send it on its blocking path.
Raising ``ClickPostAbandoned`` means the request DID go out and no usable
answer came back, and then sending it again would set a second operation going
on the same crop. That is a terminal end for the click, not a retry.

The wait itself is cancellable. Timers and queued signals run inside it, so the
session that asked for the click can end while the answer is still on its way
(the mode switched, the plugin unloaded, the user signed out). The caller hands
in ``cancel_check`` and the wait polls it, ends the reply and reports the click
as abandoned rather than letting the answer land on a session that is gone.
Anything that has no event to ride on calls ``cancel_click_wait`` instead: user
input is what the wait holds back, so a cancel that needs a gesture to be
noticed would never be noticed.
"""
from __future__ import annotations

from qgis.core import QgsNetworkAccessManager
from qgis.PyQt.QtCore import QByteArray, QEvent, QEventLoop, QObject, QTimer, QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from ..core.qt_compat import resolve_qt_enum

# Paint and timers run, user input waits. Without this a click could land
# inside the click it is already answering.
_KEEP_PAINTING = resolve_qt_enum(QEventLoop, "ProcessEventsFlag",
                                 "ExcludeUserInputEvents")
_ALL_EVENTS = resolve_qt_enum(QEventLoop, "ProcessEventsFlag", "AllEvents")

# The longest a click may hold the window with user input held back. The
# submit deadline this is capped against belongs to a tile in a background
# run, where a slow day costs nobody anything and the dial can be opened to
# minutes. Here every one of those seconds is a map that answers nothing, so
# the cap sits well below the dial.
#
# It is a dial itself (``network.click_wait_max_ms``), because a shipped
# constant here caps a live server value: if the cold path ever answers slower
# than this, every remote click fails and no deploy could raise the ceiling.
# The constant below is the fallback.
_CLICK_WAIT_MAX_MS = 45_000

# Bounds on the served cap. Below the floor the wait would expire on almost
# every click, above the ceiling one bad deploy could hold the window for
# minutes with user input held back. Outside either, the shipped constant
# stands. Same rule as the submit deadline this is capped against.
_CLICK_WAIT_FLOOR_MS = 5_000
_CLICK_WAIT_CEILING_MS = 300_000

# Qt's own transfer deadline ends the reply. This second one only ends the
# WAIT, for the case where the reply never reports anything at all, and it
# sits well past Qt's so it never fires first.
_WAIT_GUARD_MS = 5_000

# How often the wait asks whether the session that wanted this click is still
# there. Short enough that a mode switch or an unload is not held up by a
# request nobody is waiting for any more.
_CANCEL_POLL_MS = 100

# Set from anywhere, read by the poll. A plain flag rather than a gesture,
# because a gesture is exactly what the wait cannot see.
_cancel_requested = False


# UNREACHABLE (2026-08-11): no caller yet. The teardowns that must end a click
# in flight (unload, mode switch, sign-out) sit in the plugin layer. Kept
# because it is the only escape from the wait that does not need user input,
# which the wait holds back.
def cancel_click_wait() -> None:
    """End any click wait in progress. Safe from any thread."""
    global _cancel_requested
    _cancel_requested = True


def click_wait_max_ms() -> int:
    """The longest one click may hold the window, in milliseconds.

    Read per click off the served network policy, bounded by the floor and
    ceiling above. Never raises and never networks: a click path calls it.
    """
    try:
        from ..core.detection_policy import network_policy

        value = network_policy().get("click_wait_max_ms")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and _CLICK_WAIT_FLOOR_MS <= value <= _CLICK_WAIT_CEILING_MS):
            return int(value)
    except Exception:  # noqa: BLE001 -- a dial must never break a click  # nosec B110
        pass
    return _CLICK_WAIT_MAX_MS


# How long the connection warm may stay open before Qt ends it. It exists to
# leave a live socket behind, not to read anything, and a request still hanging
# long after the user started working is holding a socket nobody will use.
_CONNECTION_WARM_TIMEOUT_MS = 20_000

# The replies in flight for a warm, held only so Python does not collect one
# mid-request. Cleared as each finishes.
_warming_replies = set()


def warm_click_connection(url: str) -> bool:
    """Open the connection the first click will need, before the click.

    A click is sent on the thread that draws, through that thread's network
    manager. The crop hand-over travels on a worker thread with a manager of
    its own, and the wake ping on a third, so neither leaves a socket the click
    can reuse: the first click of every session pays a name lookup, a
    connection and a TLS handshake of its own before its first byte moves. That
    is a few hundred milliseconds on a slow link, and it is paid at the one
    moment the user is watching.

    So this sends one throwaway GET from the calling thread, which must be the
    thread that will answer the clicks. The answer is dropped. What is kept is
    the open connection Qt now holds for that host.

    Returns whether a request went out. Never raises: a warm that fails leaves
    the click exactly as slow as it is today, and nothing else.
    """
    if not url:
        return False
    try:
        manager = QgsNetworkAccessManager.instance()
        if manager is None:
            return False
        request = QNetworkRequest(QUrl(url))
        request.setTransferTimeout(_CONNECTION_WARM_TIMEOUT_MS)
        reply = manager.get(request)
        if reply is None:
            return False

        def _done() -> None:
            _warming_replies.discard(reply)
            reply.deleteLater()

        # Connected BEFORE the reply is held, so nothing can be held by a
        # connection that never happened: a reply added first and then failing
        # to connect would sit in this set for the life of the process with
        # nothing left to take it out.
        reply.finished.connect(_done)
        _warming_replies.add(reply)
        return True
    except Exception:  # noqa: BLE001 -- a warm must never reach the user
        return False


class _SwallowMousePresses(QObject):
    """Eat map clicks for as long as it is installed. See _drop_held_clicks."""

    def eventFilter(self, _obj, event):
        return event.type() in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonDblClick,
            QEvent.Type.MouseButtonRelease,
        )


def _drop_held_clicks() -> None:
    """Throw away the map clicks Qt kept back during the wait.

    Excluding user input defers it, it does not drop it, so every click made
    while the answer was travelling arrives the moment the wait ends and each
    one starts a round trip of its own. The user made those clicks because
    nothing was happening, not because they wanted four more points on the
    object. Never raises: a stray click is not worth failing a served answer.
    """
    try:
        from qgis.PyQt.QtWidgets import QApplication
        from qgis.utils import iface

        canvas = iface.mapCanvas() if iface is not None else None
        target = canvas.viewport() if canvas is not None else None
        if target is None:
            return
        swallow = _SwallowMousePresses()
        target.installEventFilter(swallow)
        try:
            QApplication.processEvents(_ALL_EVENTS, 0)
        finally:
            target.removeEventFilter(swallow)
    except Exception:  # noqa: BLE001 -- a dropped click must never fail a click  # nosec B110
        pass


class ClickPostAbandoned(Exception):
    """The request went out and no usable answer came back.

    The one thing a caller must not do with this is send the same request
    again: the far side may already be acting on the first one. ``cancelled``
    tells apart the two ways to get here, an owner that went away and a wait
    that ran out with nothing reported.
    """

    def __init__(self, cancelled: bool = False) -> None:
        super().__init__("click cancelled" if cancelled else "no answer")
        self.cancelled = cancelled


def post_and_keep_painting(
    url: str,
    body: bytes,
    auth: dict | None,
    timeout_ms: int,
    apply_redirect_policy,
    cancel_check=None,
) -> tuple[bytes, int | None, object] | None:
    """Send one POST and wait for it without stopping the window.

    Returns ``(raw body, HTTP status or None, Qt network error)``.

    Returns None when the request was never sent, and only then: the caller may
    take its blocking path. Raises ``ClickPostAbandoned`` once the request has
    gone out without an answer, because sending it a second time would act
    twice.

    ``cancel_check`` is an optional callable polled during the wait. As soon as
    it answers True the reply is ended and the click is abandoned. So is
    ``cancel_click_wait``, which any thread can call.

    ``timeout_ms`` is the caller's deadline, capped at ``click_wait_max_ms()``:
    a click holds the window, and how long a background tile may take is not
    how long a user may be left with a map that answers nothing.
    """
    global _cancel_requested
    _cancel_requested = False
    wait_ms = min(int(timeout_ms), click_wait_max_ms())
    try:
        manager = QgsNetworkAccessManager.instance()
        if manager is None:
            return None
        request = QNetworkRequest(QUrl(url))
        request.setRawHeader(b"Content-Type", b"application/json")
        request.setTransferTimeout(wait_ms)
        apply_redirect_policy(request, bool(auth))
        for key, value in (auth or {}).items():
            request.setRawHeader(key.encode("utf-8"), value.encode("utf-8"))

        reply = manager.post(request, QByteArray(body))
        if reply is None:
            return None
    except Exception:  # noqa: BLE001 -- the caller still has its own path
        return None

    # Past this line the request exists on the wire, so every way out of this
    # function is either the answer or ClickPostAbandoned. None is no longer
    # available, and the caller must not send this body again.
    loop = QEventLoop()
    guard = QTimer()
    guard.setSingleShot(True)
    guard.setInterval(wait_ms + _WAIT_GUARD_MS)
    watch = None
    held_input = False
    state = {"cancelled": False}

    def _poll_owner() -> None:
        try:
            if _cancel_requested or (cancel_check is not None and cancel_check()):
                state["cancelled"] = True
                loop.quit()
        except Exception:  # noqa: BLE001 -- an unreadable owner is not a cancel  # nosec B110
            pass

    try:
        reply.finished.connect(loop.quit)
        guard.timeout.connect(loop.quit)
        guard.start()
        watch = QTimer()
        watch.setInterval(_CANCEL_POLL_MS)
        watch.timeout.connect(_poll_owner)
        watch.start()
        if not _is_finished(reply):
            held_input = True
            loop.exec(_KEEP_PAINTING)
    except Exception:  # noqa: BLE001 -- the request is out, so this ends it
        _end(reply, guard, watch)
        raise ClickPostAbandoned() from None
    guard.stop()

    # The answer is read first and the held-back clicks are dropped on the way
    # out, whichever way that is: dropping them runs one turn of the event
    # loop, and this reply must be read before anything else can run.
    try:
        if state["cancelled"] or not _is_finished(reply):
            # Either the session that wanted this click is gone, or the wait
            # ran out with nothing reported. Leaving the reply running would
            # leak it, so end it here; the click ends with it.
            _end(reply, guard, watch)
            raise ClickPostAbandoned(cancelled=state["cancelled"])
        try:
            raw = bytes(reply.readAll())
            error = reply.error()
            status = reply.attribute(
                resolve_qt_enum(QNetworkRequest, "Attribute",
                                "HttpStatusCodeAttribute"))
            status = int(status) if status is not None else None
        except Exception:  # noqa: BLE001 -- an unreadable reply is not an answer
            _end(reply, guard, watch)
            raise ClickPostAbandoned() from None
        _end(reply, guard, watch)
        return raw, status, error
    finally:
        if held_input:
            _drop_held_clicks()


def _is_finished(reply) -> bool:
    """Whether the reply reports itself done. A reply that cannot answer that
    is treated as unfinished, which is the reading that ends it cleanly."""
    try:
        return bool(reply.isFinished())
    except Exception:  # noqa: BLE001 -- an unreadable reply is not a finished one
        return False


def _end(reply, guard, watch=None) -> None:
    """Close one reply down for good. Never raises: this runs on failure paths."""
    for timer in (guard, watch):
        if timer is None:
            continue
        try:
            timer.stop()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    try:
        if not _is_finished(reply):
            reply.abort()
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    try:
        reply.deleteLater()
    except Exception:  # noqa: BLE001  # nosec B110
        pass
