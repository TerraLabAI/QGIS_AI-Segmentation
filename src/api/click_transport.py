





































from __future__ import annotations

import threading

from qgis.core import QgsNetworkAccessManager
from qgis.PyQt.QtCore import QByteArray, QEvent, QEventLoop, QObject, QTimer, QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from ..core.click_phase_clock import active_click_clock, click_clock_now
from ..core.qt_compat import resolve_qt_enum
from ..core.server_dials import dial_in_range



_KEEP_PAINTING = resolve_qt_enum(QEventLoop, "ProcessEventsFlag",
                                 "ExcludeUserInputEvents")
_ALL_EVENTS = resolve_qt_enum(QEventLoop, "ProcessEventsFlag", "AllEvents")











_CLICK_WAIT_MAX_MS = 45_000





_CLICK_WAIT_FLOOR_MS = 5_000
_CLICK_WAIT_CEILING_MS = 300_000











_CLICK_INPUT_HOLD_MS = 2_500
_CLICK_INPUT_HOLD_FLOOR_MS = 250
_CLICK_INPUT_HOLD_CEILING_MS = 300_000




_WAIT_GUARD_MS = 5_000




_CANCEL_POLL_MS = 100





_cancel_lock = threading.Lock()
_cancel_generation = 0


def cancel_click_wait() -> None:







    global _cancel_generation
    with _cancel_lock:
        _cancel_generation += 1


def _click_wait_generation() -> int:


    return _cancel_generation


def _transport_ceiling_ms() -> int:








    try:
        value = int(QgsNetworkAccessManager.timeout())
    except Exception:  # noqa: BLE001
        return _CLICK_WAIT_CEILING_MS
    return value if value >= _CLICK_WAIT_FLOOR_MS else _CLICK_WAIT_CEILING_MS


def click_wait_max_ms() -> int:






    try:
        from ..core.detection_policy import network_policy

        value = network_policy().get("click_wait_max_ms")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and _CLICK_WAIT_FLOOR_MS <= value <= _CLICK_WAIT_CEILING_MS):
            return min(int(value), _transport_ceiling_ms())
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return min(_CLICK_WAIT_MAX_MS, _transport_ceiling_ms())


def click_input_hold_ms() -> int:





    try:
        from ..core.detection_policy import network_policy

        value = network_policy().get("click_input_hold_ms")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and _CLICK_INPUT_HOLD_FLOOR_MS <= value <= _CLICK_INPUT_HOLD_CEILING_MS):
            return int(value)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return _CLICK_INPUT_HOLD_MS





_CONNECTION_WARM_TIMEOUT_MS = 20_000



_warming_replies = set()


def warm_click_connection(url: str) -> bool:

















    if not url:
        return False
    reply = None
    try:
        manager = QgsNetworkAccessManager.instance()
        if manager is None:
            return False
        request = QNetworkRequest(QUrl(url))

        if hasattr(request, "setTransferTimeout"):
            request.setTransferTimeout(dial_in_range(
                "tuning.click.connection_warm_timeout_ms",
                _CONNECTION_WARM_TIMEOUT_MS, 2000, 60000))
        reply = manager.get(request)
        if reply is None:
            return False

        def _done() -> None:
            _warming_replies.discard(reply)
            try:
                reply.deleteLater()
            except RuntimeError:
                pass  # nosec B110





        reply.finished.connect(_done)
        reply.destroyed.connect(lambda *_args: _warming_replies.discard(reply))
        _warming_replies.add(reply)
        if reply.isFinished():
            _done()
        return True
    except Exception:  # noqa: BLE001
        if reply is not None:
            _warming_replies.discard(reply)
            try:
                reply.abort()
                reply.deleteLater()
            except RuntimeError:  # nosec B110
                pass
        return False


class _SwallowMousePresses(QObject):


    def eventFilter(self, _obj, event):
        return event.type() in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonDblClick,
            QEvent.Type.MouseButtonRelease,
        )


def _map_viewport():

    try:
        from qgis.utils import iface

        canvas = iface.mapCanvas() if iface is not None else None
        return canvas.viewport() if canvas is not None else None
    except Exception:  # noqa: BLE001
        return None


def _drop_held_clicks() -> None:








    try:
        from qgis.PyQt.QtWidgets import QApplication

        target = _map_viewport()
        if target is None:
            return
        swallow = _SwallowMousePresses()
        target.installEventFilter(swallow)
        try:
            QApplication.processEvents(_ALL_EVENTS, 0)
        finally:
            target.removeEventFilter(swallow)
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _wait_with_the_window_free(loop) -> None:












    target = _map_viewport()
    if target is None:
        loop.exec(_KEEP_PAINTING)
        return
    swallow = _SwallowMousePresses()
    target.installEventFilter(swallow)
    try:
        loop.exec(_ALL_EVENTS)
    finally:
        try:
            target.removeEventFilter(swallow)
        except RuntimeError:
            pass  # nosec B110


class ClickPostAbandoned(Exception):








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
    packed: bool = False,
) -> tuple[bytes, int | None, object] | None:

























    started_generation = _click_wait_generation()
    wait_ms = min(int(timeout_ms), click_wait_max_ms())
    try:
        cancelled = cancel_check is not None and bool(cancel_check())
    except Exception:  # noqa: BLE001
        cancelled = False
    if cancelled:
        raise ClickPostAbandoned(cancelled=True)
    try:
        manager = QgsNetworkAccessManager.instance()
        if manager is None:
            return None
        request = QNetworkRequest(QUrl(url))
        request.setRawHeader(b"Content-Type", b"application/json")
        if packed:


            request.setRawHeader(b"Content-Encoding", b"gzip")

        if hasattr(request, "setTransferTimeout"):
            request.setTransferTimeout(wait_ms)
        apply_redirect_policy(request, bool(auth))
        for key, value in (auth or {}).items():
            request.setRawHeader(key.encode("utf-8"), value.encode("utf-8"))

        payload = QByteArray(body)
    except Exception:  # noqa: BLE001
        return None
    try:
        reply = manager.post(request, payload)
    except Exception:  # noqa: BLE001
        raise ClickPostAbandoned() from None
    if reply is None:
        raise ClickPostAbandoned()
    sent_at = click_clock_now()




    loop = QEventLoop()
    guard = QTimer()
    guard.setSingleShot(True)
    guard.setInterval(wait_ms + _WAIT_GUARD_MS)

    hold = QTimer()
    hold.setSingleShot(True)
    hold.setInterval(min(click_input_hold_ms(), wait_ms))
    watch = None
    held_input = False
    state = {"cancelled": False}

    def _poll_owner() -> None:
        try:
            if (_click_wait_generation() != started_generation
                    or (cancel_check is not None and cancel_check())):
                state["cancelled"] = True
                loop.quit()
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    try:
        reply.finished.connect(loop.quit)
        guard.timeout.connect(loop.quit)
        hold.timeout.connect(loop.quit)
        guard.start()
        watch = QTimer()
        watch.setInterval(_CANCEL_POLL_MS)
        watch.timeout.connect(_poll_owner)
        watch.start()
        if not _is_finished(reply):
            held_input = True
            hold.start()
            loop.exec(_KEEP_PAINTING)
            hold.stop()




            if (not _is_finished(reply) and not state["cancelled"]
                    and guard.isActive()):
                _wait_with_the_window_free(loop)
    except Exception:  # noqa: BLE001
        _end(reply, guard, watch)
        _stop_timer(hold)
        raise ClickPostAbandoned() from None
    guard.stop()
    _stop_timer(hold)




    try:
        if state["cancelled"] or not _is_finished(reply):



            _end(reply, guard, watch)
            raise ClickPostAbandoned(cancelled=state["cancelled"])
        try:
            raw = bytes(reply.readAll())
            error = reply.error()
            status = reply.attribute(
                resolve_qt_enum(QNetworkRequest, "Attribute",
                                "HttpStatusCodeAttribute"))
        except Exception:  # noqa: BLE001
            _end(reply, guard, watch)
            raise ClickPostAbandoned() from None
        try:
            status = int(status) if status is not None else None
        except (TypeError, ValueError):



            status = None
        _end(reply, guard, watch)
        clock = active_click_clock()
        if clock is not None:
            clock.note_request(sent_at, click_clock_now(), len(body), len(raw))
        return raw, status, error
    finally:
        if held_input:
            _drop_held_clicks()


def _is_finished(reply) -> bool:


    try:
        return bool(reply.isFinished())
    except Exception:  # noqa: BLE001
        return False


def _stop_timer(timer) -> None:

    if timer is None:
        return
    try:
        timer.stop()
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _end(reply, guard, watch=None) -> None:

    for timer in (guard, watch):
        _stop_timer(timer)
    try:
        if not _is_finished(reply):
            reply.abort()
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    try:
        reply.deleteLater()
    except Exception:  # noqa: BLE001  # nosec B110
        pass
