








from __future__ import annotations

import json
import time

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtNetwork import QNetworkReply, QNetworkRequest

from ..core import transport_dials as _td

__all__ = [
    "_Attr",
    "_AuthRequired",
    "_ConnRefused",
    "_ContentDenied",
    "_HTTP_STATUS_ATTR",
    "_HostNotFound",
    "_NE",
    "_NO_LESS_SAFE_REDIRECT",
    "_NoError",
    "_OpCanceled",
    "_REDIRECT_ATTR",
    "_RedirectPolicy",
    "_SAME_ORIGIN_REDIRECT",
    "_SERVER_CONTACT_TTL_S",
    "_SslFailed",
    "_TIMEOUT_API",
    "_TIMEOUT_CHECKOUT_LINK",
    "_TIMEOUT_INTERACTIVE",
    "_TIMEOUT_POLL_DETECTION",
    "_TIMEOUT_RUN_EXPORT",
    "_TIMEOUT_SUBMIT_DETECTION",
    "_TIMEOUT_SUBMIT_DETECTION_DIRECT",
    "_TIMEOUT_TRANSLATE",
    "_TIMEOUT_WARMUP",
    "_Timeout",
    "_UnknownNetwork",
    "_WALL_CLOCK_GUARD_MS",
    "_WallClockGuard",
    "_apply_redirect_policy",
    "_http_status_of",
    "_log_warning",
    "_parse_json_body",
    "_reply_was_packed",
    "note_server_contact",
    "server_reached_recently",
]

_TIMEOUT_API = 30_000
_TIMEOUT_INTERACTIVE = 10_000

_TIMEOUT_SUBMIT_DETECTION = 45_000






_TIMEOUT_SUBMIT_DETECTION_DIRECT = 110_000
_TIMEOUT_POLL_DETECTION = 15_000
_TIMEOUT_WARMUP = 5_000




_TIMEOUT_TRANSLATE = 12_000




_TIMEOUT_CHECKOUT_LINK = 4_000

_TIMEOUT_RUN_EXPORT = 60_000




_WALL_CLOCK_GUARD_MS = 5_000


_NE = getattr(QNetworkReply, "NetworkError", QNetworkReply)
_HostNotFound = getattr(_NE, "HostNotFoundError", getattr(QNetworkReply, "HostNotFoundError", None))
_ConnRefused = getattr(_NE, "ConnectionRefusedError", getattr(QNetworkReply, "ConnectionRefusedError", None))
_Timeout = getattr(_NE, "TimeoutError", getattr(QNetworkReply, "TimeoutError", None))






_OpCanceled = getattr(_NE, "OperationCanceledError", getattr(QNetworkReply, "OperationCanceledError", None))
_SslFailed = getattr(_NE, "SslHandshakeFailedError", getattr(QNetworkReply, "SslHandshakeFailedError", None))
_ContentDenied = getattr(_NE, "ContentAccessDenied", getattr(QNetworkReply, "ContentAccessDenied", None))
_AuthRequired = getattr(_NE, "AuthenticationRequiredError", getattr(QNetworkReply, "AuthenticationRequiredError", None))
_UnknownNetwork = getattr(_NE, "UnknownNetworkError", getattr(QNetworkReply, "UnknownNetworkError", None))
_NoError = getattr(_NE, "NoError", getattr(QNetworkReply, "NoError", 0))



_Attr = getattr(QNetworkRequest, "Attribute", QNetworkRequest)
_HTTP_STATUS_ATTR = getattr(_Attr, "HttpStatusCodeAttribute", getattr(QNetworkRequest, "HttpStatusCodeAttribute", None))



_REDIRECT_ATTR = getattr(_Attr, "RedirectPolicyAttribute",
                         getattr(QNetworkRequest, "RedirectPolicyAttribute", None))
_RedirectPolicy = getattr(QNetworkRequest, "RedirectPolicy", QNetworkRequest)
_NO_LESS_SAFE_REDIRECT = getattr(_RedirectPolicy, "NoLessSafeRedirectPolicy",
                                 getattr(QNetworkRequest, "NoLessSafeRedirectPolicy", None))
_SAME_ORIGIN_REDIRECT = getattr(_RedirectPolicy, "SameOriginRedirectPolicy",
                                getattr(QNetworkRequest, "SameOriginRedirectPolicy", None))


def _log_warning(msg: str):
    QgsMessageLog.logMessage(msg, "AI Segmentation", level=Qgis.MessageLevel.Warning)


class _WallClockGuard:









    def __init__(self, blocker, timeout_ms: int) -> None:
        from qgis.PyQt.QtCore import QTimer

        self._blocker = blocker
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(max(1_000, int(timeout_ms)) + _WALL_CLOCK_GUARD_MS)
        self._timer.timeout.connect(self._end_it)
        try:
            self._timer.start()
        except (RuntimeError, TypeError):
            pass  # nosec B110

    def _end_it(self) -> None:
        try:
            self._blocker.abort()
        except (AttributeError, RuntimeError) as err:
            _log_warning(f"Could not end a stalled request: {err}")

    def stop(self) -> None:
        try:
            self._timer.stop()
        except RuntimeError:
            pass  # nosec B110








_SERVER_CONTACT_TTL_S = 180.0
_last_server_contact_monotonic: float | None = None


def note_server_contact() -> None:




    global _last_server_contact_monotonic
    _last_server_contact_monotonic = time.monotonic()


def server_reached_recently() -> bool:







    stamp = _last_server_contact_monotonic
    if stamp is None:
        return False
    age = time.monotonic() - stamp
    return 0.0 <= age <= _td.server_contact_ttl_s(_SERVER_CONTACT_TTL_S)


def _apply_redirect_policy(req: QNetworkRequest, has_auth: bool) -> None:







    if _REDIRECT_ATTR is None:
        return
    policy = _SAME_ORIGIN_REDIRECT if has_auth else _NO_LESS_SAFE_REDIRECT
    if policy is not None:
        req.setAttribute(_REDIRECT_ATTR, policy)


def _parse_json_body(raw_body: str, allow_list: bool = False):










    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    def finite_float(value):
        number = float(value)
        if abs(number) == float("inf"):
            raise ValueError("JSON number is outside the finite range")
        return number

    parsed = json.loads(raw_body, parse_constant=reject_constant, parse_float=finite_float)
    if isinstance(parsed, dict):
        return parsed
    if allow_list and isinstance(parsed, list):
        return parsed
    return None


def _http_status_of(reply) -> int | None:









    if reply is None or _HTTP_STATUS_ATTR is None:
        return None
    try:
        attr = reply.attribute(_HTTP_STATUS_ATTR)
        if attr is None:
            return None
        if isinstance(attr, bool):
            return None
        status = int(attr)
        return status if 100 <= status <= 599 else None
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return None


def _reply_was_packed(reply) -> bool:






    try:
        return bytes(reply.request().rawHeader(b"Content-Encoding")) == b"gzip"
    except Exception:  # noqa: BLE001
        return False
