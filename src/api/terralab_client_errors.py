







from __future__ import annotations

from qgis.core import QgsBlockingNetworkRequest
from qgis.PyQt.QtNetwork import QNetworkReply

from ..core.i18n import tr
from .terralab_client_primitives import (
    _NE,
    _AuthRequired,
    _ConnRefused,
    _ContentDenied,
    _HostNotFound,
    _http_status_of,
    _log_warning,
    _OpCanceled,
    _SslFailed,
    _Timeout,
    _UnknownNetwork,
    server_reached_recently,
)







_CONNECT_FAILURE_ERRORS = set(filter(None, [
    getattr(_NE, "TemporaryNetworkFailureError",
            getattr(QNetworkReply, "TemporaryNetworkFailureError", None)),
    getattr(_NE, "NetworkSessionFailedError",
            getattr(QNetworkReply, "NetworkSessionFailedError", None)),
]))

_PROXY_ERRORS = set(filter(None, [
    getattr(_NE, "ProxyConnectionRefusedError", getattr(QNetworkReply, "ProxyConnectionRefusedError", None)),
    getattr(_NE, "ProxyConnectionClosedError", getattr(QNetworkReply, "ProxyConnectionClosedError", None)),
    getattr(_NE, "ProxyNotFoundError", getattr(QNetworkReply, "ProxyNotFoundError", None)),
    getattr(_NE, "ProxyTimeoutError", getattr(QNetworkReply, "ProxyTimeoutError", None)),
    getattr(_NE, "ProxyAuthenticationRequiredError", getattr(QNetworkReply, "ProxyAuthenticationRequiredError", None)),
    getattr(_NE, "UnknownProxyError", getattr(QNetworkReply, "UnknownProxyError", None)),
]))


def _named_error_text(body: dict) -> str:







    for key in ("error", "message", "detail"):
        value = body.get(key)
        if isinstance(value, str):
            if value.strip():
                return value.strip()[:400]
            continue
        if value:
            return str(value)[:400]
    return ""


def _error_shaped(body: dict, fallback_code: str, fallback_msg: str) -> dict:














    answer = dict(body)
    if "error" not in answer:
        answer["error"] = _named_error_text(body) or fallback_msg
    if not answer.get("code"):
        answer["code"] = fallback_code
    return answer


def _unreadable_answer() -> dict:







    return {
        "error": tr(
            "The reply did not come from the service. If this network shows a "
            "sign-in page, open it in your browser first, then try again."
        ),
        "code": "UNREADABLE_RESPONSE",
    }


def _classify_network_error(blocker: QgsBlockingNetworkRequest) -> tuple[str, str]:
    reply = blocker.reply()
    qt_error = reply.error() if reply else _UnknownNetwork
    return _classify_qt_error(
        qt_error, blocker.errorMessage(), _http_status_of(reply),
        service_reachable=server_reached_recently(),
    )

















def _classify_qt_error(
    qt_error,
    error_string: str,
    http_status: int | None,
    service_reachable: bool = False,
) -> tuple[str, str]:

















    qt_error_num = getattr(qt_error, "value", qt_error)



    try:
        from ..core.log_scrub import scrub_sensitive

        detail = scrub_sensitive(error_string or "")
    except Exception:  # noqa: BLE001
        detail = ""
    _log_warning(
        f"Network error: qt_error={qt_error_num}, http_status={http_status}, "
        f"detail={detail[:500]}"
    )




    try:
        from ..core.server_dials import dial_copy
    except Exception:  # noqa: BLE001  # nosec B110
        def dial_copy(_string_id: str, fallback: str) -> str:
            return fallback

    if qt_error == _HostNotFound:
        return "DNS_ERROR", dial_copy(
            "network.server_unreachable",
            tr("Cannot reach the server. Check your internet connection."),
        )
    if qt_error == _ConnRefused:
        return "CONNECTION_REFUSED", dial_copy(
            "network.connection_refused", tr("Server refused the connection."))
    if qt_error == _Timeout or (_OpCanceled is not None and qt_error == _OpCanceled):









        if http_status is None:
            if service_reachable:
                return "SERVICE_WARMING", dial_copy(
                    "network.service_waking_up",
                    tr("The AI service is waking up. Holding your spot..."),
                )




            return "NO_INTERNET", dial_copy(
                "network.no_connection",
                tr("Network error. Check your internet connection."),
            )
        return "TIMEOUT", dial_copy(
            "network.request_timed_out",
            tr("Request timed out. Check your connection or try again."),
        )
    if qt_error == _SslFailed:
        return "SSL_ERROR", dial_copy(
            "network.secure_connection_blocked",
            tr("SSL certificate error. Your network may be blocking secure connections."),
        )
    if qt_error in _PROXY_ERRORS:
        return "PROXY_ERROR", dial_copy(
            "network.proxy_failed",
            tr(
                "Proxy connection failed. "
                "Check QGIS proxy settings (Settings > Options > Network)."
            ),
        )
    if qt_error in (_ContentDenied, _AuthRequired) and http_status == 401:






        return "AUTH_ERROR", dial_copy(
            "network.sign_in_again", tr("Authentication failed. Please sign in again."))





    if http_status is not None and http_status >= 500:
        return "SERVER_ERROR", dial_copy(
            "network.service_unavailable",
            tr(
                "The service is temporarily unavailable (server error). "
                "Your connection is fine - please try again in a few minutes."
            ),
        )





    if http_status is not None:
        return "SERVER_ERROR", dial_copy(
            "network.unexpected_response",
            tr("The server returned an unexpected response. Please try again."),
        )





    if qt_error in _CONNECT_FAILURE_ERRORS:
        return "NO_INTERNET", dial_copy(
            "network.no_connection",
            tr("Network error. Check your internet connection."),
        )




    return "SERVER_ERROR", dial_copy(
        "network.connection_interrupted",
        tr("The connection to the server was interrupted. Please try again."),
    )
