from __future__ import annotations

import json

from qgis.core import Qgis, QgsBlockingNetworkRequest, QgsMessageLog
from qgis.PyQt.QtCore import QByteArray, QUrl
from qgis.PyQt.QtNetwork import QNetworkReply, QNetworkRequest

_TIMEOUT_API = 30_000

# Qt6 (QGIS 4) uses scoped enums; Qt5 uses flat. Resolve at import time.
_NE = getattr(QNetworkReply, "NetworkError", QNetworkReply)
_HostNotFound = getattr(_NE, "HostNotFoundError", getattr(QNetworkReply, "HostNotFoundError", None))
_ConnRefused = getattr(_NE, "ConnectionRefusedError", getattr(QNetworkReply, "ConnectionRefusedError", None))
_Timeout = getattr(_NE, "TimeoutError", getattr(QNetworkReply, "TimeoutError", None))
_SslFailed = getattr(_NE, "SslHandshakeFailedError", getattr(QNetworkReply, "SslHandshakeFailedError", None))
_ContentDenied = getattr(_NE, "ContentAccessDenied", getattr(QNetworkReply, "ContentAccessDenied", None))
_AuthRequired = getattr(_NE, "AuthenticationRequiredError", getattr(QNetworkReply, "AuthenticationRequiredError", None))
_UnknownNetwork = getattr(_NE, "UnknownNetworkError", getattr(QNetworkReply, "UnknownNetworkError", None))

_PROXY_ERRORS = set(filter(None, [
    getattr(_NE, "ProxyConnectionRefusedError", getattr(QNetworkReply, "ProxyConnectionRefusedError", None)),
    getattr(_NE, "ProxyConnectionClosedError", getattr(QNetworkReply, "ProxyConnectionClosedError", None)),
    getattr(_NE, "ProxyNotFoundError", getattr(QNetworkReply, "ProxyNotFoundError", None)),
    getattr(_NE, "ProxyTimeoutError", getattr(QNetworkReply, "ProxyTimeoutError", None)),
    getattr(_NE, "ProxyAuthenticationRequiredError", getattr(QNetworkReply, "ProxyAuthenticationRequiredError", None)),
    getattr(_NE, "UnknownProxyError", getattr(QNetworkReply, "UnknownProxyError", None)),
]))

# QNetworkRequest.Attribute.HttpStatusCodeAttribute (Qt6) vs QNetworkRequest.HttpStatusCodeAttribute (Qt5)
_Attr = getattr(QNetworkRequest, "Attribute", QNetworkRequest)
_HTTP_STATUS_ATTR = getattr(_Attr, "HttpStatusCodeAttribute", getattr(QNetworkRequest, "HttpStatusCodeAttribute", None))


def _log_warning(msg: str):
    QgsMessageLog.logMessage(msg, "AI Segmentation", level=Qgis.MessageLevel.Warning)


def _classify_network_error(blocker: QgsBlockingNetworkRequest) -> tuple[str, str]:
    reply = blocker.reply()
    qt_error = reply.error() if reply else _UnknownNetwork
    error_string = blocker.errorMessage()
    http_status = None
    if reply and _HTTP_STATUS_ATTR is not None:
        attr = reply.attribute(_HTTP_STATUS_ATTR)
        if attr is not None:
            http_status = int(attr)
    _log_warning(
        f"Network error: qt_error={int(qt_error)}, http_status={http_status}, "
        f"detail={error_string[:500]}"
    )

    if qt_error == _HostNotFound:
        return "DNS_ERROR", "Cannot reach the server. Check your internet connection."
    if qt_error == _ConnRefused:
        return "CONNECTION_REFUSED", "Server refused the connection."
    if qt_error == _Timeout:
        return "TIMEOUT", "Request timed out. Check your connection or try again."
    if qt_error == _SslFailed:
        return "SSL_ERROR", "SSL certificate error. Your network may be blocking secure connections."
    if qt_error in _PROXY_ERRORS:
        return "PROXY_ERROR", (
            "Proxy connection failed. "
            "Check QGIS proxy settings (Settings > Options > Network)."
        )
    if qt_error in (_ContentDenied, _AuthRequired):
        return "AUTH_ERROR", "Authentication failed. Please sign in again."
    return "NO_INTERNET", "Network error. Check your internet connection."


class TerraLabClient:
    """HTTP client for the TerraLab backend — uses QgsBlockingNetworkRequest so
    requests pick up QGIS proxy / SSL / Network Logger settings."""

    def __init__(self, base_url: str | None = None):
        if base_url is None:
            base_url = self._read_base_url()
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def _read_base_url() -> str:
        import os
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        env_path = os.path.join(plugin_dir, ".env.local")
        if os.path.isfile(env_path):
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TERRALAB_BASE_URL="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        return "https://terra-lab.ai"

    def get_usage(self, auth: dict) -> dict:
        return self._request("GET", "/api/plugin/usage", auth=auth)

    def get_account(self, auth: dict) -> dict:
        return self._request("GET", "/api/plugin/account", auth=auth)

    def get_config(self, product: str) -> dict:
        return self._request("GET", f"/api/plugin/config?product={product}")

    def _request(
        self,
        method: str,
        path: str,
        auth: dict | None = None,
        body: bytes | None = None,
        timeout_ms: int = _TIMEOUT_API,
    ) -> dict:
        url = f"{self.base_url}{path}"
        req = QNetworkRequest(QUrl(url))
        req.setRawHeader(b"Content-Type", b"application/json")
        if hasattr(req, "setTransferTimeout"):
            req.setTransferTimeout(timeout_ms)
        if auth:
            for key, value in auth.items():
                req.setRawHeader(key.encode("utf-8"), value.encode("utf-8"))

        blocker = QgsBlockingNetworkRequest()
        if method == "GET":
            err = blocker.get(req, forceRefresh=True)
        elif method == "POST":
            payload = QByteArray(body) if body else QByteArray()
            err = blocker.post(req, payload)
        else:
            return {"error": f"Unsupported method: {method}", "code": "CLIENT_ERROR"}

        if err != QgsBlockingNetworkRequest.NoError:
            reply = blocker.reply()
            if reply:
                http_attr = reply.attribute(_HTTP_STATUS_ATTR)
                if http_attr and int(http_attr) >= 400:
                    raw = bytes(reply.content()).decode("utf-8")
                    if raw:
                        try:
                            return json.loads(raw)
                        except Exception:
                            pass  # nosec B110
            code, msg = _classify_network_error(blocker)
            return {"error": msg, "code": code}

        reply = blocker.reply()
        http_status = reply.attribute(_HTTP_STATUS_ATTR)
        raw_body = bytes(reply.content()).decode("utf-8")

        if http_status and int(http_status) >= 400:
            _log_warning(f"HTTP {http_status}: {raw_body[:500]}")
            try:
                error_body = json.loads(raw_body)
                if "error" in error_body:
                    return error_body
                return {
                    "error": error_body.get("detail", raw_body[:200]),
                    "code": "SERVER_ERROR",
                }
            except Exception:
                return {"error": f"Server error (HTTP {http_status})", "code": "SERVER_ERROR"}

        if not raw_body:
            return {}
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError:
            _log_warning(f"Invalid JSON response: {raw_body[:500]}")
            return {"error": "Invalid server response", "code": "SERVER_ERROR"}
