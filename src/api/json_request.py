








from __future__ import annotations

from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from .terralab_client_primitives import _apply_redirect_policy


def _plugin_version() -> str | None:
    try:
        from ..core.request_context import plugin_version

        return plugin_version()
    except Exception:  # noqa: BLE001
        return None


def build_json_request(
    url: str,
    auth: dict | None,
    timeout_ms: int,
    *,
    packed: bool = False,
    extra_headers: dict | None = None,
    redirect_policy=_apply_redirect_policy,
) -> QNetworkRequest:









    request = QNetworkRequest(QUrl(url))
    request.setRawHeader(b"Content-Type", b"application/json")
    if packed:
        request.setRawHeader(b"Content-Encoding", b"gzip")
    if hasattr(request, "setTransferTimeout"):
        request.setTransferTimeout(timeout_ms)
    redirect_policy(request, bool(auth))
    version = _plugin_version()
    if version:
        request.setRawHeader(b"X-Plugin-Version", version.encode("utf-8"))
    for headers in (auth, extra_headers):
        for key, value in (headers or {}).items():
            request.setRawHeader(key.encode("utf-8"), value.encode("utf-8"))
    return request
