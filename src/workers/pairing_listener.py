
















from __future__ import annotations

import hmac
import html
import re
from urllib.parse import parse_qs, urlsplit

from qgis.PyQt.QtCore import QObject, QTimer
from qgis.PyQt.QtNetwork import QHostAddress, QTcpServer

from ..core.i18n import tr
from ..core.logging_utils import log
from ..core.qt_compat import resolve_qt_enum

SIGNED_IN_PATH = "/terralab/signed-in"
_MAX_REQUEST_BYTES = 4096


_IDLE_SOCKET_MS = 10_000
_MAX_OPEN_SOCKETS = 16
_GRANT_RE = re.compile(r"^[A-Za-z0-9_-]{43,128}$")
_SECURITY_HEADERS = (
    "Cache-Control: no-store\r\n"
    "Referrer-Policy: no-referrer\r\n"
    "X-Content-Type-Options: nosniff\r\n"
    "Connection: close\r\n"
)


class PairingReply:


    def __init__(self, listener: PairingListener, socket) -> None:
        self._listener = listener
        self._socket = socket
        self._sent = False

    def redirect(self, location: str) -> None:
        self._send(f"HTTP/1.1 302 Found\r\nLocation: {location}\r\n"
                   f"{_SECURITY_HEADERS}Content-Length: 0\r\n\r\n".encode("ascii"))

    def go_back(self) -> None:
        title = html.escape(tr("Go back to QGIS"))
        body = (
            '<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width">'
            f"<title>{title}</title></head>"
            '<body style="font-family:system-ui,sans-serif;text-align:center;'
            f'padding:64px 16px"><h1 style="font-size:20px;font-weight:600">{title}</h1>'
            "</body></html>"
        ).encode()
        head = ("HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                "Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline'\r\n"
                f"{_SECURITY_HEADERS}Content-Length: {len(body)}\r\n\r\n").encode("ascii")
        self._send(head + body)

    def _send(self, payload: bytes) -> None:
        if self._sent:
            return
        self._sent = True
        self._listener._write_and_close(self._socket, payload)


class PairingListener(QObject):






    def __init__(self, on_grant) -> None:
        super().__init__(None)
        self._on_grant = on_grant
        self._server: QTcpServer | None = None
        self._code = ""
        self._sockets: set = set()
        self._handled: set = set()
        self.port = 0

    def start(self) -> int:


        server = QTcpServer(self)
        local_host = QHostAddress(resolve_qt_enum(QHostAddress, "SpecialAddress", "LocalHost"))
        if not server.listen(local_host, 0) or not 1024 <= int(server.serverPort()) <= 65535:
            log("Pairing: no local port for the browser to come back to; "
                "the sign-in will ask for the code instead")
            server.close()
            server.deleteLater()
            return 0
        server.newConnection.connect(self._on_connection)
        self._server = server
        self.port = int(server.serverPort())
        return self.port

    def set_code(self, code: str) -> None:
        self._code = code or ""

    def close(self) -> None:


        server, self._server = self._server, None
        self._code = ""
        self.port = 0
        if server is not None:
            try:
                server.close()
                server.deleteLater()
            except RuntimeError:  # nosec B110
                pass
        for socket in list(self._sockets):
            if socket not in self._handled:
                self._drop(socket)

    def abort_all(self) -> None:

        self.close()
        for socket in list(self._sockets):
            self._drop(socket)



    def _on_connection(self) -> None:
        server = self._server
        if server is None:
            return
        while server.hasPendingConnections():
            socket = server.nextPendingConnection()
            if socket is None:
                break

            socket.setParent(None)
            if len(self._sockets) >= _MAX_OPEN_SOCKETS:
                socket.abort()
                socket.deleteLater()
                continue
            self._sockets.add(socket)
            socket.readyRead.connect(lambda s=socket: self._on_ready(s))
            socket.disconnected.connect(lambda s=socket: self._forget(s))
            QTimer.singleShot(_IDLE_SOCKET_MS, lambda s=socket: self._drop_if_idle(s))

    def _on_ready(self, socket) -> None:
        if socket in self._handled or socket not in self._sockets:
            return
        try:
            data = bytes(socket.peek(_MAX_REQUEST_BYTES + 1))
        except RuntimeError:
            return
        if b"\n" not in data:
            if len(data) > _MAX_REQUEST_BYTES:
                self._not_found(socket)
            return
        socket.readAll()
        self._handled.add(socket)
        grant = self._grant_of(data.split(b"\n", 1)[0])
        if not grant or self._server is None:
            self._not_found(socket)
            return
        self._on_grant(grant, PairingReply(self, socket))

    def _grant_of(self, request_line: bytes) -> str:

        parts = request_line.strip().decode("latin-1", "replace").split(" ")
        if len(parts) != 3 or parts[0] != "GET":
            return ""

        if not parts[1].startswith("/") or parts[1].startswith("//"):
            return ""
        url = urlsplit(parts[1])
        if url.path != SIGNED_IN_PATH:
            return ""
        try:
            query = parse_qs(url.query, keep_blank_values=True, max_num_fields=4)
        except ValueError:
            return ""
        codes, grants = query.get("code", []), query.get("grant", [])
        if set(query) != {"code", "grant"} or len(codes) != 1 or len(grants) != 1:
            return ""
        if not self._code or not hmac.compare_digest(
                codes[0].encode("utf-8"), self._code.encode("utf-8")):
            return ""
        return grants[0] if _GRANT_RE.match(grants[0]) else ""

    def _not_found(self, socket) -> None:
        body = b"Not found"
        head = ("HTTP/1.1 404 Not Found\r\nContent-Type: text/plain; charset=utf-8\r\n"
                f"{_SECURITY_HEADERS}Content-Length: {len(body)}\r\n\r\n").encode("ascii")
        self._write_and_close(socket, head + body)

    def _write_and_close(self, socket, payload: bytes) -> None:
        try:
            socket.write(payload)
            socket.flush()
            socket.disconnectFromHost()
        except RuntimeError:  # nosec B110
            self._forget(socket)

    def _drop_if_idle(self, socket) -> None:
        if socket in self._sockets and socket not in self._handled:
            self._drop(socket)

    def _drop(self, socket) -> None:
        try:
            socket.abort()
        except RuntimeError:  # nosec B110
            pass
        self._forget(socket)

    def _forget(self, socket) -> None:
        if socket not in self._sockets:
            return
        self._sockets.discard(socket)
        self._handled.discard(socket)
        try:
            socket.deleteLater()
        except RuntimeError:  # nosec B110
            pass
