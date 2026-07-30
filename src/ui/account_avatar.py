








from __future__ import annotations

import hashlib
import os

from qgis.PyQt.QtCore import QObject, Qt, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QImage, QPainter, QPainterPath, QPixmap
from qgis.PyQt.QtNetwork import QNetworkReply, QNetworkRequest

from ..core.cache_paths import PLUGIN_CACHE_DIR
from ..core.qt_compat import (
    HttpStatusCodeAttribute,
    NoLessSafeRedirectPolicy,
    RedirectPolicyAttribute,
    resolve_qt_enum,
)



MAX_AVATAR_BYTES = 400 * 1024



AVATAR_OVERSAMPLE = 2

AVATAR_CACHE_DIR = os.path.join(PLUGIN_CACHE_DIR, "avatars")

_NETWORK_NO_ERROR = resolve_qt_enum(QNetworkReply, "NetworkError", "NoError")




def is_avatar_url_usable(url: object) -> bool:
    return isinstance(url, str) and url.lower().startswith("https://")




def avatar_cache_path(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    return os.path.join(AVATAR_CACHE_DIR, f"{digest}.img")


def circular_avatar_pixmap(image: QImage, diameter: int) -> QPixmap | None:





    if image.isNull() or diameter <= 0:
        return None
    side = int(diameter) * AVATAR_OVERSAMPLE
    scaled = image.scaled(
        side, side,
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )
    square = scaled.copy(
        max(0, (scaled.width() - side) // 2),
        max(0, (scaled.height() - side) // 2),
        side, side,
    )
    pixmap = QPixmap(side, side)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        path = QPainterPath()
        path.addEllipse(0, 0, side, side)
        painter.setClipPath(path)
        painter.drawImage(0, 0, square)
    finally:
        painter.end()
    pixmap.setDevicePixelRatio(float(AVATAR_OVERSAMPLE))
    return pixmap


def cached_avatar_pixmap(url: str, diameter: int) -> QPixmap | None:

    path = avatar_cache_path(url)
    try:
        if os.path.getsize(path) > MAX_AVATAR_BYTES:
            return None
        with open(path, "rb") as handle:
            data = handle.read(MAX_AVATAR_BYTES)
    except OSError:
        return None
    image = QImage()
    if not image.loadFromData(data):
        return None
    return circular_avatar_pixmap(image, diameter)


def store_avatar_bytes(url: str, data: bytes) -> None:

    path = avatar_cache_path(url)
    part = path + ".part"
    try:
        os.makedirs(AVATAR_CACHE_DIR, exist_ok=True)
        with open(part, "wb") as handle:
            handle.write(data)
        os.replace(part, path)
    except OSError:
        try:
            os.remove(part)
        except OSError:
            pass


class AccountAvatarLoader(QObject):






    loaded = pyqtSignal(QPixmap)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._reply = None
        self._url = ""
        self._diameter = 0

    def fetch(self, url: str, diameter: int) -> None:

        from qgis.core import QgsNetworkAccessManager

        if self._reply is not None or not is_avatar_url_usable(url):
            return
        self._url = url
        self._diameter = int(diameter)
        request = QNetworkRequest(QUrl(url))
        request.setAttribute(RedirectPolicyAttribute, NoLessSafeRedirectPolicy)
        try:
            reply = QgsNetworkAccessManager.instance().get(request)
        except Exception:  # noqa: BLE001
            return
        try:
            reply.setReadBufferSize(MAX_AVATAR_BYTES + 1)
        except (AttributeError, RuntimeError):

            pass
        self._reply = reply
        reply.downloadProgress.connect(self._on_download_progress)
        reply.finished.connect(self._on_reply_finished)




        reply.finished.connect(reply.deleteLater)

    def abort(self) -> None:

        reply, self._reply = self._reply, None
        if reply is None:
            return
        for signal in ("finished", "downloadProgress"):
            try:
                getattr(reply, signal).disconnect()
            except (AttributeError, RuntimeError, TypeError):

                pass
        try:
            reply.abort()
            reply.deleteLater()
        except RuntimeError:
            pass

    def _on_download_progress(self, received: int, total: int) -> None:

        if self._reply is not None and (received > MAX_AVATAR_BYTES or total > MAX_AVATAR_BYTES):
            self.abort()

    def _on_reply_finished(self) -> None:
        reply, self._reply = self._reply, None
        if reply is None:
            return
        data = b""
        try:
            status = reply.attribute(HttpStatusCodeAttribute)
            served = int(status) if status is not None else 200
            if reply.error() == _NETWORK_NO_ERROR and served == 200:
                data = bytes(reply.readAll())
        except (RuntimeError, TypeError, ValueError):
            data = b""
        try:
            reply.deleteLater()
        except RuntimeError:
            pass
        if not data or len(data) > MAX_AVATAR_BYTES:
            return
        image = QImage()
        if not image.loadFromData(data):
            return
        pixmap = circular_avatar_pixmap(image, self._diameter)
        if pixmap is None:
            return
        store_avatar_bytes(self._url, data)
        self.loaded.emit(pixmap)
