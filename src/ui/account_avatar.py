"""The account's own profile picture, drawn as the round badge in Settings.

The account endpoint hands the dialog an https picture URL. Reading it must not
stall the window, so the request goes through QgsNetworkAccessManager and
reports back on its finished signal. The bytes are then kept under the plugin's
cache folder, keyed by a hash of the URL, so every later dialog open draws the
picture with no request at all. Anything that fails here is silent: the caller
keeps the letter badge it already drew.
"""
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

# A profile picture is a small square. Past this the reply is aborted rather
# than buffered, so a wrong URL can never pull a large file into memory.
MAX_AVATAR_BYTES = 400 * 1024

# Rendered at twice the badge size and marked as such, so the circle stays
# sharp on a high-density screen without asking the widget for its ratio.
AVATAR_OVERSAMPLE = 2

AVATAR_CACHE_DIR = os.path.join(PLUGIN_CACHE_DIR, "avatars")

_NETWORK_NO_ERROR = resolve_qt_enum(QNetworkReply, "NetworkError", "NoError")


# https only: the picture is not worth a plaintext request, and a file: or
# data: URL coming back from a server would read something local instead.
def is_avatar_url_usable(url: object) -> bool:
    return isinstance(url, str) and url.lower().startswith("https://")


# The URL is the identity of the picture: a new photo comes with a new URL, so
# a hash of it is a key that never serves a stale face.
def avatar_cache_path(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    return os.path.join(AVATAR_CACHE_DIR, f"{digest}.img")


def circular_avatar_pixmap(image: QImage, diameter: int) -> QPixmap | None:
    """Crop to a centred square, clip to a circle, return it at ``diameter``.

    Same diameter and same edge treatment as the letter badge it replaces (a
    filled circle, no border), so the card keeps its layout to the pixel.
    """
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
    """The picture already on disk for this URL, or None to go and fetch it."""
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
    """Keep the picture beside the plugin's other user data, atomically."""
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
            pass  # the picture is a nicety, never a failure path


class AccountAvatarLoader(QObject):
    """One picture read, off the UI thread, reporting back on ``loaded``.

    Parent it to the dialog: a window closed before the reply lands takes the
    loader and its connections with it, so nothing reaches a deleted widget.
    """

    loaded = pyqtSignal(QPixmap)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._reply = None
        self._url = ""
        self._diameter = 0

    def fetch(self, url: str, diameter: int) -> None:
        """Start the one request. A second call while one runs does nothing."""
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
            return  # no picture: the letter badge stands
        try:
            reply.setReadBufferSize(MAX_AVATAR_BYTES + 1)
        except (AttributeError, RuntimeError):
            pass
        self._reply = reply
        reply.downloadProgress.connect(self._on_download_progress)
        reply.finished.connect(self._on_reply_finished)
        # Second, after ours so the bytes are read first: the reply belongs to
        # the network manager, and this slot has the reply itself as receiver,
        # so it still frees it when the dialog (and this loader) went away mid
        # request and our own handler never runs.
        reply.finished.connect(reply.deleteLater)

    def abort(self) -> None:
        """Drop an in-flight read so its result can never land late."""
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
        # Stop anything that is plainly not a profile picture before it is read.
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
