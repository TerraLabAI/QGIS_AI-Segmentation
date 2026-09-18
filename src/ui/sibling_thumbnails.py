











from __future__ import annotations

import hashlib
import os
from urllib.parse import urlsplit

from qgis.PyQt.QtCore import (
    QBuffer,
    QByteArray,
    QIODevice,
    QObject,
    QRect,
    QRectF,
    QSize,
    QStandardPaths,
    Qt,
    QUrl,
    pyqtSignal,
)
from qgis.PyQt.QtGui import QColor, QImage, QImageReader, QPainter, QPainterPath, QPixmap
from qgis.PyQt.QtNetwork import QNetworkReply, QNetworkRequest
from qgis.PyQt.QtWidgets import QLabel, QSizePolicy

from .dock.styles import BRAND_GREEN, RADIUS_CARD

_CARD_RADIUS = RADIUS_CARD

_CARD_MIN_W = 260



_MAX_THUMBNAIL_BYTES = 900 * 1024



_SHOT_RATIO = 630 / 1200

_LOGO_DRAW_PX = 48


def _enum(owner, scope, name):






    holder = getattr(owner, scope, None)
    if holder is not None and hasattr(holder, name):
        return getattr(holder, name)



    return getattr(owner, name)


_ALIGN_CENTER = _enum(Qt, "AlignmentFlag", "AlignCenter")
_ALIGN_LEFT = _enum(Qt, "AlignmentFlag", "AlignLeft")
_ALIGN_TOP = _enum(Qt, "AlignmentFlag", "AlignTop")
_ALIGN_VCENTER = _enum(Qt, "AlignmentFlag", "AlignVCenter")
_POINTING_HAND = _enum(Qt, "CursorShape", "PointingHandCursor")
_NO_PEN = _enum(Qt, "PenStyle", "NoPen")
_EXPAND_RATIO = _enum(Qt, "AspectRatioMode", "KeepAspectRatioByExpanding")
_SMOOTH = _enum(Qt, "TransformationMode", "SmoothTransformation")
_SIZE_EXPANDING = _enum(QSizePolicy, "Policy", "Expanding")
_SIZE_PREFERRED = _enum(QSizePolicy, "Policy", "Preferred")
_READ_ONLY = _enum(QIODevice, "OpenModeFlag", "ReadOnly")
_CACHE_LOCATION = _enum(QStandardPaths, "StandardLocation", "CacheLocation")
_NETWORK_NO_ERROR = _enum(QNetworkReply, "NetworkError", "NoError")
_HTTP_STATUS = _enum(QNetworkRequest, "Attribute", "HttpStatusCodeAttribute")
_REDIRECT_POLICY = _enum(QNetworkRequest, "Attribute", "RedirectPolicyAttribute")
_NO_LESS_SAFE = _enum(QNetworkRequest, "RedirectPolicy", "NoLessSafeRedirectPolicy")


def _thumbnail_cache_dir() -> str:


    root = QStandardPaths.writableLocation(_CACHE_LOCATION) or os.path.expanduser("~/.cache")
    return os.path.join(root, "terralab", "plugin-thumbnails")


def _cache_path(url: str) -> str:

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    return os.path.join(_thumbnail_cache_dir(), f"{digest}.img")


def _url_is_usable(url) -> bool:

    if not isinstance(url, str) or not url.lower().startswith("https://"):
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return bool(parsed.hostname) and not parsed.username and not parsed.password


def _read_bounded_image(data: bytes):

    try:
        buffer = QBuffer()
        buffer.setData(QByteArray(data))
        buffer.open(_READ_ONLY)
        reader = QImageReader(buffer)
        size = reader.size()
        if size.isValid() and size.width() * size.height() > 16 * 1024 * 1024:
            return None
        image = reader.read()
        return None if image.isNull() else image
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _cached_image(url: str):
    try:
        path = _cache_path(url)
        if os.path.getsize(path) > _MAX_THUMBNAIL_BYTES:
            return None
        with open(path, "rb") as handle:
            data = handle.read(_MAX_THUMBNAIL_BYTES)
    except OSError:
        return None
    return _read_bounded_image(data)


def _store_image(url: str, data: bytes) -> None:
    path = _cache_path(url)
    part = path + ".part"
    try:
        os.makedirs(_thumbnail_cache_dir(), exist_ok=True)
        with open(part, "wb") as handle:
            handle.write(data)
        os.replace(part, path)
    except OSError:
        try:
            os.remove(part)
        except OSError:
            pass  # nosec B110


class _ThumbnailLoader(QObject):






    loaded = pyqtSignal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._reply = None
        self._url = ""

    def fetch(self, url: str) -> None:
        from qgis.core import QgsNetworkAccessManager

        if self._reply is not None or not _url_is_usable(url):
            return
        self._url = url
        request = QNetworkRequest(QUrl(url))
        try:
            request.setAttribute(_REDIRECT_POLICY, _NO_LESS_SAFE)
            reply = QgsNetworkAccessManager.instance().get(request)
        except Exception:  # noqa: BLE001
            return  # nosec B110
        try:
            reply.setReadBufferSize(_MAX_THUMBNAIL_BYTES + 1)
        except (AttributeError, RuntimeError):
            pass  # nosec B110
        self._reply = reply
        reply.downloadProgress.connect(self._on_progress)
        reply.finished.connect(self._on_finished)



        reply.finished.connect(reply.deleteLater)

    def abort(self) -> None:
        reply, self._reply = self._reply, None
        if reply is None:
            return
        for signal in ("finished", "downloadProgress"):
            try:
                getattr(reply, signal).disconnect()
            except (AttributeError, RuntimeError, TypeError):
                pass  # nosec B110
        try:
            reply.abort()
            reply.deleteLater()
        except RuntimeError:
            pass  # nosec B110

    def _on_progress(self, received: int, total: int) -> None:
        if self._reply is not None and (received > _MAX_THUMBNAIL_BYTES
                                        or total > _MAX_THUMBNAIL_BYTES):
            self.abort()

    def _on_finished(self) -> None:
        reply, self._reply = self._reply, None
        if reply is None:
            return
        data = b""
        try:
            status = reply.attribute(_HTTP_STATUS)
            served = int(status) if status is not None else 200
            if reply.error() == _NETWORK_NO_ERROR and served == 200:
                data = bytes(reply.readAll())
        except (RuntimeError, TypeError, ValueError):
            data = b""
        try:
            reply.deleteLater()
        except RuntimeError:
            pass  # nosec B110
        if not data or len(data) > _MAX_THUMBNAIL_BYTES:
            return
        image = _read_bounded_image(data)
        if image is None:
            return
        _store_image(self._url, data)
        self.loaded.emit(image)


class SiblingShot(QLabel):








    def __init__(self, logo_path: str, parent=None, tint: str = BRAND_GREEN):
        super().__init__(parent)
        self._image = None
        self._logo = None
        self._tint = QColor(tint)
        if logo_path and os.path.isfile(logo_path):
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():


                self._logo = pixmap.scaled(_LOGO_DRAW_PX * 2, _LOGO_DRAW_PX * 2,
                                           _enum(Qt, "AspectRatioMode", "KeepAspectRatio"),
                                           _SMOOTH)


        policy = QSizePolicy(_SIZE_EXPANDING, _SIZE_PREFERRED)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setAlignment(_ALIGN_CENTER)

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return int(round(max(1, width) * _SHOT_RATIO))

    def sizeHint(self):  # noqa: N802


        return QSize(_CARD_MIN_W, self.heightForWidth(_CARD_MIN_W))

    def minimumSizeHint(self):  # noqa: N802
        return QSize(1, 1)

    def set_image(self, image: QImage) -> None:
        self._image = image if image is not None and not image.isNull() else None
        self.update()

    def _clip_path(self) -> QPainterPath:

        rect = QRectF(self.rect())
        radius = float(max(0, _CARD_RADIUS - 1))
        path = QPainterPath()
        path.moveTo(rect.left(), rect.bottom() + 1)
        path.lineTo(rect.left(), rect.top() + radius)
        path.quadTo(rect.left(), rect.top(), rect.left() + radius, rect.top())
        path.lineTo(rect.right() + 1 - radius, rect.top())
        path.quadTo(rect.right() + 1, rect.top(), rect.right() + 1, rect.top() + radius)
        path.lineTo(rect.right() + 1, rect.bottom() + 1)
        path.closeSubpath()
        return path

    def paintEvent(self, _event):  # noqa: N802
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            rect = self.rect()
            painter.setClipPath(self._clip_path())
            if self._image is not None:
                ratio = float(self.devicePixelRatioF())
                target = QSize(max(1, int(rect.width() * ratio)),
                               max(1, int(rect.height() * ratio)))
                scaled = self._image.scaled(target, _EXPAND_RATIO, _SMOOTH)
                pixmap = QPixmap.fromImage(scaled)
                pixmap.setDevicePixelRatio(ratio)
                painter.drawPixmap(
                    rect.x() - max(0, (int(scaled.width() / ratio) - rect.width())) // 2,
                    rect.y() - max(0, (int(scaled.height() / ratio) - rect.height())) // 2,
                    pixmap)
                return
            tint = QColor(self._tint)
            tint.setAlphaF(0.16)
            painter.fillRect(rect, tint)
            if self._logo is not None:
                w = self._logo.width() // 2
                h = self._logo.height() // 2
                painter.drawPixmap(QRect(int((rect.width() - w) / 2),
                                         int((rect.height() - h) / 2), w, h),
                                   self._logo)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        finally:
            painter.end()


def load_shot_image(shot: SiblingShot, owner: QObject, url: str):

    if not _url_is_usable(url):
        return None
    cached = _cached_image(url)
    if cached is not None:
        shot.set_image(cached)
        return None
    loader = _ThumbnailLoader(owner)
    loader.loaded.connect(shot.set_image)
    loader.fetch(url)
    return loader


__all__ = ["SiblingShot", "load_shot_image"]
