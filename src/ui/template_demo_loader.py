"""Async loader for the segment-library before/after demo images.

Caches PNG/JPEG bytes on disk under the platform's per-user cache dir
(via ``QStandardPaths.CacheLocation``) so the second open of the library
is instant. Fetches go through QGIS's ``QgsNetworkAccessManager`` (so they
inherit its SSL/proxy/auth config) and emit a signal per finished download
so cards can swap in the real pixmap when ready. 404s (presets not yet seeded
server-side) are remembered so we don't refetch them.

Ported from the AI Edit plugin; only the cache namespace and logging shim
differ. Production-safe logging: it only ever logs preset ids + local cache
paths, never URLs/keys/model names.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from qgis.core import Qgis, QgsNetworkAccessManager
from qgis.PyQt.QtCore import QByteArray, QObject, QStandardPaths, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtNetwork import QNetworkReply, QNetworkRequest

from ..core.logging_utils import log
from ..core.qt_compat import (
    NoLessSafeRedirectPolicy,
    RedirectPolicyAttribute,
    safe_single_shot,
)


def log_debug(message: str) -> None:
    log(message, Qgis.MessageLevel.Info)


def log_warning(message: str) -> None:
    log(message, Qgis.MessageLevel.Warning)


# Demos the server returned 404 for (not yet seeded). Module-level so the
# knowledge survives reopening the library dialog within a QGIS session and we
# don't re-issue doomed requests (each burns a concurrency slot + 15s timeout).
_KNOWN_MISSING: set[tuple[str, str]] = set()

# Bump this when the server-side demo set is re-seeded in a way that must
# invalidate every client's on-disk cache at once (the 7-day TTL is too slow).
_CACHE_DIR_NAME = "ai-segmentation-template-demos-v1"


def _cache_root() -> Path:
    """Per-platform cache dir for demo image bytes.

    Returns ``CacheLocation/<_CACHE_DIR_NAME>`` which is:
        - Windows: ``%LOCALAPPDATA%/<org>/<app>/cache/<_CACHE_DIR_NAME>``
        - macOS:   ``~/Library/Caches/<org>/<app>/<_CACHE_DIR_NAME>``
        - Linux:   ``~/.cache/<app>/<_CACHE_DIR_NAME>``
    Falls back to the historical Linux-style path when QStandardPaths
    returns nothing (rare, mostly headless test envs).
    """
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
    if base:
        return Path(base) / _CACHE_DIR_NAME
    return Path.home() / ".cache" / _CACHE_DIR_NAME


def _cache_path(template_id: str, which: str) -> Path:
    safe_id = "".join(c for c in template_id if c.isalnum() or c in "-_")
    return _cache_root() / safe_id / f"{which}.jpg"


# Demos rarely change, but a curated demo can be re-seeded server-side. Without
# expiry the on-disk cache would pin the old image forever; a 7-day TTL lets
# updates propagate while still keeping repeat opens instant.
_CACHE_TTL_SECONDS = 7 * 24 * 3600


def read_cached_pixmap(template_id: str, which: str) -> QPixmap | None:
    """Return a QPixmap from the on-disk cache, or None if absent or stale."""
    path = _cache_path(template_id, which)
    if not path.is_file():
        return None
    try:
        if (time.time() - path.stat().st_mtime) > _CACHE_TTL_SECONDS:
            return None
        pm = QPixmap(str(path))
        if pm.isNull() or pm.width() < 2:
            return None
        return pm
    except Exception as err:  # noqa: BLE001
        log_warning(f"Failed to read cached demo {path}: {err}")
        return None


class TemplateDemoLoader(QObject):
    """Async fetcher for preset demo images. One instance per dialog.

    Signals:
        loaded(template_id, which, QPixmap) - fires when a download (or cache
            hit) yields a usable pixmap. The card matching template_id + which
            installs it into the slider.
        failed(template_id, which) - fires once we've concluded the demo will
            never be available (404 server-side or persistent network error).
    """

    loaded = pyqtSignal(str, str, QPixmap)
    failed = pyqtSignal(str, str)

    # Cap simultaneous fetches so opening the library (or a popup with bigger
    # preview images) doesn't fire dozens of requests at once and choke a slow
    # link. Excess requests queue and start as in-flight ones finish. Kept low
    # so a thin pipe isn't split too many ways (each split is likelier to hit
    # the per-request transfer timeout).
    _MAX_CONCURRENT = 3

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._queue: list[tuple[str, str, str, "dict | None"]] = []
        self._in_flight = 0
        _cache_root().mkdir(parents=True, exist_ok=True)

    def request(self, template_id: str, which: str, url: str,
                headers: dict | None = None) -> None:
        """Try cache first; if miss, queue an async network fetch.

        ``which`` is normally "before"/"after" for card sliders; the detail
        popup also passes "before_preview", "after_preview" for the bigger
        image. Any non-empty token works as the on-disk cache filename.

        ``headers`` (optional) are raw request headers for authorized fetches
        (the run-history thumbnails need the account's Authorization header).
        The disk cache key stays (template_id, which) - history callers pass
        the tile's request_id as template_id, so the key is per-artifact."""
        if not template_id or not which or not url:
            return
        key = (template_id, which)
        if key in _KNOWN_MISSING:
            self.failed.emit(template_id, which)
            return
        # Defer the disk read + decode to the next event-loop turn so a burst of
        # cached cards built in one synchronous loop doesn't block the dialog's
        # first paint. Parented to self, so it can't fire after the loader dies.
        safe_single_shot(
            0, self,
            lambda t=template_id, w=which, u=url, h=headers:
                self._load_cached_or_fetch(t, w, u, h),
        )

    def _load_cached_or_fetch(self, template_id: str, which: str, url: str,
                              headers: dict | None = None) -> None:
        pm = read_cached_pixmap(template_id, which)
        if pm is not None:
            self.loaded.emit(template_id, which, pm)
            return
        self._queue.append((template_id, which, url, headers))
        self._pump()

    def _pump(self) -> None:
        """Start queued fetches up to the concurrency cap."""
        while self._in_flight < self._MAX_CONCURRENT and self._queue:
            template_id, which, url, headers = self._queue.pop(0)
            self._in_flight += 1
            self._start(template_id, which, url, headers)

    def _start(self, template_id: str, which: str, url: str,
               headers: dict | None = None) -> None:
        req = QNetworkRequest(QUrl(url))
        # Follow redirects. Resolved via qt_compat (scoped-then-flat) because
        # PyQt5 on some QGIS 3 builds exposes these enums flat, not scoped.
        req.setAttribute(RedirectPolicyAttribute, NoLessSafeRedirectPolicy)
        req.setRawHeader(b"Accept", b"image/jpeg, image/png, image/webp, image/*")
        if headers:
            for hk, hv in headers.items():
                try:
                    req.setRawHeader(str(hk).encode("utf-8"), str(hv).encode("utf-8"))
                except (UnicodeError, TypeError):
                    continue
        req.setTransferTimeout(15_000)
        # Route through QGIS's network manager so the fetch inherits its SSL CA
        # bundle, proxy, and auth config. A bare QNetworkAccessManager fails
        # silently on some CDN hosts. Parent the reply to this loader so it dies
        # with the dialog (no callback on a dead object).
        reply = QgsNetworkAccessManager.instance().get(req)
        reply.setParent(self)
        reply.finished.connect(
            lambda r=reply, t=template_id, w=which: self._on_finished(r, t, w)
        )

    def _on_finished(self, reply: QNetworkReply, template_id: str, which: str) -> None:
        try:
            err_code = reply.error()
            no_err = QNetworkReply.NetworkError.NoError
            http_status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
            try:
                http_int = int(http_status) if http_status is not None else 0
            except (TypeError, ValueError):
                http_int = 0
            if err_code != no_err or http_int >= 400:
                if http_int == 404:
                    _KNOWN_MISSING.add((template_id, which))
                else:
                    log_debug(
                        f"Demo fetch failed for {template_id}/{which}: "
                        f"err={err_code} http={http_int}"
                    )
                self.failed.emit(template_id, which)
                return
            data: QByteArray = reply.readAll()
            buf = bytes(data)
            if len(buf) < 256:
                self.failed.emit(template_id, which)
                return
            pm = QPixmap()
            if not pm.loadFromData(buf):
                log_debug(f"Demo bytes did not decode for {template_id}/{which}")
                self.failed.emit(template_id, which)
                return
            self._write_cache(template_id, which, buf)
            self.loaded.emit(template_id, which, pm)
        finally:
            reply.deleteLater()
            self._in_flight = max(0, self._in_flight - 1)
            self._pump()

    @staticmethod
    def _write_cache(template_id: str, which: str, buf: bytes) -> None:
        path = _cache_path(template_id, which)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jpg.tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(buf)
            os.replace(tmp, path)
        except OSError as err:
            log_warning(f"Failed to write demo cache {path}: {err}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass  # nosec B110
