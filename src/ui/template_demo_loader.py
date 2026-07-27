"""Async loader for the segment-library before/after images.

Caches PNG/JPEG bytes on disk under the platform's per-user cache dir (the
cache location reported by ``QStandardPaths``) so the second open of the
library is instant. Fetches go through QGIS's ``QgsNetworkAccessManager`` (so
they inherit its SSL/proxy/auth config) and emit a signal per finished download
so cards can swap in the real pixmap when ready. 404s (presets not yet seeded
server-side) are remembered so we don't refetch them.

Two kinds of image share this cache, and they age differently:

- **Template demos**, keyed by preset id. The server re-seeds a curated demo
  between plugin releases, so a cache hit is painted at once and then checked
  in the background with a conditional request (``image_cache_validators``):
  a 304 keeps the file, a 200 swaps the card in place. A server that sends no
  validator falls back to expiring the entry after ``_CACHE_TTL_SECONDS``.
- **Archived run artifacts**, keyed by the tile's request id. Written once at
  archive time and never changed, so callers pass ``immutable=True``: no
  expiry, and never a revalidation request.

The cache is bounded: a loader schedules one deferred, once-per-session sweep
that evicts least-recently-used files back under the budget
(``image_cache_budget``). Ported from the AI Edit plugin. Production-safe
logging: only preset ids and local cache paths, never URLs/keys/model names.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import NamedTuple

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
from .image_cache_budget import sweep_image_cache_once, touch_for_lru
from .image_cache_validators import (
    conditional_headers,
    mark_validator_checked,
    read_validator,
    should_revalidate,
    validator_from_reply,
    write_validator,
)


def log_debug(message: str) -> None:
    log(message, Qgis.MessageLevel.Info)


def log_warning(message: str) -> None:
    log(message, Qgis.MessageLevel.Warning)


# Images the server returned 404 for (not yet seeded). Module-level so the
# knowledge survives reopening the library within a QGIS session and we don't
# re-issue doomed requests (each burns a concurrency slot + 15s timeout).
_KNOWN_MISSING: set[tuple[str, str, str]] = set()

# Bump this when the server-side demo set is re-seeded in a way that must
# invalidate every client's on-disk cache at once (the 7-day TTL is too slow).
_CACHE_DIR_NAME = "ai-segmentation-template-demos-v1"


def _cache_root() -> Path:
    """Per-platform cache dir for image bytes, ``<cache location>/<dir name>``:

        - Windows: ``%LOCALAPPDATA%/<org>/<app>/cache/<_CACHE_DIR_NAME>``
        - macOS:   ``~/Library/Caches/<org>/<app>/<_CACHE_DIR_NAME>``
        - Linux:   ``~/.cache/<app>/<_CACHE_DIR_NAME>``, also the fallback
          when QStandardPaths returns nothing.
    """
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
    if base:
        return Path(base) / _CACHE_DIR_NAME
    return Path.home() / ".cache" / _CACHE_DIR_NAME


# Also what stops a server-supplied key from walking out of the cache dir.
def _safe_token(value: str) -> str:
    """Strip a cache key down to characters that are safe in a filename."""
    return "".join(c for c in value if c.isalnum() or c in "-_")


def _cache_path(template_id: str, which: str, variant: str | None = None) -> Path:
    """On-disk path of one cached image.

    With no variant the historical shape is kept, ``<id>/<which>.jpg``, so the
    cache already sitting on a user's disk stays valid. A variant is a size
    token such as "w512" and lands as a suffix, ``<id>/<which>@w512.jpg``, so
    two widths of one image never overwrite each other.
    """
    name = _safe_token(which)
    token = _safe_token(variant) if variant else ""
    if token:
        name = f"{name}@{token}"
    return _cache_root() / _safe_token(template_id) / f"{name}.jpg"


# Fallback expiry, for a server that sends no ETag and no Last-Modified: with
# nothing to revalidate against, dropping the file and fetching it again is the
# only way a re-seeded demo reaches the user. An entry that carries a validator
# is checked on the interval instead, an immutable one never.
_CACHE_TTL_SECONDS = 7 * 24 * 3600


def read_cached_pixmap(template_id: str, which: str, variant: str | None = None,
                       immutable: bool = False) -> QPixmap | None:
    """Return a QPixmap from the on-disk cache, or None if absent or stale.

    Never touches the network: an entry due for revalidation is still served
    from disk, and the caller checks it in the background.
    """
    path = _cache_path(template_id, which, variant)
    if not path.is_file():
        return None
    try:
        expires = not immutable and read_validator(path) is None
        if expires and (time.time() - path.stat().st_mtime) > _CACHE_TTL_SECONDS:
            return None
        pm = QPixmap(str(path))
        if pm.isNull() or pm.width() < 2:
            return None
        if not expires:
            # Only here: on the others, mtime IS the expiry stamp.
            touch_for_lru(path)
        return pm
    except Exception as err:  # noqa: BLE001
        log_warning(f"Failed to read cached demo {path}: {err}")
        return None


def _http_status(reply: QNetworkReply) -> int:
    """HTTP status of a finished reply, 0 when there is none (transport error)."""
    raw = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


class _PendingFetch(NamedTuple):
    """One queued request, with everything the fetch and its callback need."""
    template_id: str
    which: str
    url: str
    headers: dict | None
    variant: str | None
    immutable: bool = False
    # True for a conditional check of bytes already on screen.
    revalidate: bool = False


class TemplateDemoLoader(QObject):
    """Async fetcher for library images. One instance per dialog.

    Signals:
        loaded(template_id, which, QPixmap) - fires when a download (or cache
            hit) yields a usable pixmap. The card matching template_id + which
            installs it into the slider.
        failed(template_id, which) - fires once we've concluded the image will
            never be available (404 server-side or persistent network error).
    """

    loaded = pyqtSignal(str, str, QPixmap)
    failed = pyqtSignal(str, str)

    # Cap simultaneous fetches so opening the library (or a popup with bigger
    # preview images) doesn't fire dozens of requests at once and choke a slow
    # link. Excess requests queue and start as in-flight ones finish. Kept low
    # so a thin pipe isn't split too many ways (each split is likelier to time
    # out).
    _MAX_CONCURRENT = 3

    # Of those slots, at most one may ever hold a background revalidation, so a
    # user staring at a placeholder always has the other two.
    _MAX_CONCURRENT_REVALIDATE = 1

    # Let the dialog lay out, paint and load its first images before walking
    # the cache dir: a few hundred stat() calls, but the first seconds of the
    # library are when a weak PC has nothing to spare.
    _SWEEP_DELAY_MS = 4000

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._queue: list[_PendingFetch] = []
        self._revalidate_queue: list[_PendingFetch] = []
        self._in_flight = 0
        self._revalidating = 0
        _cache_root().mkdir(parents=True, exist_ok=True)
        # Parented to self: closing the dialog first just cancels it, and the
        # next dialog schedules it again. image_cache_budget holds the
        # once-per-session guard.
        safe_single_shot(self._SWEEP_DELAY_MS, self, self._sweep_cache_once)

    @staticmethod
    def _sweep_cache_once() -> None:
        """Bring the on-disk cache back under budget, once per QGIS session."""
        sweep_image_cache_once(_cache_root(), _CACHE_DIR_NAME)

    def request(self, template_id: str, which: str, url: str,
                headers: dict | None = None, *, variant: str | None = None,
                immutable: bool = False) -> None:
        """Try cache first; if miss, queue an async network fetch.

        ``which`` is normally "before"/"after" for card sliders; the detail
        popup also passes "before_preview", "after_preview" for the bigger
        image, and run history passes "input"/"preview". Any non-empty token
        works as the on-disk cache filename.

        ``headers`` (optional) are raw request headers for authorized fetches
        (the run-history thumbnails need the account's Authorization header).

        ``variant`` (optional) names a size of the same image, e.g. "w512" for
        a server-side thumbnail. It becomes part of the cache filename, so two
        widths of one image coexist instead of overwriting each other.

        ``immutable=True`` declares that the bytes behind this key can never
        change (an archived run artifact is written once, at archive time). No
        expiry and no revalidation, so reopening the library months later costs
        nothing on the network. The disk cache key is (template_id, which,
        variant), and history callers pass the tile's request_id as
        template_id, so the key is per-artifact.
        """
        if not template_id or not which or not url:
            return
        if (template_id, which, variant or "") in _KNOWN_MISSING:
            self.failed.emit(template_id, which)
            return
        # Defer the disk read + decode to the next event-loop turn so a burst of
        # cached cards built in one loop doesn't block the dialog's first paint.
        # Parented to self, so it can't fire after the loader dies.
        pending = _PendingFetch(template_id, which, url, headers, variant, immutable)
        safe_single_shot(0, self, lambda p=pending: self._load_cached_or_fetch(p))

    def _load_cached_or_fetch(self, pending: _PendingFetch) -> None:
        pm = read_cached_pixmap(pending.template_id, pending.which,
                                pending.variant, pending.immutable)
        if pm is None:
            self._queue.append(pending)
            self._pump()
            return
        # Paint first, ask later: no delay for the user, and a re-seeded demo
        # swaps itself in when the background check comes back.
        self.loaded.emit(pending.template_id, pending.which, pm)
        if not pending.immutable:
            self._queue_revalidation(pending)

    def _queue_revalidation(self, pending: _PendingFetch) -> None:
        """Queue a conditional check of an entry we just served, or skip it.

        Skipped with no validator to ask with, and skipped while the last check
        is younger than the interval (ten opens in an afternoon cost one
        request per image, not ten).
        """
        path = _cache_path(pending.template_id, pending.which, pending.variant)
        meta = read_validator(path)
        if not should_revalidate(meta):
            return
        headers = dict(pending.headers or {})
        headers.update(conditional_headers(meta))
        self._revalidate_queue.append(pending._replace(headers=headers, revalidate=True))
        self._pump()

    def _pump(self) -> None:
        """Start queued work: misses first, background checks with what is left.

        A miss is a user watching a placeholder; a revalidation is invisible to
        them. So misses take every free slot, and a check starts only once the
        miss queue is empty.
        """
        while self._in_flight < self._MAX_CONCURRENT and self._queue:
            pending = self._queue.pop(0)
            self._in_flight += 1
            self._start(pending)
        while True:
            free_slot = not self._queue and bool(self._revalidate_queue)
            free_slot = free_slot and self._in_flight < self._MAX_CONCURRENT
            free_slot = free_slot and self._revalidating < self._MAX_CONCURRENT_REVALIDATE
            if not free_slot:
                break
            pending = self._revalidate_queue.pop(0)
            self._in_flight += 1
            self._revalidating += 1
            self._start(pending)

    def _start(self, pending: _PendingFetch) -> None:
        req = QNetworkRequest(QUrl(pending.url))
        # Follow redirects. Resolved via qt_compat (scoped-then-flat) because
        # PyQt5 on some QGIS 3 builds exposes these enums flat, not scoped.
        req.setAttribute(RedirectPolicyAttribute, NoLessSafeRedirectPolicy)
        req.setRawHeader(b"Accept", b"image/jpeg, image/png, image/webp, image/*")
        if pending.headers:
            for hk, hv in pending.headers.items():
                try:
                    req.setRawHeader(str(hk).encode("utf-8"), str(hv).encode("utf-8"))
                except (UnicodeError, TypeError):
                    continue
        req.setTransferTimeout(15_000)
        # Route through QGIS's network manager so the fetch inherits its SSL CA
        # bundle, proxy, and auth config: a bare QNetworkAccessManager fails
        # silently on some CDN hosts. Parent the reply to this loader so it dies
        # with the dialog (no callback on a dead object).
        reply = QgsNetworkAccessManager.instance().get(req)
        reply.setParent(self)
        reply.finished.connect(lambda r=reply, p=pending: self._on_finished(r, p))

    def _on_finished(self, reply: QNetworkReply, pending: _PendingFetch) -> None:
        try:
            if pending.revalidate:
                self._on_revalidated(reply, pending)
            else:
                self._on_fetched(reply, pending)
        finally:
            reply.deleteLater()
            self._in_flight = max(0, self._in_flight - 1)
            if pending.revalidate:
                self._revalidating = max(0, self._revalidating - 1)
            self._pump()

    def _on_revalidated(self, reply: QNetworkReply, pending: _PendingFetch) -> None:
        """Apply the answer to a conditional check. Any failure is a no-op.

        The status is the final hop's: the image URL redirects, and the 304
        comes from the target, never from the intermediate 302. The card is
        already painted, so a timeout or a 5xx leaves the cached image where it
        is: no ``failed``, no delete, nothing to tell the user.
        """
        path = _cache_path(pending.template_id, pending.which, pending.variant)
        status = _http_status(reply)
        if status == 304:
            # Still current. Bump both stamps: the file so the LRU sweep sees
            # it used, the sidecar so we don't ask again before the interval.
            touch_for_lru(path)
            mark_validator_checked(path)
            return
        if status != 200 or reply.error() != QNetworkReply.NetworkError.NoError:
            return
        buf = bytes(reply.readAll())
        if len(buf) < 256:
            return
        pm = QPixmap()
        if not pm.loadFromData(buf):
            return
        # The image changed server-side: keep the new bytes and swap the card.
        self._write_cache(pending, buf, reply)
        self.loaded.emit(pending.template_id, pending.which, pm)

    def _on_fetched(self, reply: QNetworkReply, pending: _PendingFetch) -> None:
        """Apply the answer to a fetch someone is waiting on."""
        template_id, which = pending.template_id, pending.which
        err_code = reply.error()
        http_int = _http_status(reply)
        if err_code != QNetworkReply.NetworkError.NoError or http_int >= 400:
            if http_int == 404:
                _KNOWN_MISSING.add((template_id, which, pending.variant or ""))
            else:
                log_debug(f"Image fetch failed for {template_id}/{which}: "
                          f"err={err_code} http={http_int}")
            self.failed.emit(template_id, which)
            return
        data: QByteArray = reply.readAll()
        buf = bytes(data)
        if len(buf) < 256:
            self.failed.emit(template_id, which)
            return
        pm = QPixmap()
        if not pm.loadFromData(buf):
            log_debug(f"Image bytes did not decode for {template_id}/{which}")
            self.failed.emit(template_id, which)
            return
        self._write_cache(pending, buf, reply)
        self.loaded.emit(template_id, which, pm)

    @staticmethod
    def _write_cache(pending: _PendingFetch, buf: bytes, reply: QNetworkReply) -> None:
        """Store the bytes, and the validator that lets us revalidate them."""
        path = _cache_path(pending.template_id, pending.which, pending.variant)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jpg.tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(buf)
            os.replace(tmp, path)
        except OSError as err:
            log_warning(f"Failed to write image cache {path}: {err}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass  # nosec B110
            return
        if not pending.immutable:
            # No sidecar for immutable bytes: they can never need a check.
            write_validator(path, *validator_from_reply(reply))
