



























from __future__ import annotations

import os
import time
from pathlib import Path
from string import ascii_letters, digits
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
from ..core.server_dials import dial_in_range, read_value
from ..core.surface_dials import library_demo_cache_ttl_s
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





_KNOWN_MISSING: set[tuple[str, str, str]] = set()







_CACHE_DIR_BASE = "ai-segmentation-template-demos"
_CACHE_DIR_FALLBACK_VERSION = "1"
_CACHE_VERSION_KEY = "demo_cache_version"

_CACHE_VERSION_CHARS = frozenset(ascii_letters + digits + "._-")
_CACHE_VERSION_MAX_CHARS = 24


def _cache_dir_version() -> str:







    value = read_value(_CACHE_VERSION_KEY)
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return _CACHE_DIR_FALLBACK_VERSION
    token = value.strip()
    if not token or len(token) > _CACHE_VERSION_MAX_CHARS:
        return _CACHE_DIR_FALLBACK_VERSION
    if not all(c in _CACHE_VERSION_CHARS for c in token):
        return _CACHE_DIR_FALLBACK_VERSION
    return token


def _cache_dir_name() -> str:

    return f"{_CACHE_DIR_BASE}-v{_cache_dir_version()}"


def _cache_root() -> Path:







    name = _cache_dir_name()
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
    if base:
        return Path(base) / name
    return Path.home() / ".cache" / name


def _remove_superseded_cache_dirs(active: Path) -> None:







    import shutil

    try:
        for child in active.parent.iterdir():
            name = child.name
            if name == active.name:
                continue
            if name != _CACHE_DIR_BASE and not name.startswith(f"{_CACHE_DIR_BASE}-"):
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
    except OSError:
        pass  # nosec B110




_CACHE_SWEEP_TASK = None



def _safe_token(value: str) -> str:

    return "".join(c for c in value if c.isalnum() or c in "-_")


def _cache_path(template_id: str, which: str, variant: str | None = None) -> Path:







    name = _safe_token(which)
    token = _safe_token(variant) if variant else ""
    if token:
        name = f"{name}@{token}"
    return _cache_root() / _safe_token(template_id) / f"{name}.jpg"






_CACHE_TTL_SECONDS = 7 * 24 * 3600


def read_cached_pixmap(template_id: str, which: str, variant: str | None = None,
                       immutable: bool = False) -> QPixmap | None:





    path = _cache_path(template_id, which, variant)
    if not path.is_file():
        return None
    try:
        expires = not immutable and read_validator(path) is None
        ttl = library_demo_cache_ttl_s(_CACHE_TTL_SECONDS)
        if expires and (time.time() - path.stat().st_mtime) > ttl:
            return None
        pm = QPixmap(str(path))
        if pm.isNull() or pm.width() < 2:
            return None
        if not expires:

            touch_for_lru(path)
        return pm
    except Exception as err:  # noqa: BLE001
        log_warning(f"Failed to read cached demo {path}: {err}")
        return None


def _http_status(reply: QNetworkReply) -> int:

    raw = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


class _PendingFetch(NamedTuple):

    template_id: str
    which: str
    url: str
    headers: dict | None
    variant: str | None
    immutable: bool = False

    revalidate: bool = False




    fallback_url: str | None = None
    fallback_headers: dict | None = None


class TemplateDemoLoader(QObject):










    loaded = pyqtSignal(str, str, QPixmap)
    failed = pyqtSignal(str, str)











    _MAX_CONCURRENT = 8



    _MAX_CONCURRENT_REVALIDATE = 1




    _SWEEP_DELAY_MS = 4000

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._queue: list[_PendingFetch] = []
        self._revalidate_queue: list[_PendingFetch] = []
        self._in_flight = 0
        self._revalidating = 0
        self._max_concurrent = dial_in_range(
            "tuning.library.demo_max_concurrent", self._MAX_CONCURRENT, 1, 16)


        self._max_concurrent_revalidate = min(
            dial_in_range(
                "tuning.library.demo_max_concurrent_revalidate",
                self._MAX_CONCURRENT_REVALIDATE, 0, 4),
            max(0, self._max_concurrent - 1),
        )
        sweep_delay_ms = dial_in_range(
            "tuning.library.demo_sweep_delay_ms", self._SWEEP_DELAY_MS, 500, 30000)
        _cache_root().mkdir(parents=True, exist_ok=True)



        safe_single_shot(sweep_delay_ms, self, self._sweep_cache_once)

    @staticmethod
    def _sweep_cache_once() -> None:








        global _CACHE_SWEEP_TASK
        root = _cache_root()

        def _sweep() -> dict:
            _remove_superseded_cache_dirs(root)
            sweep_image_cache_once(root, root.name)
            return {}

        try:
            from qgis.core import QgsApplication

            from ..workers.generic_request_task import GenericRequestTask

            task = GenericRequestTask("Image cache cleanup", _sweep, hidden=True)
            _CACHE_SWEEP_TASK = task
            QgsApplication.taskManager().addTask(task)
        except Exception:  # noqa: BLE001
            _sweep()

    def request(self, template_id: str, which: str, url: str,
                headers: dict | None = None, *, variant: str | None = None,
                immutable: bool = False, fallback_url: str | None = None,
                fallback_headers: dict | None = None) -> None:




























        if not template_id or not which or not url:
            return
        if (template_id, which, variant or "") in _KNOWN_MISSING:
            self.failed.emit(template_id, which)
            return



        pending = _PendingFetch(
            template_id, which, url, headers, variant, immutable,
            fallback_url=fallback_url, fallback_headers=fallback_headers)
        safe_single_shot(0, self, lambda p=pending: self._load_cached_or_fetch(p))

    def _load_cached_or_fetch(self, pending: _PendingFetch) -> None:
        pm = read_cached_pixmap(pending.template_id, pending.which,
                                pending.variant, pending.immutable)
        if pm is None:
            self._queue.append(pending)
            self._pump()
            return


        self.loaded.emit(pending.template_id, pending.which, pm)
        if not pending.immutable:
            self._queue_revalidation(pending)

    def _queue_revalidation(self, pending: _PendingFetch) -> None:






        path = _cache_path(pending.template_id, pending.which, pending.variant)
        meta = read_validator(path)
        if not should_revalidate(meta):
            return
        headers = dict(pending.headers or {})
        headers.update(conditional_headers(meta))
        self._revalidate_queue.append(pending._replace(headers=headers, revalidate=True))
        self._pump()

    def _pump(self) -> None:






        while self._in_flight < self._max_concurrent and self._queue:
            pending = self._queue.pop(0)
            self._in_flight += 1
            self._start(pending)
        while True:
            free_slot = not self._queue and bool(self._revalidate_queue)
            free_slot = free_slot and self._in_flight < self._max_concurrent
            free_slot = free_slot and self._revalidating < self._max_concurrent_revalidate
            if not free_slot:
                break
            pending = self._revalidate_queue.pop(0)
            self._in_flight += 1
            self._revalidating += 1
            self._start(pending)

    def _start(self, pending: _PendingFetch) -> None:
        req = QNetworkRequest(QUrl(pending.url))


        req.setAttribute(RedirectPolicyAttribute, NoLessSafeRedirectPolicy)
        req.setRawHeader(b"Accept", b"image/jpeg, image/png, image/webp, image/*")
        if pending.headers:
            for hk, hv in pending.headers.items():
                try:
                    req.setRawHeader(str(hk).encode("utf-8"), str(hv).encode("utf-8"))
                except (UnicodeError, TypeError):
                    continue

        if hasattr(req, "setTransferTimeout"):
            req.setTransferTimeout(dial_in_range(
                "tuning.library.demo_transfer_timeout_ms", 15_000, 5_000, 60_000))




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







        path = _cache_path(pending.template_id, pending.which, pending.variant)
        status = _http_status(reply)
        if status == 304:


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

        self._write_cache(pending, buf, reply)
        self.loaded.emit(pending.template_id, pending.which, pm)

    def _on_fetched(self, reply: QNetworkReply, pending: _PendingFetch) -> None:

        template_id, which = pending.template_id, pending.which
        err_code = reply.error()
        http_int = _http_status(reply)
        if err_code != QNetworkReply.NetworkError.NoError or http_int >= 400:
            if self._retry_on_fallback(pending):
                return
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
            if self._retry_on_fallback(pending):
                return
            self.failed.emit(template_id, which)
            return
        pm = QPixmap()
        if not pm.loadFromData(buf):
            if self._retry_on_fallback(pending):
                return
            log_debug(f"Image bytes did not decode for {template_id}/{which}")
            self.failed.emit(template_id, which)
            return
        self._write_cache(pending, buf, reply)
        self.loaded.emit(template_id, which, pm)

    def _retry_on_fallback(self, pending: _PendingFetch) -> bool:






        if pending.revalidate or not pending.fallback_url:
            return False
        self._queue.append(pending._replace(
            url=pending.fallback_url,
            headers=pending.fallback_headers,
            fallback_url=None,
            fallback_headers=None,
        ))
        return True

    @staticmethod
    def _write_cache(pending: _PendingFetch, buf: bytes, reply: QNetworkReply) -> None:

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

            write_validator(path, *validator_from_reply(reply))
