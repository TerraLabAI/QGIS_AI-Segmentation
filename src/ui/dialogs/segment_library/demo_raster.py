



















from __future__ import annotations

import os
import time
from pathlib import Path

from qgis.core import Qgis, QgsNetworkAccessManager
from qgis.PyQt.QtCore import QCoreApplication, QObject, QStandardPaths, QUrl, pyqtSignal
from qgis.PyQt.QtNetwork import QNetworkReply, QNetworkRequest

from ....core.i18n import tr
from ....core.logging_utils import log
from ....core.presets.segmentation_presets_client import absolute_demo_url
from ....core.qt_compat import (
    NoLessSafeRedirectPolicy,
    RedirectPolicyAttribute,
    reply_http_status,
    resolve_qt_enum,
)
from ....core.server_dials import dial_in_range
from ...image_cache_validators import (
    conditional_headers,
    mark_validator_checked,
    read_validator,
    should_revalidate,
    validator_from_reply,
    validator_path,
    write_validator,
)

CACHE_DIR_NAME = "ai-segmentation-library-rasters"
RASTER_NAME = "raster.tif"
MAX_BYTES = 60 * 1024 * 1024


_TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


ERR_NETWORK = "network"
ERR_TOO_LARGE = "too_large"
ERR_INVALID = "invalid"
ERR_NOT_FOUND = "not_found"


def demo_raster_url_for(base: str, preset: dict) -> str:

    return absolute_demo_url(base, preset.get("demo_raster_url"))


def _raster_dir_name(preset_id: str) -> str:
    keep = "".join(c for c in str(preset_id) if c.isalnum() or c in "-_")
    return keep[:80] or "preset"


def demo_raster_cache_path(preset_id: str) -> Path:
    root = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
    return Path(os.path.join(root, CACHE_DIR_NAME, _raster_dir_name(preset_id), RASTER_NAME))


def cached_demo_raster(preset_id: str) -> Path | None:


    path = demo_raster_cache_path(preset_id)
    pending = path.with_name(RASTER_NAME + ".new")
    if pending.is_file():
        try:
            os.replace(str(pending), str(path))
            meta = read_validator(pending)
            if meta:
                write_validator(path, meta.get("etag", ""), meta.get("last_modified", ""))
            try:
                os.remove(str(validator_path(pending)))
            except OSError:
                pass
        except OSError:
            pass
    try:
        if path.is_file() and _is_tiff_file(path):
            return path
    except OSError:
        pass
    return None


def _is_tiff_file(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(4) in _TIFF_MAGIC


def _timeout_ms() -> int:
    return dial_in_range("tuning.library.demo_raster_timeout_ms", 30_000, 5_000, 120_000)


class DemoRasterDownload(QObject):








    progress = pyqtSignal(int)
    done = pyqtSignal(str, int)
    failed = pyqtSignal(str, int)

    def __init__(self, preset_id: str, url: str, parent=None, revalidate: bool = False):
        super().__init__(parent)
        self._preset_id = preset_id
        self._url = url
        self._revalidate = revalidate
        self._target = demo_raster_cache_path(preset_id)
        if revalidate:
            self._target = self._target.with_name(RASTER_NAME + ".new")
        self._part = self._target.with_name(self._target.name + ".part")
        self._reply = None
        self._fh = None
        self._received = 0
        self._ended = False

    @property
    def received(self) -> int:
        return self._received

    def start(self) -> None:
        req = QNetworkRequest(QUrl(self._url))
        req.setAttribute(RedirectPolicyAttribute, NoLessSafeRedirectPolicy)
        req.setRawHeader(b"Accept", b"image/tiff, application/octet-stream")



        req.setAttribute(
            resolve_qt_enum(QNetworkRequest, "Attribute", "CacheLoadControlAttribute"),
            resolve_qt_enum(QNetworkRequest, "CacheLoadControl", "AlwaysNetwork"))
        req.setAttribute(
            resolve_qt_enum(QNetworkRequest, "Attribute", "CacheSaveControlAttribute"), False)
        if self._revalidate:
            for k, v in conditional_headers(read_validator(demo_raster_cache_path(self._preset_id))).items():
                req.setRawHeader(k.encode("latin-1"), v.encode("latin-1"))
        if hasattr(req, "setTransferTimeout"):
            req.setTransferTimeout(_timeout_ms())
        try:
            self._part.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._part, "wb")
        except OSError:
            self._finish_failed(ERR_INVALID)
            return
        reply = QgsNetworkAccessManager.instance().get(req)
        reply.setParent(self)
        self._reply = reply
        reply.readyRead.connect(self._on_ready_read)
        reply.downloadProgress.connect(self._on_progress)
        reply.finished.connect(self._on_finished)

    def abort(self) -> None:
        if self._ended:
            return
        self._ended = True
        self._close_part(delete=True)
        if self._reply is not None:
            try:
                self._reply.abort()
            except RuntimeError:
                pass



    def _on_progress(self, received: int, total: int) -> None:
        if self._ended:
            return
        if total > MAX_BYTES or received > MAX_BYTES:
            self._finish_failed(ERR_TOO_LARGE)
            return
        self.progress.emit(int(received * 100 / total) if total > 0 else -1)

    def _on_ready_read(self) -> None:
        if self._ended or self._reply is None or self._fh is None:
            return
        status = reply_http_status(self._reply)
        if status is not None and status != 200:
            return
        data = bytes(self._reply.readAll())
        if self._received == 0 and data and data[:4] not in _TIFF_MAGIC:
            self._finish_failed(ERR_INVALID)
            return
        self._received += len(data)
        if self._received > MAX_BYTES:
            self._finish_failed(ERR_TOO_LARGE)
            return
        try:
            self._fh.write(data)
        except OSError:
            self._finish_failed(ERR_INVALID)

    def _on_finished(self) -> None:
        if self._ended:
            return
        reply = self._reply
        self._on_ready_read()
        if self._ended:
            return
        status = reply_http_status(reply)
        if self._revalidate and status == 304:
            self._close_part(delete=True)
            mark_validator_checked(demo_raster_cache_path(self._preset_id))
            self._ended = True
            self.done.emit(str(demo_raster_cache_path(self._preset_id)), 0)
            return
        if status in (404, 410):
            self._finish_failed(ERR_NOT_FOUND)
            return
        if reply.error() != QNetworkReply.NetworkError.NoError or status != 200:
            self._finish_failed(ERR_NETWORK)
            return
        if self._received < 8:
            self._finish_failed(ERR_INVALID)
            return
        self._close_part(delete=False)
        try:
            os.replace(str(self._part), str(self._target))
        except OSError:
            self._finish_failed(ERR_INVALID)
            return
        etag, last_modified = validator_from_reply(reply)
        write_validator(self._target, etag, last_modified)
        self._ended = True
        self.done.emit(str(self._target), self._received)

    def _finish_failed(self, kind: str) -> None:
        if self._ended:
            return
        self._ended = True
        self._close_part(delete=True)
        if self._reply is not None:
            try:
                self._reply.abort()
            except RuntimeError:
                pass
        log(f"library raster {self._preset_id}: {kind}", Qgis.MessageLevel.Info)
        self.failed.emit(kind, self._received)

    def _close_part(self, delete: bool) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        if delete:
            try:
                os.remove(str(self._part))
            except OSError:
                pass


def revalidate_demo_raster(preset_id: str, url: str) -> None:






    if preset_id in _REVALIDATING:
        return



    path = demo_raster_cache_path(preset_id)
    if path.with_name(RASTER_NAME + ".new").is_file():
        return
    meta = read_validator(demo_raster_cache_path(preset_id))

    if meta is None or not should_revalidate(meta):
        return
    app = QCoreApplication.instance()
    job = DemoRasterDownload(preset_id, url, app, revalidate=True)

    def _ended(*_a) -> None:
        _REVALIDATING.discard(preset_id)
        job.deleteLater()

    job.done.connect(_ended)
    job.failed.connect(_ended)
    _REVALIDATING.add(preset_id)
    job.start()



_REVALIDATING: set[str] = set()


def _file_size(path) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


class PresetRasterTryMixin:






    def _on_try_raster(self) -> None:
        if self._raster_job is not None or not self._raster_url:
            return
        if getattr(self.parent(), "_view_only", False):
            return
        self._raster_error.setVisible(False)
        self._raster_t0 = time.monotonic()
        pid = str(self._preset.get("id", ""))
        cached = cached_demo_raster(pid)
        if cached is not None:
            revalidate_demo_raster(pid, self._raster_url)
            self._finish_try(str(cached), "cached", _file_size(cached))
            return
        self.try_btn.setEnabled(False)
        self.try_btn.setText(tr("Downloading..."))
        job = DemoRasterDownload(pid, self._raster_url, self)
        job.progress.connect(self._on_raster_progress)
        job.done.connect(lambda path, n: self._finish_try(path, "ok", n))
        job.failed.connect(self._on_raster_failed)
        self._raster_job = job
        job.start()

    def _on_raster_progress(self, percent: int) -> None:
        if percent < 0:
            self.try_btn.setText(tr("Downloading..."))
        else:
            self.try_btn.setText(tr("Downloading {percent}%").format(percent=percent))

    def _on_raster_failed(self, kind: str, received: int) -> None:
        self._raster_job = None
        self._track_try("error", received, kind)
        messages = {
            ERR_TOO_LARGE: tr("This image is too large to download."),
            ERR_INVALID: tr("This image could not be read."),
            ERR_NOT_FOUND: tr("This example image is not available right now."),
        }
        self._raster_error.setText(messages.get(
            kind, tr("Download failed. Check your connection and try again.")))
        self._raster_error.setVisible(True)
        self._reset_try_btn()

    def _reset_try_btn(self) -> None:
        self.try_btn.setText(tr("Try on this image"))
        self.try_btn.setEnabled(True)

    def _finish_try(self, path: str, outcome: str, size: int) -> None:
        self._raster_job = None
        self._track_try(outcome, size)
        self.tried_raster_path = path
        self.chosen = True
        self.accept()

    def _track_try(self, outcome: str, size: int, error_kind: str = "") -> None:
        try:
            from ....core import telemetry_session_events
            telemetry_session_events.track_library_example_tried(
                str(self._preset.get("id", "")), outcome,
                size / (1024 * 1024),
                max(0.0, time.monotonic() - self._raster_t0),
                error_kind=error_kind)
        except Exception:
            pass  # nosec B110

    def done(self, result: int) -> None:  # noqa: N802

        job = self._raster_job
        if job is not None:
            self._raster_job = None
            received = job.received
            job.abort()
            self._track_try("cancelled", received)
        super().done(result)
