"""TileRenderBridge: main-thread render bridge for the per-tile JIT path.

QgsMapRendererParallelJob is GUI-thread only, so the AutoDetectionWorker thread
can never render directly. It asks this bridge to render a tile via a
QueuedConnection signal and blocks on a QWaitCondition until the bridge stores
the result and wakes it.
"""
from __future__ import annotations

import logging

from qgis.PyQt.QtCore import (
    QMutex,
    QObject,
    QWaitCondition,
    pyqtSignal,
    pyqtSlot,
)

logger = logging.getLogger(__name__)

# What start_tile_render_job allows a tile render when nothing is served.
_TILE_RENDER_TIMEOUT_MS = 60_000


def _resolve_render_timeout_ms() -> int:
    """The served per-tile render deadline, or the shipped one. Cache-only, so
    it never networks and is safe on any thread; falls back on any failure so a
    missing configuration renders exactly as it does today."""
    try:
        from ..core.detection_policy import tile_render_timeout_ms
        return tile_render_timeout_ms(_TILE_RENDER_TIMEOUT_MS)
    except Exception:  # noqa: BLE001 - a bad dial must never block a render
        return _TILE_RENDER_TIMEOUT_MS


class TileRenderBridge(QObject):
    """Main-thread render bridge for the per-tile JIT path.

    Lives on (and is moved to) the GUI thread. QgsMapRendererParallelJob is
    GUI-thread only, so the worker thread can never render directly; instead it
    asks this bridge to render a tile via a QueuedConnection signal and blocks on
    a QWaitCondition until the bridge stores the result and wakes it.

    The bridge owns the layer + global geo_transform: given a tile rect
    (tx, ty, tw, th) it computes that tile's bbox_native from the SAME global
    geo_transform the worker uses for georeferencing, then renders ONLY that
    sub-extent to a tw x th QImage. The render thus matches slicing one big zone
    render exactly (same CRS, same map units per pixel, same tile origin).
    """

    _render_requested = pyqtSignal(int, int, int, int, int, int, int)

    def __init__(self, layer, geo_transform: dict, parent=None):
        super().__init__(parent)
        self._layer = layer
        self._geo_transform = geo_transform
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        # request_seq -> QImage|None once the render is done; None means not ready.
        self._results: dict[int, object] = {}
        self._done: set[int] = set()
        self._seq = 0
        self._cancelled = False
        # Resampling clone of a local raster, built once per run on the main
        # thread and reused by every tile. Building one reopens the GDAL
        # dataset on the thread that also paints, so it must not happen per
        # tile. None until the first render, and it stays None for an online
        # provider, which needs no clone.
        self._render_clone = None
        self._render_clone_built = False
        # Resolved once from geo_transform["crs"], wrapped in a tuple so a
        # resolved None is told apart from "not looked up yet".
        self._run_crs_cache: tuple | None = None
        # Deadline on one tile's render. Past it the tile has no imagery and is
        # retried, then dropped uncharged, so on a slow link this decides how
        # much of a zone comes back. Served, because the value that suits a
        # 4G user is not the one that suits an office line, and a plugin
        # release is days away. Resolved once per run (cache-only read).
        self._render_timeout_ms = _resolve_render_timeout_ms()
        # The bridge's slot must run on the bridge's (main) thread even when the
        # signal is emitted from the worker thread: a queued connection marshals
        # the call onto the bridge's event loop. AutoConnection already does this
        # across threads, but we pin QueuedConnection to be explicit.
        from qgis.PyQt.QtCore import Qt as _Qt
        self._render_requested.connect(
            self._on_render_requested, _Qt.ConnectionType.QueuedConnection)

    def _run_crs(self):
        """The CRS the run's bbox is in, so the render targets the same one.

        None only when the run named no CRS at all, and the render then uses the
        layer's own, which is what it always did.

        Raises when the run DID name a CRS that this install cannot build. That
        is not a case to render through: every tile extent is in that CRS, so
        rendering in the layer's instead would send the model a picture of
        somewhere else and bill it. Failing the tile is the honest outcome.

        Built once and kept: a QgsCoordinateReferenceSystem costs a database
        lookup and this is asked once per tile."""
        cached = self._run_crs_cache
        if cached is not None:
            return cached[0]
        from qgis.core import QgsCoordinateReferenceSystem

        authid = ""
        try:
            authid = self._geo_transform.get("crs") or ""
        except (RuntimeError, AttributeError, TypeError):
            authid = ""
        crs = None
        if authid:
            candidate = QgsCoordinateReferenceSystem(authid)
            if not candidate.isValid():
                raise ValueError(
                    "the run's CRS cannot be built on this install: " + str(authid))
            crs = candidate
        self._run_crs_cache = (crs,)
        return crs

    def _tile_extent(self, tx: int, ty: int, tw: int, th: int):
        """Build the tile's bbox_native as a QgsRectangle in the run CRS, from
        the global geo_transform. Identical math to
        AutoDetectionWorker._make_tile_transform (bbox_native), so the two never
        diverge."""
        from qgis.core import QgsRectangle

        src_bbox = self._geo_transform.get("bbox", (0.0, 0.0, 1.0, 1.0))
        img_shape = self._geo_transform.get("img_shape", (1, 1))
        img_h, img_w = max(img_shape[0], 1), max(img_shape[1], 1)
        src_minx, src_miny, src_maxx, src_maxy = src_bbox
        px_w = (src_maxx - src_minx) / img_w
        px_h = (src_maxy - src_miny) / img_h
        tile_minx = src_minx + tx * px_w
        tile_maxx = src_minx + (tx + tw) * px_w
        tile_miny = src_maxy - (ty + th) * px_h
        tile_maxy = src_maxy - ty * px_h
        return QgsRectangle(tile_minx, tile_miny, tile_maxx, tile_maxy)

    @pyqtSlot(int, int, int, int, int, int, int)
    def _on_render_requested(
        self, seq: int, tx: int, ty: int, tw: int, th: int,
        out_w: int, out_h: int,
    ) -> None:
        """Start this tile's render as an ASYNC job on the main thread.

        The slot returns immediately; the job's completion stores the QImage
        and wakes the worker. Because the slot never blocks, several queued
        requests start OVERLAPPING render jobs - that concurrency is the whole
        point of the worker-side prefetch (the serialized per-tile render is
        the bottleneck of large runs, not the network or the detection host).
        (out_w, out_h) is the OUTPUT pixel size; when it differs from (tw, th)
        the same ground extent renders at a finer scale (the saturated-tile
        re-split path). 0 means "same as the rect"."""
        from ..core.cloud_detection import (
            _local_raster_render_clone,
            start_tile_render_job,
        )

        if self._cancelled:
            # request_render drops the mutex before it emits, so a request can
            # still land here after the run was torn down. Building anything
            # now would reopen the raster for a run that is over.
            self._store_result(seq, None)
            return

        started = False
        try:
            extent = self._tile_extent(tx, ty, tw, th)
            if not self._render_clone_built:
                # Main thread here, which is where a QgsRasterLayer must be
                # built and later destroyed. Once per run, not once per tile.
                try:
                    self._render_clone = _local_raster_render_clone(self._layer)
                    # Latch on SUCCESS only. Latching before the call would
                    # turn one failed clone into a whole run rendered through
                    # the raw layer, decimated nearest, which is the aliasing
                    # the clone exists to avoid. Retrying is nearly free.
                    self._render_clone_built = True
                except Exception as exc:  # noqa: BLE001 - falls back to the layer
                    self._render_clone = None
                    logger.warning(
                        "TileRenderBridge: render clone unavailable, this tile "
                        "renders through the layer itself: %s", exc)
            started = start_tile_render_job(
                self._layer, extent, out_w or tw, out_h or th,
                lambda img, s=seq: self._store_result(s, img),
                timeout_ms=self._render_timeout_ms,
                render_clone=self._render_clone,
                render_crs=self._run_crs(),
            )
        except Exception as exc:  # noqa: BLE001 - never break the handshake
            logger.warning("TileRenderBridge: render failed at (%d,%d): %s", tx, ty, exc)
            started = False
        if not started:
            self._store_result(seq, None)

    def _store_result(self, seq: int, img) -> None:
        """Store one render's outcome and wake every waiting collector."""
        self._mutex.lock()
        try:
            self._results[seq] = img
            self._done.add(seq)
            self._cond.wakeAll()
        finally:
            self._mutex.unlock()

    def cancel(self) -> None:
        """Called FROM THE WORKER THREAD (via request_stop). Mark the bridge
        cancelled and wake any render_tile blocked on the condition AT ONCE, so a
        stop never waits out the condition timeout. This is the deadlock guard:
        on unload the main thread blocks in worker.wait(); if the worker were
        parked in render_tile waiting for a main-thread render that can no longer
        run, only this immediate wake lets it exit so wait() returns cleanly."""
        self._mutex.lock()
        try:
            self._cancelled = True
            self._cond.wakeAll()
        finally:
            self._mutex.unlock()

    def release_render_clone(self) -> None:
        """Drop the cached render clone. MAIN THREAD ONLY.

        Deliberately not done in cancel(), which runs on the worker thread: a
        QgsRasterLayer has to be destroyed on the thread that built it, and
        dropping the last reference from the worker would be a cross-thread
        delete. Also releases the file handle the clone holds on the user's
        raster, which Windows needs before that file can be moved or deleted.

        The latch stays SET. Clearing it would re-arm the lazy build in
        _on_render_requested, and a request already past request_render's
        unlock can still arrive after this: the bridge would reopen the raster
        for a finished run, and nothing would ever release it again because the
        controller has already dropped its reference to this bridge.
        """
        self._render_clone = None
        self._render_clone_built = True

    def request_render(self, tx: int, ty: int, tw: int, th: int,
                       out_w: int = 0, out_h: int = 0) -> int | None:
        """Called FROM THE WORKER THREAD. Post a render request WITHOUT
        blocking and return its collect token (or None when cancelled). The
        render runs as an async job on the main thread, so several requested
        tiles render side by side while the worker keeps driving the network
        window - this is the prefetch half of the pipeline."""
        self._mutex.lock()
        try:
            if self._cancelled:
                return None
            seq = self._seq
            self._seq += 1
        finally:
            self._mutex.unlock()
        # Emit OUTSIDE the lock so the queued slot can run on the main thread.
        self._render_requested.emit(seq, tx, ty, tw, th, out_w, out_h)
        return seq

    def collect_render(self, seq: int):
        """Called FROM THE WORKER THREAD. Block until the requested render is
        done and return its QImage (or None). Returns at once when the result
        already landed, and unblocks the moment cancel() is called even if the
        render never ran (the unload deadlock guard)."""
        self._mutex.lock()
        try:
            while seq not in self._done and not self._cancelled:
                # Block until the main thread stores this render OR cancel()
                # wakes us. The wait is only a safety net and re-arms, so a
                # render deadline longer than it still resolves here; cancel()
                # wakes immediately.
                self._cond.wait(self._mutex, 30000)
            img = self._results.pop(seq, None)
            self._done.discard(seq)
            return img
        finally:
            self._mutex.unlock()

    def render_tile(self, tx: int, ty: int, tw: int, th: int,
                    out_w: int = 0, out_h: int = 0):
        """Called FROM THE WORKER THREAD. Request a main-thread render of this
        tile and block until it is done, returning the QImage (or None).
        Composition of request_render + collect_render, kept as the simple
        synchronous entry point (non-prefetched tiles, batched path, tests)."""
        seq = self.request_render(tx, ty, tw, th, out_w, out_h)
        if seq is None:
            return None
        return self.collect_render(seq)
