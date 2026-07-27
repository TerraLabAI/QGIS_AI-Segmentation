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
        # The bridge's slot must run on the bridge's (main) thread even when the
        # signal is emitted from the worker thread: a queued connection marshals
        # the call onto the bridge's event loop. AutoConnection already does this
        # across threads, but we pin QueuedConnection to be explicit.
        from qgis.PyQt.QtCore import Qt as _Qt
        self._render_requested.connect(
            self._on_render_requested, _Qt.ConnectionType.QueuedConnection)

    def _tile_extent(self, tx: int, ty: int, tw: int, th: int):
        """Build the tile's bbox_native as a QgsRectangle in the layer CRS,
        from the global geo_transform. Identical math to
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
        from ..core.cloud_detection import start_tile_render_job

        started = False
        try:
            extent = self._tile_extent(tx, ty, tw, th)
            started = start_tile_render_job(
                self._layer, extent, out_w or tw, out_h or th,
                lambda img, s=seq: self._store_result(s, img),
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
                # wakes us. The timeout is only a safety net (a render job
                # itself times out at 60s); cancel() wakes immediately.
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
