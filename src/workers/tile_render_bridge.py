"""TileRenderBridge: main-thread render bridge for the per-tile JIT path.

QgsMapRendererParallelJob is GUI-thread only, so the AutoDetectionWorker thread
can never render directly. It asks this bridge to render a tile via a
QueuedConnection signal and blocks on a QWaitCondition until the bridge stores
the result and wakes it.
"""
from __future__ import annotations

import logging
import time

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
        # seq -> monotonic instant the render was requested, and seq -> how
        # long it took. The worker reads the duration when it collects, so a
        # render that ran slowly behind a deep prefetch still counts as slow
        # even though nobody waited on it.
        self._requested_at: dict[int, float] = {}
        self._durations: dict[int, float] = {}
        self._last_collected_duration = 0.0
        # What the GUI thread itself owes this run. queue_s is emit to slot
        # entry, which is the event loop's own backlog and so a direct readout
        # of how busy the GUI thread was; slot_s is the slot body, which builds
        # the extent and starts the async job. Neither is the render itself
        # (that runs off the GUI thread), so a large queue_s means the run was
        # waiting on the GUI thread rather than on imagery.
        self._queue_s = 0.0
        self._slot_s = 0.0
        self._slot_count = 0
        self._worst_queue_s = 0.0
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
            if isinstance(cached[0], Exception):
                # The failure is kept as well as the success: a run naming a
                # CRS this install cannot build used to rebuild it and raise
                # again for every tile, on the GUI thread.
                raise cached[0]
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
                refusal = ValueError(
                    "the run's CRS cannot be built on this install: " + str(authid))
                self._run_crs_cache = (refusal,)
                raise refusal
            crs = candidate
        self._run_crs_cache = (crs,)
        return crs

    def _tile_extent(self, tx: int, ty: int, tw: int, th: int):
        """Build the tile's bbox_native as a QgsRectangle in the run CRS, from
        the global geo_transform. Identical math to
        AutoDetectionWorker._make_tile_transform (bbox_native), so the two never
        diverge.

        Raises when the geo_transform names no usable extent, for the same
        reason _run_crs raises on a CRS it cannot build: a fallback would
        render a 1x1 unit box at the CRS origin, and that picture is encoded,
        submitted and billed as this tile."""
        from qgis.core import QgsRectangle

        src_bbox = self._geo_transform.get("bbox")
        img_shape = self._geo_transform.get("img_shape")
        if (not isinstance(src_bbox, (list, tuple)) or len(src_bbox) != 4
                or not isinstance(img_shape, (list, tuple))
                or len(img_shape) < 2):
            raise ValueError("the run's geo_transform names no extent")
        src_minx, src_miny, src_maxx, src_maxy = (float(v) for v in src_bbox)
        img_h, img_w = int(img_shape[0]), int(img_shape[1])
        if (img_h <= 0 or img_w <= 0
                or src_maxx <= src_minx or src_maxy <= src_miny):
            raise ValueError("the run's geo_transform names a degenerate extent")
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

        slot_t0 = time.monotonic()
        self._mutex.lock()
        try:
            asked_at = self._requested_at.get(seq)
            if asked_at is not None:
                waited = max(0.0, slot_t0 - asked_at)
                self._queue_s += waited
                self._worst_queue_s = max(self._worst_queue_s, waited)
            self._slot_count += 1
        finally:
            self._mutex.unlock()

        if self._cancelled:
            # request_render drops the mutex before it emits, so a request can
            # still land here after the run was torn down. Building anything
            # now would reopen the raster for a run that is over.
            self._store_result(seq, None)
            self._slot_s += time.monotonic() - slot_t0
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
                clone_resolved=self._render_clone_built,
                render_crs=self._run_crs(),
            )
        except Exception as exc:  # noqa: BLE001 - never break the handshake
            logger.warning("TileRenderBridge: render failed at (%d,%d): %s", tx, ty, exc)
            started = False
        if not started:
            self._store_result(seq, None)
        self._slot_s += time.monotonic() - slot_t0

    def gui_thread_summary(self) -> dict:
        """What this run cost the GUI thread, and what it waited on it for.

        ``queue_s`` is the total emit-to-slot delay over the run: the render
        requests sat in the GUI thread's event queue for that long, which is
        the plainest measure there is of the GUI thread being the run's
        bottleneck. ``slot_s`` is the slot body, which is the only part that
        runs ON the GUI thread; the render itself does not.
        """
        self._mutex.lock()
        try:
            return {
                "renders": self._slot_count,
                "queue_s": round(self._queue_s, 2),
                "slot_s": round(self._slot_s, 2),
                "worst_queue_s": round(self._worst_queue_s, 2),
            }
        finally:
            self._mutex.unlock()

    def _store_result(self, seq: int, img) -> None:
        """Store one render's outcome and wake every waiting collector.

        A render that lands after the bridge was cancelled is dropped rather
        than stored: every collect_render exits at once once cancelled, so
        nothing will ever take it, and a whole tile bitmap per outstanding
        prefetch would sit here until the bridge itself is released.
        """
        self._mutex.lock()
        try:
            if self._cancelled:
                self._cond.wakeAll()
                return
            self._results[seq] = img
            self._done.add(seq)
            started = self._requested_at.pop(seq, None)
            if started is not None:
                self._durations[seq] = max(0.0, time.monotonic() - started)
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
        try:
            from qgis.core import Qgis, QgsMessageLog

            summary = self.gui_thread_summary()
            if summary["renders"]:
                QgsMessageLog.logMessage(
                    "Auto detection: render bridge - {renders} request(s), "
                    "{queue_s:.1f}s queued on the GUI thread (worst "
                    "{worst_queue_s:.1f}s), {slot_s:.1f}s in the slot"
                    .format(**summary),
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
        except Exception:  # noqa: BLE001 - a log line must never end a run
            pass  # nosec B110

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
            self._requested_at[seq] = time.monotonic()
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
            self._requested_at.pop(seq, None)
            self._last_collected_duration = self._durations.pop(seq, 0.0)
            return img
        finally:
            self._mutex.unlock()

    def last_render_duration(self) -> float:
        """Called FROM THE WORKER THREAD, right after collect_render: how long
        that render took from request to result, in seconds (0.0 when it was
        never timed, for example a cancelled bridge)."""
        self._mutex.lock()
        try:
            return float(self._last_collected_duration)
        finally:
            self._mutex.unlock()

    def render_ready(self, seq: int) -> bool:
        """Called FROM THE WORKER THREAD. True when this render has landed and
        collect_render would return at once. A cancelled bridge answers True
        for the same reason collect_render returns at once then: nothing is
        worth waiting for any more."""
        self._mutex.lock()
        try:
            return self._cancelled or seq in self._done
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
