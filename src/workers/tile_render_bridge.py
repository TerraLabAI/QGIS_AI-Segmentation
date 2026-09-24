






from __future__ import annotations

import logging
import math
import time

from qgis.PyQt.QtCore import (
    QMutex,
    QObject,
    QWaitCondition,
    pyqtSignal,
    pyqtSlot,
)

from ..core import run_timeline as _timeline

logger = logging.getLogger(__name__)


_TILE_RENDER_TIMEOUT_MS = 60_000


def _resolve_render_timeout_ms() -> int:



    try:
        from ..core.detection_policy import tile_render_timeout_ms
        return tile_render_timeout_ms(_TILE_RENDER_TIMEOUT_MS)
    except Exception:  # noqa: BLE001
        return _TILE_RENDER_TIMEOUT_MS


class TileRenderBridge(QObject):














    _render_requested = pyqtSignal(int, int, int, int, int, int, int)

    def __init__(self, layer, geo_transform: dict, parent=None):
        super().__init__(parent)
        self._layer = layer
        self._geo_transform = geo_transform
        self._mutex = QMutex()
        self._cond = QWaitCondition()

        self._results: dict[int, object] = {}
        self._done: set[int] = set()




        self._requested_at: dict[int, float] = {}
        self._durations: dict[int, float] = {}
        self._last_collected_duration = 0.0






        self._queue_s = 0.0
        self._slot_s = 0.0
        self._slot_count = 0
        self._worst_queue_s = 0.0
        self._seq = 0
        self._cancelled = False


        self._landed_hook = None





        self._render_clone = None
        self._render_clone_built = False


        self._run_crs_cache: tuple | None = None





        self._render_timeout_ms = _resolve_render_timeout_ms()




        self._deadline_is_final: bool | None = None
        self._rendered_any = False
        self._refuse_renders = False




        from qgis.PyQt.QtCore import Qt as _Qt
        self._render_requested.connect(
            self._on_render_requested, _Qt.ConnectionType.QueuedConnection)

    def _run_crs(self):












        cached = self._run_crs_cache
        if cached is not None:
            if isinstance(cached[0], Exception):



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









        from qgis.core import QgsRectangle

        src_bbox = self._geo_transform.get("bbox")
        img_shape = self._geo_transform.get("img_shape")
        if (not isinstance(src_bbox, (list, tuple)) or len(src_bbox) != 4
                or not isinstance(img_shape, (list, tuple))
                or len(img_shape) < 2):
            raise ValueError("the run's geo_transform names no extent")
        src_minx, src_miny, src_maxx, src_maxy = (float(v) for v in src_bbox)
        img_h, img_w = int(img_shape[0]), int(img_shape[1])
        if (not all(math.isfinite(v) for v in
                    (src_minx, src_miny, src_maxx, src_maxy))
                or img_h <= 0 or img_w <= 0 or tw <= 0 or th <= 0
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










        from ..core.cloud_detection import (
            _local_raster_render_clone,
            start_tile_render_job,
        )

        slot_t0 = time.monotonic()
        _timeline.mark("render_slot")
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

        if self._cancelled or self._refuse_renders:



            self._store_result(seq, None)
            self._slot_s += time.monotonic() - slot_t0
            return

        started = False
        try:
            extent = self._tile_extent(tx, ty, tw, th)
            if not self._render_clone_built:


                try:
                    self._render_clone = _local_raster_render_clone(self._layer)




                    self._render_clone_built = True
                except Exception as exc:  # noqa: BLE001
                    self._render_clone = None
                    logger.warning(
                        "TileRenderBridge: render clone unavailable, this tile "
                        "renders through the layer itself: %s", exc)
            job_t0 = time.monotonic()
            started = start_tile_render_job(
                self._layer, extent, out_w or tw, out_h or th,
                lambda img, s=seq: self._render_landed(s, img, job_t0),
                timeout_ms=self._render_timeout_ms,
                render_clone=self._render_clone,
                clone_resolved=self._render_clone_built,
                render_crs=self._run_crs(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("TileRenderBridge: render failed at (%d,%d): %s", tx, ty, exc)
            started = False
        if not started:
            self._store_result(seq, None)
        self._slot_s += time.monotonic() - slot_t0

    def _render_landed(self, seq: int, img, job_t0: float) -> None:



        if img is not None:
            self._rendered_any = True
        elif (not self._rendered_any and not self._refuse_renders
                and time.monotonic() - job_t0
                >= 0.9 * self._render_timeout_ms / 1000.0
                and self._missed_deadline_is_final()):
            self._refuse_renders = True
            try:
                from qgis.core import Qgis, QgsMessageLog

                QgsMessageLog.logMessage(
                    "Auto detection: a tile of this local raster took over "
                    f"{self._render_timeout_ms / 1000.0:.0f}s to render and "
                    "none has rendered yet; the remaining tiles are not "
                    "rendered",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
        self._store_result(seq, img)

    def _missed_deadline_is_final(self) -> bool:




        if self._deadline_is_final is None:
            final = False
            try:
                import os

                source = self._layer.source() or ""
                low = source.lower()
                final = (self._layer.providerType() == "gdal"
                         and not low.startswith("/vsi") and "://" not in low
                         and os.path.isfile(source))
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
                final = False
            self._deadline_is_final = final
        return self._deadline_is_final

    def gui_thread_summary(self) -> dict:








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







        self._mutex.lock()
        try:
            if self._cancelled:
                self._cond.wakeAll()
                return
            self._results[seq] = img
            self._done.add(seq)
            _timeline.mark("render_done")
            started = self._requested_at.pop(seq, None)
            if started is not None:
                self._durations[seq] = max(0.0, time.monotonic() - started)
            self._cond.wakeAll()
            hook = self._landed_hook
        finally:
            self._mutex.unlock()
        if hook is not None:
            try:
                hook(seq)
            except Exception:  # noqa: BLE001
                pass  # nosec B110

    def set_landed_hook(self, hook) -> None:


        self._mutex.lock()
        try:
            self._landed_hook = hook
        finally:
            self._mutex.unlock()

    def cancel(self) -> None:






        self._mutex.lock()
        try:
            self._cancelled = True
            self._results.clear()
            self._done.clear()
            self._requested_at.clear()
            self._durations.clear()
            self._landed_hook = None
            self._cond.wakeAll()
        finally:
            self._mutex.unlock()

    def release_render_clone(self) -> None:














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
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def request_render(self, tx: int, ty: int, tw: int, th: int,
                       out_w: int = 0, out_h: int = 0) -> int | None:





        self._mutex.lock()
        try:
            if self._cancelled:
                return None
            seq = self._seq
            self._seq += 1
            self._requested_at[seq] = time.monotonic()
        finally:
            self._mutex.unlock()

        self._render_requested.emit(seq, tx, ty, tw, th, out_w, out_h)
        return seq

    def collect_render(self, seq: int):




        self._mutex.lock()
        try:
            while seq not in self._done and not self._cancelled:




                self._cond.wait(self._mutex, 30000)
            img = self._results.pop(seq, None)
            self._done.discard(seq)
            self._requested_at.pop(seq, None)
            self._last_collected_duration = self._durations.pop(seq, 0.0)
            return img
        finally:
            self._mutex.unlock()

    def collect_render_timed(self, seq: int) -> tuple:


        self._mutex.lock()
        try:
            while seq not in self._done and not self._cancelled:
                self._cond.wait(self._mutex, 30000)
            img = self._results.pop(seq, None)
            self._done.discard(seq)
            self._requested_at.pop(seq, None)
            return img, float(self._durations.pop(seq, 0.0))
        finally:
            self._mutex.unlock()

    def last_render_duration(self) -> float:



        self._mutex.lock()
        try:
            return float(self._last_collected_duration)
        finally:
            self._mutex.unlock()

    def render_ready(self, seq: int) -> bool:




        self._mutex.lock()
        try:
            return self._cancelled or seq in self._done
        finally:
            self._mutex.unlock()

    def render_tile(self, tx: int, ty: int, tw: int, th: int,
                    out_w: int = 0, out_h: int = 0):




        seq = self.request_render(tx, ty, tw, th, out_w, out_h)
        if seq is None:
            return None
        return self.collect_render(seq)
