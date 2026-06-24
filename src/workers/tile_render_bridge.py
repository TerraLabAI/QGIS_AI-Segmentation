






from __future__ import annotations

import logging
import math
import threading
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

    _fallback_render_requested = pyqtSignal(int, int, int, int, int, int, int, int)

    def __init__(self, layer, geo_transform: dict, parent=None,
                 floor_ratio: float = 0.0, transform_context=None):
        super().__init__(parent)
        self._layer = layer
        self._geo_transform = geo_transform
        from qgis.core import QgsCoordinateTransformContext, QgsProject




        self._transform_context = QgsCoordinateTransformContext(
            transform_context if transform_context is not None
            else QgsProject.instance().transformContext())
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






        self._reach_zoom = None
        self._reach_settled = threading.Event()
        self._start_reach_probe()



        self._fallback_plan = self._build_fallback_plan()



        self._floor_cap_zoom = self._build_floor_cap_zoom(floor_ratio)
        self._fallback_clones: dict[int, object] = {}
        self._fallback_zoom_used: int | None = None




        from qgis.PyQt.QtCore import Qt as _Qt
        self._render_requested.connect(
            self._on_render_requested, _Qt.ConnectionType.QueuedConnection)
        self._fallback_render_requested.connect(
            self._on_fallback_render_requested,
            _Qt.ConnectionType.QueuedConnection)

    def _build_fallback_plan(self):

        try:
            from ..core.online_zoom_reach import fallback_plan

            _zone, mupp = self._zone_in_web_mercator()
            return fallback_plan(self._layer, mupp)
        except Exception as exc:  # noqa: BLE001
            logger.debug("TileRenderBridge: no zoom fallback: %s", exc)
            return None

    def _build_floor_cap_zoom(self, floor_ratio: float):

        if not floor_ratio or floor_ratio <= 1.0:
            return None
        try:
            from ..core.online_zoom_reach import floor_cap_zoom

            _zone, mupp = self._zone_in_web_mercator()
            return floor_cap_zoom(self._layer, mupp, floor_ratio)
        except Exception as exc:  # noqa: BLE001
            logger.debug("TileRenderBridge: no floor cap: %s", exc)
            return None

    def fallback_zooms(self) -> tuple:


        if self._fallback_plan is None:
            return ()
        from ..core.online_zoom_reach import fallback_zooms

        return fallback_zooms(self._fallback_plan,
                              self.basemap_reach_zoom())

    def render_tile_at_zoom(self, tx: int, ty: int, tw: int, th: int,
                            out_w: int, out_h: int, zoom: int):


        self._mutex.lock()
        try:
            if self._cancelled:
                return None
            seq = self._seq
            self._seq += 1
            self._requested_at[seq] = time.monotonic()
        finally:
            self._mutex.unlock()
        self._fallback_render_requested.emit(
            seq, tx, ty, tw, th, out_w, out_h, int(zoom))
        return self.collect_render(seq)

    def note_fallback_zoom(self, zoom: int) -> None:


        if self._fallback_zoom_used is None or zoom < self._fallback_zoom_used:
            self._fallback_zoom_used = int(zoom)

    def basemap_fallback_zoom(self):

        return self._fallback_zoom_used

    @pyqtSlot(int, int, int, int, int, int, int, int)
    def _on_fallback_render_requested(
        self, seq: int, tx: int, ty: int, tw: int, th: int,
        out_w: int, out_h: int, zoom: int,
    ) -> None:

        from ..core.cloud_detection import start_tile_render_job

        if self._cancelled or self._refuse_renders:
            self._store_result(seq, None)
            return
        started = False
        try:
            clone = self._fallback_clones.get(zoom)
            if clone is None:
                from ..core.online_zoom_reach import reach_clone

                clone = reach_clone(self._layer, zoom)
                if clone is None:
                    raise ValueError("no capped copy of the layer")
                self._fallback_clones[zoom] = clone
            job_t0 = time.monotonic()
            started = start_tile_render_job(
                self._layer, self._tile_extent(tx, ty, tw, th),
                out_w or tw, out_h or th,
                lambda img, s=seq: self._render_landed(s, img, job_t0),
                timeout_ms=self._render_timeout_ms,
                render_clone=clone, clone_resolved=True,
                render_crs=self._run_crs(),
                transform_context=self._transform_context,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("TileRenderBridge: fallback render failed at "
                           "(%d,%d) zoom %d: %s", tx, ty, zoom, exc)
            started = False
        if not started:
            self._store_result(seq, None)

    def _start_reach_probe(self) -> None:


        request = None
        try:
            from ..core.online_zoom_reach import reach_request

            zone, mupp = self._zone_in_web_mercator()
            if zone is not None:
                request = reach_request(self._layer, zone, mupp)
        except Exception as exc:  # noqa: BLE001
            logger.debug("TileRenderBridge: no zoom probe: %s", exc)
        if request is None:
            self._reach_settled.set()
            return

        def probe():
            try:
                from ..core.online_zoom_reach import probe_reach, remember_reach

                self._reach_zoom = probe_reach(
                    request, cancel_check=lambda: self._cancelled)


                remember_reach(request.source_key, request.points[0],
                               self._reach_zoom)
            finally:
                self._reach_settled.set()

        threading.Thread(target=probe, name="ai-seg-zoom-reach",
                         daemon=True).start()

    def _zone_in_web_mercator(self):

        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsRectangle,
        )

        bbox = self._geo_transform.get("bbox")
        shape = self._geo_transform.get("img_shape")
        crs = self._run_crs() or self._layer.crs()
        if not bbox or not shape or crs is None or not crs.isValid():
            return None, 0.0
        rect = QgsRectangle(*(float(v) for v in bbox))
        mercator = QgsCoordinateReferenceSystem("EPSG:3857")
        if crs.authid() != "EPSG:3857":
            rect = QgsCoordinateTransform(
                crs, mercator, self._transform_context).transformBoundingBox(rect)
        width_px = int(shape[1])
        if width_px <= 0 or rect.width() <= 0:
            return None, 0.0
        return ((rect.xMinimum(), rect.yMinimum(), rect.xMaximum(),
                 rect.yMaximum()), rect.width() / width_px)

    def _await_reach(self) -> None:


        if not self._reach_settled.is_set():
            self._reach_settled.wait(35.0)

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
                    if self._render_clone is None:
                        self._render_clone = self._reach_clone()




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
                transform_context=self._transform_context,
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

    def basemap_reach_zoom(self):


        if self._render_clone is None:
            return None
        reach = self._reach_zoom if self._reach_settled.is_set() else None
        if self._floor_cap_zoom is not None:
            return (self._floor_cap_zoom if reach is None
                    else min(reach, self._floor_cap_zoom))
        return reach

    def _reach_clone(self):



        zoom = self._reach_zoom if self._reach_settled.is_set() else None
        cap = self._floor_cap_zoom
        if cap is not None:
            zoom = cap if zoom is None else min(zoom, cap)
        if zoom is None:
            return None
        from ..core.online_zoom_reach import reach_clone

        clone = reach_clone(self._layer, zoom)
        if clone is not None and cap is not None and zoom == cap:
            try:
                from qgis.core import Qgis, QgsMessageLog

                QgsMessageLog.logMessage(
                    f"Auto detection: tiles read zoom {zoom}, the level the "
                    "imagery check found a picture at",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
        elif clone is not None:
            try:
                from qgis.core import Qgis, QgsMessageLog

                QgsMessageLog.logMessage(
                    f"Auto detection: the basemap host serves zoom {zoom} "
                    "here, past the layer's max zoom; tiles read that level",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
        return clone

    def release_render_clone(self) -> None:














        self._render_clone = None
        self._render_clone_built = True
        self._fallback_clones.clear()
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





        self._await_reach()
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
