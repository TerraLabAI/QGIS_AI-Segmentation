

















from __future__ import annotations

import logging
import queue
import threading
import time

from qgis.PyQt.QtCore import QThread, pyqtSignal

from ..core import transport_dials as _td

logger = logging.getLogger(__name__)






RAW_FRAGMENT_RETAIN_CAP = 40000




STITCH_JOIN_TIMEOUT_MS = 8000





_RESCALE_MIN_CHANGE = 0.05





_SHAPE_MAX_WAIT = 3


DELTA_ADD = "add"
DELTA_UPDATE = "update"
DELTA_REMOVE = "remove"


def _build_area_measurer(crs_authid: str):




    from qgis.core import QgsCoordinateReferenceSystem

    from ..core.layer_conventions import make_area_measurer

    try:
        return make_area_measurer(QgsCoordinateReferenceSystem(crs_authid))
    except Exception:  # noqa: BLE001
        return None


def _detached(geom):






    from qgis.core import QgsGeometry

    inner = geom.constGet()
    if inner is None:
        return QgsGeometry(geom)
    return QgsGeometry(inner.clone())


class LiveStitchThread(QThread):










    batch_ready = pyqtSignal()

    def __init__(
        self,
        merger,
        params: dict,
        pixel_size: float,
        metres_per_unit: float,
        crs_authid: str,
        unit_aspect: float = 1.0,
        retain_fragments: bool = False,
        retain_coverage: bool = False,
        tile_ground_area: float = 0.0,
        hard_coverage: float = 0.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._merger = merger
        self._params = dict(params or {})
        self._pixel_size = float(pixel_size or 0.0)
        self._metres_per_unit = float(metres_per_unit or 1.0)



        self._unit_aspect = float(unit_aspect or 1.0)
        self._crs_authid = crs_authid or "EPSG:4326"
        self._retain_fragments = bool(retain_fragments)
        self._retain_coverage = bool(retain_coverage)
        self._tile_ground_area = float(tile_ground_area or 0.0)
        self._hard_coverage = float(hard_coverage or 0.0)

        self._inbox: queue.Queue = queue.Queue()
        self._outbox: list = []
        self._lock = threading.Lock()
        self._aborted = False
        self._stopping = False


        self._shown: dict[int, float] = {}






        self._pending_shape: dict[int, int] = {}












        self._drawn_from: dict[int, object] = {}



        self._area_of: dict[int, tuple] = {}
        self._refiner = None

        self._shape_pool = None




        self._shaped_ahead: dict[int, tuple] = {}



        self._measurer = _build_area_measurer(self._crs_authid)
        self._rescaled = False




        self.fold_ms = 0.0
        self.tiles_folded = 0
        self.rescales = 0







        self.shaped_geoms: dict[int, object] = {}
        self.shape_pixel_size = float(pixel_size or 0.0)
        self.shape_metres_per_unit = self._metres_per_unit




        self.raw_count = 0
        self.raw_n_total = 0
        self.raw_coverage_sum = 0.0
        self.raw_coverage_sq_sum = 0.0
        self.raw_fragments: list | None = [] if retain_fragments else None



    def submit(self, detections: list, pixel_size: float = 0.0) -> None:







        if self._aborted:
            return
        self._inbox.put((list(detections or ()), float(pixel_size or 0.0)))

    def take_deltas(self) -> list:





        with self._lock:
            out = self._outbox
            self._outbox = []
        return out

    def pending(self) -> int:


        return max(0, self._inbox.qsize() - (1 if self._stopping else 0))

    def finish(self) -> None:






        if self._stopping:
            return
        self._stopping = True
        self._inbox.put(None)

    def abort(self) -> None:






        self._aborted = True
        if self._stopping:
            return
        self._stopping = True
        self._inbox.put(None)

    def join_run(self, timeout_ms: int = STITCH_JOIN_TIMEOUT_MS) -> bool:





        if not self.isRunning():
            return True
        return bool(self.wait(timeout_ms))



    def run(self) -> None:  # noqa: D102
        try:
            self._build_tools()
            while True:
                item = self._inbox.get()
                if item is None:
                    break
                if self._aborted:
                    continue
                detections, pixel_size = item
                started = time.perf_counter()
                try:
                    self._fold_tile(detections, pixel_size)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "LiveStitchThread: tile fold failed", exc_info=True)
                finally:
                    self.fold_ms += (time.perf_counter() - started) * 1000.0
                    self.tiles_folded += 1
            if not self._aborted:



                self._publish(self._drain_merger_changes(settle_all=True))
        except Exception:  # noqa: BLE001
            logger.warning("LiveStitchThread: stopped on error", exc_info=True)
        finally:


            self._close_shape_pool()

    def _build_tools(self) -> None:










        from ..core.live_refine import LiveRefiner
        from .stitch_shape_pool import (
            DEFAULT_WORKERS,
            MIN_BATCH,
            ShapeFanout,
        )

        self._refiner = LiveRefiner(
            self._params, self._pixel_size, self._metres_per_unit,
            self._unit_aspect)
        self._shape_pool = ShapeFanout(
            self._refiner.refine,
            workers=_td.stitch_shape_workers(DEFAULT_WORKERS),
            min_batch=_td.stitch_shape_min_batch(MIN_BATCH))

    def _close_shape_pool(self) -> None:

        pool, self._shape_pool = self._shape_pool, None
        if pool is not None:
            try:
                pool.close()
            except Exception:  # noqa: BLE001
                logger.warning("LiveStitchThread: shape pool would not close",
                               exc_info=True)

    def _wants_rescale(self, pixel_size: float) -> bool:









        if pixel_size <= 0 or self._rescaled:
            return False
        widest = max(pixel_size, self._pixel_size)
        if widest <= 0:
            return False
        return (abs(pixel_size - self._pixel_size) / widest
                > _td.stitch_rescale_min_change(_RESCALE_MIN_CHANGE))

    def _rescale(self, pixel_size: float) -> None:






        from ..core.live_refine import LiveRefiner

        self._pixel_size = pixel_size
        self.shape_pixel_size = float(pixel_size)
        self._rescaled = True
        self.rescales += 1




        self.shaped_geoms.clear()
        self._drawn_from.clear()
        self._refiner = LiveRefiner(
            self._params, self._pixel_size, self._metres_per_unit,
            self._unit_aspect)
        self._merger.mark_changed(list(self._shown))

    def _fold_tile(self, detections: list, pixel_size: float) -> None:

        from qgis.core import QgsGeometry

        if self._wants_rescale(pixel_size):
            self._rescale(pixel_size)

        for wkb, score in detections:
            geom = QgsGeometry()
            geom.fromWkb(wkb)
            if geom.isEmpty():
                continue
            self.raw_count += 1
            if self._retain_coverage:
                self._note_coverage(geom)
            if self.raw_fragments is not None:
                if len(self.raw_fragments) >= _td.stitch_raw_fragment_retain_cap(RAW_FRAGMENT_RETAIN_CAP):





                    logger.warning(
                        "LiveStitchThread: raw fragment retain ceiling reached; "
                        "count grouping is no longer available for this run")
                    self.raw_fragments = None
                else:
                    self.raw_fragments.append((wkb, float(score)))




            self._merger.add(geom, float(score))

        self._publish(self._drain_merger_changes())

    def _publish(self, deltas: list) -> None:






        if not deltas or self._aborted:
            return
        with self._lock:
            self._outbox.extend(deltas)
        self.batch_ready.emit()

    def _note_coverage(self, geom) -> None:



        if self._tile_ground_area <= 0:
            self.raw_n_total += 1
            return
        self.raw_n_total += 1
        cov = geom.area() / self._tile_ground_area
        if 0.0 < cov <= self._hard_coverage:
            self.raw_coverage_sum += cov
            self.raw_coverage_sq_sum += cov * cov

    def _drain_merger_changes(self, settle_all: bool = False) -> list:
























        changed, removed = self._merger.drain_changes()
        out: list = []
        for fid in removed:
            self._forget(fid)
            if self._shown.pop(fid, None) is not None:
                out.append((DELTA_REMOVE, fid, None, 0.0))
        hot = set(changed)
        max_wait = _td.stitch_shape_max_wait(_SHAPE_MAX_WAIT)




        waits = [(fid, self._pending_shape.get(fid, 0) + 1) for fid in changed]
        self._shape_ahead(
            [fid for fid, waited in waits if waited >= max_wait or settle_all])
        for fid, waited in waits:
            if waited >= max_wait or settle_all:
                self._pending_shape.pop(fid, None)
                out.append(self._object_delta(fid, shaped=True))
            else:
                out.append(self._object_delta(fid, shaped=False))
                if self._wants_shape(fid):
                    self._pending_shape[fid] = waited
                else:
                    self._pending_shape.pop(fid, None)
        settling = [f for f in self._pending_shape if settle_all or f not in hot]
        self._shape_ahead(settling)
        for fid in settling:
            self._pending_shape.pop(fid, None)
            out.append(self._object_delta(fid, shaped=True))
        self._shaped_ahead.clear()
        return [d for d in out if d is not None]

    def _shape_input(self, fid: int):







        geom, score = self._merger.keeper(fid)
        if geom is None or geom.isEmpty():
            return None
        if not self._passes_filters(score, self._object_area(fid, geom)):
            return None
        if self._already_drawn(fid, geom, True):
            return None
        return geom

    def _shape_ahead(self, fids: list) -> None:








        pool = self._shape_pool
        if pool is None or not fids:
            return
        work = []
        for fid in fids:
            geom = self._shape_input(fid)
            if geom is not None:
                work.append((fid, geom))
        if not work:
            return
        shapes = pool.map([geom for _fid, geom in work])
        for (fid, geom), shape in zip(work, shapes):
            self._shaped_ahead[fid] = (geom, shape)

    def _wants_shape(self, fid: int) -> bool:








        return fid in self._shown and fid not in self.shaped_geoms

    def _forget(self, fid: int) -> None:





        self._pending_shape.pop(fid, None)
        self._drawn_from.pop(fid, None)
        self._area_of.pop(fid, None)
        self.shaped_geoms.pop(fid, None)

    def _object_delta(self, fid: int, shaped: bool):








        try:
            return self._build_object_delta(fid, shaped)
        except Exception:  # noqa: BLE001
            logger.warning(
                "LiveStitchThread: no delta for object %s", fid, exc_info=True)
            return None

    def _build_object_delta(self, fid: int, shaped: bool):




        from ..core.layer_conventions import to_multipolygon

        geom, score = self._merger.keeper(fid)
        if geom is None or geom.isEmpty():
            return self._drop(fid)
        if not self._passes_filters(score, self._object_area(fid, geom)):
            return self._drop(fid)
        if self._already_drawn(fid, geom, shaped):
            return self._rescore_delta(fid, geom, score)
        shape = None
        if shaped:



            held = self._shaped_ahead.pop(fid, None)
            if held is not None and held[0] is geom:
                shape = held[1]
            else:
                try:
                    shape = self._refiner.refine(geom)
                except Exception:  # noqa: BLE001
                    shape = None
        refined = shaped and shape is not None and not shape.isEmpty()
        if not refined:



            self.shaped_geoms.pop(fid, None)
        if shape is None or shape.isEmpty():





            shape = _detached(geom)
        shape = to_multipolygon(shape)
        if shape is None or shape.isEmpty():
            return self._drop(fid)
        kind = DELTA_UPDATE if fid in self._shown else DELTA_ADD
        self._shown[fid] = float(score)
        self._drawn_from[fid] = geom
        if refined:





            self.shaped_geoms[fid] = shape
            shape = _detached(shape)
        return (kind, fid, shape, float(score))

    def _already_drawn(self, fid: int, geom, shaped: bool) -> bool:






        if fid not in self._shown or self._drawn_from.get(fid) is not geom:
            return False
        return (not shaped) or fid in self.shaped_geoms

    def _rescore_delta(self, fid: int, geom, score: float):







        from ..core.layer_conventions import to_multipolygon

        if self._shown.get(fid) == float(score):
            return None
        held = self.shaped_geoms.get(fid)


        shape = to_multipolygon(_detached(held if held is not None else geom))
        if shape is None or shape.isEmpty():
            return self._drop(fid)
        self._shown[fid] = float(score)
        return (DELTA_UPDATE, fid, shape, float(score))

    def _drop(self, fid: int):






        self._pending_shape.pop(fid, None)
        self._drawn_from.pop(fid, None)
        self.shaped_geoms.pop(fid, None)
        if self._shown.pop(fid, None) is None:
            return None
        return (DELTA_REMOVE, fid, None, 0.0)

    def _object_area(self, fid: int, geom) -> float:







        held = self._area_of.get(fid)
        if held is not None and held[0] is geom:
            return held[1]
        try:
            if self._measurer is not None:
                area = float(self._measurer.measureArea(geom))
            else:
                area = float(geom.area())
        except (RuntimeError, AttributeError):
            try:
                area = float(geom.area())
            except (RuntimeError, AttributeError):
                return 0.0
        self._area_of[fid] = (geom, area)
        return area

    def _passes_filters(self, score: float, area: float) -> bool:


        from ..core.review_defaults import object_passes_review_gates
        return object_passes_review_gates(score, area, self._params)
