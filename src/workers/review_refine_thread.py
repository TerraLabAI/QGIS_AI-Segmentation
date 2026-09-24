




















from __future__ import annotations

import logging
import queue
import threading

from qgis.PyQt.QtCore import QThread

logger = logging.getLogger(__name__)




REVIEW_REFINE_JOIN_TIMEOUT_MS = 3000


def detached_review_geom(geom):






    from qgis.core import QgsGeometry

    inner = geom.constGet()
    if inner is None:
        return QgsGeometry(geom)
    return QgsGeometry(inner.clone())


class ReviewRefineThread(QThread):







    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._inbox: queue.Queue = queue.Queue()
        self._outbox: list = []
        self._lock = threading.Lock()
        self._aborted = False
        self._stopping = False





        self._live_stamp = None



    def set_live_stamp(self, stamp) -> None:






        self._live_stamp = stamp

    def submit(self, det_idx: int, stamp, refiner, geom, seq: int = 0,
               spec: tuple | None = None) -> bool:

















        if self._aborted or self._stopping:
            return False
        try:
            job = (int(det_idx), stamp, int(seq), refiner,
                   detached_review_geom(geom))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
        self._inbox.put(job)
        return True

    def take_results(self) -> list:







        with self._lock:
            out = self._outbox
            self._outbox = []
        return out

    def finish(self) -> None:













        if self._stopping:
            return
        self._stopping = True
        self._inbox.put(None)

    def abort(self) -> None:







        self._aborted = True
        with self._inbox.mutex:
            self._inbox.queue.clear()
        if self._stopping:
            self._inbox.put(None)
            return
        self._stopping = True
        self._inbox.put(None)

    def join_run(self, timeout_ms: int | None = None) -> bool:








        if not self.isRunning():
            return True
        if timeout_ms is None:
            timeout_ms = REVIEW_REFINE_JOIN_TIMEOUT_MS
            try:
                from ..core.server_dials import dial_in_range
                timeout_ms = dial_in_range(
                    "tuning.review.refine_thread_join_timeout_ms",
                    REVIEW_REFINE_JOIN_TIMEOUT_MS, 500, 30000)
            except Exception:  # noqa: BLE001
                timeout_ms = REVIEW_REFINE_JOIN_TIMEOUT_MS
        return bool(self.wait(timeout_ms))



    def run(self) -> None:  # noqa: D102
        from ..core.live_refine import refine_review_geom
        from ..core.macos_activity import promote_current_thread



        promote_current_thread()
        try:
            while True:
                item = self._inbox.get()
                if item is None:
                    break
                if self._aborted:
                    continue
                det_idx, stamp, seq, refiner, geom = item
                live = self._live_stamp
                if live is not None and stamp != live:



                    continue
                try:
                    shaped, err = refine_review_geom(refiner, geom)
                except Exception as exc:  # noqa: BLE001
                    shaped, err = None, exc
                with self._lock:
                    if not self._aborted and (
                            self._live_stamp is None or stamp == self._live_stamp):
                        self._outbox.append((det_idx, stamp, seq, shaped, err))
        except Exception:  # noqa: BLE001
            logger.warning("ReviewRefineThread: stopped on error", exc_info=True)
