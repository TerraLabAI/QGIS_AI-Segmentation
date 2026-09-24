

















from __future__ import annotations

import logging

logger = logging.getLogger(__name__)



MIN_BATCH = 8




DEFAULT_WORKERS = 3


def worker_count(want: int) -> int:







    want = max(0, int(want))
    if want <= 0:
        return 0
    try:
        from .tile_convert_threads import usable_cores

        cpus = int(usable_cores())
    except (TypeError, ValueError, ImportError):
        cpus = 1
    return max(0, min(want, cpus - 2))


class ShapeFanout:









    def __init__(self, fn, workers: int = DEFAULT_WORKERS,
                 min_batch: int = MIN_BATCH) -> None:
        self._fn = fn
        self._min_batch = max(1, int(min_batch))
        self._pool = None
        self._workers = 0
        count = worker_count(workers)
        if count > 0:
            self._workers = count
            try:
                from concurrent.futures import ThreadPoolExecutor

                from ..core.macos_activity import promote_current_thread

                self._pool = ThreadPoolExecutor(
                    max_workers=count, thread_name_prefix="tl-stitch-shape",
                    initializer=promote_current_thread)
            except Exception:  # noqa: BLE001
                logger.warning("ShapeFanout: no pool, shaping in place",
                               exc_info=True)
                self._pool = None

    def _one(self, item):
        try:
            return self._fn(item)
        except Exception:  # noqa: BLE001
            return None

    def map(self, items: list) -> list:







        if not items:
            return []
        if self._pool is None or len(items) < self._min_batch:
            return [self._one(item) for item in items]
        try:
            mine = len(items) // (self._workers + 1)
            cut = len(items) - mine
            sent = [self._pool.submit(self._one, item) for item in items[:cut]]
            kept = [self._one(item) for item in items[cut:]]
            return [f.result() for f in sent] + kept
        except Exception:  # noqa: BLE001
            logger.warning("ShapeFanout: batch failed, shaping in place",
                           exc_info=True)
            return [self._one(item) for item in items]

    def close(self) -> None:

        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)
