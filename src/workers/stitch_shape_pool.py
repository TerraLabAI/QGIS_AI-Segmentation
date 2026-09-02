"""A small fan-out for the live stitcher's per-object shape pass.

The stitcher folds one tile at a time on one thread, and the shape preset it
applies to each merged object is the larger half of that thread's work. The
shape is a pure function of the object's geometry and the run's dials, so the
objects of one drain cycle can be shaped side by side and the answers are the
same whatever order they come back in.

Most of that work is inside the geometry library, which drops the interpreter
lock while it runs, so a handful of threads do overlap. The fine-grained python
around it does not, which is why the worker count is small and why a cycle with
only a few objects in it is shaped in place instead: below that size the hand-
off costs more than the overlap buys.

The pool belongs to the stitch thread. It is created when that thread starts,
used only from it, and shut down before it exits, so nothing here is ever
touched by the GUI or outlives the run.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: Objects a drain cycle must carry before the pass is worth spreading. Under
#: it the submit-and-collect round trip costs more than the overlap returns.
MIN_BATCH = 8

#: Workers the fan-out opens, on top of the stitch thread itself. Three is
#: where the measured overlap flattens: the geometry calls scale, the python
#: between them does not.
DEFAULT_WORKERS = 3


def worker_count(want: int) -> int:
    """How many workers to actually open for a machine this size.

    The stitch thread works alongside them, so the pass occupies one processor
    more than this number. Two are left free whatever the dial asks for: one
    for the interface, which has to keep painting, and one for the rest of the
    run. A machine with two processors opens none and shapes in place.
    """
    want = max(0, int(want))
    if want <= 0:
        return 0
    try:
        cpus = int(os.cpu_count() or 1)
    except (TypeError, ValueError):
        cpus = 1
    return max(0, min(want, cpus - 2))


class ShapeFanout:
    """Applies one pure per-object function to a batch, over a few threads.

    ``fn`` must be safe to call from several threads at once and must depend on
    nothing but its argument, because the batch is spread without any ordering
    between the calls. Results come back in the order they were asked for, and a
    call that raises comes back as None, exactly as a single-threaded caller's
    own try/except would leave it.
    """

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
                self._pool = ThreadPoolExecutor(
                    max_workers=count, thread_name_prefix="tl-stitch-shape")
            except Exception:  # noqa: BLE001 - shaping in place is the fallback
                logger.warning("ShapeFanout: no pool, shaping in place",
                               exc_info=True)
                self._pool = None

    def _one(self, item):
        try:
            return self._fn(item)
        except Exception:  # noqa: BLE001 - one bad object, not the batch
            return None

    def map(self, items: list) -> list:
        """``[fn(item) for item in items]``, spread when the batch is worth it.

        The calling thread keeps a share of the batch rather than waiting on
        the workers: it is going to block here either way, so the pass runs on
        one more processor than the pool has threads, and a single worker still
        halves the wall clock instead of only moving the work.
        """
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
        except Exception:  # noqa: BLE001 - a dead pool must not end the run
            logger.warning("ShapeFanout: batch failed, shaping in place",
                           exc_info=True)
            return [self._one(item) for item in items]

    def close(self) -> None:
        """Stop the workers. Called once, by the thread that owns the pool."""
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)
