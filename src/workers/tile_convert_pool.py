"""Off-loop conversion of finished tiles into ready geometry.

The streaming run loop drives the sockets and used to also convert every
finished tile's masks into geometry inline. A dense tile carries well over a
hundred masks and several hundred polygons, so that conversion took longer than
the inference the loop was waiting on: the loop fired a window of tiles, blocked
until it had converted all of them, then fired the next window. The window
stopped sliding and became a barrier, and the service sat idle for most of a
large run.

This pool takes the conversion off that loop. The run loop hands each finished
tile over and goes straight back to firing the next one; finished conversions
come back through a queue the loop drains between cycles. Every Qt signal is
still emitted from the worker thread, and the conversion itself is the same
code on the same inputs, so the run's output does not change: only when the
work happens does.

No Qt import here, so the scheduling is unit-testable off a running QGIS.
"""
from __future__ import annotations

import logging
import os
import queue
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

#: Converter threads when the caller asks for the default. The stages that
#: dominate (mask decode, polygonize, GEOS clip) drop the GIL for their C parts,
#: so a couple of threads overlap usefully, but the gain flattens quickly and
#: every extra thread holds another tile's masks alive. Two spare cores are left
#: for the GUI thread's live merge and the map renders it serves, down to a
#: floor of one worker: on a machine with 3 cores or fewer, sparing 2 AND
#: still running a converter are not both possible, and the spare cores win.
_DEFAULT_MAX = 4
_SPARE_CORES = 2
_MIN_WORKERS = 1


def default_workers() -> int:
    """Converter thread count for this machine."""
    try:
        cores = os.cpu_count() or 2
    except Exception:  # noqa: BLE001 - never let sizing break a run
        cores = 2
    return max(_MIN_WORKERS, min(_DEFAULT_MAX, cores - _SPARE_CORES))


class TileConvertPool:
    """Runs one ``convert(job)`` call per finished tile on a small thread pool.

    ``submit`` never blocks. ``drain`` returns the conversions that have
    finished since the last call, each as ``(ok, job, payload)`` where payload
    is the conversion's return value when ``ok`` is True and the exception when
    it is False. A conversion that raises is reported, never swallowed and never
    retried: the tile is already billed, so the caller decides what to do with
    it exactly as it did when the conversion ran inline.

    Only the run loop calls ``submit``/``drain``/``close``, so ``pending`` needs
    no lock of its own.
    """

    def __init__(self, convert, workers: int = 0) -> None:
        self._convert = convert
        self._workers = int(workers) if workers > 0 else default_workers()
        self._pool = ThreadPoolExecutor(
            max_workers=self._workers, thread_name_prefix="tileconv")
        self._done: queue.Queue = queue.Queue()
        self._pending = 0
        self._closed = False

    @property
    def workers(self) -> int:
        return self._workers

    @property
    def pending(self) -> int:
        """Conversions submitted but not yet drained."""
        return self._pending

    def submit(self, job) -> None:
        """Queue one tile's conversion. Returns at once."""
        if self._closed:
            raise RuntimeError("TileConvertPool is closed")
        self._pending += 1
        self._pool.submit(self._run, job)

    def _run(self, job) -> None:
        try:
            self._done.put((True, job, self._convert(job)))
        except BaseException as exc:  # noqa: BLE001 - a bad tile must not kill the pool
            self._done.put((False, job, exc))

    def drain(self, timeout: float = 0.0) -> list:
        """Finished conversions, oldest first.

        ``timeout`` waits that long for the FIRST result when nothing has landed
        yet; anything already queued behind it is taken without waiting. Use it
        when the loop has nothing else to do, and leave it at 0 otherwise.
        """
        out: list = []
        first = timeout > 0.0
        while True:
            try:
                item = (self._done.get(timeout=timeout) if first
                        else self._done.get_nowait())
            except queue.Empty:
                break
            first = False
            self._pending -= 1
            out.append(item)
        return out

    def close(self, wait: bool = False) -> list:
        """Shut the pool down and return whatever had already finished.

        ``wait`` False cancels conversions that have not started yet: on a hard
        teardown the thread must become joinable at once. Results already in the
        queue are always returned.

        Those cancelled conversions ARE billed tiles whose geometry is lost, so
        the caller drains this pool before closing it and only reaches here on
        what its budget could not cover. Waiting here instead would not save
        them: the stop has a fixed window before the GUI gives up on the run.
        """
        if self._closed:
            return self.drain()
        self._closed = True
        try:
            self._pool.shutdown(wait=wait, cancel_futures=not wait)
        except TypeError:  # Python < 3.9 has no cancel_futures
            self._pool.shutdown(wait=wait)
        except Exception:  # noqa: BLE001 - teardown must never re-raise
            logger.warning("TileConvertPool: shutdown failed", exc_info=True)
        return self.drain()
