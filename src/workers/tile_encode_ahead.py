


















from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor

logger = logging.getLogger(__name__)


class TileEncodeAhead:







    def __init__(self, collect_timed, encode) -> None:
        self._collect_timed = collect_timed
        self._encode = encode
        self._lock = threading.Lock()

        self._expected: dict[int, tuple] = {}
        self._jobs: dict[int, Future] = {}
        self._pool: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tileenc")

    def expect(self, seq: int, tile_idx: int, tw: int, th: int,
               ready=None) -> None:





        with self._lock:
            if self._pool is None:
                return
            self._expected[seq] = (tile_idx, tw, th)
        if ready is not None and ready(seq):
            self.render_landed(seq)

    def render_landed(self, seq: int) -> None:

        with self._lock:
            entry = self._expected.pop(seq, None)
            if entry is None or self._pool is None:
                return
            tile_idx, tw, th = entry
            try:
                self._jobs[tile_idx] = self._pool.submit(
                    self._collect_and_encode, seq, tw, th)
            except RuntimeError:
                self._expected[seq] = entry

    def _collect_and_encode(self, seq: int, tw: int, th: int) -> tuple:
        img, render_s = self._collect_timed(seq)
        encoded = None
        if img is not None and not img.isNull():
            try:
                encoded = self._encode(img, 0, 0, tw, th)
            except Exception:  # noqa: BLE001
                logger.debug("TileEncodeAhead: encode failed", exc_info=True)
        return img, render_s, encoded

    def handed_over(self, tile_idx: int) -> Future | None:

        with self._lock:
            return self._jobs.get(tile_idx)

    def claim(self, tile_idx: int, seq: int) -> Future | None:







        with self._lock:
            self._expected.pop(seq, None)
            return self._jobs.pop(tile_idx, None)

    def release(self, tile_idx: int, seq: int) -> bool:





        with self._lock:
            self._expected.pop(seq, None)
            job = self._jobs.pop(tile_idx, None)
        if job is None:
            return False

        return not job.cancel()

    def close(self) -> None:


        with self._lock:
            pool, self._pool = self._pool, None
            self._expected.clear()
            self._jobs.clear()
        if pool is None:
            return
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            logger.debug("TileEncodeAhead: shutdown failed", exc_info=True)
