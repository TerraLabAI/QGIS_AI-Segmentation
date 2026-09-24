






























from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .tile_convert_child import (  # noqa: F401
    STAT_FOLD,
    _child_stderr_file,
    _recv,
    _send,
    child_creation_flags,
    child_cwd,
    child_environment,
    child_main,
    child_python,
    keep_child_off_power_throttling,
    skip_unused_child_imports,
    unthrottle_this_process,
)
from .tile_convert_handshake import (  # noqa: F401
    _await_ready,
    _init_all,
    _silent_end,
)
from .tile_convert_lifecycle import TileConvertLifecycleMixin
from .tile_convert_threads import (  # noqa: F401
    _ANY_FAILURE,
    DEFAULT_MAX_WORKERS,
    SPARE_CORES,
    TileConvertPool,
    default_workers,
    usable_cores,
)

logger = logging.getLogger(__name__)



PROCESS_POOL_MIN_TILES = 24








PROCESS_MAX_WORKERS = 3



CHILD_READY_TIMEOUT_S = 45.0


def _child_ready_timeout_s() -> float:







    from ..core.server_dials import dial_in_range

    return dial_in_range(
        "tuning.convert.child_ready_timeout_s", CHILD_READY_TIMEOUT_S, 5.0, 180.0)


def process_workers(default_max: int | None = None,
                    spare_cores: int | None = None) -> int:















    cores = usable_cores()
    top = PROCESS_MAX_WORKERS if default_max is None else max(1, int(default_max))
    spare = SPARE_CORES if spare_cores is None else max(0, int(spare_cores))
    children = min(top, cores - spare)
    return children if children >= 2 else 0


class TileConvertProcessPool(TileConvertLifecycleMixin):





















    def __init__(self, convert, snapshot: dict | None, workers: int = 0,
                 ready_timeout: float | None = None) -> None:
        self._convert = convert
        self._snapshot = snapshot
        self._workers = int(workers) if workers > 0 else default_workers()
        self._ready_timeout = (
            float(ready_timeout) if ready_timeout is not None else _child_ready_timeout_s()
        )
        self._children: list = []
        self._readers: list = []






        self._outboxes: list = []
        self._feeders: list = []
        self._done: queue.Queue = queue.Queue()
        self._jobs: dict = {}
        self._next_key = 0
        self._round = 0
        self._pending = 0
        self._closed = False
        self._stats: dict = {}
        self.child_failures = 0



        self.last_failure = ""



        self._stderr_files: list = []
        self._spawn_tried = False
        self._spawned_at = 0.0




        self._owners: dict = {}
        self._dead_children: set = set()
        self._lock = threading.Lock()




        self._boot_state = ""
        self._boot_taken = False
        self._boot_s = 0.0
        self._bridged_at = 0.0
        self._live = False
        self._local_threads = 0
        self._local_busy = 0
        self._local_queue: collections.deque = collections.deque()
        self._local_pool: ThreadPoolExecutor | None = None
        self.local_converted = 0



    @property
    def workers(self) -> int:


        if self._boot_state and not self._live:
            return self._local_threads
        return len(self._children) or self._workers

    @property
    def pending(self) -> int:
        return max(0, self._pending)

    def submit(self, job) -> None:








        pump = False
        with self._lock:
            if self._closed:
                raise RuntimeError("TileConvertProcessPool is closed")
            key = self._next_key
            self._next_key += 1
            self._jobs[key] = job
            self._pending += 1
            if (not self._boot_state or self._live) and self._hand_to_child(key, job):
                return
            self._owners[key] = None
            local = self._local_pool
            if local is not None:
                self._local_queue.append(key)
                pump = self._local_busy < self._local_threads
                if pump:
                    self._local_busy += 1
        if local is None:
            self._convert_here(key, job)
        elif pump:
            try:
                local.submit(self._local_pump)
            except RuntimeError:
                with self._lock:
                    self._local_busy -= 1

    def _hand_to_child(self, key, job) -> bool:






        live = [i for i, proc in enumerate(self._children)
                if proc not in self._dead_children]
        if not live:
            return False
        n = len(self._children)
        start = self._round % n
        self._round += 1
        slot = min(live, key=lambda i: (
            self._outboxes[i].qsize(), (i - start) % n))
        self._owners[key] = self._children[slot]
        self._outboxes[slot].put((key, job))
        return True

    def _local_pump(self) -> None:









        while True:
            with self._lock:
                if self._closed or not self._local_queue:
                    self._local_busy -= 1
                    return
                key = self._local_queue.popleft()
                job = self._jobs.get(key)
            if job is None:
                continue
            try:
                self._convert_here(key, job)
            except Exception:  # noqa: BLE001
                logger.warning("TileConvertProcessPool: a local conversion "
                               "could not be published", exc_info=True)
            with self._lock:
                self.local_converted += 1

    def _feed(self, proc: Any, outbox: queue.Queue) -> None:



        try:
            while True:
                item = outbox.get()
                if item is None:
                    return
                key, job = item
                with self._lock:
                    if self._closed:
                        return
                    if self._owners.get(key) is not proc:
                        continue
                try:



                    if proc is None or proc.stdin is None:
                        raise BrokenPipeError("converter child has no input pipe")
                    _send(proc.stdin, ("job", (key, job)))
                except Exception:  # noqa: BLE001
                    self._child_failed(proc)
                    return
        finally:


            try:
                proc.stdin.close()
            except Exception:  # noqa: BLE001  # nosec B110
                pass

    def _child_failed(self, proc) -> None:




        with self._lock:
            if proc in self._dead_children:
                return
            self._dead_children.add(proc)
            if self._closed:
                return
            self.child_failures += 1
            jobs = [(key, self._jobs[key]) for key, owner in self._owners.items()
                    if owner is proc]
            for key, _job in jobs:
                self._owners[key] = None
        try:
            proc.kill()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        if jobs:
            logger.warning("TileConvertProcessPool: a converter child exited "
                           "with %d tile(s) unanswered; converting them here",
                           len(jobs))
        for key, job in jobs:
            self._convert_here(key, job)

    def _complete(self, owner, ok, key, payload, stats=None) -> None:

        with self._lock:
            if key not in self._owners or self._owners[key] is not owner:
                return
            self._owners.pop(key)
            self._done.put((ok, key, payload, stats))

    def _convert_here(self, key, job) -> None:


        with self._lock:
            if self._closed or key not in self._owners:
                return
        try:
            payload = self._convert(job)
        except _ANY_FAILURE as exc:
            self._complete(None, False, key, exc)
        else:
            self._complete(None, True, key, payload)

    def _read_from(self, proc) -> None:


        while True:
            try:
                frame = _recv(proc.stdout)
            except Exception:  # noqa: BLE001
                frame = None
            if frame is None:
                self._child_failed(proc)
                return
            kind, payload = frame
            if kind == "done":
                key, dets, stats, extra, _elapsed = payload
                self._complete(proc, True, key, dets, (stats, extra))
            elif kind == "fail":
                key, text = payload
                self._complete(proc, False, key, RuntimeError(text))

    def drain(self, timeout: float = 0.0) -> list:

        out: list = []
        first = timeout > 0.0
        while True:
            try:
                item = (self._done.get(timeout=timeout) if first
                        else self._done.get_nowait())
            except queue.Empty:
                break
            first = False
            ok, key, payload, stats = item
            with self._lock:
                if key not in self._jobs:
                    continue
                job = self._jobs.pop(key)
                self._pending -= 1
            if stats is not None:
                self._absorb(*stats)
            out.append((ok, job, payload))
        return out

    def _absorb(self, stats: dict, extra: list) -> None:

        if extra:
            logger.warning(
                "TileConvertProcessPool: converter accumulator(s) %s are not "
                "in the fold table and are being dropped", ", ".join(extra))
        for name, how in STAT_FOLD.items():
            value = stats.get(name)
            if value is None:
                continue
            if how == "extend":
                self._stats.setdefault(name, []).extend(value)
            elif how == "max":
                self._stats[name] = max(self._stats.get(name, 0.0), value)
            else:
                self._stats[name] = self._stats.get(name, 0) + value

    def fold_stats(self, worker) -> None:


        for name, how in STAT_FOLD.items():
            if name not in self._stats:
                continue
            if how == "extend":
                getattr(worker, name).extend(self._stats[name])
            elif how == "max":
                setattr(worker, name,
                        max(getattr(worker, name, 0.0), self._stats[name]))
            else:
                setattr(worker, name,
                        getattr(worker, name, 0) + self._stats[name])
        self._stats = {}

    def close(self, wait: bool = False) -> list:







        with self._lock:
            already_closed = self._closed
            self._closed = True
            booting = self._boot_state == "booting"
            if booting:
                self._boot_state = "abandoned"
                self._boot_s = time.monotonic() - self._bridged_at
            self._local_queue.clear()
        if already_closed:
            return self.drain()
        if self._local_pool is not None:
            try:
                self._local_pool.shutdown(wait=False)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        for outbox in self._outboxes:


            with outbox.mutex:
                outbox.queue.clear()
            outbox.put(None)
        if not wait or booting:





            for proc in list(self._children):
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001  # nosec B110
                    pass
        else:
            for proc in self._children:
                try:
                    proc.wait(timeout=5)
                except Exception:  # noqa: BLE001  # nosec B110
                    pass
        results = self.drain()
        if not booting:
            self._kill_all()
        with self._lock:
            self._pending = 0
            self._jobs.clear()
            self._owners.clear()
        return results
