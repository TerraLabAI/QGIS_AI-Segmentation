



















from __future__ import annotations

import logging
import os
import queue
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)















DEFAULT_MAX_WORKERS = 3
SPARE_CORES = 2
_MIN_WORKERS = 1





_ANY_FAILURE = (Exception, GeneratorExit, KeyboardInterrupt, SystemExit)


def default_workers(default_max: int | None = None,
                    spare_cores: int | None = None) -> int:



    try:


        cores = usable_cores() if os.name == "nt" else (os.cpu_count() or 2)
    except Exception:  # noqa: BLE001
        cores = 2
    top = DEFAULT_MAX_WORKERS if default_max is None else max(1, int(default_max))
    spare = SPARE_CORES if spare_cores is None else max(0, int(spare_cores))
    return max(_MIN_WORKERS, min(top, cores - spare))


class TileConvertPool:













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





        return max(0, self._pending)

    def submit(self, job) -> None:

        if self._closed:
            raise RuntimeError("TileConvertPool is closed")
        self._pool.submit(self._run, job)
        self._pending += 1

    def _run(self, job) -> None:
        try:
            self._done.put((True, job, self._convert(job)))
        except _ANY_FAILURE as exc:
            self._done.put((False, job, exc))

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
            self._pending -= 1
            out.append(item)
        return out

    def close(self, wait: bool = False) -> list:











        if self._closed:
            return self.drain()
        self._closed = True
        try:
            self._pool.shutdown(wait=wait, cancel_futures=not wait)
        except TypeError:
            self._pool.shutdown(wait=wait)
        except Exception:  # noqa: BLE001
            logger.warning("TileConvertPool: shutdown failed", exc_info=True)
        results = self.drain()




        self._pending = 0
        return results


def usable_cores() -> int:









    readings = []
    try:
        read_count = getattr(os, "process_cpu_count", None)
        count = read_count() if read_count is not None else None
        if count:
            readings.append(int(count))
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    try:
        affinity = getattr(os, "sched_getaffinity", None)
        if affinity is not None:
            readings.append(len(affinity(0)))
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    if os.name == "nt":
        try:
            import ctypes

            process_mask = ctypes.c_size_t()
            system_mask = ctypes.c_size_t()





            kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]


            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.GetProcessAffinityMask.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t)]
            if kernel32.GetProcessAffinityMask(
                    kernel32.GetCurrentProcess(),
                    ctypes.byref(process_mask), ctypes.byref(system_mask)):
                readings.append(bin(process_mask.value).count("1"))
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    try:
        readings.append(os.cpu_count() or 2)
    except Exception:  # noqa: BLE001
        readings.append(2)
    return max(1, min(r for r in readings if r > 0))
