



















from __future__ import annotations

import logging
import os
import queue
import sys
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
    return max(_MIN_WORKERS, min(top, cores - spare, physical_worker_cap(cores)))


class TileConvertPool:













    def __init__(self, convert, workers: int = 0) -> None:
        self._convert = convert
        self._workers = int(workers) if workers > 0 else default_workers()
        from ..core.macos_activity import promote_current_thread



        self._pool = ThreadPoolExecutor(
            max_workers=self._workers, thread_name_prefix="tileconv",
            initializer=promote_current_thread)
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


_PHYSICAL_UNSET = object()
_physical_cache = _PHYSICAL_UNSET


def physical_cores() -> int | None:


    global _physical_cache
    if _physical_cache is not _PHYSICAL_UNSET:
        return _physical_cache
    try:
        value = _read_physical_cores()
    except Exception:  # noqa: BLE001  # nosec B110
        value = None
    _physical_cache = value if value and value > 0 else None
    return _physical_cache


def _read_physical_cores() -> int | None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes


        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        fn = kernel32.GetLogicalProcessorInformationEx
        fn.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        fn.restype = wintypes.BOOL
        size = wintypes.DWORD(0)
        fn(0, None, ctypes.byref(size))
        if size.value == 0:
            return None
        buf = ctypes.create_string_buffer(size.value)
        if not fn(0, buf, ctypes.byref(size)):
            return None
        raw = buf.raw[:size.value]
        count = 0
        offset = 0
        while offset + 8 <= len(raw):
            relation = int.from_bytes(raw[offset:offset + 4], "little")
            record = int.from_bytes(raw[offset + 4:offset + 8], "little")
            if record <= 0:
                break
            if relation == 0:
                count += 1
            offset += record
        return count or None
    if sys.platform == "darwin":
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib")
        out = ctypes.c_int(0)
        out_size = ctypes.c_size_t(ctypes.sizeof(out))
        if libc.sysctlbyname(b"hw.physicalcpu", ctypes.byref(out),
                             ctypes.byref(out_size), None, 0) == 0:
            return int(out.value) or None
        return None
    pairs = set()
    physical_id = None
    with open("/proc/cpuinfo", encoding="utf-8") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "physical id":
                physical_id = value.strip()
            elif key == "core id":
                pairs.add((physical_id, value.strip()))
    return len(pairs) or None


def physical_worker_cap(logical_cores: int) -> int:







    try:
        physical = physical_cores()
        if physical is None or physical >= logical_cores:
            return logical_cores
        from ..core.server_dials import dial_bool

        if not dial_bool("tuning.convert.use_physical_cores", True):
            return logical_cores
        return max(0, physical - 1)
    except Exception:  # noqa: BLE001
        return logical_cores


def available_memory_mb() -> int | None:






    try:
        if os.name == "nt":
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatus)]
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys // (1024 * 1024))
            return None
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return None
