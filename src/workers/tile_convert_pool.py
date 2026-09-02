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

#: Converter threads when the caller asks for the default. Only part of the
#: conversion drops the GIL: the GEOS work does, the polygonize does not, and
#: neither does the Python that builds a geometry from its rings. So the gain
#: peaks early and then REVERSES, because past the peak the threads spend more
#: time handing the lock to each other than they save, and they take it from
#: the stitch thread and the GUI thread, which are on the same lock and are
#: what the user is waiting for. Measured on a run's own tiles with the
#: stitcher folding alongside, not on a converter pool by itself: a pool sized
#: for the second reading is the one that finishes the run sooner. This is the
#: client fallback; the served dial moves it without a release. Two spare
#: cores are left for the GUI thread's live merge and the map renders it
#: serves, down to a floor of one worker: on a machine with 3 cores or fewer,
#: sparing 2 AND still running a converter are not both possible, and the
#: spare cores win.
DEFAULT_MAX_WORKERS = 3
SPARE_CORES = 2
_MIN_WORKERS = 1


def default_workers(default_max: int | None = None,
                    spare_cores: int | None = None) -> int:
    """Converter thread count for this machine: at most ``default_max``,
    leaving ``spare_cores`` free, never under one. The caller may pass served
    values; None means the module's own."""
    try:
        cores = os.cpu_count() or 2
    except Exception:  # noqa: BLE001 - never let sizing break a run
        cores = 2
    top = DEFAULT_MAX_WORKERS if default_max is None else max(1, int(default_max))
    spare = SPARE_CORES if spare_cores is None else max(0, int(spare_cores))
    return max(_MIN_WORKERS, min(top, cores - spare))


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
        """Conversions submitted but not yet drained.

        Never negative: a conversion that lands after a cancelling close has
        already been accounted for, and the run loop reads this as a count.
        """
        return max(0, self._pending)

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
        results = self.drain()
        # A future cancelled by the shutdown never reaches _run, so its submit
        # increment would never be undone and pending would stay above zero for
        # the rest of the session. Nothing can be submitted after a close, so
        # what the final drain did not return is simply gone.
        self._pending = 0
        return results


# --------------------------------------------------------------------------
# Converter PROCESSES
# --------------------------------------------------------------------------
#
# Threads cannot take this path past about one and a half cores, whatever the
# pool size, because the polygonize holds the interpreter lock for its whole
# duration and the Python that builds a geometry from its rings holds it too.
# Separate interpreters have no shared lock, and the seam is unusually clean:
# a tile goes in as its own reply payload and comes back as WKB, both of which
# pickle, and the conversion reads nothing else that a run does not fix before
# its first tile.
#
# Everything here is best-effort. A child that cannot start, cannot answer, or
# dies mid-run falls the run back onto the thread pool above, which is exactly
# what shipped before, so the worst case is a lost handshake and today's speed.

#: Tiles under which a run stays on threads: the children have to import QGIS
#: before they can convert anything, and a short run finishes inside that.
PROCESS_POOL_MIN_TILES = 24

#: Most converter PROCESSES. Each child is a second interpreter with QGIS,
#: numpy and the geometry stack loaded: about 180 MB resident idle and 195 MB
#: mid-tile, and it holds a core for the length of the run. Past three the
#: run gets no shorter end to end, because the children take the cores the
#: stitch thread and the finalize need next: on 814 tiles the review opened
#: after 379 s with 3 children, 416 s with 4 and 465 s with 6. Three is the
#: cap; the served convert.default_max may lower it, never usefully raise it.
#: The spare-core rule in process_workers keeps a 4-core machine at 2 and a
#: 2- or 3-core machine on threads.
PROCESS_MAX_WORKERS = 3

#: How long a child may take to report that it has imported QGIS and the
#: plugin and is ready for work.
CHILD_READY_TIMEOUT_S = 45.0

#: Run-wide accumulators `_detections_to_geoms` writes under the worker's stat
#: lock, and how the parent folds a child's answer back in. A new accumulator
#: that is missing here comes back to the parent as a warning, never silently:
#: see `_unfolded_stats`.
STAT_FOLD = {
    "raw_detections_total": "sum",
    "masks_dropped_whole_tile": "sum",
    "masks_whole_tile_armed": "sum",
    "masks_dropped_hard_cover": "sum",
    "masks_dropped_tile_span": "sum",
    "masks_dropped_not_compact": "sum",
    "masks_whole_tile_kept_map": "sum",
    "masks_dropped_map_lowscore": "sum",
    "phase_convert_s": "sum",
    "map_cover_scores": "extend",
    "observed_mask_gsd": "max",
}


def usable_cores() -> int:
    """The cores THIS process may run on, not the cores the machine has.

    A QGIS pinned to four cores of a sixteen-core box, or one inside a
    container, reports the full count from ``os.cpu_count`` and would size its
    children for cores it cannot use. Every reading the platform offers is
    taken and the smallest wins, because each one can miss a restriction the
    others see (on Windows the affinity mask is one kernel call away and
    ``os.process_cpu_count`` does not read it).
    """
    readings = []
    try:
        count = getattr(os, "process_cpu_count", None)
        if count is not None and count():
            readings.append(int(count()))
    except Exception:  # noqa: BLE001 - a missing reading is not an error  # nosec B110
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
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            # Declared, or the pseudo-handle is truncated to 32 bits and
            # the call fails quietly.
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
    except Exception:  # noqa: BLE001 - never let sizing break a run
        readings.append(2)
    return max(1, min(r for r in readings if r > 0))


def process_workers(default_max: int | None = None,
                    spare_cores: int | None = None) -> int:
    """Converter PROCESS count for this machine, or 0 for none.

    Sized separately from the thread pool and larger, because the reason to cap
    threads (they contend for one interpreter lock, and past the peak they take
    it from the threads the user is waiting on) does not apply to a child with
    its own interpreter.

    The spare cores are the GUI thread's and the stitch thread's: each child
    saturates a core for the length of a run, and a machine with no core left
    for the thread the user is looking at freezes the canvas. So a four-core
    machine gets at most two children, and a machine that could only run one
    gets none: one child, plus the pipes and the pickling that feed it, does
    not beat one thread, and it costs a second QGIS in memory. The caller
    reads 0 as "stay on the thread pool".
    """
    cores = usable_cores()
    top = PROCESS_MAX_WORKERS if default_max is None else max(1, int(default_max))
    spare = SPARE_CORES if spare_cores is None else max(0, int(spare_cores))
    children = min(top, cores - spare)
    return children if children >= 2 else 0


def _send(stream, obj) -> None:
    """One length-prefixed pickle frame. The only thing either side writes."""
    import pickle  # nosec B403
    import struct

    payload = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack("<I", len(payload)))
    stream.write(payload)
    stream.flush()


def _recv(stream):
    """The next frame, or None once the other side closes."""
    import pickle  # nosec B403
    import struct

    head = stream.read(4)
    if not head or len(head) < 4:
        return None
    size = struct.unpack("<I", head)[0]
    body = stream.read(size)
    if body is None or len(body) < size:
        return None
    return pickle.loads(body)  # nosec B301 - both ends are this module


def _fresh(zero):
    """A private copy of an accumulator's zero, so no two jobs share a list."""
    return list(zero) if isinstance(zero, list) else zero


def _unfolded_stats(before: dict, after: dict) -> list:
    """Counters a job moved that STAT_FOLD does not name.

    The fold table is written by hand and the accumulators are not, so this is
    what stops a new one from being dropped in silence: the child reports it,
    the parent logs it, and the number is visibly wrong instead of quietly so.

    Numbers only, and bool is not one: they are the only attributes that can be
    compared for a change after the fact, since a container the conversion
    appended to is the same object it was before and reads as unchanged.
    """
    out = []
    for name, value in after.items():
        if name in STAT_FOLD or not isinstance(value, (int, float)):
            continue
        if isinstance(value, bool):
            continue
        if name in before and before[name] != value:
            out.append(name)
    return out


def child_main() -> None:
    """A converter process: read jobs off stdin, write conversions to stdout.

    Runs the worker's OWN conversion, on a stand-in built from the snapshot the
    parent sent, so there is one implementation of the geometry pipeline and a
    child cannot drift from the thread path.
    """
    import sys
    import threading
    import time

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    # A library that prints would land in the middle of a frame. Nothing may
    # reach the real stdout from here on.
    sys.stdout = sys.stderr

    try:
        from qgis.core import QgsApplication

        app = QgsApplication([], False)
        app.initQgis()
        from .auto_detection_worker import AutoDetectionWorker
    except BaseException as exc:  # noqa: BLE001 - the parent decides what to do
        try:
            _send(stdout, ("no", repr(exc)))
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return

    worker = None
    stats_reset: dict = {}
    baseline: dict = {}
    _send(stdout, ("ready", None))

    while True:
        frame = _recv(stdin)
        if frame is None:
            break
        kind, payload = frame
        if kind == "init":
            worker = AutoDetectionWorker.__new__(AutoDetectionWorker)
            worker.__dict__.update(payload)
            # Rebuilt rather than sent: a lock and a thread-local are what the
            # parent could not pickle, and the prepared clip engine is bound to
            # the geometry instance it was built from, so it has to be built on
            # the thread that uses it (see _clip_for_thread).
            worker._stat_lock = threading.Lock()
            worker._clip_local = threading.local()
            # Zeroed, because what travels back is one tile's SHARE. The
            # parent holds the run-wide totals and folds each answer in.
            for name, how in STAT_FOLD.items():
                if how == "extend":
                    stats_reset[name] = []
                elif isinstance(getattr(worker, name, 0), int):
                    stats_reset[name] = 0
                else:
                    stats_reset[name] = 0.0
                setattr(worker, name, _fresh(stats_reset[name]))
            # The baseline the per-job check compares against: every numeric
            # attribute as it stands before any tile is converted.
            baseline = {k: v for k, v in worker.__dict__.items()
                        if isinstance(v, (int, float))
                        and not isinstance(v, bool)}
            _send(stdout, ("init_ok", None))
            continue
        if kind != "job":
            break
        key, job = payload
        try:
            t0 = time.monotonic()
            dets = worker._convert_completed(job)
            stats = {n: getattr(worker, n) for n in STAT_FOLD}
            extra = _unfolded_stats(baseline, worker.__dict__)
            # Reset for the next job so what travels is this tile's own share.
            for name, value in stats_reset.items():
                setattr(worker, name, _fresh(value))
            _send(stdout, ("done", (key, dets, stats, extra,
                                    time.monotonic() - t0)))
        except BaseException as exc:  # noqa: BLE001 - one bad tile, not the pool
            _send(stdout, ("fail", (key, repr(exc))))


def child_python() -> str | None:
    """The interpreter a converter child runs, or None when there is none.

    QGIS embeds Python, so ``sys.executable`` is the QGIS binary on the two
    platforms that ship it that way, and launching THAT would start a second
    QGIS. What is wanted is the interpreter behind ``sys.prefix``, which is the
    same build, with the same qgis bindings and the same GEOS, so a child can
    never disagree with the parent about a geometry.

    On a macOS bundle that interpreter is not under the prefix at all: the
    prefix holds the frameworks and has no ``bin`` directory, and the only
    python binary sits beside the QGIS binary. Measured on 3.44: prefix
    Contents/Frameworks, interpreter Contents/MacOS/python3.12. So the
    directory of ``sys.executable`` is searched too.

    A guess that is wrong costs nothing: the child has to answer a handshake
    before it is given any work.
    """
    import sys

    major, minor = sys.version_info[:2]
    beside_qgis = os.path.dirname(sys.executable or "") or "."
    names = (
        os.path.join(sys.prefix, "python.exe"),
        os.path.join(sys.prefix, "bin", f"python{major}.{minor}"),
        os.path.join(sys.prefix, "bin", "python3"),
        os.path.join(sys.prefix, "bin", "python"),
        os.path.join(beside_qgis, f"python{major}.{minor}"),
        os.path.join(beside_qgis, "python3"),
        os.path.join(beside_qgis, f"python{major}.{minor}.exe"),
    )
    for path in names:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    exe = sys.executable or ""
    if exe and os.path.basename(exe).lower().startswith("python"):
        return exe
    return None


def _stdlib_home() -> str | None:
    """The prefix a child can find ``encodings`` under, or None.

    None means the interpreter is expected to find its own, which is what a
    distribution python does. A wrong home is worse than none: it aborts the
    child before its first line, so this returns a directory only when the
    standard library is really there.
    """
    import sys
    import sysconfig

    home = sys.base_prefix or sys.prefix
    if not home:
        return None
    candidates = []
    try:
        stdlib = sysconfig.get_paths().get("stdlib")
    except Exception:  # noqa: BLE001 - a broken sysconfig is not fatal here
        stdlib = None
    if stdlib:
        candidates.append(stdlib)
    major, minor = sys.version_info[:2]
    candidates.append(os.path.join(home, "Lib"))
    candidates.append(os.path.join(home, "lib", f"python{major}.{minor}"))
    root = os.path.normcase(os.path.abspath(home))
    for path in candidates:
        # Only a standard library that lives UNDER the home proves the home.
        # sysconfig answers for the running interpreter, which in a virtual
        # environment points outside it.
        inside = os.path.normcase(os.path.abspath(path)).startswith(root)
        if inside and os.path.isdir(os.path.join(path, "encodings")):
            return home
    return None


def child_environment() -> dict:
    """The environment a child needs to import qgis.core and the plugin.

    Inherited from this process, which is already a working QGIS, plus the two
    directories the child imports from, plus PYTHONHOME.

    PYTHONHOME is the one the parent never needs. The QGIS binary sets the
    home for its own embedded interpreter at startup, and a child inherits
    nothing of that. On a macOS bundle the python binary carries the build
    machine's prefix compiled in, a directory that exists on no user machine,
    so a child launched without a home looks for its standard library there,
    fails to find ``encodings`` and dies before running a line.

    The home is ``sys.base_prefix``, not ``sys.prefix``: inside a virtual
    environment they differ, and a venv holds no standard library, so pointing
    a child at one breaks it the same way. It is set only once the standard
    library is found under it, which is what a distribution python on Linux
    already gets right on its own. When nothing is found the variable is left
    alone, and the child starts exactly as it did before.
    """
    import sys
    import sysconfig

    env = dict(os.environ)
    home = _stdlib_home()
    if home:
        env["PYTHONHOME"] = home
    del sysconfig
    plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    parts = [plugin_dir]
    try:
        from qgis.core import QgsApplication

        qgis_python = os.path.join(QgsApplication.prefixPath(), "python")
        if os.path.isdir(qgis_python):
            parts.append(qgis_python)
    except Exception:  # noqa: BLE001 - the handshake is the real check  # nosec B110
        pass
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONNOUSERSITE"] = "1"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env.pop("PYTHONSTARTUP", None)
    # Every path this module logs is a directory, never a URL or a key.
    logger.debug("TileConvertProcessPool: child sys.path head %s", parts[0])
    del sys
    return env


class TileConvertProcessPool:
    """Runs the worker's own conversion on separate interpreters.

    Same three calls as ``TileConvertPool`` (``submit``, ``drain``, ``close``)
    and the same ``(ok, job, payload)`` answers, so the run loop cannot tell
    which pool it is holding. Two differences it does have to know about:

    - ``start`` may refuse. No interpreter, a child that will not import QGIS,
      a machine that will not spawn: all of them return False, and the caller
      keeps the thread pool it already knows how to use.
    - a job whose child dies is converted here instead, on this thread, so a
      billed tile is never lost to a broken child.

    ``fold_stats(worker)`` puts the run-wide accumulators back on the worker.
    They are the one thing that cannot cross the boundary on its own, because
    the conversion writes them on the object it runs on and the children each
    have their own copy of it.
    """

    def __init__(self, convert, snapshot: dict, workers: int = 0,
                 ready_timeout: float = CHILD_READY_TIMEOUT_S) -> None:
        self._convert = convert
        self._snapshot = snapshot
        self._workers = int(workers) if workers > 0 else default_workers()
        self._ready_timeout = float(ready_timeout)
        self._children: list = []
        self._readers: list = []
        # One outbox and one writer thread per child. A job is several
        # megabytes of reply and the pipe holds a few kilobytes, so a write
        # straight from submit() blocked the run loop until the child came
        # back from its current tile to read, and every socket in the window
        # went unread for that long. The writers do nothing but move frames,
        # and sleep in the OS write the rest of the time.
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
        # Why start() said no, in words the run's log and its telemetry can
        # carry: which step refused and, when the children spoke before they
        # died, the tail of what they said. Empty after a start that worked.
        self.last_failure = ""
        # One scratch file per child for its stderr. A pipe nobody reads
        # deadlocks the child on its first full buffer; a file never does,
        # and its tail is the one clue a child that died leaves behind.
        self._stderr_files: list = []

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        """Spawn the children and wait for every one to say it is ready.

        All or nothing: a pool running fewer children than it was asked for
        would silently cut a run's throughput, and the caller has a working
        thread pool to fall back to.
        """
        import subprocess  # nosec B404 - a fixed argv, never a shell
        import threading

        exe = child_python()
        if not exe:
            logger.info("TileConvertProcessPool: no child interpreter found")
            self.last_failure = "no child interpreter found"
            return False
        env = child_environment()
        boot = (
            "from src.workers.tile_convert_pool import child_main; child_main()"
        )
        import tempfile

        try:
            for _ in range(self._workers):
                errfile = tempfile.TemporaryFile()
                self._stderr_files.append(errfile)
                # -s, never -I: isolated mode also drops PYTHONPATH, which is
                # the one thing the child is given (see child_environment).
                proc = subprocess.Popen(  # nosec B603 - list argv, no shell
                    [exe, "-s", "-c", boot],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=errfile, env=env, close_fds=True,
                    # No console window per child on Windows; 0 elsewhere.
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self._children.append(proc)
        except Exception as exc:  # noqa: BLE001 - a machine that will not spawn
            logger.info("TileConvertProcessPool: could not start a child",
                        exc_info=True)
            self._fail(f"spawn error {type(exc).__name__}")
            return False

        answers = [_await_ready(proc, self._ready_timeout)
                   for proc in self._children]
        ready = [ok for ok, _why in answers]
        if not all(ready):
            came_up = sum(1 for r in ready if r)
            logger.info("TileConvertProcessPool: %d of %d children came up",
                        came_up, len(self._children))
            whys = [why for ok, why in answers if not ok and why]
            detail = f"{came_up} of {len(self._children)} came up"
            if whys:
                detail = f"{detail}; child said: {whys[0]}"
            self._fail(detail)
            return False

        for proc in self._children:
            _send(proc.stdin, ("init", self._snapshot))
            frame = _recv(proc.stdout)
            if not frame or frame[0] != "init_ok":
                logger.info("TileConvertProcessPool: a child refused the run")
                self._fail("a child refused the run")
                return False
        for proc in self._children:
            reader = threading.Thread(
                target=self._read_from, args=(proc,), daemon=True,
                name="tileconvproc")
            reader.start()
            self._readers.append(reader)
            outbox: queue.Queue = queue.Queue()
            feeder = threading.Thread(
                target=self._feed, args=(proc, outbox), daemon=True,
                name="tileconvfeed")
            feeder.start()
            self._outboxes.append(outbox)
            self._feeders.append(feeder)
        logger.info("TileConvertProcessPool: %d converter process(es) ready",
                    len(self._children))
        return True

    def _fail(self, detail: str) -> None:
        """Record why the pool is not starting, with the children's last
        words when they left any, then put the children down."""
        tail = self._stderr_tail()
        self.last_failure = f"{detail}; stderr: {tail}" if tail else detail
        self._kill_all()

    def _stderr_tail(self, limit: int = 2048) -> str:
        """The last ``limit`` bytes any child wrote to stderr, one line, or
        "" when they were silent. Read before the files are closed."""
        pieces = []
        for errfile in self._stderr_files:
            try:
                errfile.flush()
                size = errfile.seek(0, os.SEEK_END)
                errfile.seek(max(0, size - limit))
                text = errfile.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001 - a clue, never a requirement  # nosec B112
                continue
            text = " | ".join(part.strip() for part in text.splitlines() if part.strip())
            if text:
                pieces.append(text)
        return " || ".join(pieces)[-limit:]

    def _kill_all(self) -> None:
        for proc in self._children:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001 - teardown never raises  # nosec B110
                pass
        self._children = []
        for errfile in self._stderr_files:
            try:
                errfile.close()
            except Exception:  # noqa: BLE001 - teardown never raises  # nosec B110
                pass
        self._stderr_files = []

    # -- the run loop's three calls ---------------------------------------

    @property
    def workers(self) -> int:
        return len(self._children) or self._workers

    @property
    def pending(self) -> int:
        return max(0, self._pending)

    def submit(self, job) -> None:
        """Hand one tile to the next child, round robin. Never blocks.

        A child that has died takes its tile with it, so the tile is converted
        here instead: it is billed, and losing it to a broken pipe would be the
        one failure this pool must not add.
        """
        if self._closed:
            raise RuntimeError("TileConvertProcessPool is closed")
        key = self._next_key
        self._next_key += 1
        self._jobs[key] = job
        self._pending += 1
        if not self._children:
            self._convert_here(key, job)
            return
        # The child with the least waiting for it, ties broken round robin,
        # so a child on a dense tile does not pile up work while another sits
        # idle. Queued only: the writer thread does the blocking part.
        n = len(self._children)
        start = self._round % n
        self._round += 1
        slot = min(range(n),
                   key=lambda i: (self._outboxes[i].qsize(), (i - start) % n))
        self._outboxes[slot].put((key, job))

    def _feed(self, proc, outbox: queue.Queue) -> None:
        """One child's writer: frames off its outbox and into its pipe, until
        the pool closes. A pipe that breaks converts the job here instead,
        which is the fallback submit() used to take inline."""
        while True:
            item = outbox.get()
            if item is None:
                return
            key, job = item
            try:
                _send(proc.stdin, ("job", (key, job)))
            except Exception:  # noqa: BLE001 - a dead pipe is a dead child
                self.child_failures += 1
                self._convert_here(key, job)

    def _convert_here(self, key, job) -> None:
        """The fallback conversion, on the calling thread. Same code, same
        answer, just without the second interpreter."""
        try:
            self._done.put((True, key, self._convert(job), None))
        except BaseException as exc:  # noqa: BLE001 - one tile, not the run
            self._done.put((False, key, exc, None))

    def _read_from(self, proc) -> None:
        """One child's answers, until it closes. Runs on its own thread and
        does nothing but move frames: the parsing is the pickle."""
        while True:
            try:
                frame = _recv(proc.stdout)
            except Exception:  # noqa: BLE001 - a child killed mid-frame
                frame = None
            if frame is None:
                return
            kind, payload = frame
            if kind == "done":
                key, dets, stats, extra, _elapsed = payload
                self._done.put((True, key, dets, (stats, extra)))
            elif kind == "fail":
                key, text = payload
                self._done.put((False, key, RuntimeError(text), None))

    def drain(self, timeout: float = 0.0) -> list:
        """Finished conversions, oldest first, as ``(ok, job, payload)``."""
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
            job = self._jobs.pop(key, None)
            self._pending -= 1
            if stats is not None:
                self._absorb(*stats)
            out.append((ok, job, payload))
        return out

    def _absorb(self, stats: dict, extra: list) -> None:
        """Add one child answer's share to the run-wide accumulators."""
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
        """Put what the children counted back on the worker. Called once the
        pool is drained, on the thread that owns the worker."""
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
        """Stop the children and return whatever had already landed."""
        if self._closed:
            return self.drain()
        self._closed = True
        for outbox in self._outboxes:
            outbox.put(None)
        for proc in self._children:
            try:
                proc.stdin.close()
            except Exception:  # noqa: BLE001 - teardown never raises  # nosec B110
                pass
        if wait:
            for proc in self._children:
                try:
                    proc.wait(timeout=5)
                except Exception:  # noqa: BLE001 - teardown never raises  # nosec B110
                    pass
        results = self.drain()
        self._kill_all()
        self._pending = 0
        self._jobs.clear()
        return results


def _await_ready(proc, timeout: float) -> tuple[bool, str]:
    """``(ready, why)``: True once a child says it has QGIS and the plugin
    loaded; otherwise False and the reason in words when the child gave one
    ("" for a silent death or a timeout).

    The read itself is what waits, on a helper thread, because a pipe read has
    no timeout of its own and a child that hangs on an import must not hang
    the run that is waiting to start.
    """
    import threading

    answer: list = []

    def read() -> None:
        try:
            answer.append(_recv(proc.stdout))
        except Exception:  # noqa: BLE001 - a child that died mid-handshake
            answer.append(None)

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    thread.join(timeout)
    if not answer:
        return False, f"no answer within {timeout:.0f}s"
    frame = answer[0]
    if not frame:
        return False, "died before answering"
    if frame[0] == "no":
        logger.info("TileConvertProcessPool: child could not start (%s)",
                    frame[1])
        return False, str(frame[1])[:300]
    return frame[0] == "ready", ""
