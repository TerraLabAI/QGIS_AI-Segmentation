"""The review's shape refine on child interpreters.

``workers.review_refine_thread`` took the Shapes step's refine off the GUI
thread, and that was the right first move: the map kept drawing. It could not
make the pass shorter, because the work is fine-grained Python over vertices
(the footprint squaring walks every edge in the interpreter), and Python
threads share one lock. Measured on a real run's visible set: one thread and
six threads finish in about the same time, and the six starve the GUI thread
that is waiting on them.

Child interpreters do not share the lock. This pool runs the SAME refine, on
the same inputs, on several of them, and hands the answers back through the
same calls the thread offers (``submit``, ``take_results``, ``abort``,
``join_run``, ``isRunning``), so the review cannot tell which one it is
holding. Geometry crosses as WKB in both directions; the refiner crosses as the
plain values it was built from, and each child rebuilds and memoises it.

Sized like the converter pool, by the same helper, and skipped on a small
machine or a small set (``pool_children``): a child is a whole core for as
long as the pass runs, and a laptop has none to spare.

Two things the caller has to know:

- ``start`` may refuse. No interpreter, a machine that will not spawn, a child
  that will not import QGIS: it returns False, and the caller keeps the thread
  it already knows how to use. The children boot in the background, so a
  start that will succeed returns at once and the first jobs wait in the
  queue until a child is up.
- a child that dies takes nothing with it. The jobs it held go back on the
  shared queue for the others; when the last child is gone ``isRunning`` turns
  False, and the review shapes what it is still waiting for itself, which is
  the path it already runs when the thread is gone.

No Qt signals: the review polls it from the cooperative pump, exactly as it
polls the thread, so nothing queued can reach a plugin that has been torn
down. The interpreter, the environment and the frame format come from
``workers.tile_convert_pool``, which proved them on the converter.
"""
from __future__ import annotations

import logging
import os
import queue
import threading

logger = logging.getLogger(__name__)

#: Machines this small keep the thread. A child interpreter is a whole core
#: for as long as the pass runs, and on four cores the GUI thread, the map
#: renderer and the stitcher are already sharing them.
MIN_CORES_FOR_POOL = 5

#: Under this many objects the children boot for longer than they save. The
#: client fallback for the served floor; the measured break-even is in
#: docs/debug, not here.
DEFAULT_MIN_OBJECTS = 1500

#: How long a teardown waits for the children to leave. They are killed, not
#: asked, so this only ever covers the operating system reaping them.
REVIEW_POOL_JOIN_TIMEOUT_MS = 3000

# Review params that change no shape. Left out of the spec so a child keeps
# one refiner across a confidence or size move.
_FILTER_ONLY_KEYS = frozenset({"conf", "min_a", "max_a", "snap_boundaries"})


def pool_children(objects: int, min_objects: int | None = None) -> int:
    """How many child interpreters a pass of ``objects`` gets, or 0 for none.

    The count is the converter's own sizing (``tile_convert_pool.default_workers``:
    the cores minus its spare, capped at its maximum, never under one), shared
    rather than restated so the two pools agree on what the machine can carry.
    Zero when the machine has too few cores for a second interpreter to help,
    or the set is too small to repay the boot.
    """
    from . import tile_convert_pool as converter

    # The converter reads the affinity mask where the platform has one, so a
    # QGIS pinned to four cores of a big box counts as four. Its reader is
    # taken when it is there, the machine count when it is not.
    count = getattr(converter, "usable_cores", None)
    cores = int(count()) if count is not None else int(os.cpu_count() or 2)
    if cores < MIN_CORES_FOR_POOL:
        return 0
    floor = DEFAULT_MIN_OBJECTS if min_objects is None else int(min_objects)
    if int(objects) < floor:
        return 0
    return int(converter.default_workers())


def refiner_spec(refiner, params: dict, pixel_size: float) -> tuple:
    """The plain values a child needs to rebuild ``refiner``.

    ``LiveRefiner`` is built from (params, pixel_size, metres_per_unit,
    unit_aspect) and keeps the last two; the first two are the caller's. Only
    the shape keys travel, so two refiners that differ by a filter memoise as
    one in the child, exactly as they do in the parent's own refine cache.
    """
    shape = {k: v for k, v in params.items() if k not in _FILTER_ONLY_KEYS}
    return (shape, float(pixel_size),
            float(getattr(refiner, "_metres_per_unit", 1.0) or 1.0),
            float(getattr(refiner, "_unit_aspect", 1.0) or 1.0))


def _spec_key(spec: tuple) -> tuple:
    params, px, mpu, aspect = spec
    return (tuple(sorted(params.items())), px, mpu, aspect)


class ReviewRefineProcessPool:
    """Shapes review objects on child interpreters: jobs in, shapes out.

    Same surface as ``ReviewRefineThread``: ``set_live_stamp``, ``submit``,
    ``take_results``, ``finish``, ``abort``, ``join_run``, ``isRunning``.
    """

    def __init__(self, workers: int = 1) -> None:
        self._workers = max(1, int(workers))
        self._queue: queue.Queue = queue.Queue()
        self._outbox: list = []
        self._lock = threading.Lock()
        self._children: list = []       # every Popen ever started
        self._live: list = []           # children that answered the handshake
        self._sent: dict = {}           # proc -> {key: item} awaiting an answer
        self._threads: list = []
        self._next_key = 0
        self._starting = False
        self._failed = False
        self._aborted = False
        self._stopping = False
        self._live_stamp = None
        self.child_failures = 0

    # ---- lifecycle -----------------------------------------------------------

    def start(self) -> bool:
        """Spawn the children and return. False when nothing could be spawned.

        The handshake (QGIS import, the served configuration) runs on a helper
        thread so the GUI never waits on a child booting; ``isRunning`` is True
        meanwhile and jobs queue up. A pool none of whose children come up
        marks itself failed, and the caller's alive check turns False.
        """
        import subprocess  # nosec B404 - a fixed argv, never a shell

        try:
            from ..core.config_cache import get_config
            from .tile_convert_pool import child_environment, child_python
        except Exception:  # noqa: BLE001 - the thread path is still there
            logger.info("ReviewRefineProcessPool: child plumbing unavailable",
                        exc_info=True)
            return False
        exe = child_python()
        if not exe:
            logger.info("ReviewRefineProcessPool: no child interpreter found")
            return False
        env = child_environment()
        boot = "from src.workers.review_refine_pool import child_main; child_main()"
        try:
            for _ in range(self._workers):
                # -s, never -I: isolated mode drops PYTHONPATH, which is the one
                # thing the child is given.
                proc = subprocess.Popen(  # nosec B603 - list argv, no shell
                    [exe, "-s", "-c", boot],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, env=env, close_fds=True)
                self._children.append(proc)
        except Exception:  # noqa: BLE001 - a machine that will not spawn
            logger.info("ReviewRefineProcessPool: could not start a child",
                        exc_info=True)
            self._kill_all()
            return False
        self._starting = True
        config = dict(get_config() or {})
        starter = threading.Thread(
            target=self._bring_up, args=(config,), daemon=True,
            name="reviewrefine-start")
        starter.start()
        self._threads.append(starter)
        return True

    def _bring_up(self, config: dict) -> None:
        """Handshake every child; the ones that answer start serving."""
        from .tile_convert_pool import (
            CHILD_READY_TIMEOUT_S,
            _await_ready,
            _recv,
            _send,
        )

        came_up = 0
        for proc in list(self._children):
            if self._aborted:
                break
            ok, _why = _await_ready(proc, CHILD_READY_TIMEOUT_S)
            if ok:
                try:
                    _send(proc.stdin, ("init", config))
                    frame = _recv(proc.stdout)
                    ok = bool(frame) and frame[0] == "init_ok"
                except Exception:  # noqa: BLE001 - a child gone mid-handshake
                    ok = False
            if not ok or self._aborted:
                self._kill(proc)
                continue
            with self._lock:
                self._live.append(proc)
                self._sent[proc] = {}
            for target, name in ((self._feed, "reviewrefine-feed"),
                                 (self._read_from, "reviewrefine-read")):
                thread = threading.Thread(target=target, args=(proc,),
                                          daemon=True, name=name)
                thread.start()
                self._threads.append(thread)
            came_up += 1
        self._starting = False
        if came_up == 0:
            self._failed = True
            logger.info("ReviewRefineProcessPool: no child came up")
        else:
            logger.info("ReviewRefineProcessPool: %d of %d child(ren) ready",
                        came_up, len(self._children))

    def _kill(self, proc) -> None:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001 - teardown never raises  # nosec B110
            pass

    def _kill_all(self) -> None:
        for proc in self._children:
            self._kill(proc)

    def _child_alive(self, proc) -> bool:
        try:
            return proc.poll() is None
        except Exception:  # noqa: BLE001
            return False

    def isRunning(self) -> bool:  # noqa: N802 - the QThread name the review calls
        """Whether there is still something here to answer what it was handed."""
        if self._aborted or self._failed:
            return False
        if self._starting:
            return True
        return any(self._child_alive(p) for p in self._live)

    # ---- GUI-thread API ------------------------------------------------------

    def set_live_stamp(self, stamp) -> None:
        """Name the settings worth spending time on; drop the queued rest.

        A child only ever holds the job it is on plus what sits in the pipe,
        so purging the shared queue here is what the thread's per-job stamp
        check did, at a fraction of the wasted work.
        """
        self._live_stamp = stamp
        with self._queue.mutex:
            kept = [item for item in self._queue.queue
                    if item is None or item[2] == stamp]
            self._queue.queue.clear()
            self._queue.queue.extend(kept)

    def submit(self, det_idx: int, stamp, refiner, geom, seq: int = 0,
               spec: tuple | None = None) -> bool:
        """Hand one object over. Returns at once; False when the pool is
        winding down or ``spec`` (the ``(params, pixel_size)`` to rebuild
        ``refiner`` from) is missing, and the caller must shape it itself."""
        if self._aborted or self._stopping or self._failed or spec is None:
            return False
        try:
            params, pixel_size = spec
            full = refiner_spec(refiner, params, pixel_size)
            wkb = bytes(geom.asWkb())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
        key = self._next_key
        self._next_key += 1
        self._queue.put((key, int(det_idx), stamp, int(seq), full, wkb))
        return True

    def take_results(self) -> list:
        """Every shape finished since the last call, as
        ``(det_idx, stamp, seq, geometry, error)``. GUI thread only."""
        from qgis.core import QgsGeometry

        with self._lock:
            raw = self._outbox
            self._outbox = []
        out = []
        for det_idx, stamp, seq, wkb, err_text in raw:
            geom = None
            if wkb:
                geom = QgsGeometry()
                geom.fromWkb(wkb)
                if geom.isEmpty():
                    geom = None
            err = RuntimeError(err_text) if err_text else None
            out.append((det_idx, stamp, seq, geom, err))
        return out

    def finish(self) -> None:
        """Stop once everything queued is shaped. Idempotent."""
        if self._stopping:
            return
        self._stopping = True
        for _ in range(max(1, len(self._children))):
            self._queue.put(None)

    def abort(self) -> None:
        """Drop what is queued and kill the children now."""
        self._aborted = True
        self._stopping = True
        with self._queue.mutex:
            self._queue.queue.clear()
        for _ in range(max(1, len(self._children))):
            self._queue.put(None)
        self._kill_all()

    def join_run(self, timeout_ms: int = REVIEW_POOL_JOIN_TIMEOUT_MS) -> bool:
        """True once every child process has exited."""
        import time

        deadline = time.monotonic() + max(0.0, timeout_ms) / 1000.0
        for proc in self._children:
            left = deadline - time.monotonic()
            try:
                proc.wait(timeout=max(0.01, left))
            except Exception:  # noqa: BLE001 - still running, or gone
                if self._child_alive(proc):
                    return False
        return True

    # ---- per-child threads ---------------------------------------------------

    def _feed(self, proc) -> None:
        """Move jobs from the shared queue into one child, until told to stop.

        A pipe write blocks when the child is behind, and that is the whole
        load balance: this thread waits, the others keep pulling.
        """
        from .tile_convert_pool import _send

        while True:
            item = self._queue.get()
            if item is None or self._aborted:
                break
            if not self._child_alive(proc):
                self._queue.put(item)
                break
            with self._lock:
                self._sent.get(proc, {})[item[0]] = item
            try:
                key, det_idx, stamp, seq, spec, wkb = item
                _send(proc.stdin, ("job", (key, det_idx, stamp, seq, spec, wkb)))
            except Exception:  # noqa: BLE001 - a dead pipe is a dead child
                self._requeue_from(proc)
                break
        if self._stopping and not self._aborted:
            try:
                proc.stdin.close()
            except Exception:  # noqa: BLE001  # nosec B110
                pass

    def _read_from(self, proc) -> None:
        """One child's answers until it closes; then whatever it still held
        goes back to the others."""
        from .tile_convert_pool import _recv

        while True:
            try:
                frame = _recv(proc.stdout)
            except Exception:  # noqa: BLE001 - killed mid-frame
                frame = None
            if frame is None:
                break
            kind, payload = frame
            if kind != "done":
                continue
            key, det_idx, stamp, seq, wkb, err_text = payload
            with self._lock:
                self._sent.get(proc, {}).pop(key, None)
                self._outbox.append((det_idx, stamp, seq, wkb, err_text))
        if not self._stopping:
            self.child_failures += 1
            self._requeue_from(proc)

    def _requeue_from(self, proc) -> None:
        """Put a dead child's unanswered jobs back for the survivors."""
        with self._lock:
            held = list(self._sent.pop(proc, {}).values())
            try:
                self._live.remove(proc)
            except ValueError:
                pass
        if self._aborted or self._stopping:
            return
        for item in held:
            if self._live_stamp is None or item[2] == self._live_stamp:
                self._queue.put(item)


# ---- child side ------------------------------------------------------------

def child_main() -> None:
    """A refine process: jobs off stdin, shaped WKB onto stdout.

    Runs ``refine_review_geom`` on a ``LiveRefiner`` rebuilt from the values
    the parent sent, under the served configuration the parent sent, so a
    child cannot drift from the thread path.
    """
    import sys

    from .tile_convert_pool import _recv, _send

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    # A library that prints would land in the middle of a frame.
    sys.stdout = sys.stderr
    try:
        from qgis.core import QgsApplication, QgsGeometry

        QgsApplication.setPrefixPath("", True)
        app = QgsApplication([], False)  # noqa: F841 - keeps QGIS alive
        app.initQgis()
        from ..core import config_cache
        from ..core.live_refine import LiveRefiner, refine_review_geom
    except BaseException as exc:  # noqa: BLE001 - the parent decides
        try:
            _send(stdout, ("no", repr(exc)))
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return

    _send(stdout, ("ready", None))
    refiners: dict = {}
    while True:
        frame = _recv(stdin)
        if frame is None:
            break
        kind, payload = frame
        if kind == "init":
            # The parent's snapshot, published here without touching the disk
            # mirror: this process must never write over the user's cache.
            config_cache.save_config = lambda config: False
            if payload:
                config_cache.set_config(payload)
            _send(stdout, ("init_ok", None))
            continue
        if kind != "job":
            break
        key, det_idx, stamp, seq, spec, wkb = payload
        try:
            skey = _spec_key(spec)
            refiner = refiners.get(skey)
            if refiner is None:
                params, px, mpu, aspect = spec
                refiner = LiveRefiner(dict(params), px, mpu, aspect)
                refiners[skey] = refiner
            geom = QgsGeometry()
            geom.fromWkb(wkb)
            shaped, err = refine_review_geom(refiner, geom)
            out = (bytes(shaped.asWkb())
                   if shaped is not None and not shaped.isEmpty() else None)
            _send(stdout, ("done", (key, det_idx, stamp, seq, out,
                                    repr(err) if err is not None else None)))
        except BaseException as exc:  # noqa: BLE001 - one object, not the pool
            _send(stdout, ("done", (key, det_idx, stamp, seq, None, repr(exc))))
