




































from __future__ import annotations

import logging
import os
import queue
import threading

logger = logging.getLogger(__name__)




MIN_CORES_FOR_POOL = 5




DEFAULT_MIN_OBJECTS = 1500



REVIEW_POOL_JOIN_TIMEOUT_MS = 3000



_FILTER_ONLY_KEYS = frozenset({"conf", "min_a", "max_a", "snap_boundaries"})


def pool_children(objects: int, min_objects: int | None = None) -> int:











    from ..core.server_dials import dial_in_range
    from . import tile_convert_pool as converter

    count = getattr(converter, "usable_cores", None)
    cores = int(count()) if count is not None else int(os.cpu_count() or 2)
    if cores < dial_in_range("tuning.review.min_cores_for_pool", MIN_CORES_FOR_POOL, 2, 32):
        return 0
    floor = DEFAULT_MIN_OBJECTS if min_objects is None else int(min_objects)
    if int(objects) < floor:
        return 0
    return int(converter.default_workers())


def refiner_spec(refiner, params: dict, pixel_size: float) -> tuple:







    shape = {k: v for k, v in params.items() if k not in _FILTER_ONLY_KEYS}
    return (shape, float(pixel_size),
            float(getattr(refiner, "_metres_per_unit", 1.0) or 1.0),
            float(getattr(refiner, "_unit_aspect", 1.0) or 1.0))


def _spec_key(spec: tuple) -> tuple:
    params, px, mpu, aspect = spec
    return (tuple(sorted(params.items())), px, mpu, aspect)


class ReviewRefineProcessPool:






    def __init__(self, workers: int = 1) -> None:
        self._workers = max(1, int(workers))
        self._queue: queue.Queue = queue.Queue()
        self._outbox: list = []
        self._lock = threading.Lock()
        self._children: list = []
        self._live: list = []
        self._sent: dict = {}
        self._threads: list = []
        self._next_key = 0
        self._starting = False
        self._failed = False
        self._aborted = False
        self._stopping = False
        self._live_stamp = None
        self.child_failures = 0



    def start(self) -> bool:







        import subprocess  # nosec B404

        try:
            from ..core.config_cache import get_config
            from .tile_convert_pool import (
                child_creation_flags,
                child_cwd,
                child_environment,
                child_python,
                keep_child_off_power_throttling,
            )
        except Exception:  # noqa: BLE001
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


                proc = subprocess.Popen(  # nosec B603
                    [exe, "-s", "-c", boot],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, env=env, cwd=child_cwd(),
                    close_fds=True,

                    creationflags=child_creation_flags())
                keep_child_off_power_throttling(proc)
                self._children.append(proc)
        except Exception:  # noqa: BLE001
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






        results: list = []
        helpers = []
        for proc in list(self._children):
            helper = threading.Thread(
                target=lambda p=proc: results.append(self._bring_up_one(p, config)),
                daemon=True, name="reviewrefine-handshake")
            helper.start()
            helpers.append(helper)
        for helper in helpers:
            helper.join()
        came_up = sum(1 for ok in results if ok)
        self._starting = False
        if came_up == 0:
            self._failed = True
            logger.info("ReviewRefineProcessPool: no child came up")
        else:
            logger.info("ReviewRefineProcessPool: %d of %d child(ren) ready",
                        came_up, len(self._children))

    def _bring_up_one(self, proc, config: dict) -> bool:

        from .tile_convert_pool import CHILD_READY_TIMEOUT_S, _await_ready

        if self._aborted:
            self._kill(proc)
            return False
        ok, _why = _await_ready(proc, CHILD_READY_TIMEOUT_S)
        if ok:


            ok, _why = _await_ready(
                proc, CHILD_READY_TIMEOUT_S, request=("init", config))
        if not ok or self._aborted:
            self._kill(proc)
            return False
        with self._lock:
            self._live.append(proc)
            self._sent[proc] = {}
        for target, name in ((self._feed, "reviewrefine-feed"),
                             (self._read_from, "reviewrefine-read")):
            thread = threading.Thread(target=target, args=(proc,),
                                      daemon=True, name=name)
            thread.start()
            self._threads.append(thread)
        return True

    def _kill(self, proc) -> None:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _kill_all(self) -> None:
        for proc in self._children:
            self._kill(proc)

    def _child_alive(self, proc) -> bool:
        try:
            return proc.poll() is None
        except Exception:  # noqa: BLE001
            return False

    def isRunning(self) -> bool:  # noqa: N802

        if self._aborted or self._failed:
            return False
        if self._starting:
            return True
        return any(self._child_alive(p) for p in self._live)



    def set_live_stamp(self, stamp) -> None:






        self._live_stamp = stamp
        with self._queue.mutex:
            kept = [item for item in self._queue.queue
                    if item is None or item[2] == stamp]
            self._queue.queue.clear()
            self._queue.queue.extend(kept)

    def submit(self, det_idx: int, stamp, refiner, geom, seq: int = 0,
               spec: tuple | None = None) -> bool:



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

        if self._stopping:
            return
        self._stopping = True
        for _ in range(max(1, len(self._children))):
            self._queue.put(None)

    def abort(self) -> None:

        self._aborted = True
        self._stopping = True
        with self._queue.mutex:
            self._queue.queue.clear()
        for _ in range(max(1, len(self._children))):
            self._queue.put(None)
        self._kill_all()

    def join_run(self, timeout_ms: int | None = None) -> bool:





        import time

        if timeout_ms is None:
            timeout_ms = REVIEW_POOL_JOIN_TIMEOUT_MS
            try:
                from ..core.server_dials import dial_in_range
                timeout_ms = dial_in_range(
                    "tuning.review.refine_pool_join_timeout_ms",
                    REVIEW_POOL_JOIN_TIMEOUT_MS, 500, 30000)
            except Exception:  # noqa: BLE001
                timeout_ms = REVIEW_POOL_JOIN_TIMEOUT_MS
        deadline = time.monotonic() + max(0.0, timeout_ms) / 1000.0
        for proc in self._children:
            left = deadline - time.monotonic()
            try:
                proc.wait(timeout=max(0.01, left))
            except Exception:  # noqa: BLE001
                if self._child_alive(proc):
                    return False
        return True



    def _feed(self, proc) -> None:





        from .tile_convert_pool import _send

        while True:
            item = self._queue.get()
            if item is None or self._aborted:
                break
            with self._lock:
                sent = self._sent.get(proc)
                if sent is None:
                    self._queue.put(item)
                    break
                sent[item[0]] = item
            try:
                key, det_idx, stamp, seq, spec, wkb = item
                _send(proc.stdin, ("job", (key, det_idx, stamp, seq, spec, wkb)))
            except Exception:  # noqa: BLE001
                self._requeue_from(proc)
                break
        if self._stopping and not self._aborted:
            try:
                proc.stdin.close()
            except Exception:  # noqa: BLE001  # nosec B110
                pass

    def _read_from(self, proc) -> None:


        from .tile_convert_pool import _recv

        while True:
            try:
                frame = _recv(proc.stdout)
            except Exception:  # noqa: BLE001
                frame = None
            if frame is None:
                break
            kind, payload = frame
            if kind != "done":
                continue
            key, det_idx, stamp, seq, wkb, err_text = payload
            with self._lock:
                if self._sent.get(proc, {}).pop(key, None) is not None:
                    self._outbox.append((det_idx, stamp, seq, wkb, err_text))
        if not self._stopping:
            self.child_failures += 1
            self._requeue_from(proc)

    def _requeue_from(self, proc) -> None:

        with self._lock:
            held = list(self._sent.pop(proc, {}).values())
            try:
                self._live.remove(proc)
            except ValueError:
                pass
        self._kill(proc)
        if self._aborted or self._stopping:
            return
        for item in held:
            if self._live_stamp is None or item[2] == self._live_stamp:
                self._queue.put(item)




def child_main() -> None:






    import sys

    from .tile_convert_child import (
        _ANY_FAILURE,
        _recv,
        _send,
        skip_unused_child_imports,
        unthrottle_this_process,
    )

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer

    sys.stdout = sys.stderr
    skip_unused_child_imports()
    try:
        from qgis.core import QgsApplication, QgsGeometry

        QgsApplication.setPrefixPath("", True)
        app = QgsApplication([], False)  # noqa: F841
        app.initQgis()
        from ..core import config_cache
        from ..core.live_refine import LiveRefiner, refine_review_geom
    except _ANY_FAILURE as exc:
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





            config_cache.save_config = lambda config: False  # type: ignore[assignment, misc]
            if payload:
                config_cache.set_config(payload)
            _send(stdout, ("init_ok", None))
            continue
        if kind != "job":
            break
        key, det_idx, stamp, seq, spec, wkb = payload
        unthrottle_this_process()
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
        except _ANY_FAILURE as exc:
            _send(stdout, ("done", (key, det_idx, stamp, seq, None, repr(exc))))
