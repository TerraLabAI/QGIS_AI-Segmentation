



























from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)



ALIGN_MAX_WORKERS = 4


STEP_WAIT_S = 0.010


STOP_JOIN_TIMEOUT_S = 3.0




STAGE_TIMEOUT_S = 120.0

_COUNTERS = ("aligned_count", "reverted_count", "circle_count",
             "skipped_count", "simplified_count")


def align_children(objects: int, min_objects: int) -> int:






    if int(objects) < max(1, int(min_objects)):
        return 0
    try:
        from ..core.shape_policy_dials import align_process_workers
        from .tile_convert_pool import process_workers

        want = int(align_process_workers(ALIGN_MAX_WORKERS))
        if want <= 0:
            return 0
        return int(process_workers(default_max=want))
    except Exception:  # noqa: BLE001
        return 0


def _balanced_shares(wkbs: list, n: int) -> list:






    import heapq

    order = sorted(range(len(wkbs)),
                   key=lambda i: -(len(wkbs[i]) if wkbs[i] else 0))
    heap = [(0, slot) for slot in range(n)]
    shares: list = [[] for _ in range(n)]
    for i in order:
        load, slot = heapq.heappop(heap)
        shares[slot].append(i)
        heapq.heappush(heap, (load + (len(wkbs[i]) if wkbs[i] else 1), slot))
    return [sorted(share) for share in shares]


class ProcessAlignPass:








    def __init__(self, rows: list, params, frame_scale: tuple, workers: int) -> None:
        self._rows = list(rows)
        self._params = params
        self._scale = (float(frame_scale[0]), float(frame_scale[1]))
        self._workers = max(1, int(workers))
        self._wkbs = [None if g is None or g.isEmpty() else bytes(g.asWkb())
                      for _fid, g, _s in self._rows]
        self._aligned: dict = {}
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._stop = threading.Event()
        self._finished = False
        self._error: BaseException | None = None
        self._children: list = []
        self._config: dict = {}
        self._env: dict = {}
        self.mode = "processes"
        self.aligned_count = 0
        self.reverted_count = 0
        self.circle_count = 0
        self.skipped_count = 0
        self.simplified_count = 0
        self.changed_fids: set = set()
        self._thread = threading.Thread(
            target=self._run, name="align-processes", daemon=True)



    def start(self) -> ProcessAlignPass:


        try:
            from ..core.config_cache import get_config
            from .tile_convert_pool import child_environment

            self._config = dict(get_config() or {})
            self._env = child_environment()
        except Exception:  # noqa: BLE001
            self._env = {}
        self._thread.start()
        return self

    def _run(self) -> None:

        from ..core.macos_activity import promote_current_thread
        promote_current_thread()
        try:
            if not self._env or not self._run_on_children():
                if not self._stop.is_set():
                    self._run_here()
            if not self._stop.is_set():
                self._finished = True
        except Exception as exc:  # noqa: BLE001
            self._error = exc
            logger.info("ProcessAlignPass: pass failed: %r", exc)
        finally:
            self._kill_all()
            self._done.set()



    def step(self, count: int = 1) -> bool:

        if self._done.is_set():
            return True
        return self._done.wait(STEP_WAIT_S)

    def finished(self) -> bool:


        return self._done.is_set()

    def result(self) -> Any:

        if not self._done.is_set() or not self._finished or self._error is not None:
            return None
        return self._assemble()

    def stop_and_take(self, timeout_s: float | None = None) -> Any:


        self.stop(timeout_s)
        if self._thread.is_alive() or self._error is not None:
            return None
        return self._assemble()

    def stop(self, timeout_s: float | None = None) -> None:
        self._stop.set()
        self._kill_all()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            if timeout_s is None:
                from ..core.server_dials import dial_in_range
                timeout_s = dial_in_range(
                    "tuning.convert.align_stop_join_timeout_s",
                    STOP_JOIN_TIMEOUT_S, 0.5, 15)
            self._thread.join(timeout_s)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def _assemble(self) -> list:
        from qgis.core import QgsGeometry

        with self._lock:
            aligned = dict(self._aligned)
        out = list(self._rows)
        for i, wkb in aligned.items():
            geom = QgsGeometry()
            geom.fromWkb(wkb)
            if geom.isEmpty():
                continue
            fid, _geom, score = self._rows[i]
            out[i] = (fid, geom, score)
        return out



    def _spawn(self) -> bool:
        import subprocess  # nosec B404

        from .tile_convert_pool import (
            child_creation_flags,
            child_cwd,
            child_python,
            keep_child_off_power_throttling,
        )

        exe = child_python()
        if not exe:
            return False
        boot = "from src.workers.align_process_pool import child_main; child_main()"
        for _ in range(self._workers):
            proc = subprocess.Popen(  # nosec B603
                [exe, "-s", "-c", boot],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=self._env, cwd=child_cwd(),
                close_fds=True,





                creationflags=child_creation_flags())
            keep_child_off_power_throttling(proc)
            with self._lock:
                self._children.append(proc)
            if self._stop.is_set():
                return False
        return True

    def _kill_all(self) -> None:
        with self._lock:
            children = list(self._children)
        for proc in children:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001  # nosec B110
                pass

    def _exchange(self, requests: list, kind: str) -> list | None:



        from .tile_convert_pool import _recv, _send

        answers: list = [None] * len(self._children)

        def talk(slot: int) -> None:
            proc = self._children[slot]
            try:
                if requests[slot] is not None:
                    _send(proc.stdin, requests[slot])
                frame = _recv(proc.stdout)
            except Exception:  # noqa: BLE001
                return
            if frame and frame[0] == kind:
                answers[slot] = frame

        threads = [threading.Thread(target=talk, args=(k,), daemon=True,
                                    name="align-talk")
                   for k in range(len(self._children))]
        for thread in threads:
            thread.start()
        from ..core.server_dials import dial_in_range
        stage_timeout_s = dial_in_range(
            "tuning.convert.align_stage_timeout_s", STAGE_TIMEOUT_S, 10, 300)
        deadline = time.monotonic() + stage_timeout_s
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        if any(t.is_alive() for t in threads) or any(a is None for a in answers):
            return None
        return [a[1] for a in answers]

    def _run_on_children(self) -> bool:


        from ..core.footprint_alignment import FootprintAlignSweep

        try:
            if not self._spawn():
                return False
        except Exception:  # noqa: BLE001
            logger.info("ProcessAlignPass: could not start a child", exc_info=True)
            return False
        n = len(self._children)
        if self._exchange([None] * n, "ready") is None:
            return False
        init = ("init", (self._config, dataclasses.asdict(self._params),
                         self._scale))
        if self._exchange([init] * n, "init_ok") is None:
            return False

        count = len(self._rows)
        shares = _balanced_shares(self._wkbs, n)
        prepare = [("prepare", [(i, self._rows[i][0], self._wkbs[i], self._rows[i][2])
                                for i in share]) for share in shares]
        summaries = self._exchange(prepare, "prepared")
        if summaries is None or self._stop.is_set():
            return False


        shell = FootprintAlignSweep(
            [(None, None, None)] * count, self._params, self._scale)
        for part in summaries:
            for i, summary in part:
                if summary is not None:
                    own, center, perimeter = summary
                    shell._prepared[i] = {
                        "own": own, "center": center, "perimeter": perimeter}
        shell._build_neighbour_index()
        for i in range(count):
            try:
                shell._consensus_one(i)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        align = [("align", [(i, shell._consensus[i]) for i in share])
                 for share in shares]
        answers = self._exchange(align, "aligned")
        if answers is None or self._stop.is_set():
            return False
        aligned: dict = {}
        totals = dict.fromkeys(_COUNTERS, 0)
        changed: set = set()
        for rows_out, counters in answers:
            for i, wkb in rows_out:
                aligned[i] = wkb
                changed.add(self._rows[i][0])
            for name in _COUNTERS:
                totals[name] += int(counters.get(name, 0))
        with self._lock:
            self._aligned = aligned
        for name, value in totals.items():
            setattr(self, name, value)
        self.changed_fids = changed
        return True

    def _run_here(self) -> None:

        from qgis.core import QgsGeometry

        from ..core.footprint_alignment import FootprintAlignSweep

        self.mode = "thread"
        copies = []
        for (fid, _geom, score), wkb in zip(self._rows, self._wkbs):
            geom = None
            if wkb is not None:
                geom = QgsGeometry()
                geom.fromWkb(wkb)
            copies.append((fid, geom, score))
        sweep = FootprintAlignSweep(copies, self._params, self._scale)
        while not self._stop.is_set():
            if sweep.step(8):
                break
        out = sweep.result()
        aligned = {}
        for i, (row, copy) in enumerate(zip(out, copies)):
            if row is not copy and row[1] is not None:
                aligned[i] = bytes(row[1].asWkb())
        with self._lock:
            self._aligned = aligned
        for name in _COUNTERS:
            setattr(self, name, getattr(sweep, name))
        self.changed_fids = set(sweep.changed_fids)




def child_main() -> None:

    import sys

    from .tile_convert_pool import (
        _ANY_FAILURE,
        _recv,
        _send,
        unthrottle_this_process,
    )

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer

    sys.stdout = sys.stderr
    try:
        from qgis.core import QgsGeometry

        from ..core import config_cache
        from ..core import footprint_alignment as fa
    except _ANY_FAILURE as exc:
        try:
            _send(stdout, ("no", repr(exc)))
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return

    _send(stdout, ("ready", None))
    params = scale = sweep = None
    index: dict = {}
    while True:
        frame = _recv(stdin)
        if frame is None:
            break
        kind, payload = frame
        if kind == "init":
            config, fields, scale = payload




            config_cache.save_config = lambda config: False  # type: ignore[assignment, misc]
            if config:
                config_cache.set_config(config)
            params = fa.AlignmentParams(**fields)
            _send(stdout, ("init_ok", None))
        elif kind == "prepare":
            unthrottle_this_process()
            rows = []
            index = {}
            for local, (i, fid, wkb, score) in enumerate(payload):
                geom = None
                if wkb is not None:
                    geom = QgsGeometry()
                    geom.fromWkb(wkb)
                rows.append((fid, geom, score))
                index[i] = local
            if params is None or scale is None:
                raise RuntimeError("prepare arrived before init")
            sweep = fa.FootprintAlignSweep(rows, params, scale)
            summaries = []
            for i, local in index.items():
                try:
                    sweep._prepare_one(local)
                except Exception:  # noqa: BLE001  # nosec B110
                    pass
                prep = sweep._prepared[local]
                summaries.append((i, None if prep is None else (
                    prep["own"], prep["center"], prep["perimeter"])))
            _send(stdout, ("prepared", summaries))
        elif kind == "align":
            unthrottle_this_process()
            if sweep is None:
                raise RuntimeError("align arrived before prepare")
            out = []
            for i, consensus in payload:
                local = index[i]
                sweep._consensus[local] = consensus
                try:
                    sweep._align_one(local)
                except Exception:  # noqa: BLE001
                    sweep.skipped_count += 1
                row = sweep._out[local]
                if row is not sweep._rows[local] and row[1] is not None:
                    out.append((i, bytes(row[1].asWkb())))
            counters = {name: getattr(sweep, name) for name in _COUNTERS}
            _send(stdout, ("aligned", (out, counters)))
        else:
            break
