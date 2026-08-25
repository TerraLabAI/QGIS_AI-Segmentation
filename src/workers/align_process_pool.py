
































from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)



ALIGN_MAX_WORKERS = 4





ALIGN_THREAD_WORKERS = 1




ALIGN_CHILD_RAM_MB = 300


STEP_WAIT_S = 0.010


STOP_JOIN_TIMEOUT_S = 3.0




STAGE_TIMEOUT_S = 120.0

_COUNTERS = ("aligned_count", "reverted_count", "circle_count",
             "skipped_count", "simplified_count")


def align_child_ram_mb(fallback: int) -> int:

    from ..core.server_dials import dial_in_range

    return int(dial_in_range(
        "tuning.convert.align_child_ram_each_mb", fallback, 128, 4096))


def align_children(objects: int, min_objects: int) -> int:






    if int(objects) < max(1, int(min_objects)):
        return 0
    try:
        from ..core.shape_policy_dials import align_process_workers
        from .tile_convert_pool import process_workers

        want = int(align_process_workers(ALIGN_MAX_WORKERS))
        if want <= 0:
            return 0
        return int(process_workers(
            default_max=want, ram_each_mb=align_child_ram_mb(ALIGN_CHILD_RAM_MB)))
    except Exception:  # noqa: BLE001
        return 0


def align_threads(objects: int, min_objects: int) -> int:







    if int(objects) < max(1, int(min_objects)):
        return 0
    try:
        from ..core.shape_policy_dials import align_thread_workers
        from .tile_convert_pool import usable_cores

        want = int(align_thread_workers(ALIGN_THREAD_WORKERS))
        threads = min(want, int(usable_cores()) - 1)
        return threads if threads >= 2 else 0
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








    def __init__(self, rows: list, params, frame_scale: tuple, workers: int,
                 threads: int = 0) -> None:
        self._rows = list(rows)
        self._params = params
        self._scale = (float(frame_scale[0]), float(frame_scale[1]))

        self._workers = max(0, int(workers))
        self._threads = max(0, int(threads))
        self._wkbs = [None if g is None or g.isEmpty() else bytes(g.asWkb())
                      for _fid, g, _s in self._rows]
        self._aligned: dict = {}
        self._lock = threading.Lock()
        self._progress_done = 0
        self._progress_parts: dict = {}
        self._done = threading.Event()
        self._stop = threading.Event()
        self._finished = False
        self._error: BaseException | None = None
        self._children: list = []
        self._config: dict = {}
        self._env: dict = {}
        self.mode = "processes" if self._workers else "threads"
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
            done = bool(self._workers and self._env and self._run_on_children())
            if not done:
                self._kill_all()
            if not done and self._threads >= 2 and not self._stop.is_set():
                done = self._run_on_threads()
            if not done and not self._stop.is_set():
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

    def progress(self) -> tuple[int, int]:

        total = 3 * len(self._rows)
        with self._lock:
            completed = self._progress_done
        ceiling = total if self._finished else max(0, total - 1)
        return min(completed, ceiling), total

    def _note_progress(self, completed: int) -> None:
        with self._lock:


            self._progress_done = max(self._progress_done, int(completed))

    def _note_share_progress(self, stage: int, slot: int, completed: int) -> None:
        with self._lock:
            key = (stage, slot)
            self._progress_parts[key] = max(
                self._progress_parts.get(key, 0), int(completed))
            done = stage * len(self._rows) + sum(
                count for (part_stage, _slot), count in self._progress_parts.items()
                if part_stage == stage)
            self._progress_done = max(self._progress_done, done)

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
        from .tile_convert_pool import child_python, spawn_worker_child

        exe = child_python()
        if not exe:
            return False
        for _ in range(self._workers):




            proc = spawn_worker_child(exe, "src.workers.align_process_pool", self._env)
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

    def _exchange(self, requests: list, kind: str,
                  progress_stage: int | None = None) -> list | None:



        from .tile_convert_pool import _recv, _send

        answers: list = [None] * len(self._children)

        def talk(slot: int) -> None:
            proc = self._children[slot]
            try:
                if requests[slot] is not None:
                    _send(proc.stdin, requests[slot])
                while True:
                    frame = _recv(proc.stdout)
                    if not frame or frame[0] != "progress":
                        break
                    if progress_stage is not None:
                        self._note_share_progress(progress_stage, slot, frame[1])
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

        shares = _balanced_shares(self._wkbs, n)
        prepare = [("prepare", [(i, self._rows[i][0], self._wkbs[i], self._rows[i][2])
                                for i in share]) for share in shares]
        summaries = self._exchange(prepare, "prepared", progress_stage=0)
        if summaries is None or self._stop.is_set():
            return False
        consensus = self._consensus_for(summaries)
        align = [("align", [(i, consensus[i]) for i in share])
                 for share in shares]
        answers = self._exchange(align, "aligned", progress_stage=2)
        if answers is None or self._stop.is_set():
            return False
        self._publish_answers(answers)
        return True

    def _consensus_for(self, summaries: list) -> list:


        from ..core.footprint_alignment import FootprintAlignSweep

        count = len(self._rows)
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
            if (i + 1) % 64 == 0 or i + 1 == count:
                self._note_progress(count + i + 1)
        return shell._consensus

    def _publish_answers(self, answers: list) -> None:

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

    def _run_on_threads(self) -> bool:




        from qgis.core import QgsGeometry

        from ..core.footprint_alignment import FootprintAlignSweep

        self.mode = "threads"
        n = self._threads
        shares = _balanced_shares(self._wkbs, n)
        sweeps, indexes = [], []
        for share in shares:
            rows, index = [], {}
            for local, i in enumerate(share):
                geom = None
                if self._wkbs[i] is not None:
                    geom = QgsGeometry()
                    geom.fromWkb(self._wkbs[i])
                rows.append((self._rows[i][0], geom, self._rows[i][2]))
                index[i] = local
            sweeps.append(FootprintAlignSweep(rows, self._params, self._scale))
            indexes.append(index)

        def fan_out(work) -> list | None:
            answers: list = [None] * n

            def one(slot: int) -> None:
                try:
                    answers[slot] = work(slot)
                except Exception:  # noqa: BLE001
                    logger.info("ProcessAlignPass: share failed", exc_info=True)

            threads = [threading.Thread(target=one, args=(k,), daemon=True,
                                        name="align-share") for k in range(n)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if self._stop.is_set() or any(a is None for a in answers):
                return None
            return answers

        summaries = fan_out(lambda k: prepare_share(
            sweeps[k], indexes[k], self._stop,
            progress=lambda done: self._note_share_progress(0, k, done)))
        if summaries is None:
            return False
        consensus = self._consensus_for(summaries)
        answers = fan_out(lambda k: align_share(
            sweeps[k], indexes[k], [(i, consensus[i]) for i in shares[k]],
            self._stop, progress=lambda done: self._note_share_progress(2, k, done)))
        if answers is None:
            return False
        self._publish_answers(answers)
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
            done = sweep.step(8)
            self._note_progress(sweep.progress()[0])
            if done:
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




def prepare_share(sweep, index: dict, stop=None, progress=None) -> list:





    summaries = []
    total = len(index)
    for completed, (i, local) in enumerate(index.items(), 1):
        if stop is not None and stop.is_set():
            break
        try:
            sweep._prepare_one(local)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        prep = sweep._prepared[local]
        summaries.append((i, None if prep is None else (
            prep["own"], prep["center"], prep["perimeter"])))
        if progress is not None and (completed % 64 == 0 or completed == total):
            progress(completed)
    return summaries


def align_share(sweep, index: dict, payload: list, stop=None, progress=None) -> tuple:

    out = []
    total = len(payload)
    for completed, (i, consensus) in enumerate(payload, 1):
        if stop is not None and stop.is_set():
            break
        local = index[i]
        sweep._consensus[local] = consensus
        try:
            sweep._align_one(local)
        except Exception:  # noqa: BLE001
            sweep.skipped_count += 1
        row = sweep._out[local]
        if row is not sweep._rows[local] and row[1] is not None:
            out.append((i, bytes(row[1].asWkb())))
        if progress is not None and (completed % 64 == 0 or completed == total):
            progress(completed)
    counters = {name: getattr(sweep, name) for name in _COUNTERS}
    return out, counters




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
            _send(stdout, ("prepared", prepare_share(
                sweep, index, progress=lambda done: _send(stdout, ("progress", done)))))
        elif kind == "align":
            unthrottle_this_process()
            if sweep is None:
                raise RuntimeError("align arrived before prepare")
            _send(stdout, ("aligned", align_share(
                sweep, index, payload,
                progress=lambda done: _send(stdout, ("progress", done)))))
        else:
            break
