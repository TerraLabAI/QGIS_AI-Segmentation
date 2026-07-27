


















from __future__ import annotations

import logging
import os
import queue
import threading

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


class TileConvertProcessPool:


















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



    def set_snapshot(self, snapshot: dict) -> None:


        self._snapshot = snapshot

    def spawn(self) -> bool:







        if self._spawn_tried:
            return bool(self._children)
        self._spawn_tried = True
        return self._launch_children()

    def start(self, while_booting=None) -> bool:












        if not self.spawn():
            return False
        if self._snapshot is None:
            self._fail("no run snapshot")
            return False
        if while_booting is not None:
            try:
                while_booting()
            except Exception:  # noqa: BLE001
                logger.info("TileConvertProcessPool: boot-time work failed",
                            exc_info=True)
        import sys

        if sys.platform == "win32":
            answers = self._await_children_ready()
        else:
            answers = self._await_children_in_turn()
        return self._finish_start(answers)

    def _await_children_in_turn(self) -> list:


        import time




        deadline = time.monotonic() + self._ready_timeout

        floor = min(1.0, self._ready_timeout)
        answers = []
        for proc in self._children:
            answers.append(_await_ready(
                proc, max(floor, deadline - time.monotonic())))
            if not answers[-1][0]:
                break
        answers += [(False, "")] * (len(self._children) - len(answers))
        return answers

    def _await_children_ready(self) -> list:








        import time

        procs = list(self._children)
        slots: list = [None] * len(procs)
        done = threading.Event()

        def wait_one(index: int, proc) -> None:
            slots[index] = _await_ready(proc, self._ready_timeout)
            done.set()

        helpers = []
        for index, proc in enumerate(procs):
            helper = threading.Thread(target=wait_one, args=(index, proc),
                                      daemon=True, name="tileconvready")
            helper.start()
            helpers.append(helper)
        started = self._spawned_at or time.monotonic()
        deadline = time.monotonic() + self._ready_timeout
        first_ready = None
        while time.monotonic() < deadline:
            done.wait(0.05)
            done.clear()
            if all(slot is not None for slot in slots):
                break
            ready = sum(1 for slot in slots if slot is not None and slot[0])
            if ready and first_ready is None:
                first_ready = time.monotonic() - started
            if (ready >= 2 and first_ready is not None
                    and time.monotonic() - started >= 2 * first_ready + 1.0):
                break
        else:



            for helper in helpers:
                helper.join(max(0.0, deadline - time.monotonic()) + 1.5)
        return [slot if slot is not None else (False, "no answer in time")
                for slot in slots]

    def _launch_children(self) -> bool:
        import subprocess  # nosec B404

        exe = child_python()
        if not exe:
            logger.info("TileConvertProcessPool: no child interpreter found")
            self.last_failure = "no child interpreter found"
            return False
        import time

        self._spawned_at = time.monotonic()
        env = child_environment()
        boot = (
            "from src.workers.tile_convert_child import child_main; child_main()"
        )
        try:
            for _ in range(self._workers):
                errfile = _child_stderr_file()
                if errfile is not None:
                    self._stderr_files.append(errfile)


                proc = subprocess.Popen(  # nosec B603
                    [exe, "-s", "-c", boot],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=errfile if errfile is not None else subprocess.DEVNULL,
                    env=env, cwd=child_cwd(), close_fds=True,
                    creationflags=child_creation_flags())
                keep_child_off_power_throttling(proc)
                self._children.append(proc)
        except Exception as exc:  # noqa: BLE001
            logger.info("TileConvertProcessPool: could not start a child",
                        exc_info=True)
            self._fail(f"spawn error {type(exc).__name__}")
            return False
        return True

    def _finish_start(self, answers: list) -> bool:
        import sys

        ready = [ok for ok, _why in answers]
        if (sys.platform == "win32" and not all(ready)
                and sum(1 for r in ready if r) >= 2):




            logger.info("TileConvertProcessPool: keeping %d of %d children",
                        sum(1 for r in ready if r), len(self._children))
            kept = []
            for proc, ok in zip(self._children, ready):
                if ok:
                    kept.append(proc)
                else:
                    try:
                        proc.kill()
                    except Exception:  # noqa: BLE001  # nosec B110
                        pass
            self._children = kept
            ready = [True] * len(kept)
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





        answers = _init_all(self._children, ("init", self._snapshot),
                            self._ready_timeout)
        for ok, why in answers:
            if not ok:
                logger.info("TileConvertProcessPool: a child refused the run")
                self._fail("a child refused the run" + (f": {why}" if why else ""))
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


        tail = self._stderr_tail()
        self.last_failure = f"{detail}; stderr: {tail}" if tail else detail
        self._kill_all()

    def _stderr_tail(self, limit: int = 2048) -> str:


        pieces = []
        for errfile in self._stderr_files:
            try:
                errfile.flush()
                size = errfile.seek(0, os.SEEK_END)
                errfile.seek(max(0, size - limit))
                text = errfile.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001  # nosec B112
                continue
            text = " | ".join(part.strip() for part in text.splitlines() if part.strip())
            if text:
                pieces.append(text)
        return " || ".join(pieces)[-limit:]

    def _kill_all(self) -> None:
        for proc in self._children:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        self._children = []
        for errfile in self._stderr_files:
            try:
                errfile.close()
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        self._stderr_files = []



    @property
    def workers(self) -> int:
        return len(self._children) or self._workers

    @property
    def pending(self) -> int:
        return max(0, self._pending)

    def submit(self, job) -> None:






        with self._lock:
            if self._closed:
                raise RuntimeError("TileConvertProcessPool is closed")
            key = self._next_key
            self._next_key += 1
            self._jobs[key] = job
            self._pending += 1
            live = [i for i, proc in enumerate(self._children)
                    if proc not in self._dead_children]
            if live:




                n = len(self._children)
                start = self._round % n
                self._round += 1
                slot = min(live, key=lambda i: (
                    self._outboxes[i].qsize(), (i - start) % n))
                self._owners[key] = self._children[slot]
                self._outboxes[slot].put((key, job))
                return
            self._owners[key] = None
        self._convert_here(key, job)

    def _feed(self, proc, outbox: queue.Queue) -> None:



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
        if already_closed:
            return self.drain()
        for outbox in self._outboxes:


            with outbox.mutex:
                outbox.queue.clear()
            outbox.put(None)
        if not wait:





            for proc in self._children:
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
        self._kill_all()
        with self._lock:
            self._pending = 0
            self._jobs.clear()
            self._owners.clear()
        return results


def _init_all(procs: list, request, timeout: float) -> list:



    answers: list = [(False, "no answer")] * len(procs)

    def exchange(index: int, proc) -> None:
        answers[index] = _await_ready(proc, timeout, request=request)

    threads = []
    for index, proc in enumerate(procs):
        thread = threading.Thread(target=exchange, args=(index, proc),
                                  daemon=True, name="tileconvinit")
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join(timeout + 2.0)
    return list(answers)


def _await_ready(proc, timeout: float, request=None) -> tuple[bool, str]:










    answer: list = []

    def read() -> None:
        try:
            if request is not None:
                _send(proc.stdin, request)
            answer.append(_recv(proc.stdout))
        except Exception:  # noqa: BLE001
            answer.append(None)

    thread = threading.Thread(target=read, daemon=True, name="tileconv-handshake")
    thread.start()
    thread.join(timeout)
    if not answer:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        thread.join(timeout=1.0)
        if not thread.is_alive():
            for stream in (proc.stdin, proc.stdout):
                try:
                    stream.close()
                except Exception:  # noqa: BLE001  # nosec B110
                    pass
        return False, f"no answer within {timeout:.0f}s"
    frame = answer[0]
    if not frame:
        return False, "died before answering"
    if frame[0] == "no":
        logger.info("TileConvertProcessPool: child could not start (%s)",
                    frame[1])
        return False, str(frame[1])[:300]
    expected = "init_ok" if request is not None else "ready"
    return frame[0] == expected, ""
