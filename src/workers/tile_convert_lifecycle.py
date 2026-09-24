








from __future__ import annotations

import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .tile_convert_child import (
    _child_stderr_file,
    child_creation_flags,
    child_cwd,
    child_environment,
    child_python,
    keep_child_off_power_throttling,
)
from .tile_convert_handshake import _await_ready, _init_all

logger = logging.getLogger(__name__)


class TileConvertLifecycleMixin:




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

    def start_bridged(self, threads: int) -> bool:













        if not self.spawn():
            return False
        if self._snapshot is None:
            self._fail("no run snapshot")
            return False
        self._local_threads = max(1, int(threads))
        from ..core.macos_activity import promote_current_thread


        self._local_pool = ThreadPoolExecutor(
            max_workers=self._local_threads, thread_name_prefix="tileconvlocal",
            initializer=promote_current_thread)
        self._bridged_at = time.monotonic()
        self._boot_state = "booting"
        try:
            threading.Thread(target=self._boot_bridged, daemon=True,
                             name="tileconvboot").start()
        except Exception as exc:  # noqa: BLE001
            self._fail(f"boot thread error {type(exc).__name__}")
            self._boot_state = "failed"
        return True

    def _boot_bridged(self) -> None:






        import sys

        ok = False
        try:
            answers = (self._await_children_ready() if sys.platform == "win32"
                       else self._await_children_in_turn())
            ok = self._finish_start(answers, wire=False)
        except Exception as exc:  # noqa: BLE001
            logger.info("TileConvertProcessPool: bridged boot failed",
                        exc_info=True)
            self.last_failure = self.last_failure or f"boot error {type(exc).__name__}"
            ok = False
        with self._lock:
            abandoned = self._closed
            if not abandoned:
                self._boot_s = time.monotonic() - self._bridged_at
                self._boot_state = "ready" if ok else "failed"
                if ok:
                    self._wire_children()
                    self._live = True


                    while self._local_queue:
                        key = self._local_queue.popleft()
                        job = self._jobs.get(key)
                        if job is not None and not self._hand_to_child(key, job):
                            self._local_queue.appendleft(key)
                            break
        if abandoned or not ok:
            self._kill_all()

    def take_boot_outcome(self) -> tuple | None:





        with self._lock:
            if self._boot_taken or self._boot_state in ("", "booting"):
                return None
            self._boot_taken = True
            return (self._boot_state, self.last_failure, self._boot_s,
                    self.local_converted)

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
            if (first_ready is not None
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

    def _finish_start(self, answers: list, wire: bool = True) -> bool:



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
        if not self._children:
            self._fail("no child left to start")
            return False
        if wire:
            self._wire_children()
        return True

    def _wire_children(self) -> None:

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
