






















from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)



STEP_WAIT_S = 0.010


STOP_JOIN_TIMEOUT_S = 3.0




_ANY_FAILURE = (Exception, GeneratorExit, KeyboardInterrupt, SystemExit)


class SteppedPassThread:



    def __init__(self, pass_: Any, step_count: int = 64,
                 inputs: list | None = None,
                 originals: list | None = None) -> None:
        self._pass = pass_
        self._count = max(1, int(step_count))
        self._inputs = inputs
        self._originals = originals
        self._done = threading.Event()
        self._stop = threading.Event()
        self._finished = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="review-set-pass", daemon=True)



    def start(self) -> SteppedPassThread:
        self._thread.start()
        return self

    def _run(self) -> None:


        from .macos_activity import promote_current_thread
        promote_current_thread()
        try:
            while not self._stop.is_set():
                if self._pass.step(self._count):
                    self._finished = True
                    break
        except _ANY_FAILURE as exc:
            self._error = exc
            logger.info("SteppedPassThread: pass failed: %r", exc)
        finally:
            self._done.set()



    def step(self, count: int = 64) -> bool:

        if self._done.is_set():
            return True
        from .server_dials import dial_in_range
        step_wait_s = dial_in_range(
            "tuning.review.set_pass_step_wait_s", STEP_WAIT_S, 0.001, 0.1)
        return self._done.wait(step_wait_s)

    def finished(self) -> bool:



        return self._done.is_set()

    def result(self) -> Any:


        if not self._done.is_set() or not self._finished or self._error is not None:
            return None
        return self._map_back(self._pass.result())

    def stop_and_take(self, timeout_s: float | None = None) -> Any:







        if timeout_s is None:
            from .server_dials import dial_in_range
            timeout_s = dial_in_range(
                "tuning.review.set_pass_stop_join_timeout_s",
                STOP_JOIN_TIMEOUT_S, 0.5, 30.0)
        self.stop(timeout_s)
        if self._thread.is_alive() or self._error is not None:
            return None
        return self._map_back(self._pass.result())

    def _map_back(self, out: Any) -> Any:


        if (self._inputs is None or self._originals is None
                or not isinstance(out, list) or len(out) != len(self._inputs)):
            return out
        return [orig if item is copy else item
                for item, copy, orig in zip(out, self._inputs, self._originals)]

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self, timeout_s: float | None = None) -> None:

        if timeout_s is None:
            from .server_dials import dial_in_range
            timeout_s = dial_in_range(
                "tuning.review.set_pass_stop_join_timeout_s",
                STOP_JOIN_TIMEOUT_S, 0.5, 30.0)
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout_s)

    def __getattr__(self, name: str) -> Any:

        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._pass, name)
