"""A stepped pass run on its own thread, behind the same step protocol.

The finalize pump drives a whole-set pass by calling ``step(count)`` until it
answers True, then reads ``result()``. On the interface thread one step lasts
as long as its slowest geometry call, and a union over a chunk of thousands of
vertices holds the map for over a second however small the slice. The QGIS
bindings release the interpreter lock inside those calls, so a plain thread
runs the very same pass while the interface keeps drawing.

``SteppedPassThread`` wraps any object with ``step(count) -> bool`` and
``result()``. Its own ``step`` does no work: it waits one short slice for the
worker and reports whether the pass has ended, so the pump's loop and budget
stay exactly as they are. Counters and everything else forward to the pass,
so a caller that summarises the pass afterwards cannot tell the difference.

The pass must only touch what it was built with. When the caller's own
objects stay in use on the interface thread, hand the pass copies and pass
both lists here: ``result`` gives the caller's own object back wherever the
pass left its copy untouched, so identities downstream are what they were.

No Qt: the pump polls it, nothing queued can reach a torn-down plugin, and
``stop`` is how every abandon path ends it.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

#: How long one ``step`` call waits for the worker before handing the
#: interface back. Under the pump's own turn budget, so a turn ends on time.
STEP_WAIT_S = 0.010

#: How long ``stop`` waits for a worker mid-call to come out of it.
STOP_JOIN_TIMEOUT_S = 3.0


class SteppedPassThread:
    """``pass_`` stepped to the end on a daemon thread, driven from the pump
    by the same ``step`` / ``result`` calls the pass itself offers."""

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

    # ---- worker ---------------------------------------------------------------

    def start(self) -> SteppedPassThread:
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if self._pass.step(self._count):
                    self._finished = True
                    break
        except BaseException as exc:  # noqa: BLE001 -- reported through result()
            self._error = exc
            logger.info("SteppedPassThread: pass failed: %r", exc)
        finally:
            self._done.set()

    # ---- the pump's side --------------------------------------------------------

    def step(self, count: int = 64) -> bool:
        """Wait one slice for the worker. True once the pass has ended."""
        if self._done.is_set():
            return True
        return self._done.wait(STEP_WAIT_S)

    def result(self) -> Any:
        """The pass's answer, or None while it runs, after a stop, or when
        the pass raised."""
        if not self._done.is_set() or not self._finished or self._error is not None:
            return None
        return self._map_back(self._pass.result())

    def stop_and_take(self, timeout_s: float = STOP_JOIN_TIMEOUT_S) -> Any:
        """End the pass where it stands and take what it has, or None when the
        worker would not come out of its call.

        For a caller that runs the pass under a time budget. A stepped pass
        answers at every step boundary (whatever it has not reached keeps the
        input it was given), and ``stop`` leaves the worker at one, so a pass
        cut short still hands back a whole list."""
        self.stop(timeout_s)
        if self._thread.is_alive() or self._error is not None:
            return None
        return self._map_back(self._pass.result())

    def _map_back(self, out: Any) -> Any:
        """The pass's list with the caller's own objects back wherever the
        pass left its copy untouched."""
        if (self._inputs is None or self._originals is None
                or not isinstance(out, list) or len(out) != len(self._inputs)):
            return out
        return [orig if item is copy else item
                for item, copy, orig in zip(out, self._inputs, self._originals)]

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self, timeout_s: float = STOP_JOIN_TIMEOUT_S) -> None:
        """End the pass at its next step and wait for the thread to leave."""
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout_s)

    def __getattr__(self, name: str) -> Any:
        # Counters and flags the caller reads off the pass once it has ended.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._pass, name)
