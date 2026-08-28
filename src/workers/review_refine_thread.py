"""Off-GUI shape refine for the Automatic review's Shapes step.

Moving a Shape control re-shapes every object of the run, because the controls
describe the whole set. On a dense run that is tens of thousands of GEOS and
Python passes, and until this thread existed they all ran on the GUI thread in
20 ms cooperative slices. The slices kept QGIS answering, but only in the gaps
between them: the canvas got about four fifths of the frames it asked for while
a pass converged, and the pass itself owned the thread that draws the map for as
long as it took.

This thread takes the same work instead. The GUI picks the refiner (that reads
the project, so it stays where the project is), hands over one job per object,
and folds the answers back into the review's refine cache as they land. The
event loop then pays a dict write per object and nothing else.

Ownership is per job, not shared: the GUI clones each geometry before handing
it over, and the refiner is read-only once built, so several jobs may share one.

The thread carries NO Qt signals on purpose. The review polls it from the
cooperative pump it already runs, which means there is no queued cross-thread
emission that could be delivered to a plugin that has since been torn down.
"""
from __future__ import annotations

import logging
import queue
import threading

from qgis.PyQt.QtCore import QThread

logger = logging.getLogger(__name__)

#: How long a teardown waits for the thread to leave the object it is shaping.
#: One object is bounded work with no network in it, so this only ever has to
#: cover the slowest single shape.
REVIEW_REFINE_JOIN_TIMEOUT_MS = 3000


def detached_review_geom(geom):
    """A geometry that shares no data with ``geom``.

    ``QgsGeometry(other)`` is a shallow copy: both wrappers point at the same
    abstract geometry, so it is not enough to hand one across a thread. Cloning
    the inner geometry is.
    """
    from qgis.core import QgsGeometry

    inner = geom.constGet()
    if inner is None:
        return QgsGeometry(geom)
    return QgsGeometry(inner.clone())


class ReviewRefineThread(QThread):
    """Shapes review objects off the GUI thread: jobs in, finished shapes out.

    Created and started by the review when a pass is big enough to be worth a
    thread, fed one job per object, and dropped through ``abort`` plus
    ``join_run`` by every path that abandons the review.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._inbox: queue.Queue = queue.Queue()
        self._outbox: list = []
        self._lock = threading.Lock()
        self._aborted = False
        self._stopping = False
        # The shape settings the GUI is currently waiting on. A job stamped with
        # anything else is skipped without being shaped: it describes settings
        # the user has moved off, and the queue may hold a whole run of them.
        # One plain attribute, written by the GUI and read by the thread, which
        # needs no lock because a Python attribute assignment is atomic.
        self._live_stamp = None

    # ---- GUI-thread API -----------------------------------------------------

    def set_live_stamp(self, stamp) -> None:
        """Name the shape settings worth spending time on from now on.

        Everything already queued under a different stamp is dropped as the
        thread reaches it, so a control moved mid-pass is served at once
        instead of queueing behind answers nobody will read.
        """
        self._live_stamp = stamp

    def submit(self, det_idx: int, stamp, refiner, geom) -> bool:
        """Hand one object over. Returns at once; False when the thread is
        winding down and the caller must shape it itself.

        ``stamp`` is the refine cache's shape key at the moment of the hand
        over. It comes back untouched, which is what lets the review drop an
        answer computed for settings the user has since moved off.
        """
        if self._aborted or self._stopping:
            return False
        try:
            job = (int(det_idx), stamp, refiner, detached_review_geom(geom))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
        self._inbox.put(job)
        return True

    def take_results(self) -> list:
        """Every shape finished since the last call, oldest first.

        Each entry is ``(det_idx, stamp, geometry, error)``; ``geometry`` is
        None when even the repair left nothing, and ``error`` is the exception
        the refine raised, for the caller to log. Called from the GUI thread
        only.
        """
        with self._lock:
            out = self._outbox
            self._outbox = []
        return out

    def finish(self) -> None:
        """Ask the thread to stop once it has shaped everything queued.

        Idempotent: the cooperative wait calls it once per slice.
        """
        if self._stopping:
            return
        self._stopping = True
        self._inbox.put(None)

    def abort(self) -> None:
        """Drop whatever is queued and stop now (teardown, discard, a later
        pass superseding this one).

        Answers already in the outbox are left there. Nothing reads them: the
        caller that aborts also drops the thread, and a stale answer would fail
        its stamp check anyway.
        """
        self._aborted = True
        if self._stopping:
            return
        self._stopping = True
        self._inbox.put(None)

    def join_run(self, timeout_ms: int = REVIEW_REFINE_JOIN_TIMEOUT_MS) -> bool:
        """Block until the thread has exited. True when it did.

        Never called from the thread itself. A False return means it is wedged
        inside one shape, which the caller must treat as "park it, never
        delete it": destroying a running QThread aborts QGIS.
        """
        if not self.isRunning():
            return True
        return bool(self.wait(timeout_ms))

    # ---- Refine-thread body -------------------------------------------------

    def run(self) -> None:  # noqa: D102 - QThread entry point
        from ..core.live_refine import refine_review_geom

        try:
            while True:
                item = self._inbox.get()
                if item is None:
                    break
                if self._aborted:
                    continue
                det_idx, stamp, refiner, geom = item
                live = self._live_stamp
                if live is not None and stamp != live:
                    # Stale settings. Answer nothing: the GUI clears its own
                    # in-flight record when it moves the stamp, so a dropped
                    # job is not one it is still waiting for.
                    continue
                try:
                    shaped, err = refine_review_geom(refiner, geom)
                except Exception as exc:  # noqa: BLE001 - one bad object only
                    shaped, err = None, exc
                with self._lock:
                    self._outbox.append((det_idx, stamp, shaped, err))
        except Exception:  # noqa: BLE001 - the thread body must never raise out
            logger.warning("ReviewRefineThread: stopped on error", exc_info=True)
