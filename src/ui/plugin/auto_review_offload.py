"""Hand the review's shape refine to a thread, and fold the answers back.

The Shapes step's controls describe the WHOLE set, so moving one re-shapes
every object of the run. The cooperative pump that does it yields to the event
loop between slices, and QGIS keeps answering, but it answers out of the gaps:
the thread that draws the map is also the thread doing the shaping, for as long
as the pass takes. On a run with tens of thousands of objects that is long
enough that the user cannot pan, zoom or move the control while it converges.

This mixin moves the shaping to workers.review_refine_thread. The GUI keeps the
cheap half (the filters, the ground measure that reads the project, picking the
refiner) and hands over one job per object; the answers land in the same refine
cache the pump already reads, so nothing downstream changes shape or order.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py). Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog

# Under this many objects the whole pass lands inside a couple of the pump's own
# slices, so handing each object to a thread costs more than it saves. Over it
# the pass owns the GUI thread for longer than a user will sit still for.
_REVIEW_OFFLOAD_MIN_OBJECTS = 1000

# How many objects may be waiting on the thread at once. Each one is a cloned
# geometry the GUI is holding on the thread's behalf, so the ceiling is what
# stops a dense run from copying its whole object set into a queue; the pump
# tops the queue back up every slice, so the thread never runs dry.
_REVIEW_OFFLOAD_QUEUE_MAX = 3000


class AutoReviewOffloadMixin:
    """Start, feed, drain and tear down the review's off-GUI refine thread."""

    def _review_refine_thread_for(self, pending_count: int):
        """The refine thread to use for a pass of ``pending_count`` objects, or
        None when this pass should stay on the GUI thread.

        Starting it is lazy and idempotent: a review that never reaches the size
        floor never creates a thread at all, which is most reviews.
        """
        if int(pending_count or 0) < _REVIEW_OFFLOAD_MIN_OBJECTS:
            return None
        thread = getattr(self, "_review_refine_thread", None)
        if thread is not None:
            try:
                if thread.isRunning():
                    return thread
            except RuntimeError:
                pass  # the C++ half went: build a fresh one below
            self._review_refine_thread = None
        try:
            from ...workers.review_refine_thread import ReviewRefineThread

            thread = ReviewRefineThread()
            thread.start()
        except Exception as exc:  # noqa: BLE001 -- the GUI path still works
            QgsMessageLog.logMessage(
                f"Auto review: shaping stays on the interface thread ({exc})",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
            self._review_refine_thread = None
            self._review_refine_inflight = {}
            return None
        self._review_refine_thread = thread
        self._review_refine_inflight = {}
        return thread

    def _stop_review_refine_thread(self) -> None:
        """Drop the refine thread, whatever it was doing.

        Called by every path that abandons the objects it is shaping: the run
        pipeline reset, the refine-cache reset (export, discard, a rebuilt
        object set), and the review discard. Idempotent, and never nulls a live
        QThread without parking it: garbage-collecting a running QThread aborts
        QGIS.
        """
        thread = getattr(self, "_review_refine_thread", None)
        self._review_refine_thread = None
        self._review_refine_inflight = {}
        if thread is None:
            return
        try:
            from ...workers.review_refine_thread import (
                REVIEW_REFINE_JOIN_TIMEOUT_MS,
            )

            thread.abort()
            if not thread.join_run(REVIEW_REFINE_JOIN_TIMEOUT_MS):
                from .shared import park_orphaned_worker

                park_orphaned_worker(thread)
        except RuntimeError:
            pass  # the C++ half is already gone
        except Exception:  # noqa: BLE001 -- teardown must never propagate  # nosec B110
            pass

    def _offload_review_refine(self, thread, det_idx: int, base,
                               params: dict, pixel_size: float,
                               stamp) -> bool:
        """Hand ONE object to the refine thread. True when it took it.

        False means the caller must shape the object itself: the queue is full,
        the thread is winding down, or the object carries no position to measure
        the run's ground dials at. Falling back is always safe, and is exactly
        what the review did before this thread existed.

        ``stamp`` is the refine cache's current shape key. A change to it drops
        whatever is still queued for the settings the user has moved off, so a
        second nudge is served promptly instead of queueing behind a pass whose
        answers nobody wants.
        """
        inflight = getattr(self, "_review_refine_inflight", None)
        if not isinstance(inflight, dict):
            inflight = {}
            self._review_refine_inflight = inflight
        if inflight.get(det_idx) == stamp:
            return True  # already on its way under these very settings
        if getattr(self, "_review_refine_stamp", None) != stamp:
            # The settings moved. Everything queued describes the old ones.
            self._review_refine_stamp = stamp
            inflight.clear()
            try:
                thread.set_live_stamp(stamp)
            except (AttributeError, RuntimeError):
                return False
        if len(inflight) >= _REVIEW_OFFLOAD_QUEUE_MAX:
            return False
        # One object may carry its own Simplify / Right angles (the Correct
        # step's per-shape settings), merged the same way _review_refined_geom
        # merges them, so the two paths shape it identically.
        per_shape = getattr(self, "_shape_params_for_object", None)
        if per_shape is not None:
            params = per_shape(det_idx, params)
        refiner = self._review_refiner_for(base, params, pixel_size)
        if refiner is None:
            return False
        try:
            if not thread.submit(det_idx, stamp, refiner, base):
                return False
        except RuntimeError:
            return False
        inflight[det_idx] = stamp
        return True

    def _drain_review_refine_results(self, stamp) -> int:
        """Fold every finished shape into the refine cache. Returns how many
        landed under ``stamp``.

        An answer whose stamp is not the cache's current key is dropped: it was
        computed for settings the user has since moved off, and writing it would
        put a shape on the map that no control on screen describes.
        """
        thread = getattr(self, "_review_refine_thread", None)
        if thread is None:
            return 0
        try:
            results = thread.take_results()
        except RuntimeError:
            self._review_refine_thread = None
            self._review_refine_inflight = {}
            return 0
        if not results:
            return 0
        inflight = getattr(self, "_review_refine_inflight", None)
        if not isinstance(inflight, dict):
            inflight = {}
            self._review_refine_inflight = inflight
        cache = getattr(self, "_auto_reslice_cache", None)
        geoms = cache.get("geoms") if isinstance(cache, dict) else None
        current = cache.get("key") if isinstance(cache, dict) else None
        landed = 0
        for det_idx, job_stamp, geom, err in results:
            if inflight.get(det_idx) == job_stamp:
                inflight.pop(det_idx, None)
            if err is not None:
                self._log_review_refine_failure(det_idx, err)
            if job_stamp != stamp or job_stamp != current or geoms is None:
                continue
            geoms[det_idx] = geom
            landed += 1
        return landed

    def _review_refine_offload_for(self, state: dict, filter_pending: list,
                                   awaiting: list):
        """The off-GUI refine thread this pass may use, or None to shape on the
        GUI thread as before.

        Reslice only. The finalize pass runs behind its own wait screen, has the
        live stitcher's shapes seeded into the cache already, and carries the
        drain, the watchdog and the archive around it; the freeze this thread
        answers is the review's, where the user is looking at the map and moving
        a control. Resolved once per pass and remembered on the state, so a
        thread is never started twice for one pass and never started at all for
        a review small enough to shape between two drawn frames.
        """
        if state.get("mode") != "reslice":
            return None
        if "offload" in state:
            return state["offload"]
        thread = self._review_refine_thread_for(len(filter_pending) + len(awaiting))
        state["offload"] = thread
        return thread

    def _accept_review_shape(self, state: dict, det_idx: int, geom,
                             score: float, manual: bool, size_gate_on: bool,
                             measurer) -> None:
        """Put one finished shape into the pass's visible set, or account for
        why it is not there.

        Shared by the three ways an object reaches its shape (already cached,
        answered by the refine thread, shaped inline), so all three apply the
        same size gate and fill the same four parallel lists.
        """
        if geom is None:
            # An object that passed every filter and came back with no shape
            # leaves the map and the export. Count it: a total that quietly
            # disagrees with the map is the one thing the review cannot explain.
            state["refine_dropped"] = int(state.get("refine_dropped", 0) or 0) + 1
            return
        if not (manual or not size_gate_on
                or self._passes_size_filters(
                    self._object_area_m2(geom, measurer), state["params"])):
            return
        state["visible"].append(geom)
        state["visible_scores"].append(score)
        state["visible_ids"].append(self._object_fid_for(det_idx))
        state["visible_order"].append(det_idx)

    def _review_refine_queue_full(self) -> bool:
        """Whether the refine thread is holding all the work it should.

        The ceiling bounds the geometry the GUI has cloned on the thread's
        behalf, and it is also the pump's signal to stop and come round again
        rather than shape the object itself. False whenever there is no live
        thread, so nothing can wait on a queue that will never drain.
        """
        if not self._review_refine_thread_alive():
            return False
        inflight = getattr(self, "_review_refine_inflight", None)
        if not isinstance(inflight, dict):
            return False
        return len(inflight) >= _REVIEW_OFFLOAD_QUEUE_MAX

    def _review_refine_thread_alive(self) -> bool:
        """Whether the refine thread is still there to answer what it was
        handed. False means the caller must shape those objects itself."""
        thread = getattr(self, "_review_refine_thread", None)
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except RuntimeError:
            self._review_refine_thread = None
            return False

    def _review_shape_now(self, det_idx: int, params: dict, pixel_size: float):
        """Shape one object on THIS thread, by canonical index.

        The way back when the refine thread cannot answer for an object it was
        handed. Returns None when the index names no object, which the caller
        accounts for exactly like a shape that came back empty."""
        objects = getattr(self, "_auto_objects", None) or []
        if not 0 <= det_idx < len(objects):
            return None
        base = objects[det_idx][0]
        if base is None or base.isEmpty():
            return None
        return self._review_refined_geom(det_idx, base, params, pixel_size)

    def _review_refine_outstanding(self) -> int:
        """Objects the refine thread still owes an answer for.

        Counted on the GUI side, not on the thread: a job the thread skips for
        naming stale settings never comes back, so a thread-side counter would
        only ever drift up."""
        if getattr(self, "_review_refine_thread", None) is None:
            return 0
        inflight = getattr(self, "_review_refine_inflight", None)
        return len(inflight) if isinstance(inflight, dict) else 0
