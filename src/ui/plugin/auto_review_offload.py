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
# the pass owns the GUI thread for longer than a user will sit still for. The
# count that is compared against it excludes objects the refine cache already
# holds, because those cost a dict lookup and never reach the thread.
_REVIEW_OFFLOAD_MIN_OBJECTS = 300

# How many objects may be waiting on the thread at once. Each one is a cloned
# geometry the GUI is holding on the thread's behalf, so the ceiling is what
# stops a dense run from copying its whole object set into a queue; the pump
# tops the queue back up every slice, so the thread never runs dry.
_REVIEW_OFFLOAD_QUEUE_MAX = 3000


class AutoReviewOffloadMixin:
    """Start, feed, drain and tear down the review's off-GUI refine: child
    interpreters (workers.review_refine_pool) when they come up, the thread
    (workers.review_refine_thread) when they do not. Both answer the same
    calls, so everything below holds either under one name."""

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
        thread = self._start_review_refine_pool(int(pending_count or 0))
        if thread is None:
            try:
                from ...workers.review_refine_thread import ReviewRefineThread

                thread = ReviewRefineThread()
                thread.start()
            except Exception as exc:  # noqa: BLE001 -- the GUI path still works
                QgsMessageLog.logMessage(
                    f"Auto review: shaping stays on the interface thread ({exc})",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
                # Shaping on the GUI thread freezes QGIS for the pass: the
                # user gets the shapes, and a frozen window with them.
                try:
                    from ...core.telemetry_errors import track_plugin_error
                    track_plugin_error(stage="segment",
                                       error_code="review_refine_thread_failed",
                                       message=type(exc).__name__)
                except Exception:  # noqa: BLE001  # nosec B110
                    pass
                self._review_refine_thread = None
                self._review_refine_inflight = {}
                return None
        self._review_refine_thread = thread
        self._review_refine_inflight = {}
        return thread

    def _start_review_refine_pool(self, pending_count: int):
        """Child interpreters for a pass of ``pending_count`` objects, or None
        to use the thread.

        The refine is fine-grained Python over vertices, so threads share one
        lock and cannot shorten it; child interpreters can (see
        workers.review_refine_pool). Sized there like the converter pool and
        refused on a small machine or a small set; the served floor says how
        small, and 0 turns the pool off. Every refusal (no interpreter, a
        machine that will not spawn) falls back to the thread, which is the
        behaviour before this pool existed.
        """
        try:
            from ...core.server_dials import dial_in_range
            from ...workers.review_refine_pool import (
                DEFAULT_MIN_OBJECTS,
                ReviewRefineProcessPool,
                pool_children,
            )
        except Exception:  # noqa: BLE001 -- the thread path is still there
            return None
        try:
            floor = int(dial_in_range(
                "review.refine_pool_min_objects", DEFAULT_MIN_OBJECTS, 0, 1000000))
        except Exception:  # noqa: BLE001
            floor = DEFAULT_MIN_OBJECTS
        if floor <= 0:
            return None
        try:
            children = pool_children(pending_count, floor)
            if children <= 0:
                return None
            pool = ReviewRefineProcessPool(workers=children)
            if not pool.start():
                return None
        except Exception:  # noqa: BLE001 -- a refusal is the thread's cue
            return None
        QgsMessageLog.logMessage(
            f"Auto review: shaping {pending_count} object(s) on {children} "
            "child interpreter(s)", "AI Segmentation",
            level=Qgis.MessageLevel.Info)
        try:
            from .auto_client_profile import review_pass_profile
            review_pass_profile(self)["on_pool"] = True
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return pool

    def _stop_review_refine_thread(self) -> None:
        """Drop the refine thread, whatever it was doing.

        Called by every path that abandons the objects it is shaping: the run
        pipeline reset, the refine-cache reset (export, discard, a rebuilt
        object set), and the review discard. Idempotent, and never nulls a live
        QThread without parking it: garbage-collecting a running QThread aborts
        QGIS.
        """
        # The two whole-set passes ride the same abandon paths: both hold
        # copies of the shapes this call gives up on.
        from .review_align_offload import stop_align_thread
        from .review_gap_fill import stop_review_set_thread

        stop_review_set_thread(self)
        stop_align_thread(self)
        thread = getattr(self, "_review_refine_thread", None)
        self._review_refine_thread = None
        self._review_refine_inflight = {}
        # The stamp names the settings the GONE thread was told to work on. Left
        # standing, the next thread is never told its own live stamp, because
        # the first object it is handed reads the stamp as unchanged.
        self._review_refine_stamp = None
        if thread is None:
            return
        try:
            from ...workers.review_refine_thread import (
                REVIEW_REFINE_JOIN_TIMEOUT_MS,
            )

            thread.abort()
            if not thread.join_run(REVIEW_REFINE_JOIN_TIMEOUT_MS):
                # Parking waits on a QThread's finished signal. The process
                # pool has none: its children are killed, and one the system
                # has not reaped yet is left to it.
                from qgis.PyQt.QtCore import QThread

                if isinstance(thread, QThread):
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

        The job also carries the object's edit sequence. A merge or a split
        rewrites one object without touching a single shape control, so the
        stamp alone cannot tell the answer for the shape BEFORE the edit from
        the answer for the shape after it.
        """
        inflight = getattr(self, "_review_refine_inflight", None)
        if not isinstance(inflight, dict):
            inflight = {}
            self._review_refine_inflight = inflight
        seq = self._review_refine_seq_of(det_idx)
        if inflight.get(det_idx) == (stamp, seq):
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
        shared_params = params
        if per_shape is not None:
            params = per_shape(det_idx, params)
        # ``stamp`` IS the shape key of the shared params, and the caller walked
        # a dozen dials to build it once for the whole pass. Hand it over rather
        # than rebuilding the same tuple for every object of the set. An object
        # carrying its own Shape settings is a different identity, so it pays
        # for its own key, exactly as it does on the inline path.
        refiner = self._review_refiner_for(
            base, params, pixel_size,
            shape_key=stamp if params is shared_params else None)
        if refiner is None:
            return False
        try:
            if not thread.submit(det_idx, stamp, refiner, base, seq=seq,
                                 spec=(params, pixel_size)):
                return False
        except RuntimeError:
            return False
        inflight[det_idx] = (stamp, seq)
        return True

    def _review_refine_seq_of(self, det_idx: int) -> int:
        """This object's edit sequence: how many times a hand edit has rewritten
        it during this review. 0 until the first one."""
        seqs = getattr(self, "_review_refine_seq", None)
        if not isinstance(seqs, dict):
            return 0
        return int(seqs.get(int(det_idx), 0))

    def _bump_review_refine_seq(self, indices) -> None:
        """Retire every answer still in flight for ``indices``.

        Called where a hand edit rewrites objects. The shape key does not move
        (no control was touched), so without this the answer computed for the
        shape BEFORE the edit lands afterwards and puts it back on the map.
        """
        seqs = getattr(self, "_review_refine_seq", None)
        if not isinstance(seqs, dict):
            seqs = {}
            self._review_refine_seq = seqs
        inflight = getattr(self, "_review_refine_inflight", None)
        for idx in indices:
            key = int(idx)
            seqs[key] = seqs.get(key, 0) + 1
            if isinstance(inflight, dict):
                inflight.pop(key, None)

    def _drain_review_refine_results(self, stamp) -> int:
        """Fold every finished shape into the refine cache. Returns how many
        landed under ``stamp``.

        An answer whose stamp is not the cache's current key is dropped: it was
        computed for settings the user has since moved off, and writing it would
        put a shape on the map that no control on screen describes. So is an
        answer whose edit sequence has moved on, which is what a hand edit does
        without touching any control.
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
        for det_idx, job_stamp, job_seq, geom, err in results:
            expected = inflight.get(det_idx)
            fresh = expected == (job_stamp, job_seq)
            if fresh:
                inflight.pop(det_idx, None)
            if err is not None:
                self._log_review_refine_failure(det_idx, err)
            if not fresh:
                # Nobody is waiting for this shape any more: a hand edit
                # rewrote the object after the job went out. Writing it would
                # undo the edit on the map.
                continue
            if job_stamp != stamp or job_stamp != current or geoms is None:
                continue
            geoms[det_idx] = geom
            self._forget_review_object_area(det_idx)
            landed += 1
        return landed

    def _review_object_area(self, det_idx: int, geom, measurer) -> float:
        """Ground area of one refined object, memoised beside the refine cache.

        The size gate measures every visible object on every pass, including
        the passes that changed nothing about its shape (a confidence move, a
        size dial move). Measuring is a full walk of the geometry, so on a dense
        result it was the cost of a filter-only reslice. The memo is dropped
        with the refine cache and wherever a shape is rewritten, so it can never
        describe a geometry that has gone.
        """
        cache = getattr(self, "_auto_reslice_cache", None)
        if not isinstance(cache, dict):
            return self._object_area_m2(geom, measurer)
        areas = cache.get("areas")
        if not isinstance(areas, dict):
            areas = {}
            cache["areas"] = areas
        hit = areas.get(det_idx)
        if hit is not None:
            return hit
        area = self._object_area_m2(geom, measurer)
        areas[det_idx] = area
        return area

    def _forget_review_object_area(self, det_idx: int) -> None:
        """Drop one object's memoised area, because its shape has changed."""
        cache = getattr(self, "_auto_reslice_cache", None)
        if not isinstance(cache, dict):
            return
        areas = cache.get("areas")
        if isinstance(areas, dict):
            areas.pop(det_idx, None)

    def _review_refine_offload_for(self, state: dict, filter_pending: list,
                                   awaiting: list):
        """The off-GUI refine thread this pass may use, or None to shape on the
        GUI thread as before.

        Both passes. The reslice needed it first, because there the user is
        looking at the map while a control moves. The finalize needs it for the
        same reason with a worse case: it shapes the WHOLE run, the seeding that
        was meant to spare it only lands when the run's own shapes survived the
        merge, and when it does not the shaping is the longest thing between the
        last tile and the review, on the thread that draws.

        Resolved once per pass and remembered on the state, so a thread is never
        started twice for one pass and never started at all for a set small
        enough to shape between two drawn frames. The one it starts is left
        running for the review that follows, which is where the next pass
        would have paid to start it.
        """
        if "offload" in state:
            held = state["offload"]
            if held is None:
                return None
            # A thread can die under a pass (a teardown, a cache reset). The
            # pass held the object either way and kept handing it work that
            # nothing would ever answer.
            if self._review_refine_thread_alive():
                return held
            state["offload"] = None
            return None
        # Objects the cache already holds cost a dict lookup and never reach
        # the thread, so counting them decided the question on work that is
        # not there. A filter-only reslice over a warm cache has none.
        cache = getattr(self, "_auto_reslice_cache", None)
        cached = cache.get("geoms") if isinstance(cache, dict) else None
        if isinstance(cached, dict) and cached:
            pending = sum(1 for row in filter_pending if row[0] not in cached)
        else:
            pending = len(filter_pending)
        thread = self._review_refine_thread_for(pending + len(awaiting))
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
                    self._review_object_area(det_idx, geom, measurer),
                    state["params"])):
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
