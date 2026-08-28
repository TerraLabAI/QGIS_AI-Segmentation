
















from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog

from ...core.server_dials import dial_in_range






_REVIEW_OFFLOAD_MIN_OBJECTS = 300





_REVIEW_OFFLOAD_QUEUE_MAX = 3000


def _review_offload_queue_max() -> int:


    return dial_in_range(
        "tuning.review.offload_queue_max", _REVIEW_OFFLOAD_QUEUE_MAX, 100, 20000)


class AutoReviewOffloadMixin:





    def _review_refine_thread_for(self, pending_count: int):






        if int(pending_count or 0) < dial_in_range(
                "tuning.review.offload_min_objects", _REVIEW_OFFLOAD_MIN_OBJECTS, 1, 100000):
            return None
        thread = getattr(self, "_review_refine_thread", None)
        if thread is not None:
            try:
                if thread.isRunning():
                    return thread
            except RuntimeError:
                pass
            self._review_refine_thread = None
        thread = self._start_review_refine_pool(int(pending_count or 0))
        if thread is None:
            try:
                from ...workers.review_refine_thread import ReviewRefineThread

                thread = ReviewRefineThread()
                thread.start()
            except Exception as exc:  # noqa: BLE001
                QgsMessageLog.logMessage(
                    f"Auto review: shaping stays on the interface thread ({exc})",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)


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











        try:
            from ...core.server_dials import dial_in_range
            from ...workers.review_refine_pool import (
                DEFAULT_MIN_OBJECTS,
                ReviewRefineProcessPool,
                pool_children,
            )
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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










        from .review_align_offload import stop_align_thread
        from .review_gap_fill import stop_review_set_thread

        stop_review_set_thread(self)
        stop_align_thread(self)
        thread = getattr(self, "_review_refine_thread", None)
        self._review_refine_thread = None
        self._review_refine_inflight = {}



        self._review_refine_stamp = None
        if thread is None:
            return
        try:
            from ...workers.review_refine_thread import (
                REVIEW_REFINE_JOIN_TIMEOUT_MS,
            )

            thread.abort()
            if not thread.join_run(REVIEW_REFINE_JOIN_TIMEOUT_MS):



                from qgis.PyQt.QtCore import QThread

                if isinstance(thread, QThread):
                    from .shared import park_orphaned_worker

                    park_orphaned_worker(thread)
        except RuntimeError:
            pass
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _offload_review_refine(self, thread, det_idx: int, base,
                               params: dict, pixel_size: float,
                               stamp) -> bool:

















        inflight = getattr(self, "_review_refine_inflight", None)
        if not isinstance(inflight, dict):
            inflight = {}
            self._review_refine_inflight = inflight
        seq = self._review_refine_seq_of(det_idx)
        if inflight.get(det_idx) == (stamp, seq):
            return True
        if getattr(self, "_review_refine_stamp", None) != stamp:

            self._review_refine_stamp = stamp
            inflight.clear()
            try:
                thread.set_live_stamp(stamp)
            except (AttributeError, RuntimeError):
                return False
        if len(inflight) >= _review_offload_queue_max():
            return False



        per_shape = getattr(self, "_shape_params_for_object", None)
        shared_params = params
        if per_shape is not None:
            params = per_shape(det_idx, params)





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


        seqs = getattr(self, "_review_refine_seq", None)
        if not isinstance(seqs, dict):
            return 0
        return int(seqs.get(int(det_idx), 0))

    def _bump_review_refine_seq(self, indices) -> None:






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



                continue
            if job_stamp != stamp or job_stamp != current or geoms is None:
                continue
            geoms[det_idx] = geom
            self._forget_review_object_area(det_idx)
            landed += 1
        return landed

    def _review_object_area(self, det_idx: int, geom, measurer) -> float:









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

        cache = getattr(self, "_auto_reslice_cache", None)
        if not isinstance(cache, dict):
            return
        areas = cache.get("areas")
        if isinstance(areas, dict):
            areas.pop(det_idx, None)

    def _review_refine_offload_for(self, state: dict, filter_pending: list,
                                   awaiting: list):
















        if "offload" in state:
            held = state["offload"]
            if held is None:
                return None



            if self._review_refine_thread_alive():
                return held
            state["offload"] = None
            return None



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







        if geom is None:



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







        if not self._review_refine_thread_alive():
            return False
        inflight = getattr(self, "_review_refine_inflight", None)
        if not isinstance(inflight, dict):
            return False
        return len(inflight) >= _review_offload_queue_max()

    def _review_refine_thread_alive(self) -> bool:


        thread = getattr(self, "_review_refine_thread", None)
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except RuntimeError:
            self._review_refine_thread = None
            return False

    def _review_shape_now(self, det_idx: int, params: dict, pixel_size: float):





        objects = getattr(self, "_auto_objects", None) or []
        if not 0 <= det_idx < len(objects):
            return None
        base = objects[det_idx][0]
        if base is None or base.isEmpty():
            return None
        return self._review_refined_geom(det_idx, base, params, pixel_size)
