






from __future__ import annotations

import math

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr
from ...core.review_defaults import (
    AUTO_REVIEW_SIMPLIFY_DEFAULT as _AUTO_REVIEW_SIMPLIFY_DEFAULT,
)
from ...core.server_dials import dial_in_range
from ...core.shape_policy_dials import align_phase_budget_s
from .auto_results import _stitch_drain_budget_s, _stitch_join_timeout_ms
from .shared import auto_pump_budget







_ALIGN_PHASE_BUDGET_S = 10.0





_AWAIT_REFINE_POLL_MS = 15


class AutoFinalizeStepsMixin:


    def _auto_offgui_poll_ms(self) -> int:



        return int(dial_in_range(
            "tuning.auto.await_refine_poll_ms", _AWAIT_REFINE_POLL_MS, 1, 200))

    def _announce_auto_finalize_phase(self, state: dict) -> None:

        if state.get("mode") == "reslice":
            return
        phase = str(state.get("phase") or "")
        if phase == state.get("announced_phase"):
            return
        labels = {
            "drain": tr("Finishing the last tiles"),
            "remerge": tr("Joining the detected parts"),
            "sweep": tr("Removing duplicate fragments"),
            "align": tr("Aligning neighbouring outlines"),
            "build": tr("Building the shapes"),
            "filter": tr("Applying the review settings"),
            "snap": tr("Joining shared borders"),
        }
        text = labels.get(phase)
        if not text or self.dock_widget is None:
            return
        try:
            self.dock_widget.set_auto_finalize_phase(text)
            state["announced_phase"] = phase
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _step_auto_finalize_refine(self) -> None:









        try:
            self._step_auto_finalize_refine_impl()
        except Exception as exc:  # noqa: BLE001
            self._abort_auto_finalize(exc)

    def _log_finalize_drop(self, state: dict, phase: str, idx,
                           exc: Exception) -> None:



        cap = state.get("drop_log_max")
        if cap is None:
            cap = dial_in_range("tuning.auto.finalize_drop_log_max", 5, 1, 50)
            state["drop_log_max"] = cap
        dropped = int(state.get("dropped_objects", 0) or 0) + 1
        state["dropped_objects"] = dropped
        if dropped <= cap:
            try:
                QgsMessageLog.logMessage(
                    f"Auto detection: dropped object {idx} in the {phase} "
                    f"phase ({exc})",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
            except Exception:  # nosec B110
                pass



        tracked = state.setdefault("drop_tracked", set())
        if phase not in tracked:
            tracked.add(phase)
            try:
                from ...core.telemetry_errors import track_plugin_error
                track_plugin_error(stage="segment",
                                   error_code=f"finalize_drop_{phase}",
                                   message=type(exc).__name__)
            except Exception:  # noqa: BLE001  # nosec B110
                pass

    def _abort_auto_finalize(self, exc: Exception) -> None:









        state = self._auto_finalize_state
        mode = (state or {}).get("mode") or (
            "reslice" if self._auto_review is not None else "finalize")
        self._auto_finalize_state = None
        self._pop_nothing_found_notice()
        try:
            QgsMessageLog.logMessage(
                f"Auto detection: {mode} aborted: {exc!r}",
                "AI Segmentation", level=Qgis.MessageLevel.Critical)
        except Exception:  # nosec B110
            pass
        try:
            from ...core.telemetry_errors import report_exception
            report_exception(exc, stage="segment", module="auto_results")
        except Exception:  # nosec B110
            pass
        if mode == "reslice":

            if self.dock_widget:
                try:
                    self.dock_widget.set_auto_status(
                        "error",
                        tr("Could not apply the new settings. "
                           "Try a different value."))
                except (RuntimeError, AttributeError):
                    pass
            return
        try:
            self._reset_auto_live_pipeline()
        except Exception:  # nosec B110
            pass
        try:
            self._remove_auto_selection_layer()
        except Exception:  # nosec B110
            pass


        recovered = None
        try:
            from ...core import run_autosave
            run_id = self._auto_run_id or ""
            pending = run_autosave.read_pending()
            if (pending and run_id and str(pending.get("run_id") or "") == run_id):
                recovered = run_autosave.load_pending_layer(pending)
                if recovered:


                    run_autosave.clear_pending(run_id)
        except Exception as recover_exc:  # noqa: BLE001



            try:
                from ...core.telemetry_errors import track_plugin_error
                track_plugin_error(stage="segment",
                                   error_code="finalize_autosave_recovery_failed",
                                   message=type(recover_exc).__name__)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        if recovered:
            msg = tr("Something went wrong preparing the results. Your "
                     "detections were saved to the layer {name}.").format(
                name=recovered)
        else:
            msg = tr("Something went wrong preparing the results. "
                     "Please run Detect again.")
        self._last_auto_result = {"status": "error", "message": str(exc)[:200]}
        if self.dock_widget:
            try:


                self.dock_widget.set_auto_finalizing(False)
                self.dock_widget.set_auto_review_active(False)
                self.dock_widget.set_auto_status("error", msg)
            except (RuntimeError, AttributeError):
                pass
        try:
            self._set_zone_band_fill_visible(True)
            self._restore_tile_grid_after_run()
        except Exception:  # nosec B110
            pass
        try:



            self.iface.messageBar().pushMessage(
                "AI Segmentation", msg, level=Qgis.MessageLevel.Warning,
                duration=10)
        except Exception:  # nosec B110
            pass

    def _show_finalize_drain_progress(self, done: int, total: int) -> None:

        if self.dock_widget is None:
            return
        try:
            self.dock_widget.set_auto_finalize_tiles(int(done), int(total))
        except (RuntimeError, AttributeError):
            pass

    def _mark_finalize_phase(self, state: dict, phase) -> None:

        import time as _t
        now = _t.monotonic()
        spent = state.setdefault("phase_s", {})
        last = state.get("phase_t0")
        prev = state.get("phase_seen")
        if last is not None and prev is not None:
            spent[prev] = spent.get(prev, 0.0) + (now - last)
        state["phase_t0"] = now
        state["phase_seen"] = phase

    def _log_finalize_phases(self, state: dict, objects: int) -> None:





        self._mark_finalize_phase(state, None)
        spent = state.get("phase_s") or {}
        if not spent:
            return
        total = sum(spent.values())



        try:
            from .auto_client_profile import add_review_pass_seconds
            if state.get("mode") != "reslice":
                self._auto_finalize_s = float(total)
            add_review_pass_seconds(
                self, "shape_pass_s",
                float(spent.get("filter", 0.0)) + float(spent.get("align", 0.0)))
            add_review_pass_seconds(self, "snap_s", float(spent.get("snap", 0.0)))
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        if total < 1.0:
            return
        parts = " ".join(f"{name} {secs:.1f}s"
                         for name, secs in sorted(spent.items(),
                                                  key=lambda kv: -kv[1])
                         if name)
        seeded = int(state.get("seeded_shapes", 0) or 0)
        if seeded:
            parts = f"{parts}, {seeded} shape(s) reused from the run"
        try:
            QgsMessageLog.logMessage(
                f"Auto detection: finalize took {total:.1f}s on {objects} "
                f"object(s) ({parts})",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _step_auto_finalize_refine_impl(self) -> None:
















        state = self._auto_finalize_state
        if state is None or state.get("gen") != self._auto_finalize_gen:
            return
        import time as _t

        from qgis.PyQt.QtCore import QTimer

        deadline = _t.monotonic() + auto_pump_budget()





        self._mark_finalize_phase(state, state.get("phase"))
        self._announce_auto_finalize_phase(state)





        if state.get("phase") == "drain":






            folded, queued = self._auto_stitch_backlog()
            now = _t.monotonic()
            drain_budget = _stitch_drain_budget_s()
            if state.get("drain_folded") != folded:
                state["drain_folded"] = folded
                state["drain_until"] = now + drain_budget
                state["drain_total"] = max(
                    int(state.get("drain_total", 0) or 0), folded + queued)
                self._show_finalize_drain_progress(folded, state["drain_total"])
            drain_until = state["drain_until"]



            if not self._finish_auto_stitcher(timeout_ms=0):
                hard_until = state.get("drain_hard_until")
                if hard_until is None and now >= drain_until:



                    QgsMessageLog.logMessage(
                        "Auto detection: live stitcher folded no tile for "
                        f"{int(drain_budget)}s with {queued} still "
                        "queued; finalizing what it folded", "AI Segmentation",
                        level=Qgis.MessageLevel.Warning)
                    self._abort_auto_stitch_queue()
                    hard_until = now + _stitch_join_timeout_ms() / 1000.0
                    state["drain_hard_until"] = hard_until
                elif hard_until is not None and now >= hard_until:



                    QgsMessageLog.logMessage(
                        "Auto detection: live stitcher would not stop; keeping "
                        "results as they stand", "AI Segmentation",
                        level=Qgis.MessageLevel.Warning)
                    self._abort_auto_stitcher()
                    self._finalize_drain_done(state)
                    return




                self._request_auto_live_repaint()
                QTimer.singleShot(self._auto_offgui_poll_ms(), self._step_auto_finalize_refine)
                return
            total = int(state.get("drain_total", 0) or 0)
            self._show_finalize_drain_progress(total, total)
            self._finalize_drain_done(state)
            return



        if state.get("phase") in ("server", "server_apply"):
            self._step_server_finalize(state, deadline)
            return





        if state.get("phase") == "remerge":
            remerge = state["remerge"]
            if state.get("remerge_t0") is None:
                state["remerge_t0"] = _t.monotonic()
            done = False
            while not done and _t.monotonic() < deadline:
                try:
                    done = remerge.step(64)
                except Exception as exc:  # noqa: BLE001


                    self._log_finalize_drop(state, "remerge", "?", exc)
            if not done:
                QTimer.singleShot(0, self._step_auto_finalize_refine)
                return
            rows = remerge.result()
            self._log_raw_fragment_remerge(
                remerge, (_t.monotonic() - state["remerge_t0"]) * 1000)
            state.pop("remerge", None)
            state.pop("remerge_t0", None)
            self._seed_finalize_sweep_phase(state, rows)
            return



        if state.get("phase") == "sweep":
            sweep = state["sweep"]
            done = False
            while not done and _t.monotonic() < deadline:
                try:
                    done = sweep.step(128)
                except Exception as exc:  # noqa: BLE001



                    self._log_finalize_drop(state, "sweep", "?", exc)
            if not done:
                QTimer.singleShot(0, self._step_auto_finalize_refine)
                return
            merged_ided = sweep.result()
            before = state.get("sweep_before", len(merged_ided))
            if len(merged_ided) != before:
                QgsMessageLog.logMessage(
                    f"Auto detection: redundancy sweep dropped {before - len(merged_ided)} "
                    f"covered fragment(s) of {before} objects",
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )
            if not merged_ided:
                self._auto_finalize_state = None
                self._pop_nothing_found_notice()
                self._record_auto_zero_result(state["tiles_succeeded"])
                return
            state.pop("sweep", None)





            from .review_align_offload import begin_align_pass
            align, align_budget = begin_align_pass(self, merged_ided)
            if align is not None:
                state["phase"] = "align"
                state["align"] = align
                state["align_rows"] = merged_ided
                state["align_budget_s"] = align_budget
                QTimer.singleShot(0, self._step_auto_finalize_refine)
                return
            self._seed_finalize_build_phase(state, merged_ided)
            return




        if state.get("phase") == "align":
            from .review_align_offload import finish_align_pass, stop_align_thread
            align = state["align"]
            budget = float(state.get("align_budget_s")
                           or align_phase_budget_s(_ALIGN_PHASE_BUDGET_S))
            align_until = state.get("align_until")
            if align_until is None:
                align_until = _t.monotonic() + budget
                state["align_until"] = align_until
            done = False



            off_gui = callable(getattr(align, "finished", None))
            try:
                if off_gui:
                    done = bool(align.finished())



                while not off_gui and not done and _t.monotonic() < deadline:
                    done = align.step(1)
            except Exception as exc:  # noqa: BLE001
                self._log_finalize_drop(state, "align", "?", exc)
                stop_align_thread(self)
                self._seed_finalize_build_phase(
                    state, state.pop("align_rows"), align)
                state.pop("align", None)
                return
            if not done and _t.monotonic() >= align_until:



                QgsMessageLog.logMessage(
                    "Auto detection: footprint alignment ran out of its "
                    f"{int(budget)}s budget; keeping the shapes it had not "
                    "reached as they are",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
                rows = finish_align_pass(
                    self, align, state["align_rows"], timed_out=True)
                state.pop("align", None)
                state.pop("align_rows", None)
                state.pop("align_until", None)
                self._seed_finalize_build_phase(state, rows, align)
                return
            if not done:
                QTimer.singleShot(
                    self._auto_offgui_poll_ms() if off_gui else 0,
                    self._step_auto_finalize_refine)
                return
            self._log_footprint_alignment(align)
            rows = finish_align_pass(
                self, align, state["align_rows"], timed_out=False)
            state.pop("align", None)
            state.pop("align_rows", None)
            state.pop("align_until", None)
            self._seed_finalize_build_phase(state, rows, align)
            return


        if state.get("phase") == "build":
            build_pending = state["build_pending"]
            objects = state["objects"]
            object_fids = state["object_fids"]
            measurer = state["measurer"]






            floor = self._review_noise_floor()
            fp_rules = state.get("fp_rules")
            if fp_rules is None:
                fp_rules = self._auto_fp_rules()
                state["fp_rules"] = fp_rules
            while build_pending:
                fid, geom, score = build_pending.pop()
                try:
                    if geom is not None and not geom.isEmpty() and float(score) >= floor:
                        area = self._object_area_m2(geom, measurer)
                        if not self._object_is_fp(geom, area, fp_rules, measurer):
                            objects.append((geom, float(score), area))




                            object_fids.append(fid)
                except Exception as exc:  # noqa: BLE001
                    self._log_finalize_drop(state, "build", fid, exc)
                if _t.monotonic() >= deadline:
                    break
            if build_pending:
                QTimer.singleShot(0, self._step_auto_finalize_refine)
                return


            self._auto_objects = objects
            self._auto_object_fids = object_fids
            self._reset_review_refine_cache()





            raw_start_conf = self._review_start_confidence()






            start_conf = self._snap_review_start_confidence(raw_start_conf)
            self._auto_confidence = start_conf
            state["params"]["conf"] = start_conf
            if not self._auto_headless_run and self.dock_widget is not None:
                try:
                    spin = self.dock_widget.auto_confidence_spin
                    spin.blockSignals(True)
                    spin.setValue(start_conf)
                    spin.blockSignals(False)





                    self.dock_widget.seed_review_confidence(
                        int(round(start_conf * 100)))



                    floor_pct = int(math.ceil(self._review_noise_floor() * 100))
                    self.dock_widget.set_review_conf_floor(floor_pct)
                    hist = getattr(self.dock_widget, "auto_conf_histogram", None)
                    if hist is not None:







                        conf_slider = self.dock_widget.auto_review_confidence_slider
                        hist_max = dial_in_range(
                            "tuning.review.confidence_hist_max", 0.95, 0.5, 1.0)
                        hist.set_range(conf_slider.minimum() / 100.0, hist_max)
                        hist.set_scores([s for (_g, s, _a) in objects])
                        hist.set_cutoff(start_conf)



                    self.dock_widget.set_auto_review_score_useful(
                        self._run_scores_rank_objects())
                except (RuntimeError, AttributeError):
                    pass






            seeded = self._seed_review_refine_cache(
                state["params"], state["pixel_size"], objects, object_fids)
            if seeded:
                state["seeded_shapes"] = seeded
            state["filter_pending"] = list(enumerate(objects))
            state["total_filter"] = len(objects)
            state["visible"] = []
            state["visible_scores"] = []
            state["visible_ids"] = []
            state["visible_order"] = []
            state["phase"] = "filter"
            QTimer.singleShot(0, self._step_auto_finalize_refine)
            return






        if state.get("phase") == "snap":
            started = state["snap"]
            done = False
            while not done and _t.monotonic() < deadline:
                done = started[0].step(64)
            if not done:
                QTimer.singleShot(0, self._step_auto_finalize_refine)
                return
            visible = self._finish_boundary_snap(state["snap_geoms"], started)
            state.pop("snap", None)
            state.pop("snap_geoms", None)
            self._end_auto_finalize_pass(
                state, visible, state.pop("snap_scores"),
                state.pop("snap_ids"))
            return


        filter_pending = state["filter_pending"]
        visible = state["visible"]
        visible_scores = state["visible_scores"]
        visible_ids = state["visible_ids"]




        visible_order = state.setdefault("visible_order", [])
        params = state["params"]
        pixel_size = state["pixel_size"]
        removed = self._review_removed_fids()





        size_gate_on = bool(params.get("min_a", 0.0) > 0 or params.get("max_a", 0.0) > 0)
        measurer = state.get("measurer")
        if size_gate_on and measurer is None:
            measurer = self._make_auto_area_measurer()
            state["measurer"] = measurer




        cache = self._auto_reslice_cache
        shape_key = self._review_shape_key(params, pixel_size)
        if cache.get("key") != shape_key:
            self._adopt_reslice_shape_key(cache, shape_key)
        cached_geoms = cache["geoms"]

        awaiting = state.setdefault("awaiting", [])
        offload = self._review_refine_offload_for(state, filter_pending, awaiting)





        self._drain_review_refine_results(shape_key)
        if awaiting:




            alive = self._review_refine_thread_alive()
            still_waiting = []
            for det_idx, score, manual in awaiting:
                if det_idx in cached_geoms:
                    g = cached_geoms[det_idx]
                elif alive:
                    still_waiting.append((det_idx, score, manual))
                    continue
                else:
                    g = self._review_shape_now(det_idx, params, pixel_size)
                self._accept_review_shape(
                    state, det_idx, g, score, manual, size_gate_on, measurer)
            awaiting[:] = still_waiting
        while filter_pending:
            row = filter_pending.pop()
            det_idx, (base, score, area) = row
            try:
                base_ok = base is not None and not base.isEmpty()



                manual = self._object_is_manual(det_idx)
                passes = (manual or self._passes_review_filters(score, area, params))
                if det_idx not in removed and base_ok and passes:


                    if det_idx in cached_geoms:
                        self._accept_review_shape(
                            state, det_idx, cached_geoms[det_idx], score,
                            manual, size_gate_on, measurer)
                    elif offload is not None and self._review_refine_queue_full():





                        filter_pending.append(row)
                        break
                    elif offload is not None and self._offload_review_refine(
                            offload, det_idx, base, params, pixel_size,
                            shape_key):
                        awaiting.append((det_idx, score, manual))
                    else:
                        g = self._review_refined_geom(
                            det_idx, base, params, pixel_size)
                        self._accept_review_shape(
                            state, det_idx, g, score, manual, size_gate_on,
                            measurer)
            except Exception as exc:  # noqa: BLE001
                self._log_finalize_drop(state, "filter", det_idx, exc)
            if _t.monotonic() >= deadline:
                break
        if filter_pending or awaiting:










            if (state.get("mode") == "reslice" and visible and not params.get("snap_boundaries")):
                now = _t.monotonic()
                partial_interval_s = dial_in_range(
                    "tuning.auto.partial_push_interval_s", 0.25, 0.05, 5.0)
                if now - state.get("last_partial", 0.0) >= partial_interval_s:
                    state["last_partial"] = now
                    self._push_review_geoms(
                        visible, repair=False, scores=visible_scores,
                        ids=visible_ids,
                        stamp=("acc", (self._auto_reslice_cache or {}).get("key")),
                        partial=True)





            more_now = bool(filter_pending) and not self._review_refine_queue_full()
            QTimer.singleShot(
                0 if more_now else dial_in_range(
                    "tuning.auto.await_refine_poll_ms", _AWAIT_REFINE_POLL_MS, 1, 200),
                self._step_auto_finalize_refine)
            return
        vis_scores = state.get("visible_scores", [])
        vis_ids = state.get("visible_ids", [])




        parallel_lists = bool(visible_order) and len(visible_order) == len(visible)
        parallel_lists = parallel_lists and len(vis_scores) == len(visible) and len(vis_ids) == len(visible)
        if parallel_lists:
            ranked = sorted(range(len(visible)), key=visible_order.__getitem__)
            visible = [visible[i] for i in ranked]
            vis_scores = [vis_scores[i] for i in ranked]
            vis_ids = [vis_ids[i] for i in ranked]



        answer, started = self._begin_boundary_snap(visible, params)
        if started is not None:
            state["phase"] = "snap"
            state["snap"] = started
            state["snap_geoms"] = visible
            state["snap_scores"] = vis_scores
            state["snap_ids"] = vis_ids
            QTimer.singleShot(0, self._step_auto_finalize_refine)
            return
        self._end_auto_finalize_pass(state, answer, vis_scores, vis_ids)

    def _end_auto_finalize_pass(self, state: dict, visible: list,
                                vis_scores: list, vis_ids: list) -> None:



        from qgis.PyQt.QtCore import QTimer



        self._log_finalize_phases(state, len(visible))
        dropped = int(state.get("dropped_objects", 0) or 0)
        if dropped > state.get("drop_log_max", 5):
            QgsMessageLog.logMessage(
                f"Auto detection: {dropped} object(s) dropped by the "
                f"finalize guards this pass",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        refine_dropped = int(state.get("refine_dropped", 0) or 0)
        if refine_dropped:
            QgsMessageLog.logMessage(
                f"Auto detection: {refine_dropped} object(s) left no shape "
                f"under the current cleanup settings and are not on the map",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        self._auto_finalize_state = None
        self._pop_nothing_found_notice()
        if state.get("mode") == "reslice":
            self._apply_auto_reslice_result(visible, vis_scores, vis_ids)
        else:
            self._complete_auto_finalize(
                visible, state["tiles_succeeded"], vis_scores, vis_ids)


            from .auto_client_profile import stop_gui_gap_watch
            stop_gui_gap_watch(self)

            try:
                from ...core.run_log_capture import send_run_log
                send_run_log("completed")
            except Exception:  # noqa: BLE001  # nosec B110
                pass









            QTimer.singleShot(0, self._archive_auto_default_export)






            self._start_build_preview_cache(state["pixel_size"])

    def _seed_finalize_build_phase(self, state: dict, merged_ided: list,
                                   align=None) -> None:







        from qgis.PyQt.QtCore import QTimer
        if align is not None:
            self._note_stitch_shapes_dirty(getattr(align, "changed_fids", None))
        state["phase"] = "build"
        state["build_pending"] = list(merged_ided)
        state["total_build"] = len(merged_ided)
        state["objects"] = []
        state["object_fids"] = []
        QTimer.singleShot(0, self._step_auto_finalize_refine)

    def _apply_auto_reslice_result(self, geoms: list,
                                   scores: list | None = None,
                                   ids: list | None = None) -> None:








        if not self._auto_review:
            return
        self._auto_review["scores"] = scores
        self._auto_review["ids"] = ids



        self._auto_review["stamp"] = (
            "acc", (self._auto_reslice_cache or {}).get("key"))
        self._auto_review["geoms"] = geoms
        self._update_review_header(len(geoms))
        self._refresh_auto_review_preview()




        _reanchor = getattr(self, "_refresh_correct_selection_after_reslice", None)
        if _reanchor is not None:
            _reanchor()

    def _start_build_preview_cache(self, pixel_size: float) -> None:









        from qgis.PyQt.QtCore import QTimer

        self._auto_preview_build_gen += 1
        self._auto_preview_build_state = {
            "pending": [(i, g, s, a)
                        for i, (g, s, a) in enumerate(self._auto_objects)],
            "out": [],
            "pixel_size": pixel_size,
            "gen": self._auto_preview_build_gen,
        }
        QTimer.singleShot(0, self._step_build_preview_cache)

    def _step_build_preview_cache(self) -> None:



        try:
            self._step_build_preview_cache_impl()
        except Exception as exc:  # noqa: BLE001
            self._auto_preview_build_state = None
            try:
                from qgis.core import Qgis, QgsMessageLog
                QgsMessageLog.logMessage(
                    f"Preview cache build stopped: {type(exc).__name__}",
                    "AI Segmentation", Qgis.MessageLevel.Warning)
            except (RuntimeError, AttributeError, ImportError):
                pass

    def _step_build_preview_cache_impl(self) -> None:











        state = self._auto_preview_build_state
        if state is None or state.get("gen") != self._auto_preview_build_gen:
            return
        import time as _t

        from qgis.PyQt.QtCore import QTimer

        deadline = _t.monotonic() + auto_pump_budget()
        pending = state["pending"]
        out = state["out"]
        pixel_size = state["pixel_size"]




        tol = _AUTO_REVIEW_SIMPLIFY_DEFAULT * pixel_size if pixel_size > 0 else 0.0
        from ...core.layer_conventions import to_multipolygon
        while pending:
            det_idx, geom, score, area = pending.pop()
            if geom is not None and not geom.isEmpty():
                s = geom.simplify(tol) if tol > 0 else geom
                if s is None or s.isEmpty():
                    s = geom


                mp = to_multipolygon(s)
                out.append((mp if mp is not None and not mp.isEmpty() else s,
                            score, area, det_idx,
                            self._object_is_manual(det_idx)))
            if _t.monotonic() >= deadline:
                break
        if pending:
            QTimer.singleShot(0, self._step_build_preview_cache)
            return
        out.sort(key=lambda row: (row[4], row[1]), reverse=True)
        self._auto_preview_geoms = out
        self._auto_preview_build_state = None
