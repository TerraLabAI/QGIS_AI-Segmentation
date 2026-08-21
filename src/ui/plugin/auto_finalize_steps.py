"""The cooperative finalize and reslice state machine: drain, sweep, build,
filter, plus the background preview cache the confidence drag reads.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

import math

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr
from ...core.review_defaults import (
    AUTO_REVIEW_SIMPLIFY_DEFAULT as _AUTO_REVIEW_SIMPLIFY_DEFAULT,
)
from ...workers.live_stitch_thread import STITCH_JOIN_TIMEOUT_MS
from .auto_results import _STITCH_DRAIN_BUDGET_S, _STITCH_WAIT_SLICE_MS
from .shared import _AUTO_PUMP_BUDGET_S

# What the run-wide footprint alignment may cost the wait between the last
# tile and the review. One object can take tens of milliseconds, so a dense run
# would otherwise hold the finish screen for as long as it takes. The pass
# improves shapes and never gates them: past this the rows it has not reached
# keep their traced outline and the run goes on.
_ALIGN_PHASE_BUDGET_S = 10.0


class AutoFinalizeStepsMixin:
    """Time-sliced finalize and reslice, yielding to the event loop between slices."""

    def _step_auto_finalize_refine(self) -> None:
        """One cooperative slice of the finalize/reslice pipeline. Thin
        exception boundary over _step_auto_finalize_refine_impl: the slices
        run as bare QTimer.singleShot slots, so an exception escaping one
        would end the chain silently, AFTER the run UI was restored and
        BEFORE the review opened, leaving billed results in limbo behind a
        bare Detect row. Per-object failures are dropped inside the impl;
        anything that still escapes is systemic and routes to
        _abort_auto_finalize (points at the disk autosave, restores a
        consistent dock state)."""
        try:
            self._step_auto_finalize_refine_impl()
        except Exception as exc:  # noqa: BLE001 -- the timer-slot boundary
            self._abort_auto_finalize(exc)

    def _log_finalize_drop(self, state: dict, phase: str, idx,
                           exc: Exception) -> None:
        """Record one object dropped by a guarded finalize slice: count it on
        the state and log only the first few, so a systemically sick set can
        never spam the message log from the main thread."""
        dropped = int(state.get("dropped_objects", 0) or 0) + 1
        state["dropped_objects"] = dropped
        if dropped <= 5:
            try:
                QgsMessageLog.logMessage(
                    f"Auto detection: dropped object {idx} in the {phase} "
                    f"phase ({exc})",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
            except Exception:  # nosec B110
                pass

    def _abort_auto_finalize(self, exc: Exception) -> None:
        """Terminal handler for a systemic finalize/reslice failure (anything
        other than the silent generation-guard supersession). Never raises.

        A reslice failure keeps the open review on its previous visible set
        and says the new settings did not apply. A finalize failure is the
        dangerous one: the run UI is already restored and no review exists,
        so the billed results' only copy is the pre-tail disk autosave. Load
        that layer into the project right away, tell the user where their
        detections went, and land the dock on a consistent prompt step."""
        state = self._auto_finalize_state
        mode = (state or {}).get("mode") or (
            "reslice" if self._auto_review is not None else "finalize")
        self._auto_finalize_state = None
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
            # The review still holds its previous visible set: keep it.
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
        # Surface the Stage-A autosave of THIS run immediately: the user must
        # never be left staring at the prompt step with paid work invisible.
        recovered = None
        try:
            from ...core import run_autosave
            run_id = self._auto_run_id or ""
            pending = run_autosave.read_pending()
            if (pending and run_id and str(pending.get("run_id") or "") == run_id):
                recovered = run_autosave.load_pending_layer(pending)
                if recovered:
                    run_autosave.clear_pending()
        except Exception:  # nosec B110
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
                # The hand-over died: give the screen back to the prompt step
                # before saying what happened (see set_auto_finalizing).
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
            self.iface.messageBar().pushWarning("AI Segmentation", msg)
        except Exception:  # nosec B110
            pass

    def _mark_finalize_phase(self, state: dict, phase) -> None:
        """Charge the time since the last slice to the phase that spent it."""
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
        """Report where the finalize wall clock went, once, when it ends.

        Objects and phases both, because the two together are the only way to
        read the next slow run: a phase that costs per object scales with the
        count, a phase that waits on the stitcher does not."""
        self._mark_finalize_phase(state, None)
        spent = state.get("phase_s") or {}
        if not spent:
            return
        total = sum(spent.values())
        if total < 1.0:
            return  # nothing worth a log line
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
        """One cooperative slice of the finalize/reslice pipeline, yielding to the
        event loop between slices so QGIS never freezes. Two phases, both
        time-sliced:

          - "build" (finalize only): measure each merged WHOLE object's geodesic
            area and store (geom, score, area) in _auto_objects. No shape refine
            here: the Automatic path stays faithful by default.
          - "filter" (both modes): filter the canonical whole objects by the
            snapshotted confidence + min/max size, then apply the user shape
            refine to those that pass, building the VISIBLE geometry set. This is
            a pure recompute on already-merged objects: it never re-merges and so
            never re-cuts an object (the fix for confidence cutting buildings).

        On completion, "finalize" enters the post-run review; "reslice" swaps the
        review's visible geoms. Generation-guarded so a new run / teardown / later
        reslice supersedes an in-flight one (last one wins)."""
        state = self._auto_finalize_state
        if state is None or state.get("gen") != self._auto_finalize_gen:
            return  # superseded by a new run, a later reslice, or torn down
        import time as _t

        from qgis.PyQt.QtCore import QTimer

        deadline = _t.monotonic() + _AUTO_PUMP_BUDGET_S
        # Time each phase separately. "Almost done - building the shapes..."
        # covers four different jobs (waiting for the stitcher's tail, the
        # redundancy sweep, measuring, shaping), and on a run big enough for the
        # wait to be noticed there is no way to tell from outside which one is
        # eating the clock. One log line at the end names it.
        self._mark_finalize_phase(state, state.get("phase"))

        # Phase -1 (finalize only): the stitcher thread is still folding the
        # tail of the run into the merger. Wait for it in short slices so the
        # event loop keeps painting, and keep asking for a repaint so the last
        # objects appear as they land rather than all at once at the end.
        if state.get("phase") == "drain":
            drain_until = state.get("drain_until")
            if drain_until is None:
                drain_until = _t.monotonic() + _STITCH_DRAIN_BUDGET_S
                state["drain_until"] = drain_until
            if not self._finish_auto_stitcher(timeout_ms=_STITCH_WAIT_SLICE_MS):
                now = _t.monotonic()
                hard_until = state.get("drain_hard_until")
                if hard_until is None and now >= drain_until:
                    # Out of patience with the BACKLOG. Drop what is still
                    # queued and keep what was already folded, which is what
                    # the user was billed for.
                    QgsMessageLog.logMessage(
                        "Auto detection: live stitcher did not finish in "
                        f"{int(_STITCH_DRAIN_BUDGET_S)}s; finalizing what it "
                        "folded", "AI Segmentation",
                        level=Qgis.MessageLevel.Warning)
                    self._abort_auto_stitch_queue()
                    hard_until = now + STITCH_JOIN_TIMEOUT_MS / 1000.0
                    state["drain_hard_until"] = hard_until
                elif hard_until is not None and now >= hard_until:
                    # It is not even leaving the tile it is on. Park it and go
                    # on: a review that never opens is worse than a stale read
                    # of results the user already paid for.
                    QgsMessageLog.logMessage(
                        "Auto detection: live stitcher would not stop; keeping "
                        "results as they stand", "AI Segmentation",
                        level=Qgis.MessageLevel.Warning)
                    self._abort_auto_stitcher()
                    self._finalize_drain_done(state)
                    return
                # Still not ours. An aborted stitcher only has to finish the one
                # tile it is inside, so this comes back in a moment; reading the
                # merger now would race that fold and corrupt the very results
                # the timeout is trying to save.
                self._request_auto_live_repaint()
                QTimer.singleShot(0, self._step_auto_finalize_refine)
                return
            self._finalize_drain_done(state)
            return

        # Phase 0 (finalize only): the end-of-run redundancy sweep, time-sliced.
        # GEOS-heavy on dense runs, so chunk it and yield between chunks.
        if state.get("phase") == "sweep":
            sweep = state["sweep"]
            done = False
            while not done and _t.monotonic() < deadline:
                try:
                    done = sweep.step(128)
                except Exception as exc:  # noqa: BLE001 -- drop the candidate
                    # step() advances its cursor before touching a geometry,
                    # so retrying skips the failing object and always makes
                    # progress; one bad geometry can no longer end the chain.
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
                self._record_auto_zero_result(state["tiles_succeeded"])
                return
            state.pop("sweep", None)
            # Sweep done: run-wide footprint alignment when the server opted
            # this run's prompt family in (None = pass off, straight to build).
            # It needs the WHOLE deduplicated set at once (the neighbour
            # consensus), which is why it lives here and not in the per-object
            # refine.
            align = self._auto_footprint_align_sweep(merged_ided)
            if align is not None:
                state["phase"] = "align"
                state["align"] = align
                state["align_rows"] = merged_ided
                QTimer.singleShot(0, self._step_auto_finalize_refine)
                return
            self._seed_finalize_build_phase(state, merged_ided)
            return

        # Phase 0b (finalize only): the run-wide footprint alignment,
        # time-sliced like the sweep. A systemic failure falls back to the
        # unaligned rows: the pass improves shapes, it never gates them.
        if state.get("phase") == "align":
            align = state["align"]
            align_until = state.get("align_until")
            if align_until is None:
                align_until = _t.monotonic() + _ALIGN_PHASE_BUDGET_S
                state["align_until"] = align_until
            done = False
            try:
                # One object per slice: a single alignment can cost tens of
                # milliseconds and the deadline is only read between slices, so
                # a bigger step overshoots the budget the whole pump shares.
                while not done and _t.monotonic() < deadline:
                    done = align.step(1)
            except Exception as exc:  # noqa: BLE001 -- keep the raw shapes
                self._log_finalize_drop(state, "align", "?", exc)
                self._seed_finalize_build_phase(state, state.pop("align_rows"))
                state.pop("align", None)
                return
            if not done and _t.monotonic() >= align_until:
                # Out of time. The rows the pass reached keep their aligned
                # shape, the rest keep the one they came in with: a review that
                # opens late is worse than a few unsquared corners.
                QgsMessageLog.logMessage(
                    "Auto detection: footprint alignment ran out of its "
                    f"{int(_ALIGN_PHASE_BUDGET_S)}s budget; keeping the shapes "
                    "it had not reached as they are",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
                rows = align.result()
                state.pop("align", None)
                state.pop("align_rows", None)
                state.pop("align_until", None)
                self._seed_finalize_build_phase(state, rows)
                return
            if not done:
                QTimer.singleShot(0, self._step_auto_finalize_refine)
                return
            self._log_footprint_alignment(align)
            rows = align.result()
            state.pop("align", None)
            state.pop("align_rows", None)
            state.pop("align_until", None)
            self._seed_finalize_build_phase(state, rows)
            return

        # Phase 1 (finalize only): build the canonical (geom, score, area) set.
        if state.get("phase") == "build":
            build_pending = state["build_pending"]
            objects = state["objects"]
            object_fids = state["object_fids"]
            measurer = state["measurer"]
            # Below-floor detections are pure noise, and objects the server's
            # per-class geometry-attribute filter marks as false positives are
            # structural junk: drop both here so they never count in the review
            # total nor render (also fewer shapes to build). The rules are the
            # same for every object, so resolve them once per run (None until the
            # first build slice; [] means the filter is OFF).
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
                            # Parallel to objects (same append guard = same
                            # order), so the review's Random hue keys on the
                            # same merger id the live run used: an object's
                            # colour survives the run->review handoff.
                            object_fids.append(fid)
                except Exception as exc:  # noqa: BLE001 -- drop this object only
                    self._log_finalize_drop(state, "build", fid, exc)
                if _t.monotonic() >= deadline:
                    break
            if build_pending:
                QTimer.singleShot(0, self._step_auto_finalize_refine)
                return
            # Build done: publish the canonical set, seed the confidence-drag
            # preview cache, then move to the shared filter phase.
            self._auto_objects = objects
            self._auto_object_fids = object_fids
            self._reset_review_refine_cache()
            # Adaptive starting confidence: if 0.30 would hide EVERY found object,
            # drop to the highest 5% step that shows at least one, so the review
            # never opens reading "0 found". Set it BEFORE the filter phase below
            # so the visible set reflects the lowered cutoff; update the snapshot
            # params too. Headless keeps the seeded default (stable API contract).
            raw_start_conf = self._review_start_confidence()
            # Snap the cutoff to the review slider's 5% grid ONCE, at the source,
            # so the stored filter value, the histogram cutoff and the seeded
            # slider/spin all show the SAME number (the slider can only rest on
            # 5% steps, so an unsnapped 0.17 used to read as 15% while the filter
            # ran at 17%). The note flags below still read the UNSNAPPED value so
            # a pure snap never masquerades as a deliberate lowering.
            start_conf = self._snap_review_start_confidence(raw_start_conf)
            self._auto_confidence = start_conf
            state["params"]["conf"] = start_conf
            if not self._auto_headless_run and self.dock_widget is not None:
                try:
                    spin = self.dock_widget.auto_confidence_spin
                    spin.blockSignals(True)
                    spin.setValue(start_conf)
                    spin.blockSignals(False)
                    # The review page already seeded its slider/spin from the
                    # pre-run dial when it opened (before this async step), so
                    # push the REAL starting cutoff into them too: without
                    # this, a tuned/adaptive start filtered at one value while
                    # the visible handle rested at another.
                    self.dock_widget.seed_review_confidence(
                        int(round(start_conf * 100)))
                    # Clamp the confidence controls so neither the slider nor the
                    # spinbox can dial below the noise floor (sub-floor detections
                    # were already dropped, so a cutoff under it is meaningless).
                    floor_pct = int(math.ceil(self._review_noise_floor() * 100))
                    self.dock_widget.set_review_conf_floor(floor_pct)
                    hist = getattr(self.dock_widget, "auto_conf_histogram", None)
                    if hist is not None:
                        # The histogram must span EXACTLY the slider's range,
                        # so the grey/blue boundary sits above the handle at
                        # every cutoff instead of drifting on a different
                        # scale. Read that range back off the slider: the floor
                        # moves DOWN as well as up now, so re-clamping it here
                        # to the design minimum put the strip on a scale the
                        # handle no longer used, and the two could drift again.
                        conf_slider = self.dock_widget.auto_review_confidence_slider
                        hist.set_range(conf_slider.minimum() / 100.0, 0.95)
                        hist.set_scores([s for (_g, s, _a) in objects])
                        hist.set_cutoff(start_conf)
                    # Last, because it can take the controls seeded just above
                    # off the step: a run whose objects are all rated the same
                    # has no ranking for Confidence to filter on.
                    self.dock_widget.set_auto_review_score_useful(
                        self._run_scores_rank_objects())
                except (RuntimeError, AttributeError):
                    pass
            # Enumerate so each visible geom remembers WHICH canonical object it
            # came from; the visible id it records is the object's stable merger
            # fid (_object_fid_for), so the Random display colours key on the same
            # id the live run used and a reslice never reshuffles a colour.
            # Hand the filter phase the shapes the stitch thread already built.
            # Whatever is not seeded is refined below exactly as before.
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

        # Phase 2 (both modes): filter whole objects, then shape-refine the pass.
        filter_pending = state["filter_pending"]
        visible = state["visible"]
        visible_scores = state["visible_scores"]
        visible_ids = state["visible_ids"]
        # The canonical index each visible row came from. The pump refines the
        # objects on screen first, so the order they land in follows the map
        # view; this is what sorts the finished set back to something that
        # depends only on the run.
        visible_order = state.setdefault("visible_order", [])
        params = state["params"]
        pixel_size = state["pixel_size"]
        removed = self._review_removed_fids()
        # The size gate reads the shape that SHIPS. Cleanup moves an object
        # across the Min/Max line, and the saved file records the refined area,
        # so gating the merged base wrote rows whose own figure the filter says
        # is out of range. Measured only when a size filter is actually set, so
        # a run without one pays nothing for it.
        size_gate_on = bool(params.get("min_a", 0.0) > 0 or params.get("max_a", 0.0) > 0)
        measurer = state.get("measurer")
        if size_gate_on and measurer is None:
            measurer = self._make_auto_area_measurer()
            state["measurer"] = measurer
        while filter_pending:
            det_idx, (base, score, area) = filter_pending.pop()
            try:
                base_ok = base is not None and not base.isEmpty()
                # Hand-drawn / split objects skip the confidence + size gates
                # so a slider set for the detections never hides the user's own
                # geometry.
                manual = self._object_is_manual(det_idx)
                passes = (manual or self._passes_review_filters(score, area, params))
                if det_idx not in removed and base_ok and passes:
                    # Cached per object + shape key: a filter-only reslice
                    # (Confidence / Min / Max size) is pure dict lookups here.
                    g = self._review_refined_geom(det_idx, base, params, pixel_size)
                    if g is None:
                        # An object that passed every filter and came back with
                        # no shape leaves the map and the export. Count it: a
                        # total that quietly disagrees with the map is the one
                        # thing the review cannot explain.
                        state["refine_dropped"] = int(
                            state.get("refine_dropped", 0) or 0) + 1
                    elif (manual or not size_gate_on
                            or self._passes_size_filters(
                                self._object_area_m2(g, measurer), params)):
                        visible.append(g)
                        visible_scores.append(score)
                        visible_ids.append(self._object_fid_for(det_idx))
                        visible_order.append(det_idx)
            except Exception as exc:  # noqa: BLE001 -- drop this object only
                self._log_finalize_drop(state, "filter", det_idx, exc)
            if _t.monotonic() >= deadline:
                break
        if filter_pending:
            # Progressive apply (reslice only): every ~250 ms, write the geoms
            # refined SO FAR onto the layer (diff-only, objects not yet
            # processed keep their old shape). A shape-settings change then
            # visibly sweeps the map instead of freezing on the old state
            # until the whole cooperative pass ends.
            #
            # Never with shared borders on: those partial geoms are not
            # snapped yet, and they would be written under the SAME stamp the
            # final snapped push carries, so the diff would read them as
            # already current and the unsnapped shapes would stay on the map.
            if (state.get("mode") == "reslice" and visible and not params.get("snap_boundaries")):
                now = _t.monotonic()
                if now - state.get("last_partial", 0.0) >= 0.25:
                    state["last_partial"] = now
                    self._push_review_geoms(
                        visible, repair=False, scores=visible_scores,
                        ids=visible_ids,
                        stamp=("acc", (self._auto_reslice_cache or {}).get("key")),
                        partial=True)
            QTimer.singleShot(0, self._step_auto_finalize_refine)
            return
        vis_scores = state.get("visible_scores", [])
        vis_ids = state.get("visible_ids", [])
        # Back to canonical object order. The three lists are parallel and the
        # export writes them in this order, so leaving them in refine order
        # would make a saved layer's row order depend on where the map happened
        # to be pointing when the user moved a dial.
        parallel_lists = bool(visible_order) and len(visible_order) == len(visible)
        parallel_lists = parallel_lists and len(vis_scores) == len(visible) and len(vis_ids) == len(visible)
        if parallel_lists:
            ranked = sorted(range(len(visible)), key=visible_order.__getitem__)
            visible = [visible[i] for i in ranked]
            vis_scores = [vis_scores[i] for i in ranked]
            vis_ids = [vis_ids[i] for i in ranked]
        # The visible set is complete: run the one whole-set operation here,
        # where every neighbour exists, instead of inside the per-object
        # refine above. Order and length are preserved, so the parallel score
        # and id lists still line up.
        visible = self._apply_boundary_snap(visible, params)
        # The per-object guards logged only their first few drops: close the
        # pass with one honest total when more were eaten.
        self._log_finalize_phases(state, len(visible))
        dropped = int(state.get("dropped_objects", 0) or 0)
        if dropped > 5:
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
        if state.get("mode") == "reslice":
            self._apply_auto_reslice_result(visible, vis_scores, vis_ids)
        else:
            self._complete_auto_finalize(
                visible, state["tiles_succeeded"], vis_scores, vis_ids)
            # Genuine live interactive finalize only (restore and headless call
            # _complete_auto_finalize directly, not through this pump): archive
            # the clean default export so a run closed without Finish is still
            # captured. DEFERRED off the review-open frame: the archive
            # serializes the whole set to GeoJSON on the GUI thread, so running
            # it inline stacked onto the review-open cost on a huge run. A later
            # event-loop turn runs it once the review is already interactive.
            # The method already no-ops when _auto_review is gone (teardown) and
            # is once-per-run, so deferring it needs no extra guard.
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._archive_auto_default_export)
            # The confidence-drag preview cache is only wanted once the review
            # is on screen and a slider exists to drag, and until it is ready
            # the drag filters _auto_objects directly. It used to be started at
            # the end of the build phase, so its own singleShot chain
            # interleaved with the filter slices: it delayed the review it is
            # for, and the phase log charged its whole cost to "filter".
            self._start_build_preview_cache(state["pixel_size"])

    def _seed_finalize_build_phase(self, state: dict, merged_ided: list) -> None:
        """Seed the cooperative build phase from a deduplicated (and possibly
        alignment-swept) merged set, and schedule the next slice."""
        from qgis.PyQt.QtCore import QTimer
        state["phase"] = "build"
        state["build_pending"] = list(merged_ided)
        state["total_build"] = len(merged_ided)
        state["objects"] = []
        state["object_fids"] = []
        QTimer.singleShot(0, self._step_auto_finalize_refine)

    def _apply_auto_reslice_result(self, geoms: list,
                                   scores: list | None = None,
                                   ids: list | None = None) -> None:
        """Completion of a cooperative review reslice: swap in the new geoms,
        update the count, refresh the preview. No re-detection, no credits.
        ``scores`` / ``ids`` are parallel per-object lists (same order as geoms)
        feeding the review heatmap and the stable Random colours.

        Every hand edit is folded into its canonical row and its det_id skips
        the gates, so the reslice drives every polygon uniformly: no freeze, no
        separate protected set to re-merge here."""
        if not self._auto_review:
            return
        self._auto_review["scores"] = scores
        self._auto_review["ids"] = ids
        # Reslice output geoms are the cache-normalized refined objects: stamp
        # them with the shape key so the incremental push writes only the delta
        # (identical det_id + stamp = geometry unchanged).
        self._auto_review["stamp"] = (
            "acc", (self._auto_reslice_cache or {}).get("key"))
        self._auto_review["geoms"] = geoms
        self._update_review_header(len(geoms))
        self._refresh_auto_review_preview()
        # The Correct step's highlight and click targets are snapshots of the
        # shapes AS DRAWN, so they have to be re-taken from the set that was
        # just published. Without this a native QGIS edit leaves the yellow
        # outline (and the hit test) on the pre-edit shape.
        _reanchor = getattr(self, "_refresh_correct_selection_after_reslice", None)
        if _reanchor is not None:
            _reanchor()

    def _start_build_preview_cache(self, pixel_size: float) -> None:
        """Kick a cooperative, time-sliced build of the confidence-slider preview
        cache: the canonical WHOLE objects (geom, score, area, index, exempt),
        lightly simplified and sorted so a slider drag is a cheap prefix slice
        of whole objects (never fragments). The ground area and the gate
        exemption travel with each row so a drag tick applies the Min/Max size
        gate without re-measuring and without scanning past the cutoff. Runs in
        the background after the objects are built so it never blocks entering
        review; the slider drag falls back to filtering _auto_objects directly
        until it is ready."""
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
        """One cooperative slice of the preview-cache build: simplify a batch of
        scored geoms, then reschedule. On completion sort the rows once and
        publish to _auto_preview_geoms so subsequent slider drags use the cheap
        prefix-slice path. Generation-guarded against a new run / teardown.

        The sort puts the gate-exempt (hand-drawn / split) rows FIRST, then the
        rest by score desc. That is what lets a drag stop at the cutoff: an
        exempt row shows whatever its score, and after the exempt block every
        remaining row is ordered, so nothing below the cutoff is ever wanted.
        Every place that changes the exempt set rebuilds this cache, so the flag
        stored per row is the current one."""
        state = self._auto_preview_build_state
        if state is None or state.get("gen") != self._auto_preview_build_gen:
            return  # superseded or torn down
        import time as _t

        from qgis.PyQt.QtCore import QTimer

        deadline = _t.monotonic() + _AUTO_PUMP_BUDGET_S
        pending = state["pending"]
        out = state["out"]
        pixel_size = state["pixel_size"]
        # Same tolerance as the committed reslice (AUTO_REVIEW_SIMPLIFY_DEFAULT
        # px): the drag preview used a 5x coarser 2.0 px simplify, so shapes
        # visibly "sharpened" on slider release and small objects could collapse
        # during the drag. Matching tolerances makes drag == release.
        tol = _AUTO_REVIEW_SIMPLIFY_DEFAULT * pixel_size if pixel_size > 0 else 0.0
        from ...core.layer_conventions import to_multipolygon
        while pending:
            det_idx, geom, score, area = pending.pop()
            if geom is not None and not geom.isEmpty():
                s = geom.simplify(tol) if tol > 0 else geom
                if s is None or s.isEmpty():
                    s = geom
                # Coerce to MultiPolygon ONCE here so a confidence-drag tick's
                # incremental adds never pay the per-geom deep copy again.
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
