"""What happens when a run stops: the finalize funnel every terminal enters,
the run summary, and the zero-detection screens.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py); split out
so agents and humans can work on one concern per file. Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr
from .shared import park_orphaned_worker


class AutoRunTerminalMixin:
    """The terminal funnel for a finished, cancelled or empty run."""

    def _on_auto_all_finished(self, results: list) -> None:
        """Slot: all tiles finished. Export deduplicated results to a GeoPackage."""
        self._set_zone_badge_enabled(True)
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_run_active(False)
                self.dock_widget.set_auto_status("idle")
            except (RuntimeError, AttributeError):
                pass
        # Refresh the credit display now that the run consumed credits.
        self._refresh_auto_credits()

        # Tiles the run actually got an answer for, zero-mask tiles included.
        # A floor on what was charged, not the charge itself: a tile whose
        # answer never arrived can still have been billed. `results` holds one
        # entry per DETECTION, so len(results) is wrong whenever a tile yields
        # 0 or 2+ masks.
        worker = self._auto_worker
        tiles_succeeded = getattr(worker, "tiles_succeeded", 0)
        self._capture_auto_mask_gsd(worker)
        # The all-finished signal fires from inside the worker's run loop, which
        # may not have returned yet: park a strong ref before dropping ours so a
        # still-running QThread is never garbage-collected (which aborts QGIS).
        if worker is not None and worker.isRunning():
            park_orphaned_worker(worker)
        self._auto_worker = None
        self._drop_auto_tile_bridge()
        self._auto_tel_stop_reason = "completed"
        self._finalize_auto_results(tiles_succeeded)

    def _finalize_auto_results(self, tiles_succeeded: int) -> None:
        """Turn the current merged set into a result.

        Headless/MCP: export straight to a layer (stable API contract).
        Interactive: enter the post-run review (refine, then Finish).

        Shared by a completed run and a stopped run (user cancel) so the
        billed partial results are never orphaned: a cancel drops the user
        into the same review of whatever was found so far.
        """
        # This is the terminal funnel for every finished/stopped run: disarm the
        # stall watchdog here so it can never tick into the review that follows.
        self._stop_auto_stall_watchdog()
        # Same reason for the tile grid: the run started by clearing it, and the
        # review must open on the detections alone. Anything that slipped back
        # onto the canvas mid-run dies here, before any review state is set.
        self._clear_zone_tile_grid()
        # Run summary for observability: split the wall-clock into render (upfront
        # basemap fetch) vs detection (submit/inference/poll). This is the single
        # log to read when a run felt slow - it says WHERE the time went without
        # needing the user. Production-safe (durations + counts only).
        try:
            import time as _time
            detect_ms = (
                int((_time.monotonic() - self._auto_detect_t0) * 1000)
                if self._auto_detect_t0 else 0
            )
            # mask/render px ratio: 1.0 = the server answered at the sent tile
            # resolution (full detail); ~2.0 = masks came back on a half grid.
            # Production-safe (a bare ratio), and the quickest field check that
            # a run really got full-resolution masks.
            ratio = (self._auto_mask_gsd / self._auto_gsd
                     if self._auto_mask_gsd > 0 and self._auto_gsd > 0 else 0.0)
            # "processed", not "billed": the count includes re-split quadrants,
            # which are free (RESPLIT_CHARGE_EVERY=0). The charged number is the
            # quoted base grid, shown by the dock's count row. Logging it as
            # "billed" read as a 2-3x overcharge in a 2026-08-04 field report.
            QgsMessageLog.logMessage(
                "Auto detection: run summary - render {} ms, detect {} ms, "
                "{} tile(s) processed, {} raw detection(s), mask/render px ratio "
                "{:.2f}, {} saturated tile(s) re-split (free), {} tile(s) gate-skipped"
                .format(
                    self._auto_render_ms, detect_ms, tiles_succeeded,
                    self._auto_raw_count, ratio,
                    getattr(self, "_auto_subdiv_tiles", 0),
                    getattr(self, "_auto_gate_skipped_tiles", 0)),
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )
        except (RuntimeError, AttributeError):
            pass

        # Pre-submit, uncharged tile drops. They used to push a message bar
        # banner the moment the review opened: it landed on top of the result
        # the user was there to look at, wrapped onto two lines and got cut,
        # and the Precision slider already caps at the level the source can
        # serve, so the common case was noise over a run that worked. They stay
        # in the log, and the one case that changes what the user sees (nothing
        # analyzed at all) is said on the run status line instead.
        # Masks the whole-tile blob guard dropped. Its own line, not part of the
        # uncharged-tile one below: these were charged, the model returned them,
        # and a client-side guard threw them away. On a large-parcel prompt that
        # is where a tile-shaped hole in the middle of a field comes from.
        blob_n = int(getattr(self, "_auto_blob_dropped", 0) or 0)
        if blob_n:
            hard_n, span_n, shape_n = getattr(
                self, "_auto_blob_split", (0, 0, 0))
            try:
                QgsMessageLog.logMessage(
                    f"Auto detection: whole-tile guard dropped {blob_n} mask(s) "
                    f"({hard_n} over the hard coverage cap, {span_n} spanning "
                    f"the tile, {shape_n} not compact enough). Count mode only; "
                    f"these were charged.",
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )
            except (RuntimeError, AttributeError):
                pass

        # MAP-mode counterpart: kept by design, never dropped. Logged because it
        # is the one number that explains a run whose outlines are right and
        # whose result is still one shape covering the whole zone.
        kept_map_n = int(getattr(self, "_auto_blob_kept_map", 0) or 0)
        cut_map_n = int(getattr(self, "_auto_blob_map_lowscore", 0) or 0)
        if kept_map_n or cut_map_n:
            try:
                scores = sorted(getattr(self, "_auto_map_cover_scores", ()) or ())
                if scores:
                    def _q(p: float) -> str:
                        return f"{scores[min(len(scores) - 1, int(len(scores) * p))]:.2f}"
                    spread = (f"scores p10 {_q(0.10)} p50 {_q(0.50)} "
                              f"p90 {_q(0.90)}")
                else:
                    spread = "no scores recorded"
                QgsMessageLog.logMessage(
                    f"Auto detection: whole-tile mask(s) in map mode - "
                    f"{kept_map_n} kept, {cut_map_n} cut by the score floor. "
                    f"{spread}",
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )
            except (RuntimeError, AttributeError, ValueError, IndexError):
                pass

        blank_n = int(getattr(self, "_auto_skipped_blank_tiles", 0) or 0)
        holes_n = int(getattr(self, "_auto_render_failed_tiles", 0) or 0)
        # Tiles the online source answered with a "no image here" card. Named
        # apart from the other two because it is the only one whose fix is the
        # user's: lower the precision or change layer. The other two are the
        # provider having a bad moment.
        unavail_n = int(getattr(self, "_auto_unavailable_tiles", 0) or 0)
        # Tiles the degenerate prefilter proved objectless (all no-data, or too
        # little ground left to hold anything). Same family as blank_n, no
        # request and no charge, but it used to reach no user-visible surface at
        # all: the quote counts them, the bill does not, and nothing said so.
        prefilt_n = int(getattr(self, "_auto_prefiltered_tiles", 0) or 0)
        if blank_n or holes_n or unavail_n or prefilt_n:
            QgsMessageLog.logMessage(
                f"Auto detection: {blank_n} blank tile(s) skipped, "
                f"{prefilt_n} empty tile(s) settled without a request, "
                f"{holes_n} render hole(s), "
                f"{unavail_n} tile(s) with no imagery at this detail. "
                f"None of these were charged.",
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )

        # Telemetry: report degraded tiles once per run (skipped / timed out /
        # blank-skipped / render holes). A tile the source answered with a "no
        # image here" card rides in render_failed_tiles: same family (the run
        # got no picture of that ground) and no new event property, so an older
        # dashboard keeps reading the same field.
        if (self._auto_skipped_tiles or self._auto_timeout_tiles
                or blank_n or holes_n or unavail_n or prefilt_n):
            try:
                from ...core import telemetry_run_events
                telemetry_run_events.track_auto_tiles_degraded(
                    run_id=self._auto_run_id or "",
                    skipped_tiles=self._auto_skipped_tiles,
                    timeout_tiles=self._auto_timeout_tiles,
                    # Prefiltered tiles ride in blank_tiles: same family (a
                    # render with nothing on it, dropped before submit, never
                    # charged) and no new event property, so an older dashboard
                    # keeps reading the same field.
                    blank_tiles=blank_n + prefilt_n,
                    render_failed_tiles=holes_n + unavail_n,
                )
            except Exception:
                pass  # nosec B110

        # The live pump is asynchronous, so the queue can still hold tiles when
        # the run signals completion. Headless blocks its caller and has no UI
        # to protect: drain synchronously and finalize inline, as before.
        if self._auto_headless_run:
            self._drain_auto_tiles_now()
            self._reset_auto_live_pipeline()
            # Same exemplar-only count-vs-map decision as the interactive path
            # (a MAP decision keeps the live merger, a SEPARATE one re-merges the
            # retained fragments), so the MCP contract reflects the same grouping.
            merged_ided = self._resolve_exemplar_finalize_ided()
            self._auto_merger = None
            if not merged_ided:
                self._record_auto_zero_result(tiles_succeeded)
                return
            # Sweep + build synchronously and export the default-filtered
            # visible set. No forced shape refine here: the Automatic path
            # stays faithful by default (the user opts into simplify/round/
            # expand/fill in the review).
            from ...core.polygon_exporter import drop_covered_objects
            merged_ided = drop_covered_objects(merged_ided)
            if not merged_ided:
                self._record_auto_zero_result(tiles_succeeded)
                return
            self._auto_objects = self._build_auto_objects(merged_ided)
            self._reset_review_refine_cache()
            visible, vis_scores = self._compute_visible_objects(
                self._fresh_review_params(), self._auto_refine_pixel_size(),
                with_scores=True)
            self._complete_auto_finalize(
                visible, tiles_succeeded, scores=vis_scores)
            return

        # Hold the run screen until the review opens. The slices below yield to
        # the event loop, so the dock repaints while they run: without this the
        # setup step (prompt, examples, Detect) came back for the length of the
        # finalize and flashed by right before the review. Every terminal
        # reaches this line in the same call stack as its own
        # set_auto_run_active(False), so nothing is painted in between.
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_finalizing(True)
            except (RuntimeError, AttributeError):
                pass
        # Interactive: drain the leftover queue COOPERATIVELY as finalize phase
        # "drain" (time-sliced, yielding to the event loop between slices). The
        # old synchronous drain blocked the GUI for the WHOLE backlog: on a
        # dense continuous run (roads) the last tiles' merges are the most
        # expensive, the async pump falls behind, and the backlog landed here
        # in one blocking bite - the end-of-run beachball. The gen bump
        # supersedes any in-flight finalize/reslice from a prior run.
        self._auto_finalize_gen += 1
        self._auto_finalize_state = {
            "mode": "finalize",
            "phase": "drain",
            "tiles_succeeded": tiles_succeeded,
            "gen": self._auto_finalize_gen,
        }
        self._step_auto_finalize_refine()

    def _finalize_drain_done(self, state: dict) -> None:
        """Drain phase complete: read the merger, stop the live pump machinery
        and seed the sweep phase (or record a zero result). Continues the
        cooperative finalize started in _finalize_auto_results."""
        # Stop the pump WITHOUT bumping the finalize gen (that would invalidate
        # the very state machine we are running); the preview-build gen is
        # bumped so a prior run's in-flight cache build can never leak in.
        self._stop_auto_live_pump()
        self._auto_preview_build_gen += 1
        self._auto_preview_build_state = None
        # Objects were stitched incrementally by the merger (confidence-agnostic),
        # so the result is already WHOLE objects (no tile seams), not raw
        # fragments. Read them SCORED (each object's representative score = the max
        # of its fragments) so the review filters act on whole objects, never on
        # cut fragments (the fix for "confidence cuts buildings in half").
        # (stable_id, geom, score) triples: the id carries each object's live
        # colour into the review so the hue is continuous across the run->review
        # boundary, not just stable during the run. For an exemplar-only run the
        # count-vs-map policy is decided HERE from the run's own masks, which may
        # replace the merged set with a fresh SEPARATE re-merge of the fragments.
        # Charged separately from the stitcher wait above: both used to land in
        # "drain", so a run that spent a minute waiting on the thread and a run
        # that spent it inside restore_absorbed_partitions (super-linear in the
        # parents carrying parts) read the same in the log.
        self._mark_finalize_phase(state, "restore")
        merged_ided = self._resolve_exemplar_finalize_ided()
        self._auto_merger = None
        # The end-of-run redundancy sweep (drop leftover partial readings mostly
        # painted over by larger objects: patches/strips double-painting a big
        # roof, which pairwise dedup cannot catch) is GEOS-heavy on dense runs,
        # so it runs cooperatively (phase "sweep") and never freezes the GUI at
        # "run finished".
        if not merged_ided:
            self._auto_finalize_state = None
            self._record_auto_zero_result(state["tiles_succeeded"])
            return

        # Crash net: the billed merged set reaches DISK before the fragile
        # sweep/build/filter/review tail runs on it. If anything past this
        # point dies, the next plugin start offers the file back, so a paid
        # run can no longer end with nothing delivered. Best-effort: a failed
        # write logs and changes nothing.
        self._mark_finalize_phase(state, "autosave")
        self._autosave_billed_results(merged_ided)

        from ...core.polygon_exporter import CoverSweep
        state.update({
            "phase": "sweep",
            "sweep": CoverSweep(list(merged_ided)),
            "sweep_before": len(merged_ided),
            "measurer": self._make_auto_area_measurer(),
            "params": self._fresh_review_params(),
            "pixel_size": self._auto_refine_pixel_size(),
        })
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(0, self._step_auto_finalize_refine)

    def _auto_run_network_dead(self, tiles_billed: int) -> bool:
        """True when a run reached its terminal having billed ZERO tiles because
        every tile failed to reach the service (skips / timeouts), not because
        the zone was genuinely empty. Drives both the honest "could not reach the
        service" banner and the telemetry decision to record such a run as a
        NETWORK failure instead of a healthy empty completion."""
        if tiles_billed > 0:
            return False
        return bool(
            getattr(self, "_auto_skipped_tiles", 0) or getattr(self, "_auto_timeout_tiles", 0))

    def _record_auto_zero_result(self, tiles_succeeded: int) -> None:
        """Shared 'no detections' bookkeeping for the finalize paths."""
        result = {
            "status": "completed",
            "instances": 0,
            "tiles_processed": tiles_succeeded,
            "layer_name": None,
        }
        prior = self._last_auto_result
        if isinstance(prior, dict) and prior.get("status") == "credits_exhausted":
            # Exhausted with nothing kept: preserve the exhaustion signal
            # (recorded before finalize) so the caller sees the real stop cause.
            result["status"] = "credits_exhausted"
            result["credits_remaining"] = prior.get("credits_remaining", 0)
        self._last_auto_result = result
        try:
            from ...core import telemetry_run_events
            ctx = self._auto_run_ctx or {}
            total = ctx.get("total", tiles_succeeded)
            # Terminal-event invariant: exactly ONE terminal event per run. A
            # cancel/exhaust already emitted its own terminal, so those runs emit
            # neither completed nor failed here. A "completed" run that billed
            # ZERO tiles only because every tile failed to reach the service
            # (network/timeout) is a FAILED run, not a healthy empty zone: it
            # emits one auto_detect_failed(NETWORK) instead of auto_detect_completed,
            # and skips auto_zero_result (there were no processed tiles to have
            # zero RESULTS from). This is the completion-path twin of the
            # cancel-path backend_stalled flag; the report dialog opened by
            # _on_auto_zero_detections still emits its separate plugin_error, the
            # same pairing the _on_auto_error NETWORK path uses.
            completed_terminal = self._auto_tel_stop_reason in (None, "completed")
            if completed_terminal and self._auto_run_network_dead(tiles_succeeded):
                telemetry_run_events.track_auto_detect_failed(
                    run_id=self._auto_run_id or "",
                    error_class="NETWORK",
                    tiles_done=tiles_succeeded,
                    duration_ms=self._auto_duration_ms(),
                    warming_ms=self._auto_warming_wait_ms(),
                )
            else:
                telemetry_run_events.track_auto_zero_result(
                    run_id=self._auto_run_id or "",
                    tiles=tiles_succeeded,
                    object_class=ctx.get("prompt") or "Example match",
                    had_exemplar=self._auto_exemplar_store.count() > 0,
                )
                if completed_terminal:
                    telemetry_run_events.track_auto_detect_completed(
                        run_id=self._auto_run_id or "",
                        duration_ms=self._auto_duration_ms(),
                        tiles_done=tiles_succeeded,
                        tiles_failed=max(0, total - tiles_succeeded),
                        instances_found=0,
                        instances_visible_at_default=0,
                        zero_at_default=True,
                        stop_reason="completed",
                        warming_ms=self._auto_warming_wait_ms(),
                        merge_mode_final="separate" if self._auto_merge_separate else "map",
                    )
        except Exception:
            pass  # nosec B110
        # D8: a HEALTHY empty interactive run (tiles processed, nothing found)
        # enters the review directly on the Correct step, where a drawn box
        # can trigger a surgical re-detect; the selection layer stays alive so
        # results have somewhere to land. Network-dead runs and the headless
        # path keep the quiet prompt-page line (there is no human to correct,
        # or nothing was actually scanned).
        healthy_zero = tiles_succeeded > 0 and not self._auto_headless_run
        healthy_zero = healthy_zero and not self._auto_run_network_dead(tiles_succeeded)
        healthy_zero = healthy_zero and self.dock_widget is not None
        if healthy_zero:
            try:
                self._enter_zero_detection_review(tiles_succeeded)
                self._clear_auto_raw_fragments()
                return
            except (RuntimeError, AttributeError):
                pass  # fall back to the quiet prompt-page line
        self._on_auto_zero_detections(tiles_succeeded)
        self._remove_auto_selection_layer()
        self._clear_auto_raw_fragments()

    def _on_auto_zero_detections(self, tiles_billed: int = 0) -> None:
        """A run that found nothing shows one quiet status line at the top of the
        prompt step ("No detection in this zone.") and leaves the user on the
        normal step-2 screen, where the Detect + Exit row already lets them
        adjust and run again or leave. No heavy guidance box. The terminal
        handlers already called set_auto_run_active(False), so the step-2
        controls are back; this just posts the message. Headless: just log."""
        # No review to hand the screen to: end the hold the finalize put on the
        # run card, which brings the step-2 controls back (see
        # set_auto_finalizing).
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_finalizing(False)
            except (RuntimeError, AttributeError):
                pass
        # Back on the prompt step: restore the zone fill the run start dropped
        # (live/review visual parity).
        self._set_zone_band_fill_visible(True)
        # Distinguish a genuinely empty zone from a run where every tile failed
        # to reach the service (offline / server down / timeouts). Telling an
        # offline user their zone is empty is misleading, so surface the real
        # cause instead. Same predicate the telemetry uses to record this run as
        # a NETWORK failure rather than a healthy empty completion.
        network_failed = self._auto_run_network_dead(tiles_billed)
        # Nothing billed, and nothing there to bill: every tile was dropped
        # before submit because the source had no image over the zone. "No
        # matches in this zone" would blame the model for a run that never
        # looked, so this one keeps its own line. It replaces the old message
        # bar banner, which fired on every partial hole as well.
        coverage_dead = bool(
            not network_failed and tiles_billed <= 0
            and (int(getattr(self, "_auto_unavailable_tiles", 0) or 0)
                 + int(getattr(self, "_auto_render_failed_tiles", 0) or 0))
        )
        _can_add_example = False
        _has_examples = False
        if not network_failed and not coverage_dead:
            try:
                _can_add_example = not self._auto_exemplar_store.is_full_for(1)
                _has_examples = self._auto_exemplar_store.count() > 0
            except (RuntimeError, AttributeError):
                _can_add_example = False
        report_payload = None
        if network_failed:
            msg = tr("Could not reach the service. Check your connection and try again.")
            log_msg = "Auto detection: run ended with no successful tiles (network/timeout)"
            # A run whose every tile failed to reach the service is a dead end:
            # the banner carries a report link and the dialog auto-opens below.
            # This path emits no plugin_error of its own, so the dialog tracks.
            report_payload = (
                tr("Automatic detection failed"), msg, "auto_detect_network_zero")
        elif coverage_dead:
            msg = tr("No image over this zone at this precision, so nothing was "
                     "analyzed (not charged). Lower Precision, or pick a layer "
                     "that covers this area.")
            log_msg = "Auto detection: run ended with no analyzable tiles (no imagery)"
        elif _can_add_example:
            # One info per state: the banner states the fact, the rescue
            # button right below it carries the action (draw an example -
            # the proven lever against empty results).
            msg = tr("No matches in this zone.")
            log_msg = "Auto detection: run completed with zero detections"
        else:
            # Example store full (examples were already the strategy): the
            # remaining levers are the word and the detail level.
            msg = tr(
                "No detection in this zone. Try a more specific object "
                "word, or more precision.")
            log_msg = "Auto detection: run completed with zero detections"
        if self.dock_widget and not self._auto_headless_run:
            try:
                self.dock_widget.set_auto_status(
                    "error" if network_failed else "info", msg,
                    report_payload=report_payload)
                if _can_add_example:
                    # The rescue row: draw-an-example leads (a full store
                    # would make it a silent no-op, so it is not offered
                    # then), plus the server-tuned stronger-word prefill.
                    self.dock_widget.show_auto_zero_assist(
                        self.dock_widget.auto_prompt_input.text().strip(),
                        has_examples=_has_examples)
                self._set_zone_badge_enabled(True)
                # Back on the prompt step: the run cleared the tile-grid preview
                # but the cost label still reads the last estimate. Redraw the
                # grid + cost for the kept zone (the Retry path restores it the
                # same way) so the screen is consistent before another Detect.
                self._restore_tile_grid_after_run()
            except (RuntimeError, AttributeError):
                pass
            # Open the report dialog for a total network dead end (once per run,
            # helper guards the flag); this path tracks its own plugin_error.
            if report_payload is not None:
                self._open_auto_error_report(*report_payload, track=True)
        QgsMessageLog.logMessage(
            log_msg, "AI Segmentation", level=Qgis.MessageLevel.Info,
        )

    def _on_auto_zero_assist_clicked(self, kind: str, to_prompt: str) -> None:
        """A zero-result rescue chip: record it, then perform the lever.

        'draw_example' arms the one-shot example draw (all the usual no-op
        guards in _on_add_exemplar_requested apply); 'synonym' prefills the
        suggested word - setText fires textChanged, which hides the chips and
        debounce-commits the new prompt, so the detail re-seed happens on the
        normal path."""
        from_prompt = ""
        try:
            from_prompt = self.dock_widget.auto_prompt_input.text().strip()
        except (RuntimeError, AttributeError):
            pass
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_zero_assist_clicked(kind, from_prompt, to_prompt)
        except Exception:
            pass  # nosec B110
        try:
            self.dock_widget.hide_auto_zero_assist()
        except (RuntimeError, AttributeError):
            pass
        if kind == "synonym" and to_prompt:
            try:
                self.dock_widget.auto_prompt_input.setText(to_prompt)
                self.dock_widget.auto_prompt_input.setFocus()
            except (RuntimeError, AttributeError):
                pass
        elif kind == "draw_example":
            self._on_add_exemplar_requested(1)
