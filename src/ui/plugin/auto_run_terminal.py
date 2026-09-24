






from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr
from .shared import park_orphaned_worker


class AutoRunTerminalMixin:


    def _on_auto_all_finished(self, results: list) -> None:

        from ...core import run_timeline
        run_timeline.mark("all_finished_slot")
        self._set_zone_badge_enabled(True)
        if self.dock_widget:
            try:








                if not self._auto_headless_run:
                    self.dock_widget.set_auto_finalizing(True)
                self.dock_widget.set_auto_run_active(False)
                self.dock_widget.set_auto_status("idle")
            except (RuntimeError, AttributeError):
                pass




        if not self._apply_tile_balance(
                getattr(self._auto_worker, "last_tile_balance", None)):
            self._refresh_auto_credits()






        worker = self._auto_worker
        tiles_succeeded = getattr(worker, "tiles_succeeded", 0)
        self._capture_auto_mask_gsd(worker)


        from .auto_client_profile import snapshot_worker_profile
        snapshot_worker_profile(self, worker)



        if worker is not None and worker.isRunning():
            park_orphaned_worker(worker)
        self._auto_worker = None
        self._drop_auto_tile_bridge()
        self._auto_tel_stop_reason = "completed"



        try:
            from ...core.cloud_notice_seen import mark_cloud_notice_seen

            mark_cloud_notice_seen()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        self._finalize_auto_results(tiles_succeeded)

    def _finalize_auto_results(self, tiles_succeeded: int) -> None:











        self._stop_auto_stall_watchdog()


        self._reset_credits_backoff()



        self._clear_zone_tile_grid()




        try:
            import time as _time
            detect_ms = (
                int((_time.monotonic() - self._auto_detect_t0) * 1000)
                if self._auto_detect_t0 else 0
            )




            ratio = (self._auto_mask_gsd / self._auto_gsd
                     if self._auto_mask_gsd > 0 and self._auto_gsd > 0 else 0.0)








            stitcher = getattr(self, "_auto_stitcher", None)
            fold_ms = float(getattr(stitcher, "fold_ms", 0.0) or 0.0)
            folded = int(getattr(stitcher, "tiles_folded", 0) or 0)
            QgsMessageLog.logMessage(
                "Auto detection: run summary - render {} ms, detect {} ms, "
                "{} tile(s) processed, {} raw detection(s), mask/render px ratio "
                "{:.2f}, {} saturated tile(s) re-split (free), {} tile(s) gate-skipped, "
                "stitch {:.0f} ms over {} tile(s), live draw {:.0f} ms over {} tick(s)"
                .format(
                    self._auto_render_ms, detect_ms, tiles_succeeded,
                    self._auto_raw_count, ratio,
                    getattr(self, "_auto_subdiv_tiles", 0),
                    getattr(self, "_auto_gate_skipped_tiles", 0),
                    fold_ms, folded,
                    float(getattr(self, "_auto_live_draw_ms", 0.0)),
                    int(getattr(self, "_auto_live_draw_ticks", 0))),
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )
        except (RuntimeError, AttributeError):
            pass















        armed_n, blob_n, tile_m = self._auto_blob_guard_stats()
        if blob_n or armed_n:
            hard_n, span_n, shape_n = getattr(
                self, "_auto_blob_split", (0, 0, 0))
            try:
                QgsMessageLog.logMessage(
                    f"Auto detection: whole-tile guard armed on {armed_n} "
                    f"mask(s) of {self._auto_raw_count} and dropped {blob_n} "
                    f"({hard_n} over the hard coverage cap, {span_n} spanning "
                    f"the tile, {shape_n} not compact enough); tile side "
                    f"{tile_m:.0f} units. Count mode only; these were charged.",
                    "AI Segmentation", level=Qgis.MessageLevel.Info,
                )
            except (RuntimeError, AttributeError):
                pass




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




        unavail_n = int(getattr(self, "_auto_unavailable_tiles", 0) or 0)





        prefilt_n = int(getattr(self, "_auto_prefiltered_tiles", 0) or 0)
        if blank_n or holes_n or unavail_n or prefilt_n:
            QgsMessageLog.logMessage(
                f"Auto detection: {blank_n} blank tile(s) skipped, "
                f"{prefilt_n} empty tile(s) settled without a request, "
                f"{holes_n} render hole(s), "
                f"{unavail_n} tile(s) with no imagery at this detail.",
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )






        if (self._auto_skipped_tiles or self._auto_timeout_tiles
                or blank_n or holes_n or unavail_n or prefilt_n):
            try:
                from ...core import telemetry_run_events
                telemetry_run_events.track_auto_tiles_degraded(
                    run_id=self._auto_run_id or "",
                    skipped_tiles=self._auto_skipped_tiles,
                    timeout_tiles=self._auto_timeout_tiles,




                    blank_tiles=blank_n + prefilt_n,
                    render_failed_tiles=holes_n + unavail_n,
                )
            except Exception:
                pass  # nosec B110




        if self._auto_headless_run:
            try:
                self._finalize_headless_run(tiles_succeeded)
            finally:




                self._auto_headless_run = False
                self._auto_review_preset_overrides = None
            return







        if self.dock_widget:
            try:
                self.dock_widget.set_auto_finalizing(True)
            except (RuntimeError, AttributeError):
                pass







        self._auto_finalize_gen += 1
        self._auto_finalize_state = {
            "mode": "finalize",
            "phase": "drain",
            "tiles_succeeded": tiles_succeeded,
            "gen": self._auto_finalize_gen,
        }
        self._step_auto_finalize_refine()

    def _finalize_headless_run(self, tiles_succeeded: int) -> None:





        self._drain_auto_tiles_now()
        self._reset_auto_live_pipeline()



        merged_ided = self._server_finalize_rows_now()
        if merged_ided is None:



            merged_ided = self._resolve_exemplar_finalize_ided()
            self._auto_merger = None
            if not merged_ided:
                self._record_auto_zero_result(tiles_succeeded)
                return




            from ...core.polygon_exporter import drop_covered_objects
            merged_ided = drop_covered_objects(merged_ided)
            if not merged_ided:
                self._record_auto_zero_result(tiles_succeeded)
                return


            merged_ided = self._align_auto_footprints_now(merged_ided)
        self._auto_objects = self._build_auto_objects(merged_ided)
        self._reset_review_refine_cache()
        visible, vis_scores = self._compute_visible_objects(
            self._fresh_review_params(), self._auto_refine_pixel_size(),
            with_scores=True)
        self._complete_auto_finalize(
            visible, tiles_succeeded, scores=vis_scores)

    def _finalize_drain_done(self, state: dict) -> None:






        self._stop_auto_live_pump()
        self._auto_preview_build_gen += 1
        self._auto_preview_build_state = None
















        if self._begin_server_finalize_phase(state):
            return
        self._mark_finalize_phase(state, "restore")
        merged_ided, remerge = self._begin_exemplar_finalize_merge()
        self._auto_merger = None
        from qgis.PyQt.QtCore import QTimer
        if remerge is not None:




            state["phase"] = "remerge"
            state["remerge"] = remerge
            state["remerge_t0"] = None
            QTimer.singleShot(0, self._step_auto_finalize_refine)
            return
        self._seed_finalize_sweep_phase(state, merged_ided)

    def _seed_finalize_sweep_phase(self, state: dict, merged_ided: list) -> None:








        if not merged_ided:
            self._auto_finalize_state = None
            self._record_auto_zero_result(state["tiles_succeeded"])
            return






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





        if tiles_billed > 0:
            return False
        return bool(
            getattr(self, "_auto_skipped_tiles", 0) or getattr(self, "_auto_timeout_tiles", 0))

    def _record_auto_zero_result(self, tiles_succeeded: int) -> None:

        result = {
            "status": "completed",
            "instances": 0,
            "tiles_processed": tiles_succeeded,
            "layer_name": None,
        }
        prior = self._last_auto_result
        if isinstance(prior, dict) and prior.get("status") == "credits_exhausted":


            result["status"] = "credits_exhausted"
            result["credits_remaining"] = prior.get("credits_remaining", 0)
        self._last_auto_result = result
        try:
            from ...core import telemetry_run_events
            ctx = self._auto_run_ctx or {}
            total = ctx.get("total", tiles_succeeded)











            completed_terminal = self._auto_tel_stop_reason in (None, "completed")
            from .auto_client_profile import client_profile_props
            if completed_terminal and self._auto_run_network_dead(tiles_succeeded):
                telemetry_run_events.track_auto_detect_failed(
                    run_id=self._auto_run_id or "",
                    error_class="NETWORK",
                    tiles_done=tiles_succeeded,
                    duration_ms=self._auto_duration_ms(),
                    warming_ms=self._auto_warming_wait_ms(),
                    client_profile=client_profile_props(self),
                )
            else:
                telemetry_run_events.track_auto_zero_result(
                    run_id=self._auto_run_id or "",
                    tiles=tiles_succeeded,
                    object_class=ctx.get("prompt") or "Example match",
                    had_exemplar=self._auto_exemplar_store.count() > 0,
                )
                if completed_terminal:
                    blob_armed, blob_dropped, tile_m = (
                        self._auto_blob_guard_stats())
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
                        blob_armed=blob_armed,
                        blob_dropped=blob_dropped,
                        tile_ground_m=tile_m,
                        client_profile=client_profile_props(self),
                    )
        except Exception:
            pass  # nosec B110


        from .auto_client_profile import stop_gui_gap_watch
        stop_gui_gap_watch(self)


        try:
            from ...core.run_log_capture import send_run_log
            send_run_log("completed")
        except Exception:  # noqa: BLE001  # nosec B110
            pass






        healthy_zero = tiles_succeeded > 0 and not self._auto_headless_run
        healthy_zero = healthy_zero and not self._auto_run_network_dead(tiles_succeeded)
        healthy_zero = healthy_zero and self.dock_widget is not None
        if healthy_zero:
            try:
                self._enter_zero_detection_review(tiles_succeeded)
                self._clear_auto_raw_fragments()
                return
            except (RuntimeError, AttributeError):
                pass
        self._on_auto_zero_detections(tiles_succeeded)
        self._remove_auto_selection_layer()
        self._clear_auto_raw_fragments()

    def _on_auto_zero_detections(self, tiles_billed: int = 0) -> None:









        if self.dock_widget:
            try:
                self.dock_widget.set_auto_finalizing(False)
            except (RuntimeError, AttributeError):
                pass


        self._set_zone_band_fill_visible(True)









        quota_stop = getattr(self, "_auto_quota_stop_banner", None)
        self._auto_quota_stop_banner = None
        network_failed = (not quota_stop
                          and self._auto_run_network_dead(tiles_billed))














        coverage_dead = bool(
            not network_failed and not quota_stop and tiles_billed <= 0
            and (int(getattr(self, "_auto_unavailable_tiles", 0) or 0)
                 + int(getattr(self, "_auto_render_failed_tiles", 0) or 0)
                 + int(getattr(self, "_auto_skipped_blank_tiles", 0) or 0))
        )



        stopped_early = bool(
            not quota_stop and not network_failed and not coverage_dead
            and tiles_billed <= 0
            and getattr(self, "_auto_tel_stop_reason", None) in ("cancelled", "stalled"))






        convert_failed_n = int(getattr(self, "_auto_convert_failed_tiles", 0) or 0)
        convert_dead = bool(
            not quota_stop and not network_failed and not coverage_dead
            and not stopped_early and convert_failed_n > 0
            and convert_failed_n * 2 >= max(1, tiles_billed))
        _can_add_example = False
        _has_examples = False
        if (not network_failed and not coverage_dead and not quota_stop
                and not stopped_early and not convert_dead):
            try:
                _can_add_example = not self._auto_exemplar_store.is_full_for(1)
                _has_examples = self._auto_exemplar_store.count() > 0
            except (RuntimeError, AttributeError):
                _can_add_example = False
        report_payload = None
        if quota_stop:
            msg = quota_stop
            log_msg = "Auto detection: run stopped on the monthly allowance"
        elif network_failed:
            msg = tr("Could not reach the service. Check your connection and try again.")
            log_msg = "Auto detection: run ended with no successful tiles (network/timeout)"



            report_payload = (
                tr("Automatic detection failed"), msg, "auto_detect_network_zero")
        elif coverage_dead:
            msg = tr("No image over this zone at this precision, so nothing was "
                     "analyzed (not charged). Lower Precision, or pick a layer "
                     "that covers this area.")
            log_msg = "Auto detection: run ended with no analyzable tiles (no imagery)"
        elif stopped_early:
            msg = tr("Detection stopped before any result came back. "
                     "Run Detect again when you are ready.")
            log_msg = "Auto detection: stopped before any tile answered"
        elif convert_dead:
            msg = tr("The service answered for {n} tile(s) and this plugin "
                     "could not read the results, so nothing was placed on "
                     "the map. This is a fault on our side, not your zone or "
                     "your wording. Send the report and we will look at it, "
                     "and write to us so we can put the tiles back."
                     ).format(n=convert_failed_n)
            log_msg = (f"Auto detection: {convert_failed_n} answered tile(s) "
                       f"failed to convert; run ended with nothing kept")
            report_payload = (
                tr("Automatic detection failed"), msg, "auto_detect_convert_zero")
        else:








            typed = ""
            try:
                typed = self.dock_widget.auto_prompt_input.text().strip()
            except (RuntimeError, AttributeError):
                typed = "?"
            if typed:


                msg = tr('No matches in this zone. Try one plain word for the '
                         'object, "building" and not "building footprint".')
            else:
                msg = tr('No matches in this zone. Add the object\'s name, like '
                         '"building", or draw a clearer example.')
            log_msg = "Auto detection: run completed with zero detections"
        if self.dock_widget and not self._auto_headless_run:
            try:
                self.dock_widget.set_auto_status(
                    "error" if (network_failed or convert_dead) else "info", msg,
                    report_payload=report_payload)
                if _can_add_example:



                    self.dock_widget.show_auto_zero_assist(
                        self.dock_widget.auto_prompt_input.text().strip(),
                        has_examples=_has_examples)
                self._set_zone_badge_enabled(True)




                self._restore_tile_grid_after_run()
            except (RuntimeError, AttributeError):
                pass


            if report_payload is not None:
                self._open_auto_error_report(*report_payload, track=True)
        QgsMessageLog.logMessage(
            log_msg, "AI Segmentation", level=Qgis.MessageLevel.Info,
        )

    def _on_auto_zero_assist_clicked(self, kind: str, to_prompt: str) -> None:







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
