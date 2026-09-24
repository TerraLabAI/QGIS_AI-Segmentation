










from __future__ import annotations

import logging
import time

from ...core.error_policy import OFFLINE_STOP_CODE

__all__ = [
    "AutoRunLifecycleMixin",
    "_BILLED_DRAIN_STOP_REASONS",
    "_EMPTY_TILES_BEFORE_NOTICE",
    "_MAX_CONSECUTIVE_TILE_FATALS",
    "_STOP_DRAIN_BUDGET_S",
    "logger",
]

logger = logging.getLogger(__name__)









_STOP_DRAIN_BUDGET_S = 2.5













_BILLED_DRAIN_STOP_REASONS = ("user", "exhausted", "stalled")








_MAX_CONSECUTIVE_TILE_FATALS = 5











_EMPTY_TILES_BEFORE_NOTICE = 12


class AutoRunLifecycleMixin:


    def _run_detection(self) -> None:
        from ...api.terralab_client import TerraLabClient
        from ...core import run_timeline

        run_timeline.mark("worker_run")
        self._client = TerraLabClient()





        self._run_nam = self._client.acquire_predict_nam()
        run_timeline.mark("client_ready")
        total = len(self._tiles)

        if total == 0:
            self._terminal_sent = True
            self.all_tiles_finished.emit([])
            return





        self._run_started_at = time.monotonic()
        self._paid_tiles_total = total
        self._paid_tiles_done = 0
        self._resplit_deadline = 0.0
        self._resplit_dropped = 0





        self._emit_run_phase("imagery")






        self._emit_progress(0, total)



        self._prepare_stamps()






        self._build_clip_engine()



        self._resolve_ground_unit_scale()

        logger.debug(
            "AutoDetectionWorker: run_id=%s tiles=%d prompt=%r exemplars=%d",
            self._run_id, total, self._prompt, len(self._stamps),
        )









        if getattr(self._client, "detection_direct", False):




            self._prespawn_convert_children()
            run_timeline.mark("prep_done")
            self._run_gate_scan()
            run_timeline.mark("gate_done")
            if self._terminal_sent or self._stop_requested:



                self._emit_terminal()
                return
            self._run_streaming(total)
        else:
            self._run_batched(total)

    def _should_abort(self) -> bool:




        return self._stop_requested





    @property
    def tiles_awaiting_conversion(self) -> int:








        pool = getattr(self, "_convert_pool", None)
        try:
            return int(pool.pending) if pool is not None else 0
        except (AttributeError, TypeError, ValueError):
            return 0

    def remaining_tiles(self) -> list[tuple[int, int, int, int]]:






        return [t for i, t in enumerate(self._tiles) if i not in self._completed_idx]

    def run_health_summary(self) -> dict:

















        return {
            "submit_retries": int(self.submit_network_retries),
            "tiles_skipped_network": int(self.tiles_skipped_network),
            "tiles_timed_out": int(self.tiles_timed_out),
            "tiles_failed_server": int(self.tiles_failed_server),
            "renders_slow": int(self.renders_slow),
            "render_window_floor": int(self.render_window_floor),
        }

    def client_profile(self) -> dict:








        loop = {}
        try:
            loop = {str(k): float(v) for k, v in self.loop_phase_s.items()}
        except (AttributeError, TypeError, ValueError):
            loop = {}
        try:
            cap = int(self._aimd.maximum)
            setbacks = int(self._aimd.setbacks)
        except (AttributeError, TypeError, ValueError):
            cap, setbacks = 0, 0



        polygonizer, most = None, 0
        for name in ("gdal", "tracer", "fallback", "fallback_fast"):
            try:
                crops = int(getattr(self, f"polygonized_{name}", 0) or 0)
            except (TypeError, ValueError):
                crops = 0
            if crops > most:
                polygonizer, most = name, crops
        return {
            **self._density_profile(),
            "convert_pool": self._convert_pool_kind or "",
            "convert_workers": int(self._convert_pool_workers),
            "convert_fallback_reason": self._convert_fallback_reason or "",
            "tiles_convert_failed": int(self.tiles_convert_failed),
            "convert_fail_reason": self._convert_fail_reason or "",
            "polygonizer": polygonizer,
            "convert_rescued_tiles": int(self._convert_rescued_tiles),
            "render_wait_s": float(self.phase_render_s),
            "encode_s": float(self.phase_encode_s),
            "predict_s": float(self.phase_predict_s),
            "loop_wall_s": float(sum(loop.values())),
            "convert_s": float(self.phase_convert_s),
            "upload_mb": float(self.upload_bytes) / 1048576.0,
            "upload_s": float(self.phase_upload_s),
            "uploads_slow": int(self.uploads_slow),
            "inflight_cap_final": cap,
            "window_setbacks": setbacks,
            "tiles_timed_out": int(self.tiles_timed_out),
            "tiles_answered": int(self.tiles_succeeded),
            "http_429": int(self.http_429),
            "http_503": int(self.http_503),
            "loop": loop,
        }

    def _emit_run_phase(self, name: str) -> None:



        if getattr(self, "_run_phase_sent", None) == name:
            return
        self._run_phase_sent = name
        try:
            self.run_phase.emit(name)
        except RuntimeError:
            pass

    def _emit_warning(self, message: str) -> None:








        try:
            self.warning.emit(message)
        except RuntimeError:
            pass  # nosec B110

    def _note_tile_outcome(self, found_something: bool) -> None:







        if self._empty_notice_sent:
            return
        if found_something:
            self._empty_notice_sent = True
            return
        self._empty_tile_streak += 1
        if self._empty_tile_streak < self._empty_tiles_before_notice:
            return
        self._empty_notice_sent = True
        try:
            self.nothing_found_yet.emit(self._empty_tile_streak)
        except RuntimeError:
            pass  # nosec B110

    def _emit_progress(self, completed: int, total: int) -> None:







        shown_total = self._progress_total or total
        try:
            self.progress.emit(self._progress_offset + completed, shown_total)
        except RuntimeError:
            pass  # nosec B110

    def _note_busy(self, position: int, depth: int, eta_s: int) -> None:


        payload = (position, depth, eta_s)
        if payload != self._last_queue_emit:
            self._last_queue_emit = payload
            try:
                self.queue_state.emit(position, depth, eta_s)
            except RuntimeError:
                pass  # nosec B110

    def _note_flowing(self) -> None:

        if self._last_queue_emit is not None:
            self._last_queue_emit = None
            try:
                self.queue_state.emit(0, 0, 0)
            except RuntimeError:
                pass  # nosec B110

    def _note_tile_balance(self, response: dict) -> None:








        if not isinstance(response, dict):
            return
        carried = {}
        for key in ("credits_remaining", "free_detections_remaining"):
            value = response.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                carried[key] = int(value)
        if not carried:
            return



        previous = self.last_tile_balance
        if isinstance(previous, dict):
            for key, value in previous.items():
                if key not in carried or value < carried[key]:
                    carried[key] = value
        self.last_tile_balance = carried

    def _emit_terminal(self) -> None:







        self._flush_withheld()


        self._tile_clean_image.clear()




        self._clear_rescan_marks()
        if self._terminal_sent:
            return
        self._terminal_sent = True
        from ...core import run_timeline
        run_timeline.mark("worker_terminal")




        try:
            if self._stop_requested:
                if self._stop_reason == "user":
                    self.cancelled.emit()
                elif self._density_replan_asked():
                    self.density_replan.emit(self._density_decision)
            else:
                self.all_tiles_finished.emit([])
        except RuntimeError:
            pass  # nosec B110

    def _mark_stop(self, stop_payload: tuple) -> None:














        if self._stop_reason is None:
            self._stop_reason = (
                "exhausted" if stop_payload[0] == "exhausted" else "error")
        self._stop_requested = True

    def _emit_stop(self, stop_payload: tuple) -> None:

















        self._mark_stop(stop_payload)
        self._flush_withheld()



        self._clear_rescan_marks()
        self._terminal_sent = True
        try:
            if stop_payload[0] == "exhausted":
                self.credits_exhausted.emit(stop_payload[1])
            elif self._stop_reason == "user":




                self.cancelled.emit()
            else:



                self.error.emit(self._submit_error_message(
                    stop_payload[1],
                    stop_payload[2] if len(stop_payload) > 2 else "",
                ))
        except RuntimeError:
            pass

    def _settle_empty_tile(self, tile_idx: int, charged: bool = True) -> None:











        self._note_tile_outcome(False)
        try:
            self.tile_completed.emit(tile_idx, [])
        except RuntimeError:



            pass  # nosec B110
        if charged:
            self.tiles_succeeded += 1
        self._completed_idx.add(tile_idx)

    def _settle_concurrency(self, setback: bool, progress: bool) -> None:




        if setback:
            self._aimd.on_setback()
        elif progress:
            self._aimd.on_clean_cycle()

    def _idle_slice(self, resubmit, ceiling: float = 0.25) -> tuple[float, str]:












        now = time.monotonic()
        retry_due = min((entry[3] for entry in resubmit), default=None)
        render_due = min((entry[0] for entry in self._render_deferred), default=None)
        on_retry = retry_due is not None and (
            render_due is None or retry_due <= render_due)
        nearest = retry_due if on_retry else render_due
        if nearest is None:
            return ceiling, "render_wait"
        return (max(0.01, min(ceiling, nearest - now)),
                "retry_wait" if on_retry else "render_wait")





    def _interruptible_sleep(self, seconds: float) -> None:





        waited = 0.0



        while waited < seconds and not self._stop_requested:
            step = min(0.25, seconds - waited)
            time.sleep(step)
            waited += step

    def _skip_network_tile(self, tile_idx: int) -> None:



        self.tiles_skipped_network += 1
        self._release_tile_clean_image(tile_idx)
        self._emit_warning(f"Tile {tile_idx}: submit retries exhausted; skipping")

    def _submit_error_message(self, code: str, detail: str = "") -> str:









        if code == OFFLINE_STOP_CODE:
            from ...core.i18n import tr

            return tr("No internet connection. Check your connection and try again.")
        base = f"Tile submit failed: {code}"
        detail = (detail or "").strip()
        return f"{base}: {detail}" if detail else base
