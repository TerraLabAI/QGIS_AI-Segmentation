









from __future__ import annotations

import time
from collections import deque

from ...core import transport_dials as _td
from ...core.error_policy import TRANSIENT_CODES
from ..tile_convert_pool import DEFAULT_MAX_WORKERS as _CONVERT_DEFAULT_MAX
from ..tile_convert_pool import SPARE_CORES as _CONVERT_SPARE_CORES
from ..tile_convert_pool import default_workers
from .retry_policy import RATE_LIMIT_SETBACK_CODES
from .run_lifecycle import _BILLED_DRAIN_STOP_REASONS
from .tile_submit import _as_float


class AutoRunLoopsMixin:


    def _run_streaming(self, total: int) -> None:












        from qgis.PyQt.QtCore import QCoreApplication, QEventLoop

        pending: deque = deque(enumerate(self._tiles))




        resubmit: deque = deque()




        in_flight: dict = {}
        submit_attempts: dict[int, int] = {}
        busy_since: dict[int, float] = {}
        completed = 0


        terminal_stop: tuple | None = None


        fatal_streak = 0


        _ef = getattr(QEventLoop, "ProcessEventsFlag", QEventLoop)
        _wait = _ef.WaitForMoreEvents | _ef.AllEvents





        _bucket = self.loop_phase_s
        _cursor = time.monotonic()

        def charge(name: str) -> None:
            nonlocal _cursor
            now = time.monotonic()
            _bucket[name] = _bucket.get(name, 0.0) + (now - _cursor)
            _cursor = now

        def fire_next() -> bool:


            nonlocal completed

            self._pump_render_deferred(pending)
            while resubmit or pending:




                ready_i = None
                now = time.monotonic()
                for i, entry in enumerate(resubmit):
                    if entry[3] <= now:
                        ready_i = i
                        break
                if ready_i is not None:
                    tile_idx, tile_spec, png_bytes, _ = resubmit[ready_i]
                    del resubmit[ready_i]
                elif pending:






                    picked = self._pop_next_pending(
                        pending, allow_wait=not in_flight)
                    if picked is None:
                        return False
                    tile_idx, spec = picked
                    if tile_idx in self._gate_skip or tile_idx in self._prefilter_skip:






                        self._discard_prefetch(tile_idx)
                        self._settle_empty_tile(
                            tile_idx, charged=tile_idx in self._gate_skip)
                        completed += 1
                        self._emit_progress(completed, total)
                        continue
                    status, payload = self._encode_or_defer(tile_idx, spec)
                    if status == "defer":
                        continue
                    if status == "empty":



                        self._settle_empty_tile(tile_idx, charged=False)
                        completed += 1
                        self._emit_progress(completed, total)
                        continue
                    if status == "skip":
                        completed += 1
                        self._emit_progress(completed, total)
                        continue
                    tile_spec, png_bytes = payload
                else:

                    return False
                submission, tile_transform = self._build_submission(
                    tile_idx, tile_spec, png_bytes
                )
                reply = self._client.post_detection_async(submission, self._auth)
                in_flight[reply] = (
                    tile_idx, tile_spec, tile_transform, png_bytes,
                    time.monotonic() + self._stream_reply_budget_s,
                )
                self._submit_at[tile_idx] = time.monotonic()
                self._watch_upload(reply, tile_idx)


                self._emit_run_phase("detecting")


                self._request_render_prefetch(pending)
                return True
            return False





        convert_workers = self._convert_workers
        if convert_workers <= 0:
            convert_workers = default_workers(
                default_max=_td.convert_pool_default_max(_CONVERT_DEFAULT_MAX),
                spare_cores=_td.convert_pool_spare_cores(_CONVERT_SPARE_CORES))







        self._request_render_prefetch(pending)
        self._convert_pool = self._open_convert_pool(convert_workers)







        primed = []

        def prime() -> None:
            charge("startup")
            while (not self._stop_requested and len(in_flight) < self._aimd.cap
                   and fire_next()):
                pass
            charge("fire")
            primed.append(True)

        self._convert_pool = self._open_convert_pool(
            convert_workers, while_booting=prime)
        if not primed:
            prime()
        else:
            charge("startup")

        while (
            in_flight or resubmit or pending or self._render_deferred or self._convert_pool.pending
        ) and not self._stop_requested:
            self.inflight_now = len(in_flight)
            if not in_flight:
                if not (pending or resubmit or self._render_deferred):



                    self._settle_converted_batch(
                        self._convert_pool.drain(timeout=0.25))
                    charge("convert_tail")
                    continue






                idle_s, idle_bucket = self._idle_slice(resubmit)
                if self._convert_pool.pending:
                    self._settle_converted_batch(
                        self._convert_pool.drain(timeout=idle_s))
                else:
                    self._interruptible_sleep(idle_s)
                charge(idle_bucket)



                while (not self._stop_requested and len(in_flight) < self._aimd.cap and fire_next()):
                    pass
                charge("fire")
                self._settle_converted_batch(self._convert_pool.drain())
                charge("settle")
                continue
            charge("loop_other")


            QCoreApplication.processEvents(_wait, 250)
            charge("net_wait")
            if self._stop_requested:
                break

            done = [r for r in in_flight if self._reply_is_finished(r)]



            expired = self._expire_stalled_replies(in_flight)
            if expired:




                completed += expired
                self._emit_progress(completed, total)
                self._aimd.on_setback()
            if not done:






                self._settle_converted_batch(self._convert_pool.drain())
                charge("settle")




                if not self._stop_requested and len(in_flight) < self._aimd.cap:
                    while (not self._stop_requested
                           and len(in_flight) < self._aimd.cap and fire_next()):
                        pass
                    charge("fire")
                    self._request_render_prefetch(pending)
                    charge("prefetch")
                continue


            cycle_setback = False
            cycle_progress = False



            cycle_refusals = 0
            cycle_answered = 0
            stop_payload = None
            for reply in done:
                tile_idx, tile_spec, tile_transform, png_bytes, _ = in_flight.pop(reply)
                response = self._read_reply(tile_idx, reply)
                outcome = self._classify_submit_response(tile_idx, response, tile_transform)
                kind = outcome[0]
                charge("read_parse")
                if kind == "completed_inline":
                    _, resp, ttf = outcome
                    _, _, tile_w, tile_h = tile_spec
                    submitted_at = self._submit_at.pop(tile_idx, None)
                    uploaded_at = self._uploaded_at.pop(tile_idx, None)
                    if submitted_at is not None:





                        left = uploaded_at if uploaded_at is not None else submitted_at
                        self.phase_upload_s += left - submitted_at
                        self.phase_predict_s += time.monotonic() - left
                        if left - submitted_at > self._upload_slow_s:
                            self.uploads_slow += 1
                            cycle_setback = True





                    self._convert_pool.submit(
                        self._plan_completed(resp, tile_idx, tile_w, tile_h, ttf))
                    charge("read_submit")
                    self.tiles_succeeded += 1
                    self._completed_idx.add(tile_idx)
                    cycle_progress = True
                    cycle_answered += 1


                    fatal_streak = 0
                    self._fastfail.reset()
                    completed += 1
                    self._emit_progress(completed, total)
                elif kind == "retry":


                    if len(outcome) > 3 and outcome[3] in RATE_LIMIT_SETBACK_CODES:
                        cycle_refusals += 1
                    give_up, delay, setback = self._retry_decision(
                        tile_idx, outcome, busy_since, submit_attempts)
                    cycle_setback = cycle_setback or setback
                    if give_up:
                        self._skip_network_tile(tile_idx)
                        completed += 1
                        self._emit_progress(completed, total)
                    else:
                        self._uploaded_at.pop(tile_idx, None)
                        resubmit.append(
                            (tile_idx, tile_spec, png_bytes,
                             time.monotonic() + delay))
                elif kind == "ok":



                    self._emit_warning(
                        f"Tile {tile_idx}: unexpected pending on direct path; skipping"
                    )
                    self._release_tile_clean_image(tile_idx)
                    completed += 1
                    self._emit_progress(completed, total)
                elif kind == "skip":
                    self._release_tile_clean_image(tile_idx)
                    completed += 1
                    self._emit_progress(completed, total)
                elif kind == "tile_fatal":



                    bad_code = outcome[1] or "UNKNOWN"
                    fatal_streak += 1
                    self._emit_warning(
                        f"Tile {tile_idx}: rejected ({bad_code}); skipping")
                    self._release_tile_clean_image(tile_idx)
                    completed += 1
                    self._emit_progress(completed, total)
                    if stop_payload is None and fatal_streak >= self._max_tile_fatals:
                        stop_payload = ("fatal", bad_code)
                elif stop_payload is None:
                    stop_payload = outcome




            self._free_read_replies(done)
            charge("read_replies")





            stop_payload = self._offline_stop(stop_payload)
            if stop_payload is not None:
                terminal_stop = stop_payload
                self._mark_stop(stop_payload)
                break





            total += self._drain_subtiles(pending)
            if self._capacity_setback(cycle_refusals, cycle_answered):
                cycle_setback = True
            self._settle_concurrency(cycle_setback, cycle_progress)




            backlog_cap = max(1, self._convert_pool.workers * self._convert_backlog_per_worker)
            if self._convert_pool.pending >= backlog_cap:
                self._settle_converted_batch(
                    self._convert_pool.drain(timeout=0.25))
                charge("backpressure")




            while not self._stop_requested and len(in_flight) < self._aimd.cap and fire_next():
                pass
            charge("fire")



            if not self._stop_requested:
                self._request_render_prefetch(pending)
            charge("prefetch")



            self._settle_converted_batch(self._convert_pool.drain())
            charge("settle")






        wind_down_end = time.monotonic() + 2 * self._stop_drain_budget_s

















        if self._stop_reason in _BILLED_DRAIN_STOP_REASONS and in_flight:
            drain_deadline = min(
                wind_down_end, time.monotonic() + self._stop_drain_budget_s)
            while in_flight and time.monotonic() < drain_deadline:
                QCoreApplication.processEvents(_wait, 100)
                drained = [r for r in in_flight if self._reply_is_finished(r)]
                for reply in drained:
                    tile_idx, tile_spec, tile_transform, png_bytes, _ = (
                        in_flight.pop(reply))
                    response = self._read_reply(tile_idx, reply)
                    outcome = self._classify_submit_response(
                        tile_idx, response, tile_transform)
                    if outcome[0] == "completed_inline":
                        _, resp, ttf = outcome
                        _, _, tile_w, tile_h = tile_spec
                        if self._emit_completed(resp, tile_idx, tile_w, tile_h, ttf):
                            self.tiles_succeeded += 1
                            self._completed_idx.add(tile_idx)
                        completed += 1
                        self._emit_progress(completed, total)


                self._free_read_replies(drained)


        stragglers = list(in_flight.keys())
        for reply in stragglers:
            try:
                if not self._reply_is_finished(reply):
                    reply.abort()
            except (RuntimeError, AttributeError):
                pass
        in_flight.clear()
        self._free_read_replies(stragglers)






        charge("wind_down")
        self._close_convert_pool(
            max(0.0, wind_down_end - time.monotonic()) if self._stop_requested
            else self._convert_drain_budget_s)
        charge("convert_close")




        if terminal_stop is not None:
            self._emit_stop(terminal_stop)
        self._emit_terminal()

    def _run_batched(self, total: int) -> None:





        pending: deque = deque(enumerate(self._tiles))






        resubmit: deque = deque()
        submit_attempts: dict[int, int] = {}



        busy_since: dict[int, float] = {}


        in_flight: dict = {}
        completed = 0


        fatal_streak = 0

        terminal_stop: tuple | None = None






        while (
            pending or in_flight or resubmit or self._render_deferred
        ) and not self._stop_requested:

            self._pump_render_deferred(pending)



            cycle_setback = False
            cycle_progress = False


            effective_cap = self._aimd.cap




            batch = []
            while (resubmit or pending) and (len(in_flight) + len(batch)) < effective_cap:



                ready_i = None
                now = time.monotonic()
                for i, entry in enumerate(resubmit):
                    if entry[3] <= now:
                        ready_i = i
                        break
                if ready_i is not None:
                    tile_idx, tile_spec, png_bytes, _ = resubmit[ready_i]
                    del resubmit[ready_i]
                    batch.append((tile_idx, tile_spec, png_bytes))
                    continue
                if not pending:
                    break
                tile_idx, spec = pending.popleft()



                status, payload = self._encode_or_defer(tile_idx, spec)
                if status == "defer":
                    continue
                if status == "empty":


                    self._settle_empty_tile(tile_idx, charged=False)
                    completed += 1
                    self._emit_progress(completed, total)
                    continue
                if status == "skip":


                    completed += 1
                    self._emit_progress(completed, total)
                    continue
                tile_spec, png_bytes = payload
                batch.append((tile_idx, tile_spec, png_bytes))

            submit_backoff: float | None = None
            batch_stop: tuple | None = None
            if batch and not self._stop_requested:
                for (tile_idx, tile_spec, png_bytes), outcome in zip(
                    batch, self._submit_batch(batch)
                ):
                    kind = outcome[0]
                    if kind == "ok":
                        _, request_id, poll_interval, max_wait, tile_transform = outcome
                        deadline = time.monotonic() + max_wait
                        in_flight[request_id] = (
                            tile_idx, tile_spec, poll_interval, max_wait,
                            deadline, tile_transform,
                        )

                        self._emit_run_phase("detecting")
                        fatal_streak = 0
                    elif kind == "completed_inline":



                        _, response, tile_transform = outcome
                        _, _, tile_w, tile_h = tile_spec
                        if self._emit_completed(
                            response, tile_idx, tile_w, tile_h, tile_transform
                        ):
                            self.tiles_succeeded += 1
                            self._completed_idx.add(tile_idx)
                            cycle_progress = True




                        fatal_streak = 0
                        self._fastfail.reset()
                        completed += 1
                        self._emit_progress(completed, total)
                    elif kind == "skip":





                        self._release_tile_clean_image(tile_idx)
                        completed += 1
                        self._emit_progress(completed, total)
                    elif kind == "tile_fatal":





                        bad_code = outcome[1] or "UNKNOWN"
                        fatal_streak += 1
                        self._emit_warning(
                            f"Tile {tile_idx}: rejected ({bad_code}); skipping")
                        self._release_tile_clean_image(tile_idx)
                        completed += 1
                        self._emit_progress(completed, total)
                        if batch_stop is None and fatal_streak >= self._max_tile_fatals:
                            batch_stop = ("fatal", bad_code)
                    elif kind == "retry":



                        give_up, delay, setback = self._retry_decision(
                            tile_idx, outcome, busy_since, submit_attempts)
                        cycle_setback = cycle_setback or setback
                        if give_up:
                            self._skip_network_tile(tile_idx)
                            completed += 1
                            self._emit_progress(completed, total)
                        else:
                            resubmit.append(
                                (tile_idx, tile_spec, png_bytes,
                                 time.monotonic() + delay))
                            submit_backoff = (
                                delay if submit_backoff is None
                                else max(submit_backoff, delay)
                            )
                    elif batch_stop is None:
                        batch_stop = outcome




            total += self._drain_subtiles(pending)





            batch_stop = self._offline_stop(batch_stop)
            if batch_stop is not None:
                terminal_stop = batch_stop
                self._mark_stop(batch_stop)

            self.inflight_now = len(in_flight)
            if self._stop_requested:
                break



            if submit_backoff is not None and not in_flight:
                self._settle_concurrency(cycle_setback, cycle_progress)
                self._interruptible_sleep(
                    min(max(submit_backoff, self._min_poll_backoff_s), 60.0)
                )
                continue

            if not in_flight:
                self._settle_concurrency(cycle_setback, cycle_progress)



                if (self._render_deferred or resubmit) and not pending:
                    self._interruptible_sleep(0.25)
                continue








            finished_ids = []
            next_backoff: float | None = None






            poll_ids = list(in_flight.keys())
            responses = self._client.get_detection_status_many(
                poll_ids, self._auth, should_abort=self._should_abort)

            if self._stop_requested:
                break

            for request_id, resp in zip(poll_ids, responses):
                tile_idx, tile_spec, poll_interval, max_wait, deadline, tile_transform = (
                    in_flight[request_id]
                )

                status = resp.get("status")

                if status == "completed":
                    _, _, tile_w, tile_h = tile_spec
                    if self._emit_completed(
                        resp, tile_idx, tile_w, tile_h, tile_transform
                    ):
                        self.tiles_succeeded += 1
                        self._completed_idx.add(tile_idx)
                        cycle_progress = True
                    completed += 1
                    self._emit_progress(completed, total)
                    finished_ids.append(request_id)

                elif status == "failed":



                    self.tiles_failed_server += 1
                    err = resp.get("error", "unknown failure")
                    self._emit_warning(
                        f"Tile {tile_idx} failed: {err}"
                    )
                    self._release_tile_clean_image(tile_idx)
                    completed += 1
                    self._emit_progress(completed, total)
                    finished_ids.append(request_id)

                elif status == "pending":
                    retry_after = _as_float(
                        resp.get("retry_after"), poll_interval)
                    if time.monotonic() > deadline:



                        self.tiles_timed_out += 1
                        self._emit_warning(
                            f"Tile {tile_idx} timed out after {int(max_wait)}s"
                        )
                        self._release_tile_clean_image(tile_idx)
                        completed += 1
                        cycle_setback = True
                        self._emit_progress(completed, total)
                        finished_ids.append(request_id)
                    else:


                        next_backoff = (
                            retry_after if next_backoff is None
                            else min(next_backoff, retry_after)
                        )

                else:

                    code = resp.get("code", "")
                    if code in TRANSIENT_CODES:



                        if time.monotonic() > deadline:
                            self.tiles_timed_out += 1
                            self._emit_warning(
                                f"Tile {tile_idx} timed out after {int(max_wait)}s"
                            )
                            self._release_tile_clean_image(tile_idx)
                            completed += 1
                            cycle_setback = True
                            self._emit_progress(completed, total)
                            finished_ids.append(request_id)
                        else:
                            next_backoff = (
                                poll_interval if next_backoff is None
                                else min(next_backoff, poll_interval)
                            )
                    else:
                        self._emit_warning(
                            f"Tile {tile_idx}: unexpected poll response code={code}"
                        )
                        self._release_tile_clean_image(tile_idx)
                        completed += 1
                        self._emit_progress(completed, total)
                        finished_ids.append(request_id)

            for rid in finished_ids:
                in_flight.pop(rid, None)


            total += self._drain_subtiles(pending)


            self._settle_concurrency(cycle_setback, cycle_progress)





            if in_flight and not finished_ids and next_backoff is not None and not self._stop_requested:
                self._interruptible_sleep(
                    min(max(next_backoff, self._min_poll_backoff_s), 5.0)
                )






        if self._stop_reason in _BILLED_DRAIN_STOP_REASONS and in_flight:


            self._drain_polled_on_stop(in_flight, completed, total)



        if terminal_stop is not None:
            self._emit_stop(terminal_stop)
        self._emit_terminal()
