









from __future__ import annotations

import logging
import random
import time

from qgis.core import Qgis

from ...core.error_policy import BACKEND_UNAVAILABLE_CODES, OFFLINE_STOP_CODE
from ...core.server_dials import dial_in_range

__all__ = [
    "AutoRetryPolicyMixin",
    "HANDOFF_CODE",
    "HANDOFF_OVERLOAD_CODE",
    "RATE_LIMIT_SETBACK_CODES",
    "_AIMD_MIN",
    "_AIMD_START",
    "_AIMD_UPLOAD_GROW_S",
    "_BACKEND_UNAVAILABLE_DELAY_S",
    "_BACKEND_UNAVAILABLE_GIVEUP_STREAK",
    "_BACKEND_UNAVAILABLE_RETRIES",
    "_BUSY_JITTER",
    "_DEFAULT_MAX_WAIT_S",
    "_DEFAULT_POLL_INTERVAL_S",
    "_HANDOFF_MIN_DELAY_S",
    "_HANDOFF_OPEN_WINDOW_MAX_S",
    "_MAX_RATE_LIMIT_RETRIES",
    "_MIDRUN_OFFLINE_STREAK",
    "_MIN_POLL_BACKOFF_S",
    "_QUEUE_RETRY_BUDGET_S",
    "_REFUSAL_WINDOW",
    "_CachedReply",
    "_UPLOAD_SLOW_S",
    "_UPLOAD_STALL_S",
    "_DEAD_LINK_QUIET_S",
    "_ANSWER_QUIET_S",
    "_OUTAGE_PROBE_S",
    "_OUTAGE_MAX_S",
    "_OUTAGE_RESUME_MIN",
    "_OUTAGE_RESUME_FRACTION",
    "_ANSWER_QUIET_P90_FACTOR",
    "_SWEEP_ANSWER_P90_FACTOR",
    "_ABORT_REQUEUE_BASE_S",
    "logger",
]

logger = logging.getLogger(__name__)























_BACKEND_UNAVAILABLE_RETRIES = 3


_BACKEND_UNAVAILABLE_DELAY_S = 1.75












_BACKEND_UNAVAILABLE_GIVEUP_STREAK = 5





_MAX_RATE_LIMIT_RETRIES = 8







_QUEUE_RETRY_BUDGET_S = 300.0







_HANDOFF_MIN_DELAY_S = 1.0






_HANDOFF_OPEN_WINDOW_MAX_S = 2.0





HANDOFF_CODE = "CAPACITY_HANDOFF"


HANDOFF_OVERLOAD_CODE = "SERVICE_OVERLOADED"







RATE_LIMIT_SETBACK_CODES = frozenset(
    {"RATE_LIMITED", HANDOFF_CODE, HANDOFF_OVERLOAD_CODE})





_REFUSAL_WINDOW = 16




_BUSY_JITTER = (0.85, 1.30)





_UPLOAD_SLOW_S = 8.0


_DEFAULT_POLL_INTERVAL_S = 2.0


_DEFAULT_MAX_WAIT_S = 120.0





_MIN_POLL_BACKOFF_S = 0.5











_AIMD_START = 3







_AIMD_UPLOAD_GROW_S = 2.0


_AIMD_MIN = 1








_UPLOAD_STALL_S = 0.0







_DEAD_LINK_QUIET_S = 0.0







_ANSWER_QUIET_S = 0.0
_ANSWER_QUIET_P90_FACTOR = 0.0





_SWEEP_ANSWER_P90_FACTOR = 0.0











_OUTAGE_PROBE_S = 0.0
_OUTAGE_MAX_S = 0.0


_OUTAGE_RESUME_MIN = 1
_OUTAGE_RESUME_FRACTION = 1.0




_ABORT_REQUEUE_BASE_S = 1.0










_MIDRUN_OFFLINE_STREAK = 30


class _CachedReply:




    def __init__(self, answer_json: str) -> None:
        self._answer_json = answer_json

    def isFinished(self) -> bool:  # noqa: N802
        return True

    def abort(self) -> None:
        return None

    def deleteLater(self) -> None:  # noqa: N802
        return None

    def answer(self) -> dict:
        import json
        return json.loads(self._answer_json)


class AutoRetryPolicyMixin:


    def _retry_decision(
        self,
        tile_idx: int,
        outcome: tuple,
        busy_since: dict[int, float],
        submit_attempts: dict[int, int],
    ) -> tuple[bool, float, bool]:









        delay, is_busy = outcome[1], outcome[2]
        retry_code = outcome[3] if len(outcome) > 3 else ""
        now = time.monotonic()
        if is_busy:
            first = busy_since.setdefault(tile_idx, now)
            give_up = (now - first) > self._queue_retry_budget_s




            delay = min(60.0, max(1.0, delay)) * random.uniform(*self._busy_jitter)  # nosec B311







            self._fastfail.reset()
            return give_up, delay, retry_code == HANDOFF_OVERLOAD_CODE
        if retry_code in BACKEND_UNAVAILABLE_CODES:







            n = submit_attempts.get(tile_idx, 0) + 1
            submit_attempts[tile_idx] = n
            give_up = n > self._backend_unavailable_retries
            if give_up:
                self._note_unavailable_giveup(retry_code)
            self._fastfail.reset()
            jitter = random.uniform(*self._busy_jitter)  # nosec B311
            delay = self._backend_unavailable_delay_s * jitter
            return give_up, delay, False
        n = submit_attempts.get(tile_idx, 0) + 1
        submit_attempts[tile_idx] = n


        self.submit_network_retries += 1
        give_up = n > self._max_rate_limit_retries


        delay = min(30.0, delay * (2 ** min(n - 1, 4)))
        delay *= random.uniform(0.5, 1.0)  # nosec B311



        self._fastfail.record(retry_code)
        return give_up, delay, True

    def _note_unavailable_giveup(self, code: str) -> None:












        if getattr(self, "_unavailable_streak_mark", None) != self.tiles_succeeded:
            self._unavailable_streak_mark = self.tiles_succeeded
            self._unavailable_giveups = 0
        self._unavailable_giveups = getattr(self, "_unavailable_giveups", 0) + 1
        self._unavailable_stop_code = code or ""

    def _capacity_setback(self, refusals: int, answered: int) -> bool:















        recent = self._refusal_window
        recent.extend((True,) * refusals)
        recent.extend((False,) * answered)
        if len(recent) < recent.maxlen:
            return False
        if sum(recent) * 2 <= len(recent):
            return False
        recent.clear()
        return True

    def _apply_window_hint(self, response: dict) -> None:










        hint = response.get("window_hint")
        if hint is None or hint == self._window_hint:
            return
        try:
            hint = int(hint)
        except (TypeError, ValueError):
            return
        self._window_hint = hint
        maximum = max(1, min(hint, self._window_hint_max))
        if maximum == self._aimd.maximum:
            return
        self._aimd.set_maximum(maximum)



        if maximum > self._prefetch_depth:
            self._prefetch_depth = maximum
            self._render_window.set_maximum(maximum)
        if self._window_hint_logged:
            return
        self._window_hint_logged = True
        try:
            from qgis.core import QgsMessageLog

            QgsMessageLog.logMessage(
                f"Auto detection: server window hint {hint}, in-flight cap now "
                f"{maximum}",
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    @staticmethod
    def _free_read_replies(replies) -> None:














        from qgis.PyQt.QtCore import QCoreApplication, QEvent



        deferred_delete = getattr(QEvent, "Type", QEvent).DeferredDelete


        deferred_delete = getattr(deferred_delete, "value", deferred_delete)
        for reply in replies:
            try:
                reply.deleteLater()
                QCoreApplication.sendPostedEvents(reply, int(deferred_delete))
            except (RuntimeError, AttributeError, TypeError, ValueError):
                continue

    @staticmethod
    def _reply_is_finished(reply) -> bool:






        try:
            return bool(reply.isFinished())
        except RuntimeError:
            return True

    def _read_reply(self, tile_idx: int, reply) -> dict:















        if isinstance(reply, _CachedReply):



            return reply.answer()
        try:
            response = self._client.parse_reply(reply)
        except RuntimeError as err:
            first_time = tile_idx not in self._dead_reply_tiles
            self._dead_reply_tiles.add(tile_idx)



            if self._stop_requested:
                outcome = "run stopping, dropping it"
            elif first_time:
                outcome = "retrying it"
            else:
                outcome = "skipping"
            self._emit_warning(
                f"Tile {tile_idx}: reply was destroyed before it could be read "
                f"({err}); {outcome}"
            )
            if first_time:
                return {"error": "Reply destroyed before it was read",
                        "code": "TIMEOUT"}
            return {"error": "Reply destroyed before it was read",
                    "code": "REPLY_DESTROYED"}



        self._apply_window_hint(response)
        self._note_tile_balance(response)
        return response

    def _expire_stalled_replies(self, in_flight: dict, requeue=None) -> int:



























        now = time.monotonic()
        stall_s = float(getattr(self, "_upload_stall_s", 0.0) or 0.0)
        expired = []
        stalled = set()
        silent_probes = set()
        for reply, entry in in_flight.items():
            if self._reply_is_finished(reply):
                continue
            tile_idx = entry[0]
            if now > self._reply_deadline(entry):
                expired.append(reply)
            elif requeue is not None and self._outage_probe_silent(tile_idx, now):
                expired.append(reply)
                silent_probes.add(reply)
            elif (requeue is not None and stall_s > 0
                  and tile_idx not in self._uploaded_at):
                last = self._upload_progress_at.get(tile_idx)
                if last is not None and now - last > stall_s:
                    expired.append(reply)
                    stalled.add(reply)
        settled = 0
        for reply in expired:
            entry = in_flight.pop(reply)
            tile_idx = entry[0]
            try:
                reply.abort()
            except (RuntimeError, AttributeError):
                pass
            if tile_idx not in self._uploaded_at and requeue is not None:
                if reply in silent_probes:
                    self.outage_probe_cuts += 1
                    self._emit_warning(
                        f"Connection lost: probe tile {tile_idx} got nothing "
                        f"through in {self._outage_probe_allowance():.0f}s; probing again")
                if reply in stalled:
                    self.upload_stalls += 1
                    self._emit_warning(
                        f"Tile {tile_idx}: upload made no progress for "
                        f"{int(stall_s)}s; re-queued")
                if requeue(tile_idx, entry, "TIMEOUT"):
                    continue
                settled += 1
                continue
            if tile_idx in self._uploaded_at and requeue is not None:
                reposted = self._repost_uploaded_aborted(
                    tile_idx, entry, requeue)
                if reposted is True:
                    continue
                if reposted is False:
                    settled += 1
                    continue
            self._release_tile_clean_image(tile_idx)
            self.tiles_timed_out += 1
            settled += 1
            self._emit_warning(
                f"Tile {tile_idx} timed out after "
                f"{int(self._stream_reply_budget_s)}s")
        self._free_read_replies(expired)
        if silent_probes:
            self._drop_wedged_connections()
        return settled

    def _drop_wedged_connections(self) -> None:





        for nam in getattr(self, "_run_nam", None) or ():
            if nam is None:
                continue
            try:
                nam.clearConnectionCache()
            except (RuntimeError, AttributeError):
                pass

    def _repost_uploaded_aborted(self, tile_idx: int, entry, requeue):








        self.uploaded_aborts_reposted += 1
        return requeue(tile_idx, entry, "TIMEOUT")

    def _requeue_unuploaded(
        self, tile_idx: int, entry, code: str, resubmit: list,
        busy_since: dict, submit_attempts: dict,
    ) -> bool:







        give_up, delay, setback = self._retry_decision(
            tile_idx, ("retry", self._abort_requeue_base_s_value(), False, code),
            busy_since, submit_attempts)
        self._uploaded_at.pop(tile_idx, None)
        self._upload_progress_at.pop(tile_idx, None)
        if give_up:
            self._skip_network_tile(tile_idx)
            return False
        if setback:
            self._aimd.on_setback()
        self.uploads_requeued += 1
        resubmit.append((tile_idx, entry[1], entry[3], time.monotonic() + delay))
        return True

    @staticmethod
    def _reply_link_dropped(reply, outcome: tuple) -> bool:









        from ..adaptive_concurrency import OfflineFastFail
        code = outcome[3] if len(outcome) > 3 else ""
        if code in OfflineFastFail.HARD_CODES:
            return True
        try:
            from qgis.PyQt.QtNetwork import QNetworkReply

            from ...core.qt_compat import reply_http_status
            ne = getattr(QNetworkReply, "NetworkError", QNetworkReply)
            closed = getattr(ne, "RemoteHostClosedError",
                             getattr(QNetworkReply, "RemoteHostClosedError", None))
            if closed is not None and reply.error() == closed:
                return True
            hard = set(filter(None, (
                getattr(ne, name, getattr(QNetworkReply, name, None))
                for name in ("RemoteHostClosedError", "ConnectionRefusedError",
                             "HostNotFoundError", "NetworkSessionFailedError",
                             "TemporaryNetworkFailureError"))))
            return (reply.error() in hard
                    and reply_http_status(reply) is None)
        except (RuntimeError, AttributeError, TypeError, ImportError):
            return False

    def _quiet_since(self, tile_idx: int) -> float:


        posted = self._submit_at.get(tile_idx, 0.0)
        return max(posted, self._reply_byte_at.get(tile_idx, 0.0))

    def _requeue_link_victim(
        self, tile_idx: int, entry, resubmit: list, submit_attempts: dict,
    ) -> bool:







        n = submit_attempts.get(tile_idx, 0) + 1
        submit_attempts[tile_idx] = n
        self._uploaded_at.pop(tile_idx, None)
        self._upload_progress_at.pop(tile_idx, None)
        self._reply_byte_at.pop(tile_idx, None)
        if n > self._max_rate_limit_retries:
            self._skip_network_tile(tile_idx)
            return False
        self.uploads_requeued += 1
        delay = self._abort_requeue_base_s_value() * random.uniform(0.5, 1.0)  # nosec B311
        resubmit.append((tile_idx, entry[1], entry[3], time.monotonic() + delay))
        return True

    def _abort_link_victims(self, in_flight: dict, victims: list, requeue) -> int:




        settled = 0
        for reply in victims:
            entry = in_flight.pop(reply)
            tile_idx = entry[0]
            try:
                reply.abort()
            except (RuntimeError, AttributeError):
                pass
            if tile_idx in self._uploaded_at:
                ok = self._repost_uploaded_aborted(tile_idx, entry, requeue)
            else:
                ok = requeue(tile_idx, entry, "TIMEOUT")
            if not ok:
                settled += 1
        self._free_read_replies(victims)
        return settled

    def _sweep_dead_link(self, in_flight: dict, requeue) -> int:








        quiet_s = float(getattr(self, "_dead_link_quiet_s", 0.0) or 0.0)
        if quiet_s <= 0:
            return 0
        factor = float(getattr(self, "_sweep_p90_factor", _SWEEP_ANSWER_P90_FACTOR))
        answer_s = max(quiet_s, factor * self._answer_p90())
        now = time.monotonic()
        victims = [
            reply for reply, entry in in_flight.items()
            if not isinstance(reply, _CachedReply)
            and not self._reply_is_finished(reply)
            and now - self._quiet_since(entry[0]) > (
                answer_s if entry[0] in self._uploaded_at else quiet_s)]
        if not victims:
            return 0
        self.dead_link_sweeps += 1
        self._emit_warning(
            f"Connection dropped: re-sending {len(victims)} silent tile(s)")
        return self._abort_link_victims(in_flight, victims, requeue)

    def _answer_p90(self) -> float:

        times = sorted(self._answer_times)
        if not times:
            return 0.0
        return times[min(len(times) - 1, int(0.9 * len(times)))]

    def _answer_quiet_limit(self) -> float:


        floor = float(getattr(self, "_answer_quiet_s", 0.0) or 0.0)
        if floor <= 0:
            return 0.0
        factor = float(getattr(self, "_answer_quiet_p90_factor", _ANSWER_QUIET_P90_FACTOR))
        return max(floor, factor * self._answer_p90())

    def _guard_silent_drops(self, in_flight: dict, requeue) -> tuple[int, int]:




        if self.tiles_succeeded <= 0:
            return 0, 0
        limit = self._answer_quiet_limit()
        if limit <= 0:
            return 0, 0
        now = time.monotonic()
        victims = [
            reply for reply, entry in in_flight.items()
            if not isinstance(reply, _CachedReply)
            and entry[0] in self._uploaded_at
            and not self._reply_is_finished(reply)
            and now - self._quiet_since(entry[0]) > limit]
        if not victims:
            return 0, 0
        self.answer_quiet_reposts += len(victims)
        self._emit_warning(
            f"{len(victims)} tile(s) got no answer for {int(limit)}s; re-sending")
        return self._abort_link_victims(in_flight, victims, requeue), len(victims)

    def _abort_requeue_base_s_value(self) -> float:
        return float(getattr(self, "_abort_requeue_base_s", _ABORT_REQUEUE_BASE_S))

    def _outage_active(self) -> bool:
        return self._outage_since is not None

    def _window_has_room(self, in_flight: dict) -> bool:




        since = self._outage_since
        if since is None:
            return len(in_flight) < self._aimd.cap
        return not any(
            self._submit_at.get(entry[0], 0.0) >= since
            for entry in in_flight.values())

    def _enter_outage(self) -> None:



        if self._outage_since is not None or self.tiles_succeeded <= 0:
            return
        if float(getattr(self, "_outage_max_s", 0.0) or 0.0) <= 0:
            return
        self._outage_since = time.monotonic()
        self._outage_cap_before = self._aimd.cap
        self.outages += 1
        self._emit_warning("Connection lost: probing until it answers")

    def _leave_outage(self) -> None:


        if self._outage_since is None:
            return
        self._outage_s_total += time.monotonic() - self._outage_since
        self._outage_since = None
        least = int(getattr(self, "_outage_resume_min", _OUTAGE_RESUME_MIN))
        fraction = float(getattr(self, "_outage_resume_fraction", _OUTAGE_RESUME_FRACTION))
        self._aimd.restore(max(least, int(self._outage_cap_before * fraction)))

    def _outage_expired(self) -> bool:
        return (self._outage_since is not None
                and time.monotonic() - self._outage_since > self._outage_max_s)

    def _outage_probe_allowance(self) -> float:



        return float(self._outage_probe_s) + float(self._link_setup_max_s)

    def _outage_probe_silent(self, tile_idx: int, now: float) -> bool:






        since = self._outage_since
        if since is None:
            return False
        posted = self._submit_at.get(tile_idx)
        if posted is None or posted < since or tile_idx in self._uploaded_at:
            return False
        if self._reply_byte_at.get(tile_idx, 0.0) >= posted:
            return False
        return now - posted > self._outage_probe_allowance()

    def _note_link_setup(self, tile_idx: int, now: float) -> None:



        posted = self._submit_at.get(tile_idx)
        if posted is None:
            return
        last = self._upload_progress_at.get(tile_idx)
        if last is not None and last >= posted:
            return
        self._link_setup_max_s = max(self._link_setup_max_s, now - posted)

    def _outage_ms_total(self) -> int:

        total = self._outage_s_total
        if self._outage_since is not None:
            total += time.monotonic() - self._outage_since
        return int(total * 1000)

    def _requeue_probe(self, tile_idx: int, tile_spec, png_bytes, resubmit: list) -> None:



        self._uploaded_at.pop(tile_idx, None)
        self._upload_progress_at.pop(tile_idx, None)
        self._upload_last_sent.pop(tile_idx, None)
        self._reply_byte_at.pop(tile_idx, None)
        delay = self._outage_probe_s * random.uniform(0.8, 1.2)  # nosec B311
        resubmit.appendleft((tile_idx, tile_spec, png_bytes, time.monotonic() + delay))

    def _reply_deadline(self, entry) -> float:







        deadline = entry[4]
        uploaded_at = self._uploaded_at.get(entry[0])
        if uploaded_at is not None:
            deadline = max(deadline, uploaded_at + self._stream_reply_budget_s)
        return deadline

    def _watch_upload(self, reply, tile_idx: int) -> None:






        def _on_upload(sent: int, total: int, _idx: int = tile_idx) -> None:


            if total > 0 and sent != self._upload_last_sent.get(_idx):
                self._note_link_setup(_idx, time.monotonic())
                self._upload_last_sent[_idx] = sent
                self._upload_progress_at[_idx] = time.monotonic()
                self._reply_byte_at[_idx] = self._upload_progress_at[_idx]
            if total > 0 and sent >= total and _idx not in self._uploaded_at:
                now = time.monotonic()
                self._uploaded_at[_idx] = now
                self._grow_window_on_fast_upload(_idx, now)

        def _on_download(received: int, _total: int, _idx: int = tile_idx) -> None:

            if received > 0:
                self._reply_byte_at[_idx] = time.monotonic()
        try:
            reply.uploadProgress.connect(_on_upload)
            reply.downloadProgress.connect(_on_download)
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _grow_window_on_fast_upload(self, tile_idx: int, uploaded_at: float) -> None:












        limit_s = getattr(self, "_aimd_upload_grow_s", 0.0)
        if limit_s <= 0 or self._aimd.setbacks:
            return
        posted_at = self._submit_at.get(tile_idx)
        if posted_at is None or uploaded_at - posted_at > limit_s:
            return
        self._aimd.on_clean_cycle()

    def _drain_polled_on_stop(
        self, in_flight: dict, completed: int, total: int
    ) -> int:
















        drain_deadline = time.monotonic() + self._stop_drain_budget_s





        def past_budget() -> bool:
            return time.monotonic() >= drain_deadline

        while in_flight and not past_budget():
            poll_ids = list(in_flight.keys())
            try:
                responses = self._client.get_detection_status_many(
                    poll_ids, self._auth, should_abort=past_budget)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "AutoDetectionWorker: stop drain poll failed", exc_info=True)
                break
            answered = False
            for request_id, resp in zip(poll_ids, responses):
                status = resp.get("status")
                if status not in ("completed", "failed", "cancelled"):


                    continue
                entry = in_flight.pop(request_id, None)
                if entry is None:
                    continue
                answered = True
                if status != "completed":



                    continue
                tile_idx, tile_spec, _, _, _, tile_transform = entry
                _, _, tile_w, tile_h = tile_spec
                if self._emit_completed(
                    resp, tile_idx, tile_w, tile_h, tile_transform
                ):
                    self.tiles_succeeded += 1
                    self._completed_idx.add(tile_idx)
                completed += 1
                self._emit_progress(completed, total)
            if in_flight and not answered:



                time.sleep(
                    max(0.0, min(0.25, drain_deadline - time.monotonic())))
        return completed

    def _offline_stop(self, stop_payload: tuple | None) -> tuple | None:














        if stop_payload is not None:
            return stop_payload
        giveups = getattr(self, "_unavailable_giveups", 0)
        if giveups >= dial_in_range(
            "tuning.network.backend_unavailable_giveup_streak",
            _BACKEND_UNAVAILABLE_GIVEUP_STREAK, 1, 200,
        ):



            return ("fatal", getattr(self, "_unavailable_stop_code", "") or "SERVER_ERROR")
        if not self._fastfail.tripped:
            return stop_payload
        if (
            self.tiles_succeeded == 0 or self._fastfail.streak >= self._midrun_offline_streak
        ):
            self.stopped_offline = True
            return ("fatal", OFFLINE_STOP_CODE)
        return stop_payload
