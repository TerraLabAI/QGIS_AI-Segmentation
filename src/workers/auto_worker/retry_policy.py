









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
    "_UPLOAD_SLOW_S",
    "_WINDOW_HINT_MAX",
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


_AIMD_MIN = 1







_WINDOW_HINT_MAX = 12










_MIDRUN_OFFLINE_STREAK = 30


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

    def _expire_stalled_replies(self, in_flight: dict) -> int:

















        now = time.monotonic()
        expired = [
            reply for reply, entry in in_flight.items()
            if now > self._reply_deadline(entry) and not self._reply_is_finished(reply)
        ]
        for reply in expired:
            tile_idx = in_flight.pop(reply)[0]
            try:
                reply.abort()
            except (RuntimeError, AttributeError):
                pass
            self._release_tile_clean_image(tile_idx)
            self.tiles_timed_out += 1
            self._emit_warning(
                f"Tile {tile_idx} timed out after "
                f"{int(self._stream_reply_budget_s)}s")
        self._free_read_replies(expired)
        return len(expired)

    def _reply_deadline(self, entry) -> float:







        deadline = entry[4]
        uploaded_at = self._uploaded_at.get(entry[0])
        if uploaded_at is not None:
            deadline = max(deadline, uploaded_at + self._stream_reply_budget_s)
        return deadline

    def _watch_upload(self, reply, tile_idx: int) -> None:






        def _on_upload(sent: int, total: int, _idx: int = tile_idx) -> None:
            if total > 0 and sent >= total and _idx not in self._uploaded_at:
                self._uploaded_at[_idx] = time.monotonic()
        try:
            reply.uploadProgress.connect(_on_upload)
        except (RuntimeError, AttributeError, TypeError):
            pass

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
            return ("fatal", OFFLINE_STOP_CODE)
        return stop_payload
