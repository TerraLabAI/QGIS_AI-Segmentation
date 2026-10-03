






from __future__ import annotations

import threading
import time

from qgis.core import QgsTask
from qgis.PyQt.QtCore import pyqtSignal

from ..core.activation_manager import ACTIVATION_KEY_RE
from ..core.i18n import tr
from ..core.logging_utils import log
from ..core.server_dials import dial_in_range
from .adaptive_concurrency import OfflineFastFail


class LivePairingCodes:












    MAX_CODES = 5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._expiry_by_code: dict[str, float] = {}

    def add_code(self, code: str, ttl_s: float) -> None:
        if not code:
            return
        with self._lock:
            self._expiry_by_code.pop(code, None)
            self._expiry_by_code[code] = time.monotonic() + ttl_s
            while len(self._expiry_by_code) > self.MAX_CODES:
                self._expiry_by_code.pop(next(iter(self._expiry_by_code)))

    def discard_code(self, code: str) -> None:
        with self._lock:
            self._expiry_by_code.pop(code, None)

    def clear_codes(self) -> None:
        with self._lock:
            self._expiry_by_code.clear()

    def live_codes(self) -> list[str]:

        now = time.monotonic()
        with self._lock:
            for code in [c for c, t in self._expiry_by_code.items() if t <= now]:
                del self._expiry_by_code[code]
            return list(self._expiry_by_code)


class PairingPollTask(QgsTask):







    pairing_succeeded = pyqtSignal(str)
    pairing_failed = pyqtSignal(str, str)
    pairing_timeout = pyqtSignal()





    pairing_browser_seen = pyqtSignal()



    pairing_stalled = pyqtSignal(str)


    STALL_BROWSER_NOT_SEEN = "browser_not_seen"

    STALL_CODE_EXPIRED = "code_expired"



    STALL_AFTER_S = 45.0








    CODE_TTL_S = 1800.0
    EXPIRY_HINT_LEAD_S = 30.0







    OFFLINE_STREAK = 4



    OLDER_CODES_EVERY = 3


    QUIET_INTERVAL_S = 6.0

    def __init__(
        self,
        client,
        code: str,
        interval_s: float = 3.0,
        total_timeout_s: float = CODE_TTL_S,
        live_codes: LivePairingCodes | None = None,
        quiet: bool = False,
    ):
        super().__init__(tr("Connecting AI Segmentation"), QgsTask.Flag.CanCancel)
        self._client = client
        self._code = code
        self._interval_s = interval_s
        self._total_timeout_s = total_timeout_s
        self._live_codes = live_codes


        self._quiet = quiet
        self._key: str | None = None
        self._failure: tuple[str, str] | None = None
        self._timed_out = False

    @property
    def pairing_code(self) -> str:



        return self._code

    def is_active(self) -> bool:
        try:
            return self.status() in (
                QgsTask.TaskStatus.Running,
                QgsTask.TaskStatus.Queued,
                QgsTask.TaskStatus.OnHold,
            )
        except Exception:
            return False

    def _poll_older_codes(self) -> bool:





        if self._live_codes is None:
            return False
        for code in self._live_codes.live_codes():
            if code == self._code:
                continue
            if self.isCanceled():
                return False
            try:
                result = self._client.poll_pairing(code)
            except Exception:
                result = None
            if not isinstance(result, dict):
                continue
            status = result.get("status")
            if status == "ready":
                raw_key = result.get("activation_key")
                key = raw_key.strip() if isinstance(raw_key, str) else ""
                if ACTIVATION_KEY_RE.match(key):
                    self._key = key
                    log("Pairing poll: an earlier sign-in code was confirmed")
                    return True
                self._live_codes.discard_code(code)
            elif status in ("cancelled", "no_plan"):
                self._live_codes.discard_code(code)
        return False

    def _run_quiet(self) -> bool:
        while not self.isCanceled():
            if self._live_codes is None or not self._live_codes.live_codes():
                return False
            if self._poll_older_codes():
                return True
            interval_s = dial_in_range(
                "tuning.pairing.quiet_interval_s", self.QUIET_INTERVAL_S, 2.0, 60.0)
            self._sleep_cancellable(interval_s)
        return False

    def _retire_own_code(self) -> None:
        if self._live_codes is not None:
            self._live_codes.discard_code(self._code)

    def run(self) -> bool:
        if self._quiet:
            return self._run_quiet()
        started = time.monotonic()
        deadline = started + self._total_timeout_s
        browser_seen = False
        stall_hinted = False
        expiry_hinted = False
        last_logged_detail = ""
        offline_streak = 0
        rounds = 0
        while not self.isCanceled() and time.monotonic() < deadline:
            rounds += 1
            try:
                result = self._client.poll_pairing(self._code)
            except Exception:


                result = {"error": "poll failed", "code": "NO_INTERNET"}

            if self.isCanceled():
                return False

            status = result.get("status") if isinstance(result, dict) else None
            error_code = ""
            if isinstance(result, dict):
                error_code = str(result.get("code") or "").strip().upper()
            if error_code in OfflineFastFail.HARD_CODES:
                offline_streak += 1
                if offline_streak >= dial_in_range(
                        "tuning.pairing.offline_streak", self.OFFLINE_STREAK, 1, 20):
                    self._failure = (
                        tr("No connection to the sign-in service. Check your "
                           "internet connection, then click Sign in to try "
                           "again."),
                        "NO_INTERNET",
                    )
                    return False
            else:
                offline_streak = 0

            if status == "ready":
                raw_key = result.get("activation_key")
                key = raw_key.strip() if isinstance(raw_key, str) else ""
                if ACTIVATION_KEY_RE.match(key):
                    self._key = key
                    return True


                self._retire_own_code()
                self._failure = (
                    tr("Unexpected response from the server. Please try again."),
                    "BAD_KEY",
                )
                return False

            if status == "no_plan":


                self._retire_own_code()
                self._failure = (
                    tr(
                        "This account has no active AI Segmentation plan. "
                        "Reactivate it on terra-lab.ai, then click Sign in again."
                    ),
                    "NO_PLAN",
                )
                return False

            if status == "cancelled":


                self._retire_own_code()
                self._failure = (
                    tr("Sign-in was cancelled in the browser. Click Sign in to "
                       "try again."),
                    "CANCELLED",
                )
                return False











            waited_s = time.monotonic() - started
            stall_after_s = dial_in_range(
                "tuning.pairing.stall_after_s", self.STALL_AFTER_S, 10.0, 300.0)
            code_ttl_s = self._total_timeout_s
            expiry_hint_lead_s = dial_in_range(
                "tuning.pairing.expiry_hint_lead_s", self.EXPIRY_HINT_LEAD_S, 5.0, 120.0)
            if status == "pending" and not error_code and not browser_seen:
                browser_seen = True
                self.pairing_browser_seen.emit()
            elif not browser_seen and not stall_hinted and waited_s >= stall_after_s:



                stall_hinted = True
                self.pairing_stalled.emit(self.STALL_BROWSER_NOT_SEEN)
            elif (browser_seen and not expiry_hinted
                    and waited_s >= code_ttl_s - expiry_hint_lead_s):




                expiry_hinted = True
                self.pairing_stalled.emit(self.STALL_CODE_EXPIRED)

            if rounds % self.OLDER_CODES_EVERY == 0 and self._poll_older_codes():
                return True

            sleep_s = self._interval_s
            hint = result.get("retry_after") if isinstance(result, dict) else None
            if hint is not None:
                try:
                    delay = float(hint)
                    if 0 < delay < float("inf"):
                        sleep_s = max(delay, 1.0)
                        if error_code != "RATE_LIMITED":
                            sleep_s = min(sleep_s, 15.0)
                except (TypeError, ValueError):
                    pass








            detail = error_code or status
            detail = detail or "unknown"
            if detail != last_logged_detail:
                last_logged_detail = detail
                log(f"Pairing poll: waiting ({detail})")
            self._sleep_cancellable(min(sleep_s, max(0.0, deadline - time.monotonic())))

        if self.isCanceled():
            return False
        self._timed_out = True
        return False

    def _sleep_cancellable(self, seconds: float) -> None:

        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.isCanceled():
                return
            time.sleep(0.25)

    def finished(self, result: bool) -> None:
        if self.isCanceled():
            return
        if result and self._key:
            self.pairing_succeeded.emit(self._key)
        elif self._timed_out:
            self.pairing_timeout.emit()
        elif self._failure is not None:
            self.pairing_failed.emit(*self._failure)
