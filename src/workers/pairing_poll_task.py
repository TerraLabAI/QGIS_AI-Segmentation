"""QgsTask that polls the server until a browser pairing code is bound to a key.

Backs the one-click "Connect" onboarding: the plugin mints a code, opens the
browser to /connect?code=..., and this task polls /api/plugin/pair/poll until
the signed-in user binds the code to their activation key. Mirrors AI Edit so
both plugins share the same sign-in flow.
"""
from __future__ import annotations

import time

from qgis.core import QgsTask
from qgis.PyQt.QtCore import pyqtSignal

from ..core.activation_manager import ACTIVATION_KEY_RE
from ..core.i18n import tr
from ..core.logging_utils import log
from .adaptive_concurrency import OfflineFastFail


class PairingPollTask(QgsTask):
    """Poll until the browser handoff completes.

    Emits exactly one of pairing_succeeded(key) / pairing_failed(msg, code) /
    pairing_timeout(). Payloads are plain str (already copied), so finished()
    never touches live context off the worker thread.
    """

    pairing_succeeded = pyqtSignal(str)
    pairing_failed = pyqtSignal(str, str)
    pairing_timeout = pyqtSignal()
    # The server marks the code 'pending' as soon as /connect renders, so we
    # can tell "user is in the browser, signing in" from "the page never
    # loaded" (browser blocked, page error, wrong machine). Emitted at most
    # once each; older servers never report 'pending' before success, in
    # which case only the stalled hint can fire and the flow is unchanged.
    pairing_browser_seen = pyqtSignal()
    # Carries why the wait is stuck: STALL_BROWSER_NOT_SEEN or
    # STALL_CODE_EXPIRED. A slot that takes no argument still works, so the
    # reason reaches the panel only once the panel asks for it.
    pairing_stalled = pyqtSignal(str)

    # The page never opened: no browser has reached the server yet.
    STALL_BROWSER_NOT_SEEN = "browser_not_seen"
    # The browser did open, and the code it carries has run out of time.
    STALL_CODE_EXPIRED = "code_expired"

    # How long to poll without ever seeing the browser before hinting the
    # user that the page probably never opened.
    STALL_AFTER_S = 45.0

    # A pairing code lives ten minutes on the server, so a wait that reaches
    # that mark is waiting on a code nothing can bind any more. The hint goes
    # out a little early: the poll window is the same ten minutes, and a line
    # the user only sees at the moment the wait is called off is not a hint.
    CODE_TTL_S = 600.0
    EXPIRY_HINT_LEAD_S = 30.0

    # Polls in a row that never reached a server before the wait is called off.
    # The browser signs in from this same machine, so a link that cannot carry
    # the poll cannot carry the sign-in either; holding the full deadline only
    # delays the one sentence that helps, and ends on "try again" without ever
    # naming the connection. Any answer at all resets the count, so a single
    # blip inside a working sign-in costs nothing.
    OFFLINE_STREAK = 4

    def __init__(
        self,
        client,
        code: str,
        interval_s: float = 3.0,
        total_timeout_s: float = 600.0,
    ):
        super().__init__(tr("Connecting AI Segmentation"), QgsTask.Flag.CanCancel)
        self._client = client
        self._code = code
        self._interval_s = interval_s
        self._total_timeout_s = total_timeout_s
        self._key: str | None = None
        self._failure: tuple[str, str] | None = None
        self._timed_out = False

    @property
    def pairing_code(self) -> str:
        """The code this poll waits on. A cancel is cooperative, so the task
        can still read as running while a NEW code needs a listener: the caller
        compares this before reusing the worker."""
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

    def run(self) -> bool:
        started = time.monotonic()
        deadline = started + self._total_timeout_s
        browser_seen = False
        stall_hinted = False
        expiry_hinted = False
        offline_streak = 0
        while not self.isCanceled() and time.monotonic() < deadline:
            try:
                result = self._client.poll_pairing(self._code)
            except Exception:
                # The code has to be one OfflineFastFail knows, or the streak
                # below never counts and the wait runs to the full deadline.
                result = {"error": "poll failed", "code": "NO_INTERNET"}

            if self.isCanceled():
                return False

            status = result.get("status") if isinstance(result, dict) else None
            error_code = ""
            if status is None and isinstance(result, dict):
                error_code = str(result.get("code") or "").strip().upper()
            if error_code in OfflineFastFail.HARD_CODES:
                offline_streak += 1
                if offline_streak >= self.OFFLINE_STREAK:
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
                key = (result.get("activation_key") or "").strip()
                if ACTIVATION_KEY_RE.match(key):
                    self._key = key
                    return True
                # Server said ready but the key is malformed: terminal, never
                # persist garbage.
                self._failure = (
                    tr("Unexpected response from the server. Please try again."),
                    "BAD_KEY",
                )
                return False

            if status == "no_plan":
                # The signed-in account has no active plan to connect. Terminal:
                # stop now with a clear message instead of polling to timeout.
                self._failure = (
                    tr(
                        "This account has no active AI Segmentation plan. "
                        "Reactivate it on terra-lab.ai, then click Sign in again."
                    ),
                    "NO_PLAN",
                )
                return False

            if status == "cancelled":
                # The user left the browser page without confirming. Terminal:
                # stop polling right away instead of spinning until timeout.
                self._failure = (
                    tr("Sign-in was cancelled in the browser. Click Sign in to "
                       "try again."),
                    "CANCELLED",
                )
                return False

            # Everything else - "pending" (browser reached /connect, user still
            # signing in), "not_found" (the browser never reached /connect, or
            # the code expired), and transient network/server errors - just
            # means "keep waiting". The poll is idempotent, so we loop until
            # ready or the overall deadline. The server cannot tell an expired
            # code from one that never existed, so the two hints below split
            # the wait instead: which one fires depends on whether a browser
            # was ever seen. Newer servers hint how long to wait; absent or
            # junk falls back to the fixed interval so older servers behave
            # unchanged.
            waited_s = time.monotonic() - started
            if status == "pending" and not browser_seen:
                browser_seen = True
                self.pairing_browser_seen.emit()
            elif not browser_seen and not stall_hinted and waited_s >= self.STALL_AFTER_S:
                # Long wait and the server never saw the browser: the page
                # probably never opened (blocked browser, page error). Hint
                # the recovery paths instead of spinning silently.
                stall_hinted = True
                self.pairing_stalled.emit(self.STALL_BROWSER_NOT_SEEN)
            elif (browser_seen and not expiry_hinted
                    and waited_s >= self.CODE_TTL_S - self.EXPIRY_HINT_LEAD_S):
                # The browser did reach the page and the code has run out of
                # time, so the server answers the same "keep waiting" it would
                # for a code that never existed. Say which one it is: a new
                # code is one click away, and waiting is not.
                expiry_hinted = True
                self.pairing_stalled.emit(self.STALL_CODE_EXPIRED)

            sleep_s = self._interval_s
            hint = result.get("retry_after") if isinstance(result, dict) else None
            if hint is not None:
                try:
                    sleep_s = min(max(float(hint), 1.0), 15.0)
                except (TypeError, ValueError):
                    pass
            # Status detail makes user error reports diagnosable (pending =
            # browser seen, not_found = browser never seen, NO_INTERNET = the
            # poll itself failing). Never log the code itself.
            detail = status or (result.get("code") if isinstance(result, dict) else None)
            log(f"Pairing poll: waiting ({detail or 'unknown'})")
            self._sleep_cancellable(sleep_s)

        if self.isCanceled():
            return False
        self._timed_out = True
        return False

    def _sleep_cancellable(self, seconds: float) -> None:
        """Sleep in short slices so a cancel is honored quickly."""
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
