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

from ..core.activation_manager import _KEY_RE
from ..core.i18n import tr
from ..core.logging_utils import log


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
    pairing_stalled = pyqtSignal()

    # How long to poll without ever seeing the browser before hinting the
    # user that the page probably never opened.
    STALL_AFTER_S = 45.0

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
        while not self.isCanceled() and time.monotonic() < deadline:
            try:
                result = self._client.poll_pairing(self._code)
            except Exception:
                result = {"error": "poll failed", "code": "NO_NETWORK"}

            if self.isCanceled():
                return False

            status = result.get("status") if isinstance(result, dict) else None
            if status == "ready":
                key = (result.get("activation_key") or "").strip()
                if _KEY_RE.match(key):
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
                        "Reactivate it on terra-lab.ai, then click Connect again."
                    ),
                    "NO_PLAN",
                )
                return False

            if status == "cancelled":
                # The user left the browser page without confirming. Terminal:
                # stop polling right away instead of spinning until timeout.
                self._failure = (
                    tr("Sign-in was cancelled in the browser. Click Connect to try again."),
                    "CANCELLED",
                )
                return False

            # Everything else - "pending" (browser reached /connect, user still
            # signing in), "not_found" (the browser never reached /connect, or
            # the code expired), and transient network/server errors - just
            # means "keep waiting". The poll is idempotent, so we loop until
            # ready or the overall deadline. Newer servers hint how long to
            # wait; absent or junk falls back to the fixed interval so older
            # servers behave unchanged.
            if status == "pending" and not browser_seen:
                browser_seen = True
                self.pairing_browser_seen.emit()
            elif (
                not browser_seen
                and not stall_hinted
                and time.monotonic() - started >= self.STALL_AFTER_S
            ):
                # Long wait and the server never saw the browser: the page
                # probably never opened (blocked browser, page error). Hint
                # the recovery paths instead of spinning silently.
                stall_hinted = True
                self.pairing_stalled.emit()

            sleep_s = self._interval_s
            hint = result.get("retry_after") if isinstance(result, dict) else None
            if hint is not None:
                try:
                    sleep_s = min(max(float(hint), 1.0), 15.0)
                except (TypeError, ValueError):
                    pass
            # Status detail makes user error reports diagnosable (pending =
            # browser seen, not_found = browser never seen, NO_NETWORK = the
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
