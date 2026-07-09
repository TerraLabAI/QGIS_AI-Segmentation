"""Pure-Python network-resilience helpers for the auto detection worker.

These carry no Qt / QGIS import so the concurrency and fast-fail logic is
unit-testable off a running QGIS (see tests/test_adaptive_concurrency.py).
``AutoDetectionWorker`` owns one instance of each and drives them from its
submit/poll loops; the loops stay in the worker (they touch Qt), the decision
math lives here.
"""
from __future__ import annotations


class AdaptiveConcurrency:
    """AIMD controller for the number of in-flight tile requests.

    A run opens NARROW (``start``) rather than at the full width. N concurrent
    tile uploads on a slow link split the uplink into N starving trickles that
    all approach the idle timeout together, abort, and re-upload the same bytes,
    which punishes exactly the users who can least afford it. The controller
    grows the window by 1 after each clean cycle (additive increase) up to
    ``maximum``, and halves it (floor ``minimum``) on a timeout / latency setback
    (multiplicative decrease). A short cooldown after a setback avoids
    immediately re-growing into the same congestion. On a healthy, already-warm
    link this climbs to ``maximum`` within a few cycles; on a bad link it
    collapses toward 1-2. Server capacity is unaffected (the fair queue still
    bounds it).
    """

    def __init__(
        self,
        start: int = 3,
        minimum: int = 1,
        maximum: int = 6,
        cooldown_cycles: int = 1,
    ) -> None:
        self._min = max(1, int(minimum))
        self._max = max(self._min, int(maximum))
        self._cap = min(max(int(start), self._min), self._max)
        self._cooldown_cycles = max(0, int(cooldown_cycles))
        self._cooldown = 0

    @property
    def cap(self) -> int:
        """Current in-flight ceiling, always within [minimum, maximum]."""
        return self._cap

    def on_clean_cycle(self) -> None:
        """Additive increase after a cycle that made progress with no setback.

        A cooldown pending from a recent setback is spent here instead of
        growing, so the window holds one cycle before climbing again.
        """
        if self._cooldown > 0:
            self._cooldown -= 1
            return
        if self._cap < self._max:
            self._cap += 1

    def on_setback(self) -> None:
        """Multiplicative decrease (halve, floor ``minimum``) on a timeout or
        latency spike, then hold for ``cooldown_cycles`` before growing again."""
        self._cap = max(self._min, self._cap // 2)
        self._cooldown = self._cooldown_cycles


class OfflineFastFail:
    """Consecutive hard-connectivity failure counter for a fast offline abort.

    A genuinely offline / DNS-dead run otherwise grinds the full per-tile retry
    budget (~150s per tile) before the end-of-run "could not reach the service"
    message appears, so a doomed 30-tile run sits on "Detecting..." for minutes.
    This counts CONSECUTIVE hard-connectivity failures (DNS / connection refused
    / proxy) and trips once ``threshold`` of them are seen in a row. The worker
    feeds it throughout the run: before the first success it aborts at
    ``threshold``; after it, the worker only aborts on a much longer streak
    (its ``_MIDRUN_OFFLINE_STREAK``, read via :pyattr:`streak`), so a genuine
    mid-run blip is still absorbed by the normal retry budget while a link
    that stays dead ends the run instead of grinding every tile's retries.
    A non-hard failure (a pure timeout, a server-busy answer) resets the
    streak: a slow-but-alive server must never fast-fail.
    """

    #: Codes that unambiguously mean "cannot reach the host" (see
    #: terralab_client._classify_qt_error). A pure TIMEOUT is deliberately NOT
    #: here: a very slow but alive link times out without being offline.
    HARD_CODES = frozenset({"DNS_ERROR", "CONNECTION_REFUSED", "PROXY_ERROR"})

    def __init__(self, threshold: int = 3) -> None:
        self._threshold = max(1, int(threshold))
        self._streak = 0

    @property
    def tripped(self) -> bool:
        """True once ``threshold`` consecutive hard failures have been recorded."""
        return self._streak >= self._threshold

    @property
    def streak(self) -> int:
        """Current consecutive hard-failure count (for the mid-run rule)."""
        return self._streak

    def reset(self) -> None:
        """Clear the streak. Called when the server was demonstrably reached (a
        completed tile, a busy/queue answer, or any non-hard outcome)."""
        self._streak = 0

    def record(self, code: str) -> bool:
        """Record one failed tile submit and return :pyattr:`tripped`.

        A hard-connectivity ``code`` advances the streak; any other code resets
        it (only an unbroken run of hard failures counts as offline).
        """
        if code in self.HARD_CODES:
            self._streak += 1
        else:
            self._streak = 0
        return self.tripped
