"""How long an Automatic run takes, in the words a user can act on.

A run reads its tiles a few at a time, so the wall clock a user waits grows
with the tile count and not with the surface drawn. The card that shows the
zone is where the run is accepted or abandoned, and until it carries a
duration the only honest thing on screen is a bar that starts after the
decision is already made.

Two properties decide the shape of what is shown:

- **The relationship is a line.** A fixed opening cost, then a steady cost per
  tile. That per-tile cost is a served constant fitted to the fleet, so it
  already carries what the adaptive width does on a real link, and throughput
  does not improve with size.
- **The spread is too wide for one number.** Wait times vary by more than a
  factor of two between an ordinary run and a slow one. Most of that is time
  no tile is in flight at all: imagery that arrives slowly, a retry, a link
  that stalls. None of it is visible before the run starts, so no amount of
  client-side arithmetic can narrow the band. The answer is to quote the band
  instead of pretending to a single number.

So the estimate is a RANGE, from the typical run to the slow one, in one unit,
rounded coarsely. It is deliberately blunt: "8-18 min" survives being wrong,
"11 min" does not.

Every dial fails open to a shipped fallback, and the whole estimate fails to
the empty string. A cold cache, an old server or a malformed value all leave
the card exactly as it was before this existed.
"""
from __future__ import annotations

import math

from .i18n import tr

# Seconds of wall clock each tile adds, and the fixed cost of opening a run
# (waking the service, filling the pipeline). Generic fallbacks: the server
# holds the values that follow the service, and these two only have to be the
# right order of magnitude when it says nothing.
SECONDS_PER_TILE_DEFAULT = 0.40
FIXED_SECONDS_DEFAULT = 1.0

# How much slower the top of the quoted band runs than its bottom. One factor,
# not a table: the band widens with the tile count on its own, because it
# multiplies a number that already grows.
SLOW_FACTOR_DEFAULT = 2.3

# Under this many seconds nothing is shown, measured on the TYPICAL run and
# not on the slow end of the band. A short run finishes before a reader has
# finished the sentence, and an estimate on it is decoration on a state that
# already has its one piece of information (the surface). The floor sits high
# enough that the band always has whole minutes to round to, which is what
# keeps the row short enough to fit the dock at its narrowest.
MIN_SHOW_SECONDS_DEFAULT = 40.0

# Bands a served value is held inside. Wide enough for a service several times
# faster or slower than today, tight enough that a malformed deploy cannot put
# hours on the card.
_SECONDS_PER_TILE_BAND = (0.02, 30.0)
_FIXED_SECONDS_BAND = (0.0, 600.0)
_SLOW_FACTOR_BAND = (1.0, 10.0)
_MIN_SHOW_BAND = (0.0, 3600.0)

_SECTION_KEY = "run_eta"


def run_seconds(tiles: int, policy: dict | None = None,
                seconds_per_tile: float | None = None) -> tuple[float, float]:
    """Typical and slow wall clock for a run of ``tiles`` tiles, in seconds.

    ``(0.0, 0.0)`` when the tile count is not a usable positive number, which
    is every caller's signal to show nothing.

    ``seconds_per_tile`` replaces the fleet-wide dial when the caller knows
    better: the pace this account's own recent runs actually held (see
    :func:`own_pace_seconds_per_tile`). The fleet dial is measured on the
    service side, and a slow uplink or a corporate gateway can hold a run
    several times longer than that without the service ever seeing it.
    """
    count = _positive_int(tiles)
    if count <= 0:
        return (0.0, 0.0)
    dials = _run_eta_dials(policy)
    per_tile = dials["seconds_per_tile"]
    if seconds_per_tile is not None:
        per_tile = _dial({"own": seconds_per_tile}, "own", per_tile,
                         _SECONDS_PER_TILE_BAND)
    typical = dials["fixed_s"] + per_tile * count
    return (typical, typical * dials["slow_factor"])


def own_pace_seconds_per_tile(plan: object) -> float | None:
    """The pace of this account's own recent runs, from a served run plan.

    The plan MAY carry an additive ``run_pace`` object with a
    ``seconds_per_tile`` figure the server measured on the account's last
    runs, wall clock from first tile sent to last tile answered. Absent,
    malformed or out of band reads as None, and the caller keeps the fleet
    dial: an older server, a first run, or a bad deploy must all leave the
    estimate exactly as it is today.
    """
    if not isinstance(plan, dict):
        return None
    block = plan.get("run_pace")
    if not isinstance(block, dict):
        return None
    value = _dial(block, "seconds_per_tile", 0.0, _SECONDS_PER_TILE_BAND)
    return value if value > 0 else None


def friendly_run_eta(tiles: int, policy: dict | None = None) -> str:
    """A rounded duration band for ``tiles`` tiles, or ``""`` to show nothing.

    Reuses the duration sentence the queue estimate already ships, so a band
    costs no new wording in any language: the fill is a range rather than a
    single number, and every locale renders it in its own unit word.

    Whole minutes only. Seconds would need a longer sentence in every language
    for a wait the floor has already decided is not worth warning about, and
    the row it goes on is one line of a fold header on a dock that can be 260
    pixels wide.
    """
    typical, slow = run_seconds(tiles, policy)
    if typical <= 0 or typical < _min_show_seconds(policy):
        return ""
    low = max(1, int(round(typical / 60.0)))
    high = max(low, int(math.ceil(slow / 60.0)))
    return tr("{m} min").format(m=_span(low, high))


def friendly_run_eta_about(tiles: int, policy: dict | None = None,
                           seconds_per_tile: float | None = None) -> str:
    """One rounded duration for ``tiles`` tiles ("about 9 min"), or ``""``.

    The single-figure form of :func:`friendly_run_eta`, for the line under
    Detect: that line names the object the user typed, so it can afford one
    number where the fold header quoted a band. The typical run, never the
    slow end, and whole minutes only for the reason given there. Under a
    minute it says so instead of rounding up to one. ``seconds_per_tile``
    is the account's own measured pace when known (see :func:`run_seconds`).
    """
    typical, _slow = run_seconds(tiles, policy, seconds_per_tile)
    if typical <= 0:
        return ""
    if typical < 60.0:
        return tr("under a minute")
    return tr("about {m} min").format(m=max(1, int(round(typical / 60.0))))


def friendly_time_left(seconds: float) -> str:
    """The run's own remaining time as a short sentence, or ``""``.

    Coarse on purpose: the rate it comes from moves with every hiccup, and a
    figure that ticks by the second reads as a promise the run cannot keep.
    """
    if seconds is None or seconds < 0:
        return ""
    if seconds < 45.0:
        return tr("Less than a minute left")
    minutes = max(1, int(round(seconds / 60.0)))
    if minutes <= 1:
        return tr("About a minute left")
    return tr("About {m} min left").format(m=minutes)


# The live "About N min left" line, and what it must not quote. A run opens
# slower than it goes on: the first tiles wait on imagery and on the service
# accepting the pass, and none of that repeats. Quoting that opening rate over
# a long window put a figure on screen that halved in the first minute, which
# reads as a promise the run never meant. So the line stays quiet a little
# longer, and once it speaks it measures over a SHORT trailing window, where
# the opening has already fallen out.
LIVE_WINDOW_SECONDS = 30.0
LIVE_MIN_TILES = 12
LIVE_MIN_SECONDS = 15.0


class RunPace:
    """The pace of a run as it happens, from its own progress reports.

    Feed it ``note(done, total, now)`` as tiles land and ask
    :meth:`seconds_left`. The rate is measured over the last ``window_s``
    seconds of reports (the whole run while it is shorter than that), so a
    slow start does not drag on the estimate for the whole run and a hiccup
    fades out of it within a minute. Silent until ``min_tiles`` tiles and
    ``min_seconds`` seconds have been seen: before that the rate is noise.
    Pure Python, no Qt, so the dock can own one per pass and tests can drive
    it with a fake clock.
    """

    def __init__(self, window_s: float = LIVE_WINDOW_SECONDS,
                 min_tiles: int = LIVE_MIN_TILES,
                 min_seconds: float = LIVE_MIN_SECONDS) -> None:
        self._window_s = max(1.0, float(window_s))
        self._min_tiles = max(1, int(min_tiles))
        self._min_seconds = max(0.0, float(min_seconds))
        self._samples: list[tuple[float, int]] = []
        self._total = 0

    def note(self, done: int, total: int, now: float) -> None:
        """Record one progress report at instant ``now`` (monotonic seconds)."""
        done = max(0, int(done))
        self._total = max(0, int(total))
        if self._samples and done < self._samples[-1][1]:
            # A pass that restarted its count: begin again.
            self._samples = []
        self._samples.append((now, done))
        cutoff = now - self._window_s
        while len(self._samples) > 2 and self._samples[1][0] < cutoff:
            self._samples.pop(0)

    def rate(self) -> float:
        """Tiles per second over the window, or 0.0 while unknown."""
        if len(self._samples) < 2:
            return 0.0
        (t0, d0), (t1, d1) = self._samples[0], self._samples[-1]
        span = t1 - t0
        if span < self._min_seconds or d1 - d0 < self._min_tiles:
            return 0.0
        return (d1 - d0) / span

    def seconds_left(self) -> float | None:
        """Seconds to the end of the pass, or None while the pace is unknown."""
        rate = self.rate()
        if rate <= 0 or not self._samples:
            return None
        remaining = self._total - self._samples[-1][1]
        if remaining <= 0:
            return 0.0
        return remaining / rate


def _span(low: int, high: int) -> str:
    """``"8-18"``, or a single number when the two ends round together."""
    return str(low) if high <= low else f"{low}-{high}"


def _run_eta_dials(policy: dict | None) -> dict[str, float]:
    """The three numbers the estimate is built from, served then bounded."""
    section = _section(policy)
    return {
        "seconds_per_tile": _dial(
            section, "seconds_per_tile", SECONDS_PER_TILE_DEFAULT,
            _SECONDS_PER_TILE_BAND),
        "fixed_s": _dial(
            section, "fixed_s", FIXED_SECONDS_DEFAULT, _FIXED_SECONDS_BAND),
        "slow_factor": _dial(
            section, "slow_factor", SLOW_FACTOR_DEFAULT, _SLOW_FACTOR_BAND),
    }


def _min_show_seconds(policy: dict | None) -> float:
    """The wait under which the card stays silent."""
    return _dial(_section(policy), "min_show_s", MIN_SHOW_SECONDS_DEFAULT,
                 _MIN_SHOW_BAND)


def _section(policy: dict | None) -> dict:
    """The served ``run_eta`` object, or an empty dict.

    It sits under ``network`` because it describes how fast the service
    answers, which is the same thing that section's retry and poll budgets
    describe. Cache-only read, safe on the GUI thread, never raises.
    """
    from .detection_policy import network_policy

    try:
        raw = network_policy(policy).get(_SECTION_KEY)
    except (AttributeError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _dial(section: dict, key: str, fallback: float,
          band: tuple[float, float]) -> float:
    """``section[key]`` as a finite float inside ``band``, else ``fallback``.

    Server payloads arrive unvalidated, and ``bool`` is an ``int`` in Python,
    so ``True`` would otherwise read as the number 1.
    """
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    value = float(value)
    if not math.isfinite(value):
        return fallback
    low, high = band
    return value if low <= value <= high else fallback


def _positive_int(value: object) -> int:
    """``value`` as a positive int, else 0. Rejects bool and every non-number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if isinstance(value, float) and not math.isfinite(value):
        return 0
    return max(0, int(value))
