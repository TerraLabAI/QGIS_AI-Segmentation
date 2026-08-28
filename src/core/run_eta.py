"""How long an Automatic run takes, in the words a user can act on.

A run reads its tiles a few at a time, so the wall clock a user waits grows
with the tile count and not with the surface drawn. The card that shows the
zone is where the run is accepted or abandoned, and until it carries a
duration the only honest thing on screen is a bar that starts after the
decision is already made.

Two properties decide the shape of what is shown:

- **The relationship is a line.** A fixed opening cost, then a steady cost per
  tile. Concurrency is fixed, so throughput does not improve with size and the
  per-tile cost stays flat across the whole range.
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
SECONDS_PER_TILE_DEFAULT = 0.58
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


def run_seconds(tiles: int, policy: dict | None = None) -> tuple[float, float]:
    """Typical and slow wall clock for a run of ``tiles`` tiles, in seconds.

    ``(0.0, 0.0)`` when the tile count is not a usable positive number, which
    is every caller's signal to show nothing.
    """
    count = _positive_int(tiles)
    if count <= 0:
        return (0.0, 0.0)
    dials = _run_eta_dials(policy)
    typical = dials["fixed_s"] + dials["seconds_per_tile"] * count
    return (typical, typical * dials["slow_factor"])


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
