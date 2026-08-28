




























from __future__ import annotations

import math
from collections import deque

from .i18n import tr





SECONDS_PER_TILE_DEFAULT = 0.40
FIXED_SECONDS_DEFAULT = 1.0




SLOW_FACTOR_DEFAULT = 2.3







MIN_SHOW_SECONDS_DEFAULT = 40.0




_SECONDS_PER_TILE_BAND = (0.02, 30.0)
_FIXED_SECONDS_BAND = (0.0, 600.0)
_SLOW_FACTOR_BAND = (1.0, 10.0)
_MIN_SHOW_BAND = (0.0, 3600.0)

_SECTION_KEY = "run_eta"


def run_seconds(tiles: int, policy: dict | None = None,
                seconds_per_tile: float | None = None) -> tuple[float, float]:











    count = _positive_int(tiles)
    if count <= 0:
        return (0.0, 0.0)
    dials = _run_eta_dials(policy)
    per_tile = dials["seconds_per_tile"]
    if seconds_per_tile is not None:
        per_tile = _dial({"own": seconds_per_tile}, "own", per_tile,
                         _SECONDS_PER_TILE_BAND)
    try:
        typical = dials["fixed_s"] + per_tile * count
        slow = typical * dials["slow_factor"]
    except OverflowError:
        return (0.0, 0.0)
    return (typical, slow) if math.isfinite(slow) else (0.0, 0.0)


def own_pace_seconds_per_tile(plan: object) -> float | None:









    if not isinstance(plan, dict):
        return None
    block = plan.get("run_pace")
    if not isinstance(block, dict):
        return None
    value = _dial(block, "seconds_per_tile", 0.0, _SECONDS_PER_TILE_BAND)
    return value if value > 0 else None


def friendly_run_eta(tiles: int, policy: dict | None = None) -> str:











    typical, slow = run_seconds(tiles, policy)
    if typical <= 0 or typical < _min_show_seconds(policy):
        return ""
    low = max(1, int(round(typical / 60.0)))
    high = max(low, int(math.ceil(slow / 60.0)))
    return tr("{m} min").format(m=_span(low, high))


def friendly_run_eta_about(tiles: int, policy: dict | None = None,
                           seconds_per_tile: float | None = None) -> str:









    typical, _slow = run_seconds(tiles, policy, seconds_per_tile)
    if typical <= 0:
        return ""
    if typical < 60.0:
        return tr("under a minute")
    return tr("about {m} min").format(m=max(1, int(round(typical / 60.0))))


def friendly_time_left(seconds: float) -> str:





    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or seconds < 0):
        return ""
    try:
        if not math.isfinite(seconds):
            return ""
    except OverflowError:
        return ""
    if seconds < 45.0:
        return tr("Less than a minute left")
    minutes = max(1, int(round(seconds / 60.0)))
    if minutes <= 1:
        return tr("About a minute left")
    return tr("About {m} min left").format(m=minutes)









LIVE_WINDOW_SECONDS = 30.0
LIVE_MIN_TILES = 12
LIVE_MIN_SECONDS = 15.0



_LIVE_WINDOW_BAND = (5.0, 600.0)
_LIVE_MIN_TILES_BAND = (1, 200)
_LIVE_MIN_SECONDS_BAND = (0.0, 300.0)


class RunPace:












    def __init__(self, window_s: float | None = None,
                 min_tiles: int | None = None,
                 min_seconds: float | None = None) -> None:
        section = _section(None)
        if window_s is None:
            window_s = _dial(section, "live_window_s", LIVE_WINDOW_SECONDS,
                             _LIVE_WINDOW_BAND)
        if min_tiles is None:
            min_tiles = _dial(section, "live_min_tiles", LIVE_MIN_TILES,
                              _LIVE_MIN_TILES_BAND)
        if min_seconds is None:
            min_seconds = _dial(section, "live_min_seconds", LIVE_MIN_SECONDS,
                                _LIVE_MIN_SECONDS_BAND)
        self._window_s = max(1.0, float(window_s))
        self._min_tiles = max(1, int(min_tiles))
        self._min_seconds = max(0.0, float(min_seconds))
        self._samples: deque[tuple[float, int]] = deque()
        self._total = 0

    def note(self, done: int, total: int, now: float) -> None:

        if isinstance(now, bool) or not isinstance(now, (int, float)):
            return
        try:
            if not math.isfinite(now):
                return
        except OverflowError:
            return
        done = _positive_int(done)
        self._total = _positive_int(total)
        done = min(done, self._total)
        if self._samples and (done < self._samples[-1][1] or now < self._samples[-1][0]):

            self._samples.clear()
        if self._samples and now == self._samples[-1][0]:
            self._samples[-1] = (now, done)
        else:
            self._samples.append((now, done))
        cutoff = now - self._window_s
        while len(self._samples) > 2 and self._samples[1][0] < cutoff:
            self._samples.popleft()

    def rate(self) -> float:

        if len(self._samples) < 2:
            return 0.0
        (t0, d0), (t1, d1) = self._samples[0], self._samples[-1]
        span = t1 - t0



        if span <= 0 or span < self._min_seconds or d1 - d0 < self._min_tiles:
            return 0.0
        return (d1 - d0) / span

    def seconds_left(self) -> float | None:

        rate = self.rate()
        if rate <= 0 or not self._samples:
            return None
        remaining = self._total - self._samples[-1][1]
        if remaining <= 0:
            return 0.0
        return remaining / rate


def _span(low: int, high: int) -> str:

    return str(low) if high <= low else f"{low}-{high}"


def _run_eta_dials(policy: dict | None) -> dict[str, float]:

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

    return _dial(_section(policy), "min_show_s", MIN_SHOW_SECONDS_DEFAULT,
                 _MIN_SHOW_BAND)


def _section(policy: dict | None) -> dict:






    from .detection_policy import network_policy

    try:
        raw = network_policy(policy).get(_SECTION_KEY)
    except (AttributeError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _dial(section: dict, key: str, fallback: float,
          band: tuple[float, float]) -> float:





    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    try:
        value = float(value)
    except OverflowError:
        return fallback
    if not math.isfinite(value):
        return fallback
    low, high = band
    return value if low <= value <= high else fallback


def _positive_int(value: object) -> int:

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if isinstance(value, float) and not math.isfinite(value):
        return 0
    return max(0, int(value))
