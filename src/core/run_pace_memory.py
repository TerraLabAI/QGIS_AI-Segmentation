"""What this machine's own Automatic runs took, end to end, for the quote.

The duration quoted under Detect starts from a fleet-wide seconds-per-tile
dial. Two things that dial cannot know decide most of the wait on a given
machine: the link the tiles leave on, and the time this computer spends
turning answers into shapes after the last tile is in. Both are stable from
one run to the next on the same machine and differ several times over
between machines, so the best predictor of the next run here is the last
few runs here.

This module keeps those few runs in QSettings: tile count and wall clock
from the Detect click to the review opening. It is read by the quote and
written once per finished run. Nothing here touches the network, and a
machine with no history reads as None so the caller keeps its other
sources (the account's served pace, then the fleet dial).
"""
from __future__ import annotations

import json
import math

SETTINGS_KEY = "AI_Segmentation/run_pace_history"

# How many finished runs to keep. Enough to ride out one odd run (a cold
# service, a busy machine), few enough that a changed link shows within a
# session or two.
HISTORY_LENGTH = 6

# Runs below this many tiles are mostly fixed cost and say little about the
# per-tile pace; they are kept out of the memory rather than out of the
# median, so a user who only ever runs small zones still gets a figure.
MIN_TILES_FOR_PACE = 20

# Same band the served dial is held inside (run_eta): a wall clock that lands
# outside it came from a clock jump or a run that sat in a paused laptop.
_PACE_BAND = (0.02, 30.0)


def remember_run(tiles: int, seconds: float, settings=None) -> None:
    """Append one finished run. Silently ignores unusable numbers."""
    try:
        count = int(tiles)
        wall = float(seconds)
    except (TypeError, ValueError):
        return
    if count < MIN_TILES_FOR_PACE or not math.isfinite(wall) or wall <= 0:
        return
    pace = wall / count
    if not _PACE_BAND[0] <= pace <= _PACE_BAND[1]:
        return
    history = _read(settings)
    history.append([count, round(wall, 1)])
    _write(history[-HISTORY_LENGTH:], settings)


def own_machine_pace(settings=None) -> float | None:
    """Median seconds per tile over the remembered runs, or None with none."""
    paces = sorted(
        wall / count for count, wall in _read(settings)
        if count > 0 and wall > 0 and _PACE_BAND[0] <= wall / count <= _PACE_BAND[1]
    )
    if not paces:
        return None
    mid = len(paces) // 2
    if len(paces) % 2:
        return paces[mid]
    return (paces[mid - 1] + paces[mid]) / 2.0


def forget_runs(settings=None) -> None:
    """Drop the memory (a support step, or a test that wants a clean slate)."""
    _write([], settings)


def _settings(settings):
    if settings is not None:
        return settings
    from qgis.core import QgsSettings

    return QgsSettings()


def _read(settings) -> list[list]:
    try:
        raw = _settings(settings).value(SETTINGS_KEY, "", type=str) or ""
        data = json.loads(raw) if raw else []
    except Exception:  # noqa: BLE001 -- a corrupt setting reads as no history
        return []
    out: list[list] = []
    if isinstance(data, list):
        for item in data:
            if (isinstance(item, list) and len(item) == 2
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            for v in item)):
                out.append([int(item[0]), float(item[1])])
    return out


def _write(history: list[list], settings) -> None:
    try:
        _settings(settings).setValue(SETTINGS_KEY, json.dumps(history))
    except Exception:  # noqa: BLE001 -- a settings write must never break a run  # nosec B110
        pass
