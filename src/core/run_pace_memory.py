















from __future__ import annotations

import json
import math

SETTINGS_KEY = "AI_Segmentation/run_pace_history"




HISTORY_LENGTH = 6




MIN_TILES_FOR_PACE = 20



_PACE_BAND = (0.02, 30.0)
_MAX_HISTORY_CHARS = 8192
_MAX_HISTORY_ENTRIES = 30


def remember_run(tiles: int, seconds: float, settings=None) -> None:

    from .server_dials import dial_in_range

    try:
        count = int(tiles)
        wall = float(seconds)
    except (TypeError, ValueError, OverflowError):
        return
    min_tiles = dial_in_range("tuning.auto.pace_min_tiles", MIN_TILES_FOR_PACE, 1, 500)
    if count < min_tiles or not math.isfinite(wall) or wall <= 0:
        return
    pace = wall / count
    if not _PACE_BAND[0] <= pace <= _PACE_BAND[1]:
        return
    history_length = dial_in_range("tuning.auto.pace_history_length", HISTORY_LENGTH, 2, 30)
    history = _read(settings)
    history.append([count, round(wall, 1)])
    _write(history[-history_length:], settings)


def own_machine_pace(settings=None) -> float | None:

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


def _settings(settings):
    if settings is not None:
        return settings
    from qgis.core import QgsSettings

    return QgsSettings()


def _read(settings) -> list[list]:
    try:
        raw = _settings(settings).value(SETTINGS_KEY, "", type=str) or ""
        if not isinstance(raw, str) or len(raw) > _MAX_HISTORY_CHARS:
            return []
        data = json.loads(raw) if raw else []
    except Exception:  # noqa: BLE001
        return []
    out: list[list] = []
    if isinstance(data, list):
        for item in data[-_MAX_HISTORY_ENTRIES:]:
            if (isinstance(item, list) and len(item) == 2
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            for v in item)):
                try:
                    if not all(math.isfinite(v) for v in item):
                        continue
                    count, wall = int(item[0]), float(item[1])
                    if count > 0 and wall > 0 and _PACE_BAND[0] <= wall / count <= _PACE_BAND[1]:
                        out.append([count, wall])
                except (ValueError, OverflowError):
                    continue
    return out


def _write(history: list[list], settings) -> None:
    try:
        _settings(settings).setValue(SETTINGS_KEY, json.dumps(history))
    except Exception:  # noqa: BLE001  # nosec B110
        pass
