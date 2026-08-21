
























from __future__ import annotations

from contextlib import suppress




_OFF_KEY = "AISegmentation/kill_switches/off"




AUTOMATIC_MODE_NAME = "automatic_mode"



_MAX_REMEMBERED = 64
_MAX_NAME_CHARS = 64



_memory: dict[str, frozenset[str]] = {}


def _settings():
    from qgis.core import QgsSettings

    return QgsSettings()


def _clean(names) -> frozenset[str]:

    out = []
    if isinstance(names, str):
        names = [names]
    try:
        for name in names or ():
            if not isinstance(name, str):
                continue
            name = name.strip()
            if name and len(name) <= _MAX_NAME_CHARS:
                out.append(name)
            if len(out) >= _MAX_REMEMBERED:
                break
    except TypeError:
        return frozenset()
    return frozenset(out)


def remembered_off() -> frozenset[str]:




    cached = _memory.get("off")
    if cached is not None:
        return cached
    names: frozenset[str] = frozenset()
    with suppress(Exception):
        names = _clean(_settings().value(_OFF_KEY, [], type=list))
    _memory["off"] = names
    return names


def is_remembered_off(name: str) -> bool:

    try:
        return name in remembered_off()
    except Exception:  # noqa: BLE001
        return False


def _switches_off_in(config: dict) -> frozenset[str]:






    off = []
    if config.get("automatic_mode_enabled") is False:
        off.append(AUTOMATIC_MODE_NAME)
    features = config.get("features")
    if isinstance(features, dict):
        for name, on in features.items():
            if on is False and isinstance(name, str):
                off.append(name)
    return _clean(off)


def remember_from_live_config(config: dict) -> None:








    if not isinstance(config, dict):
        return
    names = _switches_off_in(config)
    _memory["off"] = names
    with suppress(Exception):
        settings = _settings()
        if names:
            settings.setValue(_OFF_KEY, sorted(names))
        else:
            settings.remove(_OFF_KEY)
        settings.sync()
