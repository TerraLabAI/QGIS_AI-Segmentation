"""What the server last turned OFF, remembered across restarts.

The disk copy of the product configuration deliberately carries no kill
switch (see ``config_cache._without_kill_switches``): that file is not
authenticated, so it may retune the plugin but must never be able to disable
it. The cost is that every cold start begins with every switch back ON, and
stays that way for the seconds it takes the fetch to land. A feature withdrawn
because it is broken is live again on every restart, on exactly the machines
the withdrawal was written for.

This module closes that window without giving the disk copy the power it was
denied. It keeps ONE thing: the set of feature names the last LIVE
configuration said false to, in the user's own QGIS settings. It is consulted
only when the configuration in force has no opinion, so a served value always
wins the moment it arrives, and the set is rewritten whole on every live
fetch, so a switch the server stops sending is on again on the next start.

The asymmetry is on purpose and is the whole safety argument: this memory can
only turn a feature off, never on. The worst a damaged or edited settings file
can do is withdraw a feature from the user who owns the file, which is the
same thing the server can already do and no more.

Pure Python plus QSettings, no network, no work on the hot path: the set is
read once per session and kept in memory.
"""
from __future__ import annotations

from contextlib import suppress

#: QSettings key holding the remembered names, one string list. Same
#: "AISegmentation/" group as the rest of the plugin's settings. Never rename
#: the literal: it sits in the user's QGIS profile.
_OFF_KEY = "AISegmentation/kill_switches/off"

#: The feature name Automatic mode is remembered under, so the historical
#: top-level ``automatic_mode_enabled`` flag and the generic
#: ``features.automatic_mode`` switch share one entry.
AUTOMATIC_MODE_NAME = "automatic_mode"

#: A served configuration names a handful of switches. The cap keeps a damaged
#: settings file from turning a membership test into a lot of work.
_MAX_REMEMBERED = 64
_MAX_NAME_CHARS = 64

#: Session cache. None until the first read. Never mutated in place, so a
#: reader always sees a complete set.
_cached: frozenset[str] | None = None


def _settings():
    from qgis.core import QgsSettings

    return QgsSettings()


def _clean(names) -> frozenset[str]:
    """The subset of ``names`` that is usable as a feature name."""
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
    """The feature names the last live configuration said false to.

    Empty on anything unreadable. Read from disk once per session.
    """
    global _cached
    if _cached is not None:
        return _cached
    names: frozenset[str] = frozenset()
    with suppress(Exception):
        names = _clean(_settings().value(_OFF_KEY, [], type=list))
    _cached = names
    return names


def is_remembered_off(name: str) -> bool:
    """Whether ``name`` was off in the last live configuration. Never raises."""
    try:
        return name in remembered_off()
    except Exception:  # noqa: BLE001 -- a memory is best-effort
        return False


def _switches_off_in(config: dict) -> frozenset[str]:
    """Every feature the given LIVE configuration explicitly turns off.

    Only an explicit ``false`` counts, exactly like the readers in
    ``server_dials``: garbage means "use what shipped" there, and must mean
    "remember nothing" here.
    """
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
    """Rewrite the memory from a configuration that came off the network.

    Whole-set replacement, never a merge: a switch the server stops sending is
    a switch the server took back, and the feature has to come back with it.
    Best-effort and silent. Losing the memory costs the seconds until the next
    fetch, which is the behaviour this module improves on, so a failure here
    can never be worse than not having it.
    """
    global _cached
    if not isinstance(config, dict):
        return
    names = _switches_off_in(config)
    _cached = names
    with suppress(Exception):
        settings = _settings()
        if names:
            settings.setValue(_OFF_KEY, sorted(names))
        else:
            settings.remove(_OFF_KEY)
        settings.sync()


def forget_all() -> None:
    """Drop the memory, on disk and in this session."""
    global _cached
    _cached = frozenset()
    with suppress(Exception):
        settings = _settings()
        settings.remove(_OFF_KEY)
        settings.sync()
