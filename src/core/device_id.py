"""Stable, pseudonymous per-machine identifier.

Produces an irreversible device hash so the backend can count how many distinct
machines use a single activation key, without ever learning the machine's real
identity. The hash is a one-way SHA256 digest, so the raw machine id never
leaves the user's computer. Mirrors AI Edit so both plugins report the same
device to the shared TerraLab account page.
"""
from __future__ import annotations

import hashlib
import uuid

from qgis.core import QgsSettings
from qgis.PyQt.QtCore import QSysInfo

# Random per-machine fallback seed, persisted when the OS machine id is unavailable.
_SETTINGS_KEY = "TerraLab/device_seed"
# 16 hex chars (64 bits): collision-safe across our user base, well under the
# server's 32-char cap, and stays lowercase hex as the route handler expects.
_HASH_LEN = 16

_cached: str | None = None


def _machine_seed(settings) -> bytes:
    """Return a stable per-machine seed.

    Prefers the OS machine id (QSysInfo). Falls back to a random UUID persisted
    in QSettings when the OS id is unavailable (some containers, locked-down hosts).
    """
    try:
        raw = bytes(QSysInfo.machineUniqueId())
        if raw:
            return raw
    except Exception:  # nosec B110
        pass

    seed = settings.value(_SETTINGS_KEY, "", type=str)
    if not seed:
        seed = uuid.uuid4().hex
        settings.setValue(_SETTINGS_KEY, seed)
    return seed.encode("utf-8")


def get_device_hash(settings=None) -> str:
    """Return a 16-char lowercase-hex hash identifying this machine.

    Irreversible (SHA256). Cached for the process lifetime.
    """
    global _cached
    if _cached is not None:
        return _cached

    s = settings or QgsSettings()
    digest = hashlib.sha256(_machine_seed(s)).hexdigest()
    _cached = digest[:_HASH_LEN]
    return _cached


# Max length the server stores for the platform label; keep payload tiny.
_PLATFORM_MAX_LEN = 48


def get_device_platform() -> str:
    """Return a human-readable OS label for this machine (e.g. "macOS 15.0").

    Lets the account page show which computer a license is active on. No PII:
    just the OS name + version. Empty string if unavailable.
    """
    try:
        name = QSysInfo.prettyProductName() or ""
    except Exception:  # nosec B110
        name = ""
    return name.strip()[:_PLATFORM_MAX_LEN]
