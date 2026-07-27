"""Stable, pseudonymous per-machine identifier.

Produces an irreversible device hash so the backend can count how many distinct
machines use a single activation key, without ever learning the machine's real
identity. The hash is a one-way SHA256 digest, so the raw machine id never
leaves the user's computer. Same shape as AI Edit's, under this plugin's own
settings key.
"""
from __future__ import annotations

import hashlib
import uuid

from qgis.core import QgsSettings
from qgis.PyQt.QtCore import QSysInfo

# Per-machine secret, persisted on first use. This IS the identity.
_SETTINGS_KEY = "TerraLab/device_seed"
# 16 hex chars (64 bits): collision-safe across our user base, well under the
# server's 32-char cap, and stays lowercase hex as the route handler expects.
_HASH_LEN = 16

_cached: str | None = None


def _inherited_seed() -> str | None:
    """Return the OS machine id as a seed string, or None when there is none.

    Installs made before the random secret existed hashed the OS machine id
    straight from ``QSysInfo.machineUniqueId()``. Reusing that id as the stored
    seed makes those machines keep the device hash they already report, so an
    update does not register every one of them a second time.

    ``machineUniqueId`` is missing on the Qt5 binding, where the call raises and
    the caller mints a random secret instead. That is the right answer there:
    that binding never had a stable OS id to inherit.
    """
    try:
        raw = bytes(QSysInfo.machineUniqueId())
    except Exception:
        return None
    if not raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    # Reuse it only when the round trip is byte-exact, since the seed is stored
    # as text and re-encoded on every read. Anything else would move the hash.
    return text if text.encode("utf-8") == raw else None


def _machine_seed(settings) -> bytes:
    """Return a stable per-machine seed, persisted in QSettings on first use.

    A machine that already has an OS machine id inherits it once, so its hash
    does not change. Everything else gets a random UUID, which is preferable:

    - ``QSysInfo.machineUniqueId()`` exists only on the Qt6 binding, so a fresh
      install that relied on it would report a different device depending on
      which QGIS major version it is running.
    - A hashed hardware address (MAC, host name) is short and structured enough
      to be brute-forced back out of the digest, which would break the promise
      that the raw machine identity never leaves the user's computer.

    A random secret has neither problem: it is unguessable, and it is read the
    same way on both bindings.
    """
    seed = settings.value(_SETTINGS_KEY, "", type=str)
    if not seed:
        seed = _inherited_seed() or uuid.uuid4().hex
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
    # This value travels as a raw HTTP header, which must stay ASCII: a
    # localized OS product name with accents, UTF-8-encoded into the header,
    # gets rejected by strict proxies/gateways. Fold instead of failing.
    name = " ".join(name.split())
    name = name.encode("ascii", "ignore").decode("ascii")
    return name.strip()[:_PLATFORM_MAX_LEN]
