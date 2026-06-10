







from __future__ import annotations

import hashlib
import uuid

from qgis.core import QgsSettings
from qgis.PyQt.QtCore import QSysInfo


_SETTINGS_KEY = "TerraLab/device_seed"


_HASH_LEN = 16

_cached: str | None = None


def _inherited_seed() -> str | None:











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


    return text if text.encode("utf-8") == raw else None


def _machine_seed(settings) -> bytes:















    seed = settings.value(_SETTINGS_KEY, "", type=str)
    if not seed:
        seed = _inherited_seed() or uuid.uuid4().hex
        settings.setValue(_SETTINGS_KEY, seed)
    return seed.encode("utf-8")


def get_device_hash(settings=None) -> str:




    global _cached
    if _cached is not None:
        return _cached

    s = settings or QgsSettings()
    digest = hashlib.sha256(_machine_seed(s)).hexdigest()
    _cached = digest[:_HASH_LEN]
    return _cached



_PLATFORM_MAX_LEN = 48

_cached_platform: str | None = None


def get_device_platform() -> str:









    global _cached_platform
    if _cached_platform is not None:
        return _cached_platform

    try:
        name = QSysInfo.prettyProductName() or ""
    except Exception:  # nosec B110
        name = ""



    name = " ".join(name.split())
    name = name.encode("ascii", "ignore").decode("ascii")
    _cached_platform = name.strip()[:_PLATFORM_MAX_LEN]
    return _cached_platform
