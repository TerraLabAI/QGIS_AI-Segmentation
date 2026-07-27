
























from __future__ import annotations

import html
import math
import re
from itertools import islice
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit




_NO_CONFIG: dict = {}




_MAX_LIST_ENTRIES = 64
_MAX_ENTRY_CHARS = 128


_MAX_URL_CHARS = 2048


_URL_FORBIDDEN = set('"<>\\^`{|}')


def _server_config() -> dict:






    try:
        from .config_cache import get_config

        config = get_config()
    except Exception:  # noqa: BLE001  # nosec B110
        return _NO_CONFIG
    return config if isinstance(config, dict) else _NO_CONFIG


def read_value(path: str) -> Any:

    try:
        value: Any = _server_config()
        for part in path.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
    except Exception:  # noqa: BLE001  # nosec B110
        return None
    return value


def _is_finite_dial_value(value: Any) -> bool:

    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def dial(path: str, fallback):





    try:
        value = read_value(path)
        if not _is_finite_dial_value(value) or value <= 0:
            return fallback
        resolved = type(fallback)(value)
        return resolved if resolved > 0 else fallback
    except Exception:  # noqa: BLE001  # nosec B110
        return fallback


def dial_in_range(path: str, fallback, low: float, high: float):





    try:
        value = read_value(path)
        if not _is_finite_dial_value(value) or not low <= value <= high:
            return fallback
        resolved = type(fallback)(value)
        return resolved if low <= resolved <= high else fallback
    except Exception:  # noqa: BLE001  # nosec B110
        return fallback


def dial_pair(path: str, fallback: tuple[float, float]) -> tuple[float, float]:





    try:
        value = read_value(path)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            low, high = value
            if _is_finite_dial_value(low) and _is_finite_dial_value(high) and 0 < low <= high:
                return (float(low), float(high))
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return fallback


def dial_list(
    path: str,
    base: Iterable[str] = (),
    normalize: Callable[[str], str] | None = None,
) -> frozenset:










    merged = set(base)
    try:
        value = read_value(path)
        if isinstance(value, (list, tuple)):
            for item in islice(value, _MAX_LIST_ENTRIES):
                if not isinstance(item, str) or len(item) > _MAX_ENTRY_CHARS:
                    continue
                entry = item.strip()
                if entry:
                    merged.add(normalize(entry) if normalize else entry)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return frozenset(merged)


def dial_str(path: str, fallback: str, allowed: Iterable[str] | None = None) -> str:


    try:
        value = read_value(path)
        if isinstance(value, str):
            value = value.strip()
            if value and (allowed is None or value in allowed):
                return value
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return fallback




_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "")


def cleartext_remote_url(url: str) -> bool:






    if not isinstance(url, str):
        return False
    try:
        parts = urlsplit(url)
        if parts.scheme.lower() != "http":
            return False
        host = (parts.hostname or "").lower()
    except Exception:  # noqa: BLE001
        return True
    return host not in _LOCAL_HOSTS


def safe_web_url(candidate: Any, fallback: str) -> str:













    try:
        if not isinstance(candidate, str):
            return fallback
        text = candidate.strip()
        if not text or len(text) > _MAX_URL_CHARS:
            return fallback


        for ch in text:
            if ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F or ch in _URL_FORBIDDEN:
                return fallback
        parts = urlsplit(text)
        if (parts.scheme.lower() != "https" or not parts.hostname
                or parts.username is not None or parts.password is not None):
            return fallback

        if parts.port is not None and parts.port <= 0:
            return fallback
        return text
    except Exception:  # noqa: BLE001  # nosec B110
        return fallback


def dial_url(path: str, fallback: str) -> str:

    return safe_web_url(read_value(path), fallback)


def dial_bool(path: str, fallback: bool) -> bool:


    try:
        value = read_value(path)
        if isinstance(value, bool):
            return value
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return fallback























_MAX_COPY_CHARS = 400


_COPY_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f<>]")


def clean_served_text(value: Any, max_chars: int = _MAX_COPY_CHARS) -> str | None:





    if not isinstance(value, str):
        return None
    text = _COPY_CTRL_RE.sub("", value[:max_chars]).strip()
    return text or None


def dial_text(container_path: str, key: str, max_chars: int = _MAX_COPY_CHARS) -> str | None:




    try:
        container = read_value(container_path)
        if isinstance(container, dict):
            return clean_served_text(container.get(key), max_chars)
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return None







_served_copy_allowed: bool | None = None


def served_copy_allowed() -> bool:




    global _served_copy_allowed
    if _served_copy_allowed is None:
        try:
            from .request_context import ui_language

            _served_copy_allowed = ui_language() is not None
        except Exception:  # noqa: BLE001
            _served_copy_allowed = False
    return _served_copy_allowed


def dial_copy(
    string_id: str,
    fallback: str,
    max_chars: int = _MAX_COPY_CHARS,
    escape: bool = False,
) -> str:














    if not served_copy_allowed():
        return fallback
    served = dial_text("copy", string_id, max_chars)
    if served is None:
        return fallback
    return html.escape(served, quote=False) if escape else served


class ServerDialSet(frozenset):







    def __new__(cls, path: str, base: Iterable[str], normalize=None):
        obj = super().__new__(cls, base)
        obj._path = path
        obj._normalize = normalize
        obj._memo = None
        return obj

    def __eq__(self, other):






        if isinstance(other, ServerDialSet):
            return (
                self._path == other._path
                and self._normalize == other._normalize
                and frozenset.__eq__(self, other)
            )
        return NotImplemented

    def __ne__(self, other):

        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    __hash__ = frozenset.__hash__

    def _served(self) -> frozenset:








        config = _server_config()
        memo = self._memo
        if memo is not None and memo[0] is config:
            return memo[1]
        served = dial_list(self._path, (), normalize=self._normalize)
        self._memo = (config, served)
        return served

    def __contains__(self, item) -> bool:
        if frozenset.__contains__(self, item):
            return True
        return item in self._served()


class ServerDialMap(dict):








    def __init__(self, path: str, defaults: dict):
        super().__init__(defaults)
        self._path = path

    def __eq__(self, other):






        if isinstance(other, ServerDialMap):
            return self._path == other._path and dict.__eq__(self, other)
        return NotImplemented

    def __ne__(self, other):

        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __getitem__(self, key):
        return dial(f"{self._path}.{key}", dict.__getitem__(self, key))

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default





def feature_enabled(name: str) -> bool:












    return feature_switch("features." + name, True)


def feature_switch(path: str, shipped: bool) -> bool:











    served = read_value(path)
    if served is False:
        return False
    if served is True:
        return True
    if not shipped:
        return False
    name = path.rsplit(".", 1)[-1]
    try:
        from .kill_switch_memory import is_remembered_off

        return not is_remembered_off(name)
    except Exception:  # noqa: BLE001  # nosec B110
        return True


def correct_ai_cloud_enabled() -> bool:














    return feature_switch("features.correct_ai_cloud", True)


def crop_webp_enabled() -> bool:








    return dial_bool("features.crop_webp", False)


def gzip_request_bodies_enabled() -> bool:









    return dial_bool("features.gzip_request_bodies", False)


def automatic_mode_enabled() -> bool:

















    if read_value("automatic_mode_enabled") is False:
        return False
    if read_value("features.automatic_mode") is False:
        return False
    try:
        from .kill_switch_memory import AUTOMATIC_MODE_NAME, is_remembered_off

        return not is_remembered_off(AUTOMATIC_MODE_NAME)
    except Exception:  # noqa: BLE001  # nosec B110
        return True





def parse_version(text) -> tuple[int, ...] | None:

    if not isinstance(text, str) or len(text) > 128:
        return None
    text = text.strip()
    if text[:1] in ("v", "V"):
        text = text[1:]
    parts = text.split(".")
    if len(parts) > 16:
        return None
    out = []
    for part in parts:
        part = part.strip()
        if not part.isascii() or not part.isdigit() or len(part) > 10:
            return None
        out.append(int(part))
    return tuple(out) if out else None


def is_served_update_recommended(installed_version: str) -> bool:


    return _served_version_above(installed_version, "min_recommended_version")


def _served_version_above(installed_version: str, key: str) -> bool:


    installed = parse_version(installed_version)
    served = parse_version(read_value(key))
    if installed is None or served is None:
        return False
    width = max(len(installed), len(served))
    installed += (0,) * (width - len(installed))
    served += (0,) * (width - len(served))
    return served > installed


def served_latest_version() -> str | None:

    value = read_value("latest_version")
    return value.strip() if parse_version(value) is not None else None


def is_served_update_available(installed_version: str) -> bool:

    return _served_version_above(installed_version, "latest_version")



_MAX_RELEASE_NOTES_CHARS = 160


def served_release_notes_line() -> str | None:

    return clean_served_text(read_value("release_notes_line"), _MAX_RELEASE_NOTES_CHARS)


MARKETPLACE_URL = "https://plugins.qgis.org/plugins/AI_Segmentation/"


def served_marketplace_url() -> str:

    return dial_url("marketplace_url", MARKETPLACE_URL)
