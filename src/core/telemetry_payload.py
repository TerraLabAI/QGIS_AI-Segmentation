
from __future__ import annotations

import math
from functools import lru_cache
from itertools import islice


_PROPERTY_MAX_CHARS = 4096
_PROPERTY_MAX_ITEMS = 200
_PROPERTY_MAX_DEPTH = 8
_PROPERTY_NODE_BUDGET = 2000

_PRIVATE_PROPERTY_KEYS = frozenset({
    "password", "secret", "apikey", "accesstoken", "refreshtoken",
    "authorization", "activationkey", "licensekey", "cookie", "setcookie",
    "clientsecret", "token", "credentials", "bbox", "extent", "coordinates",
    "pointcoords", "latitude", "longitude", "lat", "lon",
})




_PATH_SEGMENT = r"[^\s\\/'\"<>,;()=:]"


def _scrub_telemetry_properties(value, depth: int = 0, budget=None):

    if budget is None:
        budget = [_PROPERTY_NODE_BUDGET]
    if budget[0] <= 0 or depth > _PROPERTY_MAX_DEPTH:
        return None
    budget[0] -= 1
    if isinstance(value, str):

        return scrub_payload_value(value[:_PROPERTY_MAX_CHARS * 2])[:_PROPERTY_MAX_CHARS]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if value.bit_length() <= 64 else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        result = {}
        for key, child in islice(value.items(), _PROPERTY_MAX_ITEMS):
            if len(result) >= _PROPERTY_MAX_ITEMS or budget[0] <= 0:
                break
            if isinstance(key, str):
                normalized = "".join(c for c in key[:256].lower() if c.isalnum())
                if (normalized in _PRIVATE_PROPERTY_KEYS
                        or (normalized.startswith("x") and "auth" in normalized)):
                    continue
                safe_key = scrub_payload_value(key[:256])
                result[safe_key] = _scrub_telemetry_properties(child, depth + 1, budget)
        return result
    if isinstance(value, (list, tuple)):
        return [_scrub_telemetry_properties(child, depth + 1, budget)
                for child in value[:_PROPERTY_MAX_ITEMS] if budget[0] > 0]
    return None


def scrub_payload_value(value: str) -> str:



















    coord_pattern, url_pattern, email_pattern, path_pattern = _scrub_patterns()


    from .log_scrub import anonymize_paths, scrub_sensitive
    value = scrub_sensitive(anonymize_paths(value))
    value = url_pattern.sub("<URL>", value or "")
    value = email_pattern.sub("<EMAIL>", value)


    value = path_pattern.sub("<path>", value)
    return coord_pattern.sub("<COORDS>", value)


@lru_cache(maxsize=1)
def _scrub_patterns():

    import re as _re
    _COORD_PATTERN = _re.compile(
        r"(?:[-+]?\d+(?:\.\d+)?)(?:\s*,\s*[-+]?\d+(?:\.\d+)?){1,}"
    )
    _URL_PATTERN = _re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"]+")
    _EMAIL_PATTERN = _re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
    _PATH_PATTERN = _re.compile(




        r"(?:<USER>"
        r"|(?<![\w.])[A-Za-z]:(?=[\\/])"
        r"|~(?=[\\/])"
        r"|\\\\(?=" + _PATH_SEGMENT + r")"
        r"|(?<![\w.:)\]])/(?=" + _PATH_SEGMENT + r"))"


        r"(?:[^\r\n'\"<>,;()=]*[\\/])?" + _PATH_SEGMENT + r"*"
    )
    return _COORD_PATTERN, _URL_PATTERN, _EMAIL_PATTERN, _PATH_PATTERN
