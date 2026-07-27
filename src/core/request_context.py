"""Short context values sent with the product configuration request.

The server answers the configuration call per product. These four values let it
answer per audience as well: the user's language, the installed plugin version,
the QGIS version and the platform. They are all optional. A value that cannot
be determined is left out rather than sent as a placeholder, so the call never
fails because one of them is missing.

Nothing here identifies a user or a machine beyond what every plugin already
sends in its user agent. No path, no name, no coordinate.

Pure Python with no Qt at import time: the two values that need QGIS resolve
through a lazy import and return None when it is not there.
"""
from __future__ import annotations

import os
import platform
import re
from urllib.parse import urlencode

# The languages the server can answer in. The QGIS locale reduces to one of
# these (see i18n.resolve_language) or the parameter is left out.
CONFIG_LANGUAGES = ("en", "fr", "es", "pt_BR")

# Context values are short tokens. Anything longer, or carrying a character
# outside this set, is cut down before it goes on the wire.
_MAX_VALUE_CHARS = 32
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._+-]")

_plugin_version_cache: str | None = None


def sanitize(value) -> str | None:
    """A short, plain token, or None when there is nothing usable to send."""
    if not isinstance(value, str):
        return None
    cleaned = _UNSAFE_CHARS.sub("", value.strip())[:_MAX_VALUE_CHARS]
    return cleaned or None


def plugin_version() -> str | None:
    """The installed plugin version from metadata.txt, or None. Memoized."""
    global _plugin_version_cache
    if _plugin_version_cache is not None:
        return _plugin_version_cache
    try:
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(plugin_dir, "metadata.txt"), encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("version="):
                    version = sanitize(line.strip().split("=", 1)[1])
                    if version:
                        _plugin_version_cache = version
                    return version
    except Exception:  # noqa: BLE001 -- context is best-effort  # nosec B110
        pass
    return None


def os_tag() -> str | None:
    """A short platform name, for example Darwin, Windows or Linux."""
    try:
        return sanitize(platform.system())
    except Exception:  # noqa: BLE001 -- context is best-effort  # nosec B110
        return None


def qgis_version() -> str | None:
    """The QGIS version number, without its release name. None off QGIS."""
    try:
        from qgis.core import Qgis

        return sanitize(str(Qgis.QGIS_VERSION).split("-")[0])
    except Exception:  # noqa: BLE001 -- context is best-effort  # nosec B110
        return None


def ui_language() -> str | None:
    """The user's language reduced to one the server answers in, or None."""
    try:
        from .i18n import resolve_language

        return sanitize(resolve_language(CONFIG_LANGUAGES))
    except Exception:  # noqa: BLE001 -- context is best-effort  # nosec B110
        return None


def config_context() -> dict:
    """The context parameters that could be determined, in a stable order."""
    values = (
        ("lang", ui_language()),
        ("v", plugin_version()),
        ("qgis", qgis_version()),
        ("os", os_tag()),
    )
    return {name: value for name, value in values if value}


def config_query(product: str) -> str:
    """The configuration route with its product and whatever context resolved.

    Every value is URL-encoded. A missing one is absent from the query rather
    than sent empty, so an older server that ignores the extras sees the same
    call it has always seen.
    """
    params = {"product": product}
    try:
        params.update(config_context())
    except Exception:  # noqa: BLE001 -- context is best-effort  # nosec B110
        pass
    return f"/api/plugin/config?{urlencode(params)}"
