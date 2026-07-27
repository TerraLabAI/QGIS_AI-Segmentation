













from __future__ import annotations

import os
import platform
import re
from urllib.parse import urlencode



CONFIG_LANGUAGES = ("en", "fr", "es", "pt_BR")



_MAX_VALUE_CHARS = 32
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._+-]")



_VERSION_UNKNOWN = "\x00"
_plugin_version_memo: dict[str, str] = {}


def sanitize(value) -> str | None:

    if not isinstance(value, str):
        return None
    cleaned = _UNSAFE_CHARS.sub("", value.strip())[:_MAX_VALUE_CHARS]
    return cleaned or None


def plugin_version() -> str | None:





    cached = _plugin_version_memo.get("version")
    if cached is not None:
        return None if cached == _VERSION_UNKNOWN else cached
    version = None
    try:
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(plugin_dir, "metadata.txt"), encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("version="):
                    version = sanitize(line.strip().split("=", 1)[1])
                    break
    except Exception:  # noqa: BLE001  # nosec B110
        version = None
    _plugin_version_memo["version"] = version or _VERSION_UNKNOWN
    return version


def os_tag() -> str | None:

    try:
        return sanitize(platform.system())
    except Exception:  # noqa: BLE001  # nosec B110
        return None


def qgis_version() -> str | None:

    try:
        from qgis.core import Qgis

        return sanitize(str(Qgis.QGIS_VERSION).split("-")[0])
    except Exception:  # noqa: BLE001  # nosec B110
        return None


def ui_language() -> str | None:

    try:
        from .i18n import resolve_language

        return sanitize(resolve_language(CONFIG_LANGUAGES))
    except Exception:  # noqa: BLE001  # nosec B110
        return None


def config_context() -> dict:

    values = (
        ("lang", ui_language()),
        ("v", plugin_version()),
        ("qgis", qgis_version()),
        ("os", os_tag()),
    )
    return {name: value for name, value in values if value}


def config_query(product: str) -> str:






    params = {"product": product}
    try:
        params.update(config_context())
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return f"/api/plugin/config?{urlencode(params)}"
