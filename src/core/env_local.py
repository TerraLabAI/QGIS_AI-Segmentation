











from __future__ import annotations

import os
import threading



TERRALAB_BASE_URL_DEFAULT = "https://terra-lab.ai"

_lock = threading.Lock()
_values: dict[str, str] | None = None


def _plugin_dir() -> str:

    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_file() -> dict[str, str]:

    out: dict[str, str] = {}
    path = os.path.join(_plugin_dir(), ".env.local")
    try:
        if not os.path.isfile(path):
            return out


        with open(path, encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                out[name.strip()] = value.strip().strip('"').strip("'")
    except Exception:  # noqa: BLE001  # nosec B110
        return {}
    return out


def env_local_value(name: str, default: str = "") -> str:

    global _values
    with _lock:
        if _values is None:
            _values = _read_file()
        return _values.get(name) or default


def env_local_flag(name: str, default: bool = False) -> bool:

    raw = env_local_value(name).strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def terralab_base_url() -> str:

    return env_local_value("TERRALAB_BASE_URL", TERRALAB_BASE_URL_DEFAULT)
