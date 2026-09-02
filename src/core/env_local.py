"""One reader for the development-only ``.env.local`` file.

The file is gitignored and absent from every released plugin, so each value
here falls back to what ships. Three modules used to open and parse it on their
own, two of them without a cache and one of them on the GUI thread every time
an engine card was drawn, so the parse happens once per process and the answers
are kept in memory.

Pure Python with no Qt at import time, so the client, the telemetry transport
and the headless path can all reach it. Nothing here raises: an unreadable file
means no override.
"""
from __future__ import annotations

import os
import threading

# Where the backend lives when nothing overrides it. Shared, so the client and
# the telemetry transport cannot drift onto two different addresses.
TERRALAB_BASE_URL_DEFAULT = "https://terra-lab.ai"

_lock = threading.Lock()
_values: dict[str, str] | None = None


def _plugin_dir() -> str:
    """The plugin root, three levels up from this file."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_file() -> dict[str, str]:
    """Every ``NAME=value`` line in the file, quotes stripped. Empty when absent."""
    out: dict[str, str] = {}
    path = os.path.join(_plugin_dir(), ".env.local")
    try:
        if not os.path.isfile(path):
            return out
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                out[name.strip()] = value.strip().strip('"').strip("'")
    except Exception:  # noqa: BLE001 -- an unreadable file is no override  # nosec B110
        return {}
    return out


def env_local_value(name: str, default: str = "") -> str:
    """The value of one key, or ``default``. Reads the file once per process."""
    global _values
    with _lock:
        if _values is None:
            _values = _read_file()
        return _values.get(name) or default


def env_local_flag(name: str, default: bool = False) -> bool:
    """One key read as a switch. Absent or empty keeps ``default``."""
    raw = env_local_value(name).strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def terralab_base_url() -> str:
    """The backend address this build talks to."""
    return env_local_value("TERRALAB_BASE_URL", TERRALAB_BASE_URL_DEFAULT)
