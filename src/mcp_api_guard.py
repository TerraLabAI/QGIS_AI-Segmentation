








from __future__ import annotations

import functools
from typing import Any, Callable

from .core.gui_thread import on_gui_thread

_REFUSAL_MESSAGE = (
    "This call drives the AI Segmentation panel and can only run on the "
    "thread QGIS itself runs on. Call it from that thread."
)


def gui_thread_only(func: Callable[..., dict]) -> Callable[..., dict]:

    @functools.wraps(func)
    def _wrapped(*args: Any, **kwargs: Any) -> dict:
        if not on_gui_thread():
            return {"_error": _REFUSAL_MESSAGE}
        return func(*args, **kwargs)
    return _wrapped


def request_served_config(plugin: Any) -> None:






    try:
        if not on_gui_thread():
            return
        ensure = getattr(plugin, "ensure_served_config_requested", None)
        if callable(ensure):
            ensure()
    except Exception:  # noqa: BLE001  # nosec B110
        pass


SETTINGS_NOT_LOADED = "settings_not_loaded"
_SETTINGS_NOT_LOADED_MESSAGE = (
    "Connecting to load settings. Automatic needs the server's settings; "
    "they are being fetched, try again in a moment."
)


def settings_not_loaded_error() -> dict:

    return {"_error": _SETTINGS_NOT_LOADED_MESSAGE, "code": SETTINGS_NOT_LOADED}


def refuse_without_served_config(plugin: Any) -> dict | None:



    try:
        from .core.served_config import served_config_ready

        if served_config_ready():
            return None
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    request_served_config(plugin)
    try:
        if on_gui_thread():
            request = getattr(plugin, "_request_served_settings", None)
            if callable(request):
                request("")
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return settings_not_loaded_error()
