








from __future__ import annotations

import functools
from typing import Any, Callable

_REFUSAL_MESSAGE = (
    "This call drives the AI Segmentation panel and can only run on the "
    "thread QGIS itself runs on. Call it from that thread."
)


def _on_the_gui_thread() -> bool:







    try:
        from qgis.core import QgsApplication
        from qgis.PyQt.QtCore import QThread

        app = QgsApplication.instance()
        if app is None:
            return False
        return QThread.currentThread() is app.thread()
    except Exception:  # noqa: BLE001
        return False


def gui_thread_only(func: Callable[..., dict]) -> Callable[..., dict]:

    @functools.wraps(func)
    def _wrapped(*args: Any, **kwargs: Any) -> dict:
        if not _on_the_gui_thread():
            return {"_error": _REFUSAL_MESSAGE}
        return func(*args, **kwargs)
    return _wrapped
