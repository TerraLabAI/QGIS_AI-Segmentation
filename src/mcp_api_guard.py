"""Main-thread guard for the mcp_api_*.py calls that drive the dock.

The Processing algorithms already refuse to run off QGIS's own thread (see
``processing/algorithm_support.py:on_the_gui_thread``), because they drive Qt
widgets and blocking on the wrong thread takes QGIS down. An MCP call reaches
the same widgets the same way, from a caller that never went through
Processing's ``canExecute``/flag machinery, so it needs the same refusal at
its own front door.
"""
from __future__ import annotations

import functools
from typing import Any, Callable

_REFUSAL_MESSAGE = (
    "This call drives the AI Segmentation panel and can only run on the "
    "thread QGIS itself runs on. Call it from that thread."
)


def _on_the_gui_thread() -> bool:
    """True only when a proven check says so; an unanswerable question reads as no.

    Mirrors ``processing/algorithm_support.py:on_the_gui_thread``, duplicated
    rather than imported: that module imports ``mcp_api.py``, which assembles
    itself from the ``mcp_api_*.py`` mixins this guard decorates, so importing
    it back from here would be a cycle.
    """
    try:
        from qgis.core import QgsApplication
        from qgis.PyQt.QtCore import QThread

        app = QgsApplication.instance()
        if app is None:
            return False
        return QThread.currentThread() is app.thread()
    except Exception:  # noqa: BLE001 - an unanswered question is read as "not the GUI thread"
        return False


def gui_thread_only(func: Callable[..., dict]) -> Callable[..., dict]:
    """Refuse, with ``_error``, a call that touches the dock off the GUI thread."""
    @functools.wraps(func)
    def _wrapped(*args: Any, **kwargs: Any) -> dict:
        if not _on_the_gui_thread():
            return {"_error": _REFUSAL_MESSAGE}
        return func(*args, **kwargs)
    return _wrapped
