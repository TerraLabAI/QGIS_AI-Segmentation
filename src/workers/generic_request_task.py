"""QgsTask wrapper for one-shot client requests (config, usage, account, key validation).

Mirrors AI Edit's GenericRequestTask. Using a QgsTask instead of a raw QThread
matters for crash safety: the task manager cancels cooperatively and drains
run() on shutdown, so we never call QThread.terminate() on a thread blocked
inside a network call (which can crash QGIS).
"""
from __future__ import annotations

from typing import Any, Callable

from qgis.core import QgsTask
from qgis.PyQt.QtCore import pyqtSignal


class GenericRequestTask(QgsTask):
    """Run a no-args callable off the main thread. Raises or {"error",...} -> failed.

    Set ``hidden=True`` for background/startup requests so they do not clutter
    the QGIS task-manager widget with alarming "AI Segmentation ..." rows.
    """

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str)

    def __init__(self, description: str, request_fn: Callable[[], Any], hidden: bool = False):
        flags = QgsTask.Flag.CanCancel
        if hidden:
            # Hidden / Silent only exist in QGIS >= 3.26; the plugin floor is
            # 3.22. Feature-detect each flag and OR it in only when present.
            # Naming QgsTask.Flag.Hidden directly would AttributeError at import
            # on 3.22-3.24; on those builds the task degrades to a plain
            # cancellable (visible) task, which is harmless for a quick request.
            for name in ("Hidden", "Silent"):
                extra = getattr(QgsTask.Flag, name, None)
                if extra is not None:
                    flags = flags | extra
        super().__init__(description, flags)
        self._request_fn = request_fn
        self._result: Any = None
        self._failure: tuple[str, str] | None = None

    def is_active(self) -> bool:
        try:
            return self.status() in (
                QgsTask.TaskStatus.Running,
                QgsTask.TaskStatus.Queued,
                QgsTask.TaskStatus.OnHold,
            )
        except Exception:
            return False

    def run(self) -> bool:
        if self.isCanceled():
            return False
        try:
            result = self._request_fn()
        except Exception as e:
            # Preserve a usable code so consumers that branch on it (network vs
            # app error) don't misread a raised exception as a blank-code error.
            raw_code = getattr(e, "code", "")
            code = getattr(raw_code, "value", raw_code) or "UNKNOWN"
            self._failure = (str(e)[:200], str(code))
            return False

        if self.isCanceled():
            return False

        if isinstance(result, dict) and "error" in result:
            self._failure = (
                str(result.get("error", "Unknown error")),
                str(result.get("code", "")),
            )
            return False

        self._result = result
        return True

    def finished(self, result: bool) -> None:
        if self.isCanceled():
            return
        if result:
            self.succeeded.emit(self._result)
        elif self._failure is not None:
            self.failed.emit(*self._failure)
