






from __future__ import annotations

from typing import Any, Callable

from qgis.core import QgsFeedback, QgsTask
from qgis.PyQt.QtCore import pyqtSignal


class GenericRequestTask(QgsTask):






    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str)

    def __init__(self, description: str, request_fn: Callable[[], Any], hidden: bool = False):
        flags = QgsTask.Flag.CanCancel
        if hidden:





            for name in ("Hidden", "Silent"):
                extra = getattr(QgsTask.Flag, name, None)
                if extra is not None:
                    flags = flags | extra
        super().__init__(description, flags)
        self._request_fn = request_fn
        self._result: Any = None
        self._failure: tuple[str, str] | None = None



        self._feedback = QgsFeedback()

    def cancel(self) -> None:
        try:
            self._feedback.cancel()
        except Exception:  # nosec B110
            pass
        super().cancel()

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
            from ..api.request_feedback import request_feedback

            with request_feedback(self._feedback):
                result = self._request_fn()
        except Exception as e:


            raw_code = getattr(e, "code", "")
            code = getattr(raw_code, "value", raw_code) or "UNKNOWN"




            from ..core.log_scrub import scrub_sensitive

            self._failure = (scrub_sensitive(str(e))[:200], str(code))
            return False

        if self.isCanceled():
            return False

        if not isinstance(result, dict):






            self._failure = ("Invalid server response", "SERVER_ERROR")
            return False

        if "error" in result:
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
