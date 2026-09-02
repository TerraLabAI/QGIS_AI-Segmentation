



















from __future__ import annotations

import threading

from qgis.PyQt.QtCore import QThread




RUN_AUTOSAVE_JOIN_TIMEOUT_MS = 10000


class RunAutosaveThread(QThread):







    def __init__(self, job: dict, parent=None) -> None:
        super().__init__(parent)
        self._job = job
        self._lock = threading.Lock()
        self._done = False
        self._info: dict | None = None



    def is_done(self) -> bool:

        with self._lock:
            return self._done

    def take_result(self) -> dict | None:





        with self._lock:
            info = self._info
            self._info = None
        return info

    def join_run(self, timeout_ms: int = RUN_AUTOSAVE_JOIN_TIMEOUT_MS) -> bool:






        if not self.isRunning():
            return True
        return bool(self.wait(timeout_ms))



    def run(self) -> None:  # noqa: D102
        info = None
        try:
            from ..core.run_autosave import write_prepared_autosave

            info = write_prepared_autosave(self._job)
        except Exception:  # noqa: BLE001
            info = None
        finally:
            with self._lock:
                self._info = info
                self._done = True


            self._job = {}
