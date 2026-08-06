












from __future__ import annotations

import threading

from qgis.PyQt.QtCore import QThread



RUN_EXPORT_JOIN_TIMEOUT_MS = 30000


class RunExportThread(QThread):


    def __init__(self, job: dict, parent=None) -> None:
        super().__init__(parent)
        from ..core.run_export_job import RunExportProgress

        self._job = job
        self._lock = threading.Lock()
        self._done = False
        self._result: dict | None = None
        self.progress = RunExportProgress(len(job.get("rows") or ()))



    def is_done(self) -> bool:

        with self._lock:
            return self._done

    def take_result(self) -> dict | None:


        with self._lock:
            result = self._result
            self._result = None
        return result

    def join_run(self, timeout_ms: int | None = None) -> bool:




        if not self.isRunning():
            return True
        if timeout_ms is None:
            timeout_ms = RUN_EXPORT_JOIN_TIMEOUT_MS
            try:
                from ..core.server_dials import dial_in_range
                timeout_ms = dial_in_range(
                    "tuning.export.run_join_timeout_ms",
                    RUN_EXPORT_JOIN_TIMEOUT_MS, 1000, 120000)
            except Exception:  # noqa: BLE001
                timeout_ms = RUN_EXPORT_JOIN_TIMEOUT_MS
        return bool(self.wait(timeout_ms))



    def run(self) -> None:  # noqa: D102
        result = None
        try:
            from ..core.run_export_job import run_export_job

            result = run_export_job(self._job, self.progress)
        except Exception:  # noqa: BLE001
            result = {"written": None, "count": 0, "area_m2": 0.0,
                      "overlapping_pairs": None, "failure": "file_refused"}
        finally:
            with self._lock:
                self._result = result
                self._done = True


            self._job = {}
