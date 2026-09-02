"""Write a finished run's crash-net autosave off the GUI thread.

The autosave streams every merged object of the run into a GeoPackage table the
moment the merger is read at finalize. On a dense run that is thousands of
geometries, a geodesic area and a perimeter each, and it ran on the thread that
draws the map: measured at 3.6 s on a real 13 500-object building run, spent
between the last tile and the review with QGIS frozen.

This thread takes the write instead. The GUI keeps the half that reads the
project (the output CRS, the transform context, the directory, the ellipsoid)
and hands over WKB copies, so nothing here touches a geometry the run still
holds or a QgsProject the GUI is using.

It carries NO Qt signals, for the same reason ``ReviewRefineThread`` does not:
the plugin polls it from a timer it owns, so there is no queued cross-thread
emission that could be delivered to a plugin that has since been torn down.

There is no abort. A half-written GeoPackage table is worse than a late one, so
every teardown path JOINS this thread instead of cutting it short.
"""
from __future__ import annotations

import threading

from qgis.PyQt.QtCore import QThread

#: How long a teardown waits for the write to finish before parking the thread.
#: One pass over a dense run measured under four seconds, and the wait only
#: happens on a path that is abandoning the run anyway.
RUN_AUTOSAVE_JOIN_TIMEOUT_MS = 10000


class RunAutosaveThread(QThread):
    """Runs one prepared autosave job to completion, then holds its answer.

    Built with the job dict ``core.run_autosave.prepare_autosave`` returns,
    started once, and read through ``take_result`` when ``is_done`` says the
    write has finished.
    """

    def __init__(self, job: dict, parent=None) -> None:
        super().__init__(parent)
        self._job = job
        self._lock = threading.Lock()
        self._done = False
        self._info: dict | None = None

    # ---- GUI-thread API -----------------------------------------------------

    def is_done(self) -> bool:
        """Whether the write has finished, successfully or not."""
        with self._lock:
            return self._done

    def take_result(self) -> dict | None:
        """The pending-pointer dict the write produced, or None.

        Reading it twice gives None the second time, so a poll that races a
        teardown cannot record the same autosave under two pointers.
        """
        with self._lock:
            info = self._info
            self._info = None
        return info

    def join_run(self, timeout_ms: int = RUN_AUTOSAVE_JOIN_TIMEOUT_MS) -> bool:
        """Block until the thread has exited. True when it did.

        A False return means the write is wedged, which the caller must treat
        as "park it, never delete it": destroying a running QThread aborts
        QGIS.
        """
        if not self.isRunning():
            return True
        return bool(self.wait(timeout_ms))

    # ---- Writer-thread body -------------------------------------------------

    def run(self) -> None:  # noqa: D102 - QThread entry point
        info = None
        try:
            from ..core.run_autosave import write_prepared_autosave

            info = write_prepared_autosave(self._job)
        except Exception:  # noqa: BLE001 -- the crash net never breaks a run
            info = None
        finally:
            with self._lock:
                self._info = info
                self._done = True
            # The rows are the biggest thing the job holds, and the plugin may
            # keep this object parked for a while after the write.
            self._job = {}
