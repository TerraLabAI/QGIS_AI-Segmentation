"""Hand the crash-net autosave to a thread, and fold its answer back.

The autosave writes every merged object of a finished run to a GeoPackage
before the review-entry tail runs on it. It used to do that on the GUI thread,
between the last tile and the review, which is exactly where the user is
waiting: measured at 3.6 s on a real 13 500-object building run.

This mixin keeps the GUI half (reading the project for the output CRS, the
directory and the ellipsoid, and copying the geometries out as WKB) and gives
the write itself to workers.run_autosave_thread. A timer here polls the thread,
because that thread carries no Qt signals on purpose.

Part of AISegmentationPlugin (see ai_segmentation_plugin.py). Methods here are
plain mixin members: state lives on the plugin instance (self).
"""
from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog, QgsProject

#: How often the GUI asks the writer thread whether it is finished. About a
#: drawn frame apart is pointless for a write that takes seconds, and the poll
#: itself is two attribute reads.
_AUTOSAVE_POLL_MS = 250


class AutoAutosaveOffloadMixin:
    """Owns the run-autosave writer thread and its GUI-side poll."""

    def _start_billed_autosave(self, merged_ided: list) -> bool:
        """Start the crash-net write on a thread. True when the thread took it.

        False means the caller must write synchronously, exactly as before this
        thread existed. Never raises: the crash net can fail, it can never
        break finalize.
        """
        try:
            from ...core import run_autosave
            from ...workers.run_autosave_thread import RunAutosaveThread
        except Exception:  # noqa: BLE001 -- fall back to the blocking write
            return False
        # A previous run's write must be finished before this one opens the
        # same GeoPackage: two writers on one SQLite file is a locked database,
        # not a race worth taking.
        self._finish_billed_autosave()
        ctx = self._auto_run_ctx or {}
        source_layer = None
        layer_id = ctx.get("layer_id")
        if layer_id:
            source_layer = QgsProject.instance().mapLayer(layer_id)
        try:
            job = run_autosave.prepare_autosave(
                merged_ided, self._auto_crs_authid or "EPSG:4326",
                str(ctx.get("prompt") or "").strip(),
                self._auto_run_id or "", source_layer=source_layer)
        except Exception:  # noqa: BLE001 -- nothing prepared, nothing to write
            job = None
        if not job:
            return True  # nothing to save: not a reason to write it twice
        try:
            thread = RunAutosaveThread(job)
            thread.start()
        except Exception as exc:  # noqa: BLE001 -- the blocking path still works
            QgsMessageLog.logMessage(
                f"Auto detection: the run autosave stays on the interface "
                f"thread ({exc})", "AI Segmentation",
                level=Qgis.MessageLevel.Info)
            self._auto_autosave_thread = None
            return False
        self._auto_autosave_thread = thread
        self._schedule_billed_autosave_poll()
        return True

    def _schedule_billed_autosave_poll(self) -> None:
        """Arm the next look at the writer thread."""
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(_AUTOSAVE_POLL_MS, self._poll_billed_autosave)

    def _poll_billed_autosave(self) -> None:
        """Adopt the write's answer once it lands, or come back later.

        Runs on the GUI thread, which is where the pending pointer, the log
        line and the repaint belong. Never raises: it is a bare timer slot.
        """
        thread = getattr(self, "_auto_autosave_thread", None)
        if thread is None:
            return
        try:
            if not thread.is_done():
                self._schedule_billed_autosave_poll()
                return
        except RuntimeError:
            self._auto_autosave_thread = None
            return
        self._auto_autosave_thread = None
        try:
            thread.join_run(1000)
        except Exception:  # nosec B110 -- it said it was done
            pass
        self._adopt_billed_autosave(thread)

    def _adopt_billed_autosave(self, thread) -> None:
        """Record what one finished writer thread produced. Never raises."""
        try:
            info = thread.take_result()
        except Exception:  # noqa: BLE001 -- a lost answer is a lost crash net
            info = None
        if not info:
            return
        try:
            from ...core import run_autosave

            run_autosave.record_pending(info)
            QgsMessageLog.logMessage(
                "Auto detection: autosaved {n} object(s) to disk".format(
                    n=info.get("count", 0)),
                "AI Segmentation", level=Qgis.MessageLevel.Info)
            # This write is a deferred writer into the file the run's own saved
            # layers display, so it may be the last one to touch it. A commit
            # that lands inside a layer's first render leaves an empty map that
            # nothing asks to draw again.
            run_autosave.repaint_layers_over(str(info.get("path") or ""))
        except Exception as exc:  # noqa: BLE001 -- the crash net never breaks a run
            # The objects are on disk but may not be on the map; that is a
            # "my polygons vanished" report in the making.
            try:
                from ...core.telemetry_errors import track_plugin_error
                track_plugin_error(stage="segment",
                                   error_code="autosave_repaint_failed",
                                   message=type(exc).__name__)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        # The merged set is safe on disk: the raw per-tile fragments are now a
        # duplicate copy. Keep them only where the exemplar-only count-vs-map
        # re-merge still reads them.
        try:
            if not getattr(self, "_auto_retain_raw", False):
                self._auto_raw_fragments = None
        except Exception:  # nosec B110
            pass

    def _billed_autosave_running(self) -> bool:
        """Whether a crash-net write is still in flight."""
        thread = getattr(self, "_auto_autosave_thread", None)
        if thread is None:
            return False
        try:
            return not thread.is_done()
        except RuntimeError:
            return False

    def _finish_billed_autosave(self) -> None:
        """Wait for the crash-net write, then adopt its answer.

        Called by every path that abandons or restarts a run
        (_reset_auto_live_pipeline), and by the finalize error path before it
        reads the pending pointer: that path recovers the run from this very
        file, so it may not read the pointer while the write is still running.
        Idempotent, and never nulls a live QThread without parking it:
        garbage-collecting a running QThread aborts QGIS.
        """
        thread = getattr(self, "_auto_autosave_thread", None)
        self._auto_autosave_thread = None
        if thread is None:
            return
        try:
            from ...workers.run_autosave_thread import (
                RUN_AUTOSAVE_JOIN_TIMEOUT_MS,
            )

            if thread.join_run(RUN_AUTOSAVE_JOIN_TIMEOUT_MS):
                self._adopt_billed_autosave(thread)
                return
            # Wedged inside the write. Nothing may cut it short (a half-written
            # table is worse than a late one), so anchor it until it ends.
            from .shared import park_orphaned_worker

            park_orphaned_worker(thread)
        except RuntimeError:
            pass  # the C++ half is already gone
        except Exception:  # noqa: BLE001 -- teardown must never propagate  # nosec B110
            pass
