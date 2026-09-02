














from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog, QgsProject




_AUTOSAVE_POLL_MS = 250


class AutoAutosaveOffloadMixin:


    def _start_billed_autosave(self, merged_ided: list) -> bool:






        try:
            from ...core import run_autosave
            from ...workers.run_autosave_thread import RunAutosaveThread
        except Exception:  # noqa: BLE001
            return False



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
        except Exception:  # noqa: BLE001
            job = None
        if not job:
            return True
        try:
            thread = RunAutosaveThread(job)
            thread.start()
        except Exception as exc:  # noqa: BLE001
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

        from qgis.PyQt.QtCore import QTimer

        from ...core.server_dials import dial_in_range
        QTimer.singleShot(
            dial_in_range("tuning.auto.autosave_poll_ms", _AUTOSAVE_POLL_MS, 50, 2000),
            self._poll_billed_autosave)

    def _poll_billed_autosave(self) -> None:





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
        except Exception:  # nosec B110
            pass
        self._adopt_billed_autosave(thread)

    def _adopt_billed_autosave(self, thread) -> None:

        try:
            info = thread.take_result()
        except Exception:  # noqa: BLE001
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




            run_autosave.repaint_layers_over(str(info.get("path") or ""))
        except Exception as exc:  # noqa: BLE001


            try:
                from ...core.telemetry_errors import track_plugin_error
                track_plugin_error(stage="segment",
                                   error_code="autosave_repaint_failed",
                                   message=type(exc).__name__)
            except Exception:  # noqa: BLE001  # nosec B110
                pass



        try:
            if not getattr(self, "_auto_retain_raw", False):
                self._auto_raw_fragments = None
        except Exception:  # nosec B110
            pass

    def _finish_billed_autosave(self) -> None:









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


            from .shared import park_orphaned_worker

            park_orphaned_worker(thread)
        except RuntimeError:
            pass
        except Exception:  # noqa: BLE001  # nosec B110
            pass
