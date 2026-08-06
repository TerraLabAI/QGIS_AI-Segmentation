





















from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr



_EXPORT_POLL_MS = 100


class AutoExportOffloadMixin:


    def _auto_review_export_busy(self) -> bool:

        return getattr(self, "_auto_export_job", None) is not None

    def _start_auto_review_export_async(self, include_hidden: bool) -> bool:






        try:
            from ...workers.run_export_thread import RunExportThread
        except Exception:  # noqa: BLE001
            return False
        collected = self._collect_auto_review_export(include_hidden, False)
        if collected is None:
            self._set_auto_export_saving_state(False)
            return True
        review = collected["review"]
        export = self._prepare_auto_export(
            collected["refined"], review["crs"], review["source_layer_name"],
            review["prompt"], scores=collected["refined_scores"],
            confidence_applied=collected["conf_applied"])
        if export is None:


            self._land_auto_review_export(collected, None, None)
            return True
        try:
            thread = RunExportThread(export["job"])
            thread.start()
        except Exception as exc:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Export: the save stays on the interface thread ({exc})",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
            from ...core.run_export_job import run_export_job

            self._land_auto_review_export(
                collected, export, run_export_job(export["job"]))
            return True
        self._auto_export_job = {
            "thread": thread, "collected": collected, "export": export}
        self._schedule_auto_review_export_poll()
        return True

    def _schedule_auto_review_export_poll(self) -> None:

        from ...core.qt_compat import safe_single_shot
        try:
            from ...core.server_dials import dial_in_range
            poll_ms = dial_in_range("tuning.export.review_poll_ms", _EXPORT_POLL_MS, 50, 500)
        except Exception:  # noqa: BLE001
            poll_ms = _EXPORT_POLL_MS

        owner = self.dock_widget or self.iface.mainWindow()
        safe_single_shot(poll_ms, owner, self._poll_auto_review_export)

    def _poll_auto_review_export(self) -> None:


        state = getattr(self, "_auto_export_job", None)
        if state is None:
            return
        thread = state["thread"]
        try:
            done = thread.is_done()
        except RuntimeError:
            done = True
        if not done:
            try:
                self.dock_widget.set_auto_export_progress(
                    int(round(thread.progress.fraction() * 100)))
            except (RuntimeError, AttributeError):
                pass
            self._schedule_auto_review_export_poll()
            return
        try:
            thread.join_run(1000)
        except Exception:  # nosec B110
            pass
        self._complete_auto_review_export(state)

    def _finish_auto_review_export_offload(self) -> None:







        state = getattr(self, "_auto_export_job", None)
        if state is None:
            return
        thread = state["thread"]
        try:

            finished = thread.join_run()
        except RuntimeError:
            finished = True
        except Exception:  # noqa: BLE001
            finished = False
        if finished:
            self._complete_auto_review_export(state, report=False)
            return
        self._auto_export_job = None
        try:
            from .shared import park_orphaned_worker

            park_orphaned_worker(thread)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        self._set_auto_export_saving_state(False)
        QgsMessageLog.logMessage(
            "Export: the save did not finish in time; the review stays open",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)

    def _complete_auto_review_export(self, state: dict, report: bool = True) -> None:


        if getattr(self, "_auto_export_job", None) is state:
            self._auto_export_job = None
        try:
            result = state["thread"].take_result()
        except Exception:  # noqa: BLE001
            result = None
        if result is None:
            result = {"written": None, "count": 0, "area_m2": 0.0,
                      "overlapping_pairs": None, "failure": "file_refused"}
        self._land_auto_review_export(
            state["collected"], state["export"], result, report=report)

    def _land_auto_review_export(self, collected: dict, export: dict | None,
                                 result: dict | None, report: bool = True) -> None:


        exported = None
        try:
            name = None
            if export is not None and result is not None:
                name = self._adopt_auto_export(export, result)
            if collected["review"] is self._auto_review:
                exported = self._conclude_auto_review_export(collected, name)
            elif name:



                exported = (name, int(getattr(self, "_auto_export_feature_count", 0) or 0))
        except Exception as exc:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Export: landing the save failed ({type(exc).__name__})",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            try:
                from ...core.telemetry_errors import track_plugin_error
                track_plugin_error(stage="export", error_code="export_failed",
                                   message=type(exc).__name__)
            except Exception:  # noqa: BLE001  # nosec B110
                pass
            exported = (None, len(collected.get("refined") or ()))
        self._set_auto_export_saving_state(False)
        if not report:
            return
        try:
            self._finish_auto_export_click(exported)
        except Exception:  # noqa: BLE001
            try:
                from ..error_report_dialog import show_error_report
                show_error_report(
                    self.iface.mainWindow(), tr("Export Failed"),
                    tr("Something went wrong saving your detections. Please try again."),
                    error_code="export_failed")
            except Exception:  # noqa: BLE001  # nosec B110
                pass

    def _set_auto_export_saving_state(self, saving: bool) -> None:

        try:
            self.dock_widget.set_auto_export_saving(bool(saving))
        except (RuntimeError, AttributeError):
            pass
