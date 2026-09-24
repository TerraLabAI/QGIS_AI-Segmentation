














from __future__ import annotations

from qgis.core import Qgis
from qgis.PyQt.QtWidgets import QApplication

from ....core import qt_compat as QtC
from ....core.i18n import tr
from ....core.logging_utils import log
from ....core.qt_compat import safe_disconnect
from ....core.server_dials import dial_in_range
from ...plugin.shared import park_orphaned_worker
from ..confirm_dialog import success_box, warning_box
from .common import _project_layer_reading, _run_key
from .detail import _ExportRunDialog, _RunProgressDialog
from .workers import _RunFetchWorker, _RunZoneFetchWorker


class LibraryRunActionsMixin:




    def _start_run_fetch(self, run: dict, action: tuple) -> None:





        if self._hist_busy or not self._auth:
            return
        self._hist_busy = True
        self._pending_action = action
        self._set_run_actions_enabled(False)
        actor = "export" if action and action[0] == "export" else "restore"
        if self._detail_dlg is not None:
            try:
                self._detail_dlg.set_busy(True, actor)
            except RuntimeError:
                pass
        from ...plugin.run_restore import run_merge_separate


        export = tuple(action[1:4]) if action and action[0] == "export" else None


        worker = _RunFetchWorker(
            self._history_client(), self._auth, run,
            run_merge_separate(self._plugin, run), export)
        worker.fetched.connect(self._on_run_fetched)
        worker.failed.connect(self._on_run_fetch_failed)
        worker.cancelled.connect(self._on_run_fetch_cancelled)
        worker.progress.connect(self._on_run_fetch_progress)
        self._fetch_worker = worker
        self._track_live_worker(
            worker, "fetched", "failed", "cancelled", "progress")
        self._show_fetch_progress()
        park_orphaned_worker(worker)
        worker.start()

    def _end_run_fetch(self) -> None:
        self._hist_busy = False
        self._pending_action = None
        self._set_run_actions_enabled(self._run_actions_available())
        if self._detail_dlg is not None:
            try:
                self._detail_dlg.set_busy(False)
            except RuntimeError:
                pass



    def _show_fetch_progress(self) -> None:


        dlg = _RunProgressDialog(self._detail_dlg or self)
        dlg.cancelled.connect(self._on_fetch_cancel_requested)
        self._fetch_progress = dlg
        reveal_ms = dial_in_range("tuning.library.fetch_progress_reveal_ms", 350, 100, 2000)
        QtC.safe_single_shot(reveal_ms, self, self._reveal_fetch_progress)

    def _reveal_fetch_progress(self) -> None:
        dlg = self._fetch_progress
        if dlg is None or not self._hist_busy:
            return
        try:
            dlg.show()


            dlg.raise_()
            dlg.activateWindow()
        except RuntimeError:
            pass

    def _close_fetch_progress(self) -> None:






        dlg = self._fetch_progress
        self._fetch_progress = None
        if dlg is None:
            return
        try:
            dlg.finish()
            dlg.deleteLater()
        except RuntimeError:
            pass

    def _on_run_fetch_progress(self, phase: str, done: int, total: int) -> None:
        dlg = self._fetch_progress
        if dlg is None:
            return
        if phase == "decode":
            text = tr("Rebuilding shapes ({done} of {total})").format(
                done=done, total=total)
        elif phase == "write":
            text = tr("Writing the file...")
            done, total = 0, 0
        else:
            text = tr("Loading stored detections ({done} of {total})").format(
                done=done, total=total)
        try:
            dlg.set_step(text, done, total)
        except RuntimeError:
            pass

    def _on_fetch_cancel_requested(self) -> None:







        worker = self._fetch_worker
        self._fetch_worker = None


        dlg = self._fetch_progress
        self._fetch_progress = None
        if dlg is not None:
            try:
                dlg.deleteLater()
            except RuntimeError:
                pass
        if worker is not None:
            try:
                worker.requestInterruption()
            except (RuntimeError, TypeError):
                pass



            for signal_name in ("fetched", "failed", "progress"):
                safe_disconnect(worker, signal_name)
        self._end_run_fetch()

    def _on_run_fetch_cancelled(self) -> None:


        log("Run history fetch stopped by the user", Qgis.MessageLevel.Info)



    def _request_restore(self, run: dict, _detail_dlg=None) -> None:
        if self._plugin is None or self._view_only:
            return
        self._start_run_fetch(run, ("restore",))

    def _request_rerun(self, run: dict, _detail_dlg=None) -> None:






        if self._plugin is None or self._view_only or self._hist_busy:
            return
        if not self._auth:
            return
        self._hist_busy = True
        self._set_run_actions_enabled(False)
        if self._detail_dlg is not None:
            try:
                self._detail_dlg.set_busy(True, "rerun")
            except RuntimeError:
                pass
        worker = _RunZoneFetchWorker(self._history_client(), self._auth, run)
        worker.fetched.connect(self._on_rerun_zone_fetched)
        worker.failed.connect(self._on_run_fetch_failed)
        self._track_live_worker(worker, "fetched", "failed")
        self._fetch_worker = worker


        self._show_fetch_progress()
        park_orphaned_worker(worker)
        worker.start()

    def _on_rerun_zone_fetched(self, run: dict, tiles: list) -> None:
        from ...plugin.run_restore import (
            zone_extent_from_tiles,
            zone_geometry_from_run,
        )
        self._fetch_worker = None
        self._close_fetch_progress()
        self._end_run_fetch()
        zone = zone_extent_from_tiles(tiles)
        if zone is None:
            warning_box(
                self,
                tr("This run did not keep where it looked, so it cannot be "
                   "pointed at the same place. Draw the zone again."))
            return
        extent, authid = zone
        dock = self._dock_widget()
        if dock is None:
            return
        if self._detail_dlg is not None:
            try:
                self._detail_dlg.reject()
            except RuntimeError:
                pass
        self.reject()
        payload = {
            "prompt": run.get("prompt") or "",
            "extent": list(extent),
            "crs": authid,
            "tiles": (run.get("tiles")
                      if isinstance(run.get("tiles"), int) and run.get("tiles") > 0
                      else len(tiles)),
        }




        outline = zone_geometry_from_run(run, authid)
        if outline is not None:
            payload["zone_wkt"] = outline.asWkt()
        dock.history_rerun_requested.emit(payload)

    def _request_export(self, run: dict, _detail_dlg=None) -> None:
        if self._plugin is None or self._view_only:
            return
        from ...plugin.run_restore import snap_confidence
        default_start = dial_in_range(
            "tuning.library.export_default_confidence", 0.30, 0.05, 0.95)
        conf_floor = dial_in_range(
            "tuning.library.export_confidence_floor", 0.15, 0.0, 0.5)
        default_conf = snap_confidence(run.get("threshold"), default_start)
        if default_conf <= conf_floor:
            default_conf = default_start
        dlg = _ExportRunDialog(run, default_conf, self._detail_dlg or self)
        if not dlg.exec() or not dlg.path():
            return
        self._start_run_fetch(
            run, ("export", dlg.driver(), dlg.confidence(), dlg.path()))



    def _on_run_fetch_failed(self, code: str) -> None:
        self._fetch_worker = None
        self._close_fetch_progress()
        self._end_run_fetch()
        log(f"Run history fetch failed: {code}", Qgis.MessageLevel.Warning)
        warning_box(
            self, tr("Could not load this run's stored detections. Try again later."))

    def _on_run_fetched(self, run: dict, tiles: list, outcome: dict) -> None:
        action = self._pending_action or ("restore",)
        self._fetch_worker = None
        self._close_fetch_progress()
        self._end_run_fetch()
        if "export" in outcome:
            self._finish_export(run, outcome, action[1], action[2], action[3])
            return
        self._finish_restore(run, tiles, outcome)

    def _missing_tiles_note(self, outcome: dict) -> str:


        skipped = int(outcome.get("tiles_skipped") or 0)
        if skipped <= 0:
            return ""
        return tr("{n} part(s) of this run took too long to load and are "
                  "missing from this result.").format(n=skipped)

    def _finish_restore(self, run: dict, tiles: list, decoded: dict) -> None:
        from qgis.PyQt.QtCore import Qt

        from ...plugin import run_restore



        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok = run_restore.restore_run(self._plugin, run, tiles, decoded)
        finally:
            QApplication.restoreOverrideCursor()
        if not ok:
            warning_box(
                self, tr("Could not load this run's stored detections. Try again later."))
            return
        self._mark_run_done(run, "restored")
        note = self._missing_tiles_note(decoded)
        if note:
            try:
                self._plugin.iface.messageBar().pushWarning(
                    "AI Segmentation", note)
            except (RuntimeError, AttributeError):

                pass
        if self._detail_dlg is not None:
            try:
                self._detail_dlg.accept()
            except RuntimeError:
                pass
        self.reject()

    def _finish_export(self, run: dict, outcome: dict, driver: str,
                       confidence: float, path: str) -> None:


        from qgis.core import QgsProject

        from ...plugin.run_restore import load_exported_layer

        summary = outcome.get("export") or {}
        count = int(summary.get("count") or 0)
        layer = load_exported_layer(path, driver) if summary.get("written") else None
        if not count or layer is None:
            warning_box(
                self,
                tr("Nothing to export at this confidence. Lower it and try again.")
                if not count else
                tr("The export failed. Check the file path and try again."))
            if not count:


                self._reopen_export(run, driver, confidence, path)
            return



        existing = _project_layer_reading(layer.source())
        if existing is None:
            QgsProject.instance().addMapLayer(layer)
        else:
            try:
                existing.dataProvider().reloadData()
                existing.triggerRepaint()
            except (RuntimeError, AttributeError):
                pass
        self._mark_run_done(run, "exported")
        try:
            from ....core import telemetry_session_events
            telemetry_session_events.track_history_exported(driver, count, run_id=_run_key(run))
        except Exception:
            pass  # nosec B110
        note = self._missing_tiles_note(outcome)



        success_box(self, tr("Exported {n} polygon(s).").format(n=count), note)

    def _reopen_export(self, run: dict, driver: str, confidence: float,
                       path: str) -> None:





        dlg = _ExportRunDialog(run, confidence, self._detail_dlg or self)


        dlg.format_combo.blockSignals(True)
        for idx in range(dlg.format_combo.count()):
            dlg.format_combo.setCurrentIndex(idx)
            if dlg.driver() == driver:
                break
        dlg.format_combo.blockSignals(False)
        if path:
            dlg.path_edit.setText(path)
            dlg.ok_btn.setEnabled(True)
        if not dlg.exec() or not dlg.path():
            return
        self._start_run_fetch(
            run, ("export", dlg.driver(), dlg.confidence(), dlg.path()))
