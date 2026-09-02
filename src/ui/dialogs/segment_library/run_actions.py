"""What the library can do WITH a past run: bring it back, point at it again,
write it to a file.

All three read the run back from the account, which is a network call and then
real work on the result (decoding stored masks into geometry, writing a
GeoPackage). Every bit of that runs on a background thread, behind one wait
window with a Cancel: done in the click handler they froze QGIS for a minute
or more on a big run, with nothing on screen.

One at a time, on purpose. While a fetch is out, the cards hide their action
rows rather than taking a click that answers nothing.

Mixed into SegmentLibraryDialog beside LibraryRailMixin and
LibraryHistoryMixin. Never define a method name that one of those defines.
"""
from __future__ import annotations

from qgis.core import Qgis
from qgis.PyQt.QtWidgets import QApplication, QMessageBox

from ....core import qt_compat as QtC
from ....core.i18n import tr
from ....core.logging_utils import log
from ....core.qt_compat import safe_disconnect
from ...plugin.shared import park_orphaned_worker
from .common import _project_layer_reading, _run_key
from .detail import _ExportRunDialog, _RunProgressDialog
from .workers import _RunFetchWorker, _RunZoneFetchWorker


class LibraryRunActionsMixin:
    """Restore, re-run and export for one stored run."""

    # ---- the one fetch at a time -----------------------------------------

    def _start_run_fetch(self, run: dict, action: tuple) -> None:
        """Everything a restore or an export needs, on one background thread.

        The tiles, the stored masks, the decode into geometry and (for an
        export) the file write all happen there.
        """
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

        # (driver, confidence, path) for an export, None for a restore.
        export = tuple(action[1:4]) if action and action[0] == "export" else None
        # The merge policy is read HERE, on the GUI thread, and handed over as
        # a plain bool: the worker must touch neither the plugin nor its caches.
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

    # ---- the wait window --------------------------------------------------

    def _show_fetch_progress(self) -> None:
        """Arm the wait window, shown after a beat so a short run does not
        flash a dialog on screen and take it away again."""
        dlg = _RunProgressDialog(self._detail_dlg or self)
        dlg.cancelled.connect(self._on_fetch_cancel_requested)
        self._fetch_progress = dlg
        QtC.safe_single_shot(350, self, self._reveal_fetch_progress)

    def _reveal_fetch_progress(self) -> None:
        dlg = self._fetch_progress
        if dlg is None or not self._hist_busy:
            return
        try:
            dlg.show()
        except RuntimeError:
            pass

    def _close_fetch_progress(self) -> None:
        """Take the wait window down because the work ended.

        Never call this from its own cancelled signal: Qt routes a programmatic
        close through reject(), and tearing a widget down inside its own signal
        aborts QGIS on Qt6.
        """
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
        """Stop waiting for this run, now.

        The thread is usually inside a blocking network call, so it takes up to
        one call to notice. Its signals are cut here rather than left to land
        in a dialog that has moved on, and the user gets the buttons back at
        once; park_orphaned_worker owns what is left of the thread's life.
        """
        worker = self._fetch_worker
        self._fetch_worker = None
        # The window is closing itself (this runs from its own signal), so only
        # drop the handle to it.
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
            # One guard per signal: batched with the interrupt above, a thread
            # that had already finished raised on the first call and left all
            # three handlers connected.
            for signal_name in ("fetched", "failed", "progress"):
                safe_disconnect(worker, signal_name)
        self._end_run_fetch()

    def _on_run_fetch_cancelled(self) -> None:
        """The thread noticed the stop and wound down. The dialog freed itself
        the moment Cancel was clicked, so there is nothing to undo here."""
        log("Run history fetch stopped by the user", Qgis.MessageLevel.Info)

    # ---- the three actions ------------------------------------------------

    def _request_restore(self, run: dict, _detail_dlg=None) -> None:
        if self._plugin is None or self._view_only:
            return
        self._start_run_fetch(run, ("restore",))

    def _request_rerun(self, run: dict, _detail_dlg=None) -> None:
        """Point the Automatic flow back at this run: same ground, same object,
        same number of tiles, stopped one click short of spending anything.

        Only the tile rows are fetched. The stored detections are what Restore
        is for, and pointing at a zone does not need them.
        """
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
        # Same wait window as Restore and Export: this reads the run back over
        # the network too, and it used to look like nothing had happened.
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
            QMessageBox.warning(
                self, tr("Segment library"),
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
        self.reject()  # close first; the plugin work is deferred a tick
        payload = {
            "prompt": run.get("prompt") or "",
            "extent": list(extent),
            "crs": authid,
            "tiles": int(run.get("tiles") or len(tiles)),
        }
        # The shape the user drew, when the run kept it: the tile union is its
        # bounding box, and re-running a box around an L-shaped zone bills for
        # ground the first run never looked at. Carried in the same CRS as the
        # extent beside it; absent on every older run, which keeps the box.
        outline = zone_geometry_from_run(run, authid)
        if outline is not None:
            payload["zone_wkt"] = outline.asWkt()
        dock.history_rerun_requested.emit(payload)

    def _request_export(self, run: dict, _detail_dlg=None) -> None:
        if self._plugin is None or self._view_only:
            return
        from ...plugin.run_restore import snap_confidence
        default_conf = snap_confidence(run.get("threshold"), 0.30)
        if default_conf <= 0.15:
            default_conf = 0.30
        dlg = _ExportRunDialog(run, default_conf, self._detail_dlg or self)
        if not dlg.exec() or not dlg.path():
            return
        self._start_run_fetch(
            run, ("export", dlg.driver(), dlg.confidence(), dlg.path()))

    # ---- what came back ---------------------------------------------------

    def _on_run_fetch_failed(self, code: str) -> None:
        self._fetch_worker = None
        self._close_fetch_progress()
        self._end_run_fetch()
        log(f"Run history fetch failed: {code}", Qgis.MessageLevel.Warning)
        QMessageBox.warning(
            self, tr("Segment library"),
            tr("Could not load this run's stored detections. Try again later."))

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
        """One sentence when the fetch ran out of its wall-clock budget, so a
        short result is never passed off as the whole run. Empty otherwise."""
        skipped = int(outcome.get("tiles_skipped") or 0)
        if skipped <= 0:
            return ""
        return tr("{n} part(s) of this run took too long to load and are "
                  "missing from this result.").format(n=skipped)

    def _finish_restore(self, run: dict, tiles: list, decoded: dict) -> None:
        from qgis.PyQt.QtCore import Qt

        from ...plugin import run_restore
        # Building the review is still GUI work (the same tail a live run runs
        # through _complete_auto_finalize); the decode that used to dominate it
        # is already done on the thread.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok = run_restore.restore_run(self._plugin, run, tiles, decoded)
        finally:
            QApplication.restoreOverrideCursor()
        if not ok:
            QMessageBox.warning(
                self, tr("Segment library"),
                tr("Could not load this run's stored detections. Try again later."))
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
        self.reject()  # no prompt chosen; the review is now open on the map

    def _finish_export(self, run: dict, outcome: dict, driver: str,
                       confidence: float, path: str) -> None:
        """The file is already written (the fetch thread did it); what is left
        is putting it on the map and saying how it went."""
        from qgis.core import QgsProject

        from ...plugin.run_restore import load_exported_layer

        summary = outcome.get("export") or {}
        count = int(summary.get("count") or 0)
        layer = load_exported_layer(path, driver) if summary.get("written") else None
        if not count or layer is None:
            QMessageBox.warning(
                self, tr("Segment library"),
                tr("Nothing to export at this confidence. Lower it and try again.")
                if not count else
                tr("The export failed. Check the file path and try again."))
            if not count:
                # The warning lands after the dialog closed, so a retry would
                # start again from an empty format, path and confidence.
                self._reopen_export(run, driver, confidence, path)
            return
        # Exporting the same run twice used to stack a second layer on the
        # first: same file, same name, two entries the user has to tell apart.
        # Refresh the one already reading that file instead.
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
        QMessageBox.information(
            self, tr("Segment library"),
            tr("Exported {n} polygon(s).").format(n=count) + (f"\n\n{note}" if note else ""))

    def _reopen_export(self, run: dict, driver: str, confidence: float,
                       path: str) -> None:
        """Bring the export dialog back on the choices the last run used.

        The format is picked by walking the combo, since the dialog owns the
        driver list; seeding the path also arms its Export button, which only
        the file chooser turns on."""
        dlg = _ExportRunDialog(run, confidence, self._detail_dlg or self)
        # Blocked: the walk is a search, not four choices, and each step fired
        # currentIndexChanged and rewrote the path's extension.
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
