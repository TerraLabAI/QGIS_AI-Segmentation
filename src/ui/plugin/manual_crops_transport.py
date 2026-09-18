









from __future__ import annotations

import threading

from qgis.core import (
    Qgis,
    QgsMessageLog,
    QgsPointXY,
)
from qgis.PyQt.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
)
from qgis.PyQt.QtWidgets import QApplication

from ...core.i18n import tr
from ..error_report_dialog import show_error_report


class CropReadWorker(QThread):














    done = pyqtSignal(int, object)

    def __init__(self, args: dict, generation: int, cancel=None, parent=None):
        super().__init__(parent)
        self._args = dict(args)
        self._generation = generation
        self._cancel = cancel

    def _abandoned(self) -> bool:
        return self._cancel is not None and self._cancel.is_set()

    def run(self):
        from ...core.feature_encoder import extract_crop_from_raster
        if self._abandoned():


            return
        try:
            result = extract_crop_from_raster(**self._args)
        except Exception as e:  # noqa: BLE001
            result = (None, None, str(e), "crop_error_unknown")
        if self._abandoned():


            return
        self.done.emit(self._generation, result)


class DirectTileFetchWorker(QThread):


















    done = pyqtSignal(int, object)

    def __init__(self, request, generation: int, cancel_check=None, parent=None):
        super().__init__(parent)
        self._request = request
        self._generation = generation
        self._cancel_check = cancel_check

    def run(self):
        try:
            if self._cancel_check is not None and self._cancel_check():
                result = (None, "crop_error_online_cancelled", 0)
            else:
                result = self._fetch_tiles()
        except Exception as e:  # noqa: BLE001
            result = (None, str(e), 0)
        self.done.emit(self._generation, result)

    def _fetch_tiles(self):

        from ...core.feature_encoder import run_direct_tile_fetch

        return run_direct_tile_fetch(self._request,
                                     cancel_check=self._cancel_check)


class ManualCropsTransportMixin:









    def _begin_file_crop_read(self, center_point, mupp_override, on_encoded,
                              *, quiet: bool = False,
                              show_busy: bool = True) -> bool:











        self._ensure_manual_encode_state()
        args = self._file_crop_read_args(center_point, mupp_override, quiet)
        if args is None:
            return False
        self._inflight_crop_window = self._crop_window_key_for(
            center_point, mupp_override)

        from ...core.feature_encoder import crop_read_is_thread_safe

        if not crop_read_is_thread_safe(args["raster_path"]):






            from ...core.feature_encoder import extract_crop_from_raster








            self._show_inline_read_busy(show_busy)
            try:
                result = extract_crop_from_raster(**args)
            finally:
                self._hide_inline_read_busy(show_busy)
            return self._deliver_crop_read(result, on_encoded, quiet=quiet,
                                           show_busy=show_busy)

        self._manual_encode_gen += 1
        gen = self._manual_encode_gen
        cancel = threading.Event()
        try:
            from .shared import park_orphaned_worker
            worker = CropReadWorker(args, gen, cancel=cancel)
        except Exception as e:  # noqa: BLE001
            self._report_crop_error(str(e), "crop_error_unknown", quiet)
            return False

        self._crop_read = {
            "worker": worker,
            "cancel": cancel,
            "gen": gen,
            "on_encoded": on_encoded,
            "cursor": bool(show_busy),
            "quiet": bool(quiet),
            "show_busy": bool(show_busy),
        }
        self._encoding_in_progress = True
        self._encode_lock_gen = gen
        self._encode_cursor_set = bool(show_busy)
        if show_busy:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            self._set_manual_encoding_note(True)
        try:
            worker.done.connect(self._on_file_crop_read_done)
            park_orphaned_worker(worker)
            worker.start()
        except Exception as e:  # noqa: BLE001



            self._release_crop_read()
            self._discard_pending_manual_click()
            self._report_crop_error(str(e), "crop_error_unknown", quiet)
            return False
        self._arm_encode_watchdog()
        return True

    def _show_inline_read_busy(self, show_busy: bool) -> None:






        if not show_busy:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        self._set_manual_encoding_note(True)
        try:
            from qgis.PyQt.QtCore import QEventLoop

            from ...core.qt_compat import resolve_qt_enum

            QApplication.processEvents(resolve_qt_enum(
                QEventLoop, "ProcessEventsFlag", "ExcludeUserInputEvents"))
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _hide_inline_read_busy(self, show_busy: bool) -> None:



        if not show_busy:
            return
        try:
            QApplication.restoreOverrideCursor()
        except Exception:  # nosec B110
            pass
        self._set_manual_encoding_note(False)

    def _on_file_crop_read_done(self, gen: int, result) -> None:




        self._ensure_manual_encode_state()


        if gen != self._encode_lock_gen:
            return
        read = self._crop_read
        quiet = bool(read.get("quiet")) if read is not None else False
        show_busy = bool(read.get("show_busy", True)) if read is not None else True
        on_encoded = read.get("on_encoded") if read is not None else None
        self._release_crop_read()



        if self.dock_widget is None or self.predictor is None:
            return
        if gen != self._manual_encode_gen:


            if self._pending_manual_click is not None:
                self._replay_pending_manual_click()
            return

        if self._deliver_crop_read(result, on_encoded, quiet=quiet,
                                   show_busy=show_busy):
            return







        self._queued_crop_request = None
        self._discard_pending_manual_click()

    def _deliver_crop_read(self, result, on_encoded, *, quiet: bool,
                           show_busy: bool) -> bool:




        image_np, crop_info, error, error_code = result
        if error or image_np is None:
            self._inflight_crop_window = None




            setter = getattr(self, "_set_ai_session_armed_line", None)
            if setter is not None:
                setter(loading=False)
            self._report_crop_error(error or "crop read failed",
                                    error_code or "crop_error_unknown", quiet)
            return False
        self._start_manual_encode(image_np, crop_info, on_encoded,
                                  show_busy=show_busy)
        return True











    def _begin_online_crop_fetch(self, center_point, mupp_override, on_encoded,
                                 *, show_busy: bool = True,
                                 quiet: bool = False) -> bool:











        from ...core.feature_encoder import OnlineCropFetcher
        from ...core.online_layer_twin import online_layer_twin

        read_layer = self._current_layer
        if quiet:
            read_layer = online_layer_twin(read_layer)
            if read_layer is None:
                return False
        actual_mupp = self._online_crop_mupp(mupp_override)
        if mupp_override is None:








            from ...core.crop_window import snap_center_to_grid
            cx, cy = snap_center_to_grid(
                center_point.x(), center_point.y(), actual_mupp, 1.0)
            center_point = QgsPointXY(cx, cy)


        self._inflight_crop_window = self._crop_window_key_for(
            center_point, actual_mupp)
        fetcher = OnlineCropFetcher(
            read_layer, center_point.x(), center_point.y(),
            actual_mupp, crop_size=1024)
        if fetcher.error is not None:
            self._surface_online_crop_error(
                fetcher.error, fetcher.error_code, center_point, quiet=quiet)
            return False
        try:
            fetcher.begin()
        except Exception as e:  # noqa: BLE001
            fetcher.restore()
            self._surface_online_crop_error(
                str(e), "crop_error_online_exception", center_point, quiet=quiet)
            return False

        self._ensure_manual_encode_state()
        self._manual_encode_gen += 1
        gen = self._manual_encode_gen
        self._online_fetch = {
            "fetcher": fetcher,
            "gen": gen,
            "on_encoded": on_encoded,
            "cursor": bool(show_busy),



            "worker": None,
            "cancel": threading.Event(),


            "quiet": bool(quiet),


            "center": center_point,
        }






        self._encoding_in_progress = True
        self._encode_cursor_set = bool(show_busy)
        self._encode_lock_gen = gen
        if show_busy:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            self._set_manual_encoding_note(True)
        self._arm_encode_watchdog()
        if show_busy:



            QApplication.processEvents()
        self._step_online_crop_fetch()
        return True

    def _step_online_crop_fetch(self) -> None:








        self._ensure_manual_encode_state()
        if self.dock_widget is None:


            self._release_online_fetch()
            return
        fetch = self._online_fetch
        if fetch is None or fetch.get("gen") != self._manual_encode_gen:
            return
        if self._start_direct_tile_fetch(fetch):
            return
        try:
            action, delay = fetch["fetcher"].step()
        except Exception as e:  # noqa: BLE001
            self._fail_online_fetch(str(e), "crop_error_online_exception")
            return
        self._route_online_fetch_action(action, delay)

    def _route_online_fetch_action(self, action, delay) -> None:




        from qgis.PyQt.QtCore import QTimer

        if action in ("stabilized", "exhausted"):
            self._complete_online_crop_fetch()
            return


        QTimer.singleShot(max(0, int(delay * 1000)), self._step_online_crop_fetch)

    def _start_direct_tile_fetch(self, fetch) -> bool:










        if fetch.get("worker") is not None:


            return True
        fetcher = fetch.get("fetcher")
        request = fetcher.direct_tile_request() if fetcher is not None else None
        if request is None:
            return False
        cancel = fetch.get("cancel")
        try:
            from .shared import park_orphaned_worker
            worker = DirectTileFetchWorker(
                request, fetch["gen"],
                cancel_check=None if cancel is None else cancel.is_set)
            worker.done.connect(self._on_direct_tile_fetch_done)
            park_orphaned_worker(worker)
            worker.start()
        except Exception as e:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Tile download stays on the main thread: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return False



        fetcher.take_direct_tile_request()
        fetch["worker"] = worker
        fetch["request"] = request
        return True

    def _on_direct_tile_fetch_done(self, gen: int, result) -> None:








        self._ensure_manual_encode_state()
        if self.dock_widget is None:

            self._release_online_fetch()
            return
        fetch = self._online_fetch
        if fetch is None or fetch.get("gen") != gen:
            return
        if gen != self._manual_encode_gen:
            return
        fetch["worker"] = None
        request = fetch.pop("request", None)
        try:
            action, delay = fetch["fetcher"].accept_direct_tiles(request, *result)
        except Exception as e:  # noqa: BLE001
            self._fail_online_fetch(str(e), "crop_error_online_exception")
            return
        self._route_online_fetch_action(action, delay)

    def _complete_online_crop_fetch(self) -> None:



        fetch = self._online_fetch
        if fetch is None:
            return
        fetcher = fetch["fetcher"]
        on_encoded = fetch.get("on_encoded")
        try:
            image_np, crop_info, error, error_code = fetcher.finish()
        except Exception as e:  # noqa: BLE001
            self._fail_online_fetch(str(e), "crop_error_online_exception")
            return
        finally:


            try:
                fetcher.restore()
            except Exception:  # nosec B110
                pass




        if self._online_fetch is not fetch:
            return
        if self.dock_widget is None or self.predictor is None:
            self._release_online_fetch(restore_provider=False)
            return
        if fetch.get("gen") != self._manual_encode_gen:
            self._release_online_fetch(restore_provider=False)
            if self._pending_manual_click is not None:
                self._replay_pending_manual_click()
            return
        if error:

            self._fail_online_fetch(error, error_code, restore_provider=False)
            return







        self._online_fetch = None
        show_busy = bool(fetch.get("cursor"))
        if show_busy:
            try:
                QApplication.restoreOverrideCursor()
            except Exception:  # nosec B110
                pass
        self._start_manual_encode(image_np, crop_info, on_encoded,
                                  show_busy=show_busy)

    def _fail_online_fetch(self, error, error_code,
                           restore_provider: bool = True) -> None:



        fetch = getattr(self, "_online_fetch", None) or {}
        center_point = fetch.get("center")
        quiet = bool(fetch.get("quiet"))
        self._release_online_fetch(restore_provider=restore_provider)
        self._surface_online_crop_error(error, error_code, center_point,
                                        quiet=quiet)



        self._queued_crop_request = None
        self._discard_pending_manual_click()

    def _surface_online_crop_error(self, error, error_code,
                                   center_point=None,
                                   quiet: bool = False) -> None:


















        if quiet:
            QgsMessageLog.logMessage(
                f"Prepared crop not read, the click will read its own: {error}",
                "AI Segmentation", level=Qgis.MessageLevel.Info
            )
            return
        QgsMessageLog.logMessage(
            f"Crop extraction failed: {error}",
            "AI Segmentation", level=Qgis.MessageLevel.Critical
        )
        if self._crop_error_went_to_panel(error_code, center_point):
            return



        source = ""
        try:
            if self._current_layer is not None:
                source = self._current_layer.source() or ""
        except (RuntimeError, AttributeError):
            source = ""
        report_key = (source, error_code or "crop_error_unknown")
        reported = getattr(self, "_crop_errors_reported", None)
        if reported is None:
            reported = set()
            self._crop_errors_reported = reported
        if report_key in reported:
            QgsMessageLog.logMessage(
                "Same crop error already reported this session; "
                "not showing the dialog again",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return
        reported.add(report_key)
        show_error_report(
            self.iface.mainWindow(),
            tr("Crop Error"),
            error,
            error_code=error_code or "crop_error_unknown",
        )
