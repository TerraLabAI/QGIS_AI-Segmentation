










from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsMessageLog,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QApplication

from ...core.i18n import tr
from ...core.interaction_dials import (
    encode_lock_ceiling_s,
    encode_watchdog_interval_ms,
)
from ..error_report_dialog import show_error_report






ENCODE_WATCHDOG_INTERVAL_MS = 5000
ENCODE_LOCK_CEILING_S = 240.0


class ManualCropsEncodeMixin:


    def _ensure_manual_encode_state(self) -> None:


        if not hasattr(self, "_manual_encode_gen"):
            self._manual_encode_gen = 0
            self._manual_encode_worker = None
            self._pending_encode = None
            self._pending_manual_click = None
        if not hasattr(self, "_online_fetch"):


            self._online_fetch = None
        if not hasattr(self, "_crop_read"):




            self._crop_read = None
        if not hasattr(self, "_queued_crop_request"):



            self._queued_crop_request = None
        if not hasattr(self, "_inflight_crop_window"):




            self._inflight_crop_window = None
            self._encoded_crop_window = None
        if not hasattr(self, "_encode_lock_gen"):







            self._encode_lock_gen = None

    def _set_manual_encoding_note(self, active: bool, phase: str = "imagery") -> None:








        dock = self.dock_widget
        if dock is None:
            return
        if active and getattr(self, "_refine_handoff_active", False):
            return
        try:
            dock.set_manual_encoding(bool(active), phase)
        except TypeError:

            dock.set_manual_encoding(bool(active))
        except (RuntimeError, AttributeError):
            pass

    def _wear_busy_cursor_for_crop(self) -> None:













        self._ensure_manual_encode_state()
        if not getattr(self, "_encoding_in_progress", False):
            return
        stage = self._crop_read or self._pending_encode or self._online_fetch
        if stage is None or stage.get("cursor"):
            return
        stage["cursor"] = True
        if "show_busy" in stage:
            stage["show_busy"] = True
        self._encode_cursor_set = True
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        self._set_manual_encoding_note(True)

    def _drain_queued_crop_request(self) -> bool:






        self._ensure_manual_encode_state()
        request = self._queued_crop_request
        self._queued_crop_request = None
        if request is None or self._encoding_in_progress:
            return False
        if self.dock_widget is None or self.predictor is None:
            return False
        return bool(self._extract_and_encode_crop(
            request["center"], request["mupp"],
            on_encoded=request["on_encoded"],
            show_busy=request["show_busy"], quiet=request["quiet"]))

    def _encode_crop_blocking(self, center_point, mupp_override) -> bool:











        if self.predictor is None:
            return False
        if self._encoding_in_progress:
            return False
        self._encoding_in_progress = True
        if not self._headless:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            self._set_manual_encoding_note(True)
            QApplication.processEvents()
        try:
            self._inflight_crop_window = self._crop_window_key_for(
                center_point, mupp_override)
            image_np, crop_info = self._extract_crop_only(center_point, mupp_override)
            if image_np is None:
                return False
            try:
                self.predictor.set_image(image_np)
            except Exception as e:
                return self._handle_encode_error(str(e))
            self._apply_encode_result_ok(crop_info)
            return True
        finally:



            self._encoding_in_progress = False
            if not self._headless:
                QApplication.restoreOverrideCursor()
                self._set_manual_encoding_note(False)

    def _handle_encode_error(self, err_str: str) -> bool:




        QgsMessageLog.logMessage(
            f"Image encoding failed: {err_str}",
            "AI Segmentation", level=Qgis.MessageLevel.Critical
        )



        from ...core.checkpoint_manager import (
            delete_checkpoint,
            is_corrupt_checkpoint_error,
        )
        if is_corrupt_checkpoint_error(err_str):
            return self._recover_corrupt_checkpoint(delete_checkpoint())



        from ...core.venv_manager import venv_needs_repair






        if venv_needs_repair(allow_subprocess_probe=False):
            return self._recover_broken_venv(err_str)
        if self._headless:
            self._headless_error = err_str
            return False
        show_error_report(
            self.iface.mainWindow(),
            tr("Encoding Error"),
            err_str,
            error_code="encoding_error",
        )
        return False

    def _apply_encode_result_ok(self, crop_info) -> None:



        self._current_crop_info = crop_info



        baseline = getattr(self, "_pending_crop_zoom_baseline", None)
        self._pending_crop_zoom_baseline = None
        if baseline is not None:
            scale_factor, canvas_mupp, actual_mupp = baseline
            self._current_crop_canvas_mupp = canvas_mupp
            if scale_factor is not None:
                self._current_crop_scale_factor = scale_factor
            if actual_mupp is not None:
                self._current_crop_actual_mupp = actual_mupp



        self._encoded_crop_window = getattr(self, "_inflight_crop_window", None)


        self._refine_min_area = self._compute_auto_min_area()
        self._safe_restore_canvas_focus()
        QgsMessageLog.logMessage(
            "Encoded crop: bounds={}, shape={}, auto_min_area={}".format(
                crop_info["bounds"], crop_info["img_shape"],
                self._refine_min_area),
            "AI Segmentation", level=Qgis.MessageLevel.Info
        )



    def _start_manual_encode(self, image_np, crop_info, on_encoded,
                             show_busy: bool = True) -> None:














        self._ensure_manual_encode_state()
        from ..background_workers import SetImageWorker
        from .shared import park_orphaned_worker

        self._manual_encode_gen += 1
        gen = self._manual_encode_gen
        self._pending_encode = {
            "crop_info": crop_info,
            "tail": on_encoded,


            "predictor": self.predictor,
            "gen": gen,
            "cursor": bool(show_busy),
        }



        self._encode_cursor_set = bool(show_busy)
        self._encoding_in_progress = True
        self._encode_lock_gen = gen



        if show_busy:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            self._set_manual_encoding_note(True, phase="encode")

        try:
            worker = SetImageWorker(self.predictor, image_np, gen)
            self._manual_encode_worker = worker
            worker.done.connect(self._on_manual_encode_done)
            park_orphaned_worker(worker)
            worker.start()
        except Exception as e:  # noqa: BLE001



            self._manual_encode_worker = None
            self._pending_encode = None
            self._encoding_in_progress = False
            self._encode_lock_gen = None
            if show_busy:
                try:
                    QApplication.restoreOverrideCursor()
                except Exception:  # nosec B110
                    pass
                self._set_manual_encoding_note(False)
            self._encode_cursor_set = True
            self._discard_pending_manual_click()
            self._handle_encode_error(str(e))
            return
        self._arm_encode_watchdog()

    def _on_manual_encode_done(self, gen: int, ok: bool, err: str) -> None:





        self._ensure_manual_encode_state()








        if gen != self._encode_lock_gen:
            return

        pending = self._pending_encode




        torn_down = self.dock_widget is None or self.predictor is None
        torn_down = torn_down or (pending is not None and self.predictor is not pending.get("predictor"))




        self._manual_encode_worker = None
        self._pending_encode = None
        self._encoding_in_progress = False
        self._encode_lock_gen = None
        cursor_was_set = (pending.get("cursor", True) if pending is not None
                          else getattr(self, "_encode_cursor_set", True))
        self._encode_cursor_set = True
        if cursor_was_set:
            try:
                QApplication.restoreOverrideCursor()
            except Exception:  # nosec B110
                pass
            self._set_manual_encoding_note(False)

        if torn_down:
            return

        if gen != self._manual_encode_gen:




            if self._pending_manual_click is not None:
                self._replay_pending_manual_click()
            return

        if not ok:


            self._inflight_crop_window = None
            self._handle_encode_error(err)
            self._discard_pending_manual_click()
            self._drain_queued_crop_request()
            return



        if pending is not None and pending.get("crop_info") is not None:
            self._apply_encode_result_ok(pending["crop_info"])


        setter = getattr(self, "_set_ai_session_armed_line", None)
        if setter is not None:
            setter(loading=False)
        tail = pending.get("tail") if pending is not None else None
        if tail is not None:
            try:
                tail()
            except Exception as e:  # noqa: BLE001
                QgsMessageLog.logMessage(
                    f"Manual encode continuation failed: {e}",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)



        if self._drain_queued_crop_request():
            return
        if self._pending_manual_click is not None:
            self._replay_pending_manual_click()

    def _release_crop_read(self) -> None:





        self._ensure_manual_encode_state()
        read = self._crop_read
        self._crop_read = None
        if read is None:
            return



        cancel = read.get("cancel")
        if cancel is not None:
            cancel.set()
        self._encoding_in_progress = False
        self._encode_lock_gen = None
        if read.get("cursor"):
            self._encode_cursor_set = False
            try:
                QApplication.restoreOverrideCursor()
            except Exception:  # nosec B110
                pass
            self._set_manual_encoding_note(False)



    def _remember_pending_manual_click(self, polarity: str, canvas_point) -> None:





        self._ensure_manual_encode_state()
        if self.map_tool:
            self.map_tool.remove_last_marker()
        self._pending_manual_click = {"polarity": polarity, "canvas_point": canvas_point}

    def _discard_pending_manual_click(self) -> None:


        self._ensure_manual_encode_state()
        self._pending_manual_click = None

    def _replay_pending_manual_click(self) -> None:






        self._ensure_manual_encode_state()
        pending = self._pending_manual_click
        if not pending:
            return
        self._pending_manual_click = None
        point = pending["canvas_point"]
        is_positive = pending["polarity"] == "positive"
        if self.map_tool:
            self.map_tool.add_marker(point, is_positive=is_positive)
        if is_positive:
            self._on_positive_click(point)
        else:
            self._on_negative_click(point)

    def _release_online_fetch(self, restore_provider: bool = True) -> None:













        self._ensure_manual_encode_state()
        fetch = self._online_fetch
        self._online_fetch = None
        if fetch is None:
            return
        cancel = fetch.get("cancel")
        if cancel is not None:
            cancel.set()
        worker = fetch.get("worker")
        if worker is not None:
            fetch["worker"] = None
            try:
                worker.done.disconnect()
            except (TypeError, RuntimeError):
                pass
        if restore_provider:
            try:
                fetch["fetcher"].restore()
            except Exception:  # nosec B110
                pass
        self._encoding_in_progress = False
        self._encode_lock_gen = None
        if fetch.get("cursor"):
            try:
                QApplication.restoreOverrideCursor()
            except Exception:  # nosec B110
                pass
            self._set_manual_encoding_note(False)

    def _invalidate_manual_encode(self) -> None:














        self._ensure_manual_encode_state()
        self._manual_encode_gen += 1
        self._pending_encode = None
        self._discard_pending_manual_click()
        self._release_online_fetch()
        self._release_crop_read()






        if (self._manual_encode_worker is not None and getattr(self, "_encode_cursor_set", False)):
            self._encode_cursor_set = False
            try:
                QApplication.restoreOverrideCursor()
            except Exception:  # nosec B110
                pass



        self._queued_crop_request = None
        self._inflight_crop_window = None
        self._encoded_crop_window = None



        self._set_manual_encoding_note(False)

    def _drop_inflight_crop_for_gesture(self) -> None:













        self._invalidate_manual_encode()
        self._current_crop_info = None
        self._encoded_crop_window = None


        self._speculative_manual_crop = False
        QgsMessageLog.logMessage(
            "Dropped the crop being read: a shape was committed instead",
            "AI Segmentation", level=Qgis.MessageLevel.Info)









    def _arm_encode_watchdog(self) -> None:


        import time
        self._encode_lock_since = time.monotonic()
        self._encode_watchdog_strikes = 0
        if getattr(self, "_encode_watchdog_armed", False):
            return
        self._encode_watchdog_armed = True
        self._schedule_encode_watchdog()

    def _schedule_encode_watchdog(self) -> None:


        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(encode_watchdog_interval_ms(ENCODE_WATCHDOG_INTERVAL_MS),
                          self._encode_watchdog_tick)

    def _encode_watchdog_tick(self) -> None:





        self._encode_watchdog_armed = False
        self._ensure_manual_encode_state()
        if not self._encoding_in_progress:
            return
        import time
        held_s = time.monotonic() - getattr(self, "_encode_lock_since", 0.0)
        worker = self._manual_encode_worker
        owner_alive = worker is not None and bool(worker.isRunning())
        owner_alive = owner_alive or self._online_fetch is not None



        owner_alive = owner_alive or self._crop_read is not None
        if owner_alive and held_s < encode_lock_ceiling_s(ENCODE_LOCK_CEILING_S):
            self._encode_watchdog_strikes = 0
        else:
            self._encode_watchdog_strikes = getattr(
                self, "_encode_watchdog_strikes", 0) + 1
            if self._encode_watchdog_strikes >= 2:
                self._force_release_encode_lock(
                    "held past ceiling" if owner_alive else "owner gone")
                return
        self._encode_watchdog_armed = True
        self._schedule_encode_watchdog()

    def _force_release_encode_lock(self, reason: str) -> None:






        QgsMessageLog.logMessage(
            f"Encode lock force-released by watchdog ({reason})",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
        self._ensure_manual_encode_state()
        self._manual_encode_gen += 1
        self._pending_encode = None
        self._manual_encode_worker = None
        self._discard_pending_manual_click()


        crop_owned = self._online_fetch is not None or self._crop_read is not None
        self._release_online_fetch()
        self._release_crop_read()
        self._encoding_in_progress = False
        self._encode_lock_gen = None
        if not crop_owned and getattr(self, "_encode_cursor_set", False):
            try:
                QApplication.restoreOverrideCursor()
            except Exception:  # nosec B110
                pass
        self._encode_cursor_set = False
        self._set_manual_encoding_note(False)

    def _handle_reencode(self, crop_status, raster_pt):











        if self._headless:
            return self._handle_reencode_sync(crop_status, raster_pt)
        return self._begin_async_reencode(crop_status, raster_pt)

    def _reencode_plan(self, crop_status, raster_pt, include_click_in_zoom):











        if crop_status == "no_crop":
            self.current_low_res_mask = None




            all_pts = [
                (p[0], p[1]) for p in
                self.prompts.positive_points + self.prompts.negative_points
            ]
            if (raster_pt.x(), raster_pt.y()) not in all_pts:
                all_pts.append((raster_pt.x(), raster_pt.y()))
            whole = self._untouched_shape_crop_window(raster_pt, all_pts)
            if whole is not None:
                center, mupp_or_scale = whole
            elif len(all_pts) > 1:
                center, mupp_or_scale = self._manual_crop_window_for_points(
                    all_pts)
            else:
                center, mupp_or_scale = self._grid_center_for_manual_click(
                    raster_pt, self._compute_initial_scale_factor())

            def _tail():
                self._invalidate_history_logits()

            return center, mupp_or_scale, _tail

        if crop_status == "outside_bounds":
            old_crop_info = self._current_crop_info
            old_history = list(self._mask_state_history)

            def _tail():
                self._freeze_active_crop(crop_info_override=old_crop_info)



                self.prompts.clear()
                self._active_crop_points_positive = []
                self._active_crop_points_negative = []

                self._mask_state_history = old_history
                self._invalidate_history_logits()
                self.current_mask = None
                self.current_low_res_mask = None



            whole = self._untouched_shape_crop_window(raster_pt)
            if whole is not None:
                return whole[0], whole[1], _tail
            center, scale = self._grid_center_for_manual_click(
                raster_pt, self._compute_initial_scale_factor())
            return center, scale, _tail


        old_crop_info = self._current_crop_info
        old_mask = self.current_mask
        all_pts = [
            (p[0], p[1]) for p in
            self.prompts.positive_points + self.prompts.negative_points
        ]
        if include_click_in_zoom and (raster_pt.x(), raster_pt.y()) not in all_pts:
            all_pts.append((raster_pt.x(), raster_pt.y()))
        whole = self._untouched_shape_crop_window(raster_pt, all_pts)
        if whole is not None:
            new_center, mupp_or_scale = whole
        elif len(all_pts) > 1:
            new_center, mupp_or_scale = self._manual_crop_window_for_points(
                all_pts)
        else:
            new_center, mupp_or_scale = self._grid_center_for_manual_click(
                raster_pt, self._compute_initial_scale_factor())

        def _tail():
            self.current_low_res_mask = None







            self.current_mask = None
            self._invalidate_history_logits()

            if old_mask is not None and old_crop_info is not None:
                transferred = self._build_mask_input_from_previous(
                    old_mask.astype(float),
                    old_crop_info["bounds"],
                    old_crop_info["img_shape"],
                    self._current_crop_info["bounds"],
                    self._current_crop_info["img_shape"],
                )
                if transferred is not None:
                    self.current_low_res_mask = transferred

        return new_center, mupp_or_scale, _tail

    def _handle_reencode_sync(self, crop_status, raster_pt):




        center, mupp_or_scale, tail = self._reencode_plan(
            crop_status, raster_pt, include_click_in_zoom=False)
        if not self._extract_and_encode_crop(center, mupp_override=mupp_or_scale):
            return False
        tail()
        return True

    def _begin_async_reencode(self, crop_status, raster_pt):








        center, mupp_or_scale, tail = self._reencode_plan(
            crop_status, raster_pt, include_click_in_zoom=True)
        return self._extract_and_encode_crop(
            center, mupp_override=mupp_or_scale, on_encoded=tail)
