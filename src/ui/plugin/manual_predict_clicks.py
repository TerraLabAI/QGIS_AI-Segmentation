






from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog, QgsPointXY

from ...core.error_policy import LINK_FAILURE_SHIPPED
from ...core.i18n import tr
from ...core.telemetry_errors import slot_guard
from ..error_report_dialog import show_error_report



QUIET_CLICK_SUPERSEDED = "superseded"

QUIET_CLICK_REREAD = "reread"

QUIET_CLICK_REFUSED = "refused"

QUIET_CLICK_OFFLINE = "offline"










def _click_offline_codes() -> frozenset:

    try:
        from ...core.server_dials import dial_list
        return dial_list("tuning.click.offline_codes_extra", LINK_FAILURE_SHIPPED, normalize=str.upper)
    except Exception:  # noqa: BLE001
        return LINK_FAILURE_SHIPPED


def _click_was_superseded(err: Exception) -> bool:







    try:
        from ...core.cloud_sam_predictor import RefineSupersededError

        return isinstance(err, RefineSupersededError)
    except Exception:  # noqa: BLE001
        return False






MASK_UNDO_DEPTH = 30


def _click_refusal_answer(err: Exception) -> str:






    try:
        from ...core.cloud_sam_predictor import REFUSAL_OTHER, RefineRefusedError

        if not isinstance(err, RefineRefusedError):
            return ""
        answer = err.refusal_class()
        return "" if answer == REFUSAL_OTHER else str(answer)
    except Exception:  # noqa: BLE001
        return ""


def _click_connectivity_code(err: Exception) -> str:

















    offline_codes = _click_offline_codes()
    code = str(getattr(err, "code", "") or "").strip().upper()
    if code == "TIMEOUT":
        try:
            from ...core.cloud_sam_predictor import unreached_error

            return code if unreached_error(err) else ""
        except Exception:  # noqa: BLE001
            return ""
    if code:
        return code if code in offline_codes else ""
    text = str(err).upper()
    for known in offline_codes:
        if known in text:
            return known
    return ""


def _click_hit_cold_start(err: Exception) -> bool:



    try:
        from ...core import cloud_warming_state
        from ...core.cloud_sam_predictor import WARMING_CODE

        code = str(getattr(err, "code", "") or "").strip().upper()
        if code == WARMING_CODE:
            return True
        if not cloud_warming_state.recently_warming():
            return False
        return code == "TIMEOUT" or "TIMEOUT" in str(err).upper()
    except Exception:  # noqa: BLE001
        return False


class ManualClickMixin:










    _HOVER_REUSE_NEAR_PX = 32

    def _start_click_clock(self) -> None:


        from ...core.click_phase_clock import ClickPhaseClock
        carried = getattr(self, "_replayed_click_clock", None)
        self._replayed_click_clock = None
        if carried is not None:
            carried.note_crop_ready(getattr(self, "_replayed_crop_encode_s", None))
            self._click_clock_in_hand = carried
        else:
            self._click_clock_in_hand = ClickPhaseClock()

    def _report_click_without_model(self) -> None:






        if getattr(self, "_click_without_model_reported", False):
            return
        if getattr(self, "_headless", False):
            return
        self._click_without_model_reported = True
        try:
            if getattr(self, "_local_ai_load_failed", False):
                line = tr("The AI did not load, so this click was not "
                          "answered. Use the Install button in the panel to "
                          "set it up again.")
            else:
                line = tr("The AI is still loading, so this click was not "
                          "answered. Try again in a few seconds.")
            from ...core.server_dials import dial_in_range

            duration = dial_in_range(
                "tuning.click.no_model_notice_s", 6, 4, 10)
            self.iface.messageBar().pushMessage(
                "AI Segmentation", line,
                level=Qgis.MessageLevel.Warning, duration=duration)
        except (RuntimeError, AttributeError):
            pass

    def _say_click_outside_raster(self) -> None:




        if self.map_tool:
            self.map_tool.remove_last_marker()



        layer_name = ""
        try:
            if self._current_layer is not None:
                layer_name = self._current_layer.name()
        except RuntimeError:
            layer_name = ""
        try:
            from ...core.server_dials import dial_in_range

            duration = dial_in_range(
                "tuning.click.outside_raster_notice_s", 8, 4, 10)
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Click is outside the '{layer}' raster. To segment another raster, stop the current segmentation first.").format(layer=layer_name),  # noqa: E501
                level=Qgis.MessageLevel.Warning,
                duration=duration
            )
        except (RuntimeError, AttributeError):
            pass

    def _refine_click_is_stale(self) -> bool:









        if getattr(self, "_auto_review", None) is None:
            return False
        busy = getattr(self, "_refine_handoff_active", False)
        busy = busy or getattr(self, "_refine_add_mode_active", False)
        return not (busy or getattr(self, "_is_refining_saved_object", False))

    def _drop_stale_refine_click(self) -> None:

        if self.map_tool:
            self.map_tool.remove_last_marker()
        self._sweep_stale_refine_canvas()

    @slot_guard(stage="segment", user_message=tr(
        "That click could not be handled. Please try again."))
    def _flash_save_first_if_outside(self, point) -> None:




        if self._refine_handoff_active or self.current_mask is None:
            return
        band = getattr(self, "mask_rubber_band", None)
        if band is None or self.map_tool is None:
            return
        try:
            from qgis.core import QgsGeometry

            shape = band.asGeometry()
            if shape is None or shape.isEmpty():
                return
            if shape.contains(QgsGeometry.fromPointXY(point)):
                return
        except RuntimeError:
            return
        self.map_tool.flash_hint(
            tr("Press S to save first, then click the next object"))

    def _on_positive_click(self, point):














        self._start_click_clock()
        held_hover_answer = self._take_hover_preview_answer()
        self._hover_click_answer = None
        self._stop_hover_preview("click")
        if self._refine_click_is_stale():
            self._drop_stale_refine_click()
            return
        self._flash_save_first_if_outside(point)
        if self.predictor is None:
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self._report_click_without_model()
            return


        raster_pt = self._transform_to_raster_crs(point)

        if not self._is_point_in_raster_extent(raster_pt):
            self._say_click_outside_raster()
            return












        is_resting_click = self._refine_handoff_active
        is_resting_click = is_resting_click and not self._is_refining_saved_object
        is_resting_click = is_resting_click and self.current_mask is None
        is_resting_click = is_resting_click and not self._active_crop_points_positive



        is_resting_click = is_resting_click and not getattr(
            self, "_refine_add_mode_active", False)
        if is_resting_click:
            if self.map_tool:
                self.map_tool.remove_last_marker()
            idx = self._hit_test_saved_polygon(raster_pt)
            if idx is not None:
                self._select_saved_polygon(
                    idx, additive=self._click_was_additive())
            else:
                self._deselect_saved_polygons()
            return

        if self._defer_cloud_click_for_config("positive", point):
            return












        if self._encoding_in_progress and not self._abandon_speculative_manual_crop(raster_pt):
            self._remember_pending_manual_click("positive", point)
            self._wear_busy_cursor_for_crop()
            return








        self._wider_crop_fresh_click()



        crop_status = self._check_crop_status(raster_pt)

        if crop_status != "ok":




            self._remember_pending_manual_click("positive", point)
            if not self._begin_async_reencode(crop_status, raster_pt):


                self._discard_pending_manual_click()
            return








        self._hover_click_answer = held_hover_answer






        from ...core.server_dials import dial_in_range

        mask_undo_depth = dial_in_range(
            "tuning.click.mask_undo_depth", MASK_UNDO_DEPTH, 5, 100)
        if len(self._mask_state_history) >= mask_undo_depth:
            self._mask_state_history.pop(0)
        self._mask_state_history.append(self._snapshot_mask_state())

        self.prompts.add_positive_point(raster_pt.x(), raster_pt.y())
        self._active_crop_points_positive.append((raster_pt.x(), raster_pt.y()))




        self._last_click_point = (raster_pt.x(), raster_pt.y())
        self._last_click_polarity = "positive"






        try:
            predicted = self._run_prediction()
        except Exception:



            from .manual_measure_cache import rollback_click_quietly
            rollback_click_quietly(self, "positive", point)
            self._wider_crop_replay_failed()
            raise
        finally:
            self._end_click_wait_started_here()
        if not predicted:
            self._rollback_failed_click("positive", point)
            return





        undo_note = None
        if self._last_prediction_found_nothing():
            undo_note = tr("No object found here. Try clicking somewhere else.")
        elif self._last_click_stood_clear_of_shape():
            undo_note = tr("That area does not touch the object you are editing, so nothing was added. Reshaping works on one object at a time.")  # noqa: E501
        if undo_note and self._mask_state_history:
            self.prompts.undo()
            if self._active_crop_points_positive:
                self._active_crop_points_positive.pop()
            self._restore_mask_state(self._mask_state_history.pop())
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self._update_ui_after_prediction()
            from ...core.server_dials import dial_in_range

            duration = dial_in_range(
                "tuning.click.auto_revert_notice_s", 5, 4, 10)
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                undo_note,
                level=Qgis.MessageLevel.Info,
                duration=duration
            )
            return

    @slot_guard(stage="segment", user_message=tr(
        "That click could not be handled. Please try again."))
    def _on_negative_click(self, point):





        self._start_click_clock()
        self._stop_hover_preview("click")
        if self._refine_click_is_stale():
            self._drop_stale_refine_click()
            return
        if self.predictor is None:
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self._report_click_without_model()
            return







        is_resting_click = self._refine_handoff_active
        is_resting_click = is_resting_click and not self._is_refining_saved_object
        is_resting_click = is_resting_click and self.current_mask is None
        is_resting_click = is_resting_click and not self._active_crop_points_positive



        is_resting_click = is_resting_click and not getattr(
            self, "_refine_add_mode_active", False)
        if is_resting_click:
            raster_pt0 = self._transform_to_raster_crs(point)
            if self._is_point_in_raster_extent(raster_pt0):
                if self.map_tool:
                    self.map_tool.remove_last_marker()
                idx = self._hit_test_saved_polygon(raster_pt0)
                if idx is not None:
                    self._select_saved_polygon(
                        idx, additive=self._click_was_additive())
                else:
                    self._deselect_saved_polygons()
                return

        if self._defer_cloud_click_for_config("negative", point):
            return




        raster_pt = self._transform_to_raster_crs(point)
        if not self._is_point_in_raster_extent(raster_pt):
            self._say_click_outside_raster()
            return





        if self._encoding_in_progress and not self._abandon_speculative_manual_crop(raster_pt):
            self._remember_pending_manual_click("negative", point)
            self._wear_busy_cursor_for_crop()
            return





        refine_edit = self._refine_edit_session_active()


        if not refine_edit and len(self.prompts.positive_points) == 0:
            if self.map_tool:
                self.map_tool.remove_last_marker()
            QgsMessageLog.logMessage(
                "Negative point ignored - need at least one positive point first",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )



            try:
                from ...core.server_dials import dial_in_range

                duration = dial_in_range(
                    "tuning.click.negative_outside_notice_s", 4, 4, 10)
                self.iface.messageBar().pushMessage(
                    "AI Segmentation", tr("Left-click to select"),
                    level=Qgis.MessageLevel.Info, duration=duration)
            except (RuntimeError, AttributeError):
                pass
            return

        crop_status = self._check_crop_status(raster_pt)





        if crop_status == "outside_bounds" and not refine_edit:
            if self.map_tool:
                self.map_tool.remove_last_marker()
            from ...core.server_dials import dial_in_range

            duration = dial_in_range(
                "tuning.click.negative_outside_notice_s", 4, 4, 10)
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Right-click must be inside the current selection area."),
                level=Qgis.MessageLevel.Info,
                duration=duration
            )
            return

        if crop_status != "ok":




            self._remember_pending_manual_click("negative", point)
            if not self._begin_async_reencode(crop_status, raster_pt):

                self._discard_pending_manual_click()
            return



        from ...core.server_dials import dial_in_range

        mask_undo_depth = dial_in_range(
            "tuning.click.mask_undo_depth", MASK_UNDO_DEPTH, 5, 100)
        if len(self._mask_state_history) >= mask_undo_depth:
            self._mask_state_history.pop(0)
        self._mask_state_history.append(self._snapshot_mask_state())

        self.prompts.add_negative_point(raster_pt.x(), raster_pt.y())
        self._active_crop_points_negative.append((raster_pt.x(), raster_pt.y()))




        self._last_click_point = (raster_pt.x(), raster_pt.y())
        self._last_click_polarity = "negative"




        try:
            predicted = self._run_prediction()
        except Exception:


            from .manual_measure_cache import rollback_click_quietly
            rollback_click_quietly(self, "negative", point)
            raise
        finally:
            self._end_click_wait_started_here()
        if not predicted:
            self._rollback_failed_click("negative", point)
            return




        if self._last_prediction_found_nothing() and self._mask_state_history:
            self.prompts.undo()
            if self._active_crop_points_negative:
                self._active_crop_points_negative.pop()
            self._restore_mask_state(self._mask_state_history.pop())
            if self.map_tool:
                self.map_tool.remove_last_marker()
            self._update_ui_after_prediction()
            from ...core.server_dials import dial_in_range

            duration = dial_in_range(
                "tuning.click.negative_auto_revert_notice_s", 4, 4, 10)



            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("That would remove the whole selection, so it was undone."),
                level=Qgis.MessageLevel.Info,
                duration=duration
            )
            return

    def _run_prediction(self) -> bool:










        import numpy as np

        from ...core.click_phase_clock import activate_click_clock, click_clock_now






        try:
            from rasterio import transform as rio_transform
            from rasterio.transform import from_bounds as transform_from_bounds
        except ImportError:
            rio_transform = None
            transform_from_bounds = None




        self._forget_alt_outlines()

        active_pos = self._active_crop_points_positive
        active_neg = self._active_crop_points_negative
        all_active = active_pos + active_neg
        if not all_active:
            return False

        if self._current_crop_info is None:
            QgsMessageLog.logMessage(
                "No crop encoded yet - cannot predict",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )
            return False













        if not self.predictor.is_image_set:
            QgsMessageLog.logMessage(
                "Worker has no encoded image - re-encoding current crop",
                "AI Segmentation", level=Qgis.MessageLevel.Warning
            )
            b = self._current_crop_info["bounds"]
            center = QgsPointXY((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
            override = (self._current_crop_actual_mupp if self._is_online_layer
                        else self._current_crop_scale_factor)
            if not self._headless:
                if self._extract_and_encode_crop(center, mupp_override=override):
                    self._end_click_quietly(QUIET_CLICK_REREAD)
                return False
            if not self._encode_crop_blocking(center, mupp_override=override):
                return False

        crop_bounds = self._current_crop_info["bounds"]
        img_shape = self._current_crop_info["img_shape"]
        img_height, img_width = img_shape

        minx, miny, maxx, maxy = crop_bounds



        if maxx <= minx or maxy <= miny or img_width <= 0 or img_height <= 0:
            QgsMessageLog.logMessage(
                "Crop window has no size - cannot place the click in it",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            return False
        img_clip_transform = None
        if rio_transform is not None:




            try:
                img_clip_transform = transform_from_bounds(
                    minx, miny, maxx, maxy, img_width, img_height)
                rio_transform.rowcol(img_clip_transform, minx, maxy)
            except Exception:  # noqa: BLE001
                img_clip_transform = None
        if img_clip_transform is not None:

            def crop_pixel_of(px, py):



                row, col = rio_transform.rowcol(img_clip_transform, px, py)
                return (min(max(int(row), 0), img_height - 1),
                        min(max(int(col), 0), img_width - 1))
        else:
            from ...core.crop_window import crop_pixel_of_point

            def crop_pixel_of(px, py):
                return crop_pixel_of_point(crop_bounds, img_shape, px, py)





        point_coords_list = []
        point_labels_list = []
        for points, label in ((active_pos, 1), (active_neg, 0)):
            for x, y in points:
                pixel = crop_pixel_of(x, y)
                if pixel is None:
                    QgsMessageLog.logMessage(
                        "Crop window has no size - cannot place the click in it",
                        "AI Segmentation", level=Qgis.MessageLevel.Warning)
                    return False
                row, col = pixel
                point_coords_list.append([col, row])
                point_labels_list.append(label)

        point_coords = np.array(point_coords_list)
        point_labels = np.array(point_labels_list)



        mask_input = None
        if self.current_low_res_mask is not None:
            mask_input = self.current_low_res_mask
        elif self._is_refining_saved_object or self._frozen_sessions:









            mask_input = self._refine_polygon_mask_input()




        prev_mask_for_merge = self.current_mask if mask_input is not None else None
        if (getattr(self, "_wider_crop_state", None) is not None
                and self.current_mask is not None and self.current_transform_info is not None):





            on_grid = self._shape_on_wider_grid()
            if on_grid is not None:
                prev_mask_for_merge = on_grid
        if prev_mask_for_merge is None and mask_input is not None:





            try:
                shape = self._shape_in_progress_geometry()
                if shape is not None:
                    prev_mask_for_merge = self._rasterize_geom_to_crop(
                        shape, crop_bounds, img_shape)
            except Exception:  # noqa: BLE001  # nosec B110
                prev_mask_for_merge = None
        elif prev_mask_for_merge is not None and self._frozen_sessions:



            try:
                shape = self._shape_in_progress_geometry()
                frozen_here = (None if shape is None else
                               self._rasterize_geom_to_crop(
                                   shape, crop_bounds, img_shape))
                if frozen_here is not None:
                    prev_mask_for_merge = np.logical_or(
                        prev_mask_for_merge[:img_height, :img_width].astype(bool),
                        frozen_here[:img_height, :img_width])
            except Exception:  # noqa: BLE001  # nosec B110
                pass




        one_positive = len(active_pos) == 1
        no_negatives = len(active_neg) == 0
        is_first_point = one_positive and no_negatives and mask_input is None
        use_multimask = is_first_point





        held_hover = getattr(self, "_hover_click_answer", None)
        self._hover_click_answer = None
        reused = (self._reused_hover_answer(held_hover, crop_bounds, img_shape,
                                            point_coords_list)
                  if held_hover is not None and use_multimask else None)



        import time as _click_clock
        self._manual_click_fell_back = False
        click_started_at = _click_clock.monotonic()
        clock = getattr(self, "_click_clock_in_hand", None)







        self._click_wait_started_here = (
            self._begin_click_wait() if reused is None else False)

        try:
            if reused is not None:
                masks, scores, low_res_masks = reused





                self._note_manual_cloud_answer()
                try:
                    self.predictor.last_answer_was_remote = True
                except (RuntimeError, AttributeError):
                    pass  # nosec B110
            else:


                if clock is not None:
                    clock.predict_started_at = click_clock_now()
                activate_click_clock(clock)
                from ...core import cloud_warming_state
                cloud_warming_state.set_wait_listener(self._show_cloud_warming_wait)
                try:
                    masks, scores, low_res_masks = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        mask_input=mask_input,
                        multimask_output=use_multimask,
                    )
                finally:
                    cloud_warming_state.set_wait_listener(None)
                    self._show_cloud_warming_wait(None)
                    activate_click_clock(None)
                    if clock is not None:
                        clock.answered_at = click_clock_now()
        except RuntimeError as e:
            if _click_was_superseded(e):



                QgsMessageLog.logMessage(
                    "Click dropped: the crop changed while its answer was on "
                    "the way", "AI Segmentation", level=Qgis.MessageLevel.Info)
                self._end_click_quietly(QUIET_CLICK_SUPERSEDED)
                return False
            error_str = str(e)
            if _click_hit_cold_start(e) and not self._headless:



                QgsMessageLog.logMessage(
                    f"Click ended while the service was starting: {error_str}",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                try:
                    self.iface.messageBar().pushWarning(
                        "AI Segmentation",
                        tr("The AI is still waking up. Click again in a few "
                           "seconds."))
                except (RuntimeError, AttributeError):
                    pass
                self._end_click_quietly(QUIET_CLICK_OFFLINE)
                return False
            refusal = _click_refusal_answer(e)
            if refusal and not self._headless:





                QgsMessageLog.logMessage(
                    f"Click refused ({refusal})", "AI Segmentation",
                    level=Qgis.MessageLevel.Warning)
                try:
                    from ...core import telemetry_errors


                    telemetry_errors.track_plugin_error(
                        stage="segment",
                        error_code=("predict_empty_result"
                                    if refusal == "EMPTY" else "predict_refused"),
                        message=error_str)
                except Exception:
                    pass  # nosec B110
                try:
                    if refusal == "SIGN_IN":
                        line = tr("Session expired. Sign in again to continue.")
                    elif refusal == "EMPTY":
                        line = tr("That click selected nothing. Move the "
                                  "points and click again.")
                    else:
                        line = tr("You saved your cloud objects for this "
                                  "month. Switch to your own computer to keep "
                                  "working free, or upgrade from the panel.")


                    self.iface.messageBar().pushWarning(
                        "AI Segmentation", line)
                except (RuntimeError, AttributeError):
                    pass
                self._end_click_quietly(QUIET_CLICK_REFUSED)
                return False
            offline_code = _click_connectivity_code(e)
            if offline_code and not self._headless:






                QgsMessageLog.logMessage(
                    f"Click could not reach the service ({offline_code}): {error_str}",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                try:
                    from ...core import telemetry_errors



                    telemetry_errors.track_plugin_error(
                        stage="segment", error_code="predict_no_connection",
                        message=error_str)
                except Exception:
                    pass  # nosec B110



                if self._manual_cloud_predictor_active():
                    offline_line = tr(
                        "TerraLab could not be reached. Check your internet "
                        "connection, or stop and pick My computer to work "
                        "offline.")
                else:
                    offline_line = tr(
                        "Network error. Check your internet connection.")
                try:
                    self.iface.messageBar().pushWarning(
                        "AI Segmentation", offline_line)
                except (RuntimeError, AttributeError):
                    pass
                self._end_click_quietly(QUIET_CLICK_OFFLINE)
                return False
            if (str(getattr(e, "code", "") or "").strip().upper() == "TIMEOUT"
                    and not self._headless):



                QgsMessageLog.logMessage(
                    f"Click timed out on a slow link: {error_str}",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)
                try:
                    from ...core import telemetry_errors
                    telemetry_errors.track_plugin_error(
                        stage="segment", error_code="predict_no_connection",
                        message=error_str)
                except Exception:
                    pass  # nosec B110
                try:
                    self.iface.messageBar().pushWarning(
                        "AI Segmentation",
                        tr("Your connection is slow or was lost. Click again "
                           "in a moment."))
                except (RuntimeError, AttributeError):
                    pass
                self._end_click_quietly(QUIET_CLICK_OFFLINE)
                return False
            QgsMessageLog.logMessage(
                f"Prediction failed: {error_str}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            self._track_manual_run_failed()




            is_dll_error = "DLL" in error_str or "Visual C++" in error_str
            if is_dll_error:
                code = "predict_dll_error"
            elif "subprocess" in error_str.lower() or "rpc" in error_str.lower():
                code = "predict_worker_died"
            else:
                code = "predict_runtime_error"
            try:
                from ...core import telemetry_errors



                if not (is_dll_error and not self._headless):
                    telemetry_errors.track_plugin_error(
                        stage="segment", error_code=code, message=error_str)
            except Exception:
                pass  # nosec B110
            if self._headless:
                self._headless_error = error_str
                return False



            if self._degrade_correct_ai_to_manual(error_str):
                return False
            if is_dll_error:
                show_error_report(
                    self.iface.mainWindow(),
                    tr("Segmentation failed"),
                    error_str,
                    error_code="predict_dll_error",
                )
            else:




                show_error_report(
                    self.iface.mainWindow(),
                    tr("Segmentation failed"),
                    error_str,
                    error_code=code,
                    track=False,
                )
            return False
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Unexpected prediction error: {str(e)}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            self._track_manual_run_failed()
            try:
                from ...core import telemetry_errors




                telemetry_errors.track_plugin_error(
                    stage="segment",
                    error_code="predict_unexpected_error",
                    message=f"{type(e).__name__}: {e}",
                    module="manual_predict_clicks")
            except Exception:
                pass  # nosec B110
            if not self._headless and not self._degrade_correct_ai_to_manual(str(e)):
                from ...core.server_dials import dial_in_range

                duration = dial_in_range(
                    "tuning.click.segment_failed_notice_s", 5, 4, 10)
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("Segmentation failed. Please try again."),
                    level=Qgis.MessageLevel.Warning,
                    duration=duration,
                )
            return False



        predict_ms = int((_click_clock.monotonic() - click_started_at) * 1000)

        multimask_picked = None
        if use_multimask:
            total_pixels = masks[0].shape[0] * masks[0].shape[1]
            mask_areas = [int(np.count_nonzero(m)) for m in masks]
            if reused is not None:


                best_idx = 0
            else:
                from ...core.multimask_pick import pick_multimask_index
                best_idx = pick_multimask_index(mask_areas, scores, total_pixels)

            QgsMessageLog.logMessage(
                f"Multimask: areas={mask_areas}, scores={[round(float(s), 3) for s in scores]}, picked={best_idx}",
                "AI Segmentation", level=Qgis.MessageLevel.Info
            )
            self.current_mask = masks[best_idx]
            self.current_score = float(scores[best_idx])
            self.current_low_res_mask = low_res_masks[best_idx:best_idx + 1]
            multimask_picked = best_idx
        else:
            self.current_mask = masks[0]
            self.current_score = float(scores[0])
            self.current_low_res_mask = low_res_masks






        self.current_mask = np.array(
            self.current_mask[:img_height, :img_width], copy=True)



        if use_multimask and self._wider_crop_before_first_answer(
                self.current_mask, is_first_point):
            self._end_click_quietly(QUIET_CLICK_REREAD)
            return False




        raw_answer = self.current_mask



        click_rc = None
        try:
            if getattr(self, "_last_click_point", None) is not None:
                cx, cy = self._last_click_point
                crow, ccol = crop_pixel_of(cx, cy)
                click_rc = (int(crow), int(ccol))
        except Exception:  # noqa: BLE001  # nosec B110
            click_rc = None






        try:
            from ...core.detection_policy import progressive_merge_enabled


            may_merge = (prev_mask_for_merge is not None and progressive_merge_enabled()
                         and not self._wider_crop_replay_in_hand())
            if may_merge and click_rc is not None:
                from ...core.progressive_merge import progressive_merge_masks


                prev_c = prev_mask_for_merge[:img_height, :img_width]
                self.current_mask = progressive_merge_masks(
                    prev_c, self.current_mask, click_rc[0], click_rc[1])
        except Exception:  # noqa: BLE001  # nosec B110
            pass








        self._last_prediction_empty = not bool(self.current_mask.any())
        self._last_click_stood_clear = False
        if getattr(self, "_last_click_polarity", "positive") == "positive":
            if self._is_refining_saved_object:






                px_size = (maxx - minx) / float(img_width) if img_width else 0.0
                (self.current_mask, self._last_click_stood_clear) = \
                    self._grown_in_one_piece(
                        self.current_mask, prev_mask_for_merge,
                        img_height, img_width, px_size, raw_answer)
            else:
                self.current_mask = self._grown_by_shape_so_far(
                    self.current_mask, prev_mask_for_merge, img_height, img_width)
        elif prev_mask_for_merge is not None and click_rc is not None:






            try:
                from ...core.progressive_merge import subtract_click_region
                prev_c = prev_mask_for_merge[:img_height, :img_width]
                keeps = [(int(row), int(col)) for (col, row), label
                         in zip(point_coords_list, point_labels_list) if label == 1]
                if not self._reask_collapsed_trim(
                        prev_c, raw_answer, point_coords, point_labels,
                        click_rc, keeps, img_height, img_width):



                    self._end_click_quietly(QUIET_CLICK_SUPERSEDED)
                    return False
                self.current_mask = subtract_click_region(
                    prev_c, self.current_mask, click_rc[0], click_rc[1],
                    keep_points=keeps)
            except Exception:  # noqa: BLE001  # nosec B110
                pass



        if (not use_multimask
                and getattr(self, "_last_click_polarity", "positive") == "positive"
                and self._wider_crop_before_later_answer(
                    self.current_mask, raw_answer, len(all_active))):
            self._end_click_quietly(QUIET_CLICK_REREAD)
            return False




        self._freeze_display_polygon_outside_crop(crop_bounds)
        self._unfrozen_display_polygon = None


        crs_value = None
        try:
            if self._current_layer and self._current_layer.crs().isValid():
                layer_crs = self._current_layer.crs()



                crs_value = layer_crs.authid() or layer_crs.toWkt()
        except RuntimeError:
            pass

        self.current_transform_info = {
            "bbox": (minx, maxx, miny, maxy),
            "img_shape": (img_height, img_width),
            "crs": crs_value,
        }

        if reused is not None:


            self._adopt_ghost_shape(held_hover[3], getattr(self, "_hover_click_shape", None))
        self._hover_click_shape = None



        if (multimask_picked is not None and prev_mask_for_merge is None
                and not self._is_refining_saved_object and len(scores) > 1):
            self._keep_alt_outlines(masks, scores, low_res_masks,
                                    multimask_picked, img_height, img_width)
        elif (multimask_picked is not None and prev_mask_for_merge is None
                and not self._is_refining_saved_object and len(scores) == 1
                and len(point_coords_list) == 1
                and getattr(self.predictor, "last_answer_was_remote", False)):

            self._keep_remote_alt_outline_seed(
                point_coords_list[0][0], point_coords_list[0][1],
                img_height, img_width)



        self._score_goes_in_click_line = clock is not None
        try:
            self._update_ui_after_prediction()
        finally:
            self._score_goes_in_click_line = False
        if clock is not None:
            clock.drawn_at = click_clock_now()
        self._track_manual_click_answered(predict_ms, clock)
        return True

    def _reask_collapsed_trim(self, prev_mask, raw_answer, point_coords,
                              point_labels, click_rc, keeps,
                              img_height: int, img_width: int) -> bool:












        import numpy as np

        from ...core.progressive_merge import pick_trim_candidate, trim_answer_collapsed
        if not trim_answer_collapsed(prev_mask, raw_answer, keeps, click_rc):
            return True
        predictor = self.predictor
        encode_gen = getattr(self, "_manual_encode_gen", 0)

        def session_moved() -> bool:
            return (self.predictor is not predictor
                    or getattr(self, "_manual_encode_gen", 0) != encode_gen)

        try:
            masks, scores, low_res = predictor.predict(
                point_coords=point_coords, point_labels=point_labels,
                mask_input=None, multimask_output=True)
        except Exception as err:  # noqa: BLE001
            if _click_was_superseded(err) or session_moved():
                QgsMessageLog.logMessage(
                    "Click dropped: the session changed during the trim re-ask",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
                return False
            QgsMessageLog.logMessage(
                f"Trim re-ask skipped: {type(err).__name__}",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
            return True
        if session_moved():
            QgsMessageLog.logMessage(
                "Click dropped: the session changed during the trim re-ask",
                "AI Segmentation", level=Qgis.MessageLevel.Info)
            return False
        cands = [m[:img_height, :img_width] for m in masks]
        idx = pick_trim_candidate(prev_mask, cands, click_rc[0], click_rc[1], keeps)
        if idx is None:
            return True
        first_kept = int(np.count_nonzero(
            (np.asarray(prev_mask) > 0) & (np.asarray(raw_answer) > 0)))
        kept = int(np.count_nonzero(
            (np.asarray(prev_mask) > 0) & (np.asarray(cands[idx]) > 0)))
        if kept <= first_kept:
            return True
        QgsMessageLog.logMessage(
            f"Trim re-ask: kept {kept} px of the shape instead of {first_kept}",
            "AI Segmentation", level=Qgis.MessageLevel.Info)
        self.current_mask = np.array(cands[idx], copy=True)
        self.current_score = float(scores[idx])
        self.current_low_res_mask = low_res[idx:idx + 1]
        return True

    def _reused_hover_answer(self, held, crop_bounds, img_shape, points):


















        import numpy as np

        try:
            bounds, shape, asked, mask, score, logits, drawn = held
        except (TypeError, ValueError):
            return None
        if logits is None or len(points) != 1:
            return None
        if tuple(bounds) != tuple(crop_bounds):
            return None
        if (int(shape[0]), int(shape[1])) != (int(img_shape[0]), int(img_shape[1])):
            return None
        col, row = int(points[0][0]), int(points[0][1])
        if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]):
            return None
        answer = np.asarray([mask]), np.asarray([float(score)]), logits




        try:
            from ...core.hover_preview_client import (
                frame_stand_in_logits,
                preview_alternatives_for,
                preview_alternatives_frame,
            )
            frame = preview_alternatives_frame(mask)
            alternatives = [
                (m, sc) for m, sc in preview_alternatives_for(mask)
                if m.shape == mask.shape]
            if alternatives and logits.ndim == 3:
                side = int(logits.shape[-1])
                answer = (
                    np.stack([mask] + [m for m, _ in alternatives]),
                    np.asarray([float(score)] + [sc for _, sc in alternatives]),
                    np.concatenate([np.asarray(logits, dtype=np.float32)] + [
                        frame_stand_in_logits(m, frame, side)
                        for m, _ in alternatives]),
                )
        except Exception:  # noqa: BLE001  # nosec B110
            answer = np.asarray([mask]), np.asarray([float(score)]), logits




        if drawn is not None:
            try:
                minx, miny, maxx, maxy = (float(v) for v in crop_bounds)
                height, width = int(shape[0]), int(shape[1])
                if height > 0 and width > 0 and maxx > minx and maxy > miny:
                    x = minx + (col + 0.5) * (maxx - minx) / width
                    y = maxy - (row + 0.5) * (maxy - miny) / height
                    if drawn.contains(QgsPointXY(x, y)):
                        return answer
            except Exception:  # noqa: BLE001  # nosec B110
                pass
        from ...core.server_dials import dial_in_range

        hover_reuse_near_px = dial_in_range(
            "tuning.click.hover_reuse_near_px", self._HOVER_REUSE_NEAR_PX, 4, 100)
        if (abs(row - int(asked[0])) > hover_reuse_near_px
                or abs(col - int(asked[1])) > hover_reuse_near_px):
            return None
        if not mask[row, col]:


            return None
        return answer

    def _last_prediction_found_nothing(self) -> bool:


        if self.current_mask is None:
            return False
        return bool(getattr(self, "_last_prediction_empty", False))

    def _last_click_stood_clear_of_shape(self) -> bool:


        return bool(getattr(self, "_last_click_stood_clear", False))

    def _show_cloud_warming_wait(self, elapsed_s) -> None:


        item = getattr(self, "_cloud_warming_item", None)
        try:
            bar = self.iface.messageBar()
            if elapsed_s is None:
                if item is not None:
                    self._cloud_warming_item = None
                    bar.popWidget(item)
                return
            text = tr("Waking up the AI...") + f" {int(elapsed_s)}s"
            if item is None:
                item = bar.createMessage("AI Segmentation", text)
                bar.pushItem(item)
                self._cloud_warming_item = item
            else:
                item.setText(text)
        except (RuntimeError, AttributeError):
            self._cloud_warming_item = None

    def _end_click_quietly(self, reason: str) -> None:




        self._quiet_click_end = reason

    def _take_quiet_click_end(self):

        reason = getattr(self, "_quiet_click_end", None)
        self._quiet_click_end = None
        return reason

    def _rollback_failed_click(self, polarity: str, canvas_point=None):













        self.prompts.undo()
        if polarity == "positive" and self._active_crop_points_positive:
            self._active_crop_points_positive.pop()
        elif polarity == "negative" and self._active_crop_points_negative:
            self._active_crop_points_negative.pop()
        if self._mask_state_history:
            self._restore_mask_state(self._mask_state_history.pop())
        quiet = self._take_quiet_click_end()
        if quiet != QUIET_CLICK_REREAD:

            self._wider_crop_replay_failed()
        if quiet == QUIET_CLICK_REREAD and canvas_point is not None:


            self._remember_pending_manual_click(polarity, canvas_point)
            return
        if self.map_tool:
            self.map_tool.remove_last_marker()
        if quiet is None and not self._headless:
            from ...core.server_dials import dial_in_range

            duration = dial_in_range(
                "tuning.click.rollback_notice_s", 5, 4, 10)
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("Something went wrong with this click, so it was not applied. Please try again."),
                level=Qgis.MessageLevel.Warning,
                duration=duration
            )

    def _update_ui_after_prediction(self):
        if not self.dock_widget:
            return
        pos_count, neg_count = self.prompts.point_count
        self.dock_widget.set_point_count(pos_count, neg_count)


        refresh = getattr(self, "_refresh_ai_add_keep_button", None)
        if refresh is not None:
            refresh()

        if self.current_mask is not None:






            score = self.current_score if self.current_score is not None else 0.0
            if not getattr(self, "_score_goes_in_click_line", False):
                QgsMessageLog.logMessage(
                    f"Segmentation result: score={score:.3f}",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info
                )
            self._warn_if_unsure(score)
            self._update_mask_visualization()
        else:


            self._update_mask_visualization()

        self._safe_restore_canvas_focus()

    def _warn_if_unsure(self, score: float) -> None:

















        if self._unsure_warning_shown:
            return
        if not getattr(getattr(self, "predictor", None),
                       "last_answer_was_remote", False):
            return
        try:
            from ...core.detection_policy import click_unsure_below
            floor = click_unsure_below()
        except Exception:  # noqa: BLE001  # nosec B110
            return
        if floor <= 0 or score <= 0 or score >= floor:
            return
        self._unsure_warning_shown = True
        from ...core.server_dials import dial_copy, dial_in_range

        duration = dial_in_range("tuning.click.unsure_notice_s", 5, 4, 10)
        self.iface.messageBar().pushMessage(
            "AI Segmentation",
            dial_copy(
                "manual.click_unsure",
                tr("The model is unsure about this outline. Click again to correct "
                   "it, or draw it by hand.")),
            level=Qgis.MessageLevel.Info,
            duration=duration,
        )
