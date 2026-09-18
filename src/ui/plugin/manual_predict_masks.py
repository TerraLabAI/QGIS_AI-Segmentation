






from __future__ import annotations

from qgis.core import Qgis, QgsGeometry, QgsMessageLog, QgsPointXY
from qgis.gui import QgsRubberBand
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QApplication

from ...core.i18n import tr
from ...core.qt_compat import DashLine, PolygonGeometry, SolidLine, WaitCursor
from ...core.review_defaults import (
    REFINE_CLEAN_DEFAULT,
    REFINE_EXPAND_DEFAULT,
    REFINE_FILL_HOLES_DEFAULT,
    REFINE_FILL_HOLES_MAX_M2_DEFAULT,
    REFINE_MIN_AREA_DEFAULT,
    REFINE_MIN_SIZE_M2_DEFAULT,
    REFINE_ORTHO_DEFAULT,
    REFINE_POINTS_PCT_DEFAULT,
    REFINE_SIMPLIFY_DEFAULT,
    REFINE_SMOOTH_DEFAULT,
)
from ..canvas_palette import PENDING_FILL, PENDING_STROKE


class ManualMaskMixin:









    _REMOTE_CLICK_NOTE_MS = 1_200

    def _click_answer_travels(self) -> bool:





        active = getattr(self, "_cloud_correct_predictor_active", None)
        if active is None:
            return False
        try:
            return bool(active())
        except Exception:  # noqa: BLE001
            return False

    def _remote_click_wait_showing(self) -> bool:


        return bool(getattr(self, "_remote_click_wait_active", False))

    def _begin_click_wait(self) -> bool:









        if self._begin_correct_wait():
            return True
        if self._correct_wait_showing() or self._remote_click_wait_showing():
            return False




        travels = self._click_answer_travels()
        self._remote_click_wait_active = True
        self._remote_click_wait_cursor = False
        if not self._headless:
            try:
                QApplication.setOverrideCursor(WaitCursor)
                self._remote_click_wait_cursor = True
            except (RuntimeError, AttributeError):
                self._remote_click_wait_cursor = False
        if travels:
            self._arm_remote_click_note()
        self._apply_mask_band_style()
        return True

    def _arm_remote_click_note(self) -> None:


        if self._headless:
            return
        try:
            from ...core.server_dials import dial_in_range
            from .shared import _debounce_timer

            note_ms = dial_in_range(
                "tuning.manual.remote_click_note_ms", self._REMOTE_CLICK_NOTE_MS, 300, 5000)
            _debounce_timer(self, "_remote_click_note_timer", self.dock_widget,
                            note_ms, self._show_remote_click_note)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _show_remote_click_note(self) -> None:


        if not self._remote_click_wait_showing():
            return
        self._set_manual_encoding_note(True, phase="remote")

    def _clear_remote_click_note(self) -> None:

        timer = getattr(self, "_remote_click_note_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except (RuntimeError, AttributeError):
                pass
        self._set_manual_encoding_note(False)

    def _end_click_wait_started_here(self) -> None:







        if not getattr(self, "_click_wait_started_here", False):
            return
        self._click_wait_started_here = False
        self._end_click_wait()

    def _end_click_wait(self) -> None:


        if not self._remote_click_wait_showing():
            self._end_correct_wait()
            return
        self._remote_click_wait_active = False
        if getattr(self, "_remote_click_wait_cursor", False):
            self._remote_click_wait_cursor = False
            try:
                QApplication.restoreOverrideCursor()
            except (RuntimeError, AttributeError):
                pass
        self._clear_remote_click_note()
        self._apply_mask_band_style()

    def _apply_mask_band_style(self) -> None:











        if self.mask_rubber_band is None:
            return
        fill = PENDING_FILL
        if (getattr(self, "_refine_handoff_active", False)
                and getattr(self, "_auto_display_mode", "") == "outline"):
            fill = QColor(PENDING_FILL)
            fill.setAlpha(0)
        self.mask_rubber_band.setColor(fill)
        self.mask_rubber_band.setStrokeColor(PENDING_STROKE)



        editing = self._refine_handoff_active or self._is_refining_saved_object
        self.mask_rubber_band.setWidth(3 if editing else 2)




        waiting = (self._correct_wait_showing()
                   or self._remote_click_wait_showing())
        try:
            self.mask_rubber_band.setLineStyle(
                DashLine if waiting else SolidLine)
        except (RuntimeError, AttributeError):
            pass

    def _update_mask_visualization(self):
        if self.mask_rubber_band is None:

            try:
                self.mask_rubber_band = QgsRubberBand(
                    self.iface.mapCanvas(),
                    PolygonGeometry
                )
                self.mask_rubber_band.setColor(PENDING_FILL)
                self.mask_rubber_band.setStrokeColor(PENDING_STROKE)
                self.mask_rubber_band.setWidth(2)
            except Exception:
                return


        self._apply_mask_band_style()

        if self.current_mask is None or self.current_transform_info is None:

            if self._frozen_sessions or self._unfrozen_display_polygon is not None:
                self._display_frozen_composite_with_extra(
                    self._unfrozen_display_polygon)
            else:
                self._clear_mask_visualization()
            return

        try:
            from ...core.detection_policy import manual_simplify_multiple_of_px
            from ...core.polygon_exporter import count_significant_regions



            _mult = manual_simplify_multiple_of_px()
            _tol = (_mult * self._crop_pixel_size_units(self.current_transform_info)
                    if _mult > 0 else 0.0)





            fill_holes, max_hole_px = self._fill_holes_arguments()
            mask_to_display, geometries = self._manual_mask_polygons(
                fill_holes, max_hole_px, _tol)








            may_warn = not self._disjoint_warning_shown and len(self._active_crop_points_positive) >= 2
            if (may_warn and len(geometries) > 1
                    and count_significant_regions(mask_to_display) > 1):
                from ...core.server_dials import dial_in_range
                self.iface.messageBar().pushMessage(
                    "AI Segmentation",
                    tr("Disconnected parts detected. For best accuracy, segment one element at a time."),
                    level=Qgis.MessageLevel.Warning,
                    duration=dial_in_range("tuning.manual.disjoint_warning_notice_s", 6, 4, 10)
                )
                self._disjoint_warning_shown = True


            all_geoms = [s.polygon for s in self._frozen_sessions]





            if geometries:
                active_combined = self._manual_active_outline()
                if active_combined is None or active_combined.isEmpty():










                    active_combined = QgsGeometry.unaryUnion(geometries)
                if active_combined and not active_combined.isEmpty():
                    all_geoms.append(active_combined)

            if all_geoms:
                combined = QgsGeometry.unaryUnion(all_geoms)
                if combined and not combined.isEmpty():

                    self._transform_geometry_to_canvas_crs(combined)
                    self.mask_rubber_band.setToGeometry(combined, None)
                else:
                    self._clear_mask_visualization()
            else:
                self._clear_mask_visualization()

        except (ValueError, TypeError, RuntimeError) as e:
            QgsMessageLog.logMessage(
                f"Mask visualization error ({type(e).__name__}): {str(e)}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning
            )
            self._clear_mask_visualization()
        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Unexpected mask visualization error ({type(e).__name__}): {str(e)}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            self._clear_mask_visualization()

    def _clear_mask_visualization(self):
        if self.mask_rubber_band:
            try:
                self.mask_rubber_band.reset(PolygonGeometry)
            except RuntimeError:
                self.mask_rubber_band = None

    def _on_undo(self):






        try:
            self._undo_one_gesture()
        finally:
            refresh = getattr(self, "_refresh_ai_add_keep_button", None)
            if refresh is not None:
                refresh()

    def _undo_one_gesture(self):








        if self._encoding_in_progress and not self._abandon_speculative_manual_crop():










            if getattr(self, "_pending_manual_click", None) is not None:
                self._discard_pending_manual_click()
            else:


                self._set_manual_encoding_note(True, phase="encode")
            return
        self._manual_undos_session = getattr(self, "_manual_undos_session", 0) + 1






        if self._refine_edit_session_active() and self.current_mask is None:
            history = getattr(self, "_refine_geom_history", None)
            if history:
                self._unfrozen_display_polygon = history.pop()
                if self.map_tool:
                    self.map_tool.remove_last_marker()
                self._update_mask_visualization()
            return

        current_point_count = self.prompts.point_count[0] + self.prompts.point_count[1]


        should_restore_deleted = current_point_count == 0
        should_restore_deleted = should_restore_deleted and getattr(self, "_deleted_objects_stack", None)
        should_restore_deleted = should_restore_deleted and self._restore_deleted_object()
        if should_restore_deleted:
            return

        if current_point_count > 0:

            result = self.prompts.undo()
            if result is None:
                return

            if self.map_tool:
                self.map_tool.remove_last_marker()





            state = (self._mask_state_history.pop()
                     if self._mask_state_history else None)
            if state:
                self._restore_mask_state(state)
            else:
                self.current_low_res_mask = None


            if result[0] == "positive" and self._active_crop_points_positive:
                self._active_crop_points_positive.pop()
            elif result[0] == "negative" and self._active_crop_points_negative:
                self._active_crop_points_negative.pop()

            if self.prompts.point_count[0] + self.prompts.point_count[1] > 0:

                self._update_ui_after_prediction()
            elif self._unfrozen_display_polygon is not None:




                self._update_mask_visualization()
                self.dock_widget.set_point_count(0, 0)



                self._keep_save_alive_for_display_polygon()
            else:

                if self._frozen_sessions:
                    self._unfreeze_last_session()
                else:
                    self.current_mask = None
                    self.current_score = 0.0
                    self._clear_mask_visualization()
                    self.dock_widget.set_point_count(0, 0)
        elif self._frozen_sessions:

            self._unfreeze_last_session()
        elif len(self.saved_polygons) > 0 and not self._refine_handoff_active:






            from ..dialogs.confirm_dialog import WARNING, question
            if question(
                self.iface.mainWindow(),
                tr("Edit a saved polygon?"),



                tr("Undo reopens the last polygon you saved."),



                default_yes=False, tone=WARNING, yes_label=tr("Edit"),
            ):
                self._restore_last_saved_mask()
            self._safe_restore_canvas_focus()

    def _keep_save_alive_for_display_polygon(self) -> None:






        if self._unfrozen_display_polygon is None:
            return
        button = getattr(self.dock_widget, "save_mask_button", None)
        if button is None:
            return
        try:
            button.setEnabled(True)
        except RuntimeError:
            pass

    def _unfreeze_last_session(self):





        if not self._frozen_sessions:
            return

        session = self._frozen_sessions.pop()


        self.current_mask = None
        self.current_score = 0.0
        self.current_low_res_mask = None
        self._current_crop_info = None
        self._mask_state_history = []




        self.prompts.clear()
        if self.map_tool:
            self.map_tool.clear_markers()

        self._active_crop_points_positive = list(session.points_positive)
        self._active_crop_points_negative = list(session.points_negative)
        for pt in session.points_positive:
            self.prompts.add_positive_point(pt[0], pt[1])
            if self.map_tool:
                canvas_pt = self._transform_to_canvas_crs(QgsPointXY(pt[0], pt[1]))
                if canvas_pt is not None:
                    self.map_tool.add_marker(canvas_pt, is_positive=True)
        for pt in session.points_negative:
            self.prompts.add_negative_point(pt[0], pt[1])
            if self.map_tool:
                canvas_pt = self._transform_to_canvas_crs(QgsPointXY(pt[0], pt[1]))
                if canvas_pt is not None:
                    self.map_tool.add_marker(canvas_pt, is_positive=False)





        self._unfrozen_display_polygon = session.polygon
        self._display_frozen_composite_with_extra(session.polygon)

        pos_count, neg_count = self.prompts.point_count
        if self.dock_widget:
            self.dock_widget.set_point_count(pos_count, neg_count)


        self._keep_save_alive_for_display_polygon()

        QgsMessageLog.logMessage(
            f"Unfroze crop session, {len(self._frozen_sessions)} frozen remaining",
            "AI Segmentation", level=Qgis.MessageLevel.Info)

    def _display_frozen_composite_with_extra(self, extra_polygon=None):

        if self.mask_rubber_band is None:
            return

        all_geoms = [s.polygon for s in self._frozen_sessions]
        if extra_polygon is not None:
            all_geoms.append(extra_polygon)

        if not all_geoms:
            self._clear_mask_visualization()
            return

        combined = QgsGeometry.unaryUnion(all_geoms)
        if combined and not combined.isEmpty():
            self._transform_geometry_to_canvas_crs(combined)
            self.mask_rubber_band.setToGeometry(combined, None)
        else:
            self._clear_mask_visualization()

    def _restore_last_saved_mask(self):

        if not self.dock_widget:
            return
        self._ensure_polygon_rubberband_sync()

        if not self.saved_polygons or not self.saved_rubber_bands:
            return


        last_polygon = self.saved_polygons.pop()
        self._refine_geom_history = []
        self._refine_edit_pristine = None
        self._refine_edit_last_applied = None






        self._active_refine_origin_entry = dict(last_polygon)




        if self.saved_rubber_bands:
            last_rb = self.saved_rubber_bands.pop()
            self._safe_remove_rubber_band(last_rb)
        if not self._handoff_remove_entry_feature(last_polygon):
            self._rebuild_handoff_layers()



        self._is_refining_saved_object = True


        self.prompts.clear()
        self._mask_state_history = []
        self._frozen_sessions = []
        self._unfrozen_display_polygon = None
        self._active_crop_points_positive = []
        self._active_crop_points_negative = []


        self.current_low_res_mask = None
        if self.map_tool:
            self.map_tool.clear_markers()


        points_positive = last_polygon.get("points_positive", [])
        points_negative = last_polygon.get("points_negative", [])


        for pt in points_positive:
            self.prompts.add_positive_point(pt[0], pt[1])
            if self.map_tool:
                canvas_pt = self._transform_to_canvas_crs(QgsPointXY(pt[0], pt[1]))
                if canvas_pt is not None:
                    self.map_tool.add_marker(canvas_pt, is_positive=True)

        for pt in points_negative:
            self.prompts.add_negative_point(pt[0], pt[1])
            if self.map_tool:
                canvas_pt = self._transform_to_canvas_crs(QgsPointXY(pt[0], pt[1]))
                if canvas_pt is not None:
                    self.map_tool.add_marker(canvas_pt, is_positive=False)


        self.current_mask = last_polygon.get("raw_mask")
        self.current_score = last_polygon.get("score", 0.0)
        self.current_transform_info = last_polygon.get("transform_info")
        if self.current_mask is None or self.current_transform_info is None:




            geom = last_polygon.get("geom_obj")
            if geom is None:
                geom = QgsGeometry.fromWkt(last_polygon.get("geometry_wkt") or "")
            if geom is not None and not geom.isEmpty():
                self._unfrozen_display_polygon = QgsGeometry(geom)



        self._refine_simplify = float(
            last_polygon.get("refine_simplify", REFINE_SIMPLIFY_DEFAULT) or 0.0)
        self._refine_points_pct = int(
            last_polygon.get("refine_points_pct") or REFINE_POINTS_PCT_DEFAULT)
        self._refine_smooth = last_polygon.get("refine_smooth", REFINE_SMOOTH_DEFAULT)
        self._refine_clean = float(
            last_polygon.get("refine_clean") or REFINE_CLEAN_DEFAULT)
        self._refine_expand = last_polygon.get("refine_expand", REFINE_EXPAND_DEFAULT)
        self._refine_fill_holes = last_polygon.get("refine_fill_holes", REFINE_FILL_HOLES_DEFAULT)
        hole_limit = last_polygon.get("refine_fill_holes_max_m2")
        self._refine_fill_holes_max_m2 = float(
            REFINE_FILL_HOLES_MAX_M2_DEFAULT if hole_limit is None else hole_limit)
        self._refine_ortho = last_polygon.get("refine_ortho", REFINE_ORTHO_DEFAULT)
        self._refine_min_area = last_polygon.get(
            "refine_min_area", REFINE_MIN_AREA_DEFAULT)
        self._refine_min_size_m2 = float(last_polygon.get("refine_min_size_m2") or REFINE_MIN_SIZE_M2_DEFAULT)


        self._refine_max_size_m2 = 0.0


        self.dock_widget.set_refine_values(
            self._refine_simplify,
            self._refine_smooth,
            self._refine_expand,
            self._refine_fill_holes,
            self._refine_min_area,
            right_angles=self._refine_ortho,
            fill_holes_max_m2=self._refine_fill_holes_max_m2,
            clean=self._refine_clean,
            points_pct=self._refine_points_pct,
        )
        self.dock_widget.set_size_filter_values(
            self._refine_min_size_m2, self._refine_max_size_m2)

        if self._unfrozen_display_polygon is not None:
            self._refine_edit_pristine = QgsGeometry(self._unfrozen_display_polygon)
            self._refine_edit_last_applied = self._current_refine_tuple()


        self._update_mask_visualization()


        pos_count, neg_count = self.prompts.point_count
        self.dock_widget.set_point_count(pos_count, neg_count)



        self._keep_save_alive_for_display_polygon()
        self.dock_widget.set_saved_polygon_count(len(self.saved_polygons))

        QgsMessageLog.logMessage(
            f"Restored mask with {pos_count} positive, {neg_count} negative points. "
            f"Refine: simplify={self._refine_simplify}, "
            f"points_pct={self._refine_points_pct}, smooth={self._refine_smooth}, "
            f"expand={self._refine_expand}, fill_holes={self._refine_fill_holes}, "
            f"min_area={self._refine_min_area}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info
        )

    def _reset_session(self):






        self._invalidate_manual_encode()



        try:
            saves = getattr(self, "_manual_saves_session", 0)
            if saves >= 1 and not self._refine_handoff_active:
                import time as _time

                from ...core import telemetry_session_events
                t0 = getattr(self, "_manual_session_t0", None)
                telemetry_session_events.track_manual_session_summary(
                    saves=saves,
                    undos=getattr(self, "_manual_undos_session", 0),
                    duration_ms=int((_time.time() - t0) * 1000) if t0 else None,
                )
        except Exception:
            pass  # nosec B110
        self._manual_saves_session = 0
        self._manual_undos_session = 0
        self._manual_session_t0 = None

        self._is_refining_saved_object = False
        self._active_refine_origin_entry = None
        self._refine_geom_history = []
        self._deleted_objects_stack = []

        self._handoff_selected_entries = []
        self._handoff_hover_entry = None
        self._handoff_hit_index = None
        self._handoff_tok2entry = {}
        self._handoff_det_id_seq = None


        self._handoff_imported_det_ids = set()
        for attr in ("_handoff_selection_band", "_handoff_hover_band"):
            band = getattr(self, attr, None)
            if band is not None:
                self._safe_remove_rubber_band(band)
                setattr(self, attr, None)


        self._segmentation_start_ts = None
        self.prompts.clear()
        self._mask_state_history = []
        self._frozen_sessions = []
        self._unfrozen_display_polygon = None
        self._active_crop_points_positive = []
        self._active_crop_points_negative = []
        self._disjoint_warning_shown = False
        self._unsure_warning_shown = False
        self.saved_polygons = []

        for rb in self.saved_rubber_bands:
            self._safe_remove_rubber_band(rb)
        self.saved_rubber_bands = []


        self._remove_handoff_layers()

        if self.map_tool:
            self.map_tool.clear_markers()

        self._clear_mask_visualization()

        self.current_mask = None
        self.current_score = 0.0
        self.current_transform_info = None
        self.current_low_res_mask = None


        self._hover_click_answer = None


        self._last_click_point = None
        self._last_click_polarity = "positive"
        self._last_prediction_empty = False
        self._last_click_stood_clear = False
        self._last_click_took_from_another = False




        self._current_layer = None
        self._canvas_to_raster_xform = None
        self._raster_to_canvas_xform = None
        self._current_layer_name = ""
        self._current_crop_info = None
        self._current_raster_path = None
        self._current_crop_canvas_mupp = None
        self._current_crop_actual_mupp = None
        self._current_crop_scale_factor = None



        self._is_online_layer = False
        try:
            from ...core.online_layer_twin import release_online_layer_twin
            release_online_layer_twin()
        except Exception:  # noqa: BLE001  # nosec B110
            pass



        try:
            from ...core.raster_dataset_cache import release_raster_datasets
            release_raster_datasets()
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        self._manual_shape_cache_reset()


        self._refine_simplify = float(REFINE_SIMPLIFY_DEFAULT)
        self._refine_points_pct = REFINE_POINTS_PCT_DEFAULT
        self._refine_smooth = REFINE_SMOOTH_DEFAULT
        self._refine_clean = REFINE_CLEAN_DEFAULT
        self._refine_expand = REFINE_EXPAND_DEFAULT
        self._refine_fill_holes = REFINE_FILL_HOLES_DEFAULT
        self._refine_fill_holes_max_m2 = REFINE_FILL_HOLES_MAX_M2_DEFAULT
        self._refine_ortho = REFINE_ORTHO_DEFAULT

        self._refine_min_area = REFINE_MIN_AREA_DEFAULT
        self._refine_min_size_m2 = REFINE_MIN_SIZE_M2_DEFAULT
        self._refine_max_size_m2 = 0.0




        self._tool_rearm_notice_shown = False
        self._manual_session_parked = False
        self._click_without_model_reported = False


        self._crop_errors_reported = set()

        if self.dock_widget:
            self.dock_widget.set_point_count(0, 0)
            self.dock_widget.set_saved_polygon_count(0)
            try:

                self.dock_widget.clear_manual_notice()
            except (RuntimeError, AttributeError):
                pass
