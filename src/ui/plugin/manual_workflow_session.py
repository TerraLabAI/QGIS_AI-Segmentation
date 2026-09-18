






from __future__ import annotations

from qgis.core import Qgis

from ...core.i18n import tr


class ManualWorkflowSessionMixin:





    def _on_layer_combo_changed(self, layer):


        if not self._current_layer:
            return



        try:
            new_layer_id = layer.id() if layer else None
            current_layer_id = self._current_layer.id() if self._current_layer else None
        except RuntimeError:

            self._current_layer = None
            return

        if new_layer_id == current_layer_id:
            return




        has_unsaved_mask = self.current_mask is not None
        has_unsaved_mask = has_unsaved_mask or bool(self._frozen_sessions)
        has_unsaved_mask = has_unsaved_mask or self._unfrozen_display_polygon is not None
        has_saved_polygons = len(self.saved_polygons) > 0

        if has_unsaved_mask or has_saved_polygons:
            polygon_count = len(self.saved_polygons)
            if has_unsaved_mask:
                polygon_count += 1
            if polygon_count == 1:
                held = tr("You have 1 unsaved polygon.")
            else:
                held = tr("You have {count} unsaved polygons.").format(
                    count=polygon_count)
            message = "{}\n\n{}".format(
                held, tr("Saving keeps them on the map."))

            choice = self._ask_unsaved_semi_auto(tr("Change Layer?"), message)
            if choice == "save":
                if self._on_export_layer():
                    return
                choice = "cancel"
            if choice == "cancel":
                self.dock_widget.layer_combo.blockSignals(True)
                self.dock_widget.layer_combo.setLayer(self._current_layer)
                self.dock_widget.layer_combo.blockSignals(False)
                return
            try:
                from ...core import telemetry_session_events
                telemetry_session_events.track_manual_abandoned(
                    context="change_layer", polygon_count=polygon_count)
            except Exception:
                pass  # nosec B110

            self._autosave_manual_saved_polygons(include_live=True)




        self._teardown_manual_session()

    def _on_tool_deactivated(self):


        self._stop_hover_preview("tool deactivated")

        try:
            if self._shortcut_filter is not None:
                self.iface.mainWindow().removeEventFilter(self._shortcut_filter)
                canvas = self.iface.mapCanvas()
                canvas.viewport().removeEventFilter(self._shortcut_filter)
                canvas.removeEventFilter(self._shortcut_filter)
        except (RuntimeError, AttributeError):
            pass

        if self._stopping_segmentation:
            if self.dock_widget:
                self.dock_widget.set_segmentation_active(False)
            return



        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(0, self._return_to_segmentation)

    def _return_to_segmentation(self):
        if self._stopping_segmentation:
            return
        if not self.map_tool or not self.dock_widget:
            return
        self._say_tool_stays_armed_once()
        self._activate_segmentation_tool()

    def _say_tool_stays_armed_once(self) -> None:







        if getattr(self, "_tool_rearm_notice_shown", False):
            return
        self._tool_rearm_notice_shown = True
        try:
            from ...core.server_dials import dial_in_range
            duration = dial_in_range(
                "tuning.manual.tool_rearm_notice_s", 6, 3, 15)
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("The click tool stays on while this session is open. Stop "
                   "the session to use another map tool."),
                level=Qgis.MessageLevel.Info,
                duration=duration,
            )
        except (RuntimeError, AttributeError):
            pass

    def _stop_manual_session_for_hidden_dock(self) -> None:











        try:
            if getattr(self, "_refine_handoff_active", False):
                return
            main_window = self.iface.mainWindow()
            siblings = main_window.tabifiedDockWidgets(self.dock_widget) or []
            for sibling in siblings:
                if sibling.isVisible():
                    return
        except (RuntimeError, AttributeError, TypeError):
            return
        self._park_manual_session()

    def _park_manual_session(self) -> bool:








        tool = getattr(self, "map_tool", None)
        if tool is None or self._current_layer is None:
            return False
        try:
            canvas = self.iface.mapCanvas()
            if canvas.mapTool() is not tool:
                return False


            self._stopping_segmentation = True
            try:
                canvas.unsetMapTool(tool)
                self._restore_previous_map_tool()
            finally:
                self._stopping_segmentation = False
        except (RuntimeError, AttributeError):
            return False
        self._manual_session_parked = True
        return True

    def _resume_parked_manual_session(self) -> None:





        if not getattr(self, "_manual_session_parked", False):
            return
        self._manual_session_parked = False
        if self.map_tool is None or self._current_layer is None:
            return
        if not self._is_layer_valid():
            return
        self._activate_segmentation_tool()

    def _restore_previous_map_tool(self):

        if self._previous_map_tool:
            try:
                self.iface.mapCanvas().setMapTool(self._previous_map_tool)
            except RuntimeError:

                pass
        self._previous_map_tool = None

    def _on_clear_selection(self) -> None:









        if not self.map_tool or not self.map_tool.isActive():
            return




        if self._refine_edit_session_active():
            self._close_active_edit_to_pending()
            return
        discard = getattr(self, "_discard_pending_manual_click", None)
        if discard is not None:
            discard()
        self._clear_active_mask_without_saving()
        self._frozen_sessions = []
        self._active_crop_points_positive = []
        self._active_crop_points_negative = []
        refresh = getattr(self, "_refresh_ai_add_keep_button", None)
        if refresh is not None:
            refresh()

    def _ask_unsaved_semi_auto(self, title: str, held: str,
                               leaving: bool = False) -> str:













        from ..dialogs.confirm_dialog import (
            DISCARD,
            PRIMARY,
            SECONDARY,
            WARNING,
            ChoiceButton,
            ask_choice,
        )
        if leaving:
            return ask_choice(
                self.iface.mainWindow(), title, held,
                [ChoiceButton("discard", tr("Discard && exit"), DISCARD),
                 ChoiceButton("cancel", tr("Cancel"), SECONDARY),
                 ChoiceButton("save", tr("Save && exit"), PRIMARY)],
                default="save", escape="cancel")
        return ask_choice(
            self.iface.mainWindow(), title, held,
            [ChoiceButton("discard", tr("Discard"), DISCARD),
             ChoiceButton("cancel", tr("Cancel"), SECONDARY),
             ChoiceButton("save", tr("Save them"), PRIMARY)],
            default="save", escape="cancel", tone=WARNING)

    def _on_stop_segmentation(self):





        if self._refine_handoff_active:
            if self._refine_edit_session_active():
                self._close_active_edit_to_pending()
            else:
                self._on_reshape_done()
            return
        polygon_count = len(self.saved_polygons)


        if self.current_mask is not None or self._frozen_sessions or self._unfrozen_display_polygon is not None:
            polygon_count += 1

        if polygon_count > 0:


            if polygon_count == 1:
                title = tr("Keep your polygon?")
                question = tr("Save 1 polygon to a layer before leaving?")
            else:
                title = tr("Keep your polygons?")
                question = tr(
                    "Save {count} polygons to a layer before leaving?").format(
                        count=polygon_count)
            choice = self._ask_unsaved_semi_auto(title, question, True)
            if choice == "cancel":
                return
            if choice == "save":



                if self._on_export_layer():
                    self._signal_gpu_session_end("semi_auto_stop")
                return
            try:
                from ...core import telemetry_session_events
                telemetry_session_events.track_manual_abandoned(
                    context="stop", polygon_count=polygon_count)
            except Exception:
                pass  # nosec B110



            self._autosave_manual_saved_polygons(include_live=True)

        self._stop_manual_session(keep_saves=False)
        self._signal_gpu_session_end("semi_auto_stop")

    def _stop_manual_session(self, keep_saves: bool) -> None:









        has_unsaved_work = False
        if keep_saves:
            has_unsaved_work = self.saved_polygons or self.current_mask is not None
            has_unsaved_work = has_unsaved_work or self._frozen_sessions
            has_unsaved_work = has_unsaved_work or self._unfrozen_display_polygon is not None
        if has_unsaved_work:


            self._on_export_layer()



            self._autosave_manual_saved_polygons(include_live=True)
        self._teardown_manual_session()

    def _teardown_manual_session(self) -> None:






        try:
            from ...core.click_crop_encoding import flush_crop_profile
            flush_crop_profile()
        except Exception:  # noqa: BLE001  # nosec B110
            pass


        self._end_manual_credit_session()


        self._stop_canvas_crs_watch()



        if getattr(self, "_refine_add_mode_active", False):
            exit_add = getattr(self, "_exit_ai_add_mode", None)
            if exit_add is not None:
                exit_add()
            else:
                self._refine_add_mode_active = False
        if self._shortcut_filter is not None:
            try:
                self.iface.mainWindow().removeEventFilter(self._shortcut_filter)
                canvas = self.iface.mapCanvas()
                canvas.viewport().removeEventFilter(self._shortcut_filter)
                canvas.removeEventFilter(self._shortcut_filter)
            except RuntimeError:
                pass
        self._stopping_segmentation = True
        try:
            self.iface.mapCanvas().unsetMapTool(self.map_tool)
            self._restore_previous_map_tool()
        finally:
            self._stopping_segmentation = False
        self._reset_session()
        if self.dock_widget:
            try:
                self.dock_widget.reset_session()
            except (RuntimeError, AttributeError):
                pass
