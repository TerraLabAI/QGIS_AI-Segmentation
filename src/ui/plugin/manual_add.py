



















from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr


class ManualAddMixin:






    def _on_ai_add_requested(self) -> None:









        review = self._auto_review
        if not review or not self.dock_widget:
            return






        if getattr(self, "_refine_add_mode_active", False):
            try:
                self._discard_pending_manual_click()
            except (RuntimeError, AttributeError):
                pass
            self._exit_ai_add_mode()
            try:
                self._on_reshape_done()
            except (RuntimeError, AttributeError):
                pass
            return


        if getattr(self, "_refine_handoff_active", False) or getattr(
                self, "_qgis_bridge_active", False):
            return
        layer = self._resolve_auto_source_layer()
        if layer is None:
            return



        if not self._manual_env_ready():
            if self._local_ai_install_pending():
                return



            if (getattr(self, "_local_ai_load_failed", False)
                    and getattr(self, "_local_ai_install_attempted", False)):
                self._warn_local_ai_unavailable_once()
                return
            from ..dialogs.confirm_dialog import (
                PRIMARY,
                SECONDARY,
                ChoiceButton,
                ask_choice,
            )
            if ask_choice(
                self.iface.mainWindow(), tr("Adding needs a one-time setup"),
                tr(
                    "Adding an object uses the free on-device AI, which is not "
                    "installed yet. Install it now? It runs once and takes a few "
                    "minutes. The review waits for it, then arms Add for you."),
                [ChoiceButton("cancel", tr("Cancel"), SECONDARY),
                 ChoiceButton("install", tr("Install now"), PRIMARY)],
                default="install", escape="cancel",
            ) != "install":
                return
            self._begin_local_ai_install("add")
            return
        self._enter_ai_add_mode(layer)

    def _enter_ai_add_mode(self, layer) -> None:



        self._disarm_shape_tool()
        self._handoff_source_layer = layer
        self._pending_refine_import = False
        self._refine_handoff_active = True

        self._auto_refined_in_manual = True


        self._remove_auto_selection_layer()
        self._set_exemplar_bands_visible(False)
        self._set_auto_zone_overlays_visible(False)
        try:
            self.dock_widget.enter_ai_reshape_state()
        except (RuntimeError, AttributeError):
            pass





        if not self._cloud_correct_predictor_active():
            self._ensure_interactive_setup()
        self._enter_manual_refine_session()


        self._refine_add_mode_active = True



        from .correct_focus import FOCUS_ID_NEW_OBJECT
        self._begin_correct_focus(FOCUS_ID_NEW_OBJECT)
        self._ai_add_kept_count = 0
        try:
            self.dock_widget.set_add_lane_armed(True, "ai")
        except (RuntimeError, AttributeError):
            pass
        self._refresh_ai_add_keep_button()
        QgsMessageLog.logMessage(
            "AI Add: armed (point at a missed object)",
            "AI Segmentation", level=Qgis.MessageLevel.Info)





    def _ai_add_outline_in_progress(self) -> bool:








        if self.current_mask is not None and self.current_transform_info is not None:
            return True
        if getattr(self, "_frozen_sessions", None):
            return True
        return getattr(self, "_unfrozen_display_polygon", None) is not None

    def _ai_add_gesture_in_progress(self) -> bool:





        if self._ai_add_outline_in_progress():
            return True
        if self.current_mask is not None:
            return True
        if getattr(self, "_active_crop_points_positive", None):
            return True
        return bool(getattr(self, "_is_refining_saved_object", False))

    def _keep_ai_add_outline(self) -> bool:









        if not getattr(self, "_refine_add_mode_active", False):
            return False
        if not self._ai_add_outline_in_progress():
            return False



        before_last = self.saved_polygons[-1] if self.saved_polygons else None
        try:
            self._on_save_polygon()
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            QgsMessageLog.logMessage(
                f"AI Add: keep failed ({exc})",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
            self._say_add_keep_did_not_land()
            return False
        kept = bool(self.saved_polygons) and self.saved_polygons[-1] is not before_last
        if not kept:
            self._say_add_keep_did_not_land()
        self._refresh_ai_add_keep_button()
        if kept:
            self._ai_add_kept_count = getattr(self, "_ai_add_kept_count", 0) + 1
            fn = getattr(self.dock_widget, "set_add_lane_progress", None)
            if callable(fn):
                try:
                    fn(self._ai_add_kept_count)
                except (RuntimeError, AttributeError, TypeError):
                    pass
        return kept

    def _say_add_keep_did_not_land(self) -> None:


        try:
            from ...core.server_dials import dial_in_range

            duration = dial_in_range("tuning.manual.add_keep_failed_notice_s", 6, 4, 10)
            self.iface.messageBar().pushMessage(
                "AI Segmentation",
                tr("That shape was not added. Adjust it with a click and try "
                   "again."),
                level=Qgis.MessageLevel.Warning, duration=duration)
        except (RuntimeError, AttributeError):
            pass  # nosec B110

    def _route_save_add_mode(self) -> bool:







        if not getattr(self, "_refine_add_mode_active", False):
            return False
        self._keep_ai_add_outline()
        return True

    def _refresh_ai_add_keep_button(self) -> None:












        dock = getattr(self, "dock_widget", None)
        if dock is None:
            return
        armed = bool(getattr(self, "_refine_add_mode_active", False))
        fn = getattr(dock, "set_add_lane_keep_available", None)
        if callable(fn):
            try:
                fn(armed and self._ai_add_outline_in_progress())
            except (RuntimeError, AttributeError, TypeError):
                pass
        undo_fn = getattr(dock, "set_add_lane_undo_available", None)
        if callable(undo_fn):
            try:
                undo_fn(armed and self._ai_session_has_undo())
            except (RuntimeError, AttributeError, TypeError):
                pass
        self._refresh_correct_session_undo()

    def _refresh_correct_session_undo(self) -> None:




        dock = getattr(self, "dock_widget", None)
        if dock is None or not getattr(self, "_refine_handoff_active", False):
            return
        fn = getattr(dock, "set_correct_session_undo_available", None)
        if callable(fn):
            try:
                fn(self._ai_session_has_undo())
            except (RuntimeError, AttributeError, TypeError):
                pass





    def _exit_ai_add_mode(self) -> None:




        self._refine_add_mode_active = False


        self._end_correct_focus()
        if self.dock_widget is None:
            return
        try:
            method = self.dock_widget.get_correct_method()
        except (RuntimeError, AttributeError):
            method = "ai"
        try:

            self.dock_widget.set_add_lane_armed(False, method)
        except (RuntimeError, AttributeError):
            pass

    def _clear_ai_add_install_pending(self) -> None:




        if not getattr(self, "_ai_add_install_pending", False):
            return
        self._ai_add_install_pending = False
        if self.dock_widget:
            try:
                self.dock_widget.set_auto_review_installing(False)
            except (RuntimeError, AttributeError):
                pass





    def _route_escape_add_mode(self) -> bool:








        if self._ai_add_gesture_in_progress():
            try:
                self._clear_active_mask_without_saving()
            except (RuntimeError, AttributeError):
                pass
            self._is_refining_saved_object = False
            self._active_refine_origin_entry = None
            try:
                self._discard_pending_manual_click()
            except (RuntimeError, AttributeError):
                pass
            self._refresh_ai_add_keep_button()
            return True


        self._exit_ai_add_mode()
        try:
            self._on_reshape_done()
        except (RuntimeError, AttributeError):
            pass
        return True
