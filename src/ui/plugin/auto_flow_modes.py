







from __future__ import annotations


class AutoFlowModesMixin:


    def _on_mode_changed(self, mode) -> None:
        from ..ai_segmentation_dockwidget import Mode


        self._stop_hover_preview("mode changed")



        if mode == Mode.INTERACTIVE:
            self._warmup_if_manual_cloud()




        if self._refine_handoff_active:
            if mode == Mode.INTERACTIVE:
                self._ensure_interactive_setup()
                self._enter_manual_refine_session()
            else:
                self._restore_auto_review_after_handoff()
            return

        to_mode = "auto" if mode == Mode.AUTOMATIC else "manual"
        has_mask = getattr(self, "current_mask", None) is not None
        has_frozen_sessions = getattr(self, "_frozen_sessions", None)
        has_unfrozen_display = getattr(self, "_unfrozen_display_polygon", None) is not None
        has_saved_polygons = getattr(self, "saved_polygons", None)
        had_unsaved_manual = bool(has_mask or has_frozen_sessions or has_unfrozen_display or has_saved_polygons)
        auto_step = None
        try:
            auto_step = int(self.dock_widget.auto_steps.currentIndex())
        except (RuntimeError, AttributeError):
            pass





        self._reset_auto_flow_to_start(exit_path="mode_switch")
        self._reset_manual_flow_to_start()


        if self.dock_widget:
            self._refresh_auto_credits()
        if mode == Mode.INTERACTIVE and self.dock_widget:
            self._ensure_interactive_setup()



            self._resume_parked_manual_session()
        try:
            from ...core import telemetry_session_events
            telemetry_session_events.track_mode_switched(
                to_mode=to_mode,
                had_unsaved_manual=had_unsaved_manual,
                auto_step=auto_step,
            )
        except Exception:
            pass  # nosec B110

    def _reset_auto_flow_to_start(self, exit_path: str = "other") -> None:




        self._stop_auto_detection()
        self._discard_auto_review(exit_path=exit_path)
        self._restore_maptool_after_zone()


        self._auto_run_plan = None
        self._auto_attribute_filters = []
        self._cancel_task("_auto_run_plan_task")
        self._cancel_task("_auto_token_task")
        self._auto_zone = None
        self._auto_zone_polygon = None
        self._clear_auto_canvas()
        if self.dock_widget:
            try:
                self.dock_widget.reset_auto_to_start()
            except (RuntimeError, AttributeError):
                pass

    def _reset_manual_flow_to_start(self) -> None:








        has_unsaved = getattr(self, "current_mask", None) is not None
        has_unsaved = has_unsaved or getattr(self, "_frozen_sessions", None)
        has_unsaved = has_unsaved or getattr(self, "_unfrozen_display_polygon", None) is not None
        has_unsaved = has_unsaved or getattr(self, "saved_polygons", None)
        if has_unsaved:



            self._park_manual_session()
            return
        try:
            canvas = self.iface.mapCanvas()
            tool = getattr(self, "map_tool", None)
            if tool is not None and canvas.mapTool() == tool:




                self._stopping_segmentation = True
                try:
                    canvas.unsetMapTool(tool)
                finally:
                    self._stopping_segmentation = False
        except (RuntimeError, AttributeError):
            pass
        self._reset_session()
        if self.dock_widget:
            try:
                self.dock_widget.reset_session()
            except (RuntimeError, AttributeError):
                pass
