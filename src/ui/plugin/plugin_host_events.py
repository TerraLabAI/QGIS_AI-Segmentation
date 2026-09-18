









from __future__ import annotations


class HostEventsMixin:





    def _on_project_read_sweep_temp(self, *_args):







        try:
            from qgis.PyQt.QtCore import QTimer

            def _sweep():
                try:
                    from ...core.output_store import sweep_stale_temp_layers
                    sweep_stale_temp_layers()
                except Exception:  # nosec B110
                    pass

            QTimer.singleShot(0, _sweep)
        except Exception:  # nosec B110
            pass

    def _on_dock_visibility_changed(self, visible: bool):
        if not visible:


            self._stop_hover_preview("dock hidden")


            self._stop_manual_session_for_hidden_dock()

            self._set_credits_watch_paused(True)



            self._set_config_refresh_paused(True)




            if self.dock_widget.isHidden():
                self._signal_gpu_session_end("panel_closed")
            return

        self._resume_parked_manual_session()


        self._reset_credits_backoff()
        self._set_credits_watch_paused(False)
        self._resume_config_refresh()
        if self._first_time_setup_done:
            return
        self._first_time_setup_done = True


        self._arm_credits_watch()

        from qgis.PyQt.QtCore import QTimer

        from ..ai_segmentation_dockwidget import Mode


        QTimer.singleShot(0, self._prefetch_server_config)


        QTimer.singleShot(0, self._prefetch_segment_catalog)
        if self.dock_widget and self.dock_widget._mode == Mode.AUTOMATIC:

            QTimer.singleShot(0, self._refresh_activation_async)
            return
        self._interactive_setup_done = True
        QTimer.singleShot(0, self._do_first_time_setup)

    def _on_auto_enter_pressed(self) -> bool:







        tool = self._zone_selection_tool
        if tool is not None:
            try:
                if self.iface.mapCanvas().mapTool() is tool and tool.has_points():
                    if tool.finish():
                        return True
            except (RuntimeError, AttributeError):
                pass
        return self._route_enter()

    def _on_auto_undo_pressed(self) -> bool:









        tool = self._zone_selection_tool
        if tool is None:
            return False
        try:
            if self.iface.mapCanvas().mapTool() is tool:
                return tool.undo_point()
        except (RuntimeError, AttributeError):
            pass
        return False

    def _ensure_interactive_setup(self) -> None:








        if self._interactive_setup_done:
            return
        self._interactive_setup_done = True
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(0, self._do_first_time_setup)
