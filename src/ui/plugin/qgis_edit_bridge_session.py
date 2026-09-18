






from __future__ import annotations

import time

from ...core.i18n import tr


class EditBridgeSessionMixin:









    def _init_qgis_bridge_state(self) -> None:

        self._qgis_bridge_active = False


        self._qgis_bridge_finishing = False




        self._qgis_bridge_stop_queued = False




        self._qgis_bridge_closing_capture = False
        self._qgis_bridge_layer = None

        self._qgis_bridge_saved_aids: dict | None = None

        self._qgis_bridge_prev_maptool = None


        self._qgis_bridge_editing_conn = False
        self._qgis_bridge_t0 = 0.0




        self._qgis_bridge_layer_flags = None




        self._qgis_bridge_snapshot: dict[int, bytes] = {}



        self._qgis_bridge_vertex_dock_visibility: dict[int, tuple[object, bool]] = {}




        self._qgis_bridge_target_idx: int | None = None


        self._qgis_bridge_target_det_id: int | None = None




        self._qgis_bridge_saved_search_radius: tuple | None = None




        self._qgis_bridge_saved_selection_colour = None


        self._qgis_bridge_feedback_conn = False




        self._qgis_bridge_layer_watch = False



        self._qgis_bridge_isolated = False



        self._qgis_bridge_undo_stack = None


        self._qgis_bridge_poll_timer = None
        self._qgis_bridge_prev_points = 0
        self._qgis_bridge_gesture_undo_count = None




        self._qgis_bridge_hand_edited = False
        self._qgis_bridge_dial_edit = False


        self._qgis_bridge_pending_fids: list[int] = []


        self._qgis_bridge_born_det_ids: set[int] = set()


        self._qgis_bridge_id_write_marks: set[int] = set()


        self._qgis_bridge_id_edit = False

        self._qgis_bridge_prev_layer = None



        self._qgis_bridge_isolation_refused = False
        self._qgis_bridge_add_entry = False





    def enter_qgis_edit_bridge(self) -> None:




        if getattr(self, "_qgis_bridge_active", False):
            return


        self._qgis_bridge_isolation_refused = False
        layer = self._resolve_bridge_layer()
        if layer is None:
            return






        self._qgis_bridge_target_idx = getattr(self, "_correct_selected_idx", None)


        try:
            self._disarm_shape_tool()
        except (RuntimeError, AttributeError):
            pass





        try:
            self.dock_widget.set_correct_selection(0)
        except (RuntimeError, AttributeError):
            pass

        canvas = self.iface.mapCanvas()
        self._qgis_bridge_prev_maptool = canvas.mapTool()



        try:
            self._qgis_bridge_prev_layer = self.iface.activeLayer()
        except (RuntimeError, AttributeError):
            self._qgis_bridge_prev_layer = None
        self._save_bridge_editing_aids()
        if not self._expose_and_activate_bridge_layer(layer):
            self._restore_bridge_layer_presentation(layer)
            self._qgis_bridge_layer_flags = None
            self._restore_bridge_setting(self._restore_bridge_editing_aids)
            self._qgis_bridge_saved_aids = None
            self._qgis_bridge_prev_maptool = None
            self._show_bridge_unavailable()
            self._rearm_correct_select_after_bridge_bail()
            return




        self._qgis_bridge_pending_fids = []
        self._qgis_bridge_born_det_ids = set()
        self._qgis_bridge_id_write_marks = set()




        self._qgis_bridge_target_det_id = self._bridge_target_det_id()
        self._qgis_bridge_isolated = bool(
            self._isolate_bridge_target(layer, self._qgis_bridge_target_det_id))
        if (self._qgis_bridge_target_det_id is not None
                and not self._qgis_bridge_isolated):




            self._clear_bridge_isolation()
            self._restore_bridge_layer_presentation(layer)
            self._qgis_bridge_layer_flags = None
            self._restore_bridge_setting(self._restore_bridge_editing_aids)
            self._qgis_bridge_saved_aids = None
            self._qgis_bridge_prev_maptool = None
            self._qgis_bridge_target_det_id = None
            self._qgis_bridge_isolation_refused = True


            if not getattr(self, "_qgis_bridge_add_entry", False):
                self._show_bridge_isolation_failed()
            self._rearm_correct_select_after_bridge_bail()
            return
        if not self._apply_bridge_editing_config(layer):


            self._clear_bridge_isolation()
            self._restore_bridge_layer_presentation(layer)
            self._qgis_bridge_layer_flags = None
            self._restore_bridge_setting(self._restore_bridge_editing_aids)
            self._qgis_bridge_saved_aids = None
            self._qgis_bridge_prev_maptool = None


            self._qgis_bridge_target_det_id = None
            self._rearm_correct_select_after_bridge_bail()
            return




        self._qgis_bridge_snapshot = self._snapshot_bridge_layer(layer)

        self._qgis_bridge_layer = layer
        self._qgis_bridge_active = True
        self._qgis_bridge_finishing = False
        self._qgis_bridge_id_edit = False
        self._qgis_bridge_t0 = time.monotonic()

        self._remember_bridge_vertex_editor_visibility()



        has_target = self._qgis_bridge_target_det_id is not None
        self._set_bridge_shape_tools_visible(has_target)
        if has_target:
            self.activate_qgis_bridge_tool("vertex")


        try:
            layer.editingStopped.connect(self._on_qgis_bridge_editing_stopped)
            self._qgis_bridge_editing_conn = True
        except (RuntimeError, AttributeError, TypeError):
            self._qgis_bridge_editing_conn = False

        self._enter_bridge_banner()



        self._select_and_frame_bridge_target(layer)





        if has_target and self._qgis_bridge_target_idx is not None:
            self._correct_selected_idx = self._qgis_bridge_target_idx
            try:
                self.dock_widget.set_correct_selection(1)
            except (RuntimeError, AttributeError):
                pass
            _push = getattr(self, "_push_shape_only_state", None)
            if _push is not None:
                _push()



        self._qgis_bridge_hand_edited = False
        self._qgis_bridge_dial_edit = False
        dock = getattr(self, "dock_widget", None)
        try:
            reset = getattr(dock, "reset_qgis_bridge_points", None)
            if callable(reset):
                reset()
            show = getattr(dock, "set_qgis_bridge_points_visible", None)
            if callable(show):
                show(self._qgis_bridge_target_det_id is not None)



            show_delete = getattr(dock, "set_qgis_bridge_delete_visible", None)
            if callable(show_delete):
                show_delete(self._qgis_bridge_target_det_id is not None)
        except (RuntimeError, AttributeError, TypeError):
            pass
        self._connect_bridge_feedback(layer)
        self._connect_bridge_layer_watch()
        self._connect_bridge_tool_messages()
        self._start_bridge_gesture_poll()
        self._track_bridge("opened")

    def _rearm_correct_select_after_bridge_bail(self) -> None:





        try:
            self._arm_correct_select()
        except (RuntimeError, AttributeError):
            pass

    def finish_qgis_edit_bridge(self, commit: bool = True) -> None:




        if not getattr(self, "_qgis_bridge_active", False):
            return
        if getattr(self, "_qgis_bridge_finishing", False):
            return
        if commit:








            self._qgis_bridge_closing_capture = True
            try:
                self._close_open_bridge_capture()
            finally:
                self._qgis_bridge_closing_capture = False




            self._assign_bridge_born_det_ids()
        layer = self._qgis_bridge_layer
        if commit and layer is not None and self._is_layer_valid(layer):
            try:
                editable = layer.isEditable()
            except (RuntimeError, AttributeError):
                editable = False




            committed = True
            commit_raised = False
            if editable:
                self._qgis_bridge_finishing = True
                try:
                    committed = bool(layer.commitChanges())
                except (RuntimeError, AttributeError):
                    commit_raised = True
                finally:
                    self._qgis_bridge_finishing = False
            if commit_raised:






                self._teardown_qgis_edit_bridge(commit=False, external=True)
                return
            if not committed:
                errors = []
                try:
                    errors = list(layer.commitErrors() or [])
                except (RuntimeError, AttributeError):
                    pass


                self._show_bridge_commit_error(errors)
                return
        self._teardown_qgis_edit_bridge(commit=commit, external=False)

    def delete_bridge_target_polygon(self) -> None:













        det_id = getattr(self, "_qgis_bridge_target_det_id", None)
        if getattr(self, "_qgis_bridge_active", False):
            self.finish_qgis_edit_bridge(commit=False)
        if det_id is None:
            return
        idx = self._object_index_for_det_id(det_id)
        if idx is None:
            return
        self._remove_detection_index(idx)





    def _abort_qgis_edit_bridge_if_active(self) -> None:



        if not getattr(self, "_qgis_bridge_active", False):
            return
        try:
            self._teardown_qgis_edit_bridge(commit=False, external=False)
        except Exception as exc:  # noqa: BLE001


            self._log_bridge_failure("qgis_bridge_teardown", exc)


            try:



                self._clear_bridge_isolation()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._restore_bridge_editing_aids()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._restore_bridge_vertex_search_radius()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._restore_bridge_selection_colour()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._restore_bridge_attribute_form(self._qgis_bridge_layer)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._restore_bridge_vertex_editor_visibility()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:




                self._restore_bridge_layer_presentation(self._qgis_bridge_layer)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._restore_bridge_map_tool()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._leave_bridge_banner()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:



                self._stop_bridge_gesture_poll()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._disconnect_bridge_layer_watch()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:



                self._disconnect_bridge_tool_messages()
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            try:
                self._set_bridge_shape_tools_visible(True)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
            self._qgis_bridge_active = False
            self._qgis_bridge_finishing = False
            self._qgis_bridge_isolated = False
            self._qgis_bridge_layer = None
            self._qgis_bridge_saved_aids = None
            self._qgis_bridge_prev_maptool = None
            self._qgis_bridge_editing_conn = False
            self._qgis_bridge_feedback_conn = False
            self._qgis_bridge_layer_flags = None
            self._qgis_bridge_snapshot = {}
            self._qgis_bridge_vertex_dock_visibility = {}
            self._qgis_bridge_saved_search_radius = None
            self._qgis_bridge_saved_selection_colour = None
            self._qgis_bridge_saved_form_suppress = None
            self._qgis_bridge_target_idx = None
            self._qgis_bridge_target_det_id = None
            self._qgis_bridge_undo_stack = None
            self._qgis_bridge_pending_fids = []
            self._qgis_bridge_born_det_ids = set()
            self._qgis_bridge_id_write_marks = set()
            self._qgis_bridge_id_edit = False
            self._qgis_bridge_hand_edited = False
            self._qgis_bridge_dial_edit = False
            self._qgis_bridge_closing_capture = False
            try:
                self._end_correct_focus()
            except Exception:  # noqa: BLE001
                pass  # nosec B110

    def _on_qgis_bridge_editing_stopped(self) -> None:











        if not getattr(self, "_qgis_bridge_active", False):
            return
        if getattr(self, "_qgis_bridge_finishing", False):
            return
        if getattr(self, "_qgis_bridge_stop_queued", False):
            return
        self._qgis_bridge_stop_queued = True
        try:
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._teardown_bridge_after_external_stop)
        except (RuntimeError, AttributeError, ImportError):



            self._qgis_bridge_stop_queued = False
            self._teardown_qgis_edit_bridge(commit=True, external=True)

    def _teardown_bridge_after_external_stop(self) -> None:






        try:
            if not getattr(self, "_qgis_bridge_active", False):
                return
            if getattr(self, "_qgis_bridge_finishing", False):
                return
            layer = getattr(self, "_qgis_bridge_layer", None)
            if layer is not None and not self._is_layer_valid(layer):



                self._qgis_bridge_layer = None
            self._teardown_qgis_edit_bridge(commit=True, external=True)
        finally:
            self._qgis_bridge_stop_queued = False





    def _teardown_qgis_edit_bridge(self, commit: bool, external: bool) -> None:



        if not getattr(self, "_qgis_bridge_active", False):
            return
        self._qgis_bridge_finishing = True
        layer = self._qgis_bridge_layer
        committed = external or commit
        try:


            self._disconnect_bridge_editing_signal(layer)
            self._disconnect_bridge_feedback(layer)
            self._disconnect_bridge_layer_watch()
            self._disconnect_bridge_tool_messages()
            self._stop_bridge_gesture_poll()
            if (not external and layer is not None and self._is_layer_valid(layer)):
                try:
                    if layer.isEditable():
                        if commit:
                            layer.commitChanges()
                        else:
                            layer.rollBack()
                except (RuntimeError, AttributeError):
                    pass


            if layer is not None and self._is_layer_valid(layer):
                try:
                    layer.removeSelection()
                except (RuntimeError, AttributeError):
                    pass




            self._restore_bridge_setting(self._restore_bridge_editing_aids)
            self._restore_bridge_setting(self._restore_bridge_vertex_search_radius)
            self._restore_bridge_setting(self._restore_bridge_selection_colour)
            self._restore_bridge_setting(self._restore_bridge_attribute_form, layer)
            self._restore_bridge_setting(self._restore_bridge_map_tool)
            self._restore_bridge_setting(self._restore_bridge_vertex_editor_visibility)
            self._leave_bridge_banner()





            features = 0
            if committed and layer is not None and self._is_layer_valid(layer):
                features = self._bridge_fold_back(layer)



            self._restore_review_step_after_bridge()
            duration_ms = int(
                (time.monotonic() - (self._qgis_bridge_t0 or time.monotonic())) * 1000)
            self._track_bridge(
                "committed" if committed else "rolled_back",
                duration_ms=duration_ms, features=features)
        finally:






            self._clear_bridge_isolation()
            self._restore_bridge_layer_presentation(layer)
            self._disconnect_bridge_layer_watch()
            self._disconnect_bridge_tool_messages()


            self._set_bridge_shape_tools_visible(True)
            self._qgis_bridge_active = False
            self._qgis_bridge_finishing = False
            self._qgis_bridge_isolated = False
            self._qgis_bridge_layer = None
            self._qgis_bridge_saved_aids = None
            self._qgis_bridge_prev_maptool = None
            self._qgis_bridge_editing_conn = False
            self._qgis_bridge_feedback_conn = False
            self._qgis_bridge_layer_flags = None
            self._qgis_bridge_snapshot = {}
            self._qgis_bridge_vertex_dock_visibility = {}
            self._qgis_bridge_saved_search_radius = None
            self._qgis_bridge_saved_selection_colour = None
            self._qgis_bridge_saved_form_suppress = None
            self._qgis_bridge_target_idx = None
            self._qgis_bridge_target_det_id = None
            self._qgis_bridge_pending_fids = []
            self._qgis_bridge_born_det_ids = set()
            self._qgis_bridge_id_write_marks = set()
            self._qgis_bridge_id_edit = False



        try:
            self._end_correct_focus()
        except (RuntimeError, AttributeError):
            pass











    def _disconnect_bridge_editing_signal(self, layer) -> None:
        if not self._qgis_bridge_editing_conn or layer is None:
            self._qgis_bridge_editing_conn = False
            return
        try:
            layer.editingStopped.disconnect(self._on_qgis_bridge_editing_stopped)
        except (TypeError, RuntimeError, AttributeError):
            pass
        self._qgis_bridge_editing_conn = False





    def _connect_bridge_layer_watch(self) -> None:











        if getattr(self, "_qgis_bridge_layer_watch", False):
            return
        self._qgis_bridge_layer_watch = False
        try:
            from qgis.core import QgsProject

            QgsProject.instance().layersWillBeRemoved.connect(
                self._on_bridge_layer_will_be_removed)
            self._qgis_bridge_layer_watch = True
        except (RuntimeError, AttributeError, TypeError, ImportError):
            self._qgis_bridge_layer_watch = False

    def _disconnect_bridge_layer_watch(self) -> None:

        if not getattr(self, "_qgis_bridge_layer_watch", False):
            return
        self._qgis_bridge_layer_watch = False
        try:
            from qgis.core import QgsProject

            QgsProject.instance().layersWillBeRemoved.disconnect(
                self._on_bridge_layer_will_be_removed)
        except (RuntimeError, AttributeError, TypeError, ImportError):

            pass

    def _on_bridge_layer_will_be_removed(self, layer_ids) -> None:





        if not getattr(self, "_qgis_bridge_active", False):
            return
        layer = getattr(self, "_qgis_bridge_layer", None)
        if layer is None:
            return
        try:
            ids = set(layer_ids or [])
            if not ids or layer.id() not in ids:
                return
        except (RuntimeError, AttributeError, TypeError):
            return


        self._disconnect_bridge_layer_watch()
        self._abort_qgis_edit_bridge_if_active()





    def _enter_bridge_banner(self) -> None:
        self._call_dock_bridge_banner("enter_qgis_bridge_state")

    def _leave_bridge_banner(self) -> None:
        self._call_dock_bridge_banner("leave_qgis_bridge_state")

    def _call_dock_bridge_banner(self, method: str) -> None:
        dock = getattr(self, "dock_widget", None)
        if dock is None:
            return
        fn = getattr(dock, method, None)
        if not callable(fn):
            return
        try:
            fn()
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _log_bridge_failure(self, stage: str, exc: BaseException) -> None:






        try:
            from ...core import telemetry_errors
            telemetry_errors.report_exception(
                exc, stage=stage, module="qgis_edit_bridge")
        except Exception:  # noqa: BLE001
            pass  # nosec B110

    def _show_bridge_commit_error(self, errors: list) -> None:


        detail = "; ".join(str(e) for e in (errors or [])[:3])


        message = tr("QGIS could not save these edits. Fix the geometry and "
                     "click Save again.")
        try:
            from qgis.core import Qgis, QgsMessageLog
            QgsMessageLog.logMessage(
                f"QGIS edit bridge commit failed: {detail}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
        try:
            from qgis.core import Qgis

            from ...core.server_dials import dial_in_range
            self.iface.messageBar().pushMessage(
                "AI Segmentation", message,
                level=Qgis.MessageLevel.Warning,
                duration=dial_in_range("tuning.agent.bridge_commit_error_notice_s", 6, 4, 10))
        except (RuntimeError, AttributeError):
            pass


        try:
            self._bridge_feedback(message, "warning")
        except (RuntimeError, AttributeError, TypeError):
            pass





    def _bridge_fold_back(self, layer) -> int:





        hook = getattr(self, "_fold_qgis_edits_back", None)
        if callable(hook):
            try:
                hook(layer)
            except Exception as exc:  # noqa: BLE001






                try:
                    from ...core import telemetry_errors
                    telemetry_errors.report_exception(
                        exc, stage="qgis_bridge_fold_back",
                        module="qgis_edit_bridge")
                except Exception:  # noqa: BLE001
                    pass  # nosec B110
        try:
            return len(getattr(self, "_auto_objects", None) or [])
        except (TypeError, AttributeError):
            return 0

    def _restore_review_step_after_bridge(self) -> None:





        dock = getattr(self, "dock_widget", None)
        if dock is None:
            return
        fn = getattr(dock, "set_auto_review_step", None)
        if not callable(fn):
            return
        try:
            fn(int(getattr(self, "_auto_review_step", 1)))
        except (RuntimeError, AttributeError, TypeError):
            pass
        if getattr(self, "_auto_review_step", 1) == 1:
            try:
                self._arm_correct_select()
            except (RuntimeError, AttributeError):
                pass





    def _track_bridge(self, outcome: str, duration_ms: int | None = None,
                      features: int | None = None) -> None:
        try:
            from ...core import telemetry_run_events
            telemetry_run_events.track_qgis_edit_bridge(
                run_id=getattr(self, "_auto_run_id", "") or "",
                outcome=outcome, duration_ms=duration_ms, features=features)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
