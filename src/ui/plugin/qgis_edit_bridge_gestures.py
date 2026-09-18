







from __future__ import annotations

from ...core.i18n import tr
from .bridge_capture_state import live_capture_points
from .qgis_edit_bridge_tools import _VERTEX_TOOL_CLASSES




_BRIDGE_POLL_MS = 200







_SPLIT_TOOL_CLASS = "QgsMapToolSplitFeatures"

_ADD_TOOL_CLASSES = ("QgsMapToolAddFeature", "QgsMapToolDigitizeFeature")


def _bridge_capture_points(tool, class_name: str):











    if tool is None or class_name in _VERTEX_TOOL_CLASSES:
        return None
    size = getattr(tool, "size", None)
    if not callable(size):
        return None
    try:
        return int(size())
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return None


class EditBridgeGesturesMixin:









    def _connect_bridge_feedback(self, layer) -> None:



        if layer is None:
            return
        connected = False
        for signal_name, handler in (
            ("geometryChanged", self._on_bridge_geometry_changed),
            ("featureAdded", self._on_bridge_feature_added),
            ("featuresDeleted", self._on_bridge_features_deleted),
        ):
            signal = getattr(layer, signal_name, None)
            if signal is None:
                continue
            try:
                signal.connect(handler)
                connected = True
            except (RuntimeError, AttributeError, TypeError):
                pass
        stack = None
        try:
            stack = layer.undoStack()
        except (RuntimeError, AttributeError):
            stack = None
        self._qgis_bridge_undo_stack = stack
        if stack is not None:
            try:
                stack.indexChanged.connect(self._on_bridge_undo_index_changed)
                connected = True
            except (RuntimeError, AttributeError, TypeError):
                pass
        self._qgis_bridge_feedback_conn = connected

    def _disconnect_bridge_feedback(self, layer) -> None:
        if not self._qgis_bridge_feedback_conn or layer is None:




            self._qgis_bridge_undo_stack = None
            self._qgis_bridge_feedback_conn = False
            return
        for signal_name, handler in (
            ("geometryChanged", self._on_bridge_geometry_changed),
            ("featureAdded", self._on_bridge_feature_added),
            ("featuresDeleted", self._on_bridge_features_deleted),
        ):
            signal = getattr(layer, signal_name, None)
            if signal is None:
                continue
            try:
                signal.disconnect(handler)
            except (RuntimeError, AttributeError, TypeError):
                pass
        stack = getattr(self, "_qgis_bridge_undo_stack", None)
        self._qgis_bridge_undo_stack = None
        if stack is not None:
            try:
                stack.indexChanged.disconnect(
                    self._on_bridge_undo_index_changed)
            except (RuntimeError, AttributeError, TypeError):
                pass
        self._qgis_bridge_feedback_conn = False

    def _bridge_feedback(self, text: str, kind: str = "armed") -> None:



        dock = getattr(self, "dock_widget", None)
        fn = getattr(dock, "set_qgis_bridge_feedback", None)
        if callable(fn):
            try:
                fn(text, kind)
            except (RuntimeError, AttributeError, TypeError):
                pass

    def _bridge_line_open(self, open_: bool) -> None:



        dock = getattr(self, "dock_widget", None)
        fn = getattr(dock, "set_qgis_bridge_line_open", None)
        if callable(fn):
            try:
                fn(bool(open_))
            except (RuntimeError, AttributeError, TypeError):
                pass

    def _on_bridge_geometry_changed(self, *_args) -> None:
        if getattr(self, "_qgis_bridge_finishing", False):
            return


        if getattr(self, "_qgis_bridge_dial_edit", False):
            return
        self._mark_bridge_hand_edited()
        self._bridge_feedback(
            tr("Shape updated. Keep editing, or click Save."), "success")

    def _on_bridge_feature_added(self, *args) -> None:


        if getattr(self, "_qgis_bridge_finishing", False):
            return
        if getattr(self, "_qgis_bridge_dial_edit", False):
            return
        self._mark_bridge_hand_edited()
        self._bridge_feedback(tr("New shape added. Click Save to keep it."),
                              "success")
        self._queue_bridge_born_det_id(args[0] if args else None)


        layer = getattr(self, "_qgis_bridge_layer", None)
        if layer is not None:
            try:
                layer.triggerRepaint()
            except (RuntimeError, AttributeError):
                pass





    def _queue_bridge_born_det_id(self, fid) -> None:









        if not isinstance(fid, int):
            return
        pending = getattr(self, "_qgis_bridge_pending_fids", None)
        if pending is None:
            pending = []
            self._qgis_bridge_pending_fids = pending
        if fid not in pending:
            pending.append(fid)
        try:
            from qgis.PyQt.QtCore import QTimer
            QTimer.singleShot(0, self._assign_bridge_born_det_ids)
        except (RuntimeError, AttributeError, ImportError):


            pass

    def _bridge_det_id_field_index(self, layer) -> int:

        try:
            return int(layer.fields().indexFromName("det_id"))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return -1

    def _bridge_used_det_ids(self, layer, skip_fids=()) -> set[int]:








        used: set[int] = set()
        try:
            objects = getattr(self, "_auto_objects", None) or []
            for index in range(len(objects)):
                value = self._object_fid_for(index)
                if isinstance(value, int) and value >= 0:
                    used.add(int(value))
        except (RuntimeError, AttributeError, TypeError):
            pass
        try:
            from qgis.core import QgsFeatureRequest
            request = QgsFeatureRequest()
            request.setSubsetOfAttributes(["det_id"], layer.fields())
            scope = getattr(QgsFeatureRequest, "Flag", QgsFeatureRequest)
            no_geometry = getattr(scope, "NoGeometry", None)
            if no_geometry is not None:
                request.setFlags(no_geometry)
            for feature in layer.getFeatures(request):
                if feature.id() in skip_fids:
                    continue
                value = feature["det_id"]
                if isinstance(value, int) and value >= 0:
                    used.add(int(value))
        except (RuntimeError, AttributeError, TypeError, KeyError, ValueError,
                ImportError):
            pass
        used.update(getattr(self, "_qgis_bridge_born_det_ids", None) or ())
        return used

    def _assign_bridge_born_det_ids(self) -> None:




















        if (getattr(self, "_qgis_bridge_finishing", False) or getattr(self, "_qgis_bridge_closing_capture", False)):
            return
        pending = list(getattr(self, "_qgis_bridge_pending_fids", None) or ())
        self._qgis_bridge_pending_fids = []
        if not pending or not getattr(self, "_qgis_bridge_active", False):
            return
        layer = getattr(self, "_qgis_bridge_layer", None)
        if layer is None or not self._is_layer_valid(layer):
            return



        try:
            if not layer.isEditable():
                return
        except (RuntimeError, AttributeError):
            return
        field_index = self._bridge_det_id_field_index(layer)
        if field_index < 0:
            return
        used = self._bridge_used_det_ids(layer, skip_fids=set(pending))
        next_id = max(used, default=-1) + 1
        assignments: list[tuple[int, int]] = []
        for fid in pending:
            try:
                feature = layer.getFeature(fid)
                if feature is None or not feature.isValid():
                    continue
                current = feature["det_id"]
            except (RuntimeError, AttributeError, KeyError, TypeError, ValueError):
                continue
            if isinstance(current, int) and current >= 0 and current not in used:
                used.add(int(current))
                continue
            while next_id in used:
                next_id += 1
            assignments.append((fid, next_id))
            used.add(next_id)
            next_id += 1
        if not assignments:
            return
        self._write_bridge_born_det_ids(layer, field_index, assignments)

    def _write_bridge_born_det_ids(self, layer, field_index, assignments) -> None:





        born = getattr(self, "_qgis_bridge_born_det_ids", None)
        if born is None:
            born = set()
            self._qgis_bridge_born_det_ids = born
        self._qgis_bridge_id_edit = True
        wrote = False
        opened = False
        try:
            layer.beginEditCommand(tr("Identify new shape"))
            opened = True
            for fid, det_id in assignments:
                try:
                    if layer.changeAttributeValue(fid, field_index, det_id):
                        born.add(det_id)
                        wrote = True
                except (RuntimeError, AttributeError, TypeError, ValueError):

                    pass
        except (RuntimeError, AttributeError, TypeError):
            wrote = False
        finally:


            if opened:
                try:
                    layer.endEditCommand()
                except (RuntimeError, AttributeError):
                    wrote = False
            self._qgis_bridge_id_edit = False
        if not wrote:
            return
        stack = getattr(self, "_qgis_bridge_undo_stack", None)
        marks = getattr(self, "_qgis_bridge_id_write_marks", None)
        if stack is not None and marks is not None:
            try:
                marks.add(int(stack.index()))
            except (RuntimeError, AttributeError, TypeError):
                pass
        try:
            layer.triggerRepaint()
        except (RuntimeError, AttributeError):
            pass

    def _bridge_pop_identity_mark(self, stack) -> bool:




        marks = getattr(self, "_qgis_bridge_id_write_marks", None)
        if not marks:
            return False
        try:
            index = int(stack.index())
        except (RuntimeError, AttributeError, TypeError):
            return False
        for mark in [value for value in marks if value > index]:
            marks.discard(mark)
        if index not in marks:
            return False
        marks.discard(index)
        return True

    def _on_bridge_features_deleted(self, *_args) -> None:
        if getattr(self, "_qgis_bridge_finishing", False):
            return
        self._bridge_feedback(
            tr("A shape was removed. Click Save to confirm."), "success")

    def _on_bridge_undo_index_changed(self, *_args) -> None:




        if getattr(self, "_qgis_bridge_finishing", False):
            return


        if getattr(self, "_qgis_bridge_id_edit", False):
            return
        stack = getattr(self, "_qgis_bridge_undo_stack", None)
        text = ""
        can_undo = False
        if stack is not None:
            try:
                can_undo = bool(stack.canUndo())
                if can_undo:




                    text = (str(stack.undoText() or "").strip() or tr("Change recorded."))
            except (RuntimeError, AttributeError, TypeError):
                can_undo = False
        dock = getattr(self, "dock_widget", None)
        fn = getattr(dock, "set_qgis_bridge_last_change", None)
        if callable(fn):
            try:
                fn(text, can_undo)
            except (RuntimeError, AttributeError, TypeError):
                pass

    def undo_qgis_bridge_edit(self) -> None:








        if not getattr(self, "_qgis_bridge_active", False):
            return
        if self._bridge_live_capture_points() > 0:
            self._bridge_undo_capture_vertex()
            return
        stack = getattr(self, "_qgis_bridge_undo_stack", None)
        if stack is None:
            return
        try:




            paired = self._bridge_pop_identity_mark(stack)
            if stack.canUndo():
                stack.undo()
                if paired and stack.canUndo():
                    stack.undo()
        except (RuntimeError, AttributeError):
            pass

    def _route_escape_qgis_bridge(self) -> bool:








        if self._bridge_live_capture_points() > 0:
            self._bridge_cancel_capture()
            return True
        self.finish_qgis_edit_bridge()
        return True

    def _bridge_live_capture_points(self) -> int:


        return live_capture_points(self, _bridge_capture_points)

    def _bridge_cancel_capture(self) -> None:





        try:
            canvas = self.iface.mapCanvas()
            tool = canvas.mapTool() if canvas is not None else None
            if tool is None:
                return
            reset = getattr(tool, "stopCapturing", None)
            if not callable(reset):
                return
            reset()
            self._qgis_bridge_prev_points = 0
            self._bridge_line_open(False)
            self._bridge_feedback("")
            canvas.refresh()
        except (RuntimeError, AttributeError, TypeError):
            pass





    def _on_bridge_gesture_requested(self, kind: str) -> None:






        if not getattr(self, "_qgis_bridge_active", False):
            return
        kind = str(kind)
        if kind == "finish":
            self._bridge_send_right_click()
            return
        if kind == "cancel":



            self._bridge_cancel_capture()
            return
        if kind == "undo_point":





            self._bridge_undo_capture_vertex()
            return
        from qgis.PyQt.QtCore import Qt
        key = {"delete_corner": Qt.Key.Key_Delete}.get(kind)
        if key is not None:
            self._bridge_send_key(key)

    def _bridge_undo_capture_vertex(self) -> None:




        try:
            canvas = self.iface.mapCanvas()
            tool = canvas.mapTool() if canvas is not None else None
            if tool is None:
                return
            undo = getattr(tool, "undo", None)


            if not callable(undo) or not callable(getattr(tool, "size", None)):
                return
            undo()
            canvas.refresh()
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _bridge_send_key(self, key) -> None:





        try:
            from qgis.PyQt.QtCore import QEvent, Qt
            from qgis.PyQt.QtGui import QKeyEvent
            from qgis.PyQt.QtWidgets import QApplication
            canvas = self.iface.mapCanvas()
            if canvas is None:
                return
            no_mod = Qt.KeyboardModifier.NoModifier
            for phase in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
                QApplication.sendEvent(
                    canvas, QKeyEvent(phase, int(key), no_mod))
        except (RuntimeError, AttributeError, TypeError, ValueError, ImportError):
            pass

    def _bridge_send_right_click(self) -> None:







        try:
            from qgis.PyQt.QtCore import QEvent, QPointF, Qt
            from qgis.PyQt.QtGui import QMouseEvent
            from qgis.PyQt.QtWidgets import QApplication
            canvas = self.iface.mapCanvas()
            if canvas is None:
                return
            tool = canvas.mapTool()
            if tool is not None:
                try:
                    cls = str(tool.metaObject().className())
                except (RuntimeError, AttributeError):
                    cls = ""
                if cls in _VERTEX_TOOL_CLASSES:
                    return
            viewport = canvas.viewport()
            if viewport is None:
                return
            centre = viewport.rect().center()
            local = QPointF(centre)
            global_pt = QPointF(viewport.mapToGlobal(centre))
            right = Qt.MouseButton.RightButton
            no_mod = Qt.KeyboardModifier.NoModifier
            for phase in (QEvent.Type.MouseButtonPress,
                          QEvent.Type.MouseButtonRelease):
                event = QMouseEvent(
                    phase, local, global_pt, right, right, no_mod)
                QApplication.sendEvent(viewport, event)
        except (RuntimeError, AttributeError, TypeError, ValueError, ImportError):
            pass





    def _mark_bridge_hand_edited(self) -> None:





        self._qgis_bridge_hand_edited = True
        dock = getattr(self, "dock_widget", None)
        fn = getattr(dock, "set_qgis_bridge_points_visible", None)
        if callable(fn):
            try:
                fn(False)
            except (RuntimeError, AttributeError, TypeError):
                pass

    def _on_bridge_points_changed(self, pct: int) -> None:






        if not getattr(self, "_qgis_bridge_active", False):
            return
        if getattr(self, "_qgis_bridge_hand_edited", False):
            return
        layer = self._qgis_bridge_layer
        det_id = getattr(self, "_qgis_bridge_target_det_id", None)
        if layer is None or det_id is None or not self._is_layer_valid(layer):
            return
        base_wkb = (getattr(self, "_qgis_bridge_snapshot", None) or {}).get(det_id)
        if not base_wkb:
            return
        from qgis.core import QgsGeometry
        base = QgsGeometry()
        try:
            base.fromWkb(base_wkb)
        except (RuntimeError, AttributeError, TypeError):
            return
        if base is None or base.isEmpty():
            return
        try:
            pct = max(10, min(100, int(pct)))
        except (TypeError, ValueError):
            return
        geom = QgsGeometry(base) if pct >= 100 else self._bridge_thin_geometry(
            base, pct)


        from ...core.layer_conventions import repair_polygon
        geom = repair_polygon(geom) if geom is not None else None
        if geom is None or geom.isEmpty():
            geom = QgsGeometry(base)
        feature = self._bridge_feature_for_det_id(layer, det_id)
        if feature is None:
            return
        fid = feature.id()
        self._qgis_bridge_dial_edit = True
        try:
            layer.beginEditCommand(tr("Fewer points"))
            layer.changeGeometry(fid, geom)
            layer.endEditCommand()
            layer.triggerRepaint()
        except (RuntimeError, AttributeError, TypeError):
            try:
                layer.destroyEditCommand()
            except (RuntimeError, AttributeError):
                pass


            return
        finally:
            self._qgis_bridge_dial_edit = False
        try:
            count = self._bridge_ring_vertex_count(geom)
            if count:
                self._bridge_feedback(tr("Points: {n}").format(n=count))
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _apply_shape_only_to_session(self, det_idx: int) -> bool:













        if not getattr(self, "_qgis_bridge_active", False):
            return False
        if getattr(self, "_qgis_bridge_hand_edited", False):
            return False
        layer = self._qgis_bridge_layer
        det_id = getattr(self, "_qgis_bridge_target_det_id", None)
        if layer is None or det_id is None or not self._is_layer_valid(layer):
            return False
        if self._det_id_for_object_index(det_idx) != det_id:
            return False
        base_wkb = (getattr(self, "_qgis_bridge_snapshot", None) or {}).get(det_id)
        if not base_wkb:
            return False
        from qgis.core import QgsGeometry
        base = QgsGeometry()
        try:
            base.fromWkb(base_wkb)
        except (RuntimeError, AttributeError, TypeError):
            return False
        if base.isEmpty():
            return False
        review = self._auto_review or {}
        try:
            params = self._shape_params_for_object(
                int(det_idx), dict(review.get("params") or {}))
            geom = self._refine_geom_for_review(
                base, params, float(review.get("pixel_size", 1.0) or 1.0))
        except Exception:  # noqa: BLE001
            return False
        if geom is None or geom.isEmpty():
            return False
        feature = self._bridge_feature_for_det_id(layer, det_id)
        if feature is None:
            return False
        self._qgis_bridge_dial_edit = True
        try:
            layer.beginEditCommand(tr("Outline settings"))
            layer.changeGeometry(feature.id(), geom)
            layer.endEditCommand()
            layer.triggerRepaint()
        except (RuntimeError, AttributeError, TypeError):
            try:
                layer.destroyEditCommand()
            except (RuntimeError, AttributeError):
                pass
            return False
        finally:
            self._qgis_bridge_dial_edit = False
        try:
            count = self._bridge_ring_vertex_count(geom)
            if count:
                self._bridge_feedback(tr("Points: {n}").format(n=count))
        except (RuntimeError, AttributeError, TypeError):
            pass
        return True

    def _bridge_thin_geometry(self, base, pct: int):







        from qgis.core import QgsGeometry

        from ...core.detection_policy import vertex_budget_settings
        from ...core.vertex_budget import simplify_to_budget
        try:
            settings = vertex_budget_settings()
        except (RuntimeError, AttributeError, TypeError, KeyError):
            return QgsGeometry(base)
        factor = None
        try:
            centre = base.boundingBox().center()
            factor = self._auto_crs_metres_per_unit(centre.x(), centre.y())
            aspect = self._auto_crs_unit_aspect(centre.x(), centre.y())
        except (RuntimeError, AttributeError, TypeError):
            factor = None




        if not factor or factor <= 0:
            return QgsGeometry(base)
        try:
            aspect = float(aspect)
        except (TypeError, ValueError):
            aspect = 1.0
        if aspect <= 0:
            aspect = 1.0
        try:
            result = simplify_to_budget(
                base,
                spacing=0.0,
                min_vertices=int(settings["min_vertices"]),
                max_deviation=float(settings["max_deviation_m"]) / factor,
                max_deviation_fraction=float(
                    settings["max_deviation_fraction"]),


                unit_aspect=aspect,
                keep_fraction=pct / 100.0,



                dial_max_cap_fraction=float(
                    settings["dial_max_cap_fraction"]),
            )
        except (RuntimeError, AttributeError, TypeError, KeyError, ValueError):
            return QgsGeometry(base)
        if result is None or result.isEmpty():
            return QgsGeometry(base)
        return result

    def _bridge_ring_vertex_count(self, geom) -> int:


        try:
            if geom is None or geom.isEmpty():
                return 0
            if geom.isMultipart():
                polys = geom.asMultiPolygon()
                if polys and polys[0]:
                    return len(polys[0][0])
                return 0
            poly = geom.asPolygon()
            if poly:
                return len(poly[0])
            return 0
        except (RuntimeError, AttributeError, TypeError, IndexError):
            return 0





    def _start_bridge_gesture_poll(self) -> None:



        try:
            from qgis.PyQt.QtCore import QTimer
            timer = self._qgis_bridge_poll_timer
            if timer is None:
                timer = QTimer(self.dock_widget)
                from ...core.server_dials import dial_in_range
                timer.setInterval(
                    dial_in_range("tuning.agent.bridge_poll_ms", _BRIDGE_POLL_MS, 100, 2000))
                timer.timeout.connect(self._on_bridge_gesture_tick)
                self._qgis_bridge_poll_timer = timer
            self._qgis_bridge_prev_points = 0
            self._qgis_bridge_gesture_undo_count = None
            timer.start()
        except (RuntimeError, AttributeError, TypeError, ImportError):
            self._qgis_bridge_poll_timer = None

    def _stop_bridge_gesture_poll(self) -> None:
        timer = getattr(self, "_qgis_bridge_poll_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except (RuntimeError, AttributeError):
                pass
        self._qgis_bridge_prev_points = 0
        self._qgis_bridge_gesture_undo_count = None

    def _bridge_undo_depth(self):




        stack = getattr(self, "_qgis_bridge_undo_stack", None)
        if stack is None:
            return None
        try:
            return int(stack.index())
        except (RuntimeError, AttributeError, TypeError):
            return None

    def _on_bridge_gesture_tick(self) -> None:




        if not getattr(self, "_qgis_bridge_active", False):
            self._stop_bridge_gesture_poll()
            return
        tool = None
        class_name = ""
        try:
            tool = self.iface.mapCanvas().mapTool()
            if tool is not None:
                class_name = str(tool.metaObject().className())
        except (RuntimeError, AttributeError, TypeError):
            return
        points = _bridge_capture_points(tool, class_name)
        if points is None:
            if int(getattr(self, "_qgis_bridge_prev_points", 0) or 0) > 0:
                self._bridge_line_open(False)
            self._qgis_bridge_prev_points = 0


            self._sync_bridge_delete_corner(class_name)
            return

        self._sync_bridge_delete_corner("")
        prev = int(getattr(self, "_qgis_bridge_prev_points", 0))
        if points == prev:
            return
        if prev == 0 and points > 0:
            self._qgis_bridge_gesture_undo_count = self._bridge_undo_depth()
        self._qgis_bridge_prev_points = points


        self._bridge_line_open(points > 0)
        if points > 0:


            self._bridge_feedback(
                tr("{n} point placed.").format(n=points) if points == 1
                else tr("{n} points placed.").format(n=points))
        elif prev > 0:
            self._report_bridge_gesture_result(class_name)

    def _report_bridge_gesture_result(self, tool_class: str) -> None:




        start = getattr(self, "_qgis_bridge_gesture_undo_count", None)
        self._qgis_bridge_gesture_undo_count = None
        depth = self._bridge_undo_depth()
        if start is None or depth is None or depth > start:
            return
        if tool_class == _SPLIT_TOOL_CLASS:
            self._bridge_feedback(tr(
                "Nothing was split. The line has to cross the shape completely, "
                "starting and ending outside it."), "warning")
        elif tool_class in _ADD_TOOL_CLASSES:
            self._bridge_feedback(tr(
                "Nothing was added. A polygon needs at least three corners."),
                "warning")
        else:
            self._bridge_feedback(tr(
                "Nothing changed. The line has to cross the outline twice, "
                "starting and ending outside the shape."), "warning")

    def _close_open_bridge_capture(self) -> None:









        try:
            tool = self.iface.mapCanvas().mapTool()
            class_name = str(tool.metaObject().className()) if tool else ""
        except (RuntimeError, AttributeError, TypeError):
            return
        points = _bridge_capture_points(tool, class_name)
        if not points:
            return
        self._bridge_send_right_click()
        self._qgis_bridge_prev_points = 0
        self._bridge_line_open(False)
