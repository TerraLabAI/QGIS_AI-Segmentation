







from __future__ import annotations

from ...core.qt_compat import QAction
from .bridge_capture_state import vertex_editor_docks



_BRIDGE_SHAPE_TOOLS = ("vertex", "reshape", "split")




_VERTEX_TOOL_CLASSES = ("QgsVertexTool", "QgsVertexToolV2", "QgsMapToolVertexEdit")


def flags_without(flags, flag):







    try:
        masked = int(flags) & ~int(flag)
    except (TypeError, ValueError):
        return flags
    try:
        return type(flags)(masked)
    except (TypeError, ValueError):
        return masked


class EditBridgeToolsMixin:





    def _bridge_target_det_id(self):








        idx = getattr(self, "_qgis_bridge_target_idx", None)
        if idx is None:
            return None
        try:
            ids = getattr(self, "_auto_object_fids", None) or []
            if idx < 0 or idx >= len(ids):
                return None
            return ids[idx]
        except (RuntimeError, AttributeError, TypeError):
            return None

    def _select_and_frame_bridge_target(self, layer) -> None:














        det_id = self._bridge_target_det_id()
        self._qgis_bridge_target_det_id = det_id
        try:
            layer.removeSelection()
        except (RuntimeError, AttributeError):
            pass
        if det_id is None:


            self._set_bridge_target_label(None)
            return
        feature = self._bridge_feature_for_det_id(
            layer, det_id, with_geometry=True)
        if feature is None:


            self._set_bridge_target_label(self._qgis_bridge_target_idx)
            return
        try:
            layer.selectByIds([feature.id()])
        except (RuntimeError, AttributeError, TypeError):
            pass
        target_geom = feature.geometry()
        if target_geom is not None and not target_geom.isEmpty():
            self._frame_bridge_target(layer, target_geom)

        self._set_bridge_target_label(self._qgis_bridge_target_idx)

    def _frame_bridge_target(self, layer, geom) -> None:












        try:
            from qgis.core import (
                QgsCoordinateTransform,
                QgsProject,
                QgsRectangle,
            )
            canvas = self.iface.mapCanvas()
            settings = canvas.mapSettings()
            bbox = geom.boundingBox()
            layer_crs = layer.crs()
            canvas_crs = settings.destinationCrs()
            if layer_crs.authid() and canvas_crs.authid() != layer_crs.authid():
                xform = QgsCoordinateTransform(
                    layer_crs, canvas_crs, QgsProject.instance())
                bbox = xform.transformBoundingBox(bbox)
            if bbox.isEmpty():
                return
            view = canvas.extent()
            if view.intersects(bbox):
                return




            if bbox.width() <= view.width() and bbox.height() <= view.height():
                half_w, half_h = view.width() / 2.0, view.height() / 2.0
                cx, cy = bbox.center().x(), bbox.center().y()
                framed = QgsRectangle(cx - half_w, cy - half_h,
                                      cx + half_w, cy + half_h)
            else:
                from ...core.server_dials import dial_in_range
                pad_factor = dial_in_range("tuning.agent.bridge_frame_pad_factor", 1.5, 1.0, 3.0)
                pad = max(bbox.width(), bbox.height()) * pad_factor or 1.0
                framed = QgsRectangle(
                    bbox.xMinimum() - pad, bbox.yMinimum() - pad,
                    bbox.xMaximum() + pad, bbox.yMaximum() + pad)
            canvas.setExtent(framed)
            canvas.refresh()
        except (RuntimeError, AttributeError, TypeError, ImportError):

            pass

    def _set_bridge_target_label(self, idx: int | None) -> None:

        label = ""
        try:
            if idx is not None:
                review = getattr(self, "_auto_review", None) or {}
                label = str(review.get("prompt") or "").strip()
        except (RuntimeError, AttributeError, TypeError):
            label = ""
        dock = getattr(self, "dock_widget", None)
        fn = getattr(dock, "set_qgis_bridge_target", None)
        if callable(fn):
            try:
                fn(label)
            except (RuntimeError, AttributeError, TypeError):
                pass

    def _set_bridge_shape_tools_visible(self, visible: bool) -> None:






        dock = getattr(self, "dock_widget", None)
        if dock is None:
            return
        setter = getattr(dock, "set_qgis_bridge_tools_visible", None)
        if callable(setter):
            try:
                setter(bool(visible))
            except (RuntimeError, AttributeError, TypeError):
                pass
            return
        buttons = getattr(dock, "_qgis_bridge_tool_buttons", None) or {}
        for key in _BRIDGE_SHAPE_TOOLS:
            button = buttons.get(key)
            if button is None:
                continue
            try:
                button.setVisible(bool(visible))
            except (RuntimeError, AttributeError):
                pass

    def activate_qgis_bridge_tool(self, tool: str) -> None:






        if not getattr(self, "_qgis_bridge_active", False):
            return
        layer = self._qgis_bridge_layer
        if layer is None or not self._is_layer_valid(layer):
            return
        tool = str(tool).lower()
        if (tool in _BRIDGE_SHAPE_TOOLS
                and getattr(self, "_qgis_bridge_target_det_id", None) is None):



            return
        try:
            self.iface.setActiveLayer(layer)
        except (RuntimeError, AttributeError):
            pass

        action = None
        if tool == "vertex":
            action = self._bridge_iface_action(
                "actionVertexToolActiveLayer", "actionVertexTool")
        elif tool == "split":
            action = self._bridge_iface_action("actionSplitFeatures")
        elif tool == "reshape":
            action = self._bridge_named_action("mActionReshapeFeatures")
        elif tool == "add":





            action = (self._bridge_iface_action("actionAddFeature") or self._bridge_named_action("mActionAddFeature"))
        if action is None:
            return
        try:
            action.trigger()
        except (RuntimeError, AttributeError):
            return







        self._bridge_cancel_capture()
        try:
            self.dock_widget.set_qgis_bridge_tool(tool)
        except (RuntimeError, AttributeError):
            pass
        if tool == "vertex":
            self._hide_bridge_opened_vertex_editors()
            try:
                from qgis.PyQt.QtCore import QTimer
                QTimer.singleShot(0, self._hide_bridge_opened_vertex_editors)
            except (RuntimeError, AttributeError):
                pass

    def _on_add_polygon_requested(self) -> None:








        if not getattr(self, "_qgis_bridge_active", False):
            self._qgis_bridge_add_entry = True
            try:
                self.enter_qgis_edit_bridge()
            finally:
                self._qgis_bridge_add_entry = False



        if (not getattr(self, "_qgis_bridge_active", False)
                and getattr(self, "_qgis_bridge_isolation_refused", False)):



            self._correct_selected_idx = None
            self.enter_qgis_edit_bridge()
        if not getattr(self, "_qgis_bridge_active", False):
            return
        self.activate_qgis_bridge_tool("add")

    def _bridge_iface_action(self, *method_names):

        for method_name in method_names:
            getter = getattr(self.iface, method_name, None)
            if getter is None:
                continue
            try:
                action = getter()
                if action is not None:
                    return action
            except (RuntimeError, AttributeError, TypeError):
                pass
        return None

    def _bridge_named_action(self, object_name: str):

        try:
            return self.iface.mainWindow().findChild(QAction, object_name)
        except (RuntimeError, AttributeError, TypeError):
            return None





    def _resolve_bridge_layer(self):



        layer = getattr(self, "_auto_selection_layer", None)
        if layer is None or not self._is_layer_valid(layer):
            return None
        from qgis.core import QgsVectorLayer
        if not isinstance(layer, QgsVectorLayer):
            return None
        return layer

    def _bridge_feature_for_det_id(self, layer, det_id, with_geometry=False):







        try:
            from qgis.core import QgsFeatureRequest
            request = QgsFeatureRequest()
            request.setFilterExpression(f'"det_id" = {int(det_id):d}')
            request.setLimit(1)
            if not with_geometry:
                scope = getattr(QgsFeatureRequest, "Flag", QgsFeatureRequest)
                no_geometry = getattr(scope, "NoGeometry", None)
                if no_geometry is not None:
                    request.setFlags(no_geometry)
            for feature in layer.getFeatures(request):
                return feature
            return None
        except (RuntimeError, AttributeError, TypeError, ValueError, ImportError):

            pass
        try:
            for feature in layer.getFeatures():
                if feature["det_id"] == det_id:
                    return feature
        except (RuntimeError, AttributeError, KeyError, TypeError):
            return None
        return None

    def _expose_and_activate_bridge_layer(self, layer) -> bool:








        try:
            from qgis.core import QgsMapLayer

            original = layer.flags()
            self._qgis_bridge_layer_flags = original
            private = getattr(QgsMapLayer.LayerFlag, "Private", None)
            if private is not None and original & private:
                layer.setFlags(flags_without(original, private))



            view = self.iface.layerTreeView()
            if view is not None:
                view.setCurrentLayer(layer)
            self.iface.setActiveLayer(layer)
            active = self.iface.activeLayer()
            return active is not None and active.id() == layer.id()
        except (RuntimeError, AttributeError, TypeError):
            return False

    def _restore_bridge_layer_presentation(self, layer) -> None:


        previous = getattr(self, "_qgis_bridge_prev_layer", None)
        self._qgis_bridge_prev_layer = None
        flags = getattr(self, "_qgis_bridge_layer_flags", None)
        try:
            if layer is not None and flags is not None and self._is_layer_valid(layer):
                layer.setFlags(flags)
        except (RuntimeError, AttributeError, TypeError):
            pass
        try:
            if previous is not None and self._is_layer_valid(previous):
                self.iface.setActiveLayer(previous)
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _snapshot_bridge_layer(self, layer) -> dict[int, bytes]:

        snapshot: dict[int, bytes] = {}
        try:
            fields = {field.name() for field in layer.fields()}
            if "det_id" not in fields:
                return snapshot
            for feature in layer.getFeatures():
                det_id = feature["det_id"]
                geom = feature.geometry()
                if (not isinstance(det_id, int) or det_id < 0 or geom is None or geom.isEmpty() or det_id in snapshot):
                    continue
                snapshot[det_id] = bytes(geom.asWkb())
        except (RuntimeError, AttributeError, TypeError, KeyError):
            return {}
        return snapshot





    def _remember_bridge_vertex_editor_visibility(self) -> None:

        self._qgis_bridge_vertex_dock_visibility = {}
        for dock in self._bridge_vertex_editor_docks():
            try:
                self._qgis_bridge_vertex_dock_visibility[id(dock)] = (
                    dock, bool(dock.isVisible()))
            except (RuntimeError, AttributeError):
                pass

    def _hide_bridge_opened_vertex_editors(self) -> None:

        if not getattr(self, "_qgis_bridge_active", False):
            return
        saved = self._qgis_bridge_vertex_dock_visibility
        for dock in self._bridge_vertex_editor_docks():
            prior = saved.get(id(dock))
            if prior is None:
                saved[id(dock)] = (dock, False)
                was_visible = False
            else:
                was_visible = prior[1]
            if was_visible:
                continue
            try:
                dock.setVisible(False)
            except (RuntimeError, AttributeError):
                pass

    def _restore_bridge_vertex_editor_visibility(self) -> None:

        for dock, visible in self._qgis_bridge_vertex_dock_visibility.values():
            try:
                dock.setVisible(bool(visible))
            except (RuntimeError, AttributeError):
                pass

    def _sync_bridge_delete_corner(self, class_name: str) -> None:






        dock = getattr(self, "dock_widget", None)
        setter = getattr(dock, "set_qgis_bridge_delete_corner_visible", None)
        if setter is None:
            return
        picked = class_name in _VERTEX_TOOL_CLASSES and self._bridge_vertex_is_locked()
        try:
            setter(picked)
        except (RuntimeError, AttributeError):
            pass

    def _bridge_vertex_is_locked(self) -> bool:




        try:
            from qgis.PyQt.QtWidgets import QTableView
        except ImportError:
            return False
        for dock in self._bridge_vertex_editor_docks():
            try:
                for view in dock.findChildren(QTableView):
                    model = view.model()
                    if model is not None and model.rowCount() > 0:
                        return True
            except (RuntimeError, AttributeError, TypeError):
                continue
        return False

    def _bridge_vertex_editor_docks(self) -> list:


        return vertex_editor_docks(self)

    def _restore_bridge_map_tool(self) -> None:







        prev = self._qgis_bridge_prev_maptool
        try:
            canvas = self.iface.mapCanvas()
            if prev is not None:
                canvas.setMapTool(prev)
            else:
                current = canvas.mapTool()
                if current is not None:
                    canvas.unsetMapTool(current)
        except (RuntimeError, AttributeError):
            pass
