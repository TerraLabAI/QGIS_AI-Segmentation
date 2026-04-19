




from __future__ import annotations

from qgis.core import QgsCoordinateTransform, QgsGeometry, QgsProject, QgsRasterLayer

from .core.qt_compat import geometry_op_succeeded
from .mcp_api_guard import gui_thread_only, zone_geometry_from_wkt


def _transform_shape(shape, source, target):
    result = QgsGeometry(shape)
    if source is None or target is None or not source.isValid() or not target.isValid():
        raise ValueError("The drawn geometry needs a valid coordinate reference system.")
    if source != target and not geometry_op_succeeded(result.transform(
            QgsCoordinateTransform(source, target, QgsProject.instance()))):
        raise ValueError("The drawn geometry could not be transformed to the imagery CRS.")
    return result


class SegmentationInteractiveMixin:


    def _automatic_readiness(self):
        from .core.activation_manager import is_plugin_activated
        from .core.served_config import served_config_ready

        layer = self._resolve_raster_layer(None)
        rasters = [item for item in QgsProject.instance().mapLayers().values()
                   if isinstance(item, QgsRasterLayer)]
        signed_in = is_plugin_activated()
        settings_ready = served_config_ready()
        state, action = "READY", None
        if not rasters:
            state, action = "NO_RASTER_LAYER", "Load satellite or aerial imagery first."
        elif not signed_in:
            state, action = "NOT_ACTIVATED", "Open AI Segmentation and sign in."
        elif not settings_ready:
            state, action = "SETTINGS_NOT_LOADED", "Open AI Segmentation to load the cloud settings."
        return {
            "ready": state == "READY", "state": state, "action_required": action,
            "mode": "automatic", "requires_local_model": False,
            "signed_in": signed_in, "settings_ready": settings_ready,
            "source": self._interactive_source(layer),
            "available_raster_layers": [self._interactive_source(item) for item in rasters],
        }

    @staticmethod
    def _interactive_source(layer):
        if layer is None:
            return None
        return {"id": layer.id(), "name": layer.name(), "crs": layer.crs().authid() or layer.crs().toWkt()}

    @gui_thread_only
    def get_interactive_state(self) -> dict:









        plugin = self._plugin
        dock = getattr(plugin, "dock_widget", None)
        layer = self._resolve_raster_layer(None)
        source = self._interactive_source(layer)
        zone = getattr(plugin, "_auto_zone", None)
        zone_wkt, zone_crs, exemplars = None, None, []
        store = getattr(plugin, "_auto_exemplar_store", None)
        examples_match_source = True
        positives = store.positives() if store else 0
        negatives = store.excludes() if store else 0
        if zone is not None:
            stored_crs = plugin._zone_source_crs(zone)
            target_crs = layer.crs() if layer is not None else stored_crs
            shape = plugin._shared_zone_shape(zone)
            zone_wkt = _transform_shape(shape, stored_crs, target_crs).asWkt()
            zone_crs = target_crs.authid() or target_crs.toWkt()
            for example in store.list() if store else []:
                if example.region:
                    continue
                captured_source = getattr(example, "source_layer_id", None) or example.stamp_layer_id
                if captured_source and (source is None or source["id"] != captured_source):
                    examples_match_source = False
                shape = example.polygon
                precise = shape is not None and not shape.isEmpty()
                if not precise:
                    shape = QgsGeometry.fromRect(example.map_rect)
                shape = _transform_shape(shape, stored_crs, target_crs)
                box = shape.boundingBox()
                item = {"bbox": [box.xMinimum(), box.yMinimum(), box.xMaximum(), box.yMaximum()],
                        "label": int(example.label)}
                if precise:
                    item["polygon_wkt"] = shape.asWkt()
                exemplars.append(item)
        prompt = dock.auto_prompt_input.text().strip() if dock is not None else ""
        mode = getattr(getattr(dock, "_mode", None), "value", None)
        busy = self._agent_zone_run_in_flight()
        review = getattr(plugin, "_auto_review", None) is not None
        canvas_tool = plugin.iface.mapCanvas().mapTool()
        example_tool = getattr(plugin, "_exemplar_maptool", None)
        zone_tool = getattr(plugin, "_zone_selection_tool", None)
        drawing = ("example" if example_tool is not None and canvas_tool is example_tool
                   else "zone" if zone_tool is not None and canvas_tool is zone_tool else None)
        from .core.detect_gate import can_detect
        return {
            "mode": mode, "source": source, "zone_wkt": zone_wkt, "zone_crs": zone_crs,
            "object_class": prompt, "exemplars": exemplars,
            "positive_examples": positives, "negative_examples": negatives,
            "running": busy, "review_pending": review,
            "drawing": drawing, "examples_match_source": examples_match_source,
            "ready_to_detect": bool(source and zone_wkt and can_detect(bool(prompt), positives)
                                    and not busy and not review and not drawing and examples_match_source),
            "automatic": self._automatic_readiness(),
        }

    @gui_thread_only
    def prepare_interactive(self, layer_name: str | None = None,
                            object_class: str | None = None,
                            zone_wkt: str | None = None,
                            interaction: str = "review") -> dict:




















        if interaction not in ("review", "draw_zone", "add_positive", "add_negative"):
            return {"_error": "interaction must be review, draw_zone, add_positive or add_negative."}
        if object_class is not None and not isinstance(object_class, str):
            return {"_error": "object_class must be a string or None."}
        if zone_wkt is not None and not isinstance(zone_wkt, str):
            return {"_error": "zone_wkt must be a WKT string or None."}
        if zone_wkt:
            _shape, error = zone_geometry_from_wkt(zone_wkt)
            if error:
                return error
        plugin = self._plugin
        if (self._agent_zone_run_in_flight() or getattr(plugin, "_auto_review", None) is not None
                or getattr(plugin, "_auto_worker", None) is not None):
            return {"_error": "Finish the current detection or review before preparing another run.", "busy": True}
        target = None
        if layer_name is not None:
            from .mcp_api import raster_layer_by_id_or_name
            target, error = raster_layer_by_id_or_name(layer_name)
            if error:
                return error
        previous = self._resolve_raster_layer(None)
        store = getattr(plugin, "_auto_exemplar_store", None)
        if target is not None and previous is not None and target.id() != previous.id() and store and store.count():
            return {"_error": "The drawn examples belong to another imagery layer. "
                    "Remove them in the panel before changing imagery."}
        plugin._ensure_dock_widget()
        dock = plugin.dock_widget
        if dock is None:
            return {"_error": "The AI Segmentation panel is unavailable."}
        from .ui.ai_segmentation_dockwidget import Mode
        if dock._mode != Mode.AUTOMATIC:
            switched = self.set_mode("automatic")
            if "_error" in switched:
                return switched
        if target is not None:
            node = QgsProject.instance().layerTreeRoot().findLayer(target.id())
            if node is not None:
                node.setItemVisibilityChecked(True)
            combo = dock.auto_layer_combo
            combo._refresh()
            combo.setLayer(target)
            selected = plugin._get_active_raster_layer()
            if selected is None or selected.id() != target.id():
                return {"_error": "The imagery could not be selected in AI Segmentation."}
        if zone_wkt and self._resolve_raster_layer(None) is None:
            return {"_error": "Select imagery before setting a zone in the imagery CRS."}
        if zone_wkt is not None:
            before = self.get_interactive_state()
            previous_zone = QgsGeometry.fromWkt(before.get("zone_wkt") or "")
            replacement = QgsGeometry.fromWkt(zone_wkt or "")
            zone = self.set_auto_zone(zone_wkt)
            if "_error" in zone:
                return zone
            if store and store.count() and not previous_zone.equals(replacement):
                plugin._clear_exemplars()
        if object_class is not None:
            dock.set_prompt_text(object_class, source="agent")
        dock.show()
        dock.raise_()
        if interaction == "draw_zone":
            plugin._activate_zone_drawing()
        elif interaction in ("add_positive", "add_negative"):
            if getattr(plugin, "_auto_zone", None) is None:
                return {"_error": "Draw the search area before adding an example."}
            label = int(interaction == "add_positive")
            if label == 0 and (not store or not store.positives()):
                return {"_error": "Draw a positive example before an exclude example."}


            if (getattr(plugin, "_exemplar_maptool", None) is not None
                    and getattr(plugin, "_pending_exemplar_label", None) != label):
                plugin._restore_maptool_after_exemplar()
            if getattr(plugin, "_exemplar_maptool", None) is None:
                plugin._on_add_exemplar_requested(label)
            if getattr(plugin, "_exemplar_maptool", None) is None:
                return {"_error": "The panel could not add this example. Check the example limit in the panel."}
        result = self.get_interactive_state()
        if "_error" not in result:
            result.update(prepared=True, inference_started=False, interaction=interaction)
        return result

    def _retained_auto_exemplars(self, zone_wkt, layer_name):

        state = self.get_interactive_state()
        if ("_error" in state or not state.get("exemplars") or not zone_wkt
                or not state.get("examples_match_source")):
            return None
        source = self._resolve_raster_layer(layer_name)
        if source is None or not state.get("source") or source.id() != state["source"]["id"]:
            return None
        wanted = QgsGeometry.fromWkt(zone_wkt)
        stored = QgsGeometry.fromWkt(state.get("zone_wkt") or "")
        if wanted.isNull() or stored.isNull() or not wanted.equals(stored):
            return None
        return state["exemplars"]
