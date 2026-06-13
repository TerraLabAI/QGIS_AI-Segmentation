


















from __future__ import annotations

from qgis.core import QgsGeometry, QgsProject

from ...core.i18n import tr


class AutoZonePickMixin:


    def _on_auto_zone_source_picked(self, kind: str, layer_id: str) -> None:


        found = self._resolve_picked_zone(kind, layer_id)
        if found is None:
            self._refuse_picked_zone(
                tr("That zone is empty. Pick another one."))
            return
        geom, crs, label = found
        moved = self._picked_zone_in_canvas_crs(geom, crs)
        if moved is None:
            self._refuse_picked_zone(
                tr("That zone cannot be moved onto this map. "
                   "Draw it instead."))
            return


        self._drop_zone_sketch()


        self._auto_zone_pick_label = label
        self._on_zone_polygon_drawn(moved)

    def _resolve_picked_zone(self, kind: str, layer_id: str):



        from ...core import zone_of_interest as zoi

        project = QgsProject.instance()
        try:
            if kind == "zone":
                zone = zoi.read_zone(project)
                if zone is None:
                    return None
                layer = (project.mapLayer(zone.layer_id)
                         if zone.layer_id else None)
                name = zone.label or (layer.name() if layer is not None else "")
                return QgsGeometry(zone.geometry), zone.crs, str(name or "")
            layer = project.mapLayer(str(layer_id or ""))
            if layer is None:
                return None
            outline = zoi.outline_of_layer(
                layer, selected_only=(kind == "selection"))
            if outline is None:
                return None
            geom, crs, _approximate = outline
            return QgsGeometry(geom), crs, str(layer.name() or "")
        except Exception:  # noqa: BLE001
            return None

    def _picked_zone_in_canvas_crs(self, geom, crs):



        from ...core import zone_of_interest as zoi

        try:
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        except (RuntimeError, AttributeError):
            return None
        moved = zoi.to_crs(geom, crs, canvas_crs, QgsProject.instance())
        if moved is None or moved.isEmpty():
            return None
        return moved

    def _drop_zone_sketch(self) -> None:


        tool = getattr(self, "_zone_selection_tool", None)
        if tool is None:
            return
        try:
            tool.clear_selection()
        except (RuntimeError, AttributeError):
            pass

    def _refuse_picked_zone(self, message: str) -> None:



        if self.dock_widget:
            try:
                self.dock_widget.set_auto_zone_rejected(None)
                self.dock_widget.set_auto_zone_refusal(message)
                return
            except (RuntimeError, AttributeError):
                pass
        try:
            self.iface.messageBar().pushWarning("AI Segmentation", message)
        except (RuntimeError, AttributeError):
            pass
