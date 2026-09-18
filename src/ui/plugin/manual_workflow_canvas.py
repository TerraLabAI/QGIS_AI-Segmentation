







from __future__ import annotations

from qgis.core import Qgis, QgsCoordinateTransform, QgsCsException, QgsGeometry, QgsMessageLog, QgsPointXY, QgsProject


class ManualWorkflowCanvasMixin:





    def _rebuild_manual_crs_transforms(self) -> None:





        self._canvas_to_raster_xform = None
        self._raster_to_canvas_xform = None
        layer = self._current_layer
        if layer is None:
            return
        try:
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            raster_crs = layer.crs()
            if not (canvas_crs.isValid() and raster_crs.isValid()):
                return
            if canvas_crs == raster_crs:
                return
            project = QgsProject.instance()
            self._canvas_to_raster_xform = QgsCoordinateTransform(
                canvas_crs, raster_crs, project)
            self._raster_to_canvas_xform = QgsCoordinateTransform(
                raster_crs, canvas_crs, project)
            QgsMessageLog.logMessage(
                f"CRS transform enabled: {canvas_crs.authid()} -> {raster_crs.authid()}",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info
            )
        except (RuntimeError, AttributeError):
            self._canvas_to_raster_xform = None
            self._raster_to_canvas_xform = None

    def _start_canvas_crs_watch(self) -> None:


        if getattr(self, "_canvas_crs_watch_on", False):
            return
        try:
            self.iface.mapCanvas().destinationCrsChanged.connect(
                self._on_canvas_crs_changed)
        except (RuntimeError, AttributeError):
            return
        self._canvas_crs_watch_on = True

    def _stop_canvas_crs_watch(self) -> None:

        if not getattr(self, "_canvas_crs_watch_on", False):
            return
        self._canvas_crs_watch_on = False
        try:
            self.iface.mapCanvas().destinationCrsChanged.disconnect(
                self._on_canvas_crs_changed)
        except (TypeError, RuntimeError, AttributeError):
            pass

    def _on_canvas_crs_changed(self) -> None:








        if self._current_layer is None:
            self._stop_canvas_crs_watch()
            return
        self._rebuild_manual_crs_transforms()
        self._reproject_session_canvas_items()

    def _reproject_session_canvas_items(self) -> None:








        try:
            if self.map_tool is not None:
                self.map_tool.clear_markers()
                points = [(pt, True) for pt in self._active_crop_points_positive]
                points += [(pt, False) for pt in self._active_crop_points_negative]
                for raw, is_positive in points:
                    canvas_pt = self._transform_to_canvas_crs(
                        QgsPointXY(raw[0], raw[1]))
                    if canvas_pt is not None:
                        self.map_tool.add_marker(canvas_pt, is_positive=is_positive)
        except (QgsCsException, RuntimeError, AttributeError, TypeError, IndexError):

            pass
        for entry, band in zip(self.saved_polygons, self.saved_rubber_bands):
            if band is None:
                continue
            geometry = entry.get("geom_obj")
            if geometry is None:
                continue
            try:
                display_geom = QgsGeometry(geometry)
                self._transform_geometry_to_canvas_crs(display_geom)
                band.setToGeometry(display_geom, None)
            except (QgsCsException, RuntimeError, AttributeError):


                continue
        try:
            self._update_mask_visualization()
        except (QgsCsException, RuntimeError, AttributeError):

            pass

    def _active_space_pan_tool(self):


        try:
            current = self.iface.mapCanvas().mapTool()
        except (RuntimeError, AttributeError):
            return None
        for tool in (self.map_tool, self._zone_selection_tool,
                     getattr(self, "_exemplar_maptool", None)):
            if tool is not None and current == tool:
                return tool
        return None
