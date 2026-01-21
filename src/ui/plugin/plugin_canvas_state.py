







from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog


class CanvasStateMixin:





    @staticmethod
    def _safe_remove_rubber_band(rb):

        if rb is None:
            return
        try:

            scene = rb.scene()
            if scene is not None:
                scene.removeItem(rb)
        except (RuntimeError, AttributeError):
            pass

    def _is_layer_valid(self, layer=None) -> bool:

        if layer is None:
            layer = self._current_layer
        if layer is None:
            return False
        try:
            layer.id()
            return True
        except RuntimeError:
            return False

    def _is_layer_georeferenced(self, layer) -> bool:


        from .shared import is_layer_georeferenced
        return is_layer_georeferenced(layer)

    @staticmethod
    def _needs_canvas_render(layer) -> bool:







        if layer is None:
            return False
        try:
            provider = layer.dataProvider()
            if provider is None:
                return False
            from ...core.feature_encoder import CANVAS_RENDERED_PROVIDERS
            return provider.name() in CANVAS_RENDERED_PROVIDERS
        except (RuntimeError, AttributeError):
            return False

    def _ensure_polygon_rubberband_sync(self):

        n_polygons = len(self.saved_polygons)
        n_bands = len(self.saved_rubber_bands)
        if n_polygons != n_bands:
            QgsMessageLog.logMessage(
                f"BUG: polygon/rubber band mismatch: {n_polygons} vs {n_bands}. "
                "Truncating to min. Please report.",
                "AI Segmentation",
                level=Qgis.MessageLevel.Critical
            )
            min_len = min(n_polygons, n_bands)
            while len(self.saved_rubber_bands) > min_len:
                rb = self.saved_rubber_bands.pop()
                self._safe_remove_rubber_band(rb)
            self.saved_polygons = self.saved_polygons[:min_len]

    @staticmethod
    def _compute_simplification_tolerance(transform_info, simplify_value):




        if simplify_value <= 0 or transform_info is None:
            return 0
        bbox = transform_info.get("bbox", [0, 1, 0, 1])
        img_shape = transform_info.get("img_shape", (1024, 1024))
        width_pixels = max(img_shape[1], 1)
        bbox_width = bbox[1] - bbox[0]
        if bbox_width == 0:
            return 0
        pixel_size = bbox_width / width_pixels







        height_pixels = max(img_shape[0], 1)
        bbox_height = bbox[3] - bbox[2]
        if bbox_height != 0:
            pixel_size = min(pixel_size, bbox_height / height_pixels)


        return pixel_size * simplify_value
