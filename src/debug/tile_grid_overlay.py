"""Debug overlay that visualizes the tile grid on the QGIS map canvas.

Creates two temporary memory vector layers:
- Tile Grid: contour-only polygons for each 1024x1024 tile
- Overlap Zones: semi-transparent polygons where tiles overlap
"""

from typing import Optional

from qgis.core import (
    QgsFeature,
    QgsFillSymbol,
    QgsGeometry,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
)


class TileGridOverlay:
    """Snapshot-based tile grid visualization for debugging."""

    def __init__(self, tile_manager):
        self._tile_manager = tile_manager
        self._grid_layer_id: Optional[str] = None
        self._overlap_layer_id: Optional[str] = None

    def show(
        self,
        layer: QgsRasterLayer,
        zone: Optional[QgsRectangle] = None,
    ) -> None:
        """Compute tile grid and display as memory vector layers."""
        self.hide()

        extent = layer.extent()
        px_w = extent.width() / layer.width()
        px_h = extent.height() / layer.height()

        if zone:
            # Convert zone to pixel dimensions
            zone_x_px = max(0, int((zone.xMinimum() - extent.xMinimum()) / px_w))
            zone_y_px = max(0, int((extent.yMaximum() - zone.yMaximum()) / px_h))
            pixel_w = int(zone.width() / px_w)
            pixel_h = int(zone.height() / px_h)
            origin_x_px = zone_x_px
            origin_y_px = zone_y_px
        else:
            pixel_w = layer.width()
            pixel_h = layer.height()
            origin_x_px = 0
            origin_y_px = 0

        tiles = self._tile_manager.compute_grid(pixel_w, pixel_h)
        if tiles is None:
            return

        crs = layer.crs().authid()

        # Build tile rectangles in geo coordinates
        tile_rects = []
        for x, y, w, h in tiles:
            abs_x = origin_x_px + x
            abs_y = origin_y_px + y
            minx = extent.xMinimum() + abs_x * px_w
            maxx = minx + w * px_w
            maxy = extent.yMaximum() - abs_y * px_h
            miny = maxy - h * px_h
            tile_rects.append(QgsRectangle(minx, miny, maxx, maxy))

        # Create tile grid layer
        grid_layer = self._create_layer(
            "Tile Grid", crs, tile_rects, self._grid_style()
        )
        self._grid_layer_id = grid_layer.id()

        # Compute and create overlap zones
        overlap_rects = self._compute_overlaps(tile_rects)
        if overlap_rects:
            overlap_layer = self._create_layer(
                "Overlap Zones", crs, overlap_rects, self._overlap_style()
            )
            self._overlap_layer_id = overlap_layer.id()

    def hide(self) -> None:
        """Remove the tile grid and overlap layers from the project."""
        project = QgsProject.instance()
        for layer_id in (self._grid_layer_id, self._overlap_layer_id):
            if layer_id and project.mapLayer(layer_id):
                project.removeMapLayer(layer_id)
        self._grid_layer_id = None
        self._overlap_layer_id = None

    def is_visible(self) -> bool:
        """Return True if the grid layers are currently displayed."""
        if not self._grid_layer_id:
            return False
        return QgsProject.instance().mapLayer(self._grid_layer_id) is not None

    # ── Private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _create_layer(
        name: str,
        crs: str,
        rects: list[QgsRectangle],
        symbol: QgsFillSymbol,
    ) -> QgsVectorLayer:
        """Create a memory vector layer with polygon features from rectangles."""
        uri = f"Polygon?crs={crs}"
        vl = QgsVectorLayer(uri, name, "memory")
        pr = vl.dataProvider()

        features = []
        for rect in rects:
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromRect(rect))
            features.append(feat)

        pr.addFeatures(features)
        vl.updateExtents()

        renderer = QgsSingleSymbolRenderer(symbol)
        vl.setRenderer(renderer)

        QgsProject.instance().addMapLayer(vl)
        return vl

    @staticmethod
    def _compute_overlaps(
        rects: list[QgsRectangle],
    ) -> list[QgsRectangle]:
        """Find intersection rectangles between all pairs of adjacent tiles."""
        overlaps = []
        seen = set()
        for i, r1 in enumerate(rects):
            for j, r2 in enumerate(rects):
                if i >= j:
                    continue
                pair = (i, j)
                if pair in seen:
                    continue
                seen.add(pair)
                if r1.intersects(r2):
                    inter = r1.intersect(r2)
                    if inter.width() > 0 and inter.height() > 0:
                        overlaps.append(inter)
        return overlaps

    @staticmethod
    def _grid_style() -> QgsFillSymbol:
        """Tile grid style: yellow contour, no fill."""
        return QgsFillSymbol.createSimple(
            {
                "color": "0,0,0,0",
                "outline_color": "#FFC107",
                "outline_width": "0.5",
            }
        )

    @staticmethod
    def _overlap_style() -> QgsFillSymbol:
        """Overlap zone style: semi-transparent orange fill."""
        return QgsFillSymbol.createSimple(
            {
                "color": "255,152,0,40",
                "outline_color": "#FF9800",
                "outline_width": "0.3",
            }
        )
