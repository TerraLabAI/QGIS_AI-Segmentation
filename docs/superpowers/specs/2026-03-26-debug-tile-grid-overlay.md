# Debug Tile Grid Overlay

**Date:** 2026-03-26
**Branch:** LA-fal.ai-features
**Status:** Design approved

---

## Problem

When working with rasters of different resolutions (drone ortho vs satellite), the tile count is invisible. A 20000x15000 px drone image produces hundreds of tiles while a satellite crop produces 4-9. There is no way to visually understand the tiling without reading logs.

## Solution

A debug-only toggle button in the PRO dock that creates temporary vector layers showing the tile grid and overlap zones on the map canvas.

## Behavior

- **Trigger:** Small button at the bottom of the PRO dock ("Show tile grid")
- **Mode:** Snapshot — click computes and displays, re-click removes
- **Source:** Raster extent if no zone, zone extent if defined (same as `estimate_credits`)
- **Cleanup:** Layers removed on dock close
- **No live update:** Grid stays fixed until re-click

## Architecture

### Single isolated module: `src/debug/tile_grid_overlay.py`

```python
class TileGridOverlay:
    def __init__(self, tile_manager: TileManager):
        ...

    def show(self, layer: QgsRasterLayer, zone: Optional[QgsRectangle]) -> None:
        """Compute tile grid and create two memory vector layers."""
        ...

    def hide(self) -> None:
        """Remove the tile grid and overlap layers from the project."""
        ...

    def is_visible(self) -> bool:
        """Return True if the grid layers are currently displayed."""
        ...
```

**Dependencies:** Only `TileManager.compute_grid()` and QGIS core API.

### Two vector layers

1. **Tile Grid** — one polygon per tile, contour only
2. **Overlap Zones** — polygons covering the overlap bands between adjacent tiles

Both are `QgsVectorLayer("Polygon?crs=...", name, "memory")` added to `QgsProject.instance()`.

### Pixel-to-geo conversion

```python
extent = layer.extent()
px_w = extent.width() / layer.width()
px_h = extent.height() / layer.height()

# For each tile (x, y, w, h) from compute_grid():
minx = extent.xMinimum() + x * px_w
miny = extent.yMaximum() - (y + h) * px_h
maxx = minx + w * px_w
maxy = miny + h * px_h
```

If a zone is defined, offset by zone origin in pixel space.

### Style

- **Tiles:** contour `#424242`, width 1px, no fill
- **Overlap:** fill `rgba(100, 100, 100, 0.12)`, contour `#757575`, width 0.5px

Applied programmatically via `QgsSymbol` / `QgsFillSymbol`.

### Overlap zone computation

For each pair of adjacent tiles that overlap, compute the intersection rectangle. The overlap fraction is 15% (stride = 870 for tile_size 1024), so overlap bands are ~154px wide.

### Integration in dock (minimal)

- One `QPushButton("Show tile grid")` at the bottom of the dock, neutral style
- Toggle logic: `if overlay.is_visible(): overlay.hide() else: overlay.show(layer, zone)`
- On dock close / `reset_session()`: `overlay.hide()`

## Not in scope

- Labels on tiles (no tile_id, col, row displayed)
- Attribute table data beyond geometry
- Live update on raster/zone change
- User-facing feature (debug only)
