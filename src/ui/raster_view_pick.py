"""Pick the raster the user is actually looking at.

The layer drop-down used to take the FIRST raster in layer-tree order whose
extent touched the map view. Two facts make that the wrong answer most of the
time: a world basemap touches every view, and QGIS drops a newly added XYZ
layer at the TOP of the tree. So "Google Satellite" beat the orthophoto loaded
right underneath it, on every plugin reload.

What decides the pick here, strongest first:

1. How much of the map view the raster covers, in bands (the tier constants
   below). A raster filling the view beats one showing a sliver in a corner.
2. Whether the raster is a world-wide backdrop. A global basemap covers every
   view and so says nothing about where the user is working: it loses to any
   local raster that is genuinely in view.
3. The layer selected in the QGIS Layers panel.
4. Three bands or more: the model reads colour imagery, so a one-band terrain
   model or mask is a poorer guess than an RGB raster in the same band.
5. Layer-tree order, topmost first.
"""
from __future__ import annotations

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
)

# View share bands. A raster filling at least half the view is what the user is
# looking at; under a twentieth is a corner sliver and does not count as data
# in view.
VIEW_SHARE_FILLS = 0.5
VIEW_SHARE_PARTIAL = 0.05

# Degrees an extent must span to read as world-wide. A tile basemap reprojects
# to the full Mercator square (360 by about 170 degrees); no national or
# regional orthophoto comes close, and neither does a WMS that publishes its
# real bounding box.
_WORLD_SPAN_LON = 300.0
_WORLD_SPAN_LAT = 130.0

TIER_FILLS_VIEW = 4  # local raster over half the view
TIER_IN_VIEW = 3  # local raster, a usable share of the view
TIER_BACKDROP = 2  # world basemap: everywhere, so it says nothing
TIER_SLIVER = 1  # local raster clipping a corner
TIER_OUT_OF_VIEW = 0  # nothing in view, or no extent to judge by


def _reproject_extent(extent, source_crs, target_crs) -> QgsRectangle | None:
    """Extent in ``target_crs``, or None when there is nothing to compare.

    None on a failed or impossible transform, never the raw extent: coordinates
    in the wrong frame compare as a real overlap and would hand the pick to a
    raster that is nowhere near the view. Every caller treats None as "cannot
    judge", which drops the raster to tree order instead of ranking it on a
    made-up number.
    """
    if extent is None or extent.isEmpty():
        return None
    if source_crs == target_crs:
        return extent  # same frame, valid or not: raw coordinates are comparable
    if not source_crs.isValid() or not target_crs.isValid():
        return None
    try:
        transform = QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance())
        reprojected = transform.transformBoundingBox(extent)
    except Exception:
        return None
    return None if reprojected.isEmpty() else reprojected


def measure_view_share(layer: QgsRasterLayer, view_extent, view_crs) -> float:
    """Share of the map view the raster's extent covers, 0.0 when it misses."""
    if view_extent is None or view_extent.isEmpty():
        return 0.0
    view_area = view_extent.width() * view_extent.height()
    if view_area <= 0:
        return 0.0
    layer_extent = _reproject_extent(layer.extent(), layer.crs(), view_crs)
    if layer_extent is None:
        return 0.0
    overlap = layer_extent.intersect(view_extent)
    if overlap.isEmpty():
        return 0.0
    return min(1.0, (overlap.width() * overlap.height()) / view_area)


def raster_is_world_backdrop(layer: QgsRasterLayer) -> bool:
    """True when the raster spans the globe: a tile basemap, not the user's data."""
    extent = _reproject_extent(
        layer.extent(), layer.crs(), QgsCoordinateReferenceSystem("EPSG:4326"))
    if extent is None:
        return False
    return extent.width() >= _WORLD_SPAN_LON and extent.height() >= _WORLD_SPAN_LAT


def view_fit_tier(layer: QgsRasterLayer, view_extent, view_crs) -> int:
    """Which band of "is this what the user is looking at" the raster falls in."""
    share = measure_view_share(layer, view_extent, view_crs)
    if share <= 0.0:
        return TIER_OUT_OF_VIEW
    if raster_is_world_backdrop(layer):
        return TIER_BACKDROP
    if share >= VIEW_SHARE_FILLS:
        return TIER_FILLS_VIEW
    if share >= VIEW_SHARE_PARTIAL:
        return TIER_IN_VIEW
    return TIER_SLIVER


def rank_raster_for_view(
    layer: QgsRasterLayer,
    view_extent,
    view_crs,
    tree_order: int,
    active_layer_id: str | None = None,
) -> tuple[int, int, int, int]:
    """Sort key for one candidate raster, HIGHEST first.

    ``tree_order`` is the raster's position in the drop-down, 0 = topmost, so
    it enters the key negated and only ever breaks a tie.
    """
    tier = view_fit_tier(layer, view_extent, view_crs)
    is_active = 1 if active_layer_id and layer.id() == active_layer_id else 0
    try:
        looks_like_imagery = 1 if layer.bandCount() >= 3 else 0
    except Exception:
        looks_like_imagery = 0
    return (tier, is_active, looks_like_imagery, -tree_order)
