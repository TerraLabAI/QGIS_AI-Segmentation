"""Tests for TileManager and IoU deduplication utility."""

import sys
import types

# Stub QGIS modules for tests that don't need full QGIS environment
for mod_name in ("qgis", "qgis.core"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Add QgsGeometry stub for dedup tests (these will fail gracefully without real QGIS)
if not hasattr(sys.modules["qgis.core"], "QgsGeometry"):
    sys.modules["qgis.core"].QgsGeometry = type(
        "QgsGeometry", (), {"fromWkt": staticmethod(lambda wkt: None)}
    )

from qgis.core import QgsGeometry  # noqa: E402


def test_deduplicate_geometries_removes_overlapping():
    """Two overlapping squares with IoU > 0.3 should deduplicate to one."""
    g1 = QgsGeometry.fromWkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")
    g2 = QgsGeometry.fromWkt("POLYGON((2 2, 12 2, 12 12, 2 12, 2 2))")
    from src.core.polygon_exporter import deduplicate_geometries

    accepted = deduplicate_geometries([g1, g2], iou_threshold=0.3)
    assert len(accepted) == 1


def test_deduplicate_geometries_keeps_non_overlapping():
    """Two non-overlapping squares should both be kept."""
    g1 = QgsGeometry.fromWkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")
    g2 = QgsGeometry.fromWkt("POLYGON((20 20, 30 20, 30 30, 20 30, 20 20))")
    from src.core.polygon_exporter import deduplicate_geometries

    accepted = deduplicate_geometries([g1, g2], iou_threshold=0.3)
    assert len(accepted) == 2


def test_no_tiling_for_small_image():
    """Image <= 1024x1024 should produce a single tile."""
    from src.core.tile_manager import TileManager

    tm = TileManager(tile_size=1024, overlap_fraction=0.15, max_tiles=50)
    tiles = tm.compute_grid(image_width=800, image_height=600)
    assert len(tiles) == 1
    assert tiles[0] == (0, 0, 800, 600)


def test_tiling_for_large_image():
    """Image 2048x2048 should produce multiple tiles with overlap."""
    from src.core.tile_manager import TileManager

    tm = TileManager(tile_size=1024, overlap_fraction=0.15, max_tiles=50)
    tiles = tm.compute_grid(image_width=2048, image_height=2048)
    assert len(tiles) > 1
    # Each tile should be at most 1024x1024
    for x, y, w, h in tiles:
        assert w <= 1024
        assert h <= 1024


def test_estimate_credits():
    """Credits should equal number of tiles."""
    from src.core.tile_manager import TileManager

    tm = TileManager(tile_size=1024, overlap_fraction=0.15, max_tiles=50)
    credits = tm.estimate_credits(image_width=3000, image_height=3000)
    tiles = tm.compute_grid(image_width=3000, image_height=3000)
    assert credits == len(tiles)


def test_max_tiles_cap():
    """Should return None when exceeding max_tiles."""
    from src.core.tile_manager import TileManager

    tm = TileManager(tile_size=1024, overlap_fraction=0.15, max_tiles=4)
    tiles = tm.compute_grid(image_width=10000, image_height=10000)
    assert tiles is None


def test_tiles_cover_full_image():
    """Union of all tiles should cover the entire image."""
    from src.core.tile_manager import TileManager

    tm = TileManager(tile_size=1024, overlap_fraction=0.15, max_tiles=50)
    w, h = 3000, 2000
    tiles = tm.compute_grid(image_width=w, image_height=h)
    max_x = max(x + tw for x, y, tw, th in tiles)
    max_y = max(y + th for x, y, tw, th in tiles)
    assert max_x >= w
    assert max_y >= h
