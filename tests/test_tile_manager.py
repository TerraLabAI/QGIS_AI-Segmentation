"""Tests for IoU deduplication utility."""

from qgis.core import QgsGeometry


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
