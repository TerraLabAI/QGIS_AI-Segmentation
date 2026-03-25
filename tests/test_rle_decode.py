"""Tests for RLE decoding from remote endpoint (no QGIS dependency)."""

import sys
import types

# Stub QGIS modules so pro_predictor can be imported outside QGIS
for mod_name in ("qgis", "qgis.core"):
    sys.modules[mod_name] = types.ModuleType(mod_name)
sys.modules["qgis.core"].QgsMessageLog = type(
    "QgsMessageLog",
    (),
    {"logMessage": staticmethod(lambda *a, **kw: None)},
)()
sys.modules["qgis.core"].Qgis = type(
    "Qgis",
    (),
    {"MessageLevel": type("ML", (), {"Info": 0, "Warning": 1})()},
)()

from src.core.pro_predictor import decode_rle_to_mask  # noqa: E402


def test_single_run():
    """5x5 mask with a single run of 3 pixels starting at offset 2."""
    rle_str = "2 3"
    mask = decode_rle_to_mask(rle_str, 5, 5)
    assert mask.shape == (5, 5)
    assert mask.dtype == bool
    flat = mask.flatten()
    assert not flat[0] and not flat[1]
    assert flat[2] and flat[3] and flat[4]
    assert not flat[5]


def test_multiple_runs():
    """4x4 mask with two runs."""
    rle_str = "0 2 8 4"
    mask = decode_rle_to_mask(rle_str, 4, 4)
    assert mask.shape == (4, 4)
    flat = mask.flatten()
    # First run: positions 0-1
    assert flat[0] and flat[1]
    assert not flat[2]
    # Second run: positions 8-11
    assert flat[8] and flat[9] and flat[10] and flat[11]


def test_all_foreground():
    """3x3 mask, all pixels foreground."""
    rle_str = "0 9"
    mask = decode_rle_to_mask(rle_str, 3, 3)
    assert mask.shape == (3, 3)
    assert mask.all()


def test_empty_rle():
    """3x3 mask, empty RLE string means all background."""
    rle_str = ""
    mask = decode_rle_to_mask(rle_str, 3, 3)
    assert mask.shape == (3, 3)
    assert not mask.any()
