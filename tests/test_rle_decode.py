"""Tests for COCO RLE decoding (no QGIS dependency)."""
import sys
import types

import numpy as np

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
    {
        "MessageLevel": type(
            "ML", (), {"Info": 0, "Warning": 1}
        )()
    },
)()

from src.core.pro_predictor import decode_rle_to_mask  # noqa: E402


def test_single_pixel_top_left():
    """2x2 mask, only top-left pixel is foreground."""
    rle = {"counts": [0, 1, 3], "size": [2, 2]}
    mask = decode_rle_to_mask(rle)
    assert mask.shape == (2, 2)
    assert mask.dtype == bool
    assert mask[0, 0] is np.True_
    assert mask[1, 0] is np.False_
    assert mask[0, 1] is np.False_
    assert mask[1, 1] is np.False_


def test_all_foreground():
    """3x3 mask, all pixels foreground."""
    rle = {"counts": [0, 9], "size": [3, 3]}
    mask = decode_rle_to_mask(rle)
    assert mask.shape == (3, 3)
    assert mask.all()


def test_all_background():
    """3x3 mask, all pixels background."""
    rle = {"counts": [9], "size": [3, 3]}
    mask = decode_rle_to_mask(rle)
    assert mask.shape == (3, 3)
    assert not mask.any()


def test_checkerboard_column_order():
    """2x2 mask with Fortran-order checkerboard pattern."""
    rle = {"counts": [0, 1, 2, 1], "size": [2, 2]}
    mask = decode_rle_to_mask(rle)
    assert mask[0, 0] is np.True_
    assert mask[1, 0] is np.False_
    assert mask[0, 1] is np.False_
    assert mask[1, 1] is np.True_
