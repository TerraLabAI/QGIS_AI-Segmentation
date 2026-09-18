







from __future__ import annotations

try:
    from .venv_manager import ensure_venv_packages_available
except ImportError:
    pass
else:
    ensure_venv_packages_available()

import numpy as np  # noqa: E402

from .detection_masks import (  # noqa: E402
    logger,
)









_BLANK_TILE_SAMPLE_PX: int = 32
_BLANK_TILE_DOMINANT_FRAC: float = 0.995



_BLANK_TILE_QUANT: int = 16











_PREFILTER_NODATA_RGB_EPS: int = 0








_PREFILTER_MIN_VALID_PX: float = 9.0












_UNAVAILABLE_NEUTRAL_EPS: int = 6
_UNAVAILABLE_NEUTRAL_FRAC: float = 0.98
_UNAVAILABLE_DOMINANT_FRAC: float = 0.80




_DEGENERATE_QUICK_STRIDE = 4


def tile_is_blank_array(
    arr: np.ndarray, dominant_frac: float = _BLANK_TILE_DOMINANT_FRAC,
    quant: int = _BLANK_TILE_QUANT,
) -> bool:

















    if arr is None or arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
        return False
    rgb = arr[:, :, :3].astype(np.int64)
    q = rgb // max(1, int(quant))

    packed = (q[:, :, 0] << 16) | (q[:, :, 1] << 8) | q[:, :, 2]
    flat = packed.reshape(-1)
    if flat.size == 0:
        return False
    counts = np.unique(flat, return_counts=True)[1]
    dominant = int(counts.max())
    return (dominant / float(flat.size)) >= dominant_frac


def _degenerate_ruled_out(
    arr: np.ndarray, nodata_frac: float, band_eps: float,
    nodata_rgb_eps: int, min_valid_px: float,
) -> bool:












    stride = _DEGENERATE_QUICK_STRIDE
    sample = arr[::stride, ::stride, :]
    if sample.size == 0:
        return False
    rgb = sample[:, :, :3].reshape(-1, 3)
    valid = None
    if sample.shape[2] >= 4:
        valid = sample[:, :, 3].reshape(-1) != 0
    if int(nodata_rgb_eps) >= 0:
        lit = rgb.max(axis=1) > int(nodata_rgb_eps)
        valid = lit if valid is None else (valid & lit)
    n_px_full = int(arr.shape[0] * arr.shape[1])
    if valid is not None:
        n_valid = int(valid.sum())
        if n_valid == 0:
            return False
        if n_valid < max(0.0, float(min_valid_px)):
            return False


        frac_ceiling = (n_px_full - n_valid) / float(n_px_full)
        if 0.0 < float(nodata_frac) <= 1.0 and frac_ceiling >= float(nodata_frac):
            return False
        rgb = rgb[valid]
    spread = np.subtract(rgb.max(axis=0), rgb.min(axis=0), dtype=np.float64)
    return bool((spread > max(0.0, float(band_eps))).any())


def tile_is_degenerate_array(
    arr: np.ndarray, nodata_frac: float = 1.0, band_eps: float = 2.0,
    nodata_rgb_eps: int = _PREFILTER_NODATA_RGB_EPS,
    min_valid_px: float = _PREFILTER_MIN_VALID_PX,
) -> bool:

















































    if arr is None or arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
        return False
    if _degenerate_ruled_out(arr, nodata_frac, band_eps, nodata_rgb_eps,
                             min_valid_px):
        return False
    eps = max(0.0, float(band_eps))
    floor_px = max(0.0, float(min_valid_px))




    rgb = arr[:, :, :3].reshape(-1, 3)
    n_px = int(arr.shape[0] * arr.shape[1])
    valid = None
    if arr.shape[2] >= 4:
        valid = arr[:, :, 3].reshape(-1) != 0
    if int(nodata_rgb_eps) >= 0:
        lit = rgb.max(axis=1) > int(nodata_rgb_eps)
        valid = lit if valid is None else (valid & lit)
    if valid is not None:
        n_valid = int(valid.sum())
        frac = (n_px - n_valid) / float(n_px)
        if 0.0 < float(nodata_frac) <= 1.0 and frac >= float(nodata_frac):
            return True
        if n_valid == 0:
            return True
        if n_valid < floor_px:
            return True
        rgb = rgb[valid]
    spread = np.subtract(rgb.max(axis=0), rgb.min(axis=0), dtype=np.float64)
    return bool((spread <= eps).all())


def tile_is_degenerate(
    img, nodata_frac: float = 1.0, band_eps: float = 2.0,
    nodata_rgb_eps: int = _PREFILTER_NODATA_RGB_EPS,
    min_valid_px: float = _PREFILTER_MIN_VALID_PX,
) -> bool:






    try:
        from qgis.PyQt.QtGui import QImage

        if img is None or img.isNull():
            return False

        full = img.convertToFormat(QImage.Format.Format_ARGB32)
        w, h = full.width(), full.height()
        if w <= 0 or h <= 0:
            return False
        ptr = full.bits()
        ptr.setsize(h * full.bytesPerLine())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, full.bytesPerLine() // 4, 4)
        arr = arr[:, :w, :]


        if _degenerate_ruled_out(arr, nodata_frac, band_eps, nodata_rgb_eps,
                                 min_valid_px):
            return False

        rgba = arr[:, :, [2, 1, 0, 3]]
        return tile_is_degenerate_array(
            rgba, nodata_frac, band_eps, nodata_rgb_eps, min_valid_px)
    except Exception as exc:  # noqa: BLE001
        logger.debug("tile_is_degenerate: check failed: %s", exc)
        return False


def _blank_tile_dials() -> tuple[int, float, int]:




    try:
        from .detection_policy import (  # noqa: PLC0415
            blank_dominant_frac,
            blank_quant,
            blank_sample_px,
        )
        return (
            blank_sample_px(_BLANK_TILE_SAMPLE_PX),
            blank_dominant_frac(_BLANK_TILE_DOMINANT_FRAC),
            blank_quant(_BLANK_TILE_QUANT),
        )
    except Exception:  # noqa: BLE001
        return (_BLANK_TILE_SAMPLE_PX, _BLANK_TILE_DOMINANT_FRAC, _BLANK_TILE_QUANT)


def tile_is_blank(img) -> bool:




    try:
        from qgis.PyQt.QtCore import QSize, Qt
        from qgis.PyQt.QtGui import QImage

        if img is None or img.isNull():
            return False
        sample_px, dominant_frac, quant = _blank_tile_dials()
        small = img.scaled(
            QSize(sample_px, sample_px),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        ).convertToFormat(QImage.Format.Format_RGB32)
        w, h = small.width(), small.height()
        if w <= 0 or h <= 0:
            return False
        ptr = small.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)

        rgb = arr[:, :, [2, 1, 0]]
        return tile_is_blank_array(rgb, dominant_frac, quant)
    except Exception as exc:  # noqa: BLE001
        logger.debug("tile_is_blank: check failed: %s", exc)
        return False


def tile_is_unavailable_array(
    arr: np.ndarray,
    neutral_eps: int = _UNAVAILABLE_NEUTRAL_EPS,
    neutral_frac: float = _UNAVAILABLE_NEUTRAL_FRAC,
    dominant_frac: float = _UNAVAILABLE_DOMINANT_FRAC,
    quant: int = _BLANK_TILE_QUANT,
) -> bool:

























    if arr is None or arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
        return False
    rgb = arr[:, :, :3].astype(np.int16)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    neutral = float((spread <= max(0, int(neutral_eps))).mean())
    if neutral < float(neutral_frac):
        return False
    return tile_is_blank_array(arr, float(dominant_frac), quant)


def _unavailable_tile_dials() -> tuple[int, int, float, float]:




    try:
        from .detection_policy import (  # noqa: PLC0415
            blank_sample_px,
            unavailable_dominant_frac,
            unavailable_neutral_eps,
            unavailable_neutral_frac,
        )
        return (
            blank_sample_px(_BLANK_TILE_SAMPLE_PX),
            unavailable_neutral_eps(_UNAVAILABLE_NEUTRAL_EPS),
            unavailable_neutral_frac(_UNAVAILABLE_NEUTRAL_FRAC),
            unavailable_dominant_frac(_UNAVAILABLE_DOMINANT_FRAC),
        )
    except Exception:  # noqa: BLE001
        return (
            _BLANK_TILE_SAMPLE_PX,
            _UNAVAILABLE_NEUTRAL_EPS,
            _UNAVAILABLE_NEUTRAL_FRAC,
            _UNAVAILABLE_DOMINANT_FRAC,
        )


def tile_is_unavailable(img) -> bool:





    try:
        from qgis.PyQt.QtCore import QSize, Qt
        from qgis.PyQt.QtGui import QImage

        if img is None or img.isNull():
            return False
        sample_px, neutral_eps, neutral_frac, dominant_frac = _unavailable_tile_dials()
        small = img.scaled(
            QSize(sample_px, sample_px),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        ).convertToFormat(QImage.Format.Format_RGB32)
        w, h = small.width(), small.height()
        if w <= 0 or h <= 0:
            return False
        ptr = small.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)

        rgb = arr[:, :, [2, 1, 0]]
        return tile_is_unavailable_array(
            rgb, neutral_eps, neutral_frac, dominant_frac, _BLANK_TILE_QUANT)
    except Exception as exc:  # noqa: BLE001
        logger.debug("tile_is_unavailable: check failed: %s", exc)
        return False


def render_verdict(img) -> str:







    if tile_is_unavailable(img):
        return "unavailable"
    if tile_is_blank(img):
        return "blank"
    return "ok"
