
























from __future__ import annotations

import sys

import numpy as np

from ...core import tile_verdicts as _tv

__all__ = ["RenderedTileVerdicts"]


def _quick_look_rules_in(arr, nodata_frac: float, band_eps: float,
                         nodata_rgb_eps: int, min_valid_px: float) -> bool:





    stride = _tv._DEGENERATE_QUICK_STRIDE
    sample = arr[::stride, ::stride, :]
    if sample.size == 0:
        return False
    planes = [sample[:, :, c] for c in range(3)]
    valid = sample[:, :, 3] != 0
    if int(nodata_rgb_eps) >= 0:
        eps = int(nodata_rgb_eps)
        lit = (planes[0] > eps) | (planes[1] > eps) | (planes[2] > eps)
        valid &= lit
    n_px_full = int(arr.shape[0] * arr.shape[1])
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return False
    if n_valid < max(0.0, float(min_valid_px)):
        return False
    frac_ceiling = (n_px_full - n_valid) / float(n_px_full)
    if 0.0 < float(nodata_frac) <= 1.0 and frac_ceiling >= float(nodata_frac):
        return False
    every = n_valid == valid.size
    limit = max(0.0, float(band_eps))
    for plane in planes:
        values = plane if every else plane[valid]
        if float(int(values.max()) - int(values.min())) > limit:
            return True
    return False


def _blank_sample_verdict(rgb, dominant_frac: float, quant: int) -> bool:




    n = int(rgb.shape[0] * rgb.shape[1])
    if n <= 0 or rgb.ndim != 3 or rgb.shape[2] < 3:
        return _tv.tile_is_blank_array(rgb, dominant_frac, quant)
    frac = float(dominant_frac)
    if not frac > (n - 1) / float(n):
        return _tv.tile_is_blank_array(rgb, dominant_frac, quant)

    step = max(1, int(quant))
    for c in range(3):
        q = rgb[:, :, c] // step
        if int(q.min()) != int(q.max()):
            return False
    return frac <= 1.0


class RenderedTileVerdicts:







    def __init__(self, img) -> None:
        self._img = img
        self._samples: dict = {}

    def _raw_bgra(self):




        if sys.byteorder != "little":
            return None
        from qgis.PyQt.QtGui import QImage

        img = self._img
        if img is None or img.isNull():
            return None
        fmt = img.format()
        if fmt == QImage.Format.Format_ARGB32_Premultiplied:
            premultiplied = True
        elif fmt == QImage.Format.Format_ARGB32:
            premultiplied = False
        else:
            return None
        width, height = img.width(), img.height()
        if width <= 0 or height <= 0:
            return None
        bits = img.constBits()
        bits.setsize(img.sizeInBytes())
        flat = np.frombuffer(bits, dtype=np.uint8)
        line = img.bytesPerLine()
        raw = flat[: line * height].reshape(height, line)[:, : width * 4]
        return raw.reshape(height, width, 4), premultiplied

    def is_degenerate(self, nodata_frac: float, band_eps: float,
                      nodata_rgb_eps: int, min_valid_px: float) -> bool:


        try:
            got = self._raw_bgra()
            if got is not None:
                raw, premultiplied = got
                stride = _tv._DEGENERATE_QUICK_STRIDE
                alpha = raw[::stride, ::stride, 3]


                exact = (not premultiplied
                         or bool(((alpha == 255) | (alpha == 0)).all()))
                if exact and _quick_look_rules_in(
                        raw, nodata_frac, band_eps, nodata_rgb_eps, min_valid_px):
                    return False
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return _tv.tile_is_degenerate(
            self._img, nodata_frac, band_eps, nodata_rgb_eps, min_valid_px)

    def _nearest_rgb(self, side: int):


        side = max(1, int(side))
        if side in self._samples:
            return self._samples[side]
        sample = None
        try:
            got = self._raw_bgra()
            if got is not None:
                raw, _premultiplied = got
                src_h, src_w = raw.shape[0], raw.shape[1]

                rows = ((np.arange(side) + 0.5) * src_h / side).astype(int).clip(0, src_h - 1)
                cols = ((np.arange(side) + 0.5) * src_w / side).astype(int).clip(0, src_w - 1)
                picked = raw[rows][:, cols]


                if bool((picked[:, :, 3] == 255).all()):
                    sample = np.ascontiguousarray(picked[:, :, [2, 1, 0]])
        except Exception:  # noqa: BLE001
            sample = None
        self._samples[side] = sample
        return sample

    def is_blank(self) -> bool:

        try:
            sample_px, dominant_frac, quant = _tv._blank_tile_dials()
            rgb = self._nearest_rgb(sample_px)
            if rgb is not None:
                return _blank_sample_verdict(rgb, dominant_frac, quant)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return _tv.tile_is_blank(self._img)

    def is_unavailable(self) -> bool:

        try:
            sample_px, neutral_eps, neutral_frac, dominant_frac = (
                _tv._unavailable_tile_dials())
            rgb = self._nearest_rgb(sample_px)
            if rgb is not None:
                return _tv.tile_is_unavailable_array(
                    rgb, neutral_eps, neutral_frac, dominant_frac,
                    _tv._BLANK_TILE_QUANT)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return _tv.tile_is_unavailable(self._img)
