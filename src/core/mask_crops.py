


















from __future__ import annotations

import logging

import numpy as np

from .cloud_detection import _iter_mask_entries, _rle_pairs, decode_rle_to_mask
from .detection_masks import _mask_box, _mask_dimensions

logger = logging.getLogger(__name__)


class MaskCrop:









    __slots__ = ("crop", "row0", "col0", "full_shape", "set_pixels")

    def __init__(self, crop: np.ndarray, row0: int, col0: int,
                 full_shape: tuple[int, int], set_pixels: int) -> None:
        self.crop = crop
        self.row0 = int(row0)
        self.col0 = int(col0)
        self.full_shape = (int(full_shape[0]), int(full_shape[1]))
        self.set_pixels = int(set_pixels)

    @property
    def row1(self) -> int:

        return self.row0 + int(self.crop.shape[0]) - 1

    @property
    def col1(self) -> int:

        return self.col0 + int(self.crop.shape[1]) - 1

    def padded(self) -> np.ndarray:






        h, w = self.crop.shape
        out = np.zeros((h + 2, w + 2), dtype=bool)
        out[1:-1, 1:-1] = self.crop
        return out


def empty_crop(full_shape: tuple[int, int]) -> MaskCrop:

    return MaskCrop(np.zeros((0, 0), dtype=bool), 0, 0, full_shape, 0)


def crop_from_mask(mask: np.ndarray) -> MaskCrop:







    full_h, full_w = mask.shape
    set_pixels = int(np.count_nonzero(mask))
    if set_pixels == 0:
        return empty_crop((full_h, full_w))
    rows = mask.any(axis=1)
    row0 = int(rows.argmax())
    row1 = full_h - 1 - int(rows[::-1].argmax())
    cols = mask[row0:row1 + 1].any(axis=0)
    col0 = int(cols.argmax())
    col1 = full_w - 1 - int(cols[::-1].argmax())
    crop = mask[row0:row1 + 1, col0:col1 + 1]
    if crop.dtype != np.bool_:
        crop = crop.astype(bool)
    return MaskCrop(crop, row0, col0, (full_h, full_w), set_pixels)


def as_crop(mask) -> MaskCrop:

    if isinstance(mask, MaskCrop):
        return mask
    return crop_from_mask(mask)


def decode_rle_to_crop(rle, height: int, width: int,
                       strict: bool = False) -> MaskCrop:









    height, width = _mask_dimensions(height, width)
    if isinstance(rle, str) and not rle.strip():
        return empty_crop((height, width))
    if not isinstance(rle, str):
        return crop_from_mask(decode_rle_to_mask(rle, height, width, strict=strict))
    tokens = rle.split()
    if len(tokens) % 2:
        return crop_from_mask(decode_rle_to_mask(rle, height, width, strict=strict))
    pairs = _rle_pairs(tokens)
    if pairs is None:
        return crop_from_mask(decode_rle_to_mask(rle, height, width, strict=strict))
    offsets, counts = pairs
    if offsets.size == 0:
        return empty_crop((int(height), int(width)))
    total = int(height) * int(width)
    starts = offsets - 1
    ends = starts + counts
    if (starts < 0).any() or (counts <= 0).any() or (ends > total).any():
        return crop_from_mask(decode_rle_to_mask(rle, height, width, strict=strict))
    if starts.size > 1 and (ends[:-1] > starts[1:]).any():


        return crop_from_mask(decode_rle_to_mask(rle, height, width, strict=strict))

    width = int(width)
    last = ends - 1
    row_start = starts // width
    row_end = last // width
    row0 = int(row_start[0])
    row1 = int(row_end[-1])
    same_row = row_start == row_end


    col_lo = np.where(same_row, starts % width, 0)
    col_hi = np.where(same_row, last % width, width - 1)
    col0 = int(col_lo.min())
    col1 = int(col_hi.max())








    nrows = row1 - row0 + 1
    ncols = col1 - col0 + 1
    if same_row.all():




        rows = (row_start - row0).astype(np.intp, copy=False)
        opens = (starts % width - col0).astype(np.intp, copy=False)
        closes = (last % width + 1 - col0).astype(np.intp, copy=False)
        edges = np.zeros((nrows, ncols + 1), dtype=np.int8)
        edges[rows, opens] += 1
        edges[rows, closes] -= 1
        crop = np.cumsum(edges[:, :ncols], axis=1, dtype=np.int8).astype(
            bool, copy=False)
        return MaskCrop(crop, row0, col0, (int(height), width),
                        int(counts.sum()))
    base = row0 * width
    span = nrows * width
    edges = np.zeros(span + 1, dtype=np.int8)
    edges[starts - base] += 1
    edges[ends - base] -= 1
    band = np.cumsum(edges[:span], dtype=np.int8).astype(bool, copy=False)
    crop = band.reshape(nrows, width)[:, col0:col1 + 1]
    return MaskCrop(crop, row0, col0, (int(height), width), int(counts.sum()))


def iter_detection_crops(response: dict, tile_w: int, tile_h: int,
                         score_threshold: float = 0.0, strict: bool = False):







    raw_masks = response.get("masks") or []
    if not isinstance(raw_masks, list):
        logger.warning("iter_detection_crops: 'masks' is not a list; returning []")
        return iter(())
    srv_w = response.get("width")
    srv_h = response.get("height")
    decode_h, decode_w = _mask_dimensions(
        srv_h if srv_h is not None else tile_h,
        srv_w if srv_w is not None else tile_w)
    return _iter_crops(raw_masks, decode_h, decode_w, score_threshold, strict)


def _iter_crops(raw_masks: list, decode_h: int, decode_w: int,
                score_threshold: float, strict: bool):
    for entry, score in _iter_mask_entries(raw_masks, score_threshold):
        crop = decode_rle_to_crop(
            entry.get("rle", ""), decode_h, decode_w, strict=strict)
        yield crop, score, _mask_box(entry)


def crop_has_no_holes(padded: np.ndarray) -> bool:









    transitions = np.count_nonzero(padded[:, 1:] != padded[:, :-1], axis=1)
    return bool((transitions <= 2).all())
