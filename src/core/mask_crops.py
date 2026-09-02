"""Decode a detection mask straight into its bounding box.

The tile pipeline never reads a mask outside the object's own box: it crops
the full grid to that box, pads it, fills pinholes and polygonizes the crop
with a pixel offset against the full grid (see ``mask_to_polygons``). The
full-grid decode it started from was pure overhead: one HxW allocation per
mask, then three full passes (count, row projection, column projection) to
find four integers that the run list already spells out. A dense tile carries
hundreds of masks, most of them a few pixels, so those passes cost more than
the geometry work they fed.

``decode_rle_to_crop`` reads the box and the pixel count off the run offsets
and paints only the rows the object covers. Its answer is the same crop, at
the same offset, with the same pixel count that ``decode_rle_to_mask`` plus
the worker's projection gave; anything the fast path cannot vouch for
(malformed runs, overlapping runs, a payload that is not a string) goes
through ``decode_rle_to_mask`` itself, so the two never disagree, not even on
what they log or raise.
"""
from __future__ import annotations

import logging

import numpy as np

from .cloud_detection import _iter_mask_entries, _rle_pairs, decode_rle_to_mask

logger = logging.getLogger(__name__)


class MaskCrop:
    """One mask reduced to its bounding box on the tile grid.

    ``crop`` is the boolean box itself, unpadded; ``row0``/``col0`` its
    top-left pixel in the ``full_shape`` grid; ``set_pixels`` how many pixels
    of the whole mask are set. An all-background mask is an EMPTY crop:
    ``set_pixels`` 0 and a ``crop`` of shape (0, 0), so a caller still reads
    the grid it came on from ``full_shape``.
    """

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
        """Last row of the box, inclusive."""
        return self.row0 + int(self.crop.shape[0]) - 1

    @property
    def col1(self) -> int:
        """Last column of the box, inclusive."""
        return self.col0 + int(self.crop.shape[1]) - 1

    def padded(self) -> np.ndarray:
        """The crop with one ring of background around it, as a fresh array.

        The pinhole fill floods inward from the array border, and an object
        always touches its own box, so the ring is what keeps a concavity that
        opens onto the box edge from reading as a hole.
        """
        h, w = self.crop.shape
        out = np.zeros((h + 2, w + 2), dtype=bool)
        out[1:-1, 1:-1] = self.crop
        return out


def empty_crop(full_shape: tuple[int, int]) -> MaskCrop:
    """The MaskCrop of an all-background mask on a ``full_shape`` grid."""
    return MaskCrop(np.zeros((0, 0), dtype=bool), 0, 0, full_shape, 0)


def crop_from_mask(mask: np.ndarray) -> MaskCrop:
    """Reduce a full-grid mask to its MaskCrop, empty when nothing is set.

    The box comes from the row and column projections, not from index arrays:
    ``np.nonzero`` allocates two arrays as long as the object to read four
    bounds off them, while ``any`` collapses each axis to one vector and
    ``argmax`` from each end reads the same bounds.
    """
    full_h, full_w = mask.shape
    set_pixels = int(np.count_nonzero(mask))
    if set_pixels == 0:
        return empty_crop((full_h, full_w))
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    row0 = int(rows.argmax())
    row1 = full_h - 1 - int(rows[::-1].argmax())
    col0 = int(cols.argmax())
    col1 = full_w - 1 - int(cols[::-1].argmax())
    crop = mask[row0:row1 + 1, col0:col1 + 1]
    if crop.dtype != np.bool_:
        crop = crop.astype(bool)
    return MaskCrop(crop, row0, col0, (full_h, full_w), set_pixels)


def as_crop(mask) -> MaskCrop:
    """A MaskCrop from either a MaskCrop or a full-grid array."""
    if isinstance(mask, MaskCrop):
        return mask
    return crop_from_mask(mask)


def decode_rle_to_crop(rle, height: int, width: int,
                       strict: bool = False) -> MaskCrop:
    """Decode one mask RLE to its bounding-box crop, empty for all background.

    Same encoding, same validation and same ``strict`` contract as
    ``decode_rle_to_mask``: an input that function would warn about, raise on
    or treat as empty gets handed to it, and its full grid is then cropped.
    Only a clean string of in-range, ascending, non-overlapping runs takes the
    direct path, and on that path the crop, its offset and its pixel count
    are what the full decode gives.
    """
    if not isinstance(rle, str) or not rle.strip():
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
        # Out of order or overlapping: the set-pixel count is no longer the
        # sum of the counts, so let the full decode count what is set.
        return crop_from_mask(decode_rle_to_mask(rle, height, width, strict=strict))

    width = int(width)
    last = ends - 1
    row_start = starts // width
    row_end = last // width
    row0 = int(row_start[0])
    row1 = int(row_end[-1])
    same_row = row_start == row_end
    # A run that wraps onto the next row covers a full row width somewhere
    # along the way, so its column reach is the whole grid.
    col_lo = np.where(same_row, starts % width, 0)
    col_hi = np.where(same_row, last % width, width - 1)
    col0 = int(col_lo.min())
    col1 = int(col_hi.max())

    # Paint the runs with one cumulative sum instead of a Python loop over
    # them. A run is a +1 where it opens and a -1 where it closes, and the
    # running total of those is 1 exactly on the run's own cells, because the
    # checks above have already established that the runs ascend and never
    # overlap. Every start is distinct and so is every end, so both writes are
    # safe; where a run closes exactly where the next one opens the cell nets
    # to zero, which is the continuous stretch the loop drew.
    nrows = row1 - row0 + 1
    ncols = col1 - col0 + 1
    if same_row.all():
        # Every run sits inside one row, which is what a mask boundary gives
        # unless the object spans the full grid width. So the sum runs across
        # the object's own box rather than across full rows of the tile, and a
        # narrow object stops paying for the background beside it.
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
    """The masks of one completed response as ``(MaskCrop, score, box)``,
    one at a time, in server order.

    Mirrors ``cloud_detection.iter_detection_masks`` (same score gate, same
    decode dimensions, same eager payload check, same ``box`` fallback), with
    each mask decoded by ``decode_rle_to_crop`` instead of into a full grid.
    """
    raw_masks = response.get("masks") or []
    if not isinstance(raw_masks, list):
        logger.warning("iter_detection_crops: 'masks' is not a list; returning []")
        return iter(())
    srv_w = response.get("width")
    srv_h = response.get("height")
    decode_w = int(srv_w) if srv_w is not None else tile_w
    decode_h = int(srv_h) if srv_h is not None else tile_h
    return _iter_crops(raw_masks, decode_h, decode_w, score_threshold, strict)


def _iter_crops(raw_masks: list, decode_h: int, decode_w: int,
                score_threshold: float, strict: bool):
    for entry, score in _iter_mask_entries(raw_masks, score_threshold):
        crop = decode_rle_to_crop(
            entry.get("rle", ""), decode_h, decode_w, strict=strict)
        raw_box = entry.get("box")
        if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
            box = [float(v) for v in raw_box]
        else:
            box = [0.0, 0.0, 0.0, 0.0]
        yield crop, score, box


def crop_has_no_holes(padded: np.ndarray) -> bool:
    """True when every row of a background-ringed crop holds at most one run.

    A hole is background that no path of background joins to the border. In a
    row with a single run, every background pixel reaches the row's end
    through background, and the ring makes that end border. So a crop whose
    rows are all single runs has no hole to fill, whatever its columns do,
    and the fill can be skipped without changing its answer. The test is one
    pass over the crop; the fill it spares is several.
    """
    transitions = np.count_nonzero(padded[:, 1:] != padded[:, :-1], axis=1)
    return bool((transitions <= 2).all())
