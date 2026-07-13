"""Adaptive tiling for Pro automatic detection mode."""
from __future__ import annotations

import math
from typing import Optional

# Tiles are sent at the model's native processing size so the server skips an
# input resize. The grid/credit/detail math reads this symbolically (stride =
# int(TILE_SIZE * (1 - OVERLAP_FRACTION))); credits are per-tile, not per-pixel.
# Masks decode at the server-reported size and map by mask.shape, so detections
# stay geo-exact regardless of this value.
TILE_SIZE = 1008
# Tile overlap fraction. Must be wide enough for the merge step to stitch
# seam-split objects across the strip without paying for redundant inference.
OVERLAP_FRACTION = 0.20
# Hard cap on tiles per run (= credits per run). Kept high enough that the
# object-aware detail walk stays reachable on large zones; memory stays flat
# (tiles render just-in-time) and the per-run credit estimate is always shown
# before Detect, so a finer grid stays a conscious choice.
MAX_TILES = 800
# Target ground footprint per tile (meters). Used to size tiles when the source
# has no native resolution (WMS); native-resolution sources tile at their own
# deepest zoom, which is already finer than this.
DETECTION_TILE_FOOTPRINT_M = 100.0

# Coarse edge (m/px) of the model's adequate-quality band. Client fallback for
# the server policy's `sweet_spot_max_mupp`.
SWEET_SPOT_MAX_MUPP_M = 0.45
# Prompt-less default seed resolution (m/px). Client fallback for the server
# policy's `zone_seed_mupp`; the object-aware seed refines it per prompt.
DEFAULT_SEED_MUPP_M = 0.28
# Resolution (m/px) below which the imagery is too coarse for reliable
# detection: the UI shows a "raise detail / zoom in" hint above it.
QUALITY_FLOOR_MUPP_M = 0.5
# Kept for backward reference (older callers / docs); the picker uses the
# sweet-spot band rather than this single target.
DEFAULT_TARGET_MUPP_M = 0.4
# Soft tile (credit) preference the auto-picked default tries to stay within;
# it may be crossed to reach an adequate resolution. Client fallback for the
# server policy's `soft_tile_budget`.
DEFAULT_AUTO_TILE_BUDGET = 30

# Hard ceiling on tiles the auto-picked default may propose (the soft budget
# above may be crossed to reach an adequate resolution, but never past this),
# so a default run cannot drain a small credit allowance in one click. Client
# fallback for the server policy's `seed_tile_cap`.
AUTO_SEED_TILE_CAP = 100

# How far past a source's native resolution a render may go (linear factor on
# m/px): upsampling adds no pixels but enlarges each object in model space,
# which helps small-object recall; past this the imagery is pure interpolation.
# Client fallback for the server policy's `native_oversample_max`.
NATIVE_OVERSAMPLE_MAX = 2.0
# Minimum pixels across for an object to count as resolvable in the picker's
# fallback. Client fallback for the server policy's `object_min_px`.
AUTO_OBJECT_MIN_PX = 20

# Highest detail level the slider exposes (and the loop bound in
# _max_useful_detail): the longer zone side renders as this many tiles at the
# top. The real per-zone ceiling is still MAX_TILES and the native-resolution
# clamp, both applied dynamically; this is just the static upper bound.
MAX_DETAIL_LEVEL = 48

# Sibling overlap of the 2x2 sub-tiles a saturated tile is re-split into,
# as a fraction of the parent side ADDED to each half. Wide enough for the
# merger to stitch an object cut by the sub-seam; kept small so each sub-tile
# still covers ~a quarter of the parent (the whole point of the re-split).
SUBDIVIDE_OVERLAP_FRACTION = 0.05
# Never re-split a tile whose side would drop below this (pixels on the run
# grid): past that the quadrants stop containing meaningfully fewer objects.
SUBDIVIDE_MIN_PARENT_PX = 256


def subdivide_quadrants(
    x: int, y: int, w: int, h: int,
    overlap_fraction: float = SUBDIVIDE_OVERLAP_FRACTION,
) -> list[tuple[int, int, int, int]]:
    """Split one tile rect into 4 overlapping quadrants (same pixel grid).

    Each quadrant is ~half the parent per side plus a small sibling overlap,
    anchored to the parent's corners, so together they cover the parent
    exactly and an object sitting on the internal seam appears whole in at
    least one quadrant (or overlapping enough for the merger to stitch it).
    Degenerate parents (too small to split on either axis) return [].
    """
    if min(w, h) < SUBDIVIDE_MIN_PARENT_PX:
        return []
    ov_x = int(w * overlap_fraction)
    ov_y = int(h * overlap_fraction)
    qw = w - (w // 2) + ov_x  # ceil(w/2) + overlap
    qh = h - (h // 2) + ov_y
    qw = min(qw, w)
    qh = min(qh, h)
    xs = (x, x + w - qw)
    ys = (y, y + h - qh)
    quads = []
    for qy in ys:
        for qx in xs:
            spec = (qx, qy, qw, qh)
            if spec not in quads:
                quads.append(spec)
    # A parent small on one axis can collapse both rows/columns onto the same
    # origin; the dedup above keeps coverage exact without duplicate submits.
    return quads if len(quads) > 1 else []


class TileManager:
    """Computes tile grids and estimates credits for large images.

    Each tile is a (x_offset, y_offset, width, height) tuple in pixel coords.
    """

    def __init__(
        self,
        tile_size: int = TILE_SIZE,
        overlap_fraction: float = OVERLAP_FRACTION,
        max_tiles: int = MAX_TILES,
    ):
        self.tile_size = tile_size
        self.overlap_fraction = overlap_fraction
        self.max_tiles = max_tiles

    def snap_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """Round dimensions up to the nearest clean tile grid boundary.

        After snapping, compute_grid() will produce only full-size tiles
        (tile_size x tile_size). Use for zone-selected mode only, NOT
        for full-image mode where partial edge tiles are acceptable.
        """
        return self._snap_axis(width), self._snap_axis(height)

    def _snap_axis(self, size: int) -> int:
        """Snap a single axis dimension to the nearest clean grid size."""
        if size <= self.tile_size:
            return size
        stride = int(self.tile_size * (1 - self.overlap_fraction))
        if stride <= 0:
            return size
        n_extra = math.ceil((size - self.tile_size) / stride)
        return self.tile_size + n_extra * stride

    def compute_grid(
        self, image_width: int, image_height: int
    ) -> Optional[list[tuple[int, int, int, int]]]:
        """Compute tile grid for an image.

        Returns:
            List of (x, y, w, h) tuples, empty when the image has no area, or
            None if it exceeds max_tiles.
        """
        # A zero/negative dimension has no area to tile: return an empty grid
        # (0 credits, no run) rather than a degenerate (0, 0, 0, 0) tile.
        if image_width <= 0 or image_height <= 0:
            return []

        if image_width <= self.tile_size and image_height <= self.tile_size:
            return [(0, 0, image_width, image_height)]

        stride = int(self.tile_size * (1 - self.overlap_fraction))
        # Guard a non-positive stride (overlap_fraction >= 1) the same way
        # _snap_axis does, so the stepping loops below can never fail to advance.
        if stride <= 0:
            return [(0, 0, image_width, image_height)]
        tiles = []

        # Edge alignment: when a plain stride step would leave a thin partial
        # tile at the far edge (e.g. a 1008x325 bottom strip), snap that LAST
        # tile flush to the edge at full size instead. The cloud model resizes
        # every tile to a fixed square, so a thin strip is stretched (a 325px
        # strip ~3.1x vertically) and its masks come back ragged; a full tile
        # over the same ground reads the strip in true proportions with real
        # context, for cleaner outlines at equal recall. Same tile count, same
        # credits; the extra overlap with the previous row/column is resolved
        # by the merger's dedup like any overlap strip. An axis smaller than
        # one tile keeps its single
        # partial tile (there is nothing to align it against).
        y = 0
        while y < image_height:
            x = 0
            tile_h = min(self.tile_size, image_height - y)
            while x < image_width:
                tile_w = min(self.tile_size, image_width - x)
                tiles.append((x, y, tile_w, tile_h))
                if x + tile_w >= image_width:
                    break
                x += stride
                if x + self.tile_size > image_width >= self.tile_size:
                    x = image_width - self.tile_size
            if y + tile_h >= image_height:
                break
            y += stride
            if y + self.tile_size > image_height >= self.tile_size:
                y = image_height - self.tile_size

        if len(tiles) > self.max_tiles:
            return None

        return tiles

    def estimate_credits(self, image_width: int, image_height: int) -> int:
        """Return number of credits (= tiles) needed, or -1 if exceeds cap."""
        tiles = self.compute_grid(image_width, image_height)
        if tiles is None:
            return -1
        return len(tiles)

    def extract_tile_crop(self, image, x: int, y: int, w: int, h: int):
        """Extract a tile crop from the full image array.

        Args:
            image: (H, W, 3) uint8 numpy array (full image at native resolution)
            x, y: top-left pixel offset of the tile
            w, h: tile dimensions in pixels

        Returns:
            (h, w, 3) uint8 numpy array -- .copy() prevents aliasing bugs when
            the caller zeros out the array for padding.
        """
        return image[y : y + h, x : x + w].copy()  # noqa: E203
