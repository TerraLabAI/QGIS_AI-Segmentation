"""Adaptive tiling for Pro automatic detection mode."""
from __future__ import annotations

import math

# Side of a request tile, in pixels. The grid and detail math read this
# symbolically (stride = int(TILE_SIZE * (1 - OVERLAP_FRACTION))).
# It is also the model's own input square, so a tile needs no server-side
# resample.
# Masks decode at the server-reported size and map by mask.shape, so detections
# stay geo-exact regardless of this value.
TILE_SIZE = 1008
# Tile overlap fraction. Must be wide enough for the merge step to stitch
# seam-split objects across the strip without paying for redundant inference.
OVERLAP_FRACTION = 0.20
# Hard cap on tiles per run. Tiles stopped being credits when Automatic moved
# to per-km2 billing: a run is priced on the surface the user drew, and the
# tile count only moves how finely that surface is read. So this bounds the
# two things tiles still cost, service time and the user's wait, and nothing
# else. Memory stays flat (tiles render just-in-time). Client fallback
# for the server policy's `max_tiles_per_run`.
MAX_TILES = 20000
# Tiles one run may spend per km2 of the zone drawn. The ceiling above bounds a
# run's wall clock; this bounds what a run may spend for each km2 it bills, and
# that is the one ratio somebody can drive to an absurd value on purpose, since
# the price follows the surface and not the grid. Sized on the finest ground
# tile any class is measurably answered at, so it refuses no precision anybody
# can use. Client fallback for the server policy's `max_tiles_per_km2`.
MAX_TILES_PER_KM2 = 130
# Floor under the per-km2 rule: a zone smaller than one tile still needs a grid
# worth running. Client fallback for `max_tiles_floor`.
MAX_TILES_FLOOR = 16
# UNREACHABLE (2026-09-01): no reader anywhere in src/, tests/ or scripts/.
# It reads as live WMS policy and is not: the WMS path sizes its tiles through
# the detail window in auto_flow, which never looks here. Kept so a reader does
# not take it for the value that path uses.
# Target ground footprint per tile (meters), for a source with no native
# resolution; native-resolution sources tile at their own deepest zoom, which
# is already finer than this.
DETECTION_TILE_FOOTPRINT_M = 100.0

# Coarse edge (m/px) of the model's adequate-quality band. Client fallback for
# the server policy's `sweet_spot_max_mupp`.
SWEET_SPOT_MAX_MUPP_M = 0.45
# Prompt-less default seed resolution (m/px). Client fallback for the server
# policy's `zone_seed_mupp`; the object-aware seed refines it per prompt.
# Matches the value the server serves, so a cold cache reads the same ground
# as a warm one.
DEFAULT_SEED_MUPP_M = 0.142
# Resolution (m/px) below which the imagery is too coarse for reliable
# detection: the UI shows a "raise detail / zoom in" hint above it.
QUALITY_FLOOR_MUPP_M = 0.5
# Kept for backward reference (older callers / docs); the picker uses the
# sweet-spot band rather than this single target.
DEFAULT_TARGET_MUPP_M = 0.4
# Soft tile preference the auto-picked default used to stay within, back when
# a tile was a credit. The seed no longer reads it: per-km2 billing means a
# finer grid costs the user nothing, so a budget in tiles could only trade the
# result away for a saving nobody banks. Kept as the client fallback for the
# server policy's `soft_tile_budget`, which is still served, so an older
# plugin reading it keeps the number it always had.
DEFAULT_AUTO_TILE_BUDGET = 30

# Hard ceiling on tiles the auto-picked default may propose. It existed to
# stop one default run draining a credit allowance; per-km2 billing removed
# that risk, and the run cap (MAX_TILES and MAX_TILES_PER_KM2) already bounds
# the wait and the service time. It sits at MAX_TILES so the default reaches
# the object's own target tile on a large zone instead of stopping at a flat
# number that was under it. Client fallback for the server policy's
# `seed_tile_cap`.
AUTO_SEED_TILE_CAP = 20000

# Levels the Precision slider keeps open above an automatically picked level.
# A default that sits at the top of its travel leaves the user no way to ask
# for more, so the seed stops this far short of the machine ceiling and the
# object band keeps its top this far past the seed wherever a finer level
# exists. Client fallback for the server policy's `seed_headroom_levels`.
AUTO_SEED_HEADROOM_LEVELS = 1

# How far past a source's native resolution a render may go (linear factor on
# m/px): upsampling adds no pixels but enlarges each object in model space,
# which helps small-object recall; past this the imagery is pure interpolation.
# Client fallback for the server policy's `native_oversample_max`.
NATIVE_OVERSAMPLE_MAX = 2.0
# Minimum pixels across for an object to count as resolvable in the picker's
# fallback. Client fallback for the server policy's `object_min_px`.
AUTO_OBJECT_MIN_PX = 20
# Object ground size as a fraction of a tile's ground side (TILE_SIZE * m/px)
# at or above which the object can no longer be counted on to come back whole:
# an object narrower than TILE_SIZE * OVERLAP_FRACTION pixels always falls
# inside one tile, past that the run leans on the seam stitch, and near the
# tile side each piece carries too little context to stitch. Client fallback
# for the server policy's `seed.split_risk_tile_frac`.
SPLIT_RISK_TILE_FRAC = 0.5
# How far the Precision slider travels either side of the automatic pick, as
# a ratio on the tile's ground side. The coarse end is the level whose tile
# covers this many times the recommended tile, the fine end the level whose
# tile covers the recommended tile divided by this. Both ends stay inside the
# machine ceiling and the object's own served floor and ceiling. Bounding the
# travel on the pick keeps the pick inside the band on every zone size, and
# stops a run from being asked at a tile so small the model gains nothing
# from it. Client fallbacks for the server policy's
# `seed.detail_coarse_travel_ratio` and `seed.detail_fine_travel_ratio`.
DETAIL_COARSE_TRAVEL_RATIO = 2.0
DETAIL_FINE_TRAVEL_RATIO = 2.0
# Share of a tile's ground side a DRAWN example may take when it is the only
# thing describing the object (no word typed). The seed reads the example as a
# measurement and grows the tile until the object fits inside this share, so a
# large one is read whole instead of in fragments. Under the prompt-less seed
# resolution the tile never shrinks, so a small example leaves the grid exactly
# where it has always been. Client fallback for the server policy's
# `seed.drawn_object_tile_frac`.
DRAWN_OBJECT_TILE_FRAC = 0.25
# Native pixels an object's NARROW dimension must span before the coarse mask
# grid may be requested for it: under this a half-cell boundary shift eats the
# object's thin parts. Client fallback for the server policy's
# `seed.mask_scale.min_width_px`.
MASK_SCALE_MIN_WIDTH_PX = 12.0

# Highest detail level the slider exposes (and the loop bound in
# _max_useful_detail): the longer zone side renders as this many tiles at the
# top. The real per-zone ceiling is still MAX_TILES and the native-resolution
# clamp, both applied dynamically; this is just the static upper bound.
#
# It has to clear the worst case, not the typical one. A level counts tiles
# along the LONGER side, so a long thin zone reaches a given ground resolution
# at a far higher level than a square one of the same area: a 10 km by 1.4 km
# zone needs level 87 for the ground a square 14 km2 zone reaches at 33. At 48
# the walk stopped before MAX_TILES did, and the stop was invisible, reading
# as a resolution the source could not do better than. The walk breaks on the
# tile cap anyway, so a high bound costs a few extra iterations of arithmetic
# on elongated zones and nothing at all on square ones.
MAX_DETAIL_LEVEL = 240

# Structural ceiling on an uncapped grid (compute_grid(apply_cap=False)).
# It guards memory while the caller culls a bounding-box grid down to the
# polygon the user drew, and it is deliberately far above any product limit,
# so the number a user meets is always max_tiles and never this.
HARD_GRID_LIMIT = 200_000

# Sibling overlap of the 2x2 sub-tiles a saturated tile is re-split into,
# as a fraction of the parent side ADDED to each half.
#
# It matches OVERLAP_FRACTION, and it has to. The merger decides whether two
# polygons on a seam are one object from a single ground distance,
# `seam_min_dim` in auto_review.py, computed as OVERLAP_FRACTION * TILE_SIZE.
# Tile identity never reaches the merger, so that one distance judges every
# seam in the run, base and sub-seam alike. At 0.05 a sub-seam strip was a
# quarter of what the gate assumes: an object cut by it read as two objects
# that overlap too little to dedup, so the re-split meant to recover objects
# split them instead, and a narrower quadrant is also a harder upsample of the
# same ground. One overlap everywhere means the gate is right everywhere.
#
# The ladder itself is off by default (`_SUBDIV_MAX_DEPTH` in
# `workers/auto_detection_worker.py`), so this only decides the shape of a
# re-split a server dial switched back on. The server's own
# `saturation.subdivide_overlap_fraction` must agree with this value or the
# seam gate is wrong again for the sub-seams.
SUBDIVIDE_OVERLAP_FRACTION = 0.20
# Never re-split a tile whose side would drop below this (pixels on the run
# grid): past that the quadrants stop containing meaningfully fewer objects.
SUBDIVIDE_MIN_PARENT_PX = 256


def subdivide_quadrants(
    x: int, y: int, w: int, h: int,
    overlap_fraction: float = SUBDIVIDE_OVERLAP_FRACTION,
    min_parent_px: int = SUBDIVIDE_MIN_PARENT_PX,
) -> list[tuple[int, int, int, int]]:
    """Split one tile rect into 4 overlapping quadrants (same pixel grid).

    Each quadrant is ~half the parent per side plus a small sibling overlap,
    anchored to the parent's corners, so together they cover the parent
    exactly and an object sitting on the internal seam appears whole in at
    least one quadrant (or overlapping enough for the merger to stitch it).
    Degenerate parents (too small to split on either axis) return [].
    """
    if min(w, h) < min_parent_px:
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
    """Computes tile grids and counts tiles for large images.

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
        self, image_width: int, image_height: int, apply_cap: bool = True
    ) -> list[tuple[int, int, int, int]] | None:
        """Compute tile grid for an image.

        ``apply_cap=False`` returns the grid whatever its size, so the caller
        can cull it against the drawn polygon FIRST and cap what the run will
        really send. Every zone is a hand-drawn polygon and the grid covers its
        bounding box, so the two counts are far apart. Capping before the
        cull refused zones whose real run was a fraction of the ceiling.
        The uncapped path still stops at HARD_GRID_LIMIT, which is a memory
        guard and not a product limit.

        Returns:
            List of (x, y, w, h) tuples, empty when the image has no area, or
            None if it exceeds max_tiles (or HARD_GRID_LIMIT when uncapped).
        """
        # A zero/negative dimension has no area to tile: return an empty grid
        # (no tiles, no run) rather than a degenerate (0, 0, 0, 0) tile.
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
        # tiles; the extra overlap with the previous row/column is resolved
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

        if apply_cap and len(tiles) > self.max_tiles:
            return None
        if len(tiles) > HARD_GRID_LIMIT:
            return None

        return tiles

    def count_grid(self, image_width: int, image_height: int) -> int:
        """Tiles in the bounding-box grid, counted rather than built.

        Same answer as ``len(compute_grid(w, h, apply_cap=False))``, without
        the list. The seed walk asks for this at every detail level of every
        zone, and on a long thin zone the bounding-box grid runs to tens of
        thousands of tiles that the polygon cull then throws away, so building
        them to count them was the walk's whole cost.

        compute_grid emits one tile at offset 0 then one per full stride, with
        the last snapped flush to the edge, so an axis of ``span`` pixels holds
        ``ceil((span - TILE_SIZE) / stride) + 1`` tiles, and 1 when it fits in
        one tile.
        """
        if image_width <= 0 or image_height <= 0:
            return 0
        stride = int(self.tile_size * (1 - self.overlap_fraction))
        if stride <= 0:
            return 1

        def per_axis(span: int) -> int:
            if span <= self.tile_size:
                return 1
            return math.ceil((span - self.tile_size) / stride) + 1

        return per_axis(image_width) * per_axis(image_height)

    def estimate_credits(self, image_width: int, image_height: int) -> int:
        """Number of tiles in the bounding-box grid, or -1 past HARD_GRID_LIMIT.

        This counts the grid over the zone's BOUNDING BOX. The run culls it
        against the drawn polygon and sends fewer, so treat this as an upper
        bound; `_tiles_in_polygon` owns the count the run really pays, and the
        per-run cap is applied to THAT count by the caller. Capping here on the
        bounding box refused, and under-tiled, every long thin zone.
        """
        count = self.count_grid(image_width, image_height)
        return -1 if count > HARD_GRID_LIMIT else count

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
