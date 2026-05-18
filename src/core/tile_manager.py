
from __future__ import annotations







TILE_SIZE = 1008


OVERLAP_FRACTION = 0.20






MAX_TILES = 20000



MAX_TILES_PER_KM2 = 500


MAX_TILES_FLOOR = 16



QUALITY_FLOOR_MUPP_M = 0.0






DEFAULT_AUTO_TILE_BUDGET = 30








AUTO_SEED_TILE_CAP = 20000






AUTO_SEED_HEADROOM_LEVELS = 0





NATIVE_OVERSAMPLE_MAX = 2.0














MAX_DETAIL_LEVEL = 240





HARD_GRID_LIMIT = 200_000


def subdivide_quadrants(
    x: int, y: int, w: int, h: int,
    overlap_fraction: float,
    min_parent_px: int,
) -> list[tuple[int, int, int, int]]:










    if min(w, h) < min_parent_px:
        return []
    ov_x = int(w * overlap_fraction)
    ov_y = int(h * overlap_fraction)
    qw = w - (w // 2) + ov_x
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


    return quads if len(quads) > 1 else []


class TileManager:





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






        return self._snap_axis(width), self._snap_axis(height)

    def _snap_axis(self, size: int) -> int:

        if size <= self.tile_size:
            return size
        stride = int(self.tile_size * (1 - self.overlap_fraction))
        if stride <= 0:
            return size
        n_extra = (size - self.tile_size + stride - 1) // stride
        return self.tile_size + n_extra * stride

    def compute_grid(
        self, image_width: int, image_height: int, apply_cap: bool = True
    ) -> list[tuple[int, int, int, int]] | None:
















        if image_width <= 0 or image_height <= 0:
            return []

        if image_width <= self.tile_size and image_height <= self.tile_size:
            return [(0, 0, image_width, image_height)]

        stride = int(self.tile_size * (1 - self.overlap_fraction))


        if stride <= 0:
            return [(0, 0, image_width, image_height)]
        limit = min(self.max_tiles, HARD_GRID_LIMIT) if apply_cap else HARD_GRID_LIMIT
        if self.count_grid(image_width, image_height) > limit:
            return None











        def axis_offsets(span):
            if span <= self.tile_size:
                return (0,)

            last = span - self.tile_size
            return (*range(0, last, stride), last)

        columns = axis_offsets(image_width)
        rows = axis_offsets(image_height)
        tile_w = min(self.tile_size, image_width)
        tile_h = min(self.tile_size, image_height)
        return [(x, y, tile_w, tile_h) for y in rows for x in columns]

    def count_grid(self, image_width: int, image_height: int) -> int:













        if image_width <= 0 or image_height <= 0:
            return 0
        stride = int(self.tile_size * (1 - self.overlap_fraction))
        if stride <= 0:
            return 1

        def per_axis(span: int) -> int:
            if span <= self.tile_size:
                return 1
            return (span - self.tile_size + stride - 1) // stride + 1

        return per_axis(image_width) * per_axis(image_height)

    def estimate_credits(self, image_width: int, image_height: int) -> int:








        count = self.count_grid(image_width, image_height)
        return -1 if count > HARD_GRID_LIMIT else count


def margin_axis_span(
    zone_lo: float, zone_hi: float, image_span: int, margin_px: int,
    reach: tuple[int, int], tile_size: int = TILE_SIZE,
) -> tuple[int, int]:










    import math



    zone_lo = max(-1.0, float(zone_lo))
    zone_hi = min(float(image_span) + 1.0, float(zone_hi))
    if zone_hi <= zone_lo:
        zone_lo, zone_hi = 0.0, float(image_span)
    lo = max(int(reach[0]), int(math.floor(zone_lo)) - int(margin_px))
    hi = min(int(reach[1]), int(math.ceil(zone_hi)) + int(margin_px))
    if hi - lo >= tile_size:
        return lo, hi
    lo_bound = min(lo, 0)
    hi_bound = max(hi, int(image_span))
    if hi_bound - lo_bound < tile_size:
        return lo_bound, hi_bound
    start = int(round((lo + hi - tile_size) / 2.0))
    start = max(start, hi - tile_size, lo_bound)
    start = min(start, lo, hi_bound - tile_size)
    return start, start + tile_size


def margin_grid_tiles(
    manager: TileManager, limits: tuple[int, int, int, int],
) -> list | None:





    x0, y0, x1, y1 = limits
    tiles = manager.compute_grid(x1 - x0, y1 - y0, apply_cap=False)
    if tiles is None:
        return None
    return [(x + x0, y + y0, w, h) for x, y, w, h in tiles]
