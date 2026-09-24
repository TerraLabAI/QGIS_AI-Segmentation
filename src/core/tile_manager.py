
from __future__ import annotations







TILE_SIZE = 1008


OVERLAP_FRACTION = 0.20






MAX_TILES = 20000






MAX_TILES_PER_KM2 = 420


MAX_TILES_FLOOR = 16


SWEET_SPOT_MAX_MUPP_M = 0.45




DEFAULT_SEED_MUPP_M = 0.142


QUALITY_FLOOR_MUPP_M = 0.5


DEFAULT_TARGET_MUPP_M = 0.4






DEFAULT_AUTO_TILE_BUDGET = 30








AUTO_SEED_TILE_CAP = 20000






AUTO_SEED_HEADROOM_LEVELS = 1





NATIVE_OVERSAMPLE_MAX = 2.0


AUTO_OBJECT_MIN_PX = 20






SPLIT_RISK_TILE_FRAC = 0.5









DETAIL_COARSE_TRAVEL_RATIO = 2.0
DETAIL_FINE_TRAVEL_RATIO = 2.0







DRAWN_OBJECT_TILE_FRAC = 0.25




MASK_SCALE_MIN_WIDTH_PX = 12.0














MAX_DETAIL_LEVEL = 240





HARD_GRID_LIMIT = 200_000



















SUBDIVIDE_OVERLAP_FRACTION = 0.20


SUBDIVIDE_MIN_PARENT_PX = 256


def subdivide_quadrants(
    x: int, y: int, w: int, h: int,
    overlap_fraction: float = SUBDIVIDE_OVERLAP_FRACTION,
    min_parent_px: int = SUBDIVIDE_MIN_PARENT_PX,
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

    def extract_tile_crop(self, image, x: int, y: int, w: int, h: int):











        return image[y : y + h, x : x + w].copy()  # noqa: E203
