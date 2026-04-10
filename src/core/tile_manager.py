"""Adaptive tiling for PRO mode large-image segmentation."""

from typing import Optional


class TileManager:
    """Computes tile grids and estimates credits for large images.

    Each tile is a (x_offset, y_offset, width, height) tuple in pixel coords.
    """

    def __init__(
        self,
        tile_size: int = 1024,
        overlap_fraction: float = 0.15,
        max_tiles: int = 50,
    ):
        self.tile_size = tile_size
        self.overlap_fraction = overlap_fraction
        self.max_tiles = max_tiles

    def compute_grid(
        self, image_width: int, image_height: int
    ) -> Optional[list[tuple[int, int, int, int]]]:
        """Compute tile grid for an image.

        Returns:
            List of (x, y, w, h) tuples, or None if exceeds max_tiles.
        """
        if image_width <= self.tile_size and image_height <= self.tile_size:
            return [(0, 0, image_width, image_height)]

        stride = int(self.tile_size * (1 - self.overlap_fraction))
        tiles = []

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
            if y + tile_h >= image_height:
                break
            y += stride

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
            (h, w, 3) uint8 numpy array
        """
        return image[y : y + h, x : x + w].copy()
