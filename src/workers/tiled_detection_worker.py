"""Worker thread for parallel tiled PRO detection."""

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Dict, List, Tuple

from qgis.PyQt.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    import numpy as np

    from ..core.tile_manager import TileManager

logger = logging.getLogger("AISegmentation")


class TiledDetectionWorker(QThread):
    """Orchestrates parallel tile inference for PRO text detection.

    For each tile in the grid, extracts the image crop, creates a fresh
    FalPredictor, runs predict_text, decodes results, and emits signals
    for progressive display.
    """

    tile_completed = pyqtSignal(int, list)
    all_completed = pyqtSignal(list)
    progress = pyqtSignal(int, int)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        predictor,
        tiles: List[Tuple[int, int, int, int]],
        full_image: "np.ndarray",
        text_prompt: str,
        max_masks: int,
        score_threshold: float,
        geo_transform_info: Dict,
        tile_manager: "TileManager",
        max_workers: int = 4,
        parent=None,
    ):
        super().__init__(parent)
        self._fal_key = predictor._fal_key
        self._tiles = tiles
        self._full_image = full_image
        self._text_prompt = text_prompt
        self._max_masks = max_masks
        self._score_threshold = score_threshold
        self._geo_transform_info = geo_transform_info
        self._tile_manager = tile_manager
        self._max_workers = max_workers
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the tiled detection."""
        self._cancelled = True

    def run(self) -> None:
        """Execute parallel tile inference."""
        all_detections: List[Tuple[int, list]] = []
        total = len(self._tiles)

        logger.info(
            "TiledDetectionWorker starting: %d tiles, max_workers=%d, "
            "prompt='%s', max_masks=%d, score_threshold=%.2f",
            total,
            self._max_workers,
            self._text_prompt,
            self._max_masks,
            self._score_threshold,
        )
        logger.info(
            "Full image shape: %s, geo_bbox: %s",
            self._full_image.shape,
            self._geo_transform_info.get("bbox"),
        )

        try:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(self._process_tile, idx, tile): idx
                    for idx, tile in enumerate(self._tiles)
                }

                for future in as_completed(futures):
                    if self._cancelled:
                        executor.shutdown(wait=False, cancel_futures=True)
                        self.cancelled.emit()
                        return

                    try:
                        tile_idx, detections = future.result()
                    except Exception as exc:
                        logger.error(
                            "Tile %d failed: %s",
                            futures[future],
                            exc,
                        )
                        continue

                    all_detections.append((tile_idx, detections))
                    self.tile_completed.emit(tile_idx, detections)
                    self.progress.emit(len(all_detections), total)

            if self._cancelled:
                self.cancelled.emit()
                return

            # Flatten all detections into a single list
            flat: List = []
            for _idx, dets in sorted(all_detections, key=lambda x: x[0]):
                flat.extend(dets)

            self.all_completed.emit(flat)

        except Exception as exc:
            logger.error(
                "TiledDetectionWorker error: %s\n%s",
                exc,
                traceback.format_exc(),
            )
            self.error.emit(str(exc))

    def _process_tile(
        self, tile_idx: int, tile: Tuple[int, int, int, int]
    ) -> Tuple[int, list]:
        """Process a single tile: extract crop, run inference, decode masks.

        Args:
            tile_idx: Index of the tile in the grid.
            tile: (x, y, w, h) pixel coordinates.

        Returns:
            (tile_idx, list of (mask, score, transform_info) tuples)
        """
        if self._cancelled:
            return (tile_idx, [])

        x, y, w, h = tile
        logger.info(
            "Tile %d: extracting crop at (%d,%d) size %dx%d", tile_idx, x, y, w, h
        )
        crop = self._tile_manager.extract_tile_crop(self._full_image, x, y, w, h)

        # Pad non-square crops to square with zero padding.
        # SAM-3 may internally process at a fixed square resolution;
        # sending non-square images can cause RLE dimension mismatches
        # that produce diagonal mask artifacts.
        # Using zero (black) padding instead of reflect — reflect creates
        # mirrored content that confuses SAM into false detections at edges.
        import numpy as np

        pad_size = max(h, w)
        if h != pad_size or w != pad_size:
            crop = np.pad(
                crop,
                ((0, pad_size - h), (0, pad_size - w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            logger.info(
                "Tile %d: padded %dx%d -> %dx%d (square)",
                tile_idx,
                w,
                h,
                pad_size,
                pad_size,
            )

        logger.info(
            "Tile %d: crop shape=%s, dtype=%s", tile_idx, crop.shape, crop.dtype
        )

        # Each thread gets its own predictor instance (no shared state)
        from ..core.pro_predictor import FalPredictor

        predictor = FalPredictor(self._fal_key)
        predictor.set_image_from_array(crop)
        # Send the full square image — do NOT crop back to native size
        predictor.set_native_size(pad_size, pad_size)

        logger.info("Tile %d: calling predict_text...", tile_idx)
        result = predictor.predict_text(self._text_prompt, max_masks=self._max_masks)
        logger.info(
            "Tile %d: predict_text returned %d RLE masks",
            tile_idx,
            len(result.get("rle", [])),
        )

        rle_list = result.get("rle", [])
        scores = result.get("scores", [])

        # Decode RLE at the actually-sent dimensions (square),
        # then crop mask back to original tile dimensions
        from ..core.pro_predictor import decode_rle_to_mask

        sent_h = result.get("_sent_h", pad_size)
        sent_w = result.get("_sent_w", pad_size)

        detections: List = []
        for i, rle_str in enumerate(rle_list):
            score = scores[i] if i < len(scores) else 0.0
            if score < self._score_threshold:
                continue
            mask = decode_rle_to_mask(rle_str, sent_h, sent_w)
            # Crop mask back to original tile dimensions (remove padding)
            mask = mask[:h, :w]
            tile_transform = self._make_tile_transform(x, y, w, h)
            detections.append((mask, score, tile_transform))

        logger.info(
            "Tile %d (%dx%d at %d,%d): %d detections",
            tile_idx,
            w,
            h,
            x,
            y,
            len(detections),
        )
        return (tile_idx, detections)

    def _make_tile_transform(
        self, tile_x: int, tile_y: int, tile_w: int, tile_h: int
    ) -> Dict:
        """Map tile pixel coordinates to geographic coordinates.

        Uses the full zone's geo_transform_info to compute the geographic
        bounding box for a specific tile.

        Args:
            tile_x: Tile x offset in pixels from full image origin.
            tile_y: Tile y offset in pixels from full image origin.
            tile_w: Tile width in pixels.
            tile_h: Tile height in pixels.

        Returns:
            Dict with "bbox" (minx, miny, maxx, maxy) and "img_shape" (H, W).
        """
        base = self._geo_transform_info
        full_bbox = base["bbox"]  # (minx, miny, maxx, maxy)
        full_w = base["img_shape"][1]
        full_h = base["img_shape"][0]

        px_w = (full_bbox[2] - full_bbox[0]) / full_w
        px_h = (full_bbox[3] - full_bbox[1]) / full_h

        # X: pixel col 0 is at xMin, increases rightward
        tile_minx = full_bbox[0] + tile_x * px_w
        tile_maxx = tile_minx + tile_w * px_w

        # Y: pixel row 0 is at yMax (top of raster), increases downward
        # So tile_y=0 → top of bbox (yMax), tile_y increases → move toward yMin
        tile_maxy = full_bbox[3] - tile_y * px_h
        tile_miny = tile_maxy - tile_h * px_h

        logger.info(
            "Tile transform: pixel(%d,%d %dx%d) -> geo(%.4f,%.4f,%.4f,%.4f)",
            tile_x,
            tile_y,
            tile_w,
            tile_h,
            tile_minx,
            tile_miny,
            tile_maxx,
            tile_maxy,
        )

        return {
            "bbox": (tile_minx, tile_miny, tile_maxx, tile_maxy),
            "img_shape": (tile_h, tile_w),
            "crs": base.get("crs"),
        }
