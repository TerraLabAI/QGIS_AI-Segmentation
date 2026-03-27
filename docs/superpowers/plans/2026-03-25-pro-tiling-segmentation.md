# PRO Tiling Segmentation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable PRO mode to segment large images by adaptively tiling them into 1024x1024 native-resolution chunks, running each through fal.ai in parallel, and merging results with IoU deduplication.

**Architecture:** New `TileManager` computes tile grids and extracts crops. New `TiledDetectionWorker` (QThread) orchestrates parallel fal.ai calls with progressive results. Existing `FalPredictor` is reused per-tile. Existing IoU dedup logic is extracted into a shared utility. The PRO dock gets a zone selection tool and credit estimator. A new `ZoneSelectionMapTool` handles rectangle drawing on the canvas.

**Tech Stack:** Python 3.9+, PyQt5/Qt signals, QGIS API (QgsMapToolEmitPoint, QgsRubberBand, QgsRectangle), rasterio for crop extraction, existing fal.ai REST API via pro_predictor.py.

**Spec:** `docs/superpowers/specs/2026-03-25-pro-tiling-segmentation-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/core/tile_manager.py` | Tile grid computation, credit estimation, crop extraction per tile |
| Create | `src/workers/tiled_detection_worker.py` | QThread orchestrating parallel tile inference + progressive signals |
| Create | `src/ui/zone_selection_maptool.py` | Rectangle drawing tool on QGIS canvas |
| Create | `tests/test_tile_manager.py` | Unit tests for tile grid math and edge cases |
| Modify | `src/ui/ai_segmentation_pro_dockwidget.py` | Add zone selection button, credit indicator, tile progress bar |
| Modify | `src/ui/ai_segmentation_plugin.py` | Wire tiling flow into existing PRO detection pipeline |
| Modify | `src/core/pro_predictor.py` | Add method to accept raw numpy crop (skip internal extraction) |
| Modify | `src/core/polygon_exporter.py` | Extract IoU dedup into reusable function |

---

## Task 1: Extract IoU deduplication into reusable utility

**Files:**
- Modify: `src/core/polygon_exporter.py` (add function)
- Modify: `src/ui/ai_segmentation_plugin.py:3095-3198` (call extracted function)
- Test: `tests/test_tile_manager.py`

- [ ] **Step 1: Write failing test for IoU dedup utility**

```python
# tests/test_tile_manager.py
from qgis.core import QgsGeometry


def test_deduplicate_geometries_removes_overlapping():
    """Two overlapping squares with IoU > 0.3 should deduplicate to one."""
    g1 = QgsGeometry.fromWkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")
    g2 = QgsGeometry.fromWkt("POLYGON((2 2, 12 2, 12 12, 2 12, 2 2))")
    from src.core.polygon_exporter import deduplicate_geometries

    accepted = deduplicate_geometries([g1, g2], iou_threshold=0.3)
    assert len(accepted) == 1


def test_deduplicate_geometries_keeps_non_overlapping():
    """Two non-overlapping squares should both be kept."""
    g1 = QgsGeometry.fromWkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")
    g2 = QgsGeometry.fromWkt("POLYGON((20 20, 30 20, 30 30, 20 30, 20 20))")
    from src.core.polygon_exporter import deduplicate_geometries

    accepted = deduplicate_geometries([g1, g2], iou_threshold=0.3)
    assert len(accepted) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tile_manager.py -v`
Expected: ImportError — `deduplicate_geometries` not found

- [ ] **Step 3: Implement `deduplicate_geometries` in polygon_exporter.py**

Add at the end of `src/core/polygon_exporter.py`:

```python
def _iou(g1, g2):
    """Intersection over Union for two QgsGeometry objects."""
    inter = g1.intersection(g2)
    if inter.isEmpty():
        return 0.0
    union_geom = g1.combine(g2)
    if union_geom.area() == 0:
        return 0.0
    return inter.area() / union_geom.area()


def deduplicate_geometries(geometries, iou_threshold=0.3):
    """Remove duplicate geometries based on IoU overlap.

    Args:
        geometries: list of QgsGeometry, ordered by priority (first = highest)
        iou_threshold: max IoU before considering a geometry as duplicate

    Returns:
        list of accepted QgsGeometry (deduplicated)
    """
    accepted = []
    for geom in geometries:
        if geom.isEmpty():
            continue
        is_dup = False
        for ag in accepted:
            if _iou(geom, ag) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            accepted.append(geom)
    return accepted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tile_manager.py -v`
Expected: PASS

- [ ] **Step 5: Update `_run_fal_detection` to use extracted function**

In `src/ui/ai_segmentation_plugin.py`, replace the inline `_iou` function and dedup loop (lines ~3095-3160) with a call to `deduplicate_geometries`. The dedup now happens after all detections are converted to geometries — collect all geometries first, then call `deduplicate_geometries()`.

- [ ] **Step 6: Commit**

```bash
git add src/core/polygon_exporter.py src/ui/ai_segmentation_plugin.py tests/test_tile_manager.py
git commit -m "refactor: extract IoU deduplication into reusable deduplicate_geometries()"
```

---

## Task 2: Create TileManager — grid computation and credit estimation

**Files:**
- Create: `src/core/tile_manager.py`
- Test: `tests/test_tile_manager.py` (append)

- [ ] **Step 1: Write failing tests for tile grid computation**

```python
# append to tests/test_tile_manager.py

def test_no_tiling_for_small_image():
    """Image <= 1024x1024 should produce a single tile."""
    from src.core.tile_manager import TileManager

    tm = TileManager(tile_size=1024, overlap_fraction=0.15, max_tiles=50)
    tiles = tm.compute_grid(image_width=800, image_height=600)
    assert len(tiles) == 1
    assert tiles[0] == (0, 0, 800, 600)


def test_tiling_for_large_image():
    """Image 2048x2048 should produce multiple tiles with overlap."""
    from src.core.tile_manager import TileManager

    tm = TileManager(tile_size=1024, overlap_fraction=0.15, max_tiles=50)
    tiles = tm.compute_grid(image_width=2048, image_height=2048)
    assert len(tiles) > 1
    # Each tile should be at most 1024x1024
    for x, y, w, h in tiles:
        assert w <= 1024
        assert h <= 1024


def test_estimate_credits():
    """Credits should equal number of tiles."""
    from src.core.tile_manager import TileManager

    tm = TileManager(tile_size=1024, overlap_fraction=0.15, max_tiles=50)
    credits = tm.estimate_credits(image_width=3000, image_height=3000)
    tiles = tm.compute_grid(image_width=3000, image_height=3000)
    assert credits == len(tiles)


def test_max_tiles_cap():
    """Should raise or return error when exceeding max_tiles."""
    from src.core.tile_manager import TileManager

    tm = TileManager(tile_size=1024, overlap_fraction=0.15, max_tiles=4)
    tiles = tm.compute_grid(image_width=10000, image_height=10000)
    assert tiles is None  # Exceeds cap


def test_tiles_cover_full_image():
    """Union of all tiles should cover the entire image."""
    from src.core.tile_manager import TileManager

    tm = TileManager(tile_size=1024, overlap_fraction=0.15, max_tiles=50)
    w, h = 3000, 2000
    tiles = tm.compute_grid(image_width=w, image_height=h)
    # Check coverage: every pixel should be in at least one tile
    max_x = max(x + tw for x, y, tw, th in tiles)
    max_y = max(y + th for x, y, tw, th in tiles)
    assert max_x >= w
    assert max_y >= h
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tile_manager.py -v -k "tile"`
Expected: ImportError — `TileManager` not found

- [ ] **Step 3: Implement TileManager**

```python
# src/core/tile_manager.py
"""Adaptive tiling for PRO mode large-image segmentation."""

from typing import List, Optional, Tuple


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
    ) -> Optional[List[Tuple[int, int, int, int]]]:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tile_manager.py -v -k "tile"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/tile_manager.py tests/test_tile_manager.py
git commit -m "feat: add TileManager for adaptive grid computation and credit estimation"
```

---

## Task 3: Add tile crop extraction to TileManager

**Files:**
- Modify: `src/core/tile_manager.py`
- Test: `tests/test_tile_manager.py` (append)

- [ ] **Step 1: Write failing test for crop extraction**

```python
# append to tests/test_tile_manager.py
import numpy as np


def test_extract_tile_crop():
    """Should extract the correct sub-region from a full image array."""
    from src.core.tile_manager import TileManager

    tm = TileManager(tile_size=4, overlap_fraction=0.0, max_tiles=50)
    # 8x8 image with distinct values per row
    full_image = np.arange(64).reshape(8, 8).astype(np.uint8)
    full_image_rgb = np.stack([full_image] * 3, axis=-1)

    crop = tm.extract_tile_crop(full_image_rgb, x=2, y=3, w=4, h=4)
    assert crop.shape == (4, 4, 3)
    assert crop[0, 0, 0] == full_image[3, 2]


def test_extract_tile_crop_edge_padding():
    """Edge tiles smaller than tile_size should be returned as-is."""
    from src.core.tile_manager import TileManager

    tm = TileManager(tile_size=1024, overlap_fraction=0.0, max_tiles=50)
    full_image = np.zeros((500, 300, 3), dtype=np.uint8)

    crop = tm.extract_tile_crop(full_image, x=0, y=0, w=300, h=500)
    assert crop.shape == (500, 300, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tile_manager.py::test_extract_tile_crop -v`
Expected: AttributeError — `extract_tile_crop` not found

- [ ] **Step 3: Implement extract_tile_crop**

Add to `src/core/tile_manager.py` class `TileManager`:

```python
def extract_tile_crop(self, image: "np.ndarray", x: int, y: int, w: int, h: int) -> "np.ndarray":
    """Extract a tile crop from the full image array.

    Args:
        image: (H, W, 3) uint8 numpy array (full image at native resolution)
        x, y: top-left pixel offset of the tile
        w, h: tile dimensions in pixels

    Returns:
        (h, w, 3) uint8 numpy array
    """
    return image[y : y + h, x : x + w].copy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tile_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/tile_manager.py tests/test_tile_manager.py
git commit -m "feat: add tile crop extraction to TileManager"
```

---

## Task 4: Add `set_image_from_array` to FalPredictor

**Files:**
- Modify: `src/core/pro_predictor.py`

Currently `FalPredictor.set_image()` takes a file path. For tiling, we need to pass a raw numpy array (the tile crop) directly.

- [ ] **Step 1: Add `set_image_from_array` method**

Add to `src/core/pro_predictor.py` class `FalPredictor`, after the existing `set_image()` method:

```python
def set_image_from_array(self, image_array):
    """Set the image from a numpy array (H, W, 3) uint8.

    Used by tiling workflow where crops are already in memory.
    """
    self._image_array = image_array
    self._original_height, self._original_width = image_array.shape[:2]
    self._image_path = None  # No file path in tiling mode
```

- [ ] **Step 2: Update `_upload_to_cdn` to handle array source**

Modify `_upload_to_cdn()` to encode from `self._image_array` when `self._image_path` is None. The existing path reads from file; the new path encodes the numpy array to PNG bytes directly:

```python
# At the start of _upload_to_cdn, add:
if self._image_path is None and self._image_array is not None:
    import cv2
    _, png_bytes = cv2.imencode(".png", cv2.cvtColor(self._image_array, cv2.COLOR_RGB2BGR))
    png_data = png_bytes.tobytes()
else:
    # existing file read logic
    with open(self._image_path, "rb") as f:
        png_data = f.read()
```

Similarly update `_make_fallback_data_uri()` to handle array source.

- [ ] **Step 3: Commit**

```bash
git add src/core/pro_predictor.py
git commit -m "feat: add set_image_from_array to FalPredictor for tiling workflow"
```

---

## Task 5: Create ZoneSelectionMapTool

**Files:**
- Create: `src/ui/zone_selection_maptool.py`

- [ ] **Step 1: Implement rectangle selection map tool**

```python
# src/ui/zone_selection_maptool.py
"""Map tool for drawing a rectangular zone selection on the QGIS canvas."""

from qgis.PyQt.QtCore import pyqtSignal, Qt
from qgis.PyQt.QtGui import QColor
from qgis.core import QgsPointXY, QgsRectangle, QgsWkbTypes
from qgis.gui import QgsMapTool, QgsRubberBand


class ZoneSelectionMapTool(QgsMapTool):
    """Allows the user to draw a rectangle on the map canvas.

    Emits zone_selected(QgsRectangle) when the user finishes drawing.
    Emits zone_cleared() when the selection is reset.
    """

    zone_selected = pyqtSignal(QgsRectangle)
    zone_cleared = pyqtSignal()

    def __init__(self, canvas):
        super().__init__(canvas)
        self._canvas = canvas
        self._rubber_band = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
        self._rubber_band.setColor(QColor(65, 105, 225, 80))
        self._rubber_band.setStrokeColor(QColor(65, 105, 225, 200))
        self._rubber_band.setWidth(2)
        self._start_point = None
        self._is_drawing = False

    def canvasPressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._start_point = self.toMapCoordinates(event.pos())
            self._is_drawing = True
            self._rubber_band.reset(QgsWkbTypes.PolygonGeometry)

    def canvasMoveEvent(self, event):
        if not self._is_drawing or self._start_point is None:
            return
        end_point = self.toMapCoordinates(event.pos())
        self._update_rubber_band(self._start_point, end_point)

    def canvasReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._is_drawing:
            self._is_drawing = False
            end_point = self.toMapCoordinates(event.pos())
            rect = QgsRectangle(self._start_point, end_point)
            rect.normalize()
            if rect.width() > 0 and rect.height() > 0:
                self.zone_selected.emit(rect)
            else:
                self.zone_cleared.emit()

    def _update_rubber_band(self, start, end):
        self._rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        self._rubber_band.addPoint(QgsPointXY(start.x(), start.y()), False)
        self._rubber_band.addPoint(QgsPointXY(end.x(), start.y()), False)
        self._rubber_band.addPoint(QgsPointXY(end.x(), end.y()), False)
        self._rubber_band.addPoint(QgsPointXY(start.x(), end.y()), True)

    def clear_selection(self):
        """Remove the rectangle from the canvas."""
        self._rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        self._start_point = None
        self.zone_cleared.emit()

    def deactivate(self):
        self._rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        super().deactivate()
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/zone_selection_maptool.py
git commit -m "feat: add ZoneSelectionMapTool for rectangle zone selection"
```

---

## Task 6: Update PRO dock widget — zone selection + credit indicator

**Files:**
- Modify: `src/ui/ai_segmentation_pro_dockwidget.py`

- [ ] **Step 1: Add new signals**

Add to the class signals (around line 50):

```python
zone_select_requested = pyqtSignal()  # User wants to draw a zone
zone_clear_requested = pyqtSignal()   # User wants to reset to full image
```

- [ ] **Step 2: Add zone selection button and credit label**

In the `_setup_ui()` method, after the layer combo and before the Start PRO button, add:

```python
# Zone selection row
zone_row = QHBoxLayout()
self.zone_select_btn = QPushButton(tr("Select zone"))
self.zone_select_btn.setToolTip(
    tr("Draw a rectangle to limit the segmentation area")
)
self.zone_select_btn.clicked.connect(self.zone_select_requested.emit)

self.zone_clear_btn = QPushButton(tr("Full image"))
self.zone_clear_btn.setToolTip(tr("Use the entire image"))
self.zone_clear_btn.clicked.connect(self.zone_clear_requested.emit)
self.zone_clear_btn.setVisible(False)  # Hidden until a zone is drawn

zone_row.addWidget(self.zone_select_btn)
zone_row.addWidget(self.zone_clear_btn)
layout.addLayout(zone_row)

# Credit estimation label
self.credit_label = QLabel("")
self.credit_label.setWordWrap(True)
self.credit_label.setStyleSheet("color: palette(text); font-size: 11px;")
layout.addWidget(self.credit_label)

# Explanatory text (always visible)
self.credit_info_label = QLabel(
    tr("The larger the selected zone, the more credits are used.")
)
self.credit_info_label.setWordWrap(True)
self.credit_info_label.setStyleSheet("color: palette(text); font-size: 11px;")
layout.addWidget(self.credit_info_label)
```

- [ ] **Step 3: Add public methods for credit/zone updates**

```python
def set_credit_estimate(self, credits: int):
    """Update the credit estimate display."""
    if credits < 0:
        self.credit_label.setText(
            tr("Zone too large. Please reduce the selection.")
        )
        self.credit_label.setStyleSheet("color: #f44336; font-size: 11px;")
    else:
        self.credit_label.setText(
            tr("Estimated credits: {count}").format(count=credits)
        )
        self.credit_label.setStyleSheet("color: palette(text); font-size: 11px;")

def set_zone_active(self, active: bool):
    """Toggle zone selection UI state."""
    self.zone_clear_btn.setVisible(active)
    if active:
        self.zone_select_btn.setText(tr("Redraw zone"))
    else:
        self.zone_select_btn.setText(tr("Select zone"))
```

- [ ] **Step 4: Add tile progress bar**

After the existing progress/status area, add a progress bar for tiling:

```python
# Tile progress bar (hidden by default)
self.tile_progress = QProgressBar()
self.tile_progress.setVisible(False)
self.tile_progress.setTextVisible(True)
self.tile_progress.setFormat("Tile %v/%m")
layout.addWidget(self.tile_progress)
```

Add public methods:

```python
def set_tile_progress(self, current: int, total: int):
    """Update tile progress bar."""
    self.tile_progress.setMaximum(total)
    self.tile_progress.setValue(current)
    self.tile_progress.setVisible(total > 1)

def hide_tile_progress(self):
    """Hide the tile progress bar."""
    self.tile_progress.setVisible(False)
```

- [ ] **Step 5: Update i18n .ts files**

Add all new `tr()` strings to `i18n/fr.ts`, `i18n/pt_BR.ts`, `i18n/es.ts`:
- "Select zone" / "Selectionner une zone" / "Selecionar zona" / "Seleccionar zona"
- "Full image" / "Image entiere" / "Imagem completa" / "Imagen completa"
- "Draw a rectangle to limit the segmentation area"
- "Use the entire image"
- "The larger the selected zone, the more credits are used."
- "Estimated credits: {count}"
- "Zone too large. Please reduce the selection."
- "Redraw zone"

- [ ] **Step 6: Commit**

```bash
git add src/ui/ai_segmentation_pro_dockwidget.py i18n/*.ts
git commit -m "feat: add zone selection UI and credit estimator to PRO dock"
```

---

## Task 7: Create TiledDetectionWorker

**Files:**
- Create: `src/workers/tiled_detection_worker.py`

- [ ] **Step 1: Implement the QThread worker**

```python
# src/workers/tiled_detection_worker.py
"""Worker thread for parallel tiled PRO detection."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from qgis.PyQt.QtCore import QThread, pyqtSignal

logger = logging.getLogger("AISegmentation")


class TiledDetectionWorker(QThread):
    """Runs tiled PRO detection in a background thread.

    Signals:
        tile_completed(int, list): tile_index, list of (mask, score, transform_info)
        all_completed(list): all detections accumulated
        progress(int, int): (current_tile, total_tiles)
        error(str): error message
        cancelled(): emitted when cancelled by user
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
        full_image,
        text_prompt: str,
        max_masks: int,
        score_threshold: float,
        geo_transform_info: Dict,
        tile_manager,
        max_workers: int = 4,
    ):
        super().__init__()
        self._predictor_class = type(predictor)
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

    def cancel(self):
        self._cancelled = True

    def run(self):
        all_detections = []
        completed = 0
        total = len(self._tiles)

        def process_tile(tile_idx, tile):
            """Process a single tile. Called from thread pool."""
            if self._cancelled:
                return tile_idx, []

            x, y, w, h = tile
            crop = self._tile_manager.extract_tile_crop(
                self._full_image, x, y, w, h
            )

            # Create a fresh predictor per thread (avoids shared state)
            predictor = self._predictor_class(self._fal_key)
            predictor.set_image_from_array(crop)
            predictor.set_native_size(h, w)

            result = predictor.predict_text(
                self._text_prompt, max_masks=self._max_masks
            )

            if not result or "error" in result:
                return tile_idx, []

            detections = []
            rle_list = result.get("rle", [])
            scores = result.get("scores", [])

            if isinstance(rle_list, str):
                rle_list = [rle_list]

            for i, rle_str in enumerate(rle_list):
                score = scores[i] if i < len(scores) else 0.0
                if score < self._score_threshold:
                    continue

                from src.core.pro_predictor import decode_rle_to_mask

                mask = decode_rle_to_mask(rle_str, h, w)
                if mask is None:
                    continue

                # Build transform_info with tile offset applied
                tile_transform = self._make_tile_transform(x, y, w, h)
                detections.append((mask, score, tile_transform))

            return tile_idx, detections

        try:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(process_tile, idx, tile): idx
                    for idx, tile in enumerate(self._tiles)
                }

                for future in as_completed(futures):
                    if self._cancelled:
                        executor.shutdown(wait=False, cancel_futures=True)
                        self.cancelled.emit()
                        return

                    tile_idx, detections = future.result()
                    completed += 1
                    all_detections.extend(detections)
                    self.tile_completed.emit(tile_idx, detections)
                    self.progress.emit(completed, total)

            if not self._cancelled:
                self.all_completed.emit(all_detections)

        except Exception as e:
            logger.error("Tiled detection error: %s", str(e))
            self.error.emit(str(e))

    def _make_tile_transform(self, tile_x, tile_y, tile_w, tile_h):
        """Create a transform_info dict that maps tile-local coords to image coords.

        The tile's pixel (0,0) corresponds to full image pixel (tile_x, tile_y).
        We need to adjust the geographic bbox accordingly.
        """
        base = self._geo_transform_info
        full_bbox = base["bbox"]  # (minx, miny, maxx, maxy) of full image
        full_w = base["img_shape"][1]
        full_h = base["img_shape"][0]

        px_w = (full_bbox[2] - full_bbox[0]) / full_w
        px_h = (full_bbox[3] - full_bbox[1]) / full_h

        tile_minx = full_bbox[0] + tile_x * px_w
        tile_miny = full_bbox[1] + tile_y * px_h
        tile_maxx = tile_minx + tile_w * px_w
        tile_maxy = tile_miny + tile_h * px_h

        return {
            "bbox": (tile_minx, tile_miny, tile_maxx, tile_maxy),
            "img_shape": (tile_h, tile_w),
        }
```

- [ ] **Step 2: Commit**

```bash
git add src/workers/tiled_detection_worker.py
git commit -m "feat: add TiledDetectionWorker for parallel tile inference"
```

---

## Task 8: Wire tiling into the main plugin PRO flow

**Files:**
- Modify: `src/ui/ai_segmentation_plugin.py`

This is the integration task — connecting all the pieces.

- [ ] **Step 1: Add imports and state variables**

At the top of the plugin class, add:

```python
from src.core.tile_manager import TileManager
from src.workers.tiled_detection_worker import TiledDetectionWorker
from src.ui.zone_selection_maptool import ZoneSelectionMapTool
```

In `__init__`, add state:

```python
self._tile_manager = TileManager(tile_size=1024, overlap_fraction=0.15, max_tiles=50)
self._zone_selection_tool = None  # ZoneSelectionMapTool instance
self._selected_zone = None  # QgsRectangle or None (full image)
self._tiled_worker = None  # TiledDetectionWorker instance
```

- [ ] **Step 2: Connect dock signals for zone selection**

In the method that connects PRO dock signals (around lines 339-356), add:

```python
self.pro_dock.zone_select_requested.connect(self._on_zone_select)
self.pro_dock.zone_clear_requested.connect(self._on_zone_clear)
```

Implement handlers:

```python
def _on_zone_select(self):
    """Activate rectangle drawing tool on canvas."""
    canvas = self.iface.mapCanvas()
    self._zone_selection_tool = ZoneSelectionMapTool(canvas)
    self._zone_selection_tool.zone_selected.connect(self._on_zone_drawn)
    canvas.setMapTool(self._zone_selection_tool)

def _on_zone_drawn(self, rect):
    """User finished drawing a zone rectangle."""
    self._selected_zone = rect
    self.pro_dock.set_zone_active(True)
    self._update_credit_estimate()
    # Restore the normal map tool
    self.iface.mapCanvas().unsetMapTool(self._zone_selection_tool)

def _on_zone_clear(self):
    """Reset to full image."""
    self._selected_zone = None
    if self._zone_selection_tool:
        self._zone_selection_tool.clear_selection()
    self.pro_dock.set_zone_active(False)
    self._update_credit_estimate()
```

- [ ] **Step 3: Implement credit estimation**

```python
def _update_credit_estimate(self):
    """Compute and display credit estimate based on current zone."""
    layer = self.pro_dock.layer_combo.currentLayer()
    if not layer:
        return

    if self._selected_zone:
        # Convert geographic zone to pixel dimensions
        pixel_width, pixel_height = self._zone_to_pixels(
            self._selected_zone, layer
        )
    else:
        pixel_width = layer.width()
        pixel_height = layer.height()

    credits = self._tile_manager.estimate_credits(pixel_width, pixel_height)
    self.pro_dock.set_credit_estimate(credits)
```

- [ ] **Step 4: Modify `_run_fal_detection` to use tiling when needed**

The key change: when the zone is large (> 1 tile), use `TiledDetectionWorker` instead of single API call. When small, keep existing single-call behavior.

```python
def _run_fal_detection(self):
    # ... existing setup code for prompt, layer, etc. ...

    if self._selected_zone:
        pixel_w, pixel_h = self._zone_to_pixels(self._selected_zone, layer)
    else:
        pixel_w, pixel_h = layer.width(), layer.height()

    tiles = self._tile_manager.compute_grid(pixel_w, pixel_h)

    if tiles is None:
        self.iface.messageBar().pushWarning(
            "AI Segmentation",
            tr("Zone too large. Please reduce the selection."),
        )
        return

    if len(tiles) == 1:
        # Single tile — use existing flow (unchanged)
        self._run_single_fal_detection()
        return

    # Multi-tile flow
    full_image = self._extract_full_zone_image(layer)
    if full_image is None:
        return

    geo_transform = self._compute_zone_geo_transform(layer)

    self._tiled_worker = TiledDetectionWorker(
        predictor=self.predictor,
        tiles=tiles,
        full_image=full_image,
        text_prompt=self.pro_dock.get_pro_text_prompt(),
        max_masks=self.pro_dock.get_max_objects(),
        score_threshold=self.pro_dock.get_score_threshold(),
        geo_transform_info=geo_transform,
        tile_manager=self._tile_manager,
    )
    self._tiled_worker.tile_completed.connect(self._on_tile_completed)
    self._tiled_worker.all_completed.connect(self._on_all_tiles_completed)
    self._tiled_worker.progress.connect(self._on_tile_progress)
    self._tiled_worker.error.connect(self._on_tile_error)
    self._tiled_worker.cancelled.connect(self._on_tile_cancelled)

    self.pro_dock.set_tile_progress(0, len(tiles))
    self._tiled_worker.start()
```

- [ ] **Step 5: Implement tile result handlers**

```python
def _on_tile_completed(self, tile_idx, detections):
    """Progressive display: show detections from this tile immediately."""
    from src.core.polygon_exporter import (
        apply_mask_refinement,
        deduplicate_geometries,
        mask_to_polygons,
    )

    for mask, score, ti in detections:
        refined = apply_mask_refinement(
            mask, self._refine_expand, self._refine_fill_holes,
            self._refine_min_area,
        )
        polys = mask_to_polygons(refined, ti)
        if not polys:
            continue
        geom = QgsGeometry.unaryUnion(polys)
        if geom.isEmpty():
            continue

        # Check IoU against existing detections
        existing_geoms = [d["geom"] for d in self._pro_pending_detections if "geom" in d]
        from src.core.polygon_exporter import _iou
        is_dup = any(_iou(geom, ag) > 0.3 for ag in existing_geoms)
        if is_dup:
            continue

        # Create rubber band and store
        rb = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
        rb.setColor(QColor(0, 255, 0, 60))
        rb.setStrokeColor(QColor(0, 255, 0, 200))
        rb.setWidth(2)
        rb.setToGeometry(geom)

        self._pro_pending_detections.append({
            "mask": mask,
            "score": score,
            "transform_info": ti,
            "rb": rb,
            "geom": geom,
        })

def _on_tile_progress(self, current, total):
    self.pro_dock.set_tile_progress(current, total)

def _on_all_tiles_completed(self, all_detections):
    self.pro_dock.hide_tile_progress()
    count = len(self._pro_pending_detections)
    self.iface.messageBar().pushSuccess(
        "AI Segmentation",
        tr("Detected {count} object(s)").format(count=count),
    )

def _on_tile_error(self, msg):
    self.pro_dock.hide_tile_progress()
    logger.error("Tiled detection failed: %s", msg)
    self.iface.messageBar().pushCritical("AI Segmentation", msg)

def _on_tile_cancelled(self):
    """User cancelled — discard everything."""
    self.pro_dock.hide_tile_progress()
    for det in self._pro_pending_detections:
        if "rb" in det:
            self.iface.mapCanvas().scene().removeItem(det["rb"])
    self._pro_pending_detections.clear()
```

- [ ] **Step 6: Add cancel button wiring**

In the stop segmentation handler, add:

```python
if self._tiled_worker and self._tiled_worker.isRunning():
    self._tiled_worker.cancel()
    self._tiled_worker.wait(5000)
```

- [ ] **Step 7: Implement helper methods**

```python
def _zone_to_pixels(self, zone_rect, layer):
    """Convert a geographic QgsRectangle to pixel dimensions for a raster layer."""
    extent = layer.extent()
    px_w = extent.width() / layer.width()
    px_h = extent.height() / layer.height()
    pixel_width = int(zone_rect.width() / px_w)
    pixel_height = int(zone_rect.height() / px_h)
    return max(1, pixel_width), max(1, pixel_height)

def _extract_full_zone_image(self, layer):
    """Extract the selected zone (or full image) at native resolution as numpy array."""
    # Use rasterio to read the zone window at native resolution
    # Returns (H, W, 3) uint8 numpy array
    # Implementation uses rasterio windowed read similar to feature_encoder.py
    ...

def _compute_zone_geo_transform(self, layer):
    """Compute the geographic bbox and pixel dimensions for the selected zone."""
    if self._selected_zone:
        rect = self._selected_zone
    else:
        rect = layer.extent()
    pixel_w, pixel_h = self._zone_to_pixels(rect, layer)
    return {
        "bbox": (rect.xMinimum(), rect.yMinimum(), rect.xMaximum(), rect.yMaximum()),
        "img_shape": (pixel_h, pixel_w),
    }
```

- [ ] **Step 8: Commit**

```bash
git add src/ui/ai_segmentation_plugin.py
git commit -m "feat: wire tiling into PRO detection flow with progressive display"
```

---

## Task 9: Implement `_extract_full_zone_image` with rasterio

**Files:**
- Modify: `src/ui/ai_segmentation_plugin.py`

- [ ] **Step 1: Implement the zone extraction method**

```python
def _extract_full_zone_image(self, layer):
    """Read the selected zone at native resolution using rasterio.

    Returns (H, W, 3) uint8 numpy array, or None on error.
    """
    import numpy as np

    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError:
        logger.error("rasterio not available for zone extraction")
        return None

    raster_path = layer.source()
    extent = layer.extent()

    if self._selected_zone:
        zone = self._selected_zone
    else:
        zone = extent

    try:
        with rasterio.open(raster_path) as src:
            # Convert geographic zone to pixel window
            px_size_x = extent.width() / src.width
            px_size_y = extent.height() / src.height

            col_off = max(0, int((zone.xMinimum() - extent.xMinimum()) / px_size_x))
            row_off = max(0, int((extent.yMaximum() - zone.yMaximum()) / px_size_y))
            win_w = min(int(zone.width() / px_size_x), src.width - col_off)
            win_h = min(int(zone.height() / px_size_y), src.height - row_off)

            if win_w <= 0 or win_h <= 0:
                return None

            window = Window(col_off, row_off, win_w, win_h)

            # Read RGB bands at native resolution
            band_count = min(src.count, 3)
            bands = src.read(
                list(range(1, band_count + 1)),
                window=window,
            )  # shape: (bands, H, W)

            if band_count == 1:
                rgb = np.stack([bands[0]] * 3, axis=-1)
            elif band_count == 2:
                rgb = np.stack([bands[0], bands[1], bands[0]], axis=-1)
            else:
                rgb = np.transpose(bands[:3], (1, 2, 0))

            return rgb.astype(np.uint8)

    except Exception as e:
        logger.error("Zone extraction failed: %s", str(e))
        return None
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/ai_segmentation_plugin.py
git commit -m "feat: implement native-resolution zone extraction with rasterio"
```

---

## Task 10: Integration test and cleanup

**Files:**
- Modify: `tests/test_tile_manager.py` (add integration-style tests)
- All modified files (cleanup pass)

- [ ] **Step 1: Add integration test for full pipeline math**

```python
# append to tests/test_tile_manager.py

def test_tile_transforms_cover_image():
    """Tile transforms should tile the geographic space without gaps."""
    from src.core.tile_manager import TileManager

    tm = TileManager(tile_size=1024, overlap_fraction=0.15, max_tiles=50)
    img_w, img_h = 3000, 2000
    tiles = tm.compute_grid(img_w, img_h)

    # Simulate geo_transform for a 3000x2000 image at [0,0] to [300,200]
    full_bbox = (0.0, 0.0, 300.0, 200.0)
    px_w = 300.0 / 3000
    px_h = 200.0 / 2000

    for x, y, w, h in tiles:
        tile_minx = full_bbox[0] + x * px_w
        tile_miny = full_bbox[1] + y * px_h
        tile_maxx = tile_minx + w * px_w
        tile_maxy = tile_miny + h * px_h
        # All tiles should be within image bounds
        assert tile_minx >= 0
        assert tile_miny >= 0
        assert tile_maxx <= 300.0 + 0.01  # small tolerance
        assert tile_maxy <= 200.0 + 0.01
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Run linter and formatter**

Run: `ruff check src/ tests/ && ruff format src/ tests/`
Expected: No errors

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "test: add integration tests for tiling pipeline"
```

---

## Summary

| Task | Description | New Files | Modified Files |
|------|-------------|-----------|----------------|
| 1 | Extract IoU dedup utility | — | polygon_exporter.py, plugin.py |
| 2 | TileManager grid computation | tile_manager.py | — |
| 3 | Tile crop extraction | — | tile_manager.py |
| 4 | FalPredictor array input | — | pro_predictor.py |
| 5 | ZoneSelectionMapTool | zone_selection_maptool.py | — |
| 6 | PRO dock UI updates | — | pro_dockwidget.py, i18n/*.ts |
| 7 | TiledDetectionWorker | tiled_detection_worker.py | — |
| 8 | Wire tiling into plugin | — | plugin.py |
| 9 | Zone image extraction | — | plugin.py |
| 10 | Integration tests + cleanup | — | test_tile_manager.py |
