# fal.ai Local Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Azure SAM3 inference with direct fal.ai REST API calls for local testing. Text prompt only, no proxy, no auth layer.

**Architecture:** Plugin encodes the canvas crop as JPEG data URI, POSTs to `https://fal.run/fal-ai/sam-3/image-rle` with `Authorization: Key` header, decodes RLE response to numpy masks, feeds into existing `mask_to_polygons` pipeline. fal.ai key read from `.env` at plugin root.

**Tech Stack:** Python 3.9+ (urllib.request, numpy, PIL), fal.ai REST API, QSettings (existing), QGIS API

**Spec:** `docs/superpowers/specs/2026-03-24-fal-local-testing-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/core/pro_predictor.py` | **REWRITE** | `FalPredictor` class + `decode_rle_to_mask` utility |
| `src/core/model_config.py` | **MODIFY** line 75-79 | Add `SAM3_INFERENCE_URL`, keep `SAM3_CLOUD_URL` |
| `src/ui/ai_segmentation_plugin.py` | **MODIFY** | Remove warmup, simplify start PRO, replace text detection |
| `.env` | **CREATE** (gitignored) | `FAL_KEY=<key>` |
| `tests/test_rle_decode.py` | **CREATE** | Unit test for `decode_rle_to_mask` |

---

## Task 1: Create `FalPredictor` and `decode_rle_to_mask`

**Files:**
- Rewrite: `src/core/pro_predictor.py`
- Modify: `src/core/model_config.py:75-79`
- Create: `tests/test_rle_decode.py`

### Context

`CloudSam3Predictor` (304 lines) is replaced by `FalPredictor` (~80 lines). The class calls `https://fal.run/fal-ai/sam-3/image-rle` directly via `urllib.request`. No sessions, no warmup, no zlib, no retry. The key difference: `set_image` is a local no-op (stores numpy), and `predict_text` sends the full image as a JPEG data URI each time.

`decode_rle_to_mask` converts COCO uncompressed RLE (Fortran column-major order) to a numpy bool array `(H, W)`.

- [ ] **Step 1.1: Add `SAM3_INFERENCE_URL` to model_config.py**

In `src/core/model_config.py`, after line 79 (`SAM3_CLOUD_URL` closing paren), add:

```python
SAM3_INFERENCE_URL = "https://fal.run/fal-ai/sam-3/image-rle"
```

Keep `SAM3_CLOUD_URL` unchanged (no deletion).

- [ ] **Step 1.2: Write `decode_rle_to_mask` test**

Create `tests/test_rle_decode.py`:

```python
"""Tests for COCO RLE decoding (no QGIS dependency)."""
import numpy as np
import sys
import types

# Stub QGIS modules so pro_predictor can be imported outside QGIS
for mod_name in ("qgis", "qgis.core"):
    sys.modules[mod_name] = types.ModuleType(mod_name)
sys.modules["qgis.core"].QgsMessageLog = type(
    "QgsMessageLog", (), {"logMessage": staticmethod(lambda *a, **kw: None)}
)()
sys.modules["qgis.core"].Qgis = type("Qgis", (), {"MessageLevel": type("ML", (), {"Info": 0, "Warning": 1})()})()

from src.core.pro_predictor import decode_rle_to_mask


def test_single_pixel_top_left():
    """2x2 mask, only top-left pixel is foreground."""
    # Fortran order: col0=[px(0,0), px(1,0)], col1=[px(0,1), px(1,1)]
    # counts: 0 bg, 1 fg, 3 bg -> px(0,0)=1, rest=0
    rle = {"counts": [0, 1, 3], "size": [2, 2]}
    mask = decode_rle_to_mask(rle)
    assert mask.shape == (2, 2)
    assert mask.dtype == bool
    assert mask[0, 0] is np.True_
    assert mask[1, 0] is np.False_
    assert mask[0, 1] is np.False_
    assert mask[1, 1] is np.False_


def test_all_foreground():
    """3x3 mask, all pixels foreground."""
    rle = {"counts": [0, 9], "size": [3, 3]}
    mask = decode_rle_to_mask(rle)
    assert mask.shape == (3, 3)
    assert mask.all()


def test_all_background():
    """3x3 mask, all pixels background."""
    rle = {"counts": [9], "size": [3, 3]}
    mask = decode_rle_to_mask(rle)
    assert mask.shape == (3, 3)
    assert not mask.any()


def test_checkerboard_column_order():
    """2x2 mask with Fortran-order checkerboard pattern."""
    # Fortran flat: [px(0,0)=1, px(1,0)=0, px(0,1)=0, px(1,1)=1]
    # counts: 0 bg, 1 fg, 2 bg, 1 fg
    rle = {"counts": [0, 1, 2, 1], "size": [2, 2]}
    mask = decode_rle_to_mask(rle)
    assert mask[0, 0] is np.True_
    assert mask[1, 0] is np.False_
    assert mask[0, 1] is np.False_
    assert mask[1, 1] is np.True_
```

- [ ] **Step 1.3: Run test to verify it fails**

```bash
cd "/Users/lilien/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation-Pro"
python3 -m pytest tests/test_rle_decode.py -v
```

Expected: FAIL with `ImportError` or `ModuleNotFoundError` (function doesn't exist yet).

- [ ] **Step 1.4: Write `pro_predictor.py` with `FalPredictor` and `decode_rle_to_mask`**

Replace `src/core/pro_predictor.py` entirely:

```python
"""FalPredictor - calls fal.ai SAM3 image-rle endpoint directly."""
import base64
import io
import json
import urllib.error
import urllib.request
from typing import Optional, Tuple

import numpy as np
from PIL import Image as PILImage
from qgis.core import Qgis, QgsMessageLog

from .model_config import SAM3_INFERENCE_URL

_TIMEOUT_PREDICT = 60


def decode_rle_to_mask(rle: dict) -> np.ndarray:
    """Decode COCO uncompressed RLE to a boolean numpy mask (H, W).

    rle format: {"counts": [n0, n1, n2, ...], "size": [H, W]}
    Counts alternate between background (0) and foreground (1) pixels
    in Fortran (column-major) order. First count is always background.
    """
    counts = rle["counts"]
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:
            flat[pos : pos + count] = 1
        pos += count
    return flat.reshape((w, h)).T.astype(bool)


class FalPredictor:
    """Stateless predictor that calls fal.ai SAM3 image-rle endpoint."""

    def __init__(self, fal_key: str) -> None:
        self._fal_key = fal_key
        self._last_image_np: Optional[np.ndarray] = None
        self.is_image_set = False
        self.original_size: Optional[Tuple[int, int]] = None

    def set_image(self, image_np: np.ndarray) -> None:
        """Store image locally for subsequent predict_text calls."""
        self._last_image_np = image_np
        self.original_size = (image_np.shape[0], image_np.shape[1])
        self.is_image_set = True
        QgsMessageLog.logMessage(
            "FalPredictor: image stored ({}x{})".format(
                image_np.shape[1], image_np.shape[0]
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

    def predict_text(self, prompt: str) -> dict:
        """Send image + prompt to fal.ai, return raw response dict.

        Returns dict with keys: rle, scores, boxes.
        Raises RuntimeError on HTTP errors or timeout.
        """
        if self._last_image_np is None:
            raise RuntimeError("No image set. Call set_image first.")

        # Encode image as JPEG data URI
        pil_img = PILImage.fromarray(self._last_image_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_uri = "data:image/jpeg;base64,{}".format(b64)

        payload = json.dumps({
            "image_url": data_uri,
            "prompt": prompt,
            "apply_mask": False,
            "return_multiple_masks": True,
            "max_masks": 10,
            "include_scores": True,
            "include_boxes": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            SAM3_INFERENCE_URL,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Key {}".format(self._fal_key),
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_PREDICT) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                rle_data = result.get("rle", [])
                n_masks = len(rle_data) if isinstance(rle_data, list) else 1
                QgsMessageLog.logMessage(
                    "FalPredictor: prediction OK ({} masks)".format(n_masks),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )
                return result
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(
                    e.read().decode("utf-8")
                ).get("detail", "")
            except Exception:
                pass
            if e.code == 401:
                raise RuntimeError(
                    "Invalid API key. Check FAL_KEY in .env"
                )
            raise RuntimeError(
                "Inference error {}: {}".format(e.code, detail)
            )
        except urllib.error.URLError as e:
            raise RuntimeError(
                "Cannot reach inference service: {}".format(e.reason)
            )

    def reset_image(self) -> None:
        """Clear stored image state."""
        self._last_image_np = None
        self.is_image_set = False
        self.original_size = None

    def cleanup(self) -> None:
        """Release resources (compatibility with existing cleanup calls)."""
        self.reset_image()
```

- [ ] **Step 1.5: Run tests to verify they pass**

```bash
cd "/Users/lilien/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation-Pro"
python3 -m pytest tests/test_rle_decode.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 1.6: Lint check**

```bash
cd "/Users/lilien/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation-Pro"
ruff check src/core/pro_predictor.py src/core/model_config.py
```

Expected: no errors.

- [ ] **Step 1.7: Commit**

```bash
cd "/Users/lilien/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation-Pro"
git add src/core/pro_predictor.py src/core/model_config.py tests/test_rle_decode.py
git commit -m "feat: replace CloudSam3Predictor with FalPredictor (serverless inference)"
```

---

## Task 2: Simplify `_on_start_pro_segmentation` and remove warmup

**Files:**
- Modify: `src/ui/ai_segmentation_plugin.py:88-110` (remove `_CloudWarmupWorker`)
- Modify: `src/ui/ai_segmentation_plugin.py:189-190` (remove warmup attrs)
- Modify: `src/ui/ai_segmentation_plugin.py:2663-2810` (rewrite start PRO)

### Context

The current `_on_start_pro_segmentation` (lines 2663-2810, 148 lines) does: read PRO API key from QSettings, create `CloudSam3Predictor`, sync auth pre-check, show warmup dialog with retry, wait for thread, setup CRS/layer. The new version: read `FAL_KEY` from `.env`, create `FalPredictor`, setup CRS/layer. No warmup, no thread, no dialog. ~40 lines.

`_CloudWarmupWorker` (lines 88-110) and the warmup attributes (lines 189-190) become dead code once the start method is rewritten. Remove them.

- [ ] **Step 2.1: Remove `_CloudWarmupWorker` class**

Delete lines 88-110 in `src/ui/ai_segmentation_plugin.py` (the entire class):

```python
class _CloudWarmupWorker(QObject):
    finished = pyqtSignal()
    attempt_started = pyqtSignal(int, int)  # (attempt_num, max_attempts)
    ...
        self.finished.emit()
```

- [ ] **Step 2.2: Remove warmup attributes from `__init__`**

Delete lines 189-190:

```python
        self._warmup_thread = None
        self._warmup_worker = None
```

- [ ] **Step 2.3: Rewrite `_on_start_pro_segmentation`**

Replace lines 2663-2810 entirely with:

```python
    def _on_start_pro_segmentation(self, layer: QgsRasterLayer):
        """Start PRO (SAM 3) segmentation via fal.ai."""
        import pathlib

        from ..core.pro_predictor import FalPredictor
        from ..core.venv_manager import ensure_venv_packages_available

        ensure_venv_packages_available()

        # Read FAL_KEY from .env at plugin root
        env_path = pathlib.Path(__file__).parent.parent.parent / ".env"
        fal_key = ""
        if env_path.exists():
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("FAL_KEY="):
                        fal_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

        if not fal_key:
            QMessageBox.warning(
                self.iface.mainWindow(),
                tr("AI Segmentation PRO"),
                tr(
                    "No API key configured.\n\n"
                    "Add FAL_KEY=your_key to the .env file\n"
                    "at the plugin root directory."
                ),
            )
            return

        # Clean up previous predictor
        if self.predictor:
            try:
                self.predictor.cleanup()
            except Exception:
                pass
        self.predictor = FalPredictor(fal_key=fal_key)

        # Validate layer
        if not self._is_layer_valid(layer):
            QgsMessageLog.logMessage(
                "Layer was deleted before PRO segmentation could start",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            self.predictor.cleanup()
            self.predictor = None
            return

        try:
            layer_name = layer.name().replace(" ", "_")
            raster_path = os.path.normcase(layer.source())
        except RuntimeError:
            self.predictor.cleanup()
            self.predictor = None
            return

        self._reset_session()
        self._current_layer = layer
        self._current_layer_name = layer_name
        self._is_online_layer = self._is_online_provider(layer)
        self._is_non_georeferenced_mode = (
            not self._is_online_layer and not self._is_layer_georeferenced(layer)
        )
        self._current_raster_path = raster_path

        # Set up CRS transforms
        canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        raster_crs = layer.crs() if layer else None
        self._canvas_to_raster_xform = None
        self._raster_to_canvas_xform = None
        if raster_crs and canvas_crs.isValid() and raster_crs.isValid():
            if canvas_crs != raster_crs:
                self._canvas_to_raster_xform = QgsCoordinateTransform(
                    canvas_crs, raster_crs, QgsProject.instance()
                )
                self._raster_to_canvas_xform = QgsCoordinateTransform(
                    raster_crs, canvas_crs, QgsProject.instance()
                )

        self._active_mode = "pro"
        self._activate_segmentation_tool()
        QgsMessageLog.logMessage(
            "PRO mode activated",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
```

- [ ] **Step 2.4: Lint check**

```bash
cd "/Users/lilien/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation-Pro"
ruff check src/ui/ai_segmentation_plugin.py --select F401,F841,E501
```

Expected: no new errors. If the `QProgressDialog` import (previously used by warmup dialog) or other imports became unused, remove them.

- [ ] **Step 2.5: Commit**

```bash
cd "/Users/lilien/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation-Pro"
git add src/ui/ai_segmentation_plugin.py
git commit -m "refactor: remove warmup/retry logic, simplify PRO start for serverless"
```

---

## Task 3: Replace `_run_pro_text_detection` with `_run_fal_detection`

**Files:**
- Modify: `src/ui/ai_segmentation_plugin.py:3009-3230` (replace text detection)
- Modify: `src/ui/ai_segmentation_plugin.py:380` (reconnect signal)

### Context

`_run_pro_text_detection` (lines 3009-3230, 222 lines) does adaptive tiling (2-6 tiles per axis), iterates tiles, calls `predictor.predict(text_prompt=...)` per tile, deduplicates via IoU. The new `_run_fal_detection` sends a single request on the canvas crop center, decodes RLE, and feeds into `mask_to_polygons`. ~80 lines.

The signal connection at line 380 must be updated from `_run_pro_text_detection` to `_run_fal_detection`.

- [ ] **Step 3.1: Replace `_run_pro_text_detection` with `_run_fal_detection`**

Replace lines 3009-3230 entirely with:

```python
    def _run_fal_detection(self):
        """Detect objects matching the text prompt on the current canvas view."""
        import numpy as np

        from ..core.pro_predictor import decode_rle_to_mask

        if not self._active_dock or not self.predictor:
            return
        text_prompt = self._active_dock.get_pro_text_prompt()
        if not text_prompt:
            return

        raster_layer = self._current_layer
        if raster_layer is None:
            return

        canvas = self.iface.mapCanvas()
        canvas_extent = canvas.extent()
        canvas_center = canvas_extent.center()

        # Convert canvas center to raster CRS if needed
        canvas_crs = canvas.mapSettings().destinationCrs()
        layer_crs = raster_layer.crs()
        if canvas_crs != layer_crs:
            xform = QgsCoordinateTransform(
                canvas_crs, layer_crs, QgsProject.instance()
            )
            raster_center = xform.transform(canvas_center)
        else:
            raster_center = canvas_center

        # Extract and store the image (calls predictor.set_image internally)
        if not self._extract_and_encode_crop(raster_center):
            return

        if self._current_crop_info is None:
            return

        crop_bounds = self._current_crop_info["bounds"]
        img_shape = self._current_crop_info["img_shape"]
        img_height, img_width = img_shape
        minx, miny, maxx, maxy = crop_bounds

        crs_value = None
        try:
            if self._current_layer and self._current_layer.crs().isValid():
                crs_value = self._current_layer.crs().authid()
        except RuntimeError:
            pass

        transform_info = {
            "bbox": (minx, miny, maxx, maxy),
            "img_shape": (img_height, img_width),
            "crs": crs_value,
        }

        score_threshold = (
            self._active_dock.get_score_threshold()
            if self._active_dock
            else 0.0
        )

        self.iface.messageBar().pushMessage(
            tr("AI Segmentation"),
            tr("Detecting '{prompt}'...").format(prompt=text_prompt),
            level=Qgis.MessageLevel.Info,
            duration=0,
        )
        QApplication.processEvents()

        try:
            result = self.predictor.predict_text(text_prompt)
        except RuntimeError as e:
            self.iface.messageBar().clearWidgets()
            QMessageBox.warning(
                self.iface.mainWindow(),
                tr("Detection Error"),
                str(e),
            )
            return

        self.iface.messageBar().clearWidgets()

        # Normalize rle to list
        rle_list = result.get("rle", [])
        if isinstance(rle_list, dict):
            rle_list = [rle_list]
        scores_list = result.get("scores") or []

        all_detections = []
        for i, rle in enumerate(rle_list):
            score = float(scores_list[i]) if i < len(scores_list) else 1.0
            if score < score_threshold:
                continue
            try:
                mask = decode_rle_to_mask(rle)
            except Exception as e:
                QgsMessageLog.logMessage(
                    "RLE decode error for mask {}: {}".format(i, e),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Warning,
                )
                continue
            all_detections.append((mask, score, transform_info))

        if not all_detections:
            self.iface.messageBar().pushMessage(
                tr("AI Segmentation"),
                tr("No objects found for '{prompt}'.").format(
                    prompt=text_prompt
                ),
                level=Qgis.MessageLevel.Info,
                duration=5,
            )
            return

        from ..core.polygon_exporter import mask_to_polygons

        all_detections.sort(key=lambda x: x[1], reverse=True)

        IOU_THRESHOLD = 0.3

        def _iou(g1, g2):
            inter = g1.intersection(g2)
            if inter.isEmpty():
                return 0.0
            union = g1.combine(g2)
            return inter.area() / union.area() if union.area() > 0 else 0.0

        accepted_geoms = []
        batch_count = 0

        for mask, score, ti in all_detections:
            if mask.sum() < 20:
                continue
            polys = mask_to_polygons(mask, ti)
            if not polys:
                continue
            geom = QgsGeometry.unaryUnion(polys)
            if not geom or geom.isEmpty():
                continue
            if any(_iou(geom, ag) > IOU_THRESHOLD for ag in accepted_geoms):
                continue
            accepted_geoms.append(geom)

            rb = QgsRubberBand(
                self.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry
            )
            rb.setColor(QColor(0, 200, 100, 120))
            rb.setFillColor(QColor(0, 200, 100, 80))
            rb.setWidth(2)
            display_geom = QgsGeometry(geom)
            self._transform_geometry_to_canvas_crs(display_geom)
            rb.setToGeometry(display_geom, None)

            self._pro_pending_detections.append(
                {
                    "mask": mask,
                    "score": score,
                    "transform_info": ti.copy(),
                    "rb": rb,
                }
            )
            batch_count += 1

        self._clear_mask_visualization()
        self.current_mask = None
        self.current_transform_info = None

        if batch_count > 0:
            self._pro_detection_batches.append(batch_count)
            if self._active_dock:
                pos = len(self._pro_detection_batches)
                self._active_dock.set_point_count(pos, 0)
                self._active_dock.set_batch_done(batch_count)
                self._active_dock.set_mask_available(True)
            self.iface.messageBar().pushMessage(
                tr("AI Segmentation"),
                tr("{n} object(s) detected. Review and save.").format(
                    n=batch_count
                ),
                level=Qgis.MessageLevel.Success,
                duration=5,
            )
```

- [ ] **Step 3.2: Update signal connection**

At line 380, replace:

```python
self.pro_dock_widget.pro_detect_requested.connect(self._run_pro_text_detection)
```

with:

```python
self.pro_dock_widget.pro_detect_requested.connect(self._run_fal_detection)
```

- [ ] **Step 3.3: Update i18n .ts files**

The following new `tr()` strings were introduced across Tasks 2 and 3. Add `<message>` blocks for each in `i18n/ai_segmentation_fr.ts`, `i18n/ai_segmentation_pt_BR.ts`, and `i18n/ai_segmentation_es.ts` inside `<context><name>AISegmentation</name>`:

New strings:
- `"No API key configured.\n\nAdd FAL_KEY=your_key to the .env file\nat the plugin root directory."`
- `"Detecting '{prompt}'..."`
- `"No objects found for '{prompt}'."`
- `"{n} object(s) detected. Review and save."`
- `"Detection Error"`

Provide accurate translations for each language (fr, pt_BR, es).

- [ ] **Step 3.4: Lint check**

```bash
cd "/Users/lilien/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation-Pro"
ruff check src/ui/ai_segmentation_plugin.py
```

Fix any unused imports (e.g., `math` was used by tiling, may now be unused).

- [ ] **Step 3.5: Commit**

```bash
cd "/Users/lilien/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation-Pro"
git add src/ui/ai_segmentation_plugin.py i18n/ai_segmentation_fr.ts i18n/ai_segmentation_pt_BR.ts i18n/ai_segmentation_es.ts
git commit -m "feat: replace tiled text detection with single-request detection"
```

---

## Task 4: Remove dead code (`_run_pro_detection`, `_pro_reference_set`)

**Files:**
- Modify: `src/ui/ai_segmentation_plugin.py`

### Context

`_run_pro_detection` (lines 2812-2932) is a point-click handler for interactive PRO mode. It is defined but never called anywhere (dead code since text-only PRO was introduced). `_pro_reference_set` (line 188) is only used inside `_run_pro_detection` and `_run_pro_text_detection` (which was replaced in Task 3). Both can be safely removed.

- [ ] **Step 4.1: Delete `_run_pro_detection` and `_apply_pro_detection_results` methods**

Search for `def _run_pro_detection` (originally line 2812) and delete the entire method. Then search for `def _apply_pro_detection_results` (originally line 2934, only called from `_run_pro_detection`) and delete that entire method too. Both are dead code after removing point-click mode.

- [ ] **Step 4.2: Remove `_pro_reference_set` attribute**

Delete the line `self._pro_reference_set: bool = False` from `__init__` (originally line 188).

Also remove the line `self._pro_reference_set = False` from `_reset_session()` (originally line 2628).

- [ ] **Step 4.3: Verify no remaining references**

```bash
cd "/Users/lilien/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation-Pro"
grep -n "CloudSam3Predictor\|SAM3_CLOUD_URL\|_CloudWarmupWorker\|_run_pro_detection\|_pro_reference_set\|_warmup_thread\|_warmup_worker\|_run_pro_text_detection\|_apply_pro_detection_results" src/ui/ai_segmentation_plugin.py src/core/pro_predictor.py src/core/model_config.py
```

Expected: no matches.

- [ ] **Step 4.4: Lint and format**

```bash
cd "/Users/lilien/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation-Pro"
ruff check src/ui/ai_segmentation_plugin.py src/core/pro_predictor.py
ruff format src/ui/ai_segmentation_plugin.py src/core/pro_predictor.py
```

- [ ] **Step 4.5: Commit**

```bash
cd "/Users/lilien/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation-Pro"
git add src/ui/ai_segmentation_plugin.py
git commit -m "refactor: remove dead PRO point detection code and _pro_reference_set"
```

---

## Task 5: Create `.env` and run end-to-end test

**Files:**
- Create: `.env` (gitignored)

### Context

Create the `.env` file with the fal.ai key. Then verify everything works manually in QGIS.

- [ ] **Step 5.1: Create `.env` file**

Create `.env` at the plugin root:

```
FAL_KEY=<paste-your-fal-key-here>
```

Verify it's gitignored:

```bash
cd "/Users/lilien/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation-Pro"
git status
```

Expected: `.env` does NOT appear in untracked files.

- [ ] **Step 5.2: Run unit tests**

```bash
cd "/Users/lilien/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation-Pro"
python3 -m pytest tests/test_rle_decode.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5.3: Manual test in QGIS — no key**

1. Rename `.env` to `.env.bak` temporarily
2. Open QGIS, load a raster layer
3. Open the PRO dock, click "Start PRO"
4. **Expected:** Warning dialog "No API key configured. Add FAL_KEY=your_key to the .env file"
5. Restore `.env` from `.env.bak`

- [ ] **Step 5.4: Manual test in QGIS — valid detection**

1. Restart QGIS (or reload plugin)
2. Load a raster with visible buildings/objects
3. Open PRO dock, click "Start PRO" — should activate immediately (no warmup dialog!)
4. Type "building" in the text prompt field
5. Click "Detect"
6. **Expected:** "Detecting 'building'..." message bar, then rubber bands appear in ~5-15s
7. Check QGIS Log Messages > "AI Segmentation" for:
   - `FalPredictor: image stored (WxH)`
   - `FalPredictor: prediction OK (N masks)`

- [ ] **Step 5.5: Manual test — save polygon**

1. After detection, click on a rubber band to select it
2. Click "Save polygon"
3. **Expected:** Polygon saved to vector layer (same as before)

- [ ] **Step 5.6: Manual test — score threshold**

1. Run a detection
2. Increase score threshold to 0.8 in the dock widget
3. Run detection again
4. **Expected:** Fewer detections (low-confidence masks filtered)

- [ ] **Step 5.7: Note pricing and latency**

After several detections, note:
- Average latency per detection request
- Image size sent (check QGIS log for dimensions)
- Any quality differences vs Azure SAM3

---

## Execution Order

Tasks are sequential: **1 → 2 → 3 → 4 → 5**

- Task 1 creates the predictor (dependency for everything else)
- Task 2 rewrites the start method to use `FalPredictor`
- Task 3 rewrites the detection method to use `predict_text` + RLE decode
- Task 4 cleans up dead code
- Task 5 creates `.env` and validates end-to-end
