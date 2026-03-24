# fal.ai Direct Integration - Local Testing Phase

**Date:** 2026-03-24
**Status:** Approved
**Supersedes:** `2026-03-23-fal-migration-design.md` (which covered full prod with Edge Function proxy)

## Objective

Replace Azure Container Apps SAM3 inference with direct fal.ai REST API calls for local development and testing. Text prompt mode only. No proxy, no auth layer, no Supabase dependency. Goal: validate pricing, latency, UX, and mask quality before building the production auth layer.

## Scope

**In scope:**
- Direct REST calls to `https://fal.run/fal-ai/sam-3/image-rle`
- fal.ai key read from `.env` at plugin root
- Text prompt detection (single request per canvas view, no tiling)
- RLE COCO decoding to numpy masks
- Removal of warmup/retry/session logic

**Out of scope (deferred to production phase):**
- Supabase Edge Function proxy
- Per-user API key validation
- Point prompt mode via fal.ai
- Usage metering / rate limiting

## Architecture

```
User types prompt in PRO dock
    |
    v
Plugin extracts crop from canvas (_extract_and_encode_crop)
    |
    v
JPEG base64 -> data URI "data:image/jpeg;base64,..."
    |
    v
POST https://fal.run/fal-ai/sam-3/image-rle
  Authorization: Key <FAL_KEY from .env>
  Body: {image_url, prompt, return_multiple_masks, max_masks, include_scores, include_boxes}
    |
    v
Response: {rle: [...], scores: [...], boxes: [...]}
    |
    v
decode_rle_to_mask (COCO RLE Fortran order -> numpy bool H,W)
    |
    v
mask_to_polygons -> rubber bands (existing pipeline, unchanged)
```

## Components

### 1. `src/core/pro_predictor.py` (rewrite)

`CloudSam3Predictor` (304 lines) replaced by `FalPredictor` (~80 lines).

```
class FalPredictor:
    __init__(fal_key: str)
    set_image(image_np: ndarray) -> None     # stores locally, no network
    predict_text(prompt: str) -> dict         # POST fal.ai, returns {rle, scores, boxes}
    reset_image() -> None
    cleanup() -> None
```

Also contains standalone `decode_rle_to_mask(rle: dict) -> ndarray`:
- Input: single RLE dict `{"counts": [n0, n1, ...], "size": [H, W]}`
- Output: numpy bool array `(H, W)`
- Handles zero-length first run (first pixel is foreground)
- Reshapes in Fortran order (column-major) then transposes to row-major

Key differences from Azure:
- `set_image` is a local no-op (stores numpy array), not a network call
- `predict_text` sends the full image each time (stateless, no sessions)
- Auth via `Authorization: Key ...` header (not `X-Api-Key`)
- Response is RLE COCO (not base64 zlib numpy)
- No warmup, no retry, no session TTL

### 2. `src/core/model_config.py` (modify)

Add constant (generic name per CLAUDE.md confidentiality rules):
```python
SAM3_INFERENCE_URL = "https://fal.run/fal-ai/sam-3/image-rle"
```

Keep `SAM3_CLOUD_URL` for now (no deletion, avoids breaking anything).

### 3. `src/ui/ai_segmentation_plugin.py` (modify)

**Removed:**
- `_CloudWarmupWorker` class
- Warmup/retry logic in `_on_start_pro_segmentation`
- `_run_pro_detection` (interactive point mode)
- `_warmup_thread`, `_warmup_worker` attributes

**Simplified:**
- `_on_start_pro_segmentation`: read FAL_KEY from .env, create FalPredictor, activate mode (~40 lines)
- `_run_pro_text_detection` replaced by `_run_fal_detection`: single request, no tiling

**`_run_fal_detection` integration loop:**
1. Get text prompt from dock
2. Extract crop via `_extract_and_encode_crop` (calls `predictor.set_image` internally)
3. Call `predictor.predict_text(prompt)` -> raw dict
4. Normalize `result["rle"]` to list (may be single dict)
5. For each RLE: `decode_rle_to_mask(rle)` -> numpy bool mask
6. Filter by score threshold
7. `mask_to_polygons(mask, transform_info)` -> rubber bands (existing pipeline)

**Resolution note:** Without tiling, the entire canvas view is sent as a single JPEG crop. For very large extents, resolution may be low. The existing `_extract_and_encode_crop` already handles crop sizing. Testing should measure mask quality at various zoom levels.

**Unchanged:**
- `_extract_and_encode_crop`
- `_transform_geometry_to_canvas_crs`
- Polygon save pipeline
- SAM1/SAM2 local modes

### 4. `.env` (create, gitignored)

```
FAL_KEY=<your-fal-key-here>
```

`.env` reading: simple parser function in `ai_segmentation_plugin.py` (the orchestrator). Key is passed to `FalPredictor.__init__(fal_key=...)`. The predictor itself never reads `.env`.

### 5. `.gitignore` (verify)

Ensure `.env` is listed.

## fal.ai API Details

**Endpoint:** `POST https://fal.run/fal-ai/sam-3/image-rle`

**Request:**
```json
{
  "image_url": "data:image/jpeg;base64,...",
  "prompt": "building",
  "apply_mask": false,
  "return_multiple_masks": true,
  "max_masks": 10,
  "include_scores": true,
  "include_boxes": true
}
```

**Important:** Use `prompt` (not `text_prompt`, which is deprecated). Set `apply_mask: false` to avoid unnecessary image compositing in the response.

`max_masks: 10` chosen to cover large scenes with many objects. Default is 3. Performance impact of higher values to be measured during testing.

**Response:**
```json
{
  "rle": [{"counts": [n0, n1, ...], "size": [H, W]}, ...],
  "scores": [0.95, 0.87, ...],
  "boxes": [[cx, cy, w, h], ...]
}
```

**Response normalization:** When `return_multiple_masks` is `true`, `rle` is a list. When `false` or single result, it may be a single dict. The implementation must normalize to always work with a list.

**Pricing:** $0.005 per request. 50 detections = $0.25.

**JPEG quality:** `quality=90` (same as current Azure flow). Impact on mask quality vs upload size to be measured during testing.

**RLE format:** COCO uncompressed RLE, Fortran (column-major) order. Counts alternate between 0-pixels (background) and 1-pixels (foreground). First count is always background (may be 0 if the first pixel is foreground).

## Error Handling

| HTTP Code | Meaning | User Message |
|-----------|---------|-------------|
| 401 | Invalid fal.ai key | "Invalid API key. Check FAL_KEY in .env" |
| 422 | No results for prompt | "No objects found for '{prompt}'" |
| 502/503 | fal.ai service down | "Service temporarily unavailable" |
| Timeout (60s) | Slow response | "Request timed out, try again" |
| Network error | No connectivity | "Cannot reach inference service" |

No automatic retry. Serverless = no cold start visible to the user.

## What Does Not Change

- `api_keys` table in Supabase (untouched)
- User PRO keys in QSettings (remain, unused during dev)
- PRO dock widget UI (text field, Detect button, score threshold)
- `mask_to_polygons` pipeline
- SAM1/SAM2 local inference
- i18n strings (no new UI strings needed)

## Testing Plan

**Manual QGIS tests:**
- No key -> warning message
- Invalid key -> 401 error message
- Valid detection -> rubber bands in ~5-15s
- Empty prompt -> no action
- No results -> info message
- Save polygon -> works as before
- Score threshold -> filters detections

**Pricing/performance to measure:**
- Latency per request (crop sizes: 1024x1024, 2048x2048, 4096x4096)
- Cost per typical work session (~50 detections)
- Mask quality comparison vs Azure SAM3

**Unit test:**
- `decode_rle_to_mask` standalone test (pure numpy, no QGIS dependency)

## Migration Path to Production

When ready for prod, the only changes will be:
1. Create Supabase Edge Function proxy (receives user API key, calls fal.ai with TerraLab key)
2. Change `FalPredictor` to POST to Edge Function URL instead of `fal.run`
3. Send `api_key` (user's tl_pro key) instead of `Authorization: Key` header
4. Remove `.env` FAL_KEY reading, use QSettings PRO key instead
