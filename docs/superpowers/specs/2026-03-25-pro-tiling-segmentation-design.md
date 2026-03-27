# PRO Mode — Adaptive Tiling Segmentation

**Date:** 2026-03-25
**Status:** Draft

## Problem

PRO mode text-prompted segmentation has two hard limits:
1. **32 objects max** per inference (fal.ai API limit, not model limit)
2. **Quality loss** from downscaling large images to 1024x1024 before inference

Users with large rasters (drone orthoimages, high-res satellite) containing hundreds of objects (houses, trees, etc.) hit both limits simultaneously.

## Solution

Adaptive tiling: intelligently split the selected zone into 1024x1024 tiles at native resolution, run each tile through the same fal.ai SAM-3 endpoint in parallel, and merge results using existing IoU deduplication.

## User Flow

1. User loads raster in QGIS, launches AI Segmentation PRO
2. By default, the entire image is the working zone
3. User can optionally draw a rectangle on the map to reduce the zone
4. Dock displays estimated credits with explanatory text: "Plus la zone selectionnee contient de pixels, plus l'operation consomme de credits"
5. Credit count updates dynamically when zone selection changes
6. User enters text prompt ("maisons", "arbres"...)
7. User clicks Run
8. Detections appear progressively on the map, tile by tile
9. Refine panel available on the merged result (existing logic)

## Tiling Strategy

### Adaptive behavior based on image resolution

- **Zone <= 1024x1024 native pixels**: no tiling, single API call (identical to current behavior)
- **Zone > 1024x1024 native pixels**: tile grid with overlap

### Tile grid parameters

- **Tile size**: 1024x1024 native pixels (max model input)
- **Overlap**: ~15-20% between adjacent tiles (to capture objects at boundaries)
- **Edge tiles**: may be smaller than 1024x1024, sent as-is or padded
- **Max tiles per run**: 50 (hard cap)
- **Processing order**: row by row, top-left to bottom-right

### Parallel execution

- All tiles submitted in parallel to fal.ai endpoint
- fal.ai queues excess requests automatically (no rejection, built-in retry in SDK)
- Concurrency depends on account tier: 2 (new) to 40 (self-serve max)
- Each tile sent with the same text prompt

### Cost

- ~0.5 cent per tile/inference
- 1 credit = 1 tile
- User sees credit estimate before launching

## Fusion and Deduplication

### Pipeline (per tile, as results arrive)

1. Tile returns N masks (max 32 per tile)
2. Each mask converted to geometry via existing `mask_to_polygons` + `unaryUnion`
3. Polygon coordinates reprojected to full image reference frame (apply tile x,y offset)
4. Each polygon checked against all previously accepted polygons using IoU

### Deduplication (existing logic reused)

- **IoU threshold**: 0.3 (existing value from PRO mode)
- If new polygon IoU > 0.3 with any accepted polygon: skip (duplicate)
- If IoU <= 0.3: accept
- **Min pixel filter**: 20px (existing)

### Edge cases

- Objects in overlap zone: detected by both tiles, first detection wins, duplicate eliminated by IoU
- Very large objects spanning beyond overlap: may produce partial polygons with IoU < 0.3. Post-processing merge (unaryUnion on touching/slightly overlapping polygons) can be added as future iteration.

## Refinement

Existing refine panel applied to merged global result:
- expand/contract
- simplify
- fill_holes
- min_area

No changes to refine logic.

## UI Changes

### New elements in PRO dock

- **"Select zone" button**: activates rectangle drawing tool on map canvas. Drawn zone is highlighted. If not used, entire image is selected.
- **Credit indicator**: dynamic text showing estimated credits for current zone. Accompanied by explanatory message about size/credit relationship.
- **Progress bar**: appears during processing, shows "Tile X/N" and percentage. Cancel button alongside.

### Unchanged

- Text prompt field
- Run/Start button (triggers tiling automatically when needed)
- Refine panel
- Save/export polygon workflow

### Smart behavior

- Small images (<=1024x1024): identical to current behavior, no visible tiling, 1 credit
- Large images: tiling activates transparently, user sees only credits + progress

### Error messages

- Zone too large (>50 tiles): "La zone selectionnee est trop grande. Veuillez reduire la zone de selection."
- Insufficient credits: "Credits insuffisants pour cette zone. Reduisez la zone ou rechargez vos credits."

## Cancellation and Error Handling

- **User cancels mid-tiling**: all results discarded, nothing saved
- **Single tile failure**: automatic retry
- **Retry exhausted**: surface error to user

## Out of Scope (future iterations)

- Advanced validation step between detection and export
- Smart merge of partial polygons at tile boundaries
- Visual grid overlay showing tile boundaries
- Variable tile sizes based on object density
