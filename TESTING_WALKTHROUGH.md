# QGIS AI Segmentation - Testing Walkthrough

Manual testing checklist. Needs: QGIS 3.28+, internet, a GeoTIFF, ~2GB disk.

---

## 1. Setup (First Launch)

- [ ] Open plugin via toolbar icon or Plugins menu > no errors in log
- [ ] Panel toggles open/close, icon stays in sync
- [ ] Fresh install shows welcome: Step 1 (deps) + Step 2 (model)
- [ ] CUDA checkbox visible on Windows/Linux, hidden on macOS

## 2. Dependency Installation

- [ ] Click "Install Dependencies" > progress bar with package names + ETA
- [ ] Completes: status "Virtual environment ready", device info shown
- [ ] Cancel mid-install > stops, can retry
- [ ] Reopen QGIS > deps detected instantly (no re-download)
- [ ] **CUDA** (Windows/Linux): check CUDA box > installs GPU torch, falls back to CPU if fails
- [ ] **Proxy**: configure QGIS proxy > pip uses it automatically

## 3. Model Download

- [ ] Click "Download Model (~375MB)" > progress in MB + ETA
- [ ] Completes: "SAM model ready"
- [ ] Cancel > partial file cleaned up, can retry
- [ ] Reopen QGIS > model detected instantly (hash verified)

## 4. Activation

- [ ] "Get my verification code" opens browser
- [ ] Valid code > "Unlocked!", segmentation section appears
- [ ] Invalid code > "Invalid code", can retry
- [ ] Activation persists across QGIS restarts

## 5. Layer Selection

- [ ] GeoTIFF appears in dropdown, "Start" button enabled
- [ ] WMS/WMTS/XYZ/PNG-without-georef/vectors are excluded
- [ ] No rasters > warning "No compatible raster found", button disabled
- [ ] Add/remove layers dynamically > dropdown updates live

## 6. Encoding (First Time Per Raster)

- [ ] New raster: progress "Encoding tile X/Y...", then enters segmentation
- [ ] Same raster again: instant (cached), no encoding
- [ ] Cancel encoding > partial cache cleaned, can retry
- [ ] Large raster (>10k px): encodes tile-by-tile, 15min timeout

## 7. Simple Mode (Default)

- [ ] Left-click > green marker + blue mask overlay
- [ ] Multiple left-clicks > mask updates to include all
- [ ] Right-click (after positive) > red X marker, mask excludes area
- [ ] Right-click with no positive > ignored
- [ ] Ctrl+Z > removes last point, mask recomputes
- [ ] Escape > clears all points/mask, stays in segmentation mode
- [ ] Export (Enter) > GeoPackage `{Raster}_mask_1` in group, red outline, attrs: id/score/area
- [ ] Multiple exports > counter increments: `_mask_1`, `_mask_2`, `_mask_3`

## 8. Batch Mode

- [ ] Check "Batch mode" > info box changes, tutorial notification (first time)
- [ ] Save polygon (S) > green overlay, counter badge, markers clear
- [ ] Save 2-3, then Export > all in one GeoPackage, correct count
- [ ] Unsaved current mask included in export
- [ ] Ctrl+Z with no current mask > restores last saved (after confirmation)
  - Points, markers, refine settings all restored
- [ ] Uncheck Batch with saved polygons > dialog: Export / Discard / Cancel
- [ ] Escape with saved polygons > dialog: "Delete all?"

## 9. Refine Panel

- [ ] Click "Refine selection" header > toggles expand/collapse
- [ ] Expand/Contract +10 > mask grows, -10 > shrinks, 0 > original. Real-time.
- [ ] Simplify 0 = max detail, 15 = smooth edges. Real-time.
- [ ] Fill holes: check > holes fill, uncheck > holes return
- [ ] Min region size: 500 > small fragments gone, 0 > all kept
- [ ] Batch undo restores refine settings from save time

## 10. Keyboard Shortcuts

| Key | Action | Verify |
|-----|--------|--------|
| G | Start segmentation | With panel open + raster selected |
| Left click | Positive point | Green marker + blue mask |
| Right click | Negative point | Red X, mask refines |
| Ctrl+Z | Undo | Last point removed |
| S | Save polygon | Batch mode only, bare S only |
| Enter | Export | Creates layer |
| Escape | Clear / exit | Clears mask or prompts exit |

- [ ] Ctrl+S triggers QGIS Save Project, NOT polygon save
- [ ] Scroll/zoom/pan work during segmentation, markers stay positioned

## 11. Edge Cases & Stability

- [ ] Delete raster during segmentation > silent reset, no crash
- [ ] Switch layer dropdown during segmentation > session resets
- [ ] Close QGIS during segmentation > no hang, no dialog, clean exit
- [ ] Click Pan tool during segmentation > deferred dialog, can return or exit
- [ ] Stop button > confirmation if unsaved work, full reset if confirmed
- [ ] Export with no mask > button disabled, Enter does nothing

## 12. Cross-Platform

**Windows:**
- [ ] No cmd.exe flash on subprocess, paths with spaces work
- [ ] Process crash (0xC0000005) > fallback to `python -m pip`

**macOS:**
- [ ] MPS detected on Apple Silicon, trackpad gestures work
- [ ] CUDA checkbox hidden

**Linux:**
- [ ] CPU fallback if no GPU, GDAL check for rasterio

## 13. GPU & Performance

- [ ] GPU >=2GB > used, <2GB > CPU fallback (logged)
- [ ] CUDA kernel test failure > CPU fallback
- [ ] GPU OOM during encoding > "falling back to CPU...", completes
- [ ] Encoding timeout: 15min. Prediction timeouts: init 120s, predict 120s

## 14. Export Verification

- [ ] Output: .gpkg in project dir (or raster dir, or home)
- [ ] CRS matches source raster
- [ ] Attributes: id (int), score (double), area (double)
- [ ] Styling: red outline 0.5px, transparent fill
- [ ] Layer group: "{RasterName} (AI Segmentation)"
- [ ] Different CRS rasters: polygons align correctly

## 15. Translations

- [ ] French (fr): UI translated, technical terms in English
- [ ] Portuguese (pt_BR): UI translated
- [ ] Spanish (es): UI translated
- [ ] Unsupported language: falls back to English, no errors

## 16. Plugin Unload/Reload

- [ ] Disable plugin during segmentation > no crash/hang, all UI removed
- [ ] Re-enable > deps detected, works normally
- [ ] No orphan subprocesses after unload
