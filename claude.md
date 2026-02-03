# QGIS AI Segmentation Plugin

A QGIS plugin for AI-powered image segmentation using Meta's Segment Anything Model (SAM).

## Architecture Overview

### Core Components

- **`src/ui/ai_segmentation_plugin.py`**: Main plugin class, handles QGIS integration, tool management, and segmentation workflow coordination
- **`src/ui/ai_segmentation_dockwidget.py`**: Qt dock widget with all UI elements (dependency management, model download, segmentation controls, mode switching)
- **`src/ui/ai_segmentation_maptool.py`**: Custom QgsMapTool for handling map clicks (positive/negative points) and keyboard shortcuts

### Segmentation Modes

The plugin has two modes:

1. **Simple Mode (default)**:
   - Segment one object at a time
   - Each export creates a new layer named `{RasterName}_mask_{counter}`
   - No "Save mask" button - only "Export to layer"
   - After export, returns to initial state
   - Yellow info box explains: "One element per export (e.g., one building, one car). For multiple elements in one layer, use Batch Mode."

2. **Batch Mode (advanced)**:
   - Activated via "Batch Mode (advanced)" button at bottom of panel
   - Save multiple masks, then export all together
   - "Save mask" button visible to add current mask to collection
   - Shows polygon count badge
   - Tutorial notification shown on first activation

### Key Classes

- `AISegmentationPlugin`: Main plugin entry point
- `AISegmentationDockWidget`: UI widget with signals for all user actions
- `AISegmentationMapTool`: Map interaction tool with point markers

### Signal Flow

1. User clicks "Start AI Segmentation" -> `start_segmentation_requested` signal
2. User clicks on map -> `positive_click` or `negative_click` signal
3. Plugin runs SAM inference -> mask displayed as QgsRubberBand
4. User clicks "Export to layer" -> `export_layer_requested` signal
5. In Simple mode: creates new layer, resets session
6. In Batch mode: "Save mask" -> `save_polygon_requested`, then "Export to layer" -> exports all saved polygons

### State Variables (DockWidget)

- `_batch_mode`: Boolean, False = Simple mode (default)
- `_segmentation_active`: Whether segmentation session is active
- `_has_mask`: Whether current mask exists
- `_saved_polygon_count`: Number of saved polygons (Batch mode)
- `_refine_expanded`: Collapsed state of refine panel
- `_positive_count`, `_negative_count`: Point counts for UI

### Refine Panel

- **Expand/Contract slider**: Dilate/erode mask boundaries (-30 to +30 pixels)
- **Simplify slider**: Douglas-Peucker simplification tolerance (0-100)
- Collapsible via click on title (no checkbox)
- Settings applied in real-time to preview and export

### Dependencies

- PyTorch >= 2.0
- segment-anything
- rasterio, pandas, numpy
- Installed in isolated venv at `src/venv_py3.12/`

### Model

- SAM ViT-B checkpoint (~375MB)
- Stored in `src/checkpoints/sam_vit_b_01ec64.pth`
- Auto-detects GPU (CUDA/MPS) or falls back to CPU

## Development Notes

- All UI text must be in English
- Buttons hidden when not in segmentation mode (not disabled-but-visible)
- Use `_update_button_visibility()` to manage button states based on mode and session state
- QSettings used for persisting tutorial flags: `AI_Segmentation/tutorial_simple_shown`, `AI_Segmentation/tutorial_batch_shown`


