# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**QGIS AI-Segmentation**: A QGIS plugin for AI-powered geospatial image segmentation using Meta's Segment Anything Model (SAM) and SAM2.

- **GitHub name**: QGIS_AI-Segmentation (public name so users know it's for QGIS)
- **Plugin name in QGIS**: AI Segmentation (displayed inside QGIS)

### Key Features
- Interactive point-based segmentation (click to segment)
- **Multiple AI models**: Fast (SAM ViT-B), Balanced (SAM2 Base+), Precise (SAM2 Large)
- **Automatic model download** from HuggingFace (no manual setup!)
- CPU-optimized (no GPU required)
- Pre-encoding architecture for real-time performance
- Cross-platform: Windows, macOS, Linux
- Exports to GeoPackage vector format

## Architecture

```
QGIS_AI-Segmentation/              # Plugin root folder (this IS the plugin)
├── __init__.py                    # QGIS plugin entry point
├── metadata.txt                   # Plugin metadata (name, version, etc.)
├── requirements.txt               # Python dependencies
├── ai_segmentation_plugin.py      # Main plugin coordinator class
├── ai_segmentation_dockwidget.py  # PyQt5 dock panel UI (multi-model selector)
├── ai_segmentation_maptool.py     # QgsMapTool for capturing clicks
├── core/
│   ├── model_registry.py          # ★ Model configurations (SAM, SAM2 variants)
│   ├── model_manager.py           # ★ Multi-model download/storage management
│   ├── sam_model.py               # ★ Unified SAM interface (supports model switching)
│   ├── dependency_manager.py      # Auto-installs onnxruntime/numpy
│   ├── sam_encoder.py             # Image → features (model-specific caching)
│   ├── sam_decoder.py             # Features + clicks → mask (config-driven)
│   ├── image_utils.py             # Raster ↔ numpy conversion
│   └── polygon_exporter.py        # Mask → GeoPackage polygons
├── models/                        # ONNX models (per-model subdirectories)
│   ├── sam_vit_b/                 # Fast model (~109MB)
│   │   ├── encoder.onnx
│   │   └── decoder.onnx
│   ├── sam2_base_plus/            # Balanced model (~294MB)
│   │   ├── encoder.onnx
│   │   └── decoder.onnx
│   └── sam2_large/                # Precise model (~910MB)
│       ├── encoder.onnx
│       └── decoder.onnx
├── resources/icons/               # Plugin icons
├── tests/                         # Test modules
└── ui/                            # UI modules
```

### Multi-Model Architecture

The plugin supports three AI model tiers:

| Model ID | Display Name | Size | Source | Description |
|----------|-------------|------|--------|-------------|
| `sam_vit_b` | Fast (SAM ViT-B) | ~109MB | visheratin/segment-anything-vit-b | Original SAM, quantized |
| `sam2_base_plus` | Balanced (SAM2 Base+) | ~294MB | shubham0204/sam2-onnx-models | Better quality |
| `sam2_large` | Precise (SAM2 Large) | ~910MB | shubham0204/sam2-onnx-models | Highest quality |

### Unified SAM Model Interface

```python
from .core.sam_model import SAMModel

# Create model with specific variant
model = SAMModel(model_id="sam_vit_b")  # or "sam2_base_plus", "sam2_large"
model.load()                            # Load encoder + decoder
model.prepare_layer(layer)              # Auto-encode (uses model-specific cache)
mask, score = model.segment(points, labels)

# Switch models (clears features, requires re-encoding)
model.switch_model("sam2_large")
```

### Model Registry (`core/model_registry.py`)

Central configuration for all model specifications:
```python
from .core.model_registry import MODEL_REGISTRY, get_model_config, get_all_models

config = get_model_config("sam_vit_b")
print(config.display_name)      # "Fast (SAM ViT-B)"
print(config.total_size_mb)     # 109
print(config.huggingface_repo)  # "visheratin/segment-anything-vit-b"
```

### Model Manager (`core/model_manager.py`)

Handles multi-model download and storage:
```python
from .core.model_manager import (
    model_exists,           # Check if specific model is installed
    get_installed_models,   # List all installed model IDs
    download_model,         # Download specific model
    migrate_legacy_models,  # Move old files to new structure
)
```

### Two-Stage Architecture (Internal)

1. **Encoding (Heavy)**: `sam_encoder.py` processes the raster once, saves features to model-specific cache
2. **Decoding (Light)**: `sam_decoder.py` runs in ~50-200ms per click using cached features

The decoder uses config-driven tensor name mapping to support different model architectures (SAM vs SAM2).

## Map Tool Mechanism (CRITICAL)

### How Point Clicking Works in QGIS

QGIS uses a **Map Tool** system to handle mouse interactions on the map canvas:
- By default, the "Pan" tool (hand icon) is active - it only pans the map
- To place points, you must **activate a custom QgsMapTool**
- When active, the cursor changes (e.g., to a crosshair) and clicks are captured

### Our Implementation

```
ai_segmentation_maptool.py    → AISegmentationMapTool (inherits QgsMapTool)
ai_segmentation_plugin.py     → Main coordinator
ai_segmentation_dockwidget.py → UI with model selector and "Start" button
```

### Activation Flow

```
1. User selects AI Model from dropdown (must be installed)
   ↓
2. User clicks "Start AI Segmentation" button
   ↓
3. dock_widget emits start_segmentation_requested signal
   ↓
4. plugin._on_start_segmentation() is called
   ↓
5. If layer needs preparation → runs PreparationWorker (encoding)
   ↓
6. plugin._activate_segmentation_tool() is called:
   self.iface.mapCanvas().setMapTool(self.map_tool)  ← THIS IS KEY
   ↓
7. Cursor changes to crosshair, clicks are now captured
   ↓
8. AISegmentationMapTool.canvasPressEvent() handles clicks
   ↓
9. Emits positive_click or negative_click signal
   ↓
10. Plugin calls SAM model to generate mask
```

### Key Code Locations

- **Model registry**: `core/model_registry.py` - All model configurations
- **Tool creation**: `ai_segmentation_plugin.py:229` - `AISegmentationMapTool(self.iface.mapCanvas())`
- **Tool activation**: `ai_segmentation_plugin.py:593` - `self.iface.mapCanvas().setMapTool(self.map_tool)`
- **Model switching**: `ai_segmentation_plugin.py:516` - `_on_model_changed()`
- **Click handling**: `ai_segmentation_maptool.py` - `canvasPressEvent()`

### Important: Why Button Must Be Enabled

The "Start AI Segmentation" button is only enabled when:
1. Dependencies are installed (`_dependencies_ok = True`)
2. At least one model is installed (`_models_ok = True`)
3. A model is selected in the dropdown (`_current_model_id is not None`)
4. A raster layer is selected (`layer_combo.currentLayer() is not None`)

See `ai_segmentation_dockwidget.py:_update_ui_state()` for the logic.

## User Experience Flow

```
1. Install plugin
2. Enable in QGIS
3. Open plugin panel
   → Collapsible "Install AI Models" section
   → Click "Install" next to desired model (Fast recommended for first use)
   → Progress bar shows download
   → Section collapses when first model installed
4. Select AI Model from dropdown (grayed out = not installed)
5. Select raster layer from dropdown
6. Click "Start AI Segmentation" button
   → If first time with this model: "Preparing layer..." (encoding, 1-5 min)
   → If cached for this model: instant start
   → Cursor changes to CROSSHAIR ← User can now click on map!
7. Left-click on map → foreground point (green marker)
8. Right-click on map → background point (red marker)
9. Mask preview (RubberBand) appears instantly after each click
10. Use "Undo Last Point" button to remove points if needed
11. Click "Finish Segmentation" to save as a new layer
    → Layer created with name: {raster_name}_segmentation_{counter}
    → Panel resets for new session
```

### Controls (when Segmentation Mode is active)
- **Left-click**: Add foreground point (include this area)
- **Right-click**: Add background point (exclude this area)
- **Ctrl+Z**: Undo last point
- **Undo Last Point button**: Remove the last point
- **Finish Segmentation button**: Save mask as layer and reset
- **Model dropdown**: Disabled during active segmentation (finish first)

## Development Commands

### Testing the Plugin in QGIS

The plugin folder should be placed directly in QGIS plugins directory:
- **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation/`
- **Windows**: `C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\QGIS_AI-Segmentation\`
- **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation/`

Then restart QGIS and enable the plugin.

### Package Dependencies

The plugin auto-installs these, but for manual testing:
```bash
pip install onnxruntime>=1.15.0 numpy>=1.20.0
```

### Testing Model Download

```python
from .core.model_manager import download_model, model_exists, get_installed_models

# Check what's installed
print(get_installed_models())  # ['sam_vit_b', 'sam2_base_plus']

# Download a specific model
if not model_exists("sam2_large"):
    success, msg = download_model("sam2_large", lambda p, m: print(f"{p}%: {m}"))
```

## Key Considerations

- **Raster layer support**: The plugin supports both:
  - **File-based rasters** (GeoTIFF, JPEG, PNG, etc.) - uses native resolution
  - **Web layers** (XYZ, WMS, Google Satellite) - renders at max_size resolution
  - Note: For web layers, the current map extent is used and rendered at a fixed resolution
- **Language**: All code and comments in English
- **QGIS Compatibility**: Target QGIS 3.28 LTR minimum
- **No GPU Required**: Use `onnxruntime` CPU provider, not `onnxruntime-gpu`
- **Coordinate Systems**: Preserve source raster CRS throughout
- **Memory**: Handle large rasters by limiting read size (see `max_size` in `image_utils.py`)
- **Caching**: Features are cached per-image AND per-model:
  - **File-based rasters**: `.ai_segmentation_cache/{model_id}/` folder next to the file
  - **Web-based layers**: `~/.qgis_ai_segmentation_cache/<hash>/{model_id}/` in user home
  - Switching models requires re-encoding (different feature spaces)
- **Plug-and-play**: Users should never need to manually download or configure models
- **Model migration**: Existing users' models are automatically migrated to the new per-model directory structure

## Code Style

- Follow PEP 8
- Use type hints
- Docstrings for all public functions
- Log messages via `QgsMessageLog.logMessage(..., "AI Segmentation", level=Qgis.Info)`
- **Relative imports**: Always use `.core.module` not `core.module` within the plugin package

## TODO / Roadmap

- [x] Add mask preview overlay on map canvas (RubberBand only, no preview layer)
- [x] LIFO undo with prompt_history tracking
- [x] Ctrl+Z keyboard shortcut for undo (active during segmentation only)
- [x] Cancel buttons for download/preparation
- [x] Layer change detection (auto-cancel segmentation)
- [x] Simplified session flow: Start → Click → Finish (auto-creates layer)
- [x] Clean layer naming: {raster_name}_segmentation_{counter}
- [x] Multi-model support (SAM ViT-B, SAM2 Base+, SAM2 Large)
- [x] Model selector dropdown with install status
- [x] Collapsible model install section
- [x] Per-model caching (separate feature cache per model)
- [x] Legacy model migration
- [ ] Add bounding box prompt support
- [ ] Add GPU support (optional)
- [ ] Create Processing algorithm for batch mode
- [ ] Add MobileSAM option (smaller/faster)
