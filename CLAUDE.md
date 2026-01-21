# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**QGIS AI-Segmentation**: A QGIS plugin for AI-powered geospatial image segmentation using Meta's Segment Anything Model (SAM).

- **GitHub name**: QGIS_AI-Segmentation (public name so users know it's for QGIS)
- **Plugin name in QGIS**: AI Segmentation (displayed inside QGIS)

### Key Features
- Interactive point-based segmentation (click to segment)
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
├── ai_segmentation_dockwidget.py  # PyQt5 dock panel UI (simplified)
├── ai_segmentation_maptool.py     # QgsMapTool for capturing clicks
├── core/
│   ├── model_manager.py           # ★ Auto-downloads models from HuggingFace
│   ├── sam_model.py               # ★ Unified SAM interface (hides encoder/decoder)
│   ├── dependency_manager.py      # Auto-installs onnxruntime/numpy
│   ├── sam_encoder.py             # Image → features (internal, used by sam_model)
│   ├── sam_decoder.py             # Features + clicks → mask (internal)
│   ├── image_utils.py             # Raster ↔ numpy conversion
│   └── polygon_exporter.py        # Mask → GeoPackage polygons
├── models/                        # ONNX models (auto-downloaded, ~109MB)
│   ├── encoder-quant.onnx         # Quantized encoder (~100MB)
│   └── decoder-quant.onnx         # Quantized decoder (~9MB)
├── resources/icons/               # Plugin icons
├── tests/                         # Test modules
└── ui/                            # UI modules
```

### Simplified Architecture (v2)

The plugin now uses a **unified SAMModel class** that hides encoder/decoder complexity:

```python
from core.sam_model import SAMModel

model = SAMModel()
model.download_models()  # Auto-download from HuggingFace
model.load()             # Load both encoder and decoder
model.prepare_layer(layer)  # Auto-encode (uses cache if available)
mask, score = model.segment(points, labels)  # Real-time segmentation
```

### Model Management

Models are automatically downloaded from HuggingFace on first run:
- **Source**: `visheratin/segment-anything-vit-b` repository
- **Files**: `encoder-quant.onnx` (100MB) + `decoder-quant.onnx` (9MB)
- **Location**: Stored in plugin's `models/` folder
- **No manual setup required!**

### Two-Stage Architecture (Internal)

1. **Encoding (Heavy)**: `sam_encoder.py` processes the raster once, saves features to `.ai_segmentation_cache/`
2. **Decoding (Light)**: `sam_decoder.py` runs in ~50-200ms per click using cached features

This is hidden from users - they just click "Start Point Mode" and it works.

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
ai_segmentation_dockwidget.py → UI with "Start Point Mode" button
```

### Activation Flow

```
1. User clicks "▶ Start Point Mode" button
   ↓
2. dock_widget emits start_segmentation_requested signal
   ↓
3. plugin._on_start_segmentation() is called
   ↓
4. If layer needs preparation → runs PreparationWorker (encoding)
   ↓
5. plugin._activate_segmentation_tool() is called:
   self.iface.mapCanvas().setMapTool(self.map_tool)  ← THIS IS KEY
   ↓
6. Cursor changes to crosshair, clicks are now captured
   ↓
7. AISegmentationMapTool.canvasPressEvent() handles clicks
   ↓
8. Emits positive_click or negative_click signal
   ↓
9. Plugin calls SAM model to generate mask
```

### Key Code Locations

- **Tool creation**: `ai_segmentation_plugin.py:186` - `AISegmentationMapTool(self.iface.mapCanvas())`
- **Tool activation**: `ai_segmentation_plugin.py:499` - `self.iface.mapCanvas().setMapTool(self.map_tool)`
- **Click handling**: `ai_segmentation_maptool.py:120-139` - `canvasPressEvent()`
- **Signal connection**: `ai_segmentation_plugin.py:187-188` - connects click signals to handlers

### Important: Why Button Must Be Enabled

The "Start Point Mode" button is only enabled when:
1. Dependencies are installed (`_dependencies_ok = True`)
2. Models are downloaded and loaded (`_models_ok = True`)
3. A raster layer is selected (`layer_combo.currentLayer() is not None`)

See `ai_segmentation_dockwidget.py:_update_ui_state()` for the logic.

## User Experience Flow

```
1. Install plugin
2. Enable in QGIS
3. Open plugin panel
   → "Models not downloaded" → Click "Download Models"
   → Progress bar shows download (~109MB)
   → "Models ready ✓"
4. Select raster layer from dropdown
5. Click "Start AI Segmentation" button
   → If first time: "Preparing layer..." (auto-encode, 1-5 min)
   → If cached: instant start
   → Cursor changes to CROSSHAIR ← User can now click on map!
6. Left-click on map → foreground point (green marker)
7. Right-click on map → background point (red marker)
8. Mask preview (RubberBand) appears instantly after each click
9. Use "Undo Last Point" button to remove points if needed
10. Click "Finish Segmentation" to save as a new layer
    → Layer created with name: {raster_name}_segmentation_{counter}
    → Panel resets for new session
```

### Controls (when Segmentation Mode is active)
- **Left-click**: Add foreground point (include this area)
- **Right-click**: Add background point (exclude this area)
- **Ctrl+Z**: Undo last point
- **Undo Last Point button**: Remove the last point
- **Finish Segmentation button**: Save mask as layer and reset

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
from core.model_manager import download_models, models_exist

if not models_exist():
    success, msg = download_models(lambda p, m: print(f"{p}%: {m}"))
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
- **Caching**: Features are cached per-image:
  - **File-based rasters** (GeoTIFF, etc.): `.ai_segmentation_cache/` folder next to the file
  - **Web-based layers** (XYZ, WMS, Google Satellite): `~/.qgis_ai_segmentation_cache/<hash>/` in user home
- **Plug-and-play**: Users should never need to manually download or configure models

## Code Style

- Follow PEP 8
- Use type hints
- Docstrings for all public functions
- Log messages via `QgsMessageLog.logMessage(..., "AI Segmentation", level=Qgis.Info)`

## TODO / Roadmap

- [x] Add mask preview overlay on map canvas (RubberBand only, no preview layer)
- [x] LIFO undo with prompt_history tracking
- [x] Ctrl+Z keyboard shortcut for undo (active during segmentation only)
- [x] Cancel buttons for download/preparation
- [x] Layer change detection (auto-cancel segmentation)
- [x] Simplified session flow: Start → Click → Finish (auto-creates layer)
- [x] Clean layer naming: {raster_name}_segmentation_{counter}
- [ ] Add bounding box prompt support
- [ ] Add GPU support (optional)
- [ ] Create Processing algorithm for batch mode
- [ ] Add SAM2 model support
- [ ] Add MobileSAM option (smaller/faster)
