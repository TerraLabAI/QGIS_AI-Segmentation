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

This is hidden from users - they just click "Start Segmentation" and it works.

## User Experience Flow

```
1. Install plugin
2. Enable in QGIS
3. Open plugin panel
   → "Models not downloaded" → Click "Download Models"
   → Progress bar shows download (~109MB)
   → "Models ready ✓"
4. Select raster layer
5. Click "Start Segmentation"
   → If first time: "Preparing layer..." (auto-encode, 1-5 min)
   → If cached: instant start
6. Click on map → instant segmentation
7. Save masks, export to GeoPackage
```

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

- **Language**: All code and comments in English
- **QGIS Compatibility**: Target QGIS 3.28 LTR minimum
- **No GPU Required**: Use `onnxruntime` CPU provider, not `onnxruntime-gpu`
- **Coordinate Systems**: Preserve source raster CRS throughout
- **Memory**: Handle large rasters by limiting read size (see `max_size` in `image_utils.py`)
- **Caching**: Features are cached per-image in `.ai_segmentation_cache/` folders
- **Plug-and-play**: Users should never need to manually download or configure models

## Code Style

- Follow PEP 8
- Use type hints
- Docstrings for all public functions
- Log messages via `QgsMessageLog.logMessage(..., "AI Segmentation", level=Qgis.Info)`

## TODO / Roadmap

- [ ] Add mask preview overlay on map canvas
- [ ] Add bounding box prompt support
- [ ] Add GPU support (optional)
- [ ] Create Processing algorithm for batch mode
- [ ] Add SAM2 model support
- [ ] Add MobileSAM option (smaller/faster)
