# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**QGIS AI-Segmentation**: A QGIS plugin for AI-powered geospatial image segmentation using Meta's Segment Anything Model (SAM).

- **GitHub name**: QGIS_AI-Segmentation (public name so users know it's for QGIS)
- **Plugin name in QGIS**: AI Segmentation (displayed inside QGIS)

### Key Features
- Interactive point-based segmentation (click to segment)
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
├── ai_segmentation_dockwidget.py  # PyQt5 dock panel UI
├── ai_segmentation_maptool.py     # QgsMapTool for capturing clicks
├── core/
│   ├── dependency_manager.py      # Auto-installs onnxruntime/numpy
│   ├── sam_encoder.py             # Image → features (heavy, run once)
│   ├── sam_decoder.py             # Features + clicks → mask (light, real-time)
│   ├── image_utils.py             # Raster ↔ numpy conversion
│   └── polygon_exporter.py        # Mask → GeoPackage polygons
├── models/                        # ONNX models (not in repo, ~400MB)
├── resources/icons/               # Plugin icons
├── tests/                         # Test modules
└── ui/                            # UI modules
```

### Two-Stage Architecture

1. **Encoding (Heavy)**: `sam_encoder.py` processes the raster once, saves features to `.ai_segmentation_cache/`
2. **Decoding (Light)**: `sam_decoder.py` runs in ~50-200ms per click using cached features

This is the key to achieving real-time CPU performance.

## Development Commands

### Testing the Plugin in QGIS

The plugin folder should be placed directly in QGIS plugins directory:
- **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation/`
- **Windows**: `C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\QGIS_AI-Segmentation\`
- **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation/`

Then restart QGIS and enable the plugin.

### Getting ONNX Models

```bash
pip install samexporter

# Export SAM ViT-B models to the models/ folder
samexporter export-encoder --model-type vit_b --output models/sam_vit_b_encoder.onnx
samexporter export-decoder --model-type vit_b --output models/sam_vit_b_decoder.onnx
```

### Package Dependencies

The plugin auto-installs these, but for manual testing:
```bash
pip install onnxruntime>=1.15.0 numpy>=1.20.0
```

## Key Considerations

- **Language**: All code and comments in English
- **QGIS Compatibility**: Target QGIS 3.28 LTR minimum
- **No GPU Required**: Use `onnxruntime` CPU provider, not `onnxruntime-gpu`
- **Coordinate Systems**: Preserve source raster CRS throughout
- **Memory**: Handle large rasters by limiting read size (see `max_size` in `image_utils.py`)
- **Caching**: Features are cached per-image in `.ai_segmentation_cache/` folders

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
