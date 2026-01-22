# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**QGIS AI-Segmentation**: A QGIS plugin for AI-powered geospatial image segmentation using Meta's Segment Anything Model (SAM) and SAM2.

- **GitHub name**: QGIS_AI-Segmentation (public name so users know it's for QGIS)
- **Plugin name in QGIS**: AI Segmentation (displayed inside QGIS)

### Key Features
- Interactive point-based segmentation (click to segment)
- **SAM ViT-B model**: Meta's Segment Anything Model
- **Automatic setup**: Dependencies and model checkpoint auto-install
- **Hardware support**: CPU, CUDA (NVIDIA), MPS (Apple Silicon)
- Pre-encoding architecture for real-time performance
- Cross-platform: Windows, macOS, Linux
- Exports to vector layers

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
│   ├── dependency_manager.py      # Auto-installs torch/segment-anything/pandas
│   ├── checkpoint_manager.py      # SAM checkpoint download/storage
│   ├── sam_predictor.py           # SAM model inference (PyTorch)
│   ├── feature_encoder.py         # Image → features encoding
│   ├── feature_dataset.py         # Feature tile management
│   ├── device_manager.py          # CPU/CUDA/MPS device detection
│   ├── geo_utils.py               # Coordinate transformations
│   └── polygon_exporter.py        # Mask → GeoPackage polygons
├── resources/icons/               # Plugin icons
├── tests/                         # Test modules
└── ui/                            # UI modules
```

### Current Implementation: PyTorch + segment_anything

The plugin uses Meta's original SAM implementation with PyTorch:

| Component | Technology | Description |
|-----------|------------|-------------|
| ML Framework | PyTorch | Model inference |
| SAM Model | segment-anything | Meta's official SAM ViT-B |
| Checkpoint | sam_vit_b_01ec64.pth | ~375MB, downloaded from Meta |

### Two-Stage Architecture

1. **Encoding (Heavy)**: `feature_encoder.py` processes the raster once using SAM's image encoder, saves features as GeoTIFF tiles
2. **Decoding (Light)**: `sam_predictor.py` runs mask prediction in ~50-200ms per click using cached features

```python
from .core.checkpoint_manager import download_checkpoint, checkpoint_exists
from .core.sam_predictor import build_sam_vit_b_no_encoder, SamPredictorNoImgEncoder

# Download checkpoint if needed
if not checkpoint_exists():
    download_checkpoint(progress_callback=lambda p, m: print(f"{p}%: {m}"))

# Load predictor (without image encoder for faster inference)
checkpoint_path = get_checkpoint_path()
sam = build_sam_vit_b_no_encoder(checkpoint=checkpoint_path)
predictor = SamPredictorNoImgEncoder(sam)
```

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
1. User clicks "Start AI Segmentation" button
   ↓
2. dock_widget emits start_segmentation_requested signal
   ↓
3. plugin._on_start_segmentation() is called
   ↓
4. If layer needs preparation → runs EncodingWorker (feature encoding)
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
9. Plugin calls SAM predictor to generate mask
```

### Key Code Locations

- **Dependency manager**: `core/dependency_manager.py` - Auto-install torch/segment-anything
- **Checkpoint manager**: `core/checkpoint_manager.py` - SAM checkpoint download
- **SAM predictor**: `core/sam_predictor.py` - Model inference
- **Feature encoder**: `core/feature_encoder.py` - Raster to features
- **Tool creation**: `ai_segmentation_plugin.py` - `AISegmentationMapTool(self.iface.mapCanvas())`
- **Click handling**: `ai_segmentation_maptool.py` - `canvasPressEvent()`

### Important: Why Button Must Be Enabled

The "Start AI Segmentation" button is only enabled when:
1. Dependencies are installed (`_dependencies_ok = True`)
2. SAM checkpoint is downloaded (`_checkpoint_ok = True`)
3. A raster layer is selected (`layer_combo.currentLayer() is not None`)
4. Layer is file-based (not a web layer)

See `ai_segmentation_dockwidget.py:_update_ui_state()` for the logic.

## User Experience Flow

```
1. Install plugin
2. Enable in QGIS
3. Open plugin panel
   → If dependencies missing: "Install Dependencies" button shown
   → Click to auto-install (PyTorch ~2GB warning shown)
   → Progress bar shows installation
4. If SAM checkpoint missing: "Download SAM Model" button shown
   → Click to download (~375MB)
   → Progress bar shows download
5. Select raster layer from dropdown
6. Click "Start AI Segmentation" button
   → If first time with this raster: "Encoding raster..." (1-5 min)
   → If cached: instant start
   → Cursor changes to CROSSHAIR ← User can now click on map!
7. Left-click on map → foreground point (green marker)
8. Right-click on map → background point (red marker)
9. Mask preview (RubberBand) appears instantly after each click
10. Press S or click "Save Polygon" to save current mask
11. Click "Export to Layer" to create vector layer
    → Layer created with name: {raster_name}_segmentation_{counter}
    → Panel resets for new session
```

### Controls (when Segmentation Mode is active)
- **Left-click**: Add foreground point (include this area)
- **Right-click**: Add background point (exclude this area)
- **Ctrl+Z**: Undo last point
- **S key**: Save current polygon
- **Undo Last Point button**: Remove the last point
- **Save Polygon button**: Save current mask and start new one
- **Export to Layer button**: Export all saved polygons as vector layer

## Development Commands

### Testing the Plugin in QGIS

The plugin folder should be placed directly in QGIS plugins directory:
- **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation/`
- **Windows**: `C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\QGIS_AI-Segmentation\`
- **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/QGIS_AI-Segmentation/`

Then restart QGIS and enable the plugin.

### Package Dependencies

The plugin auto-installs these on first run, but for manual testing:
```bash
pip install torch>=2.0.0 segment-anything>=1.0 rasterio>=1.3.0 pandas>=1.3.0 numpy>=1.20.0
```

**Note**: PyTorch is ~2GB. The plugin warns users before downloading.

**QGIS-provided packages** (usually already installed):
- `numpy` - Array operations
- `rasterio` - Geospatial raster I/O

**Required packages** (auto-installed to plugin directory):
- `torch` - PyTorch ML framework (~2GB)
- `segment-anything` - Meta's SAM model
- `pandas` - Feature index management

### Testing Checkpoint Download

```python
from .core.checkpoint_manager import download_checkpoint, checkpoint_exists, get_checkpoint_path

# Check if SAM checkpoint is downloaded
print(checkpoint_exists())  # True/False

# Download checkpoint (~375MB)
if not checkpoint_exists():
    success, msg = download_checkpoint(lambda p, m: print(f"{p}%: {m}"))

# Get path to checkpoint
print(get_checkpoint_path())  # ~/.qgis_ai_segmentation/checkpoints/sam_vit_b_01ec64.pth
```

## Key Considerations

- MANDATORY : Don't put comments in the code or only minimal one, like really minimal

- **Raster layer support**: The plugin supports both:
  - **File-based rasters** (GeoTIFF, JPEG, PNG, etc.) - uses native resolution
  - **Web layers** (XYZ, WMS, Google Satellite) - renders at max_size resolution
  - Note: For web layers, the current map extent is used and rendered at a fixed resolution
- **Language**: All code and small comments in English
- **QGIS Compatibility**: Target QGIS 3.28 LTR minimum
- **Hardware**: CPU by default, supports CUDA (NVIDIA) and MPS (Apple Silicon) if available
- **Coordinate Systems**: Preserve source raster CRS throughout
- **Memory**: Handle large rasters by limiting read size (see `max_size` in feature_encoder.py)
- **Caching**: Features are cached per-image:
  - **File-based rasters**: `.ai_segmentation_cache/` folder next to the file
  - **Web-based layers**: `~/.qgis_ai_segmentation/features/<hash>/` in user home
- **Plug-and-play**: Dependencies auto-install, checkpoint auto-downloads
- **Large dependencies**: PyTorch is ~2GB - user is warned before auto-install

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
- [x] Simplified session flow: Start → Click → Save → Export
- [x] Clean layer naming: {raster_name}_segmentation_{counter}
- [x] Auto-install dependencies (torch, segment-anything, pandas)
- [x] Auto-download SAM checkpoint
- [x] Hardware detection (CPU/CUDA/MPS)
- [ ] Add bounding box prompt support
- [ ] Multi-model support (SAM2, MobileSAM)
- [ ] Create Processing algorithm for batch mode
- [ ] ONNX runtime option (lighter weight alternative to PyTorch)
