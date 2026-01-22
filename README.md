# QGIS AI-Segmentation

AI-powered image segmentation for QGIS using Meta's Segment Anything Model (SAM).

![AI Segmentation](resources/icons/ai_segmentation_icon.png)

## Quick Start

1. **Install the plugin** - Copy to your QGIS plugins folder
2. **Enable the plugin** - In QGIS: `Plugins > Manage and Install Plugins`
3. **Open AI Segmentation** - Click the toolbar icon or `Raster > AI Segmentation`
4. **Download models** - Click "Download Models" (one-time, ~375 MB)
5. **Select a raster layer** - Choose your satellite image or orthophoto
6. **Start segmenting** - Click "Start Segmentation", then click on objects!

That's it! No manual model setup required.

## Features

- **One-Click Setup**: SAM model downloads automatically on first run
- **Interactive Segmentation**: Left-click to include, right-click to exclude
- **CPU-Optimized**: Works on any computer without GPU
- **Geospatial-Aware**: Preserves CRS and exports georeferenced polygons
- **Cross-Platform**: Windows, macOS, and Linux
- **Caching**: Feature embeddings are cached for instant subsequent use

---

## Supported Formats

### Input Raster Formats

| Format | Status | Notes |
|--------|--------|-------|
| **GeoTIFF (.tif, .tiff)** | ✅ Fully Supported | Best format - includes CRS and georeferencing |
| **TIFF (.tif, .tiff)** | ✅ Supported | Works well if CRS is set in QGIS |
| **JPEG (.jpg, .jpeg)** | ✅ Supported* | Uses QGIS layer extent for georeferencing |
| **PNG (.png)** | ✅ Supported* | Uses QGIS layer extent for georeferencing |
| **JPEG2000 (.jp2)** | ✅ Supported* | Uses QGIS layer extent if bounds look like pixel coords |
| **ECW (.ecw)** | ✅ Supported | Enterprise format with georeferencing |
| **MrSID (.sid)** | ✅ Supported | If GDAL support is available |

**\* Important for PNG/JPEG/JP2 users:**
- These formats may not have embedded georeferencing
- The plugin automatically uses the QGIS layer extent when needed
- For best accuracy, use an accompanying world file (`.pgw` / `.jgw` / `.j2w`) 
- Or georeference manually in QGIS before starting segmentation

### Output Export Formats

| Format | Description |
|--------|-------------|
| **Memory Layer** | Created directly in QGIS as a temporary vector layer |
| **MultiPolygon Geometry** | Exported geometries are MultiPolygon type |

**Attributes exported:**
- `id` - Polygon sequential ID
- `score` - SAM confidence score (0-1)
- `area` - Polygon area in layer units

---

## Installation

### From Source

1. Download or clone this repository
2. Copy the `QGIS_AI-Segmentation` folder to your QGIS plugins directory:
   - **Windows**: `C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
   - **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
   - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
3. Restart QGIS
4. Enable the plugin in `Plugins > Manage and Install Plugins`

### Requirements

- QGIS 3.28 LTR or later
- Python 3.9+
- Internet connection (for first-time model download)

**Python dependencies** (installed manually, see plugin panel for instructions):
- `torch`
- `torchvision` 
- `segment-anything`
- `rasterio`
- `pandas`

---

## Usage

1. **Load a raster layer** (satellite image, orthophoto, drone imagery, etc.)
2. **Open AI Segmentation** from the toolbar
3. **Select your raster layer** from the dropdown
4. **Click "Start Segmentation"**
   - First time: the image will be encoded (1-5 minutes depending on size)
   - Subsequent uses: instant start (features are cached)
5. **Click on objects** to segment them:
   - **Left-click**: Include this area in the mask (positive prompt)
   - **Right-click**: Exclude this area from the mask (negative prompt)
   - **Ctrl+Z / Undo button**: Remove last point
6. **Save Polygon (S key)** to save current mask and start a new one
7. **Export to Layer** when done - creates a new vector layer with all saved polygons

---

## How It Works

The plugin uses SAM (Segment Anything Model) with a two-stage architecture optimized for CPU:

1. **Preparation** (one-time per image): The image is encoded into feature embeddings using SAM's Vision Transformer encoder. Features are cached to disk as GeoTIFF tiles.
2. **Segmentation** (real-time): Your clicks are processed instantly using the cached features and SAM's lightweight mask decoder.

This approach enables interactive segmentation on CPU without requiring a GPU.

### Cache Location

Feature embeddings are stored in:
- **macOS/Linux**: `~/.qgis_ai_segmentation/features/`
- **Windows**: `C:\Users\<username>\.qgis_ai_segmentation\features\`

Each raster gets a unique cache folder based on its file path hash.

---

## Known Limitations

- **PNG/JPEG without georeferencing**: Polygons may appear in wrong location if the raster has no CRS or world file
- **Web layers (WMS, XYZ, etc.)**: Not supported - only file-based rasters work
- **Very large images**: May require significant encoding time and disk space for cache
- **No GPU acceleration**: Currently CPU-only (GPU support planned)

---

## Roadmap / TODO

### High Priority
- [ ] **Export to GeoPackage/Shapefile**: Add direct export to common vector formats
- [ ] **GPU acceleration**: Optional CUDA/MPS support for faster encoding
- [ ] **Improved PNG/JPEG support**: Better handling of non-georeferenced rasters
- [ ] **Cancel encoding gracefully**: ✅ Implemented - cleans up partial cache

### Medium Priority
- [ ] **Box prompts**: Draw rectangles to guide segmentation
- [ ] **Automatic batch segmentation**: Process entire image automatically
- [ ] **Undo/Redo for saved polygons**: Currently only works for points
- [ ] **Style presets**: Save/load custom visualization styles
- [ ] **Progress for large images**: Show estimated time remaining

### Low Priority / Future
- [ ] **Multi-class labels**: Assign categories to polygons
- [ ] **Model selection**: Support for SAM ViT-L and ViT-H models
- [ ] **Cloud processing**: Offload encoding to cloud service
- [ ] **Training/Fine-tuning**: Custom SAM models for specific domains
- [ ] **ONNX Runtime backend**: Alternative to PyTorch for lighter deployment

---

## Troubleshooting

### Models won't download
- Check your internet connection
- Try again - the download may have timed out
- Check QGIS logs for detailed error messages

### Segmentation is slow
- The first segmentation on an image requires encoding (1-5 minutes)
- Subsequent segmentations use cached features and are instant
- Very large images may take longer

### Plugin doesn't appear
- Ensure the folder is named exactly `QGIS_AI-Segmentation`
- Check that it's in the correct plugins directory
- Try restarting QGIS

### Polygons don't appear after export (PNG/JPEG)
- Your raster likely lacks georeferencing
- Add a world file (`.pgw` / `.jgw`) or georeference manually in QGIS
- Delete the cache folder for this raster and re-encode

### "Module not found" errors
- Install dependencies manually using pip with QGIS's Python
- See the plugin panel for specific installation instructions

---

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [Meta AI](https://github.com/facebookresearch/segment-anything) - Segment Anything Model
- [Geo-SAM](https://github.com/coolzhao/Geo-SAM) - Architecture inspiration
- [TorchGeo](https://github.com/microsoft/torchgeo) - Geospatial utilities

---

**Developed by [TerraLab](https://terralab.ai)**
