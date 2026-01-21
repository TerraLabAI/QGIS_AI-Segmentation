# QGIS AI-Segmentation

AI-powered image segmentation for QGIS using Meta's Segment Anything Model (SAM).

![AI Segmentation](resources/icons/ai_segmentation_icon.png)

## Quick Start

1. **Install the plugin** - Copy to your QGIS plugins folder
2. **Enable the plugin** - In QGIS: `Plugins > Manage and Install Plugins`
3. **Open AI Segmentation** - Click the toolbar icon or `Raster > AI Segmentation`
4. **Download models** - Click "Download Models" (one-time, ~109 MB)
5. **Select a raster layer** - Choose your satellite image or orthophoto
6. **Start segmenting** - Click "Start Segmentation", then click on objects!

That's it! No manual model setup required.

## Features

- **One-Click Setup**: Models download automatically on first run
- **Interactive Segmentation**: Left-click to include, right-click to exclude
- **CPU-Optimized**: Works on any computer without GPU
- **Geospatial-Aware**: Preserves CRS and exports georeferenced polygons
- **Cross-Platform**: Windows, macOS, and Linux

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

Python dependencies (`onnxruntime`, `numpy`) are installed automatically.

## Usage

1. **Load a raster layer** (satellite image, orthophoto, drone imagery, etc.)
2. **Open AI Segmentation** from the toolbar
3. **Select your raster layer** from the dropdown
4. **Click "Start Segmentation"**
   - First time: the image will be processed (1-5 minutes depending on size)
   - Subsequent uses: instant start (features are cached)
5. **Click on objects** to segment them:
   - **Left-click**: Include this area in the mask
   - **Right-click**: Exclude this area from the mask
6. **Save masks** to build up your segmentation layer
7. **Export to GeoPackage** when done

## How It Works

The plugin uses SAM (Segment Anything Model) with a two-stage architecture optimized for CPU:

1. **Preparation** (one-time per image): The image is encoded into feature embeddings and cached to disk
2. **Segmentation** (real-time): Your clicks are processed instantly using the cached features

This approach enables interactive segmentation on CPU without requiring a GPU.

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

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [Meta AI](https://github.com/facebookresearch/segment-anything) - Segment Anything Model
- [Geo-SAM](https://github.com/coolzhao/Geo-SAM) - Architecture inspiration
- [visheratin/segment-anything-vit-b](https://huggingface.co/visheratin/segment-anything-vit-b) - ONNX model hosting
