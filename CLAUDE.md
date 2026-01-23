# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**QGIS AI-Segmentation**: A QGIS plugin for AI-powered geospatial image segmentation using Meta's Segment Anything Model (SAM).

- **GitHub name**: QGIS_AI-Segmentation
- **Plugin name in QGIS**: AI Segmentation

### Key Features
- Interactive point-based segmentation (click to segment)
- **SAM ViT-B model**: Meta's Segment Anything Model
- **Automatic setup**: Virtual environment with auto-installed dependencies
- **Hardware support**: CPU, CUDA (NVIDIA), MPS (Apple Silicon)
- **Subprocess isolation**: PyTorch runs in isolated processes (no QGIS crashes)
- Cross-platform: Windows, macOS, Linux
- Exports to vector layers

## Architecture

```
QGIS_AI-Segmentation/
├── __init__.py                    # Plugin entry point
├── metadata.txt                   # Plugin metadata
├── ai_segmentation_plugin.py      # Main coordinator
├── ai_segmentation_dockwidget.py  # UI panel
├── ai_segmentation_maptool.py     # Map click handler
├── core/
│   ├── python_manager.py          # Python standalone download/management
│   ├── venv_manager.py            # Virtual environment management
│   ├── checkpoint_manager.py      # SAM model download
│   ├── sam_predictor.py           # Prediction subprocess interface
│   ├── feature_encoder.py         # Encoding subprocess interface
│   ├── feature_dataset.py         # Feature tile loading
│   ├── device_manager.py          # CPU/CUDA/MPS detection
│   ├── geo_utils.py               # Coordinate transformations
│   └── polygon_exporter.py        # Mask → vector conversion
├── workers/
│   ├── encoding_worker.py         # Subprocess: raster encoding
│   └── prediction_worker.py       # Subprocess: SAM prediction
├── python_standalone/             # Auto-downloaded Python interpreter
├── venv_py3.XX/                   # Auto-created virtual environment
└── resources/icons/
```

### Hybrid Subprocess Architecture

The plugin uses a **hybrid architecture** to ensure stability across all QGIS versions:

```
┌─────────────────────────────────────────────────────────────┐
│                    QGIS MAIN PROCESS                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Plugin UI, feature_dataset, device_manager          │   │
│  │ → Packages accessed via sys.path injection          │   │
│  │ → Lightweight operations only                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
              │                        │
              ▼                        ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│   ENCODING SUBPROCESS   │  │  PREDICTION SUBPROCESS  │
│   workers/encoding_     │  │  workers/prediction_    │
│   worker.py             │  │  worker.py              │
│   ─────────────────     │  │  ─────────────────      │
│   • venv Python 3.12    │  │  • venv Python 3.12     │
│   • torch, SAM model    │  │  • torch, SAM model     │
│   • Full isolation      │  │  • Full isolation       │
└─────────────────────────┘  └─────────────────────────┘
```

**Why subprocess isolation?**
- PyTorch can crash QGIS if loaded directly (memory conflicts, threading issues)
- QGIS's embedded Python has limited/broken subprocess capabilities on some platforms
- Subprocess architecture isolates heavy ML workloads completely

### Python Standalone (Automatic Download)

The plugin automatically downloads a standalone Python interpreter that matches the QGIS Python version. This ensures 100% compatibility across all platforms.

| Component | Source | Size |
|-----------|--------|------|
| **Python Standalone** | [astral-sh/python-build-standalone](https://github.com/astral-sh/python-build-standalone) | ~50 MB |

**How it works:**
1. Plugin detects QGIS Python version (e.g., 3.12)
2. Downloads matching Python standalone for the platform
3. Creates venv using the standalone Python
4. All packages are guaranteed compatible with QGIS

**Why not use system Python?**
- QGIS's embedded Python cannot be used for subprocesses on macOS (vcpkg build limitation)
- System Python versions vary and may not match QGIS
- Standalone approach ensures consistent behavior across all machines

## Dependency Management

### Virtual Environment Structure

```
venv_py3.12/
├── bin/                    # macOS/Linux executables
│   ├── python3
│   └── pip
├── Scripts/                # Windows executables
│   ├── python.exe
│   └── pip.exe
└── lib/python3.12/site-packages/
    ├── torch/              # ~2GB
    ├── torchvision/
    ├── segment_anything/
    ├── pandas/
    ├── rasterio/
    └── numpy/
```

### Package Access Strategy

```python
# In core modules that need venv packages:
from .venv_manager import ensure_venv_packages_available
ensure_venv_packages_available()  # Adds venv site-packages to sys.path

import pandas as pd  # Now accessible
import torch         # Now accessible
```

### Required Packages

| Package | Version | Size | Purpose |
|---------|---------|------|---------|
| torch | ≥2.0.0 | ~2GB | ML framework |
| torchvision | ≥0.15.0 | ~50MB | Image transforms |
| segment-anything | ≥1.0 | ~5MB | SAM model |
| pandas | ≥1.3.0 | ~50MB | Feature indexing |
| rasterio | ≥1.3.0 | ~20MB | Raster I/O |
| numpy | <2.0,≥1.20 | ~30MB | Array operations |

## User Experience Flow

```
1. Install plugin → Enable in QGIS
2. Open panel → "Install Dependencies" button shown
   → Downloads Python standalone (~50MB, ~30 sec)
   → Creates virtual environment (~5 sec)
   → Installs packages (~2.5GB, ~5-10 min)
   → Progress bar shows status
3. "Download SAM Model" button shown
   → Downloads checkpoint (~375MB)
4. Select raster layer
5. Click "Start AI Segmentation"
   → First time: Encoding subprocess runs (1-5 min)
   → Cached: Instant start
   → Cursor changes to crosshair
6. Left-click = include area, Right-click = exclude area
7. Mask preview appears instantly
8. Press S or "Save Polygon" to save
9. "Export to Layer" creates vector layer
```

### Controls
- **Left-click**: Foreground point (include)
- **Right-click**: Background point (exclude)
- **Ctrl+Z**: Undo last point
- **S key**: Save polygon
- **Export to Layer**: Create vector layer

## Development

### Plugin Locations

- **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
- **Windows**: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
- **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`

### Key Files

| File | Purpose |
|------|---------|
| `python_manager.py` | Downloads/manages Python standalone interpreter |
| `venv_manager.py` | Venv creation, package installation |
| `feature_encoder.py` | Spawns encoding subprocess, handles progress |
| `sam_predictor.py` | Spawns prediction subprocess, JSON communication |
| `feature_dataset.py` | Loads cached features, spatial indexing |
| `device_manager.py` | Detects CUDA/MPS/CPU |

### Debugging

```python
# Check Python standalone status
from .core.python_manager import standalone_python_exists, get_standalone_python_path
print(f"Standalone exists: {standalone_python_exists()}")
print(f"Standalone path: {get_standalone_python_path()}")

# Check venv status
from .core.venv_manager import venv_exists, get_venv_dir, verify_venv
print(f"Venv exists: {venv_exists()}")
print(f"Venv path: {get_venv_dir()}")
print(verify_venv())
```

## Code Style

- Follow PEP 8
- Use type hints
- Minimal comments (code should be self-documenting)
- Log via `QgsMessageLog.logMessage(..., "AI Segmentation", level=Qgis.Info)`
- **Relative imports**: Always `.core.module` not `core.module`
- Reference: `pyqgis_developer_cookbook/` for QGIS patterns

## TODO / Roadmap

- [x] Subprocess isolation for PyTorch
- [x] Virtual environment management
- [x] Python standalone auto-download (no system Python required)
- [x] Mask preview with RubberBand
- [x] Ctrl+Z undo support
- [x] Hardware detection (CPU/CUDA/MPS)
- [ ] Add bounding box prompt support
- [ ] Multi-model support (SAM2, MobileSAM)
- [ ] Processing algorithm for batch mode
- [ ] ONNX runtime option
