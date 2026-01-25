# QGIS AI-Segmentation - Architecture Documentation

## Overview

This document explains the technical architecture of the QGIS AI-Segmentation plugin, including design decisions, cross-platform compatibility strategies, and the subprocess isolation system.

> **📋 See Also**: [ARCHITECTURE_RECOMMENDATIONS.md](./ARCHITECTURE_RECOMMENDATIONS.md) for recommendations on refactoring to achieve better separation of business logic from QGIS-specific code, enabling multi-platform support (QGIS, ArcGIS, TerraLab3D).

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [The QGIS Python Problem](#the-qgis-python-problem)
3. [Subprocess Isolation Strategy](#subprocess-isolation-strategy)
4. [Virtual Environment Management](#virtual-environment-management)
5. [Cross-Platform Compatibility](#cross-platform-compatibility)
6. [Data Flow](#data-flow)
7. [Component Details](#component-details)

---

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              QGIS APPLICATION                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         PLUGIN (Main Process)                         │  │
│  │                                                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │  │
│  │  │   UI Layer  │  │  Map Tool   │  │   Feature   │  │   Polygon   │ │  │
│  │  │  DockWidget │  │  Maptool    │  │   Dataset   │  │   Exporter  │ │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │  │
│  │         │                │                │                │        │  │
│  │         └────────────────┴────────────────┴────────────────┘        │  │
│  │                                   │                                  │  │
│  │                    ┌──────────────┴──────────────┐                  │  │
│  │                    │     Plugin Coordinator      │                  │  │
│  │                    │  ai_segmentation_plugin.py  │                  │  │
│  │                    └──────────────┬──────────────┘                  │  │
│  └───────────────────────────────────┼──────────────────────────────────┘  │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
         ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
         │ ENCODING PROCESS │ │PREDICTION PROCESS│ │  VENV MANAGER    │
         │  (subprocess)    │ │  (subprocess)    │ │  (in-process)    │
         │                  │ │                  │ │                  │
         │ • Python 3.12    │ │ • Python 3.12    │ │ • Creates venv   │
         │ • PyTorch        │ │ • PyTorch        │ │ • Installs deps  │
         │ • SAM Encoder    │ │ • SAM Decoder    │ │ • Verifies pkgs  │
         │ • Rasterio       │ │ • NumPy          │ │                  │
         └──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## The QGIS Python Problem

### Why Can't We Just Use QGIS's Python?

QGIS embeds a Python interpreter, but it has significant limitations:

#### macOS (QGIS 3.40+ with vcpkg)
```
Problem: QGIS's embedded Python cannot be executed via subprocess.

$ /Applications/QGIS.app/Contents/MacOS/python3.12 -c "print('hello')"
Fatal Python error: init_fs_encoding: failed to get the Python codec
ModuleNotFoundError: No module named 'encodings'

Reason: The vcpkg build hardcodes paths to the build environment:
  sys.prefix = '/Users/runner/work/QGIS/QGIS/build/vcpkg_installed/...'

This path doesn't exist on user machines.
```

#### Windows (OSGeo4W)
```
Problem: sys.executable points to qgis.exe, not python.exe

>>> import sys
>>> sys.executable
'C:\\OSGeo4W\\bin\\qgis-bin.exe'

Solution: Must find Python in OSGeo4W apps directory.
```

#### Linux
```
Generally works with sys.executable, but may have permission issues
or missing packages in system Python.
```

### Our Solution: Python Standalone + Virtual Environment

Instead of fighting QGIS's Python limitations, we:

1. **Download a standalone Python** matching the QGIS version (~50MB)
   - Source: [astral-sh/python-build-standalone](https://github.com/astral-sh/python-build-standalone)
   - Automatically downloaded on first "Install Dependencies" click
   - No system Python required
2. **Create an isolated virtual environment** using the standalone Python
3. **Run heavy workloads in subprocesses** using the venv Python
4. **Inject venv packages into QGIS's sys.path** for lightweight operations

This approach ensures 100% compatibility across all platforms and QGIS versions.

---

## Subprocess Isolation Strategy

### Why Subprocesses?

| Approach | Pros | Cons |
|----------|------|------|
| **In-process PyTorch** | Simple, fast IPC | Can crash QGIS, memory conflicts, threading issues |
| **Subprocess isolation** | Stable, crash-safe | Slower startup, IPC overhead |

We chose subprocess isolation because:
- PyTorch + QGIS in same process = frequent crashes
- Memory leaks in PyTorch don't affect QGIS
- Can restart subprocess without restarting QGIS
- Clean separation of concerns

### Process Communication

```
┌─────────────────┐         JSON/stdin          ┌─────────────────┐
│  QGIS Process   │ ──────────────────────────► │ Worker Process  │
│                 │                              │                 │
│  • Sends config │                              │ • Receives JSON │
│  • Reads stdout │ ◄────────────────────────── │ • Sends progress│
│  • Parses JSON  │      JSON/stdout/stderr      │ • Sends results │
└─────────────────┘                              └─────────────────┘
```

#### Encoding Worker Protocol
```python
# QGIS sends via stdin:
{
    "raster_path": "/path/to/image.tif",
    "output_dir": "/path/to/cache",
    "checkpoint_path": "/path/to/sam_vit_b.pth",
    "layer_crs_wkt": "EPSG:4326",
    "layer_extent": [xmin, ymin, xmax, ymax]
}

# Worker sends via stdout:
{"status": "progress", "percent": 25, "message": "Processing tile 1/4"}
{"status": "progress", "percent": 50, "message": "Processing tile 2/4"}
{"status": "complete", "tiles": 4, "message": "Encoding complete"}
# or
{"status": "error", "message": "Out of memory"}
```

#### Prediction Worker Protocol
```python
# QGIS sends via stdin:
{"command": "load_checkpoint", "path": "/path/to/sam_vit_b.pth"}
{"command": "set_features", "features_path": "/path/to/tile.tif", ...}
{"command": "predict", "points": [[x, y]], "labels": [1]}

# Worker sends via stdout:
{"status": "ready"}
{"status": "features_loaded"}
{"status": "prediction", "mask": "<base64-encoded-numpy-array>", ...}
```

---

## Virtual Environment Management

### Venv Structure

```
QGIS_AI-Segmentation/
└── venv_py3.12/                    # Named by Python version
    ├── bin/                        # macOS/Linux
    │   ├── python3 → python3.12
    │   ├── pip
    │   └── activate
    ├── Scripts/                    # Windows
    │   ├── python.exe
    │   ├── pip.exe
    │   └── activate.bat
    ├── lib/
    │   └── python3.12/
    │       └── site-packages/
    │           ├── torch/
    │           ├── torchvision/
    │           ├── segment_anything/
    │           ├── pandas/
    │           ├── rasterio/
    │           └── numpy/
    └── pyvenv.cfg
```

### Package Installation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Install Dependencies                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Find System Python                                          │
│     macOS:   /opt/homebrew/bin/python3.12                       │
│     Windows: C:\OSGeo4W\apps\Python312\python.exe               │
│     Linux:   /usr/bin/python3                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Create Virtual Environment                                   │
│     subprocess: [python, "-m", "venv", "venv_py3.12"]           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Install Packages via pip                                     │
│     subprocess: [venv/pip, "install", "torch>=2.0.0", ...]      │
│                                                                  │
│     Order matters for compatibility:                             │
│     1. numpy<2.0,>=1.20.0  (pin version for torch compatibility)│
│     2. torch>=2.0.0        (~2GB download)                      │
│     3. torchvision>=0.15.0                                      │
│     4. segment-anything>=1.0                                    │
│     5. pandas>=1.3.0                                            │
│     6. rasterio>=1.3.0                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Verify Installation                                          │
│     subprocess: [venv/python, "-c", "import torch"]             │
│     subprocess: [venv/python, "-c", "import pandas"]            │
│     ...                                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Accessing Packages in QGIS Process

For lightweight operations (loading feature index, device detection), we need packages in the main QGIS process:

```python
# core/venv_manager.py
def ensure_venv_packages_available():
    """Add venv site-packages to sys.path for in-process imports."""
    site_packages = get_venv_site_packages()  # e.g., venv_py3.12/lib/python3.12/site-packages

    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)

    return True

# core/feature_dataset.py
from .venv_manager import ensure_venv_packages_available
ensure_venv_packages_available()  # Must be called BEFORE importing packages

import pandas as pd  # Now works!
import torch         # Now works!
```

---

## Cross-Platform Compatibility

### Python Discovery Algorithm

```python
def _get_system_python() -> str:
    py_major = sys.version_info.major
    py_minor = sys.version_info.minor

    if sys.platform == "darwin":
        # macOS: Homebrew Python required (QGIS Python is broken)
        candidates = [
            f"/opt/homebrew/bin/python{py_major}.{py_minor}",  # Apple Silicon
            f"/usr/local/bin/python{py_major}.{py_minor}",     # Intel Mac
        ]

    elif sys.platform == "win32":
        # Windows: Check OSGeo4W, then standard locations
        osgeo4w = os.environ.get("OSGEO4W_ROOT", r"C:\OSGeo4W")
        candidates = [
            os.path.join(osgeo4w, "apps", f"Python{py_major}{py_minor}", "python.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", ...),
            os.path.join(os.environ.get("ProgramFiles", ""), f"Python{py_major}{py_minor}", ...),
        ]

    else:
        # Linux: System Python usually works
        return sys.executable
```

### Platform-Specific Considerations

| Platform | Python Source | Venv Location | Subprocess Spawning |
|----------|---------------|---------------|---------------------|
| macOS | Homebrew | `venv_py3.12/bin/python3` | Standard subprocess |
| Windows | OSGeo4W/System | `venv_py3.12\Scripts\python.exe` | Hidden window (STARTUPINFO) |
| Linux | System | `venv_py3.12/bin/python3` | Standard subprocess |

### Windows-Specific Handling

```python
if sys.platform == "win32":
    # Hide console window when spawning subprocesses
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE

    result = subprocess.run(cmd, startupinfo=startupinfo, ...)
```

---

## Data Flow

### Complete Segmentation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERACTION                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
           ┌───────────────┐               ┌───────────────┐
           │ First Time    │               │ Cached        │
           │ with Raster   │               │ Features      │
           └───────┬───────┘               └───────┬───────┘
                   │                               │
                   ▼                               │
┌─────────────────────────────────────┐           │
│      ENCODING SUBPROCESS            │           │
│                                     │           │
│  1. Load raster via rasterio        │           │
│  2. Load SAM encoder                │           │
│  3. Split into 1024x1024 tiles      │           │
│  4. Encode each tile → features     │           │
│  5. Save as GeoTIFF + CSV index     │           │
│                                     │           │
│  Output: .ai_segmentation_cache/    │           │
│          ├── features_vit_b_0.tif   │           │
│          ├── features_vit_b_1.tif   │           │
│          └── layer_name.csv         │           │
└─────────────────┬───────────────────┘           │
                  │                               │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  LOAD FEATURES (QGIS)   │
                    │                         │
                    │  FeatureDataset loads:  │
                    │  • CSV index → pandas   │
                    │  • Spatial index        │
                    │  • CRS, resolution      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  USER CLICKS ON MAP     │
                    │                         │
                    │  MapTool captures:      │
                    │  • Point coordinates    │
                    │  • Left/Right click     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  PREDICTION SUBPROCESS  │
                    │                         │
                    │  1. Find nearest tile   │
                    │  2. Load tile features  │
                    │  3. Transform coords    │
                    │  4. Run SAM decoder     │
                    │  5. Return mask         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  DISPLAY MASK (QGIS)    │
                    │                         │
                    │  • RubberBand overlay   │
                    │  • Semi-transparent     │
                    │  • Updates on each click│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  EXPORT (QGIS)          │
                    │                         │
                    │  • Mask → polygons      │
                    │  • Create vector layer  │
                    │  • Add to project       │
                    └─────────────────────────┘
```

### Feature Cache Structure

```
image.tif
└── .ai_segmentation_cache/
    ├── features_vit_b_0.tif      # Tile 0 features (256 channels)
    ├── features_vit_b_1.tif      # Tile 1 features
    ├── features_vit_b_2.tif      # ...
    └── image.csv                 # Spatial index
        │
        │  id,filepath,minx,maxx,miny,maxy,mint,maxt,crs,res
        │  0,features_vit_b_0.tif,100,200,100,200,0,inf,EPSG:4326,10
        │  1,features_vit_b_1.tif,200,300,100,200,0,inf,EPSG:4326,10
```

---

## Component Details

### venv_manager.py

**Responsibilities:**
- Python discovery across platforms
- Virtual environment creation
- Package installation via pip
- sys.path injection for in-process imports

**Key Functions:**
```python
_get_system_python()           # Find suitable Python executable
create_venv()                  # Create virtual environment
install_dependencies()         # Install all required packages
verify_venv()                  # Check all packages importable
ensure_venv_packages_available()  # Add to sys.path
```

### feature_encoder.py

**Responsibilities:**
- Spawn encoding subprocess
- Send raster configuration
- Parse progress updates
- Handle cancellation

**Subprocess Communication:**
```python
process = subprocess.Popen(
    [venv_python, worker_script],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Send config
process.stdin.write(json.dumps(config).encode() + b'\n')

# Read progress
for line in process.stdout:
    data = json.loads(line)
    if data["status"] == "progress":
        progress_callback(data["percent"], data["message"])
```

### sam_predictor.py

**Responsibilities:**
- Maintain long-running prediction subprocess
- Send prediction requests
- Receive and decode masks
- Handle process lifecycle

**Persistent Process:**
```python
class SamPredictorNoImgEncoder:
    def __init__(self):
        self.process = None  # Lazy initialization

    def predict(self, points, labels):
        if self.process is None:
            self._start_worker()

        # Send prediction request
        self._send({"command": "predict", "points": points, "labels": labels})

        # Receive mask
        response = self._receive()
        mask = np.frombuffer(base64.b64decode(response["mask"]), dtype=np.uint8)
        return mask
```

### feature_dataset.py

**Responsibilities:**
- Load feature tile index
- Spatial queries for point-to-tile mapping
- Cache loaded features in memory

**Uses venv packages in QGIS process:**
```python
from .venv_manager import ensure_venv_packages_available
ensure_venv_packages_available()

import pandas as pd  # For CSV index
import torch         # For feature tensors
```

---

## Summary

The QGIS AI-Segmentation plugin uses a **hybrid subprocess architecture** to achieve:

1. **Stability**: PyTorch crashes don't affect QGIS
2. **Compatibility**: Works across macOS, Windows, Linux
3. **Isolation**: Dependencies in dedicated virtual environment
4. **Performance**: Cached features for fast predictions

Key design decisions:
- External Python (Homebrew/OSGeo4W) for subprocess execution
- Virtual environment for dependency isolation
- sys.path injection for lightweight in-process operations
- JSON over stdin/stdout for IPC

---

## Future Improvements

For recommendations on refactoring this architecture to achieve better separation of business logic from QGIS-specific code (enabling multi-platform support), see [ARCHITECTURE_RECOMMENDATIONS.md](./ARCHITECTURE_RECOMMENDATIONS.md).
