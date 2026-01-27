<div align="center">

<img src="resources/icons/terralab-banner.png" alt="TerraLab" width="600"/>

<br/>
<br/>

# AI Segmentation

**Segment anything in your geospatial rasters using Meta's SAM**

<br/>

[![QGIS](https://img.shields.io/badge/QGIS-3.0+-93b023?style=flat-square&logo=qgis&logoColor=white)](https://qgis.org)
[![License](https://img.shields.io/badge/License-GPL--3.0-blue?style=flat-square)](LICENSE)
[![Windows](https://img.shields.io/badge/Windows-0078D6?style=flat-square&logo=windows&logoColor=white)]()
[![macOS](https://img.shields.io/badge/macOS-000000?style=flat-square&logo=apple&logoColor=white)]()
[![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)]()

<br/>

[Documentation](https://www.terra-lab.ai/docs/ai-segmentation) · [Report Issue](https://github.com/TerraLabAI/QGIS_AI-Segmentation/issues) · [TerraLab](https://terra-lab.ai)

<br/>

<img src="assets/demo.gif" alt="Demo" width="700"/>

</div>

<br/>

---

<br/>

## Overview

AI Segmentation brings Meta's **Segment Anything Model (SAM)** to QGIS. Point-and-click on any raster to extract precise vector polygons — no training required.

<br/>

## Features

| | |
|---|---|
| **Zero-config setup** | Dependencies install automatically on first launch |
| **Interactive segmentation** | Left-click to include, right-click to exclude |
| **Cross-platform** | Native support for Windows, macOS, and Linux |
| **GPU acceleration** | CUDA (NVIDIA) and MPS (Apple Silicon) support |
| **Vector export** | Save masks directly to GeoPackage layers |

<br/>

## Architecture

The plugin uses **subprocess isolation** to keep QGIS stable — PyTorch runs in a separate process.

```
┌─────────────────────────────────────────────────────────────┐
│                         QGIS                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Plugin UI  ·  Map Tool  ·  Layer Export              │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │ JSON / stdin-stdout
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
┌─────────────────────┐           ┌─────────────────────┐
│  Encoding Process   │           │  Prediction Process │
│  ─────────────────  │           │  ─────────────────  │
│  Python 3.12        │           │  Python 3.12        │
│  PyTorch + SAM      │           │  PyTorch + SAM      │
│  Rasterio           │           │  NumPy              │
└─────────────────────┘           └─────────────────────┘
```

> PyTorch crashes cannot affect QGIS. The subprocess can restart without restarting your session.

<br/>

## Quick Start

**1. Install**
```
QGIS → Plugins → Manage and Install Plugins → Search "AI Segmentation"
```

**2. Setup** *(first launch only)*
```
Open plugin panel → Click "Install Dependencies" → Wait ~5 min
```

**3. Segment**
```
Select raster layer → Start segmentation → Click on map → Export polygons
```

<br/>

## Requirements

| Component | Specification |
|-----------|---------------|
| QGIS | 3.0 or later |
| Disk space | ~2.5 GB (PyTorch + SAM model) |
| RAM | 8 GB minimum, 16 GB recommended |
| GPU | Optional (CUDA 11.8+ or Apple Silicon) |

<br/>

## Controls

| Input | Action |
|-------|--------|
| `Left-click` | Add foreground point (include) |
| `Right-click` | Add background point (exclude) |
| `Ctrl+Z` | Undo last point |
| `S` | Save current polygon |

<br/>

## Documentation

Full documentation available at **[terra-lab.ai/docs/ai-segmentation](https://www.terra-lab.ai/docs/ai-segmentation)**

<br/>

## License

[GPL-3.0](LICENSE) — Compatible with QGIS core.

<br/>

---

<br/>

<div align="center">

**Built by [TerraLab](https://terra-lab.ai)** — The geospatial infrastructure layer for spatial AI

<br/>

[![Email](https://img.shields.io/badge/contact@terra--lab.ai-D14836?style=flat-square&logo=mail.ru&logoColor=white)](mailto:contact@terra-lab.ai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/company/terralab-3d)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=flat-square&logo=youtube&logoColor=white)](https://youtube.com/@terra-lab)

</div>
