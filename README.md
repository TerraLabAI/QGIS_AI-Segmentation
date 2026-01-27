
# AI Segmentation in QGIS [![QGIS](https://img.shields.io/badge/QGIS-3.0+-93b023?style=flat-square&logo=qgis&logoColor=white)](https://qgis.org) [![Windows](https://img.shields.io/badge/Windows-0078D6?style=flat-square&logo=windows&logoColor=white)]() [![macOS](https://img.shields.io/badge/macOS-000000?style=flat-square&logo=apple&logoColor=white)]() [![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)]()

### Segment anything in your geospatial rasters using AI


<img src="https://github.com/user-attachments/assets/8528dc25-0dc7-4102-b242-5a223339db36" alt="Demo" width="700"/>

---
[Documentation](https://www.terra-lab.ai/docs/ai-segmentation) · [Report Issue](https://github.com/TerraLabAI/QGIS_AI-Segmentation/issues) · [TerraLab](https://terra-lab.ai)


---

## Overview

AI Segmentation brings Meta's **Segment Anything Model (SAM)** to QGIS. Point-and-click on any raster to extract precise vector polygons

<br/>

## Features

| | |
|---|---|
| **Zero-config setup** | Dependencies install automatically on first launch |
| **Interactive segmentation** | Left-click to include, right-click to exclude |
| **Cross-platform** | Native support for Windows, macOS, and Linux |
| **CPU acceleration** |  MPS (Apple Silicon) support |
| **Vector export** | Save masks directly to GeoPackage layers |

<br/>

## Architecture

The plugin uses **subprocess isolation** to keep QGIS stable, PyTorch runs in a separate process.

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

<br/>

## Quick Start
We're currently waiting that the QGIS Team accept our plugin... for now you can clone the plugin repo on your plugins/ folder in QGIS


**1. Install**
```
QGIS → Plugins → Manage and Install Plugins → Search "AI Segmentation"
```

**2. Setup** *(first launch only)*
```
- Open plugin panel → Click "Install Dependencies" → Wait ~3 min
- Install AI model
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
| Disk space | ~2 GB (PyTorch + SAM model) |
| RAM | 8 GB minimum |
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


