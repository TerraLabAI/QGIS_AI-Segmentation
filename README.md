# AI Segmentation in QGIS [![QGIS](https://img.shields.io/badge/QGIS-3.22+-93b023?style=flat-square&logo=qgis&logoColor=white)](https://qgis.org) [![Windows](https://img.shields.io/badge/Windows-0078D6?style=flat-square&logo=windows&logoColor=white)]() [![macOS](https://img.shields.io/badge/macOS-000000?style=flat-square&logo=apple&logoColor=white)]() [![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)]() [![License: GPL v2](https://img.shields.io/badge/License-GPLv2-blue.svg?style=flat-square)](LICENSE)

## AI segmentation for your geospatial rasters, inside QGIS

Click one object and get its outline as a polygon. Or name what you want
("buildings", "solar panels", "trees") and get every match across a drawn area
as vector polygons, ready to edit and export. No GPU needed.

<img src="https://github.com/user-attachments/assets/8528dc25-0dc7-4102-b242-5a223339db36" alt="AI Segmentation turning aerial imagery into building polygons in QGIS" width="700"/>

## What it does

AI Segmentation by TerraLab turns your imagery into vector polygons that replace tracing by hand. It uses its own detection models, tuned for aerial and satellite imagery.

- **Automatic mode**: draw a zone on your imagery, type the object ("buildings", "trees", "solar panels") and get every match in one run, with a live preview. It runs on TerraLab servers in Europe, so no GPU is needed.
- **Semi-Auto mode**: click an object, add positive and negative points, get a clean outline. Choose *My computer* (local, offline) or *Cloud AI* (nothing to download).
- **Your imagery**: orthophotos, drone and satellite images as GeoTIFF, JPG or PNG, and online basemaps (WMS, XYZ tiles).
- **Review before you keep**: right angles for buildings, expand, contract, simplify, round corners, fill holes.
- **Export**: GeoPackage, Shapefile or GeoJSON.
- **Free tier, no card**: 1.5 km² (Automatic) + 20 objects (Semi-Auto) a month. Semi-Auto on *My computer* has no counter. Plans and limits: https://terra-lab.ai/pricing
- **Platforms**: QGIS 3.22 to QGIS 4, on Windows, macOS and Linux. 12 languages.

## Install

In QGIS, open *Plugins > Manage and Install Plugins*, search "AI Segmentation", install. A free account unlocks the free tier, no card needed.

## Links

- Page, tutorial and plans: https://terra-lab.ai/ai-segmentation
- QGIS plugin repository: https://plugins.qgis.org/plugins/AI_Segmentation/
- Guide: [AI Segmentation for QGIS, the complete guide](https://terra-lab.ai/blog/ai-segmentation-complete-guide)
- [Detect building footprints in QGIS: 3 methods](https://terra-lab.ai/blog/detect-building-footprints-qgis)
- [Detect solar panels in QGIS: 5 methods](https://terra-lab.ai/blog/solar-panel-detection-qgis)
- Other TerraLab plugins: [AI Agent](https://github.com/TerraLabAI/QGIS_AI-Agent), [AI Edit](https://github.com/TerraLabAI/QGIS_AI-Edit)
- Bugs and requests: https://github.com/TerraLabAI/QGIS_AI-Segmentation/issues

License: GPL-2.0-or-later. Made by [TerraLab](https://terra-lab.ai), terra-lab.ai.

---

## Data & privacy

AI Segmentation has two modes with different privacy profiles. **Semi-Auto mode
asks you where the segmentation runs**, on the page you start it from. Pick *My
computer* and your imagery and clicks never leave it. Pick *Cloud AI*, the
offered choice because it needs no download, and small image crops of the area
you are working on go to our servers in Europe. The card you pick from says so
and links to this policy.

**Automatic mode is cloud-powered**: the imagery tiles inside the zone you draw
are sent to our detection service for processing during the run.

If you sign in, the plugin contacts our server to check your licence and your
credits. Usage statistics are on by default and you can switch them off in
*Account Settings*. They carry errors, versions and the words you type, linked
to your account, and never your imagery, layers or coordinates. See our
[Privacy Policy](https://terra-lab.ai/privacy-policy).

---

## License

The plugin code is GPL-2.0-or-later ([LICENSE](LICENSE)). The TerraLab names and
logos are not covered, see [TRADEMARKS.md](TRADEMARKS.md).

The hosted service (API responses, prompts, model settings) is not in this
repository and is not licensed for reuse. Text and data mining rights on it are
reserved (EU Directive 2019/790, Art. 4). See our
[Terms of Use](https://terra-lab.ai/terms-of-use).
