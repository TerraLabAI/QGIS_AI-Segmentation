# AI Segmentation in QGIS [![QGIS](https://img.shields.io/badge/QGIS-3.22+-93b023?style=flat-square&logo=qgis&logoColor=white)](https://qgis.org) [![Windows](https://img.shields.io/badge/Windows-0078D6?style=flat-square&logo=windows&logoColor=white)]() [![macOS](https://img.shields.io/badge/macOS-000000?style=flat-square&logo=apple&logoColor=white)]() [![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)]() [![License: GPL v2](https://img.shields.io/badge/License-GPLv2-blue.svg?style=flat-square)](LICENSE)

## Segment Anything in your geospatial rasters, inside QGIS

Click one object and get its outline as a polygon. Or name what you want
("buildings", "solar panels", "trees") and get every match across a drawn area
as vector polygons, ready to edit and export. No GPU needed.

<img src="https://github.com/user-attachments/assets/8528dc25-0dc7-4102-b242-5a223339db36" alt="AI Segmentation turning aerial imagery into building polygons in QGIS" width="700"/>

## Install

QGIS 3.22 or later, on Windows, macOS or Linux. In QGIS, open *Plugins >
Manage and Install Plugins*, search "AI Segmentation", install. A free account
unlocks the free tier, no card needed.

- Tutorial, documentation and plans: https://terra-lab.ai/ai-segmentation
- Plugin page on the QGIS repository: https://plugins.qgis.org/plugins/AI_Segmentation/
- Bugs and requests: https://github.com/TerraLabAI/QGIS_AI-Segmentation/issues

License: GPL-2.0-or-later. Made by [TerraLab](https://terra-lab.ai).

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
