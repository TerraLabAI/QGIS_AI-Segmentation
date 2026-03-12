
# AI Segmentation in QGIS [![QGIS](https://img.shields.io/badge/QGIS-3.0+-93b023?style=flat-square&logo=qgis&logoColor=white)](https://qgis.org) [![Windows](https://img.shields.io/badge/Windows-0078D6?style=flat-square&logo=windows&logoColor=white)]() [![macOS](https://img.shields.io/badge/macOS-000000?style=flat-square&logo=apple&logoColor=white)]() [![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)]()

## Segment anything in your geospatial rasters using AI
### Follow this tutorial/documentation to use the plugin :
 https://terra-lab.ai/ai-segmentation
---


<img src="https://github.com/user-attachments/assets/8528dc25-0dc7-4102-b242-5a223339db36" alt="Demo" width="700"/>

---

## Release

Releases are automated via GitHub Actions:

1. Create a GitHub Release with a tag matching `vX.Y.Z` (e.g. `gh release create v0.8.0`)
2. The `release.yml` workflow automatically:
   - Packages the plugin (dev files excluded via `.gitattributes`)
   - Attaches the zip to the GitHub Release
   - Publishes to [plugins.qgis.org](https://plugins.qgis.org)

**Prerequisites:** `OSGEO_USERNAME` and `OSGEO_PASSWORD` secrets must be configured in the repo's GitHub Settings > Secrets.
