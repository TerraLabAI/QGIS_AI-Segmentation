# QGIS AI-Segmentation - AI Segmentation for QGIS

A QGIS plugin that brings the power of Meta's Segment Anything Model (SAM) to geospatial image segmentation.

![AI Segmentation](resources/icons/ai_segmentation_icon.png)

## Features

- **Interactive Segmentation**: Click on objects to segment them in real-time
- **CPU-Optimized**: Works on any computer without GPU requirements
- **Geospatial-Aware**: Properly handles georeferenced rasters and exports georeferenced vector polygons
- **Cross-Platform**: Works on Windows, macOS, and Linux

## How It Works

AI Segmentation uses a two-stage architecture for efficient interactive segmentation:

1. **Encoding Stage** (one-time): The raster image is processed through SAM's image encoder, generating feature embeddings that are cached to disk.

2. **Interactive Stage** (real-time): When you click on the map, only the lightweight decoder runs, enabling instant mask generation on CPU.

```
┌─────────────────────┐     ┌─────────────────────┐
│   Raster Image      │────▶│   SAM Encoder       │────▶ Features (cached)
└─────────────────────┘     └─────────────────────┘
                                    │
                                    ▼
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   User Clicks       │────▶│   SAM Decoder       │────▶│   Segmentation      │
│   (prompts)         │     │   (real-time)       │     │   Mask              │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

## Installation

### Prerequisites

- QGIS 3.28 LTR or later
- Python 3.9+

### Install the Plugin

1. Download or clone this repository
2. Copy the entire `QGIS_AI-Segmentation` folder to your QGIS plugins directory:
   - **Windows**: `C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
   - **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
   - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`

3. Download the SAM ViT-B ONNX models and place them in `models/` folder inside the plugin:
   - `sam_vit_b_encoder.onnx` (~375 MB)
   - `sam_vit_b_decoder.onnx` (~17 MB)

4. Restart QGIS and enable the plugin in `Plugins > Manage and Install Plugins`

### Downloading the Models

You can export the models yourself using [samexporter](https://github.com/vietanhdev/samexporter):

```bash
pip install samexporter
samexporter export-encoder --model-type vit_b --output models/sam_vit_b_encoder.onnx
samexporter export-decoder --model-type vit_b --output models/sam_vit_b_decoder.onnx
```

Or download pre-exported models from the releases page.

### Understanding SAM Models: Encoder vs Decoder

#### D'où viennent ces modèles ?

Les modèles SAM (Segment Anything Model) ont été créés par **Meta AI (Facebook Research)** en 2023. Le modèle original est disponible sur GitHub :

- **Repository officiel** : [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
- **Modèles PyTorch originaux** : Les poids pré-entraînés sont disponibles en format PyTorch (`.pth`) sur le dépôt Meta

Les modèles ONNX utilisés par ce plugin sont des **conversions** du modèle PyTorch original, optimisées pour l'inférence sur CPU via ONNX Runtime.

#### Architecture Encodeur/Décodeur de SAM

SAM utilise une architecture **encodeur-décodeur** spécialement conçue pour la segmentation d'images :

**1. L'Encodeur (`sam_vit_b_encoder.onnx` - ~375 MB)**
- **Rôle** : Transforme une image brute (1024×1024 pixels RGB) en **embeddings de caractéristiques** (feature embeddings)
- **Architecture** : Basé sur Vision Transformer (ViT-B) avec 12 couches d'attention
- **Entrée** : Image RGB normalisée (1, 3, 1024, 1024)
- **Sortie** : Tenseur de caractéristiques de forme (1, 256, 64, 64) - une représentation dense de l'image
- **Pourquoi c'est lourd** : Il doit analyser toute l'image pixel par pixel pour extraire les caractéristiques visuelles (contours, textures, objets)
- **Exécution** : Prend 1-5 minutes selon la taille de l'image (c'est pourquoi on le fait une seule fois et on cache le résultat)

**2. Le Décodeur (`sam_vit_b_decoder.onnx` - ~17 MB)**
- **Rôle** : Prend les embeddings de l'encodeur + les prompts utilisateur (clics) et génère un **masque de segmentation**
- **Architecture** : Réseau de décodage léger avec mécanisme d'attention
- **Entrée** : 
  - Features de l'encodeur (déjà calculées)
  - Coordonnées des points cliqués (x, y)
  - Labels (1 = inclure, 0 = exclure)
- **Sortie** : Masque binaire (H, W) indiquant quels pixels appartiennent à l'objet
- **Pourquoi c'est léger** : Il ne traite que les prompts (quelques points) et utilise les features pré-calculées
- **Exécution** : ~50-200ms par clic (temps réel)

#### Ce qu'il y a vraiment dans ces fichiers ONNX

Les fichiers `.onnx` contiennent :
- **Les poids du réseau de neurones** : Des millions de paramètres (matrices de poids) qui ont été entraînés sur 11 millions d'images
- **L'architecture du modèle** : La structure du réseau (couches, connexions, opérations)
- **Les métadonnées** : Informations sur les formats d'entrée/sortie attendus

Ce sont essentiellement des **réseaux de neurones congelés** prêts à l'inférence, convertis depuis PyTorch vers le format ONNX (Open Neural Network Exchange) pour une exécution optimisée.

#### Liens pour télécharger les modèles

**Option 1 : Exporter depuis le modèle PyTorch original (recommandé)**
```bash
# Installer samexporter
pip install samexporter

# Télécharger automatiquement les poids PyTorch et exporter en ONNX
samexporter export-encoder --model-type vit_b --output sam_vit_b_encoder.onnx
samexporter export-decoder --model-type vit_b --output sam_vit_b_decoder.onnx
```

**Option 2 : Télécharger les modèles PyTorch originaux**
- **SAM ViT-B checkpoint** : [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- Puis convertir en ONNX avec `samexporter` ou un autre outil

**Option 3 : Télécharger des modèles ONNX pré-exportés**
- Certains dépôts GitHub proposent des modèles ONNX pré-exportés
- Vérifiez la compatibilité avec la version de `samexporter` utilisée

#### Pourquoi cette séparation Encodeur/Décodeur ?

Cette architecture permet :
- **Performance** : L'encodeur lourd ne s'exécute qu'une fois par image
- **Interactivité** : Le décodeur léger permet des réponses en temps réel
- **Efficacité CPU** : Pas besoin de GPU grâce à cette optimisation
- **Cache** : Les features peuvent être sauvegardées et réutilisées

## Usage

1. **Load a raster layer** (satellite image, orthophoto, etc.)
2. **Open AI Segmentation** from the toolbar or `Raster > AI Segmentation`
3. **Select the source layer** in the dropdown
4. **Click "Encode Image"** - this processes the image (takes 1-5 minutes depending on size)
5. **Click "Start Segmentation"** to activate the segmentation tool
6. **Left-click** on objects to include them in the segmentation
7. **Right-click** to exclude regions
8. **Click "Save Current Mask"** to add the polygon to the output layer
9. **Export to GeoPackage** when done

## Requirements

The plugin will automatically install these Python dependencies on first run:

- `onnxruntime >= 1.15.0` - ONNX inference engine
- `numpy >= 1.20.0` - Numerical operations

## Project Structure

```
QGIS_AI-Segmentation/           # Plugin root (this folder IS the plugin)
├── __init__.py                 # Plugin entry point
├── metadata.txt                # QGIS plugin metadata
├── requirements.txt            # Python dependencies
├── ai_segmentation_plugin.py   # Main plugin class
├── ai_segmentation_dockwidget.py # UI panel
├── ai_segmentation_maptool.py  # Map click tool
├── core/
│   ├── dependency_manager.py   # Auto-install dependencies
│   ├── sam_encoder.py          # Image encoding
│   ├── sam_decoder.py          # Mask decoding
│   ├── image_utils.py          # Image processing utilities
│   └── polygon_exporter.py     # Vector export
├── models/                     # ONNX model files (not included)
│   ├── sam_vit_b_encoder.onnx
│   └── sam_vit_b_decoder.onnx
├── resources/icons/            # Plugin icons
├── tests/                      # Test modules
└── ui/                         # UI modules
```

## Technical Notes

### Why SAM ViT-B?

We chose SAM ViT-B as the default model because:
- **Proven for geospatial**: Used successfully by [Geo-SAM](https://github.com/coolzhao/Geo-SAM)
- **Good balance**: Smaller than ViT-H/L but still accurate
- **Well-documented**: Extensive ONNX export support

### Pre-encoding Strategy

The key to achieving real-time performance on CPU is pre-encoding:
- The encoder (heavy) runs once per image
- Features are cached to disk (~50-100 MB per image)
- The decoder (light) runs in ~50-200ms per click

### Coordinate Systems

- All coordinates are handled in the source raster's CRS
- Output polygons are georeferenced
- Works with any projected or geographic CRS

## Roadmap

- [ ] Mask preview overlay on canvas
- [ ] Bounding box prompts
- [ ] GPU acceleration (optional)
- [ ] Processing algorithm for batch operations
- [ ] SAM2 support

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [Meta AI](https://github.com/facebookresearch/segment-anything) for the Segment Anything Model
- [Geo-SAM](https://github.com/coolzhao/Geo-SAM) for the pre-encoding architecture inspiration
- [samexporter](https://github.com/vietanhdev/samexporter) for ONNX export tools
