"""
SAM Image Encoder for AI Segmentation

Handles the heavy lifting of encoding raster images into feature embeddings
that can be used for real-time interactive segmentation.

The encoding is done once per image and cached to disk for reuse.
"""

import os
import hashlib
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Callable
import json

import numpy as np

from qgis.core import (
    QgsRasterLayer,
    QgsMessageLog,
    Qgis,
    QgsRectangle,
)

# SAM input size (fixed)
SAM_INPUT_SIZE = 1024


class SAMEncoder:
    """
    SAM Image Encoder using ONNX Runtime.

    Encodes geospatial raster images into feature embeddings that can be
    used for fast interactive segmentation.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the SAM encoder.

        Args:
            model_path: Path to the ONNX encoder model file
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None

    def load_model(self, model_path: str = None) -> bool:
        """
        Load the ONNX encoder model.

        Args:
            model_path: Path to the ONNX model file

        Returns:
            True if model loaded successfully
        """
        try:
            import onnxruntime as ort

            if model_path:
                self.model_path = model_path

            if not self.model_path or not os.path.exists(self.model_path):
                QgsMessageLog.logMessage(
                    f"Encoder model not found: {self.model_path}",
                    "AI Segmentation",
                    level=Qgis.Critical
                )
                return False

            # Create inference session with CPU provider
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )

            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            QgsMessageLog.logMessage(
                f"Encoder model loaded: {self.model_path}",
                "AI Segmentation",
                level=Qgis.Info
            )
            return True

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to load encoder model: {str(e)}",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return False

    def preprocess_image(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Preprocess image for SAM encoder.

        SAM expects:
        - RGB image (3 channels)
        - Size 1024x1024
        - Normalized with ImageNet mean/std
        - Shape: (1, 3, 1024, 1024) - NCHW format

        Args:
            image: Input image as numpy array (H, W, C) in RGB format

        Returns:
            Tuple of (preprocessed_image, transform_info)
        """
        original_size = image.shape[:2]  # (H, W)

        # Resize to 1024x1024 while maintaining aspect ratio
        h, w = original_size
        scale = SAM_INPUT_SIZE / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Simple resize using numpy (avoid opencv dependency)
        # For production, consider using PIL or skimage
        from .image_utils import resize_image
        resized = resize_image(image, (new_h, new_w))

        # Pad to 1024x1024
        padded = np.zeros((SAM_INPUT_SIZE, SAM_INPUT_SIZE, 3), dtype=np.float32)
        padded[:new_h, :new_w, :] = resized

        # Normalize with ImageNet mean and std
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        normalized = (padded - mean) / std

        # Convert to NCHW format
        preprocessed = normalized.transpose(2, 0, 1)[np.newaxis, ...]

        transform_info = {
            "original_size": original_size,
            "resized_size": (new_h, new_w),
            "scale": scale,
            "input_size": SAM_INPUT_SIZE
        }

        return preprocessed.astype(np.float32), transform_info

    def encode(
        self,
        image: np.ndarray,
        progress_callback: Callable[[int, str], None] = None
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Encode an image into SAM feature embeddings.

        Args:
            image: Input image as numpy array (H, W, C) RGB format
            progress_callback: Optional callback(percent, message)

        Returns:
            Tuple of (features, transform_info) or (None, {}) on error
        """
        if self.session is None:
            QgsMessageLog.logMessage(
                "Encoder model not loaded",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return None, {}

        try:
            if progress_callback:
                progress_callback(10, "Preprocessing image...")

            # Preprocess
            preprocessed, transform_info = self.preprocess_image(image)

            if progress_callback:
                progress_callback(30, "Running encoder (this may take a while)...")

            # Run inference
            features = self.session.run(
                [self.output_name],
                {self.input_name: preprocessed}
            )[0]

            if progress_callback:
                progress_callback(100, "Encoding complete!")

            QgsMessageLog.logMessage(
                f"Image encoded. Features shape: {features.shape}",
                "AI Segmentation",
                level=Qgis.Info
            )

            return features, transform_info

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Encoding failed: {str(e)}",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return None, {}

    def encode_raster_layer(
        self,
        layer: QgsRasterLayer,
        extent: QgsRectangle = None,
        progress_callback: Callable[[int, str], None] = None
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Encode a QGIS raster layer.

        Args:
            layer: QgsRasterLayer to encode
            extent: Optional extent to encode (defaults to full layer)
            progress_callback: Optional callback(percent, message)

        Returns:
            Tuple of (features, transform_info) or (None, {}) on error
        """
        if progress_callback:
            progress_callback(0, "Reading raster data...")

        try:
            from .image_utils import raster_to_numpy
            image, geo_transform = raster_to_numpy(layer, extent)

            if image is None:
                return None, {}

            # Store geo transform in transform_info
            features, transform_info = self.encode(image, progress_callback)

            if features is not None:
                transform_info["geo_transform"] = geo_transform
                transform_info["layer_crs"] = layer.crs().authid()
                transform_info["extent"] = [
                    extent.xMinimum() if extent else layer.extent().xMinimum(),
                    extent.yMinimum() if extent else layer.extent().yMinimum(),
                    extent.xMaximum() if extent else layer.extent().xMaximum(),
                    extent.yMaximum() if extent else layer.extent().yMaximum(),
                ]

            return features, transform_info

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to encode raster layer: {str(e)}",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return None, {}


def save_features(
    features: np.ndarray,
    transform_info: dict,
    output_path: str
) -> bool:
    """
    Save encoded features to disk.

    Args:
        features: Feature embeddings array
        transform_info: Dictionary with transform information
        output_path: Path to save the features (without extension)

    Returns:
        True if saved successfully
    """
    try:
        # Save features as .npy
        np.save(f"{output_path}.npy", features)

        # Save transform info as JSON
        with open(f"{output_path}.json", 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            info = {}
            for k, v in transform_info.items():
                if isinstance(v, tuple):
                    info[k] = list(v)
                elif isinstance(v, np.ndarray):
                    info[k] = v.tolist()
                else:
                    info[k] = v
            json.dump(info, f, indent=2)

        QgsMessageLog.logMessage(
            f"Features saved to: {output_path}",
            "AI Segmentation",
            level=Qgis.Info
        )
        return True

    except Exception as e:
        QgsMessageLog.logMessage(
            f"Failed to save features: {str(e)}",
            "AI Segmentation",
            level=Qgis.Critical
        )
        return False


def load_features(features_path: str) -> Tuple[Optional[np.ndarray], dict]:
    """
    Load encoded features from disk.

    Args:
        features_path: Path to features file (without extension)

    Returns:
        Tuple of (features, transform_info) or (None, {}) on error
    """
    try:
        features = np.load(f"{features_path}.npy")

        with open(f"{features_path}.json", 'r') as f:
            transform_info = json.load(f)

        return features, transform_info

    except Exception as e:
        QgsMessageLog.logMessage(
            f"Failed to load features: {str(e)}",
            "AI Segmentation",
            level=Qgis.Critical
        )
        return None, {}


def _is_file_based_layer(layer: QgsRasterLayer) -> bool:
    """
    Check if a layer is file-based (as opposed to web-based like XYZ, WMS).

    Args:
        layer: QgsRasterLayer

    Returns:
        True if the layer source is a local file
    """
    source = layer.source()

    # Web layers typically have these patterns
    web_indicators = ['http://', 'https://', 'type=xyz', 'type=wms', 'url=']
    for indicator in web_indicators:
        if indicator in source.lower():
            return False

    # Check if it's an actual file path
    return os.path.isfile(source)


def get_features_path(layer: QgsRasterLayer, model_id: str = None) -> str:
    """
    Get the path where features should be cached for a layer and model.

    Cache is stored per-model to support multiple model architectures.

    For file-based layers: cache in .ai_segmentation_cache/{model_id}/ next to the file
    For web-based layers: cache in a user temp directory with hashed name

    Args:
        layer: QgsRasterLayer
        model_id: Model identifier. If None, uses "default" (backward compatibility)

    Returns:
        Path for features cache (without extension)
    """
    from .model_registry import DEFAULT_MODEL_ID

    if model_id is None:
        model_id = DEFAULT_MODEL_ID

    source = layer.source()

    if _is_file_based_layer(layer):
        # File-based layer: cache next to the source file
        source_path = Path(source)
        cache_dir = source_path.parent / ".ai_segmentation_cache" / model_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir / source_path.stem)
    else:
        # Web-based layer (XYZ, WMS, etc.): use a temp directory
        # Create a unique cache name based on layer source + extent
        extent = layer.extent()
        cache_key = f"{source}_{extent.toString()}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]

        # Use a persistent cache directory in user's home
        cache_base = Path.home() / ".qgis_ai_segmentation_cache"
        cache_dir = cache_base / cache_hash / model_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Use layer name as the stem for readability
        layer_name = layer.name().replace(" ", "_").replace("/", "_")
        return str(cache_dir / layer_name)


def has_cached_features(layer: QgsRasterLayer, model_id: str = None) -> bool:
    """
    Check if a layer has cached features for a specific model.

    Args:
        layer: QgsRasterLayer
        model_id: Model identifier. If None, uses DEFAULT_MODEL_ID

    Returns:
        True if cached features exist for the specified model
    """
    features_path = get_features_path(layer, model_id)
    return (
        os.path.exists(f"{features_path}.npy") and
        os.path.exists(f"{features_path}.json")
    )
