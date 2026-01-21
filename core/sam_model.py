"""
Unified SAM Model for AI Segmentation

Provides a simple, high-level interface for SAM segmentation.
Hides the encoder/decoder complexity from the user.

Usage:
    model = SAMModel()
    model.load()  # Downloads models if needed
    model.prepare_layer(raster_layer)  # Auto-encodes
    mask, score = model.segment(points, labels)
"""

from pathlib import Path
from typing import Tuple, Optional, List, Callable
import numpy as np

from qgis.core import QgsRasterLayer, QgsMessageLog, Qgis

from .model_manager import (
    models_exist,
    download_models,
    get_encoder_path,
    get_decoder_path,
)
from .sam_encoder import (
    SAMEncoder,
    save_features,
    load_features,
    get_features_path,
    has_cached_features,
)
from .sam_decoder import SAMDecoder, PromptManager


class SAMModel:
    """
    Unified SAM Model interface.

    Combines encoder and decoder into a single, easy-to-use class.
    Handles model downloading, encoding, caching, and segmentation.
    """

    def __init__(self):
        """Initialize the SAM model."""
        self._encoder: Optional[SAMEncoder] = None
        self._decoder: Optional[SAMDecoder] = None
        self._loaded = False

        # Current layer state
        self._current_layer: Optional[QgsRasterLayer] = None
        self._features: Optional[np.ndarray] = None
        self._transform_info: dict = {}

        # Prompt manager for interactive segmentation
        self.prompts = PromptManager()

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded and ready."""
        return self._loaded

    @property
    def is_ready(self) -> bool:
        """Check if ready for segmentation (models loaded + layer prepared)."""
        return self._loaded and self._features is not None

    @property
    def models_available(self) -> bool:
        """Check if model files exist on disk."""
        return models_exist()

    def download_models(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Tuple[bool, str]:
        """
        Download SAM models if not already present.

        Args:
            progress_callback: Optional callback(percent, message)

        Returns:
            Tuple of (success, message)
        """
        return download_models(progress_callback)

    def load(self) -> Tuple[bool, str]:
        """
        Load the SAM models.

        Downloads models if not present, then loads them into memory.

        Returns:
            Tuple of (success, message)
        """
        # Check if models exist
        if not models_exist():
            return False, "Models not downloaded. Call download_models() first."

        try:
            # Load encoder
            self._encoder = SAMEncoder()
            if not self._encoder.load_model(str(get_encoder_path())):
                return False, "Failed to load encoder model"

            # Load decoder
            self._decoder = SAMDecoder()
            if not self._decoder.load_model(str(get_decoder_path())):
                return False, "Failed to load decoder model"

            self._loaded = True
            QgsMessageLog.logMessage(
                "SAM models loaded successfully",
                "AI Segmentation",
                level=Qgis.Info
            )
            return True, "Models loaded successfully"

        except Exception as e:
            self._loaded = False
            return False, f"Failed to load models: {str(e)}"

    def prepare_layer(
        self,
        layer: QgsRasterLayer,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        force_encode: bool = False
    ) -> Tuple[bool, str]:
        """
        Prepare a raster layer for segmentation.

        Automatically encodes the layer if no cache exists.
        Uses cached features if available.

        Args:
            layer: QgsRasterLayer to prepare
            progress_callback: Optional callback(percent, message)
            force_encode: If True, re-encode even if cache exists

        Returns:
            Tuple of (success, message)
        """
        if not self._loaded:
            return False, "Models not loaded. Call load() first."

        try:
            # Clear previous state
            self.prompts.clear()

            # Check for cached features
            if not force_encode and has_cached_features(layer):
                if progress_callback:
                    progress_callback(50, "Loading cached features...")

                features_path = get_features_path(layer)
                self._features, self._transform_info = load_features(features_path)

                if self._features is not None:
                    self._current_layer = layer
                    if progress_callback:
                        progress_callback(100, "Layer ready (from cache)")

                    QgsMessageLog.logMessage(
                        f"Loaded cached features for: {layer.name()}",
                        "AI Segmentation",
                        level=Qgis.Info
                    )
                    return True, "Layer ready (using cached features)"

            # Need to encode
            if progress_callback:
                progress_callback(0, "Encoding image (this may take a few minutes)...")

            self._features, self._transform_info = self._encoder.encode_raster_layer(
                layer,
                progress_callback=progress_callback
            )

            if self._features is None:
                return False, "Failed to encode layer"

            # Save to cache
            features_path = get_features_path(layer)
            save_features(self._features, self._transform_info, features_path)

            self._current_layer = layer

            if progress_callback:
                progress_callback(100, "Layer ready")

            return True, "Layer encoded and ready"

        except Exception as e:
            self._features = None
            self._transform_info = {}
            return False, f"Failed to prepare layer: {str(e)}"

    def segment(
        self,
        points: List[Tuple[float, float]],
        labels: List[int]
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Segment using point prompts.

        Args:
            points: List of (x, y) points in map coordinates
            labels: List of labels (1=positive/foreground, 0=negative/background)

        Returns:
            Tuple of (binary_mask, confidence_score) or (None, 0.0) on error
        """
        if not self.is_ready:
            QgsMessageLog.logMessage(
                "Model not ready for segmentation",
                "AI Segmentation",
                level=Qgis.Warning
            )
            return None, 0.0

        return self._decoder.predict_mask(
            self._features,
            points,
            labels,
            self._transform_info
        )

    def segment_with_prompts(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Segment using the current prompts from the PromptManager.

        Returns:
            Tuple of (binary_mask, confidence_score) or (None, 0.0) on error
        """
        if not self.prompts.has_points():
            return None, 0.0

        points, labels = self.prompts.get_all_points()
        return self.segment(points, labels)

    def add_positive_point(self, x: float, y: float) -> Tuple[Optional[np.ndarray], float]:
        """
        Add a positive (foreground) point and return updated mask.

        Args:
            x, y: Point coordinates in map CRS

        Returns:
            Tuple of (binary_mask, confidence_score)
        """
        self.prompts.add_positive(x, y)
        return self.segment_with_prompts()

    def add_negative_point(self, x: float, y: float) -> Tuple[Optional[np.ndarray], float]:
        """
        Add a negative (background) point and return updated mask.

        Args:
            x, y: Point coordinates in map CRS

        Returns:
            Tuple of (binary_mask, confidence_score)
        """
        self.prompts.add_negative(x, y)
        return self.segment_with_prompts()

    def undo_point(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Undo the last point and return updated mask.

        Returns:
            Tuple of (binary_mask, confidence_score)
        """
        self.prompts.undo()
        if self.prompts.has_points():
            return self.segment_with_prompts()
        return None, 0.0

    def clear_points(self):
        """Clear all prompt points."""
        self.prompts.clear()

    def get_transform_info(self) -> dict:
        """Get the transform info for the current layer."""
        return self._transform_info.copy()

    def get_current_layer(self) -> Optional[QgsRasterLayer]:
        """Get the currently prepared layer."""
        return self._current_layer

    def unload(self):
        """Unload models and free memory."""
        self._encoder = None
        self._decoder = None
        self._loaded = False
        self._features = None
        self._transform_info = {}
        self._current_layer = None
        self.prompts.clear()

        QgsMessageLog.logMessage(
            "SAM models unloaded",
            "AI Segmentation",
            level=Qgis.Info
        )
