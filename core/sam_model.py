"""
Unified SAM Model for AI Segmentation

Provides a simple, high-level interface for SAM segmentation.
Hides the encoder/decoder complexity from the user.
Supports multiple model architectures (SAM, SAM2).

Usage:
    model = SAMModel(model_id="sam_vit_b")  # or "sam2_base_plus", "sam2_large"
    model.load()  # Downloads models if needed
    model.prepare_layer(raster_layer)  # Auto-encodes
    mask, score = model.segment(points, labels)

    # Switch models:
    model.switch_model("sam2_large")
"""

from pathlib import Path
from typing import Tuple, Optional, List, Callable
import numpy as np

from qgis.core import QgsRasterLayer, QgsMessageLog, Qgis

from .model_registry import (
    ModelConfig,
    get_model_config,
    get_all_models,
    DEFAULT_MODEL_ID,
)
from .model_manager import (
    models_exist,
    model_exists,
    download_model,
    get_encoder_path,
    get_decoder_path,
    get_installed_models,
    get_first_installed_model,
    migrate_legacy_models,
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
    Supports multiple model architectures through model_id selection.
    """

    def __init__(self, model_id: str = None):
        """
        Initialize the SAM model.

        Args:
            model_id: Model identifier (e.g., "sam_vit_b", "sam2_base_plus").
                      If None, uses first installed model or DEFAULT_MODEL_ID.
        """
        # Run migration for legacy model files
        migrate_legacy_models()

        # Determine which model to use
        if model_id is None:
            # Try to use first installed model, otherwise default
            model_id = get_first_installed_model() or DEFAULT_MODEL_ID

        self._model_id = model_id
        self._model_config: ModelConfig = get_model_config(model_id)

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
    def model_id(self) -> str:
        """Get the current model ID."""
        return self._model_id

    @property
    def model_config(self) -> ModelConfig:
        """Get the current model configuration."""
        return self._model_config

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
        """Check if current model files exist on disk."""
        return model_exists(self._model_id)

    def download_models(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Tuple[bool, str]:
        """
        Download the current model if not already present.

        Args:
            progress_callback: Optional callback(percent, message)

        Returns:
            Tuple of (success, message)
        """
        return download_model(self._model_id, progress_callback)

    def load(self) -> Tuple[bool, str]:
        """
        Load the SAM models for the current model_id.

        Downloads models if not present, then loads them into memory.

        Returns:
            Tuple of (success, message)
        """
        # Check if models exist
        if not model_exists(self._model_id):
            return False, f"Models not downloaded for {self._model_config.display_name}. Call download_models() first."

        try:
            # Load encoder
            self._encoder = SAMEncoder()
            encoder_path = get_encoder_path(self._model_id)
            if not self._encoder.load_model(str(encoder_path)):
                return False, f"Failed to load encoder model for {self._model_config.display_name}"

            # Load decoder with model config for tensor name mapping
            self._decoder = SAMDecoder(model_config=self._model_config)
            decoder_path = get_decoder_path(self._model_id)
            if not self._decoder.load_model(str(decoder_path)):
                return False, f"Failed to load decoder model for {self._model_config.display_name}"

            self._loaded = True
            QgsMessageLog.logMessage(
                f"SAM models loaded successfully: {self._model_config.display_name}",
                "AI Segmentation",
                level=Qgis.Info
            )
            return True, f"{self._model_config.display_name} loaded successfully"

        except Exception as e:
            self._loaded = False
            return False, f"Failed to load models: {str(e)}"

    def switch_model(self, model_id: str, force_reload: bool = False) -> Tuple[bool, str]:
        """
        Switch to a different model.

        This unloads the current model, clears features, and loads the new model.
        The layer will need to be re-prepared after switching.

        Args:
            model_id: New model identifier
            force_reload: If True, reload even if same model_id

        Returns:
            Tuple of (success, message)
        """
        if model_id == self._model_id and not force_reload:
            return True, f"Already using {self._model_config.display_name}"

        # Check if new model exists
        if not model_exists(model_id):
            return False, f"Model {model_id} is not downloaded"

        # Unload current model
        self._unload_model()

        # Update model ID and config
        self._model_id = model_id
        self._model_config = get_model_config(model_id)

        # Clear features (will need re-encoding)
        self._features = None
        self._transform_info = {}
        self._current_layer = None
        self.prompts.clear()

        # Load new model
        return self.load()

    def _unload_model(self):
        """Unload current model from memory."""
        self._encoder = None
        self._decoder = None
        self._loaded = False

    def prepare_layer(
        self,
        layer: QgsRasterLayer,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        force_encode: bool = False
    ) -> Tuple[bool, str]:
        """
        Prepare a raster layer for segmentation.

        Automatically encodes the layer if no cache exists for the current model.
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

            # Check for cached features (model-specific cache)
            if not force_encode and has_cached_features(layer, self._model_id):
                if progress_callback:
                    progress_callback(50, "Loading cached features...")

                features_path = get_features_path(layer, self._model_id)
                self._features, self._transform_info = load_features(features_path)

                if self._features is not None:
                    self._current_layer = layer
                    if progress_callback:
                        progress_callback(100, "Layer ready (from cache)")

                    QgsMessageLog.logMessage(
                        f"Loaded cached features for: {layer.name()} (model: {self._model_id})",
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

            # Save to cache (model-specific)
            features_path = get_features_path(layer, self._model_id)
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

    def clear_features(self):
        """Clear cached features (force re-encoding on next prepare_layer)."""
        self._features = None
        self._transform_info = {}
        self._current_layer = None

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
