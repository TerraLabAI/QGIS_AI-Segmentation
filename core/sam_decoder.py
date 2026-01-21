"""
SAM Mask Decoder for AI Segmentation

Handles real-time mask prediction from point prompts using pre-encoded features.
This is the lightweight component that enables interactive segmentation on CPU.
"""

import os
from typing import List, Tuple, Optional
import numpy as np

from qgis.core import QgsMessageLog, Qgis

# SAM constants
SAM_INPUT_SIZE = 1024
MASK_THRESHOLD = 0.0  # Threshold for binary mask


class SAMDecoder:
    """
    SAM Mask Decoder using ONNX Runtime.

    Takes pre-encoded image features and point prompts to generate
    segmentation masks in real-time.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the SAM decoder.

        Args:
            model_path: Path to the ONNX decoder model file
        """
        self.model_path = model_path
        self.session = None
        self.input_names = None
        self.output_names = None

    def load_model(self, model_path: str = None) -> bool:
        """
        Load the ONNX decoder model.

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
                    f"Decoder model not found: {self.model_path}",
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
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]

            QgsMessageLog.logMessage(
                f"Decoder model loaded. Inputs: {self.input_names}",
                "AI Segmentation",
                level=Qgis.Info
            )
            return True

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to load decoder model: {str(e)}",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return False

    def prepare_prompts(
        self,
        points: List[Tuple[float, float]],
        labels: List[int],
        transform_info: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare point prompts for SAM decoder.

        Args:
            points: List of (x, y) points in map coordinates
            labels: List of labels (1 for positive/foreground, 0 for negative/background)
            transform_info: Transform info from encoding

        Returns:
            Tuple of (point_coords, point_labels) arrays ready for decoder
        """
        from .image_utils import map_point_to_sam_coords

        # Convert map coordinates to SAM coordinates
        sam_points = []
        for x, y in points:
            sam_x, sam_y = map_point_to_sam_coords(x, y, transform_info)
            sam_points.append([sam_x, sam_y])

        # Create arrays in the format SAM expects
        # Shape: (1, N, 2) for points, (1, N) for labels
        point_coords = np.array(sam_points, dtype=np.float32).reshape(1, -1, 2)
        point_labels = np.array(labels, dtype=np.float32).reshape(1, -1)

        return point_coords, point_labels

    def decode(
        self,
        features: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        transform_info: dict,
        multimask_output: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Decode features with point prompts to generate masks.

        Args:
            features: Pre-encoded image features
            point_coords: Point coordinates array (1, N, 2)
            point_labels: Point labels array (1, N)
            transform_info: Transform info from encoding
            multimask_output: If True, return multiple mask candidates

        Returns:
            Tuple of (masks, scores) or (None, None) on error
            masks shape: (1, num_masks, H, W) where H, W are original image size
            scores shape: (1, num_masks) confidence scores for each mask
        """
        if self.session is None:
            QgsMessageLog.logMessage(
                "Decoder model not loaded",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return None, None

        try:
            original_size = transform_info["original_size"]

            # Prepare inputs based on model requirements
            # Standard SAM decoder inputs
            inputs = self._prepare_decoder_inputs(
                features,
                point_coords,
                point_labels,
                original_size,
                multimask_output
            )

            # Validate input shapes before inference
            if not self._validate_input_shapes(inputs):
                return None, None

            # Run inference
            outputs = self.session.run(self.output_names, inputs)

            # Parse outputs (typically masks and iou_predictions)
            masks = outputs[0]  # (1, num_masks, H, W)
            scores = outputs[1] if len(outputs) > 1 else None  # (1, num_masks)

            QgsMessageLog.logMessage(
                f"Decoder raw output: masks shape={masks.shape}, original_size={original_size}",
                "AI Segmentation",
                level=Qgis.Info
            )

            # Check if masks need to be upsampled to original size
            # SAM decoder often outputs at 256x256 which needs resizing
            mask_h, mask_w = masks.shape[-2:]
            orig_h, orig_w = original_size

            if mask_h != orig_h or mask_w != orig_w:
                QgsMessageLog.logMessage(
                    f"Upsampling masks from ({mask_h}, {mask_w}) to ({orig_h}, {orig_w})",
                    "AI Segmentation",
                    level=Qgis.Info
                )
                masks = self._upsample_masks(masks, (orig_h, orig_w))

            return masks, scores

        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Decoding failed: {str(e)}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return None, None

    def _upsample_masks(
        self,
        masks: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Upsample masks to target size using bilinear interpolation.

        Args:
            masks: Input masks (1, num_masks, H, W)
            target_size: Target (height, width)

        Returns:
            Upsampled masks (1, num_masks, target_H, target_W)
        """
        from .image_utils import resize_image

        batch, num_masks, src_h, src_w = masks.shape
        tgt_h, tgt_w = target_size

        # Resize each mask
        upsampled = np.zeros((batch, num_masks, tgt_h, tgt_w), dtype=masks.dtype)

        for b in range(batch):
            for m in range(num_masks):
                mask_2d = masks[b, m]
                # Use resize_image for upsampling
                upsampled[b, m] = resize_image(mask_2d, target_size)

        return upsampled

    def _prepare_decoder_inputs(
        self,
        features: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        original_size: Tuple[int, int],
        multimask_output: bool
    ) -> dict:
        """
        Prepare inputs for the decoder model using exact name mapping.

        Uses a dictionary with exact input names to avoid substring matching
        issues (e.g., "mask_input" matching "has_mask_input").

        Args:
            features: Image embeddings from encoder
            point_coords: Point coordinates array (1, N, 2)
            point_labels: Point labels array (1, N)
            original_size: Original image size (H, W)
            multimask_output: Whether to output multiple masks

        Returns:
            Dictionary of inputs for the ONNX session
        """
        # Exact input name mapping for SAM ViT-B ONNX model
        # Source: HuggingFace visheratin/segment-anything-vit-b
        INPUT_VALUES = {
            "image_embeddings": features,                                    # (1, 256, 64, 64)
            "point_coords": point_coords,                                    # (1, N, 2)
            "point_labels": point_labels,                                    # (1, N)
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),      # (1, 1, 256, 256)
            "has_mask_input": np.array([0], dtype=np.float32),               # (1,) - rank 1!
            "orig_im_size": np.array(original_size, dtype=np.float32),       # (2,)
        }

        inputs = {}
        for name in self.input_names:
            if name in INPUT_VALUES:
                inputs[name] = INPUT_VALUES[name]
            else:
                QgsMessageLog.logMessage(
                    f"Unknown decoder input: {name}",
                    "AI Segmentation",
                    level=Qgis.Warning
                )

        return inputs

    def _validate_input_shapes(self, inputs: dict) -> bool:
        """
        Validate input tensor shapes before ONNX inference.

        Catches shape mismatches early with clear error messages instead of
        cryptic ONNX runtime errors.

        Args:
            inputs: Dictionary of input name to numpy array

        Returns:
            True if all shapes are valid, False otherwise
        """
        EXPECTED_RANKS = {
            "image_embeddings": 4,   # (1, 256, 64, 64)
            "point_coords": 3,       # (1, N, 2)
            "point_labels": 2,       # (1, N)
            "mask_input": 4,         # (1, 1, 256, 256)
            "has_mask_input": 1,     # (1,)
            "orig_im_size": 1,       # (2,)
        }

        for name, tensor in inputs.items():
            actual_rank = len(tensor.shape)
            expected_rank = EXPECTED_RANKS.get(name)

            if expected_rank is not None and actual_rank != expected_rank:
                QgsMessageLog.logMessage(
                    f"Input shape mismatch for '{name}': "
                    f"got rank {actual_rank} (shape {tensor.shape}), "
                    f"expected rank {expected_rank}",
                    "AI Segmentation",
                    level=Qgis.Critical
                )
                return False

        return True

    def predict_mask(
        self,
        features: np.ndarray,
        points: List[Tuple[float, float]],
        labels: List[int],
        transform_info: dict,
        return_best: bool = True
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        High-level method to predict a mask from points.

        Args:
            features: Pre-encoded image features
            points: List of (x, y) points in map coordinates
            labels: List of labels (1=positive, 0=negative)
            transform_info: Transform info from encoding
            return_best: If True, return only the best mask

        Returns:
            Tuple of (mask, score) where mask is binary (H, W) array
            Returns (None, 0.0) on error
        """
        if len(points) == 0:
            QgsMessageLog.logMessage(
                "predict_mask: No points provided",
                "AI Segmentation",
                level=Qgis.Warning
            )
            return None, 0.0

        # Count positive and negative points
        pos_count = sum(1 for l in labels if l == 1)
        neg_count = sum(1 for l in labels if l == 0)

        QgsMessageLog.logMessage(
            "═══════════════════════════════════════════════════════════",
            "AI Segmentation",
            level=Qgis.Info
        )
        QgsMessageLog.logMessage(
            f"SEGMENTATION REQUEST: {len(points)} points ({pos_count} positive ✓, {neg_count} negative ✗)",
            "AI Segmentation",
            level=Qgis.Info
        )

        # Log each point with its type
        for i, ((x, y), label) in enumerate(zip(points, labels)):
            point_type = "POSITIVE (include)" if label == 1 else "NEGATIVE (exclude)"
            QgsMessageLog.logMessage(
                f"  Point {i+1}: ({x:.2f}, {y:.2f}) → {point_type}",
                "AI Segmentation",
                level=Qgis.Info
            )

        # Prepare prompts
        point_coords, point_labels = self.prepare_prompts(
            points, labels, transform_info
        )

        QgsMessageLog.logMessage(
            f"Converted to SAM coords (0-1024 space):",
            "AI Segmentation",
            level=Qgis.Info
        )
        for i, (coord, label) in enumerate(zip(point_coords[0], point_labels[0])):
            point_type = "+" if label == 1 else "-"
            QgsMessageLog.logMessage(
                f"  [{point_type}] SAM({coord[0]:.1f}, {coord[1]:.1f})",
                "AI Segmentation",
                level=Qgis.Info
            )

        # Decode
        masks, scores = self.decode(
            features,
            point_coords,
            point_labels,
            transform_info,
            multimask_output=not return_best
        )

        if masks is None:
            QgsMessageLog.logMessage(
                "predict_mask: decode returned None - FAILED",
                "AI Segmentation",
                level=Qgis.Warning
            )
            return None, 0.0

        # Log all candidate masks and their scores
        QgsMessageLog.logMessage(
            f"SAM returned {masks.shape[1]} mask candidates:",
            "AI Segmentation",
            level=Qgis.Info
        )
        for i in range(masks.shape[1]):
            mask_pixels = (masks[0, i] > MASK_THRESHOLD).sum()
            score_val = scores[0, i] if scores is not None else 0.0
            QgsMessageLog.logMessage(
                f"  Mask {i+1}: score={score_val:.3f}, pixels={mask_pixels}",
                "AI Segmentation",
                level=Qgis.Info
            )

        if return_best and scores is not None:
            # Select best mask based on score
            best_idx = np.argmax(scores[0])
            mask = masks[0, best_idx]
            score = float(scores[0, best_idx])
            QgsMessageLog.logMessage(
                f"Selected BEST mask: index={best_idx+1}, score={score:.3f}",
                "AI Segmentation",
                level=Qgis.Info
            )
        else:
            # Return first mask
            mask = masks[0, 0]
            score = float(scores[0, 0]) if scores is not None else 1.0

        # Apply threshold to get binary mask
        binary_mask = (mask > MASK_THRESHOLD).astype(np.uint8)
        mask_pixels = int(binary_mask.sum())

        QgsMessageLog.logMessage(
            f"RESULT: Binary mask {binary_mask.shape[1]}x{binary_mask.shape[0]}, "
            f"{mask_pixels} pixels, score={score:.3f}",
            "AI Segmentation",
            level=Qgis.Info
        )
        QgsMessageLog.logMessage(
            "═══════════════════════════════════════════════════════════",
            "AI Segmentation",
            level=Qgis.Info
        )

        return binary_mask, score


class PromptManager:
    """
    Manages point prompts for interactive segmentation.

    Keeps track of positive and negative points, and provides
    methods for adding, removing, and clearing points.

    Note on SAM point semantics:
    - POSITIVE points (label=1): "This pixel IS part of the object I want"
    - NEGATIVE points (label=0): "This pixel is NOT part of the object"

    Negative points help refine boundaries by telling SAM what to exclude.
    They work best when placed near the boundary of an incorrectly included region.
    """

    def __init__(self):
        """Initialize the prompt manager."""
        self.positive_points: List[Tuple[float, float]] = []
        self.negative_points: List[Tuple[float, float]] = []
        # Track order of all point additions for proper LIFO undo
        self.prompt_history: List[str] = []  # "positive" or "negative"

    def add_positive(self, x: float, y: float):
        """Add a positive (foreground) point."""
        self.positive_points.append((x, y))
        self.prompt_history.append("positive")
        QgsMessageLog.logMessage(
            f"[PromptManager] Added POSITIVE point #{len(self.positive_points)} at ({x:.2f}, {y:.2f})",
            "AI Segmentation",
            level=Qgis.Info
        )
        self._log_state()

    def add_negative(self, x: float, y: float):
        """Add a negative (background) point."""
        self.negative_points.append((x, y))
        self.prompt_history.append("negative")
        QgsMessageLog.logMessage(
            f"[PromptManager] Added NEGATIVE point #{len(self.negative_points)} at ({x:.2f}, {y:.2f})",
            "AI Segmentation",
            level=Qgis.Info
        )
        self._log_state()

    def undo(self) -> bool:
        """
        Remove the last added point (either positive or negative).

        Uses prompt_history to properly implement LIFO (Last-In-First-Out)
        behavior, removing the most recently added point regardless of type.

        Returns:
            True if a point was removed
        """
        if not self.prompt_history:
            QgsMessageLog.logMessage(
                "[PromptManager] Undo: No points to undo",
                "AI Segmentation",
                level=Qgis.Info
            )
            return False

        last_type = self.prompt_history.pop()
        if last_type == "positive" and self.positive_points:
            removed = self.positive_points.pop()
            QgsMessageLog.logMessage(
                f"[PromptManager] Undo: Removed POSITIVE point at ({removed[0]:.2f}, {removed[1]:.2f})",
                "AI Segmentation",
                level=Qgis.Info
            )
            self._log_state()
            return True
        elif last_type == "negative" and self.negative_points:
            removed = self.negative_points.pop()
            QgsMessageLog.logMessage(
                f"[PromptManager] Undo: Removed NEGATIVE point at ({removed[0]:.2f}, {removed[1]:.2f})",
                "AI Segmentation",
                level=Qgis.Info
            )
            self._log_state()
            return True
        return False

    def clear(self):
        """Clear all points."""
        pos_count = len(self.positive_points)
        neg_count = len(self.negative_points)
        self.positive_points.clear()
        self.negative_points.clear()
        self.prompt_history.clear()
        QgsMessageLog.logMessage(
            f"[PromptManager] Cleared all points ({pos_count} positive, {neg_count} negative)",
            "AI Segmentation",
            level=Qgis.Info
        )

    def get_all_points(self) -> Tuple[List[Tuple[float, float]], List[int]]:
        """
        Get all points with their labels.

        Returns:
            Tuple of (points, labels) where labels are 1 for positive, 0 for negative

        Note: Points are returned with all positive points first, then all negative.
        This is the expected format for SAM - the order within each group doesn't matter.
        """
        points = self.positive_points + self.negative_points
        labels = [1] * len(self.positive_points) + [0] * len(self.negative_points)
        return points, labels

    def has_points(self) -> bool:
        """Check if there are any points."""
        return len(self.positive_points) > 0 or len(self.negative_points) > 0

    @property
    def point_count(self) -> Tuple[int, int]:
        """Get count of (positive, negative) points."""
        return len(self.positive_points), len(self.negative_points)

    def _log_state(self):
        """Log current prompt state for debugging."""
        QgsMessageLog.logMessage(
            f"[PromptManager] Current state: {len(self.positive_points)} positive, "
            f"{len(self.negative_points)} negative points",
            "AI Segmentation",
            level=Qgis.Info
        )
