

import os
from typing import List, Tuple, Optional
import numpy as np

from qgis.core import QgsMessageLog, Qgis

from .model_registry import ModelConfig, get_model_config, DEFAULT_MODEL_ID

SAM_INPUT_SIZE = 1024


class SAMDecoder:
    

    def __init__(self, model_path: str = None, model_config: ModelConfig = None):
        
        self.model_path = model_path
        self.session = None
        self.input_names = None
        self.output_names = None

        if model_config is None:
            model_config = get_model_config(DEFAULT_MODEL_ID)
        self._model_config = model_config

    def load_model(self, model_path: str = None) -> bool:
        
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

            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )

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
        from .image_utils import map_point_to_sam_coords

        original_size = transform_info["original_size"]
        input_size = transform_info.get("input_size", SAM_INPUT_SIZE)
        use_sam2_transform = self._model_config.use_sam2_coord_transform
        extent = transform_info.get("extent", [0, 0, 1, 1])

        QgsMessageLog.logMessage(
            f"[COORD] Model: {self._model_config.model_id}, SAM2 transform: {use_sam2_transform}",
            "AI Segmentation",
            level=Qgis.Info
        )
        QgsMessageLog.logMessage(
            f"[COORD] original_size (H,W): {original_size}, input_size: {input_size}",
            "AI Segmentation",
            level=Qgis.Info
        )
        QgsMessageLog.logMessage(
            f"[COORD] extent [xmin,ymin,xmax,ymax]: {extent}",
            "AI Segmentation",
            level=Qgis.Info
        )

        sam_points = []
        for i, (x, y) in enumerate(points):
            pixel_x, pixel_y = map_point_to_sam_coords(
                x, y, transform_info,
                scale_to_sam_space=False
            )

            if use_sam2_transform:
                sam_x = pixel_x / original_size[1] * input_size
                sam_y = pixel_y / original_size[0] * input_size
            else:
                scale = transform_info["scale"]
                sam_x = pixel_x * scale
                sam_y = pixel_y * scale

            QgsMessageLog.logMessage(
                f"[COORD] Point {i}: geo({x:.2f},{y:.2f}) -> pixel({pixel_x:.2f},{pixel_y:.2f}) -> sam({sam_x:.2f},{sam_y:.2f})",
                "AI Segmentation",
                level=Qgis.Info
            )

            sam_points.append([sam_x, sam_y])

        point_coords = np.array(sam_points, dtype=np.float32).reshape(1, -1, 2)
        point_labels = np.array(labels, dtype=np.float32).reshape(1, -1)

        return point_coords, point_labels

    def decode(
        self,
        features: dict,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        transform_info: dict,
        multimask_output: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        
        if self.session is None:
            QgsMessageLog.logMessage(
                "Decoder model not loaded",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return None, None

        try:
            original_size = transform_info["original_size"]

            inputs = self._prepare_decoder_inputs(
                features,
                point_coords,
                point_labels,
                original_size,
                multimask_output,
                transform_info
            )

            if not self._validate_input_shapes(inputs):
                QgsMessageLog.logMessage(
                    f"[DECODER DEBUG] Input validation FAILED!",
                    "AI Segmentation",
                    level=Qgis.Critical
                )
                return None, None

            QgsMessageLog.logMessage(
                f"[DECODER DEBUG] Running ONNX inference with outputs: {self.output_names}",
                "AI Segmentation",
                level=Qgis.Info
            )
            outputs = self.session.run(self.output_names, inputs)
            QgsMessageLog.logMessage(
                f"[DECODER DEBUG] ONNX inference completed! Got {len(outputs)} outputs",
                "AI Segmentation",
                level=Qgis.Info
            )

            masks = outputs[0]  
            scores = outputs[1] if len(outputs) > 1 else None  

            QgsMessageLog.logMessage(
                f"Decoder raw output: masks shape={masks.shape}, original_size={original_size}",
                "AI Segmentation",
                level=Qgis.Info
            )

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
        
        from .image_utils import resize_image

        batch, num_masks, src_h, src_w = masks.shape
        tgt_h, tgt_w = target_size

        upsampled = np.zeros((batch, num_masks, tgt_h, tgt_w), dtype=masks.dtype)

        for b in range(batch):
            for m in range(num_masks):
                mask_2d = masks[b, m]
                upsampled[b, m] = resize_image(mask_2d, target_size)

        return upsampled

    def _prepare_decoder_inputs(
        self,
        features: dict,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        original_size: Tuple[int, int],
        multimask_output: bool,
        transform_info: dict = None
    ) -> dict:
        mask_shape = self._model_config.mask_input_shape
        is_sam2 = self._model_config.is_sam2

        if is_sam2 and self._model_config.use_sam2_coord_transform:
            return self._prepare_sam2_inputs_by_order(
                features, point_coords, point_labels, original_size, transform_info
            )

        tensor_names = self._model_config.decoder_input_names

        LOGICAL_VALUES = {}
        for key, value in features.items():
            LOGICAL_VALUES[key] = value

        LOGICAL_VALUES.update({
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": np.zeros(mask_shape, dtype=np.float32),
            "has_mask_input": np.array([0], dtype=np.float32),
        })

        if "orig_im_size" in tensor_names:
            LOGICAL_VALUES["orig_im_size"] = np.array(original_size, dtype=np.float32)

        inputs = {}
        for name in self.input_names:
            matched = False
            for logical_name, onnx_name in tensor_names.items():
                if onnx_name == name:
                    if logical_name in LOGICAL_VALUES:
                        inputs[name] = LOGICAL_VALUES[logical_name]
                        matched = True
                    break

            if not matched and name in LOGICAL_VALUES:
                inputs[name] = LOGICAL_VALUES[name]
                matched = True

            if not matched:
                QgsMessageLog.logMessage(
                    f"Unknown/missing decoder input: {name}",
                    "AI Segmentation",
                    level=Qgis.Warning
                )

        return inputs

    def _prepare_sam2_inputs_by_order(
        self,
        features: dict,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        original_size: Tuple[int, int],
        transform_info: dict = None
    ) -> dict:
        input_size = self._model_config.input_size
        scale_factor = 4
        mask_h = input_size // scale_factor
        mask_w = input_size // scale_factor
        num_labels = point_labels.shape[0]
        mask_input = np.zeros((num_labels, 1, mask_h, mask_w), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)

        output_order = []
        if transform_info:
            output_order = transform_info.get("_output_order", [])

        if len(output_order) >= 3:
            high_res_feats_0 = features.get(output_order[0])
            high_res_feats_1 = features.get(output_order[1])
            image_embed = features.get(output_order[2])
            QgsMessageLog.logMessage(
                f"[SAM2] Using output order: {output_order}",
                "AI Segmentation",
                level=Qgis.Info
            )
        else:
            image_embed = None
            high_res_feats_0 = None
            high_res_feats_1 = None
            for key, value in features.items():
                if key == "_output_order":
                    continue
                if "image" in key.lower() or "embed" in key.lower():
                    image_embed = value
                elif "high_res" in key.lower() or "feat" in key.lower():
                    if high_res_feats_0 is None:
                        high_res_feats_0 = value
                    else:
                        high_res_feats_1 = value

            if image_embed is None:
                feature_keys = [k for k in features.keys() if k != "_output_order"]
                if len(feature_keys) >= 3:
                    high_res_feats_0 = features.get(feature_keys[0])
                    high_res_feats_1 = features.get(feature_keys[1])
                    image_embed = features.get(feature_keys[2])

        QgsMessageLog.logMessage(
            f"[SAM2] image_embed shape: {image_embed.shape if image_embed is not None else 'None'}",
            "AI Segmentation",
            level=Qgis.Info
        )
        QgsMessageLog.logMessage(
            f"[SAM2] high_res_feats_0 shape: {high_res_feats_0.shape if high_res_feats_0 is not None else 'None'}",
            "AI Segmentation",
            level=Qgis.Info
        )
        QgsMessageLog.logMessage(
            f"[SAM2] high_res_feats_1 shape: {high_res_feats_1.shape if high_res_feats_1 is not None else 'None'}",
            "AI Segmentation",
            level=Qgis.Info
        )

        ordered_values = [
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            point_coords,
            point_labels,
            mask_input,
            has_mask_input,
        ]

        inputs = {}
        for i, name in enumerate(self.input_names):
            if i < len(ordered_values) and ordered_values[i] is not None:
                inputs[name] = ordered_values[i]
                QgsMessageLog.logMessage(
                    f"[SAM2] Decoder input[{i}] '{name}': shape={ordered_values[i].shape}",
                    "AI Segmentation",
                    level=Qgis.Info
                )
            else:
                QgsMessageLog.logMessage(
                    f"[SAM2] Decoder missing input {i} ({name})",
                    "AI Segmentation",
                    level=Qgis.Warning
                )

        QgsMessageLog.logMessage(
            f"[SAM2 DECODER] point_coords: {point_coords}",
            "AI Segmentation",
            level=Qgis.Info
        )

        return inputs

    def _validate_input_shapes(self, inputs: dict) -> bool:
        
        expected_ranks = self._model_config.decoder_input_ranks
        tensor_names = self._model_config.decoder_input_names

        onnx_to_logical = {v: k for k, v in tensor_names.items()}

        for name, tensor in inputs.items():
            actual_rank = len(tensor.shape)

            logical_name = onnx_to_logical.get(name, name)
            expected_rank = expected_ranks.get(logical_name)

            if expected_rank is not None and actual_rank != expected_rank:
                QgsMessageLog.logMessage(
                    f"Input shape mismatch for '{name}' (logical: {logical_name}): "
                    f"got rank {actual_rank} (shape {tensor.shape}), "
                    f"expected rank {expected_rank}",
                    "AI Segmentation",
                    level=Qgis.Critical
                )
                return False

        return True

    def predict_mask(
        self,
        features: dict,
        points: List[Tuple[float, float]],
        labels: List[int],
        transform_info: dict,
        return_best: bool = True
    ) -> Tuple[Optional[np.ndarray], float]:
        
        if len(points) == 0:
            QgsMessageLog.logMessage(
                "predict_mask: No points provided",
                "AI Segmentation",
                level=Qgis.Warning
            )
            return None, 0.0

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

        for i, ((x, y), label) in enumerate(zip(points, labels)):
            point_type = "POSITIVE (include)" if label == 1 else "NEGATIVE (exclude)"
            QgsMessageLog.logMessage(
                f"  Point {i+1}: ({x:.2f}, {y:.2f}) → {point_type}",
                "AI Segmentation",
                level=Qgis.Info
            )

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

        mask_threshold = self._model_config.mask_threshold

        QgsMessageLog.logMessage(
            f"SAM returned {masks.shape[1]} mask candidates (threshold={mask_threshold}):",
            "AI Segmentation",
            level=Qgis.Info
        )
        for i in range(masks.shape[1]):
            mask_pixels = (masks[0, i] > mask_threshold).sum()
            score_val = scores[0, i] if scores is not None else 0.0
            QgsMessageLog.logMessage(
                f"  Mask {i+1}: score={score_val:.3f}, pixels={mask_pixels}",
                "AI Segmentation",
                level=Qgis.Info
            )

        if return_best and scores is not None:
            best_idx = np.argmax(scores[0])
            mask = masks[0, best_idx]
            score = float(scores[0, best_idx])
            QgsMessageLog.logMessage(
                f"Selected BEST mask: index={best_idx+1}, score={score:.3f}",
                "AI Segmentation",
                level=Qgis.Info
            )
        else:
            mask = masks[0, 0]
            score = float(scores[0, 0]) if scores is not None else 1.0

        binary_mask = (mask > mask_threshold).astype(np.uint8)
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
    

    def __init__(self):
        
        self.positive_points: List[Tuple[float, float]] = []
        self.negative_points: List[Tuple[float, float]] = []
        self.prompt_history: List[str] = []  

    def add_positive(self, x: float, y: float):
        
        self.positive_points.append((x, y))
        self.prompt_history.append("positive")
        QgsMessageLog.logMessage(
            f"[PromptManager] Added POSITIVE point #{len(self.positive_points)} at ({x:.2f}, {y:.2f})",
            "AI Segmentation",
            level=Qgis.Info
        )
        self._log_state()

    def add_negative(self, x: float, y: float):
        
        self.negative_points.append((x, y))
        self.prompt_history.append("negative")
        QgsMessageLog.logMessage(
            f"[PromptManager] Added NEGATIVE point #{len(self.negative_points)} at ({x:.2f}, {y:.2f})",
            "AI Segmentation",
            level=Qgis.Info
        )
        self._log_state()

    def undo(self) -> bool:
        
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
        
        points = self.positive_points + self.negative_points
        labels = [1] * len(self.positive_points) + [0] * len(self.negative_points)
        return points, labels

    def has_points(self) -> bool:
        
        return len(self.positive_points) > 0 or len(self.negative_points) > 0

    @property
    def point_count(self) -> Tuple[int, int]:
        
        return len(self.positive_points), len(self.negative_points)

    def _log_state(self):
        
        QgsMessageLog.logMessage(
            f"[PromptManager] Current state: {len(self.positive_points)} positive, "
            f"{len(self.negative_points)} negative points",
            "AI Segmentation",
            level=Qgis.Info
        )
