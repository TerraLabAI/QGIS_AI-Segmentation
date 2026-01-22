

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

from .model_registry import ModelConfig, get_model_config, DEFAULT_MODEL_ID

SAM_INPUT_SIZE = 1024


class SAMEncoder:

    def __init__(self, model_path: str = None, model_config: ModelConfig = None):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_names = []

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
                    f"Encoder model not found: {self.model_path}",
                    "AI Segmentation",
                    level=Qgis.Critical
                )
                return False

            from .execution_provider import get_optimal_providers, get_active_provider_name

            providers = get_optimal_providers()
            self.session = ort.InferenceSession(
                self.model_path,
                providers=providers
            )

            active_provider = get_active_provider_name(self.session)
            QgsMessageLog.logMessage(
                f"Encoder loaded with provider: {active_provider}",
                "AI Segmentation",
                level=Qgis.Info
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]

            QgsMessageLog.logMessage(
                f"Encoder model loaded: {self.model_path}",
                "AI Segmentation",
                level=Qgis.Info
            )
            QgsMessageLog.logMessage(
                f"Encoder outputs: {self.output_names}",
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
        from .image_utils import resize_image

        original_size = image.shape[:2]
        h, w = original_size

        if self._model_config.use_sam2_preprocessing:
            resized = resize_image(image, (SAM_INPUT_SIZE, SAM_INPUT_SIZE))

            normalized = resized / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (normalized - mean) / std

            preprocessed = normalized.transpose(2, 0, 1)[np.newaxis, ...]

            transform_info = {
                "original_size": original_size,
                "resized_size": (SAM_INPUT_SIZE, SAM_INPUT_SIZE),
                "scale": SAM_INPUT_SIZE / max(h, w),
                "scale_x": SAM_INPUT_SIZE / w,
                "scale_y": SAM_INPUT_SIZE / h,
                "input_size": SAM_INPUT_SIZE,
                "is_sam2": True,
            }
        else:
            scale = SAM_INPUT_SIZE / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)

            resized = resize_image(image, (new_h, new_w))

            padded = np.zeros((SAM_INPUT_SIZE, SAM_INPUT_SIZE, 3), dtype=np.float32)
            padded[:new_h, :new_w, :] = resized

            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            normalized = (padded - mean) / std

            preprocessed = normalized.transpose(2, 0, 1)[np.newaxis, ...]

            transform_info = {
                "original_size": original_size,
                "resized_size": (new_h, new_w),
                "scale": scale,
                "input_size": SAM_INPUT_SIZE,
                "is_sam2": False,
            }

        return preprocessed.astype(np.float32), transform_info

    def encode(
        self,
        image: np.ndarray,
        progress_callback: Callable[[int, str], None] = None
    ) -> Tuple[Optional[dict], dict]:
        import time
        from .debug_settings import get_settings
        settings = get_settings()

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

            preprocess_start = time.time()
            preprocessed, transform_info = self.preprocess_image(image)
            preprocess_time = time.time() - preprocess_start

            if settings.show_timing_info:
                QgsMessageLog.logMessage(
                    f"[TIMING] Preprocessing: {preprocess_time:.2f}s",
                    "AI Segmentation",
                    level=Qgis.Info
                )

            if progress_callback:
                progress_callback(30, "Running encoder (this may take a while)...")

            encode_start = time.time()
            outputs = self.session.run(self.output_names, {self.input_name: preprocessed})
            encode_time = time.time() - encode_start

            if settings.show_timing_info:
                QgsMessageLog.logMessage(
                    f"[TIMING] Encoding: {encode_time:.2f}s",
                    "AI Segmentation",
                    level=Qgis.Info
                )

            features_dict = {}
            for i, name in enumerate(self.output_names):
                features_dict[name] = outputs[i]
                QgsMessageLog.logMessage(
                    f"Encoder output[{i}] '{name}': shape={outputs[i].shape}",
                    "AI Segmentation",
                    level=Qgis.Info
                )

            if self._model_config.use_sam2_preprocessing and len(outputs) >= 3:
                transform_info["_output_order"] = self.output_names.copy()
                QgsMessageLog.logMessage(
                    f"SAM2 encoder output order: {self.output_names}",
                    "AI Segmentation",
                    level=Qgis.Info
                )

            if progress_callback:
                progress_callback(100, "Encoding complete!")

            return features_dict, transform_info

        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Encoding failed: {str(e)}\n{traceback.format_exc()}",
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
        
        if progress_callback:
            progress_callback(0, "Reading raster data...")

        try:
            from .image_utils import raster_to_numpy
            image, geo_transform = raster_to_numpy(layer, extent)

            if image is None:
                return None, {}

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
    features: dict,
    transform_info: dict,
    output_path: str
) -> bool:
    
    try:
        np.savez(f"{output_path}.npz", **features)

        with open(f"{output_path}.json", 'w') as f:
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
            f"Features saved to: {output_path} (keys: {list(features.keys())})",
            "AI Segmentation",
            level=Qgis.Info
        )
        return True

    except Exception as e:
        import traceback
        QgsMessageLog.logMessage(
            f"Failed to save features: {str(e)}\n{traceback.format_exc()}",
            "AI Segmentation",
            level=Qgis.Critical
        )
        return False


def load_features(features_path: str) -> Tuple[Optional[dict], dict]:
    
    try:
        npz_path = f"{features_path}.npz"
        npy_path = f"{features_path}.npy"

        if os.path.exists(npz_path):
            loaded = np.load(npz_path)
            features = {key: loaded[key] for key in loaded.files}
            for key, arr in features.items():
                QgsMessageLog.logMessage(
                    f"[CACHE] Loaded {key}: shape={arr.shape}, dtype={arr.dtype}, "
                    f"min={arr.min():.4f}, max={arr.max():.4f}",
                    "AI Segmentation",
                    level=Qgis.Info
                )
        elif os.path.exists(npy_path):
            single_features = np.load(npy_path)
            features = {"image_embeddings": single_features}
            QgsMessageLog.logMessage(
                f"Loaded features from legacy npy format",
                "AI Segmentation",
                level=Qgis.Info
            )
        else:
            QgsMessageLog.logMessage(
                f"No features file found at: {features_path}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            return None, {}

        with open(f"{features_path}.json", 'r') as f:
            transform_info = json.load(f)

        return features, transform_info

    except Exception as e:
        import traceback
        QgsMessageLog.logMessage(
            f"Failed to load features: {str(e)}\n{traceback.format_exc()}",
            "AI Segmentation",
            level=Qgis.Critical
        )
        return None, {}


def _is_file_based_layer(layer: QgsRasterLayer) -> bool:
    
    source = layer.source()

    web_indicators = ['http://', 'https://', 'type=xyz', 'type=wms', 'url=']
    for indicator in web_indicators:
        if indicator in source.lower():
            return False

    return os.path.isfile(source)


def get_features_path(layer: QgsRasterLayer, model_id: str = None) -> str:
    
    from .model_registry import DEFAULT_MODEL_ID

    if model_id is None:
        model_id = DEFAULT_MODEL_ID

    source = layer.source()

    if _is_file_based_layer(layer):
        source_path = Path(source)
        cache_dir = source_path.parent / ".ai_segmentation_cache" / model_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir / source_path.stem)
    else:
        extent = layer.extent()
        cache_key = f"{source}_{extent.toString()}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]

        cache_base = Path.home() / ".qgis_ai_segmentation_cache"
        cache_dir = cache_base / cache_hash / model_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        layer_name = layer.name().replace(" ", "_").replace("/", "_")
        return str(cache_dir / layer_name)


def has_cached_features(layer: QgsRasterLayer, model_id: str = None) -> bool:
    
    from .model_registry import get_model_config, DEFAULT_MODEL_ID

    if model_id is None:
        model_id = DEFAULT_MODEL_ID

    config = get_model_config(model_id)
    features_path = get_features_path(layer, model_id)
    json_exists = os.path.exists(f"{features_path}.json")

    if config.is_sam2:
        npz_exists = os.path.exists(f"{features_path}.npz")
        return json_exists and npz_exists
    else:
        npz_exists = os.path.exists(f"{features_path}.npz")
        npy_exists = os.path.exists(f"{features_path}.npy")
        return json_exists and (npz_exists or npy_exists)
