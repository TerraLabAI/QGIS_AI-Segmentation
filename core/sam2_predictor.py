import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from qgis.core import QgsMessageLog, Qgis, QgsRasterLayer, QgsRectangle

from .model_registry import ModelConfig, get_model_config, get_ultralytics_model_path


class SAM2Predictor:

    def __init__(self, model_config: ModelConfig = None):
        self._model_config = model_config
        self._model = None
        self._current_image = None
        self._transform_info = None
        self._is_loaded = False

    def load_model(self) -> bool:
        if self._model_config is None:
            QgsMessageLog.logMessage(
                "SAM2Predictor: No model config provided",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return False

        if not self._model_config.is_ultralytics:
            QgsMessageLog.logMessage(
                f"SAM2Predictor: Model {self._model_config.model_id} is not an Ultralytics model",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return False

        try:
            from ultralytics import SAM

            model_path = get_ultralytics_model_path(self._model_config.model_id)
            
            if model_path is None:
                model_path = self._model_config.ultralytics_model
                QgsMessageLog.logMessage(
                    f"[SAM2] No local model found, will use name: {model_path}",
                    "AI Segmentation",
                    level=Qgis.Warning
                )
            else:
                QgsMessageLog.logMessage(
                    f"[SAM2] Loading model from local path: {model_path}",
                    "AI Segmentation",
                    level=Qgis.Info
                )

            self._model = SAM(model_path)
            self._is_loaded = True

            QgsMessageLog.logMessage(
                f"[SAM2] Model loaded successfully: {self._model_config.display_name}",
                "AI Segmentation",
                level=Qgis.Success
            )
            return True

        except ImportError as e:
            QgsMessageLog.logMessage(
                f"SAM2Predictor: Ultralytics not installed - {e}",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return False
        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"SAM2Predictor: Failed to load model - {e}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return False

    def is_loaded(self) -> bool:
        return self._is_loaded and self._model is not None

    def set_image(self, image: np.ndarray, transform_info: dict = None):
        self._current_image = image
        self._transform_info = transform_info or {}
        self._transform_info["original_size"] = image.shape[:2]

        QgsMessageLog.logMessage(
            f"SAM2Predictor: Image set with shape {image.shape}",
            "AI Segmentation",
            level=Qgis.Info
        )

    def set_image_from_layer(
        self,
        layer: QgsRasterLayer,
        extent: QgsRectangle = None,
        progress_callback=None
    ) -> bool:
        try:
            from .image_utils import raster_to_numpy

            if progress_callback:
                progress_callback(10, "Reading raster data...")

            image, geo_transform = raster_to_numpy(layer, extent)

            if image is None:
                QgsMessageLog.logMessage(
                    "SAM2Predictor: Failed to read raster layer",
                    "AI Segmentation",
                    level=Qgis.Critical
                )
                return False

            transform_info = {
                "geo_transform": geo_transform,
                "layer_crs": layer.crs().authid(),
                "extent": [
                    extent.xMinimum() if extent else layer.extent().xMinimum(),
                    extent.yMinimum() if extent else layer.extent().yMinimum(),
                    extent.xMaximum() if extent else layer.extent().xMaximum(),
                    extent.yMaximum() if extent else layer.extent().yMaximum(),
                ],
            }

            self.set_image(image, transform_info)

            if progress_callback:
                progress_callback(100, "Image ready for segmentation")

            return True

        except Exception as e:
            QgsMessageLog.logMessage(
                f"SAM2Predictor: Failed to set image from layer - {e}",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return False

    def predict_mask(
        self,
        points: List[Tuple[float, float]],
        labels: List[int],
        return_best: bool = True
    ) -> Tuple[Optional[np.ndarray], float]:
        if not self.is_loaded():
            QgsMessageLog.logMessage(
                "SAM2Predictor: Model not loaded",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return None, 0.0

        if self._current_image is None:
            QgsMessageLog.logMessage(
                "SAM2Predictor: No image set",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return None, 0.0

        if len(points) == 0:
            QgsMessageLog.logMessage(
                "SAM2Predictor: No points provided",
                "AI Segmentation",
                level=Qgis.Warning
            )
            return None, 0.0

        try:
            pixel_points, pixel_labels = self._convert_geo_to_pixel(points, labels)

            QgsMessageLog.logMessage(
                f"SAM2Predictor: Predicting with {len(pixel_points)} points",
                "AI Segmentation",
                level=Qgis.Info
            )

            results = self._model.predict(
                source=self._current_image,
                points=[list(p) for p in pixel_points],
                labels=pixel_labels,
                verbose=False
            )

            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'masks') and result.masks is not None:
                    masks_tensor = result.masks.data

                    if len(masks_tensor) > 0:
                        if return_best:
                            mask_tensor = masks_tensor[0]
                        else:
                            mask_tensor = masks_tensor[0]

                        if hasattr(mask_tensor, 'cpu'):
                            mask = mask_tensor.cpu().numpy()
                        else:
                            mask = np.array(mask_tensor)

                        original_h, original_w = self._transform_info.get("original_size", mask.shape[:2])
                        if mask.shape[0] != original_h or mask.shape[1] != original_w:
                            from .image_utils import resize_image
                            mask = resize_image(mask.astype(np.float32), (original_h, original_w))

                        binary_mask = (mask > 0.5).astype(np.uint8)
                        score = 1.0

                        QgsMessageLog.logMessage(
                            f"SAM2Predictor: Generated mask with {binary_mask.sum()} pixels",
                            "AI Segmentation",
                            level=Qgis.Info
                        )

                        return binary_mask, score

            QgsMessageLog.logMessage(
                "SAM2Predictor: No mask generated",
                "AI Segmentation",
                level=Qgis.Warning
            )
            return None, 0.0

        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"SAM2Predictor: Prediction failed - {e}\n{traceback.format_exc()}",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return None, 0.0

    def _convert_geo_to_pixel(
        self,
        points: List[Tuple[float, float]],
        labels: List[int]
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        from .image_utils import map_point_to_sam_coords

        pixel_points = []
        for x, y in points:
            if self._transform_info:
                pixel_x, pixel_y = map_point_to_sam_coords(
                    x, y, self._transform_info, scale_to_sam_space=False
                )
                pixel_points.append((int(pixel_x), int(pixel_y)))
            else:
                pixel_points.append((int(x), int(y)))

        return pixel_points, labels

    def get_transform_info(self) -> dict:
        return self._transform_info or {}

    def clear(self):
        self._current_image = None
        self._transform_info = None

    def unload(self):
        self._model = None
        self._is_loaded = False
        self.clear()


def is_sam2_available() -> bool:
    try:
        from ultralytics import SAM
        return True
    except ImportError:
        return False
