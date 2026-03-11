"""CloudSam3Predictor - same interface as CloudPredictor but calls SAM3 server."""
import json
import base64
import urllib.request
import urllib.error
from typing import Tuple, Optional

import numpy as np

from qgis.core import QgsMessageLog, Qgis

from .model_config import SAM3_CLOUD_URL

_TIMEOUT_HEALTH = 300
_TIMEOUT_SET_IMAGE = 120
_TIMEOUT_PREDICT = 60
_TIMEOUT_RESET = 10


class CloudSam3Predictor:

    def __init__(self, hf_token: str = "", api_key: str = "") -> None:
        self.is_image_set = False
        self.original_size = None
        self.input_size = None
        self._session_id = None
        self._hf_token = hf_token
        self._api_key = api_key

    def _request(self, method, path, data=None, timeout=30):
        url = "{}{}".format(SAM3_CLOUD_URL, path)
        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")
        headers = {}
        if body:
            headers["Content-Type"] = "application/json"
        if self._api_key:
            headers["X-Api-Key"] = self._api_key
        if self._hf_token:
            headers["X-HF-Token"] = self._hf_token
        req = urllib.request.Request(
            url, data=body, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(
                "SAM 3 server error {}: {}".format(e.code, detail)
            )
        except urllib.error.URLError as e:
            raise RuntimeError(
                "SAM 3 server unreachable: {}".format(e.reason)
            )

    def warm_up(self) -> bool:
        try:
            resp = self._request("GET", "/health", timeout=_TIMEOUT_HEALTH)
            if resp.get("status") == "ok":
                QgsMessageLog.logMessage(
                    "SAM 3 server connected (device: {})".format(
                        resp.get("device", "unknown")),
                    "AI Segmentation", level=Qgis.Info
                )
                return True
            raise RuntimeError(
                "Unexpected health response: {}".format(resp)
            )
        except Exception as e:
            QgsMessageLog.logMessage(
                "SAM 3 warm_up failed: {}".format(e),
                "AI Segmentation", level=Qgis.Critical
            )
            return False

    def set_image(self, image_np: np.ndarray) -> None:
        image_b64 = base64.b64encode(image_np.tobytes()).decode("utf-8")
        data = {
            "image_b64": image_b64,
            "image_shape": list(image_np.shape),
            "image_dtype": str(image_np.dtype),
        }
        resp = self._request(
            "POST", "/set_image", data, timeout=_TIMEOUT_SET_IMAGE
        )
        self._session_id = resp["session_id"]
        self.original_size = tuple(resp["original_size"])
        self.input_size = None
        self.is_image_set = True
        QgsMessageLog.logMessage(
            "SAM 3 set_image: session={}, size={}".format(
                self._session_id, self.original_size),
            "AI Segmentation", level=Qgis.Info
        )

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
        text_prompt: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_image_set or self._session_id is None:
            raise RuntimeError(
                "Image has not been set. Call set_image first."
            )

        data = {
            "session_id": self._session_id,
            "point_coords": (
                point_coords.tolist() if point_coords is not None else []
            ),
            "point_labels": (
                point_labels.tolist() if point_labels is not None else []
            ),
            "multimask_output": multimask_output,
        }
        if text_prompt:
            data["text_prompt"] = text_prompt
        if mask_input is not None:
            data["mask_input"] = base64.b64encode(
                mask_input.tobytes()
            ).decode("utf-8")
            data["mask_input_shape"] = list(mask_input.shape)
            data["mask_input_dtype"] = str(mask_input.dtype)

        resp = self._request(
            "POST", "/predict", data, timeout=_TIMEOUT_PREDICT
        )

        masks_bytes = base64.b64decode(resp["masks"].encode("utf-8"))
        masks = np.frombuffer(
            masks_bytes, dtype=resp["masks_dtype"]
        ).reshape(resp["masks_shape"])

        scores = np.array(resp["scores"])

        lr_bytes = base64.b64decode(
            resp["low_res_masks"].encode("utf-8")
        )
        low_res_masks = np.frombuffer(
            lr_bytes, dtype=resp["low_res_masks_dtype"]
        ).reshape(resp["low_res_masks_shape"])

        return masks, scores, low_res_masks

    def predict_text(self, text: str) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """Text-only prediction (no points)."""
        return self.predict(text_prompt=text)

    def reset_image(self) -> None:
        if self._session_id:
            try:
                self._request(
                    "POST",
                    "/reset?session_id={}".format(self._session_id),
                    timeout=_TIMEOUT_RESET
                )
            except Exception as e:
                QgsMessageLog.logMessage(
                    "SAM 3 reset error: {}".format(e),
                    "AI Segmentation", level=Qgis.Warning
                )
        self._session_id = None
        self.is_image_set = False
        self.original_size = None
        self.input_size = None

    def cleanup(self) -> None:
        self.reset_image()
