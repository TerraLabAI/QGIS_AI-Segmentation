"""CloudSam3Predictor - same interface as CloudPredictor but calls SAM3 server."""
import io
import json
import zlib
import base64
import urllib.request
import urllib.error
from typing import Tuple, Optional

import numpy as np
from PIL import Image as PILImage

from qgis.core import QgsMessageLog, Qgis

from .model_config import SAM3_CLOUD_URL

_TIMEOUT_HEALTH = 60
_TIMEOUT_SET_IMAGE = 120
_TIMEOUT_PREDICT = 60
_TIMEOUT_PREDICT_TEXT = 120
_TIMEOUT_RESET = 10


class SessionExpiredError(RuntimeError):
    """Raised when the server returns 404 for an expired session."""
    pass


class CloudSam3Predictor:

    def __init__(self, api_key: str = "") -> None:
        self.is_image_set = False
        self.original_size = None
        self.input_size = None
        self._session_id = None
        self._api_key = api_key
        self._last_image_np = None
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

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
            if e.code == 404 and "Session not found" in detail:
                raise SessionExpiredError(detail)
            try:
                detail = json.loads(detail).get("detail", detail)
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

    def warm_up_with_retry(self, max_attempts=5, initial_delay=30, attempt_callback=None) -> Tuple[bool, str]:
        """
        Try to contact SAM 3 server with retry and exponential backoff.

        Args:
            max_attempts: Maximum number of attempts (default: 5)
            initial_delay: Initial delay in seconds between retries (default: 30s, capped at 60s)
            attempt_callback: Optional callable(attempt_num, max_attempts) called before each attempt

        Returns:
            Tuple of (success: bool, error_type: str)
            error_type can be: 'none', 'timeout', 'network', 'unknown', 'cancelled'
        """
        import time

        for attempt in range(1, max_attempts + 1):
            if self._stop_requested:
                return (False, "cancelled")
            if attempt_callback:
                attempt_callback(attempt, max_attempts)

            try:
                QgsMessageLog.logMessage(
                    "SAM 3 warm_up: attempt {}/{}".format(attempt, max_attempts),
                    "AI Segmentation", level=Qgis.Info
                )

                resp = self._request("GET", "/health", timeout=_TIMEOUT_HEALTH)
                if resp.get("status") == "ok":
                    device = resp.get("device", "unknown")
                    sessions = resp.get("active_sessions", 0)
                    QgsMessageLog.logMessage(
                        "SAM 3 warm_up: success (device={}, sessions={})".format(
                            device, sessions),
                        "AI Segmentation", level=Qgis.Info
                    )
                    # Validate API key against a protected endpoint
                    try:
                        self._request("POST", "/reset?session_id=auth-check", timeout=_TIMEOUT_RESET)
                    except Exception as auth_err:
                        if "401" in str(auth_err):
                            return (False, "auth")
                    return (True, "none")
                else:
                    raise RuntimeError("Unexpected response: {}".format(resp))

            except Exception as e:
                error_msg = str(e)
                is_timeout = (
                    "504" in error_msg or "503" in error_msg or "502" in error_msg
                    or "timeout" in error_msg.lower()
                )

                QgsMessageLog.logMessage(
                    "SAM 3 warm_up: attempt {} failed - {}".format(
                        attempt, error_msg),
                    "AI Segmentation",
                    level=Qgis.Warning
                )

                # Last attempt: give up
                if attempt == max_attempts:
                    QgsMessageLog.logMessage(
                        "SAM 3 warm_up: failed after {} attempts".format(
                            max_attempts),
                        "AI Segmentation", level=Qgis.Critical
                    )
                    if is_timeout:
                        return (False, "timeout")
                    elif "unreachable" in error_msg.lower():
                        return (False, "network")
                    else:
                        return (False, "unknown")

                # Wait before retry (exponential backoff for cold starts, capped at 60s)
                delay = min(initial_delay * (2 ** (attempt - 1)), 60) if is_timeout else 2
                if is_timeout:
                    QgsMessageLog.logMessage(
                        "SAM 3 cold start detected, retrying in {}s...".format(delay),
                        "AI Segmentation", level=Qgis.Info
                    )
                elapsed = 0
                while elapsed < delay and not self._stop_requested:
                    time.sleep(min(1, delay - elapsed))
                    elapsed += 1

        return (False, "unknown")

    def set_image(self, image_np: np.ndarray) -> None:
        self._last_image_np = image_np
        # JPEG compress for faster upload (~4MB -> ~0.3-0.5MB)
        pil_img = PILImage.fromarray(image_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data = {
            "image_b64": image_b64,
            "image_shape": list(image_np.shape),
            "image_dtype": str(image_np.dtype),
            "image_format": "jpeg",
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
        if box is not None:
            data["box"] = box.tolist()
        if text_prompt:
            data["text_prompt"] = text_prompt
        if mask_input is not None:
            data["mask_input"] = base64.b64encode(
                mask_input.tobytes()
            ).decode("utf-8")
            data["mask_input_shape"] = list(mask_input.shape)
            data["mask_input_dtype"] = str(mask_input.dtype)

        timeout = (
            _TIMEOUT_PREDICT_TEXT if text_prompt
            else _TIMEOUT_PREDICT
        )
        try:
            resp = self._request(
                "POST", "/predict", data, timeout=timeout
            )
        except SessionExpiredError:
            if self._last_image_np is None:
                raise RuntimeError(
                    "Session expired and no image available for retry"
                )
            QgsMessageLog.logMessage(
                "SAM 3 session expired, re-uploading image...",
                "AI Segmentation", level=Qgis.Info
            )
            self.set_image(self._last_image_np)
            data["session_id"] = self._session_id
            resp = self._request(
                "POST", "/predict", data, timeout=timeout
            )

        masks_bytes = base64.b64decode(resp["masks"].encode("utf-8"))
        if resp.get("masks_compressed"):
            masks_bytes = zlib.decompress(masks_bytes)
        masks = np.frombuffer(
            masks_bytes, dtype=resp["masks_dtype"]
        ).reshape(resp["masks_shape"])

        scores = np.array(resp["scores"])

        lr_bytes = base64.b64decode(
            resp["low_res_masks"].encode("utf-8")
        )
        if resp.get("low_res_masks_compressed"):
            lr_bytes = zlib.decompress(lr_bytes)
        low_res_masks = np.frombuffer(
            lr_bytes, dtype=resp["low_res_masks_dtype"]
        ).reshape(resp["low_res_masks_shape"])

        return masks, scores, low_res_masks

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
        self._last_image_np = None

    def cleanup(self) -> None:
        self.reset_image()
