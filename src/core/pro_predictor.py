"""FalPredictor - calls fal.ai SAM3 image-rle endpoint directly."""
import base64
import io
import json
import urllib.error
import urllib.request
from typing import Optional, Tuple

import numpy as np
from PIL import Image as PILImage
from qgis.core import Qgis, QgsMessageLog

from .model_config import SAM3_INFERENCE_URL

_TIMEOUT_PREDICT = 60


def decode_rle_to_mask(rle: dict) -> np.ndarray:
    """Decode COCO uncompressed RLE to a boolean numpy mask (H, W).

    rle format: {"counts": [n0, n1, n2, ...], "size": [H, W]}
    Counts alternate between background (0) and foreground (1) pixels
    in Fortran (column-major) order. First count is always background.
    """
    counts = rle["counts"]
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:
            flat[pos : pos + count] = 1
        pos += count
    return flat.reshape((w, h)).T.astype(bool)


class FalPredictor:
    """Stateless predictor that calls fal.ai SAM3 image-rle endpoint."""

    def __init__(self, fal_key: str) -> None:
        self._fal_key = fal_key
        self._last_image_np: Optional[np.ndarray] = None
        self.is_image_set = False
        self.original_size: Optional[Tuple[int, int]] = None

    def set_image(self, image_np: np.ndarray) -> None:
        """Store image locally for subsequent predict_text calls."""
        self._last_image_np = image_np
        self.original_size = (image_np.shape[0], image_np.shape[1])
        self.is_image_set = True
        QgsMessageLog.logMessage(
            "FalPredictor: image stored ({}x{})".format(
                image_np.shape[1], image_np.shape[0]
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

    def predict_text(self, prompt: str) -> dict:
        """Send image + prompt to fal.ai, return raw response dict.

        Returns dict with keys: rle, scores, boxes.
        Raises RuntimeError on HTTP errors or timeout.
        """
        if self._last_image_np is None:
            raise RuntimeError("No image set. Call set_image first.")

        # Encode image as JPEG data URI
        pil_img = PILImage.fromarray(self._last_image_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_uri = "data:image/jpeg;base64,{}".format(b64)

        payload = json.dumps(
            {
                "image_url": data_uri,
                "prompt": prompt,
                "apply_mask": False,
                "return_multiple_masks": True,
                "max_masks": 10,
                "include_scores": True,
                "include_boxes": True,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            SAM3_INFERENCE_URL,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Key {}".format(self._fal_key),
            },
        )

        try:
            with urllib.request.urlopen(
                req, timeout=_TIMEOUT_PREDICT
            ) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                rle_data = result.get("rle", [])
                n_masks = (
                    len(rle_data) if isinstance(rle_data, list) else 1
                )
                QgsMessageLog.logMessage(
                    "FalPredictor: prediction OK ({} masks)".format(
                        n_masks
                    ),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )
                return result
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read().decode("utf-8")).get(
                    "detail", ""
                )
            except Exception:
                pass
            if e.code == 401:
                raise RuntimeError(
                    "Invalid API key. Check FAL_KEY in .env"
                )
            raise RuntimeError(
                "Inference error {}: {}".format(e.code, detail)
            )
        except urllib.error.URLError as e:
            raise RuntimeError(
                "Cannot reach inference service: {}".format(e.reason)
            )

    def reset_image(self) -> None:
        """Clear stored image state."""
        self._last_image_np = None
        self.is_image_set = False
        self.original_size = None

    def cleanup(self) -> None:
        """Release resources (compatibility with existing cleanup calls)."""
        self.reset_image()
