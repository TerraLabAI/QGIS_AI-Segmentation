"""FalPredictor - calls remote SAM3 segmentation endpoint."""

import base64
import io
import json
import urllib.error
import urllib.request
from typing import Optional

import numpy as np
from qgis.core import Qgis, QgsMessageLog

from .model_config import MEDIA_UPLOAD_URL, SAM3_INFERENCE_URL, STORAGE_TOKEN_URL

_TIMEOUT_PREDICT = 120
_TIMEOUT_UPLOAD = 15


def decode_rle_to_mask(rle_string: str, height: int, width: int) -> np.ndarray:
    """Decode RLE string from remote endpoint to a boolean numpy mask (H, W).

    RLE format: space-separated pairs of "offset count" where each pair
    means 'count' foreground pixels starting at flat-array position 'offset'.
    The flat array is in row-major order (C order).

    Args:
        rle_string: Space-separated "offset count offset count ..." string.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        Boolean numpy array of shape (height, width).
    """
    import logging

    logger = logging.getLogger("AISegmentation")

    total = height * width
    flat = np.zeros(total, dtype=np.uint8)
    tokens = rle_string.split()
    if not tokens:
        return flat.reshape((height, width)).astype(bool)

    # Validate RLE dimensions match expected size
    max_rle_end = 0
    for i in range(0, len(tokens), 2):
        offset = int(tokens[i])
        count = int(tokens[i + 1])
        end = offset + count
        if end > max_rle_end:
            max_rle_end = end

    if max_rle_end > total:
        logger.warning(
            "RLE DIMENSION MISMATCH: max_offset=%d > expected_total=%d (%dx%d). "
            "fal.ai returned mask for a larger grid.",
            max_rle_end,
            total,
            width,
            height,
        )

    for i in range(0, len(tokens), 2):
        offset = int(tokens[i])
        count = int(tokens[i + 1])
        flat[offset : offset + count] = 1
    return flat.reshape((height, width)).astype(bool)


class FalPredictor:
    """Stateless predictor that calls remote SAM3 segmentation endpoint."""

    def __init__(self, fal_key: str) -> None:
        self._fal_key = fal_key
        self._last_image_np: Optional[np.ndarray] = None
        self.is_image_set = False
        self.original_size: Optional[tuple[int, int]] = None
        self._native_size: Optional[tuple[int, int]] = None

    def set_image(self, image_np: np.ndarray) -> None:
        """Store image locally for subsequent predict_text calls."""
        self._last_image_np = image_np
        self.original_size = (image_np.shape[0], image_np.shape[1])
        self.is_image_set = True
        QgsMessageLog.logMessage(
            f"FalPredictor: image stored ({image_np.shape[1]}x{image_np.shape[0]})",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

    def set_image_from_array(self, image_array: np.ndarray) -> None:
        """Set the image from a numpy array (H, W, 3) uint8.

        Used by tiling workflow where crops are already in memory.
        This is functionally equivalent to set_image() but provides
        a clear API for the tiling use case.

        Args:
            image_array: RGB image array, shape (H, W, 3), dtype uint8.
        """
        self._last_image_np = image_array
        self.original_size = (image_array.shape[0], image_array.shape[1])
        self.is_image_set = True
        QgsMessageLog.logMessage(
            f"FalPredictor: tile image stored ({image_array.shape[1]}x{image_array.shape[0]})",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

    def _get_cdn_token(self) -> Optional[str]:
        """Exchange API key for a short-lived CDN Bearer token."""
        req = urllib.request.Request(
            STORAGE_TOKEN_URL,
            method="POST",
            headers={
                "Authorization": f"Key {self._fal_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            data=b"{}",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("token")

    def _upload_to_cdn(self, image_np: np.ndarray) -> Optional[str]:
        """Upload image to media CDN, return access URL or None on failure."""
        try:
            # Encode image as PNG bytes
            from PIL import Image as PILImage

            pil_img = PILImage.fromarray(image_np)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            # Get short-lived Bearer token for CDN
            token = self._get_cdn_token()
            if not token:
                QgsMessageLog.logMessage(
                    "FalPredictor: CDN token exchange returned empty",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Warning,
                )
                return None

            # Upload with Bearer token
            req = urllib.request.Request(
                MEDIA_UPLOAD_URL,
                data=png_bytes,
                method="POST",
                headers={
                    "Content-Type": "image/png",
                    "Authorization": f"Bearer {token}",
                },
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT_UPLOAD) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                url = result.get("access_url") or result.get("url", "")
                if url:
                    QgsMessageLog.logMessage(
                        f"FalPredictor: CDN upload OK ({len(png_bytes)} bytes)",
                        "AI Segmentation",
                        level=Qgis.MessageLevel.Info,
                    )
                    return url
                QgsMessageLog.logMessage(
                    "FalPredictor: CDN upload returned no URL",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Warning,
                )
                return None
        except Exception as e:
            QgsMessageLog.logMessage(
                f"FalPredictor: CDN upload failed ({e}), using fallback",
                "AI Segmentation",
                level=Qgis.MessageLevel.Warning,
            )
            return None

    def _make_fallback_data_uri(self, image_np: np.ndarray) -> str:
        """Encode image as JPEG Q95 data URI (fallback when CDN is unavailable)."""
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        QgsMessageLog.logMessage(
            f"FalPredictor: using JPEG fallback ({len(buf.getvalue())} bytes)",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )
        return f"data:image/jpeg;base64,{b64}"

    def predict_text(self, prompt: str, max_masks: int = 10) -> dict:
        """Send image + prompt to remote endpoint, return raw response dict.

        Returns dict with keys: rle, scores, boxes.
        Raises RuntimeError on HTTP errors or timeout.
        """
        if self._last_image_np is None:
            raise RuntimeError("No image set. Call set_image first.")

        img = self._last_image_np
        # Crop to native size if set — discards reflect-padding added by feature
        # encoder so SAM-3 sees only real pixels and zero-pads internally.
        if self._native_size is not None:
            native_h, native_w = self._native_size
            if img.shape[0] != native_h or img.shape[1] != native_w:
                img = img[:native_h, :native_w]
                QgsMessageLog.logMessage(
                    f"FalPredictor: cropped padded image from "
                    f"{self._last_image_np.shape[1]}x{self._last_image_np.shape[0]} to native {native_w}x{native_h}",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )
        QgsMessageLog.logMessage(
            f"FalPredictor.predict_text: prompt='{prompt}', max_masks={max_masks}, "
            f"img_shape={img.shape}, dtype={img.dtype}, range=[{int(img.min())},{int(img.max())}]",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        # Upload to CDN (fast path) or fall back to data URI
        cdn_url = self._upload_to_cdn(img)
        if cdn_url:
            image_url = cdn_url
            QgsMessageLog.logMessage(
                "FalPredictor: using CDN URL",
                "AI Segmentation",
                level=Qgis.MessageLevel.Info,
            )
        else:
            image_url = self._make_fallback_data_uri(img)

        payload_dict = {
            "image_url": image_url[:80] + "..." if len(image_url) > 80 else image_url,
            "prompt": prompt,
            "apply_mask": False,
            "return_multiple_masks": True,
            "max_masks": max_masks,
            "include_scores": True,
            "include_boxes": True,
        }
        QgsMessageLog.logMessage(
            "FalPredictor: payload (sans image) = {}".format(
                {k: v for k, v in payload_dict.items() if k != "image_url"}
            ),
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        payload = json.dumps(
            {
                "image_url": image_url,
                "prompt": prompt,
                "apply_mask": False,
                "return_multiple_masks": True,
                "max_masks": max_masks,
                "include_scores": True,
                "include_boxes": True,
            }
        ).encode("utf-8")
        QgsMessageLog.logMessage(
            f"FalPredictor: payload size = {len(payload)} bytes",
            "AI Segmentation",
            level=Qgis.MessageLevel.Info,
        )

        req = urllib.request.Request(
            SAM3_INFERENCE_URL,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Key {self._fal_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_PREDICT) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                rle_data = result.get("rle", [])
                scores = result.get("scores") or []
                n_masks = len(rle_data) if isinstance(rle_data, list) else 1
                # Log detailed response info
                rle_preview = []
                if isinstance(rle_data, list):
                    for idx, r in enumerate(rle_data[:5]):
                        tokens = r.split() if isinstance(r, str) else []
                        rle_preview.append(
                            "mask{}:{}tokens,{}chars".format(
                                idx, len(tokens), len(r) if isinstance(r, str) else "?"
                            )
                        )
                QgsMessageLog.logMessage(
                    "FalPredictor: prediction OK — {} masks, "
                    "scores={}, rle_preview=[{}]".format(
                        n_masks,
                        [round(s, 3) for s in scores[:10]] if scores else "null",
                        ", ".join(rle_preview),
                    ),
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )
                # Store the actually-sent image dimensions so callers can
                # decode RLE at the correct size
                result["_sent_h"] = img.shape[0]
                result["_sent_w"] = img.shape[1]
                return result
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read().decode("utf-8")).get("detail", "")
            except Exception:
                pass
            # 422 with "No masks" = empty result, not an error
            if e.code == 422 and "No masks" in detail:
                QgsMessageLog.logMessage(
                    f"FalPredictor: no masks found for prompt '{prompt}'",
                    "AI Segmentation",
                    level=Qgis.MessageLevel.Info,
                )
                return {"rle": [], "scores": []}
            if e.code == 401:
                raise RuntimeError("Invalid API key. Check FAL_KEY in .env") from e
            raise RuntimeError(f"Inference error {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach inference service: {e.reason}") from e

    def set_native_size(self, height: int, width: int) -> None:
        """Set the native (pre-padding) image dimensions.

        When set, predict_text() will crop the stored image to these dimensions
        before uploading, discarding reflect-padding added by the feature encoder.
        SAM-3 will then zero-pad internally, avoiding mirror-image false detections.
        """
        self._native_size = (height, width)

    def reset_image(self) -> None:
        """Clear stored image state."""
        self._last_image_np = None
        self.is_image_set = False
        self.original_size = None
        self._native_size = None

    def cleanup(self) -> None:
        """Release resources (compatibility with existing cleanup calls)."""
        self.reset_image()
