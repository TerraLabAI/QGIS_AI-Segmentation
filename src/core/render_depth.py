

























from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)









LEVEL_AGREEMENT_MIN: float = -1.0




LEVEL_SAMPLE_PX: int = 32




LEVEL_BACKOFF_STEPS: int = 2





_FLAT_IMAGE_EPS: float = 1e-6


def level_agreement(fine_img, coarse_img, sample_px: int = 0) -> float:











    try:
        import numpy as np  # noqa: PLC0415

        side = int(sample_px) if sample_px else _sample_px()
        fine = _grey_sample(fine_img, side, np)
        coarse = _grey_sample(coarse_img, side, np)
        if fine is None or coarse is None:
            return 1.0
        fine = fine - fine.mean()
        coarse = coarse - coarse.mean()
        norm = float(np.linalg.norm(fine)) * float(np.linalg.norm(coarse))
        if norm <= _FLAT_IMAGE_EPS:
            return 1.0
        return float(fine.ravel() @ coarse.ravel() / norm)
    except Exception as exc:  # noqa: BLE001
        logger.debug("level_agreement: comparison failed: %s", exc)
        return 1.0


def agreement_min(policy: dict | None = None) -> float:





    from .detection_policy import unavailable_agreement_min

    try:
        value = float(unavailable_agreement_min(LEVEL_AGREEMENT_MIN, policy))
    except (TypeError, ValueError):
        return LEVEL_AGREEMENT_MIN
    if math.isnan(value) or not -1.0 <= value <= 1.0:
        return LEVEL_AGREEMENT_MIN
    return value


def backoff_steps(policy: dict | None = None) -> int:





    from .detection_policy import unavailable_backoff_steps

    try:
        value = int(unavailable_backoff_steps(LEVEL_BACKOFF_STEPS, policy))
    except (TypeError, ValueError):
        return LEVEL_BACKOFF_STEPS
    return max(0, min(value, 8))


def _sample_px(policy: dict | None = None) -> int:





    from .detection_policy import unavailable_agreement_sample_px

    try:
        value = int(unavailable_agreement_sample_px(LEVEL_SAMPLE_PX, policy))
    except (TypeError, ValueError):
        return LEVEL_SAMPLE_PX
    if not 8 <= value <= 256:
        return LEVEL_SAMPLE_PX
    return value


def _grey_sample(img, side: int, np):






    if img is None or img.isNull():
        return None


    from .qimage_strips import qimage_array_in_strips, smooth_scaled_in_python

    small = smooth_scaled_in_python(img, side, side)
    if small.width() != side or small.height() != side:
        return None
    arr = qimage_array_in_strips(small, small.format(), 4)
    if arr is None:
        return None

    return (arr[:, :, 2] * 0.299 + arr[:, :, 1] * 0.587
            + arr[:, :, 0] * 0.114).astype(np.float32)
