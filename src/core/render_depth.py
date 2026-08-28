

























from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)








LEVEL_AGREEMENT_MIN: float = 0.27




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


def levels_disagree(fine_img, coarse_img) -> bool:






    return level_agreement(fine_img, coarse_img) < agreement_min()


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






    from qgis.PyQt.QtCore import QSize, Qt
    from qgis.PyQt.QtGui import QImage

    if img is None or img.isNull():
        return None
    small = img.scaled(
        QSize(side, side),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    ).convertToFormat(QImage.Format.Format_RGB32)
    if small.width() != side or small.height() != side:
        return None
    ptr = small.bits()
    ptr.setsize(side * side * 4)
    arr = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape(side, side, 4)

    return (arr[:, :, 2] * 0.299 + arr[:, :, 1] * 0.587
            + arr[:, :, 0] * 0.114).astype(np.float32)
