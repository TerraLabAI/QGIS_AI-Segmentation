




from __future__ import annotations

from typing import Sequence


WHOLE_CROP_RATIO_DEFAULT = 1.0


def whole_crop_ratio() -> float:


    try:
        from .server_dials import dial_in_range

        return dial_in_range(
            "tuning.click.multimask_whole_crop_ratio",
            WHOLE_CROP_RATIO_DEFAULT, 0.5, 1.0)
    except Exception:  # noqa: BLE001
        return WHOLE_CROP_RATIO_DEFAULT


def pick_multimask_index(areas: Sequence[int], scores: Sequence[float],
                         total_pixels: int, ratio: float | None = None) -> int:








    if ratio is None:
        ratio = whole_crop_ratio()
    small_enough = [
        i for i in range(len(scores))
        if 0 < areas[i] < ratio * total_pixels
    ]
    if small_enough:
        return max(small_enough, key=lambda i: float(scores[i]))
    return min(range(len(scores)), key=lambda i: areas[i])
