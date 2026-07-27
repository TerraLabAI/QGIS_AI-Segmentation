





























from __future__ import annotations





REMOVE_ALL_RINGS = -1.0


def units2_per_m2(crs_area: float, ground_area_m2: float) -> float:











    try:
        crs_area = float(crs_area)
        ground_area_m2 = float(ground_area_m2)
    except (TypeError, ValueError):
        return 1.0
    if crs_area <= 0.0 or ground_area_m2 <= 0.0:
        return 1.0
    return crs_area / ground_area_m2


def ring_area_arg(ground_m2: float, scale_units2_per_m2: float = 1.0) -> float:







    try:
        ground_m2 = float(ground_m2)
    except (TypeError, ValueError):
        return REMOVE_ALL_RINGS
    if ground_m2 <= 0.0:
        return REMOVE_ALL_RINGS
    try:
        scale = float(scale_units2_per_m2)
    except (TypeError, ValueError):
        scale = 1.0
    if scale <= 0.0:
        scale = 1.0
    return ground_m2 * scale


def hole_pixels(ground_m2: float, m2_per_pixel: float) -> int | None:







    try:
        ground_m2 = float(ground_m2)
    except (TypeError, ValueError):
        return None
    if ground_m2 <= 0.0:
        return None
    try:
        m2_per_pixel = float(m2_per_pixel)
    except (TypeError, ValueError):
        return None
    if m2_per_pixel <= 0.0:
        return None
    return int(ground_m2 / m2_per_pixel)
