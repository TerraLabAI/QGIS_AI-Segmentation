



















from __future__ import annotations

import math

from .shape_policy_dials import crop_edge_clearance, crop_grid_cell_fraction, crop_scale_step




GRID_CELL_FRACTION = 0.25




SCALE_STEP = 1.15



EDGE_CLEARANCE = 0.05



NATIVE_SCALE = 1.0


def scale_at_least_native(exact_scale: float) -> float:














    try:
        value = float(exact_scale)
    except (TypeError, ValueError):
        return NATIVE_SCALE
    if not math.isfinite(value) or not value > 0:
        return NATIVE_SCALE
    return max(NATIVE_SCALE, value)


def ground_per_pixel_at_least_native(requested: float, native: float) -> float:







    try:
        want = float(requested)
        floor = float(native)
    except (TypeError, ValueError):
        return requested
    if not want > 0 or not floor > 0 or not math.isfinite(floor):
        return requested
    return max(want, floor)








MAX_CROP_GROUND_WIDTH_M = 2560.0


def ground_per_pixel_within_ceiling(requested: float, ceiling: float) -> float:







    try:
        want = float(requested)
        roof = float(ceiling)
    except (TypeError, ValueError):
        return requested
    if not want > 0 or not roof > 0:
        return requested
    return min(want, roof)


def scale_floor_is_usable(min_scale) -> bool:






    try:
        value = float(min_scale)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0


def crop_scale_for_bounds(
    width: float,
    height: float,
    native_pixel_size: float,
    crop_size: int = 1024,
    margin: float = 1.4,
    min_scale: float = 1.0,
    max_scale: float = 8.0,
) -> float:



















    floored = scale_floor_is_usable(min_scale)
    unsized = float(min_scale) if floored else 0.0
    if (crop_size <= 0 or native_pixel_size <= 0
            or not all(math.isfinite(v) for v in
                       (width, height, native_pixel_size, margin, max_scale))):
        return unsized
    needed_ground = max(width, height) * margin
    exact = needed_ground / (crop_size * native_pixel_size)
    if not math.isfinite(exact) or not exact > 0:


        return unsized
    if not floored:


        return min(max_scale, exact)
    if exact <= min_scale:
        return min_scale
    if exact >= max_scale:
        return max_scale
    scale_step = crop_scale_step(SCALE_STEP)
    steps = math.ceil(math.log(exact / min_scale) / math.log(scale_step))
    return min(max_scale, min_scale * scale_step ** steps)


def snap_center_to_grid(
    center_x: float,
    center_y: float,
    scale: float,
    native_pixel_size: float,
    crop_size: int = 1024,
) -> tuple[float, float]:






    if (native_pixel_size <= 0 or scale <= 0 or crop_size <= 0
            or not all(math.isfinite(v) for v in
                       (center_x, center_y, native_pixel_size, scale))):
        return center_x, center_y
    step = crop_size * scale * native_pixel_size * crop_grid_cell_fraction(GRID_CELL_FRACTION)
    if step <= 0 or not math.isfinite(step):
        return center_x, center_y
    return round(center_x / step) * step, round(center_y / step) * step


def neighborhood_crop_window(
    bounds: tuple[float, float, float, float],
    native_pixel_size: float,
    crop_size: int = 1024,
    margin: float = 1.4,
    min_scale: float = 1.0,
    max_scale: float = 8.0,
) -> tuple[float, float, float]:








    minx, miny, maxx, maxy = bounds
    exact_cx = (minx + maxx) / 2.0
    exact_cy = (miny + maxy) / 2.0
    scale = crop_scale_for_bounds(
        maxx - minx, maxy - miny, native_pixel_size,
        crop_size=crop_size, margin=margin,
        min_scale=min_scale, max_scale=max_scale)
    if native_pixel_size <= 0:
        return exact_cx, exact_cy, scale

    ground = crop_size * scale * native_pixel_size
    step = ground * crop_grid_cell_fraction(GRID_CELL_FRACTION)
    if step <= 0 or not math.isfinite(step) or not (
            math.isfinite(exact_cx) and math.isfinite(exact_cy)):
        return exact_cx, exact_cy, scale
    cx = round(exact_cx / step) * step
    cy = round(exact_cy / step) * step



    half = ground / 2.0 - ground * crop_edge_clearance(EDGE_CLEARANCE)
    if (minx < cx - half or maxx > cx + half or miny < cy - half or maxy > cy + half):
        return exact_cx, exact_cy, scale
    return cx, cy, scale


def crop_window_key(
    center_x: float, center_y: float, scale: float
) -> tuple[float, float, float]:

    return (round(float(center_x), 6), round(float(center_y), 6),
            round(float(scale), 6))


def crop_pixel_of_point(
    bounds: tuple[float, float, float, float],
    img_shape: tuple[int, int],
    x: float,
    y: float,
) -> tuple[int, int] | None:













    minx, miny, maxx, maxy = bounds
    height, width = int(img_shape[0]), int(img_shape[1])
    span_x = maxx - minx
    span_y = maxy - miny
    if (not all(math.isfinite(v) for v in (*bounds, x, y, span_x, span_y))
            or span_x <= 0 or span_y <= 0 or width <= 0 or height <= 0):
        return None
    col = math.floor((x - minx) * width / span_x)
    row = math.floor((maxy - y) * height / span_y)
    return (min(max(int(row), 0), height - 1),
            min(max(int(col), 0), width - 1))


def window_frames_bounds(
    window: tuple[float, float, float],
    bounds: tuple[float, float, float, float],
    native_pixel_size: float,
    crop_size: int = 1024,
) -> bool:




    if native_pixel_size <= 0 or window is None:
        return False
    cx, cy, scale = window
    if (scale <= 0 or crop_size <= 0 or not all(
            math.isfinite(v) for v in (*bounds, cx, cy, scale, native_pixel_size))):
        return False
    ground = crop_size * scale * native_pixel_size
    half = ground / 2.0 - ground * crop_edge_clearance(EDGE_CLEARANCE)
    minx, miny, maxx, maxy = bounds
    return (minx >= cx - half and maxx <= cx + half and miny >= cy - half and maxy <= cy + half)


def crop_window_for_object(
    bounds: tuple[float, float, float, float],
    native_pixel_size: float,
    held_window: tuple[float, float, float] | None = None,
    crop_size: int = 1024,
    margin: float = 1.4,
    min_scale: float = 1.0,
    max_scale: float = 8.0,
) -> tuple[float, float, float]:












    needed_scale = crop_scale_for_bounds(
        bounds[2] - bounds[0], bounds[3] - bounds[1], native_pixel_size,
        crop_size=crop_size, margin=margin,
        min_scale=min_scale, max_scale=max_scale)
    if (
        held_window is not None
        and round(float(held_window[2]), 6) == round(needed_scale, 6)
        and window_frames_bounds(held_window, bounds, native_pixel_size, crop_size=crop_size)
    ):
        return (float(held_window[0]), float(held_window[1]), needed_scale)
    return neighborhood_crop_window(
        bounds, native_pixel_size, crop_size=crop_size, margin=margin,
        min_scale=min_scale, max_scale=max_scale)
