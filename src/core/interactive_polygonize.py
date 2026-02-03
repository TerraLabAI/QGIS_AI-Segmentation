





from __future__ import annotations

import math


def interactive_polygonize_enabled() -> bool:

    try:
        from .server_dials import feature_enabled

        return feature_enabled("fast_ring_tracer")
    except Exception:  # noqa: BLE001
        return True


def mask_to_polygons_interactive(mask, transform_info, simplify_tolerance=0.0,
                                 pixel_offset=None, full_shape=None):






    from .polygon_masks import mask_to_polygons, mask_to_polygons_fallback

    if mask is None or not mask.any():
        return []
    try:
        from rasterio.transform import from_bounds  # noqa: F401
    except ImportError:
        pass
    else:
        return mask_to_polygons(mask, transform_info, simplify_tolerance,
                                pixel_offset, full_shape)

    if interactive_polygonize_enabled():
        traced = _traced_fallback(mask, transform_info, simplify_tolerance,
                                  pixel_offset, full_shape)
        if traced is not None:
            return traced
    return mask_to_polygons_fallback(mask, transform_info, simplify_tolerance,
                                     pixel_offset, full_shape)


def _traced_fallback(mask, transform_info, simplify_tolerance,
                     pixel_offset, full_shape):

    try:
        from .polygon_masks import _mask_grid_info, fallback_polygons_traced

        if mask.ndim != 2:
            return None
        height, width = (int(v) for v in mask.shape)
        if height <= 0 or width <= 0:
            return None
        grid = _mask_grid_info(mask, transform_info, pixel_offset, full_shape)
        bbox = grid.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None



        if any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox):
            return None
        minx, maxx, miny, maxy = bbox
        if (minx + width * ((maxx - minx) / width) != maxx
                or maxy - height * ((maxy - miny) / height) != miny):
            return None
        return fallback_polygons_traced(
            mask, None, grid, simplify_tolerance, (0, 0), (height, width))
    except Exception:  # noqa: BLE001
        return None
