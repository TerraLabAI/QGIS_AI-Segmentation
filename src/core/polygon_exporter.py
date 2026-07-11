from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import rasterio

from .venv_manager import ensure_venv_packages_available

ensure_venv_packages_available()

import numpy as np  # noqa: E402
from qgis.core import (  # noqa: E402
    Qgis,
    QgsFeature,
    QgsGeometry,
    QgsLineString,
    QgsMessageLog,
    QgsPointXY,
    QgsPolygon,
    QgsSpatialIndex,
)

from .qt_compat import field_type_double, field_type_string  # noqa: E402


def mask_to_polygons_rasterio(
    mask: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
    simplify_tolerance: float = 0.0
) -> list[QgsGeometry]:
    # Empty masks are normal in dense auto runs; return silently. This function
    # runs once per detected instance (hundreds to thousands per run), so any
    # per-call logging here floods the message panel and adds main-thread cost.
    if mask is None or mask.sum() == 0:
        return []

    try:
        from rasterio.features import shapes as get_shapes

        mask_uint8 = mask.astype(np.uint8)

        shape_generator = get_shapes(
            mask_uint8,
            mask=mask_uint8 > 0,
            connectivity=4,
            transform=transform,
        )

        geometries = []
        for geojson_geom, value in shape_generator:
            if value == 0:
                continue

            geom = _geojson_to_geometry(geojson_geom)
            if geom is not None and not geom.isEmpty() and geom.isGeosValid():
                if simplify_tolerance > 0:
                    geom = geom.simplify(simplify_tolerance)
                geometries.append(geom)

        return geometries

    except Exception as e:
        import traceback
        QgsMessageLog.logMessage(
            f"Failed to convert mask to polygons: {str(e)}\n{traceback.format_exc()}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Warning
        )
        return []


def _geojson_to_geometry(geojson: dict) -> "QgsGeometry | None":
    """Build a QgsGeometry directly from a rasterio shapes() GeoJSON dict.

    Replaces a WKT round-trip (geojson -> WKT string -> fromWkt) that formatted
    every vertex into text and re-ran the WKT parser, once per detected instance
    (hundreds-thousands per run). Constructing from the coordinate arrays yields
    the identical geometry without the string serialize/parse. Coordinate order
    is (x, y) exactly as the WKT path wrote it.
    """
    geom_type = geojson.get("type", "")
    coords = geojson.get("coordinates", [])

    if geom_type == "Polygon":
        rings = [[QgsPointXY(x, y) for x, y in ring] for ring in coords]
        return QgsGeometry.fromPolygonXY(rings)

    if geom_type == "MultiPolygon":
        polygons = [
            [[QgsPointXY(x, y) for x, y in ring] for ring in polygon]
            for polygon in coords
        ]
        return QgsGeometry.fromMultiPolygonXY(polygons)

    return None


def mask_to_polygons(
    mask: np.ndarray,
    transform_info: dict,
    simplify_tolerance: float = 0.0,
    pixel_offset: tuple[int, int] | None = None,
    full_shape: tuple[int, int] | None = None,
) -> list[QgsGeometry]:
    # Empty masks are normal and frequent; return silently (see note in
    # mask_to_polygons_rasterio about per-instance logging cost).
    if mask is None or mask.sum() == 0:
        return []

    try:
        from rasterio.transform import from_bounds as transform_from_bounds

        bbox = transform_info.get("bbox")

        if bbox:
            # polygon_exporter bbox convention: (minx, MAXX, miny, MAXY), NOT the
            # standard (minx, miny, maxx, maxy). The auto worker produces this
            # ordering under tile_transform["bbox"] on purpose (see
            # auto_detection_worker._make_tile_transform); do not "fix" it to
            # standard order without updating every caller. Locked by
            # tests/test_mask_to_polygons.py.
            minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
            # Derive the pixel grid from the MASK's actual shape, not the tile
            # dimensions we sent. The cloud model processes at a fixed internal
            # size and can return masks at a different resolution than the tile
            # uploaded; the mask still spans the full tile bbox, so using its own
            # shape keeps the pixel->ground scale exact. Using the sent tile dims
            # instead compressed every mask toward the tile origin, which read as
            # a detection offset and left seam halves too far apart to merge.
            if pixel_offset is not None and full_shape is not None:
                # `mask` is a CROP of a full_shape grid, its top-left at pixel
                # (col0, row0). Keep the pixel->ground scale from the FULL grid
                # (so the crop doesn't rescale), then place the crop by offsetting
                # its origin. This makes polygonizing only the object's pixels
                # geo-identical to polygonizing the full-tile mask, just far
                # cheaper (a small object no longer scans the whole 1024x1024).
                full_h, full_w = int(full_shape[0]), int(full_shape[1])
                col0, row0 = int(pixel_offset[0]), int(pixel_offset[1])
                px_w = (maxx - minx) / max(full_w, 1)
                px_h = (maxy - miny) / max(full_h, 1)
                sub_h, sub_w = int(mask.shape[0]), int(mask.shape[1])
                sub_minx = minx + col0 * px_w
                sub_maxy = maxy - row0 * px_h
                sub_maxx = sub_minx + sub_w * px_w
                sub_miny = sub_maxy - sub_h * px_h
                transform = transform_from_bounds(
                    sub_minx, sub_miny, sub_maxx, sub_maxy, sub_w, sub_h
                )
            else:
                height, width = int(mask.shape[0]), int(mask.shape[1])
                transform = transform_from_bounds(minx, miny, maxx, maxy, width, height)
            crs = transform_info.get("crs", "EPSG:4326")

            return mask_to_polygons_rasterio(mask, transform, crs, simplify_tolerance)

        extent = transform_info.get("extent")
        original_size = transform_info.get("original_size")

        if extent and original_size:
            x_min, y_min, x_max, y_max = extent

            if isinstance(original_size, (list, tuple)):
                height, width = original_size[0], original_size[1]
            else:
                height = width = original_size

            transform = transform_from_bounds(x_min, y_min, x_max, y_max, width, height)
            crs = transform_info.get("layer_crs", transform_info.get("crs", "EPSG:4326"))

            return mask_to_polygons_rasterio(mask, transform, crs, simplify_tolerance)

        return mask_to_polygons_fallback(mask, transform_info, simplify_tolerance)

    except ImportError:
        return mask_to_polygons_fallback(mask, transform_info, simplify_tolerance)
    except Exception as e:
        import traceback
        QgsMessageLog.logMessage(
            f"mask_to_polygons error: {str(e)}\n{traceback.format_exc()}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Warning
        )
        return mask_to_polygons_fallback(mask, transform_info, simplify_tolerance)


def mask_to_polygons_fallback(
    mask: np.ndarray,
    transform_info: dict,
    simplify_tolerance: float = 0.0
) -> list[QgsGeometry]:
    try:
        contours = find_contours(mask)

        if not contours:
            return []

        geometries = []
        for contour in contours:
            if len(contour) < 3:
                continue

            map_points = []
            for px, py in contour:
                mx, my = pixel_to_map_coords(px, py, transform_info)
                map_points.append(QgsPointXY(mx, my))

            if map_points[0] != map_points[-1]:
                map_points.append(map_points[0])

            if len(map_points) >= 4:
                line = QgsLineString(list(map_points))
                polygon = QgsPolygon()
                polygon.setExteriorRing(line)
                geom = QgsGeometry(polygon)

                if simplify_tolerance > 0:
                    geom = geom.simplify(simplify_tolerance)

                if geom.isGeosValid():
                    geometries.append(geom)

        return geometries

    except Exception as e:
        import traceback
        QgsMessageLog.logMessage(
            f"Fallback polygon conversion failed: {str(e)}\n{traceback.format_exc()}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Warning
        )
        return []


def find_contours(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    try:
        from skimage import measure
        raw_contours = measure.find_contours(mask.astype(float), 0.5)
        contours = []
        for contour in raw_contours:
            points = [(int(c[1]), int(c[0])) for c in contour]
            if len(points) >= 3:
                contours.append(points)
        return contours
    except ImportError:
        pass

    contours = []
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    visited_pad = np.pad(visited, 1, mode="constant", constant_values=True)

    directions = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1)
    ]

    for y in range(1, h + 1):
        for x in range(1, w + 1):
            if padded[y, x] == 1 and not visited_pad[y, x]:
                is_boundary = False
                for dx, dy in directions:
                    if padded[y + dy, x + dx] == 0:
                        is_boundary = True
                        break

                if is_boundary:
                    contour = trace_contour(padded, visited_pad, x, y, directions)
                    if len(contour) >= 3:
                        contour = [(px - 1, py - 1) for px, py in contour]
                        contours.append(contour)

    return contours


def trace_contour(
    mask: np.ndarray,
    visited: np.ndarray,
    start_x: int,
    start_y: int,
    directions: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    contour = [(start_x, start_y)]
    visited[start_y, start_x] = True

    x, y = start_x, start_y
    prev_dir = 0

    max_iterations = mask.shape[0] * mask.shape[1]
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        found_next = False

        for i in range(8):
            dir_idx = (prev_dir + i) % 8
            dx, dy = directions[dir_idx]
            nx, ny = x + dx, y + dy

            if mask[ny, nx] == 1:
                is_boundary = False
                for ddx, ddy in directions:
                    if mask[ny + ddy, nx + ddx] == 0:
                        is_boundary = True
                        break

                if is_boundary:
                    if nx == start_x and ny == start_y:
                        return contour

                    if not visited[ny, nx]:
                        contour.append((nx, ny))
                        visited[ny, nx] = True
                        x, y = nx, ny
                        prev_dir = (dir_idx + 5) % 8
                        found_next = True
                        break

        if not found_next:
            break

    return contour


def pixel_to_map_coords(
    pixel_x: float,
    pixel_y: float,
    transform_info: dict
) -> tuple[float, float]:
    bbox = transform_info.get("bbox")
    img_shape = transform_info.get("img_shape")

    if bbox and img_shape:
        # Same polygon_exporter (minx, MAXX, miny, MAXY) ordering as
        # mask_to_polygons above (NOT standard minx, miny, maxx, maxy).
        minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
        height, width = img_shape

        map_x = minx + (pixel_x / width) * (maxx - minx)
        map_y = maxy - (pixel_y / height) * (maxy - miny)
        return map_x, map_y

    extent = transform_info.get("extent")
    original_size = transform_info.get("original_size")

    if extent and original_size:
        x_min, y_min, x_max, y_max = extent

        if isinstance(original_size, (list, tuple)):
            height, width = original_size[0], original_size[1]
        else:
            height = width = original_size

        map_x = x_min + (pixel_x / width) * (x_max - x_min)
        map_y = y_max - (pixel_y / height) * (y_max - y_min)
        return map_x, map_y

    return pixel_x, pixel_y


def apply_mask_refinement(
    mask: np.ndarray,
    expand_value: int = 0,        # -20 to +20 (pixels)
    fill_holes: bool = False,     # Fill interior holes
    min_area: int = 0             # Remove regions smaller than this (pixels)
) -> np.ndarray:
    """
    Apply morphological operations to refine the mask.
    Pure numpy implementation - no scipy needed.
    Note: Simplification is done at the polygon level using QGIS simplify().

    Args:
        mask: Binary mask array
        expand_value: Pixels to expand (positive) or contract (negative)
        fill_holes: If True, fill interior holes in the mask
        min_area: Remove connected regions smaller than this pixel count (0 = keep all)
    """
    result = mask.copy().astype(np.uint8)

    # 1. Expand/Contract first so fill-holes operates on the adjusted mask
    if expand_value != 0:
        iterations = abs(expand_value)
        if expand_value > 0:
            result = _numpy_dilate(result, iterations)
        else:
            result = _numpy_erode(result, iterations)

    # 2. Fill holes (on already expanded/contracted mask)
    if fill_holes:
        result = _fill_holes(result)

    # 3. Remove small regions (artifacts/noise)
    if min_area > 0:
        result = _remove_small_regions(result, min_area)

    return result


def fill_small_holes(mask: np.ndarray, max_hole_px: int) -> np.ndarray:
    """Fill only interior holes up to max_hole_px pixels; KEEP larger holes.

    The per-tile pipeline needs pinholes gone (mask staircase / compression
    noise punches 1-10 px holes that become spurious inner rings), but a
    genuine interior hole (building courtyard, ring road, island) is real
    shape that must survive to the output: the review's "Fill holes" toggle
    owns the decision for those, and it can only work if the hole still
    exists when the mask is polygonized. An unconditional fill here made that
    toggle a no-op and exported every courtyard building as a solid block.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return mask
    try:
        filled = ndimage.binary_fill_holes(mask)
        holes = filled & ~mask.astype(bool)
        if not holes.any():
            return filled.astype(np.uint8)
        labels, n = ndimage.label(holes)
        if n:
            counts = np.bincount(labels.ravel())
            big = np.flatnonzero(counts > max_hole_px)
            big = big[big != 0]  # label 0 is background, never a hole
            if big.size:
                filled[np.isin(labels, big)] = False
        return filled.astype(np.uint8)
    except Exception:
        # A scipy failure (e.g. RecursionError) must never crash the per-tile
        # pipeline; leave the mask unfilled rather than aborting the run
        # (#bug-robert).
        return mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill interior holes in the mask.
    A hole is a region of 0s completely surrounded by 1s.
    Uses scipy if available (very fast), otherwise numpy fallback.
    """
    # Try scipy first - it's much faster (C implementation)
    try:
        from scipy import ndimage
        return ndimage.binary_fill_holes(mask).astype(np.uint8)
    except ImportError:
        pass
    except Exception:  # noqa: BLE001  # nosec B110
        # scipy is present but the fill failed (e.g. a RecursionError deep in
        # ndimage on a pathological mask). Fall through to the recursion-free
        # numpy implementation below rather than crashing the caller, which used
        # to bubble up and kill the click handler.
        pass

    # Numpy fallback: iterative flood fill from edges
    h, w = mask.shape
    # Create a padded version to flood fill from outside
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask

    # Start with border pixels as exterior (only background pixels)
    exterior = np.zeros_like(padded, dtype=bool)
    exterior[0, :] = (padded[0, :] == 0)
    exterior[-1, :] = (padded[-1, :] == 0)
    exterior[:, 0] = (padded[:, 0] == 0)
    exterior[:, -1] = (padded[:, -1] == 0)

    # Iteratively expand exterior into connected background pixels
    background = (padded == 0)
    for _ in range(min(max(h, w), 2048)):  # Capped to prevent excessive loops
        # Dilate exterior by 1 pixel in 4 directions using slicing
        expanded = exterior.copy()
        expanded[1:, :] |= exterior[:-1, :]
        expanded[:-1, :] |= exterior[1:, :]
        expanded[:, 1:] |= exterior[:, :-1]
        expanded[:, :-1] |= exterior[:, 1:]

        # Only keep background pixels
        expanded &= background

        # Check if anything changed
        if np.array_equal(expanded, exterior):
            break
        exterior = expanded

    # Holes are background pixels that are not exterior
    result = padded.copy()
    result[(padded == 0) & (~exterior)] = 1

    return result[1:-1, 1:-1]


def _remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove connected regions smaller than min_area pixels.
    Uses scipy if available (very fast), otherwise numpy fallback.
    """
    if min_area <= 1:
        return mask.copy()

    # Try to use scipy if available (much faster - C implementation)
    try:
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            return mask.copy()

        # Count pixels in each region using bincount (very fast)
        component_sizes = np.bincount(labeled.ravel())

        # Create lookup table: True for regions to keep
        keep_mask = component_sizes >= min_area
        keep_mask[0] = False  # Background is always 0

        # Apply lookup table directly (very fast)
        return keep_mask[labeled].astype(np.uint8)

    except ImportError:
        pass

    # Fallback: numpy-only implementation
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    mask_bool = mask.astype(bool)
    current_label = 0
    small_labels = []

    # Flood fill each component
    for start_y in range(h):
        for start_x in range(w):
            if mask_bool[start_y, start_x] and labels[start_y, start_x] == 0:
                current_label += 1
                stack = [(start_y, start_x)]
                labels[start_y, start_x] = current_label
                count = 1

                while stack:
                    y, x = stack.pop()
                    # Check 4-connected neighbors
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask_bool[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = current_label
                                stack.append((ny, nx))
                                count += 1

                if count < min_area:
                    small_labels.append(current_label)

    # Remove small regions in one operation
    if small_labels:
        remove_mask = np.isin(labels, small_labels)
        result = mask.copy()
        result[remove_mask] = 0
        return result

    return mask.copy()


def count_significant_regions(mask: np.ndarray, min_ratio: float = 0.01) -> int:
    """Count connected regions, ignoring tiny artifacts.

    Only counts regions whose area is at least min_ratio * largest_region_area.
    Uses a dilated mask to bridge 1px gaps before labeling.
    """
    if mask is None or mask.sum() == 0:
        return 0

    # Dilate by 1px to bridge only hairline gaps
    bridged = _numpy_dilate(mask.astype(np.uint8), 1)

    sizes = _label_region_sizes(bridged)
    if len(sizes) == 0:
        return 0

    largest = max(sizes)
    threshold = largest * min_ratio
    return sum(1 for s in sizes if s >= threshold)


def _label_region_sizes(mask: np.ndarray) -> list:
    """Return list of region sizes (pixel counts) for each connected component."""
    try:
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            return []
        return list(np.bincount(labeled.ravel())[1:])
    except ImportError:
        pass

    # Numpy fallback
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    mask_bool = mask.astype(bool)
    sizes = []

    for start_y in range(h):
        for start_x in range(w):
            if mask_bool[start_y, start_x] and labels[start_y, start_x] == 0:
                label = len(sizes) + 1
                stack = [(start_y, start_x)]
                labels[start_y, start_x] = label
                count = 1

                while stack:
                    y, x = stack.pop()
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask_bool[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = label
                                stack.append((ny, nx))
                                count += 1

                sizes.append(count)

    return sizes


def _numpy_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Dilate mask using numpy (expand the mask).

    Uses scipy binary_dilation when available for better performance,
    falls back to iterative numpy implementation.
    """
    try:
        from scipy.ndimage import binary_dilation
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        return binary_dilation(
            mask, structure=struct, iterations=iterations
        ).astype(np.uint8)
    except ImportError:
        pass

    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=0)
        center = padded[1:-1, 1:-1]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        dilated = center | up | down | left | right
        result = dilated.astype(np.uint8)
    return result


def _numpy_erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Erode mask using numpy (shrink the mask).

    Uses scipy binary_erosion when available for better performance,
    falls back to iterative numpy implementation.
    """
    try:
        from scipy.ndimage import binary_erosion
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        return binary_erosion(
            mask, structure=struct, iterations=iterations
        ).astype(np.uint8)
    except ImportError:
        pass

    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=0)
        center = padded[1:-1, 1:-1]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        eroded = center & up & down & left & right
        result = eroded.astype(np.uint8)
    return result


def _overlap_metrics(g1: QgsGeometry, g2: QgsGeometry) -> tuple[float, float]:
    """Return (IoU, containment) of two QgsGeometry polygons.

    IoU is intersection over union. Containment is intersection over the
    SMALLER polygon's area: it is high when one polygon is mostly inside
    the other, even if their sizes differ a lot (e.g. a mask truncated at
    a tile boundary vs the full mask from the overlapping neighbor tile,
    whose IoU stays low). Both are 0 on empty/degenerate input.
    """
    if g1 is None or g2 is None:
        return 0.0, 0.0
    if g1.isEmpty() or g2.isEmpty():
        return 0.0, 0.0
    try:
        intersection = g1.intersection(g2)
        if intersection is None or intersection.isEmpty():
            return 0.0, 0.0
        inter_area = intersection.area()
        if inter_area <= 0.0:
            return 0.0, 0.0
        a1 = g1.area()
        a2 = g2.area()
        union_area = a1 + a2 - inter_area
        min_area = min(a1, a2)
        iou = inter_area / union_area if union_area > 0.0 else 0.0
        containment = inter_area / min_area if min_area > 0.0 else 0.0
        return iou, containment
    except Exception:
        return 0.0, 0.0


def _ios_and_span(g1: QgsGeometry, g2: QgsGeometry,
                  a1: float | None = None,
                  a2: float | None = None) -> tuple[float, float]:
    """Return (IoS, overlap_span) for two polygons.

    IoS = intersection area over the SMALLER polygon's area (see _overlap_metrics).
    overlap_span = the longer side of the intersection's bounding box, in ground
    units. A genuine tile-seam split leaves a thin overlap strip whose longer
    dimension runs the length of the object along the seam, so a large
    overlap_span (comparable to the tile-overlap width) is the tell-tale of a
    seam split even when the overlap AREA (IoS) is tiny relative to a big object.
    Both are 0 on empty/degenerate input. ``a1``/``a2`` let a caller that
    already computed the polygon areas skip the O(vertices) recompute
    (QgsGeometry.area() is not cached).
    """
    if g1 is None or g2 is None or g1.isEmpty() or g2.isEmpty():
        return 0.0, 0.0
    try:
        inter = g1.intersection(g2)
        if inter is None or inter.isEmpty():
            return 0.0, 0.0
        inter_area = inter.area()
        if inter_area <= 0.0:
            return 0.0, 0.0
        if a1 is None:
            a1 = g1.area()
        if a2 is None:
            a2 = g2.area()
        min_area = a1 if a1 < a2 else a2
        ios = inter_area / min_area if min_area > 0.0 else 0.0
        bb = inter.boundingBox()
        span = bb.width() if bb.width() > bb.height() else bb.height()
        return ios, span
    except Exception:
        return 0.0, 0.0


def _buffer_square_corners(g: "QgsGeometry", dist: float) -> "QgsGeometry | None":
    """Buffer with a MITRE join so square corners stay square.

    Enum homes differ across our supported QGIS range (Qgis.JoinStyle since
    3.30, flat QgsGeometry.JoinStyleMiter before); resolve defensively and fall
    back to the plain round-join buffer rather than fail the refine step.
    """
    cap = getattr(getattr(Qgis, "EndCapStyle", None), "Round", None)
    join = getattr(getattr(Qgis, "JoinStyle", None), "Miter", None)
    if cap is None or join is None:
        cap = getattr(QgsGeometry, "CapRound", None)
        join = getattr(QgsGeometry, "JoinStyleMiter", None)
    if cap is not None and join is not None:
        try:
            return g.buffer(dist, 8, cap, join, 2.0)
        except (TypeError, AttributeError):
            pass
    return g.buffer(dist, 8)


def apply_geometry_refinement(
    geom: "QgsGeometry",
    *,
    simplify_tol: float = 0.0,
    smooth: bool = False,
    expand_dist: float = 0.0,
    fill_holes: bool = False,
    open_dist: float = 0.0,
    ortho: bool = False,
    ortho_tol: float = 0.0,
) -> "QgsGeometry":
    """Geometry-level refine for Automatic-review WHOLE objects (no mask here).

    The Manual mask-refine controls (simplify / round corners / expand-shrink /
    fill holes) act on a pixel mask; in the Automatic review the masks live on
    the worker and only merged polygons remain, so the same knobs are applied
    directly to the geometry:

      - fill_holes: drop every interior ring (native removeInteriorRings).
      - open_dist: morphological opening (shrink then grow back) that removes
        thin attached fringe / tendrils narrower than 2*open_dist, ground units.
      - simplify_tol: Douglas-Peucker simplify to a ground-unit tolerance.
      - ortho + ortho_tol: "Right angles" regularizer for man-made shapes
        (native QgsGeometry.orthogonalize snaps edges to 90 degrees). A raw
        mask outline is ALREADY all right-angle stair steps, so it must be
        de-staircased first: the simplify pass runs at max(simplify_tol,
        ortho_tol) when ortho is on (ortho_tol ~ 2.5 detection px, the
        regularization literature's simplify-then-orthogonalize tolerance).
      - expand_dist: buffer out (positive) or shrink in (negative), ground units.
      - smooth: Chaikin round the corners (native QgsGeometry.smooth).

    Order: fill, then open (strip fringe), then simplify, then right angles,
    then expand, then a final smooth. Best-effort: any failed step is skipped
    and the geometry so far is kept, so one degenerate object never breaks a
    refresh.
    """
    if geom is None or geom.isEmpty():
        return geom
    g = geom
    if ortho and ortho_tol > 0.0:
        simplify_tol = max(simplify_tol or 0.0, ortho_tol)
    if fill_holes:
        try:
            r = g.removeInteriorRings(-1.0)
            if isinstance(r, QgsGeometry) and not r.isEmpty():
                g = r
        except (AttributeError, TypeError, ValueError):
            pass
    if open_dist and open_dist > 0.0:
        # Morphological opening: shrink by open_dist then grow back. Thin fringe
        # / tendrils narrower than 2*open_dist vanish while the main shape keeps
        # its true size. buffer(-d) can empty a genuinely thin object; guard on
        # isEmpty so it is skipped, never destroyed.
        try:
            r = g.buffer(-open_dist, 8).buffer(open_dist, 8)
            if r is not None and not r.isEmpty():
                g = r
        except Exception:  # noqa: BLE001 -- refine is best-effort  # nosec B110
            pass
    if simplify_tol and simplify_tol > 0.0:
        try:
            r = g.simplify(simplify_tol)
            if r is not None and not r.isEmpty():
                g = r
        except Exception:  # noqa: BLE001 -- refine is best-effort  # nosec B110
            pass
    if ortho:
        # Tiny snap tolerance, plenty of iterations, and only nudge segments
        # already within 15 degrees of 0/90 (a genuinely diagonal wall is
        # left alone).
        try:
            r = g.orthogonalize(1.0e-8, 1000, 15.0)
            if r is not None and not r.isEmpty():
                g = r
        except Exception:  # noqa: BLE001 -- refine is best-effort  # nosec B110
            pass
    if expand_dist and expand_dist != 0.0:
        try:
            # With Right angles on, a round-join buffer would re-round every
            # corner the ortho step just squared; a mitre join preserves them.
            r = _buffer_square_corners(g, expand_dist) if ortho else g.buffer(expand_dist, 8)
            if r is not None and not r.isEmpty():
                g = r
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    if smooth:
        try:
            # minimumDistance = the simplify tolerance: a segment already at
            # sub-tolerance scale has no visible corner to round, so splitting
            # it would only mint invisible vertices.
            min_dist = simplify_tol if simplify_tol and simplify_tol > 0.0 else -1.0
            r = g.smooth(1, 0.25, min_dist, 120.0)
            if r is not None and not r.isEmpty():
                g = r
                # Vertex diet: one Chaikin pass near-doubles the vertex count,
                # which is pure render weight once thousands of review objects
                # are on canvas (every pan/zoom repaint reads every vertex).
                # Round corners is now a smart DEFAULT for vegetation prompts,
                # so the weight must be paid off here: a post-smooth simplify
                # at the same tolerance strips the sub-tolerance points the
                # rounding added while keeping every curve big enough to see.
                if simplify_tol and simplify_tol > 0.0:
                    r2 = g.simplify(simplify_tol)
                    if r2 is not None and not r2.isEmpty():
                        g = r2
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    return g


def apply_right_angles(geom: "QgsGeometry", destair_tol: float = 0.0) -> "QgsGeometry":
    """"Right angles" for the Manual pipeline: orthogonalize a polygon that
    came straight from a mask. A raw mask outline is ALREADY all right-angle
    stair steps, so orthogonalize alone is a no-op: when ``destair_tol`` is
    given, the outline is first simplified up to that tolerance (callers pass
    it only when their own simplify was weaker). Best-effort: returns the
    input on any failure, never raises."""
    if geom is None or geom.isEmpty():
        return geom
    g = geom
    if destair_tol and destair_tol > 0.0:
        try:
            r = g.simplify(destair_tol)
            if r is not None and not r.isEmpty():
                g = r
        except Exception:  # noqa: BLE001 -- refine is best-effort  # nosec B110
            pass
    try:
        r = g.orthogonalize(1.0e-8, 1000, 15.0)
        if r is not None and not r.isEmpty():
            g = r
    except Exception:  # noqa: BLE001 -- refine is best-effort
        return geom
    return g


def shape_polygon_geometry(
    geom: "QgsGeometry",
    mupp: float,
    simplify_px: int = 0,
    smooth: bool = False,
    expand_px: int = 0,
    fill_holes: bool = False,
    ortho: bool = False,
) -> "QgsGeometry":
    """Apply the Manual refine controls to an EXISTING polygon geometry.

    The Refine-in-Manual handoff edits detections that have no source mask
    (their shape came from the cloud run), so the mask-space refinement
    pipeline cannot run. This is its geometry-space twin: same controls,
    same order (fill holes, expand/contract, simplify, right angles, round
    corners), with pixel-denominated controls converted to ground units via
    ``mupp`` (map units per pixel of the source raster). Callers keep the
    PRISTINE geometry and re-shape from it on every settings change, so the
    operation stays non-destructive. Best-effort: returns the input on any
    failure, never raises.
    """
    if geom is None or geom.isEmpty() or not mupp or mupp <= 0:
        return geom
    g = QgsGeometry(geom)
    try:
        if fill_holes:
            parts = g.asMultiPolygon() if g.isMultipart() else [g.asPolygon()]
            shells = [[rings[0]] for rings in parts if rings]
            if shells:
                r = QgsGeometry.fromMultiPolygonXY(shells)
                if r is not None and not r.isEmpty():
                    g = r
        if expand_px:
            r = g.buffer(expand_px * mupp, 8)
            if r is not None and not r.isEmpty():
                g = r
        # Same px -> tolerance scale as the Manual mask pipeline
        # (_compute_simplification_tolerance), so the spinbox value means the
        # same thing whether the polygon came from a local mask or the cloud.
        tolerance = simplify_px * mupp * 0.5 if simplify_px > 0 else 0.0
        if tolerance > 0:
            r = g.simplify(tolerance)
            if r is not None and not r.isEmpty():
                g = r
        if ortho:
            g = apply_right_angles(
                g, destair_tol=max(0.0, 3 * mupp * 0.5 - tolerance))
        if smooth:
            r = g.smooth(1, 0.25, -1.0, 120.0)
            if r is not None and not r.isEmpty():
                g = r
        if g.isGeosValid() is False:
            r = g.makeValid()
            if r is not None and not r.isEmpty():
                g = r
    except Exception:  # noqa: BLE001 -- refine is best-effort
        return geom
    return g if g is not None and not g.isEmpty() else geom


def suppress_redundant_hypotheses(
    items: "list[tuple[QgsGeometry, float]]",
    ios_threshold: float = 0.5,
    dup_ios_floor: float = 0.3,
    dup_centroid_frac: float = 0.35,
) -> "list[tuple[QgsGeometry, float]]":
    """Keep ONE detection hypothesis per region: greedy score-descending NMS.

    The cloud model emits overlapping hypotheses at several granularities for
    the same region (a whole industrial complex AND its individual roofs; a
    roof AND its sections). Keeping them all lets any IoS-based dedup UNION
    the granularities into one mega-object whose score is the MAX of its
    parts: a low-score complex-wide mask (shadow fringe included) swallows the
    high-score roofs and then survives the review confidence filter at THEIR
    score. Selecting instead of unioning fixes both the mega-blob and the
    fake-confidence promotion.

    A candidate is suppressed when it conflicts with an already kept,
    higher-scored geometry under the SAME conditions the merger would later
    use to fuse the pair (so nothing the merger could union within one tile
    survives to be unioned): IoS >= ios_threshold (intersection over the
    smaller area), or IoS >= dup_ios_floor with near-coincident centroids.

    Intended for the detections of ONE tile in SEPARATE/count mode, BEFORE
    they reach the merger. Do not use in MAP/continuous mode: coverage there
    is the union of hypotheses by design, and dropping a large low-score
    canopy/road mask that overlaps one small high-score fragment would lose
    real coverage.
    """
    if len(items) < 2:
        return list(items)
    order = sorted(items, key=lambda t: -t[1])
    kept: list[tuple[QgsGeometry, float]] = []
    kept_meta: list[tuple] = []  # (bbox, area, centroid, bbox_max_dim)

    def _meta(geom):
        bb = geom.boundingBox()
        c = geom.centroid().asPoint()
        return bb, geom.area(), c, max(bb.width(), bb.height())

    for geom, score in order:
        bb, area, cen, dim = _meta(geom)
        conflict = False
        for (kbb, karea, kcen, kdim), kept_pair in zip(kept_meta, kept):
            # Cheap bbox pre-filter: the true overlap can never exceed the bbox
            # overlap, so if even that is under the lowest applicable floor the
            # pair cannot conflict.
            iw = min(bb.xMaximum(), kbb.xMaximum()) - max(bb.xMinimum(), kbb.xMinimum())
            ih = min(bb.yMaximum(), kbb.yMaximum()) - max(bb.yMinimum(), kbb.yMinimum())
            if iw <= 0.0 or ih <= 0.0:
                continue
            small = area if area < karea else karea
            if small <= 0.0 or (iw * ih) / small < dup_ios_floor:
                continue
            inter = geom.intersection(kept_pair[0])
            ia = inter.area() if inter is not None and not inter.isEmpty() else 0.0
            ios = ia / small
            if ios >= ios_threshold:
                conflict = True
                break
            if ios >= dup_ios_floor:
                smax = dim if area <= karea else kdim
                if smax > 0.0:
                    dist = ((cen.x() - kcen.x()) ** 2 + (cen.y() - kcen.y()) ** 2) ** 0.5
                    if dist < dup_centroid_frac * smax:
                        conflict = True
                        break
        if not conflict:
            kept.append((geom, score))
            kept_meta.append((bb, area, cen, dim))
    return kept


def drop_covered_objects(
    items: "list[tuple[int, QgsGeometry, float]]",
    cover_threshold: float = 0.40,
) -> "list[tuple[int, QgsGeometry, float]]":
    """End-of-run redundancy sweep on the merger's WHOLE objects.

    An object mostly painted over by LARGER objects is a leftover partial
    reading (a strip/patch on a big roof, an interlocking cross-tile crop)
    that slipped past the pairwise dedup because no single pair cleared the
    IoS floors: the union of several neighbours covers it even though each
    one alone does not. Rendering it double-paints the big object and reads
    as debris. Drop it when the larger objects' combined overlap reaches
    cover_threshold of its area, UNLESS it outscores every covering
    neighbour (a genuinely better reading of a small building squeezed
    between sloppy big masks must survive: an overpainted partial reading
    scores below its covering host).

    items are (stable_id, geometry, score) triples as produced by
    IncrementalMerger.result_scored_ided(); order is preserved.

    Candidates come from a QgsSpatialIndex, not an all-pairs scan. The per
    candidate intersection+union GEOS work is still heavy on dense runs
    (thousands of small objects), so the interactive path drives this in
    time-sliced chunks via CoverSweep; this synchronous wrapper is used by the
    headless/MCP path (no UI to protect) and keeps one implementation.
    """
    sweep = CoverSweep(items, cover_threshold=cover_threshold)
    sweep.step(len(items))  # run to completion in one call
    return sweep.result()


class CoverSweep:
    """Step-able form of the end-of-run redundancy sweep (see
    ``drop_covered_objects`` for the algorithm and rationale). ``step(k)``
    processes up to ``k`` candidates and returns True once the whole sweep is
    done, so the finalize state machine can slice it across event-loop turns
    instead of freezing the GUI at the end of a dense run."""

    def __init__(
        self,
        items: "list[tuple[int, QgsGeometry, float]]",
        cover_threshold: float = 0.40,
    ) -> None:
        self._items = items
        self._threshold = cover_threshold
        self._n = len(items)
        self._i = 0
        self._keep = [True] * self._n
        self._done = self._n < 2
        self._metas: list = []
        self._index = QgsSpatialIndex()
        if self._done:
            return
        for pos, (sid, geom, score) in enumerate(items):
            try:
                bb, area = geom.boundingBox(), geom.area()
            except Exception:
                self._metas.append((None, 0.0))
                continue
            self._metas.append((bb, area))
            feat = QgsFeature(pos)
            feat.setGeometry(QgsGeometry.fromRect(bb))
            self._index.addFeature(feat)

    def step(self, max_items: int) -> bool:
        """Process up to ``max_items`` candidates; return True when finished."""
        if self._done:
            return True
        items, metas, index, keep = self._items, self._metas, self._index, self._keep
        threshold = self._threshold
        processed = 0
        while self._i < self._n and processed < max_items:
            i = self._i
            self._i += 1
            processed += 1
            bb, area = metas[i]
            if bb is None or area <= 0.0:
                continue
            covers = []
            best_cover_score = None
            for j in index.intersects(bb):
                if j == i:
                    continue
                jbb, jarea = metas[j]
                if jbb is None or jarea <= area:
                    continue
                iw = min(bb.xMaximum(), jbb.xMaximum()) - max(bb.xMinimum(), jbb.xMinimum())
                ih = min(bb.yMaximum(), jbb.yMaximum()) - max(bb.yMinimum(), jbb.yMinimum())
                if iw <= 0.0 or ih <= 0.0 or (iw * ih) / area < threshold * 0.5:
                    continue
                try:
                    inter = items[i][1].intersection(items[j][1])
                except Exception:  # nosec B112
                    continue
                if inter is None or inter.isEmpty():
                    continue
                covers.append(inter)
                s = items[j][2]
                if best_cover_score is None or s > best_cover_score:
                    best_cover_score = s
            if not covers:
                continue
            # Union the per-neighbour overlaps: two large neighbours overlapping
            # each other over this object must not double count.
            merged = covers[0]
            for g in covers[1:]:
                u = merged.combine(g)
                if u is not None and not u.isEmpty():
                    merged = u
            if merged.area() / area >= threshold:
                if best_cover_score is not None and items[i][2] > best_cover_score:
                    continue
                keep[i] = False
        if self._i >= self._n:
            self._done = True
        return self._done

    def result(self) -> "list[tuple[int, QgsGeometry, float]]":
        return [it for it, k in zip(self._items, self._keep) if k]


class IncrementalMerger:
    """Greedy non-max merging maintained online as tile fragments arrive.

    A single batch merge at the end of a run would leave the live preview showing
    raw, cut tile fragments until then. This keeps a running merged set instead:
    each add() folds a fragment in immediately, so the preview shows whole,
    stitched objects as tiles complete.

    A fragment unions into EVERY existing object it overlaps by at least the
    pair's threshold (IoS, intersection over the smaller area), fusing them all
    into one. IoS (not IoU) is used so the thin overlap a long object (road,
    river) leaves in the tile overlap strip still registers. Merging all matches
    (not just the best) is what fully stitches a long object whose pieces arrive
    out of tile order: a bridging fragment that touches two already-grown keepers
    fuses them into one.

    Size-aware gate (the anti-over-merge safety). The low merge_ios is meant only
    to stitch one large object cut by a tile boundary. But IoS divides by the
    SMALLER area, so a sliver of overlap between two small objects also clears it,
    and the bridging then chains distinct small neighbours (solar panels, cars,
    trees) into one blob even when there is clear space between them. The fix:
    an object shorter than ``seam_min_dim`` in both bbox dimensions is guaranteed
    to fit whole inside one tile (it is <= the inter-tile overlap strip), so it
    can only ever be a cross-tile DUPLICATE, never a seam-split half. Two pieces
    therefore merge at the low merge_ios only when BOTH could span a seam (both
    >= seam_min_dim); any pair involving a small object must instead clear the
    strict dedup_ios (a near-duplicate). Distinct small neighbours have IoS well
    below dedup_ios, so they stay apart, while a true duplicate of a small object
    (the same panel seen in two overlapping tiles) sits near IoS 1.0 and still
    merges. seam_min_dim = 0.0 disables the gate (every pair uses merge_ios, the
    original behaviour) for callers without a known tile size.

    Order-independent: union is commutative, so the result does not depend on the
    order tiles complete in.
    """

    def __init__(
        self,
        merge_ios: float = 0.15,
        dedup_ios: float = 0.5,
        seam_min_dim: float = 0.0,
        dup_ios_floor: float = 0.3,
        dup_centroid_frac: float = 0.35,
        seam_span_ios: float = 0.03,
        select_duplicates: bool = False,
        gsd: float = 0.0,
    ):
        self._merge_ios = merge_ios
        self._dedup_ios = dedup_ios
        self._seam_min_dim = seam_min_dim
        # Ground units per detection pixel. Sizes the one-pixel erosion that
        # separates jitter duplicates from real seam complements in the
        # additive-union select branch (see add()). 0.0 (unknown gsd) falls
        # back to a relative added-area floor instead of the erosion test.
        self._gsd = float(gsd)
        # SEPARATE/count mode: a matched group is EITHER redundant readings of
        # one footprint (cross-tile jitter duplicate, parent vs child
        # hypothesis) OR the pieces of one object cut by a tile seam. The
        # additive-union select branch in add() tells them apart PER MEMBER
        # with an erosion test on the member's added area: redundant members
        # are skipped (union would only add outline dilation or rebuild the
        # mega-blob the per-tile NMS killed; score comes only from
        # co-extensive or contributing members so a small high-score child
        # never promotes a big low-score parent), real complements are
        # stitched in (discarding one renders the object truncated flat along
        # the tile grid). MAP/continuous callers keep the plain union
        # (coverage is a union by design).
        self._select_duplicates = select_duplicates
        # Seam-span rescue for MISSED JOINS: a large object split ~50/50 by a
        # tile seam overlaps only inside the thin (~20%) overlap strip, so its
        # IoS can fall below merge_ios and plain IoS merging leaves a straight
        # seam line through the object. When BOTH pieces are seam-eligible (each
        # big enough to be a tile-cut half) and the overlap REGION spans at least
        # a seam-strip width in its longer dimension (the tell-tale of a seam
        # split, not an incidental corner touch), merge them at this much lower
        # IoS floor. Gated to seam-eligible pairs, a finite positive
        # seam_min_dim AND union mode (select_duplicates=False), so it never
        # fuses small distinct neighbours, never fires when the size gate is
        # off, and never lets a common-wall sliver between two DISTINCT large
        # buildings reach the selection-mode stitch (see add()).
        self._seam_span_ios = seam_span_ios
        # Robust same-object dedup: a duplicate the cloud model drew slightly differently
        # across two tiles can fall below dedup_ios yet still be ONE object. Treat
        # it as a duplicate when the overlap clears dup_ios_floor AND the
        # centroids sit within dup_centroid_frac of the smaller object's size.
        # Conservative: distinct neighbours have far-apart centroids, so this
        # never fuses them. Applies to the non-seam (small-object) branch only.
        self._dup_ios_floor = dup_ios_floor
        self._dup_centroid_frac = dup_centroid_frac
        self._index = QgsSpatialIndex()
        self._keepers: dict[int, "QgsGeometry | None"] = {}
        # Representative score per keeper: the MAX of its constituent fragments'
        # scores, so a strong object survives the review confidence filter even
        # if one seam-half scored low (that is exactly the "confidence cuts
        # buildings" bug: filtering fragments before stitching drops the weak
        # half). Merges take the max across the candidate and every keeper it
        # absorbs.
        self._scores: dict[int, float] = {}
        self._next_id = 0
        # Ids of the non-retired keepers, kept in lockstep with _keepers/_scores
        # (add on _insert, drop on retirement). result()/result_scored() walk this
        # instead of scanning every id ever inserted, so the retired None slots
        # that pile up over a dense run cost nothing. Stored as a
        # dict used as an INSERTION-ORDERED set (keys only) so result order is
        # byte-identical to the old _keepers.items() scan (an object retired then
        # re-inserted lands last in both). Pure bookkeeping: the merge LOGIC is
        # unchanged. The candidate loop in add() deliberately stays driven by
        # _index.intersects()+_keepers.get(), so a retired fid still returned by
        # the index is skipped exactly as before.
        self._live_ids: dict[int, None] = {}

    def _is_seam_eligible(self, geom: "QgsGeometry") -> bool:
        """True if geom is large enough to possibly be cut by a tile boundary.

        Below seam_min_dim in both bbox dimensions an object always fits whole in
        one tile, so it can only be a duplicate, not a seam-split half. When the
        gate is off (seam_min_dim <= 0) everything is treated as seam-eligible.
        """
        if self._seam_min_dim <= 0.0:
            return True
        bb = geom.boundingBox()
        return max(bb.width(), bb.height()) >= self._seam_min_dim

    def add(self, geom: "QgsGeometry", score: float = 0.0) -> None:
        if geom is None or geom.isEmpty():
            return
        # Find every existing object this fragment overlaps enough to be part of.
        # Merging ALL of them (not just the best) stitches a long object whose
        # pieces arrive out of tile order. The size-aware threshold keeps that
        # from chaining distinct SMALL neighbours: a low merge_ios bridge is
        # allowed only when both pieces are large enough to be tile-cut halves;
        # any pair with a small object needs the strict dedup_ios (true duplicate).
        cand_bbox = geom.boundingBox()
        cand_area = geom.area()
        cand_seam = self._is_seam_eligible(geom)
        # Seam-span rescue is only armed when the size gate is a finite positive
        # span (a known GSD) AND the caller wants unions (map/continuous). In
        # selection mode it stays off: two DISTINCT adjacent buildings sharing a
        # long thin overlap (common wall, shadow fringe) have a large overlap
        # span at a tiny IoS, exactly what the rescue accepts, and the stitch
        # branch would then fuse them. Seam halves of one big object clear the
        # plain merge_ios instead (their overlap strip is a large fraction of
        # the smaller half up to ~500 m objects).
        seam_span_armed = (
            0.0 < self._seam_min_dim < float("inf") and not self._select_duplicates
        )
        matches = []
        # dict.fromkeys dedupes the candidate fids: a merge reuses the primary
        # keeper's fid (see the branches below), so after that reuse the index
        # holds TWO bbox entries under that fid (the stale retired bbox plus the
        # fresh union bbox) until the next compaction. Without this dedupe
        # intersects() would return the reused fid twice within one add(); the
        # second occurrence now sees a LIVE keeper and would self-merge the
        # object into itself (or, in the stitch branch, combine() a keeper that
        # the first occurrence already retired to None). Processing each fid at
        # most once keeps matches unique and the merge logic byte-unchanged.
        for fid in dict.fromkeys(self._index.intersects(cand_bbox)):
            keeper = self._keepers.get(fid)
            if keeper is None:
                continue
            both_large = cand_seam and self._is_seam_eligible(keeper)
            threshold = self._merge_ios if both_large else self._dedup_ios
            # The lowest IoS we might still accept, used only to size the cheap
            # bbox pre-filter below. The non-seam branch can merge a near-
            # coincident duplicate down at dup_ios_floor (centroid test); the
            # seam branch can rescue a seam split down at seam_span_ios. The
            # pre-filter must not prune below whichever floor applies.
            if both_large:
                min_threshold = (
                    min(threshold, self._seam_span_ios) if seam_span_armed else threshold
                )
            else:
                min_threshold = min(threshold, self._dup_ios_floor)
            # Cheap bbox pre-filter before the costly intersection(): the real
            # geometry overlap can never exceed the bbox overlap, so if even
            # bbox_overlap / smaller_area is below the threshold the pair cannot
            # merge. This skips intersection() for the many bbox-adjacent-but-
            # distinct neighbours in dense scenes (packed panels, urban roofs),
            # the dominant main-thread cost of a big run. No false negatives:
            # bbox overlap is an upper bound on the true overlap.
            kb = keeper.boundingBox()
            iw = min(cand_bbox.xMaximum(), kb.xMaximum()) - max(cand_bbox.xMinimum(), kb.xMinimum())
            ih = min(cand_bbox.yMaximum(), kb.yMaximum()) - max(cand_bbox.yMinimum(), kb.yMinimum())
            if iw <= 0.0 or ih <= 0.0:
                continue
            keeper_area = keeper.area()
            min_area = cand_area if cand_area < keeper_area else keeper_area
            if min_area <= 0.0 or (iw * ih) / min_area < min_threshold:
                continue
            ios, span = _ios_and_span(geom, keeper, a1=cand_area, a2=keeper_area)
            if ios >= threshold:
                matches.append(fid)
                continue
            # Seam-span rescue (large pairs only): a true tile-seam split has a
            # low IoS but its overlap strip runs the object's length along the
            # seam, so its span reaches a seam-strip width. Distinct large
            # neighbours are spatially apart, so their overlap span is small.
            # The span requirement takes a tolerance: mask edges land a pixel
            # or two short of the tile border and the simplify pass shaves the
            # strip further, so a genuine seam strip measures slightly UNDER
            # the theoretical overlap width; requiring the full width lets
            # near-exact seam splits through as overlapping duplicates. An
            # incidental corner touch between distinct neighbours stays far
            # below this scale.
            if both_large and seam_span_armed:
                if (ios >= self._seam_span_ios
                        and span >= 0.85 * self._seam_min_dim):  # noqa: W503
                    matches.append(fid)
                    continue
            # Robust same-object dedup (non-seam only): decent overlap + nearly
            # coincident centroids => the same object split/redrawn across tiles,
            # so it must not be counted twice.
            if (not both_large) and ios >= self._dup_ios_floor:
                smaller_bb = cand_bbox if cand_area <= keeper_area else kb
                smax = max(smaller_bb.width(), smaller_bb.height())
                if smax > 0.0:
                    cc = geom.centroid().asPoint()
                    kc = keeper.centroid().asPoint()
                    dist = ((cc.x() - kc.x()) ** 2 + (cc.y() - kc.y()) ** 2) ** 0.5
                    if dist < self._dup_centroid_frac * smax:
                        matches.append(fid)

        if matches and self._select_duplicates:
            # A matched group is one of two things, and the ADDITIVE UNION
            # below tells them apart PER MEMBER instead of by a single
            # union/largest ratio (a single ratio test discards a seam
            # complement that extends the object by less than the ratio margin,
            # which leaves big buildings with a flat wall along the tile grid
            # where the discarded complement would have carried the real width):
            #  - REDUNDANT readings of the same footprint (cross-tile jitter
            #    duplicate, parent/child hypothesis): the member adds no real
            #    area, unioning it would only dilate the outline or rebuild
            #    the mega-blob the per-tile NMS killed -> skipped.
            #  - PIECES of ONE object cut by a tile seam (or emitted as
            #    sections): the member contributes real new area past the
            #    growing shape -> unioned (the stitch; discarding it renders
            #    the object truncated along the tile grid).
            # The discriminator: the member's difference against the growing
            # shape must SURVIVE a one-pixel erosion. A duplicate's difference
            # is a sub-pixel jitter ring that erodes to nothing; a genuine
            # complement's difference is real width that survives, so it heals
            # the flat wall without growing the outline or rebuilding a
            # mega-blob. Cross-tile parent-vs-children
            # is unchanged: the largest member already won before, and a
            # child inside it adds no surviving difference.
            # The union carries the PRIMARY keeper's fid (lowest among the
            # matched keepers, deterministic) so an object keeps ONE stable id
            # across every merge that grows it (the id drives the live per-object
            # colour and the refine cache in the repaint loop). All matched
            # keepers are retired; the primary's fid is reused for the union.
            primary_fid = min(matches)
            members = [(geom, float(score))]
            for fid in matches:
                keeper = self._keepers[fid]
                if keeper is not None:
                    members.append((keeper, self._scores.get(fid, 0.0)))
                self._keepers[fid] = None
                self._live_ids.pop(fid, None)
            members.sort(key=lambda t: t[0].area(), reverse=True)
            largest_area = members[0][0].area()
            current = members[0][0]
            contributing = {0}
            for i in range(1, len(members)):
                g = members[i][0]
                diff = g.difference(current)
                if diff is None or diff.isEmpty():
                    continue
                if self._gsd > 0.0:
                    eroded = diff.buffer(-self._gsd, 5)
                    if eroded is None or eroded.isEmpty() or eroded.area() <= 0.0:
                        continue
                elif diff.area() < 0.02 * largest_area:
                    # Unknown gsd (no pixel size to erode by): fall back to a
                    # small relative floor so pure-jitter rings still never
                    # dilate the outline.
                    continue
                union = current.combine(g)
                if union is not None and not union.isEmpty():
                    current = union
                    contributing.add(i)
            # Score: the MAX over members that are co-extensive with the
            # largest (area >= half of it: a redundant full reading) or that
            # actually contributed area (a stitched seam half). A small
            # high-score child that added nothing never promotes the object;
            # a contributing half keeps its score (filtering a half before
            # stitching is the "confidence cuts buildings" bug).
            floor_area = 0.5 * largest_area
            best_score = max(
                s for i, (g, s) in enumerate(members)
                if i in contributing or g.area() >= floor_area
            )
            self._insert(current, best_score, fid=primary_fid)
        elif matches:
            combined = geom
            best_score = float(score)
            retired = []
            for fid in matches:
                keeper = self._keepers[fid]
                union = combined.combine(keeper)
                if union is not None and not union.isEmpty():
                    combined = union
                    # Only retire a keeper whose geometry was actually absorbed;
                    # retiring on a failed union would silently drop its area.
                    # The merged object keeps the MAX score of everything in it.
                    if self._scores.get(fid, 0.0) > best_score:
                        best_score = self._scores.get(fid, 0.0)
                    self._keepers[fid] = None
                    self._live_ids.pop(fid, None)
                    retired.append(fid)
            if retired:
                # Reuse the lowest RETIRED keeper's fid (primary) so the stitched
                # object keeps one stable id across merges. min(retired), not
                # min(matches): a matched fid whose combine() failed stays LIVE,
                # so reusing it would collide two live keepers on one id.
                self._insert(combined, best_score, fid=min(retired))
            else:
                # No union succeeded (all combine() calls failed); keep the
                # fragment as its own object rather than losing it.
                self._insert(geom, float(score))
        else:
            self._insert(geom, float(score))

    def _insert(self, geom: "QgsGeometry", score: float = 0.0, fid: "int | None" = None) -> None:
        # fid=None mints a genuinely new object from _next_id (unchanged path).
        # An explicit fid REUSES a retired keeper's id for a merge product: the
        # object keeps its stable colour/refine id across the stitch. _next_id is
        # NOT bumped on reuse (it stays a strict high-water mark, so a future new
        # object can never collide with a reused id). Reuse re-adds a bbox entry
        # under an id whose stale retired bbox still sits in _index until the next
        # compaction; add()'s candidate dedupe absorbs that duplicate.
        use_id = self._next_id if fid is None else fid
        feat = QgsFeature(use_id)
        feat.setGeometry(geom)
        self._index.insertFeature(feat)
        self._keepers[use_id] = geom
        self._scores[use_id] = float(score)
        self._live_ids[use_id] = None
        if fid is None:
            self._next_id += 1
        self._maybe_compact()

    def _maybe_compact(self) -> None:
        """Drop the retired None slots from _keepers/_scores AND rebuild _index
        from the survivors once they dominate the dicts (retired > 4x live), so a
        dense run's memory and add()'s intersects() scan both track the live
        object count instead of every fragment ever inserted.
        Behavior-preserving: _next_id is never reset (so it can never collide
        with a retired fid still living in _index), and add() reads keepers via
        .get(), so a dropped key reads identically to a None slot. _live_ids is
        the surviving id set, so the rebuilt dicts hold exactly the live objects
        in the same order."""
        live = len(self._live_ids)
        if len(self._keepers) - live <= 4 * live:
            return
        self._keepers = {fid: self._keepers[fid] for fid in self._live_ids}
        self._scores = {fid: self._scores[fid] for fid in self._live_ids}
        # Rebuild the spatial index from the survivors too: retired fids are
        # skipped by add() anyway (keepers.get() is None), but their bboxes
        # otherwise pile up and add()'s intersects() scan grows with every
        # fragment ever inserted instead of the live count. _next_id is still
        # never reset, so fresh ids can never collide with pruned ones. Each
        # feature carries the CURRENT merged geometry (_keepers[fid]) under its
        # own live fid, so the index stays consistent with the keeper dicts.
        index = QgsSpatialIndex()
        for fid in self._live_ids:
            feat = QgsFeature(fid)
            feat.setGeometry(self._keepers[fid])
            index.insertFeature(feat)
        self._index = index

    def result(self) -> list:
        """Current merged objects (one geometry per stitched object)."""
        return [self._keepers[fid] for fid in self._live_ids]

    def result_scored(self) -> list:
        """Current merged objects as (geometry, score) pairs.

        score is the representative (MAX) score of the object's constituent
        fragments, so the review confidence filter acts on WHOLE objects and a
        strong object survives even if one of its seam halves scored low.
        """
        return [
            (self._keepers[fid], self._scores.get(fid, 0.0))
            for fid in self._live_ids
        ]

    def result_scored_ided(self) -> list:
        """Current merged objects as (stable_id, geometry, score) triples.

        stable_id is the keeper's fid, assigned once when the object first
        appears and preserved across later merges (a merge folds a fragment
        INTO an existing keeper, reusing its id; only NEW objects get a fresh
        id). A caller that keys a per-object colour on it keeps an object's hue
        fixed as more tiles arrive, unlike the positional index which reshuffles
        whenever an earlier keeper retires.
        """
        return [
            (fid, self._keepers[fid], self._scores.get(fid, 0.0))
            for fid in self._live_ids
        ]


# ---------------------------------------------------------------------------
# File export (additive driver support for the Library's direct Export)
# ---------------------------------------------------------------------------

# Vector drivers the direct export supports. GPKG is the default and keeps the
# full layer conventions (minimal schema, run-level metadata, embedded style);
# the other drivers cannot embed a style, so they are written style-less.
EXPORT_DRIVERS = ("GPKG", "GeoJSON", "ESRI Shapefile", "KML")

_DRIVER_EXTENSIONS = {
    "GPKG": ".gpkg",
    "GeoJSON": ".geojson",
    "ESRI Shapefile": ".shp",
    "KML": ".kml",
}


def driver_extension(driver: str) -> str:
    """File extension (with dot) for a supported export driver."""
    return _DRIVER_EXTENSIONS.get(driver, ".gpkg")


def export_geometries_to_file(
    geoms: list,
    crs,
    output_path: str,
    driver: str = "GPKG",
    source_layer_name: str = "",
    layer_name: str | None = None,
):
    """Write polygon geometries to a vector file and return the loaded layer.

    Additive export helper used by the Library's direct Export (and reusable by
    any caller that already has final geometries). Keeps the export layer
    conventions: minimal per-feature schema (editable ``label`` + geodesic
    ``area_m2``), geometries repaired with makeValid before save (repair, never
    silently drop). For GPKG the run-level provenance metadata and the style
    are stored INTO the file; the other drivers cannot embed a style, so they
    skip it. KML is always written in EPSG:4326 (the format mandates it).

    Args:
        geoms:             QgsGeometry list (any polygonal type).
        crs:               QgsCoordinateReferenceSystem of the geometries.
        output_path:       Destination file path (extension decides nothing;
                           the ``driver`` does).
        driver:            One of EXPORT_DRIVERS. Default "GPKG".
        source_layer_name: Raster name recorded in the GPKG provenance.
        layer_name:        Layer name inside the file; defaults to the file stem.

    Returns:
        The loaded QgsVectorLayer on success (NOT added to the project), or
        None on failure.
    """
    import os

    from qgis.core import (
        QgsCoordinateReferenceSystem,
        QgsCoordinateTransform,
        QgsField,
        QgsProject,
        QgsVectorFileWriter,
        QgsVectorLayer,
    )

    from .layer_conventions import (
        apply_output_conventions,
        make_area_measurer,
        make_committed_renderer,
        repair_polygon,
        to_multipolygon,
    )

    if not geoms:
        return None
    if driver not in EXPORT_DRIVERS:
        driver = "GPKG"

    # Field-type enums: Qt6/PyQt6 (QGIS 4) scoped QMetaType vs Qt5 QVariant.
    field_str = field_type_string()
    field_dbl = field_type_double()

    stem = os.path.splitext(os.path.basename(output_path))[0]
    name = layer_name or stem or "detections"

    temp_layer = QgsVectorLayer("MultiPolygon", name, "memory")
    if not temp_layer.isValid():
        return None
    temp_layer.setCrs(crs)
    pr = temp_layer.dataProvider()
    pr.addAttributes([
        QgsField("label", field_str),
        QgsField("area_m2", field_dbl),
    ])
    temp_layer.updateFields()

    measurer = make_area_measurer(crs)
    feats = []
    for geom in geoms:
        if geom is None or geom.isEmpty():
            continue
        geom = to_multipolygon(repair_polygon(geom) or geom)
        if geom is None or geom.isEmpty():
            continue
        feat = QgsFeature(temp_layer.fields())
        feat.setGeometry(geom)
        try:
            area = float(measurer.measureArea(geom))
        except (RuntimeError, AttributeError):
            area = float(geom.area())
        feat.setAttributes(["", area])
        feats.append(feat)
    if not feats:
        return None
    pr.addFeatures(feats)
    temp_layer.updateExtents()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as err:
            QgsMessageLog.logMessage(
                "Export: cannot create directory: {}".format(err),
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return None

    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = driver
    options.fileEncoding = "UTF-8"
    options.layerName = name
    if driver == "KML":
        # KML is defined on WGS84 only; write reprojected coordinates.
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        if crs is not None and crs.isValid() and crs != wgs84:
            options.ct = QgsCoordinateTransform(crs, wgs84, QgsProject.instance())

    error = QgsVectorFileWriter.writeAsVectorFormatV3(
        temp_layer, output_path, QgsProject.instance().transformContext(), options,
    )
    if error[0] != QgsVectorFileWriter.WriterError.NoError:
        QgsMessageLog.logMessage(
            "Export failed ({}): {}".format(driver, error[1]),
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        return None

    result_layer = QgsVectorLayer(output_path, name, "ogr")
    if not result_layer.isValid():
        result_layer = QgsVectorLayer(
            "{}|layername={}".format(output_path, name), name, "ogr")
    if not result_layer.isValid():
        QgsMessageLog.logMessage(
            "Export: file saved but could not be loaded back",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        return None

    result_layer.setRenderer(make_committed_renderer())
    if driver == "GPKG":
        # Provenance metadata + style embedded in the GeoPackage, exactly like
        # the standard review export. Other drivers cannot persist either.
        apply_output_conventions(result_layer, source_layer_name)
    return result_layer
