from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

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
from qgis.PyQt.QtCore import QMetaObject, QObject, pyqtSlot  # noqa: E402

from .merger import IncrementalMerger  # noqa: E402,F401
from .polygon_packing import pack_disjoint_crops  # noqa: E402
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

            geom = _finish_polygon(geojson_geom, simplify_tolerance)
            if geom is not None:
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


def _finish_polygon(geojson: dict, simplify_tolerance: float) -> QgsGeometry | None:
    """One shapes() output -> a valid, optionally simplified QgsGeometry.

    Shared by the per-mask and the packed polygonizers so the two can never
    drift: the validity gate and the simplify-then-revalidate ladder are the
    only thing that decides what a mask's pixels become.
    """
    geom = _geojson_to_geometry(geojson)
    if geom is None or geom.isEmpty() or not geom.isGeosValid():
        return None
    if simplify_tolerance <= 0:
        return geom
    # simplify() can return a self-intersecting result that the isGeosValid()
    # gate above already passed; re-validate so an invalid geometry never
    # reaches the live merger. Repair when possible, otherwise keep the
    # unsimplified (valid) geometry.
    simplified = geom.simplify(simplify_tolerance)
    if simplified is None:
        return geom
    if not simplified.isEmpty() and simplified.isGeosValid():
        return simplified
    fixed = simplified.makeValid()
    if fixed is not None and not fixed.isEmpty() and fixed.isGeosValid():
        return fixed
    return geom


def masks_to_polygons_packed(
    crops: list,
    transform_info: dict,
    full_shape: tuple[int, int],
    simplify_tolerance: float = 0.0,
    max_side: int | None = None,
) -> list[list[QgsGeometry]]:
    """Polygonize many crops of ONE tile grid, batching the rasterio calls.

    ``crops`` is a list of ``(crop_array, (row0, col0))``: the boolean crop and
    the pixel position of its top-left corner in the ``full_shape`` grid, the
    same pair mask_to_polygons takes as ``pixel_offset``. Returns one geometry
    list per input crop, in input order.

    Why: shapes() carries a fixed GDAL setup cost of a few ms per call whatever
    it is handed, and a dense tile calls it once per mask, well over a hundred
    times. Crops whose boxes are disjoint are painted into a single int32 LABEL
    raster and polygonized in ONE call; shapes() hands back the label with each
    polygon, so every polygon still maps to exactly one crop, and a label's
    polygon is its own cells whether or not another label sits next to it.

    Batching is not free: shapes() walks the whole grid it is handed, and a
    pack's grid is the bounding box of the crops in it, so spread-out crops
    drag in background nobody needs. What keeps that bounded is max_side, which
    caps how big one pack can get however far apart its crops are. Left None it
    is the cap in force, resolved by the packer on every call.

    Output matches calling mask_to_polygons per crop: same pixels, same pixel
    scale, same validity ladder. The two build their affine from different
    origins, so coordinates can differ in the last bits of a float, orders
    below the simplify tolerance and the merger's thresholds. Only the number
    of calls changes.
    """
    out: list[list[QgsGeometry]] = [[] for _ in crops]
    if not crops:
        return out
    bbox = transform_info.get("bbox")
    if not bbox:
        return out
    try:
        from rasterio.features import shapes as get_shapes
        from rasterio.transform import from_bounds as transform_from_bounds
    except ImportError:
        # Automatic runs before any local install, and a QGIS that bundles no
        # rasterio (every Linux packaging) would otherwise raise once per crop,
        # get swallowed by the per-tile catch, and hand the user a billed run
        # with nothing on the map. mask_to_polygons already falls back here; the
        # batched path has to as well.
        # Note the offset order: crops carry (row0, col0), mask_to_polygons_fallback
        # reads pixel_offset as (col0, row0).
        return [
            mask_to_polygons_fallback(
                crop, transform_info, simplify_tolerance, (col0, row0), full_shape)
            for crop, (row0, col0) in crops
        ]

    # polygon_exporter bbox convention: (minx, MAXX, miny, MAXY). The pixel
    # scale comes from the FULL grid so a crop never rescales (see
    # mask_to_polygons).
    minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
    full_h, full_w = int(full_shape[0]), int(full_shape[1])
    px_w = (maxx - minx) / max(full_w, 1)
    px_h = (maxy - miny) / max(full_h, 1)

    boxes = []
    for crop, (row0, col0) in crops:
        h, w = int(crop.shape[0]), int(crop.shape[1])
        boxes.append((int(row0), int(row0) + h - 1, int(col0), int(col0) + w - 1))

    try:
        for indices, (br0, br1, bc0, bc1) in pack_disjoint_crops(boxes, max_side):
            lab = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), np.int32)
            for label, i in enumerate(indices, start=1):
                crop, _origin = crops[i]
                r0, _r1, c0, _c1 = boxes[i]
                ro, co = r0 - br0, c0 - bc0
                # `where` must be boolean: numpy refuses a uint8 selector under
                # safe casting, and the tile pipeline hands uint8 crops (see
                # fill_small_holes). Any non-zero pixel is object.
                stamp = crop if crop.dtype == np.bool_ else crop != 0
                np.copyto(lab[ro:ro + crop.shape[0], co:co + crop.shape[1]],
                          np.int32(label), where=stamp)
            pack_minx = minx + bc0 * px_w
            pack_maxy = maxy - br0 * px_h
            transform = transform_from_bounds(
                pack_minx, pack_maxy - lab.shape[0] * px_h,
                pack_minx + lab.shape[1] * px_w, pack_maxy,
                lab.shape[1], lab.shape[0],
            )
            for geojson_geom, value in get_shapes(
                lab, mask=lab > 0, connectivity=4, transform=transform
            ):
                label = int(value)
                if label <= 0 or label > len(indices):
                    continue
                geom = _finish_polygon(geojson_geom, simplify_tolerance)
                if geom is not None:
                    out[indices[label - 1]].append(geom)
        return out
    except Exception as exc:  # noqa: BLE001 - fall back to the per-mask path
        QgsMessageLog.logMessage(
            f"Packed polygonize failed ({exc}); falling back to per-mask",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        return [
            mask_to_polygons(
                crop, transform_info, simplify_tolerance=simplify_tolerance,
                pixel_offset=(int(col0), int(row0)), full_shape=full_shape,
            )
            for crop, (row0, col0) in crops
        ]


def _geojson_to_geometry(geojson: dict) -> QgsGeometry | None:
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

        return mask_to_polygons_fallback(
            mask, transform_info, simplify_tolerance, pixel_offset, full_shape)

    except ImportError:
        return mask_to_polygons_fallback(
            mask, transform_info, simplify_tolerance, pixel_offset, full_shape)
    except Exception as e:
        import traceback
        QgsMessageLog.logMessage(
            f"mask_to_polygons error: {str(e)}\n{traceback.format_exc()}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Warning
        )
        return mask_to_polygons_fallback(
            mask, transform_info, simplify_tolerance, pixel_offset, full_shape)


def mask_to_polygons_fallback(
    mask: np.ndarray,
    transform_info: dict,
    simplify_tolerance: float = 0.0,
    pixel_offset: tuple[int, int] | None = None,
    full_shape: tuple[int, int] | None = None,
) -> list[QgsGeometry]:
    try:
        # Honor pixel_offset/full_shape exactly like the primary path: when the
        # mask is a CROP of a full grid, place it by its offset within
        # full_shape and map with the sub-mask's own shape. Mapping a crop with
        # the full tile dimensions would pull every mask toward the tile origin
        # (the historic tile-origin offset bug).
        bbox = transform_info.get("bbox")
        if bbox and pixel_offset is not None and full_shape is not None:
            minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
            full_h, full_w = int(full_shape[0]), int(full_shape[1])
            col0, row0 = int(pixel_offset[0]), int(pixel_offset[1])
            px_w = (maxx - minx) / max(full_w, 1)
            px_h = (maxy - miny) / max(full_h, 1)
            sub_h, sub_w = int(mask.shape[0]), int(mask.shape[1])
            sub_minx = minx + col0 * px_w
            sub_maxy = maxy - row0 * px_h
            sub_maxx = sub_minx + sub_w * px_w
            sub_miny = sub_maxy - sub_h * px_h
            transform_info = dict(transform_info)
            transform_info["bbox"] = (sub_minx, sub_maxx, sub_miny, sub_maxy)
            transform_info["img_shape"] = (sub_h, sub_w)

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

    # Which pixels sit on a boundary depends only on the mask, and the mask does
    # not change while tracing, so find them all in a few numpy passes instead of
    # asking the question per pixel in Python: a full-size mask has millions of
    # pixels but only thousands on a boundary. Only the visited test has to stay
    # inside the loop, because trace_contour updates it as it walks.
    foreground = padded == 1
    is_background = padded == 0
    on_boundary = np.zeros_like(foreground)
    for dx, dy in directions:
        on_boundary[1:h + 1, 1:w + 1] |= is_background[
            1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
    on_boundary &= foreground

    # np.nonzero walks row-major, the same order the per-pixel scan used.
    ys, xs = np.nonzero(on_boundary)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if visited_pad[y, x]:
            continue
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
    min_area: int = 0,            # Remove regions smaller than this (pixels)
    max_hole_px: int | None = None,  # Only fill holes up to this many pixels
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
        max_hole_px: Size cutoff for fill_holes, in mask pixels. None fills every
            hole (what the control did before the cutoff existed). A number keeps
            holes bigger than it, so a road keeps its median and loses its parked
            cars. The user sets a ground area; core.hole_size.hole_pixels turns it
            into this pixel count.
    """
    # astype already returns a fresh array, so a .copy() before it would
    # allocate and fill a whole extra mask on every refine repaint.
    result = mask.astype(np.uint8, order="C")

    # 1. Expand/Contract first so fill-holes operates on the adjusted mask
    if expand_value != 0:
        iterations = abs(expand_value)
        if expand_value > 0:
            result = _numpy_dilate(result, iterations)
        else:
            result = _numpy_erode(result, iterations)

    # 2. Fill holes (on already expanded/contracted mask)
    if fill_holes:
        if max_hole_px is None:
            result = _fill_holes(result)
        else:
            result = fill_small_holes(result, max(0, int(max_hole_px)))

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


def _label_components(mask_bool: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labels for a boolean mask, without scipy.

    scipy is optional, so every labeling helper here needs a fallback that stays
    usable on a click. A per-pixel flood fill in Python walks a million cells on
    a full-size mask; this works on row RUNS instead, and a mask has at most
    height + components of them, so the Python part collapses to the run count
    while the pixel-level work stays in numpy.

    Returns (labels, count) with labels an int32 (H, W) array, 0 = background
    and components numbered 1..count in row-major order of their first pixel
    (the same numbering scipy.ndimage.label produces).
    """
    h, w = mask_bool.shape
    labels = np.zeros((h, w), dtype=np.int32)
    if h == 0 or w == 0 or not mask_bool.any():
        return labels, 0

    # One False column on each side so no run can straddle a row boundary.
    padded = np.zeros((h, w + 2), dtype=bool)
    padded[:, 1:-1] = mask_bool
    flat = padded.ravel()
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = edges[0::2]
    stops = edges[1::2]
    rows = starts // (w + 2)
    col_start = starts - rows * (w + 2) - 1
    col_stop = stops - rows * (w + 2) - 1  # exclusive

    n_runs = int(starts.size)
    parent = list(range(n_runs))

    def _find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    # Runs are ordered by row, so each row occupies one contiguous slice.
    row_lo = np.searchsorted(rows, np.arange(h), side="left").tolist()
    row_hi = np.searchsorted(rows, np.arange(h), side="right").tolist()
    cs = col_start.tolist()
    ce = col_stop.tolist()
    for r in range(h - 1):
        i, i_end = row_lo[r], row_hi[r]
        j, j_end = row_lo[r + 1], row_hi[r + 1]
        while i < i_end and j < j_end:
            if cs[i] < ce[j] and cs[j] < ce[i]:
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    if ri < rj:
                        parent[rj] = ri
                    else:
                        parent[ri] = rj
            if ce[i] < ce[j]:
                i += 1
            else:
                j += 1

    # Number the roots in first-pixel order, which is the run order.
    numbering: dict[int, int] = {}
    run_label = [0] * n_runs
    for i in range(n_runs):
        root = _find(i)
        lab = numbering.get(root)
        if lab is None:
            lab = len(numbering) + 1
            numbering[root] = lab
        run_label[i] = lab

    row_list = rows.tolist()
    for i in range(n_runs):
        labels[row_list[i], cs[i]:ce[i]] = run_label[i]
    return labels, len(numbering)


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

    # Numpy fallback: label the background once instead of expanding the
    # exterior one pixel per pass. The pass-per-pixel form needed as many whole-
    # array passes as the mask is wide, so an elongated shape cost a thousand of
    # them on the click path; one labeling settles it whatever the shape.
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask

    background = (padded == 0)
    labels, _count = _label_components(background)
    # The pad ring is background all the way round, so every background pixel
    # connected to the outside carries the ring's own label.
    exterior = labels == labels[0, 0]

    result = padded.copy()
    result[background & ~exterior] = 1

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

    # Fallback: run-based labeling, so a mask with no scipy still costs a few
    # numpy passes instead of a per-pixel Python flood fill on the click path.
    labels, count = _label_components(mask.astype(bool))
    if count == 0:
        return mask.copy()

    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    small = np.flatnonzero(sizes < min_area)
    small = small[small != 0]  # label 0 is background, never a region

    if small.size:
        result = mask.copy()
        result[np.isin(labels, small)] = 0
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

    # Numpy fallback: same run-based labeling as the rest of this module, so a
    # missing scipy costs numpy passes rather than a per-pixel Python walk.
    labels, count = _label_components(mask.astype(bool))
    if count == 0:
        return []
    return list(np.bincount(labels.ravel(), minlength=count + 1)[1:])


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


def _buffer_square_corners(g: QgsGeometry, dist: float) -> QgsGeometry | None:
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


def _keep_largest_part(g: QgsGeometry) -> QgsGeometry | None:
    """The largest polygon part of g by area, or the input when single-part.

    Used right after a despike opening: an opening that severs a blob joined by
    a thin neck leaves the blob as a separate part, so keeping the largest part
    drops it. A normal single-part footprint is returned unchanged. Best-effort:
    returns None only on failure."""
    try:
        if g is None or g.isEmpty() or not g.isMultipart():
            return g
        parts = g.asGeometryCollection()
        if not parts:
            return g
        best = max(parts, key=lambda p: p.area())
        return QgsGeometry(best)
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        return None


# Ceiling on how much a notch closing may grow an object before the result is
# refused. Bridging an object's own bites adds a little area; welding it to a
# neighbour adds a lot, and there is no shape reason for the second.
_CLOSE_MAX_AREA_GROWTH = 1.25


def _geometry_part_count(geom: QgsGeometry) -> int:
    """Parts in a geometry, 1 for a single polygon. Used to refuse a step that
    silently welded two objects into one."""
    try:
        if not geom.isMultipart():
            return 1
        return int(geom.constGet().numGeometries())
    except Exception:  # noqa: BLE001 -- a guard must never raise
        return 1


def despike_thin_necks(
    g: QgsGeometry,
    despike_m: float,
    *,
    preserve_parts: bool = False,
) -> QgsGeometry:
    """Cut thin spikes and necks out of a traced outline, in ground units.

    Shared by the Automatic review tail and the Manual refine tail so one
    object loses the same spikes in both modes. Runs BEFORE squaring, so the
    regularizer never snaps a spike into a rotated diamond.

    Morphological opening with MITRE joins (shrink by ``despike_m`` then grow
    back, square corners preserved), then keep the largest part so a blob
    severed at a thin neck is dropped instead of left as an island.
    ``preserve_parts`` skips that last step: a multipart input is already an
    explicit user or object decision (a Correct-step merge of disjoint roof
    pieces, a multipart Manual selection), and must never come back with a
    piece missing.

    A distance of 0 or less is the OFF state and returns the input untouched.
    Guarded on isEmpty so a genuinely thin object is skipped, never destroyed.
    Best-effort: returns the input unchanged on any failure.
    """
    if g is None or g.isEmpty() or not despike_m or despike_m <= 0.0:
        return g
    try:
        shrunk = _buffer_square_corners(g, -despike_m)
        if shrunk is None or shrunk.isEmpty():
            return g
        grown = _buffer_square_corners(shrunk, despike_m)
        if grown is None or grown.isEmpty():
            return g
        return grown if preserve_parts else (_keep_largest_part(grown) or grown)
    except Exception:  # noqa: BLE001 -- refine is best-effort  # nosec B110
        return g


# Share of its own area a geometry must keep through a smoothing step for that
# step to be accepted. A corner round trims a little; anything that eats half an
# object is the engine failing, not a shape decision.
_SMOOTH_AREA_KEEP = 0.5
# The post-smooth diet runs at this share of the simplify tolerance. At the full
# tolerance Douglas-Peucker removes exactly the points the round just added.
_SMOOTH_DIET_FRACTION = 0.5


def _mean_segment_length(g: QgsGeometry) -> float:
    """Mean length of the exterior-ring segments, in the geometry's own units.

    Tells corner rounding whether the outline is denser than the scale it works
    at. 0.0 when there is nothing measurable, which reads as "do not thin".
    """
    try:
        length = float(g.length())
        count = int(g.constGet().nCoordinates()) if g.constGet() else 0
    except (AttributeError, TypeError, ValueError):
        return 0.0
    if length <= 0.0 or count < 2:
        return 0.0
    return length / float(count)


def rounded_corner_outline(
    g: QgsGeometry, simplify_tol: float = 0.0,
    settings: dict | None = None,
) -> QgsGeometry:
    """One "Round corners" pass plus the vertex diet that pays for it.

    Shared by the Automatic review tail, the Manual refine tail and the
    Refine-in-Manual handoff, so the same object gets the same rounded outline
    whichever one produced it.

    ONE Chaikin pass, offset 0.25. Each pass roughly doubles the point count,
    so more than one undoes the point budget that ran just before.
    ``minimumDistance`` is the simplify tolerance: a segment already at
    sub-tolerance scale has no visible corner to round, so splitting it would
    only mint invisible vertices. ``maxAngle`` is 120 degrees, which leaves a
    near-straight wall vertex alone instead of bending it. All three are
    server-overridable (``review.smooth``), so how strong the tick is can be
    retuned without a plugin release; the numbers here are the fallback.

    Then the diet. The pass still near-doubles the vertex count, which is pure
    render weight once thousands of objects are on canvas (every pan or zoom
    repaint reads every vertex), so a post-smooth simplify at the same
    tolerance strips the sub-tolerance points the rounding added while keeping
    every curve big enough to see.

    ``settings`` is an already-resolved ``review.smooth`` block. A caller that
    rounds thousands of objects under one set of dials (the Automatic review's
    LiveRefiner) resolves it once and passes it; None reads the policy here, as
    a single-object caller does.

    Best-effort: returns the geometry so far on any failure.
    """
    if g is None or g.isEmpty():
        return g
    passes, offset, max_angle = 1, 0.25, 120.0
    try:
        served = settings
        if served is None:
            from .detection_policy import smooth_pass_settings

            served = smooth_pass_settings()
        passes = int(served["iterations"])
        offset = float(served["offset"])
        max_angle = float(served["max_angle_deg"])
    except Exception:  # noqa: BLE001 -- policy is best-effort  # nosec B110
        pass
    try:
        src = g
        tol = float(simplify_tol) if simplify_tol and simplify_tol > 0.0 else 0.0
        # Thin FIRST when the outline is denser than the tolerance. Chaikin cuts
        # each corner by a fraction of its adjacent edges, so on an outline whose
        # segments are already sub-tolerance the cut lands under the visible
        # scale and the tick reads as dead however many points the object has.
        # Thinning to the tolerance gives the corner an edge long enough to bite.
        if tol > 0.0 and _mean_segment_length(src) < tol:
            thinned = src.simplify(tol)
            if (thinned is not None and not thinned.isEmpty()
                    and thinned.area() >= _SMOOTH_AREA_KEEP * src.area()):
                src = thinned
        # minimumDistance is NOT a tolerance here: hand QgsGeometry.smooth a ring
        # whose segments sit under it and it returns a degenerate zero-area ring
        # instead of skipping those segments. -1 disables the knob, which is the
        # only safe value; the thinning above already sets the working scale.
        r = src.smooth(passes, offset, -1.0, max_angle)
        # isEmpty() does NOT catch that degenerate ring (it has points, just no
        # area), so the guard is on area, against the geometry actually smoothed.
        if r is None or r.isEmpty() or r.area() < _SMOOTH_AREA_KEEP * src.area():
            return src
        g = r
        if tol > 0.0:
            # Half tolerance, not the full one: at the full tolerance
            # Douglas-Peucker strips the curve the pass just added and the
            # object comes back with the corners it started with.
            r2 = g.simplify(tol * _SMOOTH_DIET_FRACTION)
            if (r2 is not None and not r2.isEmpty()
                    and r2.area() >= _SMOOTH_AREA_KEEP * g.area()):
                g = r2
    except Exception:  # noqa: BLE001 -- refine is best-effort  # nosec B110
        pass
    return g


def _regularize_shape_kwargs(
    diagonal_reduction: float | None,
    circle_threshold: float | None,
    multi_direction: bool = False,
    multi_max_groups: int | None = None,
    multi_min_separation_deg: float | None = None,
) -> dict:
    """The regularizer's shape dials as keyword arguments, omitting the ones
    the caller left unset.

    All are server-tunable (core.detection_policy.regularize_settings), but
    this module stays policy-free: it is the pure geometry layer and the Manual
    pipeline imports it offline. Omitting an unset dial hands the decision to
    building_regularizer's own default instead of copying that value here,
    where it would drift. multi_direction is only forwarded when on, so the
    single-direction engine default stands for every caller that leaves it off.
    """
    kwargs: dict = {}
    if diagonal_reduction is not None:
        kwargs["diagonal_reduction"] = float(diagonal_reduction)
    if circle_threshold is not None and circle_threshold > 0:
        kwargs["circle_threshold"] = float(circle_threshold)
    if multi_direction:
        kwargs["multi_direction"] = True
        if multi_max_groups is not None:
            kwargs["multi_max_groups"] = int(multi_max_groups)
        if multi_min_separation_deg is not None:
            kwargs["multi_min_separation_deg"] = float(multi_min_separation_deg)
    return kwargs


def apply_geometry_refinement(
    geom: QgsGeometry,
    *,
    smooth_settings: dict | None = None,
    simplify_tol: float = 0.0,
    smooth: bool = False,
    expand_dist: float = 0.0,
    fill_holes: bool = False,
    fill_holes_max_area: float = 0.0,
    open_dist: float = 0.0,
    close_dist: float = 0.0,
    despike_m: float = 0.0,
    vertex_spacing: float = 0.0,
    vertex_min: int = 8,
    vertex_max_deviation: float = 0.0,
    vertex_max_deviation_fraction: float = 0.0,
    vertex_keep_fraction: float = 0.0,
    vertex_dial_max_cap_fraction: float | None = None,
    ortho: bool = False,
    ortho_tol: float = 0.0,
    regularize: bool = False,
    regularize_tol: float = 0.0,
    allow_diagonal: bool = True,
    allow_circles: bool = False,
    regularize_min_iou: float = 0.0,
    diagonal_reduction: float | None = None,
    circle_threshold: float | None = None,
    multi_direction: bool = False,
    multi_max_groups: int | None = None,
    multi_min_separation_deg: float | None = None,
    envelope: Any = None,
    unit_aspect: float = 1.0,
) -> QgsGeometry:
    """Geometry-level refine for Automatic-review WHOLE objects (no mask here).

    The Manual mask-refine controls (simplify / round corners / expand-shrink /
    fill holes) act on a pixel mask; in the Automatic review the masks live on
    the worker and only merged polygons remain, so the same knobs are applied
    directly to the geometry:

      - fill_holes: drop interior rings (native removeInteriorRings).
      - fill_holes_max_area: size cutoff for fill_holes, in CRS UNITS SQUARED
        (0 = drop every ring, the behaviour before the cutoff existed). Only
        rings under it go, so a road keeps its median and loses the holes its
        parked cars punched. The user sets a ground area; the caller scales it
        with core.hole_size.ring_area_arg, since one CRS unit is not one metre
        outside a metric projection.
      - open_dist: morphological opening (shrink then grow back) that removes
        thin attached fringe / tendrils narrower than 2*open_dist, ground units.
      - simplify_tol: Douglas-Peucker simplify to a ground-unit tolerance.
      - vertex_spacing + vertex_min + vertex_max_deviation(_fraction) +
        vertex_keep_fraction: the point budget (core.vertex_budget), which thins
        the outline to one point per vertex_spacing of its own length, never
        below vertex_min points and never moving the boundary by more than the
        cap (the flat distance, or a share of the object's narrow dimension,
        whichever is tighter). vertex_keep_fraction is the user's dial, the
        share of its own points an outline may keep, and it can only take the
        budget lower. vertex_dial_max_cap_fraction caps how far that dial may
        let a corner travel, as a share of the object's narrow dimension; it is
        a server dial, and None means "use the client fallback", so every
        caller of the budget reads one value. Distances in CRS units.
        It runs BEFORE squaring, never
        after: thinning an already-squared footprint would un-square it, but
        squaring a thinned outline is exactly what the shape wants. When it
        runs it also serves as the de-staircase pass, so ortho_tol does not
        raise simplify on top of it.
      - ortho + ortho_tol: "Right angles" regularizer for man-made shapes
        (native QgsGeometry.orthogonalize snaps edges to 90 degrees). A raw
        mask outline is ALREADY all right-angle stair steps, so it must be
        de-staircased first: the simplify pass runs at max(simplify_tol,
        ortho_tol) when ortho is on. ortho_tol is a GROUND distance the caller
        resolves (core.detection_policy.destair_tolerance_m), so the same
        building de-staircases the same way whatever detail it was tiled at.
      - regularize: the diagonals-aware footprint regularizer (building_
        regularizer). Used INSTEAD of the plain orthogonalize when on, so a
        45-degree building wall is snapped clean, not left as stair steps. It
        de-staircases internally (regularize_tol, ground units), so no extra
        simplify bump is needed. allow_diagonal enables the 45-degree snaps;
        diagonal_reduction narrows the window an edge must fall in to land on
        one, so raising it turns chamfered corners back into square ones;
        allow_circles fits near-circular blobs, circle_threshold sets how round
        a blob must be first; regularize_min_iou reverts to the input when the
        snapped shape drifts too far. Which classes get this and the tuning
        come from the server policy; an unset dial keeps the engine default.
      - unit_aspect: ground metres per y unit over ground metres per x unit of
        the geometry's CRS, so the squaring lands on the ground rather than on
        raw coordinates. 1.0, the default, is exact for a projected CRS. A
        caller working in a geographic one (a local image is often supplied in
        degrees) MUST measure and pass it, or every square corner arrives
        tilted, the more so the further the data sits from the equator.
      - expand_dist: buffer out (positive) or shrink in (negative), ground units.
      - smooth: Chaikin round the corners (native QgsGeometry.smooth).

      - despike_m: the spike and neck cut (despike_thin_necks), ground units,
        0 = off.

    Order: fill, then de-spike, then open (strip fringe), then simplify, then
    the point budget, then right angles, then expand, then a final smooth.
    Best-effort: any failed step is skipped and the geometry so far is kept, so
    one degenerate object never breaks a refresh.

    The de-spike and the final smooth live in despike_thin_necks and
    rounded_corner_outline, shared with the Manual refine tail so one object
    gets the same outline in both modes.
    """
    if geom is None or geom.isEmpty():
        return geom
    g = geom
    # A multipart input is already an explicit user/object decision: it can be
    # a Correct-step merge of disjoint roof pieces.  The de-spike opening may
    # split a single noisy mask at a thin neck, where keeping its main part is
    # intentional; it must never turn an existing MultiPolygon into one part
    # and silently discard a piece the user deliberately merged.
    preserve_input_parts = bool(g.isMultipart())
    budget_on = bool((vertex_spacing and vertex_spacing > 0.0) or (vertex_keep_fraction and vertex_keep_fraction > 0.0))
    if ortho and ortho_tol > 0.0:
        # De-staircase before squaring, ALWAYS when Right angles is on. ortho_tol
        # is the small de-staircase distance (a couple of pixels), not a hard
        # reduction, so it only strips stair steps and never sweeps a wall into
        # a slanted chord. The point budget below thins further, but it can
        # decline every vertex when its deviation cap is strict, so leaving the
        # regularizer a raw pixel staircase in that case. This gentle pass is
        # cheap insurance that squaring always starts from a de-staircased
        # outline, budget or no budget.
        simplify_tol = max(simplify_tol or 0.0, ortho_tol)
    if fill_holes:
        from .hole_size import REMOVE_ALL_RINGS

        cutoff = (float(fill_holes_max_area) if fill_holes_max_area and fill_holes_max_area > 0.0 else REMOVE_ALL_RINGS)
        try:
            r = g.removeInteriorRings(cutoff)
            if isinstance(r, QgsGeometry) and not r.isEmpty():
                g = r
        except (AttributeError, TypeError, ValueError):
            pass
    # Cut the thin spikes and necks the raw mask carries (a tile-seam join, an
    # uncertain point) BEFORE squaring. Shared with the Manual tail.
    g = despike_thin_necks(g, despike_m, preserve_parts=preserve_input_parts)
    if close_dist and close_dist > 0.0:
        # Morphological closing: grow by close_dist then shrink back, so a bite
        # taken out of the outline from OUTSIDE is bridged while the object
        # keeps its true size. Fill holes cannot reach one of these: it is not
        # an interior ring, it opens onto the boundary. Runs AFTER the despike
        # so it does not bridge a neck that step just cut, and BEFORE the
        # opening so any fringe it leaves is trimmed again.
        #
        # Two reverts, because a closing that reaches too far is not a fix. It
        # must not weld separate objects together (the part count would drop),
        # and it must not inflate the object past what bridging its own bites
        # can explain.
        try:
            before_parts = _geometry_part_count(g)
            before_area = g.area()
            # Mitre joins, like the despike: a round join rounds off the very
            # corners the closing just bridged, so the bite comes back as a
            # dent. Square corners are also what a kerb actually looks like.
            grown = _buffer_square_corners(g, close_dist)
            r = (None if grown is None or grown.isEmpty()
                 else _buffer_square_corners(grown, -close_dist))
            if (r is not None and not r.isEmpty()
                    and _geometry_part_count(r) >= before_parts
                    and (before_area <= 0.0
                         or r.area() <= before_area * _CLOSE_MAX_AREA_GROWTH)):
                g = r
        except Exception:  # noqa: BLE001 -- refine is best-effort  # nosec B110
            pass
    if open_dist and open_dist > 0.0:
        # Morphological opening: shrink by open_dist then grow back. Thin fringe
        # / tendrils narrower than 2*open_dist vanish while the main shape keeps
        # its true size. buffer(-d) can empty a genuinely thin object; guard on
        # isEmpty so it is skipped, never destroyed.
        #
        # Same part-count refusal as the closing above and as despike's
        # preserve_parts: on a multipart object the shrink can dissolve the
        # smaller piece outright, and a piece the user merged by hand must
        # never go without a word. Keeping a spike beats losing half an object.
        try:
            before_parts = _geometry_part_count(g)
            r = g.buffer(-open_dist, 8).buffer(open_dist, 8)
            if (r is not None and not r.isEmpty()
                    and _geometry_part_count(r) >= before_parts):
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
    if budget_on:
        # The point budget: cut the traced outline down to the number of points
        # a person would have drawn. Runs under squaring too, where it replaces
        # the Douglas-Peucker de-staircase above.
        try:
            from .vertex_budget import (
                _DIAL_MAX_CAP_NARROW_FRACTION,
                simplify_to_budget,
            )

            # None = no server dial, so fall back to the engine's own value.
            # Read from there rather than restated here: one client-side
            # default for every caller of the budget.
            dial_cap = (_DIAL_MAX_CAP_NARROW_FRACTION
                        if vertex_dial_max_cap_fraction is None
                        else float(vertex_dial_max_cap_fraction))
            r = simplify_to_budget(
                g, spacing=vertex_spacing, min_vertices=vertex_min,
                max_deviation=vertex_max_deviation,
                max_deviation_fraction=vertex_max_deviation_fraction,
                keep_fraction=vertex_keep_fraction,
                dial_max_cap_fraction=dial_cap,
                # Same axis correction the regularizer below is given: without
                # it a geographic CRS counts a degree of longitude and one of
                # latitude as the same length, and the budget thins one axis
                # against the wrong ground distance.
                unit_aspect=unit_aspect)
            if r is not None and not r.isEmpty():
                g = r
        except Exception:  # noqa: BLE001 -- refine is best-effort  # nosec B110
            pass
    regularized_ok = False
    if regularize:
        # Diagonals-aware path: snap edges to the dominant direction plus its
        # perpendicular and (optionally) 45-degree diagonals, bounded by a
        # ground-unit tolerance. Falls back to the simplify tolerance when the
        # caller passes none; skipped when neither is positive. Best-effort:
        # returns the input unchanged on any failure.
        tol = regularize_tol if regularize_tol > 0.0 else simplify_tol
        if tol and tol > 0.0:
            try:
                from .building_regularizer import regularize_qgs_geometry_ex
                res = regularize_qgs_geometry_ex(
                    g,
                    tolerance_m=tol,
                    allow_diagonal=allow_diagonal,
                    allow_circles=allow_circles,
                    min_keep_iou=regularize_min_iou,
                    policy=envelope,
                    unit_aspect=unit_aspect,
                    **_regularize_shape_kwargs(
                        diagonal_reduction, circle_threshold,
                        multi_direction, multi_max_groups,
                        multi_min_separation_deg),
                )
                if res.regularized and not res.geometry.isEmpty():
                    g = res.geometry
                    regularized_ok = True
            except Exception:  # noqa: BLE001 -- refine is best-effort  # nosec B110
                pass
    # Right angles must still do SOMETHING when the plain path is in use and no
    # regularizer ran. But when the diagonals-aware regularizer was requested
    # and DECLINED this shape (its IoU guard reverted it), orthogonalizing the
    # outline manufactures a staircase on exactly the complex/off-axis footprints
    # it could not handle. The simplify above already de-staircased g (ortho
    # forces simplify_tol up to ortho_tol), so we leave it as that smooth
    # simplified outline instead.
    if ortho and not regularized_ok and not regularize:
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
            # A contract eats a small part before it eats a big one, so it gets
            # the same part-count refusal as the opening. A grow never drops a
            # part, so it pays nothing for the check.
            shrinking = expand_dist < 0.0
            before_parts = _geometry_part_count(g) if shrinking else 0
            # With Right angles or the regularizer on, a round-join buffer would
            # re-round every corner just squared; a mitre join preserves them.
            r = (
                _buffer_square_corners(g, expand_dist)
                if (ortho or regularize)
                else g.buffer(expand_dist, 8)
            )
            if (r is not None and not r.isEmpty()
                    and (not shrinking
                         or _geometry_part_count(r) >= before_parts)):
                g = r
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    if smooth:
        # Round corners plus its vertex diet, shared with the Manual tail.
        g = rounded_corner_outline(g, simplify_tol, smooth_settings)
    return g


def apply_right_angles(
    geom: QgsGeometry,
    destair_tol: float = 0.0,
    *,
    tolerance_m: float = 0.0,
    allow_diagonal: bool = True,
    allow_circles: bool = False,
    min_keep_iou: float = 0.7,
    diagonal_reduction: float | None = None,
    circle_threshold: float | None = None,
    multi_direction: bool = False,
    multi_max_groups: int | None = None,
    multi_min_separation_deg: float | None = None,
    envelope: Any = None,
    unit_aspect: float = 1.0,
) -> QgsGeometry:
    """"Right angles" for the Manual pipeline and the Refine-in-Manual handoff.

    Uses the diagonals-aware footprint regularizer (building_regularizer), the
    SAME engine the Automatic review uses, so both modes behave the same: a
    45-degree wall snaps clean, not just edges already within 15 degrees of
    0/90. When that engine is unavailable or leaves the shape empty the
    fallback is a de-staircased outline, NOT the native
    QgsGeometry.orthogonalize (see the paragraph on the guard below); an
    unavailable engine also means no squaring happened at all, which is why the
    UI asks building_regularizer.dependencies_available() before offering the
    control.

    A raw mask outline is ALREADY all right-angle stair steps, so it is
    de-staircased first: when ``destair_tol`` is given the outline is simplified
    up to it. ``tolerance_m`` is the regularizer's max snap distance (ground
    units); callers pass the server ground tolerance so the snap keeps a scale
    even when destair_tol collapses to 0. ``allow_diagonal``, ``allow_circles``,
    ``min_keep_iou``, ``diagonal_reduction``, ``circle_threshold``,
    ``multi_direction``, ``multi_max_groups`` and ``multi_min_separation_deg``
    are the server shape dials (core.detection_policy.regularize_settings), so
    Manual and the Refine-in-Manual handoff square exactly like the Automatic
    review; each defaults to the engine default, so an old caller is unchanged.
    ``unit_aspect`` squares against ground distance rather than raw coordinates
    and MUST be passed when the geometry sits in a geographic CRS, or the
    corners come out tilted (see apply_geometry_refinement).
    ``multi_direction`` off is the single-direction engine, today's behaviour:
    with it on, a building whose wing sits at an angle to the main block keeps
    each wing on its own grid instead of staircasing off one global axis.

    The regularizer's IoU guard reverts a shape the snap would distort. When it
    does, we do NOT run the native orthogonalize over the outline: on a complex
    or off-axis footprint that manufactures a staircase. The outline is already
    de-staircased, so we return that smooth simplified shape, which is never
    worse than the input. Best-effort: returns the input on any failure, never
    raises."""
    if geom is None or geom.isEmpty():
        return geom
    g = geom
    destaired = False
    if destair_tol and destair_tol > 0.0:
        try:
            r = g.simplify(destair_tol)
            if r is not None and not r.isEmpty():
                g = r
                destaired = True
        except Exception:  # noqa: BLE001 -- refine is best-effort  # nosec B110
            pass
    tol = tolerance_m if tolerance_m and tolerance_m > 0.0 else destair_tol
    if tol and tol > 0.0:
        try:
            from .building_regularizer import regularize_qgs_geometry_ex
            res = regularize_qgs_geometry_ex(
                g,
                tolerance_m=tol,
                allow_diagonal=allow_diagonal,
                allow_circles=allow_circles,
                min_keep_iou=min_keep_iou,
                policy=envelope,
                unit_aspect=unit_aspect,
                **_regularize_shape_kwargs(
                    diagonal_reduction, circle_threshold,
                    multi_direction, multi_max_groups,
                    multi_min_separation_deg),
            )
            # Only the regularized outcome short-circuits. When it changed
            # nothing (guard reverted, or the engine is unavailable) it returns
            # the INPUT; fall through to the de-staircase degradation below.
            if res.regularized and not res.geometry.isEmpty():
                return res.geometry
        except Exception:  # noqa: BLE001 -- fall back to the de-staircased outline  # nosec B110
            pass
    # Reverted or unavailable: degrade to a de-staircased simplified outline,
    # never a manufactured or raw staircase. When no destair distance was given,
    # simplify to the snap tolerance so the fallback is still a cleaned outline.
    if not destaired and tol and tol > 0.0:
        try:
            r = geom.simplify(tol)
            if r is not None and not r.isEmpty():
                return r
        except Exception:  # noqa: BLE001 -- refine is best-effort  # nosec B110
            pass
    return g


def shape_polygon_geometry(
    geom: QgsGeometry,
    mupp: float,
    simplify_px: float = 0.0,
    smooth: bool = False,
    expand_px: int = 0,
    fill_holes: bool = False,
    ortho: bool = False,
    fill_holes_max_area: float = 0.0,
    *,
    open_dist: float = 0.0,
    vertex_spacing: float = 0.0,
    vertex_min: int = 8,
    vertex_max_deviation: float = 0.0,
    vertex_max_deviation_fraction: float = 0.0,
    vertex_keep_fraction: float = 0.0,
    vertex_dial_max_cap_fraction: float | None = None,
    regularize_tol: float = 0.0,
    destair_tol: float = 0.0,
    allow_diagonal: bool = True,
    allow_circles: bool = False,
    min_keep_iou: float = 0.7,
    diagonal_reduction: float | None = None,
    circle_threshold: float | None = None,
    multi_direction: bool = False,
    multi_max_groups: int | None = None,
    multi_min_separation_deg: float | None = None,
    envelope: Any = None,
    unit_aspect: float = 1.0,
) -> QgsGeometry:
    """Apply the Manual refine controls to an EXISTING polygon geometry.

    The Refine-in-Manual handoff edits detections that have no source mask
    (their shape came from the cloud run), so the mask-space refinement
    pipeline cannot run. This is its geometry-space twin: same controls,
    same order as ``apply_geometry_refinement`` (fill holes, trim spikes,
    simplify, point budget, right angles, expand/contract, round corners),
    with pixel-denominated controls converted to ground
    units via ``mupp`` (map units per pixel of the source raster). Callers keep
    the PRISTINE geometry and re-shape from it on every settings change, so the
    operation stays non-destructive. Best-effort: returns the input on any
    failure, never raises.

    ``fill_holes_max_area`` is the fill-holes size cutoff in CRS UNITS SQUARED
    (0 = every hole, as before): only rings under it go, so a big real hole
    survives. Scale a ground-m2 setting with core.hole_size.ring_area_arg first.

    ``open_dist`` is the Trim-spikes morphological opening distance in CRS units
    (0 = off).

    ``expand_px`` runs AFTER the squaring, with mitre joins under it, the same
    slot as ``apply_geometry_refinement``: a round-join buffer before the
    squaring feeds it rounded corners, and one after it re-rounds the corners
    the squaring just produced.

    ``vertex_spacing`` + ``vertex_min`` + ``vertex_max_deviation(_fraction)`` +
    ``vertex_keep_fraction`` + ``vertex_dial_max_cap_fraction`` are the point
    budget (core.vertex_budget), the same set and the same slot as
    ``apply_geometry_refinement``: after the simplify, before the squaring,
    because squaring a thinned outline is what the shape wants while thinning an
    already-squared one un-squares it. ``vertex_keep_fraction`` is the user's
    Points dial and ``vertex_spacing`` the class density; a caller whose geometry
    already carries its class density (an object handed off from the Automatic
    review) passes spacing 0 so the dial is the only budget. All distances in CRS
    units; both budgets at 0 skips the pass, so an old caller is unchanged.

    ``regularize_tol``/``destair_tol`` are the Right-angles snap and
    de-staircase distances in CRS units, and ``allow_diagonal``,
    ``allow_circles``, ``min_keep_iou``, ``diagonal_reduction`` and
    ``circle_threshold``, ``multi_direction``, ``multi_max_groups`` and
    ``multi_min_separation_deg`` are the server shape dials
    (core.detection_policy.regularize_settings). When ``regularize_tol`` is 0
    the squaring falls back to the old pixel-anchored tolerance, so a caller
    that passes none is unchanged. ``multi_direction`` off is the
    single-direction engine, so a building whose wings sit at an angle to each
    other is snapped to one grid; the dial is what lets each wing keep its own.

    ``unit_aspect`` makes the squaring land on the ground rather than on raw
    coordinates, and MUST be passed when the geometry sits in a geographic CRS
    (see apply_geometry_refinement).
    """
    if geom is None or geom.isEmpty() or not mupp or mupp <= 0:
        return geom
    g = QgsGeometry(geom)
    try:
        if fill_holes:
            if fill_holes_max_area and fill_holes_max_area > 0.0:
                r = g.removeInteriorRings(float(fill_holes_max_area))
            else:
                parts = g.asMultiPolygon() if g.isMultipart() else [g.asPolygon()]
                shells = [[rings[0]] for rings in parts if rings]
                r = QgsGeometry.fromMultiPolygonXY(shells) if shells else None
            if r is not None and not r.isEmpty():
                g = r
        # Trim spikes: morphological opening (shrink then grow back) that drops
        # thin attached fringe narrower than 2*open_dist. The same op as the
        # Automatic review's Trim spikes; guarded on isEmpty so a genuinely thin
        # object is skipped, never destroyed, and on the part count so the
        # shrink cannot dissolve the smaller half of a merge the user made by
        # hand. Same refusal as apply_geometry_refinement, the twin tail.
        if open_dist and open_dist > 0.0:
            before_parts = _geometry_part_count(g)
            r = g.buffer(-open_dist, 8).buffer(open_dist, 8)
            if (r is not None and not r.isEmpty()
                    and _geometry_part_count(r) >= before_parts):
                g = r
        # Same px -> tolerance scale as the Manual mask pipeline
        # (_compute_simplification_tolerance), so the spinbox value means the
        # same thing whether the polygon came from a local mask or the cloud.
        tolerance = simplify_px * mupp if simplify_px > 0 else 0.0
        if tolerance > 0:
            r = g.simplify(tolerance)
            if r is not None and not r.isEmpty():
                g = r
        if ((vertex_spacing and vertex_spacing > 0.0) or (vertex_keep_fraction and vertex_keep_fraction > 0.0)):
            # The point budget, in the same slot as apply_geometry_refinement so
            # the two refine tails stay one pipeline.
            from .vertex_budget import (
                _DIAL_MAX_CAP_NARROW_FRACTION,
                simplify_to_budget,
            )

            # None = no server dial: read the engine's own value rather than
            # restating a second client default here.
            dial_cap = (_DIAL_MAX_CAP_NARROW_FRACTION
                        if vertex_dial_max_cap_fraction is None
                        else float(vertex_dial_max_cap_fraction))
            r = simplify_to_budget(
                g, spacing=vertex_spacing, min_vertices=vertex_min,
                max_deviation=vertex_max_deviation,
                max_deviation_fraction=vertex_max_deviation_fraction,
                keep_fraction=vertex_keep_fraction,
                dial_max_cap_fraction=dial_cap,
                # Same axis correction the regularizer below is given: without
                # it a geographic CRS counts a degree of longitude and one of
                # latitude as the same length, and the budget thins one axis
                # against the wrong ground distance.
                unit_aspect=unit_aspect)
            if r is not None and not r.isEmpty():
                g = r
        if ortho:
            # Ground snap + de-staircase distances (server dials), converted to
            # CRS units by the caller. Fall back to the old pixel-anchored 3-px
            # tolerance when none was passed, so an old caller is unchanged.
            reg_tol = regularize_tol if regularize_tol and regularize_tol > 0.0 else 1.5 * mupp
            destair = destair_tol if destair_tol and destair_tol > 0.0 else reg_tol
            g = apply_right_angles(
                g, destair_tol=max(0.0, destair - tolerance),
                tolerance_m=reg_tol,
                allow_diagonal=allow_diagonal,
                allow_circles=allow_circles,
                min_keep_iou=min_keep_iou,
                diagonal_reduction=diagonal_reduction,
                circle_threshold=circle_threshold,
                multi_direction=multi_direction,
                multi_max_groups=multi_max_groups,
                multi_min_separation_deg=multi_min_separation_deg,
                unit_aspect=unit_aspect,
                envelope=envelope)
        if expand_px:
            # After the squaring, never before: with Right angles on, a
            # round-join buffer would re-round every corner just produced, so
            # the join is mitre there. A contract carries the opening's
            # part-count refusal; a grow never drops a part.
            dist = expand_px * mupp
            shrinking = dist < 0.0
            before_parts = _geometry_part_count(g) if shrinking else 0
            r = _buffer_square_corners(g, dist) if ortho else g.buffer(dist, 8)
            if (r is not None and not r.isEmpty()
                    and (not shrinking
                         or _geometry_part_count(r) >= before_parts)):
                g = r
        if smooth:
            # Round corners plus its vertex diet, the same pass the Automatic
            # review runs.
            g = rounded_corner_outline(g, tolerance)
        if g.isGeosValid() is False:
            r = g.makeValid()
            if r is not None and not r.isEmpty():
                g = r
    except Exception:  # noqa: BLE001 -- refine is best-effort
        return geom
    return g if g is not None and not g.isEmpty() else geom


def suppress_redundant_hypotheses(
    items: list[tuple[QgsGeometry, float]],
    ios_threshold: float = 0.5,
    dup_ios_floor: float = 0.3,
    dup_centroid_frac: float = 0.35,
) -> list[tuple[QgsGeometry, float]]:
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
            small = min(karea, area)
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


# ONE generic client fallback for the end-of-run redundancy sweep. It is the
# share of its own area an object must have painted over by LARGER objects
# before it is read as a leftover partial. The tuned value is server policy
# (review.merge.cover_threshold); this is what a run uses with none.
COVER_THRESHOLD_DEFAULT = 0.40


def drop_covered_objects(
    items: list[tuple[int, QgsGeometry, float]],
    cover_threshold: float | None = None,
) -> list[tuple[int, QgsGeometry, float]]:
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

    Only SURVIVING neighbours can drop an object, which is why candidates are
    walked largest first: the largest object has nothing above it and is
    decided first, so every larger neighbour's fate is settled by the time an
    object is judged. Walking them in arrival order instead let a dropped
    partial take a real object down with it, and the chain compounded: on a
    dense scene a big garbage mask removed a building, and the building removed
    a garage that nothing surviving painted over. Losing a real object costs
    the user a new paid run; keeping a double-painted one costs them a click.

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
        items: list[tuple[int, QgsGeometry, float]],
        cover_threshold: float | None = None,
    ) -> None:
        self._items = items
        # None resolves the server value here rather than at each call site,
        # so the interactive sweep and the headless one cannot drift apart.
        self._threshold = (
            COVER_THRESHOLD_DEFAULT if cover_threshold is None
            else float(cover_threshold)
        )
        if cover_threshold is None:
            try:
                from .detection_policy import merge_scalar

                self._threshold = merge_scalar("cover_threshold", COVER_THRESHOLD_DEFAULT)
            except Exception:  # noqa: BLE001 -- policy is best-effort  # nosec B110
                pass
        self._n = len(items)
        self._i = 0
        self._keep = [True] * self._n
        self._done = self._n < 2
        self._metas: list = []
        self._index = QgsSpatialIndex()
        if self._done:
            self._order: list[int] = []
            return
        for pos, (_sid, geom, _score) in enumerate(items):
            try:
                bb, area = geom.boundingBox(), geom.area()
            except Exception:
                self._metas.append((None, 0.0))
                continue
            self._metas.append((bb, area))
            feat = QgsFeature(pos)
            feat.setGeometry(QgsGeometry.fromRect(bb))
            # Return ignored on purpose: a refused insert only costs one
            # covering candidate (a degenerate box), and the sweep only ever
            # drops an object, so a missing candidate keeps more, never less.
            self._index.addFeature(feat)
        # Largest first, so a candidate is only ever judged against neighbours
        # whose own fate is already settled (see drop_covered_objects). Ties keep
        # their arrival order, and an object of equal area never covers another.
        self._order = sorted(
            range(self._n), key=lambda pos: -self._metas[pos][1])

    def step(self, max_items: int) -> bool:
        """Process up to ``max_items`` candidates; return True when finished."""
        if self._done:
            return True
        items, metas, index, keep = self._items, self._metas, self._index, self._keep
        threshold = self._threshold
        processed = 0
        while self._i < self._n and processed < max_items:
            i = self._order[self._i]
            self._i += 1
            processed += 1
            bb, area = metas[i]
            if bb is None or area <= 0.0:
                continue
            covers = []
            best_cover_score = None
            # Prepared geometry for THIS candidate, built once and only when a
            # neighbour survives the box test. The bbox pre-filter below is an
            # upper bound on the shared area, and it prunes well for compact
            # objects but not at all for a long diagonal ribbon, whose box
            # covers most of the zone while its area stays small. Without the
            # prepared engine those runs pay a full GEOS overlay per pair.
            # prepareGeometry indexes the candidate's segments once, so each
            # neighbour costs a cheap predicate and the overlay runs only on a
            # real intersection.
            engine = None
            for j in index.intersects(bb):
                if j == i:
                    continue
                jbb, jarea = metas[j]
                if jbb is None or jarea <= area or not keep[j]:
                    continue
                iw = min(bb.xMaximum(), jbb.xMaximum()) - max(bb.xMinimum(), jbb.xMinimum())
                ih = min(bb.yMaximum(), jbb.yMaximum()) - max(bb.yMinimum(), jbb.yMinimum())
                if iw <= 0.0 or ih <= 0.0 or (iw * ih) / area < threshold * 0.5:
                    continue
                try:
                    if engine is None:
                        engine = QgsGeometry.createGeometryEngine(items[i][1].constGet())
                        engine.prepareGeometry()
                    jg = items[j][1].constGet()
                    if not engine.intersects(jg):
                        continue
                    raw = engine.intersection(jg)
                    inter = None if raw is None else QgsGeometry(raw)
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
            # each other over this object must not double count. One unaryUnion
            # over the whole list beats a chain of pairwise combine() (fewer
            # intermediate geometries) on a dense tile.
            if len(covers) == 1:
                merged = covers[0]
            else:
                merged = QgsGeometry.unaryUnion(covers)
                if merged is None or merged.isEmpty():
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

    def result(self) -> list[tuple[int, QgsGeometry, float]]:
        return [it for it, k in zip(self._items, self._keep) if k]


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

# Formats whose own specification fixes the coordinates on WGS84. Writing one
# of these in a projected CRS produces a file every conforming reader places
# somewhere else. The ground-metre CRS goes to the other drivers.
_WGS84_ONLY_DRIVERS = ("GeoJSON", "KML")


def driver_extension(driver: str) -> str:
    """File extension (with dot) for a supported export driver."""
    return _DRIVER_EXTENSIONS.get(driver, ".gpkg")


def _release_project_layers_on_gui(path: str) -> int:
    """Drop project layers reading this exact file. GUI thread only.

    Best effort: a failure to read one layer skips it rather than stopping the
    export. normcase before comparing, because the same Windows file has many
    spellings and one that does not match leaves the handle in place, which is
    the whole point of the call.
    """
    import os

    from qgis.core import QgsProject

    try:
        target = os.path.normcase(os.path.abspath(path))
    except (OSError, ValueError):
        return 0
    doomed = []
    for layer_id, layer in QgsProject.instance().mapLayers().items():
        try:
            # A provider URI can carry "|layername=..." and friends; the file
            # is everything before the first pipe.
            source = (layer.source() or "").split("|")[0]
            if source and os.path.normcase(os.path.abspath(source)) == target:
                doomed.append(layer_id)
        except (AttributeError, RuntimeError, OSError, ValueError):
            continue
    if doomed:
        QgsProject.instance().removeMapLayers(doomed)
    return len(doomed)


# How long a worker thread waits for the GUI thread to run a posted release.
# Long enough for a busy GUI to reach it, short enough that an export never
# hangs on a GUI thread that cannot answer at all.
_RELEASE_LAYERS_TIMEOUT_S = 5.0


class _LayerReleaseCall(QObject):
    """One release, run on the thread this object lives on.

    A slot and not a plain function because that is what QMetaObject can post
    across threads by name. It carries the answer back on ``count``, and
    ``finished`` is what the calling thread waits on.
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path
        self.count = 0
        self.finished = threading.Event()

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.count = _release_project_layers_on_gui(self._path)
        except Exception:  # noqa: BLE001 -- the release never stops an export
            self.count = 0
        finally:
            self.finished.set()
            _pending_layer_releases.discard(self)
            self.deleteLater()


# Releases posted to the GUI thread and not yet run. A posted call outlives the
# worker's own reference whenever the wait runs out, and a receiver collected
# while the GUI thread is about to call it is a crash.
_pending_layer_releases: set[_LayerReleaseCall] = set()


def _release_project_layers_at(path: str) -> int:
    """Drop project layers reading this exact file. Returns how many went.

    Safe from any thread. A map layer belongs to the GUI thread and dropping
    one from a worker corrupts the project, so off the GUI thread the work is
    posted there and waited on: the caller overwrites the file right after, and
    the handle has to be gone first. A wait that runs out returns 0 and lets the
    write try anyway, which fails loudly, rather than freezing the export.
    """
    from qgis.core import QgsApplication
    from qgis.PyQt.QtCore import Qt, QThread

    from .qt_compat import resolve_qt_enum

    try:
        app = QgsApplication.instance()
        if app is None or QThread.currentThread() == app.thread():
            return _release_project_layers_on_gui(path)
    except (RuntimeError, AttributeError):
        return 0
    call = _LayerReleaseCall(path)
    try:
        call.moveToThread(app.thread())
        _pending_layer_releases.add(call)
        queued = resolve_qt_enum(Qt, "ConnectionType", "QueuedConnection")
        # Only an explicit False is a refusal. This overload answers True on
        # PyQt5 and None on PyQt6, so reading the result as a plain boolean
        # throws the posted call away on QGIS 4 and never releases anything.
        if QMetaObject.invokeMethod(call, "run", queued) is False:
            _pending_layer_releases.discard(call)
            return 0
    except (RuntimeError, AttributeError, TypeError):
        _pending_layer_releases.discard(call)
        return 0
    if not call.finished.wait(_RELEASE_LAYERS_TIMEOUT_S):
        QgsMessageLog.logMessage(
            "Export: gave up waiting for the layers holding the target file",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        return 0
    return call.count


def export_geometries_to_file(
    geoms: list,
    crs,
    output_path: str,
    driver: str = "GPKG",
    source_layer_name: str = "",
    layer_name: str | None = None,
    stats: dict | None = None,
):
    """Write polygon geometries to a vector file and return the loaded layer.

    Additive export helper used by the Library's direct Export (and reusable by
    any caller that already has final geometries). Keeps the export layer
    conventions: lean per-feature schema (editable ``label`` + the geodesic
    ``area_m2`` and ``perimeter_m``), geometries repaired with makeValid before
    save (repair, never silently drop). For GPKG the run-level provenance metadata and the style
    are stored INTO the file; the other drivers cannot embed a style, so they
    skip it. GeoJSON and KML are always written in EPSG:4326 (both formats
    mandate it); every other driver gets the ground-metre CRS from
    pick_output_crs, so a length read off the file is a length on the ground.

    Args:
        geoms:             QgsGeometry list (any polygonal type).
        crs:               QgsCoordinateReferenceSystem of the geometries.
        output_path:       Destination file path (extension decides nothing;
                           the ``driver`` does).
        driver:            One of EXPORT_DRIVERS. Default "GPKG".
        source_layer_name: Raster name recorded in the GPKG provenance.
        layer_name:        Layer name inside the file; defaults to the file stem.
        stats:             Optional dict this fills with ``written`` (polygons
                           staged for the file) and ``skipped`` (geometries no
                           repair could save). A caller that counts what it
                           handed in reports a number the file contradicts, so
                           read ``written`` when telling the user how many
                           polygons they got. Optional on purpose: an older
                           caller passes nothing and is unchanged.

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
        pick_output_crs,
        repair_polygon,
        round_measure,
        to_multipolygon,
    )

    if not geoms:
        return None
    if driver not in EXPORT_DRIVERS:
        driver = "GPKG"
    if not str(output_path or "").strip():
        return None
    # A bare filename lands in the process working directory, which is never a
    # place the user picked (on Windows it sits under the QGIS install). Anchor
    # a relative path before anything touches the disk.
    output_path = os.path.abspath(output_path)

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
    if not pr.addAttributes([
        QgsField("label", field_str),
        QgsField("area_m2", field_dbl),
        QgsField("perimeter_m", field_dbl),
    ]):
        QgsMessageLog.logMessage(
            "Export: could not build the attribute schema",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        return None
    temp_layer.updateFields()

    measurer = make_area_measurer(crs)
    feats = []
    skipped = 0
    for geom in geoms:
        if geom is None or geom.isEmpty():
            skipped += 1
            continue
        geom = to_multipolygon(repair_polygon(geom) or geom)
        if geom is None or geom.isEmpty():
            skipped += 1
            continue
        feat = QgsFeature(temp_layer.fields())
        feat.setGeometry(geom)
        try:
            area = measurer.measureArea(geom)
            perimeter = measurer.measurePerimeter(geom)
        except (RuntimeError, AttributeError):
            area, perimeter = geom.area(), geom.length()
        feat.setAttributes(
            ["", round_measure(area), round_measure(perimeter)])
        feats.append(feat)
    # What reaches the file, not what the caller handed in. A geometry no
    # repair can save is dropped here, and a caller counting its own input
    # would name a number the file contradicts.
    if isinstance(stats, dict):
        stats["written"] = len(feats)
        stats["skipped"] = skipped
    if skipped:
        QgsMessageLog.logMessage(
            f"Export: {skipped} polygon(s) no repair could save, left out of "
            "the file",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
    if not feats:
        return None
    # A rejected batch must not be reported as a saved file: without this the
    # writer would produce an empty layer and the caller would tell the user
    # their polygons are on disk.
    if not pr.addFeatures(feats):
        QgsMessageLog.logMessage(
            f"Export: could not stage {len(feats)} polygon(s) for writing",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        return None
    temp_layer.updateExtents()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as err:
            QgsMessageLog.logMessage(
                f"Export: cannot create directory: {err}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning,
            )
            return None

    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = driver
    options.fileEncoding = "UTF-8"
    options.layerName = name
    # A GeoPackage holds several tables and is often already open as a map
    # layer: replace the one table, never the file. Recreating the file drops
    # every other table in it, and a file with a live handle cannot be unlinked
    # at all on Windows.
    options.actionOnExistingFile = (
        QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
        if driver == "GPKG" and os.path.exists(output_path)
        else QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
    )
    if driver != "GPKG" and os.path.exists(output_path):
        # Those drivers all take CreateOrOverwriteFile, which unlinks the
        # target first, and the comment above applies to them too: the OGR
        # provider holds a handle for every loaded layer. Exporting a second
        # time over a name the user already has open fails on Windows, and a
        # Shapefile unlinks its sidecars one by one, so it can leave a .shp
        # with no .dbf. The caller reloads the file right after the write, so
        # dropping the stale layer here is the same layer coming back current.
        released = _release_project_layers_at(output_path)
        if released:
            QgsMessageLog.logMessage(
                f"Export: released {released} project layer(s) holding the "
                f"target file open before overwriting it",
                "AI Segmentation", level=Qgis.MessageLevel.Info,
            )
    if driver in _WGS84_ONLY_DRIVERS:
        # Both formats define their coordinates on WGS84 and nowhere else, so
        # they get reprojected coordinates whatever the project is set to. A
        # GeoJSON written in a projected CRS carries no CRS a reader is obliged
        # to honour, and a web map drops it near 0,0.
        target = QgsCoordinateReferenceSystem("EPSG:4326")
    else:
        # Ground metres, so a length read off the file is a length on the
        # ground. See layer_conventions.pick_output_crs.
        target = pick_output_crs(crs, temp_layer.extent())
    if (crs is not None and crs.isValid() and target is not None and target.isValid() and target != crs):
        options.ct = QgsCoordinateTransform(crs, target, QgsProject.instance())

    error = QgsVectorFileWriter.writeAsVectorFormatV3(
        temp_layer, output_path, QgsProject.instance().transformContext(), options,
    )
    if error[0] != QgsVectorFileWriter.WriterError.NoError:
        QgsMessageLog.logMessage(
            f"Export failed ({driver}): {error[1]}",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        return None

    result_layer = QgsVectorLayer(output_path, name, "ogr")
    if not result_layer.isValid():
        result_layer = QgsVectorLayer(
            f"{output_path}|layername={name}", name, "ogr")
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
