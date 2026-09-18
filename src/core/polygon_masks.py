







# ruff: noqa: E402


from __future__ import annotations

from .venv_manager import ensure_venv_packages_available

ensure_venv_packages_available()

from typing import TYPE_CHECKING

import numpy as np
from qgis.core import Qgis, QgsGeometry, QgsLineString, QgsMessageLog, QgsMultiPolygon, QgsPointXY, QgsPolygon

from .polygon_packing import pack_disjoint_crops
from .polygon_refine import (
    _mask_bounding_window,
)

if TYPE_CHECKING:
    import rasterio


def mask_to_polygons_rasterio(
    mask: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
    simplify_tolerance: float = 0.0
) -> list[QgsGeometry]:



    if mask is None:
        return []

    try:
        from rasterio.features import shapes as get_shapes







        window = _mask_bounding_window(mask)
        if window is None:
            return []
        row0, row1, col0, col1 = window
        mask_uint8 = np.ascontiguousarray(
            mask[row0:row1, col0:col1], dtype=np.uint8)
        if row0 or col0:
            transform = transform * transform.translation(col0, row0)




        shape_generator = get_shapes(
            mask_uint8,
            mask=mask_uint8,
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






    geom = _geojson_to_geometry(geojson)
    if geom is None or geom.isEmpty():
        return None
    if simplify_tolerance <= 0:
        return geom if geom.isGeosValid() else None





    simplified = geom.simplify(simplify_tolerance)
    if (simplified is not None and not simplified.isEmpty()
            and simplified.isGeosValid()):
        return simplified
    if not geom.isGeosValid():
        return None
    if simplified is None:
        return geom
    return _repaired_simplification(geom, simplified)


def simplify_and_revalidate(geom: QgsGeometry,
                            tolerance: float) -> QgsGeometry:








    if tolerance <= 0:
        return geom
    simplified = geom.simplify(tolerance)
    if simplified is None:
        return geom
    if not simplified.isEmpty() and simplified.isGeosValid():
        return simplified
    return _repaired_simplification(geom, simplified)


def _repaired_simplification(geom: QgsGeometry,
                             simplified: QgsGeometry) -> QgsGeometry:


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


























    out: list[list[QgsGeometry]] = [[] for _ in crops]
    if not crops:
        return out
    bbox = transform_info.get("bbox")
    if not bbox:
        return out
    try:
        from rasterio import Env as rasterio_env
        from rasterio.features import shapes as get_shapes
        from rasterio.transform import from_bounds as transform_from_bounds
    except ImportError:







        return [
            mask_to_polygons_fallback(
                crop, transform_info, simplify_tolerance, (col0, row0), full_shape)
            for crop, (row0, col0) in crops
        ]




    minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
    full_h, full_w = int(full_shape[0]), int(full_shape[1])
    px_w = (maxx - minx) / max(full_w, 1)
    px_h = (maxy - miny) / max(full_h, 1)

    boxes = []
    for crop, (row0, col0) in crops:
        h, w = int(crop.shape[0]), int(crop.shape[1])
        boxes.append((int(row0), int(row0) + h - 1, int(col0), int(col0) + w - 1))

    try:




        with rasterio_env():
            for indices, (br0, br1, bc0, bc1) in pack_disjoint_crops(boxes, max_side):
                lab = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), np.int32)
                for label, i in enumerate(indices, start=1):
                    crop, _origin = crops[i]
                    r0, _r1, c0, _c1 = boxes[i]
                    ro, co = r0 - br0, c0 - bc0



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
    except Exception as exc:  # noqa: BLE001
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


def _polygon_from_geojson_rings(rings: list) -> QgsPolygon:







    polygon = QgsPolygon()
    for index, ring in enumerate(rings):
        line = QgsLineString([point[0] for point in ring],
                             [point[1] for point in ring])
        if index == 0:
            polygon.setExteriorRing(line)
        else:
            polygon.addInteriorRing(line)
    return polygon


def _geojson_to_geometry(geojson: dict) -> QgsGeometry | None:








    geom_type = geojson.get("type", "")
    coords = geojson.get("coordinates", [])

    if geom_type == "Polygon":
        return QgsGeometry(_polygon_from_geojson_rings(coords))

    if geom_type == "MultiPolygon":
        multi = QgsMultiPolygon()
        for polygon in coords:
            multi.addGeometry(_polygon_from_geojson_rings(polygon))
        return QgsGeometry(multi)

    return None


def mask_to_polygons(
    mask: np.ndarray,
    transform_info: dict,
    simplify_tolerance: float = 0.0,
    pixel_offset: tuple[int, int] | None = None,
    full_shape: tuple[int, int] | None = None,
) -> list[QgsGeometry]:




    if mask is None or not mask.any():
        return []

    try:
        from rasterio.transform import from_bounds as transform_from_bounds

        bbox = transform_info.get("bbox")

        if bbox:






            minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]







            if pixel_offset is not None and full_shape is not None:






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

        rings = []
        for contour in contours:
            if len(contour) < 3:
                continue

            map_points = []
            for px, py in contour:




                mx, my = pixel_to_map_coords(px + 0.5, py + 0.5, transform_info)
                map_points.append(QgsPointXY(mx, my))

            if map_points[0] != map_points[-1]:
                map_points.append(map_points[0])
            if len(map_points) >= 4:
                rings.append(map_points)




        from .contour_rings import rings_to_polygons

        geometries = []
        for geom in rings_to_polygons(rings):
            if geom is None or geom.isEmpty():
                continue
            if not geom.isGeosValid():



                fixed = polygonal_part_of(geom.makeValid())
                if fixed is None:
                    continue
                geom = fixed




            geom = simplify_and_revalidate(geom, simplify_tolerance)
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


def polygonal_part_of(geom: QgsGeometry | None) -> QgsGeometry | None:







    from .qt_compat import PolygonGeometry

    if geom is None or geom.isEmpty():
        return None
    if geom.type() == PolygonGeometry:
        return geom
    parts = [
        part for part in geom.asGeometryCollection()
        if part is not None and not part.isEmpty()
        and part.type() == PolygonGeometry
    ]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    collected = QgsGeometry.collectGeometry(parts)
    if collected is None or collected.isEmpty():
        return parts[0]
    return collected


def find_contours(mask: np.ndarray) -> list[list[tuple[float, float]]]:




    try:
        from skimage import measure





        bordered = np.pad(mask.astype(float), 1, mode="constant",
                          constant_values=0.0)
        raw_contours = measure.find_contours(bordered, 0.5)
        contours = []
        for contour in raw_contours:



            points = [(float(c[1]) - 1.0, float(c[0]) - 1.0) for c in contour]
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






    foreground = padded == 1
    is_background = padded == 0
    on_boundary = np.zeros_like(foreground)
    for dx, dy in directions:
        on_boundary[1:h + 1, 1:w + 1] |= is_background[
            1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
    on_boundary &= foreground


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
