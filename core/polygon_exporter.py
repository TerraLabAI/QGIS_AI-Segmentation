from typing import List, Optional, Tuple
import numpy as np

from qgis.core import (
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsPolygon,
    QgsLineString,
    QgsField,
    QgsFields,
    QgsProject,
    QgsVectorFileWriter,
    QgsCoordinateReferenceSystem,
    QgsMessageLog,
    Qgis,
)
from qgis.PyQt.QtCore import QVariant


def mask_to_polygons_rasterio(
    mask: np.ndarray,
    transform: 'rasterio.Affine',
    crs: str,
    simplify_tolerance: float = 0.0
) -> List[QgsGeometry]:
    from .import_guard import assert_package_isolated
    assert_package_isolated('numpy', np)

    if mask is None or mask.sum() == 0:
        QgsMessageLog.logMessage(
            "mask_to_polygons: Empty or None mask",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return []

    try:
        import rasterio
        from rasterio.features import shapes as get_shapes
        assert_package_isolated('rasterio', rasterio)

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

            geom = QgsGeometry.fromWkt(geojson_to_wkt(geojson_geom))
            if geom and not geom.isEmpty() and geom.isGeosValid():
                if simplify_tolerance > 0:
                    geom = geom.simplify(simplify_tolerance)
                geometries.append(geom)

        QgsMessageLog.logMessage(
            f"mask_to_polygons: Created {len(geometries)} polygons",
            "AI Segmentation",
            level=Qgis.Info
        )

        return geometries

    except Exception as e:
        import traceback
        QgsMessageLog.logMessage(
            f"Failed to convert mask to polygons: {str(e)}\n{traceback.format_exc()}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return []


def geojson_to_wkt(geojson: dict) -> str:
    geom_type = geojson.get("type", "")
    coords = geojson.get("coordinates", [])

    if geom_type == "Polygon":
        rings = []
        for ring in coords:
            points = ", ".join([f"{x} {y}" for x, y in ring])
            rings.append(f"({points})")
        return f"POLYGON({', '.join(rings)})"

    elif geom_type == "MultiPolygon":
        polygons = []
        for polygon in coords:
            rings = []
            for ring in polygon:
                points = ", ".join([f"{x} {y}" for x, y in ring])
                rings.append(f"({points})")
            polygons.append(f"({', '.join(rings)})")
        return f"MULTIPOLYGON({', '.join(polygons)})"

    return ""


def mask_to_polygons(
    mask: np.ndarray,
    transform_info: dict,
    simplify_tolerance: float = 0.0
) -> List[QgsGeometry]:
    if mask is None or mask.sum() == 0:
        QgsMessageLog.logMessage(
            f"mask_to_polygons: Empty or None mask (sum={mask.sum() if mask is not None else 'None'})",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return []

    try:
        import rasterio
        from rasterio.transform import from_bounds as transform_from_bounds
        from .import_guard import assert_package_isolated
        assert_package_isolated('rasterio', rasterio)

        bbox = transform_info.get("bbox")
        img_shape = transform_info.get("img_shape")

        if bbox and img_shape:
            minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
            height, width = img_shape

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
            level=Qgis.Warning
        )
        return mask_to_polygons_fallback(mask, transform_info, simplify_tolerance)


def mask_to_polygons_fallback(
    mask: np.ndarray,
    transform_info: dict,
    simplify_tolerance: float = 0.0
) -> List[QgsGeometry]:
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
                line = QgsLineString([p for p in map_points])
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
            level=Qgis.Warning
        )
        return []


def find_contours(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
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
    padded = np.pad(mask, 1, mode='constant', constant_values=0)
    visited_pad = np.pad(visited, 1, mode='constant', constant_values=True)

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
    directions: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
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
) -> Tuple[float, float]:
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


def export_to_geopackage(
    layer: QgsVectorLayer,
    output_path: str
) -> Tuple[bool, str]:
    try:
        if not output_path.lower().endswith('.gpkg'):
            output_path += '.gpkg'

        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.fileEncoding = "UTF-8"

        error = QgsVectorFileWriter.writeAsVectorFormatV3(
            layer,
            output_path,
            QgsProject.instance().transformContext(),
            options
        )

        if error[0] == QgsVectorFileWriter.NoError:
            QgsMessageLog.logMessage(
                f"Exported to: {output_path}",
                "AI Segmentation",
                level=Qgis.Success
            )
            return True, f"Successfully exported to {output_path}"
        else:
            return False, f"Export error: {error[1]}"

    except Exception as e:
        return False, f"Export failed: {str(e)}"
