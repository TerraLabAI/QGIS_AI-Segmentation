

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
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant


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
        QgsMessageLog.logMessage(
            f"mask_to_polygons: mask shape={mask.shape}, non-zero pixels={mask.sum()}, "
            f"extent={transform_info.get('extent', 'N/A')}, "
            f"original_size={transform_info.get('original_size', 'N/A')}",
            "AI Segmentation",
            level=Qgis.Info
        )

        contours = find_contours(mask)

        QgsMessageLog.logMessage(
            f"mask_to_polygons: Found {len(contours)} contours",
            "AI Segmentation",
            level=Qgis.Info
        )

        if not contours:
            return []

        geometries = []
        for i, contour in enumerate(contours):
            if len(contour) < 3:
                continue

            map_points = []
            for px, py in contour:
                mx, my = pixel_to_map_coords(px, py, transform_info)
                map_points.append(QgsPointXY(mx, my))

            if i == 0 and len(contour) > 0:
                first_px, first_py = contour[0]
                first_mx, first_my = map_points[0].x(), map_points[0].y()
                QgsMessageLog.logMessage(
                    f"First contour: {len(contour)} points, "
                    f"pixel[0]=({first_px}, {first_py}) -> map=({first_mx:.2f}, {first_my:.2f})",
                    "AI Segmentation",
                    level=Qgis.Info
                )

            if map_points[0] != map_points[-1]:
                map_points.append(map_points[0])

            if len(map_points) >= 4:  # Minimum 3 points + closing point
                line = QgsLineString([p for p in map_points])
                polygon = QgsPolygon()
                polygon.setExteriorRing(line)
                geom = QgsGeometry(polygon)

                if simplify_tolerance > 0:
                    geom = geom.simplify(simplify_tolerance)

                if geom.isGeosValid():
                    geometries.append(geom)

        QgsMessageLog.logMessage(
            f"mask_to_polygons: Created {len(geometries)} valid geometries",
            "AI Segmentation",
            level=Qgis.Info
        )

        if not geometries and mask.sum() > 0:
            QgsMessageLog.logMessage(
                "mask_to_polygons: No contours found, creating bounding box fallback",
                "AI Segmentation",
                level=Qgis.Warning
            )
            bbox_geom = create_bounding_box_polygon(mask, transform_info)
            if bbox_geom:
                geometries.append(bbox_geom)

        return geometries

    except Exception as e:
        import traceback
        QgsMessageLog.logMessage(
            f"Failed to convert mask to polygons: {str(e)}\n{traceback.format_exc()}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return []


def create_bounding_box_polygon(
    mask: np.ndarray,
    transform_info: dict
) -> Optional[QgsGeometry]:
    
    try:
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return None

        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        tl = pixel_to_map_coords(min_col, min_row, transform_info)  # top-left
        tr = pixel_to_map_coords(max_col, min_row, transform_info)  # top-right
        br = pixel_to_map_coords(max_col, max_row, transform_info)  # bottom-right
        bl = pixel_to_map_coords(min_col, max_row, transform_info)  # bottom-left

        points = [
            QgsPointXY(tl[0], tl[1]),
            QgsPointXY(tr[0], tr[1]),
            QgsPointXY(br[0], br[1]),
            QgsPointXY(bl[0], bl[1]),
            QgsPointXY(tl[0], tl[1]),  # Close the polygon
        ]

        line = QgsLineString([p for p in points])
        polygon = QgsPolygon()
        polygon.setExteriorRing(line)
        geom = QgsGeometry(polygon)

        QgsMessageLog.logMessage(
            f"Created bounding box: pixel ({min_col}, {min_row}) to ({max_col}, {max_row}), "
            f"map ({tl[0]:.2f}, {tl[1]:.2f}) to ({br[0]:.2f}, {br[1]:.2f})",
            "AI Segmentation",
            level=Qgis.Info
        )

        return geom if geom.isGeosValid() else None

    except Exception as e:
        QgsMessageLog.logMessage(
            f"Failed to create bounding box: {str(e)}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return None


def find_contours(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    
    try:
        from skimage import measure
        raw_contours = measure.find_contours(mask.astype(float), 0.5)
        contours = []
        for contour in raw_contours:
            points = [(int(c[1]), int(c[0])) for c in contour]
            if len(points) >= 3:
                contours.append(points)
        QgsMessageLog.logMessage(
            f"find_contours (skimage): Found {len(contours)} contours",
            "AI Segmentation",
            level=Qgis.Info
        )
        return contours
    except ImportError:
        pass

    QgsMessageLog.logMessage(
        "find_contours: Using simple boundary algorithm (skimage not available)",
        "AI Segmentation",
        level=Qgis.Info
    )

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
    prev_dir = 0  # Start searching from the right

    max_iterations = mask.shape[0] * mask.shape[1]  # Safety limit
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
    
    extent = transform_info["extent"]
    original_size = transform_info["original_size"]

    x_min, y_min, x_max, y_max = extent

    if isinstance(original_size, (list, tuple)):
        height, width = original_size[0], original_size[1]
    else:
        height = width = original_size

    map_x = x_min + (pixel_x / width) * (x_max - x_min)
    map_y = y_max - (pixel_y / height) * (y_max - y_min)

    return map_x, map_y


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


def export_mask_to_geopackage(
    mask: np.ndarray,
    transform_info: dict,
    score: float,
    output_path: str,
    layer_name: str = "segmentation"
) -> Tuple[bool, str]:
    
    try:
        geometries = mask_to_polygons(mask, transform_info)

        if not geometries:
            return False, "No valid geometries could be created from the mask"

        if not output_path.lower().endswith('.gpkg'):
            output_path += '.gpkg'

        crs_authid = transform_info.get("layer_crs", "EPSG:4326")
        crs = QgsCoordinateReferenceSystem(crs_authid)

        uri = f"Polygon?crs={crs.authid()}"
        temp_layer = QgsVectorLayer(uri, layer_name, "memory")

        provider = temp_layer.dataProvider()
        fields = QgsFields()
        fields.append(QgsField("id", QVariant.Int))
        fields.append(QgsField("score", QVariant.Double))
        fields.append(QgsField("area", QVariant.Double))
        provider.addAttributes(fields)
        temp_layer.updateFields()

        features = []
        for i, geom in enumerate(geometries, start=1):
            feature = QgsFeature(temp_layer.fields())
            feature.setGeometry(geom)
            feature.setAttribute("id", i)
            feature.setAttribute("score", score)
            feature.setAttribute("area", geom.area())
            features.append(feature)

        provider.addFeatures(features)
        temp_layer.updateExtents()

        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.fileEncoding = "UTF-8"
        options.layerName = layer_name

        error = QgsVectorFileWriter.writeAsVectorFormatV3(
            temp_layer,
            output_path,
            QgsProject.instance().transformContext(),
            options
        )

        if error[0] == QgsVectorFileWriter.NoError:
            QgsMessageLog.logMessage(
                f"Exported mask to: {output_path} (layer: {layer_name})",
                "AI Segmentation",
                level=Qgis.Success
            )
            return True, f"Exported to {output_path}"
        else:
            return False, f"Export error: {error[1]}"

    except Exception as e:
        import traceback
        QgsMessageLog.logMessage(
            f"Export failed: {str(e)}\n{traceback.format_exc()}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return False, f"Export failed: {str(e)}"
