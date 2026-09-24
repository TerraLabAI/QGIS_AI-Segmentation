







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
from .polygon_trace import gdal_vertex_rule, polygon_wkb, trace_crop_rings, vertex_tables_many

if TYPE_CHECKING:
    import rasterio




POLYGONIZERS = ("gdal", "tracer", "fallback", "fallback_fast")


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






    return _finish_traced_geometry(_geojson_to_geometry(geojson), simplify_tolerance)


def _finish_traced_geometry(geom: QgsGeometry | None,
                            simplify_tolerance: float) -> QgsGeometry | None:



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
    outlines: list | None = None,
    skip_below_area: float = 0.0,
    path_counts: dict | None = None,
) -> list[list[QgsGeometry]]:













































    out: list[list[QgsGeometry]] = [[] for _ in crops]
    if not crops:
        return out
    bbox = transform_info.get("bbox")
    if not bbox:
        return out




    use_tracer = _fast_ring_tracer_enabled()

    try:
        from rasterio import Env as rasterio_env
        from rasterio.features import shapes as get_shapes
        from rasterio.transform import from_bounds as transform_from_bounds
    except ImportError:
        crops = _filled_crops(crops, outlines)









        out = []
        walked = 0
        for i, (crop, (row0, col0)) in enumerate(crops):
            outline = outlines[i] if outlines is not None else None
            geoms = None
            if use_tracer:
                try:
                    geoms = fallback_polygons_traced(
                        crop, outline, transform_info, simplify_tolerance,
                        (col0, row0), full_shape)
                except Exception:  # noqa: BLE001
                    geoms = None
            if geoms is None:
                walked += 1
                geoms = mask_to_polygons_fallback(
                    crop, transform_info, simplify_tolerance, (col0, row0), full_shape)
            out.append(geoms)
        if path_counts is not None:
            _count_polygonizer(path_counts, "fallback", walked)
            _count_polygonizer(path_counts, "fallback_fast", len(crops) - walked)
        return out




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


            rule = gdal_vertex_rule() if use_tracer else None
            packs = []
            for indices, (br0, br1, bc0, bc1) in pack_disjoint_crops(boxes, max_side):
                pack_h, pack_w = br1 - br0 + 1, bc1 - bc0 + 1
                pack_minx = minx + bc0 * px_w
                pack_maxy = maxy - br0 * px_h
                transform = transform_from_bounds(
                    pack_minx, pack_maxy - pack_h * px_h,
                    pack_minx + pack_w * px_w, pack_maxy,
                    pack_w, pack_h,
                )



                px_area = abs(transform.a * transform.e)
                skip_box = (skip_below_area * (1.0 - 1e-6) / px_area
                            if skip_below_area > 0.0 and px_area > 0.0 else 0.0)
                planned = (_plan_pack_traced(crops, indices, outlines, skip_box)
                           if rule is not None else None)
                packs.append((indices, br0, bc0, pack_h, pack_w, transform, planned))
            traced = [p for p in packs if p[6] is not None]
            tables = vertex_tables_many(
                [(p[5].to_gdal(), p[3], p[4]) for p in traced], rule) if traced else []
            for (_indices, br0, bc0, _h, _w, _t, planned), table in zip(traced, tables):
                for i, polygons in planned:
                    r0, _r1, c0, _c1 = boxes[i]
                    for polygon in polygons:
                        geom = QgsGeometry()
                        geom.fromWkb(polygon_wkb(polygon, table, r0 - br0, c0 - bc0))
                        geom = _finish_traced_geometry(geom, simplify_tolerance)
                        if geom is not None:
                            out[i].append(geom)
            by_gdal = 0
            for indices, br0, bc0, pack_h, pack_w, transform, planned in packs:
                if planned is not None:
                    continue
                by_gdal += len(indices)
                lab = np.zeros((pack_h, pack_w), np.int32)
                for label, i in enumerate(indices, start=1):
                    crop, _origin = crops[i]
                    outline = outlines[i] if outlines is not None else None
                    if outline is not None and outline.pixels is not None:

                        crop = outline.pixels()
                    r0, _r1, c0, _c1 = boxes[i]
                    ro, co = r0 - br0, c0 - bc0



                    stamp = crop if crop.dtype == np.bool_ else crop != 0
                    np.copyto(lab[ro:ro + crop.shape[0], co:co + crop.shape[1]],
                              np.int32(label), where=stamp)
                for geojson_geom, value in get_shapes(
                    lab, mask=lab > 0, connectivity=4, transform=transform
                ):
                    label = int(value)
                    if label <= 0 or label > len(indices):
                        continue
                    geom = _finish_polygon(geojson_geom, simplify_tolerance)
                    if geom is not None:
                        out[indices[label - 1]].append(geom)
        if path_counts is not None:
            _count_polygonizer(path_counts, "gdal", by_gdal)
            _count_polygonizer(path_counts, "tracer", len(crops) - by_gdal)
        return out
    except Exception as exc:  # noqa: BLE001
        QgsMessageLog.logMessage(
            f"Packed polygonize failed ({exc}); falling back to per-mask",
            "AI Segmentation", level=Qgis.MessageLevel.Warning,
        )
        crops = _filled_crops(crops, outlines)
        if path_counts is not None:

            _count_polygonizer(path_counts, "gdal", len(crops))
        return [
            mask_to_polygons(
                crop, transform_info, simplify_tolerance=simplify_tolerance,
                pixel_offset=(int(col0), int(row0)), full_shape=full_shape,
            )
            for crop, (row0, col0) in crops
        ]


def _fast_ring_tracer_enabled() -> bool:

    try:
        from .server_dials import feature_enabled

        return feature_enabled("fast_ring_tracer")
    except Exception:  # noqa: BLE001  # nosec B110
        return True


def _count_polygonizer(path_counts: dict, name: str, crops: int) -> None:

    if crops:
        path_counts[name] = path_counts.get(name, 0) + crops


def _filled_crops(crops: list, outlines: list | None) -> list:



    if outlines is None:
        return crops
    return [(o.pixels(), origin) if o is not None and o.pixels is not None
            else (crop, origin)
            for (crop, origin), o in zip(crops, outlines)]


def _plan_pack_traced(crops: list, indices: list, outlines: list | None,
                      skip_box: float = 0.0) -> list | None:



    planned = []
    try:
        for i in indices:
            outline = outlines[i] if outlines is not None else None
            if outline is None:
                outline = trace_crop_rings(crops[i][0])
            polygons = outline.polygons(skip_box) if outline is not None else None
            if polygons is None:
                return None
            planned.append((i, polygons))
    except Exception:  # noqa: BLE001
        return None
    return planned


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





        transform_info = dict(transform_info)
        transform_info["img_shape"] = tuple(mask.shape)
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

        return _finish_fallback_polygons(rings_to_polygons(rings), simplify_tolerance)

    except Exception as e:
        import traceback
        QgsMessageLog.logMessage(
            f"Fallback polygon conversion failed: {str(e)}\n{traceback.format_exc()}",
            "AI Segmentation",
            level=Qgis.MessageLevel.Warning
        )
        return []


def _finish_fallback_polygons(polygons: list, simplify_tolerance: float) -> list:



    geometries = []
    for geom in polygons:
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


def fallback_polygons_traced(
    mask: np.ndarray,
    outline,
    transform_info: dict,
    simplify_tolerance: float,
    pixel_offset: tuple[int, int],
    full_shape: tuple[int, int],
) -> list[QgsGeometry] | None:














    from .contour_rings import _max_traced_rings, _oriented
    from .polygon_trace import fallback_contours, trace_crop_rings, walked_contours

    if outline is None:
        outline = trace_crop_rings(mask)
    if outline is None:
        return None
    walked = None
    if outline.saddle_free:
        contours = fallback_contours(outline)
    else:
        walked = walked_contours(outline)
        contours = walked
    if contours is None:
        return None
    if not contours:
        return []

    bbox = transform_info["bbox"]
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

    xs = sub_minx + (np.arange(sub_w + 1, dtype=np.float64) / sub_w) * (sub_maxx - sub_minx)
    ys = sub_maxy - (np.arange(sub_h + 1, dtype=np.float64) / sub_h) * (sub_maxy - sub_miny)

    if walked is not None:
        polygons = _walked_polygons(
            [(np.append(x, x[0]).tolist(), np.append(y, y[0]).tolist())
             for x, y in ((xs[cols], ys[rows]) for rows, cols in walked)])
        if polygons is None:
            return None
        return _finish_fallback_polygons(polygons, simplify_tolerance)

    def line(rows, cols):
        x = xs[cols]
        y = ys[rows]
        return QgsLineString(np.append(x, x[0]).tolist(), np.append(y, y[0]).tolist())

    def polygon(exterior, interiors=()):
        poly = QgsPolygon()
        poly.setExteriorRing(line(*exterior))
        for rows, cols in interiors:
            poly.addInteriorRing(line(rows, cols))
        return _oriented(QgsGeometry(poly))

    if len(contours) == 1 or len(contours) > _max_traced_rings():
        polygons = [polygon((rows, cols)) for rows, cols, _k in contours]
    else:


        owner = _hole_owners(outline)
        holes: dict = {}
        for rows, cols, k in contours:
            if not outline.outer[k]:
                holes.setdefault(owner[k], []).append((rows, cols))
        polygons = [polygon((rows, cols), holes.get(k, ()))
                    for rows, cols, k in contours if outline.outer[k]]
    return _finish_fallback_polygons(polygons, simplify_tolerance)


def _walked_polygons(rings: list) -> list | None:















    from .contour_rings import _containers, _max_traced_rings, _oriented

    if len(rings) > _max_traced_rings():
        return None

    def solid(x, y):
        try:
            polygon = QgsPolygon()
            polygon.setExteriorRing(QgsLineString(x, y))
            geom = QgsGeometry(polygon)
            return None if geom.isEmpty() else _oriented(geom)
        except Exception:  # noqa: BLE001
            return None

    if len(rings) == 1:
        return [g for g in (solid(*rings[0]),) if g is not None]
    try:
        solids = [solid(x, y) for x, y in rings]
        containers = _containers([[QgsPointXY(x[0], y[0])] for x, y in rings], solids)
        outers: list = []
        holes: dict = {}
        for i, inside in enumerate(containers):
            if solids[i] is not None and len(inside) % 2 == 0:
                outers.append(i)
                holes.setdefault(i, [])
        for i, inside in enumerate(containers):
            if solids[i] is None or len(inside) % 2 == 0:
                continue
            owner = max(inside, key=lambda j: len(containers[j]))
            if owner in holes:
                holes[owner].append(i)
        out = []
        for i in outers:
            polygon = QgsPolygon()
            polygon.setExteriorRing(QgsLineString(*rings[i]))
            for k in holes.get(i, []):
                polygon.addInteriorRing(QgsLineString(*rings[k]))
            out.append(_oriented(QgsGeometry(polygon)))
        return out or [g for g in solids if g is not None]
    except Exception:  # noqa: BLE001
        return None


def _hole_owners(outline) -> dict:


    from .polygon_trace import _inside_ring, _ring_pixel_area

    outer_ids = [k for k, o in enumerate(outline.outer) if o]
    owner = {}
    for k, is_outer in enumerate(outline.outer):
        if is_outer:
            continue
        rows, cols = outline.rings[k]
        r, c = int(rows[0]) - 1, int(cols[0])
        best, best_area = None, None
        for j in outer_ids:
            lo_r, hi_r, lo_c, hi_c = outline.boxes[j]
            if not (lo_r <= r < hi_r and lo_c <= c < hi_c):
                continue
            if not _inside_ring(r, c, outline.rings[j]):
                continue
            area = _ring_pixel_area(*outline.rings[j])
            if best_area is None or area < best_area:
                best, best_area = j, area
        owner[k] = best
    return owner


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






    foreground = np.asarray(mask, dtype=bool)
    bordered = np.pad(foreground, 1, mode="constant")
    boundaries = (
        foreground & ~bordered[:-2, 1:-1],
        foreground & ~bordered[1:-1, 2:],
        foreground & ~bordered[2:, 1:-1],
        foreground & ~bordered[1:-1, :-2],
    )
    offsets = ((0, 0), (1, 0), (1, 1), (0, 1))
    steps = ((1, 0), (0, 1), (-1, 0), (0, -1))
    edges: dict[tuple[int, int], dict[int, tuple[int, int]]] = {}
    for heading, (boundary, (ox, oy), (dx, dy)) in enumerate(
            zip(boundaries, offsets, steps)):
        rows, cols = np.nonzero(boundary)
        for row, col in zip(rows.tolist(), cols.tolist()):
            start = (col + ox, row + oy)
            end = (start[0] + dx, start[1] + dy)
            edges.setdefault(start, {})[heading] = end

    contours = []
    while edges:
        start = next(iter(edges))
        point = start
        heading = next(iter(edges[start]))
        contour = []
        while True:
            contour.append((point[0] - 0.5, point[1] - 0.5))
            outgoing = edges[point]


            for candidate in ((heading + 1) % 4, heading,
                              (heading - 1) % 4, (heading + 2) % 4):
                if candidate in outgoing:
                    heading = candidate
                    following = outgoing.pop(candidate)
                    break
            if not outgoing:
                del edges[point]
            point = following
            if point == start:
                break
        if len(contour) >= 3:
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
