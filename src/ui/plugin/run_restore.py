
























from __future__ import annotations

import math

from qgis.core import Qgis, QgsCoordinateReferenceSystem, QgsMessageLog

from ...core.i18n import tr
from ...core.interaction_dials import (
    restore_align_budget_s,
    restore_align_max_objects,
    restore_confidence_floor,
)
from ...core.qt_compat import (
    DistanceMeters,
    field_type_double,
    field_type_int,
    field_type_string,
)
from .run_zone_clip import (
    clip_geometry_to_zone,
    prepare_zone_engine,
    zone_geometry_from_run,
    zone_polygon_from_wkt,
)



_DEFAULT_START_CONFIDENCE = 0.30




_RESTORE_CONFIDENCE_FLOOR = 0.0





_project_export_context: dict = {}





_RESTORE_ALIGN_MAX_OBJECTS = 400




_RESTORE_ALIGN_BUDGET_S = 2.0



_STAMP_RECT_MARGIN_PX = 8.0


def _log(msg: str, level=None) -> None:
    QgsMessageLog.logMessage(
        msg, "AI Segmentation",
        level=level if level is not None else Qgis.MessageLevel.Info)


def snap_confidence(value, default: float = _DEFAULT_START_CONFIDENCE) -> float:

    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(v) or v <= 0.0 or v > 1.0:
        return default
    step = int(round(v * 100 / 5.0)) * 5
    return max(5, min(95, step)) / 100.0


def _tile_bbox(tile: dict):

    if not isinstance(tile, dict):
        return None
    bb = tile.get("tile_bbox_native") or tile.get("bbox_native")
    try:
        if isinstance(bb, dict):
            vals = (float(bb["xmin"]), float(bb["ymin"]),
                    float(bb["xmax"]), float(bb["ymax"]))
        elif isinstance(bb, (list, tuple)) and len(bb) >= 4:
            vals = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
        else:
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (not all(math.isfinite(value) for value in vals)
            or vals[2] <= vals[0] or vals[3] <= vals[1]):
        return None
    return vals


def zone_extent_from_tiles(tiles: list) -> tuple[tuple, str] | None:









    located = []
    for tile in tiles:
        box = _tile_bbox(tile)
        if box is not None:
            located.append((tile, box))
    boxes = [b for _t, b in located]
    if not boxes:
        return None
    from qgis.core import QgsCoordinateReferenceSystem

    first_crs = None
    authid = ""
    for tile, _box in located:
        crs = QgsCoordinateReferenceSystem(str(tile.get("crs_authid") or ""))
        if not crs.isValid() or (first_crs is not None and crs != first_crs):
            return None
        if first_crs is None:
            first_crs, authid = crs, str(tile["crs_authid"])
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)), authid


def _masks_list(payload) -> list:

    if isinstance(payload, dict):
        payload = payload.get("masks")
    if not isinstance(payload, list):
        return []
    return [m for m in payload if isinstance(m, dict)]


def _tile_stamp_rect(tile: dict, width, height) -> list | None:













    exemplars = tile.get("exemplars")
    if not isinstance(exemplars, list) or not exemplars:
        return None
    from ...core.tile_manager import OVERLAP_FRACTION, TILE_SIZE

    try:
        w = float(width or TILE_SIZE)
        h = float(height or TILE_SIZE)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    band = OVERLAP_FRACTION * TILE_SIZE * (h / TILE_SIZE)
    edge = None
    prev_x1 = None
    x0 = y0 = float("inf")
    x1 = y1 = float("-inf")
    for ex in exemplars:
        box = ex.get("box") if isinstance(ex, dict) else None
        if not isinstance(box, list) or len(box) != 4:
            break
        try:
            bx0, by0, bx1, by1 = (float(v) for v in box)
        except (TypeError, ValueError):
            break
        side = "top" if by1 <= band + 1 else ("bottom" if by0 >= h - band - 1 else None)
        if side is None or (edge is not None and side != edge):
            break
        if prev_x1 is not None and bx0 < prev_x1 - 1:
            break
        edge = side
        prev_x1 = bx1
        x0, y0 = min(x0, bx0), min(y0, by0)
        x1, y1 = max(x1, bx1), max(y1, by1)
    if edge is None:
        return None
    margin = _STAMP_RECT_MARGIN_PX
    return [
        max(0.0, (x0 - margin) / w),
        0.0 if edge == "top" else max(0.0, (y0 - margin) / h),
        min(1.0, (x1 + margin) / w),
        1.0 if edge == "bottom" else min(1.0, (y1 + margin) / h),
    ]


def _run_gsd(tiles: list) -> float:





    from ...core.tile_manager import TILE_SIZE

    widest = 0.0
    for tile in tiles:
        bb = _tile_bbox(tile)
        if bb is not None:
            widest = max(widest, bb[2] - bb[0])
    return widest / TILE_SIZE if widest > 0 else 0.0


def _run_ground_gsd_m(crs_authid: str, tiles: list, gsd: float) -> float:




    if gsd <= 0:
        return 0.0
    boxes = [b for b in (_tile_bbox(t) for t in tiles) if b is not None]
    if not boxes:
        return 0.0
    cx = (min(b[0] for b in boxes) + max(b[2] for b in boxes)) / 2.0
    cy = (min(b[1] for b in boxes) + max(b[3] for b in boxes)) / 2.0
    try:
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsDistanceArea,
            QgsPointXY,
            QgsProject,
        )
        crs = QgsCoordinateReferenceSystem(crs_authid)
        if not crs.isValid():
            return 0.0
        da = QgsDistanceArea()
        da.setSourceCrs(crs, QgsProject.instance().transformContext())
        da.setEllipsoid("WGS84")
        dist = max(
            da.measureLine(QgsPointXY(cx, cy), QgsPointXY(cx + gsd, cy)),
            da.measureLine(QgsPointXY(cx, cy), QgsPointXY(cx, cy + gsd)))
        metres = float(da.convertLengthMeasurement(dist, DistanceMeters))
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return 0.0
    return metres if metres > 0 else 0.0


def _run_stored_float(run: dict, tiles: list, key: str) -> float:











    sources = [run]
    sources.extend(t for t in tiles if isinstance(t, dict))
    for src in sources:
        val = src.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
            return float(val)
    return 0.0


def _run_simplify_mult(run: dict, tiles: list) -> float:

    return _run_stored_float(run, tiles, "tile_simplify_mult")


def _run_pinhole_m(run: dict, tiles: list) -> float:

    return _run_stored_float(run, tiles, "pinhole_m")


def _run_ground_unit_metres(crs, tiles: list) -> tuple[float, float]:














    import math

    box = None
    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        box = _tile_bbox(tile)
        if box is not None:
            break
    if box is None:
        return 1.0, 1.0
    try:
        from qgis.core import (
            QgsCoordinateTransformContext,
            QgsDistanceArea,
            QgsPointXY,
        )

        from ...core.qt_compat import DistanceMeters

        if crs is None or not crs.isValid():
            return 1.0, 1.0
        xmin, ymin, xmax, ymax = box
        span_x = float(xmax - xmin)
        span_y = float(ymax - ymin)
        if span_x <= 0.0 or span_y <= 0.0:
            return 1.0, 1.0
        measurer = QgsDistanceArea()
        measurer.setSourceCrs(crs, QgsCoordinateTransformContext())
        measurer.setEllipsoid("WGS84")
        xmid = (xmin + xmax) / 2.0
        ymid = (ymin + ymax) / 2.0
        width_m = measurer.convertLengthMeasurement(measurer.measureLine(
            QgsPointXY(xmin, ymid), QgsPointXY(xmax, ymid)), DistanceMeters)
        height_m = measurer.convertLengthMeasurement(measurer.measureLine(
            QgsPointXY(xmid, ymin), QgsPointXY(xmid, ymax)), DistanceMeters)



        if (math.isfinite(width_m) and width_m > 0.0
                and math.isfinite(height_m) and height_m > 0.0):
            return width_m / span_x, height_m / span_y
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    return 1.0, 1.0


def capture_project_export_context() -> dict:










    from qgis.core import QgsProject

    try:
        project = QgsProject.instance()
        _project_export_context.clear()
        _project_export_context.update({
            "project_crs": project.crs(),
            "transform_context": project.transformContext(),
            "ellipsoid": str(project.ellipsoid() or ""),
        })
    except (RuntimeError, AttributeError):
        _project_export_context.clear()
    return dict(_project_export_context)


def run_merge_separate(plugin, run: dict) -> bool:









    del plugin, run
    capture_project_export_context()
    return True


def _run_decisions(run: dict) -> dict | None:


    from ...core.run_decisions import parse_restore_decisions

    return parse_restore_decisions(run.get("decisions"))









_ZONE_MARGIN_RUNS_KEY = "AI_Segmentation/zone_margin_runs"
_ZONE_MARGIN_RUNS_KEPT = 300


def _zone_margin_runs(settings) -> list:
    import json
    try:
        raw = settings.value(_ZONE_MARGIN_RUNS_KEY, "", type=str) or ""
        data = json.loads(raw) if raw else []
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(data, list):
        return []
    return [
        item for item in data
        if isinstance(item, list) and len(item) == 2
        and isinstance(item[0], str)
        and isinstance(item[1], (int, float)) and not isinstance(item[1], bool)]


def note_run_zone_margin(run_id: str, margin_m: float) -> None:


    import json
    import math
    try:
        margin = float(margin_m)
        if not run_id or not math.isfinite(margin) or margin <= 0.0:
            return
        from qgis.core import QgsSettings
        settings = QgsSettings()
        runs = [item for item in _zone_margin_runs(settings) if item[0] != run_id]
        runs.append([str(run_id), round(margin, 3)])
        settings.setValue(
            _ZONE_MARGIN_RUNS_KEY, json.dumps(runs[-_ZONE_MARGIN_RUNS_KEPT:]))
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def run_zone_margin_m(run: dict) -> float:





    import math
    served = run.get("zone_keep_margin_m")
    if isinstance(served, (int, float)) and not isinstance(served, bool):
        value = float(served)
        if math.isfinite(value) and 0.0 < value <= 500.0:
            return value
    run_id = str(run.get("run_id") or "")
    if not run_id:
        return 0.0
    try:
        from qgis.core import QgsSettings
        for item_id, margin in reversed(_zone_margin_runs(QgsSettings())):
            if item_id == run_id:
                value = float(margin)
                return value if math.isfinite(value) and value > 0.0 else 0.0
    except Exception:  # noqa: BLE001
        return 0.0
    return 0.0


def _clip_keeping_zone_margin(geom, zone, engine, keep_margin: float):



    if keep_margin > 0.0 and zone is not None and geom is not None:
        try:
            whole = engine is not None and engine.contains(geom.constGet())
            if not whole:
                gap = geom.distance(zone)
                whole = 0.0 <= gap <= keep_margin
        except Exception:  # noqa: BLE001
            whole = False
        if whole:
            return geom
    return clip_geometry_to_zone(geom, zone, engine)


def _rows_centred_in_zone(rows: list, zone, engine) -> list:





    from ...core.zone_membership import rows_overlapping_zone
    return rows_overlapping_zone(rows, zone, engine)


def decode_run_masks(run: dict, tiles: list, masks_per_tile: dict,
                     merge_separate: bool, *, on_tile=None,
                     is_cancelled=None) -> dict | None:














    import math

    from qgis.core import QgsCoordinateReferenceSystem

    from ...core import detection_policy
    from ...core.cloud_detection import (
        mask_cell_size,
        pinhole_fill_limit_px,
        tile_simplify_tolerance,
    )
    from ...core.layer_conventions import repair_polygon, to_multipolygon
    from ...core.mask_crops import iter_detection_crops
    from ...core.polygon_exporter import (
        IncrementalMerger,
        apply_mask_refinement,
        drop_covered_objects,
        mask_to_polygons,
    )
    from ...core.tile_manager import OVERLAP_FRACTION, TILE_SIZE
    from ...workers.auto_worker.mask_geometry import AutoMaskGeometryMixin



    tiles = [tile for tile in tiles if isinstance(tile, dict)]
    crs_authid = str(run.get("crs_authid") or next(
        (tile.get("crs_authid") for tile in tiles if tile.get("crs_authid")), ""))
    run_crs = QgsCoordinateReferenceSystem(crs_authid)
    if not run_crs.isValid():
        raise ValueError("The archived run has no valid coordinate reference system")
    for tile_authid in {str(tile["crs_authid"]) for tile in tiles if tile.get("crs_authid")}:
        tile_crs = QgsCoordinateReferenceSystem(tile_authid)
        if not tile_crs.isValid() or tile_crs != run_crs:
            raise ValueError("Archived tiles use inconsistent coordinate reference systems")




    max_tile_coverage = detection_policy.max_tile_coverage()

    gsd = _run_gsd(tiles)
    simplify_mult = _run_simplify_mult(run, tiles)
    pinhole_m = _run_pinhole_m(run, tiles)




    ground_kx, ground_ky = _run_ground_unit_metres(
        QgsCoordinateReferenceSystem(crs_authid), tiles)
    area_scale = ground_kx * ground_ky
    if not math.isfinite(area_scale) or area_scale <= 0:
        area_scale = 1.0
    length_scale = math.sqrt(area_scale)







    decisions = _run_decisions(run)
    merge_separate = (
        decisions["merge_separate"] if decisions is not None else bool(merge_separate))
    if gsd > 0:
        seam_min_dim = OVERLAP_FRACTION * TILE_SIZE * gsd
    else:
        seam_min_dim = float("inf") if merge_separate else 0.0




    merge_scalars = detection_policy.merge_scalars()
    merger = IncrementalMerger(
        seam_min_dim=seam_min_dim,
        select_duplicates=merge_separate,
        gsd=gsd,
        restore_partitions=(
            merge_separate
            and decisions is not None and decisions["restore_partitions"]),
        **detection_policy.merge_scalar_kwargs(IncrementalMerger, merge_scalars),
    )







    min_keep_px = detection_policy.min_keep_px()
    min_keep_area = (
        max((min_keep_px * gsd) ** 2,
            detection_policy.min_keep_floor_m2(0.0) / area_scale)
        if gsd > 0 else 0.0
    )






    zone = zone_geometry_from_run(run, crs_authid)
    zone_engine = prepare_zone_engine(zone)
    dropped_outside = 0





    keep_margin_m = (
        run_zone_margin_m(run)
        if zone is not None and merge_separate
        and (run.get("prompt") or "").strip() else 0.0)
    keep_margin = keep_margin_m / length_scale if length_scale > 0 else 0.0








    exemplar_only = not (run.get("prompt") or "").strip()
    from ...core.hypothesis_nms import select_tile_hypotheses
    from ...core.server_dials import dial_bool
    from ...workers.auto_detection_worker import AutoDetectionWorker

    nms_kwargs = {k: merge_scalars[k] for k in (
        "ios_threshold", "dup_ios_floor", "dup_centroid_frac") if k in merge_scalars}
    frag_hard_cov = detection_policy.hard_tile_coverage()
    frag_min_fill = detection_policy.compact_min_fill()
    if exemplar_only:
        frag_tile_area = (TILE_SIZE * gsd) ** 2 if gsd > 0 else 0.0








    text_nms = merge_separate or dial_bool("features.map_hypothesis_nms", False)
    shape_escape = detection_policy.hard_cover_shape_escape()
    span_fraction = detection_policy.tile_span_fraction()
    map_cover_floor = detection_policy.map_cover_score_floor(0.0)

    decoded_tiles = 0
    total = len(tiles)
    for index, tile in enumerate(tiles):
        if is_cancelled is not None and is_cancelled():
            return None
        if on_tile is not None:
            on_tile(index, total)
        request_id = tile.get("request_id") or ""
        masks = _masks_list(masks_per_tile.get(request_id))
        if not masks:
            continue
        bb = _tile_bbox(tile)
        if bb is None:
            continue
        xmin, ymin, xmax, ymax = bb
        response = {
            "masks": masks,
            "width": tile.get("output_width"),
            "height": tile.get("output_height"),
        }
        tile_transform = {

            "bbox": (xmin, xmax, ymin, ymax),
            "crs": crs_authid,
        }


        stamp = _tile_stamp_rect(
            tile, tile.get("output_width"), tile.get("output_height"))
        tile_had_masks = False
        tile_frags: list = []






        for mask, score, box in iter_detection_crops(
                response, TILE_SIZE, TILE_SIZE, 0.0):
            if is_cancelled is not None and is_cancelled():
                return None
            if stamp and AutoMaskGeometryMixin._centroid_in_stamp(box, mask, stamp):
                continue
            if not tile_had_masks:
                tile_had_masks = True
                decoded_tiles += 1








            full_h, full_w = mask.full_shape
            cell = mask_cell_size(xmax - xmin, ymax - ymin, full_w, full_h)
            set_pixels = mask.set_pixels
            if set_pixels == 0:
                continue
            row0, col0 = mask.row0, mask.col0
            blob_check = False
            if not exemplar_only and set_pixels > max_tile_coverage * float(full_h * full_w):
                coverage = set_pixels / float(full_h * full_w)
                if merge_separate:
                    if coverage > frag_hard_cov and not shape_escape:
                        continue
                    if (mask.col1 - col0 + 1 >= span_fraction * full_w
                            and mask.row1 - row0 + 1 >= span_fraction * full_h):
                        continue
                    blob_check = True
                elif map_cover_floor > 0.0 and float(score) < map_cover_floor:
                    continue

            sub = mask.padded()




            sub = apply_mask_refinement(
                sub, expand_value=0, fill_holes=True, min_area=0,
                max_hole_px=pinhole_fill_limit_px(
                    gsd * length_scale, cell * length_scale, pinhole_m))
            for geom in mask_to_polygons(
                sub, tile_transform,
                simplify_tolerance=tile_simplify_tolerance(
                    gsd, cell, simplify_mult),
                pixel_offset=(col0 - 1, row0 - 1), full_shape=(full_h, full_w),
            ):
                if geom is None or geom.isEmpty():
                    continue





                clipped = _clip_keeping_zone_margin(
                    geom, zone, zone_engine, keep_margin)
                if clipped is None:
                    dropped_outside += 1
                    continue









                if zone is not None and clipped is geom:
                    geom = to_multipolygon(clipped)
                else:
                    geom = to_multipolygon(repair_polygon(clipped) or clipped)
                if geom is None or geom.isEmpty():
                    continue



                if min_keep_area > 0.0 and geom.area() < min_keep_area:
                    continue
                if blob_check and not AutoDetectionWorker._is_compact_shape(
                        geom, frag_min_fill):
                    continue
                tile_frags.append((geom, float(score)))
        if not exemplar_only and tile_frags:
            if text_nms:
                tile_frags = select_tile_hypotheses(tile_frags, **nms_kwargs)
            for geom, score in tile_frags:
                merger.add(geom, score)
        if exemplar_only and tile_frags:
            for geom, score in select_tile_hypotheses(tile_frags, **nms_kwargs):
                if merge_separate and frag_tile_area > 0:
                    cov = geom.area() / frag_tile_area
                    if cov > frag_hard_cov:
                        continue
                    if cov > max_tile_coverage and not AutoDetectionWorker._is_compact_shape(
                            geom, frag_min_fill):
                        continue
                merger.add(geom, score)

    if on_tile is not None:
        on_tile(total, total)



    merger.restore_absorbed_partitions()


    merged_scored = merger.result_scored_ided()
    if keep_margin > 0.0:
        before_zone = len(merged_scored)
        merged_scored = _rows_centred_in_zone(merged_scored, zone, zone_engine)
        if len(merged_scored) != before_zone:
            _log(f"Run restore: {before_zone - len(merged_scored)} object(s) "
                 f"entirely outside the run's zone dropped")




    merged_scored = drop_covered_objects(merged_scored)
    if dropped_outside:
        _log(f"Run restore: {dropped_outside} detection(s) outside the run's "
             f"zone dropped")
    _log(f"Run restore: decoded {decoded_tiles} tile(s) into {len(merged_scored)} object(s)")
    return {
        "objects": merged_scored,
        "crs_authid": crs_authid,
        "gsd": gsd,
        "merge_separate": merge_separate,
        "zone_wkt": zone.asWkt() if zone is not None else "",
    }


def _align_restore_footprints(plugin, rows: list) -> list:









    import time

    max_objects = restore_align_max_objects(_RESTORE_ALIGN_MAX_OBJECTS)
    if len(rows) > max_objects:
        _log(f"Run restore: footprint alignment skipped on {len(rows)} object(s) "
             f"(over the {max_objects} this caller can wait for)")
        return rows
    try:
        sweep = plugin._auto_footprint_align_sweep(rows)
    except (AttributeError, RuntimeError):
        return rows
    if sweep is None:
        return rows
    deadline = time.monotonic() + restore_align_budget_s(_RESTORE_ALIGN_BUDGET_S)
    from ...core.server_dials import dial_in_range
    step_size = dial_in_range("tuning.export.footprint_align_batch", 64, 8, 256)
    try:
        while not sweep.step(step_size):
            if time.monotonic() >= deadline:
                _log("Run restore: footprint alignment stopped at its time "
                     "budget; the rest keep their archived shape")
                break
        plugin._log_footprint_alignment(sweep)
        return sweep.result()
    except Exception:  # noqa: BLE001
        return rows


def _run_start_confidence(run: dict, tiles: list) -> float:





    threshold = run.get("threshold")
    if threshold is None and tiles:
        threshold = tiles[0].get("threshold")
    default = _restore_default_confidence(run)
    snapped = snap_confidence(threshold, default)
    floor = restore_confidence_floor(_RESTORE_CONFIDENCE_FLOOR)
    decisions = _run_decisions(run)
    if decisions is not None and "restore_confidence_floor" in decisions:
        floor = decisions["restore_confidence_floor"]
    if snapped <= floor + 1e-9:
        return default
    return snapped


def _restore_default_confidence(run: dict) -> float:


    decisions = _run_decisions(run)
    if decisions is not None:
        return float(decisions["start_confidence"])
    return _DEFAULT_START_CONFIDENCE


def _confidence_showing_an_object(conf: float, objects: list) -> float:









    import math

    scores = [float(s) for (_g, s, _a) in objects
              if isinstance(s, (int, float)) and not isinstance(s, bool)
              and math.isfinite(s) and 0.0 <= s <= 1.0]
    if not scores:
        return conf
    best = max(scores)
    if best >= conf:
        return conf
    return max(0, int(math.floor(best * 100 / 5.0)) * 5) / 100.0


def _make_restore_selection_layer(crs_authid: str):



    from qgis.core import QgsField, QgsProject, QgsVectorLayer

    from ...core.layer_conventions import make_review_renderer

    field_str = field_type_string()
    field_dbl = field_type_double()

    try:
        layer = QgsVectorLayer(
            f"MultiPolygon?crs={crs_authid}",
            tr("Auto detection (live)"), "memory")
        if not layer.isValid():
            return None
        pr = layer.dataProvider()





        pr.addAttributes([
            QgsField("label", field_str),
            QgsField("score", field_dbl),
            QgsField("det_id", field_type_int()),
        ])
        layer.updateFields()
        layer.setRenderer(make_review_renderer())
        try:


            from .shared import _apply_fast_render
            _apply_fast_render(layer)
        except Exception:  # nosec B110
            pass


        from ...core.output_store import drop_from_snapping, mark_temp_layer
        mark_temp_layer(layer)
        QgsProject.instance().addMapLayer(layer, False)


        drop_from_snapping(layer)
        QgsProject.instance().layerTreeRoot().insertLayer(0, layer)
        return layer
    except (RuntimeError, AttributeError):
        return None


def _zoom_to_tiles(plugin, tiles: list, crs_authid: str) -> None:

    from qgis.core import (
        QgsCoordinateReferenceSystem,
        QgsCoordinateTransform,
        QgsProject,
        QgsRectangle,
    )

    union = None
    for tile in tiles:
        bb = _tile_bbox(tile)
        if bb is None:
            continue
        rect = QgsRectangle(bb[0], bb[1], bb[2], bb[3])
        if union is None:
            union = rect
        else:
            union.combineExtentWith(rect)
    if union is None or union.isEmpty():
        return
    try:
        canvas = plugin.iface.mapCanvas()
        run_crs = QgsCoordinateReferenceSystem(crs_authid)
        canvas_crs = canvas.mapSettings().destinationCrs()
        if run_crs.isValid() and canvas_crs.isValid() and run_crs != canvas_crs:
            xform = QgsCoordinateTransform(
                run_crs, canvas_crs, QgsProject.instance())
            union = xform.transformBoundingBox(union)
        from ...core.server_dials import dial_in_range
        pad_frac = dial_in_range("tuning.export.restore_zoom_pad_fraction", 0.05, 0.0, 0.5)
        union.grow(max(union.width(), union.height()) * pad_frac)
        canvas.setExtent(union)
        canvas.refresh()
    except Exception:  # nosec B110
        pass


def _run_age_days(run: dict) -> int:
    import calendar
    import time

    ts = str(run.get("started_at") or run.get("created_at") or "")
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = time.strptime(ts[:19], fmt)
            return max(0, int((time.time() - calendar.timegm(parsed)) // 86400))
        except (ValueError, TypeError):
            continue
    return 0


def export_decoded_run(decoded: dict, confidence: float, path: str,
                       driver: str, project_context: dict | None = None) -> dict:






















    from qgis.core import QgsCoordinateReferenceSystem

    from ...core.output_values import confidence_value
    from ...core.polygon_exporter import export_geometries_to_file

    kept = []
    for fid, geom, score in decoded.get("objects") or []:
        value = confidence_value(score)
        if geom is not None and not geom.isEmpty() and (value is None or value >= confidence):
            kept.append((fid, geom, value))
    if not kept:
        return {"count": 0, "written": False}
    geoms = [g for _fid, g, _s in kept]



    det_ids = [fid for fid, _g, _s in kept]
    scores = [s for _fid, _g, s in kept]
    context = (project_context if isinstance(project_context, dict)
               else _project_export_context)
    stats: dict = {}
    layer = export_geometries_to_file(
        geoms, QgsCoordinateReferenceSystem(decoded.get("crs_authid") or ""),
        path, driver=driver, stats=stats,
        project_crs=context.get("project_crs"),
        transform_context=context.get("transform_context"),
        ellipsoid=str(context.get("ellipsoid") or ""),
        scores=scores,
        det_ids=det_ids,
        object_class=str(decoded.get("prompt") or ""),
        source_layer_name=str(decoded.get("source_layer_name") or ""),
        prompt=str(decoded.get("prompt") or ""),
        detail=decoded.get("detail"),
        confidence=confidence)
    written = layer is not None
    del layer
    return {"count": int(stats.get("written") or 0), "written": written}


def load_exported_layer(path: str, driver: str):







    import os

    from qgis.core import QgsVectorLayer

    from ...core.layer_conventions import make_committed_renderer
    from ...core.output_store import apply_fast_canvas_render

    name = os.path.splitext(os.path.basename(path))[0] or "detections"





    layer = QgsVectorLayer(f"{path}|layername={name}", name, "ogr")
    if not layer.isValid():
        layer = QgsVectorLayer(path, name, "ogr")
    if not layer.isValid():
        _log(f"Run export: file saved but could not be loaded back: {path}",
             Qgis.MessageLevel.Warning)
        return None
    if driver != "GPKG":


        layer.setRenderer(make_committed_renderer())




    apply_fast_canvas_render(layer)
    return layer


def restore_blocked_by_live_run(plugin) -> bool:


    worker = getattr(plugin, "_auto_worker", None)
    return bool(
        (worker is not None and worker.isRunning())
        or getattr(plugin, "_auto_review", None) is not None
        or getattr(plugin, "_auto_finalize_state", None) is not None
        or getattr(plugin, "_auto_start_in_progress", False))


def restore_run(plugin, run: dict, tiles: list, decoded: dict) -> bool:











    if plugin is None or not tiles:
        return False


    if restore_blocked_by_live_run(plugin):
        try:
            plugin.iface.messageBar().pushWarning(
                "AI Segmentation",
                tr("Finish or exit the current run before restoring a past one."))
        except (RuntimeError, AttributeError):
            pass
        return False

    merged_scored = (decoded or {}).get("objects") or []
    crs_authid = (decoded or {}).get("crs_authid") or ""
    if not QgsCoordinateReferenceSystem(crs_authid).isValid():
        try:
            plugin.iface.messageBar().pushWarning(
                "AI Segmentation",
                tr("Could not restore this run because its coordinate reference system is missing or invalid."))
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        return False



    forget = getattr(plugin, "_forget_auto_last_run_km2", None)
    if callable(forget):
        forget()
    gsd = float((decoded or {}).get("gsd") or 0.0)
    merge_separate = bool((decoded or {}).get("merge_separate", True))
    if not merged_scored:
        try:
            plugin.iface.messageBar().pushWarning(
                "AI Segmentation",
                tr("Could not rebuild this run's detections."))
        except (RuntimeError, AttributeError):
            pass
        return False

    prompt = (run.get("prompt") or "").strip()
    conf = _run_start_confidence(run, tiles)

    plugin._auto_start_confidence_default = _restore_default_confidence(run)


    plugin._ensure_dock_widget()
    dock = plugin.dock_widget
    if dock is None:
        return False
    try:
        from ..ai_segmentation_dockwidget import Mode
        if dock._mode != Mode.AUTOMATIC:
            if dock._on_mode_selected(Mode.AUTOMATIC) is False:
                return False
    except (RuntimeError, AttributeError, ImportError):
        pass

    plugin._reset_auto_live_pipeline()
    plugin._auto_merger = None
    plugin._auto_worker = None
    plugin._auto_headless_run = False


    plugin._auto_tel_stop_reason = "restored"
    plugin._auto_run_id = str(run.get("run_id") or run.get("group_key") or "")
    plugin._auto_crs_authid = crs_authid
    plugin._auto_gsd = gsd






    plugin._auto_mask_gsd = 0.0
    plugin._auto_gsd_m = _run_ground_gsd_m(crs_authid, tiles, gsd)
    plugin._auto_merge_separate = merge_separate


    plugin._auto_zone_keep_margin_m = 0.0
    plugin._auto_is_exemplar_only = not prompt
    plugin._auto_confidence = conf
    plugin._auto_raw_count = len(merged_scored)
    plugin._auto_dense_tiles = 0
    plugin._auto_preview_geoms = []





    plugin._auto_manual_removed = set()
    plugin._auto_correction_removed = set()
    plugin._auto_manual_object_ids = set()




    plugin._auto_clip_polygon = zone_polygon_from_wkt(
        (decoded or {}).get("zone_wkt"))
    plugin._auto_clip_engine = None
    plugin._auto_zone = None
    plugin._auto_zone_polygon = None
    from ...core.activation_manager import auth_revision
    from ...core.detection_history import account_history_dir

    plugin._auto_run_ctx = {
        "auth_revision": auth_revision(),
        "account_dir": account_history_dir(),
        "prompt": prompt,
        "crs_authid": crs_authid,
        "layer_id": None,
        "zone": None,
        "detail": None,
        "detection_threshold": conf,
        "exemplars": None,
        "total": len(tiles),
        "restored": True,
    }





    merged_scored = _align_restore_footprints(plugin, merged_scored)
    plugin._auto_objects = plugin._build_auto_objects(merged_scored)
    if not plugin._auto_objects:
        return False





    conf = _confidence_showing_an_object(conf, plugin._auto_objects)
    plugin._auto_confidence = conf



    try:
        dock.set_prompt_text(prompt, source="restore")
        spin = dock.auto_confidence_spin
        spin.blockSignals(True)
        spin.setValue(conf)
        spin.blockSignals(False)
        dock._auto_started = True
        dock.set_auto_zone_state("zone_set")


        dock.set_auto_review_score_useful(plugin._run_scores_rank_objects())



        import math as _math
        dock.set_review_conf_floor(
            int(_math.ceil(plugin._review_noise_floor() * 100 - 1e-6)))
    except (RuntimeError, AttributeError):
        pass


    plugin._remove_auto_selection_layer()
    plugin._auto_selection_layer = _make_restore_selection_layer(crs_authid)



    params = plugin._fresh_review_params()
    params["conf"] = conf
    pixel_size = gsd if gsd > 0 else 1.0
    visible = []
    vis_scores = []
    vis_ids = []
    for det_idx, (base, score, area) in enumerate(plugin._auto_objects):
        if base is None or base.isEmpty():
            continue
        if not plugin._passes_review_filters(score, area, params):
            continue
        g = plugin._refine_geom_for_review(base, params, pixel_size)
        if g is not None and not g.isEmpty():
            visible.append(g)
            vis_scores.append(score)


            vis_ids.append(plugin._object_fid_for(det_idx))


    plugin._start_build_preview_cache(pixel_size)
    try:
        hist = getattr(dock, "auto_conf_histogram", None)
        if hist is not None:
            hist.set_scores([s for (_g, s, _a) in plugin._auto_objects])
            hist.set_cutoff(conf)
    except (RuntimeError, AttributeError):
        pass


    plugin._complete_auto_finalize(visible, len(tiles), vis_scores, vis_ids)
    if plugin._auto_review is not None:



        plugin._auto_review["pixel_size"] = pixel_size





    try:
        import math as _math
        dock.seed_review_confidence(int(round(conf * 100)))
        dock.set_review_conf_floor(
            int(_math.ceil(plugin._review_noise_floor() * 100 - 1e-6)))
    except (RuntimeError, AttributeError):
        pass

    _zoom_to_tiles(plugin, tiles, crs_authid)

    try:
        dock.set_auto_status(
            "info",
            tr('Restored "{prompt}" - adjust and export below.').format(
                prompt=prompt))
    except (RuntimeError, AttributeError):
        pass

    try:
        from ...core import telemetry_session_events
        telemetry_session_events.track_history_restored(
            plugin._auto_run_id,
            len(tiles),
            len(plugin._auto_objects),
            age_days=_run_age_days(run),
        )
    except Exception:
        pass  # nosec B110

    _log(f"Run restore: review opened with {len(visible)} object(s) at {int(round(conf * 100))}%")
    return True
