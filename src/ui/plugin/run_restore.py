
























from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr
from ...core.interaction_dials import (
    restore_align_budget_s,
    restore_align_max_objects,
    restore_confidence_floor,
)
from ...core.layer_conventions import crs_measures_in_ground_metres
from ...core.qt_compat import (
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



_RESTORE_CONFIDENCE_FLOOR = 0.15





_project_export_context: dict = {}





_RESTORE_ALIGN_MAX_OBJECTS = 400




_RESTORE_ALIGN_BUDGET_S = 2.0


def _log(msg: str, level=None) -> None:
    QgsMessageLog.logMessage(
        msg, "AI Segmentation",
        level=level if level is not None else Qgis.MessageLevel.Info)


def snap_confidence(value, default: float = _DEFAULT_START_CONFIDENCE) -> float:

    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v <= 0.0 or v > 1.0:
        return default
    step = int(round(v * 100 / 5.0)) * 5
    return max(5, min(95, step)) / 100.0


def _tile_bbox(tile: dict):

    bb = tile.get("tile_bbox_native") or tile.get("bbox_native")
    try:
        if isinstance(bb, dict):
            vals = (float(bb["xmin"]), float(bb["ymin"]),
                    float(bb["xmax"]), float(bb["ymax"]))
        elif isinstance(bb, (list, tuple)) and len(bb) >= 4:
            vals = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
        else:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return None
    return vals


def zone_extent_from_tiles(tiles: list) -> tuple[tuple, str] | None:









    boxes = [b for b in (_tile_bbox(t) for t in tiles if isinstance(t, dict)) if b]
    if not boxes:
        return None
    authid = ""
    for tile in tiles:
        if isinstance(tile, dict) and tile.get("crs_authid"):
            authid = str(tile["crs_authid"])
            break
    if not authid:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)), authid


def _masks_list(payload) -> list:

    if isinstance(payload, dict):
        payload = payload.get("masks")
    if not isinstance(payload, list):
        return []
    return [m for m in payload if isinstance(m, dict)]


def _run_gsd(tiles: list) -> float:





    from ...core.tile_manager import TILE_SIZE

    widest = 0.0
    for tile in tiles:
        bb = _tile_bbox(tile)
        if bb is not None:
            widest = max(widest, bb[2] - bb[0])
    return widest / TILE_SIZE if widest > 0 else 0.0


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










    capture_project_export_context()
    try:
        return bool(plugin._default_merge_separate((run.get("prompt") or "").strip()))
    except (AttributeError, RuntimeError, TypeError):
        return True


def decode_run_masks(run: dict, tiles: list, masks_per_tile: dict,
                     merge_separate: bool, *, on_tile=None,
                     is_cancelled=None) -> dict | None:














    import math

    import numpy as np
    from qgis.core import QgsCoordinateReferenceSystem

    from ...core import detection_policy
    from ...core.cloud_detection import (
        iter_detection_masks,
        mask_cell_size,
        pinhole_fill_limit_px,
        tile_simplify_tolerance,
    )
    from ...core.layer_conventions import repair_polygon, to_multipolygon
    from ...core.polygon_exporter import (
        IncrementalMerger,
        apply_mask_refinement,
        drop_covered_objects,
        mask_to_polygons,
    )
    from ...core.tile_manager import OVERLAP_FRACTION, TILE_SIZE






    from ...workers.auto_detection_worker import _MAX_TILE_COVERAGE, _MIN_KEEP_PX

    crs_authid = run.get("crs_authid") or (tiles[0].get("crs_authid") if tiles else None) or "EPSG:4326"
    gsd = _run_gsd(tiles)
    simplify_mult = _run_simplify_mult(run, tiles)
    pinhole_m = _run_pinhole_m(run, tiles)
    prompt = (run.get("prompt") or "").strip()




    ground_kx, ground_ky = _run_ground_unit_metres(
        QgsCoordinateReferenceSystem(crs_authid), tiles)
    area_scale = ground_kx * ground_ky
    if not math.isfinite(area_scale) or area_scale <= 0:
        area_scale = 1.0
    length_scale = math.sqrt(area_scale)







    merge_separate = bool(merge_separate)
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
            and detection_policy.restore_partitions_for(
                prompt, exemplar_only=not prompt)),
        **detection_policy.merge_scalar_kwargs(IncrementalMerger, merge_scalars),
    )







    min_keep_px = detection_policy.min_keep_px(_MIN_KEEP_PX)
    min_keep_area = (
        max((min_keep_px * gsd) ** 2,
            detection_policy.min_keep_floor_m2(0.0) / area_scale)
        if gsd > 0 else 0.0
    )






    zone = zone_geometry_from_run(run, crs_authid)
    zone_engine = prepare_zone_engine(zone)
    dropped_outside = 0

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
        tile_had_masks = False





        for mask, score, _box in iter_detection_masks(
                response, TILE_SIZE, TILE_SIZE, 0.0):
            if is_cancelled is not None and is_cancelled():
                return None
            if not tile_had_masks:
                tile_had_masks = True
                decoded_tiles += 1








            full_h, full_w = mask.shape
            cell = mask_cell_size(xmax - xmin, ymax - ymin, full_w, full_h)
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue
            if merge_separate and ys.size > _MAX_TILE_COVERAGE * float(full_h * full_w):
                continue
            row0, col0 = int(ys.min()), int(xs.min())
            sub = mask[row0:int(ys.max()) + 1, col0:int(xs.max()) + 1]
            sub = np.pad(sub, 1, constant_values=False)




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





                clipped = clip_geometry_to_zone(geom, zone, zone_engine)
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
                merger.add(geom, float(score))

    if on_tile is not None:
        on_tile(total, total)



    merger.restore_absorbed_partitions()


    merged_scored = merger.result_scored_ided()




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
    if snapped <= restore_confidence_floor(_RESTORE_CONFIDENCE_FLOOR):
        return default
    return snapped


def _restore_default_confidence(run: dict) -> float:


    try:
        from ...core.review_presets import review_start_confidence_default

        prompt = (run.get("prompt") or "").strip()
        return float(review_start_confidence_default(prompt, not prompt))
    except Exception:  # noqa: BLE001  # nosec B110
        return _DEFAULT_START_CONFIDENCE


def _confidence_showing_an_object(conf: float, objects: list) -> float:









    import math

    scores = [s for (_g, s, _a) in objects if s is not None]
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

    from ...core.polygon_exporter import export_geometries_to_file

    kept = [(fid, g, s) for fid, g, s in (decoded.get("objects") or [])
            if g is not None and not g.isEmpty() and s >= confidence]
    if not kept:
        return {"count": 0, "written": False}
    geoms = [g for _fid, g, _s in kept]



    det_ids = [fid for fid, _g, _s in kept]
    scores = [s for _fid, _g, s in kept]
    context = (project_context if isinstance(project_context, dict)
               else _project_export_context)
    stats: dict = {}
    layer = export_geometries_to_file(
        geoms, QgsCoordinateReferenceSystem(decoded.get("crs_authid") or "EPSG:4326"),
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


def restore_run(plugin, run: dict, tiles: list, decoded: dict) -> bool:











    if plugin is None or not tiles:
        return False


    worker = getattr(plugin, "_auto_worker", None)
    if ((worker is not None and worker.isRunning())
            or getattr(plugin, "_auto_review", None) is not None
            or getattr(plugin, "_auto_finalize_state", None) is not None
            or getattr(plugin, "_auto_start_in_progress", False)):
        try:
            plugin.iface.messageBar().pushWarning(
                "AI Segmentation",
                tr("Finish or exit the current run before restoring a past one."))
        except (RuntimeError, AttributeError):
            pass
        return False

    merged_scored = (decoded or {}).get("objects") or []
    crs_authid = (decoded or {}).get("crs_authid") or "EPSG:4326"
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
    plugin._auto_gsd_m = 0.0
    try:
        from qgis.core import QgsCoordinateReferenceSystem
        crs = QgsCoordinateReferenceSystem(crs_authid)




        if crs_measures_in_ground_metres(crs):
            plugin._auto_gsd_m = gsd
    except (RuntimeError, AttributeError, TypeError):
        pass
    plugin._auto_merge_separate = merge_separate
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
    plugin._auto_run_ctx = {
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
        dock.set_prompt_text(prompt)
        spin = dock.auto_confidence_spin
        spin.blockSignals(True)
        spin.setValue(conf)
        spin.blockSignals(False)
        dock._auto_started = True
        dock.set_auto_zone_state("zone_set")


        dock.set_auto_review_score_useful(plugin._run_scores_rank_objects())



        import math as _math
        dock.set_review_conf_floor(
            int(_math.ceil(plugin._review_noise_floor() * 100)))
    except (RuntimeError, AttributeError):
        pass


    plugin._remove_auto_selection_layer()
    plugin._auto_selection_layer = _make_restore_selection_layer(crs_authid)



    params = plugin._fresh_review_params()
    params["conf"] = conf
    pixel_size = gsd if gsd > 0 else 1.0
    visible = []
    vis_scores = []
    for base, score, area in plugin._auto_objects:
        if base is None or base.isEmpty():
            continue
        if not plugin._passes_review_filters(score, area, params):
            continue
        g = plugin._refine_geom_for_review(base, params, pixel_size)
        if g is not None and not g.isEmpty():
            visible.append(g)
            vis_scores.append(score)


    plugin._start_build_preview_cache(pixel_size)
    try:
        hist = getattr(dock, "auto_conf_histogram", None)
        if hist is not None:
            hist.set_scores([s for (_g, s, _a) in plugin._auto_objects])
            hist.set_cutoff(conf)
    except (RuntimeError, AttributeError):
        pass


    plugin._complete_auto_finalize(visible, len(tiles), vis_scores)
    if plugin._auto_review is not None:



        plugin._auto_review["pixel_size"] = pixel_size





    try:
        import math as _math
        dock.seed_review_confidence(int(round(conf * 100)))
        dock.set_review_conf_floor(
            int(_math.ceil(plugin._review_noise_floor() * 100)))
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
