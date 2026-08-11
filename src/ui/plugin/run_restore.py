"""Rebuild a past cloud run's detections from its stored masks.

The Library's History tab uses this to "Restore to map": every archived tile's
masks are decoded with the EXISTING pipeline (RLE decode -> mask_to_polygons ->
IncrementalMerger, exactly like a live run) and the standard post-run review is
re-opened on the result. Zero credits, no re-detection: everything runs from
the stored masks + tile bboxes, raster-independent.

Direct Export reuses the same decode+merge steps headless (no review) and
hands the filtered geometries to polygon_exporter.export_geometries_to_file.

Split by thread. ``decode_run_masks`` and ``export_decoded_run`` run on the
library's fetch thread: a 100-tile run is thousands of masks at 10 to 30 ms
each, which froze QGIS for minutes when it ran in the click handler. They take
plain values (never the plugin) and report through the two callbacks so the
caller can show progress and stop. ``run_merge_separate``, ``restore_run`` and
``load_exported_layer`` are the GUI-thread half.

CRITICAL invariant kept: masks map to ground by their OWN pixel shape (the
cloud model returns masks at its internal size, not the uploaded tile size);
mask_to_polygons derives the grid from mask.shape / full_shape, never from the
tile dimensions we sent.
"""
from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog

from ...core.i18n import tr
from ...core.layer_conventions import crs_measures_in_ground_metres
from ...core.qt_compat import (
    field_type_double,
    field_type_int,
    field_type_string,
)

# Whole-tile blob guard in SEPARATE (count) mode; mirrors the worker's
# _MAX_TILE_COVERAGE (auto_detection_worker.py). Kept as a local constant so
# the core layer never imports the QThread worker module.
_MAX_TILE_COVERAGE = 0.55

_DEFAULT_START_CONFIDENCE = 0.30


def _log(msg: str, level=None) -> None:
    QgsMessageLog.logMessage(
        msg, "AI Segmentation",
        level=level if level is not None else Qgis.MessageLevel.Info)


def snap_confidence(value, default: float = _DEFAULT_START_CONFIDENCE) -> float:
    """Snap a stored run threshold to the review slider's 5% steps."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v <= 0.0 or v > 1.0:
        return default
    step = int(round(v * 100 / 5.0)) * 5
    return max(5, min(95, step)) / 100.0


def _tile_bbox(tile: dict):
    """(xmin, ymin, xmax, ymax) from a history tile row, or None."""
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
    """The zone a stored run covered, as ((xmin, ymin, xmax, ymax), crs_authid).

    A run does not keep the polygon the user drew, but it keeps every tile it
    billed, and the grid was built to cover that polygon. Their union is
    therefore the same ground the run looked at, which is what "run this again"
    has to point at. Returns None when no tile carries a usable box.
    """
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
    """Normalize a stored masks payload to a [{rle, score, box}] list."""
    if isinstance(payload, dict):
        payload = payload.get("masks")
    if not isinstance(payload, list):
        return []
    return [m for m in payload if isinstance(m, dict)]


def _run_gsd(tiles: list) -> float:
    """Ground units (run CRS) per pixel of the run's tile grid.

    The widest tile bbox spans a full TILE_SIZE-pixel tile, so its width over
    TILE_SIZE recovers the grid's ground sample distance without needing the
    (unstored) per-tile pixel dimensions."""
    from ...core.tile_manager import TILE_SIZE

    widest = 0.0
    for tile in tiles:
        bb = _tile_bbox(tile)
        if bb is not None:
            widest = max(widest, bb[2] - bb[0])
    return widest / TILE_SIZE if widest > 0 else 0.0


def _run_simplify_mult(run: dict, tiles: list) -> float:
    """The staircase simplify multiplier THIS run was vectorized with.

    A run resolves the multiplier ONCE at its start and keeps it for every one
    of its tiles. A replay has to reuse that run's own value: reading a fresh
    one would change an archived run's geometry the day the value moves, and a
    replay must reproduce the run it replays, not today's settings.

    Runs archived before the value was recorded carry nothing. 0.0 is then
    returned, which makes ``tile_simplify_tolerance`` apply the shipped
    constant, and that constant is exactly what those runs ran with.
    """
    sources = [run]
    sources.extend(t for t in tiles if isinstance(t, dict))
    for src in sources:
        val = src.get("tile_simplify_mult")
        if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
            return float(val)
    return 0.0


def run_merge_separate(plugin, run: dict) -> bool:
    """The merge policy a replay folds with, resolved on the GUI thread.

    ``_default_merge_separate`` reads the served policy blob and the preset
    catalogue, so it is answered here and handed to the decode as a plain bool:
    the decode runs on a worker thread and must touch neither the plugin nor
    anything it caches.
    """
    try:
        return bool(plugin._default_merge_separate((run.get("prompt") or "").strip()))
    except (AttributeError, RuntimeError, TypeError):
        return True  # counting-safe default, same as a live run's


def decode_run_masks(run: dict, tiles: list, masks_per_tile: dict,
                     merge_separate: bool, *, on_tile=None,
                     is_cancelled=None) -> dict | None:
    """Shared steps 1-3: decode every stored mask and fold it into a fresh
    IncrementalMerger seeded exactly like a live run (same merge-policy default
    from the prompt, same seam gate formula, same per-detection refine ->
    polygonize -> repair pipeline as the worker).

    Runs on the library's fetch thread. ``on_tile(done, total)`` reports what
    is left; ``is_cancelled()`` is polled per tile AND per mask, because one
    saturated tile is 200 masks and several seconds on its own.

    Returns the decoded bundle read by ``restore_run`` and
    ``export_decoded_run``: objects (a list of (stable_id, geometry, score),
    [] when nothing decoded), crs_authid, gsd and the merge policy it folded
    with. None means the caller cancelled part way.
    """
    import numpy as np

    from ...core.cloud_detection import (
        iter_detection_masks,
        mask_cell_size,
        tile_simplify_tolerance,
    )
    from ...core.layer_conventions import repair_polygon, to_multipolygon
    from ...core.polygon_exporter import (
        IncrementalMerger,
        apply_mask_refinement,
        mask_to_polygons,
    )
    from ...core.tile_manager import OVERLAP_FRACTION, TILE_SIZE

    crs_authid = run.get("crs_authid") or (tiles[0].get("crs_authid") if tiles else None) or "EPSG:4326"
    gsd = _run_gsd(tiles)
    simplify_mult = _run_simplify_mult(run, tiles)

    # Same smart default a live run uses (discrete objects stay SEPARATE,
    # continuous features MERGE across seams), same seam-gate formula as
    # plugin._auto_seam_min_dim, evaluated with this run's grid GSD.
    merge_separate = bool(merge_separate)
    if merge_separate:
        seam_min_dim = float("inf")
    else:
        seam_min_dim = OVERLAP_FRACTION * TILE_SIZE * gsd if gsd > 0 else 0.0
    merger = IncrementalMerger(seam_min_dim=seam_min_dim)

    min_keep_area = (1.5 * gsd) ** 2 if gsd > 0 else 0.0

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
            # polygon_exporter bbox convention: (minx, maxx, miny, maxy).
            "bbox": (xmin, xmax, ymin, ymax),
            "crs": crs_authid,
        }
        tile_had_masks = False
        # Fallback dims only apply when the server stored no output size; the
        # masks were archived at the model's output resolution, which for our
        # tiles is at most the uploaded TILE_SIZE. Streamed one mask at a time
        # (same order, same values as the whole-tile list this replaced) so a
        # saturated archived tile never holds every full-tile grid at once.
        for mask, score, _box in iter_detection_masks(
                response, TILE_SIZE, TILE_SIZE, 0.0):
            if is_cancelled is not None and is_cancelled():
                return None
            if not tile_had_masks:
                tile_had_masks = True
                decoded_tiles += 1
            # Verbatim worker pipeline (_detections_to_geoms): crop the mask to
            # the object's bbox, pad 1px, fill pinholes, polygonize with the
            # crop offset against the FULL mask grid so the mapping stays
            # pixel-exact and scale comes from the mask's own shape. Like the
            # worker, the simplify tolerance keys on each mask's own grid cell
            # (a coarser returned grid de-staircases at its true step;
            # unchanged when the grids match) and on the multiplier THIS run
            # used, so the replay reproduces the run rather than today's dial.
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
                sub, expand_value=0, fill_holes=True, min_area=0)
            for geom in mask_to_polygons(
                sub, tile_transform,
                simplify_tolerance=tile_simplify_tolerance(
                    gsd, cell, simplify_mult),
                pixel_offset=(col0 - 1, row0 - 1), full_shape=(full_h, full_w),
            ):
                if geom is None or geom.isEmpty():
                    continue
                geom = to_multipolygon(repair_polygon(geom) or geom)
                if geom is None or geom.isEmpty():
                    continue
                if min_keep_area > 0.0 and geom.area() < min_keep_area:
                    continue
                merger.add(geom, float(score))

    if on_tile is not None:
        on_tile(total, total)
    # Ided triples, because the review builder keys each object's stable colour
    # on the merger fid and reads (fid, geom, score) in that order.
    merged_scored = merger.result_scored_ided()
    _log(f"Run restore: decoded {decoded_tiles} tile(s) into {len(merged_scored)} object(s)")
    return {
        "objects": merged_scored,
        "crs_authid": crs_authid,
        "gsd": gsd,
        "merge_separate": merge_separate,
    }


def _run_start_confidence(run: dict, tiles: list) -> float:
    """Start confidence = the run's stored threshold, snapped to 5% steps.

    Recent runs store the low recall floor as their threshold (the review was
    the real cutoff); a floor-level threshold keeps the default start instead,
    matching what the user actually reviewed at."""
    threshold = run.get("threshold")
    if threshold is None and tiles:
        threshold = tiles[0].get("threshold")
    snapped = snap_confidence(threshold)
    if snapped <= 0.15:
        return _DEFAULT_START_CONFIDENCE
    return snapped


def _confidence_showing_an_object(conf: float, objects: list) -> float:
    """``conf`` lowered to the highest 5% step at or below the best score, when
    the cutoff would otherwise hide every object.

    Same rule as the live review's _review_start_confidence (auto_review_params),
    which restore cannot call because that one re-derives the run's default
    instead of reading the run's own stored threshold. Without it a restored run
    can open on an empty map, and a run whose objects all share one score has no
    Confidence control on screen to lower.
    """
    import math

    scores = [s for (_g, s, _a) in objects if s is not None]
    if not scores:
        return conf
    best = max(scores)
    if best >= conf:
        return conf
    return max(0, int(math.floor(best * 100 / 5.0)) * 5) / 100.0


def _make_restore_selection_layer(crs_authid: str):
    """In-memory review layer for the restored detections (the run CRS variant
    of the plugin's _create_auto_selection_layer, which needs a raster layer;
    restore is raster-independent)."""
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
        # The SAME three fields as _create_auto_selection_layer. This layer is
        # assigned to plugin._auto_selection_layer, so the shared review code
        # writes three-attribute features into it; with only two the provider
        # refused every one of them ("expecting 2, received 3") and a restored
        # run came back empty. det_id also carries the per-object colour.
        pr.addAttributes([
            QgsField("label", field_str),
            QgsField("score", field_dbl),
            QgsField("det_id", field_type_int()),
        ])
        layer.updateFields()
        layer.setRenderer(make_review_renderer())
        try:
            # Same smoothness helpers the live layer gets (render-time simplify
            # + spatial index); optional, restore works without them.
            from .shared import _apply_fast_render
            _apply_fast_render(layer)
        except Exception:  # nosec B110
            pass
        # Private working layer: renders via its tree node but stays out of
        # the Layers panel. Flag BEFORE the add so the panel never flashes it.
        from ...core.output_store import mark_temp_layer
        mark_temp_layer(layer)
        QgsProject.instance().addMapLayer(layer, False)
        QgsProject.instance().layerTreeRoot().insertLayer(0, layer)
        return layer
    except (RuntimeError, AttributeError):
        return None


def _zoom_to_tiles(plugin, tiles: list, crs_authid: str) -> None:
    """Zoom the canvas to the run's union bbox (transformed to canvas CRS)."""
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
        union.grow(max(union.width(), union.height()) * 0.05)
        canvas.setExtent(union)
        canvas.refresh()
    except Exception:  # nosec B110 -- zoom is best-effort, never block restore
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
                       driver: str) -> dict:
    """Filter WHOLE objects at ``confidence`` and write them to ``path``.

    Runs on the library's fetch thread, next to the decode that produced
    ``decoded``: per feature the exporter repairs the geometry and measures it
    geodesically, which is seconds of frozen GUI at a few thousand objects.

    Returns {"count": polygons that reached the file, "written": the file
    exists}. The count comes from the exporter, not from the objects above the
    cutoff: a geometry no repair can save is left out, and reporting the input
    names a number the file contradicts.
    The QgsVectorLayer the exporter loads back is dropped HERE, on the thread
    that created it (a map layer must never be destroyed from another thread);
    the GUI re-opens the finished file with ``load_exported_layer``.
    """
    from qgis.core import QgsCoordinateReferenceSystem

    from ...core.polygon_exporter import export_geometries_to_file

    geoms = [g for _fid, g, s in (decoded.get("objects") or [])
             if g is not None and not g.isEmpty() and s >= confidence]
    if not geoms:
        return {"count": 0, "written": False}
    stats: dict = {}
    layer = export_geometries_to_file(
        geoms, QgsCoordinateReferenceSystem(decoded.get("crs_authid") or "EPSG:4326"),
        path, driver=driver, stats=stats)
    written = layer is not None
    del layer
    return {"count": int(stats.get("written") or 0), "written": written}


def load_exported_layer(path: str, driver: str):
    """Re-open the file a direct Export just wrote, on the GUI thread.

    The exporter already loaded it once, on the worker thread that wrote it,
    and dropped it there. A map layer belongs to the thread that made it, so
    the copy that goes into the project is made here. Returns None when the
    file cannot be read back.
    """
    import os

    from qgis.core import QgsVectorLayer

    from ...core.layer_conventions import make_committed_renderer
    from ...core.output_store import apply_fast_canvas_render

    name = os.path.splitext(os.path.basename(path))[0] or "detections"
    layer = QgsVectorLayer(path, name, "ogr")
    if not layer.isValid():
        layer = QgsVectorLayer(f"{path}|layername={name}", name, "ogr")
    if not layer.isValid():
        _log(f"Run export: file saved but could not be loaded back: {path}",
             Qgis.MessageLevel.Warning)
        return None
    if driver != "GPKG":
        # Only the GeoPackage carries the style inside the file; every other
        # driver comes back bare, so the committed look is re-applied here.
        layer.setRenderer(make_committed_renderer())
    # This layer goes straight into the project, not through add_committed_layer,
    # so it asks for the fast render itself. A layer reopened from a file starts
    # on QGIS's own simplification defaults whatever the saved style says, so
    # this is a re-apply, not a first one.
    apply_fast_canvas_render(layer)
    return layer


def restore_run(plugin, run: dict, tiles: list, decoded: dict) -> bool:
    """Rebuild a past run's detections and open the standard post-run review.

    Zero credits. Enters the EXISTING review-open path (the same
    _complete_auto_finalize tail a live run uses), so confidence, shape
    controls, display modes and Export all behave identically to a fresh run.

    Takes the bundle ``decode_run_masks`` built on the fetch thread; everything
    from here on is GUI work.

    Returns True when the review opened, False otherwise (a user-facing
    message is shown on the failure paths)."""
    if plugin is None or not tiles:
        return False

    # A live run or an open review owns the auto state; never clobber it.
    worker = getattr(plugin, "_auto_worker", None)
    if (worker is not None and worker.isRunning()) or plugin._auto_review:
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

    # --- seed the plugin's run state exactly like a fresh finalize ----------
    plugin._ensure_dock_widget()
    dock = plugin.dock_widget
    if dock is None:
        return False
    try:
        from ..ai_segmentation_dockwidget import Mode
        if dock._mode != Mode.AUTOMATIC:
            dock._on_mode_selected(Mode.AUTOMATIC)
    except (RuntimeError, AttributeError, ImportError):
        pass

    plugin._reset_auto_live_pipeline()
    plugin._auto_merger = None
    plugin._auto_worker = None
    plugin._auto_headless_run = False
    # Not a live detection: suppress the fake auto_detect_completed terminal
    # (review_opened still fires with the original run_id for correlation).
    plugin._auto_tel_stop_reason = "restored"
    plugin._auto_run_id = str(run.get("run_id") or run.get("group_key") or "")
    plugin._auto_crs_authid = crs_authid
    plugin._auto_gsd = gsd
    # A restore is not the previous live run: drop its observed mask
    # resolution, and take the run GSD as meters only when the run CRS really
    # measures meters (else 0.0 = no resolution noise floor; the prompt-aware
    # Min size floor still applies). Both feed the smart review preset.
    plugin._auto_mask_gsd = 0.0
    plugin._auto_gsd_m = 0.0
    try:
        from qgis.core import QgsCoordinateReferenceSystem
        crs = QgsCoordinateReferenceSystem(crs_authid)
        # Not mapUnits(): Pseudo-Mercator reports metres and is not one on the
        # ground, so an archived basemap run restored through it opens the
        # noise floor by 1/cos(latitude) and hides small objects the original
        # review kept.
        if crs_measures_in_ground_metres(crs):
            plugin._auto_gsd_m = gsd
    except (RuntimeError, AttributeError, TypeError):
        pass
    plugin._auto_merge_separate = merge_separate
    plugin._auto_confidence = conf
    plugin._auto_raw_count = len(merged_scored)
    plugin._auto_dense_tiles = 0
    plugin._auto_preview_geoms = []
    plugin._auto_clip_polygon = None
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

    # Canonical whole objects (geom, score, area) via the existing builder.
    plugin._auto_objects = plugin._build_auto_objects(merged_scored)
    if not plugin._auto_objects:
        return False
    # The stored threshold can sit above every score this run kept, which opens
    # the review on nothing to see and an Export it cannot enable. Lower it to a
    # step that shows at least one object, exactly as a live run does. Every
    # reader below (the dock seed, the filter params, the histogram) takes the
    # value from here.
    conf = _confidence_showing_an_object(conf, plugin._auto_objects)
    plugin._auto_confidence = conf

    # Dock state: prompt box, the confidence seed the review slider reads, and
    # land on the prompt step (the review panel lives there).
    try:
        dock.set_prompt_text(prompt)
        spin = dock.auto_confidence_spin
        spin.blockSignals(True)
        spin.setValue(conf)
        spin.blockSignals(False)
        dock._auto_started = True
        dock.set_auto_zone_state("zone_set")
        # A restored run answers the same question as a fresh one: scores that
        # cannot order the objects give the review nothing to filter on.
        dock.set_auto_review_score_useful(plugin._run_scores_rank_objects())
    except (RuntimeError, AttributeError):
        pass

    # Fresh selection layer in the RUN's CRS (raster-independent).
    plugin._remove_auto_selection_layer()
    plugin._auto_selection_layer = _make_restore_selection_layer(crs_authid)

    # Visible set at the starting cutoff, with parallel scores for the heatmap
    # (the existing filter + shape-refine helpers, neutral fresh params).
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

    # Confidence-drag preview cache + histogram, as the live finalize does.
    plugin._start_build_preview_cache(pixel_size)
    try:
        hist = getattr(dock, "auto_conf_histogram", None)
        if hist is not None:
            hist.set_scores([s for (_g, s, _a) in plugin._auto_objects])
            hist.set_cutoff(conf)
    except (RuntimeError, AttributeError):
        pass

    # Enter the EXISTING review-open path (no parallel review).
    plugin._complete_auto_finalize(visible, len(tiles), vis_scores)
    if plugin._auto_review is not None:
        # The finalize derives pixel_size from the ACTIVE raster; a restore is
        # raster-independent, so pin the run's own grid scale for the shape
        # refine px->ground conversions.
        plugin._auto_review["pixel_size"] = pixel_size

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
