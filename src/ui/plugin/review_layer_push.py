"""Writing the review's visible set onto the live selection layer.

Three functions taking the plugin controller: the entry point, the incremental
provider diff, and the full clear-and-re-add fallback. They came out of
``auto_review.py``, which was well over its size band, and they are one
concern: nothing here decides WHAT is visible, only how that set reaches the
layer with the fewest provider writes.
"""
from __future__ import annotations

from qgis.core import Qgis, QgsFeature, QgsMessageLog

from .shared import (
    _add_features_fast,
    _add_features_with_ids,
    _clear_all_features,
    _notify_provider_write,
)


def push_review_geoms(
    plugin, geoms: list, repair: bool = True, scores: list | None = None,
    ids: list | None = None, stamp: tuple | None = None,
    partial: bool = False, update_extents: bool | None = None,
) -> None:
    """Write geoms onto the live review selection layer and update the
    review count. Shared by the accurate refresh and the fast confidence-
    drag preview.

    With ``stamp`` + ``ids`` the push is INCREMENTAL: the provider gets only
    the delta (add / delete / changed geometry) against _review_fid_map,
    keyed on det_id. ``stamp`` names the geometry provenance (preview-cache
    build vs refine-cache shape key), so identical det_id + stamp means the
    on-layer geometry is already current and nothing is written. This is
    what makes a confidence-drag tick or a filter-only reslice O(delta)
    instead of a full truncate + re-add of every feature. Without a stamp
    (unknown provenance: protected-dissolve output, handoff harvest) the
    push falls back to the full truncate + re-add.

    ``repair=False`` skips the per-geom makeValid for the fast path (raw
    geoms are usually valid; the accurate pass repairs at cache-fill).
    ``scores`` feeds the review heatmap (1.0 fallback keeps a mismatched
    case green/trusted); ``ids`` is the canonical det_id the Random display
    mode hues on (NULL fallback lets the renderer hue on $id).

    ``partial=True`` is the PROGRESSIVE mode used mid-reslice: geoms not in
    this batch are LEFT on the layer (their old shape) instead of deleted,
    and the header/extents stay untouched, so a long shape-refine visibly
    sweeps the map instead of freezing on the old state until the end. It
    only ever applies through the incremental diff (never a truncate)."""
    layer = plugin._auto_selection_layer
    if layer is None:
        return
    try:
        if not layer.isValid():
            return
        # The Manual method puts THIS layer into a QGIS edit session. Every
        # write below goes through the PROVIDER, which the edit buffer does
        # not see, and the buffer is what commit and rollback act on: the
        # write is lost, or it moves a vertex out from under the user's
        # cursor. Nothing is owed by skipping. The way out of a Manual
        # session commits first and reslices after, so the layer is
        # republished on committed geometry a moment later.
        if layer.isEditable():
            if not getattr(plugin, "_review_push_editable_logged", False):
                plugin._review_push_editable_logged = True
                QgsMessageLog.logMessage(
                    "Review push skipped: the review layer is in a QGIS "
                    "edit session; it is republished when the session ends.",
                    "AI Segmentation", level=Qgis.MessageLevel.Info)
            return
        count = -1
        if stamp is not None and ids is not None and len(ids) == len(geoms):
            count = diff_push_review_geoms(
                plugin, layer, geoms, scores, ids, stamp, keep_missing=partial)
        if count < 0:
            if partial:
                return  # progressive apply needs the diff path; skip quietly
            count = full_push_review_geoms(
                plugin, layer, geoms, repair, scores, ids, stamp)
        # updateExtents rescans every feature (O(N) on a memory provider):
        # only the accurate release pass needs it (zoom-to-layer); the 40ms
        # drag preview renders by viewport via the spatial index, so the
        # stale cached extent is invisible mid-drag. Defaults to `repair`
        # for callers that do not say (the historical coupling), but the
        # review-entry push decouples them: its geoms are already repaired
        # at cache-fill, so repair=False, yet it still needs fresh extents.
        do_extents = repair if update_extents is None else update_extents
        if do_extents and not partial:
            layer.updateExtents()
        # The rows above went in through the PROVIDER, which emits no signal
        # QGIS's snapping index or vertex tool listen to. Without this the
        # hand-edit tools keep pointing at the features from BEFORE this
        # filter change, and clicking a vertex silently does nothing.
        _notify_provider_write(layer)
        # triggerRepaint alone schedules the canvas update; the extra
        # mapCanvas().refresh() forced a full re-render of EVERY layer on
        # each (debounced) slider tick, which is what made review sliders lag.
        # Through the live run's pacer, not straight at the layer: a bare
        # triggerRepaint makes the canvas abandon the frame it is drawing and
        # start again, and a slider drag pushes faster than a dense set can
        # be drawn, so no frame ever finished and the map read as frozen for
        # the whole drag. The pacer marks the layer dirty instead and repaints
        # when the canvas says it finished.
        plugin._repaint_live_layer(layer)
        if partial:
            return  # header + extents settle on the final complete push
        # Keep the review count honest with the size filter (it can hide
        # detections): show how many are actually on the layer now.
        plugin._update_review_header(count)
    except Exception as e:  # noqa: BLE001
        # Review must never crash the UI, so the guard stays broad; but a
        # swallowed geometry error was fully silent. Log once per rebuild
        # generation (reset in _start_auto_reslice) so a real bug surfaces
        # without spamming the log on every confidence-drag tick.
        if not plugin._review_push_err_logged:
            plugin._review_push_err_logged = True
            QgsMessageLog.logMessage(
                f"Auto review: geometry rebuild error: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)


def diff_push_review_geoms(
    plugin, layer, geoms: list, scores: list | None, ids: list,
    stamp: tuple, keep_missing: bool = False,
) -> int:
    """Incremental provider update against _review_fid_map. Returns the
    visible count written, or -1 when a diff is not possible (unknown
    provider contents, a missing/duplicate det_id) so the caller falls back
    to the full push. Geometry writes coerce to MultiPolygon lazily (only
    the delta pays the copy). ``keep_missing`` (progressive mid-reslice
    batches) leaves entries absent from this batch untouched on the layer
    instead of deleting them; the final complete push reconciles."""
    from .auto_results import _diff_live_fid_map
    pr = layer.dataProvider()
    old_map = plugin._review_fid_map
    if not old_map and pr.featureCount() > 0:
        return -1  # cannot trust a diff against unknown provider contents
    current = []
    geom_by_id = {}
    score_by_id = {}
    for i, geom in enumerate(geoms):
        if geom is None or geom.isEmpty():
            continue
        det_id = ids[i]
        if det_id is None or det_id in geom_by_id:
            return -1  # unknown or ambiguous identity: full push instead
        s = (float(scores[i])
             if scores is not None and i < len(scores) and scores[i] is not None
             else 1.0)
        current.append((det_id, stamp, True, s))
        geom_by_id[det_id] = geom
        score_by_id[det_id] = s
    adds, geom_changes, attr_changes, deletes, new_map = _diff_live_fid_map(
        old_map, current)
    if keep_missing:
        # Progressive batch: entries not processed yet keep their old
        # shape on the layer and their old mapping.
        deletes = []
        for det_id, rec in old_map.items():
            if det_id not in new_map and det_id not in geom_by_id:
                new_map[det_id] = rec
    from ...core.layer_conventions import to_multipolygon

    def _mp(g):
        return to_multipolygon(g) or g

    def _diff_failed() -> int:
        # A half-applied provider write leaves the layer and the fid map out
        # of step, and a map that no longer describes the layer makes every
        # later diff wrong. Drop it so the caller rebuilds with a full push.
        plugin._review_fid_map = {}
        return -1

    if deletes:
        if not pr.deleteFeatures(deletes):
            return _diff_failed()
    if adds:
        fields = layer.fields()
        add_feats = []
        for det_id in adds:
            feat = QgsFeature(fields)
            feat.setGeometry(_mp(geom_by_id[det_id]))
            feat.setAttributes(["", score_by_id[det_id], int(det_id)])
            add_feats.append((det_id, feat))
        # The assigned provider fids (needed so a later tick can target
        # these objects) come back on the RETURNED copies; addFeatures
        # never mutates its inputs.
        ok, added = _add_features_with_ids(pr, [f for _, f in add_feats])
        if not ok or len(added) != len(add_feats):
            return _diff_failed()
        for (det_id, _f), out in zip(add_feats, added):
            pfid = out.id()
            if pfid is None or pfid < 0:
                return _diff_failed()
            new_map[det_id] = (pfid, stamp, True, score_by_id[det_id])
    if geom_changes:
        if not pr.changeGeometryValues(
                {pf: _mp(geom_by_id[d]) for pf, d in geom_changes.items()}):
            return _diff_failed()
    if attr_changes:
        score_idx = layer.fields().indexOf("score")
        if score_idx >= 0:
            if not pr.changeAttributeValues(
                    {pf: {score_idx: score_by_id[d]}
                     for pf, d in attr_changes.items()}):
                return _diff_failed()
    plugin._review_fid_map = new_map
    return len(current)


def full_push_review_geoms(
    plugin, layer, geoms: list, repair: bool, scores: list | None,
    ids: list | None, stamp: tuple | None,
) -> int:
    """Full clear + re-add of the visible set (the pre-diff behaviour).
    Rebuilds _review_fid_map when identity (ids + stamp) is available so the
    NEXT push can diff; otherwise clears it (later pushes stay full until a
    stamped one bootstraps)."""
    from ...core.layer_conventions import repair_polygon, to_multipolygon
    pr = layer.dataProvider()
    _clear_all_features(pr)
    plugin._review_fid_map = {}
    with_identity = stamp is not None and ids is not None and len(ids) == len(geoms)
    features_to_add = []
    # Hoisted: fields() hands back a fresh copy of the field set on every
    # call, and this loop runs once per object of the whole visible set.
    fields = layer.fields()
    for i, geom in enumerate(geoms):
        if geom is None or geom.isEmpty():
            continue
        if repair:
            geom = to_multipolygon(repair_polygon(geom) or geom)
        else:
            geom = to_multipolygon(geom) or geom
        if geom is None or geom.isEmpty():
            continue
        feat = QgsFeature(fields)
        feat.setGeometry(geom)
        score = float(scores[i]) if scores is not None and i < len(scores) and scores[i] is not None else 1.0
        det_id = ids[i] if ids is not None and i < len(ids) else None
        feat.setAttributes(
            ["", score, int(det_id) if det_id is not None else None])
        features_to_add.append((det_id, score, feat))
    if features_to_add:
        if with_identity:
            # The provider fids that bootstrap the next incremental push
            # come back on the RETURNED copies; addFeatures never mutates
            # its inputs.
            ok, added = _add_features_with_ids(
                pr, [f for _, _, f in features_to_add])
            fid_map = {}
            complete = ok and len(added) == len(features_to_add)
            if complete:
                for (det_id, score, _f), out in zip(features_to_add, added):
                    pfid = out.id()
                    if (det_id is None or pfid is None or pfid < 0 or det_id in fid_map):
                        complete = False
                        break
                    fid_map[det_id] = (pfid, stamp, True, score)
            if complete:
                plugin._review_fid_map = fid_map
        else:
            _add_features_fast(pr, [f for _, _, f in features_to_add])
    return len(features_to_add)
