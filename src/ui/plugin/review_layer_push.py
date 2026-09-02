







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

























    layer = plugin._auto_selection_layer
    if layer is None:
        return
    try:
        if not layer.isValid():
            return







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
                return
            count = full_push_review_geoms(
                plugin, layer, geoms, repair, scores, ids, stamp)







        do_extents = repair if update_extents is None else update_extents
        if do_extents and not partial:
            layer.updateExtents()




        _notify_provider_write(layer)









        plugin._repaint_live_layer(layer)
        if partial:
            return


        plugin._update_review_header(count)
    except Exception as e:  # noqa: BLE001




        if not plugin._review_push_err_logged:
            plugin._review_push_err_logged = True
            QgsMessageLog.logMessage(
                f"Auto review: geometry rebuild error: {e}",
                "AI Segmentation", level=Qgis.MessageLevel.Warning)


def diff_push_review_geoms(
    plugin, layer, geoms: list, scores: list | None, ids: list,
    stamp: tuple, keep_missing: bool = False,
) -> int:







    from .auto_results import _diff_live_fid_map
    pr = layer.dataProvider()
    old_map = plugin._review_fid_map
    if not old_map and pr.featureCount() > 0:
        return -1
    current = []
    geom_by_id = {}
    score_by_id = {}
    for i, geom in enumerate(geoms):
        if geom is None or geom.isEmpty():
            continue
        det_id = ids[i]
        if det_id is None or det_id in geom_by_id:
            return -1
        s = (float(scores[i])
             if scores is not None and i < len(scores) and scores[i] is not None
             else 1.0)
        current.append((det_id, stamp, True, s))
        geom_by_id[det_id] = geom
        score_by_id[det_id] = s
    adds, geom_changes, attr_changes, deletes, new_map = _diff_live_fid_map(
        old_map, current)
    if keep_missing:


        deletes = []
        for det_id, rec in old_map.items():
            if det_id not in new_map and det_id not in geom_by_id:
                new_map[det_id] = rec
    from ...core.layer_conventions import to_multipolygon

    def _mp(g):
        return to_multipolygon(g) or g

    def _diff_failed() -> int:



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




    from ...core.layer_conventions import repair_polygon, to_multipolygon
    pr = layer.dataProvider()
    _clear_all_features(pr)
    plugin._review_fid_map = {}
    with_identity = stamp is not None and ids is not None and len(ids) == len(geoms)
    features_to_add = []


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
