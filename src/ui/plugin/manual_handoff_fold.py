








from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog


class ManualHandoffFoldMixin:





    def _collect_manual_refine_into_review(self) -> None:





        if self._pending_refine_import:
            self._pending_refine_import = False
            self._handoff_crs_pair = None
            self._teardown_manual_session()
            return
        review = self._auto_review
        if review is None:

            self._handoff_crs_pair = None
        else:


            try:
                self._on_save_polygon()
            except Exception as e:  # noqa: BLE001



                QgsMessageLog.logMessage(
                    f"Refine handoff: save fold error: {e}",
                    "AI Segmentation", level=Qgis.MessageLevel.Warning)





            if self._is_refining_saved_object:
                self._close_active_edit_to_pending()
            entries = []
            for pg in self.saved_polygons:
                g = self._entry_geom(pg)
                if g is not None and not g.isEmpty():
                    entries.append((g, pg.get("det_id"), pg.get("score"),
                                    bool(pg.get("manual_touched"))))



            entries = self._handoff_entries_to_run_crs(entries)


            geoms, ids, scores = self._handoff_entry_identities(
                [(g, i, s) for g, i, s, _t in entries])
            review["geoms"] = geoms
            review["scores"] = scores
            review["ids"] = ids



            review["stamp"] = None




            self._fold_manual_refine_into_objects(entries, geoms, ids, scores)

            self._disarm_after_handoff()
        self._teardown_manual_session()

    def _fold_manual_refine_into_objects(self, entries, geoms, ids, scores) -> None:











        objects = getattr(self, "_auto_objects", None)
        if objects is None:
            return
        manual_removed_before = set(getattr(self, "_auto_manual_removed", None) or ())
        pre_len = len(objects)


        added_ids = self._register_manual_only_review_objects(geoms, ids, scores)
        appended = len(objects) - pre_len

        fids = list(getattr(self, "_auto_object_fids", None) or [])
        by_id = {fid: idx for idx, fid in enumerate(fids)}
        measurer = self._make_auto_area_measurer()
        from ...core.layer_conventions import repair_polygon, to_multipolygon
        manual_ids = self._auto_manual_object_ids
        restored: list[tuple[int, object]] = []
        exempted: list[int] = []
        for g, det_id, score, touched in entries:
            if not touched or not isinstance(det_id, int):
                continue
            index = by_id.get(det_id)
            if index is None or index >= pre_len:
                continue
            repaired = to_multipolygon(repair_polygon(g) or g)
            if repaired is None or repaired.isEmpty():
                continue
            restored.append((index, objects[index]))
            carried = objects[index][1] if score is None else float(score)
            objects[index] = (
                repaired, float(carried), self._object_area_m2(repaired, measurer))
            if det_id not in manual_ids:
                manual_ids.add(det_id)
                exempted.append(det_id)
        for det_id in added_ids:
            if det_id not in exempted:
                exempted.append(det_id)






        new_removed = manual_removed_before | self._removed_canonical_objects(geoms)
        self._auto_manual_removed = new_removed
        from ...core.shape_edits import KIND_REFINE, ShapeEdit
        edit = ShapeEdit(
            kind=KIND_REFINE,
            restored=tuple(restored),
            appended=appended,
            unremoved=(),
            exempted=tuple(exempted),
        )


        self._record_fold_edit(
            edit, fids=tuple(exempted),
            manual_removed_before=(manual_removed_before
                                   if manual_removed_before != new_removed
                                   else None))

        self._shape_hit_geoms = {}
        self._reset_review_refine_cache()
        try:
            pixel_size = (self._auto_review or {}).get("pixel_size", 1.0)
            self._start_build_preview_cache(pixel_size)
        except (RuntimeError, AttributeError):
            pass

    def _register_manual_only_review_objects(self, geoms, ids, scores) -> list:






        added: list[int] = []
        if not isinstance(ids, list):
            return added
        objects = getattr(self, "_auto_objects", None)
        if objects is None:
            return added
        known = {self._object_fid_for(index) for index in range(len(objects))}
        fids = list(getattr(self, "_auto_object_fids", None) or [])
        measurer = self._make_auto_area_measurer()
        manual_ids = self._auto_manual_object_ids
        imported = getattr(self, "_handoff_imported_det_ids", None) or set()
        for index, geom in enumerate(geoms or []):
            det_id = ids[index] if index < len(ids) else None
            if not isinstance(det_id, int) or geom is None or geom.isEmpty():
                continue
            if det_id in known:





                if det_id not in imported:
                    QgsMessageLog.logMessage(
                        f"Refine handoff: det_id {det_id} is already taken by "
                        f"another object; the added shape got no row of its own.",
                        "AI Segmentation", level=Qgis.MessageLevel.Warning)
                continue
            score = scores[index] if isinstance(scores, list) and index < len(scores) else 1.0
            try:
                score = float(score) if score is not None else 1.0
            except (TypeError, ValueError):
                score = 1.0
            objects.append((geom, score, self._object_area_m2(geom, measurer)))
            fids.append(det_id)
            known.add(det_id)
            if det_id not in manual_ids:
                manual_ids.add(det_id)
            added.append(det_id)
        self._auto_object_fids = fids
        return added

    def _removed_canonical_objects(self, kept_geoms: list) -> set:













        objects = getattr(self, "_auto_objects", None) or []
        session_ids = getattr(self, "_handoff_imported_det_ids", None)
        if not objects or not session_ids:
            return set()
        from qgis.core import QgsFeature, QgsGeometry, QgsSpatialIndex

        from ...core.server_dials import dial_in_range


        coverage_floor = dial_in_range(
            "tuning.manual.handoff_kept_coverage", 0.3, 0.05, 0.95)

        index = QgsSpatialIndex()
        kept = []
        for g in kept_geoms or []:
            if g is None or g.isEmpty():
                continue
            feat = QgsFeature(len(kept))
            feat.setGeometry(QgsGeometry.fromRect(g.boundingBox()))
            index.addFeature(feat)
            kept.append(g)
        removed = set()
        for det_idx, (base, _score, _area) in enumerate(objects):
            if base is None or base.isEmpty():
                continue
            if self._object_fid_for(det_idx) not in session_ids:
                continue
            area = base.area()
            if area <= 0:
                continue
            still_present = False
            for j in index.intersects(base.boundingBox()):
                try:
                    inter = base.intersection(kept[j])
                except Exception:  # nosec B112
                    continue
                if inter is not None and not inter.isEmpty() and inter.area() / area >= coverage_floor:
                    still_present = True
                    break
            if not still_present:
                removed.add(det_idx)
        return removed
