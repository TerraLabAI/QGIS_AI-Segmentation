from __future__ import annotations

import logging

from qgis.core import (
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsSpatialIndex,
)

from . import transport_dials as _td
from .layer_conventions import repair_polygon

logger = logging.getLogger(__name__)




_COMPACT_MIN_LIVE = 64




_ABSORBED_POOL_MULT = 8

_geos_failure = {"logged": False}


def _ios_and_span(g1: QgsGeometry, g2: QgsGeometry,
                  a1: float | None = None,
                  a2: float | None = None) -> tuple[float, float]:












    if g1 is None or g2 is None or g1.isEmpty() or g2.isEmpty():
        return 0.0, 0.0
    try:
        inter = g1.intersection(g2)
        if inter is None or inter.isEmpty():
            return 0.0, 0.0
        inter_area = inter.area()
        if inter_area <= 0.0:
            return 0.0, 0.0
        if a1 is None:
            a1 = g1.area()
        if a2 is None:
            a2 = g2.area()
        min_area = min(a2, a1)
        ios = inter_area / min_area if min_area > 0.0 else 0.0
        bb = inter.boundingBox()
        span = max(bb.height(), bb.width())
        return ios, span
    except Exception as exc:



        if not _geos_failure["logged"]:
            _geos_failure["logged"] = True
            logger.warning("IncrementalMerger: overlap test failed: %s", exc)
        return 0.0, 0.0


class IncrementalMerger:



































    def __init__(
        self,
        merge_ios: float = 0.15,
        dedup_ios: float = 0.5,
        seam_min_dim: float = 0.0,
        dup_ios_floor: float = 0.3,
        dup_centroid_frac: float = 0.35,
        seam_span_ios: float = 0.03,
        select_duplicates: bool = False,
        gsd: float = 0.0,
        seam_span_tol: float = 0.85,
        jitter_area_frac: float = 0.02,
        jitter_erode_px: float = 1.0,
        score_floor_frac: float = 0.5,
        restore_partitions: bool = False,
        part_inside: float = 0.90,
        part_max_frac: float = 0.70,
        part_sibling_ios: float = 0.20,
        part_cover_frac: float = 0.60,
        part_min_children: int = 2,
    ):
        self._merge_ios = merge_ios
        self._dedup_ios = dedup_ios
        self._seam_min_dim = seam_min_dim




        self._gsd = float(gsd)



        self._jitter_erode_px = float(jitter_erode_px)












        self._select_duplicates = select_duplicates












        self._restore_partitions = bool(restore_partitions)

        _geos_failure["logged"] = False
        self._absorbed: dict[int, list] | None = {} if restore_partitions else None
        self._part_inside = float(part_inside)
        self._part_max_frac = float(part_max_frac)
        self._part_sibling_ios = float(part_sibling_ios)
        self._part_cover_frac = float(part_cover_frac)
        self._part_min_children = int(part_min_children)












        self._seam_span_ios = seam_span_ios






        self._dup_ios_floor = dup_ios_floor
        self._dup_centroid_frac = dup_centroid_frac




        self._seam_span_tol = seam_span_tol
        self._jitter_area_frac = jitter_area_frac
        self._score_floor_frac = score_floor_frac
        self._index = QgsSpatialIndex()






        self._index_entries = 0
        self._keepers: dict[int, QgsGeometry | None] = {}






        self._scores: dict[int, float] = {}





        self._areas: dict[int, float] = {}






        self._centroids: dict[int, QgsPointXY] = {}
        self._next_id = 0










        self._live_ids: dict[int, None] = {}




        self.restored_fids: set[int] = set()







        self._dirty: dict[int, None] = {}
        self._gone: dict[int, None] = {}

    def drain_changes(self) -> tuple[list[int], list[int]]:







        changed = list(self._dirty)
        removed = list(self._gone)
        self._dirty = {}
        self._gone = {}
        return changed, removed

    def keeper(self, fid: int) -> tuple[QgsGeometry | None, float]:

        geom = self._keepers.get(fid)
        if geom is None:
            return None, 0.0
        return geom, self._scores.get(fid, 0.0)

    def mark_changed(self, fids) -> None:






        for fid in fids:
            if fid in self._live_ids:
                self._dirty[fid] = None

    def _retire_keeper(self, fid: int) -> None:






        self._keepers[fid] = None
        self._live_ids.pop(fid, None)
        self._areas.pop(fid, None)
        self._centroids.pop(fid, None)
        self._dirty.pop(fid, None)
        self._gone[fid] = None

    def _is_seam_eligible(self, geom: QgsGeometry) -> bool:






        if self._seam_min_dim <= 0.0:
            return True
        bb = geom.boundingBox()
        return max(bb.width(), bb.height()) >= self._seam_min_dim

    def add(self, geom: QgsGeometry, score: float = 0.0) -> None:
        if geom is None or geom.isEmpty():
            return





        geom = repair_polygon(geom)
        if geom is None or geom.isEmpty():
            return






        cand_bbox = geom.boundingBox()
        cand_area = geom.area()
        cand_seam = self._is_seam_eligible(geom)








        seam_span_armed = (
            0.0 < self._seam_min_dim < float("inf") and not self._select_duplicates
        )
        matches = []



        cand_centroid = None









        for fid in dict.fromkeys(self._index.intersects(cand_bbox)):
            keeper = self._keepers.get(fid)
            if keeper is None:
                continue
            both_large = cand_seam and self._is_seam_eligible(keeper)
            threshold = self._merge_ios if both_large else self._dedup_ios





            if both_large:
                min_threshold = (
                    min(threshold, self._seam_span_ios) if seam_span_armed else threshold
                )
            else:
                min_threshold = min(threshold, self._dup_ios_floor)







            kb = keeper.boundingBox()
            iw = min(cand_bbox.xMaximum(), kb.xMaximum()) - max(cand_bbox.xMinimum(), kb.xMinimum())
            ih = min(cand_bbox.yMaximum(), kb.yMaximum()) - max(cand_bbox.yMinimum(), kb.yMinimum())
            if iw <= 0.0 or ih <= 0.0:
                continue
            keeper_area = self._areas.get(fid)
            if keeper_area is None:
                keeper_area = keeper.area()
            min_area = min(keeper_area, cand_area)
            if min_area <= 0.0 or (iw * ih) / min_area < min_threshold:
                continue
            ios, span = _ios_and_span(geom, keeper, a1=cand_area, a2=keeper_area)
            if ios >= threshold:
                matches.append(fid)
                continue











            if both_large and seam_span_armed:
                if (ios >= self._seam_span_ios and span >= self._seam_span_tol * self._seam_min_dim):
                    matches.append(fid)
                    continue



            if (not both_large) and ios >= self._dup_ios_floor:
                smaller_bb = cand_bbox if cand_area <= keeper_area else kb
                smax = max(smaller_bb.width(), smaller_bb.height())
                if smax > 0.0:
                    if cand_centroid is None:
                        cand_centroid = geom.centroid().asPoint()
                    cc = cand_centroid
                    kc = self._centroids.get(fid)
                    if kc is None:
                        kc = keeper.centroid().asPoint()
                        self._centroids[fid] = kc
                    dist = ((cc.x() - kc.x()) ** 2 + (cc.y() - kc.y()) ** 2) ** 0.5
                    if dist < self._dup_centroid_frac * smax:
                        matches.append(fid)

        if matches and self._select_duplicates:



























            primary_fid = min(matches)





            members = [(geom, float(score), cand_area)]
            for fid in matches:
                keeper = self._keepers[fid]
                if keeper is not None:
                    keeper_area = self._areas.get(fid)
                    if keeper_area is None:
                        keeper_area = keeper.area()
                    members.append((keeper, self._scores.get(fid, 0.0), keeper_area))
            members.sort(key=lambda t: t[2], reverse=True)
            largest_area = members[0][2]
            current = members[0][0]
            contributing = {0}
            for i in range(1, len(members)):
                g = members[i][0]
                diff = g.difference(current)
                if diff is None or diff.isEmpty():
                    continue
                if self._gsd > 0.0:
                    eroded = diff.buffer(-self._gsd * self._jitter_erode_px, 5)
                    if eroded is None or eroded.isEmpty() or eroded.area() <= 0.0:
                        continue
                elif diff.area() < self._jitter_area_frac * largest_area:



                    continue
                union = current.combine(g)
                if union is None or union.isEmpty():


                    self._insert(geom, float(score), area=cand_area)
                    return
                current = union
                contributing.add(i)






            floor_area = self._score_floor_frac * largest_area
            best_score = max(
                s for i, (_g, s, a) in enumerate(members)
                if i in contributing or a >= floor_area
            )
            for fid in matches:
                self._retire_keeper(fid)
            if self._absorbed is not None:




                pool = []
                for fid in matches:
                    pool.extend(self._absorbed.pop(fid, ()))




                pool.extend((g.asWkb(), s, a) for i, (g, s, a) in enumerate(members)
                            if i and i not in contributing)
                if pool:
                    self._absorbed[primary_fid] = self._cap_pool(pool)
            self._insert(current, best_score, fid=primary_fid)
        elif matches:
            combined = geom
            best_score = float(score)
            retired = []
            for fid in matches:
                keeper = self._keepers[fid]
                union = combined.combine(keeper)
                if union is not None and not union.isEmpty():
                    combined = union



                    if self._scores.get(fid, 0.0) > best_score:
                        best_score = self._scores.get(fid, 0.0)
                    self._retire_keeper(fid)
                    retired.append(fid)
            if retired:




                self._insert(combined, best_score, fid=min(retired))
            else:


                self._insert(geom, float(score), area=cand_area)
        else:
            self._insert(geom, float(score), area=cand_area)

    def _insert(self, geom: QgsGeometry, score: float = 0.0, fid: int | None = None,
                area: float | None = None) -> None:







        use_id = self._next_id if fid is None else fid
        feat = QgsFeature(use_id)
        feat.setGeometry(geom)


        self._index.addFeature(feat)
        self._index_entries += 1
        self._keepers[use_id] = geom
        self._scores[use_id] = float(score)


        self._areas[use_id] = geom.area() if area is None else float(area)




        self._centroids.pop(use_id, None)
        self._live_ids[use_id] = None



        self._dirty[use_id] = None
        self._gone.pop(use_id, None)
        if fid is None:
            self._next_id += 1
        self._maybe_compact()

    def _maybe_compact(self) -> None:
















        live = len(self._live_ids)


        stale = (live >= _td.merge_compact_min_live(_COMPACT_MIN_LIVE)
                 and self._index_entries > 2 * live)
        if not stale and len(self._keepers) - live <= 4 * live:
            return
        self._keepers = {fid: self._keepers[fid] for fid in self._live_ids}
        self._scores = {fid: self._scores[fid] for fid in self._live_ids}
        self._areas = {fid: self._areas[fid] for fid in self._live_ids}


        self._centroids = {
            fid: c for fid, c in self._centroids.items() if fid in self._live_ids
        }







        index = QgsSpatialIndex()
        for fid in self._live_ids:
            feat = QgsFeature(fid)
            feat.setGeometry(self._keepers[fid])
            index.addFeature(feat)
        self._index = index
        self._index_entries = len(self._live_ids)

    def _cap_pool(self, pool: list) -> list:

        cap = max(1, self._part_min_children) * _td.merge_absorbed_pool_mult(_ABSORBED_POOL_MULT)
        if len(pool) <= cap:
            return pool
        return sorted(pool, key=lambda t: -t[2])[:cap]

    def _overlaps_taken(self, geom, bbox, area: float, taken: list) -> bool:






        for other, obox, oarea in taken:
            iw = min(bbox.xMaximum(), obox.xMaximum()) - max(bbox.xMinimum(), obox.xMinimum())
            ih = min(bbox.yMaximum(), obox.yMaximum()) - max(bbox.yMinimum(), obox.yMinimum())
            if iw <= 0.0 or ih <= 0.0:
                continue
            min_area = min(area, oarea)
            if min_area <= 0.0 or (iw * ih) / min_area < self._part_sibling_ios:
                continue
            if _ios_and_span(geom, other, a1=area, a2=oarea)[0] >= self._part_sibling_ios:
                return True
        return False

    def _partition_of(self, whole: QgsGeometry, parts: list) -> list:








        whole_area = whole.area()
        if whole_area <= 0.0:
            return []
        picked: list = []
        taken: list = []
        for wkb, score, area in sorted(parts, key=lambda t: -t[2]):
            if area <= 0.0 or area > self._part_max_frac * whole_area:
                continue
            geom = QgsGeometry()
            geom.fromWkb(wkb)
            if geom.isEmpty():
                continue
            inter = geom.intersection(whole)
            if inter is None or inter.isEmpty():
                continue
            if inter.area() / area < self._part_inside:
                continue
            bbox = geom.boundingBox()
            if self._overlaps_taken(geom, bbox, area, taken):
                continue
            picked.append((geom, score))
            taken.append((geom, bbox, area))
        if len(picked) < self._part_min_children:
            return []
        covered = QgsGeometry(picked[0][0])
        for geom, _score in picked[1:]:
            grown = covered.combine(geom)
            if grown is not None and not grown.isEmpty():
                covered = grown
        got = covered.intersection(whole)
        if got is None or got.isEmpty():
            return []
        if got.area() < self._part_cover_frac * whole_area:
            return []
        return picked

    def restore_absorbed_partitions(self) -> int:








        self.restored_fids = set()
        if not self._absorbed:
            return 0
        restored = 0
        for fid in list(self._live_ids):
            parts = self._absorbed.get(fid)
            keeper = self._keepers.get(fid)
            if not parts or keeper is None:
                continue
            picked = self._partition_of(keeper, parts)
            if not picked:
                continue
            self._retire_keeper(fid)
            self.restored_fids.add(fid)
            for geom, score in picked:




                self.restored_fids.add(self._next_id)
                self._insert(QgsGeometry(geom), float(score))
            restored += 1
        self._absorbed.clear()
        return restored

    def result(self) -> list:

        return [self._keepers[fid] for fid in self._live_ids]

    def result_scored(self) -> list:






        return [
            (self._keepers[fid], self._scores.get(fid, 0.0))
            for fid in self._live_ids
        ]

    def result_scored_ided(self) -> list:









        return [
            (fid, self._keepers[fid], self._scores.get(fid, 0.0))
            for fid in self._live_ids
        ]
