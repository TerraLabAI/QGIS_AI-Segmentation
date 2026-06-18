from __future__ import annotations

import logging

from qgis.core import QgsFeature, QgsGeometry, QgsSpatialIndex

from .fragment_graph import FragmentGraph, FragmentReference
from .layer_conventions import repair_polygon

logger = logging.getLogger(__name__)


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
        *,
        merge_ios: float,
        dedup_ios: float,
        seam_min_dim: float = 0.0,
        dup_ios_floor: float,
        dup_centroid_frac: float,
        seam_span_ios: float,
        select_duplicates: bool = False,
        gsd: float = 0.0,
        seam_span_tol: float,
        jitter_area_frac: float,
        jitter_erode_px: float,
        score_floor_frac: float,
        restore_partitions: bool = False,
        part_inside: float,
        part_max_frac: float,
        part_sibling_ios: float,
        part_cover_frac: float,
        part_min_children: int,
    ):



        self.init_kwargs = {
            name: value for name, value in locals().items() if name != "self"}
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
        self._fragment_graph = FragmentGraph()
        self._canonical_pending: set[int] = set()
        self._keepers: dict[int, QgsGeometry | None] = {}






        self._scores: dict[int, float] = {}
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






        self._keepers.pop(fid, None)
        self._scores.pop(fid, None)
        self._live_ids.pop(fid, None)
        self._dirty.pop(fid, None)
        self._gone[fid] = None

    def _fragments_match(self, candidate, reference, candidate_geom) -> bool:

        cb, rb = candidate.bbox, reference.bbox
        both_large = (self._seam_min_dim <= 0.0 or (
            max(cb.width(), cb.height()) >= self._seam_min_dim
            and max(rb.width(), rb.height()) >= self._seam_min_dim))
        threshold = self._merge_ios if both_large else self._dedup_ios
        seam_span_armed = (
            0.0 < self._seam_min_dim < float("inf") and not self._select_duplicates)
        floor = (min(threshold, self._seam_span_ios) if both_large and seam_span_armed
                 else threshold if both_large else min(threshold, self._dup_ios_floor))
        iw = min(cb.xMaximum(), rb.xMaximum()) - max(cb.xMinimum(), rb.xMinimum())
        ih = min(cb.yMaximum(), rb.yMaximum()) - max(cb.yMinimum(), rb.yMinimum())
        small_area = min(candidate.area, reference.area)
        if iw <= 0.0 or ih <= 0.0 or small_area <= 0.0 or iw * ih / small_area < floor:
            return False
        keeper = reference.geometry()
        ios, span = _ios_and_span(
            candidate_geom, keeper, candidate.area, reference.area)
        if ios >= threshold:
            return True
        if both_large and seam_span_armed:
            return (ios >= self._seam_span_ios
                    and span >= self._seam_span_tol * self._seam_min_dim)
        if not both_large and ios >= self._dup_ios_floor:
            cand_dim = max(cb.width(), cb.height())
            ref_dim = max(rb.width(), rb.height())


            dim = (cand_dim if candidate.area < reference.area else
                   ref_dim if reference.area < candidate.area else min(cand_dim, ref_dim))
            cc, rc = candidate.centre(candidate_geom), reference.centre(keeper)
            distance = ((cc.x() - rc.x()) ** 2 + (cc.y() - rc.y()) ** 2) ** 0.5
            return dim > 0.0 and distance < self._dup_centroid_frac * dim
        return False

    def _compose_component(self, root: int):






        return self._compose_members(self._fragment_graph.members(root))

    def _compose_members(self, members: list[FragmentReference]):

        if not self._select_duplicates:
            current = QgsGeometry.unaryUnion([member.geometry() for member in members])
            score = max(member.score for member in members)
            pool = []
        else:
            largest_area = members[0].area
            current = members[0].geometry()
            contributing = {0}
            pool = []
            for index, member in enumerate(members[1:], 1):
                geom = member.geometry()
                difference = geom.difference(current)
                redundant = difference is None or difference.isEmpty()
                if not redundant and self._gsd > 0.0:
                    eroded = difference.buffer(-self._gsd * self._jitter_erode_px, 5)
                    redundant = eroded is None or eroded.isEmpty() or eroded.area() <= 0.0
                elif not redundant:
                    redundant = difference.area() < self._jitter_area_frac * largest_area
                if redundant:
                    pool.append((member.wkb, member.score, member.area))
                    continue
                union = current.combine(geom)
                if union is None or union.isEmpty():
                    return None, 0.0, []
                current = union
                contributing.add(index)
            floor_area = self._score_floor_frac * largest_area
            score = max(member.score for index, member in enumerate(members)
                        if index in contributing or member.area >= floor_area)
        if current is None or current.isEmpty():
            return None, 0.0, []
        current.normalize()
        return current, score, pool

    def _install_component(self, root: int, current, score: float, pool: list) -> int:

        previous = self._fragment_graph.keepers(root)
        primary = min(previous) if previous else self._next_id
        keeper = self._keepers.get(primary)
        same_geometry = (len(previous) == 1 and keeper is not None
                         and bytes(keeper.asWkb()) == bytes(current.asWkb()))
        if same_geometry:

            if self._scores.get(primary) != score:
                self._scores[primary] = float(score)
                self._dirty[primary] = None
        else:
            for fid in previous:
                self._retire_keeper(fid)
            self._insert(current, score, fid=primary if previous else None)
        if self._absorbed is not None:
            for fid in previous:
                self._absorbed.pop(fid, None)
            if pool:



                self._absorbed[primary] = pool
        self._fragment_graph.bind(root, {primary})
        return primary

    def _canonicalize_pending(self) -> None:

        roots = {self._fragment_graph.root(root) for root in self._canonical_pending}
        self._canonical_pending.clear()
        for root in roots:
            previous = self._fragment_graph.keepers(root)
            try:
                current, score, pool = self._compose_component(root)
            except Exception:  # noqa: BLE001
                logger.exception("IncrementalMerger: canonical union failed")
                self._canonical_pending.add(root)
                continue
            if current is None:
                self._canonical_pending.add(root)
                continue
            previous_geom = self._keepers.get(min(previous)) if len(previous) == 1 else None
            changed_geometry = (previous_geom is None
                                or bytes(previous_geom.asWkb()) != bytes(current.asWkb()))
            fid = self._install_component(root, current, score, pool)
            if changed_geometry:
                self.restored_fids.update(previous | {fid})

    def add(self, geom: QgsGeometry, score: float = 0.0) -> None:
        if geom is None or geom.isEmpty():
            return
        geom = repair_polygon(geom)
        if geom is None or geom.isEmpty():
            return


        geom = QgsGeometry(geom)
        geom.normalize()
        root, changed = self._fragment_graph.add(geom, score, self._fragments_match)
        if not changed:
            return
        previous = self._fragment_graph.keepers(root)
        self._canonical_pending.add(root)
        try:
            if self._select_duplicates:




                members = [FragmentReference(self._keepers[fid], self._scores[fid])
                           for fid in previous if self._keepers.get(fid) is not None]
                members.append(FragmentReference(geom, score))
                members.sort(key=lambda member: (-member.area, member.wkb))
                current, best_score, pool = self._compose_members(members)
            else:



                pieces = [self._keepers[fid] for fid in previous
                          if self._keepers.get(fid) is not None]
                pieces.append(geom)
                current = QgsGeometry.unaryUnion(pieces) if len(pieces) > 1 else geom
                best_score = max([float(score)] + [self._scores[fid] for fid in previous])
                pool = []
                if current is not None and not current.isEmpty():
                    current.normalize()
        except Exception:  # noqa: BLE001
            logger.exception("IncrementalMerger: component union failed")
            current = None
        if current is None or current.isEmpty():


            fid = self._next_id
            self._insert(geom, float(score))
            self._fragment_graph.bind(root, previous | {fid})
            self._canonical_pending.add(root)
            return
        self._install_component(root, current, best_score, pool)

    def _insert(self, geom: QgsGeometry, score: float = 0.0, fid: int | None = None) -> None:


        use_id = self._next_id if fid is None else fid
        self._keepers[use_id] = geom
        self._scores[use_id] = float(score)
        self._live_ids[use_id] = None
        self._dirty[use_id] = None
        self._gone.pop(use_id, None)
        if fid is None:
            self._next_id += 1

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
        taken_index = QgsSpatialIndex()
        for wkb, score, area in sorted(parts, key=lambda t: (-t[2], t[0])):
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
            neighbours = (taken[index] for index in taken_index.intersects(bbox))
            if self._overlaps_taken(geom, bbox, area, neighbours):
                continue
            picked.append((geom, score))
            feature = QgsFeature(len(taken))
            feature.setGeometry(geom)
            taken_index.addFeature(feature)
            taken.append((geom, bbox, area))
        if len(picked) < self._part_min_children:
            return []
        covered = QgsGeometry.unaryUnion([geom for geom, _score in picked])
        if covered is None or covered.isEmpty():
            return []
        got = covered.intersection(whole)
        if got is None or got.isEmpty():
            return []
        if got.area() < self._part_cover_frac * whole_area:
            return []
        return picked

    def restore_absorbed_partitions(self) -> int:








        self.restored_fids = set()
        self._canonicalize_pending()
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
            replacements = []
            for geom, score in picked:




                self.restored_fids.add(self._next_id)
                replacements.append(self._next_id)
                self._insert(QgsGeometry(geom), float(score))
            self._fragment_graph.replace_keeper(fid, replacements)
            restored += 1
        self._absorbed.clear()
        return restored

    def result(self) -> list:

        self._canonicalize_pending()
        return [self._keepers[fid] for fid in self._result_fids()]

    def _result_fids(self) -> list[int]:

        return sorted(self._live_ids, key=lambda fid: bytes(self._keepers[fid].asWkb()))

    def result_scored(self) -> list:






        self._canonicalize_pending()
        return [
            (self._keepers[fid], self._scores.get(fid, 0.0))
            for fid in self._result_fids()
        ]

    def result_scored_ided(self) -> list:









        self._canonicalize_pending()
        return [
            (fid, self._keepers[fid], self._scores.get(fid, 0.0))
            for fid in self._result_fids()
        ]
