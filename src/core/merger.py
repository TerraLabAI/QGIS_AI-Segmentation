from __future__ import annotations

from qgis.core import (
    QgsFeature,
    QgsGeometry,
    QgsSpatialIndex,
)

# Live objects below which _maybe_compact never rebuilds the spatial index: the
# candidate scan is short whatever the index holds, and a rebuild per insert
# would cost more than the scan it shortens.
_COMPACT_MIN_LIVE = 64


def _ios_and_span(g1: QgsGeometry, g2: QgsGeometry,
                  a1: float | None = None,
                  a2: float | None = None) -> tuple[float, float]:
    """Return (IoS, overlap_span) for two polygons.

    IoS = intersection area over the SMALLER polygon's area (see _overlap_metrics).
    overlap_span = the longer side of the intersection's bounding box, in ground
    units. A genuine tile-seam split leaves a thin overlap strip whose longer
    dimension runs the length of the object along the seam, so a large
    overlap_span (comparable to the tile-overlap width) is the tell-tale of a
    seam split even when the overlap AREA (IoS) is tiny relative to a big object.
    Both are 0 on empty/degenerate input. ``a1``/``a2`` let a caller that
    already computed the polygon areas skip the O(vertices) recompute
    (QgsGeometry.area() is not cached).
    """
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
    except Exception:
        return 0.0, 0.0


class IncrementalMerger:
    """Greedy non-max merging maintained online as tile fragments arrive.

    A single batch merge at the end of a run would leave the live preview showing
    raw, cut tile fragments until then. This keeps a running merged set instead:
    each add() folds a fragment in immediately, so the preview shows whole,
    stitched objects as tiles complete.

    A fragment unions into EVERY existing object it overlaps by at least the
    pair's threshold (IoS, intersection over the smaller area), fusing them all
    into one. IoS (not IoU) is used so the thin overlap a long object (road,
    river) leaves in the tile overlap strip still registers. Merging all matches
    (not just the best) is what fully stitches a long object whose pieces arrive
    out of tile order: a bridging fragment that touches two already-grown keepers
    fuses them into one.

    Size-aware gate (the anti-over-merge safety). The low merge_ios is meant only
    to stitch one large object cut by a tile boundary. But IoS divides by the
    SMALLER area, so a sliver of overlap between two small objects also clears it,
    and the bridging then chains distinct small neighbours (solar panels, cars,
    trees) into one blob even when there is clear space between them. The fix:
    an object shorter than ``seam_min_dim`` in both bbox dimensions is guaranteed
    to fit whole inside one tile (it is <= the inter-tile overlap strip), so it
    can only ever be a cross-tile DUPLICATE, never a seam-split half. Two pieces
    therefore merge at the low merge_ios only when BOTH could span a seam (both
    >= seam_min_dim); any pair involving a small object must instead clear the
    strict dedup_ios (a near-duplicate). Distinct small neighbours have IoS well
    below dedup_ios, so they stay apart, while a true duplicate of a small object
    (the same panel seen in two overlapping tiles) sits near IoS 1.0 and still
    merges. seam_min_dim = 0.0 disables the gate (every pair uses merge_ios, the
    original behaviour) for callers without a known tile size.

    Order-independent: union is commutative, so the result does not depend on the
    order tiles complete in.
    """

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
        # Ground units per detection pixel. Sizes the one-pixel erosion that
        # separates jitter duplicates from real seam complements in the
        # additive-union select branch (see add()). 0.0 (unknown gsd) falls
        # back to a relative added-area floor instead of the erosion test.
        self._gsd = float(gsd)
        # SEPARATE/count mode: a matched group is EITHER redundant readings of
        # one footprint (cross-tile jitter duplicate, parent vs child
        # hypothesis) OR the pieces of one object cut by a tile seam. The
        # additive-union select branch in add() tells them apart PER MEMBER
        # with an erosion test on the member's added area: redundant members
        # are skipped (union would only add outline dilation or rebuild the
        # mega-blob the per-tile NMS killed; score comes only from
        # co-extensive or contributing members so a small high-score child
        # never promotes a big low-score parent), real complements are
        # stitched in (discarding one renders the object truncated flat along
        # the tile grid). MAP/continuous callers keep the plain union
        # (coverage is a union by design).
        self._select_duplicates = select_duplicates
        # The select branch treats "adds no new area" as "redundant reading",
        # which is right for a cross-tile jitter duplicate and wrong for the
        # individual buildings of a complex the model ALSO read as one shape:
        # each sits whole inside that shape, so each adds nothing and each is
        # skipped. Per tile the loss is invisible (some tile keeps the fine
        # reading), it lands here when the tiles meet, and it is the largest
        # single source of missed objects in this pipeline.
        #
        # On, the skipped members are REMEMBERED against the object that
        # swallowed them, and restore_absorbed_partitions() gives them back
        # when they turn out to account for it. Off (the default) nothing is
        # recorded and every output is byte-identical to before.
        self._restore_partitions = bool(restore_partitions)
        self._absorbed: dict[int, list] | None = {} if restore_partitions else None
        self._part_inside = float(part_inside)
        self._part_max_frac = float(part_max_frac)
        self._part_sibling_ios = float(part_sibling_ios)
        self._part_cover_frac = float(part_cover_frac)
        self._part_min_children = int(part_min_children)
        # Seam-span rescue for MISSED JOINS: a large object split ~50/50 by a
        # tile seam overlaps only inside the thin (~20%) overlap strip, so its
        # IoS can fall below merge_ios and plain IoS merging leaves a straight
        # seam line through the object. When BOTH pieces are seam-eligible (each
        # big enough to be a tile-cut half) and the overlap REGION spans at least
        # a seam-strip width in its longer dimension (the tell-tale of a seam
        # split, not an incidental corner touch), merge them at this much lower
        # IoS floor. Gated to seam-eligible pairs, a finite positive
        # seam_min_dim AND union mode (select_duplicates=False), so it never
        # fuses small distinct neighbours, never fires when the size gate is
        # off, and never lets a common-wall sliver between two DISTINCT large
        # buildings reach the selection-mode stitch (see add()).
        self._seam_span_ios = seam_span_ios
        # Robust same-object dedup: a duplicate the cloud model drew slightly differently
        # across two tiles can fall below dedup_ios yet still be ONE object. Treat
        # it as a duplicate when the overlap clears dup_ios_floor AND the
        # centroids sit within dup_centroid_frac of the smaller object's size.
        # Conservative: distinct neighbours have far-apart centroids, so this
        # never fuses them. Applies to the non-seam (small-object) branch only.
        self._dup_ios_floor = dup_ios_floor
        self._dup_centroid_frac = dup_centroid_frac
        # Three more merge scalars, kwargs like the five above so a run can pass
        # its server-resolved values (review.merge) and the inference service's
        # mirror of these thresholds can be tuned in lockstep. The defaults are
        # the generic client fallbacks.
        self._seam_span_tol = seam_span_tol
        self._jitter_area_frac = jitter_area_frac
        self._score_floor_frac = score_floor_frac
        self._index = QgsSpatialIndex()
        # Bbox entries currently sitting in _index. Not the same as the live
        # object count and not the same as the retired count: a merge REUSES the
        # primary keeper's fid, so it adds an entry without minting a key in
        # _keepers, and the stale entry under that fid stays until a rebuild.
        # Over a dense fold this is what the candidate scan actually walks, so it
        # is what _maybe_compact has to watch.
        self._index_entries = 0
        self._keepers: dict[int, QgsGeometry | None] = {}
        # Representative score per keeper: the MAX of its constituent fragments'
        # scores, so a strong object survives the review confidence filter even
        # if one seam-half scored low (that is exactly the "confidence cuts
        # buildings" bug: filtering fragments before stitching drops the weak
        # half). Merges take the max across the candidate and every keeper it
        # absorbs.
        self._scores: dict[int, float] = {}
        # Ground area per keeper, cached at insert time. QgsGeometry.area() is an
        # O(vertices) GEOS walk with no internal cache, and the candidate loop in
        # add() needs the keeper's area for every fragment that lands in its
        # bbox. A keeper's geometry never changes in place (a merge retires the
        # fid and inserts the union), so the cached value stays exact.
        self._areas: dict[int, float] = {}
        self._next_id = 0
        # Ids of the non-retired keepers, kept in lockstep with _keepers/_scores
        # (add on _insert, drop on retirement). result()/result_scored() walk this
        # instead of scanning every id ever inserted, so the retired None slots
        # that pile up over a dense run cost nothing. Stored as a
        # dict used as an INSERTION-ORDERED set (keys only) so result order is
        # byte-identical to the old _keepers.items() scan (an object retired then
        # re-inserted lands last in both). Pure bookkeeping: the merge LOGIC is
        # unchanged. The candidate loop in add() deliberately stays driven by
        # _index.intersects()+_keepers.get(), so a retired fid still returned by
        # the index is skipped exactly as before.
        self._live_ids: dict[int, None] = {}
        # Change log for a live consumer. _dirty holds every fid whose geometry
        # was inserted or replaced since the last drain_changes(), _gone every
        # fid retired with nothing put back under it. A preview that folds this
        # delta costs the size of the change, not the size of the whole set,
        # which is what lets the live layer keep up on a dense run. Both are
        # dicts used as insertion-ordered sets, so a delta arrives in the order
        # the objects changed.
        self._dirty: dict[int, None] = {}
        self._gone: dict[int, None] = {}

    def drain_changes(self) -> tuple[list[int], list[int]]:
        """(changed, removed) keeper fids since the last call, then reset.

        ``changed`` is every LIVE keeper whose geometry was inserted or
        replaced; ``removed`` is every keeper retired with no replacement under
        the same fid. A fid never appears in both: a merge that reuses the
        primary's fid reports it as changed, not removed.
        """
        changed = list(self._dirty)
        removed = list(self._gone)
        self._dirty = {}
        self._gone = {}
        return changed, removed

    def keeper(self, fid: int) -> tuple[QgsGeometry | None, float]:
        """(geometry, score) of one live keeper, or (None, 0.0) once retired."""
        geom = self._keepers.get(fid)
        if geom is None:
            return None, 0.0
        return geom, self._scores.get(fid, 0.0)

    def mark_changed(self, fids) -> None:
        """Report these live keepers as changed again on the next drain.

        For a consumer whose own rendering scale moved under it: the merger did
        not touch the geometry, but the consumer has to build it again. Fids
        that are no longer live are ignored.
        """
        for fid in fids:
            if fid in self._live_ids:
                self._dirty[fid] = None

    def _retire_keeper(self, fid: int) -> None:
        """Drop one keeper from the live set and record it in the change log.

        Every retirement goes through here: a site that clears the three dicts
        by hand would leave the live preview drawing an object the merger has
        already absorbed.
        """
        self._keepers[fid] = None
        self._live_ids.pop(fid, None)
        self._areas.pop(fid, None)
        self._dirty.pop(fid, None)
        self._gone[fid] = None

    def _is_seam_eligible(self, geom: QgsGeometry) -> bool:
        """True if geom is large enough to possibly be cut by a tile boundary.

        Below seam_min_dim in both bbox dimensions an object always fits whole in
        one tile, so it can only be a duplicate, not a seam-split half. When the
        gate is off (seam_min_dim <= 0) everything is treated as seam-eligible.
        """
        if self._seam_min_dim <= 0.0:
            return True
        bb = geom.boundingBox()
        return max(bb.width(), bb.height()) >= self._seam_min_dim

    def add(self, geom: QgsGeometry, score: float = 0.0) -> None:
        if geom is None or geom.isEmpty():
            return
        # Find every existing object this fragment overlaps enough to be part of.
        # Merging ALL of them (not just the best) stitches a long object whose
        # pieces arrive out of tile order. The size-aware threshold keeps that
        # from chaining distinct SMALL neighbours: a low merge_ios bridge is
        # allowed only when both pieces are large enough to be tile-cut halves;
        # any pair with a small object needs the strict dedup_ios (true duplicate).
        cand_bbox = geom.boundingBox()
        cand_area = geom.area()
        cand_seam = self._is_seam_eligible(geom)
        # Seam-span rescue is only armed when the size gate is a finite positive
        # span (a known GSD) AND the caller wants unions (map/continuous). In
        # selection mode it stays off: two DISTINCT adjacent buildings sharing a
        # long thin overlap (common wall, shadow fringe) have a large overlap
        # span at a tiny IoS, exactly what the rescue accepts, and the stitch
        # branch would then fuse them. Seam halves of one big object clear the
        # plain merge_ios instead (their overlap strip is a large fraction of
        # the smaller half up to ~500 m objects).
        seam_span_armed = (
            0.0 < self._seam_min_dim < float("inf") and not self._select_duplicates
        )
        matches = []
        # The incoming fragment's centroid, computed at most once per add(): it
        # is the same geometry for the whole loop, so recomputing it inside the
        # dup branch repeats an O(vertices) GEOS call per candidate.
        cand_centroid = None
        # dict.fromkeys dedupes the candidate fids: a merge reuses the primary
        # keeper's fid (see the branches below), so after that reuse the index
        # holds TWO bbox entries under that fid (the stale retired bbox plus the
        # fresh union bbox) until the next compaction. Without this dedupe
        # intersects() would return the reused fid twice within one add(); the
        # second occurrence now sees a LIVE keeper and would self-merge the
        # object into itself (or, in the stitch branch, combine() a keeper that
        # the first occurrence already retired to None). Processing each fid at
        # most once keeps matches unique and the merge logic byte-unchanged.
        for fid in dict.fromkeys(self._index.intersects(cand_bbox)):
            keeper = self._keepers.get(fid)
            if keeper is None:
                continue
            both_large = cand_seam and self._is_seam_eligible(keeper)
            threshold = self._merge_ios if both_large else self._dedup_ios
            # The lowest IoS we might still accept, used only to size the cheap
            # bbox pre-filter below. The non-seam branch can merge a near-
            # coincident duplicate down at dup_ios_floor (centroid test); the
            # seam branch can rescue a seam split down at seam_span_ios. The
            # pre-filter must not prune below whichever floor applies.
            if both_large:
                min_threshold = (
                    min(threshold, self._seam_span_ios) if seam_span_armed else threshold
                )
            else:
                min_threshold = min(threshold, self._dup_ios_floor)
            # Cheap bbox pre-filter before the costly intersection(): the real
            # geometry overlap can never exceed the bbox overlap, so if even
            # bbox_overlap / smaller_area is below the threshold the pair cannot
            # merge. This skips intersection() for the many bbox-adjacent-but-
            # distinct neighbours in dense scenes (packed panels, urban roofs),
            # the dominant main-thread cost of a big run. No false negatives:
            # bbox overlap is an upper bound on the true overlap.
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
            # Seam-span rescue (large pairs only): a true tile-seam split has a
            # low IoS but its overlap strip runs the object's length along the
            # seam, so its span reaches a seam-strip width. Distinct large
            # neighbours are spatially apart, so their overlap span is small.
            # The span requirement takes a tolerance: mask edges land a pixel
            # or two short of the tile border and the simplify pass shaves the
            # strip further, so a genuine seam strip measures slightly UNDER
            # the theoretical overlap width; requiring the full width lets
            # near-exact seam splits through as overlapping duplicates. An
            # incidental corner touch between distinct neighbours stays far
            # below this scale.
            if both_large and seam_span_armed:
                if (ios >= self._seam_span_ios and span >= self._seam_span_tol * self._seam_min_dim):
                    matches.append(fid)
                    continue
            # Robust same-object dedup (non-seam only): decent overlap + nearly
            # coincident centroids => the same object split/redrawn across tiles,
            # so it must not be counted twice.
            if (not both_large) and ios >= self._dup_ios_floor:
                smaller_bb = cand_bbox if cand_area <= keeper_area else kb
                smax = max(smaller_bb.width(), smaller_bb.height())
                if smax > 0.0:
                    if cand_centroid is None:
                        cand_centroid = geom.centroid().asPoint()
                    cc = cand_centroid
                    kc = keeper.centroid().asPoint()
                    dist = ((cc.x() - kc.x()) ** 2 + (cc.y() - kc.y()) ** 2) ** 0.5
                    if dist < self._dup_centroid_frac * smax:
                        matches.append(fid)

        if matches and self._select_duplicates:
            # A matched group is one of two things, and the ADDITIVE UNION
            # below tells them apart PER MEMBER instead of by a single
            # union/largest ratio (a single ratio test discards a seam
            # complement that extends the object by less than the ratio margin,
            # which leaves big buildings with a flat wall along the tile grid
            # where the discarded complement would have carried the real width):
            #  - REDUNDANT readings of the same footprint (cross-tile jitter
            #    duplicate, parent/child hypothesis): the member adds no real
            #    area, unioning it would only dilate the outline or rebuild
            #    the mega-blob the per-tile NMS killed -> skipped.
            #  - PIECES of ONE object cut by a tile seam (or emitted as
            #    sections): the member contributes real new area past the
            #    growing shape -> unioned (the stitch; discarding it renders
            #    the object truncated along the tile grid).
            # The discriminator: the member's difference against the growing
            # shape must SURVIVE a one-pixel erosion. A duplicate's difference
            # is a sub-pixel jitter ring that erodes to nothing; a genuine
            # complement's difference is real width that survives, so it heals
            # the flat wall without growing the outline or rebuilding a
            # mega-blob. Cross-tile parent-vs-children
            # is unchanged: the largest member already won before, and a
            # child inside it adds no surviving difference.
            # The union carries the PRIMARY keeper's fid (lowest among the
            # matched keepers, deterministic) so an object keeps ONE stable id
            # across every merge that grows it (the id drives the live per-object
            # colour and the refine cache in the repaint loop). All matched
            # keepers are retired; the primary's fid is reused for the union.
            primary_fid = min(matches)
            members = [(geom, float(score))]
            for fid in matches:
                keeper = self._keepers[fid]
                if keeper is not None:
                    members.append((keeper, self._scores.get(fid, 0.0)))
                self._retire_keeper(fid)
            members.sort(key=lambda t: t[0].area(), reverse=True)
            largest_area = members[0][0].area()
            current = members[0][0]
            contributing = {0}
            for i in range(1, len(members)):
                g = members[i][0]
                diff = g.difference(current)
                if diff is None or diff.isEmpty():
                    continue
                if self._gsd > 0.0:
                    eroded = diff.buffer(-self._gsd, 5)
                    if eroded is None or eroded.isEmpty() or eroded.area() <= 0.0:
                        continue
                elif diff.area() < self._jitter_area_frac * largest_area:
                    # Unknown gsd (no pixel size to erode by): fall back to a
                    # small relative floor so pure-jitter rings still never
                    # dilate the outline.
                    continue
                union = current.combine(g)
                if union is not None and not union.isEmpty():
                    current = union
                    contributing.add(i)
            # Score: the MAX over members that are co-extensive with the
            # largest (area >= half of it: a redundant full reading) or that
            # actually contributed area (a stitched seam half). A small
            # high-score child that added nothing never promotes the object;
            # a contributing half keeps its score (filtering a half before
            # stitching is the "confidence cuts buildings" bug).
            floor_area = self._score_floor_frac * largest_area
            best_score = max(
                s for i, (g, s) in enumerate(members)
                if i in contributing or g.area() >= floor_area
            )
            if self._absorbed is not None:
                # Carry each matched keeper's own history onto the survivor, so
                # a complex swallowed over three tiles still knows all its
                # parts, then add the members this union skipped. members[0]
                # seeded the union and is never a part of itself.
                pool = []
                for fid in matches:
                    pool.extend(self._absorbed.pop(fid, ()))
                pool.extend(g_s for i, g_s in enumerate(members)
                            if i and i not in contributing)
                if pool:
                    self._absorbed[primary_fid] = pool
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
                    # Only retire a keeper whose geometry was actually absorbed;
                    # retiring on a failed union would silently drop its area.
                    # The merged object keeps the MAX score of everything in it.
                    if self._scores.get(fid, 0.0) > best_score:
                        best_score = self._scores.get(fid, 0.0)
                    self._retire_keeper(fid)
                    retired.append(fid)
            if retired:
                # Reuse the lowest RETIRED keeper's fid (primary) so the stitched
                # object keeps one stable id across merges. min(retired), not
                # min(matches): a matched fid whose combine() failed stays LIVE,
                # so reusing it would collide two live keepers on one id.
                self._insert(combined, best_score, fid=min(retired))
            else:
                # No union succeeded (all combine() calls failed); keep the
                # fragment as its own object rather than losing it.
                self._insert(geom, float(score), area=cand_area)
        else:
            self._insert(geom, float(score), area=cand_area)

    def _insert(self, geom: QgsGeometry, score: float = 0.0, fid: int | None = None,
                area: float | None = None) -> None:
        # fid=None mints a genuinely new object from _next_id (unchanged path).
        # An explicit fid REUSES a retired keeper's id for a merge product: the
        # object keeps its stable colour/refine id across the stitch. _next_id is
        # NOT bumped on reuse (it stays a strict high-water mark, so a future new
        # object can never collide with a reused id). Reuse re-adds a bbox entry
        # under an id whose stale retired bbox still sits in _index until the next
        # compaction; add()'s candidate dedupe absorbs that duplicate.
        use_id = self._next_id if fid is None else fid
        feat = QgsFeature(use_id)
        feat.setGeometry(geom)
        self._index.insertFeature(feat)
        self._index_entries += 1
        self._keepers[use_id] = geom
        self._scores[use_id] = float(score)
        # ``area`` lets a caller that already measured this geometry skip the
        # recompute; it must be geom.area() exactly, never an approximation.
        self._areas[use_id] = geom.area() if area is None else float(area)
        self._live_ids[use_id] = None
        # Change log: this fid now carries a geometry the consumer has not seen.
        # A merge that reuses a just-retired primary fid lands here too, so the
        # reuse reads as a change and never as a delete followed by an add.
        self._dirty[use_id] = None
        self._gone.pop(use_id, None)
        if fid is None:
            self._next_id += 1
        self._maybe_compact()

    def _maybe_compact(self) -> None:
        """Drop the retired None slots from _keepers/_scores AND rebuild _index
        from the survivors once the stale entries dominate, so a dense run's
        memory and add()'s intersects() scan both track the live object count
        instead of every fragment ever inserted.
        Behavior-preserving: _next_id is never reset (so it can never collide
        with a retired fid still living in _index), and add() reads keepers via
        .get(), so a dropped key reads identically to a None slot. _live_ids is
        the surviving id set, so the rebuilt dicts hold exactly the live objects
        in the same order.

        The trigger reads _index_entries, not the retired count. Retired grows
        only when one add() absorbs two or more keepers at once, which is rare,
        so on a dense fold that condition can go a whole run without firing
        while the index keeps every bbox ever inserted and the candidate scan
        walks them all. The retired condition stays as the second arm, so
        nothing that compacted before stops compacting now."""
        live = len(self._live_ids)
        # Under this the scan is short whatever the index holds, and rebuilding
        # per insert would cost more than it saves.
        stale = live >= _COMPACT_MIN_LIVE and self._index_entries > 2 * live
        if not stale and len(self._keepers) - live <= 4 * live:
            return
        self._keepers = {fid: self._keepers[fid] for fid in self._live_ids}
        self._scores = {fid: self._scores[fid] for fid in self._live_ids}
        self._areas = {fid: self._areas[fid] for fid in self._live_ids}
        # Rebuild the spatial index from the survivors too: retired fids are
        # skipped by add() anyway (keepers.get() is None), but their bboxes
        # otherwise pile up and add()'s intersects() scan grows with every
        # fragment ever inserted instead of the live count. _next_id is still
        # never reset, so fresh ids can never collide with pruned ones. Each
        # feature carries the CURRENT merged geometry (_keepers[fid]) under its
        # own live fid, so the index stays consistent with the keeper dicts.
        index = QgsSpatialIndex()
        for fid in self._live_ids:
            feat = QgsFeature(fid)
            feat.setGeometry(self._keepers[fid])
            index.insertFeature(feat)
        self._index = index
        self._index_entries = len(self._live_ids)

    def _partition_of(self, whole: QgsGeometry, parts: list) -> list:
        """The largest subset of ``parts`` that reads as a partition of
        ``whole``, or [] when there is none.

        Each part must sit inside the whole, be clearly smaller than it, and
        not overlap the parts already taken; together they must account for
        most of it. Anything less and the whole stays: one building whose roof
        carried a second hypothesis must never come back as two buildings.
        """
        whole_area = whole.area()
        if whole_area <= 0.0:
            return []
        picked: list = []
        for geom, score in sorted(parts, key=lambda t: -t[0].area()):
            area = geom.area()
            if area <= 0.0 or area > self._part_max_frac * whole_area:
                continue
            inter = geom.intersection(whole)
            if inter is None or inter.isEmpty():
                continue
            if inter.area() / area < self._part_inside:
                continue
            if any(_ios_and_span(geom, p)[0] >= self._part_sibling_ios
                   for p, _ps in picked):
                continue
            picked.append((geom, score))
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
        """Give back the objects a coarse reading swallowed, and return how
        many coarse readings were replaced.

        Call ONCE when every tile is in, never during the run: mid-run the
        parts of a complex have not all arrived, so an early call would judge
        an incomplete set and keep the blob. A no-op unless the merger was
        built with restore_partitions=True.
        """
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
            for geom, score in picked:
                self._insert(QgsGeometry(geom), float(score))
            restored += 1
        self._absorbed.clear()
        return restored

    def result(self) -> list:
        """Current merged objects (one geometry per stitched object)."""
        return [self._keepers[fid] for fid in self._live_ids]

    def result_scored(self) -> list:
        """Current merged objects as (geometry, score) pairs.

        score is the representative (MAX) score of the object's constituent
        fragments, so the review confidence filter acts on WHOLE objects and a
        strong object survives even if one of its seam halves scored low.
        """
        return [
            (self._keepers[fid], self._scores.get(fid, 0.0))
            for fid in self._live_ids
        ]

    def result_scored_ided(self) -> list:
        """Current merged objects as (stable_id, geometry, score) triples.

        stable_id is the keeper's fid, assigned once when the object first
        appears and preserved across later merges (a merge folds a fragment
        INTO an existing keeper, reusing its id; only NEW objects get a fresh
        id). A caller that keys a per-object colour on it keeps an object's hue
        fixed as more tiles arrive, unlike the positional index which reshuffles
        whenever an earlier keeper retires.
        """
        return [
            (fid, self._keepers[fid], self._scores.get(fid, 0.0))
            for fid in self._live_ids
        ]
