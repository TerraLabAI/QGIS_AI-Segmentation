"""Fill the gaps a set of touching objects encloses between them.

Fill holes drops an object's own interior rings. A gap enclosed by several
neighbours, or by the parts of one object, is nobody's ring, so the per-object
pass never sees it and it survives whatever the size cutoff says. On a field
run those gaps are what the user points at: a sliver where two parcels meet, a
pocket at the junction of three. This pass runs once over the assembled
visible set. It unions the objects that touch a neighbour, takes the interior
rings of that union under the same cutoff, and hands each gap to the object it
shares the longest border with. An object that encloses no gap keeps its
identity, so nothing outside the affected ones moves.

Stepped, under the protocol the finalize pump drives (``step(count)`` until it
answers True, then ``result()``), because the union of a large set costs
seconds and this is the last thing between the user and the review. Pure
QGIS-core geometry, no Qt widgets, so it runs anywhere the refine does.
"""
from __future__ import annotations

import math
from typing import Any

from qgis.core import (
    QgsCurvePolygon,
    QgsFeature,
    QgsGeometry,
    QgsPolygon,
    QgsRectangle,
    QgsSpatialIndex,
)

from .detection_policy import review_policy
from .qt_compat import PolygonGeometry

# How many touching shapes one pass may take. Above it the gaps stay, the same
# way shared borders declines a set over its own limit: the union of a set this
# size costs more than the review can spend between two slider moves.
_FALLBACK_MAX_OBJECTS = 20000

# Vertices one union step takes in. Bounds how long a single step holds the
# interface, and fixes the cascade so the union comes out the same whatever
# slice size the caller steps with.
_UNION_STEP_VERTICES = 20000

# How far past a gap's box its neighbours' outlines are kept before the band
# is built around them, in bands. Past one band the band cannot reach the gap.
_ASSIGN_CLIP_BANDS = 4.0

# Under this many shapes the pass stays on the interface thread: it ends
# within a slice or two, and a thread would only add its own start.
_FALLBACK_THREAD_MIN_OBJECTS = 200

# Two shared-border lengths closer than this (relative) count as a tie and are
# measured again on the whole outlines. Far above what trimming moves a
# length by, far below any real difference between two borders.
_ASSIGN_TIE_MARGIN = 1e-4


def gap_fill_policy(policy: dict | None = None) -> dict:
    """The ``review.gap_fill`` sub-policy. Empty dict when absent, so each
    reader falls open to its generic client value."""
    section = review_policy(policy).get("gap_fill")
    return section if isinstance(section, dict) else {}


def gap_fill_max_objects(policy: dict | None = None) -> int:
    """How many touching shapes one pass may union (0 = no limit)."""
    val = gap_fill_policy(policy).get("max_objects")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        n = int(val)
        if n >= 0:
            return n
    return _FALLBACK_MAX_OBJECTS


def gap_fill_thread_min_objects(policy: dict | None = None) -> int:
    """From how many shapes the pass runs on its own thread instead of the
    interface thread (0 = always). Served as ``review.gap_fill.thread_min_objects``."""
    val = gap_fill_policy(policy).get("thread_min_objects")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        n = int(val)
        if n >= 0:
            return n
    return _FALLBACK_THREAD_MIN_OBJECTS


def _polygon_boundary(geom: QgsGeometry) -> QgsGeometry | None:
    """The outline of ``geom`` as a line geometry, or None."""
    try:
        boundary = geom.constGet().boundary()
    except (AttributeError, RuntimeError):
        return None
    if boundary is None:
        return None
    line = QgsGeometry(boundary)
    return None if line.isEmpty() else line


def _shared_border_length(gap_line: QgsGeometry, outline: QgsGeometry,
                          band: float, near: QgsRectangle | None) -> float:
    """Length of ``gap_line`` inside a ``band`` around ``outline``, the outline
    first trimmed to ``near`` when a rectangle is given. 0.0 when they never
    meet or the geometry cannot be read."""
    try:
        if near is not None:
            outline = outline.clipped(near)
            if outline is None or outline.isEmpty():
                return 0.0
        shared = gap_line.intersection(outline.buffer(band, 1))
        if shared is None or shared.isEmpty():
            return 0.0
        return float(shared.length())
    except (AttributeError, RuntimeError):
        return 0.0


def _shares_too_close(shares: list[tuple[int, float]]) -> bool:
    """Whether the two longest shares are within _ASSIGN_TIE_MARGIN of each
    other, so the trimmed measure cannot be trusted to order them."""
    lengths = sorted((length for _, length in shares), reverse=True)
    if len(lengths) < 2 or lengths[0] <= 0.0:
        return False
    return lengths[1] >= lengths[0] * (1.0 - _ASSIGN_TIE_MARGIN)


def union_interior_rings(union: QgsGeometry) -> list[QgsGeometry]:
    """Every interior ring of ``union`` as its own polygon, in ring order."""
    rings: list[QgsGeometry] = []
    if union is None or union.isEmpty():
        return rings
    for part in union.constParts():
        # QgsPolygon derives from QgsCurvePolygon, so one test covers both,
        # with no enum access for the Qt6 checker to flag.
        if not isinstance(part, QgsCurvePolygon):
            continue
        for index in range(part.numInteriorRings()):
            polygon = QgsPolygon()
            polygon.setExteriorRing(part.interiorRing(index).clone())
            rings.append(QgsGeometry(polygon))
    return rings


class GapFillPass:
    """One gap fill over a list of geometries, stepped.

    ``max_area`` is the cutoff in CRS UNITS SQUARED: an enclosed gap under it
    is filled, a bigger one stays. 0 or less fills every enclosed gap, which is
    what the control's "No limit" means for an object's own rings too.
    ``result()`` answers a list of the same length and order as the input, the
    input geometry itself wherever nothing changed, or None when the pass
    declined the set (over ``max_objects``) or failed.
    """

    def __init__(self, geoms: list, max_area: float,
                 max_objects: int | None = None) -> None:
        self._geoms = list(geoms)
        self._max_area = float(max_area) if max_area and max_area > 0 else 0.0
        self._max_objects = (gap_fill_max_objects() if max_objects is None
                             else int(max_objects))
        self._phase = "touch"
        self._cursor = 0
        self._index: QgsSpatialIndex | None = None
        self._touching: set[int] = set()
        self._subset: list[int] = []
        self._level: list[QgsGeometry] = []
        self._next_level: list[QgsGeometry] = []
        self._union: QgsGeometry | None = None
        self._gaps: list[QgsGeometry] = []
        self._assigned: dict[int, list[QgsGeometry]] = {}
        self._result: list | None = None
        self._done = False
        self.gaps_filled = 0
        self.objects_changed = 0
        self.merges_rejected = 0
        self.skip_reason = ""

    def result(self) -> list | None:
        return self._result if self._done else None

    def step(self, count: int = 64) -> bool:
        """Do one bounded slice of work. True when the pass has finished."""
        if self._done:
            return True
        count = max(1, int(count))
        try:
            phase = getattr(self, "_step_" + self._phase)
            phase(count)
        except Exception as exc:  # noqa: BLE001 -- keep the input shapes
            self._finish(None, f"failed ({exc})")
        return self._done

    def _finish(self, result: list | None, reason: str = "") -> None:
        self._result = result
        self.skip_reason = reason
        self._done = True

    # Phase 1: which objects touch a neighbour. Only those can enclose a gap
    # between them; an isolated object's holes are its own rings and the
    # per-object fill has already dealt with them. A multipart object counts
    # as touching itself, since its parts may enclose a gap on their own.
    def _step_touch(self, count: int) -> None:
        geoms = self._geoms
        if self._index is None:
            self._index = QgsSpatialIndex()
            for k, geom in enumerate(geoms):
                if geom is None or geom.isEmpty():
                    continue
                feature = QgsFeature(k)
                feature.setGeometry(geom)
                self._index.addFeature(feature)
        end = min(len(geoms), self._cursor + count)
        for k in range(self._cursor, end):
            geom = geoms[k]
            if geom is None or geom.isEmpty():
                continue
            if geom.constGet().partCount() > 1:
                self._touching.add(k)
            for j in self._index.intersects(geom.boundingBox()):
                if j <= k:
                    continue
                if geom.intersects(geoms[j]):
                    self._touching.add(k)
                    self._touching.add(j)
        self._cursor = end
        if end < len(geoms):
            return
        self._subset = sorted(self._touching)
        if not self._subset:
            self._finish(list(geoms))
            return
        if 0 < self._max_objects < len(self._subset):
            self._finish(None, f"{len(self._subset)} touching shapes over the "
                               f"{self._max_objects} limit")
            return
        self._level = [geoms[k] for k in self._subset]
        self._cursor = 0
        self._phase = "union"

    # Phase 2: a cascaded union of the touching subset, one chunk per step, so
    # the interface is served between chunks instead of waiting on one call.
    # A chunk is cut on a fixed vertex budget, never on the caller's slice
    # size: the cascade decides how the union is noded, and the answer must
    # not depend on how finely a caller stepped it.
    def _step_union(self, count: int) -> None:
        level = self._level
        start = self._cursor
        end = min(len(level), start + 2)
        vertices = sum(level[i].constGet().nCoordinates() for i in range(start, end))
        while end < len(level) and vertices < _UNION_STEP_VERTICES:
            vertices += level[end].constGet().nCoordinates()
            end += 1
        chunk = level[start:end]
        merged = chunk[0] if len(chunk) == 1 else QgsGeometry.unaryUnion(chunk)
        if merged is not None and not merged.isEmpty():
            self._next_level.append(merged)
        self._cursor = end
        if end < len(self._level):
            return
        if len(self._next_level) > 1:
            self._level = self._next_level
            self._next_level = []
            self._cursor = 0
            return
        self._union = self._next_level[0] if self._next_level else None
        self._next_level = []
        self._level = []
        rings = union_interior_rings(self._union) if self._union is not None else []
        self._gaps = [
            ring for ring in rings
            if ring.area() > 0.0
            and (self._max_area <= 0.0 or ring.area() < self._max_area)
        ]
        if not self._gaps:
            self._finish(list(self._geoms))
            return
        self._cursor = 0
        self._phase = "assign"

    # Phase 3: each gap goes to the touching object it shares the longest
    # border with. The border is measured inside a hair-thin band around the
    # object's outline rather than as a line-on-line intersection, because the
    # union renodes its input and a renoded edge no longer meets the original
    # one exactly. Ties break on the lowest index, so the answer is stable.
    #
    # The band is built around the stretch of outline near the gap, not the
    # whole outline: a gap is a few metres wide and a parcel outline runs to
    # thousands of vertices, and the part of the band further from the gap
    # than its own width cannot meet the gap's outline. Trimming the input
    # moves the measure by a hair, so two shares within _ASSIGN_TIE_MARGIN of
    # each other are measured again on the whole outlines, which keeps the
    # answer the untrimmed measure gives.
    def _step_assign(self, count: int) -> None:
        geoms = self._geoms
        index = self._index
        end = min(len(self._gaps), self._cursor + count)
        for gap in self._gaps[self._cursor:end]:
            area = gap.area()
            if area <= 0.0 or index is None:
                continue
            gap_line = _polygon_boundary(gap)
            if gap_line is None:
                continue
            band = 1e-4 * math.sqrt(area)
            box = gap.boundingBox()
            near = QgsRectangle(box)
            near.grow(_ASSIGN_CLIP_BANDS * band)
            candidates = [(k, _polygon_boundary(geoms[k]))
                          for k in sorted(index.intersects(box))]
            shares = [(k, _shared_border_length(gap_line, outline, band, near))
                      for k, outline in candidates if outline is not None]
            if _shares_too_close(shares):
                shares = [(k, _shared_border_length(gap_line, outline, band, None))
                          for k, outline in candidates if outline is not None]
            best = -1
            best_length = 0.0
            for k, length in shares:
                if length > best_length:
                    best, best_length = k, length
            if best >= 0:
                self._assigned.setdefault(best, []).append(gap)
        self._cursor = end
        if end < len(self._gaps):
            return
        if not self._assigned:
            self._finish(list(geoms))
            return
        self._subset = sorted(self._assigned)
        self._result = list(geoms)
        self._cursor = 0
        self._phase = "merge"

    # Phase 4: weld each object to the gaps it was given. The union of a valid
    # polygon with pieces that share its edge is the polygon with those edges
    # dissolved, so the object grows by the gap area and nothing else. A weld
    # that answers anything else keeps the original.
    def _step_merge(self, count: int) -> None:
        assert self._result is not None  # nosec B101
        end = min(len(self._subset), self._cursor + count)
        for k in self._subset[self._cursor:end]:
            original = self._geoms[k]
            pieces = self._assigned[k]
            merged = QgsGeometry.unaryUnion([original] + pieces)
            if merged is None or merged.isEmpty() or merged.type() != PolygonGeometry:
                continue
            # The gaps lie outside the object and apart from each other, so
            # the weld must grow it by their area and nothing else. Tolerance
            # relative to the object: renoding moves an area by a share of it.
            gained = merged.area() - original.area()
            allowed = sum(piece.area() for piece in pieces)
            tolerance = 1e-6 * max(original.area() + allowed, 1e-12)
            if abs(gained - allowed) > tolerance:
                self.merges_rejected += 1
                continue
            if original.isMultipart() and not merged.isMultipart():
                merged.convertToMultiType()
            self._result[k] = merged
            self.objects_changed += 1
            self.gaps_filled += len(pieces)
        self._cursor = end
        if end < len(self._subset):
            return
        self._finish(self._result)


def fill_enclosed_gaps(geoms: list, max_area: float,
                       max_objects: int | None = None) -> list | None:
    """``GapFillPass`` run to the end in one call. See the class for the
    contract; the review's finalize pump steps the class itself."""
    pass_ = GapFillPass(geoms, max_area, max_objects)
    while not pass_.step(256):
        pass
    return pass_.result()


def gap_fill_summary(pass_: Any) -> str:
    """One log line's worth of what a finished pass did."""
    if pass_.skip_reason:
        return f"gap fill skipped: {pass_.skip_reason}"
    return (f"gap fill closed {pass_.gaps_filled} gap(s) across "
            f"{pass_.objects_changed} shape(s)")
