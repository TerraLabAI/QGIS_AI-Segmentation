
















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




_FALLBACK_MAX_OBJECTS = 20000




_UNION_STEP_VERTICES = 20000



_ASSIGN_CLIP_BANDS = 4.0


def _union_step_vertices() -> int:



    from .server_dials import dial_in_range

    return int(dial_in_range(
        "tuning.review.gap_fill_union_step_vertices", _UNION_STEP_VERTICES,
        2000, 200000))


def _assign_clip_bands() -> float:


    from .server_dials import dial_in_range

    return dial_in_range(
        "tuning.review.gap_fill_assign_clip_bands", _ASSIGN_CLIP_BANDS,
        2.0, 16.0)




_FALLBACK_THREAD_MIN_OBJECTS = 200




_ASSIGN_TIE_MARGIN = 1e-4


def gap_fill_policy(policy: dict | None = None) -> dict:


    section = review_policy(policy).get("gap_fill")
    return section if isinstance(section, dict) else {}


def gap_fill_max_objects(policy: dict | None = None) -> int:

    val = gap_fill_policy(policy).get("max_objects")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        n = int(val)
        if n >= 0:
            return n
    return _FALLBACK_MAX_OBJECTS


def gap_fill_thread_min_objects(policy: dict | None = None) -> int:


    val = gap_fill_policy(policy).get("thread_min_objects")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        n = int(val)
        if n >= 0:
            return n
    return _FALLBACK_THREAD_MIN_OBJECTS


def _polygon_boundary(geom: QgsGeometry) -> QgsGeometry | None:

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


    lengths = sorted((length for _, length in shares), reverse=True)
    if len(lengths) < 2 or lengths[0] <= 0.0:
        return False
    return lengths[1] >= lengths[0] * (1.0 - _ASSIGN_TIE_MARGIN)


def union_interior_rings(union: QgsGeometry) -> list[QgsGeometry]:

    rings: list[QgsGeometry] = []
    if union is None or union.isEmpty():
        return rings
    for part in union.constParts():


        if not isinstance(part, QgsCurvePolygon):
            continue
        for index in range(part.numInteriorRings()):
            polygon = QgsPolygon()
            polygon.setExteriorRing(part.interiorRing(index).clone())
            rings.append(QgsGeometry(polygon))
    return rings


class GapFillPass:










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

        if self._done:
            return True
        count = max(1, int(count))
        try:
            phase = getattr(self, "_step_" + self._phase)
            phase(count)
        except Exception as exc:  # noqa: BLE001
            self._finish(None, f"failed ({exc})")
        return self._done

    def _finish(self, result: list | None, reason: str = "") -> None:
        self._result = result
        self.skip_reason = reason
        self._done = True





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






    def _step_union(self, count: int) -> None:
        level = self._level
        start = self._cursor
        end = min(len(level), start + 2)
        vertices = sum(level[i].constGet().nCoordinates() for i in range(start, end))
        step_budget = _union_step_vertices()
        while end < len(level) and vertices < step_budget:
            vertices += level[end].constGet().nCoordinates()
            end += 1
        chunk = level[start:end]
        merged = chunk[0] if len(chunk) == 1 else QgsGeometry.unaryUnion(chunk)
        if merged is None or merged.isEmpty():
            self._finish(None, "union failed")
            return
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
            near.grow(_assign_clip_bands() * band)
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





    def _step_merge(self, count: int) -> None:
        assert self._result is not None  # nosec B101
        end = min(len(self._subset), self._cursor + count)
        for k in self._subset[self._cursor:end]:
            original = self._geoms[k]
            pieces = self._assigned[k]
            merged = QgsGeometry.unaryUnion([original] + pieces)
            if merged is None or merged.isEmpty() or merged.type() != PolygonGeometry:
                continue



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


def gap_fill_summary(pass_: Any) -> str:

    if pass_.skip_reason:
        return f"gap fill skipped: {pass_.skip_reason}"
    return (f"gap fill closed {pass_.gaps_filled} gap(s) across "
            f"{pass_.objects_changed} shape(s)")
