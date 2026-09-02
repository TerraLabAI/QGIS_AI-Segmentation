
















from __future__ import annotations

from typing import Any

from .boundary_snap import (
    _DEFAULT_CRS,
    _FALLBACK_MAX_AREA_CHANGE,
    _FALLBACK_MIN_KEEP_SHARE,
    SnapResult,
    _axis_safe_tolerance,
    _drop_small_parts,
    _policy_value,
    _run_snap_algorithm,
    _same_place,
    _total_area,
    _unchanged,
    _unchanged_geometry,
    boundary_snap_max_area_change,
    boundary_snap_min_keep_share,
)


class PartitionCutter:









    def __init__(self, geoms: list, tolerance: float) -> None:
        from qgis.core import QgsSpatialIndex

        self._geoms = geoms


        self._min_part_area = tolerance * tolerance

        self._index = QgsSpatialIndex()
        self._kept: list = []
        self._cursor = 0
        self.slivers = 0
        self.failed = False

    def step(self, count: int = 32) -> bool:


        from qgis.core import QgsGeometry

        end = min(self._cursor + max(1, int(count)), len(self._geoms))
        while self._cursor < end:
            geom = self._geoms[self._cursor]
            self._cursor += 1
            current = QgsGeometry(geom)
            if not current.isGeosValid():
                fixed = current.makeValid()
                if fixed is not None and not fixed.isEmpty():
                    current = fixed
            covered = False
            for j in self._index.intersects(current.boundingBox()):
                earlier = self._kept[j - 1]
                if earlier is None or earlier.isEmpty():
                    continue
                if not current.intersects(earlier):
                    continue
                cut = current.difference(earlier)
                if cut is None or cut.isEmpty():
                    covered = True
                    break
                current = cut
            if not covered:
                current, dropped = _drop_small_parts(current, self._min_part_area)
                self.slivers += dropped
                covered = current is None or current.isEmpty()
            if covered:



                current = QgsGeometry(geom)
            self._kept.append(current)
            if not self._index.addFeature(self._cursor, current.boundingBox()):


                self.failed = True
                return True
        return self._cursor >= len(self._geoms)

    def result(self) -> list | None:

        return None if self.failed else self._kept


class BoundarySnapPass:







    def __init__(self, geoms: list, tolerance: float, crs: str | None,
                 cut_to_partition: bool = True) -> None:
        self._geoms = geoms
        self._tolerance = float(tolerance)
        self._crs = crs
        self._cut = bool(cut_to_partition)
        self._stage = "snap"
        self._snapped: list = []
        self._partitioned: list = []
        self._slivers = 0
        self._area_before = 0.0
        self._area_after = 0.0
        self._cursor = 0
        self._cutter: PartitionCutter | None = None
        self._all_unchanged = True
        self._result: SnapResult | None = None



    def step(self, count: int = 32) -> bool:





        try:
            return self._step(max(1, int(count)))
        except Exception as exc:  # noqa: BLE001
            self._finish(_unchanged(self._geoms, f"{type(exc).__name__}: {exc}"))
            return True

    def result(self) -> SnapResult:

        if self._result is None:
            return _unchanged(self._geoms, "pass not finished")
        return self._result

    def _finish(self, result: SnapResult) -> bool:
        self._result = result
        self._stage = "done"
        return True

    def _step(self, count: int) -> bool:
        if self._stage == "done":
            return True
        if self._stage == "snap":
            return self._step_snap()
        if self._stage == "verify_place":
            return self._step_verify_place(count)
        if self._stage == "cut":
            return self._step_cut(count)
        if self._stage == "area":
            return self._step_area(count)
        if self._stage == "keep":
            return self._step_keep(count)
        return self._finish(_unchanged(self._geoms, "unknown stage"))



    def _step_snap(self) -> bool:






        from qgis.core import QgsFeature, QgsGeometry, QgsVectorLayer

        for geom in self._geoms:
            if not isinstance(geom, QgsGeometry) or geom.isEmpty():
                return self._finish(
                    _unchanged(self._geoms, "empty or non-geometry member"))

        self._area_before = _total_area(self._geoms)
        if self._area_before <= 0.0:
            return self._finish(_unchanged(self._geoms, "zero input area"))

        self._tolerance = _axis_safe_tolerance(
            self._tolerance, self._geoms, self._crs)
        if self._tolerance <= 0.0:
            return self._finish(_unchanged(self._geoms, "no tolerance"))




        layer = QgsVectorLayer(
            f"MultiPolygon?crs={self._crs or _DEFAULT_CRS}",
            "boundary_snap", "memory")
        if not layer.isValid():
            return self._finish(
                _unchanged(self._geoms, "working layer not created"))
        feats = []
        for geom in self._geoms:
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry(geom))
            feats.append(feat)
        result = layer.dataProvider().addFeatures(feats)
        added = result[0] if isinstance(result, tuple) else result
        if not added:
            return self._finish(
                _unchanged(self._geoms, "working layer not filled"))
        layer.updateExtents()

        snapped = _run_snap_algorithm(layer, self._tolerance)
        if snapped is None:
            return self._finish(
                _unchanged(self._geoms, "snap algorithm unavailable"))
        if len(snapped) != len(self._geoms):
            return self._finish(
                _unchanged(self._geoms, "snap dropped or added geometries"))
        self._snapped = snapped
        self._stage = "verify_place"
        self._cursor = 0
        return False

    def _step_verify_place(self, count: int) -> bool:






        end = min(self._cursor + count, len(self._geoms))
        while self._cursor < end:
            i = self._cursor
            self._cursor += 1
            if not _same_place(self._geoms[i], self._snapped[i], self._tolerance):
                return self._finish(
                    _unchanged(self._geoms, "output no longer matches its input"))
        if self._cursor < len(self._geoms):
            return False
        if self._cut:
            self._cutter = PartitionCutter(self._snapped, self._tolerance)
            self._stage = "cut"
            return False


        self._partitioned = self._snapped
        self._slivers = 0
        self._stage = "area"
        self._cursor = 0
        return False

    def _step_cut(self, count: int) -> bool:
        cutter = self._cutter
        if cutter is None:
            return self._finish(_unchanged(self._geoms, "partition failed"))
        if not cutter.step(count):
            return False
        partitioned = cutter.result()
        if partitioned is None:
            return self._finish(_unchanged(self._geoms, "partition failed"))
        self._partitioned = partitioned
        self._slivers = cutter.slivers
        self._stage = "area"
        self._cursor = 0
        return False

    def _step_area(self, count: int) -> bool:

        end = min(self._cursor + count, len(self._partitioned))
        while self._cursor < end:
            geom = self._partitioned[self._cursor]
            self._cursor += 1
            try:
                if geom is not None and not geom.isEmpty():
                    self._area_after += float(geom.area())
            except Exception:  # noqa: BLE001  # nosec B112
                continue
        if self._cursor < len(self._partitioned):
            return False
        change = (self._area_after - self._area_before) / self._area_before
        if abs(change) > _policy_value(boundary_snap_max_area_change,
                                       _FALLBACK_MAX_AREA_CHANGE):
            return self._finish(_unchanged(
                self._geoms, f"area moved {change:.4f}, over the limit"))
        self._change = change
        self._keep_share = _policy_value(boundary_snap_min_keep_share,
                                         _FALLBACK_MIN_KEEP_SHARE)
        self._stage = "keep"
        self._cursor = 0
        return False

    def _step_keep(self, count: int) -> bool:





        end = min(self._cursor + count, len(self._geoms))
        while self._cursor < end:
            i = self._cursor
            self._cursor += 1
            src: Any = self._geoms[i]
            out: Any = self._partitioned[i]
            if out.area() < src.area() * self._keep_share:
                return self._finish(
                    _unchanged(self._geoms, "a polygon lost too much of itself"))
            if self._all_unchanged and not _unchanged_geometry(src, out):
                self._all_unchanged = False
        if self._cursor < len(self._geoms):
            return False
        if self._all_unchanged:


            return self._finish(
                _unchanged(self._geoms, "nothing within the tolerance"))
        return self._finish(SnapResult(
            self._partitioned, True, self._area_before, self._area_after,
            self._change, self._slivers, ""))
