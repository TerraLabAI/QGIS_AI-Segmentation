




























from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsRectangle,
    QgsSpatialIndex,
)

from .footprint_neighbour_cache import cached_neighbour_prep, sync_neighbour_cache_limit
from .footprint_ring_math import (
    angle_diff_mod90,
    circle_ring,
    ring_dominant_angle,
    ring_drop_short_edges,
    ring_is_simple,
    ring_rebuild_corners,
    ring_snap_segments,
    weighted_circular_mean_mod90,
)
from .shape_policy_dials import (
    circle_segments,
    consensus_neighbour_cap,
    min_angle_split_deg,
    save_neighbour_cap,
)


_MIN_ANGLE_SPLIT_DEG = 1.0



_CONSENSUS_NEIGHBOUR_CAP = 24
_CIRCLE_SEGMENTS = 32


_METRES_PER_DEG_LAT = 111132.0
_METRES_PER_DEG_LON_EQUATOR = 111320.0


@dataclass(frozen=True)
class AlignmentParams:




    simplify_tolerance: float
    ortho_window_deg: float
    diag_window_deg: float
    min_edge_abs_m: float
    min_edge_rel: float
    min_corner_deg: float
    parallel_threshold_m: float
    circularity_skip: float
    hist_bin_deg: float
    bin_halo: int
    mrr_when_top_below: float
    consensus_radius_m: float
    consensus_min_neighbours: int
    consensus_iou_margin: float
    guard_iou_floor: float
    guard_area_ceiling: float



    revert_to_simplified: bool = False


def compile_alignment_params(settings: dict, gsd_m: float) -> AlignmentParams:



    factor = float(settings["simplify_gsd_factor"])
    lo = float(settings["simplify_min_m"])
    hi = float(settings["simplify_max_m"])
    pixel = float(gsd_m) if gsd_m and gsd_m > 0 else 0.0
    tol = min(hi, max(lo, factor * pixel)) if pixel > 0 else lo
    return AlignmentParams(
        simplify_tolerance=tol,
        ortho_window_deg=float(settings["ortho_window_deg"]),
        diag_window_deg=float(settings["diag_window_deg"]),
        min_edge_abs_m=float(settings["min_edge_abs_m"]),
        min_edge_rel=float(settings["min_edge_rel"]),
        min_corner_deg=float(settings["min_corner_deg"]),
        parallel_threshold_m=float(settings["parallel_threshold_m"]),
        circularity_skip=float(settings["circularity_skip"]),
        hist_bin_deg=float(settings["hist_bin_deg"]),
        bin_halo=int(settings["bin_halo"]),
        mrr_when_top_below=float(settings["mrr_when_top_below"]),
        consensus_radius_m=float(settings["consensus_radius_m"]),
        consensus_min_neighbours=int(settings["consensus_min_neighbours"]),
        consensus_iou_margin=float(settings["consensus_iou_margin"]),
        guard_iou_floor=float(settings["guard_iou_floor"]),
        guard_area_ceiling=float(settings["guard_area_ceiling"]),
        revert_to_simplified=settings.get("revert_to_simplified") is True,
    )


def run_frame_scale(rows: list, measurer, crs_authid) -> tuple[float, float] | None:












    try:
        if isinstance(crs_authid, QgsCoordinateReferenceSystem):
            crs = crs_authid
        else:
            crs = QgsCoordinateReferenceSystem(crs_authid or "")
        if crs.isValid() and crs.isGeographic():
            lats = []
            for _fid, geom, _score in rows[:50]:
                if geom is None or geom.isEmpty():
                    continue
                lats.append(geom.centroid().asPoint().y())
            if not lats:
                return None
            lat = math.radians(sum(lats) / len(lats))
            kx = _METRES_PER_DEG_LON_EQUATOR * math.cos(lat)
            if kx <= 0:
                return None
            return kx, _METRES_PER_DEG_LAT
        ratios = []
        for _fid, geom, _score in rows[:25]:
            if geom is None or geom.isEmpty() or measurer is None:
                continue
            planar = geom.area()
            if planar <= 0:
                continue
            measured = float(measurer.measureArea(geom))
            ratio = measured / planar
            if 1e-6 < ratio < 1e6:
                ratios.append(ratio)
        if ratios:
            ratios.sort()
            k = math.sqrt(ratios[len(ratios) // 2])
            return k, k


        if crs.isValid() and _crs_unit_is_metre(crs):
            return 1.0, 1.0
        return None
    except Exception:  # noqa: BLE001
        return None





def _crs_unit_is_metre(crs: QgsCoordinateReferenceSystem) -> bool:



    from .qt_compat import DistanceMeters

    if DistanceMeters is None:
        return False
    try:
        return crs.mapUnits() == DistanceMeters
    except (RuntimeError, AttributeError):
        return False


def _geometry_rings(geom: QgsGeometry) -> list[np.ndarray]:


    rings = _wkb_polygon_rings(geom)
    if rings is not None:
        return rings
    polygon = geom.asPolygon()
    if not polygon:
        return []
    out = []
    for ring in polygon:
        if len(ring) < 4:
            continue
        out.append(np.array([[p.x(), p.y()] for p in ring], dtype=float))
    return out


def _wkb_polygon_rings(geom: QgsGeometry) -> list[np.ndarray] | None:







    try:
        buf = bytes(geom.asWkb())
    except (RuntimeError, AttributeError, TypeError):
        return None
    if len(buf) < 9 or buf[0] != 1 or int.from_bytes(buf[1:5], "little") != 3:
        return None
    ring_count = int.from_bytes(buf[5:9], "little")
    offset = 9
    out = []
    for _ in range(ring_count):
        if offset + 4 > len(buf):
            return None
        count = int.from_bytes(buf[offset:offset + 4], "little")
        offset += 4
        end = offset + 16 * count
        if end > len(buf):
            return None
        if count >= 4:
            out.append(np.frombuffer(buf, dtype="<f8", count=2 * count,
                                     offset=offset).reshape(count, 2).astype(float))
        offset = end
    return out


def _geometry_from_rings(rings: list[np.ndarray]) -> QgsGeometry | None:

    if not rings:
        return None
    geom = _geometry_from_closed_rings(rings)
    if geom is not None:
        return geom if not geom.isEmpty() else None
    qgs_rings = []
    for ring in rings:


        qgs_rings.append([QgsPointXY(x, y)
                          for x, y in np.asarray(ring, dtype=float).tolist()])
    geom = QgsGeometry.fromPolygonXY(qgs_rings)
    return geom if geom is not None and not geom.isEmpty() else None


def _geometry_from_closed_rings(rings: list[np.ndarray]) -> QgsGeometry | None:






    chunks = [struct.pack("<BII", 1, 3, len(rings))]
    for ring in rings:
        arr = np.asarray(ring, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 4:
            return None
        if not (arr[0, 0] == arr[-1, 0] and arr[0, 1] == arr[-1, 1]):
            return None
        chunks.append(struct.pack("<I", len(arr)))
        chunks.append(np.ascontiguousarray(arr, dtype="<f8").tobytes())
    geom = QgsGeometry()
    try:
        geom.fromWkb(b"".join(chunks))
    except (RuntimeError, TypeError, ValueError):
        return None
    return geom


def _scaled_rings(rings: list[np.ndarray], kx: float, ky: float) -> list[np.ndarray]:


    return [ring * np.array([kx, ky]) for ring in rings]


def _polygon_parts(geom: QgsGeometry) -> list:


    if geom.isMultipart():
        parts = geom.asMultiPolygon()
        if parts:
            return parts
    else:
        rings = geom.asPolygon()
        if rings:
            return [rings]
    parts = []
    for piece in geom.asGeometryCollection():
        if piece.isMultipart():
            parts.extend(piece.asMultiPolygon() or [])
        else:
            rings = piece.asPolygon()
            if rings:
                parts.append(rings)
    return parts


def _polygon_part_count(geom: QgsGeometry) -> int:



    try:
        abstract = geom.constGet()
        if abstract is None:
            return 1
        return int(abstract.partCount())
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return 1


def _largest_valid_polygon(geom: QgsGeometry) -> QgsGeometry | None:


    if geom is None or geom.isEmpty():
        return None
    g = geom
    if not g.isGeosValid():
        g = g.makeValid()
        if g is None or g.isEmpty():
            return None
    if _is_closed_flat_polygon(g):



        return g if g.area() > 0.0 else None
    best = None
    best_area = 0.0
    for rings in _polygon_parts(g):
        part = QgsGeometry.fromPolygonXY(rings)
        if part is None or part.isEmpty():
            continue
        area = part.area()
        if area > best_area:
            best, best_area = part, area
    return best


def _is_closed_flat_polygon(geom: QgsGeometry) -> bool:




    try:
        kind = geom.wkbType()
        if int(getattr(kind, "value", kind)) != 3:
            return False
        polygon = geom.constGet()
        rings = [polygon.exteriorRing()] + [
            polygon.interiorRing(i) for i in range(polygon.numInteriorRings())]
        return all(ring is not None and ring.isClosed() for ring in rings)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return False


def _exterior_perimeter(rings: list[np.ndarray]) -> float:

    if not rings:
        return 0.0
    ring = rings[0]
    d = ring[1:] - ring[:-1]
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def _geometry_iou(a: QgsGeometry, b: QgsGeometry,
                  a_area: float | None = None) -> float:



    try:
        inter = a.intersection(b)
        inter_area = inter.area() if inter is not None and not inter.isEmpty() else 0.0
        if a_area is None:
            a_area = a.area()
        union = a_area + b.area() - inter_area
        return inter_area / union if union > 0 else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _mitred_buffer(geom: QgsGeometry, dist: float) -> QgsGeometry | None:



    cap = getattr(getattr(Qgis, "EndCapStyle", None), "Round", None)
    join = getattr(getattr(Qgis, "JoinStyle", None), "Miter", None)
    if cap is None or join is None:
        cap = getattr(QgsGeometry, "CapRound", None)
        join = getattr(QgsGeometry, "JoinStyleMiter", None)
    if cap is not None and join is not None:
        try:
            return geom.buffer(dist, 8, cap, join, 4.0)
        except (TypeError, AttributeError):
            pass
    return geom.buffer(dist, 8)


def _mrr_angle_mod90(geom: QgsGeometry) -> float | None:

    try:
        box = geom.orientedMinimumBoundingBox()[0]
        ring = _geometry_rings(box)
        if not ring or len(ring[0]) < 2:
            return None
        d = ring[0][1] - ring[0][0]
        if not np.isfinite(d).all() or (d[0] == 0 and d[1] == 0):
            return None
        return math.degrees(math.atan2(d[1], d[0])) % 90.0
    except Exception:  # noqa: BLE001
        return None





def _prepare_footprint(raw: QgsGeometry, params: AlignmentParams):





    simp = raw.simplify(params.simplify_tolerance)
    simp = _largest_valid_polygon(simp) if simp is not None else None
    if simp is None:
        simp = raw
    rings = _geometry_rings(simp)
    if not rings or len(rings[0]) < 4:
        return None
    if len(rings) > 1:


        return None
    return simp, rings[0]


def _restore_footprint_area(raw: QgsGeometry, cand: QgsGeometry,
                            params: AlignmentParams,
                            raw_area: float | None = None) -> QgsGeometry:





    cand_rings = _geometry_rings(cand)
    perimeter = _exterior_perimeter(cand_rings)
    if perimeter <= 0:
        return cand
    if raw_area is None:
        raw_area = raw.area()
    delta = (raw_area - cand.area()) / perimeter
    tol = params.simplify_tolerance
    delta = max(-tol, min(tol, delta))
    if abs(delta) < 0.01:
        return cand
    fixed = _mitred_buffer(cand, delta)
    fixed = _largest_valid_polygon(fixed) if fixed is not None else None
    if fixed is None:
        return cand
    fixed_rings = _geometry_rings(fixed)
    if not fixed_rings or not cand_rings:
        return cand
    if len(fixed_rings[0]) > len(cand_rings[0]) + 2:
        return cand
    return fixed


def _guarded_candidate(raw: QgsGeometry, cand: QgsGeometry,
                       params: AlignmentParams,
                       raw_area: float | None = None):








    if raw_area is None:
        raw_area = raw.area()
    cand = _restore_footprint_area(raw, cand, params, raw_area)
    iou = _geometry_iou(raw, cand, raw_area)
    if iou < params.guard_iou_floor:
        return None, iou
    change = (cand.area() - raw_area) / raw_area if raw_area > 0 else 0.0
    if abs(change) > params.guard_area_ceiling:
        return None, iou


    try:
        limit = max(4.0 * params.simplify_tolerance, params.min_edge_abs_m)
        too_far = limit > 0 and float(raw.hausdorffDistance(cand)) > limit
    except Exception:  # noqa: BLE001
        too_far = False
    if too_far:
        return None, iou
    return cand, iou


def _snap_candidate(coords: np.ndarray, base_deg: float,
                    params: AlignmentParams) -> QgsGeometry | None:






    lines = ring_snap_segments(
        coords, base_deg, params.ortho_window_deg, params.diag_window_deg)
    rebuilt = ring_rebuild_corners(lines, params.parallel_threshold_m)
    if rebuilt is None:
        return None
    ring = ring_drop_short_edges(
        rebuilt, params.min_edge_abs_m, params.min_edge_rel, params.min_corner_deg)


    unchanged = len(ring) == len(rebuilt) and np.array_equal(ring, rebuilt)
    if not unchanged and not ring_is_simple(ring):
        return None
    geom = _geometry_from_rings([ring])
    if geom is None:
        return None
    return _largest_valid_polygon(geom)





class FootprintAlignSweep:














    def __init__(self, rows: list, params: AlignmentParams,
                 frame_scale: tuple[float, float]) -> None:
        self._rows = list(rows)
        self._params = params
        self._kx, self._ky = frame_scale
        self._stage = "prepare" if self._rows else "done"
        self._cursor = 0

        self._prepared: list = [None] * len(self._rows)
        self._consensus: list = [None] * len(self._rows)
        self._out: list = list(self._rows)
        self._index: QgsSpatialIndex | None = None


        self._neighbours: list = []
        self.aligned_count = 0
        self.reverted_count = 0
        self.circle_count = 0
        self.skipped_count = 0


        self.simplified_count = 0


        self.changed_fids: set = set()



    def _prepare_one(self, i: int) -> None:

        _fid, geom, _score = self._rows[i]
        if geom is None or geom.isEmpty():
            return




        if _polygon_part_count(geom) > 1:
            return
        source = _largest_valid_polygon(geom)
        if source is None:
            return
        rings = _geometry_rings(source)
        if not rings:
            return



        scaled = _scaled_rings(rings, self._kx, self._ky)


        raw = _geometry_from_rings(scaled)
        raw_area = raw.area() if raw is not None else 0.0
        if raw is None or raw_area <= 0:
            return
        prep = _prepare_footprint(raw, self._params)
        own = (0.0, 0.0)
        simp = None
        coords = None
        if prep is not None:
            simp, coords = prep
            own = ring_dominant_angle(
                coords, self._params.hist_bin_deg, self._params.bin_halo)
        center = raw.centroid().asPoint()
        self._prepared[i] = {
            "raw": raw,


            "raw_area": raw_area,
            "simp": simp,
            "coords": coords,
            "own": own,
            "center": (center.x(), center.y()),
            "perimeter": _exterior_perimeter(scaled),
        }

    def _build_neighbour_index(self) -> None:

        index = QgsSpatialIndex()
        self._neighbours = [
            None if prep is None else (*prep["center"], prep["own"][0], prep["perimeter"])
            for prep in self._prepared]
        for i, prep in enumerate(self._prepared):
            if prep is None:
                continue
            feature = QgsFeature(i)
            feature.setGeometry(QgsGeometry.fromPointXY(
                QgsPointXY(*prep["center"])))
            index.addFeature(feature)
        self._index = index

    def _consensus_one(self, i: int) -> None:


        prep = self._prepared[i]
        if prep is None or self._index is None:
            return
        radius = self._params.consensus_radius_m
        cx, cy = prep["center"]
        hits = self._index.intersects(
            QgsRectangle(cx - radius, cy - radius, cx + radius, cy + radius))
        near = []
        neighbours = self._neighbours
        for j in hits:
            if j == i:
                continue
            other = neighbours[j]
            if other is None:
                continue
            ox, oy, angle, perimeter = other
            dist = math.hypot(ox - cx, oy - cy)
            if dist > radius:
                continue
            near.append((dist, angle, perimeter))
        if len(near) < self._params.consensus_min_neighbours:
            return





        cap = max(consensus_neighbour_cap(_CONSENSUS_NEIGHBOUR_CAP),
                  int(self._params.consensus_min_neighbours))
        if len(near) > cap:
            near.sort(key=lambda item: item[0])
            near = near[:cap]
        self._consensus[i] = weighted_circular_mean_mod90(
            np.asarray([item[1] for item in near]),
            np.asarray([item[2] for item in near]))

    def _align_one(self, i: int) -> None:


        prep = self._prepared[i]
        if prep is None:
            self.skipped_count += 1
            return




        self._prepared[i] = None
        params = self._params
        raw = prep["raw"]
        raw_area = prep["raw_area"]
        simp = prep["simp"]
        coords = prep["coords"]
        if simp is None or coords is None:
            self.skipped_count += 1
            return





        perimeter = _exterior_perimeter([coords])
        area = simp.area()
        circularity = (4 * math.pi * area / (perimeter * perimeter)
                       if perimeter > 0 else 0.0)
        if circularity >= params.circularity_skip:
            center = simp.centroid().asPoint()
            circle = _geometry_from_rings([circle_ring(
                center.x(), center.y(), area, circle_segments(_CIRCLE_SEGMENTS))])
            kept = None
            if circle is not None:
                kept, _iou = _guarded_candidate(raw, circle, params, raw_area)
            if kept is None:
                self._revert(i, raw, simp, raw_area)
                return
            self._publish(i, kept)
            self.circle_count += 1
            return

        own_angle, top_fraction = prep["own"]
        consensus = self._consensus[i]

        candidates: list[tuple[QgsGeometry, float, bool]] = []

        def _try_angle(base: float, is_consensus: bool) -> None:
            cand = _snap_candidate(coords, base, params)
            if cand is None:
                return
            kept, iou = _guarded_candidate(raw, cand, params, raw_area)
            if kept is not None:
                candidates.append((kept, iou, is_consensus))

        split_deg = min_angle_split_deg(_MIN_ANGLE_SPLIT_DEG)
        _try_angle(own_angle, False)
        if top_fraction < params.mrr_when_top_below:
            mrr = _mrr_angle_mod90(raw)
            if mrr is not None and angle_diff_mod90(mrr, own_angle) > split_deg:
                _try_angle(mrr, False)
        if consensus is not None and angle_diff_mod90(consensus, own_angle) > split_deg:
            _try_angle(consensus, True)

        if not candidates:
            self._revert(i, raw, simp, raw_area)
            return
        best = max(candidates, key=lambda c: c[1])


        for cand in candidates:
            if cand[2] and cand[1] >= best[1] - params.consensus_iou_margin:
                best = cand
                break
        self._publish(i, best[0])

    def _revert(self, i: int, raw: QgsGeometry, simp: QgsGeometry | None,
                raw_area: float) -> None:






        self.reverted_count += 1
        if not self._params.revert_to_simplified or simp is None:
            return
        kept, _iou = _guarded_candidate(raw, simp, self._params, raw_area)
        if kept is not None and self._store_frame_geom(i, kept):
            self.simplified_count += 1

    def _store_frame_geom(self, i: int, frame_geom: QgsGeometry) -> bool:


        rings = _geometry_rings(frame_geom)
        back = _geometry_from_rings(
            _scaled_rings(rings, 1.0 / self._kx, 1.0 / self._ky))
        if back is None or back.isEmpty():
            return False
        fid, _geom, score = self._rows[i]
        self._out[i] = (fid, back, score)
        self.changed_fids.add(fid)
        return True

    def _publish(self, i: int, frame_geom: QgsGeometry) -> None:


        if not self._store_frame_geom(i, frame_geom):
            self.reverted_count += 1
            return
        self.aligned_count += 1



    def step(self, count: int = 24) -> bool:

        handlers = {
            "prepare": self._prepare_one,
            "consensus": self._consensus_one,
            "align": self._align_one,
        }
        while count > 0 and self._stage != "done":
            handler = handlers[self._stage]
            while count > 0 and self._cursor < len(self._rows):
                i = self._cursor
                self._cursor += 1
                count -= 1
                try:
                    handler(i)
                except Exception:  # noqa: BLE001
                    if self._stage == "align":
                        self.skipped_count += 1
            if self._cursor >= len(self._rows):
                if self._stage == "prepare":
                    self._build_neighbour_index()
                    self._stage = "consensus"
                elif self._stage == "consensus":
                    self._stage = "align"
                else:
                    self._stage = "done"
                self._cursor = 0
        return self._stage == "done"

    def result(self) -> list:

        return list(self._out)








_SAVE_NEIGHBOUR_CAP = 24


def align_saved_footprint(geom: QgsGeometry, neighbour_geoms: list,
                          settings: dict, gsd_m: float,
                          crs_authid: str, measurer) -> QgsGeometry | None:











    try:
        if geom is None or geom.isEmpty():
            return None
        params = compile_alignment_params(settings, gsd_m)
        target_row = (0, geom, None)
        scale = run_frame_scale([target_row], measurer, crs_authid)
        if scale is None:
            return None
        kx, ky = scale
        centre = geom.boundingBox().center()
        cx, cy = centre.x() * kx, centre.y() * ky
        near = []
        for other in neighbour_geoms:
            if other is None or other.isEmpty():
                continue
            oc = other.boundingBox().center()
            dist = math.hypot(oc.x() * kx - cx, oc.y() * ky - cy)
            if dist <= params.consensus_radius_m:
                near.append((dist, other))
        near.sort(key=lambda item: item[0])
        rows = [target_row] + [
            (i + 1, other, None)
            for i, (_dist, other) in enumerate(near[:save_neighbour_cap(_SAVE_NEIGHBOUR_CAP)])]
        sweep = FootprintAlignSweep(rows, params, scale)
        sync_neighbour_cache_limit()
        try:
            sweep._prepare_one(0)
        except Exception:  # noqa: BLE001
            return None
        for i in range(1, len(rows)):
            try:
                sweep._prepared[i] = cached_neighbour_prep(
                    sweep, i, params, scale)
            except Exception:  # noqa: BLE001
                sweep._prepared[i] = None
        sweep._build_neighbour_index()
        sweep._consensus_one(0)
        sweep._align_one(0)
        if sweep.aligned_count != 1:
            return None
        aligned = sweep.result()[0][1]
        if aligned is None or aligned.isEmpty():
            return None
        return aligned
    except Exception:  # noqa: BLE001
        return None
