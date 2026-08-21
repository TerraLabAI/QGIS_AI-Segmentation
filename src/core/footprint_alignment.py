"""Run-wide footprint alignment for the Automatic pipeline (QGIS geometry).

Runs ONCE per run, at finalize, on the merged whole objects of a run whose
prompt family the server opted in (core.detection_policy.auto_regularize_
settings). Each footprint is simplified, snapped to its dominant edge grid
(90-degree steps, 45-degree edges in a tighter window), and rebuilt with clean
corners; a neighbourhood consensus angle competes with the object's own angle
and the minimum rotated rectangle, so one street aligns as a block instead of
each roof snapping a few degrees apart. Give-up guards revert any object the
snap would distort, so a shape the model traced right is never made worse.

Distinct from core.building_regularizer, which answers the review's Right
angles control per object: this pass is run-wide (it needs every neighbour),
policy-gated per prompt family, and applied to the canonical bases BEFORE the
review opens, so the review and the export both see the aligned shapes.
align_saved_footprint at the bottom is the single-object form of the same
competition, run at a Semi-Auto save with the session's already saved shapes
as the neighbourhood (ui.plugin.manual_save_alignment).

All tolerances are GROUND METRES. The driver builds a per-run metre frame
(run_frame_scale) and does every measure inside it, so a pseudo-Mercator or
geographic run aligns the same way as a metric one. The ring math lives in
core.footprint_ring_math (numpy only); this module owns the QGIS geometry
ops, the candidate competition, and the step-able per-run sweep.

Import this module lazily (at finalize), never at plugin load: it pulls numpy
through the ring math.
"""
from __future__ import annotations

import math
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

from .footprint_ring_math import (
    angle_diff_mod90,
    circle_ring,
    ring_dominant_angle,
    ring_drop_short_edges,
    ring_rebuild_corners,
    ring_snap_segments,
    weighted_circular_mean_mod90,
)

# An angle candidate closer than this to one already tried adds nothing.
_MIN_ANGLE_SPLIT_DEG = 1.0
# The most neighbours one consensus angle averages, nearest first. The
# consensus needs a few agreeing neighbours, not the whole block, and a dense
# parcel run puts hundreds inside the radius of every object.
_CONSENSUS_NEIGHBOUR_CAP = 24
_CIRCLE_SEGMENTS = 32
# Ground-metre span of one degree of latitude, and of one degree of longitude
# at the equator, for the geographic-CRS metre frame.
_METRES_PER_DEG_LAT = 111132.0
_METRES_PER_DEG_LON_EQUATOR = 111320.0


@dataclass(frozen=True)
class AlignmentParams:
    """Compiled per-run dials of the footprint alignment, all ground metres or
    degrees. Built by compile_alignment_params from the served settings dict;
    the one derived field is simplify_tolerance."""

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


def compile_alignment_params(settings: dict, gsd_m: float) -> AlignmentParams:
    """Turn the served auto_regularize settings plus the run's ground pixel
    into the compiled dials. The simplify tolerance follows the pixel between
    the two served bounds, so a fine run keeps detail a coarse one cannot."""
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
    )


def run_frame_scale(rows: list, measurer, crs_authid) -> tuple[float, float] | None:
    """Ground metres per run-CRS unit along x and y, or None (frame unknown).

    ``crs_authid`` is an authid string, a full CRS definition string, or a
    QgsCoordinateReferenceSystem (a session CRS with no authority code only
    round-trips as an object or a definition, never as an authid).

    Geographic CRS: the cos-latitude pair at the run's centroid. Projected
    CRS: one isotropic factor from the geodesic/planar area ratio, sampled
    over a few objects (this is what absorbs the pseudo-Mercator inflation).
    None disables the alignment for the run: guessing a frame would scale
    every metre dial by an unknown factor.
    """
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
        # No measurer to ask: only a CRS that already measures in metres is
        # safe to take at face value.
        if crs.isValid() and _crs_unit_is_metre(crs):
            return 1.0, 1.0
        return None
    except Exception:  # noqa: BLE001 -- frame detection is best-effort
        return None


# --------------------------------------------------------------- geometry ops


def _crs_unit_is_metre(crs: QgsCoordinateReferenceSystem) -> bool:
    """Whether the CRS measures in metres. The enum moved across the supported
    QGIS range, so it is resolved once in qt_compat; unresolvable means False,
    so the caller fails closed."""
    from .qt_compat import DistanceMeters

    if DistanceMeters is None:
        return False
    try:
        return crs.mapUnits() == DistanceMeters
    except (RuntimeError, AttributeError):
        return False


def _geometry_rings(geom: QgsGeometry) -> list[np.ndarray]:
    """The rings of a single-polygon geometry as closed (n, 2) arrays,
    exterior first. Empty list when the geometry is not a plain polygon."""
    polygon = geom.asPolygon()
    if not polygon:
        return []
    out = []
    for ring in polygon:
        if len(ring) < 4:
            continue
        out.append(np.array([[p.x(), p.y()] for p in ring], dtype=float))
    return out


def _geometry_from_rings(rings: list[np.ndarray]) -> QgsGeometry | None:
    """A polygon geometry from closed coordinate rings (exterior first)."""
    if not rings:
        return None
    qgs_rings = []
    for ring in rings:
        qgs_rings.append([QgsPointXY(float(x), float(y)) for x, y in ring])
    geom = QgsGeometry.fromPolygonXY(qgs_rings)
    return geom if geom is not None and not geom.isEmpty() else None


def _scaled_rings(rings: list[np.ndarray], kx: float, ky: float) -> list[np.ndarray]:
    """Rings mapped by (x*kx, y*ky), the run-CRS to metre-frame map (and its
    inverse with the reciprocal factors)."""
    return [ring * np.array([kx, ky]) for ring in rings]


def _polygon_parts(geom: QgsGeometry) -> list:
    """Every polygon part of a geometry as ring lists (asPolygon shape),
    walking one level of geometry collection when repair produced one."""
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
    """How many parts a geometry carries, read off the abstract geometry so no
    coordinate is copied. 1 for a plain polygon; 1 as well when it cannot be
    read, so an unreadable geometry keeps the behaviour it had."""
    try:
        abstract = geom.constGet()
        if abstract is None:
            return 1
        return int(abstract.partCount())
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return 1


def _largest_valid_polygon(geom: QgsGeometry) -> QgsGeometry | None:
    """The geometry repaired and reduced to its largest single polygon part
    (holes kept), or None when nothing usable remains."""
    if geom is None or geom.isEmpty():
        return None
    g = geom
    if not g.isGeosValid():
        g = g.makeValid()
        if g is None or g.isEmpty():
            return None
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


def _exterior_perimeter(rings: list[np.ndarray]) -> float:
    """Length of the exterior ring."""
    if not rings:
        return 0.0
    d = np.diff(rings[0], axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def _geometry_iou(a: QgsGeometry, b: QgsGeometry) -> float:
    """Intersection over union of two polygon geometries (planar)."""
    try:
        inter = a.intersection(b)
        inter_area = inter.area() if inter is not None and not inter.isEmpty() else 0.0
        union = a.area() + b.area() - inter_area
        return inter_area / union if union > 0 else 0.0
    except Exception:  # noqa: BLE001 -- treat a GEOS failure as no overlap
        return 0.0


def _mitred_buffer(geom: QgsGeometry, dist: float) -> QgsGeometry | None:
    """Buffer with a MITRE join so square corners stay square. Enum homes
    differ across the supported QGIS range; resolve defensively and fall back
    to the round-join buffer rather than fail the pass."""
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
    """Azimuth mod 90 of the minimum rotated rectangle (staircase-robust)."""
    try:
        box = geom.orientedMinimumBoundingBox()[0]
        ring = _geometry_rings(box)
        if not ring or len(ring[0]) < 2:
            return None
        d = ring[0][1] - ring[0][0]
        if not np.isfinite(d).all() or (d[0] == 0 and d[1] == 0):
            return None
        return math.degrees(math.atan2(d[1], d[0])) % 90.0
    except Exception:  # noqa: BLE001 -- no rectangle, no candidate
        return None


# ---------------------------------------------------------- per-object passes


def _prepare_footprint(raw: QgsGeometry, params: AlignmentParams):
    """Simplify one repaired footprint and expose its exterior ring.

    Returns (simplified geometry, exterior ring coords) or None when the
    simplify leaves nothing usable (the caller then keeps the raw shape).
    """
    simp = raw.simplify(params.simplify_tolerance)
    simp = _largest_valid_polygon(simp) if simp is not None else None
    if simp is None:
        simp = raw
    rings = _geometry_rings(simp)
    if not rings or len(rings[0]) < 4:
        return None
    if len(rings) > 1:
        # The candidate rebuild carries only the exterior ring, so a footprint
        # with a hole (a courtyard) would come back filled. Keep it raw.
        return None
    return simp, rings[0]


def _restore_footprint_area(raw: QgsGeometry, cand: QgsGeometry,
                            params: AlignmentParams,
                            raw_area: float | None = None) -> QgsGeometry:
    """Counter the systematic shrink from simplify chords and corner rebuilds:
    offset the candidate by the area deficit over its perimeter, mitred so
    corners stay sharp, clamped to the simplify tolerance so it can never
    invent geometry. The corrected shape is kept only when it did not grow
    extra vertices (a mitre blow-up shows up as new corners)."""
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
    """Area-restore one candidate, then measure it against the raw shape.

    ``raw_area`` is the raw shape's area when the caller already measured it
    (the sweep does, once per object), None to measure it here.

    Returns (geometry or None, iou). None means a give-up guard fired: the
    candidate strayed too far from what the model traced, keep the input.
    """
    if raw_area is None:
        raw_area = raw.area()
    cand = _restore_footprint_area(raw, cand, params, raw_area)
    iou = _geometry_iou(raw, cand)
    if iou < params.guard_iou_floor:
        return None, iou
    change = (cand.area() - raw_area) / raw_area if raw_area > 0 else 0.0
    if abs(change) > params.guard_area_ceiling:
        return None, iou
    # IoU and area both pass on a long thin spike, so bound how far any point
    # of the candidate sits from the raw shape: a snap never travels far.
    try:
        limit = max(4.0 * params.simplify_tolerance, params.min_edge_abs_m)
        too_far = limit > 0 and float(raw.hausdorffDistance(cand)) > limit
    except Exception:  # noqa: BLE001 -- an unanswerable distance is no veto
        too_far = False
    if too_far:
        return None, iou
    return cand, iou


def _snap_candidate(coords: np.ndarray, base_deg: float,
                    params: AlignmentParams) -> QgsGeometry | None:
    """One angle candidate: snap the ring to base_deg, rebuild corners, drop
    short edges, repair. None when the rebuild degenerates."""
    lines = ring_snap_segments(
        coords, base_deg, params.ortho_window_deg, params.diag_window_deg)
    ring = ring_rebuild_corners(lines, params.parallel_threshold_m)
    if ring is None:
        return None
    ring = ring_drop_short_edges(
        ring, params.min_edge_abs_m, params.min_edge_rel, params.min_corner_deg)
    geom = _geometry_from_rings([ring])
    if geom is None:
        return None
    return _largest_valid_polygon(geom)


# ----------------------------------------------------------------- the sweep


class FootprintAlignSweep:
    """Step-able run-wide footprint alignment over (fid, geometry, score) rows.

    Three stages, all sliced so the finalize state machine can spread the work
    across event-loop turns: "prepare" (repair, metre frame, simplify, own
    dominant angle per object), "consensus" (one spatial-index pass giving each
    object its neighbourhood angle), "align" (candidate competition plus the
    give-up guards per object). ``step(k)`` advances up to k objects and
    returns True when everything is done; ``result()`` then returns the rows
    in the input order, geometries replaced only where an aligned shape passed
    the guards. Never raises out of step(): a failing object keeps its input
    geometry.
    """

    def __init__(self, rows: list, params: AlignmentParams,
                 frame_scale: tuple[float, float]) -> None:
        self._rows = list(rows)
        self._params = params
        self._kx, self._ky = frame_scale
        self._stage = "prepare" if self._rows else "done"
        self._cursor = 0
        # Per-row prepared state: None, or a dict with the frame geometries.
        self._prepared: list = [None] * len(self._rows)
        self._consensus: list = [None] * len(self._rows)
        self._out: list = list(self._rows)
        self._index: QgsSpatialIndex | None = None
        self.aligned_count = 0
        self.reverted_count = 0
        self.circle_count = 0
        self.skipped_count = 0

    # -- stages --------------------------------------------------------------

    def _prepare_one(self, i: int) -> None:
        """Stage 1 for one row: metre frame, repair, simplify, own angle."""
        _fid, geom, _score = self._rows[i]
        if geom is None or geom.isEmpty():
            return
        # The rebuild carries ONE exterior ring, so a multi-part object would
        # come back as its largest piece with the rest deleted. Refused whole,
        # the way a footprint with a hole is (see _prepare_footprint): the row
        # keeps the geometry it came in with.
        if _polygon_part_count(geom) > 1:
            return
        source = _largest_valid_polygon(geom)
        if source is None:
            return
        rings = _geometry_rings(source)
        if not rings:
            return
        # Scale once and keep the arrays: they are already the metre-frame
        # rings, so the perimeter reads off them instead of converting the
        # rebuilt geometry back to rings.
        scaled = _scaled_rings(rings, self._kx, self._ky)
        # No second repair: source came out of _largest_valid_polygon valid and
        # single-part, and scaling by two positive factors cannot invalidate it.
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
            # Measured once: the guards read it for every candidate, and
            # QgsGeometry.area() walks the ring each time.
            "raw_area": raw_area,
            "simp": simp,
            "coords": coords,
            "own": own,
            "center": (center.x(), center.y()),
            "perimeter": _exterior_perimeter(scaled),
        }

    def _build_neighbour_index(self) -> None:
        """One spatial index over the prepared centroids (metre frame)."""
        index = QgsSpatialIndex()
        for i, prep in enumerate(self._prepared):
            if prep is None:
                continue
            feature = QgsFeature(i)
            feature.setGeometry(QgsGeometry.fromPointXY(
                QgsPointXY(*prep["center"])))
            index.addFeature(feature)
        self._index = index

    def _consensus_one(self, i: int) -> None:
        """Stage 2 for one row: perimeter-weighted circular-mean angle of the
        neighbours within the consensus radius, when enough of them exist."""
        prep = self._prepared[i]
        if prep is None or self._index is None:
            return
        radius = self._params.consensus_radius_m
        cx, cy = prep["center"]
        hits = self._index.intersects(
            QgsRectangle(cx - radius, cy - radius, cx + radius, cy + radius))
        near = []
        for j in hits:
            if j == i:
                continue
            other = self._prepared[j]
            if other is None:
                continue
            ox, oy = other["center"]
            dist = math.hypot(ox - cx, oy - cy)
            if dist > radius:
                continue
            near.append((dist, other["own"][0], other["perimeter"]))
        if len(near) < self._params.consensus_min_neighbours:
            return
        # Nearest first, then capped: the objects next door are the ones the
        # angle should agree with, and the mean over hundreds of them costs the
        # whole finalize slice on a dense run. The cap never cuts below the
        # minimum the dial asks for, or a configuration wanting more agreement
        # would silently get less of it than a run with the shipped value.
        cap = max(_CONSENSUS_NEIGHBOUR_CAP,
                  int(self._params.consensus_min_neighbours))
        if len(near) > cap:
            near.sort(key=lambda item: item[0])
            near = near[:cap]
        self._consensus[i] = weighted_circular_mean_mod90(
            np.asarray([item[1] for item in near]),
            np.asarray([item[2] for item in near]))

    def _align_one(self, i: int) -> None:
        """Stage 3 for one row: circle or angle-candidate competition, guards,
        and the map back from the metre frame into the run CRS."""
        prep = self._prepared[i]
        if prep is None:
            self.skipped_count += 1
            return
        # Nothing reads this row's prepared state after its own align turn (the
        # consensus stage is finished), and it holds two geometries plus a
        # coordinate array per object. Released here so a large run does not
        # carry the whole prepared set to the end of the sweep.
        self._prepared[i] = None
        params = self._params
        raw = prep["raw"]
        raw_area = prep["raw_area"]
        simp = prep["simp"]
        coords = prep["coords"]
        if simp is None or coords is None:
            self.skipped_count += 1
            return

        # Near-circular blobs become circles instead of getting corners.
        simp_rings = _geometry_rings(simp)
        perimeter = _exterior_perimeter(simp_rings)
        area = simp.area()
        circularity = (4 * math.pi * area / (perimeter * perimeter)
                       if perimeter > 0 else 0.0)
        if circularity >= params.circularity_skip:
            center = simp.centroid().asPoint()
            circle = _geometry_from_rings([circle_ring(
                center.x(), center.y(), area, _CIRCLE_SEGMENTS)])
            kept = None
            if circle is not None:
                kept, _iou = _guarded_candidate(raw, circle, params, raw_area)
            if kept is None:
                self.reverted_count += 1
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

        _try_angle(own_angle, False)
        if top_fraction < params.mrr_when_top_below:
            mrr = _mrr_angle_mod90(raw)
            if mrr is not None and angle_diff_mod90(mrr, own_angle) > _MIN_ANGLE_SPLIT_DEG:
                _try_angle(mrr, False)
        if consensus is not None and angle_diff_mod90(consensus, own_angle) > _MIN_ANGLE_SPLIT_DEG:
            _try_angle(consensus, True)

        if not candidates:
            self.reverted_count += 1
            return
        best = max(candidates, key=lambda c: c[1])
        # The consensus candidate wins ties within the margin, so a block
        # aligns whenever alignment costs nothing.
        for cand in candidates:
            if cand[2] and cand[1] >= best[1] - params.consensus_iou_margin:
                best = cand
                break
        self._publish(i, best[0])

    def _publish(self, i: int, frame_geom: QgsGeometry) -> None:
        """Map an accepted metre-frame shape back to the run CRS and store it
        on the output row. A failed map keeps the input row."""
        rings = _geometry_rings(frame_geom)
        back = _geometry_from_rings(
            _scaled_rings(rings, 1.0 / self._kx, 1.0 / self._ky))
        if back is None or back.isEmpty():
            self.reverted_count += 1
            return
        fid, _geom, score = self._rows[i]
        self._out[i] = (fid, back, score)
        self.aligned_count += 1

    # -- driver --------------------------------------------------------------

    def step(self, count: int = 24) -> bool:
        """Advance up to ``count`` objects of the current stage. True = done."""
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
                except Exception:  # noqa: BLE001 -- this object keeps its input
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
        """The rows in input order, aligned geometries swapped in."""
        return list(self._out)


def align_footprints_now(rows: list, params: AlignmentParams,
                         frame_scale: tuple[float, float]) -> tuple[list, FootprintAlignSweep]:
    """Synchronous form of the sweep for the headless and restore paths.
    Returns (rows, finished sweep) so the caller can read the counters."""
    sweep = FootprintAlignSweep(rows, params, frame_scale)
    while not sweep.step(256):
        pass
    return sweep.result(), sweep


# The most neighbours one save-time alignment prepares. Only the pool inside
# the consensus radius counts, nearest first, so a session holding thousands
# of saved shapes still pays a bounded price per save. The consensus needs a
# few agreeing neighbours, not the whole street; every save re-prepares them
# on the GUI thread, so the cap is what bounds the per-save stall.
_SAVE_NEIGHBOUR_CAP = 24


def align_saved_footprint(geom: QgsGeometry, neighbour_geoms: list,
                          settings: dict, gsd_m: float,
                          crs_authid: str, measurer) -> QgsGeometry | None:
    """One footprint through the same candidate competition as the run-wide
    sweep, at save time, with already saved shapes standing in as the
    neighbourhood: the consensus angle exists once enough of them sit inside
    the consensus radius, so each new save snaps to the grid the earlier ones
    agree on.

    ``geom`` and ``neighbour_geoms`` share one CRS (``crs_authid``); the
    metre frame is built from the object itself. Inputs are read, never
    mutated. Returns the aligned geometry, or None when the pass must leave
    the save unchanged: no metre frame, no candidate survived the give-up
    guards, or any failure. Never raises."""
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
            for i, (_dist, other) in enumerate(near[:_SAVE_NEIGHBOUR_CAP])]
        sweep = FootprintAlignSweep(rows, params, scale)
        for i in range(len(rows)):
            try:
                sweep._prepare_one(i)
            except Exception:  # noqa: BLE001 -- a bad neighbour drops out alone
                if i == 0:
                    return None
        sweep._build_neighbour_index()
        sweep._consensus_one(0)
        sweep._align_one(0)
        if sweep.aligned_count != 1:
            return None
        aligned = sweep.result()[0][1]
        if aligned is None or aligned.isEmpty():
            return None
        return aligned
    except Exception:  # noqa: BLE001 -- a failed alignment keeps the save as is
        return None
