











from __future__ import annotations

import math
from typing import Any, NamedTuple

from .regularize_edge_pipeline import (
    _ASPECT_IDENTITY_EPSILON,
    _CHANGED_MIN_FRACTION,
    _DEFAULT_CIRCLE_THRESHOLD,
    _DEFAULT_DIAGONAL_REDUCTION,
    _DEFAULT_MIN_KEEP_IOU,
    _DEFAULT_MULTI_DIRECTION,
    _DEFAULT_MULTI_MAX_GROUPS,
    _DEFAULT_MULTI_MIN_SEPARATION_DEG,
    _DESTAIRCASE_NOOP_FRACTION,
    _ENFORCE_ANGLE_TOL_DEG,
    _TIDY_AREA_NOOP_FRACTION,
    RegularizeDials,
    _resolve_regularize_dials,
    calculate_distance,
    regularize_coordinate_array,
)
from .regularize_multi_direction import (
    regularize_coordinate_array_multi,
)
from .regularize_ring_tidy import tidy_squared_ring





_CAP_SQUARE = 3
_JOIN_MITRE = 2



np: Any = None
Polygon: Any = None
MultiPolygon: Any = None
LinearRing: Any = None
_unary_union: Any = None
_affine_transform: Any = None


def _ensure_deps() -> bool:


    global np, Polygon, MultiPolygon, LinearRing, _unary_union
    global _affine_transform
    if np is not None and Polygon is not None:
        return True
    try:
        import numpy as _numpy
        from shapely.affinity import affine_transform as _affine
        from shapely.geometry import LinearRing as _LinearRing
        from shapely.geometry import MultiPolygon as _MultiPolygon
        from shapely.geometry import Polygon as _Polygon
        from shapely.ops import unary_union as _uu
    except Exception:  # noqa: BLE001  # nosec B110
        return False
    MultiPolygon = _MultiPolygon
    LinearRing = _LinearRing
    _unary_union = _uu
    _affine_transform = _affine



    np = _numpy
    Polygon = _Polygon
    return True


def _segmentize(polygon: Any, max_segment_length: float) -> Any:










    method = getattr(polygon, "segmentize", None)
    if method is not None:
        return method(max_segment_length=max_segment_length)
    if not (max_segment_length and max_segment_length > 0.0):
        return polygon

    def densify(coords) -> list:
        pts = [(float(c[0]), float(c[1])) for c in coords]
        out = []
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            out.append((x0, y0))
            pieces = int(math.ceil(math.hypot(x1 - x0, y1 - y0) / max_segment_length))
            for k in range(1, pieces):
                t = k / pieces
                out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
        if pts:
            out.append(pts[-1])
        return out

    return Polygon(densify(polygon.exterior.coords),
                   [densify(ring.coords) for ring in polygon.interiors])




_PROBE: list = []


def _probe_regularizer() -> tuple[bool, str]:








    if _PROBE:
        return _PROBE[0]
    if not _ensure_deps():
        return False, "numpy or shapely does not import"
    try:
        import shapely

        version = str(getattr(shapely, "__version__", "?"))
    except Exception:  # noqa: BLE001
        version = "?"


    angle = math.radians(20.0)
    ca, sa = math.cos(angle), math.sin(angle)
    wobble = [(0.0, 0.0), (5.0, 0.3), (10.0, -0.2), (15.0, 0.25), (20.0, 0.0),
              (20.3, 5.0), (20.0, 10.0), (15.0, 10.25), (10.0, 9.8),
              (5.0, 10.3), (0.0, 10.0), (-0.25, 5.0)]
    ring = [(x * ca - y * sa, x * sa + y * ca) for x, y in wobble]
    try:
        probe = Polygon(ring)
        preprocess_polygon(probe, True, 1.0)
        _result, regularized, _reverted = _regularize_geometry(
            probe, 1.0, True, False, _DEFAULT_MIN_KEEP_IOU)
    except Exception as exc:  # noqa: BLE001
        answer = (False, f"shapely {version}: {type(exc).__name__}: {exc}"[:300])
    else:
        answer = ((True, "") if regularized else
                  (False, f"shapely {version}: the test outline did not square"))
    _PROBE.append(answer)
    return answer


def dependencies_problem() -> str:

    ok, reason = _probe_regularizer()
    return "" if ok else (reason or "unknown")


class RegularizeResult(NamedTuple):








    geometry: Any
    regularized: bool
    reverted: bool


class RegularizePolicy(NamedTuple):





    envelope_enabled: bool = False
    max_area_ratio: float = 0.0
    min_area_ratio: float = 0.0
    max_hausdorff_mult: float = 0.0
    max_vertex_growth: float = 0.0
    enforce_component_count: bool = False
    enforce_hole_count: bool = False

    eligibility_enabled: bool = False
    min_rectangularity: float = 0.0
    max_holes: int = -1

    rectangle_enabled: bool = False
    rectangle_area_fill: float = 0.95
    rectangle_min_aspect: float = 1.2


_NEUTRAL_POLICY = RegularizePolicy()


def iou_and_symmetric_fraction(shape_a: Any, shape_b: Any) -> tuple[float, float]:











    area_a = shape_a.area
    area_b = shape_b.area
    intersection_area = shape_a.intersection(shape_b).area
    union_area = area_a + area_b - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0
    symmetric_fraction = (
        (area_a + area_b - 2.0 * intersection_area) / area_a
        if area_a > 0 else 0.0)
    return iou, symmetric_fraction


def dependencies_available() -> bool:






















    return _probe_regularizer()[0]


def _rectangularity(geom: Any) -> float:


    try:
        hull_area = geom.convex_hull.area
        if hull_area <= 0:
            return 0.0
        return geom.area / hull_area
    except Exception:  # noqa: BLE001  # nosec B110
        return 0.0


def preprocess_polygon(polygon: Any, simplify: bool, simplify_tolerance: float) -> Any:


    if simplify:
        simplified = polygon.simplify(
            tolerance=simplify_tolerance, preserve_topology=True
        )



        if simplified.is_empty:
            return polygon
        if isinstance(simplified, Polygon):
            polygon = simplified
        else:
            return polygon




    return _segmentize(polygon, simplify_tolerance * 5)


def flatten_to_polygons(geometries) -> list[Any]:


    flat: list[Any] = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, MultiPolygon):
            flat.extend(p for p in geom.geoms if not p.is_empty)
        elif isinstance(geom, Polygon):
            flat.append(geom)
    return flat


def regularize_single_polygon(
    polygon: Any,
    parallel_threshold: float,
    allow_45_degree: bool,
    diagonal_threshold_reduction: float,
    allow_circles: bool,
    circle_threshold: float,
    simplify: bool,
    simplify_tolerance: float,
    multi_direction: bool = _DEFAULT_MULTI_DIRECTION,
    multi_max_groups: int = _DEFAULT_MULTI_MAX_GROUPS,
    multi_min_separation_deg: float = _DEFAULT_MULTI_MIN_SEPARATION_DEG,
    dials: RegularizeDials | None = None,
) -> list[Any]:









    resolved_dials = dials or _resolve_regularize_dials()
    if isinstance(polygon, MultiPolygon):
        results: list[Any] = []
        for p in polygon.geoms:
            results.extend(
                regularize_single_polygon(
                    polygon=p,
                    parallel_threshold=parallel_threshold,
                    allow_45_degree=allow_45_degree,
                    diagonal_threshold_reduction=diagonal_threshold_reduction,
                    allow_circles=allow_circles,
                    circle_threshold=circle_threshold,
                    simplify=simplify,
                    simplify_tolerance=simplify_tolerance,
                    multi_direction=multi_direction,
                    multi_max_groups=multi_max_groups,
                    multi_min_separation_deg=multi_min_separation_deg,
                    dials=resolved_dials,
                )
            )
        return results or [polygon]
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        return [polygon]

    simple_polygon = preprocess_polygon(
        polygon, simplify=simplify, simplify_tolerance=simplify_tolerance
    ).buffer(0)

    if isinstance(simple_polygon, MultiPolygon):
        results = []
        for p in simple_polygon.geoms:
            results.extend(
                regularize_single_polygon(
                    polygon=p,
                    parallel_threshold=parallel_threshold,
                    allow_45_degree=allow_45_degree,
                    diagonal_threshold_reduction=diagonal_threshold_reduction,
                    allow_circles=allow_circles,
                    circle_threshold=circle_threshold,
                    simplify=simplify,
                    simplify_tolerance=simplify_tolerance,
                    multi_direction=multi_direction,
                    multi_max_groups=multi_max_groups,
                    multi_min_separation_deg=multi_min_separation_deg,
                    dials=resolved_dials,
                )
            )
        return results or [polygon]

    if not isinstance(simple_polygon, Polygon) or simple_polygon.is_empty:
        return [polygon]

    exterior_coordinates = np.array(simple_polygon.exterior.coords)
    regularized_exterior, _main_direction = _regularize_one_ring(
        coordinates=exterior_coordinates,
        parallel_threshold=parallel_threshold,
        allow_45_degree=allow_45_degree,
        diagonal_threshold_reduction=diagonal_threshold_reduction,
        multi_direction=multi_direction,
        multi_max_groups=multi_max_groups,
        multi_min_separation_deg=multi_min_separation_deg,
        dials=resolved_dials,
    )




    if allow_circles and polygon.area > 0 and not simple_polygon.interiors:
        radius = math.sqrt(polygon.area / math.pi)


        perfect_circle = polygon.centroid.buffer(radius, 42)
        circle_iou, _ = iou_and_symmetric_fraction(perfect_circle, polygon)
        if circle_iou > circle_threshold:
            regularized_exterior = np.array(
                perfect_circle.exterior.coords, dtype=float
            )

    regularized_interiors: list[Any] = []
    for interior in simple_polygon.interiors:
        interior_coordinates = np.array(interior.coords)
        regularized_interior, _ = _regularize_one_ring(
            coordinates=interior_coordinates,
            parallel_threshold=parallel_threshold,
            allow_45_degree=allow_45_degree,
            diagonal_threshold_reduction=diagonal_threshold_reduction,
            multi_direction=multi_direction,
            multi_max_groups=multi_max_groups,
            multi_min_separation_deg=multi_min_separation_deg,
            dials=resolved_dials,
        )
        regularized_interiors.append(regularized_interior)

    try:
        exterior_ring = LinearRing(regularized_exterior)
        interior_rings = [LinearRing(r) for r in regularized_interiors]
        regularized_polygon = Polygon(exterior_ring, interior_rings).buffer(0)
        final_iou, _ = iou_and_symmetric_fraction(regularized_polygon, polygon)



        if final_iou < resolved_dials.ring_min_iou:
            return [polygon]
        pieces = flatten_to_polygons([regularized_polygon])
        return pieces or [polygon]
    except Exception:  # noqa: BLE001  # nosec B110
        return [polygon]


def _regularize_one_ring(
    coordinates: Any,
    parallel_threshold: float,
    allow_45_degree: bool,
    diagonal_threshold_reduction: float,
    multi_direction: bool,
    multi_max_groups: int,
    multi_min_separation_deg: float,
    dials: RegularizeDials | None = None,
) -> tuple[Any, float]:

    angle_tol = dials.enforce_angle_tol_deg if dials is not None else _ENFORCE_ANGLE_TOL_DEG
    if multi_direction:
        return regularize_coordinate_array_multi(
            coordinates=coordinates,
            parallel_threshold=parallel_threshold,
            allow_45_degree=allow_45_degree,
            diagonal_threshold_reduction=diagonal_threshold_reduction,
            max_groups=multi_max_groups,
            min_separation_deg=multi_min_separation_deg,
            angle_enforcement_tolerance=angle_tol,
            dials=dials,
        )
    return regularize_coordinate_array(
        coordinates=coordinates,
        parallel_threshold=parallel_threshold,
        allow_45_degree=allow_45_degree,
        diagonal_threshold_reduction=diagonal_threshold_reduction,
        angle_enforcement_tolerance=angle_tol,
    )


def _regularize_part_local(
    part: Any,
    tolerance: float,
    allow_diagonal: bool,
    diagonal_reduction: float,
    allow_circles: bool,
    circle_threshold: float,
    multi_direction: bool,
    multi_max_groups: int,
    multi_min_separation_deg: float,
    dials: RegularizeDials | None = None,
) -> list[Any]:


















    if tolerance <= 0:
        return regularize_single_polygon(
            polygon=part,
            parallel_threshold=tolerance,
            allow_45_degree=allow_diagonal,
            diagonal_threshold_reduction=diagonal_reduction,
            allow_circles=allow_circles,
            circle_threshold=circle_threshold,
            simplify=True,
            simplify_tolerance=tolerance,
            multi_direction=multi_direction,
            multi_max_groups=multi_max_groups,
            multi_min_separation_deg=multi_min_separation_deg,
            dials=dials,
        )
    centre = part.centroid
    origin_x, origin_y = float(centre.x), float(centre.y)
    scale = 1.0 / float(tolerance)
    into = (scale, 0.0, 0.0, scale, -scale * origin_x, -scale * origin_y)
    back = (tolerance, 0.0, 0.0, tolerance, origin_x, origin_y)
    results = regularize_single_polygon(
        polygon=_affine_transform(part, into),
        parallel_threshold=1.0,
        allow_45_degree=allow_diagonal,
        diagonal_threshold_reduction=diagonal_reduction,
        allow_circles=allow_circles,
        circle_threshold=circle_threshold,
        simplify=True,
        simplify_tolerance=1.0,
        multi_direction=multi_direction,
        multi_max_groups=multi_max_groups,
        multi_min_separation_deg=multi_min_separation_deg,
        dials=dials,
    )
    return [_affine_transform(piece, back) for piece in results or []]


def _tidy_polygon_corners(polygon: Any, tolerance_m: float,
                          dials: RegularizeDials, reference: Any = None) -> Any:






    min_edge = dials.tidy_min_edge_mult * tolerance_m
    chamfer = dials.tidy_chamfer_mult * tolerance_m


    shift = dials.tidy_min_edge_mult * tolerance_m
    if (polygon is None or polygon.is_empty or tolerance_m <= 0
            or (min_edge <= 0 and chamfer <= 0)
            or not isinstance(polygon, Polygon)):
        return polygon
    ref_shell = None
    if isinstance(reference, Polygon) and not reference.is_empty:
        ref_shell = np.asarray(reference.exterior.coords, dtype=float)
    try:
        shell = tidy_squared_ring(
            np.asarray(polygon.exterior.coords, dtype=float), min_edge, chamfer,
            shift, ref_shell, dials.tidy)
        holes = [
            tidy_squared_ring(
                np.asarray(r.coords, dtype=float), min_edge, chamfer, shift,
                tidy=dials.tidy)
            for r in polygon.interiors
        ]
        tidied = Polygon(shell, holes)
        if tidied.is_empty or not tidied.is_valid:
            return polygon
        return _restore_tidied_area(tidied, polygon.area, dials.tidy_area_noop_frac)
    except Exception:  # noqa: BLE001  # nosec B110
        return polygon


def _restore_tidied_area(tidied: Any, area: float,
                         noop_frac: float = _TIDY_AREA_NOOP_FRACTION) -> Any:




    try:
        if area <= 0 or tidied.length <= 0:
            return tidied
        gap = tidied.area - area
        if abs(gap) <= noop_frac * area:
            return tidied
        restored = tidied.buffer(-gap / tidied.length, join_style=_JOIN_MITRE)
        if (not isinstance(restored, Polygon) or restored.is_empty
                or not restored.is_valid
                or len(restored.interiors) != len(tidied.interiors)):
            return tidied
        return restored
    except Exception:  # noqa: BLE001  # nosec B110
        return tidied


def _cleanup_polygon(polygon: Any, simplify_tolerance: float) -> Any:











    if polygon is None or polygon.is_empty or simplify_tolerance <= 0:
        return polygon
    try:





        if (isinstance(polygon, Polygon) and not polygon.interiors
                and len(polygon.exterior.coords) == 5):
            return polygon
        buffer_size = simplify_tolerance / 50.0
        cleaned = polygon.buffer(-buffer_size, cap_style=_CAP_SQUARE, join_style=_JOIN_MITRE)
        cleaned = cleaned.buffer(
            buffer_size * 2, cap_style=_CAP_SQUARE, join_style=_JOIN_MITRE
        )
        cleaned = cleaned.buffer(-buffer_size, cap_style=_CAP_SQUARE, join_style=_JOIN_MITRE)
        if cleaned.is_empty:
            return polygon
        cleaned = cleaned.simplify(tolerance=buffer_size, preserve_topology=True)
        if cleaned.is_empty:
            return polygon
        cleaned_parts, cleaned_holes = _component_and_hole_counts(cleaned)
        original_parts, original_holes = _component_and_hole_counts(polygon)
        if cleaned_parts < original_parts or cleaned_holes < original_holes:
            return polygon
        return cleaned
    except Exception:  # noqa: BLE001  # nosec B110
        return polygon


def _assemble_parts(per_part: list[list[Any]]) -> Any:
















    merged: list[Any] = []
    for bucket in per_part:
        alive = [p for p in bucket if p is not None and not p.is_empty]
        if not alive:
            continue
        one = alive[0] if len(alive) == 1 else _unary_union(alive)
        if one is not None and not one.is_empty:
            merged.append(one)
    if not merged:
        return None
    if len(merged) == 1:
        return merged[0]
    try:
        apart = MultiPolygon(flatten_to_polygons(merged))
        if not apart.is_empty and apart.is_valid:
            return apart
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    joined = _unary_union(merged)




    if joined is None:
        return None
    if _component_count(joined) < sum(_component_count(one) for one in merged):
        return None
    return joined


def _vertex_count(geom: Any) -> int:


    try:
        total = 0
        for part in flatten_to_polygons([geom]):
            total += len(part.exterior.coords)
            for ring in part.interiors:
                total += len(ring.coords)
        return total
    except Exception:  # noqa: BLE001  # nosec B110
        return 0


def _component_count(geom: Any) -> int:

    try:
        return len(flatten_to_polygons([geom]))
    except Exception:  # noqa: BLE001  # nosec B110
        return 0


def _hole_count(geom: Any) -> int:

    try:
        return sum(len(part.interiors) for part in flatten_to_polygons([geom]))
    except Exception:  # noqa: BLE001  # nosec B110
        return 0


def _component_and_hole_counts(geom: Any) -> tuple[int, int]:



    try:
        parts = flatten_to_polygons([geom])
        return len(parts), sum(len(part.interiors) for part in parts)
    except Exception:  # noqa: BLE001  # nosec B110
        return 0, 0


def _ombb_candidate(part: Any, policy: RegularizePolicy) -> Any | None:





    try:
        if len(part.interiors) != 0:
            return None
        mrr = part.minimum_rotated_rectangle
        if not isinstance(mrr, Polygon) or not mrr.is_valid or mrr.area <= 0:
            return None
        area_fill = part.area / mrr.area
        coords = list(mrr.exterior.coords)
        if len(coords) < 5:
            return None
        side_a = calculate_distance(coords[0], coords[1])
        side_b = calculate_distance(coords[1], coords[2])
        short = min(side_a, side_b)
        if short <= 0:
            return None
        aspect = max(side_a, side_b) / short
        if area_fill >= policy.rectangle_area_fill and aspect >= policy.rectangle_min_aspect:
            return mrr
        return None
    except Exception:  # noqa: BLE001  # nosec B110
        return None


def _passes_envelope(
    regularized: Any, original: Any, tolerance_m: float, policy: RegularizePolicy
) -> bool:







    try:
        if not regularized.is_valid:
            return False
        if policy.enforce_component_count and (
            _component_count(regularized) != _component_count(original)
        ):
            return False
        if policy.enforce_hole_count and (
            _hole_count(regularized) != _hole_count(original)
        ):
            return False
        orig_area = original.area
        if orig_area > 0:
            ratio = regularized.area / orig_area
            if policy.max_area_ratio > 0 and ratio > policy.max_area_ratio:
                return False
            if policy.min_area_ratio > 0 and ratio < policy.min_area_ratio:
                return False
        if policy.max_hausdorff_mult > 0 and tolerance_m > 0:
            d = regularized.hausdorff_distance(original)
            if d > policy.max_hausdorff_mult * tolerance_m:
                return False
        if policy.max_vertex_growth > 0:
            ov = _vertex_count(original)
            rv = _vertex_count(regularized)
            if ov > 0 and rv > ov * policy.max_vertex_growth:
                return False
        return True
    except Exception:  # noqa: BLE001  # nosec B110
        return True


def _passes_guards(candidate: Any, original: Any, tolerance_m: float,
                   min_keep_iou: float, policy: RegularizePolicy) -> bool:



    try:
        iou, _sym = iou_and_symmetric_fraction(original, candidate)
        if iou < min_keep_iou:
            return False
        return not policy.envelope_enabled or _passes_envelope(
            candidate, original, tolerance_m, policy)
    except Exception:  # noqa: BLE001  # nosec B110
        return True


def _is_eligible(original: Any, policy: RegularizePolicy) -> bool:


    try:
        if policy.max_holes >= 0 and _hole_count(original) > policy.max_holes:
            return False
        return not (
            policy.min_rectangularity > 0 and _rectangularity(original) < policy.min_rectangularity
        )
    except Exception:  # noqa: BLE001  # nosec B110
        return True


def _destaircase_geometry(geometry: Any, tolerance_m: float) -> Any:



    if geometry is None or tolerance_m <= 0:
        return None
    try:
        simplified = geometry.simplify(tolerance_m, preserve_topology=True)
    except Exception:  # noqa: BLE001  # nosec B110
        return None
    if simplified is None or simplified.is_empty:
        return None
    return simplified


def _rings_already_on_grid(
    parts: list[Any],
    tolerance_m: float,
    allow_diagonal: bool,
    angle_tolerance: float = 0.1,
) -> bool:


















    if not parts:
        return False
    try:
        period = 45.0 if allow_diagonal else 90.0
        for part in parts:
            for ring in (part.exterior, *part.interiors):
                coords = np.asarray(ring.coords, dtype=float)
                if len(coords) < 4:
                    return False
                vectors = coords[1:] - coords[:-1]
                lengths = np.hypot(vectors[:, 0], vectors[:, 1])
                kept = lengths > 1e-9
                if not np.any(kept):
                    return False
                azimuths = np.degrees(
                    np.arctan2(vectors[kept, 1], vectors[kept, 0])
                )

                offsets = np.mod(azimuths - azimuths[0], period)
                gaps = np.minimum(offsets, period - offsets)
                if np.any(gaps > angle_tolerance):
                    return False
        for part in parts:
            simple = part.simplify(tolerance_m, preserve_topology=True)
            if not isinstance(simple, Polygon) or simple.is_empty:
                return False
            if part.area <= 0.0 or part.length <= 0.0:
                return False
            if abs(simple.area - part.area) > _DESTAIRCASE_NOOP_FRACTION * part.area:
                return False
            if abs(simple.length - part.length) > _DESTAIRCASE_NOOP_FRACTION * part.length:
                return False
        return True
    except Exception:  # noqa: BLE001  # nosec B110
        return False


def _regularize_geometry(
    geometry: Any,
    tolerance_m: float,
    allow_diagonal: bool,
    allow_circles: bool,
    min_keep_iou: float,
    diagonal_reduction: float = _DEFAULT_DIAGONAL_REDUCTION,
    circle_threshold: float = _DEFAULT_CIRCLE_THRESHOLD,
    multi_direction: bool = _DEFAULT_MULTI_DIRECTION,
    multi_max_groups: int = _DEFAULT_MULTI_MAX_GROUPS,
    multi_min_separation_deg: float = _DEFAULT_MULTI_MIN_SEPARATION_DEG,
    policy: RegularizePolicy | None = None,
) -> tuple[Any, bool, bool]:













    pol = policy or _NEUTRAL_POLICY
    if geometry is None:
        return geometry, False, False
    try:
        if geometry.is_empty:
            return geometry, False, False
    except Exception:  # noqa: BLE001  # nosec B110
        return geometry, False, False
    if tolerance_m <= 0:
        return geometry, False, False

    original = geometry
    try:
        if not original.is_valid:
            fixed = original.buffer(0)
            if not fixed.is_empty:
                original = fixed
    except Exception:  # noqa: BLE001  # nosec B110
        pass

    parts = flatten_to_polygons([original])
    if not parts:
        return geometry, False, False

    if pol.eligibility_enabled and not _is_eligible(original, pol):


        return geometry, False, True



    dials = _resolve_regularize_dials()




    if _rings_already_on_grid(parts, tolerance_m, allow_diagonal):
        tidied_parts = [_tidy_polygon_corners(p, tolerance_m, dials) for p in parts]
        if all(t is p for t, p in zip(tidied_parts, parts)):
            return geometry, False, False
        try:
            tidied = _assemble_parts([[t] for t in tidied_parts])
            iou, sym_frac = iou_and_symmetric_fraction(original, tidied)
        except Exception:  # noqa: BLE001  # nosec B110
            return geometry, False, False
        if tidied is None or tidied.is_empty or iou < min_keep_iou:
            return geometry, False, False
        return tidied, sym_frac > _CHANGED_MIN_FRACTION, False





    per_part: list[list[Any]] = []
    untidied: list[list[Any]] = []
    for part in parts:
        bucket: list[Any] = []
        plain_bucket: list[Any] = []
        per_part.append(bucket)
        untidied.append(plain_bucket)
        rect = _ombb_candidate(part, pol) if pol.rectangle_enabled else None
        if rect is not None:
            cleaned = _cleanup_polygon(rect, tolerance_m)
            if cleaned is not None and not cleaned.is_empty:
                bucket.append(cleaned)
                plain_bucket.append(cleaned)
            continue
        try:
            results = _regularize_part_local(
                part=part,
                tolerance=tolerance_m,
                allow_diagonal=allow_diagonal,
                diagonal_reduction=diagonal_reduction,
                allow_circles=allow_circles,
                circle_threshold=circle_threshold,
                multi_direction=multi_direction,
                multi_max_groups=multi_max_groups,
                multi_min_separation_deg=multi_min_separation_deg,
                dials=dials,
            )
        except Exception:  # noqa: BLE001  # nosec B110
            results = [part]
        for piece in results:
            plain = _cleanup_polygon(piece, tolerance_m)
            tidied = _tidy_polygon_corners(piece, tolerance_m, dials, part)
            cleaned = plain if tidied is piece else _cleanup_polygon(
                tidied, tolerance_m)
            if cleaned is not None and not cleaned.is_empty:
                bucket.append(cleaned)
            if plain is not None and not plain.is_empty:
                plain_bucket.append(plain)

    if not any(per_part):
        return geometry, False, False

    try:
        regularized = _assemble_parts(per_part)
        if regularized is None or regularized.is_empty:
            return geometry, False, False
    except Exception:  # noqa: BLE001  # nosec B110
        return geometry, False, False



    if not _passes_guards(regularized, original, tolerance_m, min_keep_iou, pol):
        try:
            fallback = _assemble_parts(untidied)
        except Exception:  # noqa: BLE001  # nosec B110
            fallback = None
        if fallback is not None and not fallback.is_empty:
            regularized = fallback



    try:
        iou, sym_frac = iou_and_symmetric_fraction(original, regularized)
        if iou < min_keep_iou:
            if multi_direction:
                destaired = _destaircase_geometry(original, tolerance_m)
                if destaired is not None:
                    return destaired, False, True
            return geometry, False, True
        changed = sym_frac > _CHANGED_MIN_FRACTION
    except Exception:  # noqa: BLE001  # nosec B110
        return geometry, False, False





    if pol.envelope_enabled and not _passes_envelope(
        regularized, original, tolerance_m, pol
    ):
        destaired = _destaircase_geometry(original, tolerance_m)
        if destaired is not None:
            return destaired, False, True
        return geometry, False, True

    return regularized, changed, False


def _regularize_geometry_isotropic(
    geometry: Any,
    tolerance_m: float,
    allow_diagonal: bool,
    allow_circles: bool,
    min_keep_iou: float,
    diagonal_reduction: float = _DEFAULT_DIAGONAL_REDUCTION,
    circle_threshold: float = _DEFAULT_CIRCLE_THRESHOLD,
    multi_direction: bool = _DEFAULT_MULTI_DIRECTION,
    multi_max_groups: int = _DEFAULT_MULTI_MAX_GROUPS,
    multi_min_separation_deg: float = _DEFAULT_MULTI_MIN_SEPARATION_DEG,
    policy: RegularizePolicy | None = None,
    unit_aspect: float = 1.0,
) -> tuple[Any, bool, bool]:


















    if not (unit_aspect and math.isfinite(unit_aspect) and unit_aspect > 0.0):
        unit_aspect = 1.0
    plain = (
        geometry, tolerance_m, allow_diagonal, allow_circles, min_keep_iou,
        diagonal_reduction, circle_threshold,
        multi_direction, multi_max_groups, multi_min_separation_deg,
    )
    if abs(unit_aspect - 1.0) < _ASPECT_IDENTITY_EPSILON or not _ensure_deps():
        return _regularize_geometry(*plain, policy=policy)
    try:
        stretched = _affine_transform(
            geometry, (1.0, 0.0, 0.0, unit_aspect, 0.0, 0.0))
    except Exception:  # noqa: BLE001  # nosec B110
        return _regularize_geometry(*plain, policy=policy)
    result, regularized, reverted = _regularize_geometry(
        stretched, tolerance_m, allow_diagonal, allow_circles, min_keep_iou,
        diagonal_reduction, circle_threshold,
        multi_direction, multi_max_groups, multi_min_separation_deg,
        policy=policy,
    )
    if result is stretched:


        return geometry, regularized, reverted
    try:
        return (
            _affine_transform(result, (1.0, 0.0, 0.0, 1.0 / unit_aspect, 0.0, 0.0)),
            regularized,
            reverted,
        )
    except Exception:  # noqa: BLE001  # nosec B110
        return geometry, False, reverted


def regularize_polygon(
    geometry: Any,
    *,
    tolerance_m: float,
    allow_diagonal: bool,
    allow_circles: bool = False,
    min_keep_iou: float = _DEFAULT_MIN_KEEP_IOU,
    diagonal_reduction: float = _DEFAULT_DIAGONAL_REDUCTION,
    circle_threshold: float = _DEFAULT_CIRCLE_THRESHOLD,
    multi_direction: bool = _DEFAULT_MULTI_DIRECTION,
    multi_max_groups: int = _DEFAULT_MULTI_MAX_GROUPS,
    multi_min_separation_deg: float = _DEFAULT_MULTI_MIN_SEPARATION_DEG,
    policy: RegularizePolicy | None = None,
    unit_aspect: float = 1.0,
) -> Any:








    if not _ensure_deps():
        return geometry
    result, _regularized, _reverted = _regularize_geometry_isotropic(
        geometry, tolerance_m, allow_diagonal, allow_circles, min_keep_iou,
        diagonal_reduction, circle_threshold,
        multi_direction, multi_max_groups, multi_min_separation_deg,
        policy=policy, unit_aspect=unit_aspect,
    )
    return result


def regularize_qgs_geometry_ex(
    geom,
    *,
    tolerance_m: float,
    allow_diagonal: bool,
    allow_circles: bool = False,
    min_keep_iou: float = _DEFAULT_MIN_KEEP_IOU,
    diagonal_reduction: float = _DEFAULT_DIAGONAL_REDUCTION,
    circle_threshold: float = _DEFAULT_CIRCLE_THRESHOLD,
    multi_direction: bool = _DEFAULT_MULTI_DIRECTION,
    multi_max_groups: int = _DEFAULT_MULTI_MAX_GROUPS,
    multi_min_separation_deg: float = _DEFAULT_MULTI_MIN_SEPARATION_DEG,
    policy: RegularizePolicy | None = None,
    unit_aspect: float = 1.0,
) -> RegularizeResult:










    if geom is None:
        return RegularizeResult(geom, False, False)
    try:
        if geom.isEmpty():
            return RegularizeResult(geom, False, False)
    except Exception:  # noqa: BLE001  # nosec B110
        return RegularizeResult(geom, False, False)
    if tolerance_m is None or tolerance_m <= 0 or not _ensure_deps():
        return RegularizeResult(geom, False, False)
    if min_keep_iou <= 0:
        min_keep_iou = _DEFAULT_MIN_KEEP_IOU
    try:
        from shapely import wkb as shapely_wkb
    except Exception:  # noqa: BLE001  # nosec B110
        return RegularizeResult(geom, False, False)
    try:
        shp = shapely_wkb.loads(bytes(geom.asWkb()))
    except Exception:  # noqa: BLE001  # nosec B110
        return RegularizeResult(geom, False, False)
    result_shp, regularized, reverted = _regularize_geometry_isotropic(
        shp, tolerance_m, allow_diagonal, allow_circles, min_keep_iou,
        diagonal_reduction, circle_threshold,
        multi_direction, multi_max_groups, multi_min_separation_deg,
        policy=policy, unit_aspect=unit_aspect,
    )
    if not regularized:


        return RegularizeResult(geom, False, reverted)
    try:
        from qgis.core import QgsGeometry
        from qgis.PyQt.QtCore import QByteArray








        out = QgsGeometry()
        out.fromWkb(QByteArray(result_shp.wkb))
        if out.isEmpty():
            out = QgsGeometry.fromWkt(result_shp.wkt)
        if out is None or out.isEmpty():
            return RegularizeResult(geom, False, reverted)
        return RegularizeResult(out, True, False)
    except Exception:  # noqa: BLE001  # nosec B110
        return RegularizeResult(geom, False, reverted)


def regularize_qgs_geometry(
    geom,
    *,
    tolerance_m: float,
    allow_diagonal: bool,
    allow_circles: bool = False,
    min_keep_iou: float = _DEFAULT_MIN_KEEP_IOU,
    diagonal_reduction: float = _DEFAULT_DIAGONAL_REDUCTION,
    circle_threshold: float = _DEFAULT_CIRCLE_THRESHOLD,
    multi_direction: bool = _DEFAULT_MULTI_DIRECTION,
    multi_max_groups: int = _DEFAULT_MULTI_MAX_GROUPS,
    multi_min_separation_deg: float = _DEFAULT_MULTI_MIN_SEPARATION_DEG,
    policy: RegularizePolicy | None = None,
    unit_aspect: float = 1.0,
):



    return regularize_qgs_geometry_ex(
        geom,
        tolerance_m=tolerance_m,
        allow_diagonal=allow_diagonal,
        allow_circles=allow_circles,
        min_keep_iou=min_keep_iou,
        diagonal_reduction=diagonal_reduction,
        circle_threshold=circle_threshold,
        multi_direction=multi_direction,
        multi_max_groups=multi_max_groups,
        multi_min_separation_deg=multi_min_separation_deg,
        policy=policy,
        unit_aspect=unit_aspect,
    ).geometry
