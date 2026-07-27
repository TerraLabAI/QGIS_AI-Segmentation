


















































from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

from .prompt_taxonomy import keyword_matches, normalize_prompt

if TYPE_CHECKING:
    from qgis.core import QgsGeometry



_SNAP_ALG = "native:snapgeometries"
_SNAP_BEHAVIOR_ANCHOR_NODES = 7



SNAP_DEFAULT_ENABLED = False




_FALLBACK_TOLERANCE_M = 0.5
_FALLBACK_MAX_AREA_CHANGE = 0.02
_FALLBACK_MIN_KEEP_SHARE = 0.5



_FALLBACK_MAX_OBJECTS = 2000



_FALLBACK_KEYWORDS: tuple[str, ...] = (
    "land cover",
    "landcover",
    "land use",
    "landuse",
    "parcel",
    "field",
    "crop",
)



_DEFAULT_CRS = "EPSG:3857"


class SnapResult(NamedTuple):











    geometries: list[QgsGeometry]
    snapped: bool
    area_before: float
    area_after: float
    area_change: float
    slivers_removed: int
    reason: str











def boundary_snap_policy(policy: dict | None = None) -> dict:


    from .detection_policy import review_policy

    val = review_policy(policy).get("boundary_snap")
    return val if isinstance(val, dict) else {}


def snap_default_enabled(policy: dict | None = None) -> bool:






    val = boundary_snap_policy(policy).get("default_enabled")
    if isinstance(val, bool):
        return val
    return SNAP_DEFAULT_ENABLED


def boundary_snap_tolerance_m(policy: dict | None = None) -> float:






    val = boundary_snap_policy(policy).get("tolerance_m")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = float(val)
        if 0.0 < f <= 100.0:
            return f
    return _FALLBACK_TOLERANCE_M


def boundary_snap_max_area_change(policy: dict | None = None) -> float:



    val = boundary_snap_policy(policy).get("max_area_change")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = float(val)
        if 0.0 < f <= 1.0:
            return f
    return _FALLBACK_MAX_AREA_CHANGE


def boundary_snap_min_keep_share(policy: dict | None = None) -> float:





    val = boundary_snap_policy(policy).get("min_keep_share")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = float(val)
        if 0.0 < f <= 1.0:
            return f
    return _FALLBACK_MIN_KEEP_SHARE


def boundary_snap_max_objects(policy: dict | None = None) -> int:






    val = boundary_snap_policy(policy).get("max_objects")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        n = int(val)
        if n >= 0:
            return n
    return _FALLBACK_MAX_OBJECTS


def boundary_snap_tolerance_units(metres_per_unit: float,
                                  policy: dict | None = None) -> float:












    tol_m = boundary_snap_tolerance_m(policy)
    try:
        factor = float(metres_per_unit)
    except (TypeError, ValueError):
        return tol_m
    if factor <= 0.0:
        return tol_m
    return tol_m / factor


def boundary_snap_offered(prompt: str, object_count: int,
                          policy: dict | None = None) -> bool:








    try:
        n = int(object_count)
    except (TypeError, ValueError):
        return False
    if n < 2:
        return False
    cap = boundary_snap_max_objects(policy)
    if cap > 0 and n > cap:
        return False
    return boundary_snap_offered_for(prompt, policy)


def boundary_snap_offered_for(prompt: str, policy: dict | None = None) -> bool:








    from .server_dials import feature_enabled

    if not feature_enabled("boundary_snap"):
        return False
    text = normalize_prompt(prompt)
    if not text:
        return False
    kws = boundary_snap_policy(policy).get("keywords")
    if isinstance(kws, list):
        keywords = tuple(k.lower() for k in kws if isinstance(k, str) and k.strip())
    else:
        keywords = ()
    if not keywords:
        keywords = _FALLBACK_KEYWORDS
    return any(keyword_matches(text, kw) for kw in keywords)







def snap_boundaries(
    geoms: list[QgsGeometry], tolerance: float, crs: str | None = None,
    cut_to_partition: bool = True,
) -> list[QgsGeometry]:




















    return snap_boundaries_ex(
        geoms, tolerance, crs=crs, cut_to_partition=cut_to_partition
    ).geometries


def snap_boundaries_ex(
    geoms: list[QgsGeometry], tolerance: float, crs: str | None = None,
    cut_to_partition: bool = True,
) -> SnapResult:

    if not isinstance(geoms, list) or len(geoms) < 2:
        return _unchanged(geoms, "fewer than two geometries")
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        return _unchanged(geoms, "no tolerance")
    tolerance = float(tolerance)
    if tolerance <= 0.0:
        return _unchanged(geoms, "no tolerance")

    try:
        return _run_snap(geoms, tolerance, crs, bool(cut_to_partition))
    except Exception as exc:  # noqa: BLE001
        return _unchanged(geoms, f"{type(exc).__name__}: {exc}")


def _unchanged(geoms: Any, reason: str) -> SnapResult:
    out = geoms if isinstance(geoms, list) else []
    total = _total_area(out)
    return SnapResult(out, False, total, total, 0.0, 0, reason)


def _total_area(geoms: list[QgsGeometry]) -> float:
    total = 0.0
    for geom in geoms:
        try:
            if geom is not None and not geom.isEmpty():
                total += float(geom.area())
        except Exception:  # noqa: BLE001  # nosec B112
            continue
    return total


def _set_centre(geoms: list[QgsGeometry]) -> tuple[float, float] | None:






    box = None
    for geom in geoms:
        try:
            if geom is None or geom.isEmpty():
                continue
            part = geom.boundingBox()
            if box is None:
                box = part
            else:
                box.combineExtentWith(part)
        except Exception:  # noqa: BLE001  # nosec B112
            continue
    if box is None or box.isEmpty():
        return None
    return float(box.center().x()), float(box.center().y())


def _axis_safe_tolerance(
    tolerance: float, geoms: list[QgsGeometry], crs: str | None
) -> float:













    try:
        from qgis.core import QgsCoordinateReferenceSystem

        ref = QgsCoordinateReferenceSystem(crs or _DEFAULT_CRS)
        if not ref.isValid() or not ref.isGeographic():
            return tolerance
        centre = _set_centre(geoms)
        if centre is None:
            return tolerance
        from .ground_frame import usable_aspect
        from .layer_conventions import ground_unit_aspect

        aspect = usable_aspect(ground_unit_aspect(ref, centre[0], centre[1]))
        return tolerance / max(1.0, aspect)
    except Exception:  # noqa: BLE001
        return tolerance


def _run_snap(
    geoms: list[QgsGeometry], tolerance: float, crs: str | None,
    cut_to_partition: bool = True,
) -> SnapResult:



    from .boundary_snap_pass import BoundarySnapPass

    pass_ = BoundarySnapPass(geoms, tolerance, crs, cut_to_partition)
    while not pass_.step(4096):
        pass
    return pass_.result()


def _unchanged_geometry(src: Any, out: Any) -> bool:
    try:
        return bool(src.equals(out))
    except Exception:  # noqa: BLE001
        return False


def _policy_value(reader: Any, fallback: float) -> float:


    try:
        return float(reader())
    except Exception:  # noqa: BLE001
        return fallback


def _run_snap_algorithm(
    layer: Any, tolerance: float
) -> list[QgsGeometry] | None:




    try:



        from qgis import processing
        from qgis.core import QgsApplication
    except ImportError:
        return None

    registry = QgsApplication.processingRegistry()
    if registry.algorithmById(_SNAP_ALG) is None:
        try:
            from processing.core.Processing import Processing

            Processing.initialize()
        except Exception:  # noqa: BLE001
            return None
        if registry.algorithmById(_SNAP_ALG) is None:
            return None

    result = processing.run(
        _SNAP_ALG,
        {
            "INPUT": layer,
            "REFERENCE_LAYER": layer,
            "TOLERANCE": tolerance,
            "BEHAVIOR": _SNAP_BEHAVIOR_ANCHOR_NODES,
            "OUTPUT": "memory:",
        },
    )
    out_layer = result.get("OUTPUT") if isinstance(result, dict) else None
    if out_layer is None:
        return None
    return [f.geometry() for f in out_layer.getFeatures()]


def _same_place(src: Any, out: Any, tolerance: float) -> bool:






    try:
        if out is None or out.isEmpty():
            return False
        allowance = tolerance * 2.0
        box = src.boundingBox()
        box.grow(allowance)
        if not box.intersects(out.boundingBox()):
            return False
        src_centre = src.centroid()
        out_centre = out.centroid()
        if (src_centre is None or src_centre.isEmpty()
                or out_centre is None or out_centre.isEmpty()):
            return True
        a = src_centre.asPoint()
        b = out_centre.asPoint()
        return math.hypot(b.x() - a.x(), b.y() - a.y()) <= allowance
    except Exception:  # noqa: BLE001
        return False


def _cut_to_partition(
    geoms: list[QgsGeometry], tolerance: float
) -> tuple[list[QgsGeometry] | None, int]:





    from .boundary_snap_pass import PartitionCutter

    cutter = PartitionCutter(geoms, tolerance)
    while not cutter.step(4096):
        pass
    return cutter.result(), cutter.slivers


def _drop_small_parts(geom: Any, min_area: float) -> tuple[Any, int]:



    from qgis.core import QgsGeometry

    try:
        if not geom.isMultipart():
            return geom, 0
        parts = geom.asGeometryCollection()
        if len(parts) < 2:
            return geom, 0
        biggest = max(range(len(parts)), key=lambda k: parts[k].area())
        keep = [p for k, p in enumerate(parts)
                if k == biggest or p.area() >= min_area]
        if len(keep) == len(parts):
            return geom, 0
        merged = QgsGeometry.unaryUnion(keep)
        if merged is None or merged.isEmpty():
            return geom, 0
        return merged, len(parts) - len(keep)
    except Exception:  # noqa: BLE001
        return geom, 0
