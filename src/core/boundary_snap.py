"""Shared boundaries for LAND COVER results. Opt-in, never automatic.

A detector returns every instance on its own, so two neighbouring land-cover
parcels almost touch instead of touching: a hairline gap here, a hairline
overlap there. A land-cover map is meant to be a planar partition, one surface
cut into classes with no gap and no double cover, so the user has to close
those seams by hand before the layer is deliverable. This module closes them.

WHEN THIS APPLIES, AND WHEN IT MUST NOT
--------------------------------------
Land cover only: fields, parcels, land-use classes, any result the user reads
as one continuous surface. For discrete objects (buildings, cars, trees) the
gap between two neighbours is REAL and correct, and pulling them together
would destroy good data. That is also the industry convention.

So this module never runs itself:

* ``SNAP_DEFAULT_ENABLED`` is False. The caller owns the control and the
  control ships off.
* ``boundary_snap_offered_for(prompt)`` answers whether the run even looks
  like land cover, so the control can stay hidden on an object run. It is
  fail-CLOSED: an empty or unknown prompt returns False.
* ``boundary_snap_offered(prompt, object_count)`` adds the size question a
  caller with a real result has to ask: one pass runs to the end, so past
  ``boundary_snap_max_objects`` the control is not offered at all.
* ``snap_boundaries`` does exactly what it is told and nothing on its own. It
  is a pure geometry call: no Qt, no UI, no layer side effects.

HOW IT WORKS
------------
``native:snapgeometries`` with BEHAVIOR 7 ("snap to anchor nodes", the one
behaviour meant for snapping a layer against itself) pulls near-coincident
vertices of neighbouring polygons onto shared anchors, which closes hairline
gaps and hairline overlaps in one pass. It is the standard QGIS tool for this
and it is a native algorithm, so it is present in every install. ``v.clean``
would be the alternative but it needs the GRASS provider, which many users do
not have, so this module never depends on it.

Snapping alone can still leave a sliver where two boundaries cross, so the
result is then cut into a real partition: each polygon gives up whatever it
still shares with an earlier one, and parts too small to be real are dropped.
That cut costs more than the snap and grows faster than the object count, so a
caller that is only redrawing the map can skip it (``cut_to_partition=False``)
and cut once on the way to disk.

Everything is best effort. Any failure, any guard trip, and the caller gets
its input list back unchanged. Area is checked before and after: snapping
moves boundaries, it must never delete a parcel.

Call this on the main thread (it runs a Processing algorithm).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from .prompt_taxonomy import keyword_matches, normalize_prompt

if TYPE_CHECKING:  # annotations only, never a QGIS import at plugin load
    from qgis.core import QgsGeometry

# The snapping algorithm and the one behaviour meant for a layer snapped
# against itself. Anything else needs a separate reference layer.
_SNAP_ALG = "native:snapgeometries"
_SNAP_BEHAVIOR_ANCHOR_NODES = 7

# The control ships OFF. Land cover wants shared boundaries; discrete objects
# do not, and a wrong default here silently rewrites correct geometry.
SNAP_DEFAULT_ENABLED = False

# Generic client fallbacks. ONE value each, used only when the server policy
# says nothing; the tuned values live in the server policy under
# ``review.boundary_snap``.
_FALLBACK_TOLERANCE_M = 0.5
_FALLBACK_MAX_AREA_CHANGE = 0.02
_FALLBACK_MIN_KEEP_SHARE = 0.5
# How many shapes one pass may take. A pass is one Processing call over the
# whole set, so it cannot be cut into slices: past this count the control is
# not offered at all, rather than freezing the panel on every toggle.
_FALLBACK_MAX_OBJECTS = 2000

# Generic client fallback keyword set for the land-cover gate. One small
# generic set, NOT a mirror of the tuned server table.
_FALLBACK_KEYWORDS: tuple[str, ...] = (
    "land cover",
    "landcover",
    "land use",
    "landuse",
    "parcel",
    "field",
    "crop",
)

# Placeholder CRS for the working layer when the caller names none. Only the
# unit matters: the tolerance is read in the geometries' own coordinate units.
_DEFAULT_CRS = "EPSG:3857"


class SnapResult(NamedTuple):
    """Outcome of one snap pass.

    geometries: the geometries to use, input order kept. The input list
        unchanged when ``snapped`` is False.
    snapped: whether the output differs from the input.
    area_before / area_after: total area, in the geometries' own units.
    area_change: signed share of the total area the pass moved (0.01 = 1%).
    slivers_removed: parts dropped as too small to be real.
    reason: why a pass did nothing, for logs and tests. Empty when it ran.
    """

    geometries: list[QgsGeometry]
    snapped: bool
    area_before: float
    area_after: float
    area_change: float
    slivers_removed: int
    reason: str


# ---------------------------------------------------------------------------
# Server policy readers
# ---------------------------------------------------------------------------
# These read ``review.boundary_snap`` off the same server-delivered policy blob
# every other tuned value comes from, through detection_policy.review_policy.
# The key is additive: an older server that does not send it leaves every
# reader on its single generic client fallback.


def boundary_snap_policy(policy: dict | None = None) -> dict:
    """The ``review.boundary_snap`` sub-policy. Empty dict when absent, so each
    reader below falls open to its generic client value."""
    from .detection_policy import review_policy

    val = review_policy(policy).get("boundary_snap")
    return val if isinstance(val, dict) else {}


def snap_default_enabled(policy: dict | None = None) -> bool:
    """Whether the control starts ticked. The shipped answer unless the server
    changes it, so the starting position can be revisited without a release.

    Read from ``review.boundary_snap.default_enabled``, the same root as every
    other value of this block. Only a real ``true`` or ``false`` counts: 0, 1,
    "yes" and every other shape leave the shipped constant in place."""
    val = boundary_snap_policy(policy).get("default_enabled")
    if isinstance(val, bool):
        return val
    return SNAP_DEFAULT_ENABLED


def boundary_snap_tolerance_m(policy: dict | None = None) -> float:
    """How far a vertex may travel to meet its neighbour, in ground metres.

    Read from ``review.boundary_snap.tolerance_m``; the fallback is ONE
    generic client value, never a mirror of a tuned table. A non-positive or
    absurd value falls back too, since the tolerance drives how much geometry
    the pass may move."""
    val = boundary_snap_policy(policy).get("tolerance_m")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = float(val)
        if 0.0 < f <= 100.0:
            return f
    return _FALLBACK_TOLERANCE_M


def boundary_snap_max_area_change(policy: dict | None = None) -> float:
    """Share of the total area the pass may move before the result is thrown
    away and the input kept (0.02 = 2%). Read from
    ``review.boundary_snap.max_area_change``, with ONE generic fallback."""
    val = boundary_snap_policy(policy).get("max_area_change")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = float(val)
        if 0.0 < f <= 1.0:
            return f
    return _FALLBACK_MAX_AREA_CHANGE


def boundary_snap_min_keep_share(policy: dict | None = None) -> float:
    """Smallest share of its own area a single polygon may keep (0.5 = half).

    The total-area guard cannot see one small parcel swallowed by its big
    neighbours, so each polygon is checked on its own too. Read from
    ``review.boundary_snap.min_keep_share``, with ONE generic fallback."""
    val = boundary_snap_policy(policy).get("min_keep_share")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = float(val)
        if 0.0 < f <= 1.0:
            return f
    return _FALLBACK_MIN_KEEP_SHARE


def boundary_snap_max_objects(policy: dict | None = None) -> int:
    """How many shapes one pass may take (0 = no limit).

    Read from ``review.boundary_snap.max_objects``, with ONE generic client
    fallback. The pass is a single Processing call over the whole set, so it
    runs to the end once started: this is what keeps a huge result from
    stalling the panel."""
    val = boundary_snap_policy(policy).get("max_objects")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        n = int(val)
        if n >= 0:
            return n
    return _FALLBACK_MAX_OBJECTS


def boundary_snap_tolerance_units(metres_per_unit: float,
                                  policy: dict | None = None) -> float:
    """The tolerance in the GEOMETRIES' own units.

    ``snap_boundaries`` reads its tolerance in whatever unit the geometries
    count in, while the dial is a ground distance. ``metres_per_unit`` is how
    many ground metres one of those units spans (1.0 in a metric projection,
    about 0.79 in Web Mercator at 38 degrees, tens of thousands in degrees).
    A missing or unusable factor reads the dial as units, which is what the
    caller did before any conversion existed."""
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
    """Whether the control may be SHOWN for this result.

    Two conditions, both fail-closed: the run looks like land cover (see
    ``boundary_snap_offered_for``), and the result is small enough for one
    pass (see ``boundary_snap_max_objects``). A single shape has no
    neighbour, so it is never offered either. Showing nothing is deliberate:
    on an object run the option must not exist, and on a result too big to
    snap a greyed control would only ask the user to guess why."""
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
    """Whether the run looks like LAND COVER, so the control may be offered.

    Fail-CLOSED: an empty prompt, or one no land-cover keyword hits as a whole
    word, returns False. This answers "may the user see the control", never
    "turn it on": the control itself still ships off (SNAP_DEFAULT_ENABLED).

    A server switch can withdraw the control entirely. It fails open, so no
    configuration leaves the behaviour exactly as shipped."""
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


# ---------------------------------------------------------------------------
# Geometry core
# ---------------------------------------------------------------------------


def snap_boundaries(
    geoms: list[QgsGeometry], tolerance: float, crs: str | None = None,
    cut_to_partition: bool = True,
) -> list[QgsGeometry]:
    """Give neighbouring LAND COVER polygons an exact shared boundary.

    ``geoms`` are QgsGeometry polygons in one projected CRS; ``tolerance`` is
    the largest gap or overlap to close, in those geometries' own units
    (metres for the CRS an Automatic run works in). ``crs`` names that CRS for
    the working layer and only matters for the unit.

    ``cut_to_partition`` runs the second half of the pass, which cuts the
    snapped shapes into a true partition (see ``_cut_to_partition``). It costs
    more than the snap itself and grows faster than the object count, so a
    caller that only has to SHOW the result may pass False and take the
    snapped shapes; what gets written to disk must always be cut. Leaving it
    out keeps the full pass.

    Returns a new list in the input order. On any failure or guard trip it
    returns the input list unchanged, and it never raises.

    This is for land cover ONLY. See the module docstring: on discrete objects
    a gap between neighbours is correct data, and closing it destroys it.
    """
    return snap_boundaries_ex(
        geoms, tolerance, crs=crs, cut_to_partition=cut_to_partition
    ).geometries


def snap_boundaries_ex(
    geoms: list[QgsGeometry], tolerance: float, crs: str | None = None,
    cut_to_partition: bool = True,
) -> SnapResult:
    """``snap_boundaries`` plus what the pass measured (see SnapResult)."""
    if not isinstance(geoms, list) or len(geoms) < 2:
        return _unchanged(geoms, "fewer than two geometries")
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        return _unchanged(geoms, "no tolerance")
    tolerance = float(tolerance)
    if tolerance <= 0.0:
        return _unchanged(geoms, "no tolerance")

    try:
        return _run_snap(geoms, tolerance, crs, bool(cut_to_partition))
    except Exception as exc:  # noqa: BLE001 -- best effort, never break a run
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
        except Exception:  # noqa: BLE001 -- a bad member never breaks the sum  # nosec B112
            continue
    return total


def _run_snap(
    geoms: list[QgsGeometry], tolerance: float, crs: str | None,
    cut_to_partition: bool = True,
) -> SnapResult:
    from qgis.core import QgsFeature, QgsGeometry, QgsVectorLayer

    for geom in geoms:
        if not isinstance(geom, QgsGeometry) or geom.isEmpty():
            return _unchanged(geoms, "empty or non-geometry member")

    area_before = _total_area(geoms)
    if area_before <= 0.0:
        return _unchanged(geoms, "zero input area")

    layer = QgsVectorLayer(
        f"Polygon?crs={crs or _DEFAULT_CRS}", "boundary_snap", "memory"
    )
    if not layer.isValid():
        return _unchanged(geoms, "working layer not created")
    feats = []
    for geom in geoms:
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry(geom))
        feats.append(feat)
    if not layer.dataProvider().addFeatures(feats):
        return _unchanged(geoms, "working layer not filled")
    layer.updateExtents()

    snapped = _run_snap_algorithm(layer, tolerance)
    if snapped is None:
        return _unchanged(geoms, "snap algorithm unavailable")
    if len(snapped) != len(geoms):
        return _unchanged(geoms, "snap dropped or added geometries")

    # The algorithm keeps input order, but a wrong pairing would silently move
    # every polygon somewhere else, so check each output still sits where its
    # input did before anything downstream trusts the list.
    for src, out in zip(geoms, snapped):
        if not _same_place(src, out, tolerance):
            return _unchanged(geoms, "output no longer matches its input")

    if cut_to_partition:
        partitioned, slivers = _cut_to_partition(snapped, tolerance)
        if partitioned is None:
            return _unchanged(geoms, "partition failed")
    else:
        # Snapped only: the borders already coincide, what is left uncut is the
        # crossing sliver. Good enough to look at, never good enough to write.
        partitioned, slivers = snapped, 0

    area_after = _total_area(partitioned)
    change = (area_after - area_before) / area_before
    if abs(change) > _policy_value(boundary_snap_max_area_change,
                                   _FALLBACK_MAX_AREA_CHANGE):
        return _unchanged(geoms, f"area moved {change:.4f}, over the limit")
    # A parcel small next to its neighbours can be swallowed whole without
    # moving the total by much, so every polygon is checked on its own.
    keep_share = _policy_value(boundary_snap_min_keep_share,
                               _FALLBACK_MIN_KEEP_SHARE)
    for src, out in zip(geoms, partitioned):
        if out.area() < src.area() * keep_share:
            return _unchanged(geoms, "a polygon lost too much of itself")
    if all(_unchanged_geometry(s, o) for s, o in zip(geoms, partitioned)):
        # Nothing was close enough to snap (neighbours further apart than the
        # tolerance). Hand back the caller's own objects, not copies.
        return _unchanged(geoms, "nothing within the tolerance")

    return SnapResult(
        partitioned, True, area_before, area_after, change, slivers, ""
    )


def _unchanged_geometry(src: Any, out: Any) -> bool:
    try:
        return bool(src.equals(out))
    except Exception:  # noqa: BLE001 -- unreadable geometry counts as changed
        return False


def _policy_value(reader: Any, fallback: float) -> float:
    """A policy reader's value, or its generic fallback when the policy cannot
    be read at all (no server configuration cached yet, for instance)."""
    try:
        return float(reader())
    except Exception:  # noqa: BLE001 -- policy is best effort
        return fallback


def _run_snap_algorithm(
    layer: Any, tolerance: float
) -> list[QgsGeometry] | None:
    """Run the snapping algorithm over the layer against itself.

    Returns the snapped geometries in the layer's feature order, or None when
    Processing or the algorithm is not available."""
    try:
        # Qualified import: the bare "import processing" only resolves once
        # QGIS has put the plugins directory on sys.path, which is not
        # guaranteed everywhere this module runs.
        from qgis import processing
        from qgis.core import QgsApplication
    except ImportError:
        return None

    registry = QgsApplication.processingRegistry()
    if registry.algorithmById(_SNAP_ALG) is None:
        try:
            from processing.core.Processing import Processing

            Processing.initialize()
        except Exception:  # noqa: BLE001 -- absent provider is a clean no-op
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
    """Whether a snapped polygon still covers the polygon it came from.

    A vertex may only travel up to the tolerance, so the two bounding boxes
    must overlap once the source box is grown by it. Cheap, and it catches a
    reordered or mispaired output list."""
    try:
        if out is None or out.isEmpty():
            return False
        box = src.boundingBox()
        box.grow(tolerance * 2.0)
        return box.intersects(out.boundingBox())
    except Exception:  # noqa: BLE001 -- unreadable geometry fails the check
        return False


def _cut_to_partition(
    geoms: list[QgsGeometry], tolerance: float
) -> tuple[list[QgsGeometry] | None, int]:
    """Turn snapped polygons into a real partition: no polygon keeps area an
    earlier one already covers, and parts too small to be real are dropped.

    Order decides who keeps a contested sliver, so the same input always gives
    the same output. Returns (geometries, slivers removed), or (None, 0) when
    a polygon comes out empty (which would delete a parcel)."""
    from qgis.core import QgsGeometry, QgsSpatialIndex

    index = QgsSpatialIndex()
    # Feature ids start at 1 so index 0 is never mistaken for "not found".
    kept: list[QgsGeometry] = []
    slivers = 0
    # A part smaller than a tolerance square cannot be a real parcel: it is
    # what a moved boundary leaves behind.
    min_part_area = tolerance * tolerance

    for i, geom in enumerate(geoms):
        current = QgsGeometry(geom)
        if not current.isGeosValid():
            fixed = current.makeValid()
            if fixed is not None and not fixed.isEmpty():
                current = fixed
        for j in index.intersects(current.boundingBox()):
            earlier = kept[j - 1]
            if earlier is None or earlier.isEmpty():
                continue
            if not current.intersects(earlier):
                continue
            cut = current.difference(earlier)
            if cut is None or cut.isEmpty():
                return None, 0
            current = cut
        current, dropped = _drop_small_parts(current, min_part_area)
        slivers += dropped
        if current is None or current.isEmpty():
            return None, 0
        kept.append(current)
        if not index.addFeature(i + 1, current.boundingBox()):
            # Without this entry the next polygons are never cut against this
            # one, so the result would silently stop being a partition.
            return None, 0

    return kept, slivers


def _drop_small_parts(geom: Any, min_area: float) -> tuple[Any, int]:
    """Drop the parts of a multipart geometry below ``min_area``.

    The largest part always survives, so a polygon is never deleted by this."""
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
    except Exception:  # noqa: BLE001 -- keep the geometry as it is
        return geom, 0
