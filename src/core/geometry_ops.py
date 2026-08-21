"""Merge and split, the two hand edits a detector's output always needs.

The model draws what it sees, one mask at a time, so two failures survive every
run and no confidence slider can fix either:

- One real object comes back as several polygons. A building whose wings were
  read as separate roofs, an object cut by a tile seam that the run-time
  stitcher did not join. The user picks the pieces and merges them.
- Two real objects come back as one polygon. Two terraced houses under one
  continuous roof line. The user draws a line across the blob and cuts it.

Both functions here are pure geometry: they take QgsGeometry in and give
QgsGeometry out, touch no layer, no canvas and no Qt widget, and carry no
user-facing text. That keeps them testable on their own and lets the review UI
stay a thin caller.

Both are best-effort. Bad input (None, empty, self-intersecting, a cut line
that misses) never raises; it returns the input unchanged so a mis-click in
the review can never lose a detection.
"""
from __future__ import annotations

from qgis.core import QgsGeometry, QgsPointXY

from .layer_conventions import repair_polygon


def _repaired(geom: QgsGeometry | None) -> QgsGeometry | None:
    """A valid polygon-only copy of geom, or None when nothing areal is left.

    Wraps ``layer_conventions.repair_polygon`` so both operations repair the
    same way the export path does (makeValid, then keep only the polygon
    parts), and so an exception inside GEOS degrades to None instead of
    escaping.
    """
    if geom is None:
        return None
    try:
        if geom.isEmpty():
            return None
        return repair_polygon(geom)
    except Exception:  # noqa: BLE001 -- best-effort
        return None


def merge_geometries(geoms: list[QgsGeometry]) -> QgsGeometry | None:
    """Union several detected polygons into one geometry.

    Returns None when there is nothing to merge (empty list, or every input
    null, empty or non-areal). A single usable input comes back repaired, not
    wrapped or altered.

    Disjoint inputs: the union NEVER bridges the gap. Two polygons that do not
    touch come back as ONE multipart geometry, one object with several parts,
    which is what a GeoPackage MultiPolygon layer stores anyway. Bridging would
    invent ground the model never saw, and the width of that invented strip
    would be a guess. This is the same primitive the run-time stitcher uses
    (``QgsGeometry.combine`` in ``core/merger.py``), which also only ever
    unions what already overlaps. A caller that wants to warn about a gap can
    test ``result.isMultipart()``.

    Touching or overlapping inputs dissolve into a single polygon, the common
    case: seam halves overlap inside the tile overlap strip, and two wings of
    one building share a wall.

    The result is repaired (makeValid, polygon parts only) so it can go
    straight onto the output layer.
    """
    if not geoms:
        return None

    parts = [fixed for fixed in (_repaired(g) for g in geoms) if fixed is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    merged = _union_all(parts)
    if merged is None:
        # Nothing unioned. Keep every part as one multipart object rather than
        # dropping area: the user asked for one object and must get one back.
        merged = _collected(parts)
    if merged is None:
        return None
    return _repaired(merged) or merged


def _union_all(parts: list[QgsGeometry]) -> QgsGeometry | None:
    """Union a list of polygons, one GEOS call first, pairwise as the fallback.

    ``unaryUnion`` is the single-call path and the one that handles a big
    selection cheaply. It returns null on some degenerate inputs, so a pairwise
    ``combine`` loop follows; a part whose combine fails is left for the
    caller's collect fallback instead of being silently dropped.
    """
    try:
        union = QgsGeometry.unaryUnion(parts)
        if union is not None and not union.isEmpty():
            return union
    except Exception:  # noqa: BLE001 -- best-effort  # nosec B110
        pass

    merged: QgsGeometry | None = None
    failed = False
    for part in parts:
        if merged is None:
            merged = part
            continue
        try:
            union = merged.combine(part)
        except Exception:  # noqa: BLE001 -- best-effort
            union = None
        if union is None or union.isEmpty():
            failed = True
            continue
        merged = union
    if failed or merged is None or merged.isEmpty():
        return None
    return merged


def _collected(parts: list[QgsGeometry]) -> QgsGeometry | None:
    """Every part in one multipart geometry, without a union pass."""
    try:
        collected = QgsGeometry.collectGeometry(parts)
    except Exception:  # noqa: BLE001 -- best-effort
        return None
    if collected is None or collected.isEmpty():
        return None
    return collected


def split_geometry(
    geom: QgsGeometry, cut_line: list[QgsPointXY]
) -> list[QgsGeometry]:
    """Cut one polygon in two with a line drawn across it.

    Return contract: a list of the resulting pieces, ALWAYS at least one item
    when there is a usable input.

    - The cut worked: two or more pieces, in a stable order (see below).
    - The cut did nothing (the line misses the polygon, only grazes an edge,
      stops inside it, or has fewer than two points): a one-item list holding
      the ORIGINAL geometry, unchanged and not repaired. A caller tests
      ``len(pieces) > 1`` to know whether anything happened.
    - No usable input (None): an empty list.

    The line must cross the polygon completely. QGIS only splits on a clean
    crossing; a line that ends inside the shape leaves it whole.

    Order: pieces come back sorted by centroid x, then centroid y, then area
    descending. The order is a property of the geometry, not of GEOS internals
    or of which piece QGIS happened to keep in place, so a caller can hand the
    original attributes to the piece it means to (usually the largest, or the
    one nearest the original centroid) and get the same answer every run.

    The input geometry is never modified. ``QgsGeometry.splitGeometry`` edits
    in place, so this works on a copy.
    """
    if geom is None:
        return []
    if not cut_line or len(cut_line) < 2:
        return [geom]

    source = _repaired(geom)
    if source is None:
        return [geom]

    work = QgsGeometry(source)
    products = _run_split(work, cut_line)
    if products is None:
        return [geom]

    pieces = [
        fixed
        for fixed in (_repaired(p) for p in [work, *products])
        if fixed is not None
    ]
    if len(pieces) < 2:
        return [geom]
    return _sorted_pieces(pieces)


def _run_split(
    work: QgsGeometry, cut_line: list[QgsPointXY]
) -> list[QgsGeometry] | None:
    """Call splitGeometry on ``work`` and return the NEW geometries it made.

    Returns None when the call itself failed. An empty list means the call ran
    and cut nothing.

    Two things move across the supported QGIS range (3.22 to 4) and both are
    handled by shape rather than by version number:

    - The trailing ``splitFeature`` argument has a default on current builds
      but not on every one, so a TypeError retries with it spelled out.
    - The return is a tuple whose first item is a result-code enum that moved
      class between versions (``QgsGeometry.OperationResult`` then
      ``Qgis.GeometryOperationResult``), and whose length has carried an extra
      topology-points item since 3.16. Reading the code would mean naming one
      of those enums, so the outcome is judged from the geometries instead:
      the pieces are the answer, whatever the code says. Only the new-geometry
      item is read, from a tuple of any length.
    """
    line = list(cut_line)
    try:
        try:
            ret = work.splitGeometry(line, False)
        except TypeError:
            ret = work.splitGeometry(line, False, True)
    except Exception:  # noqa: BLE001 -- best-effort
        return None
    return _split_products(ret)


def _split_products(ret) -> list[QgsGeometry]:
    """The new geometries out of a splitGeometry return value.

    Accepts the (code, new_geometries, topology_points) triple of current
    builds, the shorter (code, new_geometries) pair, and a bare result code
    with no geometries at all.
    """
    if ret is None:
        return []
    if not isinstance(ret, (tuple, list)):
        return []
    if len(ret) < 2:
        return []
    produced = ret[1]
    if not produced:
        return []
    try:
        return [g for g in produced if isinstance(g, QgsGeometry)]
    except TypeError:
        return []


def _sorted_pieces(pieces: list[QgsGeometry]) -> list[QgsGeometry]:
    """Sort split pieces by centroid x, then centroid y, then area descending."""
    def key(piece: QgsGeometry):
        try:
            point = piece.centroid().asPoint()
            return (point.x(), point.y(), -piece.area())
        except Exception:  # noqa: BLE001 -- best-effort
            return (0.0, 0.0, 0.0)

    return sorted(pieces, key=key)


def polygon_part_count(geom: QgsGeometry | None) -> int:
    """Polygon parts in a geometry, 1 when the count cannot be read.

    A union never bridges a gap, so pieces that do not touch come back as
    several parts. Counting them is how a Merge tells "one object stitched"
    from "two objects stacked in one row". An uncountable geometry reads as
    one part: refusing a join we cannot explain would cost the user a
    correction that is probably fine.
    """
    if geom is None:
        return 0
    try:
        if not geom.isMultipart():
            return 1
        return len(geom.asMultiPolygon())
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return 1


def bridge_seam_gap(geom: QgsGeometry | None,
                    tolerance: float) -> QgsGeometry | None:
    """Close hairline gaps between the parts of geom, or None on failure.

    The gap a merge exists to close is a tile seam: two halves of ONE object
    that a union leaves as two parts because a sliver of nothing sits between
    them. Growing by the tolerance joins them, and shrinking back by the same
    amount puts the outline where it was. Returns None when the tolerance is
    not usable, when the operation fails, or when the result is still several
    parts, which means the pieces were never one object.
    """
    if geom is None or tolerance <= 0:
        return None
    try:
        grown = geom.buffer(tolerance, 8)
        if grown is None or grown.isEmpty():
            return None
        closed = grown.buffer(-tolerance, 8)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return None
    if closed is None or closed.isEmpty():
        return None
    if polygon_part_count(closed) > 1:
        return None
    return _repaired(closed)
