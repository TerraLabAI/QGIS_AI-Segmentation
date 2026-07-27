"""The "fill holes smaller than N m2" threshold, and the two conversions it needs.

Fill holes used to be all or nothing. Segment a road and the detector rightly
leaves the parked cars out, so every car becomes an interior hole; filling them
all also fills the road median, which is a real hole. A size threshold splits
the two, the way ArcGIS "Eliminate Polygon Part" and QGIS ``native:deleteholes``
already do it: holes under N ground m2 are filled, bigger ones stay.

The control is TRUE GROUND m2, like the Min/Max size controls beside it, so the
same number means the same thing whatever CRS the run happens in. Two consumers
need it in their own units:

  - the Automatic review and the handoff work on POLYGONS and hand the value to
    ``QgsGeometry.removeInteriorRings``, which counts in CRS units squared. In
    Web Mercator one unit is well under a metre (about 0.79 m at latitude 38),
    so a squared conversion factor sits between the two;
  - base Manual works on the pixel MASK and hands the value to
    ``polygon_exporter.fill_small_holes``, which counts in pixels.

Both conversions live here, as plain arithmetic on floats, so they are unit
tested without a QGIS runtime. The callers supply the two measurements: the
CRS-unit area and the true ground area of the same shape (see
``layer_conventions.make_area_measurer``, the one area convention in the
plugin), or the ground area of one mask pixel.

0 means "fill every hole", which is what the control shipped as before the
threshold existed, so a user who never touches the new field keeps the old
result exactly.
"""

from __future__ import annotations

# What QgsGeometry.removeInteriorRings takes to mean "every interior ring".
# A 0 passed straight through removes NOTHING (no ring has an area under 0),
# which is the opposite of what the control's 0 means, so the two never meet
# without going through ring_area_arg below.
REMOVE_ALL_RINGS = -1.0


def units2_per_m2(crs_area: float, ground_area_m2: float) -> float:
    """CRS units squared per true ground m2, measured on one shape.

    Both numbers describe the SAME geometry: ``crs_area`` is what
    ``QgsGeometry.area()`` reports (CRS units squared), ``ground_area_m2`` what
    the ellipsoidal measurer reports (true m2). Their ratio is the local scale
    between the two, so it stays right in Web Mercator, in a degree CRS, and in
    a metric projection alike, with no per-CRS table.

    Answers 1.0 when either measurement is unusable, which reads a ground-m2
    setting as CRS units: the behaviour before this conversion existed.
    """
    try:
        crs_area = float(crs_area)
        ground_area_m2 = float(ground_area_m2)
    except (TypeError, ValueError):
        return 1.0
    if crs_area <= 0.0 or ground_area_m2 <= 0.0:
        return 1.0
    return crs_area / ground_area_m2


def ring_area_arg(ground_m2: float, scale_units2_per_m2: float = 1.0) -> float:
    """The argument ``QgsGeometry.removeInteriorRings`` wants.

    ``ground_m2`` is the user's field: 0 (or less) means fill every hole, which
    maps to the native "remove them all" sentinel. Anything positive is scaled
    into CRS units squared by ``scale_units2_per_m2`` (see units2_per_m2) and
    passed as the ring-area cutoff, so only rings under it are dropped.
    """
    try:
        ground_m2 = float(ground_m2)
    except (TypeError, ValueError):
        return REMOVE_ALL_RINGS
    if ground_m2 <= 0.0:
        return REMOVE_ALL_RINGS
    try:
        scale = float(scale_units2_per_m2)
    except (TypeError, ValueError):
        scale = 1.0
    if scale <= 0.0:
        scale = 1.0
    return ground_m2 * scale


def hole_pixels(ground_m2: float, m2_per_pixel: float) -> int | None:
    """The mask-pixel cutoff for ``fill_small_holes``, or None to fill them all.

    None (not 0) is the "fill every hole" answer, because 0 is a meaningful
    cutoff there: it fills nothing. A threshold finer than one pixel therefore
    returns 0 and fills nothing, which is the honest reading of "fill holes
    under an area smaller than the smallest hole that can exist".
    """
    try:
        ground_m2 = float(ground_m2)
    except (TypeError, ValueError):
        return None
    if ground_m2 <= 0.0:
        return None
    try:
        m2_per_pixel = float(m2_per_pixel)
    except (TypeError, ValueError):
        return None
    if m2_per_pixel <= 0.0:
        return None
    return int(ground_m2 / m2_per_pixel)
