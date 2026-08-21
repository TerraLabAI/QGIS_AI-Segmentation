"""One isotropic working frame for the geometry steps that read raw coordinates.

A buffer distance, a simplify tolerance, a snap tolerance: every one of them is
a single number applied to both axes. That is only a ground distance when one x
unit and one y unit of the geometry's CRS cover the same distance on the
ground. In a projected CRS they do. In a geographic one they do not, and the
gap grows with latitude, so the same dial reaches further along x than along y
and the shape comes back stretched.

The fix is the one ``building_regularizer`` already uses on its own snapping
math: stretch y into the x unit, run every step in that frame, map straight
back. The caller keeps its own CRS and every dial keeps meaning one distance.

``unit_aspect`` is ground metres per y unit over ground metres per x unit of
the geometry's CRS, which is what ``layer_conventions.ground_unit_aspect``
measures. 1.0 means the two axes already agree, and then nothing here does any
work at all.

No QGIS import at module load: the Manual pipeline imports this offline.
"""
from __future__ import annotations

import math
from typing import Any

# How far from 1.0 an aspect has to sit before the frame change is worth
# making. Under it the correction moves a corner by less than the width of the
# line it is drawn with, so a projected CRS pays nothing. Same band as
# layer_conventions.GROUND_ASPECT_DEAD_BAND, which is what decides whether a
# caller reports an aspect at all.
ASPECT_IDENTITY_EPSILON = 0.01


def usable_aspect(value: Any) -> float:
    """``value`` as a ratio this module can work with, or 1.0.

    1.0 for anything missing, negative, infinite or not a number, so a caller
    that measured nothing gets the frame it already had.
    """
    try:
        aspect = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(aspect) or aspect <= 0.0:
        return 1.0
    return aspect


def aspect_is_identity(aspect: float) -> bool:
    """Whether stretching by ``aspect`` would change nothing worth changing."""
    return abs(usable_aspect(aspect) - 1.0) < ASPECT_IDENTITY_EPSILON


def stretch_y(geom: Any, aspect: float) -> Any:
    """A copy of ``geom`` with every y multiplied by ``aspect``, or None.

    None on any failure and on an identity aspect, which tells the caller to
    stay in the frame it already has. Never mutates the input.
    """
    aspect = usable_aspect(aspect)
    if geom is None or aspect_is_identity(aspect):
        return None
    return _scaled_copy(geom, aspect)


def unstretch_y(geom: Any, aspect: float) -> Any:
    """Undo ``stretch_y``: divide every y by ``aspect``. None on any failure.

    A None here must never be shipped as a result: it means the shape is still
    in the stretched frame, and the caller has to fall back to the geometry it
    started from.

    The gate is ``aspect`` itself, the same number ``stretch_y`` was gated on,
    not its reciprocal: an aspect just outside the dead band has a reciprocal
    just inside it, and testing the reciprocal would refuse to bring back a
    shape that was stretched.
    """
    aspect = usable_aspect(aspect)
    if geom is None or aspect_is_identity(aspect):
        return None
    return _scaled_copy(geom, 1.0 / aspect)


def _scaled_copy(geom: Any, factor: float) -> Any:
    """A copy of ``geom`` with every y multiplied by ``factor``, or None.

    Two ways of applying the matrix, because neither is guaranteed on its own
    across the supported QGIS range, and each one is checked against the
    geometry that came out rather than against what it answered.
    """
    try:
        from qgis.core import QgsGeometry
        from qgis.PyQt.QtGui import QTransform

        matrix = QTransform.fromScale(1.0, factor)
    except Exception:  # noqa: BLE001 -- the caller keeps its own frame
        return None
    for apply_matrix in (_scale_through_geometry, _scale_through_inner_geometry):
        out = _scaled_once(QgsGeometry, geom, factor, matrix, apply_matrix)
        if out is not None:
            return out
    return None


def _scaled_once(builder: Any, geom: Any, factor: float, matrix: Any,
                 apply_matrix: Any) -> Any:
    """One way of applying the matrix, or None when it did not take."""
    try:
        out = builder(geom)
        if out.isEmpty():
            return None
        if apply_matrix(out, matrix) and _y_was_scaled(geom, out, factor):
            return out
    except Exception:  # noqa: BLE001 -- the caller tries the next way
        return None
    return None


def _scale_through_geometry(out: Any, matrix: Any) -> bool:
    """``QgsGeometry.transform`` with a matrix, present on every build here."""
    return not _reports_failure(out.transform(matrix))


def _scale_through_inner_geometry(out: Any, matrix: Any) -> bool:
    """The same matrix through the geometry underneath, which answers nothing.

    ``QgsGeometry.get()`` detaches first, so the copy is what gets moved and
    the caller's geometry is left alone. Whether it worked is settled by
    ``_y_was_scaled``.
    """
    inner = out.get()
    if inner is None:
        return False
    inner.transform(matrix)
    return True


def _reports_failure(result: Any) -> bool:
    """Whether a geometry operation result says outright that it failed.

    ``QgsGeometry.transform`` answers with an operation CODE whose success
    value is 0, so a plain truth test on it reads every success as a failure
    and the frame change never happens. Anything that is not a code and not a
    bool says nothing, and the geometry check decides instead.
    """
    if result is None:
        return False
    if isinstance(result, bool):
        return not result
    try:
        return int(result) != 0
    except (TypeError, ValueError):
        return False


def _y_was_scaled(before: Any, after: Any, factor: float) -> bool:
    """Whether ``after`` is ``before`` with its y axis multiplied.

    Read off the two bounding boxes, so it holds whatever the transform call
    reported and whichever way the matrix was applied.
    """
    try:
        if after is None or after.isEmpty():
            return False
        source = before.boundingBox()
        moved = after.boundingBox()
        pairs = (
            (source.xMinimum(), moved.xMinimum()),
            (source.xMaximum(), moved.xMaximum()),
            (source.yMinimum() * factor, moved.yMinimum()),
            (source.yMaximum() * factor, moved.yMaximum()),
        )
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False
    for want, got in pairs:
        if not math.isfinite(want) or not math.isfinite(got):
            return False
        if abs(got - want) > 1e-9 * max(1.0, abs(want)):
            return False
    return True


def conservative_ground_factor(metres_per_x_unit: float, aspect: float) -> float:
    """Ground metres per unit to divide a ground dial by so the dial is never
    exceeded on EITHER axis.

    A tolerance converted with the x factor alone reaches ``aspect`` times
    further along y, which on a geographic CRS is most of a metre out of half
    a metre. Taking the longer of the two axes keeps the converted value inside
    the dial everywhere, which is the safe side for a tolerance: it snaps a
    little less rather than a little too much.
    """
    try:
        factor = float(metres_per_x_unit)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(factor) or factor <= 0.0:
        return 0.0
    return factor * max(1.0, usable_aspect(aspect))
