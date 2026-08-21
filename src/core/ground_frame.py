



















from __future__ import annotations

import math
from typing import Any






ASPECT_IDENTITY_EPSILON = 0.01


def usable_aspect(value: Any) -> float:





    try:
        aspect = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(aspect) or aspect <= 0.0:
        return 1.0
    return aspect


def aspect_is_identity(aspect: float) -> bool:

    return abs(usable_aspect(aspect) - 1.0) < ASPECT_IDENTITY_EPSILON


def stretch_y(geom: Any, aspect: float) -> Any:





    aspect = usable_aspect(aspect)
    if geom is None or aspect_is_identity(aspect):
        return None
    return _scaled_copy(geom, aspect)


def unstretch_y(geom: Any, aspect: float) -> Any:











    aspect = usable_aspect(aspect)
    if geom is None or aspect_is_identity(aspect):
        return None
    return _scaled_copy(geom, 1.0 / aspect)


def _scaled_copy(geom: Any, factor: float) -> Any:






    try:
        from qgis.core import QgsGeometry
        from qgis.PyQt.QtGui import QTransform

        matrix = QTransform.fromScale(1.0, factor)
    except Exception:  # noqa: BLE001
        return None
    for apply_matrix in (_scale_through_geometry, _scale_through_inner_geometry):
        out = _scaled_once(QgsGeometry, geom, factor, matrix, apply_matrix)
        if out is not None:
            return out
    return None


def _scaled_once(builder: Any, geom: Any, factor: float, matrix: Any,
                 apply_matrix: Any) -> Any:

    try:
        out = builder(geom)
        if out.isEmpty():
            return None
        if apply_matrix(out, matrix) and _y_was_scaled(geom, out, factor):
            return out
    except Exception:  # noqa: BLE001
        return None
    return None


def _scale_through_geometry(out: Any, matrix: Any) -> bool:

    return not _reports_failure(out.transform(matrix))


def _scale_through_inner_geometry(out: Any, matrix: Any) -> bool:






    inner = out.get()
    if inner is None:
        return False
    inner.transform(matrix)
    return True


def _reports_failure(result: Any) -> bool:







    if result is None:
        return False
    if isinstance(result, bool):
        return not result
    try:
        return int(result) != 0
    except (TypeError, ValueError):
        return False


def _y_was_scaled(before: Any, after: Any, factor: float) -> bool:





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
