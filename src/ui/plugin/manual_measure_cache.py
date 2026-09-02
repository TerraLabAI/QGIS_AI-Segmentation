"""What a Semi-Auto preview repaint should not pay for twice.

A repaint of the live outline asks the same three questions over and over: how
long a ground metre is here, how wide the object is across its short side, and
whether a click rollback went through. The first two each build or measure
something the session already knows, and a slider dragged across its range asks
for them once per step.

Everything here is a plain function taking the plugin instance, so no name
lands in the mixin namespace and nothing has to be assembled into the class.
The caches key on what they were built from, so a session that changes raster
or CRS rebuilds them instead of answering with the old one.
"""
from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog


def session_area_measurer(owner, crs=None):
    """A geodesic measurer for this session's CRS, built once.

    ``setEllipsoid`` reads the SRS database, which is slow enough to show when
    one preview repaint asks for four of them. None when there is no usable
    CRS, which is what every caller already handles.
    """
    try:
        if crs is None:
            layer = getattr(owner, "_current_layer", None)
            crs = layer.crs() if layer is not None else None
        if crs is None or not crs.isValid():
            return None
        key = crs.authid() or crs.toWkt()
    except (RuntimeError, AttributeError):
        return None
    if getattr(owner, "_manual_measurer_key", None) == key:
        cached = getattr(owner, "_manual_measurer", None)
        if cached is not None:
            return cached
    try:
        from ...core.layer_conventions import make_area_measurer

        measurer = make_area_measurer(crs)
    except (RuntimeError, AttributeError):
        return None
    owner._manual_measurer = measurer
    owner._manual_measurer_key = key
    return measurer


def narrow_dimension(owner, geom) -> float:
    """The short side of the object's oriented bounding box, once per outline.

    0.0 when it cannot be measured, which callers read as "no bound".
    """
    memo = getattr(owner, "_manual_narrow_memo", None)
    if memo is not None and memo[0] is geom:
        return memo[1]
    try:
        _pt, _area, _angle, width, height = geom.orientedMinimumBoundingBox()
        narrow = min(float(width), float(height))
    except Exception:  # noqa: BLE001 -- unmeasurable, the caller keeps its value
        narrow = 0.0
    owner._manual_narrow_memo = (geom, narrow)
    return narrow


def rollback_click_quietly(owner, polarity: str, canvas_point=None) -> None:
    """Take a failed click's point back down without replacing its error.

    The rollback runs inside an ``except`` on its way to the slot guard, so an
    exception raised here would be the one the user is told about, and the
    real cause would never be reported at all.
    """
    try:
        owner._rollback_failed_click(polarity, canvas_point)
    except Exception:  # noqa: BLE001 -- the original error is what matters
        QgsMessageLog.logMessage(
            "Could not take back a failed click",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
