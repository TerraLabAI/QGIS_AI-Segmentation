












from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog


def session_area_measurer(owner, crs=None):






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




    memo = getattr(owner, "_manual_narrow_memo", None)
    if memo is not None and memo[0] is geom:
        return memo[1]
    try:
        _pt, _area, _angle, width, height = geom.orientedMinimumBoundingBox()
        narrow = min(float(width), float(height))
    except Exception:  # noqa: BLE001
        narrow = 0.0
    owner._manual_narrow_memo = (geom, narrow)
    return narrow


def rollback_click_quietly(owner, polarity: str, canvas_point=None) -> None:






    try:
        owner._rollback_failed_click(polarity, canvas_point)
    except Exception:  # noqa: BLE001
        QgsMessageLog.logMessage(
            "Could not take back a failed click",
            "AI Segmentation", level=Qgis.MessageLevel.Warning)
