











from __future__ import annotations

from qgis.PyQt.QtWidgets import QWidget





_FLOOR_TEXT_LINES = 16
_FLOOR_MIN_PX = 260


def dock_minimum_height(widget: QWidget) -> int:

    try:
        line_px = widget.fontMetrics().height()
    except Exception:  # noqa: BLE001
        return _FLOOR_MIN_PX
    return max(_FLOOR_MIN_PX, int(line_px) * _FLOOR_TEXT_LINES)
