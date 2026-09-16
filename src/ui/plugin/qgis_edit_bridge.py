














































from __future__ import annotations

from .qgis_edit_bridge_aids import (
    _SNAP_TOLERANCE_PX,
    EditBridgeAidsMixin,
    _avoid_mode,
    _snap_mode_all_layers,
    _snap_type_flags,
    _tolerance_pixels_unit,
)
from .qgis_edit_bridge_gestures import (
    _ADD_TOOL_CLASSES,
    _BRIDGE_POLL_MS,
    _SPLIT_TOOL_CLASS,
    EditBridgeGesturesMixin,
    _bridge_capture_points,
)
from .qgis_edit_bridge_session import (
    EditBridgeSessionMixin,
)
from .qgis_edit_bridge_tools import (
    _BRIDGE_SHAPE_TOOLS,
    _VERTEX_TOOL_CLASSES,
    EditBridgeToolsMixin,
    flags_without,
)


class QgisEditBridgeMixin(
    EditBridgeSessionMixin,
    EditBridgeToolsMixin,
    EditBridgeGesturesMixin,
    EditBridgeAidsMixin,
):
    pass








__all__ = [
    "QgisEditBridgeMixin",
    "_ADD_TOOL_CLASSES",
    "_BRIDGE_POLL_MS",
    "_BRIDGE_SHAPE_TOOLS",
    "_SNAP_TOLERANCE_PX",
    "_SPLIT_TOOL_CLASS",
    "_VERTEX_TOOL_CLASSES",
    "_avoid_mode",
    "_bridge_capture_points",
    "_snap_mode_all_layers",
    "_snap_type_flags",
    "_tolerance_pixels_unit",
    "flags_without",
]
