






from __future__ import annotations

from ...core.i18n import tr
from ...core.qt_compat import DistanceMeters
from ...core.telemetry_errors import slot_guard
from .auto_flow_credits import (
    AutoFlowCreditsMixin,
)
from .auto_flow_detail import (
    AutoFlowDetailMixin,
)
from .auto_flow_errors import (
    AutoFlowErrorsMixin,
)
from .auto_flow_grid import (
    AutoFlowGridMixin,
    _crs_run_identifier,
)
from .auto_flow_modes import (
    AutoFlowModesMixin,
)
from .auto_flow_run_plan import (
    AutoFlowRunPlanMixin,
)
from .auto_flow_wiring import (
    AutoFlowWiringMixin,
)
from .shared import (
    _WEBMERC_MUPP_Z0,
    _debounce_timer,
    clip_served_hint,
)


class AutoFlowMixin(
    AutoFlowModesMixin,
    AutoFlowCreditsMixin,
    AutoFlowErrorsMixin,
    AutoFlowWiringMixin,
    AutoFlowGridMixin,
    AutoFlowDetailMixin,
    AutoFlowRunPlanMixin,
):
    pass


__all__ = [
    "tr",
    "DistanceMeters",
    "slot_guard",
    "_WEBMERC_MUPP_Z0",
    "_debounce_timer",
    "clip_served_hint",
    "_crs_run_identifier",
    "AutoFlowMixin",
    "AutoFlowModesMixin",
    "AutoFlowCreditsMixin",
    "AutoFlowErrorsMixin",
    "AutoFlowWiringMixin",
    "AutoFlowGridMixin",
    "AutoFlowDetailMixin",
    "AutoFlowRunPlanMixin",
]
