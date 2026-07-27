






from __future__ import annotations

from .ui_refresh_auto_page import DockAutoPageMixin
from .ui_refresh_credits import (
    DockCreditsDisplayMixin,
    _gauge_percent_used,
    format_km2_left,
    format_km2_surface,
    format_quota_count,
)
from .ui_refresh_exemplars import DockExemplarsMixin
from .ui_refresh_install import (
    _INSTALL_ETA_CEILING_S,
    _INSTALL_ETA_HONEST_S,
    DockInstallStatusMixin,
)
from .ui_refresh_instructions import (
    _SIGN_ADD,
    _SIGN_TRIM,
    DockInstructionsMixin,
)
from .ui_refresh_lifecycle import DockLifecycleMixin
from .ui_refresh_session import DockSessionStateMixin


class DockStateMixin(
    DockInstallStatusMixin,
    DockSessionStateMixin,
    DockInstructionsMixin,
    DockExemplarsMixin,
    DockAutoPageMixin,
    DockCreditsDisplayMixin,
    DockLifecycleMixin,
):
    pass


__all__ = [
    "DockStateMixin",
    "_INSTALL_ETA_HONEST_S",
    "_INSTALL_ETA_CEILING_S",
    "_SIGN_ADD",
    "_SIGN_TRIM",
    "format_km2_left",
    "_gauge_percent_used",
    "format_quota_count",
    "format_km2_surface",
]
