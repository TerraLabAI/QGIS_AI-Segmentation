






from __future__ import annotations

from typing import Callable

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
)

from ...core.i18n import tr
from ...core.interaction_dials import cancel_watchdog_ms, lost_terminal_grace_s
from ...core.telemetry_errors import slot_guard
from .auto_run_cancel import (
    _CANCEL_WATCHDOG_MS,
    AutoRunCancelMixin,
)
from .auto_run_handoff import (
    AutoRunHandoffMixin,
)
from .auto_run_headless import (
    _HEADLESS_CANCEL_GRACE_MS,
    _HEADLESS_CANCEL_POLL_MS,
    AutoRunHeadlessMixin,
    _headless_cancel_grace_ms,
    _headless_cancel_poll_ms,
)
from .auto_run_preflight import (
    AutoRunPreflightMixin,
)
from .auto_run_progress import (
    _LOST_TERMINAL_GRACE_S,
    _SLOW_NOTICE_S,
    _STALL_CHECK_INTERVAL_MS,
    _STALL_TIMEOUT_S,
    _STALL_WIND_DOWN_DETACH,
    _WIND_DOWN_DETACH,
    AutoRunProgressMixin,
)
from .auto_run_start import (
    AutoRunStartMixin,
)
from .shared import (
    _RECALL_FLOOR,
    _RECALL_FLOOR_EXEMPLAR_ONLY,
    _provider_name_for_log,
    park_orphaned_worker,
)


class AutoRunMixin(
    AutoRunPreflightMixin,
    AutoRunStartMixin,
    AutoRunHandoffMixin,
    AutoRunProgressMixin,
    AutoRunCancelMixin,
    AutoRunHeadlessMixin,
):
    pass


__all__ = [
    "Callable",
    "Qgis",
    "QgsCoordinateReferenceSystem",
    "QgsCoordinateTransform",
    "QgsGeometry",
    "QgsMessageLog",
    "QgsProject",
    "QgsRasterLayer",
    "QgsRectangle",
    "tr",
    "cancel_watchdog_ms",
    "lost_terminal_grace_s",
    "slot_guard",
    "_RECALL_FLOOR",
    "_RECALL_FLOOR_EXEMPLAR_ONLY",
    "_provider_name_for_log",
    "park_orphaned_worker",
    "_CANCEL_WATCHDOG_MS",
    "_STALL_TIMEOUT_S",
    "_STALL_CHECK_INTERVAL_MS",
    "_SLOW_NOTICE_S",
    "_LOST_TERMINAL_GRACE_S",
    "_HEADLESS_CANCEL_POLL_MS",
    "_HEADLESS_CANCEL_GRACE_MS",
    "_headless_cancel_grace_ms",
    "_headless_cancel_poll_ms",
    "_WIND_DOWN_DETACH",
    "_STALL_WIND_DOWN_DETACH",
    "AutoRunMixin",
    "AutoRunPreflightMixin",
    "AutoRunStartMixin",
    "AutoRunHandoffMixin",
    "AutoRunProgressMixin",
    "AutoRunCancelMixin",
    "AutoRunHeadlessMixin",
]
