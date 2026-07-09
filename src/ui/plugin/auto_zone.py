






from __future__ import annotations

from .auto_zone_canvas import (
    AutoZoneCanvasMixin,
)
from .auto_zone_crs import (
    AutoZoneCrsMixin,
)
from .auto_zone_draw import (
    AutoZoneDrawMixin,
)
from .auto_zone_free_fit import (
    AutoZoneFreeFitMixin,
)
from .auto_zone_grid import (
    _ZONE_GRID_CACHE_SIZE,
    AutoZoneGridMixin,
)
from .auto_zone_history import (
    AutoZoneHistoryMixin,
)


class AutoZoneMixin(
    AutoZoneDrawMixin,
    AutoZoneFreeFitMixin,
    AutoZoneHistoryMixin,
    AutoZoneGridMixin,
    AutoZoneCrsMixin,
    AutoZoneCanvasMixin,
):
    pass


__all__ = [
    "AutoZoneMixin",
    "_ZONE_GRID_CACHE_SIZE",
]
