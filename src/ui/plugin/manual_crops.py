










from __future__ import annotations

from .manual_crops_encode import (
    ENCODE_LOCK_CEILING_S,
    ENCODE_WATCHDOG_INTERVAL_MS,
    ManualCropsEncodeMixin,
)
from .manual_crops_extract import ManualCropsExtractMixin
from .manual_crops_geometry import ManualCropsGeometryMixin
from .manual_crops_transport import (
    CropReadWorker,
    DirectTileFetchWorker,
    ManualCropsTransportMixin,
)


class ManualCropsMixin(
    ManualCropsGeometryMixin,
    ManualCropsExtractMixin,
    ManualCropsEncodeMixin,
    ManualCropsTransportMixin,
):
    pass


__all__ = [
    "ManualCropsMixin",
    "ENCODE_WATCHDOG_INTERVAL_MS",
    "ENCODE_LOCK_CEILING_S",
    "CropReadWorker",
    "DirectTileFetchWorker",
]
