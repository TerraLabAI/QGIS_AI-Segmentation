






from __future__ import annotations

import os

from qgis.core import QgsProcessingProvider

from .algorithm_segment_point import SegmentPointAlgorithm
from .algorithm_segment_zone import SegmentZoneAlgorithm
from .algorithm_segmentation_status import SegmentationStatusAlgorithm
from .algorithm_support import FACADE_MISSING_MESSAGE, segmentation_facade



TERRALAB_PROVIDER_ID = "terralab"


_PROVIDER_ICON = None


class TerraLabProcessingProvider(QgsProcessingProvider):


    def id(self):
        return TERRALAB_PROVIDER_ID

    def name(self):
        return "TerraLab AI"

    def longName(self):
        return "TerraLab AI segmentation for QGIS"

    def icon(self):



        global _PROVIDER_ICON
        if _PROVIDER_ICON is None:
            from ..core.qimage_strips import icon_from_file
            icon_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "resources", "icons", "icon.png",
            )
            _PROVIDER_ICON = icon_from_file(icon_path) if os.path.exists(icon_path) else False
        if _PROVIDER_ICON is False or _PROVIDER_ICON.isNull():
            return super().icon()
        return _PROVIDER_ICON

    def loadAlgorithms(self):
        self.addAlgorithm(SegmentationStatusAlgorithm())
        self.addAlgorithm(SegmentZoneAlgorithm())
        self.addAlgorithm(SegmentPointAlgorithm())

    def isActive(self):







        return segmentation_facade() is not None

    def warningMessage(self):

        if segmentation_facade() is None:
            return FACADE_MISSING_MESSAGE
        return ""
