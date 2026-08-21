"""The QGIS Processing provider that publishes AI segmentation to everything.

Registering here is what makes the plugin reachable from the Processing
Toolbox, the Graphical Modeler, batch mode, PyQGIS scripts and every
third-party MCP server that exposes a generic "run a processing algorithm"
tool. Nothing else has to be written on either side.
"""
from __future__ import annotations

import os

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .algorithm_segment_point import SegmentPointAlgorithm
from .algorithm_segment_zone import SegmentZoneAlgorithm
from .algorithm_segmentation_status import SegmentationStatusAlgorithm
from .algorithm_support import FACADE_MISSING_MESSAGE, segmentation_facade

# Published interface: an algorithm is addressed as "terralab:<name>". Callers
# hardcode it, so this string never changes.
TERRALAB_PROVIDER_ID = "terralab"


class TerraLabProcessingProvider(QgsProcessingProvider):
    """Holds the AI segmentation algorithms and their shared identity."""

    def id(self):
        return TERRALAB_PROVIDER_ID

    def name(self):
        return "TerraLab AI"

    def longName(self):
        return "TerraLab AI segmentation for QGIS"

    def icon(self):
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "resources", "icons", "icon.png",
        )
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return super().icon()

    def loadAlgorithms(self):
        self.addAlgorithm(SegmentationStatusAlgorithm())
        self.addAlgorithm(SegmentZoneAlgorithm())
        self.addAlgorithm(SegmentPointAlgorithm())

    def isActive(self):
        """Grey the whole provider out when the plugin is gone.

        QgsProcessingProvider has no canExecute hook: the pair a provider is
        asked for is isActive, a plain bool, and warningMessage, the sentence
        shown beside it. An override of a name the base class does not carry is
        never called, so the provider stayed active with the plugin unloaded.
        """
        return segmentation_facade() is not None

    def warningMessage(self):
        """Why the provider is greyed out, in the words the algorithms use."""
        if segmentation_facade() is None:
            return FACADE_MISSING_MESSAGE
        return ""
