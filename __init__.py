"""AI Segmentation by TerraLab: QGIS plugin entry point (classFactory)."""

import os

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))


# The legacy libs/ tree from the pre-venv layout is removed by the startup
# check worker, off the UI thread. Deleting it here would run a multi-GB,
# tens-of-thousands-of-files delete during plugin import, freezing the QGIS
# splash screen with no message and no way to cancel.


def classFactory(iface):
    from .src.ui.ai_segmentation_plugin import AISegmentationPlugin

    return AISegmentationPlugin(iface)
