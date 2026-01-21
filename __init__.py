

import os

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))








def classFactory(iface):
    from .src.ui.ai_segmentation_plugin import AISegmentationPlugin

    return AISegmentationPlugin(iface)
