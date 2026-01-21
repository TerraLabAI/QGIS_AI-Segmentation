# AI Segmentation - AI-powered segmentation for QGIS
# Copyright (C) 2025 QGIS AI-Segmentation Team
# Licensed under MIT License


def classFactory(iface):
    """
    Load the AI Segmentation plugin class.

    This is the entry point for QGIS to load the plugin.

    Args:
        iface: QgsInterface instance providing access to QGIS components

    Returns:
        AISegmentationPlugin instance
    """
    from .ai_segmentation_plugin import AISegmentationPlugin
    return AISegmentationPlugin(iface)
