# AI Segmentation - AI-powered segmentation for QGIS
# Copyright (C) 2025 QGIS AI-Segmentation Team
# Licensed under MIT License

"""
AI Segmentation Plugin for QGIS

This plugin requires external Python packages (numpy, onnxruntime) that are
not bundled with QGIS. On first startup, an installation dialog will assist
with installing these dependencies.

The packages are installed to a local directory within the plugin folder
to avoid conflicts with the system Python environment.
"""

# Initialize the packages directory early, before any imports that might need them
# This is done here to ensure the path is set up before classFactory is called
from .core.dependency_manager import init_packages_path
init_packages_path()


def classFactory(iface):
    """
    Load the AI Segmentation plugin class.

    This is the entry point for QGIS to load the plugin.
    The plugin is designed to load even if dependencies are missing,
    and will prompt the user to install them when the panel is opened.

    Args:
        iface: QgsInterface instance providing access to QGIS components

    Returns:
        AISegmentationPlugin instance
    """
    from .ai_segmentation_plugin import AISegmentationPlugin
    return AISegmentationPlugin(iface)
