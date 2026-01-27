import os

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))


def _cleanup_old_installation():
    """Clean up old libs/ installation if it exists."""
    from .core.venv_manager import cleanup_old_libs
    try:
        cleanup_old_libs()
    except Exception:
        pass


_cleanup_old_installation()


def classFactory(iface):
    from .ui.ai_segmentation_plugin import AISegmentationPlugin
    return AISegmentationPlugin(iface)
