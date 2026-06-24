import os

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))


def _cleanup_old_installation():
    """Clean up old libs/ installation if it exists."""
    try:
        from .src.core.venv_manager import cleanup_old_libs

        cleanup_old_libs()
    except (ImportError, Exception):
        pass


try:
    _cleanup_old_installation()
except Exception:  # nosec B110 - best-effort cleanup, must never block plugin load
    pass


def classFactory(iface):
    from .src.ui.ai_segmentation_plugin import AISegmentationPlugin

    return AISegmentationPlugin(iface)
