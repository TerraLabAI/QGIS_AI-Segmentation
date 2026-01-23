
import sys
import os

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
LIBS_DIR = os.path.join(PLUGIN_DIR, 'libs')


def _isolate_dependencies():
    """Ensure plugin dependencies are loaded first."""

    packages_to_isolate = ['numpy', 'rasterio', 'pandas', 'torch', 'segment_anything']
    for pkg in packages_to_isolate:
        if pkg in sys.modules:
            del sys.modules[pkg]

    if os.path.exists(LIBS_DIR) and LIBS_DIR not in sys.path:
        sys.path.insert(0, LIBS_DIR)


_isolate_dependencies()

from .core.dependency_manager import init_packages_path
init_packages_path()


def classFactory(iface):
    from .ai_segmentation_plugin import AISegmentationPlugin
    return AISegmentationPlugin(iface)
