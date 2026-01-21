


from .core.dependency_manager import init_packages_path
init_packages_path()


def classFactory(iface):
    
    from .ai_segmentation_plugin import AISegmentationPlugin
    return AISegmentationPlugin(iface)
