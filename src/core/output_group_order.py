

















from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsLayerTree,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
)

_LOG_TAG = "AI Segmentation"


def _node_is_effectively_visible(node) -> bool:






    current = node
    while current is not None:
        try:
            if not current.itemVisibilityChecked():
                return False
        except AttributeError:
            return True
        current = current.parent()
    return True


def _holds_visible_raster(node) -> bool:

    try:
        if QgsLayerTree.isLayer(node):
            layer = node.layer()
            return (isinstance(layer, QgsRasterLayer)
                    and layer.isValid()
                    and _node_is_effectively_visible(node))
        for child in node.findLayers():
            layer = child.layer()
            if (isinstance(layer, QgsRasterLayer)
                    and layer.isValid()
                    and _node_is_effectively_visible(child)):
                return True
    except (AttributeError, RuntimeError):
        return False
    return False


def imagery_covers_group(root, group) -> bool:







    try:
        children = root.children()
        position = children.index(group)
    except (ValueError, AttributeError, RuntimeError):
        return False
    return any(_holds_visible_raster(node) for node in children[:position])


def _raise_in_custom_order(project, group) -> None:






    root = project.layerTreeRoot()
    if not root.hasCustomLayerOrder():
        return
    ours = [node.layer() for node in group.findLayers() if node.layer()]
    if not ours:
        return
    ids = {layer.id() for layer in ours}
    rest = [layer for layer in root.customLayerOrder() if layer.id() not in ids]
    root.setCustomLayerOrder(ours + rest)


def raise_group_to_top(root, group):











    project = QgsProject.instance()
    bridge = None
    try:
        bridge = project.layerTreeRegistryBridge()
    except AttributeError:
        bridge = None
    try:
        if bridge is not None:
            bridge.setEnabled(False)
        clone = group.clone()
        expanded = group.isExpanded()
        checked = group.itemVisibilityChecked()
        root.insertChildNode(0, clone)
        root.removeChildNode(group)
        clone.setExpanded(expanded)
        clone.setItemVisibilityChecked(checked)
    except (AttributeError, RuntimeError, TypeError) as err:
        QgsMessageLog.logMessage(
            f"Could not move the results above the imagery: {err}",
            _LOG_TAG, level=Qgis.MessageLevel.Warning)
        return group
    finally:
        if bridge is not None:
            try:
                bridge.setEnabled(True)
            except (AttributeError, RuntimeError):
                pass
    _raise_in_custom_order(project, clone)
    return clone


def keep_group_above_imagery(group):






    try:
        root = QgsProject.instance().layerTreeRoot()
    except (AttributeError, RuntimeError):
        return group
    if group is None:
        return group
    try:
        if not imagery_covers_group(root, group):
            _raise_in_custom_order(QgsProject.instance(), group)
            return group
    except (AttributeError, RuntimeError):
        return group
    return raise_group_to_top(root, group)
