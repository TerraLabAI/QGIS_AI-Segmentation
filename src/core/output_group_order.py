

















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
    order = root.customLayerOrder()
    raster_positions = []
    for index, layer in enumerate(order):
        if not isinstance(layer, QgsRasterLayer) or not layer.isValid():
            continue
        node = root.findLayer(layer.id())
        if node is not None and _node_is_effectively_visible(node):
            raster_positions.append(index)
    if not raster_positions:
        return
    first_raster = raster_positions[0]
    positions = {layer.id(): i for i, layer in enumerate(order)}
    if all(positions.get(layer.id(), len(order)) < first_raster for layer in ours):
        return
    first_raster_id = order[first_raster].id()
    rest = [layer for layer in order if layer.id() not in ids]
    insertion = next(i for i, layer in enumerate(rest) if layer.id() == first_raster_id)
    root.setCustomLayerOrder(rest[:insertion] + ours + rest[insertion:])


def raise_group_to_top(root, group):











    project = QgsProject.instance()
    bridge = None
    try:
        bridge = project.layerTreeRegistryBridge()
    except AttributeError:
        bridge = None
    bridge_was_enabled = True
    try:
        if bridge is not None:
            bridge_was_enabled = bridge.isEnabled()
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
                bridge.setEnabled(bridge_was_enabled)
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
