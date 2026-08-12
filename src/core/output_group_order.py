"""Keep the "AI Segmentation" group drawn above the imagery it was made from.

QGIS draws the layer tree top down: the first row of the Layers panel is the
last thing painted, so it covers everything under it. A committed run is a
vector outline over a raster, and a raster is opaque, so a group that sits
below the imagery paints under it and the canvas looks unchanged. The user
saves a run and sees nothing.

The rule this module enforces is the ordinary GIS one, not a plugin habit:
**vector results belong above the imagery they describe.** It is applied at
one moment only, when a run is committed, and only when something would
actually hide it. A group already in view is never moved, so a user who
organised their tree keeps it.

Main thread only (layer tree access), and best-effort throughout: every
failure leaves the tree exactly as it was rather than raising into a save that
has already written its file.
"""
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
    """Whether this node paints: it is checked AND so is every group above it.

    ``isVisible`` on its own reports the node's own checkbox, so a layer ticked
    inside an unticked group reads as visible and would count as something that
    hides our results when it paints nothing at all.
    """
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
    """Whether this top-level row paints raster pixels somewhere inside it."""
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
    """Whether visible imagery is painted over ``group``.

    Only rows ABOVE the group are looked at, because those are the only ones
    QGIS paints after it. Vector layers are not counted: they leave the ground
    they do not cover, so an outline under them stays readable, and moving the
    group for one would reshuffle a tree nobody asked us to touch.
    """
    try:
        children = root.children()
        position = children.index(group)
    except (ValueError, AttributeError, RuntimeError):
        return False
    return any(_holds_visible_raster(node) for node in children[:position])


def _raise_in_custom_order(project, group) -> None:
    """Move the group's layers to the front of a custom drawing order.

    A project with "Control rendering order" ticked draws from that list and
    ignores the tree entirely, so moving the row would look right in the panel
    and change nothing on the canvas.
    """
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
    """Move ``group`` to the first row of the Layers panel. Returns the new node.

    The layer tree has no move: a node is cloned, the copy is inserted where it
    belongs, and the original is dropped. The insert comes FIRST so the layers
    never spend an instant without a row, and the registry bridge is muted for
    the whole swap because it deletes a layer from the project when its last
    row disappears. Getting that order wrong loses the run the user just saved.

    Returns the original node untouched when the move could not be made, so a
    caller always holds a node that is really in the tree.
    """
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
    """Raise the results above the imagery when they would be hidden by it.

    The one entry point. Returns the group node to keep using: the same one
    when nothing moved, the new node when it did, because the original is
    destroyed by the move and touching it afterwards crashes QGIS.
    """
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
