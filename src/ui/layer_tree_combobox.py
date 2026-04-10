"""Raster layer combo box that mirrors the QGIS Layer panel tree.

Reusable widget: shows visible raster layers grouped by their layer tree
structure.  Non-selectable group headers provide visual hierarchy.
Auto-refreshes on layer add/remove, visibility, and reorder.
"""

from qgis.core import QgsLayerTree, QgsProject
from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal
from qgis.PyQt.QtWidgets import QComboBox, QStyle, QStyledItemDelegate, QStyleOptionViewItem


class _IndentDelegate(QStyledItemDelegate):
    """Delegate that shifts icon+text to the right based on a depth role."""

    DEPTH_ROLE = Qt.ItemDataRole.UserRole + 100
    INDENT_PX = 20

    def paint(self, painter, option, index):
        depth = index.data(self.DEPTH_ROLE) or 0
        shift = depth * self.INDENT_PX
        if shift:
            shifted = QStyleOptionViewItem(option)
            shifted.rect = shifted.rect.adjusted(shift, 0, 0, 0)
            super().paint(painter, shifted, index)
        else:
            super().paint(painter, option, index)

    def sizeHint(self, option, index):
        hint = super().sizeHint(option, index)
        depth = index.data(self.DEPTH_ROLE) or 0
        hint.setWidth(hint.width() + depth * self.INDENT_PX)
        return hint


class LayerTreeComboBox(QComboBox):
    """Drop-down that mirrors the QGIS Layer panel order with group headers.

    Only visible raster layers are selectable.  Group names appear as
    non-selectable, indented headers.  Auto-refreshes on layer add/remove,
    visibility toggle, and tree reorder.
    """

    layerChanged = pyqtSignal(object)  # emits QgsMapLayer or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_layer_id = None  # track selection across refreshes
        self._layer_ids = []  # ordered list of selectable layer IDs
        self._refreshing = False

        from qgis.PyQt.QtCore import QSize
        self.setIconSize(QSize(16, 16))
        self.setItemDelegate(_IndentDelegate(self))
        self.currentIndexChanged.connect(self._on_index_changed)

        # Connect to project signals for auto-refresh
        proj = QgsProject.instance()
        proj.layersAdded.connect(self._schedule_refresh)
        proj.layersRemoved.connect(self._schedule_refresh)

        root = proj.layerTreeRoot()
        root.visibilityChanged.connect(self._schedule_refresh)
        root.addedChildren.connect(self._schedule_refresh)
        root.removedChildren.connect(self._schedule_refresh)
        root.nameChanged.connect(self._schedule_refresh)

        # Debounce refresh (group visibility fires per-node)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._refresh)

        # Initial population
        self._refresh()

    # -- public API (matches QgsMapLayerComboBox interface) --

    def currentLayer(self):
        """Return the currently selected QgsMapLayer, or None."""
        idx = self.currentIndex()
        if idx < 0:
            return None
        layer_id = self.itemData(idx)
        if layer_id is None:
            return None
        return QgsProject.instance().mapLayer(layer_id)

    def setLayer(self, layer):
        """Programmatically select a layer."""
        if layer is None:
            return
        target_id = layer.id()
        for i in range(self.count()):
            if self.itemData(i) == target_id:
                self.setCurrentIndex(i)
                return

    def count_layers(self):
        """Return the number of selectable (non-header) items."""
        return len(self._layer_ids)

    def cleanup(self):
        """Disconnect project signals."""
        try:
            proj = QgsProject.instance()
            proj.layersAdded.disconnect(self._schedule_refresh)
            proj.layersRemoved.disconnect(self._schedule_refresh)
            root = proj.layerTreeRoot()
            root.visibilityChanged.disconnect(self._schedule_refresh)
            root.addedChildren.disconnect(self._schedule_refresh)
            root.removedChildren.disconnect(self._schedule_refresh)
            root.nameChanged.disconnect(self._schedule_refresh)
        except (TypeError, RuntimeError):
            pass
        try:
            self.currentIndexChanged.disconnect(self._on_index_changed)
        except (TypeError, RuntimeError):
            pass
        try:
            self._refresh_timer.stop()
        except RuntimeError:
            pass

    # -- internals --

    def _schedule_refresh(self, *_args):
        """Debounced refresh (100 ms)."""
        self._refresh_timer.start(100)

    def _refresh(self):
        """Rebuild the combo items from the layer tree."""
        self._refreshing = True
        prev_id = self._current_layer_id
        self.clear()
        self._layer_ids = []

        root = QgsProject.instance().layerTreeRoot()
        self._traverse(root)

        # Restore previous selection
        restored = False
        if prev_id:
            for i in range(self.count()):
                if self.itemData(i) == prev_id:
                    self.setCurrentIndex(i)
                    restored = True
                    break

        # If not restored, pick best raster: prefer one visible in current map view
        if not restored:
            best_idx = None
            try:
                from qgis.utils import iface
                canvas_extent = iface.mapCanvas().extent()
                canvas_crs = iface.mapCanvas().mapSettings().destinationCrs()
                from qgis.core import QgsCoordinateTransform
                for i in range(self.count()):
                    layer_id = self.itemData(i)
                    if layer_id is None:
                        continue
                    if best_idx is None:
                        best_idx = i  # fallback: first selectable
                    layer = QgsProject.instance().mapLayer(layer_id)
                    if layer and layer.extent().isEmpty():
                        continue
                    # Transform layer extent to canvas CRS for comparison
                    try:
                        xform = QgsCoordinateTransform(
                            layer.crs(), canvas_crs, QgsProject.instance())
                        layer_extent = xform.transformBoundingBox(layer.extent())
                    except Exception:
                        layer_extent = layer.extent()
                    if layer_extent.intersects(canvas_extent):
                        best_idx = i
                        break  # topmost visible raster in current view
            except Exception:
                # Fallback: just pick first selectable
                for i in range(self.count()):
                    if self.itemData(i) is not None:
                        best_idx = i
                        break
            if best_idx is not None:
                self.setCurrentIndex(best_idx)

        self._refreshing = False

        # Emit if selection actually changed
        new_layer = self.currentLayer()
        new_id = new_layer.id() if new_layer else None
        if new_id != prev_id:
            self._current_layer_id = new_id
            self.layerChanged.emit(new_layer)

    def _has_visible_rasters(self, node):
        """Check if a tree node has any visible raster layer descendants."""
        for child in node.children():
            if QgsLayerTree.isLayer(child):
                layer = child.layer()
                if (layer and layer.type() == layer.RasterLayer
                        and child.isVisible()):  # noqa: W503
                    return True
            elif QgsLayerTree.isGroup(child):
                if child.isVisible() and self._has_visible_rasters(child):
                    return True
        return False

    def _traverse(self, node, depth=0):
        """Recursively walk the layer tree and add items."""
        from qgis.core import QgsApplication, QgsIconUtils

        visible_children = []
        for child in node.children():
            if QgsLayerTree.isGroup(child):
                if child.isVisible() and self._has_visible_rasters(child):
                    visible_children.append(child)
            elif QgsLayerTree.isLayer(child):
                layer = child.layer()
                if (layer and layer.type() == layer.RasterLayer
                        and child.isVisible()):  # noqa: W503
                    visible_children.append(child)

        depth_role = _IndentDelegate.DEPTH_ROLE
        for child in visible_children:
            if QgsLayerTree.isGroup(child):
                folder_icon = QgsApplication.getThemeIcon("/mActionFolder.svg")
                if folder_icon.isNull():
                    folder_icon = self.style().standardIcon(
                        QStyle.StandardPixmap.SP_DirIcon)
                self.addItem(folder_icon, child.name())
                idx = self.count() - 1
                item = self.model().item(idx)
                if item:
                    item.setEnabled(False)
                    item.setSelectable(False)
                    item.setData(depth, depth_role)
                self._traverse(child, depth + 1)

            elif QgsLayerTree.isLayer(child):
                layer = child.layer()
                layer_icon = QgsIconUtils.iconRaster()
                self.addItem(layer_icon, layer.name(), layer.id())
                idx = self.count() - 1
                item = self.model().item(idx)
                if item:
                    item.setData(depth, depth_role)
                self._layer_ids.append(layer.id())

    def _on_index_changed(self, index):
        """Handle user selection change."""
        if self._refreshing:
            return
        layer = self.currentLayer()
        layer_id = layer.id() if layer else None
        if layer_id != self._current_layer_id:
            self._current_layer_id = layer_id
            self.layerChanged.emit(layer)
