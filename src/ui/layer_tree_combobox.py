"""Raster layer combo box that mirrors the QGIS Layer panel tree.

Reusable widget: shows visible raster layers grouped by their layer tree
structure.  Non-selectable group headers provide visual hierarchy.
Auto-refreshes on layer add/remove, visibility, and reorder.
"""
from __future__ import annotations

from qgis.core import QgsLayerTree, QgsProject, QgsRasterLayer
from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal
from qgis.PyQt.QtWidgets import QComboBox, QStyle, QStyledItemDelegate, QStyleOptionViewItem

from ..core.qt_compat import safe_disconnect


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


# Layer-tree group names whose rasters never win the DEFAULT pick: the AI Edit
# plugin parks its generated rasters in an "AI-Edit" group at the top of the
# tree, right over the map view, so the "topmost raster in view" heuristic
# always grabbed one of them - and an AI Edit output is rarely the imagery the
# user wants to segment. They stay listed and selectable; they just lose the
# automatic pick unless they are the only rasters in the project.
_DEPRIORITIZED_GROUP_NAMES = {"ai-edit", "ai edit"}


def _is_deprioritized_group(name: str) -> bool:
    return (name or "").strip().lower() in _DEPRIORITIZED_GROUP_NAMES


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
        self._deprioritized_ids = set()  # rasters under an AI Edit group
        self._refreshing = False
        self._frozen = False  # when True, ignore layer-tree changes (locked flow)
        # No deliberate choice yet, so the pick stays free to follow the map view.
        # Any user pick, or any setLayer() from the code, ends that for good.
        self._pick_is_automatic = True
        self._view_tracking = True  # suspended while a segmentation session runs

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

        # Follow the map view: panning off one orthophoto and onto another is a
        # layer change the user should not have to make by hand. Long debounce,
        # since extentsChanged fires on every step of a pan.
        self._view_timer = QTimer(self)
        self._view_timer.setSingleShot(True)
        self._view_timer.timeout.connect(self._repick_for_view)
        try:
            from qgis.utils import iface
            iface.mapCanvas().extentsChanged.connect(self._schedule_view_repick)
        except Exception:  # nosec B110
            pass  # headless (tests): no canvas to follow

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
        """Programmatically select a layer. Counts as a deliberate choice, so
        the combo stops following the map view from here on."""
        if layer is None:
            return
        target_id = layer.id()
        for i in range(self.count()):
            if self.itemData(i) == target_id:
                self._pick_is_automatic = False
                self.setCurrentIndex(i)
                return

    def count_layers(self):
        """Return the number of selectable (non-header) items."""
        return len(self._layer_ids)

    def set_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze auto-refresh. While frozen, layer-tree changes (add,
        remove, visibility, reorder) are ignored so the current list + selection
        stay put; used while the Automatic flow has a locked source raster, where
        hiding a layer to peek underneath must not drop the locked source or
        re-pick another. Unfreezing resyncs once (the tree may have changed)."""
        if frozen == self._frozen:
            return
        self._frozen = frozen
        if not frozen:
            self._refresh()
            # _refresh() restores the locked raster before it ever looks at the
            # map, so a view that moved during the run needs its own pass.
            if self._pick_is_automatic:
                self._schedule_view_repick()

    def set_view_tracking(self, enabled: bool) -> None:
        """Stop or resume following the map view.

        A live segmentation session owns its raster: re-picking under it on a
        simple pan would throw away the work in progress, and pop the "Change
        Layer?" prompt to say so. Ending the session resumes tracking, unless
        the user has meanwhile picked a raster by hand.
        """
        self._view_tracking = enabled
        # The view may have moved a long way during the session, and nothing
        # else re-reads it: without this pass the next run would open on the
        # raster the last one used, wherever the user has since gone.
        if enabled and self._pick_is_automatic:
            self._schedule_view_repick()

    def cleanup(self):
        """Disconnect the project signals this combo hooked in __init__.

        One guard per signal, never one around the batch. All six used to share
        a single try block, so the first failure skipped every disconnect after
        it: one slot that was never connected (a second cleanup() call, or a
        project reload that swapped the layer tree root out from under us) left
        the four root signals wired to _schedule_refresh. Nothing else cleans
        this widget up, so those connections then fired the refresh timer into
        a destroyed combo box every time the user touched the Layers panel.
        """
        proj = QgsProject.instance()
        safe_disconnect(proj, "layersAdded", self._schedule_refresh)
        safe_disconnect(proj, "layersRemoved", self._schedule_refresh)
        # Re-read the root here: the connect side took it from the same call,
        # and fetching it inside the old try block meant a raise above skipped
        # the lookup entirely.
        root = proj.layerTreeRoot()
        for signal_name in (
            "visibilityChanged", "addedChildren", "removedChildren", "nameChanged"
        ):
            safe_disconnect(root, signal_name, self._schedule_refresh)
        safe_disconnect(self, "currentIndexChanged", self._on_index_changed)
        try:
            from qgis.utils import iface
            safe_disconnect(iface.mapCanvas(), "extentsChanged", self._schedule_view_repick)
        except Exception:  # nosec B110
            pass  # headless (tests): nothing was connected
        for timer in (self._refresh_timer, self._view_timer):
            try:
                timer.stop()
            except RuntimeError:
                pass

    # -- internals --

    def _schedule_refresh(self, *_args):
        """Debounced refresh (100 ms)."""
        self._refresh_timer.start(100)

    def _refresh(self):
        """Rebuild the combo items from the layer tree."""
        # Frozen while the Automatic flow has a locked source raster: a layer-tree
        # change (add/remove/visibility/reorder) must NOT rebuild the list or
        # re-pick the selection, so hiding a layer keeps the locked source intact.
        if self._frozen:
            return
        self._refreshing = True
        prev_id = self._current_layer_id
        try:
            self.clear()
            self._layer_ids = []
            self._deprioritized_ids = set()

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

            if not restored:
                best_idx = self._best_index_for_view()
                if best_idx is not None:
                    self.setCurrentIndex(best_idx)
        finally:
            # Never leave the flag stuck: a raise inside _traverse used to leave
            # the combo permanently deaf to the user's own picks.
            self._refreshing = False

        # Emit if selection actually changed
        new_layer = self.currentLayer()
        new_id = new_layer.id() if new_layer else None
        if new_id != prev_id:
            self._current_layer_id = new_id
            self.layerChanged.emit(new_layer)

    def _best_index_for_view(self):
        """Index of the raster that best matches the current map view, or None.

        Candidates are the NON-deprioritized rasters first, so an AI Edit output
        only ever wins when it is the only raster around. Ranking is in
        ``raster_view_pick``; on a headless run, with no canvas to read, the
        topmost raster stands.
        """
        selectable = [i for i in range(self.count()) if self.itemData(i) is not None]
        preferred = [i for i in selectable if self.itemData(i) not in self._deprioritized_ids]
        pool = preferred or selectable
        if not pool:
            return None
        try:
            from qgis.utils import iface

            from .raster_view_pick import rank_raster_for_view
            canvas = iface.mapCanvas()
            view_extent = canvas.extent()
            view_crs = canvas.mapSettings().destinationCrs()
            active_layer = iface.activeLayer()
            active_id = active_layer.id() if active_layer is not None else None
        except Exception:  # nosec B110
            return pool[0]

        project = QgsProject.instance()
        best_idx = None
        best_key = None
        for tree_order, i in enumerate(pool):
            layer = project.mapLayer(self.itemData(i))
            if layer is None:
                continue
            try:
                key = rank_raster_for_view(
                    layer, view_extent, view_crs, tree_order, active_id)
            except Exception:
                key = None
            if key is None:
                continue
            if best_key is None or key > best_key:
                best_key, best_idx = key, i
        return best_idx if best_idx is not None else pool[0]

    def _schedule_view_repick(self, *_args):
        """Debounced re-pick after a pan or zoom (600 ms)."""
        self._view_timer.start(600)

    def _repick_for_view(self):
        """Move the selection to the raster the new map view is showing.

        Only while the pick is still the plugin's own guess: once the user has
        chosen a raster, or a session has claimed one, the view stops deciding.
        """
        if self._frozen or not self._pick_is_automatic or not self._view_tracking:
            return
        best_idx = self._best_index_for_view()
        if best_idx is None or best_idx == self.currentIndex():
            return
        self._refreshing = True
        try:
            self.setCurrentIndex(best_idx)
        finally:
            self._refreshing = False
        layer = self.currentLayer()
        new_id = layer.id() if layer else None
        if new_id != self._current_layer_id:
            self._current_layer_id = new_id
            self.layerChanged.emit(layer)

    def _has_visible_rasters(self, node):
        """Check if a tree node has any visible raster layer descendants."""
        for child in node.children():
            if QgsLayerTree.isLayer(child):
                layer = child.layer()
                if layer and isinstance(layer, QgsRasterLayer) and layer.isValid() and child.isVisible():
                    return True
            elif QgsLayerTree.isGroup(child):
                if child.isVisible() and self._has_visible_rasters(child):
                    return True
        return False

    def _traverse(self, node, depth=0, deprioritized=False):
        """Recursively walk the layer tree and add items. ``deprioritized``
        marks the whole subtree of an AI Edit output group: its rasters are
        listed as usual but recorded so the default pick skips them."""
        from qgis.core import QgsApplication

        visible_children = []
        for child in node.children():
            if QgsLayerTree.isGroup(child):
                if child.isVisible() and self._has_visible_rasters(child):
                    visible_children.append(child)
            elif QgsLayerTree.isLayer(child):
                layer = child.layer()
                if layer and isinstance(layer, QgsRasterLayer) and layer.isValid() and child.isVisible():
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
                self._traverse(
                    child, depth + 1,
                    deprioritized or _is_deprioritized_group(child.name()))

            elif QgsLayerTree.isLayer(child):
                layer = child.layer()
                # getThemeIcon works on all supported QGIS (3.0+), unlike
                # QgsIconUtils.iconRaster() which only exists since 3.20.
                layer_icon = QgsApplication.getThemeIcon("/mIconRaster.svg")
                self.addItem(layer_icon, layer.name(), layer.id())
                idx = self.count() - 1
                item = self.model().item(idx)
                if item:
                    item.setData(depth, depth_role)
                self._layer_ids.append(layer.id())
                if deprioritized:
                    self._deprioritized_ids.add(layer.id())

    def _on_index_changed(self, index):
        """Handle user selection change."""
        if self._refreshing:
            return
        # The user opened the drop-down and chose: the map view no longer decides.
        self._pick_is_automatic = False
        layer = self.currentLayer()
        layer_id = layer.id() if layer else None
        if layer_id != self._current_layer_id:
            self._current_layer_id = layer_id
            self.layerChanged.emit(layer)
