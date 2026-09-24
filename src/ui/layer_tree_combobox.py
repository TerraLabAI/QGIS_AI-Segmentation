





from __future__ import annotations

from qgis.core import QgsLayerTree, QgsProject, QgsRasterLayer
from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal
from qgis.PyQt.QtWidgets import QComboBox, QStyle, QStyledItemDelegate, QStyleOptionViewItem

from ..core.qt_compat import safe_disconnect
from ..core.server_dials import dial_in_range


class _IndentDelegate(QStyledItemDelegate):


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








_DEPRIORITIZED_GROUP_NAMES = {"ai-edit", "ai edit"}


def _is_deprioritized_group(name: str) -> bool:
    return (name or "").strip().lower() in _DEPRIORITIZED_GROUP_NAMES


def _raster_tree_nodes(root=None):

    if root is None:
        root = QgsProject.instance().layerTreeRoot()
    nodes = []
    for node in root.findLayers():
        try:
            layer = node.layer()
        except RuntimeError:
            continue
        if layer is not None and isinstance(layer, QgsRasterLayer):
            nodes.append(node)
    return nodes


def project_raster_presence() -> tuple[int, int]:




    visible = hidden = 0
    for node in _raster_tree_nodes():
        try:
            if not node.layer().isValid():
                continue
            if node.isVisible():
                visible += 1
            else:
                hidden += 1
        except RuntimeError:
            continue
    return visible, hidden


def reveal_hidden_rasters() -> int:



    revealed = 0
    for node in _raster_tree_nodes():
        try:
            if not node.layer().isValid() or node.isVisible():
                continue
            node.setItemVisibilityChecked(True)
            parent = node.parent()
            while parent is not None:
                if not parent.itemVisibilityChecked():
                    parent.setItemVisibilityChecked(True)
                parent = parent.parent()
            revealed += 1
        except RuntimeError:
            continue
    return revealed


class LayerTreeComboBox(QComboBox):







    layerChanged = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_layer_id = None
        self._layer_ids = []
        self._deprioritized_ids = set()
        self._refreshing = False
        self._frozen = False


        self._pick_is_automatic = True
        self._view_tracking = True



        self._watched_invalid_ids = set()

        from qgis.PyQt.QtCore import QSize
        self.setIconSize(QSize(16, 16))


        if hasattr(self, "setPlaceholderText"):
            from ..core.i18n import tr
            self.setPlaceholderText(tr("Choose the imagery to segment"))
        self.setItemDelegate(_IndentDelegate(self))
        self.currentIndexChanged.connect(self._on_index_changed)


        proj = QgsProject.instance()
        proj.layersAdded.connect(self._schedule_refresh)
        proj.layersRemoved.connect(self._schedule_refresh)

        root = proj.layerTreeRoot()
        root.visibilityChanged.connect(self._schedule_refresh)
        root.addedChildren.connect(self._schedule_refresh)
        root.removedChildren.connect(self._schedule_refresh)
        root.nameChanged.connect(self._schedule_refresh)


        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._refresh)




        self._view_timer = QTimer(self)
        self._view_timer.setSingleShot(True)
        self._view_timer.timeout.connect(self._repick_for_view)
        try:
            from qgis.utils import iface
            iface.mapCanvas().extentsChanged.connect(self._schedule_view_repick)
        except Exception:  # nosec B110
            pass


        self._refresh()



    def currentLayer(self):

        idx = self.currentIndex()
        if idx < 0:
            return None
        layer_id = self.itemData(idx)
        if layer_id is None:
            return None
        return QgsProject.instance().mapLayer(layer_id)

    def setLayer(self, layer):


        if layer is None:
            return
        target_id = layer.id()
        for i in range(self.count()):
            if self.itemData(i) == target_id:
                self._pick_is_automatic = False
                self.setCurrentIndex(i)
                return

    def count_layers(self):

        return len(self._layer_ids)

    def set_frozen(self, frozen: bool) -> None:





        if frozen == self._frozen:
            return
        self._frozen = frozen
        if not frozen:
            self._refresh()


            if self._pick_is_automatic:
                self._schedule_view_repick()

    def set_view_tracking(self, enabled: bool) -> None:







        self._view_tracking = enabled



        if enabled and self._pick_is_automatic:
            self._schedule_view_repick()

    def cleanup(self):










        proj = QgsProject.instance()
        safe_disconnect(proj, "layersAdded", self._schedule_refresh)
        safe_disconnect(proj, "layersRemoved", self._schedule_refresh)



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
            pass
        for timer in (self._refresh_timer, self._view_timer):
            try:
                timer.stop()
            except RuntimeError:
                pass



    def _schedule_refresh(self, *_args):

        self._refresh_timer.start(
            dial_in_range("tuning.ui.tree_refresh_debounce_ms", 100, 25, 500))

    def _refresh(self):




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


            self._refreshing = False


        new_layer = self.currentLayer()
        new_id = new_layer.id() if new_layer else None
        if new_id != prev_id:
            self._current_layer_id = new_id
            self.layerChanged.emit(new_layer)

    def _best_index_for_view(self):







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

        self._view_timer.start(dial_in_range("tuning.ui.view_repick_debounce_ms", 600, 100, 3000))

    def _repick_for_view(self):





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

    def _watch_invalid_raster(self, layer) -> None:


        try:
            layer_id = layer.id()
            if layer_id in self._watched_invalid_ids:
                return
            self._watched_invalid_ids.add(layer_id)
            layer.dataSourceChanged.connect(self._schedule_refresh)
        except (RuntimeError, AttributeError, TypeError):
            pass  # nosec B110

    def _has_visible_rasters(self, node):

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
                elif layer and isinstance(layer, QgsRasterLayer) and not layer.isValid():
                    self._watch_invalid_raster(layer)

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

        if self._refreshing:
            return

        self._pick_is_automatic = False
        layer = self.currentLayer()
        layer_id = layer.id() if layer else None
        if layer_id != self._current_layer_id:
            self._current_layer_id = layer_id
            self.layerChanged.emit(layer)
