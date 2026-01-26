





from __future__ import annotations

from qgis.core import QgsPointXY, QgsVectorLayer
from qgis.PyQt.QtCore import QEvent, QObject, Qt
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QTextEdit,
)

from ..core.server_dials import dial_in_range


class ShortcutFilter(QObject):








    def __init__(self, plugin, parent=None):
        super().__init__(parent)
        self._plugin = plugin

    def _typing_in_text_field(self) -> bool:


        app = QApplication.instance()
        if not app:
            return True
        focused = app.focusWidget()
        return isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit,
                                    QSpinBox, QDoubleSpinBox))

    def _automatic_flow_owns_keys(self) -> bool:



        dock = getattr(self._plugin, "dock_widget", None)
        owns = getattr(dock, "auto_flow_owns_keys", None)
        if not callable(owns):
            return False
        try:
            return bool(owns())
        except (RuntimeError, AttributeError):
            return False

    def _clear_selection_offered(self) -> bool:








        dock = getattr(self._plugin, "dock_widget", None)
        button = getattr(dock, "clear_selection_button", None)
        if button is None:
            return False
        try:
            return bool(button.isVisible() and button.isEnabled())
        except (RuntimeError, AttributeError):
            return False

    def _session_owns_key(self, key, modifiers) -> bool:









        ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        if key == Qt.Key.Key_Delete or (key == Qt.Key.Key_Backspace and ctrl):
            return True
        if key == Qt.Key.Key_Z and ctrl:
            return not self._vector_edit_owns_undo()
        if key == Qt.Key.Key_Backspace and not modifiers:
            return True
        blocking = Qt.KeyboardModifier.ControlModifier
        blocking |= Qt.KeyboardModifier.AltModifier
        blocking |= Qt.KeyboardModifier.ShiftModifier
        if key == Qt.Key.Key_C and not (modifiers & blocking):


            return self._clear_selection_offered()
        if key in (Qt.Key.Key_S, Qt.Key.Key_E) and not (modifiers & blocking):
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape):
            return not self._automatic_flow_owns_keys()
        return False

    def _vector_edit_owns_undo(self) -> bool:








        try:
            layer = self._plugin.iface.activeLayer()
            if not isinstance(layer, QgsVectorLayer) or not layer.isEditable():
                return False
            stack = layer.undoStack()
            return stack is not None and stack.canUndo()
        except (RuntimeError, AttributeError):
            return False

    def _end_lost_space_pan(self) -> None:







        try:
            pan_tool = self._plugin._active_space_pan_tool()
            if pan_tool is None or not pan_tool.is_space_panning():
                return
            pan_tool.stop_space_pan()
        except (RuntimeError, AttributeError):
            pass

    def eventFilter(self, _obj, event):




        try:
            return self._route_event(event)
        except Exception:
            return False

    def _route_event(self, event):
        event_type = event.type()
        plugin = self._plugin




        if event_type in (QEvent.Type.ShortcutOverride,
                          QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            if event.key() == Qt.Key.Key_Space:
                if event_type != QEvent.Type.KeyRelease and self._typing_in_text_field():
                    return False
                pan_tool = plugin._active_space_pan_tool()
                if pan_tool is not None:
                    if event.isAutoRepeat():
                        event.accept()
                        return True
                    if event_type == QEvent.Type.ShortcutOverride:
                        event.accept()
                        return True
                    if event_type == QEvent.Type.KeyPress:
                        pan_tool.start_space_pan()
                        return True
                    pan_tool.stop_space_pan()
                    return True




        if event_type in (QEvent.Type.WindowDeactivate, QEvent.Type.FocusOut):
            self._end_lost_space_pan()
            return False





        if event_type == QEvent.Type.ShortcutOverride:
            if not plugin.map_tool or not plugin.map_tool.isActive():
                return False
            if self._typing_in_text_field():
                return False
            if self._session_owns_key(event.key(), event.modifiers()):
                event.accept()
                return True
            return False

        if event_type != QEvent.Type.KeyPress:
            return False
        if not plugin.map_tool or not plugin.map_tool.isActive():
            return False

        if self._typing_in_text_field():
            return False
        app = QApplication.instance()
        if not app:
            return False
        focused = app.focusWidget()


        from qgis.PyQt.QtWidgets import QAbstractItemView, QListView, QTableView, QTreeView
        if isinstance(focused, (QAbstractItemView, QListView,
                                QTableView, QTreeView)):
            return False

        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key.Key_Z and modifiers & Qt.KeyboardModifier.ControlModifier:
            if self._vector_edit_owns_undo():
                return False
            plugin._on_undo()
            return True



        if key == Qt.Key.Key_Delete or (
                key == Qt.Key.Key_Backspace and modifiers & Qt.KeyboardModifier.ControlModifier):
            if getattr(plugin, "_on_delete_active_object", None):
                plugin._on_delete_active_object()
                return True






        if key == Qt.Key.Key_Backspace and not modifiers:
            can_delete_active = getattr(plugin, "_handoff_selected_entries", None)
            can_delete_active = can_delete_active and plugin.current_mask is None
            can_delete_active = can_delete_active and not plugin._active_crop_points_positive
            can_delete_active = can_delete_active and getattr(plugin, "_on_delete_active_object", None)
            if can_delete_active:
                plugin._on_delete_active_object()
                return True
            plugin._on_undo()
            return True
        blocking_mods = Qt.KeyboardModifier.ControlModifier
        blocking_mods |= Qt.KeyboardModifier.AltModifier
        blocking_mods |= Qt.KeyboardModifier.ShiftModifier
        if key == Qt.Key.Key_S and not (modifiers & blocking_mods):







            if getattr(plugin, "_refine_handoff_active", False):





                add_save = getattr(plugin, "_route_save_add_mode", None)
                if callable(add_save) and add_save():
                    return True
                if (getattr(plugin, "_refine_edit_session_active", None) and plugin._refine_edit_session_active()):
                    plugin._on_reshape_done()
                return True
            plugin._on_save_polygon()
            return True


        if key == Qt.Key.Key_E and not (modifiers & blocking_mods) and getattr(
                plugin, "_edit_selected_saved_polygon", None):



            if getattr(plugin, "_encode_blocks_ui", None) and plugin._encode_blocks_ui():
                return True
            if plugin._edit_selected_saved_polygon():
                return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            plugin._on_export_layer()
            return True
        if key == Qt.Key.Key_C and not modifiers:



            if not self._clear_selection_offered():
                return False
            plugin._on_clear_selection()
            return True
        if key == Qt.Key.Key_Escape:


            if getattr(plugin, "_handoff_selected_entries", None) and getattr(plugin, "_deselect_saved_polygons", None):
                plugin._deselect_saved_polygons()
                return True
            plugin._on_stop_segmentation()
            return True
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right,
                   Qt.Key.Key_Up, Qt.Key.Key_Down):
            canvas = plugin.iface.mapCanvas()
            extent = canvas.extent()
            pan_step = dial_in_range("tuning.ui.arrow_pan_step", 0.25, 0.05, 0.9)
            dx = extent.width() * pan_step
            dy = extent.height() * pan_step
            cx, cy = canvas.center().x(), canvas.center().y()
            if key == Qt.Key.Key_Left:
                cx -= dx
            elif key == Qt.Key.Key_Right:
                cx += dx
            elif key == Qt.Key.Key_Up:
                cy += dy
            elif key == Qt.Key.Key_Down:
                cy -= dy
            canvas.setCenter(QgsPointXY(cx, cy))
            canvas.refresh()
            return True

        return False
