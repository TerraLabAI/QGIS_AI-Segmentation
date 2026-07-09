"""Global keyboard shortcut filter for the segmentation map tool.

Intercepts shortcuts (Space, Ctrl+Z, Backspace, S, Enter, Esc, Delete,
arrows) regardless of which widget has focus, solving the issue where
dock widget updates steal keyboard focus from the map canvas.
"""
from __future__ import annotations


from qgis.core import QgsPointXY
from qgis.PyQt.QtCore import QEvent, QObject, Qt
from qgis.PyQt.QtWidgets import QApplication, QDoubleSpinBox, QLineEdit, QPlainTextEdit, QSpinBox, QTextEdit


class ShortcutFilter(QObject):
    """Event filter that intercepts keyboard shortcuts on the main window.

    QgsMapTool.keyPressEvent only fires when the canvas has keyboard
    focus, which is unreliable after encoding/prediction (dock widget
    updates steal focus).  This filter catches shortcuts regardless of
    which widget has focus.
    """

    def __init__(self, plugin, parent=None):
        super().__init__(parent)
        self._plugin = plugin

    def eventFilter(self, _obj, event):
        event_type = event.type()
        plugin = self._plugin

        # --- Space key: handle press AND release for temporary pan ---
        # Also intercept ShortcutOverride to prevent QGIS from activating
        # its own pan-tool shortcut when Space is pressed.
        if event_type in (QEvent.Type.ShortcutOverride,
                          QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            if (event.key() == Qt.Key.Key_Space
                    and not event.isAutoRepeat()):  # noqa: W503
                pan_tool = plugin._active_space_pan_tool()
                if pan_tool is not None:
                    if event_type == QEvent.Type.ShortcutOverride:
                        event.accept()
                        return True
                    if event_type == QEvent.Type.KeyPress:
                        pan_tool.start_space_pan()
                        return True
                    pan_tool.stop_space_pan()
                    return True

        if event_type != QEvent.Type.KeyPress:
            return False
        if not plugin.map_tool or not plugin.map_tool.isActive():
            return False

        app = QApplication.instance()
        if not app:
            return False
        focused = app.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit,
                                QSpinBox, QDoubleSpinBox)):
            return False
        # Don't intercept arrow keys in table/tree views (attribute table, etc.)
        # but allow them on the map canvas (QGraphicsView subclass).
        from qgis.PyQt.QtWidgets import QAbstractItemView, QListView, QTableView, QTreeView
        if isinstance(focused, (QAbstractItemView, QListView,
                                QTableView, QTreeView)):
            return False

        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key.Key_Z and modifiers & Qt.KeyboardModifier.ControlModifier:
            plugin._on_undo()
            return True
        # Delete the active (open-for-editing) object: Delete, or Ctrl/Cmd+Backspace (the
        # big key on Mac keyboards; Qt maps Cmd to ControlModifier on macOS).
        # Plain Backspace without a modifier falls through (never deletes).
        if key == Qt.Key.Key_Delete or (
                key == Qt.Key.Key_Backspace and modifiers & Qt.KeyboardModifier.ControlModifier):
            if getattr(plugin, "_on_delete_active_object", None):
                plugin._on_delete_active_object()
                return True
        # Plain Backspace (no modifier) = undo the last click, mirroring the
        # zone-draw tool's Backspace-undo. Kept AFTER the modifier branch
        # above so Ctrl/Cmd+Backspace keeps meaning delete, never undo.
        if key == Qt.Key.Key_Backspace and not modifiers:
            plugin._on_undo()
            return True
        if (key == Qt.Key.Key_S
                and not (modifiers & (Qt.KeyboardModifier.ControlModifier  # noqa: W503
                                      | Qt.KeyboardModifier.AltModifier  # noqa: W503
                                      | Qt.KeyboardModifier.ShiftModifier))):  # noqa: W503
            plugin._on_save_polygon()
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            plugin._on_export_layer()
            return True
        if key == Qt.Key.Key_Escape:
            plugin._on_stop_segmentation()
            return True
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right,
                   Qt.Key.Key_Up, Qt.Key.Key_Down):
            canvas = plugin.iface.mapCanvas()
            extent = canvas.extent()
            dx = extent.width() * 0.25
            dy = extent.height() * 0.25
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
