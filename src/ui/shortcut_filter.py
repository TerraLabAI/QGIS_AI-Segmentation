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

    def _typing_in_text_field(self) -> bool:
        """True while a text or spin widget holds focus (or focus is unknown):
        typing keys must never be stolen from an input field."""
        app = QApplication.instance()
        if not app:
            return True
        focused = app.focusWidget()
        return isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit,
                                    QSpinBox, QDoubleSpinBox))

    def _automatic_flow_owns_keys(self) -> bool:
        """True while the Automatic flow's own Escape / Enter dispatcher is
        live. There the two keys belong to the run and the review (soft
        cancel, Detect, Export), so the armed session must not claim them."""
        dock = getattr(self._plugin, "dock_widget", None)
        owns = getattr(dock, "auto_flow_owns_keys", None)
        if not callable(owns):
            return False
        try:
            return bool(owns())
        except (RuntimeError, AttributeError):
            return False

    def _session_owns_key(self, key, modifiers) -> bool:
        """Whether the armed session handles this key in the KeyPress branch.

        Every key listed here is also claimed by a window-level shortcut that
        would consume the press before it could reach this filter: QGIS binds a
        bare S (toggle snapping) and a bare T, and the dock binds Escape and
        Enter for the Automatic flow, whose gate is down in Manual yet still
        eats the key. Accepting the ShortcutOverride for these skips the
        shortcut map and routes the key here instead.
        """
        ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        if key == Qt.Key.Key_Delete or (key == Qt.Key.Key_Backspace and ctrl):
            return True
        if key == Qt.Key.Key_Z and ctrl:
            return True
        if key == Qt.Key.Key_Backspace and not modifiers:
            return True
        blocking = Qt.KeyboardModifier.ControlModifier
        blocking |= Qt.KeyboardModifier.AltModifier
        blocking |= Qt.KeyboardModifier.ShiftModifier
        if key in (Qt.Key.Key_S, Qt.Key.Key_E) and not (modifiers & blocking):
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape):
            return not self._automatic_flow_owns_keys()
        return False

    def _end_lost_space_pan(self) -> None:
        """End a Space pan whose KeyRelease will never arrive.

        The release only reaches this filter while the window is in front and
        the canvas holds focus, and dock updates take that focus away. A lost
        release leaves the tool panning for good: every click ignored and the
        map sliding under the cursor until the user switches tool.
        """
        try:
            pan_tool = self._plugin._active_space_pan_tool()
            if pan_tool is None or not pan_tool.is_space_panning():
                return
            pan_tool.stop_space_pan()
        except (RuntimeError, AttributeError):
            pass

    def eventFilter(self, _obj, event):
        # Qt calls this from C++ for every event the main window and the canvas
        # receive, and the branches below call plugin slots (undo, delete, save,
        # export, stop). A raise would travel back into Qt's event dispatch, so
        # any failure drops the key instead: the shortcut is lost, nothing else.
        try:
            return self._route_event(event)
        except Exception:
            return False

    def _route_event(self, event):
        event_type = event.type()
        plugin = self._plugin

        # --- Space key: handle press AND release for temporary pan ---
        # Also intercept ShortcutOverride to prevent QGIS from activating
        # its own pan-tool shortcut when Space is pressed.
        if event_type in (QEvent.Type.ShortcutOverride,
                          QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
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

        # The window going to the back, or the canvas losing focus, ends a pan
        # the same way its release would. Never consumed: both events belong to
        # whoever else is watching them.
        if event_type in (QEvent.Type.WindowDeactivate, QEvent.Type.FocusOut):
            self._end_lost_space_pan()
            return False

        # --- Session-owned keys. While the segmentation tool is armed, every
        # key this filter handles belongs to the session, and a window-level
        # shortcut elsewhere would otherwise consume the press before the
        # KeyPress branch below ever ran (see _session_owns_key).
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
        # Plain Backspace (no modifier): on Mac keyboards the big delete key
        # IS Backspace, so with detections SELECTED and nothing open for
        # editing it must delete the selection (like Key_Delete), never fall
        # into undo and resurrect an unrelated saved polygon. Otherwise it
        # stays undo-the-last-click, mirroring the zone-draw tool. Kept AFTER
        # the modifier branch above so Ctrl/Cmd+Backspace keeps meaning delete.
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
            # In a Correct-step reshape session S is the Save BUTTON, which
            # both saves AND closes the session back to the resting select
            # state (_on_reshape_done). The bare _on_save_polygon is the
            # base-Manual save that keeps segmenting, so during a handoff it
            # saved the edit but left the session armed and the panel stuck
            # open. Mirror the Esc handling in manual_workflow: an open edit
            # routes to the full done; a resting selection has nothing to save.
            if getattr(plugin, "_refine_handoff_active", False):
                # The AI Add lane owns Save first: it keeps the outline as its
                # own new object and stays armed for the next one. Its outline
                # is NOT an open edit session (no saved object was reopened),
                # so without this branch the gate below saw nothing to save and
                # S did nothing at all.
                add_save = getattr(plugin, "_route_save_add_mode", None)
                if callable(add_save) and add_save():
                    return True
                if (getattr(plugin, "_refine_edit_session_active", None) and plugin._refine_edit_session_active()):
                    plugin._on_reshape_done()
                return True
            plugin._on_save_polygon()
            return True
        # E opens the single selected detection for SAM editing (the keyboard
        # twin of the second click / double-click).
        if key == Qt.Key.Key_E and not (modifiers & blocking_mods) and getattr(
                plugin, "_edit_selected_saved_polygon", None):
            # Mirror the double-click open: while a foreground (busy-cursor)
            # encode owns the pipe, E defers to it and no-ops, so a stray press
            # during a busy encode can never race the open (see _encode_blocks_ui).
            if getattr(plugin, "_encode_blocks_ui", None) and plugin._encode_blocks_ui():
                return True
            if plugin._edit_selected_saved_polygon():
                return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            plugin._on_export_layer()
            return True
        if key == Qt.Key.Key_Escape:
            # Selection-first: Esc clears the selection before it ever means
            # "stop the session".
            if getattr(plugin, "_handoff_selected_entries", None) and getattr(plugin, "_deselect_saved_polygons", None):
                plugin._deselect_saved_polygons()
                return True
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
