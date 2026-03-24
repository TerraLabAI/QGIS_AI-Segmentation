"""Global keyboard shortcut filter for AI Segmentation map tool.

QgsMapTool.keyPressEvent only fires when the canvas has keyboard
focus, which is unreliable after encoding/prediction (dock widget
updates steal focus).  This filter catches shortcuts regardless of
which widget has focus.
"""

from qgis.PyQt.QtCore import QEvent, QObject, Qt
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QTextEdit,
)


class ShortcutFilter(QObject):
    """Event filter that intercepts keyboard shortcuts on the main window."""

    def __init__(self, plugin, parent=None):
        super().__init__(parent)
        self._plugin = plugin

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.KeyPress:
            return False
        plugin = self._plugin
        if not plugin.map_tool or not plugin.map_tool.isActive():
            return False

        app = QApplication.instance()
        if not app:
            return False
        focused = app.focusWidget()
        if isinstance(
            focused,
            (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox),
        ):
            return False

        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key.Key_Z and modifiers & Qt.KeyboardModifier.ControlModifier:
            plugin._on_undo()
            return True
        elif key == Qt.Key.Key_S and not (
            modifiers
            & (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.AltModifier
                | Qt.KeyboardModifier.ShiftModifier
            )
        ):
            plugin._on_save_polygon()
            return True
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            plugin._on_export_layer()
            return True
        elif key == Qt.Key.Key_Escape:
            plugin._on_stop_segmentation()
            return True

        return False
