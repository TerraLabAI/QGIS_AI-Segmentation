"""Cross-plugin discovery: expose sibling TerraLab plugins in the UI to boost
conversion (#30). If the sibling is installed, activate its dock; if not,
open the QGIS Plugin Manager.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QTimer, QUrl
from qgis.PyQt.QtGui import QDesktopServices, QIcon
from qgis.PyQt.QtWidgets import QAction, QLineEdit

_AI_EDIT_KEYS = ("AI_Edit", "QGIS_AI-Edit", "QGIS_AI-Edit-Team")
_AI_EDIT_PLUGINS_URL = "https://plugins.qgis.org/plugins/AI_Edit/"


def _find_installed_plugin(keys: tuple[str, ...]):
    try:
        import qgis.utils
        for key in keys:
            plugin = qgis.utils.plugins.get(key)
            if plugin is not None:
                return plugin
    except Exception:
        pass
    return None


def _open_plugin_manager(iface, search_text: str):
    try:
        # Tab 0 = "All" — best for searching across installed and available
        iface.pluginManagerInterface().showPluginManager(0)
    except Exception:
        QDesktopServices.openUrl(QUrl(_AI_EDIT_PLUGINS_URL))
        return

    # Best-effort: find the search field in the Plugin Manager dialog and type the query.
    def _fill_search():
        try:
            from qgis.PyQt.QtWidgets import QDialog
            for dlg in iface.mainWindow().findChildren(QDialog):
                if "pluginmanager" in type(dlg).__name__.lower() or "plugin" in (dlg.objectName() or "").lower():
                    line_edits = dlg.findChildren(QLineEdit)
                    if line_edits:
                        line_edits[0].setText(search_text)
                        return
        except Exception:
            pass

    QTimer.singleShot(200, _fill_search)


def _activate_ai_edit_dock(plugin) -> bool:
    for attr in ("show_dock_widget", "toggle_dock_widget", "run", "activate"):
        fn = getattr(plugin, attr, None)
        if callable(fn):
            try:
                fn()
                return True
            except Exception:
                continue
    dock = getattr(plugin, "dock_widget", None)
    if dock is not None:
        try:
            dock.show()
            dock.raise_()
            return True
        except Exception:
            pass
    return False


def make_ai_edit_action(parent, iface, label: str, tooltip: str,
                        icon: QIcon | None = None) -> QAction:
    """Create a QAction that opens AI Edit if installed, else the plugin manager."""
    action = QAction(icon or QIcon(), label, parent)
    action.setToolTip(tooltip)

    def triggered():
        plugin = _find_installed_plugin(_AI_EDIT_KEYS)
        if plugin is not None and _activate_ai_edit_dock(plugin):
            return
        _open_plugin_manager(iface, "AI Edit")

    action.triggered.connect(triggered)
    return action
