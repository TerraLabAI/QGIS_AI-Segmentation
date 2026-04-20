"""Cross-plugin discovery: expose sibling TerraLab plugins in the UI to boost
conversion (#30). If the sibling is installed, activate its dock; if not,
open the QGIS Plugin Manager.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtGui import QDesktopServices, QIcon
from qgis.PyQt.QtWidgets import QAction

# Plugin IDs the QGIS plugin system uses for the sibling plugin.
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
        iface.pluginManagerInterface().showPluginManager()
    except Exception:
        QDesktopServices.openUrl(QUrl(_AI_EDIT_PLUGINS_URL))
        return
    # Plugin Manager has no stable API to pre-fill the search field, so just
    # open it and let the user see the TerraLab plugins.
    _ = search_text


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
