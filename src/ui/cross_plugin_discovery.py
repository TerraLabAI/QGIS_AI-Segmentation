"""Cross-plugin discovery: expose sibling TerraLab plugins in the UI to boost
conversion (#30). If the sibling is installed, activate its dock; if not,
open the product page.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtGui import QDesktopServices, QIcon
from qgis.PyQt.QtWidgets import QAction

_AI_EDIT_KEYS = ("AI_Edit", "QGIS_AI-Edit", "QGIS_AI-Edit-Team")
_AI_EDIT_PRODUCT_URL = (
    "https://terra-lab.ai/ai-edit"
    "?utm_source=qgis&utm_medium=plugin&utm_campaign=ai_segmentation_cross_promo"
)


def _find_installed_plugin(keys: tuple[str, ...]):
    try:
        import qgis.utils
        for key in keys:
            plugin = qgis.utils.plugins.get(key)
            if plugin is not None:
                return plugin
    except Exception:
        pass  # nosec B110
    return None


def _activate_dock(plugin) -> bool:
    """Ensure a sibling plugin's dock widget is visible."""
    for attr in ("dock_widget", "_dock_widget"):
        dock = getattr(plugin, attr, None)
        if dock is not None:
            try:
                dock.show()
                dock.raise_()
                return True
            except Exception:
                continue  # nosec B112
    for attr in ("toggle_dock_widget", "show_dock_widget", "_toggle_dock", "run"):
        fn = getattr(plugin, attr, None)
        if callable(fn):
            try:
                fn()
                return True
            except Exception:
                continue  # nosec B112
    return False


def open_plugin_manager(plugin_name: str, fallback_url: str) -> None:
    """Open the QGIS Plugin Manager on the 'Not installed' tab, pre-filtered to
    plugin_name. Falls back to the product page if the manager can't open."""
    try:
        from qgis.PyQt.QtCore import QTimer
        from qgis.utils import iface
        mgr = iface.pluginManagerInterface()
        if mgr is None:
            raise RuntimeError("plugin manager unavailable")
        mgr.showPluginManager(2)  # 0 All, 1 Installed, 2 Not installed, 3 Upgradeable
        QTimer.singleShot(0, lambda: _prefill_plugin_filter(plugin_name))
    except Exception:
        QDesktopServices.openUrl(QUrl(fallback_url))


def _prefill_plugin_filter(text: str) -> None:
    """Best-effort: set the Plugin Manager's search field (C++ internal dialog,
    no public API). The dialog has several line edits (e.g. a hidden install-from
    -ZIP picker), so target the named search box, then the first VISIBLE filter;
    no-ops silently if not found, leaving the manager open."""
    try:
        from qgis.PyQt.QtWidgets import QApplication, QLineEdit
        dialog = next(
            (w for w in QApplication.instance().topLevelWidgets()
             if w.metaObject().className() == "QgsPluginManager"),
            None,
        )
        if dialog is None:
            return
        edit = None
        for name in ("leFilter", "mLeFilter", "mFilterLineEdit"):
            edit = dialog.findChild(QLineEdit, name)
            if edit is not None:
                break
        if edit is None:
            try:
                from qgis.gui import QgsFilterLineEdit
                edit = next((e for e in dialog.findChildren(QgsFilterLineEdit) if e.isVisible()), None)
            except Exception:
                edit = None
        if edit is None:
            edit = next((e for e in dialog.findChildren(QLineEdit) if e.isVisible()), None)
        if edit is not None:
            edit.setText(text)
    except Exception:
        pass  # nosec B110


def open_ai_edit_page() -> None:
    """Open the Plugin Manager pre-filtered to AI Edit so it can be installed in
    one click, falling back to the product page. Used by the in-dock footer CTA.
    """
    open_plugin_manager("AI Edit by TerraLab", _AI_EDIT_PRODUCT_URL)


def make_ai_edit_action(parent, iface, label: str, tooltip: str,
                        icon: QIcon | None = None) -> QAction:
    """Create a QAction that opens AI Edit if installed, else the Plugin Manager."""
    action = QAction(icon or QIcon(), label, parent)
    action.setToolTip(tooltip)

    def triggered():
        plugin = _find_installed_plugin(_AI_EDIT_KEYS)
        if plugin is not None and _activate_dock(plugin):
            return
        open_plugin_manager("AI Edit by TerraLab", _AI_EDIT_PRODUCT_URL)

    action.triggered.connect(triggered)
    return action
