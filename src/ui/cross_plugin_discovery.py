"""Cross-plugin discovery: expose sibling TerraLab plugins in the UI to boost
conversion (#30). If the sibling is installed, activate its dock; if not,
open the product page.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtGui import QDesktopServices, QIcon
from qgis.PyQt.QtWidgets import QAction

_AI_EDIT_KEYS = ("AI_Edit", "QGIS_AI-Edit", "QGIS_AI-Edit-Team")
_AI_EDIT_PRODUCT_URL = "https://terra-lab.ai/ai-edit?utm_source=qgis&utm_medium=plugin&utm_campaign=ai_segmentation_cross_promo"


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
                continue
    for attr in ("toggle_dock_widget", "show_dock_widget", "_toggle_dock", "run"):
        fn = getattr(plugin, attr, None)
        if callable(fn):
            try:
                fn()
                return True
            except Exception:
                continue
    return False


def make_ai_edit_action(parent, iface, label: str, tooltip: str,
                        icon: QIcon | None = None) -> QAction:
    """Create a QAction that opens AI Edit if installed, else the product page."""
    action = QAction(icon or QIcon(), label, parent)
    action.setToolTip(tooltip)

    def triggered():
        plugin = _find_installed_plugin(_AI_EDIT_KEYS)
        if plugin is not None and _activate_dock(plugin):
            return
        QDesktopServices.openUrl(QUrl(_AI_EDIT_PRODUCT_URL))

    action.triggered.connect(triggered)
    return action
