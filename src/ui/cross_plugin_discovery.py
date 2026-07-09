"""Cross-plugin discovery: expose sibling TerraLab plugins in the UI to boost
conversion (#30). If the sibling is installed, activate its dock; if not, open
the QGIS Plugin Manager pre-filtered to it so the user can install it in one
place (with the product page as a fallback when the manager can't open).
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtGui import QAction, QDesktopServices, QIcon

_AI_EDIT_KEYS = ("AI_Edit", "QGIS_AI-Edit", "QGIS_AI-Edit-Team")
# Must match the sibling's metadata.txt name= so the Plugin Manager filter lands
# on the right row.
_AI_EDIT_PLUGIN_NAME = "AI Edit by TerraLab"
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


def open_ai_edit_page() -> None:
    """Open the AI Edit product page in the browser - always the website, with
    no installed-plugin detection. Used by the in-dock footer CTA.
    """
    QDesktopServices.openUrl(QUrl(_AI_EDIT_PRODUCT_URL))


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
        # Start the poll (see _prefill_plugin_filter) rather than a lone setText:
        # on the FIRST open the dialog is still being built and repopulated.
        QTimer.singleShot(0, lambda: _prefill_plugin_filter(plugin_name))
    except Exception:
        QDesktopServices.openUrl(QUrl(fallback_url))


def _prefill_plugin_filter(text: str, attempts: int = 14, confirmed: int = 0) -> None:
    """Set the Plugin Manager's search field, retrying until it sticks.

    The manager is a C++ internal dialog with no public filter API. On its FIRST
    open it is constructed and then repopulates its plugin list across several
    event-loop turns, and each repopulation clears the filter, so a lone
    ``singleShot(0)`` setText landed too early and got wiped, which is why the
    filter only appeared on the SECOND click (once the dialog already existed).

    We poll on a short cadence and re-assert the text whenever the box does not
    already hold it, overriding BOTH an empty box (fresh open or a mid-open
    repopulation that wiped the filter) AND a stale filter left over from a
    previous session: the manager is a persistent singleton that keeps its last
    search, so clicking this CTA after searching something else must replace
    that old query, not defer to it. The poll runs only right after the CTA
    click, so there is no in-progress user typing to preserve. It stops once the
    text has survived two consecutive polls. The dialog has several line edits
    (e.g. a hidden install-from-ZIP picker), so target the named search box
    first, then the first VISIBLE filter; no-ops silently if never found."""
    from qgis.PyQt.QtCore import QTimer
    from qgis.PyQt.QtWidgets import QApplication, QLineEdit

    try:
        dialog = next(
            (w for w in QApplication.instance().topLevelWidgets()
             if w.metaObject().className() == "QgsPluginManager"),
            None,
        )
        edit = None
        if dialog is not None:
            for name in ("leFilter", "mLeFilter", "mFilterLineEdit"):
                edit = dialog.findChild(QLineEdit, name)
                if edit is not None:
                    break
            if edit is None:
                try:
                    from qgis.gui import QgsFilterLineEdit
                    edit = next(
                        (e for e in dialog.findChildren(QgsFilterLineEdit)
                         if e.isVisible()),
                        None,
                    )
                except Exception:
                    edit = None
            if edit is None:
                edit = next(
                    (e for e in dialog.findChildren(QLineEdit) if e.isVisible()),
                    None,
                )
        if edit is not None:
            current = edit.text()
            if current == text:
                confirmed += 1  # it stuck; make sure it survives one more poll
            else:
                # (Re)assert our text, overriding whatever is there: an empty box
                # (fresh open / repopulation clear) OR a stale query the manager
                # kept from a previous open. The CTA means "take me to THIS
                # plugin", so a leftover filter must be replaced.
                edit.setText(text)
                confirmed = 0
    except Exception:
        pass  # nosec B110
    if confirmed < 2 and attempts > 0:
        QTimer.singleShot(
            70, lambda: _prefill_plugin_filter(text, attempts - 1, confirmed))


def make_ai_edit_action(parent, iface, label: str, tooltip: str,
                        icon: QIcon | None = None) -> QAction:
    """Create a QAction that opens AI Edit if installed, else the Plugin Manager
    pre-filtered to it (product page as a last-resort fallback)."""
    action = QAction(icon or QIcon(), label, parent)
    action.setToolTip(tooltip)

    def triggered():
        plugin = _find_installed_plugin(_AI_EDIT_KEYS)
        if plugin is not None and _activate_dock(plugin):
            return
        open_plugin_manager(_AI_EDIT_PLUGIN_NAME, _AI_EDIT_PRODUCT_URL)

    action.triggered.connect(triggered)
    return action
