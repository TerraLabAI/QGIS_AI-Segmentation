"""Cooperative TerraLab menu management for QGIS plugins.

SHARED: keep in sync with the copy in the sibling TerraLab plugin.
"""

from __future__ import annotations

import os

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QMenu

from ..core.i18n import tr
from .external_links import open_external_url

TERRALAB_URL = (
    "https://terra-lab.ai"
    "?utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation&utm_content=menu_more"
)
_UTILITY_SEPARATOR = "_terralab_utility_sep"
_PLUGINS_MENU_NAME = "TerraLab"
# Stable identity for the shared menus, robust to translation or to a
# third-party QMenu("TerraLab") colliding on display text (the toolbar already
# keys on an objectName; the menus now match). The text match stays as a
# fallback so a sibling plugin from an older release is still found.
_MENU_OBJECT_NAME = "TerraLabMenu"
_SUBMENU_OBJECT_NAME = "TerraLabPluginsSubmenu"


def _find_terralab_logo():
    ui_dir = os.path.dirname(__file__)
    plugin_dir = os.path.dirname(os.path.dirname(ui_dir))
    logo_path = os.path.join(plugin_dir, "resources", "icons", "terralab-logo.png")
    if os.path.isfile(logo_path):
        return logo_path
    return None


def open_plugin_manager_updates(fallback_url: str | None = None) -> bool:
    """Open the Plugin Manager on its Upgradeable tab. True when it opened.

    Never a silent no-op: the manager interface can be absent or refuse to
    open, and the menu entry then swallowed the click with nothing on screen.
    The generic Manage Plugins action is tried next, and a web page last, so
    the user always lands somewhere they can act.
    """
    try:
        from qgis.utils import iface
        manager = iface.pluginManagerInterface()
        if manager is None:
            raise RuntimeError("plugin manager unavailable")
        manager.showPluginManager(3)
        return True
    except Exception:  # noqa: BLE001 -- fall through to the next way in
        pass  # nosec B110
    try:
        from qgis.utils import iface
        iface.actionManagePlugins().trigger()
        return True
    except Exception:  # noqa: BLE001 -- last resort below
        pass  # nosec B110
    open_external_url(fallback_url or TERRALAB_URL)
    return False


def _open_plugin_manager_updates(_checked=False):
    open_plugin_manager_updates()


def get_or_create_terralab_menu(main_window) -> QMenu:
    menu_bar = main_window.menuBar()
    for action in menu_bar.actions():
        menu = action.menu()
        if menu and (menu.objectName() == _MENU_OBJECT_NAME or action.text() == "TerraLab"):
            return menu
    menu = QMenu("TerraLab", main_window)
    menu.setObjectName(_MENU_OBJECT_NAME)
    menu_bar.addMenu(menu)
    sep = menu.addSeparator()
    sep.setObjectName(_UTILITY_SEPARATOR)
    update_icon = QIcon(":/images/themes/default/mActionRefresh.svg")
    check_update = menu.addAction(update_icon, tr("Check for Updates"))
    check_update.triggered.connect(_open_plugin_manager_updates)
    logo_path = _find_terralab_logo()
    website_icon = QIcon(logo_path) if logo_path else QIcon()
    more_action = menu.addAction(website_icon, tr("More from TerraLab..."))
    more_action.triggered.connect(lambda: open_external_url(TERRALAB_URL, parent=menu))
    return menu


def _sort_key(action) -> str:
    """Order menu entries by product id, never by their displayed label.

    The label is translated, so the same two plugins landed in a different
    order in every language, and the insertion point moved with the UI
    language rather than staying put.
    """
    return str(action.property("terralab_product_id") or action.text())


def add_plugin_to_menu(menu: QMenu, action, product_id: str, is_cross_promo: bool = False):
    action.setProperty("terralab_product_id", product_id)
    action.setProperty("terralab_is_cross_promo", is_cross_promo)
    # A real plugin replaces a cross-promo placeholder, but a placeholder must
    # never replace a real plugin's action (mirrors add_action_to_toolbar), or
    # load order could let our cross-promo entry clobber the sibling's real one.
    for a in menu.actions():
        if a.objectName() == _UTILITY_SEPARATOR:
            break
        if a.property("terralab_product_id") == product_id and a is not action:
            if is_cross_promo and not a.property("terralab_is_cross_promo"):
                return
            menu.removeAction(a)
            break
    sep_action = None
    plugin_actions = []
    for a in menu.actions():
        if a.objectName() == _UTILITY_SEPARATOR:
            sep_action = a
            break
        if not a.isSeparator():
            plugin_actions.append(a)
    insert_before = sep_action
    action_key = _sort_key(action)
    for existing in plugin_actions:
        if _sort_key(existing) > action_key:
            insert_before = existing
            break
    if insert_before:
        menu.insertAction(insert_before, action)
    else:
        menu.addAction(action)


def remove_plugin_from_menu(menu: QMenu, action, main_window):
    menu.removeAction(action)
    has_plugins = False
    for a in menu.actions():
        if a.objectName() == _UTILITY_SEPARATOR:
            break
        if not a.isSeparator():
            has_plugins = True
            break
    if not has_plugins:
        main_window.menuBar().removeAction(menu.menuAction())
        # Removing the action only unhooks the menu from the bar. Without this
        # the QMenu stays alive and a re-enable builds a second one.
        menu.deleteLater()


def _get_or_create_plugins_submenu(iface) -> QMenu:
    plugin_menu = iface.pluginMenu()
    for a in plugin_menu.actions():
        sub = a.menu()
        if sub and (sub.objectName() == _SUBMENU_OBJECT_NAME
                    or a.text() == _PLUGINS_MENU_NAME):
            return sub
    logo_path = _find_terralab_logo()
    logo_icon = QIcon(logo_path) if logo_path else QIcon()
    submenu = plugin_menu.addMenu(logo_icon, _PLUGINS_MENU_NAME)
    submenu.setObjectName(_SUBMENU_OBJECT_NAME)
    sep = submenu.addSeparator()
    sep.setObjectName(_UTILITY_SEPARATOR)
    update_icon = QIcon(":/images/themes/default/mActionRefresh.svg")
    check_update = submenu.addAction(update_icon, tr("Check for Updates"))
    check_update.triggered.connect(_open_plugin_manager_updates)
    website_icon = QIcon(logo_path) if logo_path else QIcon()
    more_action = submenu.addAction(website_icon, tr("More from TerraLab..."))
    more_action.triggered.connect(lambda: open_external_url(TERRALAB_URL, parent=submenu))
    return submenu


def add_to_plugins_menu(iface, action):
    submenu = _get_or_create_plugins_submenu(iface)
    product_id = action.property("terralab_product_id")
    if product_id:
        for a in submenu.actions():
            if a.objectName() == _UTILITY_SEPARATOR:
                break
            if a.property("terralab_product_id") == product_id and a is not action:
                submenu.removeAction(a)
                break
    sep_action = None
    plugin_actions = []
    for a in submenu.actions():
        if a.objectName() == _UTILITY_SEPARATOR:
            sep_action = a
            break
        if not a.isSeparator():
            plugin_actions.append(a)
    insert_before = sep_action
    action_key = _sort_key(action)
    for existing in plugin_actions:
        if _sort_key(existing) > action_key:
            insert_before = existing
            break
    if insert_before:
        submenu.insertAction(insert_before, action)
    else:
        submenu.addAction(action)


def remove_from_plugins_menu(iface, action):
    plugin_menu = iface.pluginMenu()
    for a in plugin_menu.actions():
        submenu = a.menu()
        if submenu and (submenu.objectName() == _SUBMENU_OBJECT_NAME
                        or a.text() == _PLUGINS_MENU_NAME):
            submenu.removeAction(action)
            if not submenu.actions():
                plugin_menu.removeAction(submenu.menuAction())
                submenu.deleteLater()
            break
