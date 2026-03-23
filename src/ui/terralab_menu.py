"""Cooperative TerraLab menu management for QGIS plugins."""

import os

from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtGui import QDesktopServices, QIcon
from qgis.PyQt.QtWidgets import QMenu

TERRALAB_URL = "https://terra-lab.ai"
_UTILITY_SEPARATOR = "_terralab_utility_sep"
_PLUGINS_MENU_NAME = "TerraLab"


def _find_terralab_logo():
    ui_dir = os.path.dirname(__file__)
    plugin_dir = os.path.dirname(os.path.dirname(ui_dir))
    logo_path = os.path.join(plugin_dir, "resources", "icons", "terralab-logo.png")
    if os.path.isfile(logo_path):
        return logo_path
    return None


def _open_plugin_manager_updates():
    try:
        from qgis.utils import iface

        iface.pluginManagerInterface().showPluginManager(3)
    except Exception:
        pass


def get_or_create_terralab_menu(main_window) -> QMenu:
    """Get existing TerraLab menu or create one with utility actions."""
    menu_bar = main_window.menuBar()
    for action in menu_bar.actions():
        if action.menu() and action.text().replace("&", "") == "TerraLab":
            return action.menu()
    menu = QMenu("TerraLab", main_window)
    menu_bar.addMenu(menu)
    sep = menu.addSeparator()
    sep.setObjectName(_UTILITY_SEPARATOR)
    update_icon = QIcon(":/images/themes/default/mActionRefresh.svg")
    check_update = menu.addAction(update_icon, "Check for Updates")
    check_update.triggered.connect(_open_plugin_manager_updates)
    logo_path = _find_terralab_logo()
    website_icon = QIcon(logo_path) if logo_path else QIcon()
    more_action = menu.addAction(website_icon, "More from TerraLab...")
    more_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(TERRALAB_URL)))
    return menu


def add_plugin_to_menu(menu: QMenu, action, product_id: str):
    """Insert plugin action alphabetically before the utility separator."""
    sep_action = None
    plugin_actions = []
    for a in menu.actions():
        if a.objectName() == _UTILITY_SEPARATOR:
            sep_action = a
            break
        if not a.isSeparator():
            plugin_actions.append(a)
    insert_before = sep_action
    for existing in plugin_actions:
        if existing.text() > action.text():
            insert_before = existing
            break
    if insert_before:
        menu.insertAction(insert_before, action)
    else:
        menu.addAction(action)


def remove_plugin_from_menu(menu: QMenu, action, main_window):
    """Remove plugin action; delete menu if no plugins remain."""
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


def add_to_plugins_menu(iface, action):
    """Add action under Plugins > TerraLab submenu."""
    plugin_menu = iface.pluginMenu()
    submenu = None
    for a in plugin_menu.actions():
        if a.menu() and a.text() == _PLUGINS_MENU_NAME:
            submenu = a.menu()
            break
    if submenu is None:
        logo_path = _find_terralab_logo()
        logo_icon = QIcon(logo_path) if logo_path else QIcon()
        submenu = plugin_menu.addMenu(logo_icon, _PLUGINS_MENU_NAME)
    insert_before = None
    for existing in submenu.actions():
        if existing.text() > action.text():
            insert_before = existing
            break
    if insert_before:
        submenu.insertAction(insert_before, action)
    else:
        submenu.addAction(action)


def remove_from_plugins_menu(iface, action):
    """Remove action from Plugins > TerraLab submenu; delete submenu if empty."""
    plugin_menu = iface.pluginMenu()
    for a in plugin_menu.actions():
        if a.menu() and a.text() == _PLUGINS_MENU_NAME:
            submenu = a.menu()
            submenu.removeAction(action)
            if not submenu.actions():
                plugin_menu.removeAction(submenu.menuAction())
            break
