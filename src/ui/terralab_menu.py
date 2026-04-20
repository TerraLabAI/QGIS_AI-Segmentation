"""Cooperative TerraLab menu management for QGIS plugins.

SHARED: keep in sync with the copy in the sibling TerraLab plugin.
"""

import os

from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtGui import QDesktopServices, QIcon
from qgis.PyQt.QtWidgets import QMenu

TERRALAB_URL = (
    "https://terra-lab.ai"
    "?utm_source=qgis&utm_medium=plugin&utm_campaign=ai-segmentation&utm_content=menu_more"
)
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
        pass  # nosec B110


def get_or_create_terralab_menu(main_window) -> QMenu:
    menu_bar = main_window.menuBar()
    for action in menu_bar.actions():
        if action.menu() and action.text() == "TerraLab":
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
    action.setProperty("terralab_product_id", product_id)
    # Remove existing action with same product_id (real plugin replaces promo)
    for a in menu.actions():
        if a.objectName() == _UTILITY_SEPARATOR:
            break
        if a.property("terralab_product_id") == product_id and a is not action:
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
    for existing in plugin_actions:
        if existing.text() > action.text():
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


def _get_or_create_plugins_submenu(iface) -> QMenu:
    plugin_menu = iface.pluginMenu()
    for a in plugin_menu.actions():
        if a.menu() and a.text() == _PLUGINS_MENU_NAME:
            return a.menu()
    logo_path = _find_terralab_logo()
    logo_icon = QIcon(logo_path) if logo_path else QIcon()
    submenu = plugin_menu.addMenu(logo_icon, _PLUGINS_MENU_NAME)
    sep = submenu.addSeparator()
    sep.setObjectName(_UTILITY_SEPARATOR)
    update_icon = QIcon(":/images/themes/default/mActionRefresh.svg")
    check_update = submenu.addAction(update_icon, "Check for Updates")
    check_update.triggered.connect(_open_plugin_manager_updates)
    website_icon = QIcon(logo_path) if logo_path else QIcon()
    more_action = submenu.addAction(website_icon, "More from TerraLab...")
    more_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(TERRALAB_URL)))
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
    for existing in plugin_actions:
        if existing.text() > action.text():
            insert_before = existing
            break
    if insert_before:
        submenu.insertAction(insert_before, action)
    else:
        submenu.addAction(action)


def remove_from_plugins_menu(iface, action):
    plugin_menu = iface.pluginMenu()
    for a in plugin_menu.actions():
        if a.menu() and a.text() == _PLUGINS_MENU_NAME:
            submenu = a.menu()
            submenu.removeAction(action)
            if not submenu.actions():
                plugin_menu.removeAction(submenu.menuAction())
            break
