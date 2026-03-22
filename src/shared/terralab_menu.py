# SHARED MODULE v1.0 — keep in sync between AI Canvas and AI Segmentation
"""Cooperative TerraLab menu management for QGIS plugins."""
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtWidgets import QMenu
from qgis.PyQt.QtGui import QDesktopServices

from .constants import TERRALAB_URL

_UTILITY_SEPARATOR = "_terralab_utility_sep"


def get_or_create_terralab_menu(main_window) -> QMenu:
    """Find existing TerraLab menu or create one with utility actions.

    Multiple plugins can call this independently. The first one creates
    the menu; subsequent calls find and reuse it.
    """
    menu_bar = main_window.menuBar()
    for action in menu_bar.actions():
        if action.menu() and action.text() == "TerraLab":
            return action.menu()

    menu = QMenu("TerraLab", main_window)
    menu_bar.addMenu(menu)

    sep = menu.addSeparator()
    sep.setObjectName(_UTILITY_SEPARATOR)

    more_action = menu.addAction("More from TerraLab...")
    more_action.triggered.connect(
        lambda: QDesktopServices.openUrl(QUrl(TERRALAB_URL))
    )

    return menu


def add_plugin_to_menu(menu: QMenu, action, product_id: str):
    """Add a plugin action before the utility separator, in alphabetical order."""
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
    """Remove a plugin action. Remove menu entirely if no plugin actions remain."""
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
