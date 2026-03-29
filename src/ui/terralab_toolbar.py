"""Cooperative TerraLab toolbar management for QGIS plugins.

SHARED: keep in sync with the copy in the sibling TerraLab plugin.
"""

from qgis.PyQt.QtWidgets import QToolBar

_TOOLBAR_OBJECT_NAME = "TerraLabToolbar"
_TOOLBAR_TITLE = "TerraLab"


def get_or_create_terralab_toolbar(iface):
    """Find existing TerraLab toolbar or create one via iface.addToolBar()."""
    main_window = iface.mainWindow()
    for tb in main_window.findChildren(QToolBar):
        if tb.objectName() == _TOOLBAR_OBJECT_NAME:
            return tb
    toolbar = QToolBar(_TOOLBAR_TITLE)
    toolbar.setObjectName(_TOOLBAR_OBJECT_NAME)
    iface.addToolBar(toolbar)
    return toolbar


def add_action_to_toolbar(toolbar, action, product_id):
    """Add a plugin action alphabetically to the shared toolbar."""
    action.setProperty("terralab_product_id", product_id)
    for existing in toolbar.actions():
        if existing.text() > action.text():
            toolbar.insertAction(existing, action)
            return
    toolbar.addAction(action)


def remove_action_from_toolbar(toolbar, action, main_window):
    """Remove action; delete toolbar if no actions remain."""
    toolbar.removeAction(action)
    remaining = [a for a in toolbar.actions() if not a.isSeparator()]
    if not remaining:
        main_window.removeToolBar(toolbar)
        toolbar.deleteLater()
