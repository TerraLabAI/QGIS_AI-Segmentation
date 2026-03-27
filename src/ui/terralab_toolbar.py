"""Cooperative TerraLab toolbar management for QGIS plugins.

SHARED: keep in sync with the copy in the sibling TerraLab plugin.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QToolBar

_TOOLBAR_OBJECT_NAME = "TerraLabToolbar"
_TOOLBAR_TITLE = "TerraLab Toolbar"


def _first_row_has_space(main_window, extra_width=80):
    """Check if the first toolbar row has enough space for our toolbar."""
    first_row_width = 0
    for tb in main_window.findChildren(QToolBar):
        if tb.isVisible() and main_window.toolBarArea(tb) == Qt.TopToolBarArea:
            if not main_window.toolBarBreak(tb):
                first_row_width += tb.sizeHint().width()
    return first_row_width + extra_width < main_window.width()


def get_or_create_terralab_toolbar(main_window):
    """Find existing TerraLab toolbar or create one, on first row if space allows."""
    for tb in main_window.findChildren(QToolBar):
        if tb.objectName() == _TOOLBAR_OBJECT_NAME:
            return tb
    toolbar = QToolBar(_TOOLBAR_TITLE, main_window)
    toolbar.setObjectName(_TOOLBAR_OBJECT_NAME)
    if not _first_row_has_space(main_window):
        main_window.addToolBarBreak(Qt.TopToolBarArea)
    main_window.addToolBar(Qt.TopToolBarArea, toolbar)
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
