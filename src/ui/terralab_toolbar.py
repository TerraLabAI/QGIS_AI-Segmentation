



from __future__ import annotations

from qgis.PyQt.QtWidgets import QToolBar

_TOOLBAR_OBJECT_NAME = "TerraLabToolbar"
_TOOLBAR_TITLE = "TerraLab Toolbar"


def get_or_create_terralab_toolbar(iface):

    main_window = iface.mainWindow()
    for tb in main_window.findChildren(QToolBar):
        if tb.objectName() == _TOOLBAR_OBJECT_NAME:
            return tb
    toolbar = QToolBar(_TOOLBAR_TITLE)
    toolbar.setObjectName(_TOOLBAR_OBJECT_NAME)
    iface.addToolBar(toolbar)
    return toolbar


def add_action_to_toolbar(toolbar, action, product_id, is_cross_promo=False):





    action.setProperty("terralab_product_id", product_id)
    action.setProperty("terralab_is_cross_promo", is_cross_promo)
    for existing in toolbar.actions():
        if existing.property("terralab_product_id") == product_id and existing is not action:
            if is_cross_promo and not existing.property("terralab_is_cross_promo"):
                return
            toolbar.removeAction(existing)
            break
    for existing in toolbar.actions():
        if _toolbar_sort_key(existing) > _toolbar_sort_key(action):
            toolbar.insertAction(existing, action)
            return
    toolbar.addAction(action)


def _toolbar_sort_key(action) -> str:





    return str(action.property("terralab_product_id") or action.text())


def remove_action_from_toolbar(toolbar, action, main_window):

    toolbar.removeAction(action)
    remaining = [a for a in toolbar.actions() if not a.isSeparator()]
    if not remaining:
        main_window.removeToolBar(toolbar)
        toolbar.deleteLater()
