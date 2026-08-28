

















from __future__ import annotations

from ..core.i18n import tr


URL = "url"
EMAIL = "email"
FOLDER = "folder"


def _copy_to_clipboard(text: str) -> bool:

    try:
        from qgis.PyQt.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        if clipboard is None:
            return False
        clipboard.setText(text)
        return True
    except (RuntimeError, AttributeError, ImportError):
        return False


def _wording(kind: str, copied: bool) -> tuple[str, str]:

    if kind == FOLDER:
        return (
            tr("QGIS could not open a file manager."),
            tr("The folder is copied to your clipboard: paste it into your "
               "file manager.") if copied else
            tr("Copy the folder below and paste it into your file manager."))
    if kind == EMAIL:
        return (
            tr("QGIS could not open your email app."),
            tr("The support address is copied to your clipboard: paste it "
               "into your email app.") if copied else
            tr("Copy the support address below into your email app."))
    return (
        tr("QGIS could not open a browser."),
        tr("The address is copied to your clipboard: paste it into a browser "
           "to continue.") if copied else
        tr("Copy the address below and paste it into a browser."))


def _address_box_owner(parent):






    try:
        from qgis.PyQt.QtWidgets import QApplication, QMenu

        if parent is not None and not isinstance(parent, QMenu) and parent.isVisible():
            return parent
        active = QApplication.activeModalWidget() or QApplication.activeWindow()
        if active is not None and not isinstance(active, QMenu):
            return active
        from qgis.utils import iface

        return iface.mainWindow() if iface is not None else parent
    except (RuntimeError, AttributeError, ImportError):
        return None


def _show_address(address: str, parent=None, kind: str = URL) -> None:

    copied = _copy_to_clipboard(address)
    opening, instruction = _wording(kind, copied)
    try:
        from .dialogs.confirm_dialog import info_box



        info_box(_address_box_owner(parent), tr("Open it yourself"),
                 f"{opening} {instruction}", detail=address, selectable=True)
    except (RuntimeError, AttributeError, ImportError):
        pass  # nosec B110


def _try_open(url_object) -> bool:

    try:
        from qgis.PyQt.QtGui import QDesktopServices

        return bool(QDesktopServices.openUrl(url_object))
    except (RuntimeError, AttributeError, ImportError):
        return False


def open_external_url(url: str, parent=None) -> bool:





    from qgis.PyQt.QtCore import QUrl

    if _try_open(QUrl(url)):
        return True
    _show_address(url, parent=parent, kind=URL)
    return False


def open_local_path(path: str, parent=None) -> bool:

    from qgis.PyQt.QtCore import QUrl

    if _try_open(QUrl.fromLocalFile(path)):
        return True
    _show_address(path, parent=parent, kind=FOLDER)
    return False


def open_email(mailto_url: str, address: str, parent=None) -> bool:





    from qgis.PyQt.QtCore import QUrl

    if _try_open(QUrl(mailto_url)):
        return True
    _show_address(address, parent=parent, kind=EMAIL)
    return False
