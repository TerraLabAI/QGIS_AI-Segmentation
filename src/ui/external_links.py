"""Open a web address, a mail client or a local folder, and say so when nothing opened.

``QDesktopServices.openUrl`` returns False when the desktop has no handler for
the address. On Linux that is routine: no ``xdg-open``, no registered browser,
no portal inside a Flatpak or Snap sandbox. A managed Windows desktop can
refuse the same way. The call raises nothing, so a click on Upgrade, Dashboard,
Help or Open folder did nothing at all and said nothing at all.

Every external open goes through one of the three functions here. When the
desktop declines, the address lands on the clipboard and a small window shows
it, selectable, so the user can paste it in by hand. The address was already on
its way to a browser, so the clipboard is no wider a place than where it was
going.

The sign-in flow keeps its own fallback: it has a panel to write into and a
short-lived code to explain, which a generic window cannot do.
"""

from __future__ import annotations

from ..core.i18n import tr

# What the desktop failed to open, which decides what the window says.
URL = "url"
EMAIL = "email"
FOLDER = "folder"


def _copy_to_clipboard(text: str) -> bool:
    """Put ``text`` on the clipboard, reporting whether it landed there."""
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
    """The two sentences for this kind of address: what failed, what to do."""
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


def _show_address(address: str, parent=None, kind: str = URL) -> None:
    """Show an address the desktop refused to open, ready to be copied."""
    copied = _copy_to_clipboard(address)
    opening, instruction = _wording(kind, copied)
    try:
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import QMessageBox

        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle(tr("Open it yourself"))
        box.setText(f"{opening} {instruction}")
        box.setInformativeText(address)
        # Selectable, so the address can still be read and copied by hand
        # when the clipboard itself is unavailable.
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.exec()
    except (RuntimeError, AttributeError, ImportError):
        pass  # nosec B110 - no window available; the clipboard already has it


def _try_open(url_object) -> bool:
    """Hand one QUrl to the desktop, treating any Qt failure as a refusal."""
    try:
        from qgis.PyQt.QtGui import QDesktopServices

        return bool(QDesktopServices.openUrl(url_object))
    except (RuntimeError, AttributeError, ImportError):
        return False


def open_external_url(url: str, parent=None) -> bool:
    """Open ``url`` in the browser, or hand the user the address instead.

    Returns True when the desktop took the address. A False return has already
    been shown to the user, so a caller does not have to report it again.
    """
    from qgis.PyQt.QtCore import QUrl

    if _try_open(QUrl(url)):
        return True
    _show_address(url, parent=parent, kind=URL)
    return False


def open_local_path(path: str, parent=None) -> bool:
    """Open a local file or folder, or hand the user the path instead."""
    from qgis.PyQt.QtCore import QUrl

    if _try_open(QUrl.fromLocalFile(path)):
        return True
    _show_address(path, parent=parent, kind=FOLDER)
    return False


def open_email(mailto_url: str, address: str, parent=None) -> bool:
    """Open the mail client, or show the plain address instead.

    The window shows ``address`` rather than ``mailto_url``: a user pasting by
    hand wants the address, not a scheme and a percent-encoded subject.
    """
    from qgis.PyQt.QtCore import QUrl

    if _try_open(QUrl(mailto_url)):
        return True
    _show_address(address, parent=parent, kind=EMAIL)
    return False
