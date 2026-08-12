"""The one dialog that asks before any imagery leaves the machine.

Shown at the FIRST Semi-Auto start on Cloud AI, and never again once answered
yes. It is not part of sign-in on purpose: an account form is the wrong place
to agree to something that happens minutes later, and a user who reads it there
has forgotten it by the time it applies.

Two facts, in the order a user asks them: what is sent, and what it costs.
The way out is a real button of its own, because a consent screen whose only
alternative is Cancel is not a choice.

Free function rather than a dock mixin: it holds no state, it is called from
one place, and a dialog that can be opened without a dock is a dialog that can
be looked at in isolation.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ...core.i18n import tr
from .styles import _BTN_GHOST, _BTN_GREEN, BRAND_BLUE, _btn_start_qss

# Width the dock itself uses. The dialog says the same things the panel does,
# so it wraps its lines the same way rather than sprawling across a monitor.
_DIALOG_WIDTH = 380

_PRIVACY_URL = (
    "https://terra-lab.ai/privacy-policy"
    "?utm_source=qgis&utm_medium=plugin"
    "&utm_campaign=ai-segmentation&utm_content=cloud_consent"
)


def ask_cloud_consent(parent=None) -> bool:
    """Put the cloud notice on screen. True when the user said yes.

    Closing the window with its own X is a no, like the offline button: the
    only path that returns True is the one where somebody read the lines and
    pressed the green button.
    """
    return build_cloud_consent_dialog(parent).exec() == QDialog.DialogCode.Accepted


def build_cloud_consent_dialog(parent=None) -> QDialog:
    """The dialog, built and not yet shown.

    Split from the asking so the window can be rendered and looked at without
    a human to answer it. A consent screen nobody has seen is a consent screen
    nobody has checked.
    """
    dialog = QDialog(parent)
    dialog.setWindowTitle(tr("Cloud AI"))
    dialog.setModal(True)
    dialog.setMinimumWidth(_DIALOG_WIDTH)

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(8)

    title = QLabel(tr("How Cloud AI works"))
    title.setWordWrap(True)
    title.setStyleSheet(
        "font-size: 14px; font-weight: bold; color: palette(text);")
    layout.addWidget(title)
    layout.addSpacing(4)

    for line in (
        # Name the destination. A GIS user's default assumption is that a
        # plugin works on their own machine, and most of them have never had
        # one send imagery anywhere, so "our servers" is not enough: say where
        # they are.
        tr("Each click sends a small square of the image to our servers in "
           "Europe, and the outline comes back."),
        # What stays, said as a fact rather than as a denial. "Nothing else
        # leaves your computer" answers a fear by repeating it, and the line
        # about going back only restated the button underneath.
        tr("Your project and your files stay on your computer. One cloud "
           "detection per object you save."),
    ):
        body = QLabel(line)
        body.setWordWrap(True)
        body.setStyleSheet("font-size: 12px; color: palette(text);")
        layout.addWidget(body)

    layout.addSpacing(10)

    accept_btn = QPushButton(tr("Continue with Cloud AI"))
    accept_btn.setMinimumHeight(38)
    accept_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    accept_btn.setStyleSheet(_btn_start_qss(_BTN_GREEN))
    accept_btn.setDefault(True)
    accept_btn.clicked.connect(dialog.accept)
    layout.addWidget(accept_btn)

    decline_btn = QPushButton(tr("Use my computer instead"))
    decline_btn.setMinimumHeight(34)
    decline_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    decline_btn.setStyleSheet(_BTN_GHOST)
    decline_btn.clicked.connect(dialog.reject)
    layout.addWidget(decline_btn)

    # Name the colour in the anchor. Qt paints a bare <a> in its own default
    # hyperlink blue, which is nearly black against the dark QGIS theme.
    privacy = QLabel(
        f'<a href="{_PRIVACY_URL}" style="color: {BRAND_BLUE};">'
        f'{tr("Read the privacy policy")}</a>')
    privacy.setOpenExternalLinks(True)
    privacy.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    privacy.setStyleSheet("font-size: 11px;")
    layout.addWidget(privacy)

    return dialog
