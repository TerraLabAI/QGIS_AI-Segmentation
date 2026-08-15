"""The one sentence saying where a cloud run sends the image, and its widget.

Both panels show it, word for word. Semi-Auto uploads the crop under a click,
Automatic uploads the tiles of the drawn zone, and both land on the same
servers, so the sentence is the same and there is one of it. One served id, one
builder: a reword on the site reaches both panels at once, in every language,
with no plugin release.

It is a permanent line on the card rather than a modal at the first cloud
Start, so it is visible before the click instead of blocking it.

It retires itself once a cloud run has completed (see core/cloud_notice_seen.py
for why that is safe, and for the rule that the flag may only ever change what
is on screen).

Never loose on the dock: every caller drops the label into a card that is
already there.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QLabel

from ...core.i18n import tr
from ...core.server_dials import dial_copy

# The separator the engine lines already use between two clauses.
_DOT = "·"

# Muted, and one step under the note it sits beneath. A disclosure has to be
# readable, not loud. Same pair as the Account Settings legal footer.
_CLOUD_NOTICE_QSS = ("font-size: 10px; color: rgba(128, 128, 128, 0.85);"
                     " background: transparent;")
_CLOUD_NOTICE_LINK_COLOR = "rgba(128, 128, 128, 0.85)"


def build_cloud_notice_line() -> QLabel:
    """The label, built and hidden. The caller adds it to its own card."""
    label = QLabel("")
    label.setObjectName("cloudNoticeLine")
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setOpenExternalLinks(True)
    label.setStyleSheet(_CLOUD_NOTICE_QSS)
    label.setVisible(False)
    return label


def cloud_notice_line_html() -> str:
    """The sentence, served or shipped, with the Privacy link on the end.

    ``str.replace``, never ``str.format``: a served sentence may carry a stray
    brace and this runs on a paint path.

    The link colour is named in the anchor. Qt paints a bare <a> in its own
    hyperlink blue, which is nearly black against the dark QGIS theme.
    """
    from ...core.activation_manager import get_privacy_url

    link = (f'<a href="{get_privacy_url()}"'
            f' style="color: {_CLOUD_NOTICE_LINK_COLOR};">{tr("Privacy")}</a>')
    text = dial_copy(
        "engine.privacy_line",
        tr("Your selection is sent to our servers in Europe {dot} {privacy}"))
    return text.replace("{dot}", _DOT).replace("{privacy}", link)
