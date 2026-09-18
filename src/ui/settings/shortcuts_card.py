











from __future__ import annotations

import sys

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QKeySequence
from qgis.PyQt.QtWidgets import (
    QBoxLayout,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from ..dock.font_scale import apply_font_scale_to_tree, scale_px_length, scale_qss_font_px
from ..dock.styles import (
    _BTN_SETTINGS_ACCENT,
    FIELD,
    FONT_BASE,
    FONT_BODY,
    FONT_HINT,
    INK,
    INK_2,
    LINE,
    RADIUS_CARD,
    RADIUS_CHIP,
    SURFACE,
)
from ..dock.widgets import native_key
from .settings_widgets import SCROLL_QSS, TITLE_PX, settings_button

_CAP_PX = 22

_TWO_COLUMNS_MIN_W = 700
SHORTCUTS_QSS = scale_qss_font_px(
    "QDialog#shortcutsDialog { background: palette(window); }"
    f"QLabel#shTitle {{ font-size: {TITLE_PX}px; font-weight: 600; color: {INK}; background: transparent; }}"
    f"QLabel#shSub {{ font-size: {FONT_BODY}px; color: {INK_2}; background: transparent; }}"
    f"QFrame#shCard {{ background: {SURFACE}; border: 1px solid {LINE};"
    f" border-radius: {RADIUS_CARD}px; }}"
    f"QFrame#shColumnRule {{ background: {LINE}; border: none; min-width: 1px; max-width: 1px; }}"
    f"QLabel#shSection {{ font-size: {FONT_HINT}px; font-weight: 600; color: {INK_2};"
    " background: transparent; }"
    f"QLabel#shAction {{ font-size: {FONT_BASE}px; color: {INK}; background: transparent; }}"
    f"QLabel#shJoin {{ font-size: {FONT_HINT}px; color: {INK_2}; background: transparent; }}"



    f"QLabel#shKey {{ font-size: {FONT_BODY}px; color: {INK_2};"
    f" background: {FIELD}; border: 1px solid {LINE}; border-radius: {RADIUS_CHIP}px;"
    " padding: 0 7px; }"
    f"QLabel#shWord {{ font-size: {FONT_HINT}px; color: {INK_2}; background: {FIELD};"
    f" border: 1px solid {LINE}; border-radius: {RADIUS_CHIP}px; padding: 0 7px; }}"
)


def _key_cap(text: str, parent: QWidget, mono: bool) -> QLabel:
    cap = QLabel(text, parent)
    cap.setObjectName("shKey" if mono else "shWord")
    cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
    cap.setFixedHeight(scale_px_length(_CAP_PX))
    return cap


def _shortcut_row(parent: QWidget, action: str, caps: tuple) -> QWidget:

    row = QWidget(parent)
    line = QHBoxLayout(row)
    line.setContentsMargins(0, 0, 0, 0)
    line.setSpacing(10)
    title = QLabel(action, row)
    title.setObjectName("shAction")
    title.setWordWrap(True)
    line.addWidget(title, 1)
    keys = QHBoxLayout()
    keys.setContentsMargins(0, 0, 0, 0)
    keys.setSpacing(4)
    for index, (label, mono) in enumerate(caps):
        if index:
            joiner = QLabel(tr("or"), row)
            joiner.setObjectName("shJoin")
            keys.addWidget(joiner)
        keys.addWidget(_key_cap(label, row, mono))
    line.addLayout(keys, 0)
    line.setAlignment(keys, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
    return row


def _shortcut_column(parent: QWidget, sections: tuple) -> QWidget:
    column = QWidget(parent)
    col = QVBoxLayout(column)
    col.setContentsMargins(0, 0, 0, 0)
    col.setSpacing(6)
    for index, (heading, rows) in enumerate(sections):

        label = QLabel(heading, column)
        label.setObjectName("shSection")
        label.setContentsMargins(0, 14 if index else 0, 0, 6)
        col.addWidget(label)
        for action, caps in rows:
            col.addWidget(_shortcut_row(column, action, caps))
    col.addStretch(1)
    return column


def shortcut_sections() -> tuple:





    def key(*labels: str) -> tuple:
        return tuple((label, True) for label in labels)

    def word(label: str) -> tuple:
        return ((label, False),)

    undo = native_key(QKeySequence.StandardKey.Undo)
    backspace = native_key("Backspace")
    enter = native_key("Return")
    esc = native_key("Esc")


    delete = (native_key("Del"),)
    if sys.platform == "darwin":
        delete = (native_key("Del"), native_key("Ctrl+Backspace"))

    general = ((tr("Start (the visible mode's Start button)"), key("G")),)
    semi_auto = (
        (tr("Add area"), word(tr("Left-click"))),
        (tr("Remove area"), word(tr("Right-click"))),
        (tr("Undo last point"), key(undo, backspace)),
        (tr("Save polygon"), key("S")),
        (tr("Open the selected saved polygon for AI editing"), key("E")),
        (tr("Delete the active object"), key(*delete)),
        (tr("Export polygon to a layer"), key(enter)),
        (tr("Clear the selection in progress"), key("C")),
        (tr("Clear the selection, or stop the segmentation"), key(esc)),
    )
    zone = (
        (tr("Add a point"), word(tr("Click"))),



        (tr("Finish the zone"), word(tr("Double-click")) + key(enter)),
        (tr("Undo last point"), key(undo, backspace)),
        (tr("Clear the points, then exit Automatic"), key(esc)),
    )
    detect = (
        (tr("Run the detection"), key(enter)),

        (tr("Cancel the example, the detection, or exit Automatic"), key(esc)),
    )
    review = (
        (tr("Export polygons to a layer"), key(enter)),
        (tr("Remove the selected detection"), key(*delete)),
        (tr("Undo the last correction"), key(undo)),
        (tr("Save the fix and go back to the review"), key("S")),
        (tr("Close the fix, clear the selection, or exit the review"), key(esc)),
    )
    merge = (
        (tr("Pick or un-pick an object"), word(tr("Click"))),
        (tr("Confirm the merge"), key(enter)),
        (tr("Cancel the merge"), key(esc)),
    )
    navigation = (
        (tr("Hold and move to pan the map"), word(tr("Space"))),
        (tr("Pan the map"), word(tr("Arrow keys"))),
    )
    return (
        ((tr("General"), general), (tr("Semi-Auto"), semi_auto),
         (tr("Automatic: draw the zone"), zone)),
        ((tr("Automatic: detect"), detect), (tr("Automatic: review and Correct"), review),
         (tr("Automatic: merge with neighbours"), merge),
         (tr("Navigation (while a tool is armed)"), navigation)),
    )


class ShortcutsCard(QFrame):


    def __init__(self, parent=None):
        super().__init__(parent)
        left, right = shortcut_sections()
        self.setObjectName("shCard")
        self.setStyleSheet(SHORTCUTS_QSS)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._columns = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        self._columns.setContentsMargins(16, 14, 16, 14)
        self._columns.setSpacing(24)
        self._columns.addWidget(_shortcut_column(self, left), 1)
        self._rule = QFrame(self)
        self._rule.setObjectName("shColumnRule")
        self._rule.setFrameShape(QFrame.Shape.NoFrame)
        self._columns.addWidget(self._rule)
        self._columns.addWidget(_shortcut_column(self, right), 1)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        stacked = event.size().width() < scale_px_length(_TWO_COLUMNS_MIN_W)
        direction = (QBoxLayout.Direction.TopToBottom if stacked
                     else QBoxLayout.Direction.LeftToRight)
        if self._columns.direction() != direction:
            self._columns.setDirection(direction)
            self._rule.setVisible(not stacked)


def build_shortcuts_card(parent: QWidget) -> QFrame:

    return ShortcutsCard(parent)


def show_shortcuts_dialog(parent=None) -> None:

    dlg = build_shortcuts_dialog(parent)
    dlg.exec()
    dlg.deleteLater()


def build_shortcuts_dialog(parent=None) -> QDialog:
    dlg = QDialog(parent)
    dlg.setObjectName("shortcutsDialog")
    dlg.setWindowTitle(tr("Keyboard shortcuts"))
    dlg.setStyleSheet(SHORTCUTS_QSS)
    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 18, 20, 16)
    layout.setSpacing(12)
    title = QLabel(tr("Keyboard shortcuts"), dlg)
    title.setObjectName("shTitle")
    from .category_tile import category_icon_tile, tile_beside



    layout.addLayout(tile_beside(category_icon_tile("terminal", "leaf", dlg), title))
    subtitle = QLabel(tr("Every key the panel answers, grouped by where it works."), dlg)
    subtitle.setObjectName("shSub")
    subtitle.setWordWrap(True)
    layout.addWidget(subtitle)

    scroll = QScrollArea(dlg)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setStyleSheet(SCROLL_QSS)
    card = build_shortcuts_card(scroll)
    scroll.setWidget(card)
    layout.addWidget(scroll, 1)
    foot = QHBoxLayout()
    foot.addStretch(1)

    done = settings_button(tr("Done"), _BTN_SETTINGS_ACCENT, dlg)
    done.setMinimumWidth(scale_px_length(96))
    done.clicked.connect(dlg.accept)
    foot.addWidget(done)
    layout.addLayout(foot)
    apply_font_scale_to_tree(dlg)


    width = scale_px_length(760)
    height = card.sizeHint().height() + scale_px_length(170)
    try:
        available = dlg.screen().availableGeometry()
        width = min(width, available.width() - 48)
        height = min(height, available.height() - 80)
    except (AttributeError, RuntimeError):
        pass  # nosec B110
    dlg.setMinimumSize(min(width, scale_px_length(420)), min(height, scale_px_length(320)))
    dlg.resize(width, height)
    return dlg
