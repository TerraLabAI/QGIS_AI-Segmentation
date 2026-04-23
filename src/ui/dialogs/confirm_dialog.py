


























from __future__ import annotations

from typing import NamedTuple

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ...core.i18n import tr
from ..dock.font_scale import scale_px_length, scale_qss_font_px
from ..dock.styles import (
    _BTN_GHOST,
    _BTN_PRIMARY,
    _BTN_RED_OUTLINE,
    ACCENT_BORDER,
    BTN_PILL_PX,
    FONT_BASE,
    FONT_BODY,
    INK,
    INK_2,
    INSET,
    LINE,
    RADIUS_CONTROL,
    RADIUS_PILL,
    RED_INK,
    RED_TINT,
    SURFACE,
)


PRIMARY = "primary"
DANGER = "danger"
SECONDARY = "secondary"
DISCARD = "discard"



INFO = "info"
WARNING = "warning"
SUCCESS = "success"
ERROR = "error"
_TONE_GLYPHS = {INFO: "info", WARNING: "warning", SUCCESS: "check", ERROR: "warning"}


_TILE_PX = 30
_GLYPH_PX = 17

_WIDTH_PX = 440
_WIDTH_MAX_PX = 600



_WINDOW_TITLE = "AI Segmentation"

_FOCUS_RING = f"QPushButton:focus {{ border: 2px solid {ACCENT_BORDER}; }}"

_FOCUS_RING_THIN = f"QPushButton:focus {{ border: 1px solid {ACCENT_BORDER}; }}"

_DIALOG_QSS = f"QDialog#confirmDialog {{ background: {SURFACE}; }}"



_TITLE_QSS = (f"font-size: {FONT_BASE + 3}px; font-weight: 600; color: {INK};"
              " background: transparent;")
_BODY_QSS = (f"font-size: {FONT_BASE}px; color: {INK_2};"
             " background: transparent;")
_DETAIL_QSS = (f"font-size: {FONT_BODY}px; color: {INK}; background: {INSET};"
               f" border: 1px solid {LINE}; border-radius: {RADIUS_CONTROL}px;"
               " padding: 8px 10px;")



_BTN_DISCARD = (
    f"QPushButton {{ background: transparent; color: {RED_INK};"
    f" padding: 0 10px; min-height: {BTN_PILL_PX}px;"
    f" border: 2px solid transparent; border-radius: {RADIUS_PILL}px;"
    f" font-size: {FONT_BODY}px; font-weight: 600; }}"
    f"QPushButton:hover {{ background: {RED_TINT}; }}"
    f"QPushButton:pressed {{ background: {RED_TINT}; }}"
)

_KIND_QSS = {
    PRIMARY: _BTN_PRIMARY,

    DANGER: _BTN_RED_OUTLINE + _FOCUS_RING_THIN,
    SECONDARY: _BTN_GHOST,
    DISCARD: _BTN_DISCARD + _FOCUS_RING,
}



_KIND_BORDER_PX = {PRIMARY: 0, DANGER: 1, SECONDARY: 1, DISCARD: 2}


class ChoiceButton(NamedTuple):


    key: str
    label: str
    kind: str = SECONDARY


class ConfirmDialog(QDialog):


    def __init__(self, parent, title: str, body: str, buttons,
                 default: str | None = None, escape: str | None = None,
                 tone: str | None = None, detail: str = "",
                 selectable: bool = False):
        super().__init__(parent)
        buttons = list(buttons)
        keys = [b.key for b in buttons]
        if escape is None:
            escape = keys[-1] if len(keys) == 1 else next(
                (b.key for b in buttons if b.kind == SECONDARY), keys[0])
        self._escape = escape
        self.chosen = escape
        self.buttons: dict[str, QPushButton] = {}

        self.setObjectName("confirmDialog")
        self.setWindowTitle(_WINDOW_TITLE)
        self.setStyleSheet(_DIALOG_QSS)
        self.setModal(True)
        self.setAccessibleName(title)
        self.setAccessibleDescription(body)


        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 18)
        layout.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(10)
        glyph_name = _TONE_GLYPHS.get(tone or "")
        indent = 0
        if glyph_name:
            from ..settings.category_tile import TONE_CATEGORIES, category_icon_tile

            glyph = category_icon_tile(glyph_name, TONE_CATEGORIES.get(tone or ""),
                                       self, _TILE_PX, _GLYPH_PX)
            glyph.setObjectName("confirmDialogGlyph")
            head.addWidget(glyph, 0, Qt.AlignmentFlag.AlignTop)
            indent = glyph.width() + head.spacing()
        self.title_label = QLabel(title)
        self.title_label.setObjectName("confirmDialogTitle")
        self.title_label.setWordWrap(True)
        self.title_label.setTextFormat(Qt.TextFormat.PlainText)
        self.title_label.setStyleSheet(scale_qss_font_px(_TITLE_QSS))
        if indent:

            self.title_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.title_label.setMinimumHeight(indent - head.spacing())
        else:
            self.title_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        head.addWidget(self.title_label, 1)
        layout.addLayout(head)

        self.body_label = None
        if body:
            self.body_label = QLabel(body)
            self.body_label.setObjectName("confirmDialogBody")
            self.body_label.setWordWrap(True)
            self.body_label.setTextFormat(Qt.TextFormat.PlainText)
            self.body_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            self.body_label.setStyleSheet(scale_qss_font_px(_BODY_QSS))
            if selectable:
                self.body_label.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(self.body_label)
            if indent:
                layout.itemAt(layout.count() - 1).widget().setContentsMargins(
                    indent, 0, 0, 0)

        self.detail_label = None
        if detail:
            self.detail_label = QLabel(detail)
            self.detail_label.setObjectName("confirmDialogDetail")
            self.detail_label.setWordWrap(True)
            self.detail_label.setTextFormat(Qt.TextFormat.PlainText)
            self.detail_label.setStyleSheet(scale_qss_font_px(_DETAIL_QSS))
            self.detail_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.TextSelectableByKeyboard)
            row = QHBoxLayout()
            row.setContentsMargins(indent, 4, 0, 0)
            row.addWidget(self.detail_label)
            layout.addLayout(row)

        layout.addSpacing(12)
        left = [b for b in buttons if b.kind == DISCARD]
        right = [b for b in buttons if b.kind != DISCARD]


        right.sort(key=lambda b: b.kind in (PRIMARY, DANGER))
        made = {spec.key: self._make_button(spec, default)
                for spec in left + right}




        spacing = 8
        margins = layout.contentsMargins()
        row_need = (sum(b.sizeHint().width() for b in made.values())
                    + spacing * (len(made) + (1 if left else 0))
                    + margins.left() + margins.right())
        widest = scale_px_length(_WIDTH_MAX_PX)
        self.stacked = row_need > widest
        if self.stacked:
            column = QVBoxLayout()
            column.setSpacing(spacing)
            for spec in list(reversed(right)) + left:
                column.addWidget(made[spec.key])
            layout.addLayout(column)
            width = widest
        else:
            row = QHBoxLayout()
            row.setSpacing(spacing)
            for spec in left:
                row.addWidget(made[spec.key])
            row.addStretch(1)
            for spec in right:
                row.addWidget(made[spec.key])
            layout.addLayout(row)
            width = max(scale_px_length(_WIDTH_PX), row_need)



        self.setFixedWidth(width)
        layout.activate()
        if layout.hasHeightForWidth():
            self.setFixedHeight(layout.totalHeightForWidth(width))

    def _make_button(self, spec: ChoiceButton, default: str | None) -> QPushButton:
        btn = QPushButton(spec.label)
        btn.setObjectName(f"confirmDialogBtn_{spec.key}")
        box = (scale_px_length(BTN_PILL_PX)
               - 2 * _KIND_BORDER_PX.get(spec.kind, 1))
        btn.setStyleSheet(scale_qss_font_px(
            _KIND_QSS.get(spec.kind, _BTN_GHOST)
            + f"QPushButton {{ min-height: {box}px; max-height: {box}px; }}"))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)

        btn.setAccessibleName(spec.label.replace("&&", "\0").replace(
            "&", "").replace("\0", "&"))
        btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)


        is_default = (spec.kind == PRIMARY and default is not None
                      and spec.key == default)
        btn.setAutoDefault(False)
        btn.setDefault(is_default)
        btn.clicked.connect(lambda _checked=False, k=spec.key: self._answer(k))
        self.buttons[spec.key] = btn
        return btn

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def reject(self) -> None:

        self.chosen = self._escape
        super().reject()

    def _answer(self, key: str) -> None:
        self.chosen = key
        self.accept()


def _parent_ok(parent):
    try:
        if parent is not None:
            parent.isVisible()
        return parent
    except (RuntimeError, AttributeError):
        return None


def ask_choice(parent, title: str, body: str, buttons, default: str | None = None,
               escape: str | None = None, tone: str | None = None,
               detail: str = "", selectable: bool = False) -> str:






    dlg = ConfirmDialog(_parent_ok(parent), title, body, buttons,
                        default=default, escape=escape, tone=tone,
                        detail=detail, selectable=selectable)
    try:
        dlg.exec()
        return dlg.chosen
    finally:
        dlg.deleteLater()


def info_box(parent, title: str, body: str = "", detail: str = "",
             selectable: bool = False, tone: str = INFO) -> None:

    ask_choice(parent, title, body, [ChoiceButton("ok", tr("OK"), PRIMARY)],
               default="ok", escape="ok", tone=tone, detail=detail,
               selectable=selectable)


def warning_box(parent, title: str, body: str = "", detail: str = "",
                selectable: bool = False) -> None:

    info_box(parent, title, body, detail=detail, selectable=selectable,
             tone=WARNING)


def error_box(parent, title: str, body: str = "", detail: str = "",
              selectable: bool = False) -> None:

    info_box(parent, title, body, detail=detail, selectable=selectable,
             tone=ERROR)


def success_box(parent, title: str, body: str = "", detail: str = "") -> None:

    info_box(parent, title, body, detail=detail, tone=SUCCESS)


def question(parent, title: str, body: str, default_yes: bool = True,
             destructive: bool = False, yes_label: str | None = None,
             no_label: str | None = None, tone: str | None = None) -> bool:





    kind = DANGER if destructive else PRIMARY
    choice = ask_choice(
        parent, title, body,
        [ChoiceButton("no", no_label or tr("Cancel"), SECONDARY),
         ChoiceButton("yes", yes_label or tr("OK"), kind)],
        default="yes" if default_yes else None, escape="no", tone=tone)
    return choice == "yes"


__all__ = [
    "DANGER",
    "DISCARD",
    "ERROR",
    "INFO",
    "PRIMARY",
    "SECONDARY",
    "SUCCESS",
    "WARNING",
    "ChoiceButton",
    "ConfirmDialog",
    "ask_choice",
    "error_box",
    "info_box",
    "question",
    "success_box",
    "warning_box",
]
