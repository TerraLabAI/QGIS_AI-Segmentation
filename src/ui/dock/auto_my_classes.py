










from __future__ import annotations

import json

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QAbstractButton,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from .auto_flow_look import _BTN_AUTO_QUIET
from .font_scale import scale_px_length, scale_qss_font_px
from .styles import (
    FIELD,
    FONT_BODY,
    FONT_HINT,
    INK,
    INK_2,
    LINE,
    RADIUS_CONTROL,
    _segmented_switch_qss,
)

__all__ = ["DockMyClassesMixin"]

_SETTINGS_ROWS = "AI_Segmentation/my_classes_rows"
_SETTINGS_CHOICE = "AI_Segmentation/land_cover_choice"


def everything_else_name() -> str:
    return tr("Everything else")


class _Swatch(QAbstractButton):


    def __init__(self, color: str, editable: bool = True, parent=None):
        super().__init__(parent)
        self.color = color
        side = scale_px_length(20)
        self.setFixedSize(side, side)
        self.setEnabled(editable)
        if editable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip(tr("Change colour"))
            self.setAccessibleName(tr("Change colour"))
            self.clicked.connect(self._pick)

    def _pick(self) -> None:
        from qgis.PyQt.QtWidgets import QColorDialog
        picked = QColorDialog.getColor(QColor(self.color), self.window(), tr("Class colour"))
        if picked.isValid():
            self.color = picked.name().lower()
            self.update()

    def paintEvent(self, _event):  # noqa: N802
        try:
            from qgis.PyQt.QtCore import QRectF
            from qgis.PyQt.QtGui import QPainter, QPen

            from .auto_target_mode import land_cover_edge_color
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(land_cover_edge_color(), 1))
            painter.setBrush(QColor(self.color))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5), 4, 4)
            painter.end()
        except Exception:  # noqa: BLE001  # nosec B110
            pass


def _table_qss() -> str:
    return scale_qss_font_px(
        "QWidget#myClassesTable { background: transparent; }"
        f"QLineEdit#myClassName {{ background-color: {FIELD}; color: {INK};"
        f" border: 1px solid {LINE}; border-radius: {RADIUS_CONTROL}px;"
        f" padding: 2px 6px; font-size: {FONT_BODY}px; }}"
        f"QLabel#myClassFixed {{ color: {INK_2}; font-size: {FONT_BODY}px;"
        " padding: 2px 7px; background: transparent; }"
        f"QLabel#myClassNote {{ color: {INK_2}; font-size: {FONT_HINT}px; background: transparent; }}"
    )


class DockMyClassesMixin:


    def _build_my_classes_section(self, col) -> None:
        host = QWidget()
        host.setObjectName("myClassesSection")
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        switch = QFrame()
        switch.setObjectName("myClassesSwitch")
        switch.setStyleSheet(_segmented_switch_qss("myClassesSwitch"))
        row = QHBoxLayout(switch)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(2)
        self._my_classes_choice_btns = {}
        for key, text in (("standard", tr("6 standard classes")), ("mine", tr("My classes"))):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            btn.setAutoDefault(False)
            btn.clicked.connect(lambda _c=False, k=key: self._set_my_classes_choice(k, remember=True))
            row.addWidget(btn, 1)
            self._my_classes_choice_btns[key] = btn
        lay.addWidget(switch)

        table_host = QWidget()
        table_host.setObjectName("myClassesTable")
        table_host.setStyleSheet(_table_qss())
        tcol = QVBoxLayout(table_host)
        tcol.setContentsMargins(0, 0, 0, 0)
        tcol.setSpacing(4)
        self._my_classes_rows_box = QVBoxLayout()
        self._my_classes_rows_box.setSpacing(4)
        tcol.addLayout(self._my_classes_rows_box)
        fixed = QHBoxLayout()
        fixed.setSpacing(6)
        from ...core.my_classes import EVERYTHING_ELSE_COLOR
        fixed.addWidget(_Swatch(EVERYTHING_ELSE_COLOR, editable=False))
        fixed_label = QLabel(everything_else_name())
        fixed_label.setObjectName("myClassFixed")
        fixed_label.setToolTip(tr("Every pixel no class above describes. Always the last row."))
        fixed.addWidget(fixed_label, 1)
        tcol.addLayout(fixed)

        actions = QHBoxLayout()
        actions.setSpacing(2)
        for key, text, tip in (
                ("add", tr("Add class"), ""),
                ("paste", tr("Paste list"), tr("One class per line, like: Palm trees #1B5E20")),
                ("standard", tr("Start from: 6 standard classes"),
                 tr("Fill the table with the six standard classes and their colours."))):
            btn = QPushButton(text)
            btn.setStyleSheet(_BTN_AUTO_QUIET)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            btn.setAutoDefault(False)
            if tip:
                btn.setToolTip(tip)
            btn.clicked.connect(lambda _c=False, k=key: self._on_my_classes_action(k))
            actions.addWidget(btn)
        actions.addStretch(1)
        tcol.addLayout(actions)
        self._my_classes_note = QLabel("")
        self._my_classes_note.setObjectName("myClassNote")
        self._my_classes_note.setWordWrap(True)
        self._my_classes_note.setVisible(False)
        tcol.addWidget(self._my_classes_note)
        lay.addWidget(table_host)

        self._my_classes_table = table_host
        self._my_classes_section = host
        self._my_classes_row_widgets: list = []
        col.addWidget(host)
        self._my_classes_load()
        self._refresh_my_classes_section()



    def my_classes_active(self) -> bool:

        from ...core.my_classes import enabled
        return self.__dict__.get("_my_classes_choice") == "mine" and enabled()

    def my_classes_rows(self) -> list[tuple[str, str]]:

        out = []
        for widget in self.__dict__.get("_my_classes_row_widgets") or ():
            out.append((widget._name.text().strip(), widget._swatch.color))
        return out

    def set_my_classes_note(self, text: str) -> None:
        note = self.__dict__.get("_my_classes_note")
        if note is not None:
            note.setText(text or "")
            note.setVisible(bool(text))

    def _refresh_my_classes_section(self) -> None:
        host = self.__dict__.get("_my_classes_section")
        if host is None:
            return
        from ...core.my_classes import enabled
        on = enabled()
        host.setVisible(on)
        mine = on and self.__dict__.get("_my_classes_choice") == "mine"
        for key, btn in (self.__dict__.get("_my_classes_choice_btns") or {}).items():
            btn.blockSignals(True)
            btn.setChecked((key == "mine") == mine)
            btn.blockSignals(False)
        self._my_classes_table.setVisible(mine)
        line = self.__dict__.get("_auto_lc_line")
        if line is not None:
            line.setText(tr("One layer, your classes, no gaps.") if mine
                         else tr("One layer, 6 classes, no gaps."))
        chips = self.__dict__.get("_auto_lc_chips")
        if chips is not None:
            chips.setVisible(not mine and bool(getattr(chips, "_items", None)))

    def _set_my_classes_choice(self, key: str, remember: bool = False) -> None:
        self._my_classes_choice = "mine" if key == "mine" else "standard"
        if self._my_classes_choice == "mine" and not self._my_classes_row_widgets:
            self._my_classes_fill_standard()
        self._refresh_my_classes_section()
        if remember:
            self._my_classes_save()



    def _my_classes_add_row(self, name: str = "", color: str | None = None) -> None:
        from ...core.my_classes import default_colors, max_classes
        if len(self._my_classes_row_widgets) >= max_classes():
            self.set_my_classes_note(tr("Use at most {n} classes.").format(n=max_classes()))
            return
        if not color:
            used = {w._swatch.color for w in self._my_classes_row_widgets}
            ramp = default_colors(max_classes())
            color = next((c for c in ramp if c not in used), ramp[len(used) % len(ramp)])
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        row._swatch = _Swatch(color)
        lay.addWidget(row._swatch)
        row._name = QLineEdit(name)
        row._name.setObjectName("myClassName")
        row._name.setPlaceholderText(tr("Class name"))
        row._name.setMaxLength(40)
        row._name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row._name.editingFinished.connect(self._my_classes_save)
        lay.addWidget(row._name, 1)
        remove = QPushButton("×")
        remove.setStyleSheet(_BTN_AUTO_QUIET)
        remove.setCursor(Qt.CursorShape.PointingHandCursor)
        remove.setToolTip(tr("Remove class"))
        remove.setAccessibleName(tr("Remove class"))
        remove.setAutoDefault(False)
        remove.clicked.connect(lambda _c=False, r=row: self._my_classes_remove_row(r))
        lay.addWidget(remove)
        self._my_classes_rows_box.addWidget(row)
        self._my_classes_row_widgets.append(row)
        self.set_my_classes_note("")

    def _my_classes_remove_row(self, row) -> None:
        if row in self._my_classes_row_widgets:
            self._my_classes_row_widgets.remove(row)
        row.setParent(None)
        row.deleteLater()
        self._my_classes_save()

    def _my_classes_clear(self) -> None:
        for row in list(self._my_classes_row_widgets):
            row.setParent(None)
            row.deleteLater()
        self._my_classes_row_widgets = []

    def _my_classes_fill_standard(self) -> bool:
        from .auto_target_mode import last_land_cover_legend
        legend = [e for e in last_land_cover_legend() if isinstance(e, dict)]
        if not legend:
            self.set_my_classes_note(tr("The standard classes are still loading. Try again in a moment."))
            return False
        self._my_classes_clear()
        for entry in legend[:12]:
            self._my_classes_add_row(str(entry.get("name") or ""), entry.get("color") or None)
        self._my_classes_save()
        return True

    def _on_my_classes_action(self, key: str) -> None:
        if key == "add":
            self._my_classes_add_row()
            if self._my_classes_row_widgets:
                self._my_classes_row_widgets[-1]._name.setFocus()
        elif key == "standard":
            self._my_classes_fill_standard()
        elif key == "paste":
            self._my_classes_paste()
        self._my_classes_save()

    def _my_classes_paste(self) -> None:
        from qgis.PyQt.QtWidgets import QInputDialog

        from ...core.my_classes import max_classes, parse_paste_list
        text, ok = QInputDialog.getMultiLineText(
            self, tr("Paste list"),
            tr("One class per line, a name then a colour, like:\nPalm trees #1B5E20"), "")
        if not ok:
            return
        rows = parse_paste_list(text)
        if not rows:
            self.set_my_classes_note(tr("No class found in the pasted text."))
            return
        self._my_classes_clear()
        for name, color in rows[:max_classes()]:
            self._my_classes_add_row(name, color)
        if len(rows) > max_classes():
            self.set_my_classes_note(
                tr("Kept the first {n} classes.").format(n=max_classes()))



    def _my_classes_save(self) -> None:
        from qgis.core import QgsSettings
        try:
            QgsSettings().setValue(_SETTINGS_ROWS, json.dumps(self.my_classes_rows()))
            QgsSettings().setValue(_SETTINGS_CHOICE, self.__dict__.get("_my_classes_choice") or "standard")
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def _my_classes_load(self) -> None:
        from qgis.core import QgsSettings
        self._my_classes_choice = "standard"
        try:
            self._my_classes_choice = (
                "mine" if str(QgsSettings().value(_SETTINGS_CHOICE, "standard")) == "mine" else "standard")
            rows = json.loads(str(QgsSettings().value(_SETTINGS_ROWS, "[]") or "[]"))
        except Exception:  # noqa: BLE001
            rows = []
        for entry in rows if isinstance(rows, list) else ():
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                self._my_classes_add_row(str(entry[0]), str(entry[1]))
