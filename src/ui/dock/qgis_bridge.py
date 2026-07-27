




















from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from .font_scale import fit_spin_width, scale_qss_font_px
from .styles import (
    _BTN_BLUE_OUTLINE,
    _BTN_GHOST,
    _BTN_GREEN,
    _BTN_REMOVE_ROW,
    _FIELD_LABEL_QSS,
    _SUBCARD_MARGINS,
    _SUBCARD_QSS,
    BTN_PRIMARY_WIDE_PX,
    FONT_BASE,
    INK,
    ON_ACCENT,
    SPACE_CARD,
    _btn_toggle_qss,
    _msg_label_qss,
    msg_rich,
)
from .wrapping_button_row import WrappingButtonRow

_MANUAL_TOOL_RGB = (30, 136, 229)










_RESTING_CORRECT_WIDGETS = (
    "auto_correct_pick_hero",
    "auto_correct_or_row",
    "auto_add_lane_card",
    "auto_correct_delete_tip",
    "auto_correct_status",
    "auto_correct_summary_row",
)











_SESSION_HIDDEN_CORRECT_WIDGETS = (
    "auto_shape_merge_btn",
    "auto_correct_remove_row",
)


def _tool_instruction(tool: str) -> str:





    if tool == "add":
        return tr("Click each corner of the object, then Finish.")
    if tool == "vertex":
        return tr("Drag a corner to move it. Double-click an edge to add one.")
    if tool == "reshape":
        return tr("Draw the new edge: start outside the shape, cross it, end "
                  "outside, then Finish.")
    if tool == "split":
        return tr("Draw a line right across the shape, then Finish.")
    return tr("Pick a tool above, then edit the highlighted object.")


class DockQgisBridgeMixin:


    def _setup_qgis_bridge_banner(self, layout) -> None:





        self.qgis_bridge_banner = QWidget()
        self.qgis_bridge_banner.setObjectName("qgisBridgeBanner")
        self.qgis_bridge_banner.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.qgis_bridge_banner.setStyleSheet(
            _SUBCARD_QSS.format(name="qgisBridgeBanner") + "QLabel { background: transparent; border: none; }")
        col = QVBoxLayout(self.qgis_bridge_banner)
        col.setContentsMargins(*_SUBCARD_MARGINS)
        col.setSpacing(6)



        self.qgis_bridge_title = QLabel(tr("This polygon"))
        self.qgis_bridge_title.setWordWrap(True)
        self.qgis_bridge_title.setStyleSheet(scale_qss_font_px(
            f"font-weight: 600; font-size: {FONT_BASE}px; color: {INK};"))
        col.addWidget(self.qgis_bridge_title)







        self.qgis_bridge_points_row = QWidget()
        _points = QHBoxLayout(self.qgis_bridge_points_row)
        _points.setContentsMargins(0, 0, 0, 0)
        _points.setSpacing(6)
        _points_lbl = QLabel(tr("Points"))

        _points_lbl.setStyleSheet(_FIELD_LABEL_QSS)
        _points_tip = tr(
            "Thin this polygon's points before you edit them by hand. 100% "
            "keeps the outline as it is.")
        _points_lbl.setToolTip(_points_tip)
        _points.addWidget(_points_lbl)
        _points.addStretch(1)
        self.qgis_bridge_points_spin = QSpinBox()
        self.qgis_bridge_points_spin.setRange(10, 100)
        self.qgis_bridge_points_spin.setSingleStep(10)
        self.qgis_bridge_points_spin.setValue(100)
        self.qgis_bridge_points_spin.setSuffix(" %")
        fit_spin_width(self.qgis_bridge_points_spin, 62, 78)
        self.qgis_bridge_points_spin.setToolTip(_points_tip)
        self.qgis_bridge_points_spin.valueChanged.connect(
            lambda v: self.auto_qgis_bridge_points_changed.emit(int(v)))
        _points.addWidget(self.qgis_bridge_points_spin)
        self.qgis_bridge_points_row.setVisible(False)
        col.addWidget(self.qgis_bridge_points_row)


        tools = WrappingButtonRow(spacing=SPACE_CARD)
        self._qgis_bridge_tool_buttons = {}



        for key, text, tooltip in (
            ("vertex", tr("Move points"), tr(
                "Drag, add or delete the object's corners by hand.")),
            ("reshape", tr("Redraw edge"), tr(
                "Replace one side by drawing a new line across the outline.")),
            ("split", tr("Split"), tr(
                "Draw a line across the object to cut it into two.")),
        ):
            btn = QPushButton(text)
            btn.setStyleSheet(_btn_toggle_qss(
                _MANUAL_TOOL_RGB, INK, ON_ACCENT, quiet=True))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(tooltip)
            btn.clicked.connect(
                lambda _checked=False, tool=key:
                self.auto_qgis_bridge_tool_requested.emit(tool))
            self._qgis_bridge_tool_buttons[key] = btn
            tools.add_row_item(btn)
        col.addWidget(tools)




        self.qgis_bridge_line = QLabel("")
        self.qgis_bridge_line.setWordWrap(True)
        self.qgis_bridge_line.setStyleSheet(_msg_label_qss("armed"))
        col.addWidget(self.qgis_bridge_line)





        _actions = QHBoxLayout()
        _actions.setContentsMargins(0, 0, 0, 0)
        _actions.setSpacing(6)
        self.qgis_bridge_finish_btn = QPushButton(tr("Finish the line"))
        self.qgis_bridge_finish_btn.setStyleSheet(_BTN_BLUE_OUTLINE)
        self.qgis_bridge_finish_btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)
        self.qgis_bridge_finish_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.qgis_bridge_finish_btn.setToolTip(tr(
            "Close the line you are drawing. A right-click on the map does "
            "the same."))
        self.qgis_bridge_finish_btn.clicked.connect(
            lambda: self.auto_qgis_bridge_gesture_requested.emit("finish"))
        self.qgis_bridge_finish_btn.setVisible(False)
        self.qgis_bridge_done_btn = QPushButton(tr("Save"))
        self.qgis_bridge_done_btn.setStyleSheet(_BTN_GREEN)
        self.qgis_bridge_done_btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)
        self.qgis_bridge_done_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.qgis_bridge_done_btn.setToolTip(tr(
            "Keep these edits and go back to picking polygons."))
        self.qgis_bridge_done_btn.clicked.connect(
            self.auto_qgis_bridge_done_requested.emit)
        self.qgis_bridge_undo_btn = QPushButton(tr("Undo"))
        self.qgis_bridge_undo_btn.setStyleSheet(_BTN_GHOST)
        self.qgis_bridge_undo_btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)
        self.qgis_bridge_undo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.qgis_bridge_undo_btn.setToolTip(tr(
            "Undo the last thing you did here: the point you just placed, or "
            "the last edit."))
        self.qgis_bridge_undo_btn.clicked.connect(
            self.auto_qgis_bridge_undo_requested.emit)




        self.qgis_bridge_undo_btn.setVisible(False)
        _actions.addWidget(self.qgis_bridge_finish_btn, 1)
        _actions.addWidget(self.qgis_bridge_done_btn, 1)
        _actions.addWidget(self.qgis_bridge_undo_btn, 1)
        col.addLayout(_actions)




        self.qgis_bridge_delete_corner_btn = QPushButton(
            tr("Delete this corner"))
        self.qgis_bridge_delete_corner_btn.setStyleSheet(_BTN_REMOVE_ROW)
        self.qgis_bridge_delete_corner_btn.setCursor(
            Qt.CursorShape.PointingHandCursor)
        self.qgis_bridge_delete_corner_btn.setToolTip(tr(
            "Remove the corner you picked. The Delete key does the same."))
        self.qgis_bridge_delete_corner_btn.clicked.connect(
            lambda: self.auto_qgis_bridge_gesture_requested.emit(
                "delete_corner"))
        self.qgis_bridge_delete_corner_btn.setVisible(False)
        col.addWidget(self.qgis_bridge_delete_corner_btn)







        self.qgis_bridge_delete_btn = QPushButton(
            tr("Delete this polygon"))
        self.qgis_bridge_delete_btn.setStyleSheet(_BTN_REMOVE_ROW)
        self.qgis_bridge_delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.qgis_bridge_delete_btn.setToolTip(tr(
            "Delete this polygon and leave the manual edit. Anything you "
            "changed here and did not save goes with it. Undo brings the "
            "polygon back."))
        self.qgis_bridge_delete_btn.clicked.connect(
            self.auto_qgis_bridge_delete_requested.emit)
        self.qgis_bridge_delete_btn.setVisible(False)
        col.addWidget(self.qgis_bridge_delete_btn)

        self.qgis_bridge_banner.setVisible(False)
        layout.addWidget(self.qgis_bridge_banner)

    def enter_qgis_bridge_state(self) -> None:








        self._qgis_bridge_active_ui = True







        try:
            self.set_correct_session_active(True)
        except (RuntimeError, AttributeError):
            pass



        self._set_bridge_widget_visible("auto_correct_session_row", False)
        self.set_qgis_bridge_tool("vertex")


        self.set_qgis_bridge_undo_available(False)



        self.reset_qgis_bridge_points()
        self.set_qgis_bridge_points_visible(False)


        self.set_qgis_bridge_delete_visible(False)



        self.set_qgis_bridge_target("")
        for name in _RESTING_CORRECT_WIDGETS + _SESSION_HIDDEN_CORRECT_WIDGETS:
            self._set_bridge_widget_visible(name, False)
        self._set_bridge_merge_enabled(False)
        try:
            self._set_review_dials_locked(True, 1)
        except (RuntimeError, AttributeError):
            pass
        try:
            self.qgis_bridge_banner.setVisible(True)
        except (RuntimeError, AttributeError):

            pass

    def leave_qgis_bridge_state(self) -> None:





        self._qgis_bridge_active_ui = False



        try:
            self.set_correct_session_active(False)
        except (RuntimeError, AttributeError):
            pass
        try:
            self.qgis_bridge_banner.setVisible(False)
        except (RuntimeError, AttributeError):

            pass
        self.set_qgis_bridge_delete_corner_visible(False)
        self.set_qgis_bridge_delete_visible(False)



        self.set_qgis_bridge_line_open(False)
        self.set_qgis_bridge_undo_available(False)



        self._set_bridge_widget_visible("auto_correct_remove_row", True)
        self._set_bridge_merge_enabled(True)
        try:
            self._apply_merge_tile()
        except (RuntimeError, AttributeError):
            pass
        try:
            self.set_auto_review_step(1)



            self._refresh_correct_panels()
            self._refresh_correct_summary_row()
        except (RuntimeError, AttributeError):
            pass


        for name in ("auto_correct_status",):
            widget = getattr(self, name, None)
            if widget is None:
                continue
            try:
                self._set_bridge_widget_visible(name, bool(widget.text()))
            except (RuntimeError, AttributeError):
                pass

    def set_qgis_bridge_tool(self, tool: str) -> None:


        self._qgis_bridge_tool = str(tool)
        for key, btn in getattr(self, "_qgis_bridge_tool_buttons", {}).items():
            try:



                self._set_btn_armed(btn, key == tool)
            except (RuntimeError, AttributeError):
                pass


        if tool == "add":
            try:
                self.qgis_bridge_title.setText(tr("New polygon"))
            except (RuntimeError, AttributeError):
                pass
        else:
            self.set_qgis_bridge_target(
                getattr(self, "_qgis_bridge_target_label", ""))
        self.set_qgis_bridge_feedback("")


        self.set_qgis_bridge_line_open(False)
        self.set_qgis_bridge_delete_corner_visible(False)

    def set_qgis_bridge_line_open(self, open_: bool) -> None:





        open_ = bool(open_)
        self._qgis_bridge_line_open = open_
        for name, show in (("qgis_bridge_finish_btn", open_),
                           ("qgis_bridge_done_btn", not open_)):
            btn = getattr(self, name, None)
            if btn is None:
                continue
            try:
                btn.setVisible(show)
            except (RuntimeError, AttributeError):
                pass


        self._apply_qgis_bridge_undo_visible()

    def set_qgis_bridge_undo_available(self, available: bool) -> None:



        self._qgis_bridge_undo_ready = bool(available)
        self._apply_qgis_bridge_undo_visible()

    def _apply_qgis_bridge_undo_visible(self) -> None:




        btn = getattr(self, "qgis_bridge_undo_btn", None)
        if btn is None:
            return
        try:
            btn.setVisible(
                bool(getattr(self, "_qgis_bridge_undo_ready", False))
                or bool(getattr(self, "_qgis_bridge_line_open", False)))
        except (RuntimeError, AttributeError):
            pass

    def set_qgis_bridge_target(self, label: str) -> None:



        self._qgis_bridge_target_label = str(label or "")
        try:
            text = str(label or "").strip()
            self.qgis_bridge_title.setText(text or tr("This polygon"))
        except (RuntimeError, AttributeError, TypeError):

            pass

    def set_qgis_bridge_last_change(self, text: str, can_undo: bool) -> None:






        self.set_qgis_bridge_undo_available(can_undo)
        self.set_qgis_bridge_feedback(text, "success" if text else "armed")

    def set_qgis_bridge_feedback(self, text: str, kind: str = "armed") -> None:


        try:
            lbl = self.qgis_bridge_line
        except AttributeError:
            return
        if not text:
            text = _tool_instruction(getattr(self, "_qgis_bridge_tool", ""))
            kind = "armed"
        try:
            lbl.setStyleSheet(_msg_label_qss(kind))
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setText(msg_rich(kind, text))
            lbl.setVisible(True)
        except (RuntimeError, AttributeError, KeyError):

            pass

    def set_qgis_bridge_delete_corner_visible(self, visible: bool) -> None:



        try:
            self.qgis_bridge_delete_corner_btn.setVisible(bool(visible))
        except (RuntimeError, AttributeError):

            pass

    def set_qgis_bridge_delete_visible(self, visible: bool) -> None:




        try:
            self.qgis_bridge_delete_btn.setVisible(bool(visible))
        except (RuntimeError, AttributeError):

            pass

    def set_qgis_bridge_points_visible(self, visible: bool) -> None:


        try:
            self.qgis_bridge_points_row.setVisible(bool(visible))
        except (RuntimeError, AttributeError):
            pass

    def reset_qgis_bridge_points(self) -> None:


        try:
            self.qgis_bridge_points_spin.blockSignals(True)
            self.qgis_bridge_points_spin.setValue(100)
            self.qgis_bridge_points_spin.blockSignals(False)
        except (RuntimeError, AttributeError):
            pass

    def _set_bridge_widget_visible(self, name: str, visible: bool) -> None:
        widget = getattr(self, name, None)
        if widget is None:
            return
        try:
            widget.setVisible(visible)
        except (RuntimeError, AttributeError):

            pass

    def _set_bridge_merge_enabled(self, enabled: bool) -> None:












        btn = getattr(self, "auto_shape_merge_btn", None)
        if btn is None:
            return
        try:
            if enabled and bool(getattr(self, "_auto_review_installing", False)):
                enabled = False
            btn.setEnabled(bool(enabled))
        except (RuntimeError, AttributeError):

            pass
