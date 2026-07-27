"""The Manual fix session: the same panel shape as the AI one.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py). The
plugin-side bridge selects the object being edited, arms editing and snapping,
and drives the native tool. This file owns the panel the user reads while that
runs, and it is deliberately the twin of the AI session panel built in
auto_correct_build: one card, a title, one instruction line, one filled
primary and a quiet row under it.

The session sits INSIDE the Correct step, under the AI | Manual switch, so
picking the other method never swaps the whole screen. While it is up, the
resting hero, the per-polygon panel and the add lane step aside.

One line carries the words: what the armed tool is for, then what the gesture
in progress is waiting for, then what QGIS recorded. Those are three states of
the same question ("what now?"), never three labels at once. They have to be
spelled out because they are native QGIS gestures a non-GIS user cannot guess
(a capture line ends on a right-click, a reshape line has to cross the outline
twice, a split has to cross it completely) and QGIS says none of it outside
the status bar, which users miss.
"""
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
from .styles import (
    _BTN_BLUE_OUTLINE,
    _BTN_GHOST,
    _BTN_GREEN,
    _BTN_REMOVE_ROW,
    _SUBCARD_MARGINS,
    _SUBCARD_QSS,
    _btn_toggle_qss,
    _msg_label_qss,
    _msg_text,
)

_MANUAL_TOOL_RGB = (30, 136, 229)

# What the Correct step shows at rest and the session replaces. One thing per
# state: while a hand edit runs, the panel is the step.
_RESTING_CORRECT_WIDGETS = (
    "auto_correct_pick_hero",
    "auto_correct_method_info_hint",
    "auto_correct_select_card",
    "auto_add_lane_card",
    "auto_correct_status",
    "auto_correct_summary_row",
)


def _tool_instruction(tool: str) -> str:
    """The one line an armed tool needs: its gesture, in plain words.

    No keys and no mouse buttons here. Every gesture that used to be spelled
    out as a shortcut is a button in the panel now, and the shortcut lives on
    that button's tooltip."""
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
    """The Manual fix-session panel and its enter/leave state."""

    def _setup_qgis_bridge_banner(self, layout) -> None:
        """Build the Manual session panel (hidden until a session opens).

        Same card, same order and same button treatment as the AI session
        panel: the title, the tool in hand, one instruction line, the filled
        primary that ends the session, and the quiet row under it."""
        self.qgis_bridge_banner = QWidget()
        self.qgis_bridge_banner.setObjectName("qgisBridgeBanner")
        self.qgis_bridge_banner.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.qgis_bridge_banner.setStyleSheet(
            _SUBCARD_QSS.format(name="qgisBridgeBanner") + "QLabel { background: transparent; border: none; }")
        col = QVBoxLayout(self.qgis_bridge_banner)
        col.setContentsMargins(*_SUBCARD_MARGINS)
        col.setSpacing(8)

        # Title: the run's class names the polygon, the way the AI panel names
        # it. set_qgis_bridge_target fills the class in.
        self.qgis_bridge_title = QLabel(tr("This polygon"))
        self.qgis_bridge_title.setWordWrap(True)
        self.qgis_bridge_title.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: palette(text);")
        col.addWidget(self.qgis_bridge_title)

        # Points: thin the target's outline BEFORE the user drags vertices, so
        # a mask traced pixel by pixel does not fight them with hundreds of
        # points. Same engine as the review's Points dial. Hidden until the
        # plugin resolves a target and revealed only then; it retires the
        # moment the shape is hand-edited (the dial rewrites from the pre-edit
        # snapshot and would wipe those corners).
        self.qgis_bridge_points_row = QWidget()
        _points = QHBoxLayout(self.qgis_bridge_points_row)
        _points.setContentsMargins(0, 0, 0, 0)
        _points.setSpacing(6)
        _points_lbl = QLabel(tr("Points"))
        _points_lbl.setStyleSheet("font-size: 11px;")
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
        self.qgis_bridge_points_spin.setMinimumWidth(62)
        self.qgis_bridge_points_spin.setMaximumWidth(78)
        self.qgis_bridge_points_spin.setToolTip(_points_tip)
        self.qgis_bridge_points_spin.valueChanged.connect(
            lambda v: self.auto_qgis_bridge_points_changed.emit(int(v)))
        _points.addWidget(self.qgis_bridge_points_spin)
        self.qgis_bridge_points_row.setVisible(False)
        col.addWidget(self.qgis_bridge_points_row)

        tools = QHBoxLayout()
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setSpacing(6)
        self._qgis_bridge_tool_buttons = {}
        # Plain names, not GIS jargon. Vertex -> "Move points", Reshape ->
        # "Redraw edge", Split stays (it is already plain). Tooltips add the
        # one-line "what for".
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
                _MANUAL_TOOL_RGB, "palette(text)", "#000000", quiet=True))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(tooltip)
            btn.clicked.connect(
                lambda _checked=False, tool=key:
                self.auto_qgis_bridge_tool_requested.emit(tool))
            self._qgis_bridge_tool_buttons[key] = btn
            tools.addWidget(btn, 1)
        col.addLayout(tools)

        # The one line: the armed tool's gesture, then the live count while a
        # line is being drawn, then what QGIS recorded. Same widget and same
        # taxonomy styling as the AI panel's armed line.
        self.qgis_bridge_line = QLabel("")
        self.qgis_bridge_line.setWordWrap(True)
        self.qgis_bridge_line.setStyleSheet(_msg_label_qss("armed"))
        col.addWidget(self.qgis_bridge_line)

        # Save ends the session, the same green primary the AI session uses.
        # Finish ends the LINE being drawn and takes Save's slot while one is
        # open, so those two are never on screen together. Undo steps back one
        # gesture, or one point while a line is open.
        _actions = QHBoxLayout()
        _actions.setContentsMargins(0, 0, 0, 0)
        _actions.setSpacing(6)
        self.qgis_bridge_finish_btn = QPushButton(tr("Finish the line"))
        self.qgis_bridge_finish_btn.setStyleSheet(_BTN_BLUE_OUTLINE)
        self.qgis_bridge_finish_btn.setMinimumHeight(36)
        self.qgis_bridge_finish_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.qgis_bridge_finish_btn.setToolTip(tr(
            "Close the line you are drawing. A right-click on the map does "
            "the same."))
        self.qgis_bridge_finish_btn.clicked.connect(
            lambda: self.auto_qgis_bridge_gesture_requested.emit("finish"))
        self.qgis_bridge_finish_btn.setVisible(False)
        self.qgis_bridge_done_btn = QPushButton("✓  " + tr("Save"))
        self.qgis_bridge_done_btn.setStyleSheet(_BTN_GREEN)
        self.qgis_bridge_done_btn.setMinimumHeight(36)
        self.qgis_bridge_done_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.qgis_bridge_done_btn.setToolTip(tr(
            "Keep these edits and go back to picking polygons."))
        self.qgis_bridge_done_btn.clicked.connect(
            self.auto_qgis_bridge_done_requested.emit)
        self.qgis_bridge_undo_btn = QPushButton(tr("Undo"))
        self.qgis_bridge_undo_btn.setStyleSheet(_BTN_GHOST)
        self.qgis_bridge_undo_btn.setMinimumHeight(36)
        self.qgis_bridge_undo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.qgis_bridge_undo_btn.setToolTip(tr(
            "Undo the last thing you did here: the point you just placed, or "
            "the last edit."))
        self.qgis_bridge_undo_btn.clicked.connect(
            self.auto_qgis_bridge_undo_requested.emit)
        _actions.addWidget(self.qgis_bridge_finish_btn, 1)
        _actions.addWidget(self.qgis_bridge_done_btn, 1)
        _actions.addWidget(self.qgis_bridge_undo_btn, 1)
        col.addLayout(_actions)

        # Delete a corner: the quiet row the resting panel gives to "Delete
        # this polygon", in the same place with the same treatment. Shown only
        # while the vertex tool holds a picked corner, so it is never inert.
        self.qgis_bridge_delete_corner_btn = QPushButton(
            "✕  " + tr("Delete this corner"))
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

        self.qgis_bridge_banner.setVisible(False)
        layout.addWidget(self.qgis_bridge_banner)

    def enter_qgis_bridge_state(self) -> None:
        """Open the Manual session: the panel takes the resting step's place.

        The AI | Manual switch, the step ladder and the pinned step primary all
        stay put: a session changes the fix method, not the screen, and the way
        to the next step has to stay on screen even mid-edit (leaving Correct
        folds the live session first). Best-effort per widget so a partial
        build never raises."""
        self._qgis_bridge_active_ui = True
        self.set_qgis_bridge_tool("vertex")
        # Points starts at 100% and hidden. The plugin reveals the row once it
        # has resolved which polygon the edit works on (a whole-layer session
        # with no target has nothing to thin).
        self.reset_qgis_bridge_points()
        self.set_qgis_bridge_points_visible(False)
        # Reset the title too: the plugin re-names the target right after, but a
        # session that cannot resolve one must not inherit the previous run's
        # word (a building on a later tree run).
        self.set_qgis_bridge_target("")
        for name in _RESTING_CORRECT_WIDGETS:
            self._set_bridge_widget_visible(name, False)
        try:
            self._set_review_dials_locked(True, 1)
        except (RuntimeError, AttributeError):
            pass
        try:
            self.qgis_bridge_banner.setVisible(True)
        except (RuntimeError, AttributeError):
            pass

    def leave_qgis_bridge_state(self) -> None:
        """Close the Manual session and bring the resting Correct step back.

        Which primary shows on which step is the plugin's call, so
        set_auto_review_step re-drives that; the panels follow the selection
        through _refresh_correct_panels."""
        self._qgis_bridge_active_ui = False
        try:
            self.qgis_bridge_banner.setVisible(False)
        except (RuntimeError, AttributeError):
            pass
        self.set_qgis_bridge_delete_corner_visible(False)
        try:
            self.set_auto_review_step(1)
            # Hero, info line and add lane follow the selection; the journal
            # summary follows the journal. The per-polygon panel is the
            # plugin's call: it re-selects the polygon it came in with.
            self._refresh_correct_panels()
            self._refresh_correct_summary_row()
        except (RuntimeError, AttributeError):
            pass
        # It carries its own content, so it comes back only if it had
        # something to say before the session hid it.
        for name in ("auto_correct_status",):
            widget = getattr(self, name, None)
            if widget is None:
                continue
            try:
                self._set_bridge_widget_visible(name, bool(widget.text()))
            except (RuntimeError, AttributeError):
                pass

    def set_qgis_bridge_tool(self, tool: str) -> None:
        """Show which native geometry tool owns the map and put its gesture on
        the panel's one line."""
        self._qgis_bridge_tool = str(tool)
        for key, btn in getattr(self, "_qgis_bridge_tool_buttons", {}).items():
            try:
                btn.setProperty("armed", key == tool)
                btn.style().unpolish(btn)
                btn.style().polish(btn)
            except (RuntimeError, AttributeError):
                pass
        # Drawing a new object is not editing the picked one, so the title says
        # so instead of naming a polygon that is not under the cursor.
        if tool == "add":
            try:
                self.qgis_bridge_title.setText(tr("New polygon"))
            except (RuntimeError, AttributeError):
                pass
        else:
            self.set_qgis_bridge_target(
                getattr(self, "_qgis_bridge_target_label", ""))
        self.set_qgis_bridge_feedback("")
        # Changing tool abandons any line in progress, so Finish goes back to
        # hiding until the poll counts a point again.
        self.set_qgis_bridge_line_open(False)
        self.set_qgis_bridge_delete_corner_visible(False)

    def set_qgis_bridge_line_open(self, open_: bool) -> None:
        """Swap the primary between Finish (a line is being drawn) and Save.

        They share one slot on purpose: "finish this line" and "save this
        polygon" read as the same act when both are on screen, and that is the
        one confusion this panel has to avoid."""
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

    def set_qgis_bridge_target(self, label: str) -> None:
        """Name the polygon under edit the way the AI panel names it: by the
        run's class. Empty means the session opened with nothing picked and
        the tools reach every polygon, so the title stays neutral."""
        self._qgis_bridge_target_label = str(label or "")
        try:
            text = str(label or "").strip()
            self.qgis_bridge_title.setText(text or tr("This polygon"))
        except (RuntimeError, AttributeError, TypeError):
            pass

    def set_qgis_bridge_last_change(self, text: str, can_undo: bool) -> None:
        """Report the last native edit by name on the panel's one line.

        ``text`` comes from QGIS's own undo stack, so it names the real
        operation ("Moved vertex", "Split features"). Empty text puts the armed
        tool's instruction back, which is the state on entry. Undo is always on
        screen now, so ``can_undo`` no longer drives it."""
        self.set_qgis_bridge_feedback(text, "success" if text else "armed")

    def set_qgis_bridge_feedback(self, text: str, kind: str = "armed") -> None:
        """Write the panel's one line. Empty text restores the armed tool's
        instruction, so the line is never blank while a tool is held."""
        try:
            lbl = self.qgis_bridge_line
        except AttributeError:
            return
        if not text:
            text = _tool_instruction(getattr(self, "_qgis_bridge_tool", ""))
            kind = "armed"
        try:
            lbl.setStyleSheet(_msg_label_qss(kind))
            lbl.setText(_msg_text(kind, text))
            lbl.setVisible(True)
        except (RuntimeError, AttributeError, KeyError):
            pass

    def set_qgis_bridge_delete_corner_visible(self, visible: bool) -> None:
        """Show or hide Delete this corner. Driven by the bridge poll: shown
        only while the vertex tool has a corner picked, so the row is never
        present when clicking it would do nothing."""
        try:
            self.qgis_bridge_delete_corner_btn.setVisible(bool(visible))
        except (RuntimeError, AttributeError):
            pass

    def set_qgis_bridge_points_visible(self, visible: bool) -> None:
        """Show or hide the Points row. The plugin shows it only once a target
        polygon is resolved, and hides it the moment the shape is hand-edited."""
        try:
            self.qgis_bridge_points_row.setVisible(bool(visible))
        except (RuntimeError, AttributeError):
            pass

    def reset_qgis_bridge_points(self) -> None:
        """Put the Points dial back to 100% without firing its signal (a value
        set here is a UI reset, not a user request to thin the polygon)."""
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
