"""Automatic review, "Correct" page construction (selection-first).

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py); this
page slots into the review card's step stack built by auto_review_build.
Dumb widgets + signal emits only: the map tools, selection and edit journal
live plugin-side; auto_state.py holds the store-only setters that drive these
widgets.

One model: a click selects a polygon and opens its panel. The AI | Manual
switch at the top changes only HOW you fix it (on-device AI points vs QGIS
vertices); the panel, the per-polygon settings, Merge and Delete stay the same
around it. Add a missed object rides the same switch (AI outline or hand
draw). Nothing is ever frozen: every edit becomes the polygon's new base and
the Shapes step keeps driving all polygons alike.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QSettings, Qt
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from ...core.review_defaults import (
    AUTO_REVIEW_POINTS_PCT_DEFAULT as _AUTO_REVIEW_POINTS_PCT_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_SIMPLIFY_DEFAULT as _AUTO_REVIEW_SIMPLIFY_DEFAULT,
)
from ...core.server_dials import dial_copy
from .correct_gesture_art import CorrectGestureArt
from .correct_method_default import correct_ai_method_enabled, correct_default_method
from .guidance import BLUE_TINT, HINT_REVIEW_RIGHT_CLICK_DELETE, DismissibleHint
from .styles import (
    _BTN_GHOST,
    _BTN_GREEN,
    _BTN_LINK_MUTED,
    _BTN_REMOVE_ROW,
    _BTN_TILE,
    _PROGRESS_THIN_QSS,
    _SECTION_TOGGLE_QSS,
    _SUBCARD_MARGINS,
    _SUBCARD_QSS,
    _card_divider,
    _choice_divider,
    _msg_card_qss,
    _msg_label_qss,
    _msg_text,
)
from .widgets import _MethodSwitch

# The step heading sits where the card's static "Review detections" title used
# to, so the head of the card names the step the user is ON. Title weight, with
# the note beside it rather than under it: the card head costs one line total.
_REVIEW_HEADING_QSS = (
    "font-size: 13px; font-weight: bold; color: palette(text);"
    " background: transparent; border: none;")
_REVIEW_HEADING_NOTE_QSS = (
    "font-size: 11px; color: rgba(128,128,128,0.95);"
    " background: transparent; border: none;")

_MUTED_LINE_QSS = (
    "font-size: 11px; color: rgba(128,128,128,0.95);"
    " background: transparent; border: none;")


class _CorrectMethodSwitch(_MethodSwitch):
    """The AI | Manual switch with its AI half under the server kill switch.

    ``set_correct_method("ai")`` is called from places that never hear about
    the switch (every review reset opens on it), so refusing the method at the
    widget is the one point that covers them all. A refused set lands on Manual
    and is ANNOUNCED down the same channel a click uses, so the dock's mirror
    of the method and the plugin's both follow it. With the feature on, this
    class changes nothing.
    """

    def set_method(self, method: str) -> None:
        enabled = correct_ai_method_enabled()
        # One method left is not a choice, so the switch goes with the AI half.
        # Re-checked here rather than at build time only: the dock is built
        # before the first configuration fetch lands.
        self.setVisible(enabled)
        if enabled or method == "manual":
            super().set_method(method)
            return
        super().set_method("manual")
        if getattr(self, "_announcing_refusal", False):
            return
        self._announcing_refusal = True
        try:
            self.method_selected.emit("manual")
        finally:
            self._announcing_refusal = False


def _muted_line(text: str = "") -> QLabel:
    """A quiet 11px line: measured facts, key hints, panel notes."""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(_MUTED_LINE_QSS)
    return lbl


def _review_step_heading(title: str, note: str = "") -> QWidget:
    """The one way to head a step page: the step's title, with an optional
    muted note on the same row.

    It replaced a floating grey question that named no owner and a static card
    title that repeated what the step dials already said. ``title_label`` and
    ``note_label`` are exposed for call sites that retitle a step later."""
    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    lbl = QLabel(title)
    lbl.setStyleSheet(_REVIEW_HEADING_QSS)
    row.addWidget(lbl)
    w.title_label = lbl
    if note:
        note_lbl = QLabel(note)
        note_lbl.setWordWrap(True)
        note_lbl.setStyleSheet(_REVIEW_HEADING_NOTE_QSS)
        row.addWidget(note_lbl, 1)
        w.note_label = note_lbl
    row.addStretch(0 if note else 1)
    return w


def _action_tile(glyph: str, label: str, tooltip: str) -> QPushButton:
    """A full-width tile action (Merge in the panel, the Add lane button).

    Nothing here is filled: blue = armed/open per the taxonomy, so the tile
    reads as a choice, not a call to action. The glyph carries the meaning and
    the detail lives in the tooltip.
    """
    btn = QPushButton(f"{glyph}  {label}")
    btn.setStyleSheet(_BTN_TILE)
    btn.setMinimumHeight(38)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setToolTip(tooltip)
    return btn


class DockAutoCorrectBuildMixin:
    """Builds the review card's Correct page (step 1 of the linear review:
    Keep -> Correct -> Shapes), selection-first, one panel per polygon."""

    # ------------------------------------------------------------------
    # The two branch cards
    # ------------------------------------------------------------------
    # A resting Correct page does exactly two things: edit a polygon that
    # exists, or add one that is missing. They used to share a centred hero
    # that named only the first, with the second hanging under it as an
    # unexplained offer. One card each, built from the same two helpers, so
    # they read as the two halves of one choice rather than two unrelated
    # boxes.

    def _branch_card(self, object_name: str) -> QWidget:
        """An empty branch card: the standard sub-card frame and spacing."""
        card = QWidget()
        card.setObjectName(object_name)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(
            _SUBCARD_QSS.format(name=object_name) + "QLabel { background: transparent; border: none; }")
        col = QVBoxLayout(card)
        col.setContentsMargins(*_SUBCARD_MARGINS)
        col.setSpacing(6)
        return card

    def _branch_head(self, glyph: str, title: str):
        """A branch card's head: the brand-blue glyph, then the branch name.

        Returns ``(row, glyph_label, title_label)``; the two labels are handed
        back because the glyph follows the fix method (the one visual that
        separates AI from Manual at rest) and the title is retitled on the
        zero-detection entry.
        """
        from ..canvas_palette import CHROME_BLUE

        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        rgb = f"{CHROME_BLUE.red()},{CHROME_BLUE.green()},{CHROME_BLUE.blue()}"
        glyph_lbl = QLabel(glyph)
        glyph_lbl.setStyleSheet(
            f"font-size: 17px; color: rgb({rgb});"
            " background: transparent; border: none;")
        title_lbl = QLabel(title)
        title_lbl.setWordWrap(True)
        title_lbl.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: palette(text);"
            " background: transparent; border: none;")
        lay.addWidget(glyph_lbl)
        lay.addWidget(title_lbl, 1)
        return row, glyph_lbl, title_lbl

    def _build_auto_correct_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._build_correct_normal_block(lay)
        lay.addStretch(1)
        return page

    # ------------------------------------------------------------------
    # The Correct column: method switch, resting hero, the one panel, the
    # add lane and the quiet Delete row.
    # ------------------------------------------------------------------

    def _build_correct_normal_block(self, parent_lay) -> None:
        self.auto_correct_normal = QWidget()
        lay = QVBoxLayout(self.auto_correct_normal)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # AI | Manual: the switch swaps only the fix method, never the panel.
        # Which half it opens on is served, and the AI half has a kill switch.
        method = correct_default_method()
        self.auto_correct_method_switch = _CorrectMethodSwitch(current=method)
        self._correct_method = method
        self.auto_correct_method_switch.method_selected.connect(
            self._on_correct_method_toggled)
        self.auto_correct_method_switch.setVisible(correct_ai_method_enabled())
        lay.addWidget(self.auto_correct_method_switch)

        # Background local-AI install banner (first AI use). Step-level so it
        # shows whether or not a polygon is picked.
        self._build_reshape_install_banner(lay)

        # Zero-detection count line: an empty run lands here directly. Kept as
        # the dock's own line, no new count plumbing.
        #
        # It says what happened and the two things that can be done about it,
        # both on this screen: the Add card right below, and Exit at the foot
        # of the review. It used to name Confidence, a control that lives on a
        # step this entry never opens, and to stop there.
        #
        # Served copy: this is what a user reads after paying for a run that
        # gave them nothing, so the advice has to be fixable the same day.
        self.auto_correct_zero_line = QLabel(_msg_text("neutral", dial_copy(
            "correct.zero_detection",
            tr("This run found nothing. Add the object yourself below, or press "
               "Exit and run again with another word or a smaller zone."))))
        self.auto_correct_zero_line.setWordWrap(True)
        self.auto_correct_zero_line.setStyleSheet(_msg_label_qss("neutral"))
        self.auto_correct_zero_line.setVisible(False)
        lay.addWidget(self.auto_correct_zero_line)

        # The Manual fix session (dock/qgis_bridge.py), hidden until a session
        # opens. It lives HERE, right under the switch that chose it, and not
        # in place of the whole review: the switch picks the fix method, so
        # only what is below it may change.
        self._setup_qgis_bridge_banner(lay)

        # Branch 1 of 2: edit a polygon that is already there. The resting step
        # offers exactly two things, one card each, so the fork is on screen
        # instead of implied. Keeps the `pick_hero` names: every visibility
        # gate, the bridge's resting-widget list and the zero-detection entry
        # already address this widget, and the branch it carries did not change.
        self.auto_correct_pick_hero = self._branch_card("autoCorrectEditCard")
        _hero = self.auto_correct_pick_hero.layout()
        _head, self.auto_correct_pick_glyph, self.auto_correct_pick_title = (
            self._branch_head("◎", tr("Edit an existing polygon")))
        _hero.addWidget(_head)
        # The gesture plate: the drawn difference between the two methods.
        # AI and Manual otherwise share every widget on this page, so the
        # switch read as if it changed nothing. The plate shows the same
        # outline marked the way each method really marks it on the canvas.
        self.auto_correct_gesture_art = CorrectGestureArt(self._correct_method)
        _hero.addWidget(self.auto_correct_gesture_art)
        # The method line: what THIS method is waiting for, in its own framed
        # box. It says in words what the plate above it shows.
        self.auto_correct_pick_hint = QLabel("")
        self.auto_correct_pick_hint.setWordWrap(True)
        self.auto_correct_pick_hint.setStyleSheet(_msg_label_qss("armed"))
        _hero.addWidget(self.auto_correct_pick_hint)
        lay.addWidget(self.auto_correct_pick_hero)

        # The word between the two branches. They are alternatives, not a list,
        # and two identical cards stacked read as steps to do in order.
        self.auto_correct_or_row = _choice_divider(tr("or"))
        self.auto_correct_or_row.setVisible(False)
        lay.addWidget(self.auto_correct_or_row)

        self._build_correct_select_card(lay)
        self._build_add_lane(lay)

        # The one gesture the step has no button for. Under both branch cards,
        # because it works on any polygon on the map whichever method is live,
        # and dismissible like every other tip.
        self.auto_correct_delete_tip = DismissibleHint(
            HINT_REVIEW_RIGHT_CLICK_DELETE,
            tr("Right-click a polygon on the map to delete it."),
            tint=BLUE_TINT,
            show_glyph=True,
            visibility_gate=self._correct_info_line_gate,
        )
        self.auto_correct_delete_tip.setVisible(False)
        lay.addWidget(self.auto_correct_delete_tip)

        # Per-edit status: an in-progress state (armed / neutral / warning),
        # with an optional secondary action link (set_correct_status). The
        # persistent summary row below reports outcomes and carries Undo.
        self.auto_correct_status = QLabel("")
        self.auto_correct_status.setWordWrap(True)
        self.auto_correct_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.auto_correct_status.linkActivated.connect(
            self._on_correct_status_link)
        self.auto_correct_status.setVisible(False)
        lay.addWidget(self.auto_correct_status)

        self._build_correct_summary_row(lay)

        # Debug-only tile grid toggle (developer aid): hidden unless the
        # QSettings flag is on.
        self._auto_tiles_debug_row = QWidget()
        _tiles_row = QHBoxLayout(self._auto_tiles_debug_row)
        _tiles_row.setContentsMargins(0, 0, 0, 0)
        _tiles_lbl = QLabel(tr("Show tiles (debug)"))
        _tiles_lbl.setStyleSheet("font-size: 11px;")
        self.auto_show_tiles_check = QCheckBox()
        self.auto_show_tiles_check.setChecked(False)
        self.auto_show_tiles_check.stateChanged.connect(
            lambda s: self.auto_show_tiles_changed.emit(bool(s)))
        _tiles_row.addWidget(_tiles_lbl)
        _tiles_row.addStretch()
        _tiles_row.addWidget(self.auto_show_tiles_check)
        lay.addWidget(self._auto_tiles_debug_row)
        self._auto_tiles_debug_row.setVisible(
            QSettings().value("TerraLab/auto_debug_tiles", False, type=bool))

        parent_lay.addWidget(self.auto_correct_normal)

    def _build_correct_select_card(self, lay) -> None:
        """The one panel for the selected polygon: title + measures, the armed
        method line, the Done/Undo session row, the per-polygon settings and
        Merge. The switch above decides whether the armed line and gestures are
        AI points or QGIS vertices; everything else is identical in both."""
        self.auto_correct_select_card = QWidget()
        self.auto_correct_select_card.setObjectName("autoCorrectSelectCard")
        self.auto_correct_select_card.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_correct_select_card.setStyleSheet(
            _SUBCARD_QSS.format(name="autoCorrectSelectCard") + "QLabel { background: transparent; border: none; }")
        _col = QVBoxLayout(self.auto_correct_select_card)
        _col.setContentsMargins(*_SUBCARD_MARGINS)
        _col.setSpacing(8)

        # Title row: the run's class names the polygon; the right side carries
        # the two numbers that decide what to do next (ground area, point
        # count, the number Simplify moves).
        _title_row = QHBoxLayout()
        _title_row.setContentsMargins(0, 0, 0, 0)
        _title_row.setSpacing(6)
        self.auto_correct_selected_label = QLabel(tr("This polygon"))
        self.auto_correct_selected_label.setStyleSheet(
            "font-weight: bold; font-size: 13px;")
        self.auto_correct_selected_info = _muted_line("")
        self.auto_correct_selected_info.setWordWrap(False)
        self.auto_correct_selected_info.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        _title_row.addWidget(self.auto_correct_selected_label)
        _title_row.addStretch(1)
        _title_row.addWidget(self.auto_correct_selected_info)
        _col.addLayout(_title_row)

        # Armed line: what the active method is waiting for. The plugin sets
        # the text (set_correct_armed_line); empty hides it.
        self.auto_correct_armed_line = QLabel("")
        self.auto_correct_armed_line.setWordWrap(True)
        self.auto_correct_armed_line.setStyleSheet(_msg_label_qss("armed"))
        self.auto_correct_armed_line.setVisible(False)
        _col.addWidget(self.auto_correct_armed_line)

        # Save + Undo, shown only while a fix session runs
        # (set_correct_session_active). Save is the session's one filled
        # primary: the green step Next hides for the duration
        # (_apply_step_next_visibility), so the way out of the edit is always
        # the most visible button on screen.
        self.auto_correct_session_row = QWidget()
        _sess = QHBoxLayout(self.auto_correct_session_row)
        _sess.setContentsMargins(0, 0, 0, 0)
        _sess.setSpacing(6)
        self.auto_reshape_done_btn = QPushButton("✓  " + tr("Save"))
        self.auto_reshape_done_btn.setStyleSheet(_BTN_GREEN)
        self.auto_reshape_done_btn.setMinimumHeight(36)
        self.auto_reshape_done_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_reshape_done_btn.setToolTip(tr(
            "Keep this shape. The polygon stays picked, so you can still "
            "adjust, merge or delete it."))
        self.auto_reshape_done_btn.clicked.connect(
            self.auto_reshape_done_requested.emit)
        self.auto_correct_session_undo_btn = QPushButton(tr("Undo"))
        self.auto_correct_session_undo_btn.setStyleSheet(_BTN_GHOST)
        self.auto_correct_session_undo_btn.setMinimumHeight(32)
        self.auto_correct_session_undo_btn.setCursor(
            Qt.CursorShape.PointingHandCursor)
        self.auto_correct_session_undo_btn.clicked.connect(
            self.auto_correction_undo_requested.emit)
        _sess.addWidget(self.auto_reshape_done_btn, 1)
        _sess.addWidget(self.auto_correct_session_undo_btn, 1)
        self.auto_correct_session_row.setVisible(False)
        _col.addWidget(self.auto_correct_session_row)

        # The per-polygon half of the panel: the shape dials and Merge. It used
        # to be the RESTED half, hidden for the whole of a fix session. Since a
        # map click opens the session at once, that made it unreachable, so it
        # now stays up throughout and only Merge waits for the session to end
        # (set_correct_session_active).
        self.auto_correct_rest_box = QWidget()
        _rest = QVBoxLayout(self.auto_correct_rest_box)
        _rest.setContentsMargins(0, 0, 0, 0)
        _rest.setSpacing(8)
        _rest.addWidget(_card_divider())

        # This polygon's own shape dials, folded behind a head that names the
        # job in the current method (_apply_shape_only_mode).
        self._build_shape_only_controls(_rest)

        # Merge acts on the selected polygon: it is the FIRST piece, the user
        # then clicks the others. Hidden unless a neighbour touches the
        # selection (set_merge_available).
        self.auto_shape_merge_btn = _action_tile(
            "⧉", tr("Merge with neighbours"), tr(
                "One object came back split into several polygons. Click the "
                "others on the map, then confirm to merge them into one."))
        self.auto_shape_merge_btn.clicked.connect(
            lambda: self.auto_shape_edit_requested.emit("merge"))
        _rest.addWidget(self.auto_shape_merge_btn)
        _col.addWidget(self.auto_correct_rest_box)

        # Delete lives IN the panel: it acts on the selected polygon, so it
        # exists exactly when the panel does.
        self._build_correct_remove_row(_col)

        self.auto_correct_select_card.setVisible(False)
        lay.addWidget(self.auto_correct_select_card)

    def _shape_only_spin_row(self, col, label: str, tooltip: str, widget,
                             suffix: str) -> None:
        """One labelled dial in the per-polygon settings: name left, value
        right, the tooltip on both. The Shapes step lays its shared twin out the
        same way, so a user who has seen one can read the other."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setStyleSheet("font-size: 11px;")
        lbl.setToolTip(tooltip)
        widget.setSuffix(suffix)
        widget.setMinimumWidth(62)
        widget.setMaximumWidth(78)
        widget.setToolTip(tooltip)
        row.addWidget(lbl)
        row.addStretch(1)
        row.addWidget(widget)
        col.addLayout(row)

    def _shape_only_check_row(self, col, label: str, tooltip: str, widget,
                              trailing=None):
        """One labelled switch in the per-polygon settings, with an optional
        trailing link (the Reset-to-shared escape hatch rides the last row).

        Returns the row's label, which a caller needs only when it has to grey
        the text with the switch (Right angles, when its geometry library is
        missing)."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        widget.setToolTip(tooltip)
        lbl = QLabel(label)
        lbl.setStyleSheet("font-size: 11px;")
        lbl.setToolTip(tooltip)
        row.addWidget(widget)
        row.addWidget(lbl)
        row.addStretch(1)
        if trailing is not None:
            row.addWidget(trailing)
        col.addLayout(row)
        return lbl

    def _build_shape_only_controls(self, col) -> None:
        """This polygon's own copy of the Shapes step, folded behind one head.

        The Shapes step settles the whole set, which is right for a run of
        buildings and wrong for the one odd roof in it. Every shared control has
        a twin here, so the exception can be made where the user is already
        looking at the polygon instead of by moving a dial that wrecks the other
        thousand. Folded shut by default: it is the rare act on this step, and
        open by default it pushed Merge and Delete off a short dock.

        Both methods carry it, and each opens it differently: see
        _apply_shape_only_mode. "Reset to shared" drops the override.
        """
        self.auto_shape_only_toggle = QPushButton()
        self.auto_shape_only_toggle.setStyleSheet(_SECTION_TOGGLE_QSS)
        self.auto_shape_only_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        # Never steal focus from the dials it opens.
        self.auto_shape_only_toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.auto_shape_only_toggle.clicked.connect(
            self._on_shape_only_toggle_clicked)
        col.addWidget(self.auto_shape_only_toggle)

        self.auto_shape_only_box = QWidget()
        _box = QVBoxLayout(self.auto_shape_only_box)
        _box.setContentsMargins(0, 2, 0, 2)
        _box.setSpacing(6)

        # What the fold is FOR, in one line. Without it the dials read as a
        # second copy of the Shapes step and the user cannot tell which one
        # wins. The wording follows the method (_apply_shape_only_mode).
        self.auto_shape_only_scope_line = _muted_line("")
        _box.addWidget(self.auto_shape_only_scope_line)

        _points_tip = tr(
            "How many of this polygon's points to keep. The count in the title "
            "row follows it. It runs before Right angles, so lowering it gives "
            "the squaring straight walls instead of a staircase.")
        self.auto_shape_only_points = QSpinBox()
        self.auto_shape_only_points.setSingleStep(5)
        # Down to 1% (like the shared control): with the dial's own low floor
        # and the deviation cap it relaxes (core.vertex_budget), a low % now
        # visibly simplifies instead of plateauing at the class density.
        self.auto_shape_only_points.setRange(1, 100)
        self.auto_shape_only_points.setValue(_AUTO_REVIEW_POINTS_PCT_DEFAULT)
        self._shape_only_spin_row(
            _box, tr("Points"), _points_tip, self.auto_shape_only_points, " %")

        # Simplify (px) for this polygon only: the Douglas-Peucker twin of the
        # shared control, kept so the two reducers can be compared on one shape.
        self.auto_shape_only_simplify = QDoubleSpinBox()
        self.auto_shape_only_simplify.setDecimals(1)
        self.auto_shape_only_simplify.setSingleStep(0.5)
        self.auto_shape_only_simplify.setRange(0.0, 1000.0)
        self.auto_shape_only_simplify.setValue(_AUTO_REVIEW_SIMPLIFY_DEFAULT)
        self._shape_only_spin_row(
            _box, tr("Simplify"), tr(
                "Drop this polygon's points closer than this distance to a "
                "straight edge (0 = off). A distance, not a count; Points is "
                "usually the better dial."),
            self.auto_shape_only_simplify, " px")

        self.auto_shape_only_clean = QDoubleSpinBox()
        self.auto_shape_only_clean.setDecimals(1)
        self.auto_shape_only_clean.setSingleStep(0.5)
        self.auto_shape_only_clean.setRange(0.0, 50.0)
        self._shape_only_spin_row(
            _box, tr("Trim spikes"), tr(
                "Cut thin spurs off this polygon (0 = off). Raise it on a "
                "single ragged outline instead of eroding the whole layer."),
            self.auto_shape_only_clean, " px")

        self.auto_shape_only_expand = QSpinBox()
        self.auto_shape_only_expand.setRange(-1000, 1000)
        self._shape_only_spin_row(
            _box, tr("Grow / shrink"), tr(
                "Push this polygon's edge out (positive) or in (negative), "
                "for the one footprint the model cut short or overran."),
            self.auto_shape_only_expand, " px")

        self.auto_shape_only_smooth = QCheckBox()
        self._shape_only_check_row(
            _box, tr("Round corners"), tr(
                "Round this polygon's corners, for a tree or a pond among "
                "squared neighbours."),
            self.auto_shape_only_smooth)

        self.auto_shape_only_fill = QCheckBox()
        self._shape_only_check_row(
            _box, tr("Fill holes"), tr(
                "Close the gaps inside this polygon, without filling the "
                "courtyards the rest of the layer is meant to keep."),
            self.auto_shape_only_fill)

        # Right angles + the Reset-to-shared link share the last row (the
        # switch on the left, the escape hatch on the right).
        self.auto_shape_only_ortho = QCheckBox()
        self.auto_shape_only_reset = QPushButton(tr("Reset to shared"))
        self.auto_shape_only_reset.setStyleSheet(_BTN_LINK_MUTED)
        self.auto_shape_only_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_shape_only_reset.setVisible(False)
        self.auto_shape_only_reset.clicked.connect(
            self.auto_shape_only_reset_requested.emit)
        self.auto_shape_only_ortho_label = self._shape_only_check_row(
            _box, tr("Right angles"), tr(
                "Square this polygon's edges, or leave them as traced while "
                "the rest of the layer stays squared."),
            self.auto_shape_only_ortho, trailing=self.auto_shape_only_reset)

        col.addWidget(self.auto_shape_only_box)

        self._auto_shape_only_expanded = False
        self._apply_shape_only_mode(self._correct_method)

        for widget, control in (
            (self.auto_shape_only_points, "shape_only_points"),
            (self.auto_shape_only_simplify, "shape_only_simplify"),
            (self.auto_shape_only_clean, "shape_only_trim_spikes"),
            (self.auto_shape_only_expand, "shape_only_grow_shrink"),
        ):
            widget.valueChanged.connect(
                lambda _v, c=control: self._emit_shape_only_changed(c, _v))
        for widget, control in (
            (self.auto_shape_only_smooth, "shape_only_round_corners"),
            (self.auto_shape_only_fill, "shape_only_fill_holes"),
            (self.auto_shape_only_ortho, "shape_only_right_angles"),
        ):
            widget.stateChanged.connect(
                lambda s, c=control: self._emit_shape_only_changed(c, s))
        # This polygon's own Right angles refuses on the same terms as the
        # shared one: ticking it is what tests the geometry library behind it.
        self.auto_shape_only_ortho.stateChanged.connect(
            lambda _s: self._sync_shape_only_right_angles())

    def _build_add_lane(self, lay) -> None:
        """Branch 2 of 2: add a polygon the run missed. Same card shape as the
        edit branch, so the two read as one choice. The tile's label and target
        follow the AI | Manual switch: AI points the on-device model at the
        object (auto_ai_add_requested), Manual hand-draws it corner by corner
        (auto_add_polygon_requested). Visible only when nothing is selected
        (and in the zero-detection entry)."""
        self.auto_add_lane_card = self._branch_card("autoAddLaneCard")
        _col = self.auto_add_lane_card.layout()
        _head, _, self.auto_add_lane_title = self._branch_head(
            "＋", tr("Add a missing polygon"))
        _col.addWidget(_head)
        # What the method costs and where it runs, one muted line. It is the
        # second thing that differs between AI and Manual at rest, and the one
        # that answers "why would I pick this one".
        self.auto_add_lane_method_line = _muted_line("")
        _col.addWidget(self.auto_add_lane_method_line)

        # Armed line above the button, shown while the add tool is live
        # (set_add_lane_armed).
        self.auto_add_lane_line = QLabel("")
        self.auto_add_lane_line.setWordWrap(True)
        self.auto_add_lane_line.setStyleSheet(_msg_label_qss("armed"))
        self.auto_add_lane_line.setVisible(False)
        _col.addWidget(self.auto_add_lane_line)

        # Keep, the way to commit the outline on screen without leaving the
        # lane. The per-polygon panel (which carries the session Save) is
        # hidden while nothing is selected, so without this button an AI
        # outline could only be kept from the keyboard. Shown exactly while
        # there is something to keep (set_add_lane_keep_available).
        self.auto_add_lane_keep_btn = QPushButton("✓  " + tr("Keep this one"))
        self.auto_add_lane_keep_btn.setStyleSheet(_BTN_GREEN)
        self.auto_add_lane_keep_btn.setMinimumHeight(36)
        self.auto_add_lane_keep_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_add_lane_keep_btn.setToolTip(tr(
            "Keep this outline and point at the next object. Shortcut: S"))
        self.auto_add_lane_keep_btn.clicked.connect(
            self.auto_ai_add_keep_requested.emit)
        self.auto_add_lane_keep_btn.setVisible(False)
        _col.addWidget(self.auto_add_lane_keep_btn)

        self.auto_add_lane_btn = _action_tile(
            "＋", tr("Point at it on the map"), tr(
                "Add an object the AI missed. In AI, point at it and the "
                "model outlines it, free; in Manual, draw its corners."))
        self.auto_add_lane_btn.clicked.connect(self._on_add_lane_clicked)
        _col.addWidget(self.auto_add_lane_btn)

        self.auto_add_lane_card.setVisible(False)
        lay.addWidget(self.auto_add_lane_card)

    def _build_correct_remove_row(self, lay) -> None:
        """Delete, inside the panel: one quiet text button, shown exactly when
        a polygon is selected (it follows the card). Neutral grey at rest,
        warms red under the pointer; Undo brings the polygon back. No key hint
        on screen: the Delete key works and lives in the tooltip."""
        self.auto_correct_remove_row = QWidget()
        _row = QHBoxLayout(self.auto_correct_remove_row)
        _row.setContentsMargins(0, 0, 0, 0)
        _row.setSpacing(6)
        self.auto_remove_btn = QPushButton("✕  " + tr("Delete this polygon"))
        self.auto_remove_btn.setStyleSheet(_BTN_REMOVE_ROW)
        self.auto_remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_remove_btn.setToolTip(tr(
            "Delete this polygon (the Delete key works too, and a right-click "
            "on the map deletes the shape under the cursor). "
            "Undo brings it back."))
        self.auto_remove_btn.clicked.connect(self.auto_remove_requested.emit)
        _row.addWidget(self.auto_remove_btn, 1)
        lay.addWidget(self.auto_correct_remove_row)

    def _build_correct_summary_row(self, lay) -> None:
        """Persistent journal summary: "N corrections this round · Undo last ·
        Clear all". Hidden while the journal is empty (set_correction_summary)."""
        self.auto_correct_summary_row = QWidget()
        _sum_row = QHBoxLayout(self.auto_correct_summary_row)
        _sum_row.setContentsMargins(0, 0, 0, 0)
        _sum_row.setSpacing(2)
        self.auto_correct_summary_label = QLabel("")
        self.auto_correct_summary_label.setStyleSheet(
            "font-size: 11px; color: rgba(128,128,128,0.95);"
            " background: transparent; border: none;")
        self.auto_correct_undo_btn = QPushButton(tr("Undo last"))
        self.auto_correct_undo_btn.setStyleSheet(_BTN_LINK_MUTED)
        self.auto_correct_undo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_correct_undo_btn.clicked.connect(
            self.auto_correction_undo_requested.emit)
        self.auto_correct_clear_btn = QPushButton(tr("Clear all"))
        self.auto_correct_clear_btn.setStyleSheet(_BTN_LINK_MUTED)
        self.auto_correct_clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_correct_clear_btn.setToolTip(tr(
            "Undo every correction of this round at once. The count is in the "
            "label, so you can see what goes."))
        self.auto_correct_clear_btn.clicked.connect(
            self.auto_correction_clear_requested.emit)
        _sum_row.addWidget(self.auto_correct_summary_label)
        for _w in (self.auto_correct_undo_btn, self.auto_correct_clear_btn):
            _dot = QLabel("·")
            _dot.setStyleSheet(
                "font-size: 11px; color: rgba(128,128,128,0.9);"
                " background: transparent; border: none;")
            _sum_row.addWidget(_dot)
            _sum_row.addWidget(_w)
        _sum_row.addStretch(1)
        self.auto_correct_summary_row.setVisible(False)
        lay.addWidget(self.auto_correct_summary_row)

    def _build_reshape_install_banner(self, lay) -> None:
        """The setup banner for the local AI, shown while a first-time install
        runs (set_auto_review_installing).

        It is the only live thing on the review while that install runs: the
        setter locks the ladder, the method switch and the way out around it,
        so the banner has to carry both the progress and the way to stop. Its
        Cancel is that way out; without it a locked review would have none."""
        self._auto_review_installing = False
        self.auto_review_install_banner = QWidget()
        self.auto_review_install_banner.setObjectName("autoReviewInstallBanner")
        self.auto_review_install_banner.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_review_install_banner.setStyleSheet(
            _msg_card_qss("autoReviewInstallBanner", "info"))
        _col = QVBoxLayout(self.auto_review_install_banner)
        _col.setContentsMargins(10, 8, 10, 8)
        _col.setSpacing(4)
        self.auto_review_install_label = QLabel(
            tr("Setting up the on-device AI..."))
        self.auto_review_install_label.setWordWrap(True)
        self.auto_review_install_label.setStyleSheet(
            "font-size: 11px; color: palette(text);")
        _col.addWidget(self.auto_review_install_label)
        self.auto_review_install_progress = QProgressBar()
        self.auto_review_install_progress.setRange(0, 100)
        self.auto_review_install_progress.setValue(0)
        self.auto_review_install_progress.setTextVisible(False)
        self.auto_review_install_progress.setStyleSheet(_PROGRESS_THIN_QSS)
        _col.addWidget(self.auto_review_install_progress)
        _cancel_row = QHBoxLayout()
        _cancel_row.setContentsMargins(0, 0, 0, 0)
        _cancel_row.addStretch(1)
        self.auto_review_install_cancel_btn = QPushButton(tr("Cancel setup"))
        self.auto_review_install_cancel_btn.setStyleSheet(_BTN_LINK_MUTED)
        self.auto_review_install_cancel_btn.setToolTip(tr(
            "Stop the setup and go back to the review. The AI fix stays "
            "unavailable until you install it."))
        self.auto_review_install_cancel_btn.clicked.connect(
            self.auto_review_install_cancel_requested.emit)
        _cancel_row.addWidget(self.auto_review_install_cancel_btn)
        _col.addLayout(_cancel_row)
        self.auto_review_install_banner.setVisible(False)
        lay.addWidget(self.auto_review_install_banner)
