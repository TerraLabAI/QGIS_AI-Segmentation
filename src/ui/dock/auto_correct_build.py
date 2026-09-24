














from __future__ import annotations

from qgis.PyQt.QtCore import QEvent, QSettings, Qt
from qgis.PyQt.QtWidgets import (
    QBoxLayout,
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
from ..icons import icon_for, pixmap_for
from .auto_flow_look import token_qcolor
from .correct_gesture_art import CorrectGestureArt
from .correct_method_default import correct_ai_method_enabled, correct_default_method
from .correct_summary_row import build_correction_summary_row
from .font_scale import fit_spin_width, scale_qss_font_px
from .guidance import BLUE_TINT, HINT_REVIEW_RIGHT_CLICK_DELETE, DismissibleHint
from .styles import (
    _BTN_GHOST,
    _BTN_GREEN,
    _BTN_LINK_MUTED,
    _BTN_LINK_QUIET,
    _BTN_REMOVE_ROW,
    _CARD_MARGINS,
    BRAND_BLUE,
    BRAND_BLUE_HOVER,
    BTN_PRIMARY_WIDE_PX,
    FONT_BODY,
    FONT_HINT,
    HUE_LOCAL,
    INK,
    INK_2,
    LINE,
    ON_ACCENT,
    RADIUS_CARD,
    SPACE_STAGE,
    SURFACE,
    _card_divider,
    _msg_label_qss,
    category_card_qss,
    category_progress_qss,
    msg_rich,
)
from .widgets import _MethodSwitch, label_with_target_hint





_REVIEW_HEADING_QSS = scale_qss_font_px(
    f"font-size: {FONT_BODY}px; font-weight: 600; color: {INK};"
    " background: transparent; border: none;")
_REVIEW_HEADING_NOTE_QSS = scale_qss_font_px(
    f"font-size: {FONT_HINT}px; color: {INK_2};"
    " background: transparent; border: none;")

_BODY_LINE_QSS = scale_qss_font_px(
    f"font-size: {FONT_BODY}px; color: {INK};"
    " background: transparent; border: none;")

_MUTED_LINE_QSS = scale_qss_font_px(
    f"font-size: {FONT_HINT}px; color: {INK_2};"
    " background: transparent; border: none;")


def branch_card_qss(object_name: str) -> str:





    return (f"QWidget#{object_name} {{ background: {SURFACE};"
            f" border: 1px solid {LINE}; border-radius: {RADIUS_CARD}px; }}"
            "QLabel { background: transparent; border: none; }")


class _CorrectMethodSwitch(_MethodSwitch):
























    def __init__(self, current: str = "manual", parent=None):
        super().__init__(current=current, parent=parent)
        self.setStyleSheet(self.styleSheet() + (
            "QPushButton:checked {"
            f"  background: {BRAND_BLUE};"
            f"  border: 1px solid {BRAND_BLUE};"
            "  color: #ffffff;"
            "}"
            f"QPushButton:checked:hover {{ background: {BRAND_BLUE_HOVER};"
            f" border-color: {BRAND_BLUE_HOVER}; }}"


            "QPushButton:checked:focus { border: 1px solid #ffffff; }"
        ))

    def set_method(self, method: str) -> None:
        enabled = correct_ai_method_enabled()



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


class _MethodLineLabel(QLabel):







    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self._alternatives: list[str] = []

    def set_alternatives(self, texts) -> None:
        self._alternatives = [str(t) for t in texts if t]
        self.updateGeometry()

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        try:
            own = super().heightForWidth(width)
            if not self._alternatives or width <= 0:
                return own
            from qgis.PyQt.QtCore import QRect

            margins = self.contentsMargins()
            inner = max(1, width - margins.left() - margins.right())
            metrics = self.fontMetrics()
            flags = int(Qt.TextFlag.TextWordWrap)
            tallest = max(
                metrics.boundingRect(QRect(0, 0, inner, 100000), flags, text).height()
                for text in self._alternatives)
            return max(own, tallest + margins.top() + margins.bottom())
        except Exception:  # noqa: BLE001
            return super().heightForWidth(width)


class _BranchCardsRow(QWidget):












    def __init__(self, spacing: int, parent=None):
        super().__init__(parent)
        self._cards: list[QWidget] = []
        self._box = QBoxLayout(QBoxLayout.Direction.TopToBottom, self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(spacing)

    def add_card(self, card: QWidget) -> None:
        self._cards.append(card)
        self._box.addWidget(card, 1)
        card.installEventFilter(self)
        self._sync_direction()

    def _side_by_side_width(self) -> int:
        shown = [c for c in self._cards if not c.isHidden()]
        if len(shown) < 2:
            return 1 << 30
        return (sum(c.sizeHint().width() for c in shown)
                + self._box.spacing() * (len(shown) - 1))

    def _sync_direction(self) -> None:
        wanted = (QBoxLayout.Direction.LeftToRight
                  if self.width() >= self._side_by_side_width()
                  else QBoxLayout.Direction.TopToBottom)
        if self._box.direction() != wanted:
            self._box.setDirection(wanted)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        try:
            self._sync_direction()
        except RuntimeError:
            pass  # nosec B110

    def eventFilter(self, watched, event):  # noqa: N802
        try:
            if event.type() in (QEvent.Type.Show, QEvent.Type.Hide):
                self._sync_direction()
        except RuntimeError:
            pass  # nosec B110
        return False


def _curly_quotes(text: str) -> str:

    out, opening = [], True
    for char in str(text or ""):
        if char == '"':
            out.append("\u201c" if opening else "\u201d")
            opening = not opening
        else:
            out.append(char)
    return "".join(out)


def _muted_line(text: str = "") -> QLabel:

    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(_MUTED_LINE_QSS)
    return lbl


def set_branch_glyph(label: QLabel, name: str) -> None:

    try:
        label.setPixmap(pixmap_for(label, name, 18, token_qcolor(BRAND_BLUE)))
    except (RuntimeError, AttributeError):
        pass



CORRECT_METHOD_GLYPH = {"ai": "spark", "manual": "polygon"}


EDIT_BRANCH_GLYPH = "pencil"


def _setting_glyph(icon_name: str) -> QLabel:



    glyph = QLabel()
    glyph.setPixmap(pixmap_for(glyph, icon_name, 16, token_qcolor(INK)))
    glyph.setFixedSize(16, 16)
    glyph.setStyleSheet("background: transparent; border: none;")
    return glyph


def _review_step_heading(title: str, note: str = "") -> QWidget:






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






    btn = QPushButton(label)


    btn.setStyleSheet(_BTN_GHOST)
    btn.setIcon(icon_for(btn, glyph, 16))

    btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
    btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setToolTip(tooltip)
    return btn


class DockAutoCorrectBuildMixin:













    def _branch_card(self, object_name: str) -> QWidget:





        card = QWidget()
        card.setObjectName(object_name)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(branch_card_qss(object_name))
        col = QVBoxLayout(card)
        col.setContentsMargins(*_CARD_MARGINS)
        col.setSpacing(8)
        return card

    def _branch_head(self, glyph: str, title: str):







        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        glyph_lbl = QLabel()
        glyph_lbl.setStyleSheet("background: transparent; border: none;")
        set_branch_glyph(glyph_lbl, glyph)
        title_lbl = QLabel(title)
        title_lbl.setWordWrap(True)
        title_lbl.setStyleSheet(_REVIEW_HEADING_QSS)
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






    def _build_correct_normal_block(self, parent_lay) -> None:
        self.auto_correct_normal = QWidget()
        lay = QVBoxLayout(self.auto_correct_normal)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)



        method = correct_default_method()
        self.auto_correct_method_switch = _CorrectMethodSwitch(current=method)
        self._correct_method = method
        self.auto_correct_method_switch.method_selected.connect(
            self._on_correct_method_toggled)
        self.auto_correct_method_switch.setVisible(correct_ai_method_enabled())
        lay.addWidget(self.auto_correct_method_switch)








        self._build_correct_summary_row(lay)



        self._build_reshape_install_banner(lay)















        import html as _html
        _zero_body = _curly_quotes(dial_copy(
            "correct.zero_detection",
            tr("This run found nothing. Add the object yourself below, or use "
               '"Re-run the whole zone" with another word.')))
        self.auto_correct_zero_line = QLabel(msg_rich(
            "info",
            f"<b>{_html.escape(tr('No objects found'))}</b><br>"
            f"{_html.escape(_zero_body)}",
            is_html=True))
        self.auto_correct_zero_line.setTextFormat(Qt.TextFormat.RichText)
        self.auto_correct_zero_line.setWordWrap(True)
        self.auto_correct_zero_line.setStyleSheet(_msg_label_qss("neutral"))
        self.auto_correct_zero_line.setVisible(False)
        lay.addWidget(self.auto_correct_zero_line)





        self._setup_qgis_bridge_banner(lay)














        self.auto_correct_pick_hero = self._branch_card("autoCorrectEditCard")
        _hero = self.auto_correct_pick_hero.layout()
        _head, self.auto_correct_pick_glyph, self.auto_correct_pick_title = (
            self._branch_head(EDIT_BRANCH_GLYPH,
                              tr("Edit an existing polygon")))
        _hero.addWidget(_head)





        self.auto_correct_pick_hint = _MethodLineLabel("")
        self.auto_correct_pick_hint.setWordWrap(True)
        self.auto_correct_pick_hint.setStyleSheet(_BODY_LINE_QSS)
        self.auto_correct_pick_hint.set_alternatives([
            tr("Click a polygon, then drag any corner."),
            tr("Click a polygon, then click the spot the AI missed.")])
        _hero.addWidget(self.auto_correct_pick_hint)





        self.auto_correct_gesture_art = CorrectGestureArt(self._correct_method)
        _hero.addWidget(self.auto_correct_gesture_art)
        self._build_correct_select_card(lay)







        self.auto_add_lane_card = self._build_add_lane_card()
        self._auto_correct_branches_row = _BranchCardsRow(spacing=SPACE_STAGE)
        self._auto_correct_branches_row.add_card(self.auto_correct_pick_hero)
        self._auto_correct_branches_row.add_card(self.auto_add_lane_card)
        lay.addWidget(self._auto_correct_branches_row)






        self.auto_correct_delete_tip = DismissibleHint(
            HINT_REVIEW_RIGHT_CLICK_DELETE,
            tr("Right-click a polygon on the map to delete it."),
            tint=BLUE_TINT,
            show_glyph=True,
            visibility_gate=self._correct_info_line_gate,
        )


        self.auto_correct_delete_tip.set_flat(False)
        self.auto_correct_delete_tip.setVisible(False)
        lay.addWidget(self.auto_correct_delete_tip)





        self.auto_correct_status = QLabel("")
        self.auto_correct_status.setWordWrap(True)
        self.auto_correct_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.auto_correct_status.linkActivated.connect(
            self._on_correct_status_link)
        self.auto_correct_status.setVisible(False)
        lay.addWidget(self.auto_correct_status)



        self._auto_tiles_debug_row = QWidget()
        _tiles_row = QHBoxLayout(self._auto_tiles_debug_row)
        _tiles_row.setContentsMargins(0, 0, 0, 0)
        _tiles_lbl = QLabel(tr("Show tiles (debug)"))
        _tiles_lbl.setObjectName("autoFieldLabel")
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




        self.auto_correct_select_card = QWidget()
        self.auto_correct_select_card.setObjectName("autoCorrectSelectCard")
        self.auto_correct_select_card.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_correct_select_card.setStyleSheet(
            branch_card_qss("autoCorrectSelectCard"))
        _col = QVBoxLayout(self.auto_correct_select_card)
        _col.setContentsMargins(*_CARD_MARGINS)
        _col.setSpacing(8)




        _title_row = QHBoxLayout()
        _title_row.setContentsMargins(0, 0, 0, 0)
        _title_row.setSpacing(6)
        self.auto_correct_selected_label = QLabel(tr("This polygon"))
        self.auto_correct_selected_label.setStyleSheet(_REVIEW_HEADING_QSS)
        self.auto_correct_selected_info = _muted_line("")
        self.auto_correct_selected_info.setWordWrap(False)
        self.auto_correct_selected_info.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        _title_row.addWidget(self.auto_correct_selected_label)
        _title_row.addStretch(1)
        _title_row.addWidget(self.auto_correct_selected_info)
        _col.addLayout(_title_row)



        self.auto_correct_armed_line = QLabel("")
        self.auto_correct_armed_line.setWordWrap(True)
        self.auto_correct_armed_line.setStyleSheet(_msg_label_qss("armed"))
        self.auto_correct_armed_line.setVisible(False)
        _col.addWidget(self.auto_correct_armed_line)






        self.auto_correct_session_row = QWidget()
        _sess = QHBoxLayout(self.auto_correct_session_row)
        _sess.setContentsMargins(0, 0, 0, 0)
        _sess.setSpacing(6)



        self.auto_reshape_done_btn = QPushButton(tr("Keep"))
        self.auto_reshape_done_btn.setStyleSheet(_BTN_GREEN)
        self.auto_reshape_done_btn.setIcon(icon_for(
            self.auto_reshape_done_btn, "check", 16, token_qcolor(ON_ACCENT)))
        self.auto_reshape_done_btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)
        self.auto_reshape_done_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_reshape_done_btn.setToolTip(tr(
            "Keep this shape. The polygon stays picked, so you can still "
            "adjust, merge or delete it."))
        self.auto_reshape_done_btn.clicked.connect(
            self.auto_reshape_done_requested.emit)
        self.auto_correct_session_undo_btn = QPushButton(tr("Undo"))
        self.auto_correct_session_undo_btn.setStyleSheet(_BTN_GHOST)


        self.auto_correct_session_undo_btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)
        self.auto_correct_session_undo_btn.setCursor(
            Qt.CursorShape.PointingHandCursor)
        self.auto_correct_session_undo_btn.clicked.connect(
            self.auto_correction_undo_requested.emit)
        _sess.addWidget(self.auto_reshape_done_btn, 1)
        _sess.addWidget(self.auto_correct_session_undo_btn, 1)
        self.auto_correct_session_row.setVisible(False)
        _col.addWidget(self.auto_correct_session_row)






        self.auto_correct_rest_box = QWidget()
        _rest = QVBoxLayout(self.auto_correct_rest_box)
        _rest.setContentsMargins(0, 0, 0, 0)
        _rest.setSpacing(8)
        _rest.addWidget(_card_divider())



        self._build_shape_only_controls(_rest)




        self.auto_shape_merge_btn = _action_tile(
            "merge", tr("Merge with neighbours"), tr(
                "One object came back split into several polygons. Click the "
                "others on the map, then confirm to merge them into one."))
        self.auto_shape_merge_btn.clicked.connect(
            lambda: self.auto_shape_edit_requested.emit("merge"))
        _rest.addWidget(self.auto_shape_merge_btn)
        _col.addWidget(self.auto_correct_rest_box)



        self._build_correct_remove_row(_col)

        self.auto_correct_select_card.setVisible(False)
        lay.addWidget(self.auto_correct_select_card)

    def _shape_only_spin_row(self, col, label: str, tooltip: str, widget,
                             suffix: str, glyph: str = "") -> None:



        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        if glyph:
            row.addWidget(_setting_glyph(glyph), 0, Qt.AlignmentFlag.AlignVCenter)
        lbl = QLabel(label)
        lbl.setObjectName("autoFieldLabel")
        lbl.setToolTip(tooltip)
        widget.setSuffix(suffix)
        fit_spin_width(widget, 84, 96)
        widget.setToolTip(tooltip)
        row.addWidget(lbl)
        row.addStretch(1)
        row.addWidget(widget)
        col.addLayout(row)

    def _shape_only_check_row(self, col, label: str, tooltip: str, widget,
                              trailing=None, glyph: str = ""):






        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        if glyph:
            row.addWidget(_setting_glyph(glyph), 0, Qt.AlignmentFlag.AlignVCenter)
        widget.setToolTip(tooltip)
        lbl = QLabel(label)
        lbl.setObjectName("autoFieldLabel")
        lbl.setToolTip(tooltip)


        row.addWidget(lbl)
        row.addStretch(1)
        if trailing is not None:
            row.addWidget(trailing)
        row.addWidget(widget)
        col.addLayout(row)
        return lbl

    def _build_shape_only_controls(self, col) -> None:














        from .fold_row import FoldRow
        self.auto_shape_only_toggle = FoldRow("", managed=False)
        self.auto_shape_only_toggle.clicked.connect(
            self._on_shape_only_toggle_clicked)
        col.addWidget(self.auto_shape_only_toggle)

        self.auto_shape_only_box = QWidget()
        _box = QVBoxLayout(self.auto_shape_only_box)
        _box.setContentsMargins(0, 2, 0, 2)
        _box.setSpacing(6)




        self.auto_shape_only_scope_line = _muted_line("")
        _box.addWidget(self.auto_shape_only_scope_line)



        self.auto_shape_only_ortho = QCheckBox()
        self.auto_shape_only_ortho_label = self._shape_only_check_row(
            _box, label_with_target_hint(tr("Right angles"), tr("buildings")),
            tr("Square this polygon's edges, or leave them as traced while "
               "the rest of the layer stays squared."),
            self.auto_shape_only_ortho, glyph="right_angle")

        self.auto_shape_only_smooth = QCheckBox()
        self._shape_only_check_row(
            _box, label_with_target_hint(tr("Round corners"), tr("trees")), tr(
                "Round this polygon's corners, for a tree or a pond among "
                "squared neighbours."),
            self.auto_shape_only_smooth, glyph="round_corner")

        self.auto_shape_only_fill = QCheckBox()
        self._shape_only_check_row(
            _box, tr("Fill holes"), tr(
                "Close the gaps inside this polygon, without filling the "
                "courtyards the rest of the layer is meant to keep."),
            self.auto_shape_only_fill, glyph="fill_holes")

        _points_tip = tr(
            "How many of this polygon's points to keep. The count in the title "
            "row follows it. It runs before Right angles, so lowering it gives "
            "the squaring straight walls instead of a staircase.")
        self.auto_shape_only_points = QSpinBox()
        self.auto_shape_only_points.setSingleStep(5)



        self.auto_shape_only_points.setRange(1, 100)
        self.auto_shape_only_points.setValue(_AUTO_REVIEW_POINTS_PCT_DEFAULT)
        self._shape_only_spin_row(
            _box, tr("Points"), _points_tip, self.auto_shape_only_points, " %",
            glyph="points")



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
            self.auto_shape_only_simplify, " px", glyph="simplify")

        self.auto_shape_only_clean = QDoubleSpinBox()
        self.auto_shape_only_clean.setDecimals(1)
        self.auto_shape_only_clean.setSingleStep(0.5)
        self.auto_shape_only_clean.setRange(0.0, 50.0)
        self._shape_only_spin_row(
            _box, tr("Trim spikes"), tr(
                "Cut thin spurs off this polygon (0 = off). Raise it on a "
                "single ragged outline instead of eroding the whole layer."),
            self.auto_shape_only_clean, " px", glyph="trim")

        self.auto_shape_only_expand = QSpinBox()
        self.auto_shape_only_expand.setRange(-1000, 1000)
        self._shape_only_spin_row(
            _box, tr("Grow / shrink"), tr(
                "Push this polygon's edge out (positive) or in (negative), "
                "for the one footprint the model cut short or overran."),
            self.auto_shape_only_expand, " px", glyph="grow_shrink")



        self.auto_shape_only_reset = QPushButton(tr("Reset to shared"))
        self.auto_shape_only_reset.setStyleSheet(_BTN_LINK_QUIET)
        self.auto_shape_only_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_shape_only_reset.setVisible(False)
        self.auto_shape_only_reset.clicked.connect(
            self.auto_shape_only_reset_requested.emit)
        _box.addWidget(self.auto_shape_only_reset, 0, Qt.AlignmentFlag.AlignRight)

        col.addWidget(self.auto_shape_only_box)

        self._auto_shape_only_expanded = False
        self._apply_shape_only_mode(self._correct_method)




        for widget, control, param in (
            (self.auto_shape_only_points, "shape_only_points", "points_pct"),
            (self.auto_shape_only_simplify, "shape_only_simplify", "simplify_px"),
            (self.auto_shape_only_clean, "shape_only_trim_spikes", "open_px"),
            (self.auto_shape_only_expand, "shape_only_grow_shrink", "expand_px"),
        ):
            widget.valueChanged.connect(
                lambda _v, c=control, k=param: self._emit_shape_only_changed(c, k))
        for widget, control, param in (
            (self.auto_shape_only_smooth, "shape_only_round_corners", "smooth"),
            (self.auto_shape_only_fill, "shape_only_fill_holes", "fill_holes"),
            (self.auto_shape_only_ortho, "shape_only_right_angles", "ortho"),
        ):
            widget.stateChanged.connect(
                lambda _s, c=control, k=param: self._emit_shape_only_changed(c, k))


        self.auto_shape_only_ortho.stateChanged.connect(
            lambda _s: self._sync_shape_only_right_angles())

    def _build_add_lane_card(self) -> QWidget:







        self.auto_add_lane_card = self._branch_card("autoAddLaneCard")
        _col = self.auto_add_lane_card.layout()
        _head, _, self.auto_add_lane_title = self._branch_head(
            "polygon_add", tr("Add a missing polygon"))
        _col.addWidget(_head)



        self.auto_add_lane_method_line = _MethodLineLabel("")
        self.auto_add_lane_method_line.setWordWrap(True)
        self.auto_add_lane_method_line.setStyleSheet(_MUTED_LINE_QSS)
        self.auto_add_lane_method_line.set_alternatives([
            tr("You place the corners."), tr("The AI outlines it.")])
        _col.addWidget(self.auto_add_lane_method_line)



        self.auto_add_lane_line = QLabel("")
        self.auto_add_lane_line.setWordWrap(True)
        self.auto_add_lane_line.setStyleSheet(_msg_label_qss("armed"))
        self.auto_add_lane_line.setVisible(False)
        _col.addWidget(self.auto_add_lane_line)







        self.auto_add_lane_action_row = QWidget()
        _act = QHBoxLayout(self.auto_add_lane_action_row)
        _act.setContentsMargins(0, 0, 0, 0)
        _act.setSpacing(6)
        self.auto_add_lane_keep_btn = QPushButton(tr("Keep this one"))
        self.auto_add_lane_keep_btn.setStyleSheet(_BTN_GREEN)
        self.auto_add_lane_keep_btn.setIcon(icon_for(
            self.auto_add_lane_keep_btn, "check", 16, token_qcolor(ON_ACCENT)))
        self.auto_add_lane_keep_btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)
        self.auto_add_lane_keep_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_add_lane_keep_btn.setToolTip(tr(
            "Keep this outline and point at the next object. Shortcut: S"))
        self.auto_add_lane_keep_btn.clicked.connect(
            self.auto_ai_add_keep_requested.emit)
        self.auto_add_lane_keep_btn.setVisible(False)
        self.auto_add_lane_undo_btn = QPushButton(tr("Undo point"))
        self.auto_add_lane_undo_btn.setStyleSheet(_BTN_GHOST)
        self.auto_add_lane_undo_btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)
        self.auto_add_lane_undo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_add_lane_undo_btn.setToolTip(tr(
            "Take back the last point you placed. Shortcut: Ctrl+Z"))
        self.auto_add_lane_undo_btn.clicked.connect(
            self.auto_correction_undo_requested.emit)
        self.auto_add_lane_undo_btn.setVisible(False)
        _act.addWidget(self.auto_add_lane_keep_btn, 1)
        _act.addWidget(self.auto_add_lane_undo_btn, 1)
        self.auto_add_lane_action_row.setVisible(False)
        _col.addWidget(self.auto_add_lane_action_row)

        self.auto_add_lane_btn = _action_tile(
            "spark", tr("Point at it on the map"), tr(
                "Add an object the AI missed. In AI, point at it and the "
                "model outlines it for one cloud detection; in Manual, draw "
                "its corners for free."))
        self.auto_add_lane_btn.clicked.connect(self._on_add_lane_clicked)
        _col.addWidget(self.auto_add_lane_btn)

        self.auto_add_lane_card.setVisible(False)
        return self.auto_add_lane_card

    def _build_correct_remove_row(self, lay) -> None:




        self.auto_correct_remove_row = QWidget()
        _row = QHBoxLayout(self.auto_correct_remove_row)
        _row.setContentsMargins(0, 0, 0, 0)
        _row.setSpacing(6)
        self.auto_remove_btn = QPushButton(tr("Delete this polygon"))
        self.auto_remove_btn.setStyleSheet(_BTN_REMOVE_ROW)
        self.auto_remove_btn.setIcon(icon_for(self.auto_remove_btn, "trash", 14))
        self.auto_remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_remove_btn.setToolTip(tr(
            "Delete this polygon (the Delete key works too, and a right-click "
            "on the map deletes the shape under the cursor). "
            "Undo brings it back."))
        self.auto_remove_btn.clicked.connect(self.auto_remove_requested.emit)
        _row.addWidget(self.auto_remove_btn, 1)
        lay.addWidget(self.auto_correct_remove_row)

    def _build_correct_summary_row(self, lay) -> None:



        build_correction_summary_row(self, lay)

    def _on_correct_clear_clicked(self) -> None:


        guard = getattr(self, "_correct_clear_confirm", None)
        if guard is None or guard.clicked():
            self.auto_correction_clear_requested.emit()

    def _build_reshape_install_banner(self, lay) -> None:







        self._auto_review_installing = False
        self.auto_review_install_banner = QWidget()
        self.auto_review_install_banner.setObjectName("autoReviewInstallBanner")
        self.auto_review_install_banner.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_review_install_banner.setStyleSheet(
            category_card_qss("autoReviewInstallBanner", HUE_LOCAL))
        _col = QVBoxLayout(self.auto_review_install_banner)
        _col.setContentsMargins(*_CARD_MARGINS)
        _col.setSpacing(6)
        self.auto_review_install_label = QLabel(
            tr("Setting up the on-device AI..."))
        self.auto_review_install_label.setWordWrap(True)
        self.auto_review_install_label.setStyleSheet(_BODY_LINE_QSS)
        _col.addWidget(self.auto_review_install_label)
        self.auto_review_install_progress = QProgressBar()
        self.auto_review_install_progress.setRange(0, 100)
        self.auto_review_install_progress.setValue(0)
        self.auto_review_install_progress.setTextVisible(False)
        self.auto_review_install_progress.setStyleSheet(category_progress_qss(HUE_LOCAL))
        _col.addWidget(self.auto_review_install_progress)
        _cancel_row = QHBoxLayout()
        _cancel_row.setContentsMargins(0, 0, 0, 0)
        _cancel_row.addStretch(1)
        self.auto_review_install_cancel_btn = QPushButton(tr("Cancel setup"))
        self.auto_review_install_cancel_btn.setStyleSheet(_BTN_LINK_MUTED)
        self.auto_review_install_cancel_btn.setCursor(
            Qt.CursorShape.PointingHandCursor)
        self.auto_review_install_cancel_btn.setToolTip(tr(
            "Stop the setup and go back to the review. The AI fix stays "
            "unavailable until you install it."))
        self.auto_review_install_cancel_btn.clicked.connect(
            self.auto_review_install_cancel_requested.emit)
        _cancel_row.addWidget(self.auto_review_install_cancel_btn)
        _col.addLayout(_cancel_row)
        self.auto_review_install_banner.setVisible(False)
        lay.addWidget(self.auto_review_install_banner)
