















from __future__ import annotations

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QAbstractSpinBox,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from ..icons import icon_for, pixmap_for
from .auto_correct_build import _REVIEW_HEADING_QSS
from .auto_flow_look import (
    _BTN_AUTO_QUIET,
    TaskDisc,
    name_unlabelled_controls,
    repolish_widget,
    toggle_indicator_qss,
    token_qcolor,
)
from .font_scale import fit_spin_width, scale_px_length, scale_qss_font_px
from .guidance import (
    BLUE_TINT,
    HINT_REVIEW_CLOSED_CANOPY,
    HINT_REVIEW_CONFIDENCE,
    DismissibleHint,
)
from .review_step_ladder import REVIEW_LADDER_ROW_PX, ReviewLadderStrip, ReviewStepChip


from .review_view_block import display_legend_html  # noqa: F401
from .styles import (
    _BTN_CHIP,
    _BTN_GREEN_STEP,
    _REVIEW_CONF_SPIN_MIN,
    _SLIDER_QSS,
    BTN_PRIMARY_WIDE_PX,
    BTN_SMALL_PX,
    FONT_HINT,
    INK,
    RED_TEXT,
    SPACE_CARD,
    review_conf_max,
    review_conf_min,
    review_conf_step,
)

__all__ = [
    "DockAutoReviewBuildMixin",
    "_BTN_LINK_CONFIRM",
    "_export_btn_label",
    "display_legend_html",
]








_REVIEW_CONTENT_MAX_PX = 640





_BTN_LINK_CONFIRM = scale_qss_font_px(
    "QPushButton { background: transparent; border: none;"
    f" color: {RED_TEXT}; font-size: {FONT_HINT}px; padding: 4px 8px;"
    " text-decoration: underline; }"
)


class _CurrentPageStack(QWidget):












    currentChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pages: list[QWidget] = []
        self._current = -1
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

    def addWidget(self, page: QWidget) -> int:
        self._pages.append(page)
        page.setParent(self)
        page.hide()
        if self._current < 0:
            self.setCurrentIndex(0)
        return len(self._pages) - 1

    def count(self) -> int:
        return len(self._pages)

    def widget(self, index: int):
        if 0 <= index < len(self._pages):
            return self._pages[index]
        return None

    def currentIndex(self) -> int:
        return self._current

    def currentWidget(self):
        return self.widget(self._current)

    def setCurrentIndex(self, index: int) -> None:
        if index == self._current or not 0 <= index < len(self._pages):
            return
        lay = self.layout()
        old = self.currentWidget()
        if old is not None:
            lay.removeWidget(old)
            old.hide()
        self._current = index
        page = self._pages[index]
        lay.addWidget(page)
        page.show()
        self.updateGeometry()
        self.currentChanged.emit(index)


def _export_btn_label(n: int) -> str:


    if n == 1:
        return tr("Export 1 polygon")
    return tr("Export {n} polygons").format(n=n)


class DockAutoReviewBuildMixin:


    def _build_deferred_review_panel(self, parent_layout, after_widget) -> None:







        self._setup_auto_review_panel(
            parent_layout, parent_layout.indexOf(after_widget) + 1)
        name_unlabelled_controls(self.auto_review_panel)
        self._setup_auto_review_view_block(self.main_layout)
        self._finish_dock_part(self.auto_review_panel)

    def _setup_auto_review_panel(self, parent_layout, index: int = -1):






        self.auto_review_panel = QWidget()
        self.auto_review_panel.setVisible(False)






        _outer = QHBoxLayout(self.auto_review_panel)
        _outer.setContentsMargins(0, 0, 0, 0)
        _outer.setSpacing(0)
        _content = QWidget()
        _content.setMaximumWidth(scale_px_length(_REVIEW_CONTENT_MAX_PX))
        _review_layout = QVBoxLayout(_content)
        _review_layout.setContentsMargins(0, 0, 0, 0)


        _review_layout.setSpacing(8)





        _content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        _outer.addStretch(0)
        _outer.addWidget(_content, 1)
        _outer.addStretch(0)




        self._auto_review_step = 0
        self._auto_zero_entry = False















        from .widgets import checkbox_indicator_qss
        _card = QWidget()
        _card.setObjectName("autoReviewCard")
        _card_qss = "QLabel { background: transparent; border: none; }"
        _card_qss += _SLIDER_QSS
        _card_qss += checkbox_indicator_qss(self)

        _card_qss += toggle_indicator_qss(self)
        _card.setStyleSheet(_card_qss)
        _card_layout = QVBoxLayout(_card)


        _card_layout.setContentsMargins(0, 0, 0, 4)
        _card_layout.setSpacing(SPACE_CARD)









        self._auto_review_card_layout = _card_layout
        self._auto_review_dials_row = self._build_review_dials()
        _card_layout.addWidget(self._auto_review_dials_row)










        from .auto_review_steps import build_shapes_page
        self.auto_review_step_stack = _CurrentPageStack()
        self.auto_review_step_stack.addWidget(self._build_auto_keep_page())
        self.auto_review_step_stack.addWidget(self._build_auto_correct_page())
        self.auto_review_step_stack.addWidget(build_shapes_page(self))




        for _i in range(self.auto_review_step_stack.count()):
            _page_lay = self.auto_review_step_stack.widget(_i).layout()
            if _page_lay is not None:
                _page_lay.setContentsMargins(0, 2, 0, 6)
        self.auto_review_step_stack.currentChanged.connect(
            self._sync_review_stack_height)
        self._sync_review_stack_height()





        self.auto_review_step_stack.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)


        self._place_review_stack(self.auto_review_step_stack.currentIndex())
        _card_layout.addWidget(self._build_review_busy_row())



        self._auto_review_card = _card
        _review_layout.addWidget(_card)











        self.auto_step_next_btn = QPushButton("")
        self.auto_step_next_btn.setStyleSheet(_BTN_GREEN_STEP)
        self.auto_step_next_btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)

        self.auto_step_next_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.auto_step_next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_step_next_btn.clicked.connect(
            self._on_auto_step_next_clicked)
        _review_layout.addWidget(self.auto_step_next_btn)




        self.auto_export_btn = QPushButton(_export_btn_label(0))
        self.auto_export_btn.setStyleSheet(_BTN_GREEN_STEP)


        self.auto_export_btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)
        self.auto_export_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.auto_export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_export_btn.clicked.connect(self.auto_export_requested.emit)
        self.auto_export_btn.setVisible(False)
        _review_layout.addWidget(self.auto_export_btn)








        self.auto_retry_btn = QPushButton(tr("Re-run the whole zone"))
        self.auto_retry_btn.setStyleSheet(_BTN_AUTO_QUIET)
        self.auto_retry_btn.setIcon(icon_for(self.auto_retry_btn, "refresh", 14))
        self.auto_retry_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_retry_btn.setToolTip(tr(
            "Go back to your zone, references and settings, then detect the "
            "whole zone again. Nothing is saved."))
        self.auto_retry_btn.clicked.connect(self.auto_retry_requested.emit)
        self.auto_retry_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.auto_review_exit_btn = QPushButton(tr("Exit"))
        self.auto_review_exit_btn.setStyleSheet(_BTN_AUTO_QUIET)
        self.auto_review_exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_review_exit_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.auto_review_exit_btn.clicked.connect(
            self.auto_review_exit_requested.emit)
        _review_actions_row = QHBoxLayout()
        _review_actions_row.setContentsMargins(0, 0, 0, 0)
        _review_actions_row.setSpacing(2)
        self._auto_review_links_sep = QLabel("·")
        self._auto_review_links_sep.setObjectName("autoMicro")
        _review_actions_row.addStretch(1)
        _review_actions_row.addWidget(self.auto_retry_btn)
        _review_actions_row.addWidget(self._auto_review_links_sep)
        _review_actions_row.addWidget(self.auto_review_exit_btn)
        _review_actions_row.addStretch(1)
        _review_layout.addLayout(_review_actions_row)





        from .upsell_card import UpsellCard
        self.auto_review_free_fit_card = UpsellCard(
            "autoReviewFreeFitCard", "full",
            on_cta=self._on_free_zone_fit_offer_clicked)
        self.auto_review_free_fit_card.dismissed.connect(
            self._on_free_zone_fit_offer_dismissed)
        self.auto_review_free_fit_card.setVisible(False)
        _review_layout.addWidget(self.auto_review_free_fit_card)


        self._auto_review_column_layout = _review_layout

        parent_layout.insertWidget(index, self.auto_review_panel)

    def _setup_auto_review_view_block(self, parent_layout):










        from .review_view_block import build_review_view_block

        build_review_view_block(self)



        column = getattr(self, "_auto_review_column_layout", None)
        if column is not None:
            column.addWidget(self.auto_review_view_row)
        else:
            parent_layout.addWidget(self.auto_review_view_row)



    def _build_auto_keep_page(self) -> QWidget:















        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)

        lay.setSpacing(8)





        from .review_card_rows import review_zone
        self.auto_review_confidence_zone, _conf_col = review_zone(
            "autoKeepConfidenceZone")
        lay.addWidget(self.auto_review_confidence_zone)





        self.auto_review_confidence_header = QWidget()
        _conf_hdr = QHBoxLayout(self.auto_review_confidence_header)
        _conf_hdr.setContentsMargins(0, 0, 0, 0)
        _conf_hdr.setSpacing(8)


        _conf_glyph = QLabel()
        _conf_glyph.setPixmap(pixmap_for(_conf_glyph, "chart", 16, token_qcolor(INK)))
        _conf_glyph.setFixedSize(16, 16)
        _conf_glyph.setStyleSheet("background: transparent; border: none;")
        _conf_hdr.addWidget(_conf_glyph, 0, Qt.AlignmentFlag.AlignVCenter)
        _conf_review_lbl = QLabel(tr("Confidence"))
        _conf_review_lbl.setStyleSheet(_REVIEW_HEADING_QSS)
        _conf_hdr.addWidget(_conf_review_lbl)
        _conf_hdr.addStretch()





        self.auto_review_confidence_spin = QSpinBox()
        self.auto_review_confidence_spin.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.auto_review_confidence_spin.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.auto_review_confidence_spin.setRange(
            _REVIEW_CONF_SPIN_MIN, review_conf_max())


        self.auto_review_confidence_spin.setSingleStep(1)
        self.auto_review_confidence_spin.setValue(30)
        self.auto_review_confidence_spin.setSuffix("%")
        fit_spin_width(self.auto_review_confidence_spin, 56, 72)
        self.auto_review_confidence_spin.setAccessibleName(tr("Confidence cutoff"))

        self.auto_review_confidence_spin.setAttribute(
            Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        _conf_hdr.addWidget(self.auto_review_confidence_spin)
        _conf_col.addWidget(self.auto_review_confidence_header)






        self._auto_review_count_label = QLabel("")
        self._auto_review_count_label.setWordWrap(True)
        self._auto_review_count_label.setObjectName("autoHint")
        _conf_col.addWidget(self._auto_review_count_label)






        self.auto_review_reveal_btn = QPushButton(tr("Show them"))
        self.auto_review_reveal_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_review_reveal_btn.setStyleSheet(_BTN_CHIP)

        self.auto_review_reveal_btn.setMinimumHeight(BTN_SMALL_PX)
        self.auto_review_reveal_btn.setAutoDefault(False)
        self.auto_review_reveal_btn.setVisible(False)
        self.auto_review_reveal_btn.clicked.connect(
            self._on_auto_review_reveal_clicked)


        _conf_col.addWidget(self.auto_review_reveal_btn, 0,
                            Qt.AlignmentFlag.AlignLeft)






        self.auto_review_flat_score_note = QLabel("")
        self.auto_review_flat_score_note.setWordWrap(True)
        self.auto_review_flat_score_note.setVisible(False)
        _conf_col.addWidget(self.auto_review_flat_score_note)



        from ..confidence_histogram import ConfidenceHistogram
        self.auto_conf_histogram = ConfidenceHistogram()
        self.auto_conf_histogram.setToolTip(
            tr("How many objects sit at each confidence level."))
        _conf_col.addWidget(self.auto_conf_histogram)
        self.auto_review_confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.auto_review_confidence_slider.setRange(
            review_conf_min(), review_conf_max())
        self.auto_review_confidence_slider.setValue(30)
        self.auto_review_confidence_slider.setSingleStep(review_conf_step())
        self.auto_review_confidence_slider.setPageStep(review_conf_step())
        self.auto_review_confidence_slider.setMinimumHeight(26)
        self.auto_review_confidence_slider.setToolTip(tr(
            "Filter detections by confidence. Lower shows more (and noisier),"
            " higher keeps only the strongest. Free and instant."))


        self.auto_review_confidence_slider.valueChanged.connect(
            self._on_conf_slider_moved)
        self.auto_review_confidence_slider.sliderReleased.connect(
            self._schedule_conf_refilter)
        self.auto_review_confidence_spin.valueChanged.connect(
            self._on_conf_spin_changed)
        _conf_col.addWidget(self.auto_review_confidence_slider)




        self.auto_review_confidence_ends = QWidget()
        _conf_ends = QHBoxLayout(self.auto_review_confidence_ends)
        _conf_ends.setContentsMargins(2, 0, 2, 0)
        _conf_left = QLabel(tr("More objects"))
        _conf_left.setObjectName("autoMicro")
        _conf_right = QLabel(tr("Only confident"))
        _conf_right.setObjectName("autoMicro")
        _conf_ends.addWidget(_conf_left)
        _conf_ends.addStretch()
        _conf_ends.addWidget(_conf_right)
        _conf_col.addWidget(self.auto_review_confidence_ends)









        self.auto_closed_canopy_hint = DismissibleHint(
            HINT_REVIEW_CLOSED_CANOPY,
            tr("Closed forest: the AI takes it as one cover and does not "
               "separate its trees. For the forest as one area, re-run "
               'with "forest".'),
            tint=BLUE_TINT,
        )


        self.auto_closed_canopy_hint.set_flat(True)
        self.auto_closed_canopy_hint.setVisible(False)
        _conf_col.addWidget(self.auto_closed_canopy_hint)

        self.auto_confidence_hint = DismissibleHint(
            HINT_REVIEW_CONFIDENCE,
            tr("How sure the AI is about each object. Lower shows more, "
               "higher keeps only the sure ones."),
            tint=BLUE_TINT,
        )


        self.auto_confidence_hint.set_flat(True)
        _conf_col.addWidget(self.auto_confidence_hint)

        lay.addWidget(self._build_size_filter_block())
        lay.addWidget(self._build_boundary_snap_block())

        lay.addStretch(1)
        return page

    def _build_size_filter_block(self) -> QWidget:


        from .review_card_rows import build_size_filter_block

        return build_size_filter_block(self)

    def _build_boundary_snap_block(self) -> QWidget:


        from .review_card_rows import build_boundary_snap_block

        return build_boundary_snap_block(self)

    def _build_review_busy_row(self) -> QWidget:


        from .review_card_rows import build_review_busy_row

        return build_review_busy_row(self)



    def _build_review_dials(self) -> QWidget:






        ladder = ReviewLadderStrip()
        outer = QHBoxLayout(ladder)
        outer.setContentsMargins(2, 0, 2, 0)
        outer.setSpacing(0)
        self._auto_review_dials = []
        labels = []
        for i, name in enumerate(
                (tr("Keep"), tr("Correct"), tr("Shapes"))):
            col = ReviewStepChip(i)

            if i != 0:
                col.setToolTip(tr("Click to open this step"))
            col.clicked.connect(self._on_review_dial_clicked)


            col.setFixedHeight(scale_px_length(REVIEW_LADDER_ROW_PX))
            row = QHBoxLayout(col)


            row.setContentsMargins(4, 4, 4, 4)
            row.setSpacing(6)
            dial = TaskDisc(i + 1, "active" if i == 0 else "todo")
            row.addWidget(dial, 0, Qt.AlignmentFlag.AlignVCenter)
            lab = QLabel(name)
            lab.setObjectName("autoTaskLabel")
            lab.setProperty("state", "active" if i == 0 else "todo")
            row.addWidget(lab, 0, Qt.AlignmentFlag.AlignVCenter)
            outer.addWidget(col, 0)
            self._auto_review_dials.append((col, dial, lab))
            labels.append(lab)
            if i < 2:
                rule = QFrame()
                rule.setObjectName("autoTaskRule")
                rule.setFixedHeight(1)
                outer.addWidget(rule, 1, Qt.AlignmentFlag.AlignVCenter)
        ladder.register(labels)
        self.auto_review_dials_row = ladder
        return ladder

    def _place_review_stack(self, index: int = -1) -> None:







        try:
            lay = self._auto_review_card_layout
            stack = self.auto_review_step_stack
            row = self._auto_review_dials_row
        except AttributeError:
            return
        if lay.indexOf(stack) >= 0:
            return
        at = lay.indexOf(row)
        if at < 0:
            return
        lay.insertWidget(at + 1, stack)
        stack.show()

    def _on_review_dial_clicked(self, step: int) -> None:



        try:
            if step == getattr(self, "_auto_review_step", 0):
                return
            self.auto_review_step_requested.emit(int(step))
        except (RuntimeError, AttributeError):
            pass

    def _set_review_dial(self, idx: int, state: str) -> None:


        try:
            _col, dial, lab = self._auto_review_dials[idx]
        except (AttributeError, IndexError):
            return
        dial.set_disc_state(state)
        if lab.property("state") != state:
            lab.setProperty("state", state)
            repolish_widget(lab)
        if state == "active":
            try:
                self.auto_review_dials_row.set_current(idx)
            except (AttributeError, RuntimeError):
                pass

    def _set_review_dials_locked(self, locked: bool, current: int) -> None:






        try:
            dials = self._auto_review_dials
        except AttributeError:
            return
        for i, (col, _dial, _lab) in enumerate(dials):
            if locked and i != current:


                if not isinstance(col.graphicsEffect(), QGraphicsOpacityEffect):
                    dim = QGraphicsOpacityEffect(col)
                    dim.setOpacity(0.45)
                    col.setGraphicsEffect(dim)
            elif col.graphicsEffect() is not None:
                col.setGraphicsEffect(None)
            navigable = not locked and i != current
            try:
                col.set_navigable(navigable)
                col.setToolTip(tr("Click to open this step") if navigable
                               else "")
            except AttributeError:
                pass

    def _sync_review_stack_height(self, _index: int = -1) -> None:













        try:
            if _index >= 0:
                self._place_review_stack(_index)
            self.auto_review_step_stack.updateGeometry()
        except (RuntimeError, AttributeError):
            pass








