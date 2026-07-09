





from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from ...core.pro_ceiling import pro_ceiling_contact_email
from ...core.review_defaults import (
    AUTO_DEFAULT_CONFIDENCE as _AUTO_DEFAULT_CONFIDENCE,
)
from ...core.review_defaults import (
    AUTO_REVIEW_CLEAN_DEFAULT as _AUTO_REVIEW_CLEAN_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_EXPAND_DEFAULT as _AUTO_REVIEW_EXPAND_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_FILL_HOLES_DEFAULT as _AUTO_REVIEW_FILL_HOLES_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT as _AUTO_REVIEW_FILL_MAX_M2_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_ORTHO_DEFAULT as _AUTO_REVIEW_ORTHO_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_POINTS_PCT_DEFAULT as _AUTO_REVIEW_POINTS_PCT_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_SIMPLIFY_DEFAULT as _AUTO_REVIEW_SIMPLIFY_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_SMOOTH_DEFAULT as _AUTO_REVIEW_SMOOTH_DEFAULT,
)
from ...core.server_dials import dial_copy
from ...core.tile_manager import MAX_DETAIL_LEVEL
from ..icons import icon_for, pixmap_for
from ..layer_tree_combobox import LayerTreeComboBox
from .auto_flow_look import (
    _BTN_AUTO_CHIP,
    _BTN_GHOST_WIDE,
    COMPOSER_INPUT_QSS,
    FitTextButton,
    NeverShownWidget,
    VisibilityTwin,
    auto_flow_sheet,
    name_unlabelled_controls,
    token_qcolor,
)
from .auto_flow_look import ComposerFrame as _ComposerFrame
from .auto_run_status import _auto_progress_bar_qss
from .auto_run_summary import AutoRunSummaryCard
from .cloud_notice_line import build_cloud_notice_line
from .font_scale import fit_spin_width
from .guidance import (
    BLUE_TINT,
    GREEN_TINT,
    HINT_EXEMPLAR_DRAW_BOX,
    HINT_EXEMPLAR_TIP,
    HINT_INPUT_RULE,
    HINT_PROMPT_TREE_OR_FOREST,
    HINT_RERUN_SAME_SETUP,
    HINT_START_AUTO,
    DismissibleHint,
)
from .run_status_loader import RunDotsGrid, RunElapsedClock
from .styles import (
    _BTN_BLUE_OUTLINE,
    _BTN_BLUE_STEP,
    _BTN_GHOST,
    _BTN_GREEN_STEP,
    _CARD_CHILD_BTN_RESET_QSS,
    _CARD_MARGINS,
    _CARD_QSS,
    _FIELD_LABEL_QSS,
    _FOLD_ROW_QSS,
    _FOLD_TITLE_QSS,
    _HINT_LINE_QSS,
    _SLIDER_QSS,
    _SUBCARD_MARGINS,
    BTN_CHIP_PX,
    BTN_PILL_PX,
    BTN_PRIMARY_WIDE_PX,
    HUE_RESULT,
    INK_2,
    MODE_HUES,
    ORANGE_TEXT,
    _btn_start_qss,
    _btn_toggle_qss,
    _micro_header,
    _msg_card_qss,
    _msg_label_qss,
    category_label_qss,
    combo_theme_qss,
)
from .upsell_card import UpsellCard, keep_working_cta
from .widgets import (
    Mode,
    _ZoneGestureGlyph,
    build_no_imagery_hero,
    make_shortcut_hint,
    native_key,
)
from .wrapping_button_row import WrappingButtonRow


class ExampleCardWithSeparator(QWidget):











    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._companion = None

    def set_companion(self, widget) -> None:

        self._companion = widget

    def setVisible(self, visible: bool) -> None:  # noqa: N802
        super().setVisible(visible)
        companion = self._companion
        if companion is None:
            return
        try:
            companion.setVisible(visible)
        except RuntimeError:
            self._companion = None


def _sentence_case(text: str) -> str:





    return text[:1].upper() + text[1:] if text else text




_BRAND_BLUE_RGB = (30, 136, 229)
_ON_BRAND_BLUE_INK = "#000000"


class DockAutoBuildMixin:


    def _setup_automatic_page(self):

        self.auto_page = QWidget()
        self.auto_page.setObjectName("autoPage")


        self.auto_page.setStyleSheet(auto_flow_sheet())
        auto_layout = QVBoxLayout(self.auto_page)
        auto_layout.setContentsMargins(0, 8, 0, 0)
        auto_layout.setSpacing(8)

        from qgis.PyQt.QtWidgets import QSizePolicy as _QSizePolicy




        self.auto_upsell_card = QFrame()
        self.auto_upsell_card.setObjectName("autoUpsellCard")
        self.auto_upsell_card.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)





        self.auto_upsell_card.setStyleSheet(
            _CARD_QSS.format(name="autoUpsellCard")
            + "QLabel { background: transparent; border: none; }"
            + _CARD_CHILD_BTN_RESET_QSS)
        upsell_layout = QVBoxLayout(self.auto_upsell_card)
        upsell_layout.setContentsMargins(*_SUBCARD_MARGINS)
        upsell_layout.setSpacing(4)






        _wall = UpsellCard("autoUpsellOffer", "wall", self._on_upgrade_clicked)


        self.auto_upgrade_btn = _wall.button



        self._auto_upsell_title = _wall.title



        self._auto_upsell_reset = _wall.note
        _wall.set_text(



            dial_copy(
                "trial.exhausted_no_count",
                tr("Your free cloud detections are used up")),




            dial_copy(
                "upsell.wall_body",
                tr("Draw a whole city and let it run, at the finest "
                   "precision.")),
            keep_working_cta(),
            star=dial_copy(
                "upsell.bullet_quota",
                tr("Pro unlocks far more Automatic surface every month, on "
                   "zones of any size.")),
        )

        _wall.set_pro_offer("plugin_free_exhausted_wall")



        self._auto_upsell_wall = _wall
        _wall.set_contact_email(pro_ceiling_contact_email())
        upsell_layout.addWidget(_wall)



        upsell_layout.addSpacing(6)
        _rule = QFrame()
        _rule.setObjectName("autoTaskRule")
        _rule.setFixedHeight(1)
        upsell_layout.addWidget(_rule)

        _upsell_free = QLabel(dial_copy(
            "upsell.manual_free",
            tr("Or click objects one by one in Semi-Auto.")))
        _upsell_free.setObjectName("autoHint")
        _upsell_free.setWordWrap(True)
        upsell_layout.addWidget(_upsell_free)



        self.auto_upsell_manual_btn = QPushButton(dial_copy(
            "upsell.manual_cta", tr("Use Semi-Auto")))
        self.auto_upsell_manual_btn.setMinimumHeight(BTN_CHIP_PX)
        self.auto_upsell_manual_btn.setCursor(Qt.CursorShape.PointingHandCursor)


        self.auto_upsell_manual_btn.setStyleSheet(_BTN_AUTO_CHIP)
        self.auto_upsell_manual_btn.clicked.connect(
            self._on_auto_upsell_manual_clicked)
        upsell_layout.addWidget(self.auto_upsell_manual_btn)



        self.auto_upsell_card.setSizePolicy(
            _QSizePolicy.Policy.Preferred, _QSizePolicy.Policy.Maximum)
        auto_layout.addWidget(self.auto_upsell_card)





        self._setup_auto_run_block(auto_layout)











        self.auto_controls_section = QWidget()
        controls_layout = QVBoxLayout(self.auto_controls_section)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)










        self.auto_layer_label = QLabel(tr("Image to segment"))
        self.auto_layer_label.setStyleSheet(_FIELD_LABEL_QSS)
        controls_layout.addWidget(self.auto_layer_label)

        self.auto_layer_combo = LayerTreeComboBox()
        self.auto_layer_combo.setToolTip(
            tr("Select a raster layer (GeoTIFF, WMS, XYZ tiles, etc.)"))


        self.auto_layer_combo.setStyleSheet(combo_theme_qss())
        self.auto_layer_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.auto_layer_combo.setMinimumWidth(0)
        self.auto_layer_combo.layerChanged.connect(self._on_auto_layer_changed)
        controls_layout.addWidget(self.auto_layer_combo)








        self.auto_no_rasters_widget, self.auto_demo_btn = build_no_imagery_hero(
            on_demo=self.auto_demo_requested.emit,
        )
        self.auto_no_rasters_widget.setVisible(False)


        controls_layout.addWidget(self.auto_no_rasters_widget)



        self._auto_hero_twin = VisibilityTwin(self.auto_no_rasters_widget)
        controls_layout.addWidget(self._auto_hero_twin)








        self.auto_steps = QStackedWidget()
        controls_layout.addWidget(self.auto_steps, 1)

        def _make_page():
            page = QWidget()
            lay = QVBoxLayout(page)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(8)
            self.auto_steps.addWidget(page)
            return lay

        _s1_layout = _make_page()





        _s1_layout.setContentsMargins(0, 8, 0, 0)

        _s2_layout = _make_page()
        _s3_layout = _make_page()


        self.auto_start_btn = QPushButton(tr("Start Automatic AI Segmentation"))




        self.auto_start_btn.setStyleSheet(_btn_start_qss(_BTN_BLUE_STEP, full_width=True))
        self.auto_start_btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)


        from qgis.PyQt.QtWidgets import QSizePolicy as _StartPolicy
        self.auto_start_btn.setSizePolicy(
            _StartPolicy.Policy.Ignored, _StartPolicy.Policy.Fixed)
        self.auto_start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_start_btn.setEnabled(False)
        self.auto_start_btn.clicked.connect(self._on_auto_start_clicked)
        _s1_layout.addWidget(self.auto_start_btn)







        self.auto_start_caption = DismissibleHint(
            HINT_START_AUTO,
            tr("Draw a zone, name one kind of object, and get all of them in "
               "one run. Use Semi-Auto mode to work one object at a time."),
            tint=GREEN_TINT,


            hue=MODE_HUES["automatic"],
            glyph="sparkles",
        )
        _s1_layout.addWidget(self.auto_start_caption)
















        self.auto_export_success = QLabel()
        self.auto_export_success.setWordWrap(True)
        self.auto_export_success.setTextFormat(Qt.TextFormat.RichText)
        self.auto_export_success.setOpenExternalLinks(False)
        self.auto_export_success.linkActivated.connect(self._on_auto_recap_link)
        self.auto_export_success.setStyleSheet(category_label_qss(HUE_RESULT))
        self.auto_export_success.setVisible(False)
        _s1_layout.addWidget(self.auto_export_success)


        from .pro_nudges import build_pro_after_success_label
        self.auto_pro_after_success = build_pro_after_success_label(
            self._on_pro_after_success_link)
        _s1_layout.addWidget(self.auto_pro_after_success)



        from .tutorial_link import build_home_tutorial_link
        self.auto_tutorial_link = build_home_tutorial_link(
            self._on_open_guide_footer, "autoTutorialRow")
        _s1_layout.addWidget(self.auto_tutorial_link)




        from ..canvas_palette import CHROME_BLUE
        self.auto_zone_hero = QWidget()
        _hero_layout = QVBoxLayout(self.auto_zone_hero)
        _hero_layout.setContentsMargins(16, 8, 16, 0)
        _hero_layout.setSpacing(10)
        self._auto_zone_glyph = _ZoneGestureGlyph(CHROME_BLUE)
        _hero_layout.addWidget(
            self._auto_zone_glyph, 0, Qt.AlignmentFlag.AlignHCenter)
        self._auto_zone_title = QLabel(tr("Draw your zone"))
        self._auto_zone_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._auto_zone_title.setObjectName("autoTitle")
        _hero_layout.addWidget(self._auto_zone_title)
        self._auto_zone_hint = QLabel(
            tr("Click on the map to outline your zone."))
        self._auto_zone_hint.setWordWrap(True)
        self._auto_zone_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._auto_zone_hint.setObjectName("autoHint")
        _hero_layout.addWidget(self._auto_zone_hint)




        _s2_layout.addSpacing(24)
        _s2_layout.addWidget(self.auto_zone_hero)



        _zone_exit_row = QHBoxLayout()
        _zone_exit_row.addStretch()
        self.auto_zone_exit_btn = QPushButton(tr("Exit"))
        self.auto_zone_exit_btn.setStyleSheet(_BTN_GHOST)
        self.auto_zone_exit_btn.setMinimumHeight(BTN_PILL_PX)
        self.auto_zone_exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_zone_exit_btn.clicked.connect(self.auto_exit_requested.emit)
        _zone_exit_row.addWidget(self.auto_zone_exit_btn)
        _zone_exit_row.addStretch()
        _s2_layout.addLayout(_zone_exit_row)



        _s2_layout.addSpacing(4)
        self._auto_zone_keys = make_shortcut_hint([
            (native_key(Qt.Key.Key_Backspace), _sentence_case(tr("undo point"))),
            (native_key(Qt.Key.Key_Escape), _sentence_case(tr("cancel"))),
        ])
        self._auto_zone_keys.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._auto_zone_keys.setObjectName("autoMicro")
        _s2_layout.addWidget(self._auto_zone_keys)














        self.auto_run_summary_card = AutoRunSummaryCard()
        _s3_layout.addWidget(self.auto_run_summary_card)












        self.auto_prompt_card = QWidget()
        self.auto_prompt_card.setObjectName("autoPromptCard")
        self.auto_prompt_card.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_prompt_card.setStyleSheet(
            _CARD_QSS.format(name="autoPromptCard"))
        _prompt_card_layout = QVBoxLayout(self.auto_prompt_card)
        _prompt_card_layout.setContentsMargins(*_CARD_MARGINS)
        _prompt_card_layout.setSpacing(6)
        self._auto_prompt_header = QLabel(tr("Describe what to detect"))



        self._auto_prompt_header.setStyleSheet(_FOLD_TITLE_QSS)

        self._auto_prompt_header.setWordWrap(True)
        _prompt_card_layout.addWidget(self._auto_prompt_header)

        self.auto_prompt_composer = _ComposerFrame()
        self.auto_prompt_composer.setObjectName("autoComposer")
        _composer_col = QVBoxLayout(self.auto_prompt_composer)
        _composer_col.setContentsMargins(12, 8, 8, 8)
        _composer_col.setSpacing(4)
        self.auto_prompt_input = QLineEdit()
        self.auto_prompt_input.setObjectName("autoComposerInput")


        self.auto_prompt_input.setStyleSheet(COMPOSER_INPUT_QSS)


        self.auto_prompt_input.setAttribute(
            Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.auto_prompt_input.setPlaceholderText(
            tr("e.g. building, tree, road, car"))


        self.auto_prompt_input.setMaxLength(200)
        self.auto_prompt_input.setAccessibleName(tr("Describe what to detect"))
        self.auto_prompt_input.setClearButtonEnabled(True)


        try:
            for _clear_act in self.auto_prompt_input.actions():
                _clear_act.setIcon(icon_for(
                    self.auto_prompt_input, "close", 14, token_qcolor(INK_2)))
        except (RuntimeError, AttributeError):
            pass
        self.auto_prompt_input.setMinimumHeight(30)
        self.auto_prompt_composer.watch_focus(self.auto_prompt_input)
        self.auto_prompt_input.textChanged.connect(self._on_auto_search_text_changed)
        self.auto_prompt_input.returnPressed.connect(self._on_auto_search_return_pressed)




        self.auto_prompt_input.editingFinished.connect(
            self._on_auto_prompt_editing_finished)



        self.install_prompt_suggest()
        _composer_col.addWidget(self.auto_prompt_input)



        _chip_row = QHBoxLayout()
        _chip_row.setContentsMargins(0, 0, 0, 0)
        _chip_row.setSpacing(6)
        self.auto_library_btn = QPushButton(tr("Library"))
        self.auto_library_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_library_btn.setToolTip(
            tr("Browse ready-to-use objects with before / after previews."))
        self.auto_library_btn.setStyleSheet(_BTN_AUTO_CHIP)
        self.auto_library_btn.setIcon(icon_for(self.auto_library_btn, "book", 16))
        self.auto_library_btn.setAutoDefault(False)
        self.auto_library_btn.clicked.connect(self.auto_library_requested.emit)
        _chip_row.addWidget(self.auto_library_btn, 0)
        _chip_row.addStretch(1)
        _composer_col.addLayout(_chip_row)
        _prompt_card_layout.addWidget(self.auto_prompt_composer)




        self.auto_prompt_info = QLabel()
        self.auto_prompt_info.setWordWrap(True)
        self.auto_prompt_info.setVisible(False)
        _prompt_card_layout.addWidget(self.auto_prompt_info)







        self.auto_prompt_tip = DismissibleHint(
            HINT_PROMPT_TREE_OR_FOREST,
            tr('Dense forest? "Forest" takes it as one block; '
               '"Tree" picks individual trees.'),
            tint=BLUE_TINT,
            visibility_gate=lambda: False,
        )
        self.auto_prompt_tip.setVisible(False)
        _prompt_card_layout.addWidget(self.auto_prompt_tip)
        self._set_prompt_info()

        _s3_layout.addWidget(self.auto_prompt_card)










        self.auto_input_joiner = NeverShownWidget()
        _s3_layout.addWidget(self.auto_input_joiner)












        self.auto_exemplar_panel = ExampleCardWithSeparator()
        self.auto_exemplar_panel.set_companion(self.auto_input_joiner)
        self.auto_exemplar_panel.setObjectName("autoExemplarCard")
        self.auto_exemplar_panel.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_exemplar_panel.setStyleSheet(
            _CARD_QSS.format(name="autoExemplarCard"))
        _ex_outer = QVBoxLayout(self.auto_exemplar_panel)
        _ex_outer.setContentsMargins(*_CARD_MARGINS)
        _ex_outer.setSpacing(6)



        self._auto_exemplar_expanded = True
        self._auto_exemplar_header = QWidget()
        _ex_hdr_row = QHBoxLayout(self._auto_exemplar_header)
        _ex_hdr_row.setContentsMargins(0, 0, 0, 0)
        _ex_hdr_row.setSpacing(6)





        _ex_title = QLabel(tr("Show what it looks like"))
        _ex_title.setStyleSheet(_FOLD_TITLE_QSS)
        _ex_title.setWordWrap(True)
        _ex_hdr_row.addWidget(_ex_title)
        _ex_hdr_row.addStretch(1)




        self.auto_exemplar_quality_dots = QLabel("")
        self.auto_exemplar_quality_dots.setTextFormat(Qt.TextFormat.RichText)
        self.auto_exemplar_quality_dots.setToolTip(tr(
            "Two references give the strongest detection. Draw a second to "
            "reach best quality."))
        self.auto_exemplar_quality_dots.setStyleSheet(
            "background: transparent; border: none;")
        self.auto_exemplar_quality_dots.setVisible(False)
        _ex_hdr_row.addWidget(self.auto_exemplar_quality_dots)
        _ex_outer.addWidget(self._auto_exemplar_header)


        self.auto_exemplar_content = QWidget()
        _ex_card_col = QVBoxLayout(self.auto_exemplar_content)
        _ex_card_col.setContentsMargins(0, 0, 0, 0)
        _ex_card_col.setSpacing(6)



        self.auto_exemplar_edit_controls = QWidget()
        _ex_edit_col = QVBoxLayout(self.auto_exemplar_edit_controls)
        _ex_edit_col.setContentsMargins(0, 0, 0, 0)
        _ex_edit_col.setSpacing(6)

        self._auto_exemplar_count = 0


















        _ex_inc_style = _btn_toggle_qss(
            _BRAND_BLUE_RGB, "palette(text)", _ON_BRAND_BLUE_INK, weight=600,
            quiet=True)









        _ex_exc_style = _btn_toggle_qss(
            _BRAND_BLUE_RGB, "palette(text)", _ON_BRAND_BLUE_INK, weight=600,
            quiet=True)
        _ex_mode_row = QHBoxLayout()
        _ex_mode_row.setContentsMargins(0, 0, 0, 0)
        _ex_mode_row.setSpacing(8)



        self.auto_ex_inc_btn = QPushButton()
        self.auto_ex_inc_btn.setStyleSheet(_ex_inc_style)
        self.auto_ex_inc_btn.setMinimumHeight(BTN_PILL_PX)
        self.auto_ex_inc_btn.setCursor(Qt.CursorShape.PointingHandCursor)


        self.auto_ex_inc_btn.setIcon(
            icon_for(self.auto_ex_inc_btn, "polygon", 16))
        self.auto_ex_inc_btn.setToolTip(tr("Mark an object to detect more like it."))
        self.auto_ex_inc_btn.clicked.connect(
            lambda: self.auto_add_exemplar_requested.emit(1))
        _ex_mode_row.addWidget(self.auto_ex_inc_btn, 1)
        self.auto_ex_exc_btn = QPushButton()
        self.auto_ex_exc_btn.setStyleSheet(_ex_exc_style)

        self.auto_ex_exc_btn.setMinimumHeight(BTN_PILL_PX)
        self.auto_ex_exc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_ex_exc_btn.setIcon(
            icon_for(self.auto_ex_exc_btn, "thumbs_down", 16))
        self.auto_ex_exc_btn.setToolTip(
            tr("Mark a false positive to drop things like it."))
        self.auto_ex_exc_btn.clicked.connect(
            lambda: self.auto_add_exemplar_requested.emit(0))

        self.auto_ex_exc_btn.setVisible(False)
        _ex_mode_row.addWidget(self.auto_ex_exc_btn, 0)
        self._refresh_exemplar_button_labels()
        _ex_edit_col.addLayout(_ex_mode_row)





        self.auto_exemplar_explainer = DismissibleHint(
            HINT_EXEMPLAR_TIP,
            tr("The AI detects every object that looks like your examples."),
            tint=BLUE_TINT,
        )
        self.auto_exemplar_explainer.set_flat(True)
        _ex_edit_col.addWidget(self.auto_exemplar_explainer)





        self.auto_exemplar_size_warning = QLabel("")
        self.auto_exemplar_size_warning.setWordWrap(True)
        self.auto_exemplar_size_warning.setStyleSheet(_msg_label_qss("warning"))
        self.auto_exemplar_size_warning.setVisible(False)
        _ex_edit_col.addWidget(self.auto_exemplar_size_warning)













        self.auto_exemplar_armed_tip = DismissibleHint(
            HINT_EXEMPLAR_DRAW_BOX,
            tr("Click points around one object, then double-click to close."),
            tint=BLUE_TINT,
            show_glyph=False,
            visibility_gate=lambda: False,
            closable=False,
        )
        self.auto_exemplar_armed_tip.set_flat(True)
        self.auto_exemplar_armed_tip.setVisible(False)
        _ex_edit_col.addWidget(self.auto_exemplar_armed_tip)







        self.auto_exemplar_quality_line = QLabel("")
        self.auto_exemplar_quality_line.setWordWrap(True)
        self.auto_exemplar_quality_line.setVisible(False)
        _ex_edit_col.addWidget(self.auto_exemplar_quality_line)



        self._auto_exemplar_edit_layout = _ex_edit_col
        self._auto_exemplar_upsell_card = None
        _ex_card_col.addWidget(self.auto_exemplar_edit_controls)




        self.auto_exemplar_chips = QWidget()
        self._auto_exemplar_chips_layout = QHBoxLayout(self.auto_exemplar_chips)
        self._auto_exemplar_chips_layout.setContentsMargins(0, 2, 0, 0)
        self._auto_exemplar_chips_layout.setSpacing(6)
        self._auto_exemplar_chips_layout.addStretch()
        _ex_card_col.addWidget(self.auto_exemplar_chips)

        _ex_outer.addWidget(self.auto_exemplar_content)






        self.auto_exemplar_panel.setVisible(False)
        _s3_layout.addWidget(self.auto_exemplar_panel)


















        self.auto_detail_row = QWidget()
        _detail_outer = QVBoxLayout(self.auto_detail_row)
        _detail_outer.setContentsMargins(0, 0, 0, 0)
        _detail_outer.setSpacing(6)






        self._auto_advanced_fold = QWidget()
        self._auto_advanced_fold.setObjectName("autoDetailFoldCard")
        self._auto_advanced_fold.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self._auto_advanced_fold.setStyleSheet(
            _CARD_QSS.format(name="autoDetailFoldCard"))
        _fold_col = QVBoxLayout(self._auto_advanced_fold)
        _fold_col.setContentsMargins(_CARD_MARGINS[0], 4, _CARD_MARGINS[2], 4)
        _fold_col.setSpacing(0)
        self._auto_advanced_open = False
        self.auto_advanced_toggle_btn = QPushButton()






        self.auto_advanced_toggle_btn.setStyleSheet(_FOLD_ROW_QSS)
        self.auto_advanced_toggle_btn.setMinimumHeight(32)
        self.auto_advanced_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self.auto_advanced_toggle_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.auto_advanced_toggle_btn.clicked.connect(
            self._on_auto_advanced_toggle_clicked)
        _fold_col.addWidget(self.auto_advanced_toggle_btn)













        _hdr_row = QHBoxLayout(self.auto_advanced_toggle_btn)
        _hdr_row.setContentsMargins(0, 0, 0, 0)
        _hdr_row.setSpacing(6)
        self.auto_advanced_toggle_title = QLabel("")



        self.auto_advanced_toggle_title.setStyleSheet(_FOLD_TITLE_QSS)


        from qgis.PyQt.QtWidgets import QSizePolicy as _SP
        self.auto_advanced_toggle_title.setSizePolicy(_SP.Policy.Minimum, _SP.Policy.Preferred)
        self.auto_advanced_toggle_title.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        _hdr_row.addWidget(self.auto_advanced_toggle_title, 0)

        self.auto_advanced_toggle_chevron = QLabel("")
        self.auto_advanced_toggle_chevron.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.auto_advanced_toggle_chevron.setStyleSheet(
            "background: transparent; border: none;")

        _hdr_row.addStretch(1)



        self.auto_credit_cost_label = QLabel("")
        self.auto_credit_cost_label.setObjectName("autoHint")
        self.auto_credit_cost_label.setWordWrap(True)
        self.auto_credit_cost_label.setVisible(False)


        _hdr_row.addWidget(self.auto_advanced_toggle_chevron)



        self.auto_advanced_body = QWidget()
        self.auto_advanced_body.setObjectName("autoDetailCard")
        self.auto_advanced_body.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)


        self.auto_advanced_body.setVisible(False)
        _adv_layout = QVBoxLayout(self.auto_advanced_body)
        _adv_layout.setContentsMargins(0, 0, 0, 8)
        _adv_layout.setSpacing(4)
        _fold_col.addWidget(self.auto_advanced_body)
        _detail_outer.addWidget(self._auto_advanced_fold)
        _detail_outer.addWidget(self.auto_credit_cost_label)
        self._refresh_auto_advanced_header()

















        self.auto_detail_sub = QLabel(tr(
            "Finer tiles find smaller objects. The grid shows on the map."))
        self.auto_detail_sub.setWordWrap(True)



        self.auto_detail_sub.setStyleSheet(_HINT_LINE_QSS)
        _adv_layout.addWidget(self.auto_detail_sub)





        self.auto_privacy_line = build_cloud_notice_line()
        _detail_outer.addWidget(self.auto_privacy_line)









        self._auto_rerun_guard_applies = False
        self.auto_rerun_guard_hint = DismissibleHint(
            HINT_RERUN_SAME_SETUP,
            tr("Same setup as your last run - the result will match. "
               "Add an example or change the precision for a different result."),
            tint=BLUE_TINT,
            visibility_gate=self._should_show_rerun_guard,
        )
        self.auto_rerun_guard_hint.set_flat(True)
        self.auto_rerun_guard_hint.setVisible(False)
        _detail_outer.addWidget(self.auto_rerun_guard_hint)






        self.auto_detail_slider_row = QWidget()
        _slider_row = QHBoxLayout(self.auto_detail_slider_row)
        _slider_row.setContentsMargins(0, 0, 0, 0)
        _slider_row.setSpacing(6)
        _coarse_lbl = QLabel(tr("Less"))
        _coarse_lbl.setObjectName("autoHint")
        _slider_row.addWidget(_coarse_lbl)
        self.auto_detail_slider = QSlider(Qt.Orientation.Horizontal)
        self.auto_detail_slider.setRange(1, MAX_DETAIL_LEVEL)
        self.auto_detail_slider.setValue(1)
        self.auto_detail_slider.setPageStep(1)
        self.auto_detail_slider.setSingleStep(1)
        self.auto_detail_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.auto_detail_slider.setTickInterval(1)
        self.auto_detail_slider.setMinimumHeight(26)
        self.auto_detail_slider.setStyleSheet(_SLIDER_QSS)
        self.auto_detail_slider.setToolTip(tr(
            "More precision sweeps your zone in a finer grid, so it catches"
            " smaller objects."))
        self.auto_detail_slider.valueChanged.connect(self._on_auto_detail_changed)
        _slider_row.addWidget(self.auto_detail_slider, 1)
        _fine_lbl = QLabel(tr("More"))
        _fine_lbl.setObjectName("autoHint")
        _slider_row.addWidget(_fine_lbl)
        _adv_layout.addWidget(self.auto_detail_slider_row)




        self.auto_zone_fit_btn = QPushButton(dial_copy(
            "zone.fit_precision_cta", tr("Lower precision to fit")))

        self.auto_zone_fit_btn.setMinimumHeight(BTN_PILL_PX)
        self.auto_zone_fit_btn.setCursor(Qt.CursorShape.PointingHandCursor)



        self.auto_zone_fit_btn.setObjectName("autoZoneFitBtn")
        self.auto_zone_fit_btn.setStyleSheet(_BTN_BLUE_OUTLINE)
        self.auto_zone_fit_btn.setToolTip(tr(
            "Sweeps the same zone in a coarser grid, so it fits in one run."))
        self.auto_zone_fit_btn.clicked.connect(self._on_auto_zone_fit_clicked)
        self.auto_zone_fit_btn.setVisible(False)
        _adv_layout.addWidget(self.auto_zone_fit_btn)



        self.auto_detail_hint = QLabel("")
        self.auto_detail_hint.setWordWrap(True)


        self._auto_zone_km2 = None
        self._auto_km2_exceeded = False


        self._auto_detail_feedback = None
        _adv_layout.addWidget(self.auto_detail_hint)








        self.auto_detail_warning = QWidget()
        self.auto_detail_warning.setObjectName("autoDetailWarning")
        self.auto_detail_warning.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_detail_warning.setStyleSheet(
            _msg_card_qss("autoDetailWarning", "warning"))
        _warn_layout = QHBoxLayout(self.auto_detail_warning)

        _warn_layout.setContentsMargins(*_SUBCARD_MARGINS)
        _warn_layout.setSpacing(8)


        _warn_icon = QLabel()


        _warn_icon.setPixmap(pixmap_for(
            _warn_icon, "warning", 16, token_qcolor(ORANGE_TEXT)))
        _warn_icon.setStyleSheet("background: transparent; border: none;")
        _warn_layout.addWidget(_warn_icon, 0, Qt.AlignmentFlag.AlignTop)


        self.auto_detail_warning_label = QLabel(tr(
            "Each tile covers a lot of ground at this precision. Raise the"
            " precision for sharper detections."))
        self.auto_detail_warning_label.setWordWrap(True)
        self.auto_detail_warning_label.setObjectName("autoText")
        _warn_layout.addWidget(self.auto_detail_warning_label, 1)
        self.auto_detail_warning.setVisible(False)
        _detail_outer.insertWidget(1, self.auto_detail_warning)
        self.auto_detail_row.setVisible(False)





        self._apply_auto_detail_gate(False)



        _s3_layout.addWidget(self.auto_detail_row)


















        self.auto_settings_box = QWidget()
        self.auto_settings_box.setObjectName("autoSettingsBox")
        self.auto_settings_box.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_settings_box.setStyleSheet(
            _CARD_QSS.format(name="autoSettingsBox") + "QLabel { background: transparent; border: none; }"
        )
        _settings_layout = QVBoxLayout(self.auto_settings_box)
        _settings_layout.setContentsMargins(*_CARD_MARGINS)
        _settings_layout.setSpacing(6)

        _settings_layout.addWidget(_micro_header(tr("Detection")))

        _conf_row = QHBoxLayout()
        _conf_label = QLabel(tr("Confidence:"))
        _conf_tip = tr(
            "Minimum confidence to keep a detected object. Lower finds more "
            "objects but may add false positives; raise it for cleaner results "
            "on large, distinct features.")
        _conf_label.setToolTip(_conf_tip)
        self.auto_confidence_spin = QDoubleSpinBox()
        self.auto_confidence_spin.setRange(0.05, 0.95)
        self.auto_confidence_spin.setSingleStep(0.05)
        self.auto_confidence_spin.setDecimals(2)
        self.auto_confidence_spin.setValue(_AUTO_DEFAULT_CONFIDENCE)
        self.auto_confidence_spin.setToolTip(_conf_tip)
        fit_spin_width(self.auto_confidence_spin, 62, 78)
        _conf_row.addWidget(_conf_label)
        _conf_row.addStretch()
        _conf_row.addWidget(self.auto_confidence_spin)
        _settings_layout.addLayout(_conf_row)
        _s3_layout.addWidget(self.auto_settings_box)


        self.auto_settings_box.setVisible(False)



























        self.auto_input_rule_hint = DismissibleHint(
            HINT_INPUT_RULE,
            tr("A name plus an example works best."),
            tint=BLUE_TINT,
        )






        _detect_row = WrappingButtonRow(spacing=6)
        self.auto_detect_btn = FitTextButton(tr("Detect objects"))



        self.auto_detect_btn.setStyleSheet(_BTN_GREEN_STEP)
        self.auto_detect_btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)
        self.auto_detect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_detect_btn.setEnabled(False)
        self.auto_detect_btn.clicked.connect(self.auto_detect_requested.emit)
        _detect_row.add_row_item(self.auto_detect_btn, 1)
        self.auto_exit_btn = QPushButton(tr("Exit"))

        self.auto_exit_btn.setStyleSheet(_BTN_GHOST_WIDE)
        self.auto_exit_btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)
        self.auto_exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_exit_btn.clicked.connect(self.auto_exit_requested.emit)
        _detect_row.add_row_item(self.auto_exit_btn, 0)




        _detect_col = QVBoxLayout()
        _detect_col.setContentsMargins(0, 0, 0, 0)
        _detect_col.setSpacing(4)
        _detect_col.addWidget(self.auto_input_rule_hint)
        _detect_col.addWidget(_detect_row)



        self.auto_detect_row = QWidget()
        self.auto_detect_row.setLayout(_detect_col)
        _s3_layout.addWidget(self.auto_detect_row)









        self.auto_progress_card = QWidget()
        self.auto_progress_card.setObjectName("autoProgressCard")
        self.auto_progress_card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_progress_card.setStyleSheet(
            _CARD_QSS.format(name="autoProgressCard"))
        _prog_col = QVBoxLayout(self.auto_progress_card)
        _prog_col.setContentsMargins(*_CARD_MARGINS)
        _prog_col.setSpacing(8)
        _prog_row1 = QHBoxLayout()
        _prog_row1.setContentsMargins(0, 0, 0, 0)
        _prog_row1.setSpacing(8)
        self.auto_progress_dots = RunDotsGrid()
        _prog_row1.addWidget(self.auto_progress_dots, 0, Qt.AlignmentFlag.AlignVCenter)
        self.auto_progress_count_label = QLabel("")
        self.auto_progress_count_label.setTextFormat(Qt.TextFormat.PlainText)


        self.auto_progress_count_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.auto_progress_count_label.setObjectName("autoRunVerb")
        _prog_row1.addWidget(self.auto_progress_count_label, 1)
        self.auto_progress_clock = RunElapsedClock()
        _prog_row1.addWidget(self.auto_progress_clock, 0, Qt.AlignmentFlag.AlignVCenter)
        self.auto_progress_pct_label = QLabel("")
        self.auto_progress_pct_label.setObjectName("autoStatusFigure")
        self.auto_progress_pct_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        _prog_row1.addWidget(self.auto_progress_pct_label, 0)
        _prog_col.addLayout(_prog_row1)
        self.auto_tile_progress = QProgressBar()
        self.auto_tile_progress.setTextVisible(False)
        self.auto_tile_progress.setStyleSheet(_auto_progress_bar_qss(None))
        _prog_col.addWidget(self.auto_tile_progress)


        self.auto_progress_label = QLabel("")
        self.auto_progress_label.setWordWrap(True)
        self.auto_progress_label.setObjectName("autoHint")
        self.auto_progress_label.setVisible(False)
        _prog_col.addWidget(self.auto_progress_label)





        self.auto_cancel_btn = QPushButton(tr("Cancel detection"))
        self.auto_cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)


        self.auto_cancel_btn.setStyleSheet(_BTN_GHOST)
        self.auto_cancel_btn.setAutoDefault(False)
        self.auto_cancel_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.auto_cancel_btn.setIcon(icon_for(
            self.auto_cancel_btn, "stop", 14, token_qcolor(INK_2)))
        self.auto_cancel_btn.setVisible(False)
        self.auto_cancel_btn.clicked.connect(self._on_auto_cancel_clicked)
        _prog_col.addWidget(self.auto_cancel_btn)
        self.auto_progress_card.setVisible(False)
        _s3_layout.addWidget(self.auto_progress_card)





        self.auto_status_banner = QLabel("")
        self.auto_status_banner.setWordWrap(True)
        self.auto_status_banner.setStyleSheet(_msg_label_qss("info"))
        self.auto_status_banner.setOpenExternalLinks(False)
        self.auto_status_banner.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction)
        self.auto_status_banner.linkActivated.connect(
            self._on_auto_status_link_activated)
        self.auto_status_banner.setVisible(False)
        _s3_layout.addWidget(self.auto_status_banner)










        self.auto_zero_assist_row = QWidget()
        _za_col = QVBoxLayout(self.auto_zero_assist_row)
        _za_col.setContentsMargins(0, 0, 0, 0)
        _za_col.setSpacing(4)
        self.auto_zero_example_chip = QPushButton("")


        self.auto_zero_example_chip.setStyleSheet(_BTN_AUTO_CHIP)
        self.auto_zero_example_chip.setIcon(
            icon_for(self.auto_zero_example_chip, "pencil", 16))
        self.auto_zero_example_chip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_zero_example_chip.clicked.connect(
            lambda: self.auto_zero_assist_clicked.emit("draw_example", ""))
        _za_col.addWidget(self.auto_zero_example_chip)
        self.auto_zero_synonym_chip = QPushButton("")



        self.auto_zero_synonym_chip.setStyleSheet(_BTN_AUTO_CHIP)
        self.auto_zero_synonym_chip.setIcon(
            icon_for(self.auto_zero_synonym_chip, "search", 16))
        self.auto_zero_synonym_chip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_zero_synonym_chip.clicked.connect(
            lambda: self.auto_zero_assist_clicked.emit(
                "synonym", getattr(self, "_auto_zero_synonym", "") or ""))
        _za_col.addWidget(self.auto_zero_synonym_chip)
        self._auto_zero_synonym = ""
        self.auto_zero_assist_row.setVisible(False)
        _s3_layout.addWidget(self.auto_zero_assist_row)






        self.auto_exhausted_subscribe = UpsellCard(
            "autoExhaustedOffer", "compact", self._on_upgrade_clicked)


        self.auto_exhausted_subscribe_link = self.auto_exhausted_subscribe.button
        self.auto_exhausted_subscribe.set_text(
            dial_copy(
                "upsell.exhausted_title",
                tr("Your Automatic allowance ran out mid-zone.")),



            dial_copy(
                "upsell.exhausted_link",
                tr("Pro picks it up where it stopped and finishes the zone.")),
            dial_copy("upsell.exhausted_cta", tr("Finish with Pro")),
        )
        self.auto_exhausted_subscribe.set_pro_offer("plugin_exhausted_offer")
        self.auto_exhausted_subscribe.setVisible(False)
        _s3_layout.addWidget(self.auto_exhausted_subscribe)




        self._setup_auto_review_panel(_s3_layout)





        _s3_layout.addStretch(1)


        _s1_layout.addStretch()
        _s2_layout.addStretch(1)
        _s3_layout.addStretch()




        auto_layout.addWidget(self.auto_controls_section, 1)


        auto_layout.addStretch()


        name_unlabelled_controls(self.auto_page)
        self.auto_page.setVisible(False)
        self.main_layout.addWidget(self.auto_page, 1)

    def _on_auto_upsell_manual_clicked(self) -> None:






        try:
            self._on_mode_selected(Mode.INTERACTIVE)
        except (RuntimeError, AttributeError):
            return

    def get_auto_confidence(self) -> float:





        spin = getattr(self, "auto_confidence_spin", None)
        if spin is None:
            return _AUTO_DEFAULT_CONFIDENCE
        return float(spin.value())

    def get_auto_min_size(self) -> float:

        spin = getattr(self, "auto_min_size_spin", None)
        return float(spin.value()) if spin is not None else 0.0

    def get_auto_max_size(self) -> float:

        spin = getattr(self, "auto_max_size_spin", None)
        return float(spin.value()) if spin is not None else 0.0

    def get_auto_fill_holes_max(self) -> float:





        spin = getattr(self, "auto_fill_max_spin", None)
        if spin is None:
            return _AUTO_REVIEW_FILL_MAX_M2_DEFAULT
        return max(0.0, float(spin.value()))

    def _sync_auto_fill_max_row(self) -> None:


        row = getattr(self, "auto_fill_max_row", None)
        check = getattr(self, "auto_fill_holes_check", None)
        if row is not None and check is not None:
            row.setVisible(check.isChecked())

    def get_auto_points_pct(self) -> int:





        spin = getattr(self, "auto_points_spin", None)
        if spin is None:
            return _AUTO_REVIEW_POINTS_PCT_DEFAULT
        try:
            return int(spin.value())
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return _AUTO_REVIEW_POINTS_PCT_DEFAULT

    def get_auto_refine_params(self) -> tuple[float, bool, int, bool, float, bool]:







        simplify = getattr(self, "auto_simplify_spin", None)
        round_c = getattr(self, "auto_round_corners_check", None)
        expand = getattr(self, "auto_expand_spin", None)
        fill = getattr(self, "auto_fill_holes_check", None)
        clean = getattr(self, "auto_clean_spin", None)
        ortho = getattr(self, "auto_ortho_check", None)
        right_angles = bool(ortho.isChecked()) if ortho is not None else _AUTO_REVIEW_ORTHO_DEFAULT





        return (
            (float(simplify.value()) if simplify is not None
             else _AUTO_REVIEW_SIMPLIFY_DEFAULT),
            (False if right_angles else
             (bool(round_c.isChecked()) if round_c is not None else _AUTO_REVIEW_SMOOTH_DEFAULT)),
            int(expand.value()) if expand is not None else _AUTO_REVIEW_EXPAND_DEFAULT,
            bool(fill.isChecked()) if fill is not None else _AUTO_REVIEW_FILL_HOLES_DEFAULT,
            (0.0 if right_angles else
             (float(clean.value()) if clean is not None else _AUTO_REVIEW_CLEAN_DEFAULT)),
            right_angles,
        )
