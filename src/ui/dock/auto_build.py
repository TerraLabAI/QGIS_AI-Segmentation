"""Automatic page construction (3-step stacked flow) and its value getters.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.activation_manager import has_tos_accepted, has_tos_locked
from ...core.i18n import tr
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
from ..layer_tree_combobox import LayerTreeComboBox
from .cloud_notice_line import build_cloud_notice_line
from .guidance import (
    BLUE_TINT,
    GREEN_TINT,
    HINT_EXEMPLAR_DRAW_BOX,
    HINT_EXEMPLAR_TIP,
    HINT_PROMPT_TREE_OR_FOREST,
    HINT_RERUN_SAME_SETUP,
    HINT_START_AUTO,
    HINT_TUTORIAL_FIRST_STEPS,
    NEUTRAL_TINT,
    DismissibleHint,
    open_guide,
)
from .styles import (
    _BTN_BLUE,
    _BTN_BLUE_PRIMARY,
    _BTN_CHIP,
    _BTN_GHOST,
    _BTN_GREEN,
    _BTN_LINK_MUTED,
    _CARD_CHILD_BTN_RESET_QSS,
    _CARD_JOINED_QSS,
    _CARD_MARGINS,
    _CARD_QSS,
    _CHIP_QSS,
    _INPUT_THEME_QSS,
    _MSG_GLYPHS,
    _PROGRESS_THIN_QSS,
    _SECTION_TOGGLE_QSS,
    _SLIDER_QSS,
    _SUBCARD_MARGINS,
    _btn_start_qss,
    _btn_toggle_qss,
    _micro_header,
    _msg_card_qss,
    _msg_label_qss,
    _step_dial,
)
from .upsell_card import UpsellCard
from .widgets import (
    Mode,
    _ZoneGestureGlyph,
    build_no_imagery_hero,
    make_shortcut_hint,
    native_key,
)


class DockAutoBuildMixin:
    """Automatic page construction (3-step stacked flow) and its value getters."""

    def _setup_automatic_page(self):
        """Build the entire Automatic mode page container (hidden in Interactive mode)."""
        self.auto_page = QWidget()
        auto_layout = QVBoxLayout(self.auto_page)
        auto_layout.setContentsMargins(0, 8, 0, 0)
        auto_layout.setSpacing(8)

        from qgis.PyQt.QtWidgets import QSizePolicy as _QSizePolicy

        # A. Upsell card - shown ONLY once the lifetime free detections are
        # exhausted (non-subscribers). Until then the upsell stays out of the
        # way: a credit ring + Subscribe pill live in the dock footer instead.
        self.auto_upsell_card = QFrame()
        self.auto_upsell_card.setObjectName("autoUpsellCard")
        self.auto_upsell_card.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        # Neutral card, and the offer inside it wears the tint. Flat and blue
        # all through, every line weighed the same and the reader met seven of
        # them at once. Pixel-for-pixel the Semi-Auto credit card
        # (manual_credit_gate.py): a user who runs out in both modes must not
        # have to read two different screens to learn one fact.
        self.auto_upsell_card.setStyleSheet(
            _CARD_QSS.format(name="autoUpsellCard")
            + "QLabel { background: transparent; border: none; }"
            + _CARD_CHILD_BTN_RESET_QSS)
        upsell_layout = QVBoxLayout(self.auto_upsell_card)
        upsell_layout.setContentsMargins(*_SUBCARD_MARGINS)
        upsell_layout.setSpacing(4)

        # The offer IS a card now, the one shape every Pro CTA in the dock
        # wears (upsell_card.py). The star variant, because this wall owns the
        # page. The lines stay served with the shipped ones as fallback: what
        # Pro gives is a commercial fact that can change any week, and a plugin
        # release takes days to reach a user.
        _wall = UpsellCard("autoUpsellOffer", "wall", self._on_upgrade_clicked)
        # Kept as an attribute name: other code reads the button by it, and the
        # upgrade handler tells this surface apart by the sender's identity.
        self.auto_upgrade_btn = _wall.button
        # The title carries the count, filled from the fetched free-detection
        # total by _refresh_auto_upsell_title, which reads this attribute. A
        # number-free line stands until the total is known.
        self._auto_upsell_title = _wall.title
        # When the quota comes back, right under the fact it belongs to.
        # It used to be its own label below the offer, where it read as a
        # second escape route next to "Or click objects one by one".
        self._auto_upsell_reset = _wall.note
        _wall.set_text(
            # The same served id _refresh_auto_upsell_title writes here once
            # the totals land. One sentence, one id: a deploy that rewords the
            # wall must not leave the build-time line saying something else.
            dial_copy(
                "trial.exhausted_no_count",
                tr("Your free detections are used up")),
            # ONE line, and it answers the question this wall raises: the
            # month ran out of surface, so what does Pro give instead. The two
            # lines here before it sold small objects and run history, neither
            # of which is what the reader just hit.
            dial_copy(
                "upsell.wall_body",
                tr("Draw a whole city and let it run, at the finest "
                   "precision.")),
            dial_copy("upsell.cta", tr("Upgrade to Pro")),
            # The price ships in the line and stays served (see the same read
            # in manual_credit_gate.py): served alone it is absent on a cold
            # cache, which is the launch where a buyer first meets this card.
            escape=dial_copy(
                "upsell.cta_hint",
                tr("39 EUR a month, cancel anytime.")),
            star=dial_copy(
                "upsell.bullet_quota",
                tr("300 km² of Automatic every month, on zones of any size")),
        )
        upsell_layout.addWidget(_wall)

        # The free way out, under a hairline and in grey: named so nobody
        # feels stuck, never weighed like the offer above it.
        upsell_layout.addSpacing(6)
        _rule = QFrame()
        _rule.setFrameShape(QFrame.Shape.HLine)
        _rule.setStyleSheet(
            "background: rgba(128,128,128,0.25); border: none; max-height: 1px;")
        upsell_layout.addWidget(_rule)

        _upsell_free = QLabel(dial_copy(
            "upsell.manual_free",
            tr("Or click objects one by one in Semi-Auto.")))
        _upsell_free.setStyleSheet("font-size: 11px; color: rgba(128,128,128,0.95);")
        _upsell_free.setWordWrap(True)
        upsell_layout.addWidget(_upsell_free)

        # The free way out is named right beside the paid one, so its label is
        # served too: the wall reads as a dead end the day the two disagree.
        self.auto_upsell_manual_btn = QPushButton(dial_copy(
            "upsell.manual_cta", tr("Use Semi-Auto")))
        self.auto_upsell_manual_btn.setMinimumHeight(30)
        self.auto_upsell_manual_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_upsell_manual_btn.setStyleSheet(_BTN_CHIP)
        self.auto_upsell_manual_btn.clicked.connect(
            self._on_auto_upsell_manual_clicked)
        upsell_layout.addWidget(self.auto_upsell_manual_btn)

        # The auto page absorbs the panel height (stretch factor below); the
        # card itself must never stretch with it.
        self.auto_upsell_card.setSizePolicy(
            _QSizePolicy.Policy.Preferred, _QSizePolicy.Policy.Maximum)
        auto_layout.addWidget(self.auto_upsell_card)

        # C. Controls section - a 3-step flow. Each step is a page of a
        # QStackedWidget so the user never sees the next step's controls
        # before completing the current one (mirrors AI Edit's paged dock):
        #   step 0  What to detect      (object combo)
        #   step 1  Where to look       (raster combo + draw-zone hero)
        #   step 2  Launch              (cost, Detect, progress, status)
        # Steps advance automatically on completion events (object picked,
        # zone drawn). There is no back arrow: the canvas x badge drops the
        # zone (returning to the zone step) and the breadcrumb summary of
        # the earlier choices is itself clickable to revisit them.
        self.auto_controls_section = QWidget()
        controls_layout = QVBoxLayout(self.auto_controls_section)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        # Persistent layer header: the chosen raster lives above the step
        # stack so it stays visible (greyed + locked) once the user starts,
        # exactly like the Interactive panel. On step 0 it is editable under
        # its label; from step 1 on it is locked (see _refresh_auto_layer_lock).
        self.auto_layer_label = QLabel(tr("Select a raster layer to segment:"))
        self.auto_layer_label.setStyleSheet(
            "font-weight: bold; color: palette(text);")
        controls_layout.addWidget(self.auto_layer_label)

        self.auto_layer_combo = LayerTreeComboBox()
        self.auto_layer_combo.setToolTip(
            tr("Select a raster layer (GeoTIFF, WMS, XYZ tiles, etc.)"))
        self.auto_layer_combo.setStyleSheet("QComboBox { color: palette(text); }")
        from qgis.PyQt.QtWidgets import QSizePolicy
        self.auto_layer_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.auto_layer_combo.setMinimumWidth(0)
        self.auto_layer_combo.layerChanged.connect(self._on_auto_layer_changed)
        controls_layout.addWidget(self.auto_layer_combo)

        # No-rasters state = the first-run hero (lives with the layer header it
        # replaces). The empty canvas is the top first-session dead end: most
        # curiosity installs have no imagery loaded at all. The screen leads
        # with the truth - the imagery is the USER's to bring (any GeoTIFF /
        # WMS / XYZ) - and keeps a one-click demo place as the reassurance
        # fallback for someone with no data on hand. Built
        # via the shared helper so Manual shows the identical card.
        self.auto_no_rasters_widget, self.auto_demo_btn = build_no_imagery_hero(
            on_demo=self.auto_demo_requested.emit,
        )
        self.auto_no_rasters_widget.setVisible(False)
        controls_layout.addWidget(self.auto_no_rasters_widget, 1)

        # Three-step flow below the layer header. Each step is a bare page of
        # a QStackedWidget (no titles, no breadcrumbs): the layer header shows
        # which raster is locked, the canvas x badge re-draws the zone, and the
        # Exit button leaves the flow.
        #   step 0  Start    (blue "Start Automatic Segmentation")
        #   step 1  Zone     (draw-zone hero)
        #   step 2  Prompt   (what to segment + detail + confidence + Detect/Exit)
        self.auto_steps = QStackedWidget()
        controls_layout.addWidget(self.auto_steps, 1)

        def _make_page():
            page = QWidget()
            lay = QVBoxLayout(page)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(8)
            self.auto_steps.addWidget(page)
            return lay

        _s1_layout = _make_page()  # step 0: start
        # Match the Manual page's gap between the layer combo and its Start
        # button: there, 8px layout spacing + an 8px start_container top margin
        # put the button 16px below the combo. The shared _make_page() uses a
        # 0 top margin, which left the Automatic Start button 8px too high, so
        # restore parity by giving step 0 the same 8px top margin.
        _s1_layout.setContentsMargins(0, 8, 0, 0)

        _s2_layout = _make_page()  # step 1: draw zone
        _s3_layout = _make_page()  # step 2: prompt + settings

        # ---- Step 0: Start (mirrors the Interactive start, in Automatic blue) ----
        self.auto_start_btn = QPushButton(tr("Start Automatic AI Segmentation"))
        self.auto_start_btn.setStyleSheet(_btn_start_qss(_BTN_BLUE_PRIMARY))
        self.auto_start_btn.setMinimumHeight(40)
        self.auto_start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_start_btn.setEnabled(False)
        self.auto_start_btn.clicked.connect(self._on_auto_start_clicked)
        _s1_layout.addWidget(self.auto_start_btn)

        # First-steps nudge: a one-time post-sign-in banner pointing
        # new users at the written guide. Built here (with the Automatic Start
        # step) but NOT added to _s1_layout: like the Manual Try-Automatic hint it
        # is pinned to the very BOTTOM of the dock, just above the footer CTAs, via
        # main_layout after the stretch. Its visibility
        # is driven by _update_auto_tutorial_banner_visibility so it only shows on
        # the Automatic Start step, never mid-flow or in Manual mode.
        # QSettings-remembered, so it shows once per user until dismissed.
        self.auto_tutorial_banner = DismissibleHint(
            HINT_TUTORIAL_FIRST_STEPS,
            # Says "tutorial", medium-neutral (the tutorial page has a video
            # too, so no "read"); quiet grey card + small blue button so it
            # never shouts. show_glyph=False: a neutral-tinted card must not
            # carry the blue info lightbulb (that glyph belongs to the info
            # tint only).
            tr("New here? Our 5-minute tutorial walks you through a full "
               "detection, step by step."),
            tint=NEUTRAL_TINT,
            action_text=tr("Open the tutorial"),
            action_color=BLUE_TINT,
            visibility_gate=self._should_show_auto_tutorial,
            show_glyph=False,
        )
        self.auto_tutorial_banner.action.connect(
            lambda: open_guide("post_signin"))

        # "What is this mode for" caption: a quiet framed card
        # under the Start button. One plain sentence; no free/paid wording,
        # no cloud/local wording. Dismissible (small x); only
        # on step 0, so it never shows mid-flow. Re-enable from Account Settings.
        # Green like its Manual sibling: the two mode descriptions are the
        # same kind of information, so they wear the same coat.
        self.auto_start_caption = DismissibleHint(
            HINT_START_AUTO,
            tr("Draw a zone, name one kind of object, and get all of them in "
               "one run. Use Semi-Auto mode to work one object at a time."),
            tint=GREEN_TINT,
            show_glyph=False,  # a mode description, not a tip
        )
        _s1_layout.addWidget(self.auto_start_caption)

        # A last-run recap card lived here and was removed on 2026-08-11. It
        # said "Last run: 69 building in Building 4 (11 Aug) · 14 credits" and
        # sat on the Start page for the rest of the session. The saved layer is
        # in the legend and the balance is on the footer ring, so it repeated
        # two things the user could already see, on the one screen that should
        # be about the next run. The success line below still names the layer
        # right after a Finish, which is the moment that needed an answer.

        # Post-export success line: after Finish the flow returns here and the
        # run status is wiped, so without this the user never learns WHERE the
        # result went. A one-line lime success message naming the layer, set
        # AFTER the reset (which clears it), dismissed on the next Start / mode
        # switch. RichText, because the layer name is the link that frames the
        # result on the map; the name is escaped where the text is built
        # (auto_recap.py), so an & in it still reads as an &.
        self.auto_export_success = QLabel()
        self.auto_export_success.setWordWrap(True)
        self.auto_export_success.setTextFormat(Qt.TextFormat.RichText)
        self.auto_export_success.setOpenExternalLinks(False)
        self.auto_export_success.linkActivated.connect(self._on_auto_recap_link)
        self.auto_export_success.setStyleSheet(_msg_label_qss("success"))
        self.auto_export_success.setVisible(False)
        _s1_layout.addWidget(self.auto_export_success)

        # ---- Step 1: Draw-zone hero (mirrors AI Edit's empty state) ----
        # Drawing arms automatically when the step opens, so the page shows a
        # gesture glyph + title + instruction inviting the drag, not a button.
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
        self._auto_zone_title.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: palette(text);")
        _hero_layout.addWidget(self._auto_zone_title)
        self._auto_zone_hint = QLabel(
            tr("Click on the map to outline the area to scan."))
        self._auto_zone_hint.setWordWrap(True)
        self._auto_zone_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._auto_zone_hint.setStyleSheet(
            "font-size: 12px; color: palette(text);")
        _hero_layout.addWidget(self._auto_zone_hint)
        # The hero stays minimal on purpose: glyph + title + one instruction
        # that pushes a single path (close on the first point). The optional
        # keyboard shortcuts live in a quiet badge pinned at the panel bottom
        # (built after the matched stretches), not stacked here in the center.
        # Center the hero vertically: equal stretch above and below (the
        # matching bottom stretch is added with the other pages' below).
        _s2_layout.addStretch(1)
        _s2_layout.addWidget(self.auto_zone_hero)
        # A compact Exit under the hero so the user always has a way back from
        # the draw step (Ctrl+Z on the empty canvas also leaves). Reuses the
        # same exit path as step 2's Exit (back to Start, layer unlocked).
        _zone_exit_row = QHBoxLayout()
        _zone_exit_row.addStretch()
        self.auto_zone_exit_btn = QPushButton(tr("Exit"))
        self.auto_zone_exit_btn.setStyleSheet(_BTN_GHOST)
        self.auto_zone_exit_btn.setMinimumHeight(30)
        self.auto_zone_exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_zone_exit_btn.clicked.connect(self.auto_exit_requested.emit)
        _zone_exit_row.addWidget(self.auto_zone_exit_btn)
        _zone_exit_row.addStretch()
        _s2_layout.addLayout(_zone_exit_row)

        # ---- Step 2: describe, then show an example, then detail.
        # Three calm cards, one job each, read top to bottom so the user does
        # one thing at a time instead of facing a wall of parameters. All three
        # are numbered (1 describe, 2 example, 3 detail): prompt PLUS example
        # is the model's most accurate mode, so the cards read in that order.
        # Detect enables on the floor: a valid prompt, or one example, or both.
        # A reference image narrows what the model looks for, which is wrong
        # often enough that requiring one cost more runs than it saved.

        # --- Card 1: describe what to find (the text prompt). ---
        self.auto_prompt_card = QWidget()
        self.auto_prompt_card.setObjectName("autoPromptCard")
        self.auto_prompt_card.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_prompt_card.setStyleSheet(_CARD_QSS.format(name="autoPromptCard"))
        _prompt_card_layout = QVBoxLayout(self.auto_prompt_card)
        _prompt_card_layout.setContentsMargins(*_CARD_MARGINS)
        _prompt_card_layout.setSpacing(6)
        # Step 1 header: a filled step dial + bold title (design-system D11
        # ordered-step treatment), read top to bottom as a checklist.
        _prompt_hdr_row = QHBoxLayout()
        _prompt_hdr_row.setContentsMargins(0, 0, 0, 0)
        _prompt_hdr_row.setSpacing(6)
        _prompt_hdr_row.addWidget(_step_dial(1, "active"))
        self._auto_prompt_header = QLabel(tr("Describe what to find"))
        self._auto_prompt_header.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: palette(text);")
        _prompt_hdr_row.addWidget(self._auto_prompt_header)
        _prompt_hdr_row.addStretch(1)
        _prompt_card_layout.addLayout(_prompt_hdr_row)

        # Input row: the prompt box and the Library button side by side, equal
        # heights. The Library is the guided path to a working prompt (curated
        # English tokens with before/after previews), so it sits right where
        # the eye lands instead of a lost ghost button below.
        _prompt_row = QHBoxLayout()
        _prompt_row.setContentsMargins(0, 0, 0, 0)
        _prompt_row.setSpacing(6)
        self.auto_prompt_input = QLineEdit()
        self.auto_prompt_input.setPlaceholderText(tr("e.g. building, tree, road, car"))
        # The backend rejects prompts over 200 chars (422), and the model only
        # reads a short phrase anyway; without a cap a long paste dies server-side.
        self.auto_prompt_input.setMaxLength(200)
        self.auto_prompt_input.setClearButtonEnabled(True)
        self.auto_prompt_input.setStyleSheet(_INPUT_THEME_QSS)
        self.auto_prompt_input.textChanged.connect(self._on_auto_search_text_changed)
        self.auto_prompt_input.returnPressed.connect(self._on_auto_search_return_pressed)
        # Enter / focus-out = the prompt is settled: flush the debounce and
        # commit immediately, so unknown words (which skip the mid-typing
        # debounce commit, see _prompt_plausibly_complete) still seed the
        # detail default and fire their one commit before Detect.
        self.auto_prompt_input.editingFinished.connect(
            self._on_auto_prompt_editing_finished)
        # The drop-down of catalogue objects, in the user's own language. It
        # hangs off this box and opens on the first letter; see
        # auto_prompt_suggest.py for why the rows carry the label alone.
        self.install_prompt_suggest()
        _prompt_row.addWidget(self.auto_prompt_input, 1)
        # The AI Edit prompt-row look: a quiet neutral chip named "Library"
        # (the place, not the content - "Browse objects" read as jargon), so
        # the guided path is there without competing with the input.
        self.auto_library_btn = QPushButton(tr("Library"))
        self.auto_library_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_library_btn.setToolTip(
            tr("Browse ready-to-use objects with before / after previews."))
        self.auto_library_btn.setStyleSheet(_BTN_CHIP)
        self.auto_library_btn.clicked.connect(self.auto_library_requested.emit)
        _prompt_row.addWidget(self.auto_library_btn, 0)
        _prompt_card_layout.addLayout(_prompt_row)

        # Guard-rail message: hidden when the prompt is empty or valid, an amber
        # callout only when the typed prompt is off the rails. No persistent
        # banner (see _set_prompt_info).
        self.auto_prompt_info = QLabel()
        self.auto_prompt_info.setWordWrap(True)
        self.auto_prompt_info.setVisible(False)
        _prompt_card_layout.addWidget(self.auto_prompt_info)
        # The advisory half of the same line: every closable tip about the
        # prompt (the tree-versus-forest heads-up, one object per run, the swap
        # note, the example nudges, the server plan hint) is written here, one
        # at a time, with its own hint id. One widget, so a tip and the guard
        # can never show together. The gate keeps a guidance reset from
        # flashing a stale tip back onto a step it does not belong to: the next
        # message writes this line itself.
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

        # --- Example card, step 2 of the default path (prompt + example is
        # the model's most accurate mode, so it is numbered like its siblings,
        # no longer marked Optional). The title stays a plain noun ("Add an
        # example") and the button inside keeps the map verb ("Draw on the
        # map"), so no two lines repeat each other. The explainer under the
        # header says why/how; it yields to the armed instruction or the drawn
        # thumbnails. Gated behind _EXEMPLARS_ENABLED. Skipping it costs
        # nothing: Detect is green on the prompt alone.
        self.auto_exemplar_panel = QWidget()
        self.auto_exemplar_panel.setObjectName("autoExemplarCard")
        self.auto_exemplar_panel.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_exemplar_panel.setStyleSheet(
            _CARD_QSS.format(name="autoExemplarCard"))
        _ex_outer = QVBoxLayout(self.auto_exemplar_panel)
        _ex_outer.setContentsMargins(*_CARD_MARGINS)
        _ex_outer.setSpacing(6)

        # Header row: step dial + bold title, the same ordered-step treatment
        # as the describe and detail cards. Wrapped in one widget so the
        # in-run read-only swap can hide the whole header at once.
        self._auto_exemplar_expanded = True
        self._auto_exemplar_header = QWidget()
        _ex_hdr_row = QHBoxLayout(self._auto_exemplar_header)
        _ex_hdr_row.setContentsMargins(0, 0, 0, 0)
        _ex_hdr_row.setSpacing(6)
        _ex_hdr_row.addWidget(_step_dial(2, "active"))
        # "Show what it looks like" pairs with step 1's "Describe what to
        # find" (words, then visuals) and says the PURPOSE - point the AI at
        # a real instance - where "Add an example" read as one abstract
        # attachment. The tip below carries the plural (up to 3).
        _ex_title = QLabel(tr("Show what it looks like"))
        _ex_title.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: palette(text);")
        _ex_hdr_row.addWidget(_ex_title)
        # Marked optional right on the title, because the step number beside it
        # says the opposite. Detect needs the word above and nothing here (see
        # core/detect_gate.can_detect), and a user who reads step 2 as a thing
        # they owe stops on a card they could have walked past. Quiet weight:
        # it qualifies the title, it is not a second title.
        _ex_optional = QLabel(tr("(optional)"))
        _ex_optional.setStyleSheet(
            "font-size: 11px; color: rgba(128, 128, 128, 0.95);")
        _ex_hdr_row.addWidget(_ex_optional)
        _ex_hdr_row.addStretch(1)
        # Quality dots: two small dots that fill lime as positive examples are
        # drawn, so the "aim for two" goal (the model's strongest mode) reads
        # at a glance without a sentence. Hidden until the first example, then
        # driven by _set_exemplar_quality. Right-aligned in the header.
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

        # Card content (editing controls + thumbnails), always visible; the
        # container survives so the in-run read-only swap keeps working.
        self.auto_exemplar_content = QWidget()
        _ex_card_col = QVBoxLayout(self.auto_exemplar_content)
        _ex_card_col.setContentsMargins(0, 0, 0, 0)
        _ex_card_col.setSpacing(6)

        # Read-only caption, shown during a run: the reference stays on
        # screen (browsable) but every editing affordance is gone.
        self.auto_exemplar_readonly_caption = QLabel(tr("Your reference"))
        self.auto_exemplar_readonly_caption.setStyleSheet(
            "font-size: 11px; color: palette(text);")
        self.auto_exemplar_readonly_caption.setVisible(False)
        _ex_card_col.addWidget(self.auto_exemplar_readonly_caption)
        # All the editing controls (draw/exclude buttons + armed line) live in
        # one container so a single toggle removes them for the read-only
        # in-run variant, leaving just the reference thumbnails.
        self.auto_exemplar_edit_controls = QWidget()
        _ex_edit_col = QVBoxLayout(self.auto_exemplar_edit_controls)
        _ex_edit_col.setContentsMargins(0, 0, 0, 0)
        _ex_edit_col.setSpacing(6)

        self._auto_exemplar_count = 0

        # The draw-example button IS the action button: clicking it arms the draw
        # tool directly (no separate "Draw" step). It is big, full-width and
        # coloured so it reads as the action of this step.
        # The armed state is driven by the plugin via set_auto_exemplar_armed, so
        # a cancel (Escape) or a finished draw both clear it. The [armed] dynamic
        # property toggles the look without rebuilding the stylesheet.
        # The OUTLINED variant, not the filled one: filled is already solid
        # green at rest, so arming only darkened it and users read the button as
        # green before and green after. Outlined at rest, solid fill while
        # drawing, which is the one distinction the toggle generator exists for.
        _ex_inc_style = _btn_toggle_qss(
            (67, 160, 71), "palette(text)", "#06210b")
        # The exclude button is the red counterpart: it drops false positives
        # by pointing at a look-alike the model should NOT return. It is a bonus
        # refinement, unlocked ONLY once two positive examples exist (a single
        # reference is too weak to refine, and reference-image detection needs a
        # pair to work well): it starts HIDDEN and set_exemplars reveals it at
        # two positives. Quiet even then, so the primary flow stays one green
        # button.
        _ex_exc_style = _btn_toggle_qss(
            (229, 57, 53), "#e57373", "#2a0606", weight=600, quiet=True)
        _ex_mode_row = QHBoxLayout()
        _ex_mode_row.setContentsMargins(0, 0, 0, 0)
        _ex_mode_row.setSpacing(8)
        # Both labels follow the state (ready / drawing now / one example
        # already drawn): _refresh_exemplar_button_labels owns every wording,
        # and is called once below to seed them.
        self.auto_ex_inc_btn = QPushButton()
        self.auto_ex_inc_btn.setStyleSheet(_ex_inc_style)
        self.auto_ex_inc_btn.setMinimumHeight(28)
        self.auto_ex_inc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_ex_inc_btn.setToolTip(tr("Mark an object to find more like it."))
        self.auto_ex_inc_btn.clicked.connect(
            lambda: self.auto_add_exemplar_requested.emit(1))
        _ex_mode_row.addWidget(self.auto_ex_inc_btn, 1)
        self.auto_ex_exc_btn = QPushButton()
        self.auto_ex_exc_btn.setStyleSheet(_ex_exc_style)
        self.auto_ex_exc_btn.setMinimumHeight(28)
        self.auto_ex_exc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_ex_exc_btn.setToolTip(
            tr("Mark a false positive to drop things like it."))
        self.auto_ex_exc_btn.clicked.connect(
            lambda: self.auto_add_exemplar_requested.emit(0))
        # Hidden until two positive examples exist (set_exemplars reveals it).
        self.auto_ex_exc_btn.setVisible(False)
        _ex_mode_row.addWidget(self.auto_ex_exc_btn, 0)
        self._refresh_exemplar_button_labels()
        _ex_edit_col.addLayout(_ex_mode_row)

        # One short blue tip UNDER the button (dismissible with the tiny x,
        # like every other blue hint): what an example buys, in one line. It
        # also yields to the armed instruction or the drawn thumbnails (see
        # _refresh_auto_exemplar_explainer).
        self.auto_exemplar_explainer = DismissibleHint(
            HINT_EXEMPLAR_TIP,
            tr("The AI finds every object that looks like your examples - "
               "you can draw up to 3."),
            tint=BLUE_TINT,
        )
        _ex_edit_col.addWidget(self.auto_exemplar_explainer)

        # The too-small warning line: an amber callout when a drawn example
        # renders below a usable pixel size. Never dismissible, it says why a
        # run will miss things. Kept separate from the armed instruction below
        # so one of the two can close and the other cannot.
        self.auto_exemplar_size_warning = QLabel("")
        self.auto_exemplar_size_warning.setWordWrap(True)
        self.auto_exemplar_size_warning.setStyleSheet(_msg_label_qss("warning"))
        self.auto_exemplar_size_warning.setVisible(False)
        _ex_edit_col.addWidget(self.auto_exemplar_size_warning)

        # Armed instruction: hidden until a button arms the draw tool, then a
        # denser blue callout giving the gesture (click points, double-click to
        # close). The button says the tool is live; this line says how to trace.
        # This is the "in-between" feedback that the click started a draw
        # action. Its two wordings, example and exclude, carry one hint id
        # each. Same gate reason as the prompt tip: the next armed draw writes
        # this line.
        #
        # No close button. It is the ONLY statement of the gesture anywhere on
        # screen, so closing it once left every later arming explaining
        # nothing, the same reason the Correct step's resting prompt is not
        # dismissible either.
        self.auto_exemplar_armed_tip = DismissibleHint(
            HINT_EXEMPLAR_DRAW_BOX,
            tr("Click points around one object, then double-click to close."),
            tint=BLUE_TINT,
            show_glyph=False,  # the armed glyph, not the info lightbulb
            visibility_gate=lambda: False,
            closable=False,
        )
        self.auto_exemplar_armed_tip.setVisible(False)
        _ex_edit_col.addWidget(self.auto_exemplar_armed_tip)

        # Quality line: the subtle push toward the recommended pair. One
        # example -> a quiet muted nudge to add a second; two or more -> a
        # calm lime confirmation that the best-quality setup is reached. Plain
        # tinted text (no boxed callout) so it stays minimal, and it yields
        # while a draw is armed or the size warning shows (one message at a
        # time). Driven by _set_exemplar_quality.
        self.auto_exemplar_quality_line = QLabel("")
        self.auto_exemplar_quality_line.setWordWrap(True)
        self.auto_exemplar_quality_line.setVisible(False)
        _ex_edit_col.addWidget(self.auto_exemplar_quality_line)
        _ex_card_col.addWidget(self.auto_exemplar_edit_controls)

        # Reference thumbnail strip: one card per drawn example (AI-Edit
        # _ThumbWidget look - thumbnail + numbered badge + hover-x), rebuilt by
        # set_exemplars().
        self.auto_exemplar_chips = QWidget()
        self._auto_exemplar_chips_layout = QHBoxLayout(self.auto_exemplar_chips)
        self._auto_exemplar_chips_layout.setContentsMargins(0, 2, 0, 0)
        self._auto_exemplar_chips_layout.setSpacing(6)
        self._auto_exemplar_chips_layout.addStretch()
        _ex_card_col.addWidget(self.auto_exemplar_chips)

        _ex_outer.addWidget(self.auto_exemplar_content)

        # The exemplar-only count-vs-map policy is no longer asked up front: an
        # empty-prompt run streams as continuous cover and the client decides
        # count-vs-map automatically from the run's own masks at the end, with a
        # one-click override offered in the post-run review (see the review panel).

        self.auto_exemplar_panel.setVisible(False)
        _s3_layout.addWidget(self.auto_exemplar_panel)

        # 5a. Advanced settings: the Precision control, folded shut by default.
        #
        # Precision is the one control here that most runs never need. The
        # level is seeded from the object the user named and the band is
        # already cut to what that object can use, so the slider's job is to
        # let someone overrule a good default, not to be answered.
        #
        # The fold owns the tile grid with it. Open, the canvas shows how the
        # zone is cut; shut, it does not. The grid is the heaviest thing this
        # screen draws over the user's imagery, and a user who has not asked
        # how the zone is split reads it as damage rather than as a plan. So
        # the two travel together: one click gives the slider AND the picture
        # of what it does (see AutoZoneMixin._tile_grid_revealed).
        #
        # What never folds: the zone's surface (what the run is billed on),
        # the monthly-envelope wall, the cloud disclosure and the re-run guard.
        # Money and disclosure do not hide behind a click.
        self.auto_detail_row = QWidget()
        _detail_outer = QVBoxLayout(self.auto_detail_row)
        _detail_outer.setContentsMargins(0, 0, 0, 0)
        _detail_outer.setSpacing(6)

        # Head over its joined body: the shared collapsible pattern (the Manual
        # refine panel and the review's Shape settings wear the same one).
        # Spacing 0, so open the pair draws as a single box.
        self._auto_advanced_fold = QWidget()
        _fold_col = QVBoxLayout(self._auto_advanced_fold)
        _fold_col.setContentsMargins(0, 0, 0, 0)
        _fold_col.setSpacing(0)
        self._auto_advanced_open = False
        self.auto_advanced_toggle_btn = QPushButton()
        self.auto_advanced_toggle_btn.setStyleSheet(_SECTION_TOGGLE_QSS)
        self.auto_advanced_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # Never steal focus from the prompt field on a toggle.
        self.auto_advanced_toggle_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.auto_advanced_toggle_btn.clicked.connect(
            self._on_auto_advanced_toggle_clicked)
        _fold_col.addWidget(self.auto_advanced_toggle_btn)
        # The zone's surface rides at the right end of the head, inside the
        # button. It is the figure the run is billed on, so it has to read with
        # the fold shut, and a label set beside the button would cut the head
        # short of the body it opens. Transparent to the mouse, so a click
        # anywhere along the row still flips the fold. Filled by
        # set_auto_zone_surface; same widget name as the old cost line, so
        # every existing visibility rule keeps working.
        #
        # The chevron and title live in their own label, laid out beside the
        # surface figure rather than painted by the button itself: a button's
        # own text ignores the child layout's geometry, so on a narrow dock
        # the two can draw on top of each other. A label in the same row
        # cannot, because the layout gives each its own rect.
        _hdr_row = QHBoxLayout(self.auto_advanced_toggle_btn)
        _hdr_row.setContentsMargins(10, 0, 10, 0)
        _hdr_row.setSpacing(6)
        self.auto_advanced_toggle_title = QLabel("")
        self.auto_advanced_toggle_title.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.auto_advanced_toggle_title.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: palette(text);"
            " background: transparent; border: none;")
        _hdr_row.addWidget(self.auto_advanced_toggle_title)
        _hdr_row.addStretch()
        self.auto_credit_cost_label = QLabel("")
        self.auto_credit_cost_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.auto_credit_cost_label.setStyleSheet(
            "font-size: 11px; color: palette(text); background: transparent;"
            " border: none;")
        self.auto_credit_cost_label.setVisible(False)
        _hdr_row.addWidget(self.auto_credit_cost_label)

        # The body the head opens. Joined card: no top edge and square top
        # corners, so head and body read as one box rather than two.
        self.auto_advanced_body = QWidget()
        self.auto_advanced_body.setObjectName("autoDetailCard")
        self.auto_advanced_body.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_advanced_body.setStyleSheet(
            _CARD_JOINED_QSS.format(name="autoDetailCard")
            + "QLabel { background: transparent; border: none; }")
        self.auto_advanced_body.setVisible(False)
        _adv_layout = QVBoxLayout(self.auto_advanced_body)
        _adv_layout.setContentsMargins(*_CARD_MARGINS)
        _adv_layout.setSpacing(4)
        _fold_col.addWidget(self.auto_advanced_body)
        _detail_outer.addWidget(self._auto_advanced_fold)
        self._refresh_auto_advanced_header()

        # Inside the fold: the control's own name. No step dial on it any more.
        # A numbered step is a thing the user has to do, and this one is
        # optional now.
        _detail_lbl = QLabel(tr("Precision"))
        _detail_lbl.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: palette(text);")
        _adv_layout.addWidget(_detail_lbl)
        # Right under the surface it contradicts: the drawn zone is bigger than
        # the km² the account has left this month. The same offer card as every
        # other Pro CTA in the dock (upsell_card.py), in its "full" variant,
        # and it greys Detect the same way it always did. Driven by
        # set_auto_km2_block; hidden whenever the account did not tell us both
        # figures, so an unknown envelope never refuses a run.
        self.auto_km2_block = UpsellCard(
            "autoKm2Block", "full", self._on_upgrade_clicked)
        # The button keeps its old attribute name: the upgrade handler tells
        # this surface apart by the sender's identity.
        self.auto_km2_block_upgrade = self.auto_km2_block.button
        self.auto_km2_block.setVisible(False)
        _detail_outer.addWidget(self.auto_km2_block)
        # Always-on subtitle under the title: what the control does, once, in
        # the muted-hint style. It sits ABOVE the slider so it never stacks with
        # the state hint under it (_refresh_auto_detail_hint), which says what
        # THIS level does to THIS object.
        # Kept on self: an object whose useful band holds a single level has no
        # choice to offer, so set_auto_detail_range hides this line with the
        # slider rather than leaving a promise about a control that is gone.
        self.auto_detail_sub = QLabel(tr(
            "More precision finds smaller objects."))
        self.auto_detail_sub.setWordWrap(True)
        self.auto_detail_sub.setStyleSheet(
            "font-size: 11px; color: rgba(128, 128, 128, 0.95);")
        _adv_layout.addWidget(self.auto_detail_sub)
        # Outside the fold, and it stays there. This is the last screen
        # before Detect sends anything, so the disclosure has to be on it
        # whether or not the user opened a settings panel. Word for word the
        # Semi-Auto engine card's line: same data, same destination, same
        # sentence. Retires itself once a run has completed.
        self.auto_privacy_line = build_cloud_notice_line()
        _detail_outer.addWidget(self.auto_privacy_line)
        # Non-blocking tip shown right under the credit estimate when the next
        # Detect would repeat the last run exactly (same prompt, detail and
        # example count): that re-run returns the same masks and only spends
        # credits. Points at the two levers that change the output. Never
        # intercepts Detect; clears itself the moment any input changes. Driven
        # by show/hide_auto_rerun_guard. Dismissible like every other blue tip:
        # the x closes it for good, Account Settings brings it back. The gate
        # tracks whether the condition still holds, so restoring the tips never
        # reveals it over a setup that changed since.
        self._auto_rerun_guard_applies = False
        self.auto_rerun_guard_hint = DismissibleHint(
            HINT_RERUN_SAME_SETUP,
            tr("Same setup as your last run - the result will match. "
               "Add an example or change the precision for a different result."),
            tint=BLUE_TINT,
            visibility_gate=self._should_show_rerun_guard,
        )
        self.auto_rerun_guard_hint.setVisible(False)
        _detail_outer.addWidget(self.auto_rerun_guard_hint)
        # Slider row: plain "Less <-> More" ends (paired with the "Precision" title
        # above) replace the abstract grid numbers, and read simpler than the old
        # Coarse/Fine. The slider still drives the tile subdivision under the hood.
        # The row lives in its own widget so the whole control can be hidden in
        # one call: a band with a single useful level is not a slider the user
        # should be dragging.
        self.auto_detail_slider_row = QWidget()
        _slider_row = QHBoxLayout(self.auto_detail_slider_row)
        _slider_row.setContentsMargins(0, 0, 0, 0)
        _slider_row.setSpacing(6)
        _coarse_lbl = QLabel(tr("Less"))
        _coarse_lbl.setStyleSheet("font-size: 10px; color: palette(text);")
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
        _fine_lbl.setStyleSheet("font-size: 10px; color: palette(text);")
        _slider_row.addWidget(_fine_lbl)
        _adv_layout.addWidget(self.auto_detail_slider_row)
        # One-line plain-language hint instead of a m/px figure. Starts on the
        # gated wording (slider disabled above); _apply_auto_detail_gate swaps
        # it once a prompt or an example exists.
        self.auto_detail_hint = QLabel(
            tr("Name the object (or draw an example) first - Precision "
               "then tunes itself to it."))
        self.auto_detail_hint.setWordWrap(True)
        self.auto_detail_hint.setStyleSheet(
            "font-size: 10px; color: palette(text);")
        # The drawn zone's surface (km²) and whether it is over what the
        # account has left this month. See set_auto_zone_surface.
        self._auto_zone_km2 = None
        self._auto_km2_exceeded = False
        # Object-aware slider verdict (state, object word), pushed by the
        # plugin from the credit-estimate chokepoint; None until known.
        self._auto_detail_feedback = None
        self.auto_detail_hint.linkActivated.connect(
            self._on_detail_cap_upgrade_link)
        _adv_layout.addWidget(self.auto_detail_hint)

        # Conditional amber warning, shown by set_auto_detail_gsd_warning when
        # the chosen precision leaves the imagery too coarse for the cloud model. A proper
        # boxed alert with a warning icon (mirrors the no-rasters warning) so it
        # reads as a real callout, not recoloured hint text. Hidden by default;
        # the neutral hint hides while it shows so guidance never stacks.
        # Lives outside the fold body: it has to read even with Advanced
        # settings collapsed, since that is the state most runs are in.
        self.auto_detail_warning = QWidget()
        self.auto_detail_warning.setObjectName("autoDetailWarning")
        self.auto_detail_warning.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_detail_warning.setStyleSheet(
            _msg_card_qss("autoDetailWarning", "warning"))
        _warn_layout = QHBoxLayout(self.auto_detail_warning)
        _warn_layout.setContentsMargins(8, 6, 8, 6)
        _warn_layout.setSpacing(8)
        # Monochrome text glyph, tinted by the label color (never the
        # colored system icon; the taxonomy glyphs stay black and white).
        _warn_icon = QLabel(_MSG_GLYPHS["warning"])
        _warn_icon.setStyleSheet("font-size: 12px;")
        _warn_layout.addWidget(_warn_icon, 0, Qt.AlignmentFlag.AlignTop)
        self.auto_detail_warning_label = QLabel(tr(
            "This area is large for this precision. Raise the precision or zoom"
            " in for sharper detections."))
        self.auto_detail_warning_label.setWordWrap(True)
        self.auto_detail_warning_label.setStyleSheet("font-size: 11px;")
        _warn_layout.addWidget(self.auto_detail_warning_label, 1)
        self.auto_detail_warning.setVisible(False)
        _detail_outer.insertWidget(1, self.auto_detail_warning)
        self.auto_detail_row.setVisible(False)
        # Gated (whole card disabled + dimmed) until the object is defined
        # (typed prompt or drawn example): the default is object-aware, so an
        # adjustment made before naming the object got thrown away by the
        # prompt-commit re-seed. See _apply_auto_detail_gate (driven from
        # _update_auto_detect_enabled).
        self._apply_auto_detail_gate(False)

        # The locked layer header above the stack already names the raster the
        # run reads, so no separate recap label is needed on this step.
        _s3_layout.addWidget(self.auto_detail_row)

        # 5c. Detection settings box (the confidence dial). Confidence is now a
        # POST-run control only: it appears in the review after detection, never
        # before Detect (where it read as a knob the user had to set up front).
        # The box is still built so the spin holds the default cutoff the run
        # starts from, but it is kept hidden in the prompt step.
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
        self.auto_confidence_spin.setMinimumWidth(62)
        self.auto_confidence_spin.setMaximumWidth(78)
        _conf_row.addWidget(_conf_label)
        _conf_row.addStretch()
        _conf_row.addWidget(self.auto_confidence_spin)
        _settings_layout.addLayout(_conf_row)
        _s3_layout.addWidget(self.auto_settings_box)
        # Confidence is a post-run control only: never show this box before
        # Detect. The spin keeps the default cutoff the run starts from.
        self.auto_settings_box.setVisible(False)

        # Terms + Privacy consent, right above Detect: the ONE moment the user
        # is about to spend a detection, so the friction sits as late as
        # possible. Same
        # GLOBAL state as the Manual checkbox (has_tos_accepted / lock_tos):
        # accepting in one mode reflects in the other, and the row disappears
        # forever once consent is sealed by the first Detect here or the first
        # Manual Start.
        _tos_terms_url = (
            "https://terra-lab.ai/terms-of-use"
            "?utm_source=qgis&utm_medium=plugin"
            "&utm_campaign=ai-segmentation&utm_content=consent_terms_auto"
        )
        _tos_privacy_url = (
            "https://terra-lab.ai/privacy-policy"
            "?utm_source=qgis&utm_medium=plugin"
            "&utm_campaign=ai-segmentation&utm_content=consent_privacy_auto"
        )
        self.auto_tos_container = QWidget()
        _auto_tos_row = QHBoxLayout(self.auto_tos_container)
        _auto_tos_row.setContentsMargins(0, 0, 0, 0)
        _auto_tos_row.setSpacing(4)
        self.auto_tos_checkbox = QCheckBox()
        self.auto_tos_checkbox.setChecked(has_tos_accepted())
        self.auto_tos_checkbox.toggled.connect(self._on_tos_toggled)
        _auto_tos_row.addWidget(self.auto_tos_checkbox, 0)
        self.auto_tos_label = QLabel(
            tr('I agree to the <a href="{terms}">Terms</a> '
               'and <a href="{privacy}">Privacy Policy</a>').format(
                terms=_tos_terms_url, privacy=_tos_privacy_url
            )
        )
        self.auto_tos_label.setOpenExternalLinks(True)
        self.auto_tos_label.setWordWrap(True)
        self.auto_tos_label.setStyleSheet("font-size: 11px; color: palette(text);")
        _auto_tos_row.addWidget(self.auto_tos_label, 1)
        if has_tos_locked():
            self.auto_tos_container.setVisible(False)
        _s3_layout.addWidget(self.auto_tos_container)

        # No balance callout above Detect any more. It refused a run when the
        # zone's TILE count passed the wallet, which is a unit Automatic stopped
        # billing in: precision moves tiles and never moves the price, so it
        # blocked zones that fitted the month easily and stacked a second
        # Upgrade button under the surface wall that was already saying it.
        # The monthly-surface wall in the Advanced settings card is the one
        # refusal, and it speaks km2 (see set_auto_zone_surface).

        # 6. Detect + Exit row (mirrors AI Edit's Generate + Exit): the green
        # primary grows, the ghost Exit stays compact beside it. Exit leaves
        # the whole flow (back to the Start step, layer unlocked).
        _detect_row = QHBoxLayout()
        _detect_row.setContentsMargins(0, 0, 0, 0)
        _detect_row.setSpacing(6)
        self.auto_detect_btn = QPushButton(tr("Detect objects"))
        self.auto_detect_btn.setStyleSheet(_BTN_GREEN)
        self.auto_detect_btn.setMinimumHeight(36)
        self.auto_detect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_detect_btn.setEnabled(False)
        self.auto_detect_btn.clicked.connect(self.auto_detect_requested.emit)
        _detect_row.addWidget(self.auto_detect_btn, 1)
        self.auto_exit_btn = QPushButton(tr("Exit"))
        self.auto_exit_btn.setStyleSheet(_BTN_GHOST)
        self.auto_exit_btn.setMinimumHeight(36)
        self.auto_exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_exit_btn.clicked.connect(self.auto_exit_requested.emit)
        _detect_row.addWidget(self.auto_exit_btn, 0)
        # The prompt page stays uncluttered: no keyboard legend here (the
        # Detect/Exit buttons speak for themselves). auto_detect_row remains a
        # QWidget so the existing show/hide (run active, review) still works.
        self.auto_detect_row = QWidget()
        self.auto_detect_row.setLayout(_detect_row)
        _s3_layout.addWidget(self.auto_detect_row)

        # 9. Progress card: an information-rich framed card (same card family as
        # the step cards) so a long tiled run always shows real movement - tile
        # count, live found count and percent - instead of a bare bar that reads
        # as dead. The prompt card + reference stay visible above it, so the user
        # keeps full context of what is being detected. Never timer-animated:
        # only real state changes repaint it.
        self.auto_progress_card = QWidget()
        self.auto_progress_card.setObjectName("autoProgressCard")
        self.auto_progress_card.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_progress_card.setStyleSheet(
            _CARD_QSS.format(name="autoProgressCard"))
        _prog_col = QVBoxLayout(self.auto_progress_card)
        _prog_col.setContentsMargins(*_CARD_MARGINS)
        _prog_col.setSpacing(6)
        # Row 1: tile count (+ live found count) on the left, percent right.
        _prog_row1 = QHBoxLayout()
        _prog_row1.setContentsMargins(0, 0, 0, 0)
        _prog_row1.setSpacing(6)
        self.auto_progress_count_label = QLabel("")
        self.auto_progress_count_label.setTextFormat(Qt.TextFormat.RichText)
        self.auto_progress_count_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: palette(text);"
            " background: transparent;")
        _prog_row1.addWidget(self.auto_progress_count_label, 1)
        self.auto_progress_pct_label = QLabel("")
        self.auto_progress_pct_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: palette(text);"
            " background: transparent;")
        self.auto_progress_pct_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        _prog_row1.addWidget(self.auto_progress_pct_label, 0)
        _prog_col.addLayout(_prog_row1)
        # Row 2: a thin instrument progress line (3px, brand blue on a faint
        # track); the measured status text lives in the labels beside it.
        self.auto_tile_progress = QProgressBar()
        self.auto_tile_progress.setTextVisible(False)
        self.auto_tile_progress.setStyleSheet(_PROGRESS_THIN_QSS)
        _prog_col.addWidget(self.auto_tile_progress)
        # Row 3 (conditional): the queue / cold-start status line (Sending to
        # the AI…, spot reserved / ETA). Hidden while tiles flow normally.
        self.auto_progress_label = QLabel("")
        self.auto_progress_label.setWordWrap(True)
        self.auto_progress_label.setStyleSheet(
            "font-size: 11px; color: palette(text); background: transparent;")
        self.auto_progress_label.setVisible(False)
        _prog_col.addWidget(self.auto_progress_label)
        self.auto_progress_card.setVisible(False)
        _s3_layout.addWidget(self.auto_progress_card)

        # 10. Cancel detection: a quiet centered text link, not a full-width
        # red button. The run is paid for and usually worth finishing, so the
        # escape hatch stays discoverable without inviting a click (AI Edit
        # hides cancel entirely; long tiled runs still need one).
        self.auto_cancel_btn = QPushButton(tr("Cancel detection"))
        self.auto_cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_cancel_btn.setStyleSheet(_BTN_LINK_MUTED)
        self.auto_cancel_btn.setVisible(False)
        self.auto_cancel_btn.clicked.connect(self._on_auto_cancel_clicked)
        _cancel_row = QHBoxLayout()
        _cancel_row.setContentsMargins(0, 0, 0, 0)
        _cancel_row.addStretch(1)
        _cancel_row.addWidget(self.auto_cancel_btn, 0)
        _cancel_row.addStretch(1)
        _s3_layout.addLayout(_cancel_row)

        # 11. Status banner. Wired for links once here: a terminal-error status
        # can carry a persistent "Report this problem" link (set_auto_status with
        # a report_payload renders RichText); the sentinel href is intercepted,
        # never opened as a URL.
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

        # 11a. Zero-result rescue, right under the status banner. A paid run
        # that found nothing is the worst moment of the flow, and the drawn
        # example is the proven lever that rescues it (runs with an example come
        # back empty far less often), so it leads as a full-width FILLED-blue
        # call, a primary, with the object named in its label. The synonym
        # prefill stays a quiet chip below it, only when the server steer
        # table knows a stronger word. Hidden by default; driven by
        # show/hide_auto_zero_assist. The row never outlives its status:
        # set_auto_status hides it on every call.
        self.auto_zero_assist_row = QWidget()
        _za_col = QVBoxLayout(self.auto_zero_assist_row)
        _za_col.setContentsMargins(0, 0, 0, 0)
        _za_col.setSpacing(4)
        self.auto_zero_example_chip = QPushButton("")
        self.auto_zero_example_chip.setStyleSheet(_BTN_BLUE)
        self.auto_zero_example_chip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_zero_example_chip.clicked.connect(
            lambda: self.auto_zero_assist_clicked.emit("draw_example", ""))
        _za_col.addWidget(self.auto_zero_example_chip)
        self.auto_zero_synonym_chip = QPushButton("")
        self.auto_zero_synonym_chip.setStyleSheet(_CHIP_QSS)
        self.auto_zero_synonym_chip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_zero_synonym_chip.clicked.connect(
            lambda: self.auto_zero_assist_clicked.emit(
                "synonym", getattr(self, "_auto_zero_synonym", "") or ""))
        _za_col.addWidget(self.auto_zero_synonym_chip)
        self._auto_zero_synonym = ""
        self.auto_zero_assist_row.setVisible(False)
        _s3_layout.addWidget(self.auto_zero_assist_row)

        # 11b. The offer shown to a free user when a run stops on an exhausted
        # allowance (Moment C). The partial results are still kept in review,
        # so this is a nudge and not a wall: the compact variant of the shared
        # offer card (upsell_card.py), one line and an outline button. Hidden
        # by default; shown by set_auto_exhausted_subscribe_visible.
        self.auto_exhausted_subscribe = UpsellCard(
            "autoExhaustedOffer", "compact", self._on_upgrade_clicked)
        # The button keeps the old attribute name: the upgrade handler tells
        # this surface apart by the sender's identity.
        self.auto_exhausted_subscribe_link = self.auto_exhausted_subscribe.button
        self.auto_exhausted_subscribe.set_text(
            dial_copy(
                "upsell.exhausted_title",
                tr("Your Automatic allowance ran out mid-zone.")),
            # Served, like the rest of the upsell copy: the sentence quotes the
            # Pro monthly quota, so a hardcoded number makes the plugin
            # misstate the offer until the next release.
            dial_copy(
                "upsell.exhausted_link",
                tr("Pro picks it up where it stopped and finishes the zone.")),
            dial_copy("upsell.exhausted_cta", tr("Finish with Pro")),
        )
        self.auto_exhausted_subscribe.setVisible(False)
        _s3_layout.addWidget(self.auto_exhausted_subscribe)

        # 12. Post-run review panel, built by DockAutoReviewBuildMixin
        # (auto_review_build.py) so this construction module stays a readable
        # size. A zero-detection run reuses the status banner above, no box.
        self._setup_auto_review_panel(_s3_layout)

        # Top-align the prompt/review page too: without a trailing stretch
        # the layout hands its surplus height to the review card (the only
        # growable child), which pads the card with dead space and pushes
        # the step primary to the bottom of the panel.
        _s3_layout.addStretch(1)

        # Top-align every page's content inside the stacked widget (step 2
        # uses matched stretches so its draw hero floats mid-panel).
        _s1_layout.addStretch()
        _s2_layout.addStretch(1)
        # Quiet keyboard badge pinned at the panel bottom for the draw step,
        # away from the centered hero. Only the two discreet helpers (undo,
        # cancel); finishing is taught by the hero's "close on first point".
        self._auto_zone_keys = make_shortcut_hint([
            (native_key(Qt.Key.Key_Backspace), tr("undo point")),
            (native_key(Qt.Key.Key_Escape), tr("cancel")),
        ])
        self._auto_zone_keys.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._auto_zone_keys.setStyleSheet(
            "font-size: 11px; color: rgba(128,128,128,0.75);")
        _s2_layout.addWidget(self._auto_zone_keys)
        _s3_layout.addStretch()

        # Stretch factors so the visible step page absorbs the panel height
        # (the steps' internal stretches can then center their content; the
        # main layout's own trailing stretch has factor 0 and yields).
        auto_layout.addWidget(self.auto_controls_section, 1)
        # Absorbs the page height when the controls are hidden (upsell card
        # state), keeping the card compact at the top.
        auto_layout.addStretch()
        self.auto_page.setVisible(False)
        self.main_layout.addWidget(self.auto_page, 1)

    def _on_auto_upsell_manual_clicked(self) -> None:
        """Take the free way out of the exhausted card: the other mode.

        Through the switch's own handler, so a run in progress still blocks
        the move and the segmented control cannot end up showing a mode the
        page is not on.
        """
        try:
            self._on_mode_selected(Mode.INTERACTIVE)
        except (RuntimeError, AttributeError):
            return

    def get_auto_confidence(self) -> float:
        """Current cloud-model detection-confidence threshold from the Automatic panel.

        Falls back to the default if the widget was not built yet (e.g. early
        startup), so callers never need a None check.
        """
        spin = getattr(self, "auto_confidence_spin", None)
        if spin is None:
            return _AUTO_DEFAULT_CONFIDENCE
        return float(spin.value())

    def get_auto_min_size(self) -> float:
        """Review Min-size filter in m2 (0 = off). Falls back to 0 pre-build."""
        spin = getattr(self, "auto_min_size_spin", None)
        return float(spin.value()) if spin is not None else 0.0

    def get_auto_max_size(self) -> float:
        """Review Max-size filter in m2 (0 = no limit). Falls back to 0 pre-build."""
        spin = getattr(self, "auto_max_size_spin", None)
        return float(spin.value()) if spin is not None else 0.0

    def get_auto_fill_holes_max(self) -> float:
        """Review Fill-holes size threshold in ground m2 (0 = fill every hole).

        Its own accessor rather than a seventh slot in get_auto_refine_params,
        so the published tuple keeps its shape. Falls back to the generic
        client default pre-build."""
        spin = getattr(self, "auto_fill_max_spin", None)
        if spin is None:
            return _AUTO_REVIEW_FILL_MAX_M2_DEFAULT
        return max(0.0, float(spin.value()))

    def _sync_auto_fill_max_row(self) -> None:
        """Show the size threshold only while Fill holes is on (an irrelevant
        control is hidden, never greyed)."""
        row = getattr(self, "auto_fill_max_row", None)
        check = getattr(self, "auto_fill_holes_check", None)
        if row is not None and check is not None:
            row.setVisible(check.isChecked())

    def get_auto_points_pct(self) -> int:
        """The Points control: the share of its own points each outline keeps.

        Its own accessor rather than a seventh slot in get_auto_refine_params,
        whose tuple shape other call sites unpack positionally.
        """
        spin = getattr(self, "auto_points_spin", None)
        if spin is None:
            return _AUTO_REVIEW_POINTS_PCT_DEFAULT
        try:
            return int(spin.value())
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return _AUTO_REVIEW_POINTS_PCT_DEFAULT

    def get_auto_refine_params(self) -> tuple[float, bool, int, bool, float, bool]:
        """Current Automatic-review shape-refine controls as
        (simplify_px, round_corners, expand_px, fill_holes, clean_px,
        right_angles). Falls back to the faithful-by-default values pre-build
        (simplify low, no round, expand 0, fill holes off so holes are
        preserved, light clean, no right angles). simplify_px and clean_px are
        floats (sub-pixel tolerances allowed). Points is read separately via
        get_auto_points_pct, so this tuple keeps its positional shape."""
        simplify = getattr(self, "auto_simplify_spin", None)
        round_c = getattr(self, "auto_round_corners_check", None)
        expand = getattr(self, "auto_expand_spin", None)
        fill = getattr(self, "auto_fill_holes_check", None)
        clean = getattr(self, "auto_clean_spin", None)
        ortho = getattr(self, "auto_ortho_check", None)
        right_angles = bool(ortho.isChecked()) if ortho is not None else _AUTO_REVIEW_ORTHO_DEFAULT
        # The UI disables Trim spikes and Round corners under Right angles.
        # Repeat the rule here so programmatic callers and stale widget values
        # cannot stack a second cleanup/rounding pass onto the controlled
        # regularizer. Simplify and Points both stay live: squaring runs on a
        # de-staircased outline, and both are passes that produce one.
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
