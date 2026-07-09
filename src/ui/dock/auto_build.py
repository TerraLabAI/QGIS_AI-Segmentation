"""Automatic page construction (3-step stacked flow) and its value getters.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations


from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QDoubleSpinBox,
    QSlider,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)


from ..layer_tree_combobox import LayerTreeComboBox

from ...core.activation_manager import has_tos_accepted, has_tos_locked
from ...core.i18n import tr
from ...core.review_defaults import (
    AUTO_DEFAULT_CONFIDENCE as _AUTO_DEFAULT_CONFIDENCE,
    AUTO_REVIEW_CLEAN_DEFAULT as _AUTO_REVIEW_CLEAN_DEFAULT,
    AUTO_REVIEW_EXPAND_DEFAULT as _AUTO_REVIEW_EXPAND_DEFAULT,
    AUTO_REVIEW_FILL_HOLES_DEFAULT as _AUTO_REVIEW_FILL_HOLES_DEFAULT,
    AUTO_REVIEW_ORTHO_DEFAULT as _AUTO_REVIEW_ORTHO_DEFAULT,
    AUTO_REVIEW_SIMPLIFY_DEFAULT as _AUTO_REVIEW_SIMPLIFY_DEFAULT,
    AUTO_REVIEW_SMOOTH_DEFAULT as _AUTO_REVIEW_SMOOTH_DEFAULT,
)
from ...core.tile_manager import MAX_DETAIL_LEVEL
from .styles import (
    _BTN_BLUE,
    _BTN_BLUE_PRIMARY,
    _BTN_GHOST,
    _BTN_GREEN,
    _CARD_QSS,
    _SLIDER_QSS,
    _auto_step_header,
)
from .guidance import (
    BLUE_TINT,
    HINT_START_AUTO,
    HINT_TUTORIAL_FIRST_STEPS,
    NEUTRAL_TINT,
    DismissibleHint,
    open_guide,
)
from .widgets import (
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
        self.auto_upsell_card.setStyleSheet(
            "QFrame { background-color: rgba(25, 118, 210, 0.08);"
            " border: 1px solid rgba(25, 118, 210, 0.2); border-radius: 4px; }"
            "QLabel { background: transparent; border: none; }"
            "QPushButton { border: none; }"
        )
        upsell_layout = QVBoxLayout(self.auto_upsell_card)
        upsell_layout.setContentsMargins(10, 10, 10, 10)
        upsell_layout.setSpacing(8)

        _upsell_title = QLabel(tr("Your 300 free detections are used up"))
        _upsell_title.setStyleSheet("font-weight: bold; font-size: 12px; color: palette(text);")
        _upsell_title.setWordWrap(True)
        upsell_layout.addWidget(_upsell_title)

        _upsell_sub = QLabel(tr("Subscribe to keep detecting without limits:"))
        _upsell_sub.setStyleSheet("font-size: 11px; color: palette(text);")
        _upsell_sub.setWordWrap(True)
        upsell_layout.addWidget(_upsell_sub)

        for bullet in [
            tr("10,000 detections every month (~1,700 km2)"),
            tr("Every building, tree, or road as clean polygons"),
            tr("Cancel anytime; your exported layers stay yours"),
        ]:
            _lbl = QLabel(bullet)
            _lbl.setStyleSheet("font-size: 11px; color: palette(text);")
            _lbl.setWordWrap(True)
            upsell_layout.addWidget(_lbl)

        # Reassurance: even out of free detections, the local Manual mode never
        # stops. Keeps the exhausted card from reading as a hard wall.
        _upsell_free = QLabel(
            tr("Manual mode stays free and unlimited on your computer."))
        _upsell_free.setStyleSheet("font-size: 11px; color: rgba(128,128,128,0.95);")
        _upsell_free.setWordWrap(True)
        upsell_layout.addWidget(_upsell_free)

        self.auto_upgrade_btn = QPushButton(tr("Upgrade to Pro"))
        self.auto_upgrade_btn.setStyleSheet(_BTN_BLUE)
        self.auto_upgrade_btn.setMinimumHeight(36)
        self.auto_upgrade_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_upgrade_btn.clicked.connect(self._on_upgrade_clicked)
        upsell_layout.addWidget(self.auto_upgrade_btn)

        _upsell_hint = QLabel(tr("Opens your TerraLab dashboard"))
        _upsell_hint.setStyleSheet("font-size: 10px; color: rgba(128,128,128,0.95);")
        _upsell_hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        _upsell_hint.setWordWrap(True)
        upsell_layout.addWidget(_upsell_hint)

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
        self.auto_layer_label = QLabel(tr("Select a Raster Layer to Segment:"))
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
        self.auto_start_btn.setStyleSheet(_BTN_BLUE_PRIMARY)
        self.auto_start_btn.setMinimumHeight(36)
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
            # never shouts.
            tr("New here? Our 5-minute tutorial walks you through a full "
               "detection, step by step."),
            tint=NEUTRAL_TINT,
            action_text=tr("Open the tutorial"),
            action_color=BLUE_TINT,
            visibility_gate=self._should_show_auto_tutorial,
        )
        self.auto_tutorial_banner.action.connect(
            lambda: open_guide("post_signin"))

        # "What is this mode for" caption: a quiet framed card
        # under the Start button. One plain sentence; no free/paid wording,
        # no cloud/local wording. Dismissible (small x); only
        # on step 0, so it never shows mid-flow. Re-enable from Account Settings.
        self.auto_start_caption = DismissibleHint(
            HINT_START_AUTO,
            tr("Finds every object of one kind in your zone - draw a zone, "
               "name the object, get all the polygons at once."),
            tint=BLUE_TINT,
        )
        _s1_layout.addWidget(self.auto_start_caption)

        # Last-run value recap: a quiet one-line card shown on the Start page
        # right after a successful Finish, so the value a run created stays
        # visible ("137 building exported") instead of vanishing on the return
        # to Start. Session only, no persistence, no close button (the next
        # Finish replaces it; clicking Start hides it). Filled + shown by
        # set_last_run_recap(); hidden by default and by clear_last_run_recap().
        # Placement: directly UNDER the what-is-this-mode
        # caption, so the reading order is Start action, what the mode does,
        # then what the last run produced; when the caption is dismissed Qt
        # collapses it and the recap sits right under the Start button. Living
        # inside step 0 also keeps it off the mid-flow steps for free.
        self.auto_last_run_recap = QLabel()
        self.auto_last_run_recap.setWordWrap(True)
        self.auto_last_run_recap.setTextFormat(Qt.TextFormat.PlainText)
        self.auto_last_run_recap.setStyleSheet(
            "font-size: 11px; color: palette(text);"
            " border: 1px solid rgba(120, 144, 156, 90);"
            " border-radius: 6px; padding: 6px 8px;"
            " background: rgba(120, 144, 156, 22);"
        )
        self.auto_last_run_recap.setVisible(False)
        _s1_layout.addWidget(self.auto_last_run_recap)

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

        # ---- Step 2: describe, then (optionally) show an example, then detail.
        # Three calm cards, one job each, read top to bottom as an ordered
        # checklist so the user does one thing at a time instead of facing a
        # wall of parameters. Card 1 (text prompt) is the primary, always-visible
        # input; card 2 (a drawn example) is an optional quality booster; card 3
        # is the detail level. Detect enables on EITHER a valid prompt or one
        # positive example, so neither card 1 nor card 2 is strictly mandatory.

        # --- Card 1: describe what to find (the text prompt). ---
        self.auto_prompt_card = QWidget()
        self.auto_prompt_card.setObjectName("autoPromptCard")
        self.auto_prompt_card.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_prompt_card.setStyleSheet(_CARD_QSS.format(name="autoPromptCard"))
        _prompt_card_layout = QVBoxLayout(self.auto_prompt_card)
        _prompt_card_layout.setContentsMargins(10, 8, 10, 10)
        _prompt_card_layout.setSpacing(6)
        self._auto_prompt_header = QLabel(
            _auto_step_header(1, tr("Describe what to find")))
        self._auto_prompt_header.setStyleSheet(
            "font-size: 11px; color: palette(text);")
        _prompt_card_layout.addWidget(self._auto_prompt_header)

        # Input row: the prompt box and the Library button side by side, equal
        # heights. The Library is the guided path to a working prompt (curated
        # English tokens with before/after previews), so it sits right where
        # the eye lands instead of a lost ghost button below.
        self._auto_prompt_valid = False
        _prompt_row = QHBoxLayout()
        _prompt_row.setContentsMargins(0, 0, 0, 0)
        _prompt_row.setSpacing(6)
        self.auto_prompt_input = QLineEdit()
        self.auto_prompt_input.setPlaceholderText(tr("e.g. building, tree, road, car"))
        self.auto_prompt_input.setClearButtonEnabled(True)
        self.auto_prompt_input.setStyleSheet(
            "QLineEdit { border: 1px solid rgba(128,128,128,0.35);"
            " border-radius: 6px; padding: 7px 10px; background: palette(base);"
            " color: palette(text); }"
            "QLineEdit:focus { border: 1px solid #1e88e5; }")
        self.auto_prompt_input.textChanged.connect(self._on_auto_search_text_changed)
        self.auto_prompt_input.returnPressed.connect(self._on_auto_search_return_pressed)
        _prompt_row.addWidget(self.auto_prompt_input, 1)
        self.auto_library_btn = QPushButton("▦  " + tr("Library"))
        self.auto_library_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_library_btn.setToolTip(
            tr("Browse ready-to-use objects with before / after previews."))
        self.auto_library_btn.setStyleSheet(
            "QPushButton { background: rgba(30,136,229,0.10); color: #1e88e5;"
            " border: 1px solid rgba(30,136,229,0.55); border-radius: 6px;"
            " padding: 7px 12px; font-size: 12px; font-weight: 600; }"
            "QPushButton:hover { background: rgba(30,136,229,0.20); }"
            "QPushButton:disabled { background: transparent;"
            " color: rgba(128,128,128,0.5); border-color: rgba(128,128,128,0.3); }")
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
        self._set_prompt_info()

        _s3_layout.addWidget(self.auto_prompt_card)

        # --- Card 2: show an example (optional). A drawn example is the cloud model's
        # single biggest quality lever (it matches on appearance, so it wins on
        # rare / hard-to-name / aerial objects where text alone fails). A single
        # Draw-example button arms the draw tool; the object is masked to the
        # drawn outline so surrounding ground never leaks into the reference.
        # Everything lives in auto_exemplar_panel so a single visibility toggle
        # controls the whole card. Gated behind _EXEMPLARS_ENABLED.
        self.auto_exemplar_panel = QWidget()
        self.auto_exemplar_panel.setObjectName("autoExemplarCard")
        self.auto_exemplar_panel.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_exemplar_panel.setStyleSheet(
            _CARD_QSS.format(name="autoExemplarCard"))
        _ex_outer = QVBoxLayout(self.auto_exemplar_panel)
        _ex_outer.setContentsMargins(10, 8, 10, 10)
        _ex_outer.setSpacing(6)
        _ex_header = QLabel(
            _auto_step_header(2, tr("Show an example"), optional=True))
        _ex_header.setStyleSheet("font-size: 11px; color: palette(text);")
        _ex_outer.addWidget(_ex_header)
        # Read-only caption, shown in its place during a run: the reference stays
        # on screen (browsable) but every editing affordance is gone.
        self._auto_exemplar_header = _ex_header
        self.auto_exemplar_readonly_caption = QLabel(tr("Your reference"))
        self.auto_exemplar_readonly_caption.setStyleSheet(
            "font-size: 11px; color: palette(text);")
        self.auto_exemplar_readonly_caption.setVisible(False)
        _ex_outer.addWidget(self.auto_exemplar_readonly_caption)
        # All the editing controls (hint + draw/exclude buttons + armed line)
        # live in one container so a single toggle removes them for the
        # read-only in-run variant, leaving just the reference thumbnails.
        self.auto_exemplar_edit_controls = QWidget()
        _ex_edit_col = QVBoxLayout(self.auto_exemplar_edit_controls)
        _ex_edit_col.setContentsMargins(0, 0, 0, 0)
        _ex_edit_col.setSpacing(6)
        _ex_hint = QLabel(tr("Outline one object; the AI finds the rest."))
        _ex_hint.setWordWrap(True)
        _ex_hint.setStyleSheet("font-size: 10px; color: rgba(128,128,128,0.9);")
        _ex_edit_col.addWidget(_ex_hint)

        # The draw-example button IS the action button: clicking it arms the draw
        # tool directly (no separate "Draw" step). It is big, full-width and
        # coloured so it reads as the primary action. While armed it fills in
        # solid and a blue instruction line tells the user to outline on the map.
        # The armed state is driven by the plugin via set_auto_exemplar_armed, so
        # a cancel (Escape) or a finished draw both clear it. The [armed] dynamic
        # property toggles the filled look without rebuilding the stylesheet.
        _ex_inc_style = (
            "QPushButton { background: rgba(67,160,71,0.14); color: #6bbf6f;"
            " border: 1px solid rgba(67,160,71,0.55); border-radius: 6px;"
            " padding: 9px 16px; font-size: 12px; font-weight: 700; }"
            "QPushButton:hover { background: rgba(67,160,71,0.24); }"
            'QPushButton[armed="true"] { background: #43a047; color: #06210b;'
            " border: 1px solid #43a047; }"
            "QPushButton:disabled { background: transparent;"
            " color: rgba(128,128,128,0.5); border-color: rgba(128,128,128,0.3); }"
        )
        # The exclude button is the red counterpart: it drops false positives
        # by pointing at a look-alike the model should NOT return. Secondary and
        # rare, so it appears only once at least one positive example exists (see
        # set_exemplars); the primary flow stays a single green button.
        _ex_exc_style = (
            "QPushButton { background: rgba(229,57,53,0.10); color: #e57373;"
            " border: 1px solid rgba(229,57,53,0.50); border-radius: 6px;"
            " padding: 9px 16px; font-size: 12px; font-weight: 600; }"
            "QPushButton:hover { background: rgba(229,57,53,0.20); }"
            'QPushButton[armed="true"] { background: #e53935; color: #2a0606;'
            " border: 1px solid #e53935; }"
            "QPushButton:disabled { background: transparent;"
            " color: rgba(128,128,128,0.5); border-color: rgba(128,128,128,0.3); }"
        )
        _ex_mode_row = QHBoxLayout()
        _ex_mode_row.setContentsMargins(0, 0, 0, 0)
        _ex_mode_row.setSpacing(8)
        self.auto_ex_inc_btn = QPushButton(tr("Draw an example"))
        self.auto_ex_inc_btn.setStyleSheet(_ex_inc_style)
        self.auto_ex_inc_btn.setMinimumHeight(34)
        self.auto_ex_inc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_ex_inc_btn.setToolTip(tr("Mark an object to find more like it."))
        self.auto_ex_inc_btn.clicked.connect(
            lambda: self.auto_add_exemplar_requested.emit(1))
        _ex_mode_row.addWidget(self.auto_ex_inc_btn, 1)
        self.auto_ex_exc_btn = QPushButton(tr("Exclude a look-alike"))
        self.auto_ex_exc_btn.setStyleSheet(_ex_exc_style)
        self.auto_ex_exc_btn.setMinimumHeight(34)
        self.auto_ex_exc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_ex_exc_btn.setToolTip(
            tr("Mark a false positive to drop things like it."))
        self.auto_ex_exc_btn.clicked.connect(
            lambda: self.auto_add_exemplar_requested.emit(0))
        self.auto_ex_exc_btn.setVisible(False)
        _ex_mode_row.addWidget(self.auto_ex_exc_btn, 0)
        _ex_edit_col.addLayout(_ex_mode_row)

        # Armed instruction line: hidden until a button arms the draw tool, then
        # a blue callout telling the user to outline an object on the map. This
        # is the "in-between" feedback that the click started a draw action.
        self.auto_exemplar_armed_hint = QLabel("")
        self.auto_exemplar_armed_hint.setWordWrap(True)
        self.auto_exemplar_armed_hint.setStyleSheet(
            "QLabel { background-color: rgba(30,136,229,0.12);"
            " border: 1px solid rgba(30,136,229,0.40); border-radius: 6px;"
            " padding: 7px 9px; font-size: 11px; color: palette(text); }")
        self.auto_exemplar_armed_hint.setVisible(False)
        _ex_edit_col.addWidget(self.auto_exemplar_armed_hint)
        _ex_outer.addWidget(self.auto_exemplar_edit_controls)

        # Reference thumbnail strip: one card per drawn example (AI-Edit
        # _ThumbWidget look - thumbnail + numbered badge + hover-x), rebuilt by
        # set_exemplars().
        self.auto_exemplar_chips = QWidget()
        self._auto_exemplar_chips_layout = QHBoxLayout(self.auto_exemplar_chips)
        self._auto_exemplar_chips_layout.setContentsMargins(0, 2, 0, 0)
        self._auto_exemplar_chips_layout.setSpacing(6)
        self._auto_exemplar_chips_layout.addStretch()
        _ex_outer.addWidget(self.auto_exemplar_chips)
        self.auto_exemplar_panel.setVisible(False)
        _s3_layout.addWidget(self.auto_exemplar_panel)

        # 5a. Detail slider (visible whenever a zone is drawn). The user picks
        # the grid subdivision n: the zone's longer side is rendered as exactly
        # n tiles, so a square zone costs n x n credits. Square-grid steps
        # (1x1 .. 7x7 = 49) keep every position under MAX_TILES = 50.
        # For local rasters the resolution is clamped to the native pixel size,
        # so the render never upsamples the source pixels.
        self.auto_detail_row = QWidget()
        self.auto_detail_row.setObjectName("autoDetailCard")
        self.auto_detail_row.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_detail_row.setStyleSheet(_CARD_QSS.format(name="autoDetailCard"))
        _detail_outer = QVBoxLayout(self.auto_detail_row)
        _detail_outer.setContentsMargins(10, 8, 10, 10)
        _detail_outer.setSpacing(4)
        # Header row: "Detail" on the left, the live credit cost on the right,
        # so the price of the chosen level is read at a glance (the map shows
        # the matching tile grid). No tile-count or m/px jargon here.
        _detail_hdr = QHBoxLayout()
        _detail_hdr.setContentsMargins(0, 0, 0, 0)
        _detail_lbl = QLabel(_auto_step_header(3, tr("Detail")))
        _detail_lbl.setStyleSheet(
            "font-size: 11px; color: palette(text);")
        _detail_hdr.addWidget(_detail_lbl)
        _detail_hdr.addStretch()
        # The live credit cost sits in the detail header (it tracks the slider
        # directly). Created here so all its existing visibility wiring keeps
        # working; the old separate placement below is removed.
        self.auto_credit_cost_label = QLabel("")
        self.auto_credit_cost_label.setStyleSheet(
            "font-size: 11px; color: palette(text);")
        self.auto_credit_cost_label.setVisible(False)
        _detail_hdr.addWidget(self.auto_credit_cost_label)
        _detail_outer.addLayout(_detail_hdr)
        # Slider row: plain "Less <-> More" ends (paired with the "Detail" title
        # above) replace the abstract grid numbers, and read simpler than the old
        # Coarse/Fine. The slider still drives the tile subdivision under the hood.
        _slider_row = QHBoxLayout()
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
            "Higher detail splits the zone into more tiles. Each tile costs"
            " 1 credit and captures smaller objects."))
        self.auto_detail_slider.valueChanged.connect(self._on_auto_detail_changed)
        # Gated until the object is defined (typed prompt or drawn example):
        # the default is object-aware, so an adjustment made before naming the
        # object got thrown away by the prompt-commit re-seed. See
        # _apply_auto_detail_gate (driven from _update_auto_detect_enabled).
        self.auto_detail_slider.setEnabled(False)
        _slider_row.addWidget(self.auto_detail_slider, 1)
        _fine_lbl = QLabel(tr("More"))
        _fine_lbl.setStyleSheet("font-size: 10px; color: palette(text);")
        _slider_row.addWidget(_fine_lbl)
        _detail_outer.addLayout(_slider_row)
        # One-line plain-language hint instead of a m/px figure. Starts on the
        # gated wording (slider disabled above); _apply_auto_detail_gate swaps
        # it once a prompt or an example exists.
        self.auto_detail_hint = QLabel(
            tr("Name the object (or draw an example) first - Detail "
               "then tunes itself to it."))
        self.auto_detail_hint.setWordWrap(True)
        self.auto_detail_hint.setStyleSheet(
            "font-size: 10px; color: palette(text);")
        _detail_outer.addWidget(self.auto_detail_hint)

        # Conditional amber warning, shown by set_auto_detail_gsd_warning when
        # the chosen detail leaves the imagery too coarse for the cloud model. A proper
        # boxed alert with a warning icon (mirrors the no-rasters warning) so it
        # reads as a real callout, not recoloured hint text. Hidden by default;
        # the neutral hint hides while it shows so guidance never stacks.
        self.auto_detail_warning = QWidget()
        self.auto_detail_warning.setStyleSheet(
            "QWidget { background-color: rgb(255, 230, 150);"
            " border: 1px solid rgba(255, 152, 0, 0.6); border-radius: 4px; }"
            "QLabel { background: transparent; border: none; color: #333333; }"
        )
        _warn_layout = QHBoxLayout(self.auto_detail_warning)
        _warn_layout.setContentsMargins(8, 6, 8, 6)
        _warn_layout.setSpacing(8)
        _warn_icon = QLabel()
        _wstyle = self.auto_detail_warning.style()
        _wico = _wstyle.pixelMetric(QStyle.PixelMetric.PM_SmallIconSize)
        _warn_icon.setPixmap(
            _wstyle.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
            .pixmap(_wico, _wico))
        _warn_icon.setFixedSize(_wico, _wico)
        _warn_layout.addWidget(_warn_icon, 0, Qt.AlignmentFlag.AlignTop)
        self.auto_detail_warning_label = QLabel(tr(
            "This area is large for this detail level. Raise detail or zoom"
            " in for sharper detections."))
        self.auto_detail_warning_label.setWordWrap(True)
        self.auto_detail_warning_label.setStyleSheet("font-size: 11px;")
        _warn_layout.addWidget(self.auto_detail_warning_label, 1)
        self.auto_detail_warning.setVisible(False)
        _detail_outer.addWidget(self.auto_detail_warning)
        self.auto_detail_row.setVisible(False)

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
        self.auto_settings_box.setStyleSheet(
            "QWidget#autoSettingsBox { background-color: rgba(128, 128, 128, 0.08);"
            " border: 1px solid rgba(128, 128, 128, 0.2); border-radius: 4px; }"
            "QLabel { background: transparent; border: none; }"
        )
        _settings_layout = QVBoxLayout(self.auto_settings_box)
        _settings_layout.setContentsMargins(10, 8, 10, 10)
        _settings_layout.setSpacing(6)

        _settings_hdr = QLabel(tr("Detection").upper())
        _settings_hdr.setStyleSheet(
            "font-size: 10px; color: palette(text); font-weight: bold; "
            "background: transparent; border: none; "
            "border-bottom: 1px solid rgba(128, 128, 128, 0.35); "
            "padding: 2px 0px 4px 0px; margin-bottom: 2px; letter-spacing: 1px;")
        _settings_layout.addWidget(_settings_hdr)

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
        _prog_col.setContentsMargins(10, 8, 10, 10)
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
        # Row 2: the bar (taller than the old slim one, brand blue, rounded).
        self.auto_tile_progress = QProgressBar()
        self.auto_tile_progress.setTextVisible(False)
        self.auto_tile_progress.setFixedHeight(8)
        self.auto_tile_progress.setStyleSheet(
            "QProgressBar { background: rgba(128,128,128,0.15); border: none;"
            " border-radius: 4px; }"
            "QProgressBar::chunk { background: #1e88e5; border-radius: 4px; }")
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
        self.auto_cancel_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none;"
            " color: rgba(128,128,128,0.9); font-size: 11px; padding: 4px 8px; }"
            "QPushButton:hover { color: #ef5350; text-decoration: underline; }")
        self.auto_cancel_btn.setVisible(False)
        self.auto_cancel_btn.clicked.connect(self._on_auto_cancel_clicked)
        _cancel_row = QHBoxLayout()
        _cancel_row.setContentsMargins(0, 0, 0, 0)
        _cancel_row.addStretch(1)
        _cancel_row.addWidget(self.auto_cancel_btn, 0)
        _cancel_row.addStretch(1)
        _s3_layout.addLayout(_cancel_row)

        # 11. Status banner
        self.auto_status_banner = QLabel("")
        self.auto_status_banner.setWordWrap(True)
        self.auto_status_banner.setStyleSheet(
            "background-color: rgba(128, 128, 128, 0.08);"
            " border: 1px solid rgba(128, 128, 128, 0.2);"
            " border-radius: 4px; padding: 8px; color: palette(text); font-size: 12px;"
        )
        self.auto_status_banner.setVisible(False)
        _s3_layout.addWidget(self.auto_status_banner)

        # 11b. Subscribe link for free users when a run stops on exhausted
        # credits (Moment C). A quiet text-link under the status; the partial
        # results are still kept in review. Hidden by default; shown by
        # set_auto_exhausted_subscribe_visible.
        self.auto_exhausted_subscribe_link = QPushButton(
            tr("Subscribe to finish this zone: 10,000 credits/month."))
        self.auto_exhausted_subscribe_link.setStyleSheet(
            "QPushButton { border: none; background: transparent; color: #1e88e5;"
            " font-size: 11px; text-align: left; padding: 2px 0px; }"
            "QPushButton:hover { text-decoration: underline; }")
        self.auto_exhausted_subscribe_link.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_exhausted_subscribe_link.setVisible(False)
        self.auto_exhausted_subscribe_link.clicked.connect(self._on_upgrade_clicked)
        _s3_layout.addWidget(self.auto_exhausted_subscribe_link)

        # 12. Post-run review panel, built by DockAutoReviewBuildMixin
        # (auto_review_build.py) so this construction module stays a readable
        # size. A zero-detection run reuses the status banner above, no box.
        self._setup_auto_review_panel(_s3_layout)

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

    def get_auto_refine_params(self) -> tuple[float, bool, int, bool, float, bool]:
        """Current Automatic-review shape-refine controls as
        (simplify_px, round_corners, expand_px, fill_holes, clean_px,
        right_angles). Falls back to the faithful-by-default values pre-build
        (simplify low, no round, expand 0, fill holes off so holes are
        preserved, light clean, no right angles). simplify_px and clean_px are
        floats (sub-pixel tolerances allowed)."""
        simplify = getattr(self, "auto_simplify_spin", None)
        round_c = getattr(self, "auto_round_corners_check", None)
        expand = getattr(self, "auto_expand_spin", None)
        fill = getattr(self, "auto_fill_holes_check", None)
        clean = getattr(self, "auto_clean_spin", None)
        ortho = getattr(self, "auto_ortho_check", None)
        return (
            float(simplify.value()) if simplify is not None else _AUTO_REVIEW_SIMPLIFY_DEFAULT,
            bool(round_c.isChecked()) if round_c is not None else _AUTO_REVIEW_SMOOTH_DEFAULT,
            int(expand.value()) if expand is not None else _AUTO_REVIEW_EXPAND_DEFAULT,
            bool(fill.isChecked()) if fill is not None else _AUTO_REVIEW_FILL_HOLES_DEFAULT,
            float(clean.value()) if clean is not None else _AUTO_REVIEW_CLEAN_DEFAULT,
            bool(ortho.isChecked()) if ortho is not None else _AUTO_REVIEW_ORTHO_DEFAULT,
        )
