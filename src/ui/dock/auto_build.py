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
    BRAND_BLUE,
    _BTN_BLUE,
    _BTN_BLUE_OUTLINE,
    _BTN_BLUE_PRIMARY,
    _BTN_CHIP,
    _BTN_GHOST,
    _BTN_GREEN,
    _BTN_LINK,
    _BTN_LINK_MUTED,
    _CARD_MARGINS,
    _CARD_QSS,
    _CHIP_QSS,
    _MSG_GLYPHS,
    _PROGRESS_THIN_QSS,
    _RECAP_CARD_QSS,
    _SLIDER_QSS,
    _btn_toggle_qss,
    _micro_header,
    _msg_card_qss,
    _msg_label_qss,
    _step_dial,
)
from .guidance import (
    BLUE_TINT,
    GREEN_TINT,
    HINT_EXEMPLAR_TIP,
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
        self.auto_upsell_card.setObjectName("autoUpsellCard")
        self.auto_upsell_card.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        _card_btn_qss = "QPushButton { border: none; }"  # ui-ok: child-button border reset inside the card
        self.auto_upsell_card.setStyleSheet(
            _msg_card_qss("autoUpsellCard", "info") + _card_btn_qss)
        upsell_layout = QVBoxLayout(self.auto_upsell_card)
        upsell_layout.setContentsMargins(10, 10, 10, 10)
        upsell_layout.setSpacing(8)

        # The count is filled in from the fetched free-detection total by
        # _refresh_auto_upsell_title (a number-free fallback until it is known).
        self._auto_upsell_title = QLabel(tr("Your free detections are used up"))
        self._auto_upsell_title.setStyleSheet("font-weight: bold; font-size: 12px; color: palette(text);")
        self._auto_upsell_title.setWordWrap(True)
        upsell_layout.addWidget(self._auto_upsell_title)

        _upsell_sub = QLabel(tr("Subscribe to keep detecting without limits:"))
        _upsell_sub.setStyleSheet("font-size: 11px; color: palette(text);")
        _upsell_sub.setWordWrap(True)
        upsell_layout.addWidget(_upsell_sub)

        for bullet in [
            tr("10,000 detections every month (~1,700 km²)"),
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
        # Green like its Manual sibling: the two mode descriptions are the
        # same kind of information, so they wear the same coat.
        self.auto_start_caption = DismissibleHint(
            HINT_START_AUTO,
            tr("Finds every object of one kind in your zone - draw a zone, "
               "name the object, get all the polygons at once."),
            tint=GREEN_TINT,
            show_glyph=False,  # a mode description, not a tip
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
        self.auto_last_run_recap.setStyleSheet(_RECAP_CARD_QSS)
        self.auto_last_run_recap.setVisible(False)
        _s1_layout.addWidget(self.auto_last_run_recap)

        # Post-export success line: after Finish the flow returns here and the
        # run status is wiped, so without this the user never learns WHERE the
        # result went. A one-line lime success message naming the layer, set
        # AFTER the reset (which clears it), dismissed on the next Start / mode
        # switch. PlainText so a layer name with an & stays literal.
        self.auto_export_success = QLabel()
        self.auto_export_success.setWordWrap(True)
        self.auto_export_success.setTextFormat(Qt.TextFormat.PlainText)
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

        # ---- Step 2: describe, then (optionally) show an example, then detail.
        # Three calm cards, one job each, read top to bottom so the user does
        # one thing at a time instead of facing a wall of parameters. Only the
        # required path is numbered (1 describe, 2 detail); the example card
        # sits between them UNNUMBERED and marked Optional, so it never reads
        # as a mandatory step (a plain description is the recommended path).
        # Detect enables on EITHER a valid prompt or one positive example, so
        # neither the prompt nor the example card is strictly mandatory.

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
            f"QLineEdit:focus {{ border: 1px solid {BRAND_BLUE}; }}")
        self.auto_prompt_input.textChanged.connect(self._on_auto_search_text_changed)
        self.auto_prompt_input.returnPressed.connect(self._on_auto_search_return_pressed)
        # Enter / focus-out = the prompt is settled: flush the debounce and
        # commit immediately, so unknown words (which skip the mid-typing
        # debounce commit, see _prompt_plausibly_complete) still seed the
        # detail default and fire their one commit before Detect.
        self.auto_prompt_input.editingFinished.connect(
            self._on_auto_prompt_editing_finished)
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
        self._set_prompt_info()

        _s3_layout.addWidget(self.auto_prompt_card)

        # --- Optional example card, ALWAYS VISIBLE like its step siblings
        # (a collapsed row read as noise, not as an option). Optionality is a
        # clearly readable "Optional" pill on the header, the title stays a
        # plain noun ("Add an example") and the button inside keeps the map
        # verb ("Draw an example"), so no two lines repeat each other. The
        # explainer under the header says why/how; it yields to the armed
        # instruction or the drawn thumbnails. Gated behind _EXEMPLARS_ENABLED.
        self.auto_exemplar_panel = QWidget()
        self.auto_exemplar_panel.setObjectName("autoExemplarCard")
        self.auto_exemplar_panel.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_exemplar_panel.setStyleSheet(
            _CARD_QSS.format(name="autoExemplarCard"))
        _ex_outer = QVBoxLayout(self.auto_exemplar_panel)
        _ex_outer.setContentsMargins(*_CARD_MARGINS)
        _ex_outer.setSpacing(6)

        # Header row: bold title + a bordered "Optional" pill (readable in
        # both themes, unlike the old small grey word). Wrapped in one widget
        # so the in-run read-only swap can hide the whole header at once.
        self._auto_exemplar_expanded = True
        self._auto_exemplar_header = QWidget()
        _ex_hdr_row = QHBoxLayout(self._auto_exemplar_header)
        _ex_hdr_row.setContentsMargins(0, 0, 0, 0)
        _ex_hdr_row.setSpacing(8)
        _ex_title = QLabel(tr("Add an example"))
        _ex_title.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: palette(text);")
        _ex_hdr_row.addWidget(_ex_title)
        _ex_optional = QLabel(tr("Optional"))
        _ex_optional.setStyleSheet(
            "font-size: 10px; color: palette(text);"
            " border: 1px solid rgba(128,128,128,0.45); border-radius: 8px;"
            " padding: 1px 8px;")
        _ex_hdr_row.addWidget(_ex_optional)
        _ex_hdr_row.addStretch(1)
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
        # coloured so it reads as the primary action. While armed it fills in
        # solid and a blue instruction line tells the user to outline on the map.
        # The armed state is driven by the plugin via set_auto_exemplar_armed, so
        # a cancel (Escape) or a finished draw both clear it. The [armed] dynamic
        # property toggles the filled look without rebuilding the stylesheet.
        # Quiet ghost rest state: the example path is optional, so its button
        # must never compete with Detect (the screen's one loud primary). It
        # takes the green only on hover, and fills solid while armed.
        _ex_inc_style = _btn_toggle_qss(
            (67, 160, 71), "#6bbf6f", "#06210b", quiet=True)
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
        # "Draw on the map" (the how), never a re-statement of the card title
        # "Add an example" (the what): the two lines must not repeat.
        self.auto_ex_inc_btn = QPushButton(tr("Draw on the map"))
        self.auto_ex_inc_btn.setStyleSheet(_ex_inc_style)
        self.auto_ex_inc_btn.setMinimumHeight(28)
        self.auto_ex_inc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_ex_inc_btn.setToolTip(tr("Mark an object to find more like it."))
        self.auto_ex_inc_btn.clicked.connect(
            lambda: self.auto_add_exemplar_requested.emit(1))
        _ex_mode_row.addWidget(self.auto_ex_inc_btn, 1)
        self.auto_ex_exc_btn = QPushButton(tr("Exclude a look-alike"))
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
        _ex_edit_col.addLayout(_ex_mode_row)

        # One short blue tip UNDER the button (dismissible with the tiny x,
        # like every other blue hint): what an example buys, in one line. It
        # also yields to the armed instruction or the drawn thumbnails (see
        # _refresh_auto_exemplar_explainer).
        self.auto_exemplar_explainer = DismissibleHint(
            HINT_EXEMPLAR_TIP,
            tr("The AI finds every object similar to your example."),
            tint=BLUE_TINT,
        )
        _ex_edit_col.addWidget(self.auto_exemplar_explainer)

        # Armed instruction line: hidden until a button arms the draw tool, then
        # a blue callout telling the user to outline an object on the map. This
        # is the "in-between" feedback that the click started a draw action.
        self.auto_exemplar_armed_hint = QLabel("")
        self.auto_exemplar_armed_hint.setWordWrap(True)
        self.auto_exemplar_armed_hint.setStyleSheet(_msg_label_qss("armed"))
        self.auto_exemplar_armed_hint.setVisible(False)
        _ex_edit_col.addWidget(self.auto_exemplar_armed_hint)
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
        _detail_outer.setContentsMargins(*_CARD_MARGINS)
        _detail_outer.setSpacing(4)
        # Header row: step dial + "Detail" on the left, the live credit cost on
        # the right, so the price of the chosen level is read at a glance (the
        # map shows the matching tile grid). No tile-count or m/px jargon here.
        _detail_hdr = QHBoxLayout()
        _detail_hdr.setContentsMargins(0, 0, 0, 0)
        _detail_hdr.setSpacing(6)
        _detail_hdr.addWidget(_step_dial(2, "active"))
        _detail_lbl = QLabel(tr("Detail"))
        _detail_lbl.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: palette(text);")
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
        # Free-plan per-run credit cap state: the slider keeps its full (Pro)
        # travel; past the cap the ESTIMATE gates Detect (red cost line, see
        # set_auto_credit_estimate) and this hint becomes an upgrade link.
        self._auto_free_run_cap = None
        self._auto_premium_gated = False
        self._detail_cap_upsell_tracked = False
        # Object-aware slider verdict (state, object word), pushed by the
        # plugin from the credit-estimate chokepoint; None until known.
        self._auto_detail_feedback = None
        self.auto_detail_hint.linkActivated.connect(
            self._on_detail_cap_upgrade_link)
        _detail_outer.addWidget(self.auto_detail_hint)

        # Conditional amber warning, shown by set_auto_detail_gsd_warning when
        # the chosen detail leaves the imagery too coarse for the cloud model. A proper
        # boxed alert with a warning icon (mirrors the no-rasters warning) so it
        # reads as a real callout, not recoloured hint text. Hidden by default;
        # the neutral hint hides while it shows so guidance never stacks.
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
            "This area is large for this detail level. Raise detail or zoom"
            " in for sharper detections."))
        self.auto_detail_warning_label.setWordWrap(True)
        self.auto_detail_warning_label.setStyleSheet("font-size: 11px;")
        _warn_layout.addWidget(self.auto_detail_warning_label, 1)
        self.auto_detail_warning.setVisible(False)
        _detail_outer.addWidget(self.auto_detail_warning)
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

        # 11. Status banner
        self.auto_status_banner = QLabel("")
        self.auto_status_banner.setWordWrap(True)
        self.auto_status_banner.setStyleSheet(_msg_label_qss("info"))
        self.auto_status_banner.setVisible(False)
        _s3_layout.addWidget(self.auto_status_banner)

        # 11a. Zero-result rescue, right under the status banner. A paid run
        # that found nothing is the worst moment of the flow, and the drawn
        # example is the proven lever that rescues it, so it leads: a
        # full-width blue-outline call (the strong-secondary family, same as
        # "Refine in Manual mode") with the outcome in its label. The synonym
        # prefill stays a quiet chip below it, only when the server steer
        # table knows a stronger word. Hidden by default; driven by
        # show/hide_auto_zero_assist. The row never outlives its status:
        # set_auto_status hides it on every call.
        self.auto_zero_assist_row = QWidget()
        _za_col = QVBoxLayout(self.auto_zero_assist_row)
        _za_col.setContentsMargins(0, 0, 0, 0)
        _za_col.setSpacing(4)
        self.auto_zero_example_chip = QPushButton("")
        self.auto_zero_example_chip.setStyleSheet(_BTN_BLUE_OUTLINE)
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

        # 11b. Subscribe link for free users when a run stops on exhausted
        # credits (Moment C). A quiet text-link under the status; the partial
        # results are still kept in review. Hidden by default; shown by
        # set_auto_exhausted_subscribe_visible.
        self.auto_exhausted_subscribe_link = QPushButton(
            tr("Subscribe to finish this zone: 10,000 credits/month."))
        self.auto_exhausted_subscribe_link.setStyleSheet(_BTN_LINK)
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
